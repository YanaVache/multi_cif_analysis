import io
import math
import re
import tempfile
import zipfile
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull

DEFAULT_MAX_COLUMNS = 5

FVAR_START_RE = re.compile(r'^FVAR\s+(.*)', re.IGNORECASE)
NUMBER_RE = re.compile(r'^-?\d+(?:\.\d+)?$')
ATOM_LINE_RE = re.compile(r'^(\S+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')


def get_res_text(source, data_name=None):
    source_str = str(source)
    if source_str.lower().endswith((".ins", ".res")):
        with open(source, encoding="utf-8", errors="replace") as f:
            return f.readlines()

    doc = gemmi.cif.read(source_str)
    if data_name is not None and len(doc) > 1:
        block = doc.find_block(data_name)
        if block is None:
            block = doc[0]
    else:
        block = doc.sole_block() if len(doc) == 1 else doc[0]

    text = block.find_value("_shelx_res_file")
    if text is None:
        return None
    return text.split("\n")

def parse_fvar(lines):
    values = []
    collecting = False
    for raw in lines:
        stripped = raw.rstrip("\r\n").strip()
        if collecting:
            tokens = stripped.split()
            if tokens and all(NUMBER_RE.match(t) for t in tokens):
                values.extend(float(t) for t in tokens)
                continue
            else:
                collecting = False
        m = FVAR_START_RE.match(stripped)
        if m:
            values.extend(float(t) for t in m.group(1).split())
            collecting = True
    return values

def find_atom_sof(lines, atom_name):
    for raw in lines:
        line = raw.rstrip("\r\n").strip()
        m = ATOM_LINE_RE.match(line)
        if m and m.group(1).upper() == atom_name.upper():
            return float(m.group(6))
    return None

def decode_occupacy(sof_encoded, fvar_values):
    sign = -1 if sof_encoded < 0 else 1
    absval = abs(sof_encoded)
    m = int(absval // 10)
    p = absval - 10 * m
    if m == 1:
        return p
    fvar_value = fvar_values[m - 1]
    if sign < 0:
        return 1 - p * fvar_value
    return p * fvar_value

def get_occupancy(cif_path, atom_name, data_name=None):
    try:
        lines = get_res_text(cif_path, data_name)
    except Exception as e:
        return None, f"blad odczytu pliku: {e}"
    if lines is None:
        return None, "brak _shelx_res_file w tym pliku"
    fvar_values = parse_fvar(lines)
    sof = find_atom_sof(lines, atom_name)
    if sof is None:
        return None, "atom nie znaleziony"
    try:
        return decode_occupacy(sof, fvar_values), None
    except (IndexError, ZeroDivisionError) as e:
        return None, f"blad dekodowania OSF: {e}"

SHELX_NON_ATOM_KEYWORDS = {
    "ZERR", "CELL", "LATT", "SYMM", "SFAC", "UNIT", "TEMP", "SIZE", "FVAR", "HKLF",
    "OMIT", "SHEL", "MPLA", "HFIX", "AFIX", "PART", "SAME", "SADI", "DELU", "SIMU",
    "ISOR", "DFIX", "DANG", "FLAT", "CHIV", "EADP", "EXYZ", "EXTI", "SWAT", "TWIN",
    "BASF", "SUMP", "BUMP", "SPEC", "RESI", "MOVE", "ANIS", "DISP", "RIGU", "WGHT",
    "BOND", "CONF", "EQIV", "LIST", "TITL", "L.S.", "ACTA", "PLAN", "FMAP", "MORE",
    "TIME", "REM", "END",
}
    
def get_available_atom_names(cif_path, data_name=None):
    try:
        lines = get_res_text(cif_path, data_name)
    except Exception:
        return[]
    if lines is None:
        return []

    names = []
    for raw in lines:
        line = raw.rstrip("\r\n").strip()
        m = ATOM_LINE_RE.match(line)
        if m:
            label = m.group(1)
            if label.upper() in SHELX_NON_ATOM_KEYWORDS:
                continue
            names.append(label)
    return names

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


SHELX_SYMBOLS = {
    r'\a': 'α', r'\b': 'β', r'\g': 'γ', r'\d': 'δ',
    r'\q': 'θ', r'\S': 'Σ', r'\%': '%',
}


def decode_shelx_symbols(s):
    for code, symbol in SHELX_SYMBOLS.items():
        s = s.replace(code, symbol)
    return s


def extract_semicolon_block(lines, tag):
    for i, line in enumerate(lines):
        if line.rstrip("\r\n").strip() == tag:
            j = i + 1
            if j < len(lines) and lines[j].rstrip("\r\n").strip() == ";":
                block = []
                k = j + 1
                while k < len(lines) and lines[k].rstrip("\r\n").strip() != ";":
                    block.append(lines[k].rstrip("\r\n"))
                    k += 1
                return block
    return []


RUN_ROW_RE = re.compile(r'^\s*\d+\s+\S+')


def compute_total_exposure_time(lines):
    block = extract_semicolon_block(lines, "_diffrn_measurement_details")
    if not block:
        return None

    total = 0.0
    found_any = False
    for raw_line in block:
        if not RUN_ROW_RE.match(raw_line):
            continue
        tokens = raw_line.split()
        if "--" in tokens:
            idx = tokens.index("--")
            texp_token = tokens[idx - 1]
        elif len(tokens) > 5:
            texp_token = tokens[5]
        else:
            continue
        frames_token = tokens[-1]
        try:
            texp = float(texp_token)
            frames = int(frames_token)
        except ValueError:
            continue
        total += texp * frames
        found_any = True

    return total if found_any else None


def split_cif_into_blocks(lines):
    blocks = []
    current_data_name = None
    current_block = []

    for line in lines:
        if line.rstrip("\r\n").strip().startswith("data_"):
            if current_data_name is not None:
                blocks.append((current_data_name, current_block))
            current_data_name = line.rstrip("\r\n").strip()[5:]
            current_block = [line]
        else:
            if current_data_name is not None:
                current_block.append(line)

    if current_data_name is not None:
        blocks.append((current_data_name, current_block))

    return blocks


def parse_cif_block(block_lines, data_name):
    data = {}
    data["_identification_code"] = data_name

    for line in block_lines:
        line = line.rstrip("\r\n")
        if line.startswith("_"):
            parts = line.split(None, 1)
            tag = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if rest == "":
                continue
            rest = rest.strip("'").strip('"')
            data[tag] = decode_shelx_symbols(rest)

    total_exposure = compute_total_exposure_time(block_lines)
    if total_exposure is not None:
        data["_computed_total_exposure_time"] = f"{total_exposure:.2f}"

    return data


def parse_cif(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    blocks = split_cif_into_blocks(lines)
    all_data = [parse_cif_block(block_lines, data_name) for data_name, block_lines in blocks]
    return all_data


def get(d, tag, default="?"):
    return d.get(tag.lower(), default)


def round2(value_str):
    try:
        return f"{float(value_str):.2f}"
    except (TypeError, ValueError):
        return value_str


def crystal_size(d):
    size_max = round2(get(d, '_exptl_crystal_size_max'))
    size_mid = round2(get(d, '_exptl_crystal_size_mid'))
    size_min = round2(get(d, '_exptl_crystal_size_min'))
    return f"{size_max} × {size_mid} × {size_min}"


def radiation(d):
    return f"{get(d, '_diffrn_radiation_type')} (λ = {get(d, '_diffrn_radiation_wavelength')} Å)"


def theta_range(d):
    try:
        t_min = float(get(d, "_diffrn_reflns_theta_min"))
        t_max = float(get(d, "_diffrn_reflns_theta_max"))
        return f"{2 * t_min:.3f} to {2 * t_max:.3f}"
    except (TypeError, ValueError):
        return "?"


def index_ranges(d):
    return (
        f"-{get(d, '_diffrn_reflns_limit_h_min')} <= h <= {get(d, '_diffrn_reflns_limit_h_max')}\n"
        f"{get(d, '_diffrn_reflns_limit_k_min')} <= k <= {get(d, '_diffrn_reflns_limit_k_max')}\n"
        f"{get(d, '_diffrn_reflns_limit_l_min')} <= l <= {get(d, '_diffrn_reflns_limit_l_max')}"
    ).replace("--", "-")


TEMPERATURE_RE = re.compile(r'^(-?\d+(?:\.\d+)?)(\(.*\))?$')


def round_temperature(value_str):
    m = TEMPERATURE_RE.match(value_str)
    if not m:
        return value_str
    main = round(float(m.group(1)))
    return f"{main}"


def independent_reflns(d):
    return (
        f"{get(d, '_reflns_number_total')} "
        f"[R~int~ = {get(d, '_diffrn_reflns_av_R_equivalents')}, "
        f"R~sigma~ = {get(d, '_diffrn_reflns_av_unetI/netI')}]"
    )


def data_restraints_params(d):
    return f"{get(d, '_refine_ls_number_reflns')}/{get(d, '_refine_ls_number_restraints')}/{get(d, '_refine_ls_number_parameters')}"


def final_r_gt(d):
    return f"R~1~ = {get(d, '_refine_ls_R_factor_gt')}, wR~2~ = {get(d, '_refine_ls_wR_factor_gt')}"


def final_r_all(d):
    return f"R~1~ = {get(d, '_refine_ls_R_factor_all')}, wR~2~ = {get(d, '_refine_ls_wR_factor_ref')}"


def diff_peak_hole(d):
    return f"{round2(get(d, '_refine_diff_density_max'))}/{round2(get(d, '_refine_diff_density_min'))}"


def flack_parameter(d):
    return get(d, "_refine_ls_abs_structure_flack", default="")


def has_any_flack(all_data):
    return any(flack_parameter(d).strip() not in ("", "?") for d in all_data)


def exposure_time_row(d):
    val = get(d, "_computed_total_exposure_time", default="")
    if val in ("", "?"):
        return "?"
    return f"{float(val) / 60:.2f}"


def cumulative_exposure_time_row(d):
    return get(d, "_computed_cumulative_exposure_time", default="?")


# Etykiety wierszy z markerami "^N" -> indeks GORNY, "~tekst~" -> indeks DOLNY
ROWS = [
    ("Identification code", lambda d: get(d, "_identification_code")),
    ("Empirical formula", lambda d: get(d, "_chemical_formula_sum").replace(" ", "")),  # cyfry -> subscript automatycznie
    ("Formula weight/g mol^-1", lambda d: get(d, "_chemical_formula_weight")),
    ("Temperature/K", lambda d: round_temperature(get(d, "_cell_measurement_temperature"))),
    ("Crystal system", lambda d: get(d, "_space_group_crystal_system")),
    ("Space group", lambda d: get(d, "_space_group_name_H-M_alt")),
    ("a/Å", lambda d: get(d, "_cell_length_a")),
    ("b/Å", lambda d: get(d, "_cell_length_b")),
    ("c/Å", lambda d: get(d, "_cell_length_c")),
    ("α/°", lambda d: get(d, "_cell_angle_alpha")),
    ("β/°", lambda d: get(d, "_cell_angle_beta")),
    ("γ/°", lambda d: get(d, "_cell_angle_gamma")),
    ("Volume/Å^3", lambda d: get(d, "_cell_volume")),
    ("Z", lambda d: get(d, "_cell_formula_units_Z")),
    ("ρ~calc~/g cm^-3", lambda d: get(d, "_exptl_crystal_density_diffrn")),
    ("μ/mm^-1", lambda d: get(d, "_exptl_absorpt_coefficient_mu")),
    ("F(000)", lambda d: get(d, "_exptl_crystal_F_000")),
    ("Crystal size/mm^3", crystal_size),
    ("Radiation", radiation),
    ("2Θ range for data collection/°", theta_range),
    ("Index ranges", index_ranges),
    ("Reflections collected", lambda d: get(d, "_diffrn_reflns_number")),
    ("Independent reflections", independent_reflns),
    ("Data/restraints/parameters", data_restraints_params),
    ("Goodness-of-fit on F^2", lambda d: get(d, "_refine_ls_goodness_of_fit_ref")),
    ("Final R indexes [I>=2σ (I)]", final_r_gt),
    ("Final R indexes [all data]", final_r_all),
    ("Largest diff. peak/hole / e Å^-3", diff_peak_hole),
    ("Exp exposure time / minutes", exposure_time_row),
    ("Total exposure time / minutes", cumulative_exposure_time_row),  # <- NOWY WIERSZ
    ("Flack parameter", flack_parameter),
]

SUBSCRIPT_DIGIT_ROWS = {"Empirical formula"}
ITALIC_SPACE_GROUP_ROWS = {"Space group"}

TOKEN_RE = re.compile(r'\^(-?\d+)|~([^~]+)~')


def html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_space_group(text):
    parts = ['<i>']
    for chunk in re.split(r'(\d+)', text):
        if chunk == "":
            continue
        if chunk.isdigit():
            if len(chunk) == 1:
                parts.append(html_escape(chunk))
            else:
                parts.append(html_escape(chunk[0]))
                parts.append(f"<sub>{html_escape(chunk[1:])}</sub>")
        else:
            parts.append(html_escape(chunk))
    parts.append('</i>')
    return "".join(parts)


def cell_html(text, subscript_digits=False, italic_space_group=False):
    if italic_space_group:
        return "<br>".join(format_space_group(line) for line in text.split("\n"))

    lines_out = []
    for line in text.split("\n"):
        if subscript_digits:
            parts = []
            for chunk in re.split(r'(\d+)', line):
                if chunk == "":
                    continue
                if chunk.isdigit():
                    parts.append(f"<sub>{chunk}</sub>")
                else:
                    parts.append(html_escape(chunk))
            lines_out.append("".join(parts))
        else:
            parts = []
            pos = 0
            for m in TOKEN_RE.finditer(line):
                if m.start() > pos:
                    parts.append(html_escape(line[pos:m.start()]))
                if m.group(1) is not None:
                    parts.append(f"<sup>{m.group(1)}</sup>")
                else:
                    parts.append(f"<sub>{html_escape(m.group(2))}</sub>")
                pos = m.end()
            if pos < len(line):
                parts.append(html_escape(line[pos:]))
            lines_out.append("".join(parts))
    return "<br>".join(lines_out)


TABLE_STYLE = "border-collapse:collapse;font-family:Calibri,Arial,sans-serif;font-size:13px;"
TH_STYLE = ("border:1px solid #444;padding:5px 9px;text-align:left;"
            "white-space:nowrap;background-color:#D9E2F3;font-weight:bold;")
TD_STYLE = "border:1px solid #444;padding:5px 9px;text-align:left;white-space:nowrap;"
TD_FIRST_STYLE = TD_STYLE + "font-weight:bold;background-color:#F7F9FC;"


def build_matrix(all_data, file_names, selected_labels):
    row_labels_html = []
    cell_matrix = []

    for label, fn in ROWS:
        if label not in selected_labels:
            continue

        row_labels_html.append(cell_html(label))
        row = []
        for d in all_data:
            if label == "Flack parameter":
                raw = flack_parameter(d)
                value = raw if raw.strip() not in ("", "?") else "N/a"
            else:
                value = fn(d)
            row.append(cell_html(
                value,
                subscript_digits=(label in SUBSCRIPT_DIGIT_ROWS),
                italic_space_group=(label in ITALIC_SPACE_GROUP_ROWS),
            ))
        cell_matrix.append(row)

    return row_labels_html, cell_matrix

def build_occupancy_rows(atom_names, all_structures, path_by_name):
    row_labels_html = []
    cell_matrix = []

    for atom_name in atom_names:
        row_labels_html.append(cell_html(f"Occupancy {atom_name}"))
        row = []
        for file_name, data_name, _ , _ in all_structures:
            file_path = path_by_name.get(file_name)
            if file_path is None:
                row.append(cell_html("?"))
                continue
            occ, err = get_occupancy(file_path, atom_name, data_name)
            if occ is None:
                row.append(cell_html("?"))
            else:
                row.append(cell_html(f"{occ:.5f}"))
        cell_matrix.append(row)

    return row_labels_html, cell_matrix


def render_table_html(row_labels_html, file_names, cell_matrix, orientation, heading):
    file_headers_html = [html_escape(n) for n in file_names]

    html = [f'<h4>{html_escape(heading)}</h4>']
    html.append(f'<div style="overflow-x:auto;"><table style="{TABLE_STYLE}">')

    if orientation == "files_cols":
        html.append("<tr>")
        html.append(f'<th style="{TH_STYLE}">Parametr</th>')
        for h in file_headers_html:
            html.append(f'<th style="{TH_STYLE}">{h}</th>')
        html.append("</tr>")

        for row_label, row in zip(row_labels_html, cell_matrix):
            html.append("<tr>")
            html.append(f'<td style="{TD_FIRST_STYLE}">{row_label}</td>')
            for cell in row:
                html.append(f'<td style="{TD_STYLE}">{cell}</td>')
            html.append("</tr>")
    else:
        html.append("<tr>")
        html.append(f'<th style="{TH_STYLE}">Plik</th>')
        for row_label in row_labels_html:
            html.append(f'<th style="{TH_STYLE}">{row_label}</th>')
        html.append("</tr>")

        for j, file_header in enumerate(file_headers_html):
            html.append("<tr>")
            html.append(f'<td style="{TD_FIRST_STYLE}">{file_header}</td>')
            for row in cell_matrix:
                html.append(f'<td style="{TD_STYLE}">{row[j]}</td>')
            html.append("</tr>")

    html.append("</table></div>")
    return "\n".join(html)

def get_shared_cif_paths():
    if "shared_uploaded_bytes" not in st.session_state:
        st.session_state.shared_uploaded_bytes = {}
    if "shared_folder_paths" not in st.session_state:
        st.session_state.shared_folder_paths = []

    st.subheader("Wybierz pliki CIF")

    loaded_names = list(st.session_state.shared_uploaded_bytes.keys()) + \
        [p.name for p in st.session_state.shared_folder_paths]
    if loaded_names:
        st.caption(f"Aktualnie wczytane pliki (wspolne dla wszystkich stron, {len(loaded_names)}): "
                    f"{', '.join(sorted(loaded_names, key=natural_key))}")
        if st.button("Wyczysc wszystkie wczytane pliki", key="shared_clear_btn"):
            st.session_state.shared_uploaded_bytes = {}
            st.session_state.shared_folder_paths = []
            st.rerun()
    input_mode = st.radio(
        "Dodaj kolejne pliki",
        ["Dodaj pojedyncze pliki", "Podaj sciezke do folderu"],
        horizontal=True,
        key="shared_input_mode",
    )

    if input_mode == "Dodaj pojedyncze pliki":
        uploaded = st.file_uploader(
            "Wybierz plik(i) CIF", type=["cif"], accept_multiple_files=True,
            key="shared_uploader",
        )
        if uploaded:
            for uf in uploaded:
                st.session_state.shared_uploaded_bytes[uf.name] = uf.getvalue()
    else:
        folder_str = st.text_input("Sciezka do folderu z plikami .cif", value="",
                                   key="shared_folder_input")
        if folder_str:
            folder = Path(folder_str)
            if folder.is_dir():
                found =  sorted(folder.glob("*.cif"), key = lambda f: natural_key(f.name))
                if not found:
                    st.warning("W tym folderze nie znaleziono zadnych plikow .cif")
                else:
                    st.session_state.shared_folder_paths = found
            else:
                st.error("Podana sciezka nie istnieje albo nie jest folderem")

    cif_paths = []
    if st.session_state.shared_uploaded_bytes:
        tmp_dir = Path(tempfile.mkdtemp(prefix="cif_shared_"))
        for fname, data in st.session_state.shared_uploaded_bytes.items():
            p = tmp_dir / fname
            p.write_bytes(data)
            cif_paths.append(p)
    cif_paths += list(st.session_state.shared_folder_paths)
    return sorted(cif_paths, key=lambda f: natural_key(f.name))



def load_cif_paths_from_uploads(uploaded_files):
    if not uploaded_files:
        return []
    tmp_dir = Path(tempfile.mkdtemp(prefix="cif_upload_"))
    paths = []
    for uf in uploaded_files:
        p = tmp_dir / uf.name
        p.write_bytes(uf.getvalue())
        paths.append(p)
    return paths


def build_all_structures(cif_paths):
    all_structures = []
    for file_path in cif_paths:
        try:
            file_structures = parse_cif(file_path)
        except Exception as e:
            st.warning(f"Nie udało sie wczytać {file_path.name}: {e}")
            continue
        for struct_index, data_dict in enumerate(file_structures):
            data_name = data_dict.get("_identification_code", "unknown")
            all_structures.append((file_path.name, data_name, data_dict, struct_index))

    all_structures.sort(key=lambda x: (natural_key(x[0]), x[3]))
    return all_structures


def page_multi_report():
    st.title("Multi report")
    
    cif_paths = get_shared_cif_paths()

    if not cif_paths:
        st.info("Wgraj pliki albo podaj folder, żeby zobaczyc tabele")
        return

    all_structures = build_all_structures(cif_paths)
    if not all_structures:
        st.warning("Nie udało się wczytać zadnej struktury z podanych plików")
        return

    structure_count_per_file = {}
    for file_name, _, _, _ in all_structures:
        structure_count_per_file[file_name] = structure_count_per_file.get(file_name, 0) + 1

    all_files_data = [item[2] for item in all_structures]
    all_file_names = []
    for file_name, data_name, _, _ in all_structures:
        if structure_count_per_file[file_name] == 1:
            all_file_names.append(file_name)
        else:
            all_file_names.append(f"{file_name}:{data_name}")

    running_total = 0.0
    for d in all_files_data:
        val = get(d, "_computed_total_exposure_time", default="")
        if val not in ("", "?"):
            running_total += float(val) / 60.0
        d["_computed_cumulative_exposure_time"] = f"{running_total:.2f}"

    st.success(f"Wczytano {len(all_structures)} struktur(y) z {len(cif_paths)} pliku(ów)")

    st.subheader("Ustawienia tabeli")

    col1, col2 = st.columns(2)
    with col1:
        orientation_label = st.radio(
            "Sposob prezentacji tabeli",
            ["Pliki w kolumnach, parametry w wierszach",
             "Parametry w kolumnach, pliki w wierszach"],
        )
        orientation = "files_cols" if orientation_label.startswith("Pliki") else "params_cols"

    all_labels = [label for label, _ in ROWS]
    with col2:
        if "selected_params" not in st.session_state:
            st.session_state.selected_params = []

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Zaznacz wszystkie", use_container_width=True):
                st.session_state.selected_params = list(all_labels)
        with btn_col2:
            if st.button("Wyczyść", use_container_width=True):
                st.session_state.selected_params = []

        selected_labels = st.multiselect(
            "Parametry do wyswietlenia w tabeli",
            options=all_labels,
            key="selected_params",
        )

    if not selected_labels:
        st.info("Wybierz przynajmniej jeden parametr albo kliknij 'Zaznacz wszystkie'")
        return

    max_columns = st.number_input(
        "Maksymalna liczba plikow w jednej tabeli (przy wiekszej liczbie - podzial na czesci)",
        min_value=1, max_value=50, value=DEFAULT_MAX_COLUMNS, step=1,
    )

    st.subheader("Tabela")
    row_labels_html, cell_matrix = build_matrix(all_files_data, all_file_names, selected_labels)

    if orientation == "files_cols" and len(all_file_names) > max_columns:
        chunks = []
        for start in range(0, len(all_file_names), max_columns):
            end = start + max_columns
            chunk_names = all_file_names[start:end]
            chunk_matrix = [row[start:end] for row in cell_matrix]
            chunks.append((chunk_names, chunk_matrix, start))
        total_parts = len(chunks)
    else:
        chunks = [(all_file_names, cell_matrix, 0)]
        total_parts = 1

    html_parts = []
    for part_index, (chunk_names, chunk_matrix, start) in enumerate(chunks, start=1):
        if total_parts > 1:
            first_no, last_no = start + 1, start + len(chunk_names)
            part_label = f"czesc {part_index} z {total_parts} (pliki {first_no}-{last_no} z {len(all_file_names)})"
            st.caption(part_label)
            heading = f"Crystal data and structure refinement - {part_label}"
        else:
            heading = "Crystal data and structure refinement"

        table_html = render_table_html(row_labels_html, chunk_names, chunk_matrix, orientation, heading=heading)
        st.markdown(table_html, unsafe_allow_html=True)

        file_name = f"cif_table_part{part_index}.html" if total_parts > 1 else "cif_table.html"
        full_doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{table_html}</body></html>"
        html_parts.append((file_name, full_doc))

    if total_parts == 1:
        st.download_button(
            "Pobierz jako plik HTML",
            data=html_parts[0][1],
            file_name=html_parts[0][0],
            mime="text/html",
        )
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for file_name, content in html_parts:
                zf.writestr(file_name, content)
        st.download_button(
            f"Pobierz wszystkie {total_parts} części jako ZIP",
            data=zip_buffer.getvalue(),
            file_name="cif_table_parts.zip",
            mime="application/zip",
        )

# ==================== GEOMETRIA (z super_cell_v132.py) ====================

def get_base_site(structure, name):
    """Znajduje atom o podanej etykiecie WPROST w asymetrycznej czesci (structure.sites)."""
    for site in structure.sites:
        if site.label.upper() == name.upper():
            return site
    raise KeyError(f"Atom '{name}' nie znaleziony w strukturze")


def get_lattice_vectors(structure):
    def vec(x, y, z):
        p = structure.cell.orthogonalize(gemmi.Fractional(x, y, z))
        return np.array([p.x, p.y, p.z])
    return vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)


def transformed_position(structure, name, symcode):
    site = get_base_site(structure, name)
    op = gemmi.Op(symcode)
    x, y, z = op.apply_to_xyz([site.fract.x, site.fract.y, site.fract.z])
    pos = structure.cell.orthogonalize(gemmi.Fractional(x, y, z))
    return np.array([pos.x, pos.y, pos.z])


def gemmi_position(structure, name, symcode):
    return gemmi.Position(*transformed_position(structure, name, symcode))


def symmetry_code_from_op(op, shift):
    i, j, k = shift
    return op.translated([i * op.DEN, j * op.DEN, k * op.DEN]).triplet()


def distance(structure, name1, symcode1, name2, symcode2):
    r1 = transformed_position(structure, name1, symcode1)
    r2 = transformed_position(structure, name2, symcode2)
    return abs(float(np.linalg.norm(r1 - r2)))


def angle(structure, name1, symcode1, name2, symcode2, name3, symcode3):
    p1 = gemmi_position(structure, name1, symcode1)
    p2 = gemmi_position(structure, name2, symcode2)
    p3 = gemmi_position(structure, name3, symcode3)
    return np.degrees(gemmi.calculate_angle(p1, p2, p3))


def torsion_angle(structure, name1, symcode1, name2, symcode2, name3, symcode3, name4, symcode4):
    p1 = gemmi_position(structure, name1, symcode1)
    p2 = gemmi_position(structure, name2, symcode2)
    p3 = gemmi_position(structure, name3, symcode3)
    p4 = gemmi_position(structure, name4, symcode4)
    return np.degrees(gemmi.calculate_dihedral(p1, p2, p3, p4))


def make_plane_from_coords(coords):
    atoms = []
    for c in coords:
        a = gemmi.Atom()
        a.pos = gemmi.Position(*c)
        atoms.append(a)
    return gemmi.find_best_plane(atoms)


def plane_from_atoms(structure, names, symcodes):
    coords = [transformed_position(structure, n, s) for n, s in zip(names, symcodes)]
    return make_plane_from_coords(coords)


def distance_to_plane(structure, name, symcode, plane_coeff):
    pos = gemmi_position(structure, name, symcode)
    return abs(gemmi.get_distance_from_plane(pos, plane_coeff))


def centroid(structure, names, symcodes):
    coords = np.array([transformed_position(structure, n, s) for n, s in zip(names, symcodes)])
    return coords.mean(axis=0)


def centroid_to_plane_distance(structure, names1, symcodes1, names2, symcodes2):
    c = centroid(structure, names1, symcodes1)
    plane = plane_from_atoms(structure, names2, symcodes2)
    return abs(gemmi.get_distance_from_plane(gemmi.Position(*c), plane))


def distance_between_centroids(structure, names1, symcodes1, names2, symcodes2):
    c1 = centroid(structure, names1, symcodes1)
    c2 = centroid(structure, names2, symcodes2)
    return abs(float(np.linalg.norm(c1 - c2)))


def poly_volume(structure, names, symcodes):
    coords = np.array([transformed_position(structure, n, s) for n, s in zip(names, symcodes)])
    hull = ConvexHull(coords)
    return hull.volume


def find_best_symcode(structure, name, ref_name, ref_symcode="x,y,z", search_range=1):
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    ops = list(sg.operations())
    ref_pos = transformed_position(structure, ref_name, ref_symcode)

    best_code, best_dist = None, None
    r = range(-search_range, search_range + 1)
    for op in ops:
        for i in r:
            for j in r:
                for k in r:
                    code = symmetry_code_from_op(op, (i, j, k))
                    pos = transformed_position(structure, name, code)
                    d = float(np.linalg.norm(pos - ref_pos))
                    if best_dist is None or d < best_dist:
                        best_code, best_dist = code, d
    return best_code, best_dist


def poly_volume_auto(structure, central_name, central_symcode, vertex_names, search_range=1):
    names = [central_name]
    symcodes = [central_symcode]
    for name in vertex_names:
        code, _ = find_best_symcode(structure, name, central_name, central_symcode,
                                     search_range=search_range)
        names.append(name)
        symcodes.append(code)
    return poly_volume(structure, names, symcodes)


def vdw_radius(element):
    return gemmi.Element(element).vdw_r


def find_weak_contacts(structure, name1, symcode1, name2, symcode2,
                        search_element, dist_range=(3, 4), angle_range=(110, 180),
                        tolerance=0.2, search_range=1):
    ref_pos = transformed_position(structure, name1, symcode1)
    r1 = vdw_radius(get_base_site(structure, name1).element.name)

    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    ops = list(sg.operations())
    matching_sites = [s for s in structure.sites if s.element.name.upper() == search_element.upper()]

    dist_min, dist_max = dist_range
    angle_min, angle_max = angle_range
    rng = range(-search_range, search_range + 1)

    results = []
    for site in matching_sites:
        r2 = vdw_radius(site.element.name)
        vdw_sum = r1 + r2

        for op in ops:
            for i in rng:
                for j in rng:
                    for k in rng:
                        code = symmetry_code_from_op(op, (i, j, k))
                        if site.label.upper() == name1.upper() and code == symcode1:
                            continue

                        pos = transformed_position(structure, site.label, code)
                        d = float(np.linalg.norm(pos - ref_pos))

                        if not (dist_min <= d <= dist_max):
                            continue
                        if not (vdw_sum - tolerance <= d <= vdw_sum + tolerance):
                            continue

                        ang = angle(structure, name1, symcode1, name2, symcode2, site.label, code)
                        if not (angle_min <= ang <= angle_max):
                            continue

                        results.append({"name": site.label, "symcode": code, "distance": d,
                                         "angle": ang, "vdw_sum": vdw_sum})

    results.sort(key=lambda r: r["distance"])
    return results


def find_n_shortest_distances(structure, central_name, central_symcode, search_target,
                               n=6, search_range=2, by_element=False):
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    ops = list(sg.operations())

    if by_element:
        matching_sites = [s for s in structure.sites if s.element.name.upper() == search_target.upper()]
    else:
        matching_sites = [s for s in structure.sites if s.label.upper() == search_target.upper()]

    ref_pos = transformed_position(structure, central_name, central_symcode)
    rng = range(-search_range, search_range + 1)

    results = []
    for site in matching_sites:
        for op in ops:
            for i in rng:
                for j in rng:
                    for k in rng:
                        code = symmetry_code_from_op(op, (i, j, k))
                        if site.label.upper() == central_name.upper() and code == central_symcode:
                            continue
                        pos = transformed_position(structure, site.label, code)
                        d = float(np.linalg.norm(pos - ref_pos))
                        results.append({"name": site.label, "symcode": code, "distance": d})

    results.sort(key=lambda r: r["distance"])
    return results[:n]


def pi_pi_stacking(structure, names1, symcodes1, names2=None, symcodes2=None, dist_range=(2.9, 4.1)):
    """names1/symcodes1, names2/symcodes2 - rownolegle listy (atom, symcode) - PER ATOM,
    tak jak przechowuje to teraz nazwana grupa (zamiast jednego wspolnego symcode1/symcode2
    na caly pierscien) - bo skoro symcode juz definiuje sie przy budowaniu grupy, nie ma
    sensu wymagac go jeszcze raz osobno tutaj."""
    if names2 is None:
        names2 = names1
        symcodes2 = symcodes1
    coords1 = np.array([transformed_position(structure, n, s) for n, s in zip(names1, symcodes1)])
    coords2 = np.array([transformed_position(structure, n, s) for n, s in zip(names2, symcodes2)])

    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)
    centroid_dist = float(np.linalg.norm(centroid1 - centroid2))

    if dist_range is not None:
        dmin, dmax = dist_range
        if not (dmin <= centroid_dist <= dmax):
            return "Poza zakresem"

    plane2 = make_plane_from_coords(coords2)
    a = gemmi.get_distance_from_plane(gemmi.Position(*centroid1), plane2)
    offset = np.sqrt(max(centroid_dist ** 2 - a ** 2, 0.0))
    ratio = min(abs(a) / centroid_dist, 1.0) if centroid_dist > 0 else 1.0
    twist_angle = float(np.degrees(np.arccos(ratio)))
    return {
        "centroid_to_centroid": centroid_dist,
        "centroid1_to_plane2": abs(a),
        "offset": float(offset),
        "twist_angle": twist_angle,
    }


def list_symmetry_operations_data(structure):
    """Jak list_symmetry_operations() z oryginalu, ale BEZ print() - zwraca liste
    (indeks, symcode) do wyswietlenia w tabeli Streamlit."""
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    ops = sg.operations()
    return [(idx, op.triplet()) for idx, op in enumerate(ops, start=1)]


def get_summary_dict(structure, cif_path):
    """Jak print_summary() z oryginalu, ale BEZ print() - zwraca slownik do wyswietlenia."""
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    def get_tag(block, name, default="-"):
        v = block.find_value(name)
        return v.strip("'\"") if v is not None else default

    def strip_esd(value_str):
        return value_str if value_str == "-" else value_str.split("(")[0]

    temperature = get_tag(block, "_cell_measurement_temperature")
    if temperature == "-":
        temperature = get_tag(block, "_diffrn_ambient_temperature")
    temperature = strip_esd(temperature)
    volume = strip_esd(get_tag(block, "_cell_volume"))
    exposure_time = compute_total_exposure_time_seconds(block)

    return {
        "a (Å)": f"{structure.cell.a:.4f}", "b (Å)": f"{structure.cell.b:.4f}", "c (Å)": f"{structure.cell.c:.4f}",
        "α (°)": f"{structure.cell.alpha:.3f}", "β (°)": f"{structure.cell.beta:.3f}",
        "γ (°)": f"{structure.cell.gamma:.3f}",
        "Volume (Å³)": volume, "Temperature (K)": temperature,
        "Total exposure time (s)": exposure_time, "Space group": structure.spacegroup_hm,
    }


def compute_total_exposure_time_seconds(block):
    """To samo co compute_exposure_time() z oryginalu, ale dziala na bloku gemmi
    (uzywane przez get_summary_dict); zwraca string sekund albo '-'."""
    text = block.find_value("_diffrn_measurement_details")
    if text is None:
        return "-"
    total, found = 0.0, False
    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\r")
        if not RUN_ROW_RE.match(line):
            continue
        tokens = line.split()
        if "--" in tokens:
            texp_token = tokens[tokens.index("--") - 1]
        elif len(tokens) > 5:
            texp_token = tokens[5]
        else:
            continue
        frames_token = tokens[-1]
        try:
            texp, frames = float(texp_token), int(frames_token)
        except ValueError:
            continue
        total += texp * frames
        found = True
    return f"{total:.2f}" if found else "-"


# ==================== ODCHYLENIA STANDARDOWE (esd) - z super_cell_v132.py ====================

ESD_RE = re.compile(r'^(-?\d+\.\d+)\((\d+)\)$')


def format_with_esd(value, sigma, sig_figs=2):
    if sigma is None or sigma < 1e-8:
        return f"{value:.4f}"
    exponent = math.floor(math.log10(abs(sigma)))
    decimals = -(exponent - (sig_figs - 1))
    if decimals < 0:
        decimals = 0
    sigma_rounded = round(sigma, decimals)
    value_rounded = round(value, decimals)
    sigma_digits = int(round(sigma_rounded * 10 ** decimals))
    value_str = f"{value_rounded:.{decimals}f}"
    return f"{value_str}({sigma_digits})"


def get_esd_table(cif_path):
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    labels = list(block.find_loop("_atom_site_label"))
    xs = list(block.find_loop("_atom_site_fract_x"))
    ys = list(block.find_loop("_atom_site_fract_y"))
    zs = list(block.find_loop("_atom_site_fract_z"))

    def parse_esd(s):
        m = ESD_RE.match(s)
        if not m:
            return 0.0
        value_str, esd_str = m.groups()
        decimals = len(value_str.split(".")[1])
        return int(esd_str) * 10 ** (-decimals)

    table = {}
    for lbl, x, y, z in zip(labels, xs, ys, zs):
        table[lbl.upper()] = (parse_esd(x), parse_esd(y), parse_esd(z))
    return table


def get_cell_esd(cif_path):
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    def parse_tag(tag):
        v = block.find_value(tag)
        if v is None:
            return 0.0
        m = ESD_RE.match(v.strip("'\""))
        if not m:
            return 0.0
        value_str, esd_str = m.groups()
        decimals = len(value_str.split(".")[1])
        return int(esd_str) * 10 ** (-decimals)

    return {
        "a": parse_tag("_cell_length_a"), "b": parse_tag("_cell_length_b"), "c": parse_tag("_cell_length_c"),
        "alpha": parse_tag("_cell_angle_alpha"), "beta": parse_tag("_cell_angle_beta"),
        "gamma": parse_tag("_cell_angle_gamma"),
    }


def make_perturbed_cell(structure, param, delta):
    vals = {"a": structure.cell.a, "b": structure.cell.b, "c": structure.cell.c,
            "alpha": structure.cell.alpha, "beta": structure.cell.beta, "gamma": structure.cell.gamma}
    vals[param] += delta
    return gemmi.UnitCell(vals["a"], vals["b"], vals["c"], vals["alpha"], vals["beta"], vals["gamma"])


def _transformed_position_override(structure, name, symcode, overrides, cell=None):
    site = get_base_site(structure, name)
    x0, y0, z0 = site.fract.x, site.fract.y, site.fract.z
    key = name.upper()
    if key in overrides:
        dx, dy, dz = overrides[key]
        x0, y0, z0 = x0 + dx, y0 + dy, z0 + dz
    op = gemmi.Op(symcode)
    x, y, z = op.apply_to_xyz([x0, y0, z0])
    use_cell = cell if cell is not None else structure.cell
    pos = use_cell.orthogonalize(gemmi.Fractional(x, y, z))
    return np.array([pos.x, pos.y, pos.z])


def _has_hydrogen(structure, names):
    for name in names:
        el = get_base_site(structure, name).element.name
        if el.upper() in ("H", "D"):
            return True
    return False


CELL_PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")


def _wrap_angle_diff(diff):
    """Normalizuje roznice katow (w stopniach) do zakresu (-180, 180] - potrzebne dla
    wielkosci CYKLICZNYCH jak kat torsyjny, ktory 'zawija sie' na +-180 stopni: naiwna
    roznica np. 179.999 - (-179.999) = 359.998 jest bledna (prawdziwa zmiana geometryczna
    to tylko 0.002 stopnia w druga strone) - bez tego sigma torsji wychodziloby ogromne
    i bledne w rzadkich przypadkach, gdy prawdziwy kat siedzi bardzo blisko +-180."""
    return (diff + 180) % 360 - 180


def _propagate_esd(compute_fn, unique_names, esd_table, structure, cell_esd, eps_factor=0.01, wrap_diff=False):
    variance_sum = 0.0
    any_esd = False

    for name in unique_names:
        sx, sy, sz = esd_table.get(name.upper(), (0.0, 0.0, 0.0))
        for axis, esd in zip("xyz", (sx, sy, sz)):
            if esd == 0.0:
                continue
            any_esd = True
            delta = eps_factor * esd
            d_plus = {"x": (delta, 0.0, 0.0), "y": (0.0, delta, 0.0), "z": (0.0, 0.0, delta)}[axis]
            d_minus = {"x": (-delta, 0.0, 0.0), "y": (0.0, -delta, 0.0), "z": (0.0, 0.0, -delta)}[axis]
            val_plus = compute_fn({name.upper(): d_plus}, None)
            val_minus = compute_fn({name.upper(): d_minus}, None)
            diff = _wrap_angle_diff(val_plus - val_minus) if wrap_diff else (val_plus - val_minus)
            deriv = diff / (2 * delta)
            variance_sum += (deriv * esd) ** 2

    for pname in CELL_PARAMS:
        esd = cell_esd.get(pname, 0.0)
        if esd == 0.0:
            continue
        any_esd = True
        delta = eps_factor * esd
        cell_plus = make_perturbed_cell(structure, pname, delta)
        cell_minus = make_perturbed_cell(structure, pname, -delta)
        val_plus = compute_fn({}, cell_plus)
        val_minus = compute_fn({}, cell_minus)
        diff = _wrap_angle_diff(val_plus - val_minus) if wrap_diff else (val_plus - val_minus)
        deriv = diff / (2 * delta)
        variance_sum += (deriv * esd) ** 2

    if not any_esd:
        return None
    return float(np.sqrt(variance_sum))


def distance_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2):
    def compute(overrides, cell=None):
        r1 = _transformed_position_override(structure, name1, symcode1, overrides, cell)
        r2 = _transformed_position_override(structure, name2, symcode2, overrides, cell)
        return float(np.linalg.norm(r1 - r2))

    value = compute({}, None)
    if _has_hydrogen(structure, [name1, name2]):
        return value, None
    sigma = _propagate_esd(compute, {name1.upper(), name2.upper()}, esd_table, structure, cell_esd)
    return value, sigma


def angle_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2, name3, symcode3):
    def compute(overrides, cell=None):
        p1 = gemmi.Position(*_transformed_position_override(structure, name1, symcode1, overrides, cell))
        p2 = gemmi.Position(*_transformed_position_override(structure, name2, symcode2, overrides, cell))
        p3 = gemmi.Position(*_transformed_position_override(structure, name3, symcode3, overrides, cell))
        return np.degrees(gemmi.calculate_angle(p1, p2, p3))

    value = compute({}, None)
    if _has_hydrogen(structure, [name1, name2, name3]):
        return value, None
    sigma = _propagate_esd(compute, {name1.upper(), name2.upper(), name3.upper()}, esd_table, structure, cell_esd)
    return value, sigma


def torsion_angle_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2,
                       name3, symcode3, name4, symcode4):
    def compute(overrides, cell=None):
        p1 = gemmi.Position(*_transformed_position_override(structure, name1, symcode1, overrides, cell))
        p2 = gemmi.Position(*_transformed_position_override(structure, name2, symcode2, overrides, cell))
        p3 = gemmi.Position(*_transformed_position_override(structure, name3, symcode3, overrides, cell))
        p4 = gemmi.Position(*_transformed_position_override(structure, name4, symcode4, overrides, cell))
        return np.degrees(gemmi.calculate_dihedral(p1, p2, p3, p4))

    value = compute({}, None)
    if _has_hydrogen(structure, [name1, name2, name3, name4]):
        return value, None
    sigma = _propagate_esd(compute, {name1.upper(), name2.upper(), name3.upper(), name4.upper()},
                            esd_table, structure, cell_esd, wrap_diff=True)
    return value, sigma


def distance_to_plane_esd(structure, esd_table, cell_esd, name, symcode, plane_names, plane_symcodes):
    def compute(overrides, cell=None):
        coords = [_transformed_position_override(structure, n, s, overrides, cell)
                  for n, s in zip(plane_names, plane_symcodes)]
        plane = make_plane_from_coords(coords)
        pos = gemmi.Position(*_transformed_position_override(structure, name, symcode, overrides, cell))
        return abs(gemmi.get_distance_from_plane(pos, plane))

    value = compute({}, None)
    all_names = [name] + list(plane_names)
    if _has_hydrogen(structure, all_names):
        return value, None
    unique_names = {name.upper()} | {n.upper() for n in plane_names}
    sigma = _propagate_esd(compute, unique_names, esd_table, structure, cell_esd)
    return value, sigma


def centroid_to_plane_distance_esd(structure, esd_table, cell_esd, names1, symcodes1, names2, symcodes2):
    def compute(overrides, cell=None):
        coords1 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names1, symcodes1)]
        c = np.mean(coords1, axis=0)
        coords2 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names2, symcodes2)]
        plane = make_plane_from_coords(coords2)
        return abs(gemmi.get_distance_from_plane(gemmi.Position(*c), plane))

    value = compute({}, None)
    all_names = list(names1) + list(names2)
    if _has_hydrogen(structure, all_names):
        return value, None
    unique_names = {n.upper() for n in names1} | {n.upper() for n in names2}
    sigma = _propagate_esd(compute, unique_names, esd_table, structure, cell_esd)
    return value, sigma


def distance_between_centroids_esd(structure, esd_table, cell_esd, names1, symcodes1, names2, symcodes2):
    def compute(overrides, cell=None):
        coords1 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names1, symcodes1)]
        coords2 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names2, symcodes2)]
        c1 = np.mean(coords1, axis=0)
        c2 = np.mean(coords2, axis=0)
        return float(np.linalg.norm(c1 - c2))

    value = compute({}, None)
    all_names = list(names1) + list(names2)
    if _has_hydrogen(structure, all_names):
        return value, None
    unique_names = {n.upper() for n in names1} | {n.upper() for n in names2}
    sigma = _propagate_esd(compute, unique_names, esd_table, structure, cell_esd)
    return value, sigma


def pi_pi_stacking_esd(structure, esd_table, cell_esd, names1, symcodes1, names2=None, symcodes2=None,
                        dist_range=(2.9, 4.1)):
    """names1/symcodes1, names2/symcodes2 - rownolegle listy (atom, symcode) - PER ATOM,
    tak jak przechowuje to nazwana grupa (nie jeden wspolny symcode1/symcode2 na caly pierscien)."""
    if names2 is None:
        names2 = names1
        symcodes2 = symcodes1

    def compute_centroid_dist(overrides, cell=None):
        coords1 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names1, symcodes1)]
        coords2 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names2, symcodes2)]
        c1 = np.mean(coords1, axis=0)
        c2 = np.mean(coords2, axis=0)
        return float(np.linalg.norm(c1 - c2))

    def compute_centroid1_to_plane2(overrides, cell=None):
        coords1 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names1, symcodes1)]
        coords2 = [_transformed_position_override(structure, n, s, overrides, cell)
                   for n, s in zip(names2, symcodes2)]
        c1 = np.mean(coords1, axis=0)
        plane2 = make_plane_from_coords(coords2)
        return abs(gemmi.get_distance_from_plane(gemmi.Position(*c1), plane2))

    def compute_offset(overrides, cell=None):
        cd = compute_centroid_dist(overrides, cell)
        a_val = compute_centroid1_to_plane2(overrides, cell)
        return float(np.sqrt(max(cd ** 2 - a_val ** 2, 0.0)))

    def compute_twist_angle(overrides, cell=None):
        cd = compute_centroid_dist(overrides, cell)
        a_val = compute_centroid1_to_plane2(overrides, cell)
        ratio = min(a_val / cd, 1.0) if cd > 0 else 1.0
        return float(np.degrees(np.arccos(ratio)))

    centroid_dist = compute_centroid_dist({}, None)
    if dist_range is not None:
        dmin, dmax = dist_range
        if not (dmin <= centroid_dist <= dmax):
            return "Poza zakresem"

    a_val = compute_centroid1_to_plane2({}, None)
    offset_val = compute_offset({}, None)
    twist_val = compute_twist_angle({}, None)

    all_names = list(names1) + list(names2)
    if _has_hydrogen(structure, all_names):
        return {
            "centroid_to_centroid": (centroid_dist, None),
            "centroid1_to_plane2": (a_val, None),
            "offset": (offset_val, None),
            "twist_angle": (twist_val, None),
        }

    unique_names = {n.upper() for n in names1} | {n.upper() for n in names2}
    sigma_cd = _propagate_esd(compute_centroid_dist, unique_names, esd_table, structure, cell_esd)
    sigma_a = _propagate_esd(compute_centroid1_to_plane2, unique_names, esd_table, structure, cell_esd)
    sigma_offset = _propagate_esd(compute_offset, unique_names, esd_table, structure, cell_esd)
    sigma_twist = _propagate_esd(compute_twist_angle, unique_names, esd_table, structure, cell_esd)

    return {
        "centroid_to_centroid": (centroid_dist, sigma_cd),
        "centroid1_to_plane2": (a_val, sigma_a),
        "offset": (offset_val, sigma_offset),
        "twist_angle": (twist_val, sigma_twist),
    }


def three_atom_summary_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2,
                            name3, symcode3):
    """Dla trzech atomow (1,2,3): odleglosc 1-3, odleglosc 1-2, kat 1-2-3 (wierzcholek=2)
    - kazde z osobna sigma. Prosty 'skrot' laczacy distance_esd + angle_esd w jednym wywolaniu."""
    d13, s13 = distance_esd(structure, esd_table, cell_esd, name1, symcode1, name3, symcode3)
    d12, s12 = distance_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2)
    ang, sang = angle_esd(structure, esd_table, cell_esd, name1, symcode1, name2, symcode2,
                           name3, symcode3)
    return {
        "dist_1_3": (d13, s13),
        "dist_1_2": (d12, s12),
        "angle_1_2_3": (ang, sang),
    }


def parse_atom_list(text):
    """Parsuje wieloliniowe pole tekstowe do (names, symcodes). Kazda linijka:
    'NAZWA' (symcode domyslnie 'x,y,z') albo 'NAZWA,symcode'. Puste linijki pomijane."""
    names, symcodes = [], []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "," in line:
            name, symcode = line.split(",", 1)
            names.append(name.strip())
            symcodes.append(symcode.strip())
        else:
            names.append(line)
            symcodes.append("x,y,z")
    return names, symcodes


def atom_picker(label, key_prefix, default_name="", default_symcode="x,y,z"):
    """Dwa pola obok siebie: nazwa atomu + symcode (domyslnie 'x,y,z')."""
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(f"{label} - nazwa atomu", value=default_name, key=f"{key_prefix}_name")
    with col2:
        symcode = st.text_input(f"{label} - symcode", value=default_symcode, key=f"{key_prefix}_symcode")
    return name, symcode


def parse_semicolon_atom_list(text):
    """Parsuje JEDNA komorke tabeli (data_editor) z lista atomow oddzielonych ';'.
    Kazdy atom: 'NAZWA' (symcode domyslnie x,y,z) albo 'NAZWA,symcode'.
    Uzywane dla kolumn typu 'lista atomow' (plaszczyzny, poly_volume, pi-pi).
    Pusta komorka (None/NaN/pusty string) daje puste listy - NIE 'nan' jako atom."""
    if is_empty_cell(text):
        return [], []
    names, symcodes = [], []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            name, symcode = part.split(",", 1)
            names.append(name.strip())
            symcodes.append(symcode.strip())
        else:
            names.append(part)
            symcodes.append("x,y,z")
    return names, symcodes


def build_query_rows(query_specs, compute_fn, all_structures, path_by_name):
    """query_specs: lista (etykieta_wiersza, args) - args rozpakowywane do compute_fn.
    compute_fn(structure, esd_table, cell_esd, *args) -> (wartosc, sigma) - ZAWSZE
    wersja z sigma (odchylenie standardowe liczy sie zawsze, nie jest opcja).
    Liczy dla KAZDEGO zapytania x KAZDEGO wczytanego pliku naraz.
    Zwraca (row_labels_html, cell_matrix) w formacie build_matrix/build_occupancy_rows,
    gotowym do wrzucenia w render_table_html (ta sama tabela HTML co w Multi report)."""
    row_labels_html = []
    cell_matrix = []
    cache = {}  # file_name -> (structure, esd_table, cell_esd) - liczone RAZ na plik

    for row_label, args in query_specs:
        row_labels_html.append(cell_html(row_label))
        row = []
        for file_name, data_name, _, _ in all_structures:
            file_path = path_by_name.get(file_name)
            if file_path is None:
                row.append(cell_html("?"))
                continue
            if file_name not in cache:
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    et = get_esd_table(str(file_path))
                    ce = get_cell_esd(str(file_path))
                    cache[file_name] = (s, et, ce)
                except Exception:
                    cache[file_name] = None
            entry = cache[file_name]
            if entry is None:
                row.append(cell_html("błąd wczytania"))
                continue
            s, et, ce = entry
            try:
                val, sigma = compute_fn(s, et, ce, *args)
                row.append(cell_html(format_with_esd(val, sigma)))
            except Exception as e:
                row.append(cell_html(f"błąd: {e}"))
        cell_matrix.append(row)
    return row_labels_html, cell_matrix


def show_immediate_preview(row_labels_html, all_file_names, cell_matrix):
    """Pokazuje wynik OD RAZU pod przyciskiem (oprocz dopisania do wspolnej tabeli
    na dole). To znika przy nastepnym odswiezeniu (bo jest w if st.button()) - to
    normalne, trwaly zapis wyniku jest w sekcji 'Wyniki' na dole (patrz append_results)."""
    if not row_labels_html:
        return
    table_html = render_table_html(row_labels_html, all_file_names, cell_matrix, "files_cols",
                                    heading="Podglad wyniku")
    st.markdown(table_html, unsafe_allow_html=True)


def append_results(section_label, row_labels_html, cell_matrix, all_file_names):
    """Dopisuje wyniki DOWOLNEJ funkcji do WSPOLNEJ, trwalej listy w session_state
    (zamiast pokazywac osobna tabele od razu pod przyciskiem - ta by znikala przy
    kazdym innym kliknieciu, bo st.button() jest True tylko w JEDNYM odswiezeniu).
    Jesli zestaw plikow zmienil sie od ostatniego razu, czysci stare wyniki
    (zeby kolumny sie nie rozjechaly) i informuje o tym.
    section_label - dopisywany jako prefiks etykiety wiersza, zeby bylo widac
    z jakiej funkcji pochodzi kazdy wiersz we wspolnej tabeli."""
    if "sc_results_files" not in st.session_state:
        st.session_state.sc_results_files = list(all_file_names)
        st.session_state.sc_results_rows = []

    if st.session_state.sc_results_files != list(all_file_names):
        st.session_state.sc_results_files = list(all_file_names)
        st.session_state.sc_results_rows = []
        st.warning("Zestaw plikow sie zmienil - wyczyszczono poprzednie wyniki (zeby kolumny sie nie rozjechaly).")

    for label, row in zip(row_labels_html, cell_matrix):
        prefixed_label = f"{html_escape(section_label)}: {label}"
        st.session_state.sc_results_rows.append((prefixed_label, row))


def render_accumulated_results():
    """Jedna wspolna tabela na dole strony - WSZYSTKIE wyniki ze WSZYSTKICH funkcji
    naraz, poza jakimkolwiek if st.button(), wiec przelaczanie orientacji faktycznie
    dziala (nie znika po kliknieciu)."""
    st.header("Wyniki")

    rows = st.session_state.get("sc_results_rows", [])
    file_names = st.session_state.get("sc_results_files", [])

    if not rows:
        st.info("Brak wynikow - policz cokolwiek powyzej, pojawi sie tutaj.")
        return

    if st.button("Wyczysc wszystkie wyniki"):
        st.session_state.sc_results_rows = []
        st.rerun()

    row_labels_html = [r[0] for r in rows]
    cell_matrix = [r[1] for r in rows]

    orientation_label = st.radio(
        "Sposob prezentacji tabeli",
        ["Pliki w kolumnach, parametry w wierszach", "Parametry w kolumnach, pliki w wierszach"],
        key="sc_results_orientation",
    )
    orientation = "files_cols" if orientation_label.startswith("Pliki") else "params_cols"
    table_html = render_table_html(row_labels_html, file_names, cell_matrix, orientation,
                                    heading="Wyniki - wszystkie obliczenia")
    st.markdown(table_html, unsafe_allow_html=True)

    full_doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{table_html}</body></html>"
    st.download_button("Pobierz jako HTML", data=full_doc, file_name="super_cell_wyniki.html",
                        mime="text/html", key="sc_results_download")


# ==================== SUPER-CELL (z super_cell_v132.py, przystosowane do dowolnego zakresu) ====================

SUPERCELL_TOL = 1e-3  # tolerancja przy usuwaniu duplikatow (pozycje specjalne)


def get_base_site(structure, name):
    """Znajduje atom o podanej etykiecie WPROST w asymetrycznej czesci (structure.sites)."""
    for site in structure.sites:
        if site.label.upper() == name.upper():
            return site
    raise KeyError(f"Atom '{name}' nie znaleziony w strukturze")


def symmetry_code_from_op(op, shift):
    """Laczy operacje gemmi.Op z przesunieciem (i,j,k) w gotowy string kodu symetrii."""
    i, j, k = shift
    return op.translated([i * op.DEN, j * op.DEN, k * op.DEN]).triplet()


def make_supercell(structure, i_range=(-1, 1), j_range=(-1, 1), k_range=(-1, 1)):
    """Jak oryginalny make_supercell(structure, n), ale zamiast JEDNEGO 'n' (zawsze
    symetryczny zakres wokol zera, ten sam dla a/b/c) przyjmuje NIEZALEZNY zakres
    (min,max) dla kazdej z 3 osi. Dzieki temu user moze wybrac DOWOLNY, niekoniecznie
    symetryczny, niekoniecznie taki sam dla a/b/c zakres przesuniec sieci - to jest
    jednoczesnie odpowiedz na 'dowolny przedzial wektora' i 'dowolny wymiar komorki',
    bo rozmiar wzdluz kazdej osi (max-min+1) jest teraz calkowicie niezalezny.
    Zwraca (all_names, all_elements, all_cart, all_symcodes, all_shifts)."""
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    ops = list(sg.operations())

    i_vals = range(i_range[0], i_range[1] + 1)
    j_vals = range(j_range[0], j_range[1] + 1)
    k_vals = range(k_range[0], k_range[1] + 1)

    all_names, all_elements, all_cart, all_symcodes, all_shifts = [], [], [], [], []

    for i in i_vals:
        for j in j_vals:
            for k in k_vals:
                for site in structure.sites:
                    x0, y0, z0 = site.fract.x, site.fract.y, site.fract.z
                    generated = []  # dedup pozycji specjalnych w tej podkomorce

                    for op in ops:
                        code = symmetry_code_from_op(op, (i, j, k))
                        op_full = gemmi.Op(code)
                        x, y, z = op_full.apply_to_xyz([x0, y0, z0])
                        xw, yw, zw = x % 1.0, y % 1.0, z % 1.0

                        is_dup = any(
                            abs(xw - gx) < SUPERCELL_TOL and abs(yw - gy) < SUPERCELL_TOL
                            and abs(zw - gz) < SUPERCELL_TOL
                            for gx, gy, gz in generated
                        )
                        if is_dup:
                            continue
                        generated.append((xw, yw, zw))

                        pos = structure.cell.orthogonalize(gemmi.Fractional(x, y, z))
                        r = np.array([pos.x, pos.y, pos.z])

                        all_names.append(site.label)
                        all_elements.append(site.element.name)
                        all_cart.append(r)
                        all_symcodes.append(code)
                        all_shifts.append((i, j, k))

    return all_names, all_elements, np.array(all_cart), all_symcodes, all_shifts


def is_empty_cell(v):
    """Sprawdza czy komorka z data_editor jest pusta - obejmuje None, NaN (pandas
    zamienia None na NaN w mieszanej kolumnie) i pusty string."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


def safe_symcode(v, default="x,y,z"):
    """Jak 'v or default', ale POPRAWNIE dziala z NaN - 'nan or default' zwraca nan
    (bo nan jest 'prawdziwe' w Pythonie), wiec zwykle 'or' tu nie wystarcza."""
    return default if is_empty_cell(v) else str(v).strip()


def format_qid(qid):
    """Formatuje numer zapytania ladnie - pandas czasem zamienia kolumne numerow
    na float (np. 1.0 zamiast 1), gdy sa w niej puste wartosci."""
    try:
        f = float(qid)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(qid)


def _group_summary_line(name, atoms):
    parts = ", ".join(f"{a}({s})" for a, s in atoms)
    return f"**{name}**: {parts}"


def group_col(label, group_names):
    """Kolumna z wyborem NAZWY zdefiniowanej wczesniej grupy atomow (zamiast
    wpisywania listy atomow od nowa w kazdej funkcji)."""
    return st.column_config.SelectboxColumn(label, options=group_names, required=False)


def render_group_builder(available_atoms_sorted, available_symcodes_sorted, key_prefix="gb"):
    """Budowniczy nazwanych grup atomow (pierscienie/plaszczyzny) - STALA wysokosc
    niezaleznie od liczby atomow czy grup: jeden multiselect (tryb prosty, wspolny
    symcode) albo jedna mala tabelka (tryb 'rozne symcode', tylko dla EDYTOWANEJ
    wlasnie grupy), a ponizej GESTA lista juz zdefiniowanych grup - jedna linijka
    na grupe, z [Edytuj]/[Usun]. Grupy trzymane w st.session_state.sc_named_groups
    ({nazwa: [(atom, symcode), ...]}) - zwracane, zeby funkcje ponizej mogly sie
    do nich odwolywac przez SAMA NAZWE zamiast wpisywac atomy na nowo za kazdym razem."""
    st.markdown("### Grupy atomow (pierscienie / plaszczyzny) - zdefiniuj raz, uzywaj wszedzie ponizej")

    if "sc_named_groups" not in st.session_state:
        st.session_state.sc_named_groups = {}
    if "sc_gb_edit_target" not in st.session_state:
        st.session_state.sc_gb_edit_target = None
    if "sc_gb_nonce" not in st.session_state:
        st.session_state.sc_gb_nonce = 0

    editing = st.session_state.sc_gb_edit_target
    editing_atoms = st.session_state.sc_named_groups.get(editing, []) if editing else []

    name_key = f"{key_prefix}_name"
    if editing and st.session_state.get(f"{key_prefix}_loaded_for") != editing:
        st.session_state[name_key] = editing
        st.session_state[f"{key_prefix}_loaded_for"] = editing
    if not editing:
        st.session_state[f"{key_prefix}_loaded_for"] = None

    name = st.text_input("Nazwa grupy", key=name_key)

    same_symcode = len({s for _, s in editing_atoms}) <= 1
    advanced_default = bool(editing) and not same_symcode
    advanced = st.checkbox("Rozne symcode dla roznych atomow", value=advanced_default,
                            key=f"{key_prefix}_advanced_{st.session_state.sc_gb_nonce}")

    if not advanced:
        default_atoms = [a for a, s in editing_atoms] if editing else []
        default_symcode = editing_atoms[0][1] if editing_atoms else "x,y,z"
        selected_atoms = st.multiselect(
            "Atomy grupy", options=available_atoms_sorted, default=default_atoms,
            key=f"{key_prefix}_multiselect_{st.session_state.sc_gb_nonce}",
        )
        symcode_options = available_symcodes_sorted if available_symcodes_sorted else ["x,y,z"]
        default_idx = symcode_options.index(default_symcode) if default_symcode in symcode_options else 0
        shared_symcode = st.selectbox(
            "Symcode (wspolny dla calej grupy)", options=symcode_options, index=default_idx,
            key=f"{key_prefix}_shared_symcode_{st.session_state.sc_gb_nonce}",
        )
        current_atoms = [(a, shared_symcode) for a in selected_atoms]
    else:
        default_rows = [{"Atom": a, "Symcode": s} for a, s in editing_atoms] or [{"Atom": None, "Symcode": "x,y,z"}]
        table_df = pd.DataFrame(default_rows)
        edited_table = st.data_editor(
            table_df, num_rows="dynamic", use_container_width=True,
            key=f"{key_prefix}_table_{st.session_state.sc_gb_nonce}",
            column_config={"Atom": atom_col("Atom", available_atoms_sorted),
                            "Symcode": symcode_col("Symcode", available_symcodes_sorted)},
        )
        current_atoms = [(r["Atom"], safe_symcode(r["Symcode"])) for _, r in edited_table.iterrows()
                          if not is_empty_cell(r["Atom"])]

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_label = "Zapisz zmiany" if editing else "Dodaj grupe"
        if st.button(btn_label, key=f"{key_prefix}_save_btn"):
            if is_empty_cell(name):
                st.error("Podaj nazwe grupy.")
            elif not current_atoms:
                st.error("Grupa musi miec co najmniej 1 atom.")
            elif name != editing and name in st.session_state.sc_named_groups:
                st.error(f"Grupa o nazwie '{name}' juz istnieje - wybierz inna nazwe.")
            else:
                if editing and editing != name:
                    del st.session_state.sc_named_groups[editing]
                st.session_state.sc_named_groups[name] = current_atoms
                st.session_state.sc_gb_edit_target = None
                st.session_state.sc_gb_nonce += 1
                st.rerun()
    with btn_col2:
        if editing and st.button("Anuluj edycje", key=f"{key_prefix}_cancel_btn"):
            st.session_state.sc_gb_edit_target = None
            st.session_state.sc_gb_nonce += 1
            st.rerun()

    if st.session_state.sc_named_groups:
        st.markdown("**Zdefiniowane grupy:**")
        for gname, atoms in list(st.session_state.sc_named_groups.items()):
            row_col1, row_col2, row_col3 = st.columns([8, 1, 1])
            with row_col1:
                st.markdown(_group_summary_line(gname, atoms))
            with row_col2:
                if st.button("Edytuj", key=f"{key_prefix}_edit_{gname}"):
                    st.session_state.sc_gb_edit_target = gname
                    st.session_state.sc_gb_nonce += 1
                    st.rerun()
            with row_col3:
                if st.button("Usun", key=f"{key_prefix}_del_{gname}"):
                    del st.session_state.sc_named_groups[gname]
                    st.rerun()
    else:
        st.caption("Brak zdefiniowanych grup - dodaj pierwsza powyzej.")

    return st.session_state.sc_named_groups


def group_by_query(edited_df, id_col="Zapytanie"):
    """Grupuje wiersze tabeli data_editor po numerze zapytania w kolumnie id_col.
    Wiersze z pustym/brakujacym ID sa pomijane (np. nowy pusty wiersz po '+').
    Zwraca liste (query_id, [wiersze]) posortowana po query_id."""
    groups = {}
    for _, row in edited_df.iterrows():
        qid = row[id_col]
        if is_empty_cell(qid):
            continue
        groups.setdefault(qid, []).append(row)
    return sorted(groups.items(), key=lambda x: x[0])


def query_id_col(label="Zapytanie"):
    """Kolumna numeru zapytania - te same numery grupuja wiersze w jedno zapytanie."""
    return st.column_config.NumberColumn(label, help="Wiersze z tym samym numerem naleza do jednego zapytania.",
                                          step=1, required=False)


def role_col(label, options):
    return st.column_config.SelectboxColumn(label, options=options, required=False)


def atom_col(label, available_atoms_sorted):
    """Kolumna z podpowiedziami (wyszukiwanie przy wpisywaniu) - do pojedynczego atomu."""
    return st.column_config.SelectboxColumn(label, options=available_atoms_sorted, required=False)


def get_available_symcodes(structures, i_range=(-1, 1), j_range=(-1, 1), k_range=(-1, 1)):
    """Zbiera WSZYSTKIE operacje symetrii wczytanych struktur, kazda z przesunieciem
    w zakresie i_range/j_range/k_range - DOKLADNIE TYM SAMYM zakresie, ktory user
    ustawil w sekcji 'Generowanie supercelu' (nie sztywne -1..1) - do wypelnienia
    listy wyboru symcode."""
    codes = set()
    for s in structures:
        try:
            sg = gemmi.SpaceGroup(s.spacegroup_hm)
        except Exception:
            continue
        for op in sg.operations():
            for i in range(i_range[0], i_range[1] + 1):
                for j in range(j_range[0], j_range[1] + 1):
                    for k in range(k_range[0], k_range[1] + 1):
                        codes.add(op.translated([i * op.DEN, j * op.DEN, k * op.DEN]).triplet())
    codes = sorted(codes)
    if "x,y,z" in codes:
        codes.remove("x,y,z")
        codes = ["x,y,z"] + codes
    return codes


def symcode_col(label, available_symcodes_sorted):
    """Kolumna z podpowiedziami dla symcode - lista operacji symetrii wczytanych
    plikow (+ przesuniecia -1..1). Poczatek listy to zawsze 'x,y,z'."""
    return st.column_config.SelectboxColumn(label, options=available_symcodes_sorted, required=False,
                                              default="x,y,z")


def atom_list_col(label, example_atoms):
    """Kolumna tekstowa 'lista atomow w jednej komorce' - z podpowiedzia (help)
    pokazujaca prawdziwy przyklad z wczytanych plikow (albo ogolny, jesli brak)."""
    example = ";".join(example_atoms[:3]) if example_atoms else "C1,x,y,z;C2;C3,-x,-y,-z"
    return st.column_config.TextColumn(
        label, required=False,
        help=f"Kilka atomow oddzielonych ';'. Symcode domyslnie x,y,z jesli pominiety. "
             f"Przyklad: {example}"
    )

def page_occupancy():
    st.title("Occupancy")

    cif_paths = get_shared_cif_paths()

    if not cif_paths:
        st.info("Wgraj pliki albo podaj folder")
        return

    all_structures = build_all_structures(cif_paths)
    if not all_structures:
        st.warning("Nie udało się wczytać zadnej struktury z podanych plików")
        return

    path_by_name = {p.name: p for p in cif_paths}

    structure_count_per_file = {}
    for file_name, _, _, _ in all_structures:
        structure_count_per_file[file_name] = structure_count_per_file.get(file_name, 0) + 1

    all_file_names = []
    for file_name, data_name, _, _ in all_structures:
        if structure_count_per_file[file_name] == 1:
            all_file_names.append(file_name)
        else:
            all_file_names.append(f"{file_name}:{data_name}")

    st.success(f"Wczytano {len(all_structures)} struktur(y) z {len(cif_paths)} pliku(ów)")

    available_atoms = set()
    for file_name, data_name, _, _ in all_structures:
        file_path = path_by_name.get(file_name)
        if file_path is not None:
            available_atoms.update(get_available_atom_names(file_path, data_name))
    available_atoms_sorted = sorted(available_atoms, key=natural_key)

    if not available_atoms_sorted:
        st.caption("Brak _shelx_res_file w wgranych plikach.")
        return

    occupancy_names = st.multiselect("Wybierz atomy do occupancy", options=available_atoms_sorted)
    if not occupancy_names:
        st.info("Wybierz przynajmniej jeden atom.")
        return

    orientation_label = st.radio(
        "Sposob prezentacji tabeli",
        ["Pliki w kolumnach, parametry w wierszach", "Parametry w kolumnach, pliki w wierszach"],
    )
    orientation = "files_cols" if orientation_label.startswith("Pliki") else "params_cols"

    occ_row_labels, occ_matrix = build_occupancy_rows(occupancy_names, all_structures, path_by_name)

    st.subheader("Tabela")
    table_html = render_table_html(occ_row_labels, all_file_names, occ_matrix, orientation,
                                    heading="Occupancy")
    st.markdown(table_html, unsafe_allow_html=True)

    full_doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{table_html}</body></html>"
    st.download_button("Pobierz jako plik HTML", data=full_doc, file_name="occupancy.html", mime="text/html")


def page_super_cell():
    st.title("Super-cell")

    st.subheader("Wybierz pliki CIF")
    input_mode = st.radio(
        "Zrodlo plikow",
        ["Dodaj pojedyncze pliki", "Podaj sciezke do folderu"],
        horizontal=True,
        key="sc_input_mode",
    )

    cif_paths = []
    if input_mode == "Dodaj pojedyncze pliki":
        uploaded = st.file_uploader(
            "Wybierz plik(i) CIF", type=["cif"], accept_multiple_files=True, key="sc_uploader"
        )
        cif_paths = load_cif_paths_from_uploads(uploaded)
    else:
        folder_str = st.text_input("Sciezka do folderu z plikami .cif", value="", key="sc_folder")
        if folder_str:
            folder = Path(folder_str)
            if folder.is_dir():
                cif_paths = sorted(folder.glob("*.cif"), key=lambda f: natural_key(f.name))
                if not cif_paths:
                    st.warning("W tym folderze nie znaleziono zadnych plikow .cif")
            else:
                st.error("Podana sciezka nie istnieje albo nie jest folderem")

    if not cif_paths:
        st.info("Wgraj pliki albo podaj folder")
        return

    all_structures = build_all_structures(cif_paths)
    if not all_structures:
        st.warning("Nie udalo sie wczytac zadnej struktury z podanych plikow")
        return

    path_by_name = {p.name: p for p in cif_paths}
    st.success(f"Wczytano {len(all_structures)} struktur(y) z {len(cif_paths)} pliku(ow)")

    structure_count_per_file = {}
    for file_name, _, _, _ in all_structures:
        structure_count_per_file[file_name] = structure_count_per_file.get(file_name, 0) + 1

    all_file_names = []
    for file_name, data_name, _, _ in all_structures:
        if structure_count_per_file[file_name] == 1:
            all_file_names.append(file_name)
        else:
            all_file_names.append(f"{file_name}:{data_name}")

    # zbior WSZYSTKICH atomow ze WSZYSTKICH wczytanych plikow - do podpowiedzi
    # (wyszukiwanie przy wpisywaniu) we wszystkich polach z pojedynczym atomem
    available_atoms = set()
    for file_name, data_name, _, _ in all_structures:
        file_path = path_by_name.get(file_name)
        if file_path is not None:
            available_atoms.update(get_available_atom_names(file_path, data_name))
    available_atoms_sorted = sorted(available_atoms, key=natural_key)

    all_gemmi_structures = []
    for file_name, data_name, _, _ in all_structures:
        file_path = path_by_name.get(file_name)
        if file_path is not None:
            try:
                all_gemmi_structures.append(gemmi.read_small_structure(str(file_path)))
            except Exception:
                pass

    example_atoms = available_atoms_sorted[:6]

    st.divider()

    # ==================== GENEROWANIE SUPERCELU (dziala na JEDNEJ wybranej strukturze) ====================
    st.subheader("Generowanie supercelu")

    selected_label = st.selectbox(
        "Wybierz strukture do wygenerowania supercelu (make_supercell dziala na jednej strukturze naraz)",
        all_file_names,
    )
    selected_index = all_file_names.index(selected_label)
    selected_file_name = all_structures[selected_index][0]
    selected_path = path_by_name[selected_file_name]

    try:
        sc_structure = gemmi.read_small_structure(str(selected_path))
    except Exception as e:
        st.error(f"Nie udalo sie wczytac struktury przez gemmi: {e}")
        return

    st.caption(
        f"Grupa przestrzenna: {sc_structure.spacegroup_hm}  |  "
        f"a,b,c = {sc_structure.cell.a:.4f}, {sc_structure.cell.b:.4f}, {sc_structure.cell.c:.4f} A  |  "
        f"alpha,beta,gamma = {sc_structure.cell.alpha:.3f}, {sc_structure.cell.beta:.3f}, {sc_structure.cell.gamma:.3f} deg"
    )

    st.write("Zakres przesuniec sieci (i, j, k) - kazda os NIEZALEZNIE, niekoniecznie symetrycznie wokol zera:")
    col_i, col_j, col_k = st.columns(3)
    with col_i:
        i_min = st.number_input("i min", value=-1, step=1, key="sc_i_min")
        i_max = st.number_input("i max", value=1, step=1, key="sc_i_max")
    with col_j:
        j_min = st.number_input("j min", value=-1, step=1, key="sc_j_min")
        j_max = st.number_input("j max", value=1, step=1, key="sc_j_max")
    with col_k:
        k_min = st.number_input("k min", value=-1, step=1, key="sc_k_min")
        k_max = st.number_input("k max", value=1, step=1, key="sc_k_max")

    if i_min > i_max or j_min > j_max or k_min > k_max:
        st.error("Wartosc 'min' nie moze byc wieksza od 'max' dla zadnej osi.")
        return

    # lista podpowiedzi symcode - UZYWA DOKLADNIE TEGO ZAKRESU (i_min..i_max itd.),
    # ktory user wlasnie ustawil powyzej, a nie sztywnego -1..1. Zabezpieczenie:
    # przy bardzo duzym zakresie (duzo operacji symetrii x duzy zakres) lista
    # eksplodowalaby do tysiecy pozycji - wtedy przycinamy do -2..2 z ostrzezeniem.
    n_ops_guess = len(gemmi.SpaceGroup(sc_structure.spacegroup_hm).operations()) if all_gemmi_structures else 1
    combo_count = n_ops_guess * (i_max - i_min + 1) * (j_max - j_min + 1) * (k_max - k_min + 1)
    if combo_count > 1000:
        st.caption("Zakres i/j/k jest bardzo duzy - lista podpowiedzi symcode ograniczona do -2..2, "
                   "zeby nie zamulic strony. Mozesz nadal wpisac dowolny symcode recznie.")
        symcode_i_range, symcode_j_range, symcode_k_range = (-2, 2), (-2, 2), (-2, 2)
    else:
        symcode_i_range, symcode_j_range, symcode_k_range = (i_min, i_max), (j_min, j_max), (k_min, k_max)

    available_symcodes_sorted = get_available_symcodes(
        all_gemmi_structures, symcode_i_range, symcode_j_range, symcode_k_range
    )

    n_i, n_j, n_k = i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1
    st.caption(f"To da {n_i} x {n_j} x {n_k} = {n_i * n_j * n_k} pod-komorek "
               f"(i={i_min}..{i_max}, j={j_min}..{j_max}, k={k_min}..{k_max}).")

    if st.button("Generuj supercele"):
        with st.spinner("Licze..."):
            names, elements, cart, symcodes, shifts = make_supercell(
                sc_structure, (i_min, i_max), (j_min, j_max), (k_min, k_max)
            )
        st.success(f"Wygenerowano {len(names)} pozycji atomow.")

        df_rows = [
            {"Atom": name, "Pierwiastek": el, "Symmetry code": code, "shift (i,j,k)": str(shift),
             "x": f"{pos[0]:.5f}", "y": f"{pos[1]:.5f}", "z": f"{pos[2]:.5f}"}
            for name, el, pos, code, shift in zip(names, elements, cart, symcodes, shifts)
        ]
        st.dataframe(df_rows, use_container_width=True, height=400)

        csv_lines = ["Atom,Pierwiastek,Symmetry code,shift (i,j,k),x,y,z"]
        txt_lines = [f"{'Atom':10s} {'Pierw.':6s} {'Symmetry code':20s} {'shift':14s} {'x':>10s} {'y':>10s} {'z':>10s}"]
        for row in df_rows:
            csv_lines.append(f'{row["Atom"]},{row["Pierwiastek"]},{row["Symmetry code"]},'
                              f'"{row["shift (i,j,k)"]}",{row["x"]},{row["y"]},{row["z"]}')
            txt_lines.append(f'{row["Atom"]:10s} {row["Pierwiastek"]:6s} {row["Symmetry code"]:20s} '
                              f'{row["shift (i,j,k)"]:14s} {row["x"]:>10s} {row["y"]:>10s} {row["z"]:>10s}')

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Pobierz jako CSV",
                data="\n".join(csv_lines).encode("utf-8"),
                file_name=f"supercell_{selected_file_name}.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                "Pobierz jako TXT",
                data="\n".join(txt_lines).encode("utf-8"),
                file_name=f"supercell_{selected_file_name}.txt",
                mime="text/plain",
            )

    st.divider()
    st.subheader("Obliczenia geometryczne")

    # ==================== ODLEGLOSC ====================
    st.markdown("### Odleglosc")
    default_df = pd.DataFrame([{"Atom 1": None, "Symcode 1": "x,y,z", "Atom 2": None, "Symcode 2": "x,y,z"}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="dist_editor",
        column_config={"Atom 1": atom_col("Atom 1", available_atoms_sorted),
                        "Symcode 1": symcode_col("Symcode 1", available_symcodes_sorted),
                        "Atom 2": atom_col("Atom 2", available_atoms_sorted),
                        "Symcode 2": symcode_col("Symcode 2", available_symcodes_sorted)},
    )
    if st.button("Oblicz odleglosci", key="dist_btn"):
        queries = []
        for _, row in edited.iterrows():
            n1, s1, n2, s2 = row["Atom 1"], row["Symcode 1"], row["Atom 2"], row["Symcode 2"]
            if is_empty_cell(n1) or is_empty_cell(n2):
                continue
            queries.append((f"{n1}-{n2}", (n1, safe_symcode(s1), n2, safe_symcode(s2))))
        row_labels, cell_matrix = build_query_rows(queries, distance_esd, all_structures, path_by_name)
        append_results("Odleglosc", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== KAT ====================
    st.markdown("### Kat")
    default_df = pd.DataFrame([{"Atom 1": None, "Symcode 1": "x,y,z", "Atom 2 (wierzcholek)": None,
                                 "Symcode 2": "x,y,z", "Atom 3": None, "Symcode 3": "x,y,z"}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="angle_editor",
        column_config={"Atom 1": atom_col("Atom 1", available_atoms_sorted),
                        "Symcode 1": symcode_col("Symcode 1", available_symcodes_sorted),
                        "Atom 2 (wierzcholek)": atom_col("Atom 2 (wierzcholek)", available_atoms_sorted),
                        "Symcode 2": symcode_col("Symcode 2", available_symcodes_sorted),
                        "Atom 3": atom_col("Atom 3", available_atoms_sorted),
                        "Symcode 3": symcode_col("Symcode 3", available_symcodes_sorted)},
    )
    if st.button("Oblicz katy", key="angle_btn"):
        queries = []
        for _, row in edited.iterrows():
            n1, s1 = row["Atom 1"], row["Symcode 1"]
            n2, s2 = row["Atom 2 (wierzcholek)"], row["Symcode 2"]
            n3, s3 = row["Atom 3"], row["Symcode 3"]
            if is_empty_cell(n1) or is_empty_cell(n2) or is_empty_cell(n3):
                continue
            queries.append((f"{n1}-{n2}-{n3}", (n1, safe_symcode(s1), n2, safe_symcode(s2), n3, safe_symcode(s3))))
        row_labels, cell_matrix = build_query_rows(queries, angle_esd, all_structures, path_by_name)
        append_results("Kat", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== KAT TORSYJNY ====================
    st.markdown("### Kat torsyjny")
    default_df = pd.DataFrame([{"Atom 1": None, "Symcode 1": "x,y,z", "Atom 2": None, "Symcode 2": "x,y,z",
                                 "Atom 3": None, "Symcode 3": "x,y,z", "Atom 4": None, "Symcode 4": "x,y,z"}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="tors_editor",
        column_config={"Atom 1": atom_col("Atom 1", available_atoms_sorted),
                        "Symcode 1": symcode_col("Symcode 1", available_symcodes_sorted),
                        "Atom 2": atom_col("Atom 2", available_atoms_sorted),
                        "Symcode 2": symcode_col("Symcode 2", available_symcodes_sorted),
                        "Atom 3": atom_col("Atom 3", available_atoms_sorted),
                        "Symcode 3": symcode_col("Symcode 3", available_symcodes_sorted),
                        "Atom 4": atom_col("Atom 4", available_atoms_sorted),
                        "Symcode 4": symcode_col("Symcode 4", available_symcodes_sorted)},
    )
    if st.button("Oblicz katy torsyjne", key="tors_btn"):
        queries = []
        for _, row in edited.iterrows():
            n1, s1 = row["Atom 1"], row["Symcode 1"]
            n2, s2 = row["Atom 2"], row["Symcode 2"]
            n3, s3 = row["Atom 3"], row["Symcode 3"]
            n4, s4 = row["Atom 4"], row["Symcode 4"]
            if is_empty_cell(n1) or is_empty_cell(n2) or is_empty_cell(n3) or is_empty_cell(n4):
                continue
            queries.append((f"{n1}-{n2}-{n3}-{n4}",
                             (n1, safe_symcode(s1), n2, safe_symcode(s2), n3, safe_symcode(s3), n4, safe_symcode(s4))))
        row_labels, cell_matrix = build_query_rows(queries, torsion_angle_esd, all_structures, path_by_name)
        append_results("Kat torsyjny", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== TRZY ATOMY: ODLEGLOSCI + KAT ====================
    st.markdown("### Trzy atomy: dist(1,3), dist(1,2), kat(1,2,3)")
    default_df = pd.DataFrame([{"Atom 1": None, "Symcode 1": "x,y,z", "Atom 2": None, "Symcode 2": "x,y,z",
                                 "Atom 3": None, "Symcode 3": "x,y,z"}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="triple_editor",
        column_config={"Atom 1": atom_col("Atom 1", available_atoms_sorted),
                        "Symcode 1": symcode_col("Symcode 1", available_symcodes_sorted),
                        "Atom 2": atom_col("Atom 2", available_atoms_sorted),
                        "Symcode 2": symcode_col("Symcode 2", available_symcodes_sorted),
                        "Atom 3": atom_col("Atom 3", available_atoms_sorted),
                        "Symcode 3": symcode_col("Symcode 3", available_symcodes_sorted)},
    )
    if st.button("Oblicz (3 atomy)", key="triple_btn"):
        for _, row in edited.iterrows():
            n1, s1 = row["Atom 1"], row["Symcode 1"]
            n2, s2 = row["Atom 2"], row["Symcode 2"]
            n3, s3 = row["Atom 3"], row["Symcode 3"]
            if is_empty_cell(n1) or is_empty_cell(n2) or is_empty_cell(n3):
                continue
            s1, s2, s3 = safe_symcode(s1), safe_symcode(s2), safe_symcode(s3)

            keys = ["dist_1_3", "dist_1_2", "angle_1_2_3"]
            key_labels = {"dist_1_3": f"dist({n1},{n3})", "dist_1_2": f"dist({n1},{n2})",
                          "angle_1_2_3": f"angle({n1}-{n2}-{n3})"}
            rows_by_key = {k: [] for k in keys}
            for file_name, data_name, _, _ in all_structures:
                file_path = path_by_name.get(file_name)
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    et = get_esd_table(str(file_path))
                    ce = get_cell_esd(str(file_path))
                    result = three_atom_summary_esd(s, et, ce, n1, s1, n2, s2, n3, s3)
                except Exception as e:
                    for key in keys:
                        rows_by_key[key].append(cell_html(f"blad: {e}"))
                    continue
                for key in keys:
                    val, sigma = result[key]
                    rows_by_key[key].append(cell_html(format_with_esd(val, sigma)))

            row_labels_html_list = [cell_html(key_labels[k]) for k in keys]
            cell_matrix = [rows_by_key[k] for k in keys]
            append_results("3 atomy", row_labels_html_list, cell_matrix, all_file_names)
            show_immediate_preview(row_labels_html_list, all_file_names, cell_matrix)
        st.success("Dodano wyniki do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== ODLEGLOSC OD PLASZCZYZNY ====================
    named_groups = render_group_builder(available_atoms_sorted, available_symcodes_sorted)
    group_names = list(named_groups.keys())

    st.divider()

    st.markdown("### Odleglosc punktu od plaszczyzny")
    st.caption("Kazdy wiersz to JEDNO zapytanie. 'Plaszczyzna' wybierasz z listy grup zdefiniowanych powyzej.")
    default_df = pd.DataFrame([{"Punkt": None, "Symcode punktu": "x,y,z", "Plaszczyzna (grupa)": None}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="d2p_editor",
        column_config={"Punkt": atom_col("Punkt", available_atoms_sorted),
                        "Symcode punktu": symcode_col("Symcode punktu", available_symcodes_sorted),
                        "Plaszczyzna (grupa)": group_col("Plaszczyzna (grupa)", group_names)},
    )
    if st.button("Oblicz odleglosci od plaszczyzny", key="d2p_btn"):
        queries = []
        for _, row in edited.iterrows():
            point, point_sym, gname = row["Punkt"], row["Symcode punktu"], row["Plaszczyzna (grupa)"]
            if is_empty_cell(point) or is_empty_cell(gname):
                continue
            group_atoms = named_groups.get(gname, [])
            if len(group_atoms) < 3:
                st.warning(f"Grupa '{gname}' ma za malo atomow (min. 3, jest {len(group_atoms)}).")
                continue
            plane_names = [a for a, s in group_atoms]
            plane_symcodes = [s for a, s in group_atoms]
            label = f"{point}->plane({gname})"
            queries.append((label, (point, safe_symcode(point_sym), plane_names, plane_symcodes)))
        row_labels, cell_matrix = build_query_rows(queries, distance_to_plane_esd, all_structures, path_by_name)
        append_results("Odleglosc od plaszczyzny", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== CENTROID -> PLASZCZYZNA ====================
    st.markdown("### Centroid -> plaszczyzna")
    st.caption("Kazdy wiersz to JEDNO zapytanie. Obie grupy wybierasz z listy zdefiniowanej powyzej.")
    default_df = pd.DataFrame([{"Grupa 1 (centroid)": None, "Grupa 2 (plaszczyzna)": None}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="c2p_editor",
        column_config={"Grupa 1 (centroid)": group_col("Grupa 1 (centroid)", group_names),
                        "Grupa 2 (plaszczyzna)": group_col("Grupa 2 (plaszczyzna)", group_names)},
    )
    if st.button("Oblicz centroid -> plaszczyzna", key="c2p_btn"):
        queries = []
        for _, row in edited.iterrows():
            g1name, g2name = row["Grupa 1 (centroid)"], row["Grupa 2 (plaszczyzna)"]
            if is_empty_cell(g1name) or is_empty_cell(g2name):
                continue
            g1 = named_groups.get(g1name, [])
            g2 = named_groups.get(g2name, [])
            if not g1 or len(g2) < 3:
                st.warning(f"'{g1name}' potrzebuje min. 1 atomu, '{g2name}' min. 3 "
                           f"(jest {len(g1)} i {len(g2)}).")
                continue
            names1 = [a for a, s in g1]; symcodes1 = [s for a, s in g1]
            names2 = [a for a, s in g2]; symcodes2 = [s for a, s in g2]
            label = f"centroid({g1name})->plane({g2name})"
            queries.append((label, (names1, symcodes1, names2, symcodes2)))
        row_labels, cell_matrix = build_query_rows(queries, centroid_to_plane_distance_esd,
                                                    all_structures, path_by_name)
        append_results("Centroid -> plaszczyzna", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== CENTROID <-> CENTROID ====================
    st.markdown("### Centroid <-> centroid")
    st.caption("Kazdy wiersz to JEDNO zapytanie. Obie grupy wybierasz z listy zdefiniowanej powyzej")
    default_df = pd.DataFrame([{"Grupa 1": None, "Grupa 2": None}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="c2c_editor",
        column_config={"Grupa 1": group_col("Grupa 1", group_names),
                        "Grupa 2": group_col("Grupa 2", group_names)},
    )
    if st.button("Oblicz centroid <-> centroid", key="c2c_btn"):
        queries = []
        for _, row in edited.iterrows():
            g1name, g2name = row["Grupa 1"], row["Grupa 2"]
            if is_empty_cell(g1name) or is_empty_cell(g2name):
                continue
            g1 = named_groups.get(g1name, [])
            g2 = named_groups.get(g2name, [])
            if not g1 or not g2:
                st.warning(f"'{g1name}' i '{g2name}' musza miec min. 1 atom kazda")
                continue
            names1 = [a for a, s in g1]; symcodes1 = [s for a, s in g1]
            names2 = [a for a, s in g2]; symcodes2 = [s for a, s in g2]
            label = f"centroid({g1name})<->centroid({g2name})"
            queries.append((label, (names1, symcodes1, names2, symcodes2)))
        row_labels, cell_matrix = build_query_rows(queries, distance_between_centroids_esd,
                                                    all_structures, path_by_name)
        append_results("Centroid <-> centroid", row_labels, cell_matrix, all_file_names)
        show_immediate_preview(row_labels, all_file_names, cell_matrix)
        st.success(f"Dodano {len(row_labels)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # ==================== OBJETOSC WIELOSCIANU ====================
    st.markdown("### Objetosc wielosciamu")
    st.caption("Kazdy wiersz to JEDNO zapytanie, wybierz grupe zdefiniowana powyzej (min. 4 atomy)")
    default_df = pd.DataFrame([{"Grupa": None}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="vol_editor",
        column_config={"Grupa": group_col("Grupa", group_names)},
    )
    if st.button("Oblicz objetosci", key="vol_btn"):
        row_labels_html_list, rows = [], []
        for _, row in edited.iterrows():
            gname = row["Grupa"]
            if is_empty_cell(gname):
                continue
            group_atoms = named_groups.get(gname, [])
            if len(group_atoms) < 4:
                st.warning(f"Grupa '{gname}' ma za malo atomow (min. 4, jest {len(group_atoms)})")
                continue
            names = [a for a, s in group_atoms]
            symcodes = [s for a, s in group_atoms]
            row_labels_html_list.append(cell_html(f"poly_volume({gname})"))
            vals = []
            for file_name, data_name, _, _ in all_structures:
                file_path = path_by_name.get(file_name)
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    v = poly_volume(s, names, symcodes)
                    vals.append(cell_html(f"{v:.4f}"))
                except Exception as e:
                    vals.append(cell_html(f"blad: {e}"))
            rows.append(vals)
        append_results("Objetosc (poly_volume)", row_labels_html_list, rows, all_file_names)
        show_immediate_preview(row_labels_html_list, all_file_names, rows)
        st.success(f"Dodano {len(row_labels_html_list)} wynik(ow) do wspolnej tabeli na dole strony.")

    st.divider()

    # # ==================== OBJETOSC WIELOSCIANU (AUTO) ====================
    # st.markdown("### Objetosc wielosciamu - poly_volume_auto (sam szuka symcode wierzcholkow)")
    # st.caption("Kazdy wiersz to JEDNO zapytanie. Symcode atomow w grupie 'Wierzcholki' jest ignorowany "
    #            "(funkcja sama znajdzie najlepszy).")
    # search_range_vol = st.number_input("search_range", value=1, min_value=0, step=1, key="vol_auto_range")
    # default_df2 = pd.DataFrame([{"Atom centralny": None, "Symcode centralny": "x,y,z", "Wierzcholki (grupa)": None}])
    # edited = st.data_editor(
    #     default_df2, num_rows="dynamic", use_container_width=True, key="vol_auto_editor",
    #     column_config={"Atom centralny": atom_col("Atom centralny", available_atoms_sorted),
    #                     "Symcode centralny": symcode_col("Symcode centralny", available_symcodes_sorted),
    #                     "Wierzcholki (grupa)": group_col("Wierzcholki (grupa)", group_names)},
    # )
    # if st.button("Oblicz objetosci (auto)", key="vol_auto_btn"):
    #     row_labels_html_list, rows = [], []
    #     for _, row in edited.iterrows():
    #         central_name, gname = row["Atom centralny"], row["Wierzcholki (grupa)"]
    #         central_symcode = safe_symcode(row["Symcode centralny"])
    #         if is_empty_cell(central_name) or is_empty_cell(gname):
    #             continue
    #         group_atoms = named_groups.get(gname, [])
    #         if not group_atoms:
    #             st.warning(f"Grupa '{gname}' jest pusta.")
    #             continue
    #         vertex_names = [a for a, s in group_atoms]
    #         row_labels_html_list.append(cell_html(f"{central_name}+auto({gname})"))
    #         vals = []
    #         for file_name, data_name, _, _ in all_structures:
    #             file_path = path_by_name.get(file_name)
    #             try:
    #                 s = gemmi.read_small_structure(str(file_path))
    #                 v = poly_volume_auto(s, central_name, central_symcode, vertex_names,
    #                                       search_range=search_range_vol)
    #                 vals.append(cell_html(f"{v:.4f}"))
    #             except Exception as e:
    #                 vals.append(cell_html(f"blad: {e}"))
    #         rows.append(vals)
    #     append_results("Objetosc (auto)", row_labels_html_list, rows, all_file_names)
    #     show_immediate_preview(row_labels_html_list, all_file_names, rows)
    #     st.success(f"Dodano {len(row_labels_html_list)} wynik(ow) do wspolnej tabeli na dole strony.")

    # st.divider()

    # ==================== SLABE KONTAKTY ====================
    st.markdown("### Slabe kontakty")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dist_min = st.number_input("dystans min", value=3.0, key="wc_dmin")
        angle_min = st.number_input("kat min", value=110.0, key="wc_amin")
    with col2:
        dist_max = st.number_input("dystans max", value=4.0, key="wc_dmax")
        angle_max = st.number_input("kat max", value=180.0, key="wc_amax")
    with col3:
        tolerance = st.number_input("tolerancja vdW", value=0.2, key="wc_tol")
    with col4:
        search_range_wc = st.number_input("search_range", value=1, min_value=0, step=1, key="wc_range")
    st.caption("Ustawienia powyzej sa WSPOLNE dla wszystkich wierszy ponizej")
    default_df = pd.DataFrame([{"Atom 1 (donor)": None, "Symcode 1": "x,y,z", "Atom 2 (rodzic)": None,
                                 "Symcode 2": "x,y,z", "Szukany pierwiastek": "O"}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="wc_editor",
        column_config={"Atom 1 (donor)": atom_col("Atom 1 (donor)", available_atoms_sorted),
                        "Symcode 1": symcode_col("Symcode 1", available_symcodes_sorted),
                        "Atom 2 (rodzic)": atom_col("Atom 2 (rodzic)", available_atoms_sorted),
                        "Symcode 2": symcode_col("Symcode 2", available_symcodes_sorted)},
    )
    if st.button("Szukaj slabych kontaktow", key="wc_btn"):
        for i, row in edited.iterrows():
            n1, s1 = row["Atom 1 (donor)"], safe_symcode(row["Symcode 1"])
            n2, s2 = row["Atom 2 (rodzic)"], safe_symcode(row["Symcode 2"])
            element = row["Szukany pierwiastek"]
            if is_empty_cell(n1) or is_empty_cell(n2) or is_empty_cell(element):
                continue
            st.markdown(f"**Zapytanie {i+1}: {n1}...{element} (via {n2})**")
            for file_name, data_name, _, _ in all_structures:
                file_path = path_by_name.get(file_name)
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    et = get_esd_table(str(file_path))
                    ce = get_cell_esd(str(file_path))
                    results = find_weak_contacts(s, n1, s1, n2, s2, element,
                                                  dist_range=(dist_min, dist_max),
                                                  angle_range=(angle_min, angle_max),
                                                  tolerance=tolerance, search_range=search_range_wc)
                except Exception as e:
                    st.write(f"{file_name}: blad ({e})")
                    continue
                st.write(f"*{file_name}* - znaleziono {len(results)}")
                if results:
                    table_rows = []
                    for r in results:
                        d1, sd1 = distance_esd(s, et, ce, n1, s1, r["name"], r["symcode"])
                        d2, sd2 = distance_esd(s, et, ce, n2, s2, r["name"], r["symcode"])
                        ang_val, ang_sig = angle_esd(s, et, ce, n1, s1, n2, s2, r["name"], r["symcode"])
                        table_rows.append({
                            "Atom": r["name"], "Symcode": r["symcode"],
                            f"Odleglosc {n1}-znaleziony": format_with_esd(d1, sd1),
                            f"Odleglosc {n2}-znaleziony": format_with_esd(d2, sd2),
                            "Kat (deg)": format_with_esd(ang_val, ang_sig),
                            "vdW sum": f"{r['vdw_sum']:.3f}",
                        })
                    st.dataframe(table_rows, use_container_width=True)

    st.divider()

    # ==================== N NAJKROTSZYCH ODLEGLOSCI ====================
    st.markdown("### N najkrotszych odleglosci")
    col1, col2 = st.columns(2)
    with col1:
        n_results = st.number_input("Ile najkrotszych", value=6, min_value=1, step=1, key="nshort_n")
        by_element = st.checkbox("Szukaj po pierwiastku (nie po dokladnej nazwie)", key="nshort_by_element")
    with col2:
        search_range_ns = st.number_input("search_range", value=2, min_value=0, step=1, key="nshort_range")
    st.caption("Ustawienia powyzej sa WSPOLNE dla wszystkich wierszy ponizej.")
    default_df = pd.DataFrame([{"Atom centralny": None, "Symcode centralny": "x,y,z",
                                 "Szukana nazwa/pierwiastek": ""}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="nshort_editor",
        column_config={"Atom centralny": atom_col("Atom centralny", available_atoms_sorted),
                        "Symcode centralny": symcode_col("Symcode centralny", available_symcodes_sorted)},
    )
    if st.button("Szukaj najkrotszych odleglosci", key="nshort_btn"):
        for i, row in edited.iterrows():
            central_name = row["Atom centralny"]
            central_symcode = safe_symcode(row["Symcode centralny"])
            target = row["Szukana nazwa/pierwiastek"]
            if is_empty_cell(central_name) or is_empty_cell(target):
                continue
            st.markdown(f"**Zapytanie {i+1}: {central_name} -> {target}**")
            for file_name, data_name, _, _ in all_structures:
                file_path = path_by_name.get(file_name)
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    results = find_n_shortest_distances(s, central_name, central_symcode, target,
                                                         n=n_results, search_range=search_range_ns,
                                                         by_element=by_element)
                except Exception as e:
                    st.write(f"{file_name}: blad ({e})")
                    continue
                st.write(f"*{file_name}*")
                st.dataframe([{"Atom": r["name"], "Symcode": r["symcode"],
                                "Odleglosc (A)": f"{r['distance']:.4f}"}
                               for r in results], use_container_width=True)

    st.divider()

    # ==================== PI-PI STACKING ====================
    st.markdown("### Pi-pi stacking")
    st.caption("Kazdy wiersz to JEDNO zapytanie. Pierscienie wybierasz z listy grup zdefiniowanej powyzej")
    default_df = pd.DataFrame([{"Ring 1 (grupa)": None, "Ring 2 (grupa)": None}])
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="pi_editor",
        column_config={"Ring 1 (grupa)": group_col("Ring 1 (grupa)", group_names),
                        "Ring 2 (grupa)": group_col("Ring 2 (grupa)", group_names)},
    )

    col1, col2 = st.columns(2)
    with col1:
        dist_min = st.number_input("dist_range min", value=2.9, key="pi_dmin")
    with col2:
        dist_max = st.number_input("dist_range max", value=4.1, key="pi_dmax")

    if st.button("Oblicz pi-pi stacking", key="pi_btn"):
        for idx, row in edited.iterrows():
            g1name = row["Ring 1 (grupa)"]
            g2name = row["Ring 2 (grupa)"]
            if is_empty_cell(g1name):
                continue
            g1 = named_groups.get(g1name, [])
            names1 = [a for a, s in g1]
            symcodes1 = [s for a, s in g1]
            if len(names1) < 3:
                st.warning(f"'{g1name}' potrzebuje min. 3 atomow (jest {len(names1)})")
                continue
            names2, symcodes2 = None, None
            if not is_empty_cell(g2name):
                g2 = named_groups.get(g2name, [])
                names2 = [a for a, s in g2]
                symcodes2 = [s for a, s in g2]

            keys = ["centroid_to_centroid", "centroid1_to_plane2", "offset", "twist_angle"]
            rows_by_key = {k: [] for k in keys}
            for file_name, data_name, _, _ in all_structures:
                file_path = path_by_name.get(file_name)
                try:
                    s = gemmi.read_small_structure(str(file_path))
                    et = get_esd_table(str(file_path))
                    ce = get_cell_esd(str(file_path))
                    result = pi_pi_stacking_esd(s, et, ce, names1, symcodes1, names2, symcodes2,
                                                 dist_range=(dist_min, dist_max))
                except Exception as e:
                    for key in keys:
                        rows_by_key[key].append(cell_html(f"blad: {e}"))
                    continue
                if isinstance(result, str):
                    for key in keys:
                        rows_by_key[key].append(cell_html(result))
                else:
                    for key in keys:
                        val, sigma = result[key]
                        rows_by_key[key].append(cell_html(format_with_esd(val, sigma)))

            ring_label = g1name if names2 is None else f"{g1name}/{g2name}"
            row_labels_html_list = [cell_html(f"{ring_label}: {k}") for k in keys]
            cell_matrix = [rows_by_key[k] for k in keys]
            append_results("Pi-pi", row_labels_html_list, cell_matrix, all_file_names)
            show_immediate_preview(row_labels_html_list, all_file_names, cell_matrix)
        st.success("Dodano wyniki pi-pi do wspolnej tabeli na dole strony")

    st.divider()
    render_accumulated_results()