import re
import hmac
import hashlib
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "v3_nominales_forzado_2026_06_24"


# ============================================================
# Utilidades generales
# ============================================================

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def norm_text(value) -> str:
    """Normaliza textos para comparar nombres de columnas."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "[": " ", "]": " ", "(": " ", ")": " ", "/": " ", "_": " ", "-": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_isin(value):
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text if ISIN_RE.match(text) else np.nan


def parse_number(value):
    """Convierte números con formato español o anglosajón a float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace("%", "")
    if text == "":
        return np.nan

    # Formato español: 1.234.567,89
    if re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", text):
        text = text.replace(".", "").replace(",", ".")
    else:
        # Formato anglosajón: 1,234,567.89
        text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def to_number_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_number).astype(float)


def detect_header_row(raw: pd.DataFrame, required_terms, max_rows: int = 30):
    """Busca una fila de cabecera que contenga alguno de los términos requeridos."""
    max_rows = min(max_rows, len(raw))
    required_norm = [norm_text(t) for t in required_terms]
    for i in range(max_rows):
        row_norm = [norm_text(x) for x in raw.iloc[i].tolist()]
        row_join = " | ".join(row_norm)
        if any(term in row_join for term in required_norm):
            return i
    return None


def find_column(columns, candidates, required=True):
    """Encuentra una columna por igualdad o inclusión normalizada, respetando la prioridad de candidates."""
    columns_norm = {col: norm_text(col) for col in columns}
    candidates_norm = [norm_text(c) for c in candidates]

    # Primero igualdad exacta normalizada, respetando la prioridad de los candidatos.
    for cand in candidates_norm:
        for col, col_norm in columns_norm.items():
            if col_norm == cand:
                return col

    # Luego inclusión, también respetando la prioridad de los candidatos.
    for cand in candidates_norm:
        for col, col_norm in columns_norm.items():
            if cand and (cand in col_norm or col_norm in cand):
                return col

    if required:
        raise ValueError(f"No se ha encontrado ninguna columna compatible con: {candidates}")
    return None


def first_non_null(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def safe_sum(series: pd.Series) -> float:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if nums.empty:
        return np.nan
    return float(nums.sum())


# ============================================================
# Lectura Depositario
# ============================================================


def parse_depositario(file_bytes: bytes):
    warnings = []
    xls = pd.ExcelFile(BytesIO(file_bytes))
    parsed = None
    raw_selected = None
    sheet_selected = None

    for sheet in xls.sheet_names:
        raw = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, header=None)
        header_row = detect_header_row(raw, ["c_codigo_isin", "codigo isin", "isin"])
        if header_row is None:
            continue

        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, header=header_row)
        if df.empty:
            continue

        parsed = df
        raw_selected = raw
        sheet_selected = sheet
        break

    if parsed is None:
        raise ValueError("No he podido identificar la tabla del depositario. Necesito una columna de ISIN, por ejemplo c_codigo_isin o ISIN.")

    df = parsed.copy()
    df.columns = [str(c).strip() for c in df.columns]

    isin_col = find_column(df.columns, ["c_codigo_isin", "codigo isin", "isin", "isin codigo"])
    name_col = find_column(df.columns, ["c_nombre_instrumento", "nombre instrumento", "instrumento", "descripcion", "nombre"], required=False)
    sector_col = find_column(df.columns, ["c_nombre_sector", "sector", "asset class", "clase activo"], required=False)
    tipo_col = find_column(df.columns, ["c_rf_rv", "tipo activo", "rf rv", "renta fija renta variable"], required=False)
    titulos_col = find_column(df.columns, ["titulos", "títulos", "cantidad", "pos", "posicion", "posición", "unidades"], required=False)
    nominal_col = find_column(df.columns, ["nominal", "nominal actual", "nominal div"], required=False)
    efectivo_col = find_column(df.columns, ["efectivo", "valor mercado", "valor de mercado", "market value", "valmrc"], required=False)

    cash_col = find_column(
        df.columns,
        ["c_total_tesoreria_dep", "total tesoreria dep", "c_total_tesoreria", "total tesoreria", "tesoreria", "efectivo cartera"],
        required=False,
    )

    out = pd.DataFrame()
    out["ISIN"] = df[isin_col].apply(normalize_isin)
    out["Nombre_DEP"] = df[name_col] if name_col else ""
    out["Sector_DEP"] = df[sector_col] if sector_col else ""
    out["Tipo_DEP"] = df[tipo_col] if tipo_col else ""
    out["Titulos_DEP"] = to_number_series(df[titulos_col]) if titulos_col else np.nan
    out["Nominal_DEP"] = to_number_series(df[nominal_col]) if nominal_col else np.nan
    out["Efectivo_DEP"] = to_number_series(df[efectivo_col]) if efectivo_col else np.nan

    out = out.dropna(subset=["ISIN"]).copy()

    if out.empty:
        raise ValueError("He encontrado columna de ISIN en el depositario, pero no hay ISINs válidos.")

    # Agrupar por ISIN por si el depositario trae líneas duplicadas.
    grouped = (
        out.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_DEP": first_non_null,
            "Sector_DEP": first_non_null,
            "Tipo_DEP": first_non_null,
            "Titulos_DEP": safe_sum,
            "Nominal_DEP": safe_sum,
            "Efectivo_DEP": safe_sum,
        })
    )

    cash_detected = np.nan
    if cash_col:
        cash_values = df[cash_col].apply(parse_number).dropna()
        if not cash_values.empty:
            cash_detected = float(cash_values.iloc[0])
    else:
        warnings.append("No he detectado una columna clara de efectivo/tesorería en el depositario.")

    return {
        "positions": grouped,
        "raw": raw_selected,
        "sheet": sheet_selected,
        "cash": cash_detected,
        "warnings": warnings,
    }


# ============================================================
# Lectura Bloomberg
# ============================================================


def find_bbg_layout(raw: pd.DataFrame):
    """Detecta la fila de cabecera Bloomberg con 'ISIN' y columnas Cart."""
    max_rows = min(40, len(raw))
    max_cols = raw.shape[1]
    for r in range(max_rows):
        row_norm = [norm_text(x) for x in raw.iloc[r].tolist()]
        for c, val in enumerate(row_norm):
            if val == "isin" or "isin" == val.strip():
                return {"header_row": r, "subheader_row": r + 1, "isin_col": c, "max_cols": max_cols}
    return None


def find_bbg_metric_col(raw: pd.DataFrame, header_row: int, subheader_row: int, candidates, prefer_cart=True, required=True):
    candidates_norm = [norm_text(c) for c in candidates]
    matches = []
    for c in range(raw.shape[1]):
        cell_norm = norm_text(raw.iat[header_row, c])
        if not cell_norm:
            continue
        if any(cand in cell_norm or cell_norm in cand for cand in candidates_norm):
            sub = norm_text(raw.iat[subheader_row, c]) if subheader_row < len(raw) else ""
            matches.append((c, sub))

    if not matches:
        if required:
            raise ValueError(f"No he encontrado la métrica Bloomberg: {candidates}")
        return None

    if prefer_cart:
        for c, sub in matches:
            if sub == "cart":
                return c
    return matches[0][0]


def parse_bloomberg(file_bytes: bytes):
    warnings = []
    xls = pd.ExcelFile(BytesIO(file_bytes))
    selected = None
    sheet_selected = None

    for sheet in xls.sheet_names:
        raw = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, header=None)
        layout = find_bbg_layout(raw)
        if layout:
            selected = (raw, layout)
            sheet_selected = sheet
            break

    if selected is None:
        raise ValueError("No he podido identificar la tabla de Bloomberg. Necesito una columna con cabecera ISIN.")

    raw, layout = selected
    h = layout["header_row"]
    sh = layout["subheader_row"]
    data_start = sh + 1

    isin_col = layout["isin_col"]
    class_col = 0
    name_col = find_bbg_metric_col(raw, h, sh, ["Nombre corto", "short name", "name"], prefer_cart=True, required=False)
    # En los exports de Bloomberg suele coexistir "Valor de mercado (%)" con "ValMrc".
    # Para efectivo/valor absoluto debemos priorizar ValMrc y no la columna porcentual.
    val_col = find_bbg_metric_col(raw, h, sh, ["ValMrc"], prefer_cart=True, required=False)
    if val_col is None:
        val_col = find_bbg_metric_col(raw, h, sh, ["market value", "valor mercado", "valor de mercado"], prefer_cart=True, required=False)
    nominal_col = find_bbg_metric_col(raw, h, sh, ["Nominal actual", "current amount", "par amount", "nominal"], prefer_cart=True, required=False)
    titulos_col = find_bbg_metric_col(raw, h, sh, ["titulos", "títulos", "cantidad", "shares", "pos", "posicion", "posición"], prefer_cart=True, required=False)

    if val_col is None:
        warnings.append("No he encontrado ValMrc/valor de mercado en Bloomberg; se dejará a cero.")
    if nominal_col is None:
        warnings.append("No he encontrado Nominal actual en Bloomberg; se dejará a cero.")
    if titulos_col is None:
        warnings.append("No he encontrado una columna específica de títulos en Bloomberg; se dejará vacía salvo que coincida con nominal.")

    rows = raw.iloc[data_start:].copy()
    out = pd.DataFrame()
    out["ISIN"] = rows.iloc[:, isin_col].apply(normalize_isin)
    out["Nombre_BBG"] = rows.iloc[:, class_col].astype(str).str.strip()
    if name_col is not None:
        out["Nombre_Corto_BBG"] = rows.iloc[:, name_col].astype(str).str.strip()
    else:
        out["Nombre_Corto_BBG"] = ""
    out["Titulos_BBG"] = to_number_series(rows.iloc[:, titulos_col]) if titulos_col is not None else np.nan
    out["Nominal_BBG"] = to_number_series(rows.iloc[:, nominal_col]) if nominal_col is not None else np.nan
    out["Efectivo_BBG"] = to_number_series(rows.iloc[:, val_col]) if val_col is not None else np.nan

    out = out.dropna(subset=["ISIN"]).copy()

    # Quita líneas residuales: ISIN con nominal y valor a cero o vacío.
    for col in ["Titulos_BBG", "Nominal_BBG", "Efectivo_BBG"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid_position = (
        out[["Titulos_BBG", "Nominal_BBG", "Efectivo_BBG"]]
        .fillna(0)
        .abs()
        .sum(axis=1)
        > 0
    )
    dropped = int((~valid_position).sum())
    if dropped:
        warnings.append(f"He descartado {dropped} línea(s) Bloomberg con ISIN pero sin posición/valor.")
    out = out.loc[valid_position].copy()

    if out.empty:
        raise ValueError("He encontrado la columna ISIN en Bloomberg, pero no hay posiciones válidas.")

    grouped = (
        out.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_BBG": first_non_null,
            "Nombre_Corto_BBG": first_non_null,
            "Titulos_BBG": safe_sum,
            "Nominal_BBG": safe_sum,
            "Efectivo_BBG": safe_sum,
        })
    )

    cash_detected = np.nan
    # Detección muy conservadora: evita interpretar la categoría "Tesorería" de Bloomberg como cash,
    # porque muchas veces contiene soberanos o monetarios con ISIN.
    if val_col is not None:
        for idx in range(data_start, len(raw)):
            label = norm_text(raw.iat[idx, class_col])
            isin = normalize_isin(raw.iat[idx, isin_col])
            if pd.isna(isin) and re.search(r"\b(cash|efectivo|liquidez)\b", label):
                cash_val = parse_number(raw.iat[idx, val_col])
                if not pd.isna(cash_val):
                    cash_detected = float(cash_val)
                    break

    return {
        "positions": grouped,
        "raw": raw,
        "sheet": sheet_selected,
        "cash": cash_detected,
        "warnings": warnings,
    }


# ============================================================
# Conciliación y Excel de salida
# ============================================================


def build_reconciliation(dep_pos, bbg_pos, cash_dep=np.nan, cash_bbg=np.nan, tol_titulos=0.0, tol_nominal=1.0, tol_efectivo=1.0):
    merged = dep_pos.merge(bbg_pos, how="outer", on="ISIN", indicator=True)

    numeric_cols = ["Titulos_DEP", "Nominal_DEP", "Efectivo_DEP", "Titulos_BBG", "Nominal_BBG", "Efectivo_BBG"]
    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Los títulos se calculan y se muestran solo como información auxiliar.
    # El estado de conciliación se determina por nominal y valor de mercado/efectivo,
    # porque en estos ficheros los títulos pueden venir con escalas o criterios distintos
    # y no deben gobernar el resumen OK / Diferencia.
    compare_titulos = merged["Titulos_DEP"].notna().any() and merged["Titulos_BBG"].notna().any()
    compare_nominal = merged["Nominal_DEP"].notna().any() and merged["Nominal_BBG"].notna().any()
    compare_efectivo = merged["Efectivo_DEP"].notna().any() and merged["Efectivo_BBG"].notna().any()

    for col in ["Nombre_DEP", "Sector_DEP", "Tipo_DEP", "Nombre_BBG", "Nombre_Corto_BBG"]:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")

    merged["Dif_Titulos"] = merged["Titulos_DEP"].fillna(0) - merged["Titulos_BBG"].fillna(0) if compare_titulos else np.nan
    merged["Dif_Nominal"] = merged["Nominal_DEP"].fillna(0) - merged["Nominal_BBG"].fillna(0) if compare_nominal else np.nan
    merged["Dif_Efectivo"] = merged["Efectivo_DEP"].fillna(0) - merged["Efectivo_BBG"].fillna(0) if compare_efectivo else np.nan

    def estado(row):
        if row["_merge"] == "left_only":
            return "Solo Depositario"
        if row["_merge"] == "right_only":
            return "Solo Bloomberg"

        # CRITERIO CORRECTO PARA EL RESUMEN:
        # Las posiciones se consideran conciliadas por NOMINAL.
        # Los títulos y el valor de mercado/efectivo por línea se muestran como información,
        # pero NO gobiernan el estado OK/Diferencia de las posiciones.
        # El efectivo total se controla después en una fila independiente "EFECTIVO".
        if compare_nominal:
            return "OK" if abs(row["Dif_Nominal"]) <= tol_nominal else "Diferencia"

        # Fallback conservador si algún fichero no trae nominal: usar valor de mercado.
        if compare_efectivo:
            return "OK" if abs(row["Dif_Efectivo"]) <= tol_efectivo else "Diferencia"

        return "OK"

    merged["Estado"] = merged.apply(estado, axis=1)
    merged["Origen"] = merged["_merge"].map({"both": "Ambos", "left_only": "Depositario", "right_only": "Bloomberg"})

    final_cols = [
        "ISIN",
        "Nombre_DEP",
        "Nombre_BBG",
        "Nombre_Corto_BBG",
        "Sector_DEP",
        "Tipo_DEP",
        "Titulos_DEP",
        "Titulos_BBG",
        "Dif_Titulos",
        "Nominal_DEP",
        "Nominal_BBG",
        "Dif_Nominal",
        "Efectivo_DEP",
        "Efectivo_BBG",
        "Dif_Efectivo",
        "Estado",
        "Origen",
    ]
    merged = merged[final_cols].sort_values(["Estado", "ISIN"], kind="stable")

    # Fila de efectivo, si hay algún dato o si el usuario lo introduce manualmente.
    if not pd.isna(cash_dep) or not pd.isna(cash_bbg):
        cash_dep_val = 0.0 if pd.isna(cash_dep) else float(cash_dep)
        cash_bbg_val = 0.0 if pd.isna(cash_bbg) else float(cash_bbg)
        cash_status = "OK" if abs(cash_dep_val - cash_bbg_val) <= tol_efectivo else "Diferencia"
        cash_row = pd.DataFrame([{
            "ISIN": "EFECTIVO",
            "Nombre_DEP": "Efectivo",
            "Nombre_BBG": "Efectivo",
            "Nombre_Corto_BBG": "",
            "Sector_DEP": "Efectivo",
            "Tipo_DEP": "Efectivo",
            "Titulos_DEP": 0.0,
            "Titulos_BBG": 0.0,
            "Dif_Titulos": 0.0,
            "Nominal_DEP": 0.0,
            "Nominal_BBG": 0.0,
            "Dif_Nominal": 0.0,
            "Efectivo_DEP": cash_dep_val,
            "Efectivo_BBG": cash_bbg_val,
            "Dif_Efectivo": cash_dep_val - cash_bbg_val,
            "Estado": cash_status,
            "Origen": "Efectivo",
        }])
        merged = pd.concat([cash_row, merged], ignore_index=True)

    return merged


def write_raw_sheet(writer, sheet_name: str, raw_df: pd.DataFrame):
    raw_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 0)
    for col_idx in range(min(raw_df.shape[1], 60)):
        # Anchura aproximada limitada para evitar hojas inmanejables.
        values = raw_df.iloc[:, col_idx].dropna().astype(str).head(100).tolist()
        max_len = min(max([len(str(v)) for v in values] + [10]), 35)
        worksheet.set_column(col_idx, col_idx, max_len + 2)


def build_excel_output(recon: pd.DataFrame, dep_raw: pd.DataFrame, bbg_raw: pd.DataFrame, dep_sheet_name: str, bbg_sheet_name: str):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        ws_name = "Conciliación"
        recon.to_excel(writer, sheet_name=ws_name, index=False, startrow=10)
        worksheet = writer.sheets[ws_name]

        title_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#17365D"})
        subtitle_fmt = workbook.add_format({"font_size": 10, "font_color": "#666666"})
        header_fmt = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1, "align": "center"})
        metric_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        value_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
        integer_fmt = workbook.add_format({"num_format": "#,##0", "border": 1})
        money_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
        text_fmt = workbook.add_format({"border": 1})

        worksheet.merge_range("A1:Q1", "Conciliación de cartera por ISIN", title_fmt)
        worksheet.write("A2", f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_fmt)
        worksheet.write("A3", f"Hoja depositario: {dep_sheet_name} | Hoja Bloomberg: {bbg_sheet_name}", subtitle_fmt)
        worksheet.write("A4", "Criterio: posiciones conciliadas por nominal; títulos y valor de mercado por línea son informativos. Efectivo total en fila independiente.", subtitle_fmt)

        summary = {
            "Posiciones conciliadas": int((recon["Origen"] == "Ambos").sum()),
            "Solo depositario": int((recon["Estado"] == "Solo Depositario").sum()),
            "Solo Bloomberg": int((recon["Estado"] == "Solo Bloomberg").sum()),
            "Diferencias": int((recon["Estado"] == "Diferencia").sum()),
            "OK": int((recon["Estado"] == "OK").sum()),
        }
        row = 5
        for k, v in summary.items():
            worksheet.write(row, 0, k, metric_fmt)
            worksheet.write(row, 1, v, integer_fmt)
            row += 1

        # Formato de tabla
        for col_num, value in enumerate(recon.columns.values):
            worksheet.write(10, col_num, value, header_fmt)

        worksheet.freeze_panes(11, 0)
        worksheet.autofilter(10, 0, 10 + len(recon), len(recon.columns) - 1)

        widths = {
            "A": 16, "B": 28, "C": 28, "D": 22, "E": 18, "F": 14,
            "G": 14, "H": 14, "I": 14, "J": 16, "K": 16, "L": 16,
            "M": 16, "N": 16, "O": 16, "P": 18, "Q": 14,
        }
        for col_letter, width in widths.items():
            worksheet.set_column(f"{col_letter}:{col_letter}", width)

        # Formatos numéricos
        worksheet.set_column("G:O", 16, money_fmt)
        worksheet.set_column("A:F", 20, text_fmt)
        worksheet.set_column("P:Q", 18, text_fmt)

        start_data = 11
        end_data = 10 + len(recon)
        if len(recon) > 0:
            worksheet.conditional_format(start_data, 15, end_data, 15, {
                "type": "text", "criteria": "containing", "value": "OK",
                "format": workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"}),
            })
            worksheet.conditional_format(start_data, 15, end_data, 15, {
                "type": "text", "criteria": "containing", "value": "Diferencia",
                "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"}),
            })
            worksheet.conditional_format(start_data, 15, end_data, 15, {
                "type": "text", "criteria": "containing", "value": "Solo",
                "format": workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#990000"}),
            })

        write_raw_sheet(writer, "Depositario", dep_raw)
        write_raw_sheet(writer, "Bloomberg", bbg_raw)

    output.seek(0)
    return output



# ============================================================
# Autenticación simple
# ============================================================

def hash_password(password: str) -> str:
    """Devuelve el hash SHA-256 de una contraseña."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_auth_config():
    """Lee usuario y contraseña desde Streamlit Secrets.

    En Streamlit Cloud, configurar en Advanced settings > Secrets:

    [auth]
    username = "usuario"
    password_hash = "hash_sha256_de_la_contraseña"
    """
    try:
        auth = st.secrets.get("auth", {})
    except Exception:
        auth = {}

    username = str(auth.get("username", "")).strip()
    password_hash = str(auth.get("password_hash", "")).strip()
    return username, password_hash


def require_login():
    """Bloquea la app hasta que el usuario introduzca credenciales válidas."""
    username, password_hash = get_auth_config()

    if not username or not password_hash:
        st.error("La autenticación no está configurada. Define [auth] username y password_hash en Streamlit Secrets.")
        st.stop()

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.success(f"Sesión iniciada: {username}")
            if st.button("Cerrar sesión"):
                st.session_state["authenticated"] = False
                st.rerun()
        return

    st.title("Acceso privado")
    st.caption("Introduce usuario y contraseña para acceder al conciliador de carteras.")

    with st.form("login_form"):
        input_user = st.text_input("Usuario")
        input_password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        user_ok = hmac.compare_digest(input_user.strip(), username)
        pass_ok = hmac.compare_digest(hash_password(input_password), password_hash)
        if user_ok and pass_ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")

    st.stop()


# ============================================================
# Interfaz Streamlit
# ============================================================

st.set_page_config(page_title="Conciliador de carteras", layout="wide")

require_login()

st.title("Conciliador de carteras por ISIN")
st.success(f"CÓDIGO ACTIVO: {APP_VERSION} — resumen calculado exclusivamente por nominales; títulos ignorados")
st.caption("El resumen OK/Diferencia de posiciones se calcula por Dif_Nominal. Los títulos y el valor de mercado por línea se muestran solo como información. El efectivo se controla en una fila independiente.")
st.caption("Sube el Excel del depositario y el Excel de Bloomberg. La aplicación cruza las posiciones por ISIN y genera un Excel con Conciliación, Depositario y Bloomberg.")

with st.sidebar:
    st.header("Tolerancias")
    st.caption("Versión corregida: el resumen OK/Diferencia de posiciones se calcula por nominal. Los títulos y el valor de mercado por línea quedan solo como dato informativo; el efectivo total se controla en su propia fila.")
    tol_titulos = st.number_input("Tolerancia títulos informativa", min_value=0.0, value=0.0, step=1.0)
    tol_nominal = st.number_input("Tolerancia nominal", min_value=0.0, value=1.0, step=1.0)
    tol_efectivo = st.number_input("Tolerancia efectivo / valor mercado", min_value=0.0, value=1.0, step=10.0)

col1, col2 = st.columns(2)
with col1:
    dep_file = st.file_uploader("Excel depositario", type=["xlsx", "xls"], key="dep")
with col2:
    bbg_file = st.file_uploader("Excel Bloomberg", type=["xlsx", "xls"], key="bbg")

if dep_file and bbg_file:
    try:
        dep_bytes = dep_file.getvalue()
        bbg_bytes = bbg_file.getvalue()

        dep_data = parse_depositario(dep_bytes)
        bbg_data = parse_bloomberg(bbg_bytes)

        st.success("Archivos leídos correctamente.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Posiciones depositario", len(dep_data["positions"]))
        c2.metric("Posiciones Bloomberg", len(bbg_data["positions"]))
        c3.metric("Efectivo depositario detectado", "-" if pd.isna(dep_data["cash"]) else f"{dep_data['cash']:,.2f}")
        c4.metric("Efectivo Bloomberg detectado", "-" if pd.isna(bbg_data["cash"]) else f"{bbg_data['cash']:,.2f}")

        warnings = dep_data["warnings"] + bbg_data["warnings"]
        if warnings:
            with st.expander("Avisos de lectura"):
                for w in warnings:
                    st.warning(w)

        st.subheader("Efectivo")
        st.caption("Puedes dejar los importes detectados o introducirlos a mano si Bloomberg/depositario no trae el efectivo como una línea clara.")
        cash_col1, cash_col2 = st.columns(2)
        default_cash_dep = 0.0 if pd.isna(dep_data["cash"]) else float(dep_data["cash"])
        default_cash_bbg = 0.0 if pd.isna(bbg_data["cash"]) else float(bbg_data["cash"])
        with cash_col1:
            cash_dep = st.number_input("Efectivo depositario", value=default_cash_dep, step=1000.0, format="%.2f")
        with cash_col2:
            cash_bbg = st.number_input("Efectivo Bloomberg", value=default_cash_bbg, step=1000.0, format="%.2f")

        recon = build_reconciliation(
            dep_data["positions"],
            bbg_data["positions"],
            cash_dep=cash_dep,
            cash_bbg=cash_bbg,
            tol_titulos=tol_titulos,
            tol_nominal=tol_nominal,
            tol_efectivo=tol_efectivo,
        )

        st.subheader(f"Resultado de conciliación por nominal — {APP_VERSION}")

        # Resumen calculado de forma explícita por nominal, no por títulos ni por valoración de mercado por línea.
        posiciones = recon[recon["ISIN"].astype(str).str.upper() != "EFECTIVO"].copy()
        efectivo_row = recon[recon["ISIN"].astype(str).str.upper() == "EFECTIVO"].copy()

        ambos = posiciones["Origen"].eq("Ambos")
        ok_nominal = int((ambos & (posiciones["Dif_Nominal"].abs() <= tol_nominal)).sum())
        dif_nominal = int((ambos & (posiciones["Dif_Nominal"].abs() > tol_nominal)).sum())
        solo_dep = int(posiciones["Estado"].eq("Solo Depositario").sum())
        solo_bbg = int(posiciones["Estado"].eq("Solo Bloomberg").sum())

        # El efectivo total se cuenta aparte, en su propia fila.
        ok_efectivo = 0
        dif_efectivo_total = 0
        if not efectivo_row.empty:
            dif_cash = float(efectivo_row["Dif_Efectivo"].iloc[0])
            if abs(dif_cash) <= tol_efectivo:
                ok_efectivo = 1
            else:
                dif_efectivo_total = 1

        resumen_ok = ok_nominal + ok_efectivo
        resumen_dif = dif_nominal + dif_efectivo_total

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OK", resumen_ok)
        m2.metric("Diferencias", resumen_dif)
        m3.metric("Solo depositario", solo_dep)
        m4.metric("Solo Bloomberg", solo_bbg)

        st.caption(f"Control interno: OK nominal={ok_nominal}, diferencias nominal={dif_nominal}, efectivo OK={ok_efectivo}, efectivo diferencia={dif_efectivo_total}.")
        st.dataframe(recon, use_container_width=True, hide_index=True)

        output = build_excel_output(
            recon,
            dep_data["raw"],
            bbg_data["raw"],
            dep_data["sheet"],
            bbg_data["sheet"],
        )

        st.download_button(
            label="Descargar Excel de conciliación",
            data=output.getvalue(),
            file_name=f"conciliacion_cartera_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error("No he podido generar la conciliación.")
        st.exception(exc)
else:
    st.info("Sube los dos Excel para generar la conciliación.")
