import re
import hmac
import hashlib
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "v1_rf_pagares_bonos_2026_06_24"

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# ============================================================
# Utilidades generales
# ============================================================


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "[": " ", "]": " ", "(": " ", ")": " ", "/": " ", "_": " ", "-": " ", ".": " ",
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


def first_non_null(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    for v in values:
        if str(v).strip().lower() not in {"", "nan", "none"}:
            return v
    return values.iloc[0]


def safe_sum(series: pd.Series) -> float:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if nums.empty:
        return np.nan
    return float(nums.sum())


def join_unique(series: pd.Series) -> str:
    vals = []
    for v in series.dropna().astype(str):
        v = v.strip()
        if v and v.lower() not in {"nan", "none"} and v not in vals:
            vals.append(v)
    return " + ".join(vals)


def detect_cash_from_raw(raw: pd.DataFrame, max_rows: int = 20):
    """Detecta una cifra de efectivo/liquidez en las primeras filas de una hoja no tabular."""
    max_rows = min(max_rows, len(raw))
    for r in range(max_rows):
        for c in range(raw.shape[1]):
            label = norm_text(raw.iat[r, c])
            if label in {"liquidez", "efectivo", "cash", "tesoreria"}:
                candidates = []
                for cc in range(c + 1, min(c + 5, raw.shape[1])):
                    val = parse_number(raw.iat[r, cc])
                    if not pd.isna(val):
                        candidates.append(float(val))
                if candidates:
                    # Evita capturar porcentajes/tipos; el importe suele ser la magnitud mayor.
                    return max(candidates, key=lambda x: abs(x))
    return np.nan


def detect_header_row(raw: pd.DataFrame, required_terms, max_rows: int = 40):
    max_rows = min(max_rows, len(raw))
    required_norm = [norm_text(t) for t in required_terms]
    for i in range(max_rows):
        row_norm = [norm_text(x) for x in raw.iloc[i].tolist()]
        row_join = " | ".join(row_norm)
        if any(term in row_join for term in required_norm):
            return i
    return None


def find_column(columns, candidates, required=True):
    columns_norm = {col: norm_text(col) for col in columns}
    candidates_norm = [norm_text(c) for c in candidates]

    # Igualdad exacta normalizada, respetando prioridad de candidatos.
    for cand in candidates_norm:
        for col, col_norm in columns_norm.items():
            if col_norm == cand:
                return col

    # Inclusión normalizada, respetando prioridad de candidatos.
    for cand in candidates_norm:
        for col, col_norm in columns_norm.items():
            if cand and (cand in col_norm or col_norm in cand):
                return col

    if required:
        raise ValueError(f"No se ha encontrado ninguna columna compatible con: {candidates}")
    return None


def find_col_index_in_row(raw: pd.DataFrame, row: int, candidates, required=True):
    candidates_norm = [norm_text(c) for c in candidates]
    row_values = [norm_text(x) for x in raw.iloc[row].tolist()]

    for cand in candidates_norm:
        for idx, value in enumerate(row_values):
            if value == cand:
                return idx
    for cand in candidates_norm:
        for idx, value in enumerate(row_values):
            if cand and value and (cand in value or value in cand):
                return idx

    if required:
        raise ValueError(f"No se ha encontrado columna en la fila {row + 1}: {candidates}")
    return None


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
        if not df.empty:
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
    tipo_col = find_column(df.columns, ["c_tipo_valor", "tipo valor", "c_rf_rv", "tipo activo", "rf rv"], required=False)
    titulos_col = find_column(df.columns, ["titulos", "títulos", "cantidad", "pos", "posicion", "posición", "unidades"], required=False)
    nominal_col = find_column(df.columns, ["nominal", "nominal actual", "nominal div"], required=False)
    efectivo_col = find_column(df.columns, ["efectivo", "valor mercado", "valor de mercado", "market value", "valmrc"], required=False)
    tir_col = find_column(df.columns, ["tir", "ytm", "yield"], required=False)

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
    out["TIR_DEP"] = to_number_series(df[tir_col]) if tir_col else np.nan

    # Solo conciliamos por ISIN. Las líneas sin ISIN se informan como aviso, no se cruzan.
    sin_isin = int(out["ISIN"].isna().sum())
    out = out.dropna(subset=["ISIN"]).copy()

    if out.empty:
        raise ValueError("He encontrado columna de ISIN en el depositario, pero no hay ISINs válidos.")

    grouped = (
        out.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_DEP": first_non_null,
            "Sector_DEP": first_non_null,
            "Tipo_DEP": first_non_null,
            "Titulos_DEP": safe_sum,
            "Nominal_DEP": safe_sum,
            "Efectivo_DEP": safe_sum,
            "TIR_DEP": first_non_null,
        })
    )

    if sin_isin:
        warnings.append(f"El depositario contiene {sin_isin} línea(s) sin ISIN. No se cruzan por posición; revisa si alguna corresponde a depósitos/tesorería.")

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
# Lectura Bonos Bloomberg
# ============================================================


def find_bbg_layout(raw: pd.DataFrame):
    max_rows = min(40, len(raw))
    for r in range(max_rows):
        row_norm = [norm_text(x) for x in raw.iloc[r].tolist()]
        for c, val in enumerate(row_norm):
            if val == "isin":
                return {"header_row": r, "subheader_row": r + 1, "isin_col": c}
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


def parse_bonos_bloomberg(file_bytes: bytes):
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
        raise ValueError("No he podido identificar la tabla de bonos Bloomberg. Necesito una columna con cabecera ISIN.")

    raw, layout = selected
    h = layout["header_row"]
    sh = layout["subheader_row"]
    data_start = sh + 1

    isin_col = layout["isin_col"]
    class_col = 0
    name_col = find_bbg_metric_col(raw, h, sh, ["Nombre corto", "short name", "name"], prefer_cart=True, required=False)
    val_col = find_bbg_metric_col(raw, h, sh, ["ValMrc"], prefer_cart=True, required=False)
    if val_col is None:
        val_col = find_bbg_metric_col(raw, h, sh, ["market value", "valor mercado", "valor de mercado"], prefer_cart=True, required=False)
    nominal_col = find_bbg_metric_col(raw, h, sh, ["Nominal actual", "current amount", "par amount", "nominal"], prefer_cart=True, required=False)
    tir_col = find_bbg_metric_col(raw, h, sh, ["Rendimiento a vmto", "yield to maturity", "ytm", "rendimiento"], prefer_cart=True, required=False)
    precio_col = find_bbg_metric_col(raw, h, sh, ["Cierre", "precio", "price"], prefer_cart=True, required=False)

    if val_col is None:
        warnings.append("No he encontrado ValMrc/valor de mercado en Bloomberg bonos; se dejará vacío.")
    if nominal_col is None:
        warnings.append("No he encontrado Nominal actual en Bloomberg bonos; se dejará vacío.")

    rows = raw.iloc[data_start:].copy()
    out = pd.DataFrame()
    out["ISIN"] = rows.iloc[:, isin_col].apply(normalize_isin)
    out["Nombre_Input"] = rows.iloc[:, class_col].astype(str).str.strip()
    out["Nombre_Corto_Input"] = rows.iloc[:, name_col].astype(str).str.strip() if name_col is not None else ""
    out["Fuente_Input"] = "Bonos Bloomberg"
    out["Titulos_Input"] = np.nan
    out["Nominal_Input"] = to_number_series(rows.iloc[:, nominal_col]) if nominal_col is not None else np.nan
    out["Efectivo_Input"] = to_number_series(rows.iloc[:, val_col]) if val_col is not None else np.nan
    out["Precio_Input"] = to_number_series(rows.iloc[:, precio_col]) if precio_col is not None else np.nan
    out["TIR_Input"] = to_number_series(rows.iloc[:, tir_col]) if tir_col is not None else np.nan

    out = out.dropna(subset=["ISIN"]).copy()
    for col in ["Nominal_Input", "Efectivo_Input", "Precio_Input", "TIR_Input"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid_position = out[["Nominal_Input", "Efectivo_Input"]].fillna(0).abs().sum(axis=1) > 0
    dropped = int((~valid_position).sum())
    if dropped:
        warnings.append(f"He descartado {dropped} línea(s) de bonos Bloomberg con ISIN pero sin nominal/valor.")
    out = out.loc[valid_position].copy()

    grouped = (
        out.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_Input": first_non_null,
            "Nombre_Corto_Input": first_non_null,
            "Fuente_Input": join_unique,
            "Titulos_Input": safe_sum,
            "Nominal_Input": safe_sum,
            "Efectivo_Input": safe_sum,
            "Precio_Input": first_non_null,
            "TIR_Input": first_non_null,
        })
    )

    cash_detected = np.nan
    if val_col is not None:
        for idx in range(data_start, len(raw)):
            label = norm_text(raw.iat[idx, class_col])
            isin = normalize_isin(raw.iat[idx, isin_col])
            if pd.isna(isin) and re.search(r"\b(cash|efectivo|liquidez|tesoreria)\b", label):
                cash_val = parse_number(raw.iat[idx, val_col])
                if not pd.isna(cash_val):
                    cash_detected = float(cash_val)
                    break

    return {"positions": grouped, "raw": raw, "sheet": sheet_selected, "cash": cash_detected, "warnings": warnings}


# ============================================================
# Lectura Pagarés
# ============================================================


def parse_pagares(file_bytes: bytes):
    warnings = []
    xls = pd.ExcelFile(BytesIO(file_bytes))
    selected = None
    sheet_selected = None

    for sheet in xls.sheet_names:
        raw = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, header=None)
        header_row = detect_header_row(raw, ["isin"], max_rows=40)
        if header_row is not None:
            # Evitar falsos positivos: debe tener ISINs reales debajo.
            isin_col = find_col_index_in_row(raw, header_row, ["ISIN"], required=False)
            if isin_col is not None:
                below = raw.iloc[header_row + 1: header_row + 20, isin_col].apply(normalize_isin)
                if below.notna().any():
                    selected = (raw, header_row, isin_col)
                    sheet_selected = sheet
                    break

    if selected is None:
        raise ValueError("No he podido identificar la tabla de pagarés. Necesito una fila de cabecera con la columna ISIN.")

    raw, h, isin_col = selected
    data_start = h + 1

    # En la plantilla de pagarés, el emisor suele estar dos columnas a la derecha del ISIN,
    # el nominal tres columnas a la derecha, y las columnas Precio/Efectivo/TIR sí suelen tener cabecera.
    nombre_col = isin_col + 2 if isin_col + 2 < raw.shape[1] else None
    nominal_col = isin_col + 3 if isin_col + 3 < raw.shape[1] else None
    ytm_col = find_col_index_in_row(raw, h, ["YTM", "TIR", "Yield"], required=False)
    venc_col = find_col_index_in_row(raw, h, ["Vencimiento", "Maturity"], required=False)
    rating_col = find_col_index_in_row(raw, h, ["RATING", "Rating"], required=False)
    precio_col = find_col_index_in_row(raw, h, ["Precios", "Precio", "Price"], required=False)
    efectivo_col = find_col_index_in_row(raw, h, ["Efectivo", "Valor de mercado", "ValMrc"], required=False)
    duracion_col = find_col_index_in_row(raw, h, ["Duración", "Duracion", "Duration"], required=False)

    if nominal_col is None:
        warnings.append("No he podido inferir la columna de nominal en pagarés; se dejará vacía.")
    if efectivo_col is None:
        warnings.append("No he encontrado columna Efectivo en pagarés; se dejará vacía.")

    rows = raw.iloc[data_start:].copy()
    out = pd.DataFrame()
    out["ISIN"] = rows.iloc[:, isin_col].apply(normalize_isin)
    out["Nombre_Input"] = rows.iloc[:, nombre_col].astype(str).str.strip() if nombre_col is not None else ""
    out["Nombre_Corto_Input"] = out["Nombre_Input"]
    out["Fuente_Input"] = "Pagarés"
    out["Titulos_Input"] = np.nan
    out["Nominal_Input"] = to_number_series(rows.iloc[:, nominal_col]) if nominal_col is not None else np.nan
    out["Efectivo_Input"] = to_number_series(rows.iloc[:, efectivo_col]) if efectivo_col is not None else np.nan
    out["Precio_Input"] = to_number_series(rows.iloc[:, precio_col]) if precio_col is not None else np.nan
    out["TIR_Input"] = to_number_series(rows.iloc[:, ytm_col]) if ytm_col is not None else np.nan
    out["Vencimiento_Input"] = rows.iloc[:, venc_col] if venc_col is not None else np.nan
    out["Rating_Input"] = rows.iloc[:, rating_col] if rating_col is not None else ""
    out["Duracion_Input"] = to_number_series(rows.iloc[:, duracion_col]) if duracion_col is not None else np.nan

    out = out.dropna(subset=["ISIN"]).copy()
    for col in ["Nominal_Input", "Efectivo_Input", "Precio_Input", "TIR_Input", "Duracion_Input"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid_position = out[["Nominal_Input", "Efectivo_Input"]].fillna(0).abs().sum(axis=1) > 0
    dropped = int((~valid_position).sum())
    if dropped:
        warnings.append(f"He descartado {dropped} línea(s) de pagarés con ISIN pero sin nominal/efectivo.")
    out = out.loc[valid_position].copy()

    grouped = (
        out.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_Input": first_non_null,
            "Nombre_Corto_Input": first_non_null,
            "Fuente_Input": join_unique,
            "Titulos_Input": safe_sum,
            "Nominal_Input": safe_sum,
            "Efectivo_Input": safe_sum,
            "Precio_Input": first_non_null,
            "TIR_Input": first_non_null,
            "Vencimiento_Input": first_non_null,
            "Rating_Input": first_non_null,
            "Duracion_Input": first_non_null,
        })
    )

    cash_detected = detect_cash_from_raw(raw)

    return {"positions": grouped, "raw": raw, "sheet": sheet_selected, "cash": cash_detected, "warnings": warnings}


# ============================================================
# Conciliación y Excel de salida
# ============================================================


def combine_inputs(bonos_pos: pd.DataFrame, pagares_pos: pd.DataFrame) -> pd.DataFrame:
    common_cols = [
        "ISIN", "Nombre_Input", "Nombre_Corto_Input", "Fuente_Input", "Titulos_Input",
        "Nominal_Input", "Efectivo_Input", "Precio_Input", "TIR_Input",
        "Vencimiento_Input", "Rating_Input", "Duracion_Input",
    ]
    for df in [bonos_pos, pagares_pos]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = np.nan
    combined = pd.concat([bonos_pos[common_cols], pagares_pos[common_cols]], ignore_index=True)
    if combined.empty:
        return combined
    grouped = (
        combined.groupby("ISIN", as_index=False)
        .agg({
            "Nombre_Input": first_non_null,
            "Nombre_Corto_Input": first_non_null,
            "Fuente_Input": join_unique,
            "Titulos_Input": safe_sum,
            "Nominal_Input": safe_sum,
            "Efectivo_Input": safe_sum,
            "Precio_Input": first_non_null,
            "TIR_Input": first_non_null,
            "Vencimiento_Input": first_non_null,
            "Rating_Input": first_non_null,
            "Duracion_Input": first_non_null,
        })
    )
    return grouped


def build_reconciliation(dep_pos, input_pos, cash_dep=np.nan, cash_inputs=np.nan, tol_nominal=1.0, tol_efectivo=1.0):
    merged = dep_pos.merge(input_pos, how="outer", on="ISIN", indicator=True)

    numeric_cols = [
        "Titulos_DEP", "Nominal_DEP", "Efectivo_DEP", "TIR_DEP",
        "Titulos_Input", "Nominal_Input", "Efectivo_Input", "Precio_Input", "TIR_Input", "Duracion_Input",
    ]
    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    text_cols = [
        "Nombre_DEP", "Sector_DEP", "Tipo_DEP", "Nombre_Input", "Nombre_Corto_Input",
        "Fuente_Input", "Vencimiento_Input", "Rating_Input",
    ]
    for col in text_cols:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")

    merged["Dif_Titulos"] = merged["Titulos_DEP"].fillna(0) - merged["Titulos_Input"].fillna(0)
    merged["Dif_Nominal"] = merged["Nominal_DEP"].fillna(0) - merged["Nominal_Input"].fillna(0)
    merged["Dif_Efectivo"] = merged["Efectivo_DEP"].fillna(0) - merged["Efectivo_Input"].fillna(0)

    def estado(row):
        if row["_merge"] == "left_only":
            return "Solo Depositario"
        if row["_merge"] == "right_only":
            return "Solo Inputs"
        return "OK" if abs(row["Dif_Nominal"]) <= tol_nominal else "Diferencia"

    merged["Estado"] = merged.apply(estado, axis=1)
    merged["Origen"] = merged["_merge"].map({"both": "Ambos", "left_only": "Depositario", "right_only": "Inputs"})

    final_cols = [
        "ISIN",
        "Fuente_Input",
        "Nombre_DEP",
        "Nombre_Input",
        "Nombre_Corto_Input",
        "Sector_DEP",
        "Tipo_DEP",
        "Titulos_DEP",
        "Titulos_Input",
        "Dif_Titulos",
        "Nominal_DEP",
        "Nominal_Input",
        "Dif_Nominal",
        "Efectivo_DEP",
        "Efectivo_Input",
        "Dif_Efectivo",
        "Precio_Input",
        "TIR_DEP",
        "TIR_Input",
        "Vencimiento_Input",
        "Rating_Input",
        "Duracion_Input",
        "Estado",
        "Origen",
    ]
    merged = merged[final_cols].sort_values(["Estado", "Fuente_Input", "ISIN"], kind="stable")

    if not pd.isna(cash_dep) or not pd.isna(cash_inputs):
        cash_dep_val = 0.0 if pd.isna(cash_dep) else float(cash_dep)
        cash_in_val = 0.0 if pd.isna(cash_inputs) else float(cash_inputs)
        cash_status = "OK" if abs(cash_dep_val - cash_in_val) <= tol_efectivo else "Diferencia"
        cash_row = pd.DataFrame([{
            "ISIN": "EFECTIVO",
            "Fuente_Input": "Efectivo",
            "Nombre_DEP": "Efectivo",
            "Nombre_Input": "Efectivo",
            "Nombre_Corto_Input": "",
            "Sector_DEP": "Efectivo",
            "Tipo_DEP": "Efectivo",
            "Titulos_DEP": 0.0,
            "Titulos_Input": 0.0,
            "Dif_Titulos": 0.0,
            "Nominal_DEP": 0.0,
            "Nominal_Input": 0.0,
            "Dif_Nominal": 0.0,
            "Efectivo_DEP": cash_dep_val,
            "Efectivo_Input": cash_in_val,
            "Dif_Efectivo": cash_dep_val - cash_in_val,
            "Precio_Input": np.nan,
            "TIR_DEP": np.nan,
            "TIR_Input": np.nan,
            "Vencimiento_Input": "",
            "Rating_Input": "",
            "Duracion_Input": np.nan,
            "Estado": cash_status,
            "Origen": "Efectivo",
        }])
        merged = pd.concat([cash_row, merged], ignore_index=True)

    return merged


def write_raw_sheet(writer, sheet_name: str, raw_df: pd.DataFrame):
    safe_name = sheet_name[:31]
    raw_df.to_excel(writer, sheet_name=safe_name, index=False, header=False)
    worksheet = writer.sheets[safe_name]
    worksheet.freeze_panes(1, 0)
    for col_idx in range(min(raw_df.shape[1], 80)):
        values = raw_df.iloc[:, col_idx].dropna().astype(str).head(120).tolist()
        max_len = min(max([len(str(v)) for v in values] + [10]), 38)
        worksheet.set_column(col_idx, col_idx, max_len + 2)


def build_excel_output(recon, dep_raw, pag_raw, bonos_raw, dep_sheet_name, pag_sheet_name, bonos_sheet_name):
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
        integer_fmt = workbook.add_format({"num_format": "#,##0", "border": 1})
        money_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
        text_fmt = workbook.add_format({"border": 1})

        worksheet.merge_range("A1:X1", "Conciliación RF: depositario vs pagarés + bonos", title_fmt)
        worksheet.write("A2", f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_fmt)
        worksheet.write("A3", f"Depositario: {dep_sheet_name} | Pagarés: {pag_sheet_name} | Bonos: {bonos_sheet_name}", subtitle_fmt)
        worksheet.write("A4", "Criterio: posiciones conciliadas por nominal. El valor de mercado por línea y los títulos son informativos. El efectivo total se controla en fila independiente.", subtitle_fmt)

        posiciones = recon[recon["ISIN"].astype(str).str.upper() != "EFECTIVO"].copy()
        efectivo = recon[recon["ISIN"].astype(str).str.upper() == "EFECTIVO"].copy()
        summary = {
            "OK": int((posiciones["Estado"] == "OK").sum()) + int((efectivo["Estado"] == "OK").sum()) if not efectivo.empty else int((posiciones["Estado"] == "OK").sum()),
            "Diferencias": int((posiciones["Estado"] == "Diferencia").sum()) + int((efectivo["Estado"] == "Diferencia").sum()) if not efectivo.empty else int((posiciones["Estado"] == "Diferencia").sum()),
            "Solo depositario": int((posiciones["Estado"] == "Solo Depositario").sum()),
            "Solo inputs": int((posiciones["Estado"] == "Solo Inputs").sum()),
            "Total líneas": int(len(recon)),
        }
        row = 5
        for k, v in summary.items():
            worksheet.write(row, 0, k, metric_fmt)
            worksheet.write(row, 1, v, integer_fmt)
            row += 1

        for col_num, value in enumerate(recon.columns.values):
            worksheet.write(10, col_num, value, header_fmt)

        worksheet.freeze_panes(11, 0)
        worksheet.autofilter(10, 0, 10 + len(recon), len(recon.columns) - 1)

        widths = {
            "A": 16, "B": 18, "C": 30, "D": 24, "E": 24, "F": 18, "G": 16,
            "H": 14, "I": 14, "J": 14, "K": 16, "L": 16, "M": 16,
            "N": 16, "O": 16, "P": 16, "Q": 14, "R": 12, "S": 12,
            "T": 16, "U": 12, "V": 12, "W": 18, "X": 14,
        }
        for col_letter, width in widths.items():
            worksheet.set_column(f"{col_letter}:{col_letter}", width)
        worksheet.set_column("H:S", 16, money_fmt)
        worksheet.set_column("A:G", 20, text_fmt)
        worksheet.set_column("T:X", 16, text_fmt)

        start_data = 11
        end_data = 10 + len(recon)
        if len(recon) > 0:
            worksheet.conditional_format(start_data, 22, end_data, 22, {
                "type": "text", "criteria": "containing", "value": "OK",
                "format": workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"}),
            })
            worksheet.conditional_format(start_data, 22, end_data, 22, {
                "type": "text", "criteria": "containing", "value": "Diferencia",
                "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"}),
            })
            worksheet.conditional_format(start_data, 22, end_data, 22, {
                "type": "text", "criteria": "containing", "value": "Solo",
                "format": workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#990000"}),
            })

        write_raw_sheet(writer, "Depositario", dep_raw)
        write_raw_sheet(writer, "Pagares", pag_raw)
        write_raw_sheet(writer, "Bonos Bloomberg", bonos_raw)

    output.seek(0)
    return output


# ============================================================
# Autenticación simple
# ============================================================


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_auth_config():
    try:
        auth = st.secrets.get("auth", {})
    except Exception:
        auth = {}
    username = str(auth.get("username", "")).strip()
    password_hash = str(auth.get("password_hash", "")).strip()
    return username, password_hash


def require_login():
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
    st.caption("Introduce usuario y contraseña para acceder al conciliador RF.")
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

st.set_page_config(page_title="Conciliador RF pagarés + bonos", layout="wide")
require_login()

st.title("Conciliador RF: pagarés + bonos contra depositario")
st.success(f"CÓDIGO ACTIVO: {APP_VERSION}")
st.caption("Sube tres Excel: depositario, cartera de pagarés y cartera de bonos Bloomberg. La app cruza por ISIN y genera un Excel con Conciliación, Depositario, Pagarés y Bonos Bloomberg.")
st.caption("Criterio operativo: el estado OK/Diferencia de posiciones se calcula por nominal. El efectivo total se controla en una fila independiente.")

with st.sidebar:
    st.header("Tolerancias")
    tol_nominal = st.number_input("Tolerancia nominal", min_value=0.0, value=1.0, step=1.0)
    tol_efectivo = st.number_input("Tolerancia efectivo total", min_value=0.0, value=1.0, step=10.0)

c1, c2, c3 = st.columns(3)
with c1:
    dep_file = st.file_uploader("Excel depositario", type=["xlsx", "xls"], key="dep")
with c2:
    pag_file = st.file_uploader("Excel cartera de pagarés", type=["xlsx", "xls"], key="pagares")
with c3:
    bonos_file = st.file_uploader("Excel cartera de bonos Bloomberg", type=["xlsx", "xls"], key="bonos")

if dep_file and pag_file and bonos_file:
    try:
        dep_bytes = dep_file.getvalue()
        pag_bytes = pag_file.getvalue()
        bonos_bytes = bonos_file.getvalue()

        dep_data = parse_depositario(dep_bytes)
        pag_data = parse_pagares(pag_bytes)
        bonos_data = parse_bonos_bloomberg(bonos_bytes)
        input_positions = combine_inputs(bonos_data["positions"], pag_data["positions"])

        st.success("Archivos leídos correctamente.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Posiciones depositario", len(dep_data["positions"]))
        m2.metric("Pagarés", len(pag_data["positions"]))
        m3.metric("Bonos Bloomberg", len(bonos_data["positions"]))
        m4.metric("Inputs combinados", len(input_positions))

        warnings = dep_data["warnings"] + pag_data["warnings"] + bonos_data["warnings"]
        if warnings:
            with st.expander("Avisos de lectura"):
                for w in warnings:
                    st.warning(w)

        st.subheader("Efectivo")
        st.caption("Puedes dejar el efectivo detectado en el depositario e introducir a mano el efectivo de los inputs si no viene como línea clara.")
        cash_col1, cash_col2 = st.columns(2)
        default_cash_dep = 0.0 if pd.isna(dep_data["cash"]) else float(dep_data["cash"])
        cash_pag = 0.0 if pd.isna(pag_data.get("cash", np.nan)) else float(pag_data.get("cash"))
        cash_bonos = 0.0 if pd.isna(bonos_data.get("cash", np.nan)) else float(bonos_data.get("cash"))
        default_cash_inputs = cash_pag + cash_bonos
        with cash_col1:
            cash_dep = st.number_input("Efectivo depositario", value=default_cash_dep, step=1000.0, format="%.2f")
        with cash_col2:
            cash_inputs = st.number_input("Efectivo pagarés + bonos", value=default_cash_inputs, step=1000.0, format="%.2f")
        st.caption(f"Efectivo detectado inputs: pagarés={cash_pag:,.2f}; bonos={cash_bonos:,.2f}.")

        recon = build_reconciliation(
            dep_data["positions"], input_positions,
            cash_dep=cash_dep, cash_inputs=cash_inputs,
            tol_nominal=tol_nominal, tol_efectivo=tol_efectivo,
        )

        st.subheader("Resultado de conciliación por nominal")
        posiciones = recon[recon["ISIN"].astype(str).str.upper() != "EFECTIVO"].copy()
        efectivo_row = recon[recon["ISIN"].astype(str).str.upper() == "EFECTIVO"].copy()
        ok_pos = int((posiciones["Estado"] == "OK").sum())
        dif_pos = int((posiciones["Estado"] == "Diferencia").sum())
        solo_dep = int((posiciones["Estado"] == "Solo Depositario").sum())
        solo_inputs = int((posiciones["Estado"] == "Solo Inputs").sum())
        ok_cash = int((efectivo_row["Estado"] == "OK").sum()) if not efectivo_row.empty else 0
        dif_cash = int((efectivo_row["Estado"] == "Diferencia").sum()) if not efectivo_row.empty else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("OK", ok_pos + ok_cash)
        r2.metric("Diferencias", dif_pos + dif_cash)
        r3.metric("Solo depositario", solo_dep)
        r4.metric("Solo inputs", solo_inputs)

        st.caption(f"Control interno: OK nominal={ok_pos}, diferencias nominal={dif_pos}, efectivo OK={ok_cash}, efectivo diferencia={dif_cash}.")
        st.dataframe(recon, use_container_width=True, hide_index=True)

        output = build_excel_output(
            recon,
            dep_data["raw"],
            pag_data["raw"],
            bonos_data["raw"],
            dep_data["sheet"],
            pag_data["sheet"],
            bonos_data["sheet"],
        )

        st.download_button(
            label="Descargar Excel de conciliación",
            data=output.getvalue(),
            file_name=f"conciliacion_rf_pagares_bonos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.error("No he podido generar la conciliación.")
        st.exception(exc)
else:
    st.info("Sube los tres Excel para generar la conciliación.")
