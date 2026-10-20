"""Cálculos de vistas: filtros reales, denominadores y exportaciones coherentes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .models import CAPACITIES, bootstrap_comparison, capture, metrics

DIMENSIONS = {"nomb_sede": "Sede / campus al ingreso", "nomb_carrera": "Carrera al ingreso", "jornada": "Jornada al ingreso"}
DISPLAY_NAMES = {
    "CAMPUS PROVIDENCIA": "Providencia", "CASA CENTRAL (TEMUCO)": "Temuco", "SEDE TALCA": "Talca",
    "CAMPUS EL LLANO": "San Miguel / El Llano", "SEDE SANTIAGO": "Santiago (denominación histórica)",
    "INGENIERIA COMERCIAL": "Ingeniería Comercial", "INGENIERIA EN ADMINISTRACION": "Ingeniería en Administración",
    "INGENIERIA EN CONTROL DE GESTION": "Ingeniería en Control de Gestión",
    "AUDITORIA E INGENIERIA EN CONTROL DE GESTION": "Auditoría e Ing. en Control de Gestión (histórica)",
    "CONTADOR PUBLICO AUDITOR": "Contador Público Auditor (histórica)",
    "INGENIERIA DE EJECUCION EN ADMINISTRACION": "Ingeniería de Ejecución en Administración (histórica)",
}


def display_name(value):
    return "Todos" if value is None else DISPLAY_NAMES.get(value, value)


def filtered(data: pd.DataFrame, selection: dict) -> pd.DataFrame:
    out = data.copy()
    for col, value in selection.items():
        if value is not None:
            out = out[out[col].eq(value)]
    return out


def number(value, digits=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def percent(value):
    return "—" if value is None or pd.isna(value) else number(100 * value, 1) + "%"


def profiles(data: pd.DataFrame, minimum=10) -> pd.DataFrame:
    rows = []
    for label, group in data.groupby("segment", dropna=True):
        n = len(group)
        if n < minimum:
            continue
        row = {"Perfil": f"Perfil {int(label)}", "Estudiantes": n,
               "Participación (%)": 100 * n / len(data),
               "Seguimientos disponibles": int(group.target.notna().sum()),
               "Continuidad UA (%)": 100 * (1 - group.target.mean()) if group.target.notna().sum() >= minimum else np.nan,
               "Cobertura admisión (%)": 100 * group.percentil_admision.notna().mean(),
               "Percentil admisión (0–100)": 100 * group.percentil_admision.mean(),
               "Percentil NEM (0–100)": 100 * group.percentil_nem.mean(),
               "Percentil ranking (0–100)": 100 * group.percentil_ranking.mean(),
               "Duración nominal (años)": group.duracion_nominal_anios.mean()}
        rows.append(row)
    return pd.DataFrame(rows)


def series(data: pd.DataFrame, minimum=10) -> pd.DataFrame:
    rows = []
    for year, group in data.groupby("cohorte"):
        known = group.target.dropna()
        rows.append({"Cohorte de ingreso": int(year), "Estudiantes": len(group),
                     "Seguimientos disponibles": len(known),
                     "Continuidad UA (%)": 100 * (1 - known.mean()) if len(known) >= minimum else np.nan})
    return pd.DataFrame(rows)


def filtered_case(predictions: pd.DataFrame, case: dict, cfg: dict):
    if len(predictions) < cfg["minimum_display_cell"]:
        return None
    d = predictions
    result = metrics(d.target, d.risk_score, case["reference_prevalence"])
    capacities = []
    for q in CAPACITIES:
        slots = int(np.ceil(len(d) * q))
        c = capture(d.target, d.risk_score, q)
        capacities.append({"Capacidad (%)": 100*q, "Cupos": slots,
                           "Captura modelo (%)": 100*c,
                           "Captura regla (%)": 100*capture(d.target, d.historical_rule, q),
                           "Captura azar (%)": 100*slots/len(d) if d.target.sum() else np.nan,
                           "Precisión (%)": 100*c*d.target.sum()/slots if d.target.sum() else 0.0})
    ci = None
    if min(int(d.target.sum()), int(len(d)-d.target.sum())) >= cfg["minimum_class"]:
        ci = bootstrap_comparison(d.target.to_numpy(int), d.risk_score.to_numpy(), d.historical_rule.to_numpy(), cfg)
    truth, predicted = calibration_curve(d.target.astype(int), d.risk_score, n_bins=5, strategy="quantile")
    return {"metrics": result, "capacity": pd.DataFrame(capacities), "uncertainty": ci,
            "calibration": pd.DataFrame({"Predicción": predicted*100, "Frecuencia observada": truth*100})}


def export_csv(table: pd.DataFrame, selection: dict, context: dict) -> bytes:
    """CSV Excel español; solo recibe tablas agregadas, con contexto de filtros."""
    if {"mrun", "student_id", "risk_score", "prediction_id"} & set(table.columns):
        raise ValueError("La exportación de la vista debe ser agregada.")
    out = table.copy()
    for col, value in context.items():
        out[col] = value
    for col, value in selection.items():
        out["Filtro: " + DIMENSIONS.get(col, col)] = value if value is not None else "Todos"
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: "'"+v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
    return out.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
