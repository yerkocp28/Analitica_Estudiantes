"""Construccion de features al cierre de una semana de scoring.

Grano de salida: student x course x term, con solo informacion observable
hasta la semana N inclusive.

Este modulo es AGNOSTICO A LA FUENTE. Recibe el conjunto canonico de
tablas y no sabe si vienen del generador sintetico, de OULAD o de
Banner/Canvas. Esa es toda la apuesta arquitectonica del proyecto: si el
mismo codigo corre sobre tres fuentes distintas, correra sobre la cuarta.

Disciplina anti-leakage (doc maestro 16.2, 74, 75):
  - solo filas con week <= N;
  - las notas se filtran por `available_from_week`, nunca por `week`: una
    nota cargada con atraso existe en la base pero no estaba disponible al
    momento del scoring;
  - ninguna columna del outcome entra al set de features;
  - las tablas ausentes producen features ausentes, no ceros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["student_id", "course_id", "term_id"]

# Ventana para las features de tendencia. Es lo que separa a quien se esta
# cayendo de quien siempre estuvo bajo: mismo nivel, trayectoria opuesta.
TREND_WINDOW = 4

# Contadores: ausencia significa cero real (no hubo tareas, no hubo notas).
COUNT_COLS = ("assigned", "submitted", "missing", "late", "n_grades")

FEATURE_COLS = [
    "attendance_cum", "attendance_delta",
    "activity_mean", "activity_days", "activity_delta",
    "submission_rate", "missing", "late",
    "mean_grade", "last_grade", "n_grades", "has_grades",
    "historical_pass_rate", "prior_attempts", "credits_load",
]

# Columnas que jamas pueden entrar como feature.
FORBIDDEN = {
    "final_grade", "final_attendance", "fail_by_grade", "fail_by_attendance",
    "failed", "withdrawn", "final_result", "_p_fail_true",
    "_latent_profile", "_latent_ability",
}


def _agg_window(df: pd.DataFrame, week: int, col: str, name: str) -> pd.Series:
    """Promedio de `col` en las ultimas TREND_WINDOW semanas hasta `week`."""
    recent = df[df["week"] > week - TREND_WINDOW]
    return recent.groupby(KEYS, sort=False)[col].mean().rename(name)


def build_features_at_week(data: dict[str, pd.DataFrame], week: int) -> pd.DataFrame:
    """Features observables al cierre de la semana `week`.

    `data` puede omitir tablas (OULAD no tiene asistencia, por ejemplo).
    Las features correspondientes salen como NaN, no como cero.
    """
    base = data["enrollment"][KEYS].drop_duplicates()
    f = base.copy()

    # --- Asistencia -------------------------------------------------
    if "attendance_weekly" in data:
        att = data["attendance_weekly"]
        att_w = att[att["week"] <= week]
        a = att_w.groupby(KEYS, sort=False).agg(
            attendance_cum=("attendance_rate", "mean"),
            attendance_recorded=("attendance_recorded", "max"),
        ).reset_index()
        a = a.join(_agg_window(att_w, week, "attendance_rate", "attendance_recent"), on=KEYS)
        a["attendance_delta"] = a["attendance_recent"] - a["attendance_cum"]
        f = f.merge(a.drop(columns=["attendance_recent"]), on=KEYS, how="left")
    else:
        # Fuente sin asistencia. Explicito, para que sea visible en el
        # analisis en vez de silenciosamente imputado.
        f["attendance_cum"] = np.nan
        f["attendance_delta"] = np.nan
        f["attendance_recorded"] = False

    # --- Actividad en la plataforma ---------------------------------
    act = data["activity_weekly"]
    act_w = act[act["week"] <= week]
    c = act_w.groupby(KEYS, sort=False).agg(
        activity_mean=("page_views", "mean"),
        activity_days=("days_active", "mean"),
        activity_reliable=("activity_reliable", "max"),
    ).reset_index()
    c = c.join(_agg_window(act_w, week, "page_views", "activity_recent"), on=KEYS)
    c["activity_delta"] = c["activity_recent"] - c["activity_mean"]
    # Donde el docente no usa la plataforma, la senal no es interpretable.
    # Se neutraliza en vez de leerse como desenganche (doc maestro 73).
    for col in ("activity_mean", "activity_days", "activity_delta"):
        c[col] = c[col].where(c["activity_reliable"].astype(bool), np.nan)
    f = f.merge(c.drop(columns=["activity_recent"]), on=KEYS, how="left")

    # --- Entregas ---------------------------------------------------
    asg = data["assignments_weekly"]
    asg_w = asg[asg["week"] <= week]
    g = asg_w.groupby(KEYS, sort=False).agg(
        assigned=("assigned", "sum"), submitted=("submitted", "sum"),
        missing=("missing", "sum"), late=("late", "sum"),
    ).reset_index()
    g["submission_rate"] = np.where(g["assigned"] > 0, g["submitted"] / g["assigned"], np.nan)
    f = f.merge(g, on=KEYS, how="left")

    # --- Notas parciales --------------------------------------------
    gra = data["grades_weekly"]
    # Filtro por disponibilidad real, no por fecha de la evaluacion.
    gra_w = gra[gra["available_from_week"] <= week]
    q = gra_w.groupby(KEYS, sort=False).agg(
        mean_grade=("assessment_grade", "mean"),
        last_grade=("assessment_grade", "last"),
        n_grades=("assessment_grade", "size"),
    ).reset_index()
    f = f.merge(q, on=KEYS, how="left")

    # --- Contexto del estudiante y del curso ------------------------
    sm = data["student_master"]
    cols = [c for c in ("student_id", "historical_pass_rate", "prior_attempts",
                        "credits_load", "campus", "career_code") if c in sm.columns]
    f = f.merge(sm[cols], on="student_id", how="left")
    for col in ("historical_pass_rate", "prior_attempts", "credits_load"):
        if col not in f.columns:
            f[col] = np.nan

    cm = data["course_master"]
    cols = [c for c in ("course_id", "difficulty_tier", "level") if c in cm.columns]
    if len(cols) > 1:
        f = f.merge(cm[cols], on="course_id", how="left")

    # Un conteo ausente es un cero real; un promedio ausente es DESCONOCIDO.
    # Confundirlos es el error clasico: imputar 0 en `mean_grade` equivale a
    # inventar la peor nota posible.
    for col in COUNT_COLS:
        if col in f.columns:
            f[col] = f[col].fillna(0)
    f["has_grades"] = (f["n_grades"] > 0).astype(int)

    leaked = set(f.columns) & FORBIDDEN
    if leaked:
        raise AssertionError(f"Columnas de resultado en las features: {sorted(leaked)}")
    return f


def feature_matrix(train: pd.DataFrame, test: pd.DataFrame,
                   cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Matrices numericas, imputando con la mediana de TRAIN.

    Imputar con la mediana del conjunto completo seria leakage. En la
    semana 3 aun no hay evaluaciones y la mediana de `mean_grade` es NaN:
    se rellena con un valor neutro y la bandera `has_grades` le indica al
    modelo que esa columna todavia no lleva informacion.
    """
    cols = cols or [c for c in FEATURE_COLS if c in train.columns]
    usable = [c for c in cols if train[c].notna().any()]
    med = train[usable].median().fillna(0.0)
    Xtr = train[usable].fillna(med).fillna(0.0).to_numpy(dtype=float)
    Xte = test[usable].fillna(med).fillna(0.0).to_numpy(dtype=float)
    return Xtr, Xte, usable
