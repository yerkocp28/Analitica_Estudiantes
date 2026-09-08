"""Curva de poder predictivo vs anticipacion (doc maestro 68).

Responde la pregunta que el informe ULagos dejo planteada y que define el
valor de este proyecto:

    Si la direccion de carrera revisa al 10% / 20% de sus estudiantes en la
    semana N, que porcentaje de los que van a reprobar alcanza a ver?

DIAGNOSTICO, no produccion. Construye features inline para validar el
generador y dimensionar el problema antes de que exista el mart real
(Sprint 2). La logica de features definitiva vive en src/, no aqui.

Disciplina anti-leakage aplicada (doc maestro 16.2, 74, 75):
  - solo se usan filas con week <= N;
  - las notas se filtran por `available_from_week`, no por `week`: una nota
    cargada con atraso existe en la base pero NO estaba disponible al
    momento del scoring. Filtrar por `week` seria leakage silencioso;
  - la validacion es temporal (semestres antiguos entrenan, recientes
    testean), nunca aleatoria.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from student_analytics.config import Settings  # noqa: E402
from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("validate_lead_time")

KEYS = ["student_id", "course_id", "term_id"]


def build_features_at_week(data: dict[str, pd.DataFrame], week: int) -> pd.DataFrame:
    """Features observables al cierre de la semana `week`. Nada posterior."""
    att = data["attendance_weekly"]
    can = data["canvas_weekly"]
    asg = data["assignments_weekly"]
    gra = data["grades_weekly"]

    att_w = att[att["week"] <= week]
    can_w = can[can["week"] <= week]
    asg_w = asg[asg["week"] <= week]
    # Filtro por disponibilidad real, no por fecha de la evaluacion.
    gra_w = gra[gra["available_from_week"] <= week]

    f = att_w.groupby(KEYS, sort=False).agg(
        attendance_cum=("attendance_rate", "mean"),
        attendance_recorded=("attendance_recorded", "max"),
    ).reset_index()

    # Tendencia: ultimas 4 semanas contra las anteriores. Es lo que separa
    # al que se esta cayendo del que siempre estuvo bajo.
    recent = att_w[att_w["week"] > week - 4]
    f_recent = recent.groupby(KEYS, sort=False)["attendance_rate"].mean()
    f = f.join(f_recent.rename("attendance_recent"), on=KEYS)
    f["attendance_delta"] = f["attendance_recent"] - f["attendance_cum"]

    c = can_w.groupby(KEYS, sort=False).agg(
        canvas_views=("page_views", "mean"),
        canvas_days=("days_active", "mean"),
        canvas_reliable=("canvas_reliable", "max"),
    ).reset_index()
    can_recent = can_w[can_w["week"] > week - 4].groupby(KEYS, sort=False)["page_views"].mean()
    c = c.join(can_recent.rename("canvas_recent"), on=KEYS)
    c["canvas_delta"] = c["canvas_recent"] - c["canvas_views"]
    # En cursos de baja adopcion docente la senal Canvas no es
    # interpretable: se neutraliza en vez de leerse como desenganche.
    for col in ("canvas_views", "canvas_days", "canvas_delta", "canvas_recent"):
        c[col] = c[col].where(c["canvas_reliable"], np.nan)
    f = f.merge(c, on=KEYS, how="left")

    a = asg_w.groupby(KEYS, sort=False).agg(
        assigned=("assigned", "sum"),
        submitted=("submitted", "sum"),
        missing=("missing", "sum"),
        late=("late", "sum"),
    ).reset_index()
    a["submission_rate"] = np.where(a["assigned"] > 0, a["submitted"] / a["assigned"], np.nan)
    f = f.merge(a, on=KEYS, how="left")

    g = gra_w.groupby(KEYS, sort=False).agg(
        mean_grade=("assessment_grade", "mean"),
        last_grade=("assessment_grade", "last"),
        n_grades=("assessment_grade", "size"),
    ).reset_index()
    f = f.merge(g, on=KEYS, how="left")

    # Distincion que importa: un conteo ausente es un CERO real (no hubo
    # tareas, no hubo notas), mientras que un promedio ausente es
    # DESCONOCIDO. Confundirlos es el error clasico en estos pipelines:
    # imputar 0 en `mean_grade` equivale a inventar un 1.0.
    for col in ("assigned", "submitted", "missing", "late", "n_grades"):
        f[col] = f[col].fillna(0)
    f["has_grades"] = (f["n_grades"] > 0).astype(int)

    f = f.merge(
        data["student_master"][["student_id", "historical_pass_rate", "admission_score",
                                "hs_gpa", "campus", "career_code"]],
        on="student_id", how="left")
    f = f.merge(
        data["course_master"][["course_id", "difficulty_tier", "level"]],
        on="course_id", how="left")
    return f


FEATURE_COLS = [
    "attendance_cum", "attendance_delta", "canvas_views", "canvas_days",
    "canvas_delta", "submission_rate", "missing", "late", "mean_grade",
    "last_grade", "n_grades", "has_grades", "historical_pass_rate",
    "admission_score", "hs_gpa", "level",
]


def lift_at(y_true: np.ndarray, score: np.ndarray, pct: float) -> float:
    """Fraccion de casos positivos capturada al priorizar el top `pct`."""
    n = max(1, int(round(len(score) * pct)))
    idx = np.argsort(-score)[:n]
    total = y_true.sum()
    return float(y_true[idx].sum() / total) if total > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--target", default="failed", choices=["failed", "fail_by_grade"])
    args = ap.parse_args()

    settings = Settings.load()
    data_dir = Path(args.data) if args.data else settings.path("data_synthetic")
    names = ["student_master", "course_master", "attendance_weekly", "grades_weekly",
             "canvas_weekly", "assignments_weekly", "outcomes"]
    data = {n: pd.read_parquet(data_dir / f"{n}.parquet") for n in names}
    outcomes = data["outcomes"]

    terms = sorted(outcomes["term_id"].unique())
    train_terms, test_terms = terms[:-2], terms[-2:]
    log.info("Validacion temporal: entrena %s, testea %s", train_terms, test_terms)

    scoring_weeks = settings.academic["scoring_weeks"]
    print("\n" + "=" * 74)
    print(f"PODER PREDICTIVO vs ANTICIPACION   target={args.target}   "
          f"prevalencia={outcomes[args.target].mean():.1%}")
    print("=" * 74)
    print(f"{'Semana':>7} {'AUC':>7} {'Top 5%':>9} {'Top 10%':>9} {'Top 20%':>9} "
          f"{'Semanas restantes':>19}")
    print("-" * 74)

    weeks_total = settings.academic["weeks_per_term"]
    for week in scoring_weeks:
        feats = build_features_at_week(data, week)
        df = feats.merge(outcomes[KEYS + [args.target]], on=KEYS, how="inner")

        tr = df[df["term_id"].isin(train_terms)]
        te = df[df["term_id"].isin(test_terms)]

        # Imputacion por mediana de TRAIN (nunca de test: seria leakage).
        # Los NaN de Canvas no confiable se imputan a proposito: significan
        # "sin informacion", no "sin actividad".
        med = tr[FEATURE_COLS].median()
        # En la semana 3 aun no existe ninguna evaluacion, asi que la
        # mediana de `mean_grade` es NaN y no imputa nada. Se rellena con
        # un valor neutro y la bandera `has_grades` le dice al modelo que
        # esa columna no lleva informacion todavia.
        med = med.fillna(0.0)
        Xtr = tr[FEATURE_COLS].fillna(med).to_numpy()
        Xte = te[FEATURE_COLS].fillna(med).to_numpy()
        ytr = tr[args.target].to_numpy()
        yte = te[args.target].to_numpy()

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]

        auc = roc_auc_score(yte, p)
        print(f"{week:>7} {auc:>7.3f} {lift_at(yte, p, 0.05):>8.1%} "
              f"{lift_at(yte, p, 0.10):>8.1%} {lift_at(yte, p, 0.20):>8.1%} "
              f"{weeks_total - week:>18}")

    print("=" * 74)
    print("Lectura: 'Top 20%' = de los estudiantes que efectivamente reprobaron,")
    print("que fraccion queda dentro del 20% priorizado por el modelo esa semana.")
    print("Baseline aleatorio = el mismo porcentaje que se prioriza (20% -> 20%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
