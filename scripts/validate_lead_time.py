"""Curva de poder predictivo vs anticipacion, sobre cualquier fuente.

    python scripts/validate_lead_time.py --source synthetic
    python scripts/validate_lead_time.py --source oulad --target failed

El mismo pipeline corre sobre datos sinteticos y sobre OULAD sin cambiar
una linea de la logica de features ni de evaluacion: solo cambia el
adaptador de ingesta. Esa es la propiedad que hara barata la migracion a
Banner/Canvas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.config import Settings  # noqa: E402
from student_analytics.ingestion.oulad import eligible_at_week, load_oulad  # noqa: E402
from student_analytics.modeling.lead_time import evaluate_lead_time  # noqa: E402

SYNTHETIC_TABLES = [
    "student_master", "course_master", "enrollment", "attendance_weekly",
    "grades_weekly", "activity_weekly", "assignments_weekly", "outcomes",
]

# OULAD son modulos de ~9 meses, no semestres de 18 semanas. Se scorea en
# puntos comparables en PROPORCION del curso, no en numero de semana.
OULAD_WEEKS = [4, 8, 12, 16, 20, 26]
OULAD_LENGTH = 39


def load_source(source: str, path: Path | None, settings: Settings):
    if source == "synthetic":
        d = path or settings.path("data_synthetic")
        data = {n: pd.read_parquet(d / f"{n}.parquet") for n in SYNTHETIC_TABLES}
        return data, settings.academic["scoring_weeks"], \
            settings.academic["weeks_per_term"], None
    d = path or (REPO_ROOT / "data" / "external" / "oulad")
    return load_oulad(d), OULAD_WEEKS, OULAD_LENGTH, eligible_at_week


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "oulad"], default="synthetic")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--target", default="failed")
    ap.add_argument("--out", type=str, default=None,
                    help="Guarda metricas y predicciones para la UI")
    args = ap.parse_args()

    settings = Settings.load()
    data, weeks, total, elig = load_source(args.source, Path(args.data) if args.data else None,
                                           settings)

    metrics, preds = evaluate_lead_time(
        data, scoring_weeks=weeks, weeks_total=total,
        target=args.target, eligibility=elig)

    print("\n" + "=" * 78)
    print(f"PODER PREDICTIVO vs ANTICIPACION   fuente={args.source}   target={args.target}")
    if args.source == "oulad":
        print("OULAD no tiene asistencia (educacion a distancia): estos numeros son un PISO.")
    print("=" * 78)
    print(f"{'Semana':>7} {'% curso':>8} {'AUC':>7} {'Brier':>7} "
          f"{'Top 5%':>8} {'Top 10%':>8} {'Top 20%':>8} {'Restan':>7}")
    print("-" * 78)
    for r in metrics.itertuples():
        print(f"{r.week:>7} {r.week / total:>7.0%} {r.auc:>7.3f} {r.brier:>7.3f} "
              f"{r.lift_05:>7.1%} {r.lift_10:>7.1%} {r.lift_20:>7.1%} "
              f"{r.weeks_remaining:>7}")
    print("=" * 78)
    print("Top 20% = de quienes efectivamente reprobaron, que fraccion queda dentro")
    print("del 20% priorizado esa semana. Baseline aleatorio = 20%.")
    print(f"Prevalencia en test: {metrics['prevalence'].iloc[0]:.1%}   "
          f"n_test={metrics['n_test'].iloc[0]:,}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(out / f"metrics_{args.source}.parquet", index=False)
        pd.concat(preds.values(), ignore_index=True).to_parquet(
            out / f"predictions_{args.source}.parquet", index=False)
        print(f"\nGuardado en {out} (usable por la UI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
