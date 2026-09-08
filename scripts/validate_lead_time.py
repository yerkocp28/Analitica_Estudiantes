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
    ap.add_argument("--population", choices=["fixed", "rolling"], default="fixed",
                    help="fixed: misma cohorte en todas las semanas (comparacion "
                         "honesta). rolling: elegibilidad recalculada cada semana.")
    ap.add_argument("--compare-population", action="store_true",
                    help="Corre ambos modos y muestra la diferencia")
    ap.add_argument("--out", type=str, default=None,
                    help="Guarda metricas y predicciones para la UI")
    args = ap.parse_args()

    settings = Settings.load()
    data, weeks, total, elig = load_source(args.source, Path(args.data) if args.data else None,
                                           settings)

    metrics, preds = evaluate_lead_time(
        data, scoring_weeks=weeks, weeks_total=total,
        target=args.target, eligibility=elig, population=args.population)

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
    if args.population == "fixed":
        print(f"Poblacion FIJA: {metrics['n_test'].iloc[0]:,} casos y prevalencia "
              f"{metrics['prevalence'].iloc[0]:.1%} en todas las semanas, asi que lo "
              f"unico\nque cambia entre filas es la informacion disponible.")
    else:
        print(f"Poblacion ROLLING: n y prevalencia varian por semana "
              f"({metrics['n_test'].iloc[0]:,} -> {metrics['n_test'].iloc[-1]:,}; "
              f"{metrics['prevalence'].iloc[0]:.1%} -> "
              f"{metrics['prevalence'].iloc[-1]:.1%}).\nLa pendiente mezcla mas "
              f"informacion con una poblacion distinta.")

    if args.compare_population:
        otro = "rolling" if args.population == "fixed" else "fixed"
        m2, _ = evaluate_lead_time(
            data, scoring_weeks=weeks, weeks_total=total,
            target=args.target, eligibility=elig, population=otro)
        comp = metrics[["week", "auc", "lift_20", "n_test", "prevalence"]].merge(
            m2[["week", "auc", "lift_20", "n_test", "prevalence"]],
            on="week", suffixes=(f"_{args.population}", f"_{otro}"))
        print(f"\n{'-' * 78}\nCOMPARACION DE POBLACION\n{'-' * 78}")
        print(f"{'Semana':>7} {'AUC fija':>9} {'AUC roll':>9} {'delta':>7} "
              f"{'n fija':>8} {'n roll':>8} {'prev fija':>10} {'prev roll':>10}")
        a, b = args.population, otro
        fija, roll = (a, b) if a == "fixed" else (b, a)
        for r in comp.itertuples():
            auc_f = getattr(r, f"auc_{fija}")
            auc_r = getattr(r, f"auc_{roll}")
            print(f"{r.week:>7} {auc_f:>9.3f} {auc_r:>9.3f} {auc_r - auc_f:>+7.3f} "
                  f"{getattr(r, f'n_test_{fija}'):>8,} "
                  f"{getattr(r, f'n_test_{roll}'):>8,} "
                  f"{getattr(r, f'prevalence_{fija}'):>9.1%} "
                  f"{getattr(r, f'prevalence_{roll}'):>9.1%}")
        print("Un delta positivo y creciente indica cuanto de la mejora aparente")
        print("venia del cambio de poblacion y no de mas informacion.")

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
