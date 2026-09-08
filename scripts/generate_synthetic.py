"""Genera el dataset sintetico y lo escribe en data/synthetic/.

Uso:
    python scripts/generate_synthetic.py                 # volumen de config
    python scripts/generate_synthetic.py --students 500  # smoke test
    python scripts/generate_synthetic.py --seed 7 --format csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.config import Settings, load_synthetic_config  # noqa: E402
from student_analytics.logging_setup import get_logger  # noqa: E402
from student_analytics.synthetic.generator import SyntheticGenerator  # noqa: E402

log = get_logger("generate_synthetic")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generador sintetico Student Analytics UA")
    ap.add_argument("--students", type=int, default=None,
                    help="Numero de estudiantes (default: config/synthetic.yml)")
    ap.add_argument("--seed", type=int, default=None, help="Override del seed")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--out", type=str, default=None, help="Directorio de salida")
    args = ap.parse_args()

    settings = Settings.load()
    cfg = load_synthetic_config()

    gen = SyntheticGenerator(cfg, settings.raw, n_students=args.students, seed=args.seed)
    ds = gen.generate()

    out_dir = Path(args.out) if args.out else settings.path("data_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Escribiendo en %s", out_dir)
    for name, df in ds.tables().items():
        path = out_dir / f"{name}.{args.format}"
        if args.format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        log.info("  %-20s %10s filas  %2d cols", name, f"{len(df):,}", df.shape[1])

    _report(ds, settings)
    return 0


def _report(ds, settings: Settings) -> None:
    """Chequeos de sanidad frente a los anclajes de calibracion."""
    out = ds.outcomes
    pass_line = settings.academic["grade_pass"]

    print("\n" + "=" * 62)
    print("SANITY CHECK  (comparar contra anclajes de config/synthetic.yml)")
    print("=" * 62)
    print(f"  Reprobacion por nota (asignatura)   {out['fail_by_grade'].mean():6.1%}")
    print(f"  Reprobacion por inasistencia        {out['fail_by_attendance'].mean():6.1%}")
    print(f"  Reprobacion total (cualquier causa) {out['failed'].mean():6.1%}")

    by_student = out.groupby(["student_id", "term_id"])["failed"].max()
    print(f"  Estudiantes que reprueban >=1 ramo  {by_student.mean():6.1%}")
    print(f"  Nota final promedio                 {out['final_grade'].mean():6.2f}"
          f"   (escala 1.0-7.0, corte {pass_line})")

    # Retencion de primer anio: presente en el termino siguiente al ingreso.
    sm = ds.student_master
    enr = ds.enrollment
    terms_by_student = enr.groupby("student_id")["term_id"].max()
    entry = sm.set_index("student_id")["entry_term"]
    eligible = entry[entry < sm["entry_term"].max()]
    retained = (terms_by_student.reindex(eligible.index).fillna(0) > eligible).mean()
    print(f"  Retencion al termino siguiente      {retained:6.1%}   (ancla SIES 82.8%)")

    print("\n  Reprobacion por dificultad de curso:")
    cm = ds.course_master[["course_id", "difficulty_tier"]]
    merged = out.merge(cm, on="course_id", how="left")
    for tier, rate in merged.groupby("difficulty_tier")["failed"].mean().items():
        print(f"    {tier:20s} {rate:6.1%}")

    print("\n  Reprobacion por perfil latente (debe ordenarse de forma coherente):")
    prof = sm[["student_id", "_latent_profile"]]
    mp = out.merge(prof, on="student_id", how="left")
    for p, rate in mp.groupby("_latent_profile")["failed"].mean().sort_values().items():
        print(f"    {p:22s} {rate:6.1%}")
    print("=" * 62)


if __name__ == "__main__":
    raise SystemExit(main())
