"""Construye los marts de la capa gold: python scripts/build_marts.py

Lee la capa silver y escribe `mart_student_course_week` y `mart_student_week`
por fuente en `data/results/marts/<fuente>/`.

Requiere antes:
    python scripts/build_silver.py

Cada fila del mart contiene **solo lo observable al cierre de esa semana**.
Los targets se adjuntan al mart por asignatura porque sirven para entrenar y
para backtesting, pero nunca entran como feature: esa frontera la impone
`features/builder.py` con su lista de columnas prohibidas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.features.marts import (  # noqa: E402
    mart_student_course_week,
    mart_student_week,
)
from student_analytics.ingestion.layers import fuentes_disponibles, leer  # noqa: E402
from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("build_marts")
SALIDA = REPO_ROOT / "data" / "results" / "marts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fuentes", nargs="*", help="por omisión, todas las de silver")
    ap.add_argument("--sin-targets", action="store_true",
                    help="no adjunta los resultados del período al mart")
    args = ap.parse_args()

    disponibles = fuentes_disponibles()
    if not disponibles:
        print("No hay capa silver. Corre antes: python scripts/build_silver.py")
        return 1
    desconocidas = set(args.fuentes) - set(disponibles)
    if desconocidas:
        print(f"Sin silver para: {', '.join(sorted(desconocidas))}. "
              f"Disponibles: {', '.join(disponibles)}")
        return 2

    for fuente in (args.fuentes or disponibles):
        datos = leer(fuente)
        mcw = mart_student_course_week(datos, con_targets=not args.sin_targets)
        msw = mart_student_week(mcw)
        destino = SALIDA / fuente
        destino.mkdir(parents=True, exist_ok=True)
        mcw.to_parquet(destino / "mart_student_course_week.parquet", index=False)
        msw.to_parquet(destino / "mart_student_week.parquet", index=False)
        peso = sum(f.stat().st_size for f in destino.glob("*.parquet")) / 1e6
        print(f"\n{fuente}")
        print(f"  mart_student_course_week  {len(mcw):>10,} filas x {len(mcw.columns):2d} cols")
        print(f"  mart_student_week         {len(msw):>10,} filas x {len(msw.columns):2d} cols")
        print(f"  -> {destino.relative_to(REPO_ROOT)}  ({peso:.1f} MB)")

    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
