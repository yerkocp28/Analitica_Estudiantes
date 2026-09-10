"""Titulación por cohorte de ingreso: python scripts/build_cohorts.py

Cruza la matrícula histórica (capa slim en `data/interim/matricula`) contra
las bases de titulados por MRUN, y calcula qué proporción de cada cohorte de
ingreso llegó a titularse dentro de su plazo.

Requiere antes:
    python scripts/download_matricula.py
    python scripts/download_titulados.py 2007 ... 2025

Sólo se procesan las cohortes cuyo horizonte cabe en los datos disponibles.
Con titulados hasta 2025 y holgura de 2 años sobre la nominal, la última
cohorte con carreras de 5 años completamente observada es la de 2018.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_analytics.ingestion.cohortes import (  # noqa: E402
    agregar_por_institucion,
    cargar_cohorte,
    indice_titulacion,
    seguir_cohorte,
)

SLIM = ROOT / "data" / "interim" / "matricula"
TITULADOS = ROOT / "data" / "external" / "titulados"
SALIDA = ROOT / "data" / "results" / "cohort_completion.parquet"
DETALLE = ROOT / "data" / "results" / "cohort_tracking.parquet"


def _titulados_por_anio() -> dict[int, Path]:
    salida = {}
    for carpeta in sorted(TITULADOS.glob("titulados_*")):
        try:
            anio = int(carpeta.name.split("_")[-1])
        except ValueError:
            continue
        csvs = sorted(carpeta.rglob("*.csv"))
        if csvs:
            salida[anio] = max(csvs, key=lambda p: p.stat().st_size)
    return salida


def _matricula_por_anio() -> dict[int, Path]:
    salida = {}
    for p in sorted(SLIM.glob("matricula_*.parquet")):
        try:
            salida[int(p.stem.split("_")[-1])] = p
        except ValueError:
            continue
    return salida


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holgura", type=int, default=2,
                    help="años de gracia sobre la duración nominal")
    ap.add_argument("--minimo", type=int, default=30,
                    help="ingresantes observables mínimos por institución")
    ap.add_argument("--min-observables", type=float, default=0.80,
                    help="fracción mínima de la cohorte que debe ser observable")
    args = ap.parse_args()

    matriculas = _matricula_por_anio()
    titulados = _titulados_por_anio()
    if not matriculas:
        raise SystemExit("Faltan los parquet slim. Corre scripts/download_matricula.py")
    if not titulados:
        raise SystemExit("Faltan titulados. Corre scripts/download_titulados.py")

    ultimo = max(titulados)
    print(f"Matrícula slim: {min(matriculas)}–{max(matriculas)} ({len(matriculas)} años)")
    print(f"Titulados:      {min(titulados)}–{ultimo} ({len(titulados)} años)")

    print("\nConstruyendo el índice de titulación por MRUN...", flush=True)
    indice = indice_titulacion(titulados)
    print(f"  {len(indice):,} títulos de pregrado universitario, "
          f"{indice.mrun.nunique():,} personas distintas")

    # Una carrera corta (4 semestres = 2 anios) se observa mucho antes que
    # una de 12 semestres. El corte por cohorte usa la mas corta plausible;
    # la censura fina la aplica `seguir_cohorte` estudiante por estudiante.
    filas, detalle = [], []
    for anio in sorted(matriculas):
        if anio + 2 + args.holgura > ultimo:
            print(f"\ncohorte {anio}: sin horizonte observable, se omite")
            continue
        cohorte = cargar_cohorte(matriculas[anio], anio)
        if cohorte.empty:
            print(f"\ncohorte {anio}: sin ingresantes")
            continue
        seg = seguir_cohorte(cohorte, indice, anio, ultimo, holgura=args.holgura)
        obs = int(seg.observable.sum())
        if obs == 0:
            print(f"\ncohorte {anio}: nadie alcanza el horizonte")
            continue
        # Corregir la censura arregla el TIEMPO, no la COMPOSICION. Cuando
        # quedan pocos anios de seguimiento, los unicos observables son los
        # de carreras cortas, que se titulan a tasas distintas: la cifra deja
        # de describir a la cohorte y pasa a describir a las carreras que
        # alcanzaron a cerrarse. Por eso hay un piso de observabilidad ademas
        # del filtro por estudiante.
        cobertura = obs / len(seg)
        if cobertura < args.min_observables:
            print(f"\ncohorte {anio}: solo {cobertura:.0%} observable "
                  f"(< {args.min_observables:.0%}); el subconjunto es de carreras "
                  f"cortas y no representa a la cohorte, se omite")
            continue
        agg = agregar_por_institucion(seg, minimo=args.minimo)
        filas.append(agg)
        detalle.append(seg.groupby("cohorte", observed=True).agg(
            ingresantes=("mrun", "size"), observables=("observable", "sum"),
            carrera=("misma_carrera_en_plazo", "sum"),
            universidad=("misma_universidad_en_plazo", "sum"),
            sistema=("sistema_en_plazo", "sum")).reset_index())
        pct = 100 * seg.misma_universidad_en_plazo.sum() / obs
        print(f"\ncohorte {anio}: {len(seg):,} ingresantes, {obs:,} observables "
              f"({obs / len(seg):.0%}) -> {pct:.1f}% titulados en su universidad",
              flush=True)

    if not filas:
        raise SystemExit("Ninguna cohorte tiene horizonte observable.")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(filas, ignore_index=True).to_parquet(SALIDA, index=False)
    pd.concat(detalle, ignore_index=True).to_parquet(DETALLE, index=False)
    print(f"\nGuardado en {SALIDA.relative_to(ROOT)}")
    print(f"           y {DETALLE.relative_to(ROOT)}")
    print("Siguiente paso: python scripts/build_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
