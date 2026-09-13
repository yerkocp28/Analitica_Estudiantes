"""Materializa la capa silver: python scripts/build_silver.py

Corre cada adaptador disponible, valida su salida contra
`config/data_contracts.yml` y la escribe en `data/silver/<fuente>/`.

Los datos que violan el contrato **no se escriben**. Esa es la frontera: si el
grano no es único o un tipo no calza, el error aparece aquí, con nombre y
apellido, en vez de propagarse a los marts y aparecer semanas después como
«resultados raros».

El día que exista `ingestion/banner.py` basta con agregarlo a ADAPTADORES.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.ingestion.layers import materializar  # noqa: E402
from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("build_silver")


def _sintetico() -> dict[str, pd.DataFrame] | None:
    carpeta = REPO_ROOT / "data" / "synthetic"
    archivos = sorted(carpeta.glob("*.parquet"))
    if not archivos:
        return None
    return {p.stem: pd.read_parquet(p) for p in archivos}


def _oulad() -> dict[str, pd.DataFrame] | None:
    carpeta = REPO_ROOT / "data" / "external" / "oulad"
    if not carpeta.exists() or not any(carpeta.rglob("*.csv")):
        return None
    from student_analytics.ingestion.oulad import load_oulad
    return load_oulad(carpeta)


# Al agregar Banner o Canvas, entran aqui y nada mas cambia.
ADAPTADORES = {"sintetico": _sintetico, "oulad": _oulad}


def main() -> int:
    ap = argparse.ArgumentParser()
    # Sin `choices`: argparse valida tambien la lista vacia por omision y
    # falla antes de entrar a main.
    ap.add_argument("fuentes", nargs="*",
                    help=f"por omisión todas: {', '.join(sorted(ADAPTADORES))}")
    ap.add_argument("--sin-contrato", action="store_true",
                    help="escribe aunque el contrato falle; solo para explorar")
    args = ap.parse_args()

    desconocidas = set(args.fuentes) - set(ADAPTADORES)
    if desconocidas:
        print(f"Fuente desconocida: {', '.join(sorted(desconocidas))}. "
              f"Disponibles: {', '.join(sorted(ADAPTADORES))}")
        return 2

    problemas = 0
    for fuente in (args.fuentes or sorted(ADAPTADORES)):
        datos = ADAPTADORES[fuente]()
        if datos is None:
            log.warning("%s: sin datos de origen, se omite", fuente)
            continue
        try:
            destino, informe = materializar(datos, fuente,
                                            exigir_contrato=not args.sin_contrato)
        except ValueError as exc:
            log.error("%s", exc)
            problemas += 1
            continue
        print(f"\n{informe.resumen()}")
        for nombre, campos in sorted(informe.huecos.items()):
            print(f"    {nombre}: sin {', '.join(campos)}")
        print(f"    -> {destino.relative_to(REPO_ROOT)}")

    if problemas:
        print(f"\n{problemas} fuente(s) no cumplen el contrato y no se escribieron.")
        return 1
    print("\nSiguiente paso: python scripts/build_marts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
