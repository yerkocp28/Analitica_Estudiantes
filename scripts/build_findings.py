"""Exporta los hallazgos a documentacion/datos/: python scripts/build_findings.py

El informe metodológico no calcula: lee estos CSV. El dashboard llama a las
mismas funciones de `modeling/findings.py`. Así ninguna cifra existe dos veces,
que es la única forma de que el informe y la app no se contradigan cuando
cambie una base.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402
from student_analytics.modeling import findings as F  # noqa: E402

log = get_logger("build_findings")
RESULTS = REPO_ROOT / "data" / "results"
OUT = REPO_ROOT / "documentacion" / "datos"
PERFILES = RESULTS / "benchmark_profiles.parquet"


def _guardar(nombre: str, d: pd.DataFrame | None) -> None:
    if d is None or d.empty:
        log.warning("  %-34s sin datos, se omite", nombre)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"{nombre}.csv", index=False, encoding="utf-8")
    log.info("  %-34s %3d filas", nombre, len(d))


def main() -> int:
    perfiles = pd.read_parquet(PERFILES) if PERFILES.exists() else None
    if perfiles is None:
        log.warning("Sin perfiles del benchmark; parte de los hallazgos se omite")

    log.info("Hallazgos")
    _guardar("hallazgo_argumento_central", F.argumento_central(RESULTS))
    _guardar("hallazgo_serie_retencion", F.serie_retencion(RESULTS))
    _guardar("hallazgo_retencion_sede",
             F.retencion_por_sede(RESULTS, area="Administración y Comercio"))
    _guardar("hallazgo_trampas", F.trampas_datos(RESULTS, perfiles))
    _guardar("hallazgo_fuentes", F.aporte_fuentes(RESULTS, perfiles))
    if perfiles is not None:
        _guardar("hallazgo_selectividad", F.posicion_selectividad(perfiles))
        contraste = F.contraste_titulacion(perfiles)
        if contraste is not None:
            contraste = contraste.assign(cohorte=contraste.attrs.get("cohorte"))
        _guardar("hallazgo_contraste_titulacion", contraste)
    log.info("Listo en %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
