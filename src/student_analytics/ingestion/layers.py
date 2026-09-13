"""Capas de datos: bronze inmutable, silver canónico, gold consumible.

EL MAPA

    bronze   data/external/   las bases tal como las publica la fuente, sin tocar
             data/interim/    recortes mecánicos de bronze (la capa slim de matrícula)
    silver   data/silver/     esquema canónico, una carpeta por fuente
    gold     data/results/    agregados, perfiles, métricas y predicciones

Bronze no se renombra ni se mueve: son 22 GB y moverlos sería churn sin valor.
Lo que faltaba era **silver materializado**.

POR QUÉ MATERIALIZAR SILVER

Hasta ahora las tablas canónicas se derivaban en memoria en cada corrida. Para
OULAD eso significa releer y reagrupar 10,6 millones de filas cada vez que algo
las necesita, y que no exista un artefacto inspeccionable entre la fuente cruda
y el modelo.

Con silver en disco, los marts y las features leen parquet por columnas, el
adaptador corre una vez, y hay un punto concreto donde mirar cuando un número
no cuadra.

SILVER ES EL PUNTO DONDE SE HACE CUMPLIR EL CONTRATO

`materializar` **valida antes de escribir** y se niega a persistir datos que lo
violen. Esa es la diferencia entre un contrato declarado y uno aplicado: el día
que exista `ingestion/banner.py`, si su grano no es único, los datos no llegan a
silver y el error aparece en la frontera, con nombre y apellido, en vez de
propagarse a los marts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_setup import get_logger
from .contract import Informe, validar_adaptador

log = get_logger(__name__)

RAIZ = Path(__file__).resolve().parents[3] / "data"
BRONZE = RAIZ / "external"
SILVER = RAIZ / "silver"
GOLD = RAIZ / "results"


def ruta_silver(fuente: str, raiz: Path | None = None) -> Path:
    return (raiz or SILVER) / fuente


def materializar(datos: dict[str, pd.DataFrame], fuente: str,
                 raiz: Path | None = None,
                 exigir_contrato: bool = True) -> tuple[Path, Informe]:
    """Escribe el esquema canónico de una fuente en silver.

    `exigir_contrato` es la razón de ser de esta función: con él activo —que es
    lo normal— los datos que violan el contrato **no se escriben**. Se puede
    desactivar para inspeccionar una fuente nueva mientras se la está
    entendiendo, nunca para producir un artefacto que otro consuma.
    """
    informe = validar_adaptador(datos, fuente)
    if exigir_contrato and not informe.valido:
        raise ValueError(
            f"'{fuente}' no cumple el contrato de datos y no se escribe en silver:\n  "
            + "\n  ".join(informe.errores))

    destino = ruta_silver(fuente, raiz)
    destino.mkdir(parents=True, exist_ok=True)
    # Se limpian los parquet previos: un dataset que la fuente dejó de entregar
    # no debe quedar como fantasma de una corrida vieja.
    for viejo in destino.glob("*.parquet"):
        viejo.unlink()
    for nombre, df in datos.items():
        df.to_parquet(destino / f"{nombre}.parquet", index=False)
    log.info("silver/%s: %d tablas, %s filas", fuente, len(datos),
             f"{sum(len(d) for d in datos.values()):,}")
    return destino, informe


def leer(fuente: str, tablas: list[str] | None = None,
         raiz: Path | None = None) -> dict[str, pd.DataFrame]:
    """Lee el esquema canónico de una fuente desde silver."""
    origen = ruta_silver(fuente, raiz)
    if not origen.exists():
        raise FileNotFoundError(
            f"No hay silver para '{fuente}'. Genéralo con scripts/build_silver.py")
    archivos = sorted(origen.glob("*.parquet"))
    if tablas is not None:
        archivos = [p for p in archivos if p.stem in set(tablas)]
    return {p.stem: pd.read_parquet(p) for p in archivos}


def fuentes_disponibles(raiz: Path | None = None) -> list[str]:
    base = raiz or SILVER
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and any(p.glob("*.parquet")))
