"""Selectividad de admisión, cruzando PAES con la matrícula por MRUN.

Es la tercera limitación que el benchmark declaraba. No requiere descargas
nuevas: usa las bases PAES que ya bajó `scripts/download_mineduc.py` y la
cohorte de ingreso que `build_benchmark` ya tiene en memoria.

QUÉ MIDE Y QUÉ NO

Mide el puntaje de quienes ingresaron, no el puntaje de corte ni la razón
postulantes/vacantes. Una universidad puede tener media alta porque
selecciona, o porque atrae a un público que ya venía con puntajes altos.
Selectividad aquí es descriptiva.

LA COBERTURA ES LA CLAVE

Solo rinde PAES una parte de quienes entran a la universidad: los ingresos
por vías especiales, extranjeros, mayores de edad o continuidad de estudios
no la rinden. La media se calcula sobre quienes sí, y por eso se expone
`paes_cobertura`: en una institución con cobertura baja, la media describe a
una minoría de su cohorte y no debe leerse como el perfil de ingreso.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

# Promedio oficial de Competencia Lectora y Matemática M1: es el indicador
# que usa el sistema de admisión y el más comparable entre carreras.
SCORE = "PROMEDIO_CM_MAX"
COLUMNS = ["MRUN", SCORE, "PTJE_NEM", "PTJE_RANKING"]

SELECTIVITY_LABELS = {
    "paes_promedio": "Puntaje PAES promedio de la cohorte",
    "paes_p25": "PAES percentil 25 de la cohorte",
    "paes_p75": "PAES percentil 75 de la cohorte",
    "paes_rango_intercuartil": "Dispersión de puntajes (p75 − p25)",
    "nem_promedio": "Puntaje NEM promedio",
    "ranking_promedio": "Puntaje ranking promedio",
    "paes_cobertura": "Cohorte con puntaje PAES",
}


def _decimal_comma(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")


def load_scores(path: str | Path) -> pd.DataFrame:
    """Puntajes PAES por MRUN, con los ceros tratados como ausencia."""
    path = Path(path)
    log.info("Leyendo puntajes PAES (%s)", path.name)
    head = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=1)
    cols = [c for c in COLUMNS if c in head.columns]
    if SCORE not in cols:
        raise ValueError(f"{path.name} no trae la columna {SCORE}")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=cols,
                     dtype=str, low_memory=False)
    out = pd.DataFrame({"mrun": pd.to_numeric(df.MRUN, errors="coerce")})
    for origen, destino in [(SCORE, "paes"), ("PTJE_NEM", "nem"),
                            ("PTJE_RANKING", "ranking")]:
        if origen in df.columns:
            valores = _decimal_comma(df[origen])
            # En estas bases el 0 significa "no rindió", no "obtuvo cero".
            out[destino] = valores.where(valores > 0)
    out = out.dropna(subset=["mrun"]).drop_duplicates("mrun")
    log.info("  %s personas con registro PAES", f"{len(out):,}")
    return out


def institutional_selectivity(cohort: pd.DataFrame, scores: pd.DataFrame,
                              min_takers: int = 30,
                              min_coverage: float = .10) -> pd.DataFrame:
    """Selectividad por institución sobre la cohorte de ingreso.

    `cohort` debe traer `mrun` y `cod_inst` a nivel de inscripción. Una
    persona inscrita en dos carreras de la misma universidad cuenta una vez:
    lo que se describe es el perfil de las personas que ingresaron, no el de
    sus matrículas.

    `min_takers` y `min_coverage` son el piso bajo el cual el promedio deja
    de ser informativo y se anula. Son parámetros y no constantes porque el
    umbral razonable depende del tamaño de las cohortes que se comparen.
    """
    base = cohort[["mrun", "cod_inst"]].copy()
    base["mrun"] = pd.to_numeric(base.mrun, errors="coerce")
    base["cod_inst"] = base.cod_inst.astype(str)
    base = base.dropna(subset=["mrun"]).drop_duplicates(["mrun", "cod_inst"])

    unido = base.merge(scores, on="mrun", how="left")
    grupo = unido.groupby("cod_inst")
    resultado = pd.DataFrame({
        "paes_promedio": grupo.paes.mean(),
        "paes_p25": grupo.paes.quantile(.25),
        "paes_p75": grupo.paes.quantile(.75),
        "paes_cobertura": grupo.paes.apply(lambda s: s.notna().mean()),
        "personas_cohorte": grupo.size(),
    })
    if "nem" in unido:
        resultado["nem_promedio"] = grupo.nem.mean()
    if "ranking" in unido:
        resultado["ranking_promedio"] = grupo.ranking.mean()
    resultado["paes_rango_intercuartil"] = resultado.paes_p75 - resultado.paes_p25

    # Con muy pocos rendidores la media es ruido. Se anula el indicador pero
    # se conserva la cobertura, que es justamente la señal de que la
    # institución recibe a su cohorte por otras vías.
    escaso = (grupo.paes.count() < min_takers) | (resultado.paes_cobertura < min_coverage)
    for col in ("paes_promedio", "paes_p25", "paes_p75", "paes_rango_intercuartil",
                "nem_promedio", "ranking_promedio"):
        if col in resultado:
            resultado.loc[escaso, col] = np.nan
    if escaso.any():
        log.info("  %s instituciones sin selectividad utilizable (pocos rendidores)",
                 int(escaso.sum()))
    return resultado.reset_index()


def attach_selectivity(profiles: pd.DataFrame, selectivity: pd.DataFrame) -> pd.DataFrame:
    out = profiles.copy()
    out["cod_inst"] = out.cod_inst.astype(str)
    columnas = [c for c in selectivity.columns if c != "personas_cohorte"]
    return out.merge(selectivity[columnas], on="cod_inst", how="left")
