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
# que usa el sistema de admisión y el más comparable entre carreras. En la
# PAES viene calculado; se verificó que es exactamente (CLEC_MAX + MATE1_MAX)/2.
SCORE = "PROMEDIO_CM_MAX"

# LA SERIE TIENE UN CORTE DE ESCALA EN 2023
#
# 2021-2022 fue la Prueba de Transición (PDT): rango 150-850, media 500, una
# sola prueba de matemática. Desde 2023 la PAES usa 100-1000 y separa M1 de M2.
# Los puntajes NO son comparables entre ambos lados: un promedio institucional
# salta unos 100 puntos entre 2022 y 2023 sin que haya cambiado nada real.
#
# Por eso se reconstruye el promedio CM para la PDT con la misma definición
# —el promedio de lectora y matemática, tomando el máximo entre el puntaje del
# año y el del anterior, como hace la PAES— y además se calcula el PERCENTIL
# nacional dentro de cada cohorte, que sí cruza el corte.
PARTES = {
    "PAES": {"clec": ["CLEC_MAX"], "mate": ["MATE1_MAX"]},
    "PDT": {"clec": ["CLEC_ACTUAL", "CLEC_ANTERIOR"],
            "mate": ["MATE_ACTUAL", "MATE_ANTERIOR"]},
}
COLUMNS = ["MRUN", SCORE, "PTJE_NEM", "PTJE_RANKING", *(
    c for partes in PARTES.values() for cols in partes.values() for c in cols)]

SELECTIVITY_LABELS = {
    "paes_promedio": "Puntaje de admisión promedio de la cohorte",
    "paes_p25": "Puntaje de admisión percentil 25 de la cohorte",
    "paes_p75": "Puntaje de admisión percentil 75 de la cohorte",
    "paes_rango_intercuartil": "Dispersión de puntajes (p75 − p25)",
    "paes_percentil_promedio": "Percentil nacional promedio de la cohorte",
    "nem_promedio": "Puntaje NEM promedio",
    "ranking_promedio": "Puntaje ranking promedio",
    "paes_cobertura": "Cohorte con puntaje de admisión",
    "paes_instrumento": "Prueba rendida por esa cohorte (PDT o PAES)",
}


def _decimal_comma(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")


def _mejor(df: pd.DataFrame, columnas: list[str]) -> pd.Series | None:
    """El mejor puntaje entre las columnas presentes; 0 es ausencia, no cero.

    La PAES admite arrastrar el puntaje del año anterior si fue mejor, y sus
    columnas `_MAX` ya vienen resueltas. Para la PDT hay que hacerlo aquí,
    entre el puntaje del año y el del anterior, que están en la misma escala.
    """
    presentes = [c for c in columnas if c in df.columns]
    if not presentes:
        return None
    valores = pd.concat([_decimal_comma(df[c]).where(lambda s: s > 0)
                         for c in presentes], axis=1)
    return valores.max(axis=1)


def load_scores(path: str | Path) -> pd.DataFrame:
    """Puntajes de admisión por MRUN, con los ceros tratados como ausencia.

    Sirve tanto para la PAES (2023 en adelante), que trae el promedio ya
    calculado, como para la PDT (2021-2022), donde hay que reconstruirlo. La
    columna `instrumento` deja registrado cuál se leyó, porque sus escalas no
    son comparables.
    """
    path = Path(path)
    log.info("Leyendo puntajes de admisión (%s)", path.name)
    head = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=1)
    cols = [c for c in COLUMNS if c in head.columns]
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=cols,
                     dtype=str, low_memory=False)

    out = pd.DataFrame({"mrun": pd.to_numeric(df.MRUN, errors="coerce")})
    if SCORE in df.columns:
        instrumento = "PAES"
        out["paes"] = _decimal_comma(df[SCORE]).where(lambda s: s > 0)
    else:
        clec = _mejor(df, PARTES["PDT"]["clec"])
        mate = _mejor(df, PARTES["PDT"]["mate"])
        if clec is None or mate is None:
            raise ValueError(f"{path.name} no trae {SCORE} ni las pruebas de la PDT")
        instrumento = "PDT"
        # Misma definicion que PROMEDIO_CM_MAX, verificada contra la PAES: el
        # promedio simple de lectora y matematica. Exige ambas, como el
        # sistema de admision.
        out["paes"] = (clec + mate) / 2
    out["instrumento"] = instrumento

    for origen, destino in [("PTJE_NEM", "nem"), ("PTJE_RANKING", "ranking")]:
        if origen in df.columns:
            valores = _decimal_comma(df[origen])
            # En estas bases el 0 significa "no rindió", no "obtuvo cero".
            out[destino] = valores.where(valores > 0)

    out = out.dropna(subset=["mrun"]).drop_duplicates("mrun")
    # Percentil nacional entre quienes rindieron ese año. Es lo unico que
    # cruza el cambio de escala de 2023: 150-850 y 100-1000 no se comparan,
    # pero "quedar sobre el 70% del pais" significa lo mismo en ambos.
    out["percentil"] = out.paes.rank(pct=True)
    log.info("  %s personas con registro %s, %s con puntaje",
             f"{len(out):,}", instrumento, f"{int(out.paes.notna().sum()):,}")
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
    if "percentil" in unido:
        # Comparable entre cohortes aunque cambie la escala de la prueba.
        resultado["paes_percentil_promedio"] = grupo.percentil.mean()
    if "nem" in unido:
        resultado["nem_promedio"] = grupo.nem.mean()
    if "ranking" in unido:
        resultado["ranking_promedio"] = grupo.ranking.mean()
    resultado["paes_rango_intercuartil"] = resultado.paes_p75 - resultado.paes_p25
    if "instrumento" in unido:
        resultado["paes_instrumento"] = unido.instrumento.dropna().iloc[0] \
            if unido.instrumento.notna().any() else pd.NA

    # Con muy pocos rendidores la media es ruido. Se anula el indicador pero
    # se conserva la cobertura, que es justamente la señal de que la
    # institución recibe a su cohorte por otras vías.
    escaso = (grupo.paes.count() < min_takers) | (resultado.paes_cobertura < min_coverage)
    for col in ("paes_promedio", "paes_p25", "paes_p75", "paes_rango_intercuartil",
                "paes_percentil_promedio", "nem_promedio", "ranking_promedio"):
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
