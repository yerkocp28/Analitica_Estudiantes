"""Retención descriptiva de cohortes universitarias desde matrícula pública."""
from __future__ import annotations

import pandas as pd

DIMENSIONS = ["cod_inst", "nomb_inst", "nomb_sede", "area_conocimiento", "nomb_carrera"]
COUNTS = ["n", "misma_carrera", "misma_universidad", "sistema", "sin_mrun",
          "con_cod_carrera"]


def aggregate_retention(current: pd.DataFrame, following: pd.DataFrame, year: int) -> pd.DataFrame:
    """Cuenta inscripciones de ingreso; excluye MRUN ausente del denominador.

    Una persona con varias carreras cuenta en cada inscripción. Duplicados
    exactos se eliminan; continuidad de carrera exige institución y código
    de carrera, pero no sede. Seguimiento considera todo el sistema.
    """
    cohort = current.loc[
        current["nivel_global"].eq("Pregrado")
        & current["tipo_inst_1"].eq("Universidades")
        & current["anio_ing_carr_ori"].eq(year)
    ].copy().drop_duplicates()
    future = following.dropna(subset=["mrun"])
    valid = cohort["mrun"].notna()
    cohort["n"] = valid.astype(int)
    cohort["sin_mrun"] = (~valid).astype(int)
    # `cod_carrera` viene ausente en una quinta parte de la matricula de 2007 y
    # 2008, y bajo el 2% desde 2009. Sin codigo, la fila no puede calzar a
    # nivel de carrera aunque la persona haya seguido en la misma: la
    # continuidad de carrera aparece hundida sin que nadie haya desertado. Se
    # cuenta la cobertura para que el consumidor pueda descartar esos anios en
    # vez de leer la caida como un hecho.
    cohort["con_cod_carrera"] = (valid & cohort["cod_carrera"].notna()).astype(int)
    for name, keys in [("misma_carrera", ["mrun", "cod_inst", "cod_carrera"]),
                       ("misma_universidad", ["mrun", "cod_inst"]),
                       ("sistema", ["mrun"])]:
        available = future.dropna(subset=keys)
        index = pd.MultiIndex.from_frame(available[keys].drop_duplicates())
        cohort[name] = (valid & pd.MultiIndex.from_frame(cohort[keys]).isin(index)).astype(int)
    for col in DIMENSIONS:
        cohort[col] = cohort[col].astype("string").fillna("Sin información")
    result = cohort.groupby(DIMENSIONS, dropna=False, observed=True)[COUNTS].sum().reset_index()
    result.insert(0, "cohorte", year)
    return result


def summarize_retention(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Agrega numeradores y denominadores antes de calcular porcentajes."""
    disponibles = [c for c in COUNTS if c in data.columns]
    result = data.groupby(groups, observed=True, dropna=False)[disponibles].sum().reset_index()
    for col in ("misma_carrera", "misma_universidad", "sistema"):
        result[f"retencion_{col}"] = result[col].div(result["n"].where(result["n"].gt(0)))
    if "con_cod_carrera" in result:
        result["cobertura_cod_carrera"] = result.con_cod_carrera.div(
            result["n"].where(result["n"].gt(0)))
    return result
