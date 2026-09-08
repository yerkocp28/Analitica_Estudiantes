"""Retención descriptiva de cohortes universitarias desde matrícula pública."""
from __future__ import annotations

import pandas as pd

DIMENSIONS = ["cod_inst", "nomb_inst", "nomb_sede", "area_conocimiento", "nomb_carrera"]
COUNTS = ["n", "misma_carrera", "misma_universidad", "sistema", "sin_mrun"]


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
    result = data.groupby(groups, observed=True, dropna=False)[COUNTS].sum().reset_index()
    for col in ("misma_carrera", "misma_universidad", "sistema"):
        result[f"retencion_{col}"] = result[col].div(result["n"].where(result["n"].gt(0)))
    return result
