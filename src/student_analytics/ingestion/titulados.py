"""Indicadores de titulación de la promoción que egresa (Mineduc).

Fuente: https://datosabiertos.mineduc.cl/titulados-en-educacion-superior/

QUÉ DESCRIBEN ESTOS INDICADORES Y QUÉ NO

Quien se titula en 2024 ingresó alrededor de 2017-2019, así que estas cifras
describen a la **promoción que egresa**, no a la cohorte de ingreso que
retrata el perfil del benchmark. De la cohorte 2024 todavía no se ha titulado
nadie.

Por eso NO son una tasa de titulación: no dicen qué proporción de quienes
entraron llegará a titularse. Dicen cuánto se demoró y qué proporción cumplió
el plazo entre quienes SÍ se titularon. Calcular una tasa por cohorte exige
matrícula histórica y seguimiento longitudinal.

Todas estas variables son descriptivas: no participan en la distancia entre
universidades, porque describen un período distinto del que define los pares.

TRES TRAMPAS DE ESTA BASE

1. `anio_ing_carr_ori` usa 1900 como centinela de dato ausente, y es el 10%
   de los titulados de pregrado universitario. Sin filtrarlo la duración
   media salta de 5 a 17 años, con máximos de 124.
2. Los planes de continuidad reconocen estudios previos: su sobreduración
   media es 3,5 años contra 0,6 del plan regular. Se miden aparte.
3. La base mezcla pregrado, postítulo y posgrado, y los tres tipos de
   institución.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

COLUMNS = ["cat_periodo", "cod_inst", "nomb_inst", "nivel_global", "tipo_inst_1",
           "anio_ing_carr_ori", "dur_total_carr", "tipo_plan_carr", "area_conocimiento"]

COMPLETION_LABELS = {
    "titulados_total": "Titulados de pregrado en el año",
    "titulacion_duracion_mediana": "Duración real mediana hasta titularse (años)",
    "titulacion_sobreduracion": "Sobreduración mediana sobre la nominal (años)",
    "titulacion_oportuna": "Titulados dentro de la duración nominal",
    "titulacion_oportuna_holgada": "Titulados dentro de la nominal más un año",
    "titulacion_cobertura": "Titulados con duración calculable",
}


def load_graduates(path: str | Path, level: str = "Pregrado",
                   institution: str = "Universidades") -> pd.DataFrame:
    """Titulados del nivel y tipo de institución pedidos."""
    path = Path(path)
    log.info("Leyendo titulados (%s)", path.name)
    df = pd.read_csv(path, sep=";", encoding="utf-8",
                     usecols=lambda c: c in COLUMNS, dtype=str, low_memory=False)
    df = df.loc[df.nivel_global.eq(level) & df.tipo_inst_1.eq(institution)].copy()
    for c in ("cat_periodo", "anio_ing_carr_ori", "dur_total_carr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.info("  %s titulados de %s en %s", f"{len(df):,}", level.lower(), institution.lower())
    return df


def institutional_completion(graduates: pd.DataFrame, min_entry_year: int = 1980,
                             max_years: int = 20, min_graduates: int = 30,
                             regular_only: bool = True) -> pd.DataFrame:
    """Indicadores de titulación por institución.

    `min_entry_year` descarta el centinela 1900 y los años imposibles;
    `max_years` acota trayectorias que no describen un paso normal por la
    carrera. `regular_only` deja fuera los planes de continuidad y especiales,
    cuya duración no es comparable porque reconocen estudios previos.
    """
    df = graduates.copy()
    df["cod_inst"] = df.cod_inst.astype(str)

    # El volumen se cuenta sobre TODOS los titulados: excluir a quienes no
    # tienen duracion calculable falsearia el tamanio de la promocion.
    volumen = df.groupby("cod_inst").size().rename("titulados_total")

    duracion = df.cat_periodo - df.anio_ing_carr_ori
    nominal_anios = df.dur_total_carr / 2  # la base expresa la nominal en semestres
    usable = (df.anio_ing_carr_ori.ge(min_entry_year)
              & duracion.between(1, max_years)
              & nominal_anios.gt(0))
    if regular_only and "tipo_plan_carr" in df:
        usable &= df.tipo_plan_carr.astype("string").eq("Plan Regular")

    df["_duracion"] = duracion.where(usable)
    df["_sobre"] = (duracion - nominal_anios).where(usable)
    df["_usable"] = usable.astype(float)

    grupo = df.groupby("cod_inst")
    resultado = pd.DataFrame({
        "titulacion_duracion_mediana": grupo._duracion.median(),
        "titulacion_sobreduracion": grupo._sobre.median(),
        "titulacion_oportuna": grupo._sobre.apply(lambda s: (s <= 0).sum() / s.notna().sum()
                                                  if s.notna().any() else np.nan),
        "titulacion_oportuna_holgada": grupo._sobre.apply(
            lambda s: (s <= 1).sum() / s.notna().sum() if s.notna().any() else np.nan),
        "titulacion_cobertura": grupo._usable.mean(),
    }).join(volumen)

    # Con pocos titulados utilizables las medianas son ruido. Se anulan los
    # indicadores pero se conservan el volumen y la cobertura: esa cobertura
    # baja es en si misma informacion sobre como registra esa institucion.
    escaso = grupo._duracion.count() < min_graduates
    for col in ("titulacion_duracion_mediana", "titulacion_sobreduracion",
                "titulacion_oportuna", "titulacion_oportuna_holgada"):
        resultado.loc[escaso, col] = np.nan
    if escaso.any():
        log.info("  %s instituciones con muy pocos titulados utilizables", int(escaso.sum()))
    return resultado.reset_index()


def attach_completion(profiles: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    """Une los indicadores al perfil por código de institución del SIES."""
    out = profiles.copy()
    out["cod_inst"] = out.cod_inst.astype(str)
    unido = out.merge(completion, on="cod_inst", how="left")
    if "titulados_total" in unido and "estudiantes_total" in unido:
        # Razon cruda entre la promocion que egresa y la matricula del mismo
        # anio. NO es una tasa de titulacion: numerador y denominador son
        # cohortes distintas. Sirve para dimensionar el flujo de salida.
        unido["titulados_por_100_estudiantes"] = 100 * unido.titulados_total.div(
            unido.estudiantes_total.where(unido.estudiantes_total.gt(0)))
    return unido
