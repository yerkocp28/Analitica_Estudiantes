"""Marts de la capa gold: `mart_student_course_week` y `mart_student_week`.

Son las dos tablas que el documento maestro especifica en §21 y que el sprint 2
declaraba pendientes.

QUÉ SON Y POR QUÉ NO SON UN `groupby` MÁS

Un mart aquí no es "los datos agregados": es **los datos tal como se veían al
cierre de cada semana**. Una fila de `mart_student_course_week` en la semana 5
contiene solo lo observable en la semana 5, y nada de lo que ocurrió después.
Esa propiedad es la que permite entrenar sin fuga y hacer backtesting honesto.

Por eso el mart **reusa `build_features_at_week`** en vez de reimplementar las
agregaciones: toda la disciplina anti-leakage —el filtro por
`available_from_week` para las notas cargadas con atraso, la lista de columnas
prohibidas, ausencia distinta de cero— ya vive ahí y está cubierta por tests.
Duplicarla sería crear una segunda versión que se desincroniza.

EL PROBLEMA DE LA ESCALA DE NOTAS

Las fuentes no comparten escala: el generador usa 1,0–7,0 y OULAD 0–100. Un
umbral absoluto de nota —"en riesgo si va bajo 4,0"— significaría cosas
distintas en cada una y produciría un indicador que parece comparable y no lo
es.

Por eso `courses_at_risk` se define **sin escala**: por posición relativa
dentro de la misma asignatura y semana, y por asistencia, que ya es una tasa
en [0,1]. El umbral es un parámetro, no una constante escondida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from .builder import KEYS, build_features_at_week

log = get_logger(__name__)

# Percentil dentro de la asignatura-semana bajo el cual se considera que la
# nota acumulada es senial de riesgo. Es una heuristica observable, NO la
# salida de un modelo: sirve para describir carga, no para priorizar personas.
PERCENTIL_RIESGO = 0.25
ASISTENCIA_RIESGO = 0.70


def _semanas(datos: dict[str, pd.DataFrame]) -> list[int]:
    """Las semanas presentes en cualquier tabla semanal de la fuente."""
    semanas: set[int] = set()
    for nombre, df in datos.items():
        if nombre.endswith("_weekly") and "week" in df.columns:
            semanas |= set(pd.to_numeric(df.week, errors="coerce").dropna().astype(int))
    return sorted(w for w in semanas if w >= 1)


def mart_student_course_week(datos: dict[str, pd.DataFrame],
                             semanas: list[int] | None = None,
                             con_targets: bool = True) -> pd.DataFrame:
    """Una fila por estudiante × asignatura × período × semana.

    Cada fila contiene **solo lo observable al cierre de esa semana**. Los
    targets se adjuntan aparte y solo si `con_targets`: son el resultado del
    período, no información de la semana, y jamás deben entrar como feature.
    """
    semanas = semanas if semanas is not None else _semanas(datos)
    if not semanas:
        raise ValueError("La fuente no tiene tablas semanales con semana >= 1.")

    piezas = []
    for w in semanas:
        f = build_features_at_week(datos, w)
        if f.empty:
            continue
        f.insert(len(KEYS), "week", w)
        piezas.append(f)
    if not piezas:
        raise ValueError("Ninguna semana produjo features.")
    mart = pd.concat(piezas, ignore_index=True)

    # Estructura del curso y del estudiante: constante en el periodo, asi que
    # no puede filtrar informacion futura. Se traen SOLO las columnas que el
    # builder no adjunto ya: unir sin mirar producia pares campus_x/campus_y.
    for tabla, llave, deseadas in [
            ("course_master", "course_id", ("subject_code", "credits", "difficulty_tier")),
            ("student_master", "student_id", ("cohort", "campus", "career_code"))]:
        origen = datos.get(tabla)
        if origen is None:
            continue
        nuevas = [c for c in deseadas if c in origen.columns and c not in mart.columns]
        if nuevas:
            mart = mart.merge(origen[[llave, *nuevas]], on=llave, how="left")

    mart = _dias_desde_actividad(mart, datos)

    if con_targets and "outcomes" in datos:
        objetivo = [c for c in ("fail_by_grade", "fail_by_attendance", "failed")
                    if c in datos["outcomes"].columns]
        if objetivo:
            mart = mart.merge(datos["outcomes"][KEYS + objetivo], on=KEYS, how="left")

    log.info("mart_student_course_week: %s filas, %d semanas, %d columnas",
             f"{len(mart):,}", len(semanas), len(mart.columns))
    return mart


def _dias_desde_actividad(mart: pd.DataFrame,
                          datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Semanas transcurridas desde la última con actividad registrada.

    El documento maestro pide `days_since_last_activity`. El grano de estas
    fuentes es semanal, así que se mide en semanas y el nombre lo dice: usar
    «days» sobre un dato semanal sería prometer una precisión que no existe.

    Es la señal de desenganche más directa que el dato permite: no cuánta
    actividad hubo, sino **cuánto lleva sin haberla**.
    """
    act = datos.get("activity_weekly")
    if act is None or "page_views" not in act.columns:
        return mart
    activas = (act.loc[act.page_views.fillna(0) > 0, KEYS + ["week"]]
               .drop_duplicates().sort_values("week"))
    if activas.empty:
        return mart

    # `merge_asof` es exactamente "la ultima activa <= esta semana", y conserva
    # TODAS las filas de la izquierda. Hacerlo con merge + filtro + dedup
    # descartaba las combinaciones sin actividad previa: 1.536 filas perdidas
    # en el generador, que deben quedar en NaN y no desaparecer.
    # `week` llega como int64 desde el mart y como Int64 nullable desde la
    # fuente; merge_asof exige el mismo tipo exacto en la clave.
    izq = mart.sort_values("week").reset_index()
    izq["week"] = izq.week.astype("int64")
    der = activas.assign(_ultima_activa=activas.week.astype("int64"))
    der["week"] = der.week.astype("int64")
    unido = pd.merge_asof(izq, der, on="week", by=KEYS, direction="backward")
    unido["weeks_since_last_activity"] = unido.week - unido._ultima_activa
    return (mart.join(unido.set_index("index")["weeks_since_last_activity"])
            .reset_index(drop=True))


def _en_riesgo(mart: pd.DataFrame, percentil: float, asistencia: float) -> pd.Series:
    """Señal de riesgo observable, sin depender de la escala de notas.

    Una asignatura cuenta como en riesgo esa semana si la asistencia acumulada
    cae bajo el umbral, o si la nota acumulada queda en el cuartil inferior de
    **esa misma asignatura y semana**. Lo segundo es relativo a propósito: es
    lo único comparable entre fuentes con escalas distintas.
    """
    riesgo = pd.Series(False, index=mart.index)
    if "attendance_cum" in mart:
        riesgo |= mart.attendance_cum.lt(asistencia).fillna(False)
    if "mean_grade" in mart and mart.mean_grade.notna().any():
        corte = (mart.groupby(["course_id", "week"], sort=False).mean_grade
                 .transform(lambda s: s.quantile(percentil) if s.notna().any() else np.nan))
        riesgo |= (mart.mean_grade <= corte).fillna(False)
    return riesgo


def mart_student_week(mart_curso: pd.DataFrame,
                      percentil_riesgo: float = PERCENTIL_RIESGO,
                      asistencia_riesgo: float = ASISTENCIA_RIESGO) -> pd.DataFrame:
    """Una fila por estudiante × período × semana, con la carga del estudiante.

    Implementa las agregaciones del §21.2. La lógica es la del documento: la
    vista por asignatura dice cómo va un ramo; ésta dice cómo va la persona,
    que es la unidad sobre la que una dirección de carrera decide a quién
    contactar.
    """
    d = mart_curso.copy()
    d["_en_riesgo"] = _en_riesgo(d, percentil_riesgo, asistencia_riesgo).astype(int)

    llaves = ["student_id", "term_id", "week"]
    agregaciones = {
        "courses_enrolled": ("course_id", "nunique"),
        "courses_at_risk": ("_en_riesgo", "sum"),
    }
    opcionales = {
        "credits_enrolled": ("credits", "sum"),
        "mean_grade": ("mean_grade", "mean"),
        "min_grade": ("mean_grade", "min"),
        "mean_attendance": ("attendance_cum", "mean"),
        "min_attendance": ("attendance_cum", "min"),
        "submission_rate": ("submission_rate", "mean"),
        "missing_assignments": ("missing", "sum"),
        "canvas_activity": ("activity_mean", "mean"),
        # El MINIMO entre asignaturas: lo que importa es cuanto lleva la
        # persona sin aparecer en ninguna parte, no el promedio de sus ramos.
        "weeks_since_last_activity": ("weeks_since_last_activity", "min"),
    }
    for salida, (col, como) in opcionales.items():
        if col in d.columns:
            agregaciones[salida] = (col, como)

    out = d.groupby(llaves, sort=False).agg(**agregaciones).reset_index()

    # Carga academica: cuanta de la carga del estudiante esta en riesgo esa
    # semana. Es una proporcion, asi que no depende del numero de ramos.
    out["academic_load_risk"] = out.courses_at_risk.div(
        out.courses_enrolled.where(out.courses_enrolled.gt(0)))

    log.info("mart_student_week: %s filas, %d columnas", f"{len(out):,}", len(out.columns))
    return out
