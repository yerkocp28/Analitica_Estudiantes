"""Capa silver y marts: que lo que se materializa sea cierto."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_analytics.features.builder import build_features_at_week
from student_analytics.features.marts import (
    mart_student_course_week,
    mart_student_week,
)
from student_analytics.ingestion.layers import fuentes_disponibles, leer, materializar

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["student_id", "course_id", "term_id"]


def _fuente_minima(semanas=3, alumnos=4, cursos=2) -> dict[str, pd.DataFrame]:
    """Una fuente canónica chica, suficiente para las invariantes."""
    filas = [(f"S{s}", f"C{c}", 1, w)
             for s in range(alumnos) for c in range(cursos)
             for w in range(1, semanas + 1)]
    base = pd.DataFrame(filas, columns=KEYS + ["week"])
    rng = np.random.default_rng(3)
    n = len(base)
    return {
        "student_master": pd.DataFrame({
            "student_id": [f"S{s}" for s in range(alumnos)],
            "cohort": 2024, "campus": "Centro", "career_code": "ADM"}),
        "course_master": pd.DataFrame({
            "course_id": [f"C{c}" for c in range(cursos)],
            "credits": 5.0, "difficulty_tier": "media"}),
        "enrollment": base[KEYS].drop_duplicates().assign(status="RE"),
        # `attendance_recorded` y `activity_reliable` son banderas de calidad
        # del REGISTRO, no del estudiante, y el builder las exige.
        "attendance_weekly": base.assign(
            sessions_held=2, sessions_attended=rng.integers(0, 3, n),
            attendance_rate=rng.uniform(0, 1, n), attendance_recorded=True),
        "activity_weekly": base.assign(
            page_views=rng.integers(0, 30, n), days_active=rng.integers(0, 5, n),
            platform_adoption="high", activity_reliable=True),
        "assignments_weekly": base.assign(
            assigned=1, submitted=rng.integers(0, 2, n), late=0, missing=0),
        "grades_weekly": base.assign(
            assessment_grade=rng.uniform(1, 7, n), assessment_seq=1,
            available_from_week=base.week),
        "outcomes": base[KEYS].drop_duplicates().assign(
            final_grade=4.5, fail_by_grade=0, fail_by_attendance=0, failed=0),
    }


# ----------------------------------------------------------------------
# Silver
# ----------------------------------------------------------------------
def test_silver_rechaza_datos_que_violan_el_contrato(tmp_path):
    """Es el punto donde el contrato deja de ser declarativo.

    El día que exista `ingestion/banner.py`, si su grano no es único los datos
    no llegan a silver y el error aparece en la frontera, no en los marts.
    """
    datos = _fuente_minima()
    datos["student_master"] = pd.concat([datos["student_master"]] * 2)  # grano roto
    with pytest.raises(ValueError, match="no cumple el contrato"):
        materializar(datos, "rota", raiz=tmp_path)
    assert not (tmp_path / "rota").exists()


def test_silver_escribe_y_relee_lo_mismo(tmp_path):
    datos = _fuente_minima()
    destino, informe = materializar(datos, "prueba", raiz=tmp_path)
    assert informe.valido
    vuelta = leer("prueba", raiz=tmp_path)
    assert set(vuelta) == set(datos)
    for k in datos:
        assert len(vuelta[k]) == len(datos[k]), k


def test_silver_borra_las_tablas_que_la_fuente_dejo_de_entregar(tmp_path):
    """Un parquet viejo no debe sobrevivir como fantasma de otra corrida."""
    materializar(_fuente_minima(), "prueba", raiz=tmp_path)
    assert "attendance_weekly" in leer("prueba", raiz=tmp_path)
    sin_asistencia = {k: v for k, v in _fuente_minima().items()
                      if k != "attendance_weekly"}
    materializar(sin_asistencia, "prueba", raiz=tmp_path)
    assert "attendance_weekly" not in leer("prueba", raiz=tmp_path)


def test_leer_una_fuente_inexistente_dice_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_silver"):
        leer("no_existe", raiz=tmp_path)


# ----------------------------------------------------------------------
# Marts
# ----------------------------------------------------------------------
def test_el_mart_por_asignatura_conserva_el_grano():
    m = mart_student_course_week(_fuente_minima())
    assert m.duplicated(subset=KEYS + ["week"]).sum() == 0


def test_el_mart_no_pierde_ni_inventa_filas_frente_al_builder():
    """El mart es el builder materializado: si difiere, uno de los dos miente.

    Una versión previa perdía 1.536 filas al calcular las semanas sin
    actividad con merge + filtro en vez de merge_asof.
    """
    datos = _fuente_minima()
    semanas = sorted(datos["activity_weekly"].week.unique())
    esperado = sum(len(build_features_at_week(datos, w)) for w in semanas)
    assert len(mart_student_course_week(datos)) == esperado


def test_los_targets_se_adjuntan_solo_si_se_piden():
    datos = _fuente_minima()
    con = mart_student_course_week(datos, con_targets=True)
    sin = mart_student_course_week(datos, con_targets=False)
    assert "failed" in con.columns
    assert "failed" not in sin.columns
    # Y ninguna columna de resultado entra jamás como feature.
    from student_analytics.features.builder import FORBIDDEN
    assert not (FORBIDDEN & set(sin.columns))


def test_no_quedan_columnas_duplicadas_al_unir_las_maestras():
    """El builder ya adjunta campus y carrera: unir sin mirar dejaba campus_x."""
    m = mart_student_course_week(_fuente_minima())
    assert not [c for c in m.columns if c.endswith(("_x", "_y"))]


def test_las_semanas_sin_actividad_cuentan_desde_la_ultima_con_actividad():
    datos = _fuente_minima(semanas=5, alumnos=1, cursos=1)
    # Actividad solo en la semana 2: en la 5 llevan 3 semanas sin aparecer.
    datos["activity_weekly"]["page_views"] = [0, 10, 0, 0, 0]
    m = mart_student_course_week(datos).set_index("week")
    assert m.loc[2, "weeks_since_last_activity"] == 0
    assert m.loc[5, "weeks_since_last_activity"] == 3
    # Antes de la primera actividad no se inventa un valor.
    assert pd.isna(m.loc[1, "weeks_since_last_activity"])


def test_el_riesgo_no_depende_de_la_escala_de_notas():
    """El generador usa 1,0-7,0 y OULAD 0-100.

    Un umbral absoluto significaría cosas distintas en cada fuente. El
    indicador se define por posición relativa, así que reescalar las notas no
    debe cambiar cuántas asignaturas quedan en riesgo.
    """
    datos = _fuente_minima(semanas=3, alumnos=8, cursos=2)
    a = mart_student_week(mart_student_course_week(datos))
    escalado = {k: v.copy() for k, v in datos.items()}
    escalado["grades_weekly"]["assessment_grade"] *= 100 / 7
    b = mart_student_week(mart_student_course_week(escalado))
    assert a.courses_at_risk.tolist() == b.courses_at_risk.tolist()


def test_el_mart_por_estudiante_trae_las_agregaciones_del_documento():
    m = mart_student_week(mart_student_course_week(_fuente_minima()))
    esperadas = {"courses_enrolled", "courses_at_risk", "credits_enrolled",
                 "mean_grade", "min_grade", "mean_attendance", "min_attendance",
                 "submission_rate", "missing_assignments", "canvas_activity",
                 "weeks_since_last_activity", "academic_load_risk"}
    assert esperadas <= set(m.columns)
    assert m.duplicated(subset=["student_id", "term_id", "week"]).sum() == 0
    assert m.academic_load_risk.dropna().between(0, 1).all()


def test_una_fuente_sin_asistencia_no_inventa_las_columnas():
    """OULAD no tiene asistencia: debe faltar, no salir en cero."""
    datos = {k: v for k, v in _fuente_minima().items() if k != "attendance_weekly"}
    m = mart_student_week(mart_student_course_week(datos))
    assert m.mean_attendance.isna().all() if "mean_attendance" in m else True


def test_una_fuente_sin_semanas_falla_fuerte():
    datos = {k: v for k, v in _fuente_minima().items() if not k.endswith("_weekly")}
    with pytest.raises(ValueError, match="tablas semanales"):
        mart_student_course_week(datos)


# ----------------------------------------------------------------------
# Sobre las fuentes reales
# ----------------------------------------------------------------------
@pytest.mark.skipif(not fuentes_disponibles(), reason="sin capa silver")
@pytest.mark.parametrize("fuente", fuentes_disponibles() or ["-"])
def test_los_marts_reales_son_coherentes(fuente):
    datos = leer(fuente)
    mcw = mart_student_course_week(datos)
    assert mcw.duplicated(subset=KEYS + ["week"]).sum() == 0
    assert not [c for c in mcw.columns if c.endswith(("_x", "_y"))]
    msw = mart_student_week(mcw)
    assert msw.duplicated(subset=["student_id", "term_id", "week"]).sum() == 0
    assert msw.courses_at_risk.le(msw.courses_enrolled).all()
    assert msw.academic_load_risk.dropna().between(0, 1).all()
