"""Todo adaptador debe cumplir el contrato de datos.

Este es el test que el adaptador de Banner tendrá que pasar el día uno. Su
valor no es cubrir líneas: es que un esquema **parecido pero mal** —otra
columna clave, otro tipo, un grano que no es único— falle aquí y no tres
semanas después como «resultados raros».

Escribirlo ya encontró cuatro derivas reales:

- el contrato declaraba `canvas_weekly`, un dataset que ningún adaptador
  producía ni el pipeline consumía;
- el grano de las tablas semanales omitía `term_id`, y sin él había 295.884
  filas duplicadas en el generador;
- `grades_weekly` es por EVALUACIÓN, no por semana: OULAD tenía 22.653
  duplicados porque le faltaba `assessment_seq`;
- `student_id` llegaba como `int64` desde OULAD y como `str` desde el
  generador. Consistente dentro de cada uno, incompatible entre ambos.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from student_analytics.ingestion.contract import (
    cargar_contrato,
    grano,
    validar_adaptador,
    validar_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
OULAD = ROOT / "data" / "external" / "oulad"

# Lo que cada adaptador NO entrega, declarado a propósito. Si un adaptador
# deja de entregar algo más, el test falla y hay que decidir si es una pérdida
# real o una decisión — que es exactamente la conversación que debe ocurrir.
HUECOS_ESPERADOS = {
    "sintetico": {
        # El generador no simula fechas de matrícula ni de carga de notas, ni
        # el código de asignatura: son campos que llegarán de Banner.
        "attendance_weekly": {"last_attend_date"},
        "course_master": {"subject_code"},
        "enrollment": {"registered_at", "dropped_at"},
        "grades_weekly": {"graded_at"},
    },
    "oulad": {
        # OULAD es educación a distancia del Reino Unido: no tiene asistencia,
        # ni sedes, ni carreras, ni puntaje de admisión, ni escala 1,0-7,0.
        "activity_weekly": {"participations", "platform_adoption"},
        "course_master": {"subject_code", "credits", "difficulty_tier"},
        "enrollment": {"registered_at", "dropped_at"},
        "grades_weekly": {"graded_at"},
        "outcomes": {"final_grade", "final_attendance", "fail_by_attendance"},
        "student_master": {"cohort", "campus", "career_code", "admission_score",
                           "hs_gpa", "funding_type"},
    },
}
DATASETS_AUSENTES = {"sintetico": set(), "oulad": {"attendance_weekly"}}


def _sintetico() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in SYN.glob("*.parquet")}


def _oulad() -> dict[str, pd.DataFrame]:
    from student_analytics.ingestion.oulad import load_oulad
    return load_oulad(OULAD)


ADAPTADORES = {"sintetico": (_sintetico, SYN), "oulad": (_oulad, OULAD)}


# ----------------------------------------------------------------------
# El validador, sobre casos construidos
# ----------------------------------------------------------------------
def test_un_grano_no_unico_es_error():
    """Es la propiedad más importante: sin ella los joins se abren solos."""
    spec = {"grain": ["a", "b"], "fields": {}}
    bueno = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    malo = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    assert validar_dataset("t", bueno, spec)[0] == []
    errores, _ = validar_dataset("t", malo, spec)
    assert len(errores) == 1 and "no es único" in errores[0]


def test_una_clave_nula_es_error():
    spec = {"grain": ["a"], "fields": {}}
    d = pd.DataFrame({"a": [1, None, 3]})
    errores, _ = validar_dataset("t", d, spec)
    assert any("clave nula" in e for e in errores)


def test_falta_una_columna_del_grano():
    errores, _ = validar_dataset("t", pd.DataFrame({"a": [1]}),
                                 {"grain": ["a", "b"], "fields": {}})
    assert any("faltan columnas del grano" in e for e in errores)


def test_un_tipo_incompatible_es_error_y_uno_ausente_es_hueco():
    spec = {"grain": ["a"], "fields": {"a": {"type": "str"}, "b": {"type": "float"}}}
    d = pd.DataFrame({"a": [1, 2]})          # a debería ser str; b no viene
    errores, huecos = validar_dataset("t", d, spec)
    assert any("se declaró str" in e for e in errores)
    assert huecos == ["b"]


def test_un_conteo_con_ausentes_llega_como_float_y_no_es_error():
    """Ausencia no es cero: un entero con nulos se vuelve float, y está bien."""
    spec = {"grain": ["a"], "fields": {"n": {"type": "int"}}}
    d = pd.DataFrame({"a": [1, 2], "n": [3.0, None]})
    assert validar_dataset("t", d, spec)[0] == []


def test_el_grano_escalar_y_el_de_lista_se_leen_igual():
    assert grano("student_id") == ["student_id"]
    assert grano(["a", "b"]) == ["a", "b"]


def test_un_dataset_fuera_del_contrato_se_reporta_sin_fallar():
    informe = validar_adaptador({"inventado": pd.DataFrame({"x": [1]})}, "raro")
    assert "inventado" in informe.extra
    assert informe.valido            # no es error: es información


def test_algo_que_no_es_dataframe_es_error():
    informe = validar_adaptador({"student_master": {"student_id": [1]}}, "raro")
    assert not informe.valido
    assert any("se esperaba un DataFrame" in e for e in informe.errores)


# ----------------------------------------------------------------------
# El contrato mismo
# ----------------------------------------------------------------------
def test_el_contrato_declara_grano_y_campos_en_todos_sus_datasets():
    for nombre, spec in cargar_contrato().items():
        assert spec.get("grain"), f"{nombre} sin grano declarado"
        assert spec.get("fields"), f"{nombre} sin campos declarados"
        assert spec.get("source_real"), f"{nombre} sin fuente real declarada"


def test_las_tablas_semanales_llevan_term_id_en_el_grano():
    """Sin term_id el grano no es único: la misma asignatura en dos períodos."""
    for nombre, spec in cargar_contrato().items():
        claves = grano(spec["grain"])
        if "week" in claves:
            assert "term_id" in claves, f"{nombre}: grano semanal sin term_id"


# ----------------------------------------------------------------------
# Los adaptadores reales
# ----------------------------------------------------------------------
@pytest.mark.parametrize("adaptador", sorted(ADAPTADORES))
def test_el_adaptador_cumple_el_contrato(adaptador):
    cargar, ruta = ADAPTADORES[adaptador]
    if not ruta.exists() or not any(ruta.rglob("*")):
        pytest.skip(f"{adaptador} sin datos descargados")
    informe = validar_adaptador(cargar(), adaptador)
    assert informe.valido, "\n".join(informe.errores)


@pytest.mark.parametrize("adaptador", sorted(ADAPTADORES))
def test_los_huecos_del_adaptador_son_los_declarados(adaptador):
    """Un hueco nuevo es una decisión, no un accidente: que se note."""
    cargar, ruta = ADAPTADORES[adaptador]
    if not ruta.exists() or not any(ruta.rglob("*")):
        pytest.skip(f"{adaptador} sin datos descargados")
    informe = validar_adaptador(cargar(), adaptador)
    obtenidos = {k: set(v) for k, v in informe.huecos.items()}
    assert obtenidos == HUECOS_ESPERADOS[adaptador]
    assert set(informe.ausentes) == DATASETS_AUSENTES[adaptador]


@pytest.mark.parametrize("adaptador", sorted(ADAPTADORES))
def test_student_id_es_str_en_todas_las_tablas(adaptador):
    """Dos adaptadores con convenciones distintas no se pueden comparar.

    OULAD lo entregaba como int64 y el generador como str.
    """
    cargar, ruta = ADAPTADORES[adaptador]
    if not ruta.exists() or not any(ruta.rglob("*")):
        pytest.skip(f"{adaptador} sin datos descargados")
    for nombre, df in cargar().items():
        if "student_id" in df.columns:
            assert pd.api.types.is_string_dtype(df.student_id) \
                or pd.api.types.is_object_dtype(df.student_id), \
                f"{adaptador}.{nombre}.student_id es {df.student_id.dtype}"
