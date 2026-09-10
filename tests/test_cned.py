"""Tests de la ingesta de recursos institucionales del CNED."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from student_analytics.ingestion.cned import (
    attach_resources,
    load_institutional,
    normalize_name,
    _closest_year,
    _sum_columns,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/cned/INDICES_Institucional_2005-2025.xlsx"


# ----------------------------------------------------------------------
# Normalizacion de nombres: es lo unico que une CNED con el SIES
# ----------------------------------------------------------------------
def test_abreviatura_de_universidad_se_expande():
    assert normalize_name("U. AUTÓNOMA DE CHILE") == normalize_name("UNIVERSIDAD AUTONOMA DE CHILE")


def test_acentos_y_mayusculas_no_importan():
    assert normalize_name("U. Andrés Bello") == normalize_name("UNIVERSIDAD ANDRES BELLO")


def test_apostrofo_y_acento_grave_se_unifican():
    """CNED escribe O`HIGGINS y el SIES O'HIGGINS.

    Sin unificar las comillas esas universidades no calzan, y el cruce
    pierde instituciones sin emitir ningun aviso.
    """
    assert normalize_name("U. BERNARDO O`HIGGINS") == normalize_name("UNIVERSIDAD BERNARDO O'HIGGINS")
    assert normalize_name("U. DE O`HIGGINS") == normalize_name("UNIVERSIDAD DE O'HIGGINS")


def test_pontificia_abreviada_calza():
    assert normalize_name("PONT. U. CATÓLICA DE CHILE") == \
        normalize_name("PONTIFICIA UNIVERSIDAD CATOLICA DE CHILE")


def test_instituciones_distintas_no_colapsan():
    """La normalizacion no puede ser tan agresiva que junte instituciones."""
    assert normalize_name("U. DE CHILE") != normalize_name("U. DE SANTIAGO DE CHILE")
    assert normalize_name("U. DIEGO PORTALES") != normalize_name("I.P. DIEGO PORTALES")
    assert normalize_name("U. CENTRAL DE CHILE") != normalize_name("U. AUSTRAL DE CHILE")


# ----------------------------------------------------------------------
# Carga y union, solo si la base esta descargada
# ----------------------------------------------------------------------
pytestmark_base = pytest.mark.skipif(
    not BASE.exists(), reason="base CNED no descargada (python scripts/download_cned.py)")


@pytest.fixture(scope="module")
def recursos():
    if not BASE.exists():
        pytest.skip("base CNED no descargada")
    return load_institutional(BASE, 2024)


@pytestmark_base
def test_carga_las_instituciones(recursos):
    assert len(recursos) > 100
    assert recursos.clave.is_unique


@pytestmark_base
def test_proporciones_en_rango(recursos):
    """Un share fuera de [0,1] delata un denominador equivocado."""
    for col in ("share_doctorado", "share_magister", "share_jornada_completa",
                "share_docentes_mujeres"):
        serie = recursos[col].dropna()
        assert not serie.empty, col
        assert serie.between(0, 1).all(), f"{col} fuera de [0,1]"


@pytestmark_base
def test_magister_incluye_doctorado(recursos):
    """share_magister se define como magíster O doctorado, así que lo contiene."""
    comp = recursos[["share_doctorado", "share_magister"]].dropna()
    assert (comp.share_magister >= comp.share_doctorado - 1e-9).all()


@pytestmark_base
def test_cruch_identifica_universidades_reales(recursos):
    """El CRUCH tiene 30 universidades e incluye privadas desde 2019."""
    assert recursos.cruch.sum() == 30
    claves = set(recursos.loc[recursos.cruch.eq(1), "clave"])
    assert normalize_name("U. DE CHILE") in claves
    # Entraron al CRUCH en 2019 tras la Ley 21.091: pertenecer al CRUCH ya no
    # equivale a ser "tradicional".
    assert normalize_name("U. DIEGO PORTALES") in claves
    assert normalize_name("U. AUTÓNOMA DE CHILE") not in claves


@pytestmark_base
def test_anio_de_creacion_es_plausible(recursos):
    anios = recursos.anio_creacion.dropna()
    assert anios.min() == 1842, "la Universidad de Chile es de 1842"
    assert anios.max() <= 2026


@pytestmark_base
def test_union_con_el_perfil_no_pierde_universidades(recursos):
    """El cruce se hace por nombre: si se degrada, hay que enterarse."""
    perfiles = ROOT / "data/results/benchmark_profiles.parquet"
    if not perfiles.exists():
        pytest.skip("perfiles no generados")
    p = pd.read_parquet(perfiles)
    p = p.loc[p.cohorte.eq(2024), ["cod_inst", "nomb_inst", "cohorte_total"]]
    unido = attach_resources(p, recursos)
    calce = unido.anio_creacion.notna().mean()
    assert calce >= .95, f"solo calza el {calce:.0%} de las universidades"


@pytestmark_base
def test_ratios_usan_matricula_institucional_no_cohorte(recursos):
    p = pd.DataFrame({"cod_inst": ["1"], "nomb_inst": ["U. DE CHILE"],
                      "cohorte_total": [1000], "estudiantes_total": [10000]})
    unido = attach_resources(p, recursos)
    fila = unido.iloc[0]
    if pd.notna(fila.get("docentes_jce")):
        esperado = 100 * fila.docentes_jce / 10000
        assert fila.docentes_por_100_alumnos == pytest.approx(esperado)


def test_recursos_sin_denominador_no_usa_cohorte():
    p = pd.DataFrame({"nomb_inst": ["U. A"], "cohorte_total": [100]})
    r = pd.DataFrame({"clave": [normalize_name("U. A")], "docentes_jce": [50.]})
    assert pd.isna(attach_resources(p, r).docentes_por_100_alumnos.iloc[0])


def test_no_sustituye_anio_por_observaciones_futuras():
    data = pd.DataFrame({"year": [2022, 2024], "value": [1, 9]})
    assert _closest_year(data, "year", 2023).empty
    assert _closest_year(data, "year", 2024).value.tolist() == [9]


def test_ausencia_de_dotacion_no_se_convierte_en_cero():
    data = pd.DataFrame({"otra": [3]})
    assert _sum_columns(data, "Docentes").isna().all()
