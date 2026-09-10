"""Tests de los indicadores de titulación."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_analytics.ingestion.titulados import (
    attach_completion,
    institutional_completion,
    load_graduates,
)

ROOT = Path(__file__).resolve().parents[1]
TITULADOS = sorted((ROOT / "data/external/titulados/titulados_2024").rglob("*.csv")) \
    if (ROOT / "data/external/titulados/titulados_2024").exists() else []


def _titulados(**cambios):
    """Promoción sintética: 40 titulados de una carrera de 10 semestres."""
    n = 40
    base = pd.DataFrame({
        "cat_periodo": 2024, "cod_inst": "A", "nomb_inst": "U. A",
        "nivel_global": "Pregrado", "tipo_inst_1": "Universidades",
        "anio_ing_carr_ori": 2019, "dur_total_carr": 10,
        "tipo_plan_carr": "Plan Regular", "area_conocimiento": "Salud",
    }, index=range(n))
    for k, v in cambios.items():
        base[k] = v
    return base


def test_el_centinela_1900_no_entra_en_la_duracion():
    """1900 marca dato ausente y es el 10% de los registros reales.

    Sin filtrarlo, la duración mediana salta de 5 a decenas de años.
    """
    d = _titulados()
    d.loc[:19, "anio_ing_carr_ori"] = 1900
    r = institutional_completion(d, min_graduates=1).set_index("cod_inst")
    assert r.loc["A", "titulacion_duracion_mediana"] == pytest.approx(5.0)
    # El volumen sí los cuenta: son titulados reales, solo sin duración.
    assert r.loc["A", "titulados_total"] == 40
    assert r.loc["A", "titulacion_cobertura"] == pytest.approx(.5)


def test_los_planes_de_continuidad_quedan_fuera():
    """Reconocen estudios previos: su duración no es comparable."""
    d = _titulados()
    d.loc[:19, "tipo_plan_carr"] = "Plan Regular de Continuidad"
    d.loc[:19, "anio_ing_carr_ori"] = 2022          # duración aparente de 2 años
    r = institutional_completion(d, min_graduates=1).set_index("cod_inst")
    assert r.loc["A", "titulacion_duracion_mediana"] == pytest.approx(5.0)
    r2 = institutional_completion(d, min_graduates=1, regular_only=False).set_index("cod_inst")
    assert r2.loc["A", "titulacion_duracion_mediana"] < 5.0


def test_la_nominal_viene_en_semestres():
    """dur_total_carr son semestres; la duración real, años."""
    d = _titulados()  # 10 semestres = 5 años nominales, ingreso 2019, titula 2024
    r = institutional_completion(d, min_graduates=1).set_index("cod_inst")
    assert r.loc["A", "titulacion_sobreduracion"] == pytest.approx(0.0)
    assert r.loc["A", "titulacion_oportuna"] == pytest.approx(1.0)


def test_sobreduracion_positiva_cuando_se_alargan():
    d = _titulados(anio_ing_carr_ori=2017)  # 7 años para una carrera de 5
    r = institutional_completion(d, min_graduates=1).set_index("cod_inst")
    assert r.loc["A", "titulacion_sobreduracion"] == pytest.approx(2.0)
    assert r.loc["A", "titulacion_oportuna"] == pytest.approx(0.0)
    assert r.loc["A", "titulacion_oportuna_holgada"] == pytest.approx(0.0)


def test_la_holgada_incluye_a_la_oportuna():
    d = _titulados()
    d.loc[:19, "anio_ing_carr_ori"] = 2018  # 6 años: un año sobre lo nominal
    r = institutional_completion(d, min_graduates=1).set_index("cod_inst")
    assert r.loc["A", "titulacion_oportuna"] == pytest.approx(.5)
    assert r.loc["A", "titulacion_oportuna_holgada"] == pytest.approx(1.0)


def test_con_pocos_titulados_se_anula_pero_se_conserva_el_volumen():
    d = _titulados().head(5)
    r = institutional_completion(d, min_graduates=30).set_index("cod_inst")
    assert np.isnan(r.loc["A", "titulacion_oportuna"])
    assert r.loc["A", "titulados_total"] == 5


def test_el_flujo_de_salida_usa_la_matricula_total():
    perfil = pd.DataFrame({"cod_inst": ["A"], "nomb_inst": ["U. A"],
                           "cohorte_total": [500], "estudiantes_total": [4000]})
    comp = pd.DataFrame({"cod_inst": ["A"], "titulados_total": [400]})
    unido = attach_completion(perfil, comp)
    assert unido.titulados_por_100_estudiantes.iloc[0] == pytest.approx(10.0)


def test_sin_matricula_total_no_se_inventa_el_flujo():
    perfil = pd.DataFrame({"cod_inst": ["A"], "nomb_inst": ["U. A"], "cohorte_total": [500]})
    comp = pd.DataFrame({"cod_inst": ["A"], "titulados_total": [400]})
    assert "titulados_por_100_estudiantes" not in attach_completion(perfil, comp).columns


# ----------------------------------------------------------------------
# Sobre la base real
# ----------------------------------------------------------------------
@pytest.mark.skipif(not TITULADOS, reason="titulados 2024 no descargados")
def test_la_carga_filtra_nivel_y_tipo():
    d = load_graduates(TITULADOS[0])
    assert d.nivel_global.eq("Pregrado").all()
    assert d.tipo_inst_1.eq("Universidades").all()
    assert len(d) > 50_000


@pytest.mark.skipif(not (ROOT / "data/results/benchmark_profiles.parquet").exists(),
                    reason="perfiles no generados")
def test_los_indicadores_del_perfil_son_plausibles():
    d = pd.read_parquet(ROOT / "data/results/benchmark_profiles.parquet")
    d = d[d.cohorte.eq(2024)]
    if "titulacion_oportuna" not in d.columns:
        pytest.skip("perfiles generados sin titulación")
    assert d.titulacion_oportuna.dropna().between(0, 1).all()
    assert d.titulacion_cobertura.dropna().between(0, 1).all()
    # Una carrera de pregrado no se completa en menos de tres años ni la
    # mediana institucional se va sobre los diez.
    assert d.titulacion_duracion_mediana.dropna().between(3, 10).all()
    assert (d.titulacion_oportuna_holgada.dropna()
            >= d.titulacion_oportuna.dropna()).all()
