"""Pares independientes del resultado, ajuste comparable e interacciones UA."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from student_analytics.modeling.benchmark import (
    compare_outcomes, fit_peers, neighbors, profile_matrix, target_code,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    c = yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
    c["k_values"] = [2, 3]
    return c


@pytest.fixture
def profiles():
    rng = np.random.default_rng(42)
    p = pd.DataFrame({"cod_inst": [str(i) for i in range(12)],
                      "nomb_inst": ["Universidad Autónoma de Chile"] + [f"Universidad {i}" for i in range(1, 12)],
                      "cohorte_total": [200] * 12, "sedes": [2] * 12,
                      "log_cohorte": np.r_[rng.normal(6, .1, 6), rng.normal(8, .1, 6)],
                      "log_sedes": np.r_[rng.normal(1, .1, 6), rng.normal(2, .1, 6)]})
    for prefix in ["area", "region", "jornada", "modalidad"]:
        p[prefix + "::A"] = np.r_[rng.uniform(.7, .9, 6), rng.uniform(.1, .3, 6)]
        p[prefix + "::B"] = 1 - p[prefix + "::A"]
    return p


def test_los_resultados_no_definen_los_pares(profiles, config):
    a, _ = profile_matrix(profiles, config["block_weights"])
    changed = profiles.assign(retencion=np.linspace(0, 1, len(profiles)), sistema=9999, misma_carrera=0)
    b, _ = profile_matrix(changed, config["block_weights"])
    np.testing.assert_array_equal(a, b)


def test_modelo_reproducible_sin_ua_como_su_propio_par(profiles, config):
    model = fit_peers(profiles, config)
    other = fit_peers(profiles.sample(frac=1, random_state=99), config)
    pd.testing.assert_frame_equal(model.universities, other.universities)
    target = target_code(profiles, config["target_name"])
    peers = neighbors(model, target)
    assert target not in peers.cod_inst.values
    assert peers.distancia.is_monotonic_increasing
    assert model.universities.groupby("grupo").size().min() >= config["minimum_cluster"]
    assert set(model.diagnostics.algoritmo) == {"K-means", "Jerárquico Ward", "Mezcla gaussiana"}


def test_pocas_universidades_no_inventa_clusters(profiles, config):
    with pytest.raises(ValueError, match="pocas universidades"):
        fit_peers(profiles.head(3), config)
    with pytest.raises(ValueError, match="única"):
        target_code(profiles.iloc[1:], config["target_name"])


def test_pesos_invalidos_no_producen_distancias_arbitrarias(profiles, config):
    config["block_weights"]["escala"] = -1
    with pytest.raises(ValueError, match="pesos"):
        profile_matrix(profiles, config["block_weights"])


def outcome_data():
    # UA concentra 90% en A; pares solo 10% en A. Tasas por área iguales:
    # la brecha cruda existe por composición, la ajustada debe desaparecer.
    return pd.DataFrame({"cohorte": [2024] * 5, "cod_inst": ["UA", "UA", "P", "P", "N"],
                         "nomb_inst": ["UA", "UA", "Par", "Par", "Nacional"],
                         "area_conocimiento": ["A", "B", "A", "B", "A"],
                         "n": [90, 10, 10, 90, 100], "misma_carrera": [81, 5, 9, 45, 10],
                         "misma_universidad": [81, 5, 9, 45, 10], "sistema": [81, 5, 9, 45, 10],
                         "sin_mrun": [1, 0, 0, 0, 0]})


def test_ajuste_por_areas_y_exclusion_ua_de_referencias():
    _, stats = compare_outcomes(outcome_data(), 2024, "UA", ["P", "UA"], "misma_carrera")
    assert stats["ua"] == pytest.approx(.86)
    assert stats["pares"] == pytest.approx(.54)
    assert stats["nacional"] == pytest.approx(.32)
    assert stats["ajustada"] == pytest.approx(.86)
    assert stats["cobertura"] == 1
    assert stats["pares_disponibles"] == 1


def test_areas_sin_pares_no_son_ceros_y_ajuste_usa_soporte_comun():
    data = outcome_data()
    data = data.loc[~(data.cod_inst.eq("P") & data.area_conocimiento.eq("B"))]
    _, stats = compare_outcomes(data, 2024, "UA", ["P"], "misma_carrera")
    assert stats["cobertura"] == pytest.approx(.9)
    assert stats["ua_comun"] == pytest.approx(.9)
    assert stats["ajustada"] == pytest.approx(.9)
    _, missing = compare_outcomes(data, 2024, "UA", ["P"], "misma_carrera", "B")
    assert np.isnan(missing["pares"])
    assert missing["cobertura"] == 0


def test_benchmark_ua_cargado_y_filtros_independientes():
    from streamlit.testing.v1 import AppTest
    if not (ROOT / "data/results/benchmark_profiles.parquet").exists():
        pytest.skip("perfiles no generados")
    app = AppTest.from_file(str(ROOT / "src/student_analytics/ui/app.py"), default_timeout=180).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert "Benchmark" in app.title[0].value
    assert len(app.tabs) == 4
    assert len(app.metric) == 4
    original_peers = app.dataframe[1].value["Universidad"].tolist()
    app.selectbox(key="bench_metric").set_value("sistema").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.dataframe[1].value["Universidad"].tolist() == original_peers
    app.selectbox(key="bench_year").set_value(2023).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    app.selectbox(key="bench_year").set_value(2024).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    app.selectbox(key="bench_area").set_value("Administración y Comercio").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.dataframe[1].value["Universidad"].tolist() == original_peers
    app.radio(key="bench_mode").set_value("Mismo cluster que la UA").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    app.radio(key="bench_mode").set_value("Selección manual").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No hay pares" in info.value for info in app.info)
