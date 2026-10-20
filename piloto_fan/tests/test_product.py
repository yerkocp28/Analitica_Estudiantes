import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from piloto_fan.data import digest
from piloto_fan.reporting import executive_html

HOME = Path(__file__).parents[1]


def latest():
    pointer = HOME / "artifacts/latest.json"
    if not pointer.exists():
        pytest.skip("Requiere generar primero los resultados locales del piloto.")
    return HOME / "artifacts" / json.loads(pointer.read_text())["run_id"]


def test_manifest_and_prediction_store_are_consistent():
    folder = latest()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        assert digest(folder / name) == expected
    for path in folder.glob("*_predictions.parquet"):
        p = pd.read_parquet(path)
        assert p.mrun.notna().all()
        assert not p.duplicated(["mrun", "case", "run_id"]).any()
        assert p.risk_score.between(0, 1).all()
        assert p.model_version.notna().all()


def test_executive_export_escapes_text_and_contains_no_individual_keys():
    report = json.loads((latest() / "report.json").read_text(encoding="utf-8"))
    report["run_id"] = "<script>alert(1)</script>"
    html = executive_html(report)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "risk_score" not in html


def test_all_sections_and_capacity_interactions():
    latest()
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=30).run()
    assert not app.exception
    for section in ["Perfiles de ingreso", "Continuidad", "Titulación", "Calidad y método", "Demostración", "Benchmark"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, [e.message for e in app.exception]
        if section in ("Continuidad", "Titulación") and app.select_slider:
            app.select_slider[0].set_value(30).run()
            assert not app.exception


def test_cascading_filters_change_metrics_and_reset():
    folder = latest()
    population = pd.read_parquet(folder / "poblacion_dataset.parquet")
    predictions = pd.read_parquet(folder / "continuidad_predictions.parquet")
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=30).run()
    site = "CAMPUS PROVIDENCIA"
    career = "INGENIERIA COMERCIAL"
    app.selectbox(key="filter_nomb_sede").set_value(site).run()
    app.selectbox(key="filter_nomb_carrera").set_value(career).run()
    assert not app.exception
    expected = population[population.nomb_sede.eq(site) & population.nomb_carrera.eq(career) & population.cohorte.eq(2025)]
    assert int(app.metric[0].value.replace(".", "")) == len(expected)
    app.sidebar.radio[0].set_value("Continuidad").run()
    expected = predictions[predictions.nomb_sede.eq(site) & predictions.nomb_carrera.eq(career)]
    assert int(app.metric[0].value.replace(".", "")) == len(expected)
    app.select_slider[0].set_value(30).run()
    assert int(app.metric[4].value.replace(".", "")) == int(__import__("numpy").ceil(len(expected)*.3))
    next(b for b in app.button if b.label == "Restablecer filtros").click().run()
    assert int(app.metric[0].value.replace(".", "")) == len(predictions)


def test_profiles_cohort_and_filtered_empty_model_are_explicit():
    folder = latest()
    population = pd.read_parquet(folder / "poblacion_dataset.parquet")
    path = folder / "titulacion_predictions.parquet"
    predictions = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["nomb_carrera"])
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Perfiles de ingreso").run()
    app.selectbox(key="cohort").set_value(2024).run()
    assert not app.exception
    missing = sorted(set(population.nomb_carrera) - set(predictions.nomb_carrera))
    if missing and path.exists():
        app.selectbox(key="filter_nomb_carrera").set_value(missing[0]).run()
        app.sidebar.radio[0].set_value("Titulación").run()
        assert not app.exception
        assert any("no tiene casos" in x.value for x in app.info)
        assert not app.metric
