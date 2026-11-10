import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from piloto_fan import presentacion
from piloto_fan.institucional import HOME


def require_all():
    for pointer in ["artifacts/latest.json", "artifacts/institucional/latest.json",
                    "artifacts/institucional_modelos/latest.json"]:
        if not (HOME / pointer).exists():
            pytest.skip("Requiere las corridas locales del piloto.")


def test_landing_opens_first_and_each_block_leads_to_its_view():
    require_all()
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=90).run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.sidebar.radio[0].value == presentacion.LANDING
    assert len(app.metric) == 4  # Una cifra principal por producto, sin avisos de datos faltantes.
    targets = {"go_1": presentacion.POSITION, "go_2": presentacion.MAP,
               "go_3": presentacion.PROFILES, "go_4": presentacion.RADAR}
    for key, view in targets.items():
        app.sidebar.radio[0].set_value(presentacion.LANDING).run()
        app.button(key=key).click().run()
        assert not app.exception, [e.message for e in app.exception]
        assert app.sidebar.radio[0].value == view


def test_story_views_render_and_map_filters_by_year():
    require_all()
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value(presentacion.POSITION).run()
    assert not app.exception, [e.message for e in app.exception]
    assert [b.label for b in app.get("download_button")] == ["Descargar informe HTML"]
    app.sidebar.radio[0].set_value(presentacion.MAP).run()
    assert not app.exception, [e.message for e in app.exception]
    total = int(app.metric[2].value.replace(".", ""))
    app.selectbox(key="map_year").set_value(2025).run()
    assert not app.exception and int(app.metric[2].value.replace(".", "")) < total


def rate(text):
    return float(text.rstrip("%").replace(",", "."))


def test_full_benchmark_opens_from_view_1_with_the_same_figures():
    # Solo usa datos públicos versionados: corre también sin las corridas locales.
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value(presentacion.POSITION).run()
    ua, peers, system = (rate(m.value) for m in app.metric)
    app.button(key="go_benchmark").click().run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.sidebar.radio[0].value == presentacion.BENCHMARK
    assert app.selectbox(key="bench_area").value == presentacion.AREA
    assert [rate(m.value) for m in app.metric[:3]] == [ua, peers, system]
    assert [t.label for t in app.tabs] == [
        "Posición de la UA", "Clusters por variables", "Mapa de posicionamiento", "Por qué son comparables",
        "Modelos de agrupamiento", "Evolución y áreas", "Método y cobertura"]
    assert [b.label for b in app.get("download_button")] == [
        "Descargar benchmark con contexto", "Descargar informe ejecutivo (HTML)",
        "Descargar grupos y variables (CSV)", "Descargar ficha metodológica (JSON)"]
    # La UA aparece en su grupo y el modelo que lo forma queda a la vista.
    labels = {m.label: m.value for m in app.metric}
    assert labels["Grupo de la UA"].startswith("Grupo ")
    assert labels["Modelo que forma los grupos"] in ("K-means", "Jerárquico Ward", "Mezcla gaussiana")


def test_full_benchmark_keeps_every_control():
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value(presentacion.BENCHMARK).run()
    steps = [lambda: app.radio(key="bench_mode").set_value("Mismo cluster que la UA"),
             lambda: app.radio(key="bench_mode").set_value("Selección manual"),
             lambda: app.selectbox(key="bench_metric").set_value("sistema"),
             lambda: app.selectbox(key="bench_area").set_value("Todas las áreas"),
             lambda: app.selectbox(key="bench_x").set_value("paes_percentil_promedio"),
             lambda: app.selectbox(key="bench_profile_block").set_value("region"),
             lambda: app.toggle(key="bench_cluster_extra").set_value(True),
             lambda: app.radio(key="bench_cluster_focus").set_value("Grupo de la UA frente al resto"),
             lambda: app.selectbox(key="bench_cluster_x").set_value("mapa_x"),
             lambda: app.selectbox(key="bench_cluster_y").set_value("arancel_mediano"),
             lambda: app.radio(key="bench_mode").set_value("Más cercanas por perfil"),
             lambda: app.number_input(key="bench_count").set_value(10),
             lambda: app.selectbox(key="bench_year").set_value(2008)]
    for step in steps:
        step().run()
        assert not app.exception, [e.message for e in app.exception]
    # 2008 es una cohorte con código de carrera incompleto: el aviso tiene que aparecer.
    app.selectbox(key="bench_metric").set_value("misma_carrera").run()
    assert any("código de carrera" in w.value for w in app.warning)


def test_summaries_follow_the_data_not_fixed_text():
    h = pd.DataFrame([{"student_key": f"{c}{i}", "campus_code": c, "campus": "Sede " + c,
                       "course_campus_code": c, "course_code": code, "course_name": name,
                       "grade_numeric": 3.0 if i < below else 5.5, "numeric_below4": float(i < below)}
                      for code, name, c, below in [("M1", "CÁLCULO I", "A", 30), ("M1", "CÁLCULO I", "B", 6),
                                                   ("H1", "HISTORIA", "A", 2)] for i in range(40)])
    s = presentacion.map_summary(h, 10)
    assert s["course"] == "CÁLCULO I" and s["top10_math"] == 1 and s["courses"] == 2
    assert s["low"] == ("Sede B", pytest.approx(.15)) and s["high"] == ("Sede A", pytest.approx(.75))
    d = pd.DataFrame({"student_key": [f"k{i:02d}" for i in range(20)], "career": "C", "campus": "T",
                      "risk": np.linspace(1, 0, 20), "below4_fraction": 0.,
                      "target": [1.] * 4 + [0.] * 16})
    r = presentacion.radar_summary(d)
    assert (r["slots"], r["hits"]) == (4, 4) and r["random"] == pytest.approx(.8)
