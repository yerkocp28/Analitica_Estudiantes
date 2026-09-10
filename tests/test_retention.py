"""Continuidad, movilidad, denominadores y ponderación de matrícula pública."""
import pandas as pd
import pytest

from student_analytics.ingestion.retention import aggregate_retention, summarize_retention


def test_movilidad_duplicados_y_mrun_ausente():
    current = pd.DataFrame({
        "mrun": ["1", "1", "2", "3", "4", None, "9"],
        "cod_inst": ["A"] * 7, "cod_carrera": ["C"] * 7,
        "nomb_inst": ["Universidad A"] * 7, "nomb_sede": ["Centro"] * 7,
        "nomb_carrera": ["Carrera C"] * 7, "area_conocimiento": ["Salud"] * 7,
        "nivel_global": ["Pregrado"] * 7, "tipo_inst_1": ["Universidades"] * 7,
        "anio_ing_carr_ori": [2024] * 6 + [2023],
    })
    following = pd.DataFrame({"mrun": ["1", "1", "2", "3", None],
                              "cod_inst": ["A", "A", "A", "IP", "A"],
                              "cod_carrera": ["C", "C", "D", "E", "C"]})
    result = aggregate_retention(current, following, 2024).iloc[0]
    assert result["n"] == 4
    assert result["sin_mrun"] == 1
    assert result["misma_carrera"] == 1
    assert result["misma_universidad"] == 2
    assert result["sistema"] == 3


def test_la_cobertura_de_cod_carrera_queda_registrada():
    """En 2007-2008 falta el código en ~20% de la matrícula.

    Sin código, la fila no puede calzar a nivel de carrera aunque la persona
    haya seguido en la misma, y la continuidad de carrera aparece hundida sin
    que nadie haya desertado. La cobertura se cuenta para poder descartar esos
    años en vez de leer la caída como un hecho.
    """
    current = pd.DataFrame({
        "mrun": ["1", "2", "3", "4"],
        "cod_inst": ["A"] * 4, "cod_carrera": ["C", "C", None, None],
        "nomb_inst": ["Universidad A"] * 4, "nomb_sede": ["Centro"] * 4,
        "nomb_carrera": ["Carrera C"] * 4, "area_conocimiento": ["Salud"] * 4,
        "nivel_global": ["Pregrado"] * 4, "tipo_inst_1": ["Universidades"] * 4,
        "anio_ing_carr_ori": [2007] * 4,
    })
    following = pd.DataFrame({"mrun": ["1", "2", "3", "4"], "cod_inst": ["A"] * 4,
                              "cod_carrera": ["C", "C", "C", "C"]})
    result = aggregate_retention(current, following, 2007).iloc[0]
    assert result["n"] == 4
    assert result["con_cod_carrera"] == 2
    # Los cuatro siguen en la misma universidad; solo dos pueden calzar carrera.
    assert result["misma_universidad"] == 4
    assert result["misma_carrera"] == 2
    resumen = summarize_retention(
        aggregate_retention(current, following, 2007), ["cohorte"]).iloc[0]
    assert resumen["cobertura_cod_carrera"] == pytest.approx(.5)


def test_tasas_ponderadas_y_denominador_cero():
    data = pd.DataFrame({"universidad": ["A", "A", "B"], "n": [10, 90, 0],
                         "misma_carrera": [10, 45, 0], "misma_universidad": [10, 60, 0],
                         "sistema": [10, 80, 0], "sin_mrun": [0, 1, 2]})
    summary = summarize_retention(data, ["universidad"]).set_index("universidad")
    assert summary.loc["A", "retencion_misma_carrera"] == pytest.approx(.55)
    assert pd.isna(summary.loc["B", "retencion_sistema"])


def test_seccion_publica_funciona_sin_predicciones(tmp_path):
    from streamlit.testing.v1 import AppTest
    # La vista pública debe mostrar instrucciones claras sin artefactos.
    script = "from pathlib import Path\nfrom student_analytics.ui.retention import render_retention\n"
    script += f"render_retention(Path({str(tmp_path)!r}))\n"
    app = AppTest.from_string(script, default_timeout=90).run()
    assert not app.exception
    assert "Aún no" in app.info[0].value


def test_navegacion_y_filtros_retencion():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    if not (root / "data/results/retention_universities.parquet").exists():
        pytest.skip("agregados públicos no generados")
    app = AppTest.from_file(str(root / "src/student_analytics/ui/app.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Retención universitaria").run()
    assert not app.exception
    assert app.title[0].value == "Retención universitaria"
    university = app.multiselect[0].options[0]
    app.multiselect[0].set_value([university]).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    app.selectbox[2].set_value("retencion_sistema").run()
    assert not app.exception
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Selecciona universidades" in info.value for info in app.info)
