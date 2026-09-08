"""Tests de la UI con el framework oficial de Streamlit.

`AppTest` ejecuta el script completo y captura cualquier excepcion, que es
lo que un `streamlit run` no revela: el servidor levanta igual y el error
solo aparece cuando un humano abre el navegador.

Se saltan si no hay resultados generados todavia.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "src" / "student_analytics" / "ui" / "app.py"
RESULTS = REPO_ROOT / "data" / "results"

pytestmark = pytest.mark.skipif(
    not list(RESULTS.glob("metrics_*.parquet")),
    reason="sin resultados (correr scripts/validate_lead_time.py --out data/results)",
)


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Alerta temprana").run()
    return at


def test_la_app_corre_sin_excepciones(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_tiene_las_tres_vistas(app):
    """Cockpit, Estudiante 360 y desempeno del modelo."""
    assert len(app.tabs) == 3


def test_muestra_kpis_y_graficos(app):
    assert len(app.metric) >= 4
    assert len(app.get("vega_lite_chart")) >= 3
    assert len(app.dataframe) >= 1


def test_la_capacidad_de_revision_cambia_la_captura(app):
    """El slider de capacidad debe mover la metrica de cobertura.

    Es la interaccion central del cockpit: responde 'cuanto alcanzo a
    cubrir con la gente que tengo'. Si no reacciona, la vista no sirve
    para decidir.
    """
    def captura(at):
        for m in at.metric:
            if "capturadas" in m.label.lower():
                return m.value
        pytest.fail("no se encontro la metrica de reprobaciones capturadas")

    baja = captura(app)
    app.sidebar.slider[0].set_value(45).run()
    alta = captura(app)
    assert not app.exception, [str(e.value) for e in app.exception]
    assert float(alta.strip("%")) > float(baja.strip("%"))


def test_cambiar_la_semana_no_rompe_la_vista(app):
    # El widget se re-obtiene en cada vuelta: tras cada run() los ids de
    # sesion cambian y una referencia guardada queda invalida.
    semanas = list(app.sidebar.get("select_slider")[0].options)
    for opcion in semanas:
        app.sidebar.get("select_slider")[0].set_value(opcion).run()
        assert not app.exception, f"semana {opcion}: {[str(e.value) for e in app.exception]}"


def test_advierte_sobre_la_naturaleza_de_los_datos(app):
    """Ninguna vista puede presentar estas cifras como hallazgos reales."""
    avisos = " ".join([w.value for w in app.warning] + [i.value for i in app.info]).lower()
    assert "sintetic" in avisos or "oulad" in avisos
