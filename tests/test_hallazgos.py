"""Tests de los hallazgos: que las cifras se calculen y no se escriban a mano."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_analytics.modeling import findings as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PERFILES = RESULTS / "benchmark_profiles.parquet"


def _perfil(n=40, cohorte=2024, **cambios):
    rng = np.random.default_rng(7)
    d = pd.DataFrame({
        "cohorte": cohorte,
        "cod_inst": [str(i) for i in range(n)],
        "nomb_inst": ["UNIVERSIDAD AUTONOMA DE CHILE"] + [f"UNIVERSIDAD {i}" for i in range(1, n)],
        "paes_promedio": rng.normal(600, 40, n),
        "paes_percentil_promedio": rng.uniform(.2, .9, n),
        "paes_cobertura": rng.uniform(.5, 1, n),
        "paes_instrumento": "PAES",
        "acreditacion_anios": rng.integers(2, 8, n).astype(float),
        "titulacion_oportuna": rng.uniform(.3, .8, n),
        "titulacion_cohorte_carrera": rng.uniform(.2, .7, n),
        "titulacion_cohorte_universidad": rng.uniform(.3, .8, n),
        "titulacion_cohorte_sistema": rng.uniform(.4, .9, n),
    })
    for k, v in cambios.items():
        d[k] = v
    return d


# ----------------------------------------------------------------------
def test_sin_artefactos_devuelve_none_en_vez_de_reventar(tmp_path):
    """El proyecto se arma por partes; la vista debe sobrevivir a lo que falte."""
    assert F.argumento_central(tmp_path) is None
    assert F.serie_retencion(tmp_path) is None
    assert F.retencion_por_sede(tmp_path) is None
    assert F.posicion_selectividad(pd.DataFrame()) is None
    assert F.contraste_titulacion(pd.DataFrame()) is None


def test_un_perfil_sin_selectividad_no_inventa_posicion():
    d = _perfil().drop(columns=["paes_percentil_promedio"])
    assert F.posicion_selectividad(d) is None


def test_la_posicion_se_calcula_sobre_el_percentil_no_sobre_el_puntaje():
    """El puntaje crudo no cruza el cambio de escala de 2023; el percentil sí."""
    d = _perfil(n=5)
    d["paes_percentil_promedio"] = [.9, .1, .2, .3, .4]   # la UA es la primera
    d["paes_promedio"] = [500, 900, 900, 900, 900]        # y la peor en puntaje
    r = F.posicion_selectividad(d).iloc[0]
    assert r.ua_percentil == pytest.approx(.9)
    assert r.posicion_ua == pytest.approx(.8)             # supera a 4 de 5


def test_el_contraste_conserva_el_signo_de_cada_indicador():
    """El hallazgo es que los dos indicadores apuntan al revés."""
    d = _perfil(n=60)
    # Transversal en contra de la selectividad; por cohorte a favor.
    d["titulacion_oportuna"] = 1 - d.paes_percentil_promedio
    d["titulacion_cohorte_universidad"] = d.paes_percentil_promedio
    r = F.contraste_titulacion(d).set_index("variable")
    assert r.loc["titulacion_oportuna", "paes_percentil_promedio"] < -0.9
    assert r.loc["titulacion_cohorte_universidad", "paes_percentil_promedio"] > 0.9


def test_con_pocas_universidades_no_se_reporta_correlacion():
    """Una correlación sobre diez puntos no es un hallazgo."""
    assert F.contraste_titulacion(_perfil(n=12)) is None


def test_las_trampas_documentan_el_limite_permanente():
    d = F.trampas_datos(RESULTS, None)
    assert len(d) > 10
    # La PSU no es un bug por arreglar: es una restricción de la fuente.
    psu = d[d.trampa.str.contains("PSU")]
    assert len(psu) == 1
    assert psu.estado.iloc[0] == "Límite permanente"


def test_el_aporte_de_fuentes_cuenta_variables_del_perfil_real():
    d = F.aporte_fuentes(RESULTS, _perfil())
    assert not d.empty
    paes = d[d.fuente.str.contains("admisión")].iloc[0]
    assert paes.variables >= 4          # paes_promedio, percentil, cobertura, instrumento


# ----------------------------------------------------------------------
# Sobre los artefactos reales
# ----------------------------------------------------------------------
@pytest.mark.skipif(not (RESULTS / "metrics_oulad.parquet").exists(),
                    reason="OULAD no evaluado")
def test_el_argumento_central_ordena_conducta_sobre_expediente():
    """Si esto se invierte, el argumento del proyecto deja de sostenerse."""
    d = F.argumento_central(RESULTS)
    if d is None or len(d) < 2:
        pytest.skip("faltan insumos del argumento")
    previo, conducta = d.iloc[0], d.iloc[1]
    assert conducta.auc > previo.auc
    assert previo.auc > 0.5


@pytest.mark.skipif(not (RESULTS / "retention_universities.parquet").exists(),
                    reason="retención no generada")
def test_la_serie_marca_las_cohortes_sin_codigo_de_carrera():
    d = F.serie_retencion(RESULTS)
    assert d is not None and not d.empty
    sin_codigo = d[d.cohorte.isin(F.COHORTES_SIN_CODIGO)]
    if not sin_codigo.empty:
        assert not sin_codigo.carrera_comparable.any()
        # Y la continuidad de universidad de esos años sí es usable.
        assert sin_codigo.retencion_misma_universidad_ua.between(0, 1).all()
    assert d.retencion_misma_universidad_ua.between(0, 1).all()


@pytest.mark.skipif(not PERFILES.exists(), reason="perfiles no generados")
def test_la_vista_de_hallazgos_renderiza():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src/student_analytics/ui/app.py"),
                            default_timeout=300).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    seccion = [r for r in app.radio if "Hallazgos" in (r.options or [])]
    assert seccion, "la sección Hallazgos no está en la barra lateral"
    seccion[0].set_value("Hallazgos").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.title[0].value == "Hallazgos"
    # Las tres métricas del argumento central más las de cada pestaña.
    assert len(app.metric) >= 6
    etiquetas = [m.label for m in app.metric]
    assert any("expediente previo" in e for e in etiquetas)
