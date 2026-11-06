import numpy as np
import pandas as pd
import pytest

from student_analytics.modeling.benchmark_reporting import (continuity_breakdown, executive_html, report_html,
                                                            sensitivity)
from student_analytics.modeling.benchmark import build_profiles


def test_estados_sum_an_uno_y_separan_movilidad():
    frame = pd.DataFrame({"es_ua": [True, False], "n": [100, 200],
                          "misma_carrera": [70, 100], "misma_universidad": [80, 140], "sistema": [90, 180]})
    out = continuity_breakdown(frame)
    assert out.groupby("Referencia")["Proporción"].sum().eq(1).all()
    ua = out.loc[out.Referencia.eq("Universidad Autónoma")].set_index("Estado")
    assert ua.loc["Otra institución", "Inscripciones"] == 10
    assert ua.loc["Sin matrícula observada", "Inscripciones"] == 10
    frame.loc[0, "misma_carrera"] = 101
    with pytest.raises(ValueError, match="consistentes"):
        continuity_breakdown(frame)


def test_sensibilidad_pondera_sin_incluir_ua():
    frame = pd.DataFrame({"cohorte": [2024] * 5, "cod_inst": ["UA", "A", "B", "C", "D"],
                          "nomb_inst": ["UA", "A", "B", "C", "D"], "area_conocimiento": ["Salud"] * 5,
                          "n": [100, 10, 20, 30, 40], "misma_carrera": [90, 5, 10, 15, 40],
                          "misma_universidad": [90, 5, 10, 15, 40], "sistema": [90, 5, 10, 15, 40], "sin_mrun": [0] * 5})
    result = sensitivity(frame, 2024, "UA", ["A", "B", "C", "D"], "misma_carrera")
    assert result["Pares seleccionados"].tolist() == [3, 4]
    assert result["Continuidad pares (%)"].tolist() == [50., 70.]
    assert result["Brecha UA (pp)"].tolist() == pytest.approx([40., 20.])


def test_informe_escapa_nombres_y_preserva_filtros():
    stats = dict(ua=.8, pares=.7, nacional=.75, ajustada=.72, ua_comun=.8, cobertura=1.)
    table = pd.DataFrame({"Universidad": ["<script>bad</script>"]})
    html = executive_html(2024, "Salud & Educación", "Misma carrera", "Manual", stats, table, table, "2026-09-08")
    assert "<script>" not in html
    assert "Salud &amp; Educación" in html
    assert "80.0%" in html


def test_informe_generico_escapa_hallazgos_secciones_y_celdas():
    tabla = pd.DataFrame({"Universidad": ["<script>x</script>"], "Valor": [None]})
    html = report_html("Título & más", "Sub", ["<b>hallazgo</b>"], [("Tabla", tabla), ("Nota", "a < b")],
                       ["fuente <i>"], "06/11/2026")
    assert "<script" not in html and "<b>" not in html and "<i>" not in html
    assert "Título &amp; más" in html and "a &lt; b" in html and "Sin datos" in html


def test_informe_fan_se_genera_con_los_datos_versionados(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not (root / "deploy/data/retention_universities.parquet").exists():
        pytest.skip("Faltan los agregados versionados.")
    spec = importlib.util.spec_from_file_location("informe_pares_fan", root / "scripts/informe_pares_fan.py")
    informe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(informe)
    monkeypatch.setattr(informe, "SALIDA", tmp_path / "informe.html")
    assert informe.main() == 0
    html = (tmp_path / "informe.html").read_text(encoding="utf-8")
    assert "<script" not in html and "Lectura ejecutiva" in html
    assert "Universidad Autonoma de Chile" in html  # La UA aparece, pero nunca como su propio par.
    assert html.count("Universidad Autonoma de Chile") == 1


def test_ingreso_desconocido_no_equivale_a_no_regular():
    frame = pd.DataFrame({"cod_inst": ["1"] * 3, "nomb_inst": ["UA"] * 3, "cod_sede": ["S"] * 3,
                          "area_conocimiento": ["Salud"] * 3, "region_sede": ["Maule"] * 3,
                          "jornada": ["Diurna"] * 3, "modalidad": ["Presencial"] * 3,
                          "forma_ingreso": ["1-Regular", "2-PACE", None]})
    profile = build_profiles(frame, 2024).iloc[0]
    assert profile.ingreso_no_regular == .5
    assert profile.ingreso_pace == .5
    assert profile.cobertura_ingreso == pytest.approx(2 / 3)
