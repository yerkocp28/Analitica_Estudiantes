"""Tests de la selectividad de admisión (PAES × matrícula por MRUN)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_analytics.ingestion.paes import (
    attach_selectivity,
    institutional_selectivity,
    load_scores,
)

ROOT = Path(__file__).resolve().parents[1]
PAES = sorted((ROOT / "data/external/mineduc/paes_2024_puntajes").rglob("*.csv")) \
    if (ROOT / "data/external/mineduc/paes_2024_puntajes").exists() else []


def _cohorte(n_por_inst=40):
    """Cohorte sintética: dos instituciones con perfiles distintos."""
    filas = []
    for inst, base in [("A", 700), ("B", 550)]:
        for i in range(n_por_inst):
            filas.append({"mrun": int(f"{ord(inst)}{i:04d}"), "cod_inst": inst,
                          "_esperado": base + i})
    return pd.DataFrame(filas)


def test_una_persona_en_dos_carreras_cuenta_una_vez():
    """El perfil describe personas que ingresaron, no inscripciones.

    Sin deduplicar, quien se matricula en dos carreras de la misma
    universidad pesaría el doble en el promedio institucional.
    """
    cohorte = pd.DataFrame({"mrun": [1, 1, 2], "cod_inst": ["A", "A", "A"]})
    scores = pd.DataFrame({"mrun": [1, 2], "paes": [800.0, 400.0]})
    # Umbral en 1 para aislar la deduplicacion del filtro por tamanio.
    r = institutional_selectivity(cohorte, scores, min_takers=1).set_index("cod_inst")
    assert r.loc["A", "paes_promedio"] == pytest.approx(600.0)


def test_el_cero_no_se_toma_como_puntaje(tmp_path):
    """En estas bases 0 significa 'no rindió', no 'obtuvo cero'."""
    csv = tmp_path / "paes.csv"
    csv.write_text("MRUN;PROMEDIO_CM_MAX;PTJE_NEM;PTJE_RANKING\n"
                   "1;700;650;660\n2;0;0;0\n3;500;480;490\n",
                   encoding="utf-8-sig")
    d = load_scores(csv)
    assert len(d) == 3
    assert d.paes.notna().sum() == 2
    assert d.loc[d.mrun.eq(2), "paes"].isna().all()
    assert d.paes.mean() == pytest.approx(600.0)


def test_coma_decimal_se_interpreta():
    """Las bases del Mineduc usan coma decimal."""
    import io
    csv = io.StringIO("MRUN;PROMEDIO_CM_MAX\n1;650,5\n")
    tmp = Path(__file__).parent / "_tmp_paes.csv"
    tmp.write_text(csv.getvalue(), encoding="utf-8-sig")
    try:
        d = load_scores(tmp)
        assert d.paes.iloc[0] == pytest.approx(650.5)
    finally:
        tmp.unlink(missing_ok=True)


def test_cobertura_refleja_quienes_rindieron():
    cohorte = pd.DataFrame({"mrun": [1, 2, 3, 4], "cod_inst": ["A"] * 4})
    scores = pd.DataFrame({"mrun": [1, 2], "paes": [700.0, 600.0]})
    r = institutional_selectivity(cohorte, scores, min_takers=1).set_index("cod_inst")
    assert r.loc["A", "paes_cobertura"] == pytest.approx(.5)


def test_con_pocos_rendidores_el_promedio_se_anula():
    """Una media sobre cuatro personas es ruido, no selectividad.

    Se anula el indicador pero se conserva la cobertura: esa cifra baja es
    justamente la señal de que la institución recibe su cohorte por otras
    vías, y es información que hay que mostrar.
    """
    cohorte = pd.DataFrame({"mrun": range(200), "cod_inst": ["A"] * 200})
    scores = pd.DataFrame({"mrun": [0, 1, 2, 3], "paes": [700.0] * 4})
    r = institutional_selectivity(cohorte, scores).set_index("cod_inst")
    assert np.isnan(r.loc["A", "paes_promedio"])
    assert r.loc["A", "paes_cobertura"] == pytest.approx(4 / 200)


def test_los_cuartiles_ordenan():
    cohorte = _cohorte()
    scores = pd.DataFrame({"mrun": cohorte.mrun, "paes": cohorte._esperado.astype(float)})
    r = institutional_selectivity(cohorte, scores, min_coverage=0.0).set_index("cod_inst")
    for inst in ("A", "B"):
        assert r.loc[inst, "paes_p25"] <= r.loc[inst, "paes_promedio"] <= r.loc[inst, "paes_p75"]
        assert r.loc[inst, "paes_rango_intercuartil"] == pytest.approx(
            r.loc[inst, "paes_p75"] - r.loc[inst, "paes_p25"])
    assert r.loc["A", "paes_promedio"] > r.loc["B", "paes_promedio"]


def test_attach_no_pierde_universidades():
    perfil = pd.DataFrame({"cod_inst": ["A", "B", "C"], "nomb_inst": list("ABC")})
    sel = pd.DataFrame({"cod_inst": ["A", "B"], "paes_promedio": [700.0, 600.0],
                        "paes_cobertura": [.9, .8], "personas_cohorte": [10, 10]})
    unido = attach_selectivity(perfil, sel)
    assert len(unido) == 3
    assert unido.loc[unido.cod_inst.eq("C"), "paes_promedio"].isna().all()
    # `personas_cohorte` es de control interno y no debe filtrarse al perfil.
    assert "personas_cohorte" not in unido.columns


# ----------------------------------------------------------------------
# Sobre la base real, si está descargada
# ----------------------------------------------------------------------
@pytest.mark.skipif(not PAES, reason="PAES 2024 no descargada")
def test_los_puntajes_reales_estan_en_rango():
    d = load_scores(PAES[0])
    p = d.paes.dropna()
    assert len(p) > 100_000
    assert p.between(100, 1000).all(), "la PAES se mide entre 100 y 1000 puntos"


@pytest.mark.skipif(not (ROOT / "data/results/benchmark_profiles.parquet").exists(),
                    reason="perfiles no generados")
def test_el_perfil_trae_selectividad_coherente():
    d = pd.read_parquet(ROOT / "data/results/benchmark_profiles.parquet")
    d = d[d.cohorte.eq(2024)]
    if "paes_promedio" not in d.columns:
        pytest.skip("perfiles generados sin PAES")
    p = d.paes_promedio.dropna()
    assert p.between(400, 900).all()
    assert d.paes_cobertura.between(0, 1).all()
    # Validez de cara: las universidades mas selectivas del pais deben
    # quedar arriba. Si esto se rompe, el cruce por MRUN esta mal.
    top = d.nlargest(3, "paes_promedio").nomb_inst.str.upper().tolist()
    assert any("CATOLICA DE CHILE" in n for n in top)
