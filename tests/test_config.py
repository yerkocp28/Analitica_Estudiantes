"""Tests de configuracion y perfiles."""
from __future__ import annotations

import pytest

from student_analytics.config import Settings, _validate_synthetic, load_synthetic_config
from student_analytics.synthetic.profiles import PROFILES, get_profile


def test_settings_carga():
    s = Settings.load()
    assert s.academic["grade_pass"] == 4.0
    assert s.academic["grade_min"] < s.academic["grade_max"]
    assert s.academic["scoring_weeks"] == sorted(s.academic["scoring_weeks"])
    assert max(s.academic["scoring_weeks"]) < s.academic["weeks_per_term"]


def test_config_sintetica_es_consistente():
    cfg = load_synthetic_config()
    assert cfg["volume"]["n_students"] > 0
    lo, hi = cfg["volume"]["courses_per_student_term"]
    assert 0 < lo <= hi


@pytest.mark.parametrize("clave", ["campus", "profiles", "careers", "canvas_adoption"])
def test_validacion_detecta_proporciones_que_no_suman_uno(clave):
    cfg = load_synthetic_config()
    if clave == "profiles":
        cfg[clave]["A_high_performer"]["prevalence"] += 0.2
    elif clave == "careers":
        cfg[clave][0]["weight"] += 0.2
    else:
        primera = next(iter(cfg[clave]))
        cfg[clave][primera] += 0.2
    with pytest.raises(ValueError, match="1.0"):
        _validate_synthetic(cfg)


def test_hay_siete_perfiles():
    assert len(PROFILES) == 7


def test_parametros_de_perfil_en_rango():
    for nombre, p in PROFILES.items():
        assert 0.0 <= p.attendance_start <= 1.0, nombre
        assert 0.0 <= p.submission_start <= 1.0, nombre
        assert 1.0 <= p.ability_mean <= 7.0, nombre
        assert 0.0 <= p.historical_pass_rate <= 1.0, nombre
        assert p.attendance_volatility >= 0, nombre


def test_los_perfiles_se_distinguen_por_trayectoria_no_solo_por_nivel():
    """E se cae y G remonta: si ambos derivaran igual, el modelo no podria
    distinguir un desenganche de una recuperacion."""
    assert PROFILES["E_disengaging"].attendance_drift < -0.2
    assert PROFILES["G_recovering"].attendance_drift > 0.2
    assert PROFILES["C_struggling_engaged"].submission_start > 0.8
    assert PROFILES["B_silent_high"].canvas_start < 0.4


def test_get_profile_falla_con_nombre_desconocido():
    with pytest.raises(KeyError):
        get_profile("Z_inexistente")
