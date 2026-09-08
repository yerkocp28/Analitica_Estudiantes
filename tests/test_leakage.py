"""Tests anti-leakage temporal (doc maestro 16.2, 74, 75).

El leakage es el Riesgo 4 del documento maestro y el unico error que no se
detecta mirando metricas: produce modelos con AUC excelente que fracasan
en produccion, porque en produccion el futuro todavia no ocurrio.

La prueba definitiva es de invarianza, no de inspeccion: se calculan las
features de la semana N, se CORROMPE todo lo posterior a N, y se recalculan.
Si algo cambia, el pipeline estaba mirando hacia adelante.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_lead_time import build_features_at_week  # noqa: E402

from student_analytics.config import Settings, load_synthetic_config  # noqa: E402
from student_analytics.synthetic.generator import SyntheticGenerator  # noqa: E402


@pytest.fixture(scope="module")
def data():
    cfg = load_synthetic_config()
    settings = Settings.load()
    ds = SyntheticGenerator(cfg, settings.raw, n_students=300, seed=11).generate()
    return ds.tables()


SCORING_WEEK = 8


def _corromper_futuro(data: dict, week: int) -> dict:
    """Reemplaza con basura todo dato posterior a `week`.

    Si las features de la semana `week` sobreviven intactas a esto, no hay
    forma de que esten leyendo el futuro.
    """
    rng = np.random.default_rng(999)
    out = {k: v.copy() for k, v in data.items()}

    for name in ("attendance_weekly", "canvas_weekly", "assignments_weekly"):
        df = out[name]
        futuro = df["week"] > week
        for col in df.columns:
            if col in ("student_id", "course_id", "term_id", "week"):
                continue
            if df[col].dtype.kind in "fi":
                df.loc[futuro, col] = rng.integers(500, 999, futuro.sum())
            elif df[col].dtype == bool:
                df.loc[futuro, col] = ~df.loc[futuro, col]
        out[name] = df

    g = out["grades_weekly"]
    futuro_g = g["available_from_week"] > week
    g.loc[futuro_g, "assessment_grade"] = 7.0
    out["grades_weekly"] = g
    return out


def test_features_no_cambian_si_se_corrompe_el_futuro(data):
    antes = build_features_at_week(data, SCORING_WEEK)
    despues = build_features_at_week(_corromper_futuro(data, SCORING_WEEK), SCORING_WEEK)

    antes = antes.sort_values(["student_id", "course_id", "term_id"]).reset_index(drop=True)
    despues = despues.sort_values(["student_id", "course_id", "term_id"]).reset_index(drop=True)

    assert list(antes.columns) == list(despues.columns)
    assert len(antes) == len(despues)
    for col in antes.columns:
        if antes[col].dtype.kind in "fc":
            np.testing.assert_allclose(
                antes[col].to_numpy(dtype=float), despues[col].to_numpy(dtype=float),
                equal_nan=True, err_msg=f"LEAKAGE: la columna '{col}' cambio")
        else:
            assert antes[col].equals(despues[col]), f"LEAKAGE: la columna '{col}' cambio"


def test_notas_atrasadas_no_entran_antes_de_estar_disponibles(data):
    """Filtrar por `week` en vez de `available_from_week` es leakage silencioso.

    Una nota de la semana 8 cargada en la semana 10 existe en la base con
    week=8. Usarla en el scoring de la semana 8 significa entrenar con
    informacion que en produccion no habria estado.
    """
    g = data["grades_weekly"]
    atrasadas = g[g["available_from_week"] > g["week"]]
    assert len(atrasadas) > 0, "el escenario no se esta simulando"

    week = int(atrasadas["week"].iloc[0])
    feats = build_features_at_week(data, week)

    visibles = g[g["available_from_week"] <= week]
    esperado = visibles.groupby(["student_id", "course_id", "term_id"]).size()

    obtenido = feats.set_index(["student_id", "course_id", "term_id"])["n_grades"]
    comun = esperado.index.intersection(obtenido.index)
    np.testing.assert_array_equal(
        esperado.loc[comun].to_numpy(), obtenido.loc[comun].to_numpy().astype(int))


def test_ninguna_feature_usa_columnas_de_resultado(data):
    """Las columnas del outcome no pueden aparecer en el set de features."""
    feats = build_features_at_week(data, SCORING_WEEK)
    prohibidas = {"final_grade", "final_attendance", "fail_by_grade",
                  "fail_by_attendance", "failed", "_p_fail_true",
                  "_latent_profile", "_latent_ability"}
    assert not (set(feats.columns) & prohibidas)


def test_features_mas_tardias_son_al_menos_tan_informativas(data):
    """Mas semanas observadas no pueden significar menos informacion."""
    n3 = build_features_at_week(data, 3)["n_grades"].sum()
    n12 = build_features_at_week(data, 12)["n_grades"].sum()
    assert n12 > n3
