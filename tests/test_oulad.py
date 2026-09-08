"""Tests del adaptador OULAD.

Se saltan si el dataset no esta descargado, para que la suite corra en
cualquier maquina:

    python scripts/download_oulad.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from student_analytics.features.builder import build_features_at_week
from student_analytics.ingestion.oulad import eligible_at_week, load_oulad

OULAD_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "oulad"
pytestmark = pytest.mark.skipif(
    not (OULAD_DIR / "studentInfo.csv").exists()
    and not list(OULAD_DIR.rglob("studentInfo.csv")),
    reason="OULAD no descargado (python scripts/download_oulad.py)",
)

KEYS = ["student_id", "course_id", "term_id"]


@pytest.fixture(scope="module")
def data():
    return load_oulad(OULAD_DIR)


def test_produce_el_esquema_canonico(data):
    """El adaptador debe entregar las mismas tablas que el generador.

    Es la propiedad que hace barata la migracion a Banner/Canvas: si el
    mismo feature builder corre sobre dos fuentes sin cambios, correra
    sobre la tercera.
    """
    esperadas = {"student_master", "course_master", "enrollment",
                 "activity_weekly", "assignments_weekly", "grades_weekly", "outcomes"}
    assert esperadas <= set(data)
    # OULAD es educacion a distancia: no existe asistencia. Debe estar
    # AUSENTE, no presente y vacia.
    assert "attendance_weekly" not in data


def test_claves_sin_nulos(data):
    for name, df in data.items():
        for k in [c for c in KEYS + ["week"] if c in df.columns]:
            assert df[k].notna().all(), f"{name}.{k} tiene nulos"


def test_semanas_son_positivas(data):
    for name in ("activity_weekly", "assignments_weekly", "grades_weekly"):
        assert (data[name]["week"] >= 1).all(), name


def test_terminos_estan_ordenados_temporalmente(data):
    """La validacion temporal depende de este orden."""
    terms = sorted(data["outcomes"]["term_id"].dropna().unique())
    assert terms == list(range(1, len(terms) + 1))


def test_targets_binarios_y_coherentes(data):
    o = data["outcomes"]
    for col in ("fail_by_grade", "withdrawn", "failed"):
        assert set(o[col].unique()) <= {0, 1}
    # `failed` es la union: reprobar o retirarse.
    assert (o["failed"] >= o["fail_by_grade"]).all()
    assert (o["failed"] >= o["withdrawn"]).all()
    # Un mismo caso no puede ser Fail y Withdrawn a la vez.
    assert not ((o["fail_by_grade"] == 1) & (o["withdrawn"] == 1)).any()


def test_entregas_atrasadas_se_registran_en_su_semana(data):
    a = data["assignments_weekly"]
    assert (a["late"] > 0).any(), "no hay entregas atrasadas"
    assert (a["missing"] >= 0).all()


def test_elegibilidad_excluye_a_quienes_ya_se_retiraron(data):
    """Quien ya se dio de baja no es un caso a predecir: ya ocurrio."""
    o = data["outcomes"]
    for week in (4, 12, 26):
        elegibles = eligible_at_week(o, week)
        assert len(elegibles) <= len(o)
        ya_fuera = elegibles["unreg_week"].dropna()
        assert (ya_fuera > week).all(), f"semana {week}: hay bajas previas incluidas"
    # Debe filtrar de verdad, no ser un no-op.
    assert len(eligible_at_week(o, 26)) < len(eligible_at_week(o, 1))


def test_el_feature_builder_corre_sin_asistencia(data):
    """La fuente sin asistencia produce NaN explicito, nunca ceros.

    Imputar cero equivaldria a afirmar que nadie asistio nunca.
    """
    feats = build_features_at_week(data, 12)
    assert len(feats) > 0
    assert feats["attendance_cum"].isna().all()
    assert feats["attendance_delta"].isna().all()
    # El resto de las familias si debe tener senal.
    assert feats["activity_mean"].notna().any()
    assert feats["submission_rate"].notna().any()
    assert feats["mean_grade"].notna().any()


def test_mas_semanas_acumulan_mas_informacion(data):
    temprano = build_features_at_week(data, 4)
    tarde = build_features_at_week(data, 20)
    assert tarde["n_grades"].sum() > temprano["n_grades"].sum()
    assert tarde["activity_mean"].notna().sum() >= temprano["activity_mean"].notna().sum()


def test_poblacion_fija_mantiene_n_y_prevalencia_constantes(data):
    """Con cohorte fija, lo unico que cambia entre semanas es la informacion.

    Si el n o la prevalencia varian, la curva de lead time mezcla 'mas datos'
    con 'poblacion distinta' y sus filas dejan de ser comparables.
    """
    from student_analytics.modeling.lead_time import evaluate_lead_time

    m, _ = evaluate_lead_time(
        data, scoring_weeks=[8, 16, 26], weeks_total=39,
        eligibility=eligible_at_week, population="fixed")
    assert m["n_test"].nunique() == 1, m["n_test"].tolist()
    assert m["prevalence"].nunique() == 1, m["prevalence"].tolist()
    # Y la curva debe seguir subiendo: mas semanas, mas poder predictivo.
    assert m["auc"].is_monotonic_increasing


def test_poblacion_rolling_encoge_semana_a_semana(data):
    """El modo alternativo si deja variar la poblacion, a proposito."""
    from student_analytics.modeling.lead_time import evaluate_lead_time

    m, _ = evaluate_lead_time(
        data, scoring_weeks=[8, 16, 26], weeks_total=39,
        eligibility=eligible_at_week, population="rolling")
    assert m["n_test"].is_monotonic_decreasing
    assert m["n_test"].nunique() > 1


def test_population_rechaza_valores_invalidos(data):
    from student_analytics.modeling.lead_time import evaluate_lead_time

    with pytest.raises(ValueError, match="population"):
        evaluate_lead_time(data, scoring_weeks=[8], weeks_total=39,
                           eligibility=eligible_at_week, population="cualquiera")
