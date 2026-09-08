"""Tests del generador sintetico."""
from __future__ import annotations

import numpy as np
import pytest

from student_analytics.config import Settings, load_synthetic_config
from student_analytics.synthetic.generator import (
    SyntheticGenerator,
    solve_intercept,
    solve_threshold,
    _sigmoid,
)
from student_analytics.synthetic.profiles import PROFILES

KEYS = ["student_id", "course_id", "term_id"]


@pytest.fixture(scope="module")
def cfg():
    return load_synthetic_config()


@pytest.fixture(scope="module")
def settings():
    return Settings.load()


@pytest.fixture(scope="module")
def ds(cfg, settings):
    """Dataset chico pero suficiente para que las tasas sean estables."""
    return SyntheticGenerator(cfg, settings.raw, n_students=800, seed=42).generate()


# ----------------------------------------------------------------------
# Estructura
# ----------------------------------------------------------------------
def test_todas_las_tablas_no_vacias(ds):
    for name, df in ds.tables().items():
        assert len(df) > 0, f"tabla vacia: {name}"


def test_grano_unico_por_tabla(ds):
    assert ds.student_master["student_id"].is_unique
    assert ds.course_master["course_id"].is_unique
    assert not ds.outcomes.duplicated(KEYS).any()
    assert not ds.enrollment.duplicated(KEYS).any()
    for name in ("attendance_weekly", "activity_weekly", "assignments_weekly"):
        df = ds.tables()[name]
        assert not df.duplicated(KEYS + ["week"]).any(), f"grano duplicado en {name}"


def test_sin_nulos_en_claves(ds):
    for name, df in ds.tables().items():
        for key in [c for c in KEYS + ["week"] if c in df.columns]:
            assert df[key].notna().all(), f"{name}.{key} tiene nulos"


def test_integridad_referencial(ds):
    students = set(ds.student_master["student_id"])
    courses = set(ds.course_master["course_id"])
    assert set(ds.enrollment["student_id"]) <= students
    assert set(ds.enrollment["course_id"]) <= courses
    assert set(ds.outcomes["student_id"]) <= students


def test_variables_latentes_van_con_guion_bajo(ds):
    """Las latentes no existen en datos reales: deben ser inconfundibles.

    El prefijo `_` es el contrato que permite excluirlas de las features
    con una sola regla en vez de una lista que hay que mantener.
    """
    latentes = {"_latent_profile", "_latent_ability", "_p_fail_true"}
    for df in ds.tables().values():
        for col in df.columns:
            if col in latentes:
                assert col.startswith("_")


# ----------------------------------------------------------------------
# Rangos y dominios
# ----------------------------------------------------------------------
def test_notas_en_escala_chilena(ds, settings):
    lo = settings.academic["grade_min"]
    hi = settings.academic["grade_max"]
    for col, df in (("assessment_grade", ds.grades_weekly),
                    ("final_grade", ds.outcomes)):
        v = df[col].dropna()
        assert v.between(lo, hi).all(), f"{col} fuera de [{lo}, {hi}]"


def test_nota_final_coherente_con_reprobacion(ds, settings):
    """Un reprobado por nota no puede tener nota final de aprobacion."""
    pass_line = settings.academic["grade_pass"]
    fallados = ds.outcomes[ds.outcomes["fail_by_grade"] == 1]
    aprobados = ds.outcomes[ds.outcomes["fail_by_grade"] == 0]
    assert (fallados["final_grade"] < pass_line).all()
    assert (aprobados["final_grade"] >= pass_line).all()


def test_asistencia_y_proporciones_en_rango(ds):
    att = ds.attendance_weekly["attendance_rate"].dropna()
    assert att.between(0, 1).all()
    assert ds.outcomes["final_attendance"].between(0, 1).all()


def test_entregas_son_consistentes(ds):
    a = ds.assignments_weekly
    assert (a["submitted"] <= a["assigned"]).all()
    assert (a["late"] <= a["submitted"]).all()
    assert (a["missing"] == a["assigned"] - a["submitted"]).all()
    assert (a[["assigned", "submitted", "late", "missing"]] >= 0).all().all()


def test_targets_son_binarios(ds):
    for col in ("fail_by_grade", "fail_by_attendance", "failed"):
        assert set(ds.outcomes[col].unique()) <= {0, 1}


def test_targets_separados_no_se_mezclan(ds):
    """`failed` debe ser la union de ambas causales, no una de ellas."""
    o = ds.outcomes
    esperado = ((o["fail_by_grade"] == 1) | (o["fail_by_attendance"] == 1)).astype(int)
    assert (o["failed"] == esperado).all()


# ----------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------
def test_mismo_seed_produce_datos_identicos(cfg, settings):
    a = SyntheticGenerator(cfg, settings.raw, n_students=200, seed=7).generate()
    b = SyntheticGenerator(cfg, settings.raw, n_students=200, seed=7).generate()
    for name in a.tables():
        assert a.tables()[name].equals(b.tables()[name]), f"{name} no es reproducible"


def test_distinto_seed_produce_datos_distintos(cfg, settings):
    a = SyntheticGenerator(cfg, settings.raw, n_students=200, seed=7).generate()
    b = SyntheticGenerator(cfg, settings.raw, n_students=200, seed=8).generate()
    assert not a.outcomes["final_grade"].equals(b.outcomes["final_grade"])


# ----------------------------------------------------------------------
# Calibracion: las anclas de config deben ser vinculantes
# ----------------------------------------------------------------------
def test_solve_intercept_alcanza_la_tasa_objetivo():
    rng = np.random.default_rng(0)
    score = rng.normal(0, 1.2, 20_000)
    for target in (0.05, 0.20, 0.50):
        b = solve_intercept(score, target)
        assert abs(float(_sigmoid(score + b).mean()) - target) < 1e-3


def test_solve_intercept_rechaza_objetivos_invalidos():
    score = np.zeros(10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            solve_intercept(score, bad)


def test_solve_threshold_deja_la_proporcion_pedida_abajo():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 20_000)
    thr = solve_threshold(v, 0.07)
    assert abs(float((v < thr).mean()) - 0.07) < 0.01


def test_tasa_de_reprobacion_respeta_el_ancla(ds, cfg):
    """El ancla de config manda sobre los pesos del riesgo latente."""
    objetivo = cfg["calibration"]["course_fail_rate"]
    obtenido = ds.outcomes["fail_by_grade"].mean()
    assert abs(obtenido - objetivo) < 0.04, f"esperado ~{objetivo}, obtenido {obtenido:.3f}"


def test_reprobacion_por_inasistencia_es_minoritaria(ds, cfg):
    objetivo = cfg["calibration"]["attendance_fail_rate"]
    obtenido = ds.outcomes["fail_by_attendance"].mean()
    assert abs(obtenido - objetivo) < 0.03
    assert obtenido < ds.outcomes["fail_by_grade"].mean()


def test_ningun_perfil_esta_saturado(ds):
    """Ningun perfil debe reprobar 0% ni 100%.

    La saturacion vuelve trivial el problema: cualquier modelo luce
    excelente y las metricas dejan de informar sobre el pipeline.
    """
    sm = ds.student_master[["student_id", "_latent_profile"]]
    tasas = ds.outcomes.merge(sm, on="student_id").groupby("_latent_profile")["failed"].mean()
    assert tasas.min() > 0.01, f"perfil sin reprobaciones: {tasas.idxmin()}"
    assert tasas.max() < 0.95, f"perfil saturado: {tasas.idxmax()}"


def test_los_perfiles_de_riesgo_reprueban_mas_que_los_buenos(ds):
    sm = ds.student_master[["student_id", "_latent_profile"]]
    tasas = ds.outcomes.merge(sm, on="student_id").groupby("_latent_profile")["failed"].mean()
    assert tasas["A_high_performer"] < tasas["D_academic_risk"]
    assert tasas["A_high_performer"] < tasas["E_disengaging"]


def test_ramos_cuantitativos_reprueban_mas(ds):
    cm = ds.course_master[["course_id", "difficulty_tier"]]
    tasas = ds.outcomes.merge(cm, on="course_id").groupby("difficulty_tier")["failed"].mean()
    assert tasas["quantitative_core"] > tasas["general"]


def test_variables_de_ingreso_predicen_debil(ds):
    """Replica el hallazgo de ULagos: PAES y NEM casi no predicen.

    Si esta correlacion se vuelve fuerte, el generador dejo de reproducir
    el fenomeno que motiva el proyecto y las conclusiones sobre lead time
    dejarian de ser trasladables.
    """
    o = ds.outcomes.merge(
        ds.student_master[["student_id", "admission_score", "hs_gpa"]], on="student_id")
    for col in ("admission_score", "hs_gpa"):
        r = abs(float(np.corrcoef(o[col], o["failed"])[0, 1]))
        assert r < 0.25, f"{col} correlaciona {r:.3f} con reprobacion (demasiado)"


# ----------------------------------------------------------------------
# Calidad de datos simulada
# ----------------------------------------------------------------------
def test_cursos_sin_registro_de_asistencia_vienen_nulos(ds):
    """Ausencia de registro debe ser NULL, jamas cero.

    Un cero se lee aguas abajo como 'no asistio' y convierte un problema
    de calidad de datos en una alerta de riesgo falsa.
    """
    att = ds.attendance_weekly
    sin_registro = att[~att["attendance_recorded"]]
    assert len(sin_registro) > 0, "el escenario no se esta simulando"
    assert sin_registro["attendance_rate"].isna().all()
    assert sin_registro["sessions_attended"].isna().all()


def test_plataforma_de_baja_adopcion_queda_marcada(ds):
    can = ds.activity_weekly
    baja = can[can["platform_adoption"] == "low"]
    assert len(baja) > 0
    assert not baja["activity_reliable"].any()
    assert can[can["platform_adoption"] == "high"]["activity_reliable"].all()


def test_hay_notas_cargadas_con_atraso(ds):
    g = ds.grades_weekly
    assert (g["available_from_week"] > g["week"]).any()
    assert (g["available_from_week"] >= g["week"]).all()


def test_perfiles_declarados_calzan_con_los_implementados(cfg):
    assert set(cfg["profiles"]) == set(PROFILES)
