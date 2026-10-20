import json

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from piloto_fan.institucional import HOME
from piloto_fan.modelos_institucionales import (
    NUMERIC, SPEC, TRANSITIONS, datasets, load_run, partition,
)
from piloto_fan.models import design


def mini_frames():
    base = pd.DataFrame({
        "student_key": ["a", "b", "c", "d"], "term": [202410]*4,
        "career": ["Comercial"]*4, "campus": ["Talca"]*4,
        "career_code": ["INCO"]*4, "campus_code": ["TA"]*4, "schedule": ["Diurno"]*4,
        "fan_proposed": [True]*4, "plans": [1]*4, "regime": ["Semestral"]*4,
        "observed_results": [5]*4, "numeric_results": [5]*4,
        "numeric_mean": [5.0]*4, "numeric_sd": [1.0]*4, "numeric_coverage": [1.0]*4,
        "numeric_below4_count": [1.0]*4,
    })
    courses = pd.DataFrame({
        "student_key": ["a", "a", "b", "b", "c", "c"], "term": [202420]*6,
        "nrc": [1, 2, 1, 2, 1, 2], "grade_numeric": [3.0, np.nan, 5.0, np.nan, 5.0, 6.0],
    })
    enrollment = pd.DataFrame({"student_key": ["a", "b", "c"], "term": [202420]*3})
    return {"students": base, "courses": courses, "enrollment": enrollment}


def test_partial_and_missing_outcomes_are_not_successes():
    data, base, audit = datasets(mini_frames())
    y = data["academico"].set_index("student_key").target
    assert y["a"] == 1  # Observed low numeric grade establishes the event, despite another code.
    assert pd.isna(y["b"])  # A coded grade prevents declaring a full negative.
    assert y["c"] == 0
    assert pd.isna(y["d"])  # No next-semester history is not academic success/failure.
    assert len(base) == 4
    assert audit[0]["academic_unknown"] == 2
    assert data["matricula"].set_index("student_key").target["d"] == 1


def test_future_notes_do_not_change_source_features_and_missing_term_is_unknown():
    frames = mini_frames()
    initial, _, _ = datasets(frames)
    frames["courses"].loc[:, "grade_numeric"] = 1.0
    changed, _, _ = datasets(frames)
    pd.testing.assert_frame_equal(design(initial["academico"], SPEC), design(changed["academico"], SPEC))
    assert changed["academico"].target.notna().sum() == 3
    frames["enrollment"] = frames["enrollment"].iloc[0:0]
    missing, _, _ = datasets(frames)
    assert missing["matricula"].target.isna().all()
    assert not (set(SPEC["numeric"]+SPEC["categorical"]) &
                {"student_key", "outcome_term", "target", "plan_status", "student_status"})


def test_temporal_splits_use_explicit_source_and_outcome_periods():
    rows = []
    for source, target in TRANSITIONS:
        for i in range(200):
            rows.append({"student_key": str(i), "source_term": source, "outcome_term": target, "target": i % 2})
    parts = partition(pd.DataFrame(rows))
    assert [p.source_term.unique().tolist() for p in parts] == [[202410], [202420], [202510]]
    assert [p.outcome_term.unique().tolist() for p in parts] == [[202420], [202510], [202520]]
    # Existing students can legitimately reappear; identity is not a predictor.
    assert set(parts[0].student_key) == set(parts[2].student_key)


def latest():
    base = HOME/"artifacts/institucional_modelos"
    if not (base/"latest.json").exists():
        pytest.skip("Requiere entrenar modelos institucionales.")
    return base/json.loads((base/"latest.json").read_text())["run_id"]


def test_saved_models_reproduce_reserved_test_predictions_and_scope():
    folder = latest()
    report, frames = load_run(folder)
    for case, c in report["cases"].items():
        assert len(c["validation_candidates"]) == 3
        test = frames[case+"_dataset"]
        test = test.loc[test.source_term.eq(202510) & test.target.notna()]
        model = joblib.load(folder/(case+"_model.joblib"))["model"]
        scored = model.predict_proba(design(test, SPEC))[:, 1]
        pred = frames[case+"_predictions"]
        assert np.allclose(scored, pred.risk_score)
        assert not pred.student_key.duplicated().any()
        assert pred.outcome_term.eq(202520).all()
        assert pred.source_term.eq(202510).all()
        assert c["status"] == "experimental"
    s = report["segments"]
    assert len({c["algorithm"] for c in s["candidates"]}) == 3
    assert s["features"] == NUMERIC
    assert s["fit_term"] == 202410
    assert "student_key" not in json.dumps(report)


def test_new_views_navigate_filter_and_preserve_public_models():
    latest()
    app = AppTest.from_file(str(HOME/"app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Modelos institucionales").run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.selectbox(key="im_case").value == "academico"
    initial = int(app.metric[0].value.replace(".", ""))
    app.selectbox(key="im_campus").set_value("Talca").run()
    assert int(app.metric[0].value.replace(".", "")) < initial
    app.select_slider(key="im_capacity").set_value(30).run()
    assert not app.exception
    app.selectbox(key="im_case").set_value("matricula").run()
    assert not app.exception
    assert any("30" in warning.value for warning in app.warning)
    app.selectbox(key="im_case").set_value("segmentos").run()
    assert not app.exception, [e.message for e in app.exception]
    app.selectbox(key="im_profile_term").set_value(202410).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Resumen").run()
    assert not app.exception
