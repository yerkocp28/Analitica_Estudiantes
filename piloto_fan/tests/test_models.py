import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from piloto_fan.models import FEATURES, capture, design, evaluate_case, temporal_split
from piloto_fan.segments import segment

CFG = {**json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8")),
       "minimum_fit": 40, "minimum_evaluation": 40, "minimum_class": 5, "bootstrap_repetitions": 30,
       "segment_minimum": 5, "segment_maximum_share": .95, "segment_k": [2, 3]}


def sample():
    rng = np.random.default_rng(10)
    rows = []
    for year in range(2019, 2026):
        for i in range(60):
            score = rng.uniform()
            rows.append(dict(mrun=f"{year}-{i}", cohorte=year, score_year=year, outcome_year=year+1,
                             label_ready_year=year+2, target=int(rng.random() < .6-.4*score),
                             nomb_carrera="A" if i % 2 else "B", nomb_sede="S", jornada="Diurna",
                             duracion_nominal_anios=5, percentil_admision=score,
                             percentil_nem=rng.uniform(), percentil_ranking=rng.uniform()))
    return pd.DataFrame(rows)


def test_constant_predictions_have_random_capture_even_with_sorted_labels():
    y = np.r_[np.ones(20), np.zeros(80)]
    assert capture(y, np.ones(100)) == pytest.approx(.2)
    assert capture(y, np.ones(100), .1) == pytest.approx(.1)


def test_maturity_is_respected_for_training_and_model_selection():
    a, v, final, t = temporal_split(sample(), CFG)
    assert a.label_ready_year.max() <= v.score_year.min()
    assert v.label_ready_year.max() <= t.score_year.min()
    assert final.label_ready_year.max() <= t.score_year.min()
    assert not set(final.mrun) & set(t.mrun)
    assert t.score_year.iloc[0] == 2025
    assert v.score_year.iloc[0] == 2023


def test_incomplete_test_year_is_not_selected():
    d = sample()
    d.loc[d.score_year.eq(2025) & d.index.to_series().mod(2).eq(0), "target"] = np.nan
    _, _, _, t = temporal_split(d, CFG)
    assert t.score_year.iloc[0] == 2024


def test_feature_whitelist_drops_targets_identifiers_and_future_fields():
    d = sample().assign(final_grade=7, future_title=1)
    x = design(d, FEATURES["continuidad"])
    assert not {"mrun", "target", "final_grade", "future_title", "outcome_year"} & set(x.columns)


def test_test_outcomes_cannot_select_algorithm():
    d = sample()
    with threadpool_limits(limits=1):
        a, _, pa = evaluate_case(d, "continuidad", CFG)
        modified = d.copy()
        modified.loc[modified.score_year.eq(2025), "target"] = 1 - modified.loc[modified.score_year.eq(2025), "target"]
        b, _, pb = evaluate_case(modified, "continuidad", CFG)
    assert a["winner"] == b["winner"]
    assert a["validation_candidates"] == b["validation_candidates"]
    np.testing.assert_allclose(pa.risk_score, pb.risk_score)
    assert len(a["validation_candidates"]) == 3


def test_insufficient_data_is_reported_without_inventing_a_model():
    result, model, predictions = evaluate_case(sample().head(10), "continuidad", CFG)
    assert result["status"] == "insuficiente" and model is None and predictions is None


def test_segments_compare_three_families_and_ignore_outcomes():
    d = sample()
    with threadpool_limits(limits=1):
        a, _, assignments = segment(d, CFG)
        b, _, changed = segment(d.assign(target=1-d.target), CFG)
    assert len({c["algorithm"] for c in a["candidates"]}) == 3
    pd.testing.assert_frame_equal(assignments, changed)
    assert a["fit_cohorts"] == [2019, 2020, 2021, 2022, 2023]
    assert a["holdout_cohorts"] == [2024, 2025]
