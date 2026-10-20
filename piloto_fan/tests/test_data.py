import json
from pathlib import Path

import numpy as np
import pandas as pd

from piloto_fan.data import continuity, graduation, prepare_entries

CFG = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
CFG = {k: v for k, v in CFG.items() if k != "faculty_catalog"}


def entry(person="1", year=2020, **changes):
    row = dict(mrun=person, cod_inst="31", cohorte=year,
               area_conocimiento="Administración y Comercio", tipo_plan_carr="Plan Regular",
               nivel_carrera_1="Profesional Con Licenciatura", nomb_carrera="COMERCIAL",
               nomb_sede="SEDE A", jornada="Diurna", duracion_nominal_anios=5,
               percentil_admision=.5, percentil_nem=.6, percentil_ranking=.7,
               sigue_universidad=0)
    return {**row, **changes}


def history(years, person="1"):
    return pd.DataFrame([dict(mrun=person, year=y, cod_inst="31", cod_carrera="A",
                             nomb_sede="SEDE A", jornada="Diurna") for y in years])


def test_missing_identifier_is_not_a_negative_outcome_and_input_is_unchanged():
    original = pd.DataFrame([entry(None), entry()])
    before = original.copy(deep=True)
    out, audit = prepare_entries(original, CFG)
    assert len(out) == 1
    assert audit["missing_identifier"] == 1
    pd.testing.assert_frame_equal(original, before)


def test_first_entry_is_defined_before_filtering_faculty_and_ambiguities_are_excluded():
    raw = pd.DataFrame([entry("1", 2019, area_conocimiento="Salud"), entry("1", 2021),
                        entry("2", 2021), entry("2", 2021, nomb_carrera="OTRA"), entry("3", 2021)])
    out, audit = prepare_entries(raw, CFG)
    assert out.mrun.tolist() == ["3"]
    assert audit["later_entries_excluded"] == 1
    assert audit["ambiguous_first_entries_excluded"] == 2


def test_continuity_is_recomputed_instead_of_trusting_old_labels():
    out = continuity(pd.DataFrame([entry(year=2021)]), history([2021, 2022]), CFG)
    assert out.target.iloc[0] == 0
    assert out.label_ready_year.iloc[0] == 2023
    unknown = continuity(pd.DataFrame([entry(year=2021)]), history([2021]), CFG)
    assert unknown.target.isna().all()


def test_recent_graduation_cohorts_are_censored_not_labelled_as_failures():
    d, audit = graduation(pd.DataFrame([entry()]), history([2020, 2021, 2022, 2023]),
                          pd.DataFrame(columns=["mrun", "year"]), CFG,
                          list(range(2020, 2027)), list(range(2020, 2026)))
    assert np.isnan(d.target.iloc[0])
    assert d.outcome_year.iloc[0] == 2027
    assert audit["right_censored"] == 1


def test_future_enrollments_do_not_change_features_and_prior_degrees_do_not_hide_later_degrees():
    base = history([2020, 2022, 2023])
    degrees = pd.DataFrame({"mrun": ["1", "1"], "year": [2015, 2026]})
    args = (CFG, list(range(2015, 2030)), list(range(2015, 2030)))
    a, _ = graduation(pd.DataFrame([entry()]), base, degrees, *args)
    future = history([2028]).assign(cod_carrera="OTRA", nomb_sede="OTRA", jornada="Otra")
    b, _ = graduation(pd.DataFrame([entry()]), pd.concat([base, future]), degrees, *args)
    pd.testing.assert_frame_equal(a, b)
    assert a.target.iloc[0] == 0
    assert a.years_without_ua.iloc[0] == 1
    assert a.career_codes_observed.iloc[0] == 1


def test_already_graduated_or_not_active_at_landmark_are_outside_population():
    degrees = pd.DataFrame({"mrun": ["1"], "year": [2022]})
    d, audit = graduation(pd.DataFrame([entry()]), history([2020, 2023]), degrees, CFG,
                          list(range(2020, 2028)), list(range(2020, 2028)))
    assert d.empty and audit["already_graduated"] == 1
    d, audit = graduation(pd.DataFrame([entry()]), history([2020, 2022]), degrees, CFG,
                          list(range(2020, 2028)), list(range(2020, 2028)))
    assert d.empty and audit["not_enrolled_at_landmark"] == 1


def test_gap_in_title_files_leaves_outcome_unknown():
    d, _ = graduation(pd.DataFrame([entry()]), history([2020, 2023]),
                       pd.DataFrame(columns=["mrun", "year"]), CFG,
                       list(range(2020, 2028)), [2020, 2021, 2022, 2023, 2025, 2026, 2027])
    assert d.target.isna().all()
