import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piloto_fan.dashboard import export_csv, filtered, filtered_case, number, percent, profiles
from piloto_fan.data import continuity, prepare_entries

HOME = Path(__file__).parents[1]
CFG = json.loads((HOME / "config.json").read_text(encoding="utf-8"))


def test_spanish_formatting_distinguishes_missing_from_zero():
    assert number(1234.5, 1) == "1.234,5"
    assert percent(.436) == "43,6%"
    assert percent(0) == "0,0%"
    assert percent(np.nan) == "—"
    assert number(None, 3) == "—"


def test_filters_never_fall_back_to_whole_population():
    d = pd.DataFrame({"nomb_sede": ["A", "B"], "nomb_carrera": ["X", "Y"], "jornada": ["D", "V"]})
    assert len(filtered(d, {"nomb_sede": "A", "nomb_carrera": "Y"})) == 0
    assert len(filtered(d, {"nomb_sede": "A", "nomb_carrera": None})) == 1
    assert len(d) == 2


def test_suppressed_profiles_keep_full_denominator():
    d = pd.DataFrame({"segment": [1]*20+[2]*5, "target": [0]*25,
                      "percentil_admision": [.5]*25, "percentil_nem": [.4]*25,
                      "percentil_ranking": [.6]*25, "duracion_nominal_anios": [5]*25})
    p = profiles(d)
    assert len(p) == 1
    assert p["Participación (%)"].iloc[0] == 80
    assert p["Continuidad UA (%)"].iloc[0] == 100


def test_filtered_capacity_uses_selected_population_and_single_class_is_safe():
    d = pd.DataFrame({"target": [0]*20, "risk_score": np.linspace(0, 1, 20), "historical_rule": [.3]*20})
    view = filtered_case(d, {"reference_prevalence": .3}, CFG)
    assert view["metrics"]["auc"] is None
    assert view["capacity"]["Cupos"].tolist() == [1, 2, 4, 6, 10]
    assert view["capacity"]["Precisión (%)"].eq(0).all()
    assert view["capacity"]["Captura modelo (%)"].isna().all()
    assert view["uncertainty"] is None
    assert filtered_case(d.head(5), {"reference_prevalence": .3}, CFG) is None


def test_csv_is_aggregate_spanish_and_has_filter_context():
    table = pd.DataFrame({"Captura (%)": [43.6], "Etiqueta": ["=1+2"]})
    raw = export_csv(table, {"nomb_sede": "Talca"}, {"Cohorte": 2025})
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "43,6" in text and "'=1+2" in text
    loaded = pd.read_csv(io.StringIO(text), sep=";", decimal=",")
    assert loaded["Filtro: Sede / campus al ingreso"].iloc[0] == "Talca"
    with pytest.raises(ValueError, match="agregada"):
        export_csv(pd.DataFrame({"mrun": [1]}), {}, {})


def test_catalog_is_an_explicit_allowlist_not_the_whole_administration_area():
    base = dict(cod_inst="31", cohorte=2021, area_conocimiento="Administración y Comercio",
                tipo_plan_carr="Plan Regular", nivel_carrera_1="Profesional Con Licenciatura",
                nomb_sede="Talca", duracion_nominal_anios=5, percentil_admision=.5,
                percentil_nem=.5, percentil_ranking=.5)
    rows = pd.DataFrame([{**base, "mrun": "1", "nomb_carrera": "INGENIERIA COMERCIAL"},
                         {**base, "mrun": "2", "nomb_carrera": "OTRO PROGRAMA NO DOCUMENTADO"}])
    d, audit = prepare_entries(rows, CFG)
    assert d.mrun.tolist() == ["1"]
    assert audit["careers_excluded_pending_confirmation"] == {"OTRO PROGRAMA NO DOCUMENTADO": 1}


def test_empty_followup_population_is_not_confused_with_missing_source_year():
    entries = pd.DataFrame({"mrun": ["1"], "cohorte": [2021]})
    history = pd.DataFrame({"mrun": ["1"], "year": [2021]})
    d = continuity(entries, history, CFG, history_years=[2021, 2022])
    assert d.target.iloc[0] == 1
