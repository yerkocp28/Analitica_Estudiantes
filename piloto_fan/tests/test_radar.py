import json

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from piloto_fan import presentacion, radar
from piloto_fan.institucional import HOME
from piloto_fan.modelos_institucionales import CATEGORICAL, NUMERIC, SPEC, load_run
from piloto_fan.models import design, estimator


def synthetic(n=400, seed=0):
    """Riesgo que depende solo del promedio: es la única señal que debe aparecer."""
    rng = np.random.default_rng(seed)
    rows = []
    for term in radar.TRAIN_TERMS + [radar.ORIGIN]:
        mean = rng.uniform(3, 6.5, n)
        rows.append(pd.DataFrame({
            "student_key": [f"{term}-{i:04d}" for i in range(n)], "source_term": term,
            "career": rng.choice(["Comercial", "Auditoría"], n), "campus": rng.choice(["Talca", "Temuco"], n),
            "observed_results": 6.0, "numeric_mean": mean, "numeric_sd": .7, "numeric_coverage": 1.0,
            "below4_fraction": 0.0, "career_code": "C", "campus_code": "T", "schedule": "Diurno",
            "target": (rng.uniform(size=n) < 1/(1+np.exp(4*(mean-4.5)))).astype(float)}))
    data = pd.concat(rows, ignore_index=True)
    train = data.loc[data.source_term.isin(radar.TRAIN_TERMS)]
    model = estimator("Regresión logística", SPEC, 42).fit(design(train, SPEC), train.target.astype(int))
    return data, {"model": model, "spec": SPEC}


def test_signals_name_what_raises_risk_and_stay_silent_at_typical_values():
    data, bundle = synthetic()
    d = radar.radar(data, bundle)
    low = d.loc[d.numeric_mean.idxmin()]
    assert low.signals.startswith("Promedio 3,0") and "típico" in low.signals
    assert all("Promedio" in s or s == radar.NO_SIGNAL for s in d.signals)  # Nada que no mueva el riesgo.
    near_typical = d.loc[(d.numeric_mean - radar.typical(data).numeric_mean).abs().idxmin()]
    assert near_typical.signals == radar.NO_SIGNAL
    assert set(d.source_term) == {radar.ORIGIN}
    assert d.outcome.isin(["Nota bajo 4", "Sin notas bajo 4"]).all()


def test_each_direction_is_cut_at_its_own_capacity():
    data, bundle = synthetic()
    d = radar.prioritize(radar.radar(data, bundle), .2)
    for _, g in d.groupby(radar.GROUP):
        assert g.listed.sum() == int(np.ceil(.2 * len(g)))
        assert g.loc[g.listed, "risk"].min() >= g.loc[~g.listed, "risk"].max()
        assert sorted(g.priority) == list(range(1, len(g) + 1))


def test_ties_are_shared_and_unknown_outcomes_stay_out_of_the_count():
    y = np.array([1., 0., 1., 0.])
    assert radar.ranked_hits(y, np.array([.9, .5, .5, .5]), 2) == pytest.approx(1 + 1/3)
    d = pd.DataFrame({"career": "C", "campus": "T", "risk": np.linspace(1, 0, 14),
                      "below4_fraction": 0., "target": [1.] * 3 + [np.nan] * 2 + [0.] * 9})
    row = radar.retrospective(d, .2).iloc[0]
    assert row["Estudiantes con resultado"] == 12 and row["Cupos"] == 3
    assert row["Aciertos del radar"] == 3 and row["Aciertos de la regla simple"] == pytest.approx(.75)


def latest_models():
    pointer = HOME / "artifacts/institucional_modelos/latest.json"
    if not pointer.exists():
        pytest.skip("Requiere modelos institucionales locales.")
    return pointer.parent / json.loads(pointer.read_text())["run_id"]


def test_radar_reuses_the_saved_model_without_refitting():
    folder = latest_models()
    _, frames = load_run(folder)
    d = radar.radar(frames["academico_dataset"], joblib.load(folder / "academico_model.joblib"),
                    frames["matricula_dataset"])
    saved = frames["academico_predictions"].set_index("student_key").risk_score
    assert np.allclose(d.set_index("student_key").risk.loc[saved.index], saved)
    assert len(d) > len(saved)  # También quienes no tienen resultado conocido.
    assert set(NUMERIC + CATEGORICAL) <= set(d.columns)


def test_radar_view_shows_no_probabilities_and_no_individual_download():
    latest_models()
    app = AppTest.from_file(str(HOME / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value(presentacion.RADAR).run()
    assert not app.exception, [e.message for e in app.exception]
    app.selectbox(key="im_campus").set_value("Talca").run()
    app.selectbox(key="im_career").set_value("Ingeniería Comercial").run()
    assert not app.exception, [e.message for e in app.exception]
    frames = [f.value for f in app.dataframe]
    listing = next(f for f in frames if "Señales" in f.columns)
    assert not any("riesgo" in c.lower() or "probab" in c.lower() for c in listing.columns)
    assert listing["Estudiante (seudónimo)"].str.len().eq(8).all()
    assert [b.label for b in app.get("download_button")] == ["Descargar resultado por dirección"]
