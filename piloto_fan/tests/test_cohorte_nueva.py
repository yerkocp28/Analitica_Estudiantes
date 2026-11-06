import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from piloto_fan import cohorte_nueva
from piloto_fan.dashboard import site_mix
from piloto_fan.segments import assign

HOME = cohorte_nueva.HOME


def latest_run():
    pointer = HOME / "artifacts/latest.json"
    if not pointer.exists():
        pytest.skip("Requiere una corrida local del piloto.")
    return json.loads(pointer.read_text())["run_id"]


def test_assign_reproduces_the_run_on_its_holdout_cohorts():
    run = latest_run()
    folder = cohorte_nueva.verified_run(run)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    holdout = report["segments"]["holdout_cohorts"]
    data = pd.read_parquet(folder / "continuidad_dataset.parquet")
    data = data[data.cohorte.isin(holdout)].reset_index(drop=True)
    saved = pd.read_parquet(folder / "segmentacion_assignments.parquet")
    expected = data[["mrun", "cohorte"]].merge(saved, on=["mrun", "cohorte"], validate="one_to_one")
    assert assign(data, joblib.load(folder / "segmentacion_model.joblib")).tolist() == expected.segment.tolist()


def test_later_cohorts_have_no_outcome_and_never_repeat_known_ones():
    run = latest_run()
    later = cohorte_nueva.load(run)
    if later is None:
        pytest.skip("Requiere python -m piloto_fan.cohorte_nueva.")
    known = pd.read_parquet(HOME / "artifacts" / run / "continuidad_dataset.parquet")
    assert later.cohorte.min() > known.cohorte.max()
    assert later.target.isna().all()
    assert not later.mrun.isin(known.mrun).any()  # Solo primer ingreso a la UA.
    assert set(later.segment) <= set(range(1, 1 + json.loads(
        (HOME / "artifacts" / run / "report.json").read_text(encoding="utf-8"))["segments"]["k"]))


def test_later_cohorts_are_rejected_if_altered_or_from_another_run(tmp_path, monkeypatch):
    run = latest_run()
    if cohorte_nueva.load(run) is None:
        pytest.skip("Requiere python -m piloto_fan.cohorte_nueva.")
    monkeypatch.setattr(cohorte_nueva, "OUT", tmp_path)
    shutil.copytree(HOME / "artifacts/cohortes_nuevas" / run, tmp_path / run)
    assert cohorte_nueva.load(run) is not None
    shutil.copytree(tmp_path / run, tmp_path / "otra-corrida")
    with pytest.raises(ValueError, match="no corresponden"):
        cohorte_nueva.load("otra-corrida")
    frame = pd.read_parquet(tmp_path / run / cohorte_nueva.DATASET)
    frame.loc[0, "segment"] = 99
    frame.to_parquet(tmp_path / run / cohorte_nueva.DATASET, index=False)
    with pytest.raises(ValueError, match="alteradas"):
        cohorte_nueva.load(run)


def test_site_mix_compares_shares_and_suppresses_small_cells():
    rows = ([("A", 2026, 1, np.nan)] * 30 + [("A", 2026, 2, np.nan)] * 10 + [("A", 2026, 3, np.nan)] * 4
            + [("A", 2025, 1, 0)] * 40 + [("A", 2025, 2, 1)] * 40
            + [("B", 2026, 1, np.nan)] * 9 + [("B", 2025, 1, 0)] * 50)
    d = pd.DataFrame(rows, columns=["nomb_sede", "cohorte", "segment", "target"])
    mix = site_mix(d, 2026, [2025]).set_index("Perfil")
    assert set(mix.Sede) == {"A"}  # B no llega a diez en 2026: no se muestra.
    assert "Perfil 3" not in mix.index  # Cuatro estudiantes: suprimido.
    assert mix.loc["Perfil 1", "Participación (%)"] == pytest.approx(100 * 30 / 44)  # Denominador completo.
    assert mix.loc["Perfil 2", "Cambio (pp)"] == pytest.approx(100 * 10 / 44 - 50)
    assert mix.loc["Perfil 2", "Continuidad histórica del perfil (%)"] == 0
    assert mix.loc["Perfil 1", "Continuidad histórica del perfil (%)"] == 100
