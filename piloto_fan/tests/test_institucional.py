import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from piloto_fan.institucional import (
    HOME, MAPPING_H, MAPPING_M, export_aggregate, grouped_results, load,
    prepare, read_config, select_data, summary,
)


def enrollment(person, program="A", **changes):
    row = {key: "x" for key in MAPPING_M}
    row.update(PIDM=person, TERM_CODE=202410, PERIODO="Primer Semestre - 2024",
               LEVL_CODE="PR", NIVEL="Pregrado", CAMP_CODE="TA", CAMPUS="Talca",
               MAJR_CODE="INCO", CARRERA="Ingeniería Comercial", PROGRAM=program,
               PROGRAMA=program, FACULTAD="Fac. Administración y Negocios", JORNADA="Diurno",
               STATUS_PLAN_ESTUDIO="Activo", STATUS_ALUMNO="Activo")
    row.update(changes)
    return {"UWVMAHI_"+k: v for k, v in row.items()}


def history(person, nrc, grade, **changes):
    row = {key: "x" for key in MAPPING_H}
    row.update(PIDM=person, TERM_CODE=202410, NRC=nrc, CAMP_CODE="TA",
               LEVL_CODE="PR", SUBJ_CODE="MAT", CRSE_NUMB="101",
               CRSE_TITLE="Matemática", GRDE_CODE_FINAL=grade, CREDIT_HOURS=72)
    row.update(changes)
    return {"UWVHIAC_"+k: v for k, v in row.items()}


@pytest.fixture
def prepared():
    m = pd.DataFrame([enrollment(101), enrollment(102),
                      enrollment(102, "B", FACULTAD="Otra facultad"),
                      enrollment(103), enrollment(104)])
    h = pd.DataFrame([history(101, 1, "5,0"), history(101, 2, "3,0"),
                      history(101, 3, "RI"), history(102, 1, "2,0"),
                      history(104, 1, "RR"), history(999, 1, "6,0")])
    return prepare(h, m, read_config(), b"test-only-local-secret")


def test_join_preserves_facts_without_arbitrary_plan_attribution(prepared):
    frames, audit = prepared
    assert len(frames["courses"]) == 6
    assert audit["multiple_plan_person_periods"] == 1
    assert audit["history_attribution"] == {
        "Única": 4, "Varios planes": 1, "Sin matrícula coincidente": 1}
    m, h, ambiguous = select_data(frames)
    assert len(m) == 4 and len(h) == 4 and len(ambiguous) == 1
    assert ambiguous.career.isna().all()
    assert not any("PIDM" in c.upper() for df in frames.values() for c in df)
    assert frames["courses"].student_key.str.len().eq(64).all()
    assert not frames["courses"].student_key.isin(["101", "102"]).any()


def test_grade_codes_missing_history_and_denominators_are_distinct(prepared):
    frames, _ = prepared
    m, h, _ = select_data(frames)
    result = summary(m, h)
    assert result["numeric_results"] == 2
    assert result["numeric_mean"] == 4
    assert result["numeric_coverage"] == .5
    assert result["numeric_below4_rate"] == .5
    assert h.loc[h.result_code.notna(), "grade_numeric"].isna().all()
    unknown = frames["students"].loc[~frames["students"].has_observed_history]
    assert len(unknown) == 1 and unknown.numeric_mean.isna().all()
    assert unknown.observed_results.isna().all()
    only_codes = frames["students"].loc[frames["students"].numeric_results.eq(0)]
    assert only_codes.numeric_below4_count.isna().all()


def test_filter_empty_does_not_fall_back_and_small_groups_are_suppressed(prepared):
    frames, _ = prepared
    m, h, unresolved = select_data(frames, filters={"campus": "Campus inexistente"})
    assert m.empty and h.empty and unresolved.empty
    _, h, _ = select_data(frames)
    assert grouped_results(h, ["course_code"], minimum=10).empty
    report = grouped_results(h, ["course_code"], minimum=1)
    assert report.iloc[0]["Notas numéricas bajo 4 (%)"] == 50


def test_invalid_grain_and_out_of_range_grade():
    h = pd.DataFrame([history(1, 1, "8,0")])
    m = pd.DataFrame([enrollment(1)])
    frames, audit = prepare(h, m, read_config(), b"test-secret")
    assert frames["courses"].grade_numeric.isna().all()
    assert audit["grade_kinds"] == {"Numérica fuera de rango": 1}
    with pytest.raises(ValueError, match="Grano"):
        prepare(pd.concat([h, h]), m, read_config(), b"test-secret")
    with pytest.raises(ValueError, match="Clave ausente"):
        prepare(pd.DataFrame([history(None, 1, "4,0")]), m, read_config(), b"test-secret")


def test_export_allowlist_and_spreadsheet_formula_protection(prepared):
    _, h, _ = select_data(prepared[0])
    table = grouped_results(h, ["course_name"], minimum=1)
    table["Asignatura"] = "=1+2"
    data = export_aggregate(table, {"Año observado": 2024})
    assert data.startswith(b"\xef\xbb\xbf")
    assert "'=1+2" in data.decode("utf-8-sig")
    for column in ["student_key", "UWVMAHI_PIDM", "risk_score"]:
        with pytest.raises(ValueError, match="agregadas"):
            export_aggregate(pd.DataFrame({column: ["sensitive"]}), {})


def latest_institutional():
    pointer = HOME / "artifacts/institucional/latest.json"
    if not pointer.exists():
        pytest.skip("Requiere generar las tablas institucionales locales.")
    return pointer.parent / json.loads(pointer.read_text())["run_id"]


def test_real_artifacts_reconcile_and_manifest_detects_tampering(tmp_path):
    import shutil
    folder = latest_institutional()
    frames, quality, _ = load(folder)
    assert len(frames["courses"]) == quality["history_rows"]
    assert len(frames["enrollment"]) == quality["enrollment_rows"]
    assert not frames["students"].duplicated(["student_key", "term"]).any()
    assert quality["fan"]["fan_proposed"]["people"] == 4308
    assert quality["fan"]["fan_proposed"]["unambiguous_history_rows"] == 63264
    copied = tmp_path / "altered"
    shutil.copytree(folder, copied)
    with (copied/"quality.json").open("a") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="alterado"):
        load(copied)


def test_institutional_navigation_and_independent_filters():
    latest_institutional()
    app = AppTest.from_file(str(HOME/"app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Datos institucionales FAN").run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.selectbox(key="inst_year").value == 2025
    assert app.selectbox(key="inst_regime").value == "Semestral"
    assert not any(s.key == "filter_nomb_sede" for s in app.selectbox)
    initial = int(app.metric[0].value.replace(".", ""))
    app.selectbox(key="inst_campus").set_value("Talca").run()
    assert not app.exception, [e.message for e in app.exception]
    assert int(app.metric[0].value.replace(".", "")) < initial
    assert len(app.expander) == 10
    app.selectbox(key="inst_scope").set_value("fan_label").run()
    next(b for b in app.button if b.label == "Restablecer filtros institucionales").click().run()
    assert not app.exception
    assert app.selectbox(key="inst_scope").value == "fan_proposed"
    assert int(app.metric[0].value.replace(".", "")) == initial
    app.sidebar.radio[0].set_value("Resumen").run()
    assert not app.exception


def test_institutional_available_without_public_model_artifacts(tmp_path):
    import shutil
    folder = latest_institutional()
    base = tmp_path/"artifacts/institucional"
    base.mkdir(parents=True)
    shutil.copytree(folder, base/folder.name)
    (base/"latest.json").write_text(json.dumps({"run_id": folder.name}))
    # Direct view does not depend on the public-model latest pointer.
    code = ("from pathlib import Path\nfrom piloto_fan.institucional_ui import render\n"
            f"render(Path({str(tmp_path)!r}))\n")
    app = AppTest.from_string(code, default_timeout=45).run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.metric
