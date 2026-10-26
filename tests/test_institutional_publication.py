"""Regresiones de denominadores, formato y publicación de agregados UA."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from student_analytics.institutional_publication import CUTS, NUMERIC_MEASURES, load_publication
from student_analytics.ui.institucional import _presentacion

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("institutional_export", ROOT / "scripts/export_institutional_aggregates.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def records(numeric=12):
    return pd.DataFrame([
        {"student_key": f"ficticio-{i}", "grade_numeric": (3.0 if i < 3 else 5.0) if i < numeric else None,
         "numeric_below4": float(i < 3) if i < numeric else None,
         "year": 2025, "period": "Primer Semestre - 2025", "career_code": "INCO",
         "career": "Ingeniería Comercial", "campus_code": "TA", "campus": "Talca",
         "course_code": "MAT:101", "course_name": "Matemática"} for i in range(12)
    ])


def publication(folder, numeric=12):
    hashes = {}
    for name, dims in CUTS.items():
        file = folder / (name + ".parquet")
        export._agregar(records(numeric), dims, 10).to_parquet(file, index=False)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (folder / "meta.json").write_text(json.dumps({"minimo_personas_celda": 10, "archivos_sha256": hashes}))


def test_numeric_denominator_excludes_codes_and_percentages_are_scaled_once():
    table = export._agregar(records(10), ["career"], 10)
    assert table.iloc[0].bajo4 == pytest.approx(.3)
    assert table.iloc[0].cobertura_nota == pytest.approx(.8333)
    shown = _presentacion(table)
    assert shown.iloc[0]["Notas bajo 4 (%)"] == pytest.approx(30)
    assert shown.iloc[0]["Cobertura de nota (%)"] == pytest.approx(83.33)
    assert table.iloc[0].bajo4 == pytest.approx(.3)  # No modifica datos para los gráficos.


@pytest.mark.parametrize("numeric", [0, 1, 9])
def test_numeric_statistics_require_ten_distinct_contributors(numeric):
    d = records(numeric)
    # Más registros de una misma persona no habilitan publicar sus notas.
    d = pd.concat([d] * 10, ignore_index=True)
    table = export._agregar(d, ["career"], 10)
    assert table.iloc[0].personas == 12
    assert table.iloc[0].registros == 120
    assert table[NUMERIC_MEASURES].isna().all().all()
    assert table.attrs["estadisticas_suprimidas"] == 1


def test_small_groups_and_lowered_threshold_cannot_be_published():
    assert export._agregar(records().iloc[:9], ["career"], 10).empty
    with pytest.raises(ValueError, match="10"):
        export._agregar(records(), ["career"], 1)


@pytest.mark.parametrize("numeric", [0, 9, 10, 12])
def test_publication_round_trip_preserves_suppression(tmp_path, numeric):
    publication(tmp_path, numeric)
    tables, _ = load_publication(tmp_path)
    assert set(tables) == set(CUTS)
    assert bool(tables["por_carrera"].nota_media.isna().all()) == (numeric < 10)


@pytest.mark.parametrize("change", ["extra_file", "identifier", "small_cell", "small_numeric_group", "altered_hash"])
def test_publication_rejects_unapproved_or_inconsistent_content(tmp_path, change):
    publication(tmp_path)
    filename = "por_carrera.parquet"
    path = tmp_path / filename
    if change == "extra_file":
        (tmp_path / "individuales.csv").write_text("student_key,nota\nficticio,3")
    elif change == "altered_hash":
        path.write_bytes(path.read_bytes() + b"alterado")
    else:
        frame = pd.read_parquet(path)
        if change == "identifier":
            frame["student_key"] = "ficticio"
        elif change == "small_cell":
            frame["personas"] = 9
        else:
            frame["personas_con_nota"] = 1
        frame.to_parquet(path, index=False)
        meta = json.loads((tmp_path / "meta.json").read_text())
        meta["archivos_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_publication(tmp_path)


def test_distributed_institutional_tables_satisfy_contract():
    tables, meta = load_publication(ROOT / "deploy/data/institucional_agregado")
    assert meta["alcance"] == "todo el extracto"
    assert all("student_key" not in frame for frame in tables.values())


def test_packaging_rejects_extra_file_before_modifying_deployment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("package_deployment", ROOT / "scripts/package_deployment.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    source = tmp_path / "data/results"
    source.mkdir(parents=True)
    manifest = {}
    for filename, key in [("benchmark_profiles.parquet", "perfiles_sha256"),
                          ("retention_universities.parquet", "retencion_sha256")]:
        path = source / filename
        pd.DataFrame({"total": [100]}).to_parquet(path)
        manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "benchmark_manifest.json").write_text(json.dumps(manifest))
    institution = source / "institucional_agregado"
    institution.mkdir()
    publication(institution)
    (institution / "individuales.csv").write_text("student_key\nficticio")
    destination = tmp_path / "deploy/data"
    destination.mkdir(parents=True)
    marker = destination / "benchmark_manifest.json"
    marker.write_text("conservar")
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "FILES", ("benchmark_profiles.parquet", "retention_universities.parquet", "benchmark_manifest.json"))
    monkeypatch.setattr(package, "CARPETAS", {"institucional_agregado": institution})
    monkeypatch.setattr(package, "OPCIONALES", {})
    with pytest.raises(ValueError, match="solo los cinco"):
        package.main()
    assert marker.read_text() == "conservar"
    assert not (destination / "institucional_agregado").exists()
