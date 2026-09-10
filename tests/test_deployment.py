"""La distribución funciona sin datos locales ni fuentes originales."""
from pathlib import Path
import hashlib
import json
import shutil

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_manifiesto_distribuido():
    folder = ROOT / "deploy/data"
    manifest = json.loads((folder / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for filename, key in [("benchmark_profiles.parquet", "perfiles_sha256"),
                           ("retention_universities.parquet", "retencion_sha256")]:
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == manifest[key]


def test_app_distribuida_sin_data_local(tmp_path, monkeypatch):
    for folder in ["src", "config", "deploy", "documentacion/assets"]:
        shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "streamlit_app.py", tmp_path / "streamlit_app.py")
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("STUDENT_ANALYTICS_RESULTS_DIR", str(tmp_path / "deploy/data"))
    app = AppTest.from_file(str(tmp_path / "streamlit_app.py"), default_timeout=180).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    # La sección se elige explícitamente: atarse a la vista por defecto hacía
    # fallar este test al agregar «Hallazgos» delante, sin que nada del
    # despliegue hubiera cambiado.
    app.sidebar.radio[0].set_value("Benchmark UA").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert "Benchmark" in app.title[0].value
    app.sidebar.radio[0].set_value("Hallazgos").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.title[0].value == "Hallazgos"
    app.sidebar.radio[0].set_value("Retención universitaria").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.dataframe) >= 1
    app.sidebar.radio[0].set_value("Alerta temprana").run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.metric) >= 4
    app.sidebar.radio[1].set_value("oulad").run()
    assert not app.exception, [str(e.value) for e in app.exception]
