"""Entrada de despliegue: usa exclusivamente los artefactos distribuidos."""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("STUDENT_ANALYTICS_RESULTS_DIR", str(ROOT / "deploy" / "data"))
runpy.run_path(str(ROOT / "src" / "student_analytics" / "ui" / "app.py"), run_name="__main__")
