"""Ejecutar desde la raíz: python -m piloto_fan.build. No escribe fuera del piloto."""
from __future__ import annotations

import os

# Limita paralelismo antes de importar BLAS; evita bloquear la estación de trabajo.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from . import __version__
from .data import continuity, digest, graduation, read_sources
from .models import FEATURES, evaluate_case
from .segments import segment

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HOME / "config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = HOME / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=False)
    print(f"Piloto separado · ejecución {run_id}", flush=True)
    write_json(out / "config.json", cfg)
    entries, history, titles, quality, sources = read_sources(ROOT, cfg)
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    first = continuity(entries, history, cfg, quality["history_years"])
    population = continuity(entries, history, {**cfg, "admission_from": int(entries.cohorte.min())}, quality["history_years"])
    population.to_parquet(out / "poblacion_dataset.parquet", index=False)
    second, quality["graduation"] = graduation(entries, history, titles, cfg,
                                              quality["history_years"], quality["title_years"])
    coverage = []
    for year, g in first.groupby("cohorte"):
        row = {"cohort": int(year), "n": len(g)}
        for col in ("percentil_admision", "percentil_nem", "percentil_ranking"):
            row[col] = float(g[col].notna().mean())
        coverage.append(row)
    quality["admission_coverage"] = coverage
    # Archivos de personas SOLO locales e ignorados por Git.
    first.to_parquet(out / "continuidad_dataset.parquet", index=False)
    second.to_parquet(out / "titulacion_dataset.parquet", index=False)
    cases = {}
    with threadpool_limits(limits=1):
        for name, dataset in (("continuidad", first), ("titulacion", second)):
            print(f"Evaluando {name}: {len(dataset)} casos elegibles", flush=True)
            result, model, predictions = evaluate_case(dataset, name, cfg)
            cases[name] = result
            write_json(out / f"{name}_model_card.json", result)
            if model is not None:
                joblib.dump(model, out / f"{name}_model.joblib")
                predictions["run_id"] = run_id
                predictions["model_version"] = __version__
                predictions["feature_set_version"] = __version__
                predictions["score_date"] = now.isoformat()
                predictions["prediction_id"] = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{name}/{person}"))
                                                 for person in predictions.mrun]
                predictions.to_parquet(out / f"{name}_predictions.parquet", index=False)
            print(f"  {result['status']} · {result.get('winner', result.get('reason'))}", flush=True)
        clusters, model, assignments = segment(first, cfg)
        if model is not None:
            joblib.dump(model, out / "segmentacion_model.joblib")
            assignments.to_parquet(out / "segmentacion_assignments.parquet", index=False)
    assumptions = [
        "Alcance FAN: carreras actuales e históricas del catálogo documentado, planes regulares y nivel profesional; pendiente de ratificación institucional.",
        "Una persona por primer ingreso observado a UA en 2007–2025. Reingresos posteriores y primeras entradas ambiguas se excluyen.",
        "Se conservan los nombres originales de sede; no se equipara SEDE SANTIAGO a San Miguel sin validación institucional.",
        "Reconstrucción retrospectiva de registros académicos; no reproduce las fechas reales de publicación de las bases públicas.",
        f"Disponibilidad de etiquetas asumida: año de resultado + {cfg['label_lag_years']}; revisable cuando existan fechas institucionales.",
        "No continuidad no equivale a abandono definitivo. No titulación en plazo no identifica asignaturas reprobadas ni sus causas.",
        "La titulación se observa en la UA; no obtener un título UA no excluye titularse en otra institución.",
        "OULAD y la simulación existente no se usan para entrenar ni medir estos dos modelos UA.",
        "Las predicciones son pruebas históricas. No hay identificación para contactos ni activación de alertas sobre estudiantes actuales.",
        "No se ha medido efecto de intervenciones ni validado todavía el uso con direcciones de carrera.",
    ]
    report = {"run_id": run_id, "created_utc": now.isoformat(), "version": __version__,
              "config": cfg, "quality": quality, "cases": cases, "segments": clusters,
              "assumptions": assumptions}
    write_json(out / "report.json", report)
    from .reporting import executive_html
    (out / "informe_ejecutivo.html").write_text(executive_html(report), encoding="utf-8")
    write_json(out / "feature_registry.json", {"version": __version__, "cases": FEATURES,
               "excluded": ["MRUN", "resultados posteriores", "notas/asistencia universitarias inexistentes",
                            "socioeconomía sin cobertura temporal consistente"],
               "fit_preprocessing": "Imputadores y codificadores ajustados solo sobre entrenamiento."})
    # Verificar que ningún insumo cambió mientras se ejecutaba el piloto.
    for path in sources:
        key = str(path.relative_to(ROOT))
        if digest(path) != input_hashes[key]:
            raise RuntimeError(f"El insumo cambió durante la ejecución: {key}. No se activa esta corrida.")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
    code = sorted(HOME.glob("*.py")) + [args.config]
    manifest = {"run_id": run_id, "created_utc": now.isoformat(), "base_git_revision": revision,
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__, "inputs_sha256": input_hashes,
                "code_sha256": {p.name: digest(p) for p in code},
                "artifacts_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out / "manifest.json", manifest)
    # Solo se cambia el puntero cuando la ejecución completa termina correctamente.
    pointer = HOME / "artifacts" / "latest.tmp"
    write_json(pointer, {"run_id": run_id})
    pointer.replace(HOME / "artifacts" / "latest.json")
    print(f"Listo: {out.relative_to(ROOT)}", flush=True)
    print("Abrir: streamlit run piloto_fan/app.py --server.port 8502", flush=True)


if __name__ == "__main__":
    main()
