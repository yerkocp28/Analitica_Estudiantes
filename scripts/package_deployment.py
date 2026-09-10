"""Empaqueta una lista explícita de resultados públicos para Streamlit Cloud."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "benchmark_profiles.parquet", "retention_universities.parquet", "benchmark_manifest.json",
    "metrics_synthetic.parquet", "predictions_synthetic.parquet",
    "metrics_oulad.parquet", "predictions_oulad.parquet",
)
# Insumos del argumento central del proyecto. Van aparte porque son opcionales
# —el resto de la app funciona sin ellos— y porque la ablacion la escribe
# analyze_mineduc.py en documentacion/datos, no en data/results.
OPCIONALES = {
    "piso_preingreso_nacional.parquet": ROOT / "data/results/piso_preingreso_nacional.parquet",
    "piso_preingreso_temporal.parquet": ROOT / "data/results/piso_preingreso_temporal.parquet",
    "ablacion_piso.csv": ROOT / "documentacion/datos/ablacion_piso.csv",
}


def main() -> None:
    source = ROOT / "data/results"
    destination = ROOT / "deploy/data"
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"Faltan artefactos: {missing}. Genera primero los resultados.")
    manifest = json.loads((source / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for name, field in [("benchmark_profiles.parquet", "perfiles_sha256"),
                         ("retention_universities.parquet", "retencion_sha256")]:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest[field]:
            raise SystemExit("El manifiesto no corresponde a los agregados. Ejecuta build_benchmark.py.")
    for name in FILES:
        if name.endswith(".parquet"):
            frame = pd.read_parquet(source / name)
            forbidden = {"mrun", "rut", "email", "correo", "nombre_estudiante", "fecha_nacimiento"}
            if forbidden & {c.lower() for c in frame.columns}:
                raise SystemExit(f"Identificadores personales no admitidos en {name}")
            if "student_id" in frame and not name.startswith("predictions_"):
                raise SystemExit(f"Se esperaba un agregado institucional en {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, destination / name)
    llevados = list(FILES)
    for name, origen in OPCIONALES.items():
        if origen.is_file():
            shutil.copyfile(origen, destination / name)
            llevados.append(name)
        else:
            print(f"  sin {name}; la vista de hallazgos lo dirá en vez de inventarlo")
    sizes = {name: (destination / name).stat().st_size for name in llevados}
    print(f"Empaquetados {len(llevados)} archivos, {sum(sizes.values()) / 1e6:.2f} MB, en {destination}")


if __name__ == "__main__":
    main()
