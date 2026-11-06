"""Ubica las cohortes de ingreso posteriores a la corrida en sus perfiles congelados.

    python -m piloto_fan.cohorte_nueva

POR QUÉ EXISTE

Un perfil de ingreso sirve el día que llega una cohorte nueva: es cuando una
dirección todavía puede ajustar nivelación y acompañamiento de primer año. La
corrida del piloto aprendió los perfiles con 2021–2023 y los asignó sin reajuste
a 2024–2025. Este paso hace lo mismo con las cohortes que llegaron después de la
corrida, con el mismo modelo: no reentrena, no cambia la numeración de los
perfiles y no modifica la corrida.

La continuidad de una cohorte nueva todavía no existe —se observa con la
matrícula del año siguiente— y queda vacía, nunca en cero.

Escribe solo en artifacts/, que Git ignora: son datos por persona y de uso local.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import digest, prepare_entries
from .segments import assign

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
SOURCE = ROOT / "data/results/ua_ingresantes.parquet"
OUT = HOME / "artifacts" / "cohortes_nuevas"
DATASET = "cohortes_nuevas.parquet"
COLUMNS = ["mrun", "cohorte", "nomb_sede", "nomb_carrera", "jornada", "percentil_admision",
           "percentil_nem", "percentil_ranking", "duracion_nominal_anios", "target", "segment"]


def verified_run(run_id: str) -> Path:
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Identificador de ejecución inválido.")
    folder = HOME / "artifacts" / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name in ("report.json", "segmentacion_model.joblib", "continuidad_dataset.parquet"):
        if digest(folder / name) != manifest["artifacts_sha256"].get(name):
            raise ValueError(f"Artefacto ausente o alterado en la corrida {run_id}: {name}.")
    return folder


def later_cohorts(raw: pd.DataFrame, cfg: dict, bundle: dict, last_known: int) -> pd.DataFrame:
    # El filtro de primer ingreso se aplica sobre todas las cohortes, igual que en la
    # corrida: quien ya había entrado antes a la UA no es ingresante de la cohorte nueva.
    entries, _ = prepare_entries(raw, cfg)
    d = entries.loc[entries.cohorte.gt(last_known) & entries.cohorte.ge(cfg["admission_from"])].copy()
    d["target"] = np.nan
    d["segment"] = assign(d, bundle) if len(d) else pd.Series(dtype=int)
    return d[COLUMNS].reset_index(drop=True)


def build(run_id: str | None = None, source: Path = SOURCE) -> Path | None:
    run_id = run_id or json.loads((HOME / "artifacts/latest.json").read_text(encoding="utf-8"))["run_id"]
    folder = verified_run(run_id)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if "winner" not in report["segments"]:
        raise ValueError("La corrida no tiene una segmentación admisible que asignar.")
    known = pd.read_parquet(folder / "continuidad_dataset.parquet", columns=["cohorte"]).cohorte
    new = later_cohorts(pd.read_parquet(source), report["config"],
                        joblib.load(folder / "segmentacion_model.joblib"), int(known.max()))
    if new.empty:
        print(f"No hay cohortes posteriores a {int(known.max())} en {source.name}.")
        return None
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out / DATASET, index=False)
    (out / "manifest.json").write_text(json.dumps({
        "model_run": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source.name, "source_sha256": digest(source),
        "cohorts": {str(int(c)): int(n) for c, n in new.cohorte.value_counts().sort_index().items()},
        "artifacts_sha256": {DATASET: digest(out / DATASET)},
        "note": "Perfiles asignados con el modelo congelado de la corrida; sin resultado observado todavía.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for cohort, g in new.groupby("cohorte"):
        shares = g.segment.value_counts(normalize=True).sort_index()
        print(f"Cohorte {int(cohort)}: {len(g)} ingresantes FAN · "
              + " · ".join(f"perfil {int(k)} {100*v:.1f}%" for k, v in shares.items()))
    print(f"-> {out.relative_to(ROOT)}")
    return out


def load(run_id: str) -> pd.DataFrame | None:
    """Cohortes nuevas de esta corrida, si existen y no fueron alteradas."""
    out = OUT / run_id
    if not (out / "manifest.json").exists():
        return None
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_run") != run_id or digest(out / DATASET) != manifest["artifacts_sha256"][DATASET]:
        raise ValueError("Las cohortes nuevas no corresponden a esta corrida o fueron alteradas.")
    return pd.read_parquet(out / DATASET)


if __name__ == "__main__":
    raise SystemExit(0 if build() else 1)
