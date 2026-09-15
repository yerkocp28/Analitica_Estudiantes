"""Segmenta a los ingresantes de la UA: python scripts/build_segments.py

Entregable del mes 3, sobre estudiantes reales. Requiere antes:
    python scripts/build_ua_cohorts.py

Escribe en data/results/:
    ua_segmentos.parquet             la asignación, una fila por ingresante
    ua_segmentos_perfil.csv          qué es cada segmento y qué continuidad tiene
    ua_segmentos_diagnostico.csv     todos los candidatos comparados
    ua_segmentos_estabilidad.csv     cómo se reparte cada cohorte
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402
from student_analytics.modeling.segmentation import (  # noqa: E402
    estabilidad,
    perfilar,
    segmentar,
)

log = get_logger("build_segments")
ENTRADA = REPO_ROOT / "data" / "results" / "ua_ingresantes.parquet"
SALIDA = REPO_ROOT / "data" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, help="cohorte inicial (default: config)")
    ap.add_argument("--hasta", type=int, help="cohorte final (default: config)")
    args = ap.parse_args()

    if not ENTRADA.exists():
        print("Falta el dataset de ingresantes. Corre: python scripts/build_ua_cohorts.py")
        return 1
    config = yaml.safe_load((REPO_ROOT / "config/segmentation.yml").read_text(encoding="utf-8"))
    desde = args.desde or config["cohorte_desde"]
    hasta = args.hasta or config["cohorte_hasta"]

    d = pd.read_parquet(ENTRADA)
    d = d[d.cohorte.between(desde, hasta)].reset_index(drop=True)
    if d.empty:
        print(f"Sin ingresantes en {desde}-{hasta}.")
        return 1
    print(f"Segmentando {len(d):,} ingresantes UA · cohortes {desde}–{hasta}\n")

    modelo = segmentar(d, config)
    perfil = perfilar(d, modelo.etiquetas)
    estab = estabilidad(d, modelo.etiquetas)

    salida = d[["mrun", "cohorte", "nomb_sede", "nomb_carrera"]].copy()
    salida["segmento"] = modelo.etiquetas.to_numpy()
    salida.to_parquet(SALIDA / "ua_segmentos.parquet", index=False)
    perfil.to_csv(SALIDA / "ua_segmentos_perfil.csv", index=False, encoding="utf-8")
    modelo.diagnostico.to_csv(SALIDA / "ua_segmentos_diagnostico.csv",
                              index=False, encoding="utf-8")
    estab.to_csv(SALIDA / "ua_segmentos_estabilidad.csv", index=False, encoding="utf-8")

    print(f"Elegido: {modelo.algoritmo} con k={modelo.k} · silueta {modelo.silueta:.3f}")
    admisibles = modelo.diagnostico.admisible.sum()
    print(f"Candidatos comparados: {len(modelo.diagnostico)} "
          f"({admisibles} admisibles tras el tamaño mínimo y el techo de concentración)\n")

    cols = [c for c in ("segmento", "n", "share", "sede_principal", "area_principal",
                        "percentil_admision", "sigue_carrera", "brecha_carrera_pp")
            if c in perfil.columns]
    vista = perfil[cols].copy()
    for c in ("share", "percentil_admision", "sigue_carrera"):
        if c in vista:
            vista[c] = vista[c].map("{:.1%}".format)
    if "brecha_carrera_pp" in vista:
        vista["brecha_carrera_pp"] = vista.brecha_carrera_pp.map("{:+.1f}".format)
    print(vista.to_string(index=False))
    print(f"\n-> {SALIDA.relative_to(REPO_ROOT)}/ua_segmentos*.{{parquet,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
