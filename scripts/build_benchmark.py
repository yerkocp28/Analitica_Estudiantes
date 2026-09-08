"""Construye perfiles de ingreso y trazabilidad del benchmark universitario."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from student_analytics.modeling.benchmark import build_profiles


def main() -> None:
    source = ROOT / "data/external/mineduc"
    retention = pd.read_parquet(ROOT / "data/results/retention_universities.parquet")
    base = ["mrun", "anio_ing_carr_ori", "nivel_global", "tipo_inst_1", "cod_inst", "nomb_inst",
            "nomb_sede", "cod_carrera", "nomb_carrera", "area_conocimiento"]
    extra = ["cod_sede", "region_sede", "modalidad", "jornada"]
    profiles, sources = [], []
    for year in sorted(retention.cohorte.unique()):
        matches = sorted((source / f"matricula_{year}").rglob("*.csv"))
        if len(matches) != 1:
            raise ValueError(f"Se esperaba un CSV de matrícula para {year}; encontrados: {len(matches)}")
        path = matches[0]
        print(f"Perfil de ingreso {year}", flush=True)
        chunks = pd.read_csv(path, sep=";", encoding="utf-8", usecols=base + extra,
                             dtype={c: "string" for c in base + extra if c != "anio_ing_carr_ori"}, chunksize=100_000)
        cohort = pd.concat([c.loc[c.anio_ing_carr_ori.eq(year) & c.nivel_global.eq("Pregrado")
                                  & c.tipo_inst_1.eq("Universidades")] for c in chunks])
        cohort = cohort.drop_duplicates(subset=base)
        p = build_profiles(cohort, int(year))
        expected = retention.loc[retention.cohorte.eq(year)].groupby("cod_inst")[["n", "sin_mrun"]].sum().sum(axis=1)
        actual = p.set_index("cod_inst").cohorte_total
        if not actual.sort_index().equals(expected.reindex(actual.index).sort_index()):
            raise ValueError("El universo de perfiles no coincide con el de retención")
        profiles.append(p)
        sources.append({"cohorte": int(year), "archivo": path.name, "bytes": path.stat().st_size,
                        "modificado_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    out = ROOT / "data/results"
    result = pd.concat(profiles, ignore_index=True)
    shares = [c for c in result if "::" in c]
    result[shares] = result[shares].fillna(0)
    result.to_parquet(out / "benchmark_profiles.parquet", index=False)
    manifest = {"generado_utc": datetime.now(timezone.utc).isoformat(), "version": 1,
                "fuentes": sources, "perfil": "cohorte de ingreso a carrera, pregrado universitario",
                "retencion_sha256": hashlib.sha256((out / "retention_universities.parquet").read_bytes()).hexdigest(),
                "perfiles_sha256": hashlib.sha256((out / "benchmark_profiles.parquet").read_bytes()).hexdigest()}
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Perfiles guardados: {len(result)} universidad-cohorte", flush=True)


if __name__ == "__main__":
    main()
