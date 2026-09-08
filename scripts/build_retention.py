"""Genera agregados públicos para Streamlit: python scripts/build_retention.py."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from student_analytics.ingestion.retention import aggregate_retention


def main() -> None:
    source = ROOT / "data" / "external" / "mineduc"
    files = {int(p.name.split("_")[-1]): next(iter(sorted(p.rglob("*.csv"))), None)
             for p in source.glob("matricula_*") if p.is_dir()}
    cols = ["mrun", "anio_ing_carr_ori", "nivel_global", "tipo_inst_1", "cod_inst",
            "nomb_inst", "nomb_sede", "cod_carrera", "nomb_carrera", "area_conocimiento"]
    results = []
    for year in sorted(files):
        if not files[year] or not files.get(year + 1):
            continue
        print(f"Procesando cohorte {year} -> {year + 1}", flush=True)
        dtype = {c: "string" for c in cols if c != "anio_ing_carr_ori"}
        # Filtrar por bloques evita mantener la matrícula completa en memoria.
        chunks = pd.read_csv(files[year], sep=";", encoding="utf-8", usecols=cols,
                             dtype=dtype, chunksize=100_000)
        current = pd.concat([c.loc[c["anio_ing_carr_ori"].eq(year)
                                 & c["nivel_global"].eq("Pregrado")
                                 & c["tipo_inst_1"].eq("Universidades")] for c in chunks])
        keys = ["mrun", "cod_inst", "cod_carrera"]
        following = pd.read_csv(files[year + 1], sep=";", encoding="utf-8",
                                usecols=keys, dtype={c: "string" for c in keys})
        results.append(aggregate_retention(current, following, year))
    if not results:
        raise SystemExit("Se necesitan al menos dos matrículas de años consecutivos en data/external/mineduc.")
    out = ROOT / "data" / "results" / "retention_universities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results, ignore_index=True).to_parquet(out, index=False)
    print(f"Agregados guardados en {out}")


if __name__ == "__main__":
    main()
