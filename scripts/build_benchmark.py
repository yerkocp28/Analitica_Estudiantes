"""Construye perfiles de ingreso y trazabilidad del benchmark universitario."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from student_analytics.ingestion.cned import attach_resources, load_institutional
from student_analytics.ingestion.paes import (attach_selectivity, institutional_selectivity,
                                              load_scores)
from student_analytics.modeling.benchmark import build_profiles


def main() -> None:
    source = ROOT / "data/external/mineduc"
    retention = pd.read_parquet(ROOT / "data/results/retention_universities.parquet")
    base = ["mrun", "anio_ing_carr_ori", "nivel_global", "tipo_inst_1", "cod_inst", "nomb_inst",
            "nomb_sede", "cod_carrera", "nomb_carrera", "area_conocimiento"]
    extra = ["cod_sede", "region_sede", "modalidad", "jornada",
             # Descriptivas: ya vienen en la matricula publica y no cuestan
             # una descarga adicional. No participan en la distancia.
             "acre_inst_anio", "acreditada_carr", "formato_valores",
             "valor_matricula", "valor_arancel", "dur_total_carr", "forma_ingreso"]
    uf = (yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
          .get("uf_clp") or {})
    profiles, sources = [], []
    for year in sorted(retention.cohorte.unique()):
        matches = sorted((source / f"matricula_{year}").rglob("*.csv"))
        if len(matches) != 1:
            raise ValueError(f"Se esperaba un CSV de matrícula para {year}; encontrados: {len(matches)}")
        path = matches[0]
        print(f"Perfil de ingreso {year}", flush=True)
        chunks = pd.read_csv(path, sep=";", encoding="utf-8", usecols=base + extra,
                             dtype={c: "string" for c in base + extra if c != "anio_ing_carr_ori"}, chunksize=100_000)
        cohort_parts, enrollment_parts = [], []
        for chunk in chunks:
            universities = chunk.loc[chunk.tipo_inst_1.eq("Universidades")]
            # Todos los niveles y años de ingreso para recursos institucionales.
            enrollment_parts.append(universities[["cod_inst", "mrun"]])
            cohort_parts.append(universities.loc[universities.anio_ing_carr_ori.eq(year)
                                                & universities.nivel_global.eq("Pregrado")])
        cohort = pd.concat(cohort_parts)
        all_enrollment = pd.concat(enrollment_parts, ignore_index=True)
        total = all_enrollment.groupby("cod_inst").agg(
            estudiantes_total=("mrun", "nunique"),
            matriculas_total=("cod_inst", "size"),
            matriculas_sin_mrun=("mrun", lambda x: int(x.isna().sum())),
        )
        cohort = cohort.drop_duplicates(subset=base)
        p = build_profiles(cohort, int(year), uf_clp=uf.get(int(year)))
        p = p.merge(total, on="cod_inst", how="left", validate="one_to_one")

        # Selectividad de admision: se calcula aqui porque la cohorte ya esta
        # en memoria con mrun y cod_inst. No requiere descargas nuevas.
        paes_dir = source / f"paes_{year}_puntajes"
        paes_files = sorted(paes_dir.rglob("*.csv")) if paes_dir.exists() else []
        if paes_files:
            scores = load_scores(paes_files[0])
            selectividad = institutional_selectivity(cohort, scores)
            p = attach_selectivity(p, selectividad)
            cobertura = p.paes_cobertura.median() if "paes_cobertura" in p else float("nan")
            print(f"  selectividad PAES: cobertura mediana {cobertura:.0%}", flush=True)
            sources.append({"cohorte": int(year), "archivo": paes_files[0].name,
                            "bytes": paes_files[0].stat().st_size,
                            "uso": "selectividad de admision"})
            del scores
        else:
            print(f"  sin PAES {year}; se omite la selectividad", flush=True)

        del all_enrollment, enrollment_parts, cohort_parts
        expected = retention.loc[retention.cohorte.eq(year)].groupby("cod_inst")[["n", "sin_mrun"]].sum().sum(axis=1)
        actual = p.set_index("cod_inst").cohorte_total
        if not actual.sort_index().equals(expected.reindex(actual.index).sort_index()):
            raise ValueError("El universo de perfiles no coincide con el de retención")
        profiles.append(p)
        sources.append({"cohorte": int(year), "archivo": path.name, "bytes": path.stat().st_size,
                        "modificado_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    out = ROOT / "data/results"
    result = pd.concat(profiles, ignore_index=True)

    # Recursos institucionales del CNED, si estan descargados. Se unen por
    # nombre normalizado porque los codigos de institucion del CNED no son
    # los del SIES. Son descriptivos: no entran en la distancia.
    cned = ROOT / "data/external/cned/INDICES_Institucional_2005-2025.xlsx"
    if cned.exists():
        piezas = []
        for year, grupo in result.groupby("cohorte"):
            recursos = load_institutional(cned, int(year))
            unido = attach_resources(grupo, recursos)
            calce = unido.docentes_jce.notna().mean() if "docentes_jce" in unido else 0
            print(f"  cohorte {year}: {calce:.0%} de las universidades con datos CNED",
                  flush=True)
            piezas.append(unido)
        result = pd.concat(piezas, ignore_index=True)
        sources.append({"fuente": "CNED INDICES Institucional", "archivo": cned.name,
                        "bytes": cned.stat().st_size})
    else:
        print("  sin base CNED; se omiten los recursos institucionales "
              "(python scripts/download_cned.py)", flush=True)

    shares = [c for c in result if "::" in c]
    result[shares] = result[shares].fillna(0)
    result.to_parquet(out / "benchmark_profiles.parquet", index=False)
    manifest = {"generado_utc": datetime.now(timezone.utc).isoformat(), "version": 2,
                "recursos_denominador": "MRUN únicos por universidad y año, todos los niveles; excluye MRUN ausente",
                "recursos_temporalidad": "solo año exacto; acreditación CNED conserva fecha del catálogo",
                "fuentes": sources, "perfil": "cohorte de ingreso a carrera, pregrado universitario",
                "retencion_sha256": hashlib.sha256((out / "retention_universities.parquet").read_bytes()).hexdigest(),
                "perfiles_sha256": hashlib.sha256((out / "benchmark_profiles.parquet").read_bytes()).hexdigest()}
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Perfiles guardados: {len(result)} universidad-cohorte", flush=True)


if __name__ == "__main__":
    main()
