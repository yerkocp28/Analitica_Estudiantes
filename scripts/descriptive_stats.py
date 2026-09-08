"""Precomputa las estadisticas descriptivas del informe metodologico.

    python scripts/descriptive_stats.py

Deja artefactos pequenios en documentacion/datos/ para que el .qmd se
renderice en segundos y siga funcionando aunque las bases originales (~3 GB)
no esten en la maquina. Cada cifra del informe sale de aca, no escrita a
mano: si los datos cambian, el informe cambia.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("descriptive_stats")

SYN = REPO_ROOT / "data" / "synthetic"
OULAD = REPO_ROOT / "data" / "external" / "oulad"
MINEDUC = REPO_ROOT / "data" / "external" / "mineduc"
RESULTS = REPO_ROOT / "data" / "results"
OUT = REPO_ROOT / "documentacion" / "datos"


def _save(name: str, df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    log.info("  %-32s %5d filas", name, len(df))


def stats_sinteticos(meta: dict) -> None:
    if not (SYN / "student_master.parquet").exists():
        log.warning("Sin datos sinteticos; se omite")
        return
    log.info("Sinteticos")
    sm = pd.read_parquet(SYN / "student_master.parquet")
    out = pd.read_parquet(SYN / "outcomes.parquet")
    cm = pd.read_parquet(SYN / "course_master.parquet")
    aw = pd.read_parquet(SYN / "attendance_weekly.parquet")
    gw = pd.read_parquet(SYN / "grades_weekly.parquet")

    meta["sintetico"] = {
        "estudiantes": int(len(sm)),
        "asignaturas": int(len(cm)),
        "inscripciones": int(len(out)),
        "filas_panel_semanal": int(len(aw)),
        "notas_parciales": int(len(gw)),
        "nota_final_media": round(float(out["final_grade"].mean()), 2),
        "nota_final_sd": round(float(out["final_grade"].std()), 2),
        "reprobacion_por_nota": round(float(out["fail_by_grade"].mean()), 4),
        "reprobacion_por_inasistencia": round(float(out["fail_by_attendance"].mean()), 4),
        "reprobacion_total": round(float(out["failed"].mean()), 4),
        "asistencia_media": round(float(aw["attendance_rate"].mean()), 4),
        "cursos_sin_registro_asistencia": round(
            float(1 - cm["attendance_recorded"].mean()), 4),
    }

    _save("sint_por_sede", sm.groupby("campus", as_index=False)
          .size().rename(columns={"size": "estudiantes"}))
    _save("sint_por_carrera", sm.groupby("career_code", as_index=False)
          .size().rename(columns={"size": "estudiantes"}))

    perf = (out.merge(sm[["student_id", "_latent_profile"]], on="student_id")
            .groupby("_latent_profile")
            .agg(inscripciones=("failed", "size"), reprobacion=("failed", "mean"))
            .reset_index().sort_values("reprobacion"))
    perf["reprobacion"] = perf["reprobacion"].round(4)
    _save("sint_por_perfil", perf)

    tier = (out.merge(cm[["course_id", "difficulty_tier"]], on="course_id")
            .groupby("difficulty_tier")
            .agg(inscripciones=("failed", "size"), reprobacion=("failed", "mean"))
            .reset_index())
    tier["reprobacion"] = tier["reprobacion"].round(4)
    _save("sint_por_dificultad", tier)

    _save("sint_adopcion_plataforma", cm.groupby("canvas_adoption", as_index=False)
          .size().rename(columns={"size": "asignaturas"}))


def stats_oulad(meta: dict) -> None:
    info_files = list(OULAD.rglob("studentInfo.csv"))
    if not info_files:
        log.warning("Sin OULAD; se omite")
        return
    log.info("OULAD")
    root = info_files[0].parent
    info = pd.read_csv(root / "studentInfo.csv")
    assess = pd.read_csv(root / "studentAssessment.csv")

    meta["oulad"] = {
        "estudiantes_unicos": int(info["id_student"].nunique()),
        "inscripciones": int(len(info)),
        "modulos": int(info["code_module"].nunique()),
        "presentaciones": sorted(info["code_presentation"].unique().tolist()),
        "regiones": int(info["region"].nunique()),
        "entregas_evaluadas": int(len(assess)),
    }
    res = info["final_result"].value_counts().reset_index()
    res.columns = ["resultado_final", "n"]
    res["proporcion"] = (res["n"] / res["n"].sum()).round(4)
    _save("oulad_resultado_final", res)

    dem = info["age_band"].value_counts().reset_index()
    dem.columns = ["tramo_edad", "n"]
    _save("oulad_tramo_edad", dem)


def stats_mineduc(meta: dict) -> None:
    from analyze_mineduc import MAT_DTYPES  # noqa

    files = {y: list((MINEDUC / f"matricula_{y}").rglob("*.csv"))
             for y in (2023, 2024, 2025)}
    if not all(files.values()):
        log.warning("Sin bases Mineduc completas; se omite")
        return
    log.info("Mineduc")
    cols = ["mrun", "nivel_global", "tipo_inst_1", "nomb_inst", "nomb_sede",
            "nomb_carrera", "area_conocimiento", "anio_ing_carr_ori", "gen_alu"]
    filas = []
    for y, f in files.items():
        n = sum(1 for _ in open(f[0], encoding="utf-8")) - 1
        filas.append({"anio": y, "filas": n})
    _save("mineduc_volumen", pd.DataFrame(filas))

    m24 = pd.read_csv(files[2024][0], sep=";", encoding="utf-8", usecols=cols,
                      dtype={k: v for k, v in MAT_DTYPES.items() if k in cols})
    ua = m24[m24["nomb_inst"].str.contains("AUTONOMA", case=False, na=False)
             & (m24["nivel_global"] == "Pregrado")]
    meta["mineduc"] = {
        "matricula_total_2024": int(len(m24)),
        "ua_pregrado_2024": int(len(ua)),
        "ua_cohorte_2024": int((ua["anio_ing_carr_ori"] == 2024).sum()),
        "ua_sedes": sorted(ua["nomb_sede"].dropna().unique().tolist()),
    }
    adm = ua[ua["area_conocimiento"].str.contains("Administraci", case=False, na=False)]
    _save("mineduc_ua_admin_por_carrera",
          adm.groupby("nomb_carrera", as_index=False).size()
          .rename(columns={"size": "matriculados_2024"})
          .sort_values("matriculados_2024", ascending=False))


def stats_paes(meta: dict) -> None:
    """Descriptivas de los puntajes PAES y del expediente escolar."""
    from analyze_mineduc import PAES_COLS, SCORE_COLS, _decimal_comma, _find  # noqa

    try:
        path = _find("paes_2024_puntajes", "*.csv")
    except FileNotFoundError:
        log.warning("Sin PAES; se omite")
        return
    log.info("PAES 2024")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig",
                     usecols=[c for c in PAES_COLS], dtype=str, low_memory=False)
    for c in SCORE_COLS:
        df[c] = _decimal_comma(df[c])
        df.loc[df[c] <= 0, c] = None   # 0 = no rindio esa prueba

    filas = []
    etiquetas = {
        "PROMEDIO_NOTAS": "Promedio de notas de ensenanza media",
        "PTJE_NEM": "Puntaje NEM", "PTJE_RANKING": "Puntaje ranking",
        "CLEC_MAX": "PAES Competencia Lectora", "MATE1_MAX": "PAES Matematica M1",
        "MATE2_MAX": "PAES Matematica M2", "HCSOC_MAX": "PAES Historia y Cs. Sociales",
        "CIEN_MAX": "PAES Ciencias",
    }
    for c, etiq in etiquetas.items():
        s = df[c]
        filas.append({
            "variable": etiq, "rindio": round(float(s.notna().mean()), 4),
            "media": round(float(s.mean()), 1), "sd": round(float(s.std()), 1),
            "p25": round(float(s.quantile(.25)), 1),
            "p50": round(float(s.quantile(.50)), 1),
            "p75": round(float(s.quantile(.75)), 1),
        })
    _save("paes_descriptivas", pd.DataFrame(filas))

    dep = pd.to_numeric(df["DEPENDENCIA"], errors="coerce").map(
        {1: "Corporacion Municipal", 2: "Municipal",
         3: "Particular Subvencionado", 4: "Particular Pagado",
         5: "Corp. de Administracion Delegada",
         6: "Servicio Local de Educacion"})
    d = dep.value_counts(dropna=False).reset_index()
    d.columns = ["dependencia", "n"]
    d["proporcion"] = (d["n"] / d["n"].sum()).round(4)
    _save("paes_dependencia", d)

    meta["paes"] = {
        "inscritos_2024": int(len(df)),
        "rindio_lectora": round(float(df["CLEC_MAX"].notna().mean()), 4),
        "rindio_m1": round(float(df["MATE1_MAX"].notna().mean()), 4),
    }


def stats_resultados(meta: dict) -> None:
    log.info("Resultados de modelos")
    for src in ("synthetic", "oulad"):
        f = RESULTS / f"metrics_{src}.parquet"
        if f.exists():
            _save(f"metricas_{src}", pd.read_parquet(f))
    for f in ("piso_preingreso_nacional", "piso_preingreso_temporal"):
        p = RESULTS / f"{f}.parquet"
        if p.exists():
            _save(f, pd.read_parquet(p))
    # ablacion_piso.csv y descriptivas_trayectoria_escolar.csv los produce
    # analyze_mineduc.py directamente en documentacion/datos/.


def main() -> int:
    meta: dict = {}
    stats_sinteticos(meta)
    stats_oulad(meta)
    stats_mineduc(meta)
    stats_paes(meta)
    stats_resultados(meta)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resumen.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Listo en %s", OUT)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
