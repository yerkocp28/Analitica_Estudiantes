"""Integración descriptiva por período. No inventa etiquetas ni fechas semanales."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import uuid

import numpy as np
import pandas as pd

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
SOURCE = ROOT / "data/interim/Matricula e Historia Académica Pregrado 2024 2025.xlsx"
CONFIG = HOME / "institucional_pendientes.json"
FILES = {
    "enrollment": "matricula_plan_periodo.parquet",
    "courses": "resultados_asignatura_periodo.parquet",
    "students": "mart_estudiante_periodo.parquet",
}
KEYS = ["student_key", "term"]
MAPPING_M = {
    "PIDM": "student_key", "TERM_CODE": "term", "PERIODO": "period",
    "LEVL_CODE": "level_code", "NIVEL": "level",
    "CAMP_CODE": "campus_code", "CAMPUS": "campus",
    "MAJR_CODE": "career_code", "CARRERA": "career",
    "PROGRAM": "program_code", "PROGRAMA": "program",
    "FACULTAD": "faculty", "JORNADA": "schedule",
    "STATUS_PLAN_ESTUDIO": "plan_status", "STATUS_ALUMNO": "student_status",
    "TIPO_ALUMNO": "student_type", "TIPO_ADMISION": "admission_type",
    "TERM_CODE_ADMIT": "admission_term",
}
MAPPING_H = {
    "PIDM": "student_key", "TERM_CODE": "term", "NRC": "nrc",
    "CAMP_CODE": "course_campus_code", "LEVL_CODE": "course_level_code",
    "SUBJ_CODE": "subject_code", "CRSE_NUMB": "course_number",
    "CRSE_TITLE": "course_name", "GRDE_CODE_FINAL": "grade_raw",
    "CREDIT_HOURS": "workload_raw",
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _normalized(frame, mapping, prefix, secret):
    required = [prefix + c for c in mapping]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError("Faltan columnas del extracto: " + ", ".join(missing))
    out = frame[required].rename(columns={prefix+k: v for k, v in mapping.items()}).copy()
    for c in out:
        out[c] = out[c].astype("string").str.strip().replace("", pd.NA)
    for c in ("student_key", "term"):
        if out[c].isna().any():
            raise ValueError(f"Clave ausente: {c}; revisar la fuente antes de unir tablas.")
    # Normalize integral Excel identifiers without conflating different identifiers.
    ids = out.student_key.str.replace(r"\.0$", "", regex=True)
    lookup = {value: hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()
              for value in ids.unique()}
    out["student_key"] = ids.map(lookup)
    term = pd.to_numeric(out.term, errors="raise")
    if not term.mod(1).eq(0).all():
        raise ValueError("Períodos no enteros.")
    out["term"] = term.astype("int64")
    return out


def regime(label):
    label = str(label).casefold()
    if "causales" in label or "bloque" in label:
        return "Otros períodos"
    if "verano" in label or "tav" in label:
        return "Verano / TAV"
    if "trimestre" in label:
        return "Trimestral"
    if "semestre" in label:
        return "Semestral"
    return "Sin clasificación"


def prepare(history, enrollment, cfg, secret):
    """Devuelve tablas locales sin PIDM, junto a una auditoría agregada."""
    m = _normalized(enrollment, MAPPING_M, "UWVMAHI_", secret)
    h = _normalized(history, MAPPING_H, "UWVHIAC_", secret)
    for frame, keys in [(m, KEYS + ["program_code"]), (h, KEYS + ["nrc"])]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Grano de origen inválido: " + ", ".join(keys))
    m["year"] = m.term // 100
    m["regime"] = m.period.map(regime)
    scope = cfg["alcance"]
    m["fan_label"] = m.faculty.eq(scope["facultad"]).fillna(False)
    m["fan_proposed"] = (m.fan_label & m.career_code.isin(scope["carreras_propuestas"])
                         & m.campus_code.isin(scope["campus_propuestos"])
                         & m.level_code.eq(scope["nivel_propuesto"])).fillna(False)
    multiplicity = m.groupby(KEYS).size().rename("plans").reset_index()
    m = m.merge(multiplicity, on=KEYS, how="left", validate="many_to_one")
    safe = m.loc[m.plans.eq(1)].drop(columns="plans")
    raw_numeric = pd.to_numeric(h.grade_raw.str.replace(",", ".", regex=False),
                                errors="coerce").astype(float)
    h["grade_numeric"] = raw_numeric.where(raw_numeric.between(1, 7))
    h["grade_kind"] = np.select(
        [h.grade_numeric.notna(), raw_numeric.notna(), h.grade_raw.isna()],
        ["Numérica 1–7", "Numérica fuera de rango", "Sin resultado"],
        default="Código institucional")
    h["result_code"] = h.grade_raw.where(h.grade_kind.eq("Código institucional"))
    h["numeric_below4"] = h.grade_numeric.lt(4).astype(float).where(h.grade_numeric.notna())
    h["workload_raw"] = pd.to_numeric(
        h.workload_raw.str.replace(",", ".", regex=False), errors="coerce")
    h["course_code"] = h.subject_code + ":" + h.course_number
    h = h.merge(multiplicity, on=KEYS, how="left", validate="many_to_one")
    h["attribution"] = np.select(
        [h.plans.eq(1), h.plans.gt(1)], ["Única", "Varios planes"], default="Sin matrícula coincidente")
    h = h.merge(safe, on=KEYS, how="left", validate="many_to_one")
    h["campus_mismatch"] = (h.attribution.eq("Única") & h.course_campus_code.notna()
                            & h.campus_code.notna() & h.course_campus_code.ne(h.campus_code))
    # Results are observed facts, not complete course registrations.
    academic = h.groupby(KEYS).agg(
        observed_results=("nrc", "size"), numeric_results=("grade_numeric", "count"),
        numeric_mean=("grade_numeric", "mean"), numeric_sd=("grade_numeric", "std"),
        numeric_below4_count=("numeric_below4", "sum"),
        coded_results=("result_code", "count")).reset_index()
    academic["numeric_below4_count"] = academic.numeric_below4_count.where(
        academic.numeric_results.gt(0))
    students = multiplicity.merge(safe, on=KEYS, how="left", validate="one_to_one")
    students = students.merge(academic, on=KEYS, how="left", validate="one_to_one")
    students["has_observed_history"] = students.observed_results.notna()
    students["numeric_coverage"] = students.numeric_results / students.observed_results
    quality = {
        "history_rows": len(h), "enrollment_rows": len(m),
        "history_people": int(h.student_key.nunique()),
        "enrollment_people": int(m.student_key.nunique()),
        "multiple_plan_person_periods": int(multiplicity.plans.gt(1).sum()),
        "history_attribution": {str(k): int(v) for k,v in h.attribution.value_counts().items()},
        "enrollment_person_periods_without_history": int((~students.has_observed_history).sum()),
        "grade_kinds": {str(k): int(v) for k,v in h.grade_kind.value_counts().items()},
        "coded_results": {str(k): int(v) for k,v in h.result_code.value_counts().items()},
        "campus_mismatch_rows": int(h.campus_mismatch.sum()),
        "periods": sorted(int(x) for x in m.term.unique()),
        "fan": {},
    }
    for label in ("fan_label", "fan_proposed"):
        sub = m.loc[m[label]]
        matches = h.merge(sub[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one")
        quality["fan"][label] = {
            "people": int(sub.student_key.nunique()), "enrollment_rows": len(sub),
            "person_periods": len(sub[KEYS].drop_duplicates()),
            "observed_history_rows": len(matches),
            "unambiguous_history_rows": int(matches.attribution.eq("Única").sum()),
        }
    return {"enrollment": m, "courses": h, "students": students}, quality


def select_data(frames, scope="fan_proposed", filters=None):
    if scope not in ("fan_proposed", "fan_label"):
        raise ValueError("Alcance desconocido.")
    m = frames["enrollment"].loc[frames["enrollment"][scope]].copy()
    for column, value in (filters or {}).items():
        if value is not None:
            m = m.loc[m[column].eq(value)]
    keys = m[KEYS].drop_duplicates()
    h = frames["courses"].merge(keys, on=KEYS, how="inner", validate="many_to_one")
    unresolved = h.loc[~h.attribution.eq("Única")]
    return m, h.loc[h.attribution.eq("Única")].copy(), unresolved


def summary(m, h):
    known = h.grade_numeric.notna()
    return {
        "people": int(m.student_key.nunique()),
        "plan_rows": len(m), "person_periods": len(m[KEYS].drop_duplicates()),
        "history_people": int(h.student_key.nunique()), "history_rows": len(h),
        "numeric_results": int(known.sum()),
        "numeric_coverage": float(known.mean()) if len(h) else None,
        "numeric_mean": float(h.loc[known, "grade_numeric"].mean()) if known.any() else None,
        "numeric_below4_rate": float(h.loc[known, "grade_numeric"].lt(4).mean()) if known.any() else None,
    }


GROUP_LABELS = {
    "period": "Período", "campus": "Campus", "career": "Carrera",
    "course_code": "Código de asignatura", "course_name": "Asignatura",
    "result_code": "Código de resultado", "schedule": "Jornada",
}
MEASURES = ["Personas con resultados", "Registros de asignatura", "Notas numéricas",
            "Cobertura numérica (%)", "Promedio numérico", "Notas numéricas bajo 4 (%)"]


def grouped_results(h, groups, minimum=10):
    rows = []
    for values, g in h.groupby(groups, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        if g.student_key.nunique() < minimum:
            continue
        n = g.grade_numeric.notna()
        # Do not expose a numeric subgroup unless enough distinct people contribute.
        enough = g.loc[n, "student_key"].nunique() >= minimum
        rows.append({
            **{GROUP_LABELS[c]: v for c, v in zip(groups, values)},
            "Personas con resultados": int(g.student_key.nunique()),
            "Registros de asignatura": len(g),
            "Notas numéricas": int(n.sum()) if enough else np.nan,
            "Cobertura numérica (%)": 100*n.mean() if enough else np.nan,
            "Promedio numérico": float(g.loc[n, "grade_numeric"].mean()) if enough else np.nan,
            "Notas numéricas bajo 4 (%)": 100*g.loc[n, "grade_numeric"].lt(4).mean() if enough else np.nan,
        })
    return pd.DataFrame(rows, columns=[GROUP_LABELS[c] for c in groups] + MEASURES)


def export_aggregate(table, context):
    from .dashboard import export_csv
    if set(table) - (set(GROUP_LABELS.values()) | set(MEASURES)):
        raise ValueError("Solo se exportan columnas agregadas aprobadas.")
    return export_csv(table, {}, context)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def load(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name in [*FILES.values(), "quality.json", "pendientes.json"]:
        if digest(folder / name) != manifest["artifacts_sha256"][name]:
            raise ValueError("Artefacto institucional alterado: " + name)
    frames = {key: pd.read_parquet(folder/name) for key, name in FILES.items()}
    return frames, json.loads((folder/"quality.json").read_text(encoding="utf-8")), manifest


def build(source=SOURCE, base=None):
    base = Path(base) if base else HOME / "artifacts/institucional"
    base.mkdir(parents=True, exist_ok=True)
    secret_path = base / ".identity_key"
    if not secret_path.exists():
        with secret_path.open("xb") as f:
            f.write(secrets.token_bytes(32))
    secret = secret_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("Clave de seudonimización inválida.")
    source_sha = digest(source)
    config_sha = digest(CONFIG)
    cfg = read_config()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = base / run_id
    out.mkdir(exist_ok=False)
    print("Leyendo historia académica completa...", flush=True)
    h = pd.read_excel(source, sheet_name="STG UWVHIAC", engine="openpyxl")
    print("Leyendo matrícula y planes completos...", flush=True)
    m = pd.read_excel(source, sheet_name="STG UWVMAHI", engine="openpyxl")
    print("Construyendo tablas por período y auditoría...", flush=True)
    frames, quality = prepare(h, m, cfg, secret)
    for key, name in FILES.items():
        frames[key].to_parquet(out/name, index=False)
    write_json(out/"quality.json", quality)
    write_json(out/"pendientes.json", cfg)
    if digest(source) != source_sha or digest(CONFIG) != config_sha:
        raise ValueError("La fuente o configuración cambió durante la generación; no se actualizó latest.")
    manifest = {
        "run_id": run_id, "created_utc": now.isoformat(),
        "source_file": Path(source).name, "source_sha256": source_sha,
        "extraction_cutoff": None, "source_origin": cfg["procedencia"],
        "config_sha256": config_sha, "build_code_sha256": digest(__file__),
        "schema_version": "institutional-term-v1",
        "use": "Descriptivo retrospectivo 2024–2025; sin modelos predictivos internos.",
        "artifacts_sha256": {name: digest(out/name) for name in
                            [*FILES.values(), "quality.json", "pendientes.json"]},
    }
    write_json(out/"manifest.json", manifest)
    load(out)
    pointer = base / ("latest." + uuid.uuid4().hex + ".tmp")
    write_json(pointer, {"run_id": run_id})
    os.replace(pointer, base/"latest.json")
    print(json.dumps({"run_id": run_id, "fan": quality["fan"]}, ensure_ascii=True), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    build(parser.parse_args().source)
