"""Auditoría local de solo lectura. No modifica los modelos ni publica datos."""
from pathlib import Path
import hashlib
import json
import time
import html
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/Matricula e Historia Académica Pregrado 2024 2025.xlsx"
OUT = ROOT / "piloto_fan/artifacts/auditoria_excel_20261020"
OUT.mkdir(parents=True, exist_ok=True)


def counts(s, limit=40):
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).head(limit).items()}


def numeric(s):
    """Excel mezcla códigos y notas con coma decimal: no convertirlos en ceros."""
    return pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False),
                         errors="coerce").astype(float)


def write_json(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def profile(df):
    result = {}
    for c in df:
        s = df[c]
        p = {"dtype": str(s.dtype), "missing": int(s.isna().sum()),
             "missing_pct": float(s.isna().mean() * 100),
             "distinct": int(s.nunique()), "blank_strings": int(s.eq("").sum())}
        # Never print identifiers or examples of individual records.
        if "PIDM" not in c:
            if s.nunique() <= 100:
                p["values"] = counts(s, 105)
            elif any(x in c for x in ("TITLE", "CARRERA", "PROGRAMA")):
                p["top_values"] = counts(s, 25)
            n = numeric(s)
            if n.notna().any():
                p["numeric"] = {"count": int(n.notna().sum()),
                                "min": float(n.min()), "max": float(n.max())}
        result[c] = p
    return result


def key_summary(df, keys):
    valid = df.dropna(subset=keys)
    g = valid.groupby(keys, dropna=False).size()
    return {"keys": keys, "rows_missing_key": len(df) - len(valid),
            "unique_keys": len(g), "duplicate_keys": int(g.gt(1).sum()),
            "rows_in_duplicate_keys": int(g[g.gt(1)].sum()),
            "excess_rows": int((g - 1).clip(lower=0).sum()),
            "max_multiplicity": int(g.max()) if len(g) else 0}


def main():
    t = time.time()
    sha = hashlib.file_digest(SOURCE.open("rb"), "sha256").hexdigest()
    cached = OUT / "source_sha256.txt"
    cache_valid = cached.exists() and cached.read_text() == sha
    tables = {}
    for name, short in [("STG UWVHIAC", "h"), ("STG UWVMAHI", "m")]:
        cache = OUT / f"{short}_local.parquet"
        if cache_valid and cache.exists():
            df = pd.read_parquet(cache)
        else:
            print(f"Leyendo completa {name}...", flush=True)
            df = pd.read_excel(SOURCE, sheet_name=name, engine="openpyxl")
            df.to_parquet(cache, index=False)
        tables[short] = df
        print(f"{name}: {len(df):,} filas, {len(df.columns)} columnas, "
              f"{time.time()-t:.1f}s", flush=True)
    cached.write_text(sha)
    h, m = tables["h"], tables["m"]
    report = {"source": SOURCE.name, "sha256": sha, "bytes": SOURCE.stat().st_size,
              "sheets": {}, "quality": {}, "joins": {}, "fan": {}, "temporal": {}}
    for short, df in tables.items():
        p = "UWVHIAC_" if short == "h" else "UWVMAHI_"
        report["sheets"][short] = {
            "rows": len(df), "columns": len(df.columns),
            "people": int(df[p+"PIDM"].nunique()),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "periods": counts(df[p+"TERM_CODE"], 100),
            "profile": profile(df),
        }
    hp, ht, hn = "UWVHIAC_PIDM", "UWVHIAC_TERM_CODE", "UWVHIAC_NRC"
    mp, mt = "UWVMAHI_PIDM", "UWVMAHI_TERM_CODE"
    report["quality"]["history_grains"] = [
        key_summary(h, keys) for keys in [
            [hp, ht, hn], [hp, ht, "UWVHIAC_SUBJ_CODE", "UWVHIAC_CRSE_NUMB"],
            [ht, hn], [ht, hn, "UWVHIAC_CAMP_CODE"]]]
    report["quality"]["enrollment_grains"] = [
        key_summary(m, keys) for keys in [
            [mp, mt], [mp, mt, "UWVMAHI_PROGRAM"],
            [mp, mt, "UWVMAHI_PROGRAM", "UWVMAHI_SEQNO"],
            [mp, mt, "UWVMAHI_KEY_SEQNO", "UWVMAHI_SECUENCIA_PLAN",
             "UWVMAHI_SEQNO"]]]
    hkeys = h[[hp, ht]].drop_duplicates().rename(columns={hp: "pid", ht: "term"})
    mkeys = m[[mp, mt]].drop_duplicates().rename(columns={mp: "pid", mt: "term"})
    joined = hkeys.merge(mkeys, on=["pid", "term"], how="outer", indicator=True)
    report["joins"]["person_period"] = counts(joined["_merge"])
    report["joins"]["people_in_both"] = len(set(h[hp].dropna()) & set(m[mp].dropna()))
    mc = m.groupby([mp, mt]).size().rename("m_rows").reset_index()
    hc = h.groupby([hp, ht]).size().rename("h_rows").reset_index()
    expansion = hc.merge(mc, left_on=[hp, ht], right_on=[mp, mt], how="left")
    report["joins"]["naive_join_matched_history_rows"] = int(
        expansion.loc[expansion.m_rows.notna(), "h_rows"].sum())
    report["joins"]["naive_inner_join_output_rows"] = int(
        (expansion.h_rows * expansion.m_rows.fillna(0)).sum())
    report["joins"]["history_rows_no_enrollment"] = int(
        expansion.loc[expansion.m_rows.isna(), "h_rows"].sum())
    for label, cols in {
        "faculty_codes": ["UWVMAHI_COLL_CODE", "UWVMAHI_FACULTAD"],
        "campus_codes": ["UWVMAHI_CAMP_CODE", "UWVMAHI_CAMPUS"],
        "period_codes": [mt, "UWVMAHI_PERIODO", "UWVMAHI_ACYR_CODE"],
        "student_status": ["UWVMAHI_STST_CODE", "UWVMAHI_STATUS_ALUMNO"],
        "plan_status": ["UWVMAHI_STSP_CODE", "UWVMAHI_STATUS_PLAN_ESTUDIO"],
    }.items():
        report["quality"][label] = m.groupby(cols, dropna=False).agg(
            rows=(mp, "size"), people=(mp, "nunique")).reset_index().to_dict("records")
    # Identify FAN by the explicit faculty descriptor, not career-name guesses.
    mask = m["UWVMAHI_FACULTAD"].astype(str).str.contains(
        "ADMINISTRACI|NEGOCIO", case=False, regex=True, na=False)
    fm = m.loc[mask].copy()
    fkeys = fm[[mp, mt]].drop_duplicates()
    fh = h.merge(fkeys, left_on=[hp, ht], right_on=[mp, mt], how="inner",
                 validate="many_to_one").drop(columns=[mp, mt])
    report["fan"] = {
        "faculty_values": counts(fm["UWVMAHI_FACULTAD"]),
        "enrollment_rows": len(fm), "people": int(fm[mp].nunique()),
        "person_periods": len(fkeys), "history_rows": len(fh),
        "history_people": int(fh[hp].nunique()),
        "periods": counts(fm[mt]), "history_periods": counts(fh[ht]),
        "careers": fm.groupby(["UWVMAHI_MAJR_CODE", "UWVMAHI_CARRERA"],
                             dropna=False).agg(rows=(mp, "size"), people=(mp, "nunique")
                             ).reset_index().to_dict("records"),
        "campuses": fm.groupby(["UWVMAHI_CAMP_CODE", "UWVMAHI_CAMPUS"],
                              dropna=False).agg(rows=(mp, "size"), people=(mp, "nunique")
                              ).reset_index().to_dict("records"),
        "enrollment_grain": key_summary(fm, [mp, mt]),
        "history_grain": key_summary(fh, [hp, ht, hn]),
    }
    # Actual distribution of grades/modes; no rule that nonnumeric means failure.
    for label, df in [("all", h), ("fan", fh)]:
        grade = numeric(df["UWVHIAC_GRDE_CODE_FINAL"])
        numeric_mask = grade.notna()
        conventional = grade.between(1, 7)
        report["quality"][f"grades_{label}"] = {
            "total": len(df), "numeric": int(numeric_mask.sum()),
            "missing": int(df["UWVHIAC_GRDE_CODE_FINAL"].isna().sum()),
            "numeric_outside_1_7": int((numeric_mask & ~conventional).sum()),
            "numeric_in_1_7": int(conventional.sum()),
            "below_4_in_1_7": int((conventional & grade.lt(4)).sum()),
            "distribution": counts(df["UWVHIAC_GRDE_CODE_FINAL"], 150),
            "mode_grade": df.groupby(
                ["UWVHIAC_MODO_CALIF", "UWVHIAC_GRDE_CODE_FINAL"], dropna=False
            ).size().rename("rows").reset_index().to_dict("records"),
        }
    for label, df, pid, term in [("all", m, mp, mt), ("fan", fm, mp, mt)]:
        year = df[term].astype(str).str[:4]
        byyear = {y: set(df.loc[year.eq(y), pid].dropna()) for y in sorted(year.unique())}
        report["temporal"][label] = {
            "people_by_year": {y: len(ids) for y, ids in byyear.items()},
            "overlaps": [
                {"from": a, "to": b, "both": len(byyear[a] & byyear[b]),
                 "only_from": len(byyear[a] - byyear[b]),
                 "only_to": len(byyear[b] - byyear[a])}
                for a, b in zip(sorted(byyear), sorted(byyear)[1:])],
        }
    # Workbook structure and presence of formulas without emitting cell values.
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    prior_report = OUT / "perfil_completo.json"
    previous = json.loads(prior_report.read_text(encoding="utf-8")) if prior_report.exists() else {}
    if previous.get("sha256") == sha and previous.get("workbook"):
        report["workbook"] = previous["workbook"]
    else:
        report["workbook"] = scan_workbook()
    final_sha = hashlib.file_digest(SOURCE.open("rb"), "sha256").hexdigest()
    assert final_sha == sha, "La fuente cambió durante la lectura"
    write_json("perfil_completo.json", report)
    sections = []
    for short, data in report["sheets"].items():
        rows = []
        for column, p in data["profile"].items():
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                        [column, p["dtype"], p["missing"], round(p["missing_pct"], 2),
                         p["distinct"], json.dumps(p.get("values", p.get("numeric", {})),
                                                  ensure_ascii=False)]) + "</tr>")
        sections.append(f"<h2>{short}: {data['rows']:,} filas</h2><table><tr>"
                        "<th>Campo</th><th>Tipo leído</th><th>Ausentes</th><th>%</th>"
                        "<th>Distintos</th><th>Distribución / rango</th></tr>"
                        + "".join(rows) + "</table>")
    (OUT / "diccionario_calidad.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Auditoría Excel UA</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#243044}table{border-collapse:collapse}'
        'td,th{border:1px solid #ccd3dc;padding:8px;vertical-align:top}td:last-child{max-width:650px;'
        'overflow-wrap:anywhere}th{background:#eef2f7}</style><h1>Inventario y calidad de campos</h1>'
        '<p>20 de octubre de 2026 · Uso interno · Perfil completo, sin identificadores individuales.'
        ' Los conteos son filas, salvo indicación expresa. Las tablas pueden tener duplicados.</p>'
        + "".join(sections) + "</html>", encoding="utf-8")
    print(json.dumps({"source_sha256": sha, "output": str(OUT),
                      "seconds": round(time.time()-t, 1),
                      "summary": {s: {k:v for k,v in d.items() if k != "profile"}
                                  for s,d in report["sheets"].items()},
                      "fan": report["fan"], "joins": report["joins"]},
                     ensure_ascii=False, default=str), flush=True)


def scan_workbook():
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(SOURCE) as z:
        workbook = {"external_links": [n for n in z.namelist()
                                               if n.startswith("xl/externalLinks/")],
                              "worksheets": {}}
        for file in z.namelist():
            if file.startswith("xl/worksheets/sheet") and file.endswith(".xml"):
                formulas = errors = 0
                with z.open(file) as stream:
                    for _, element in ET.iterparse(stream, events=("end",)):
                        if element.tag == "{"+ns["m"]+"}f":
                            formulas += 1
                        if element.tag == "{"+ns["m"]+"}c":
                            errors += element.attrib.get("t") == "e"
                        element.clear()
                workbook["worksheets"][file] = {
                    "formulas": formulas, "error_cells": errors}
    return workbook


if __name__ == "__main__":
    main()
