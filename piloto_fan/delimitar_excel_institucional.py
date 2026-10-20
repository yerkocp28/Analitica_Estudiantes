"""Cuantifica un alcance FAN provisional; no reemplaza la ratificación institucional."""
from explorar_excel_institucional import OUT, write_json, numeric
import pandas as pd
import json

h = pd.read_parquet(OUT / "h_local.parquet")
m = pd.read_parquet(OUT / "m_local.parquet")
mp, mt = "UWVMAHI_PIDM", "UWVMAHI_TERM_CODE"
hp, ht = "UWVHIAC_PIDM", "UWVHIAC_TERM_CODE"
fan = m.UWVMAHI_FACULTAD.str.contains("Negocios", case=False, na=False)
fm = m.loc[fan].copy()
core_careers = [
    "Ingeniería Comercial", "Aud. e Ing En Control de Gest.",
    "Ing. en Control de Gestión (D)", "Ing. en Control de Gestión (T)",
    "Ingeniería en Administración", "Contador Público Auditor"]
core = fm.loc[fm.UWVMAHI_NIVEL.eq("Pregrado") &
              fm.UWVMAHI_CARRERA.isin(core_careers)].copy()
unique_m = m.groupby([mp, mt]).size().eq(1).rename("single_plan")
scopes = {"FAN_etiquetada": fm, "FAN_propuesta_presencial": core}
r = {"scope_not_approved": True, "scopes": {}, "join_quality": {}}
for label, d in scopes.items():
    fk = d[[mp, mt]].drop_duplicates()
    fh = h.merge(fk, left_on=[hp, ht], right_on=[mp, mt], validate="many_to_one")
    valid_grade = numeric(fh.UWVHIAC_GRDE_CODE_FINAL)
    gvalid = valid_grade.between(1, 7)
    per = []
    for term, part in d.groupby(mt):
        ids = set(part[mp])
        term_hist = fh.loc[fh[ht].eq(term)]
        per.append({"period": int(term), "rows": len(part), "people": len(ids),
                    "people_with_history": int(term_hist[hp].nunique()),
                    "history_rows": len(term_hist),
                    "active_plan_people": int(part.loc[
                        part.UWVMAHI_STATUS_PLAN_ESTUDIO.eq("Activo"), mp].nunique())})
    r["scopes"][label] = {
        "rows": len(d), "people": int(d[mp].nunique()),
        "person_periods": len(fk), "history_rows": len(fh),
        "people_with_history": int(fh[hp].nunique()),
        "numeric_grade_rows": int(gvalid.sum()),
        "numeric_below4": int((gvalid & valid_grade.lt(4)).sum()),
        "coded_grades": fh.loc[~gvalid, "UWVHIAC_GRDE_CODE_FINAL"].value_counts().to_dict(),
        "by_period": per,
        "by_career": d.groupby("UWVMAHI_CARRERA").agg(
            rows=(mp, "size"), people=(mp, "nunique")).reset_index().to_dict("records"),
        "by_campus": d.groupby("UWVMAHI_CAMPUS").agg(
            rows=(mp, "size"), people=(mp, "nunique")).reset_index().to_dict("records"),
    }
    # Safe unambiguous analytical subset: exactly one record in the entire
    # institution per person-period. Remaining records require plan attribution.
    safe = d.merge(unique_m.reset_index(), on=[mp, mt], validate="many_to_one")
    safe = safe.loc[safe.single_plan]
    hs = h.merge(safe, left_on=[hp, ht], right_on=[mp, mt], validate="many_to_one")
    r["scopes"][label]["unambiguous_rows"] = len(safe)
    r["scopes"][label]["unambiguous_history_rows"] = len(hs)
    r["scopes"][label]["campus_mismatch_history_rows"] = int(
        hs.UWVHIAC_CAMP_CODE.ne(hs.UWVMAHI_CAMP_CODE).sum())
    # Provisional descriptive comparison within ordinary semesters only.
    hs = hs.loc[hs[ht].isin([202410, 202420, 202510, 202520])].copy()
    hs["grade"] = numeric(hs.UWVHIAC_GRDE_CODE_FINAL)
    hs["numeric_grade"] = hs.grade.between(1, 7)
    hs["below4"] = hs.numeric_grade & hs.grade.lt(4)
    r["scopes"][label]["semester_grade_summary_unambiguous"] = hs.groupby(
        ht).agg(rows=(hp, "size"), people=(hp, "nunique"),
                numeric_grades=("numeric_grade", "sum"), below4=("below4", "sum"),
                mean_grade=("grade", "mean")).reset_index().to_dict("records")
    r["scopes"][label]["admission_missing_rows"] = int(d.UWVMAHI_ADMT_CODE.isna().sum())
# Canonical ownership of sections must include level and subject, not just NRC.
for keys in [[ht, "UWVHIAC_NRC", "UWVHIAC_LEVL_CODE"],
             [ht, "UWVHIAC_NRC", "UWVHIAC_LEVL_CODE", "UWVHIAC_CAMP_CODE"]]:
    n = h.groupby(keys, dropna=False)[["UWVHIAC_SUBJ_CODE", "UWVHIAC_CRSE_NUMB",
                                      "UWVHIAC_CREDIT_HOURS", "UWVHIAC_CRSE_TITLE"]].nunique()
    r["join_quality"]["section_key_"+str(len(keys))] = {
        c: int(n[c].gt(1).sum()) for c in n}
write_json("alcance_fan_propuesto.json", r)
print(json.dumps(r, ensure_ascii=True, indent=2))
