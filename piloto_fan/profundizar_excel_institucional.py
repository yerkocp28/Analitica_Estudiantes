"""Comprobaciones de uniones, resultados y cobertura sobre la caché local auditada."""
from explorar_excel_institucional import OUT, write_json, counts, key_summary, numeric
import pandas as pd
import json

h = pd.read_parquet(OUT / "h_local.parquet")
m = pd.read_parquet(OUT / "m_local.parquet")
hp, ht, hn = "UWVHIAC_PIDM", "UWVHIAC_TERM_CODE", "UWVHIAC_NRC"
mp, mt = "UWVMAHI_PIDM", "UWVMAHI_TERM_CODE"
hk, mk = [hp, ht, hn], [mp, mt]
hd, md = h.drop_duplicates(), m.drop_duplicates()
fan = md["UWVMAHI_FACULTAD"].astype(str).str.contains(
    "ADMINISTRACI|NEGOCIO", case=False, na=False)
fm = md.loc[fan].copy()
fkeys = fm[mk].drop_duplicates()
fh = hd.merge(fkeys, left_on=[hp, ht], right_on=mk, validate="many_to_one").drop(columns=mk)
r = {"temporal_diagnostic_warning": "Orden numérico del código, no disponibilidad temporal. "
     "No usar como corte de predicción sin calendario por régimen y fechas de cierre.",
     "deduplicated": {"history": len(hd), "enrollment": len(md),
                     "history_key": key_summary(hd, hk), "enrollment_key": key_summary(md, mk)},
     "history_key_conflicting_fields": {}, "enrollment_multirecord_fields": {},
     "fan": {}, "periods": {}, "prospective": {}}
for label, df, keys in [("history_key_conflicting_fields", hd, hk),
                         ("enrollment_multirecord_fields", md, mk)]:
    dup = df.loc[df.duplicated(keys, keep=False)]
    varied = dup.groupby(keys).nunique(dropna=False)
    r[label] = {c: int(varied[c].gt(1).sum()) for c in varied if varied[c].gt(1).any()}
faculties = md.groupby(mk)["UWVMAHI_FACULTAD"].nunique(dropna=False)
fc = fkeys.merge(faculties.rename("n_faculties").reset_index(), on=mk)
r["fan"]["person_periods_multiple_faculties"] = int(fc.n_faculties.gt(1).sum())
r["fan"]["deduplicated_history_rows"] = len(fh)
r["fan"]["history_unresolved_keys"] = key_summary(fh, hk)
r["fan"]["deduplicated_enrollment_rows"] = len(fm)
r["fan"]["faculty_career_campus"] = fm.groupby(
    ["UWVMAHI_CARRERA", "UWVMAHI_CAMPUS"], dropna=False).agg(
        rows=(mp, "size"), people=(mp, "nunique")).reset_index().to_dict("records")
r["fan"]["statuses_by_period"] = fm.groupby(
    [mt, "UWVMAHI_STATUS_ALUMNO", "UWVMAHI_STATUS_PLAN_ESTUDIO"],
    dropna=False).agg(rows=(mp, "size"), people=(mp, "nunique")).reset_index().to_dict("records")
for label, df in [("all", hd), ("fan", fh)]:
    g = numeric(df.UWVHIAC_GRDE_CODE_FINAL)
    credits = pd.to_numeric(df.UWVHIAC_CREDIT_HOURS, errors="coerce")
    d = df.assign(numeric_grade=g, valid_grade=g.between(1, 7),
                  below4=g.between(1, 7) & g.lt(4), numeric_credits=credits)
    r["periods"][label] = d.groupby(ht, dropna=False).agg(
        rows=(hp, "size"), people=(hp, "nunique"), numeric_valid=("valid_grade", "sum"),
        below4=("below4", "sum")).reset_index().to_dict("records")
    r[f"credits_{label}"] = {
        "missing": int(credits.isna().sum()), "zero": int(credits.eq(0).sum()),
        "negative": int(credits.lt(0).sum()), "min": float(credits.min()),
        "max": float(credits.max()), "distribution": counts(credits, 60)}
    r[f"grades_by_mode_{label}"] = d.groupby("UWVHIAC_MODO_CALIF", dropna=False).agg(
        rows=(hp, "size"), numeric_valid=("valid_grade", "sum"), below4=("below4", "sum")
    ).reset_index().to_dict("records")
    r[f"repeated_{label}"] = counts(df.UWVHIAC_REPETIDO)
    r[f"popular_courses_{label}"] = d.groupby(
        ["UWVHIAC_SUBJ_CODE", "UWVHIAC_CRSE_NUMB", "UWVHIAC_CRSE_TITLE"],
        dropna=False).agg(rows=(hp, "size"), people=(hp, "nunique"),
                         numeric_valid=("valid_grade", "sum"), below4=("below4", "sum")
                         ).sort_values("rows", ascending=False).head(25).reset_index().to_dict("records")
# Same NRC is usually reused across terms. Test the actual section grain.
for key in [[ht, hn], [ht, hn, "UWVHIAC_CAMP_CODE"]]:
    variability = hd.groupby(key)[["UWVHIAC_SUBJ_CODE", "UWVHIAC_CRSE_NUMB",
                                   "UWVHIAC_CREDIT_HOURS", "UWVHIAC_CAMP_CODE",
                                   "UWVHIAC_CRSE_TITLE"]].nunique(dropna=False)
    r["section_"+str(len(key))] = {
        c: int(variability[c].gt(1).sum()) for c in variability}
# Diagnose FAN identities and enrollment coverage term by term.
for term in sorted(fm[mt].dropna().unique()):
    current = set(fm.loc[fm[mt].eq(term), mp])
    historical = set(hd.loc[hd[ht].lt(term), hp])
    current_h = set(hd.loc[hd[ht].eq(term), hp])
    r["prospective"][str(term)] = {
        "fan_enrolled_people": len(current),
        "with_lower_term_code_history_not_availability": len(current & historical),
        "with_same_term_academic_history": len(current & current_h),
        "without_lower_term_code_history": len(current - historical),
    }
admit = pd.to_numeric(fm.UWVMAHI_TERM_CODE_ADMIT, errors="coerce")
r["fan"]["admission_periods"] = counts(fm.UWVMAHI_TERM_CODE_ADMIT, 100)
r["fan"]["admission_after_observed_term_rows"] = int(admit.gt(fm[mt]).sum())
r["fan"]["same_person_multiple_careers"] = int(
    fm.groupby(mp).UWVMAHI_CARRERA.nunique().gt(1).sum())
r["fan"]["same_person_multiple_campuses"] = int(
    fm.groupby(mp).UWVMAHI_CAMPUS.nunique().gt(1).sum())
r["fan"]["row_priority"] = counts(fm.UWVMAHI_PRIORIDAD)
r["fan"]["row_sequence"] = counts(fm.UWVMAHI_SEQNO)
r["fan"]["row_plan_sequence"] = counts(fm.UWVMAHI_SECUENCIA_PLAN)
r["fan"]["student_types"] = counts(fm.UWVMAHI_TIPO_ALUMNO)
write_json("diagnostico_uniones_y_cobertura.json", r)
print(json.dumps({k:v for k,v in r.items() if "popular_courses" not in k},
                 ensure_ascii=True, indent=2))
