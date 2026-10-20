"""Modelos retrospectivos por semestre, con cortes explícitos y sin notas futuras."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits

from .institucional import HOME, SOURCE, digest, load, read_config
from .models import ALGORITHMS, design, estimator, metrics, bootstrap_comparison

NUMERIC = ["observed_results", "numeric_mean", "numeric_sd", "numeric_coverage", "below4_fraction"]
CATEGORICAL = ["career_code", "campus_code", "schedule"]
SPEC = {"numeric": NUMERIC, "categorical": CATEGORICAL}
TRANSITIONS = [(202410, 202420), (202420, 202510), (202510, 202520)]
CFG = {"seed": 42, "bootstrap_repetitions": 300, "minimum_class": 10,
       "minimum_fit": 150, "minimum_evaluation": 100}
TITLES = {
    "academico": "Al menos una nota numérica bajo 4 en el siguiente semestre",
    "matricula": "No observar registro de matrícula UA en el siguiente semestre ordinario",
}
LIMITATIONS = [
    "Experimento retrospectivo, no alerta individual ni resultado validado para uso operativo.",
    "Se supone que resultados de cada semestre estaban disponibles a su cierre, antes del siguiente. "
    "No se conocen fechas reales de publicación; este supuesto necesita confirmación institucional.",
    "Solo hay tres transiciones y dos años. Los intervalos no cubren deriva temporal ni errores de extracción.",
    "La población tiene historia numérica previa, un único plan y alcance FAN provisional. No representa nuevos "
    "estudiantes sin historia ni modalidades trimestrales/online.",
    "Las mismas personas pueden aparecer en semestres sucesivos: evaluación longitudinal, no independencia entre "
    "personas de entrenamiento y prueba. Los identificadores nunca son predictores.",
    "Estados del plan/alumno, secuencias, CREDIT_HOURS, códigos administrativos, PIDM/MRUN y notas futuras "
    "se excluyen de las variables. Los atributos del plan se toman del período de origen del extracto.",
    "Ausencia de matrícula en el semestre ordinario no equivale a deserción: incluye posibles egresos, "
    "cambios de régimen y otras salidas; se consideran todos los estados de matrícula del extracto.",
    "Una nota numérica bajo 4 no equivale a toda la reprobación institucional. Los códigos no se interpretan "
    "como aprobado/reprobado. Resultados negativos solo con historia observada completamente numérica.",
    "No se entrenan sobre datos sintéticos, no se vincula PIDM con MRUN y no se generan prioridades para 2026.",
]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def snapshots(frames):
    d = frames["students"].copy()
    valid = (d.fan_proposed.fillna(False) & d.plans.eq(1) & d.regime.eq("Semestral")
             & d.numeric_results.gt(0) & d.term.isin([202410, 202420, 202510, 202520]))
    d = d.loc[valid].copy()
    d["below4_fraction"] = d.numeric_below4_count / d.numeric_results
    return d[["student_key", "term", "career", "campus"] + NUMERIC + CATEGORICAL]


def datasets(frames):
    """No requiere matrícula ni resultado futuro para entrar a la población base."""
    base = snapshots(frames)
    courses = frames["courses"]
    future = courses.groupby(["student_key", "term"]).agg(
        observed=("nrc", "size"), numeric=("grade_numeric", "count"),
        minimum=("grade_numeric", "min")).reset_index()
    presence = set(map(tuple, frames["enrollment"][["student_key", "term"]].drop_duplicates().to_numpy()))
    observed_terms = set(frames["enrollment"].term)
    academic_terms = set(courses.term)
    rows = []
    for origin, target in TRANSITIONS:
        d = base.loc[base.term.eq(origin)].rename(columns={"term": "source_term"}).copy()
        d["outcome_term"] = target
        f = future.loc[future.term.eq(target)].drop(columns="term")
        d = d.merge(f, on="student_key", how="left", validate="one_to_one")
        d["academic_target"] = np.nan
        positive = d.minimum.lt(4)
        negative = d.observed.gt(0) & d.observed.eq(d.numeric) & d.minimum.ge(4)
        if target in academic_terms:
            d.loc[positive, "academic_target"] = 1.
            d.loc[negative, "academic_target"] = 0.
        d["enrollment_target"] = [float((pid, target) not in presence) if target in observed_terms else np.nan
                                  for pid in d.student_key]
        # Keep outcome observability for auditing, never for model features.
        rows.append(d)
    all_rows = pd.concat(rows, ignore_index=True)
    audit = []
    for (origin, target), g in all_rows.groupby(["source_term", "outcome_term"]):
        audit.append({"source_term": int(origin), "outcome_term": int(target),
                      "base_people": len(g), "academic_known": int(g.academic_target.notna().sum()),
                      "academic_unknown": int(g.academic_target.isna().sum()),
                      "enrollment_known": int(g.enrollment_target.notna().sum())})
    result = {}
    for case, column in [("academico", "academic_target"), ("matricula", "enrollment_target")]:
        # Output labels and explicit feature list only; future counts cannot leak to design().
        result[case] = all_rows[["student_key", "source_term", "outcome_term", "career", "campus"]
                                 + NUMERIC + CATEGORICAL + [column]].rename(columns={column: "target"})
    return result, base, audit


def historical(train, other):
    rates = train.groupby("career_code").target.agg(["sum", "count"])
    smooth = (rates["sum"] + 30*train.target.mean()) / (rates["count"]+30)
    return other.career_code.map(smooth).fillna(train.target.mean()).to_numpy(float)


def rule(train, other, case):
    return (other.below4_fraction.to_numpy(float) if case == "academico" else historical(train, other))


def partition(data):
    observed = data.dropna(subset=["target"])
    parts = [observed.loc[observed.source_term.eq(t)].copy() for t in [202410, 202420, 202510]]
    for i, part in enumerate(parts):
        required = CFG["minimum_fit"] if i == 0 else CFG["minimum_evaluation"]
        vc = part.target.value_counts()
        if len(part) < required or len(vc) != 2 or vc.min() < CFG["minimum_class"]:
            raise ValueError(f"Partición {i+1}: casos o clases insuficientes.")
        if part.student_key.duplicated().any():
            raise ValueError("Más de una observación por persona en un semestre.")
    return parts


def fit_case(data, case):
    result = {"title": TITLES[case], "status": "experimental",
              "features": SPEC, "unknown_labels": int(data.target.isna().sum()),
              "rule": ("Fracción previa de notas numéricas bajo 4" if case == "academico"
                       else "Tasa histórica por carrera, suavizada con 30 casos")}
    try:
        train, validation, test = partition(data)
    except ValueError as exc:
        return {**result, "status": "insuficiente", "reason": str(exc)}, None, None
    result["split"] = {"train": [202410, 202420], "validation": [202420, 202510],
                       "test": [202510, 202520], "train_n": len(train), "validation_n": len(validation),
                       "test_n": len(test),
                       "test_people_seen_earlier": len(set(test.student_key) &
                                                     (set(train.student_key) | set(validation.student_key)))}
    candidates = []
    for name in ALGORITHMS:
        model = estimator(name, SPEC, CFG["seed"])
        model.fit(design(train, SPEC), train.target.astype(int))
        score = model.predict_proba(design(validation, SPEC))[:, 1]
        candidates.append({"algorithm": name, **metrics(validation.target, score, train.target.mean())})
    winner = max(candidates, key=lambda r: (r["capture_20"], -r["brier"], r["auc"] or 0))["algorithm"]
    final_train = pd.concat([train, validation], ignore_index=True)
    chosen = estimator(winner, SPEC, CFG["seed"])
    chosen.fit(design(final_train, SPEC), final_train.target.astype(int))
    score = chosen.predict_proba(design(test, SPEC))[:, 1]
    baseline = rule(final_train, test, case)
    prior = float(final_train.target.mean())
    # Predefined ablation, selected algorithm unchanged: quantify academic-data contribution.
    contextual_spec = {"numeric": [], "categorical": CATEGORICAL}
    contextual = estimator(winner, contextual_spec, CFG["seed"])
    contextual.fit(design(final_train, contextual_spec), final_train.target.astype(int))
    context_score = contextual.predict_proba(design(test, contextual_spec))[:, 1]
    ci = bootstrap_comparison(test.target, score, baseline, CFG)
    m = metrics(test.target, score, prior)
    result.update(
        winner=winner, validation_candidates=candidates, selection="Captura al 20% en validación; desempate Brier y AUC",
        final_train_n=len(final_train), reference_prevalence=prior, test=m,
        baseline=metrics(test.target, baseline, prior),
        historical_career=metrics(test.target, historical(final_train, test), prior),
        contextual_only=metrics(test.target, context_score, prior), uncertainty=ci,
        incremental_uncertainty=bootstrap_comparison(test.target, score, context_score, CFG),
        evidence_vs_rule=("Mejora de captura en esta prueba" if ci["capture_difference_vs_rule_ci"][0] > 0
                          else "Mejora no concluyente en esta prueba"),
    )
    result["validation_prevalence"] = float(validation.target.mean())
    result["selection_training_prevalence"] = float(train.target.mean())
    result["diagnostics"] = []
    if min(m["events"], m["n"]-m["events"]) < 30:
        result["diagnostics"].append("Menos de 30 eventos de una clase en la prueba: incertidumbre elevada.")
    if m["brier_skill"] is None or m["brier_skill"] <= 0:
        result["diagnostics"].append("El error de probabilidad no mejora la constante de entrenamiento.")
    if abs(m["mean_prediction"]-m["prevalence"]) >= .05:
        result["diagnostics"].append("Descalibración: diferencia de al menos 5 puntos entre predicción media y frecuencia observada.")
    if abs(result["validation_prevalence"]-m["prevalence"]) >= .05:
        result["diagnostics"].append("La frecuencia del resultado cambia al menos 5 puntos entre validación y prueba.")
    actual, predicted = calibration_curve(test.target.astype(int), score, n_bins=5, strategy="quantile")
    result["calibration"] = [{"predicted": float(p), "observed": float(y)} for p,y in zip(predicted, actual)]
    pred = test[["student_key", "source_term", "outcome_term", "career", "campus", "schedule", "target"]].copy()
    pred["risk_score"], pred["rule_score"], pred["contextual_score"] = score, baseline, context_score
    return result, {"model": chosen, "contextual_model": contextual, "spec": SPEC}, pred


def segment(base):
    train = base.loc[base.term.eq(202410)]
    if len(train) < 160:
        return {"status": "insuficiente", "reason": "Pocos casos de entrenamiento."}, None, None
    prep = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler())])
    a = prep.fit_transform(train[NUMERIC])
    all_x = prep.transform(base[NUMERIC])
    candidates, eligible = [], []
    for k in [2, 3, 4, 5]:
        options = {
            "K-means": KMeans(n_clusters=k, n_init=10, random_state=42),
            "Mezcla gaussiana": GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                                              reg_covar=1e-4, random_state=42),
            "K-means bisectriz": BisectingKMeans(n_clusters=k, random_state=42),
        }
        for name, model in options.items():
            labels = model.fit_predict(a)
            counts = pd.Series(labels).value_counts()
            sil = float(silhouette_score(a, labels, sample_size=min(1000, len(a)), random_state=42))
            admissible = (len(counts) == k and counts.min() >= 80
                          and counts.max()/len(train) <= .8 and getattr(model, "converged_", True))
            candidates.append({"algorithm": name, "k": k, "silhouette": sil,
                               "smallest": int(counts.min()), "largest_share": float(counts.max()/len(train)),
                               "admissible": bool(admissible)})
            if admissible:
                eligible.append((sil, -k, name, model))
    if not eligible:
        return {"status": "insuficiente", "candidates": candidates}, None, None
    sil, minus_k, name, model = max(eligible, key=lambda r: (r[0], r[1]))
    mapping = {int(label): i+1 for i,label in enumerate(pd.Series(model.predict(a)).value_counts().index)}
    assigned = base.copy()
    assigned["segment"] = [mapping[int(label)] for label in model.predict(all_x)]
    profiles = []
    for (term, label), g in assigned.groupby(["term", "segment"]):
        if len(g) >= 10:
            profiles.append({"term": int(term), "segment": int(label), "n": len(g),
                             **{c: float(g[c].mean()) for c in NUMERIC}})
    report = {"status": "descriptivo", "winner": name, "k": -minus_k, "silhouette": sil,
              "fit_term": 202410, "fit_n": len(train), "features": NUMERIC,
              "candidates": candidates, "profiles": profiles,
              "interpretation": "Perfiles de resultados observados; no niveles de riesgo ni necesidades diagnosticadas. "
              "Cobertura e imputación pueden influir. Carrera, campus y resultados futuros no definen los grupos."}
    return report, {"preprocess": prep, "model": model, "mapping": mapping}, assigned


def load_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        if Path(name).name != name or digest(folder/name) != expected:
            raise ValueError("Artefacto de modelos alterado: " + name)
    report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
    frames = {p.stem: pd.read_parquet(p) for p in folder.glob("*.parquet")}
    return report, frames


def build():
    base = HOME / "artifacts/institucional"
    source_id = json.loads((base/"latest.json").read_text())["run_id"]
    if Path(source_id).name != source_id or "/" in source_id or "\\" in source_id:
        raise ValueError("Ejecución fuente inválida.")
    folder = base/source_id
    frames, _, source_manifest = load(folder)
    if SOURCE.exists() and digest(SOURCE) != source_manifest["source_sha256"]:
        raise ValueError("El Excel cambió: regenerar primero la integración institucional.")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    dest = HOME/"artifacts/institucional_modelos"
    out = dest/run_id
    out.mkdir(parents=True, exist_ok=False)
    data, base_population, audit = datasets(frames)
    report = {"run_id": run_id, "created_utc": now.isoformat(), "source_run_id": source_id,
              "source_sha256": source_manifest["source_sha256"], "config": CFG,
              "limitations": LIMITATIONS, "coverage": audit, "cases": {},
              "pending": read_config()["preguntas"]}
    with threadpool_limits(limits=1):
        for case, dataset in data.items():
            print("Entrenando " + case, flush=True)
            result, model, predictions = fit_case(dataset, case)
            report["cases"][case] = result
            write(out/(case+"_model_card.json"), {**result, "limitations": LIMITATIONS,
                                                "source_run_id": source_id, "run_id": run_id})
            if model is not None:
                joblib.dump(model, out/(case+"_model.joblib"))
                predictions["run_id"] = run_id
                predictions.to_parquet(out/(case+"_predictions.parquet"), index=False)
            dataset.to_parquet(out/(case+"_dataset.parquet"), index=False)
            print(json.dumps(clean({k:result.get(k) for k in ["winner", "status", "test", "baseline"]})), flush=True)
        print("Comparando segmentaciones", flush=True)
        report["segments"], model, assignments = segment(base_population)
        if model is not None:
            joblib.dump(model, out/"segmentacion_model.joblib")
            assignments.to_parquet(out/"segmentacion_assignments.parquet", index=False)
    write(out/"report.json", report)
    write(out/"feature_registry.json", {"predictive": SPEC, "segmentation": NUMERIC,
                                       "timing": "Resultados del semestre de origen, supuesto cerrado; nunca del objetivo.",
                                       "excluded": ["student_key", "target", "outcome_term", "plan_status", "student_status",
                                                    "workload_raw", "admission_type"]})
    write(out/"manifest.json", {
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "source_run_id": source_id, "source_manifest_sha256": digest(folder/"manifest.json"),
        "source_artifacts_sha256": source_manifest["artifacts_sha256"],
        "source_sha256": source_manifest["source_sha256"],
        "code_sha256": {name: digest(HOME/name) for name in ["modelos_institucionales.py", "models.py"]},
        "artifacts_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    })
    load_run(out)
    pointer = dest/(run_id+".tmp")
    write(pointer, {"run_id": run_id})
    os.replace(pointer, dest/"latest.json")
    print("Modelos guardados: " + str(out), flush=True)
    return out


if __name__ == "__main__":
    build()
