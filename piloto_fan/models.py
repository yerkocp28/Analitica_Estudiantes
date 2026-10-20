"""Comparación temporal: selección con años anteriores al test y etiquetas maduras."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = {
    "continuidad": {
        "numeric": ["percentil_admision", "percentil_nem", "percentil_ranking", "duracion_nominal_anios"],
        "categorical": ["nomb_carrera", "nomb_sede", "jornada"],
        "target": "No observar matrícula en la UA al año siguiente",
        "timing": "Características registradas en la cohorte de ingreso; sin conducta universitaria semanal.",
    },
    "titulacion": {
        "numeric": ["duracion_nominal_anios", "years_enrolled", "years_without_ua",
                    "career_codes_observed", "sites_observed", "schedules_observed"],
        "categorical": ["nomb_carrera", "nomb_sede", "jornada"],
        "target": "No observar un título de pregrado UA hasta duración nominal más holgura",
        "timing": "Historia UA entre el ingreso y el año de ingreso + 3, inclusive; perfil de carrera/sede al ingreso.",
    },
}
ALGORITHMS = ["Regresión logística", "Random forest", "Gradient boosting"]
CAPACITIES = [0.05, 0.10, 0.20, 0.30, 0.50]


def design(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    x = d[spec["numeric"] + spec["categorical"]].copy()
    for col in spec["numeric"]:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for col in spec["categorical"]:
        x[col] = x[col].astype("string").fillna("Sin información").astype(str)
    return x


def estimator(name: str, spec: dict, seed: int):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                 keep_empty_features=True)),
                        ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("numeric", numeric, spec["numeric"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                       sparse_output=False), spec["categorical"]),
    ])
    algorithms = {
        "Regresión logística": LogisticRegression(C=0.3, max_iter=2000, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=15,
                                                 max_depth=8, n_jobs=1, random_state=seed),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                               l2_regularization=5, random_state=seed),
    }
    return Pipeline([("preprocess", prep), ("model", algorithms[name])])


def capture(y, p, capacity=0.2) -> float:
    """Desempate fraccional: probabilidades constantes no inventan un orden útil."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y) or y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y) * capacity)))
    threshold = np.sort(p)[-k]
    high, tied = p > threshold, p == threshold
    expected = y[high].sum() + (k - high.sum()) * y[tied].mean()
    return float(expected / y.sum())


def metrics(y, p, reference_prevalence: float) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    brier = float(brier_score_loss(y, p))
    baseline = float(brier_score_loss(y, np.full(len(y), reference_prevalence)))
    return {"n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "brier": brier, "brier_skill": 1 - brier / baseline if baseline else None,
            "capture_20": capture(y, p), "mean_prediction": float(p.mean())}


def historical_rule(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Referencia: tasas por carrera con contracción equivalente a 30 casos."""
    rates = train.groupby("nomb_carrera", dropna=False).target.agg(["sum", "count"])
    smoothed = (rates["sum"] + 30 * train.target.mean()) / (rates["count"] + 30)
    return other.nomb_carrera.map(smoothed).fillna(train.target.mean()).to_numpy(float)


def enough(d: pd.DataFrame, cfg: dict, fit=False) -> bool:
    counts = d.target.value_counts()
    return (len(d) >= cfg["minimum_fit" if fit else "minimum_evaluation"]
            and len(counts) == 2 and int(counts.min()) >= cfg["minimum_class"])


def temporal_split(data: pd.DataFrame, cfg: dict):
    """Una fecha test con cobertura de resultado completa, y validación ya resuelta.

    No basta con que la cohorte train sea anterior: su resultado, incluida
    la holgura de publicación asumida, debe conocerse en la fecha simulada.
    """
    observed = data.dropna(subset=["target"]).copy()
    complete = data.groupby("score_year").target.apply(lambda s: s.notna().all())
    for test_year in sorted(complete[complete].index, reverse=True):
        test = observed[observed.score_year.eq(test_year)]
        if not enough(test, cfg):
            continue
        for validation_year in sorted(complete[complete].index, reverse=True):
            if validation_year >= test_year:
                continue
            val = observed[observed.score_year.eq(validation_year)]
            if not enough(val, cfg) or val.label_ready_year.max() > test_year:
                continue
            train = observed[(observed.score_year < validation_year)
                             & (observed.label_ready_year <= validation_year)]
            train = train[~train.mrun.isin(set(val.mrun) | set(test.mrun))]
            if enough(train, cfg, fit=True):
                final_train = observed[(observed.score_year < test_year)
                                       & (observed.label_ready_year <= test_year)]
                final_train = final_train[~final_train.mrun.isin(test.mrun)]
                return train, val, final_train, test
    raise ValueError("No hay partición temporal con etiquetas maduras, ambas clases y tamaño suficiente.")


def bootstrap_comparison(y, p, rule, cfg):
    rng = np.random.default_rng(cfg["seed"])
    y, p, rule = np.asarray(y), np.asarray(p), np.asarray(rule)
    captures, differences = [], []
    for _ in range(cfg["bootstrap_repetitions"]):
        idx = rng.integers(0, len(y), len(y))
        c = capture(y[idx], p[idx])
        if np.isfinite(c):
            captures.append(c)
            differences.append(c - capture(y[idx], rule[idx]))
    return {"capture_20_ci": np.quantile(captures, [.025, .975]).tolist(),
            "capture_difference_vs_rule_ci": np.quantile(differences, [.025, .975]).tolist(),
            "interpretation": "Bootstrap por persona del test; no cubre deriva entre años ni errores de registro."}


def evaluate_case(data: pd.DataFrame, case: str, cfg: dict):
    spec = FEATURES[case]
    base = {"case": case, "features": spec, "eligible": len(data),
            "censored": int(data.target.isna().sum()) if len(data) else 0,
            "intended_use": "Evaluación retrospectiva y discusión con direcciones de carrera.",
            "prohibited_use": "Contactar o sancionar estudiantes, diagnosticar causas o afirmar impacto de intervenciones.",
            "limitations": ["Población del área y plan definidos, pendiente de validar pertenencia a la facultad.",
                            "Una cohorte final de test no demuestra estabilidad en cohortes futuras.",
                            "Las probabilidades requieren revisar calibración y deriva antes de uso institucional.",
                            "Disponibilidad temporal reconstruida con demora anual asumida; no son fechas de publicación verificadas."]}
    if data.empty:
        return {**base, "status": "insuficiente", "reason": "Sin población elegible."}, None, None
    try:
        train, val, final_train, test = temporal_split(data, cfg)
    except ValueError as exc:
        return {**base, "status": "insuficiente", "reason": str(exc)}, None, None
    xtr, xv = design(train, spec), design(val, spec)
    candidates, fitted = [], {}
    for name in ALGORITHMS:
        model = estimator(name, spec, cfg["seed"])
        model.fit(xtr, train.target.astype(int))
        p = model.predict_proba(xv)[:, 1]
        candidates.append({"algorithm": name, **metrics(val.target, p, train.target.mean())})
        fitted[name] = model
    # Se decide exclusivamente con validación, NUNCA se elige por el test final.
    winner = sorted(candidates, key=lambda r: (-r["capture_20"], r["brier"], -r["auc"]))[0]["algorithm"]
    importance = permutation_importance(fitted[winner], xv, val.target.astype(int),
                                       n_repeats=5, random_state=cfg["seed"], scoring="roc_auc", n_jobs=1)
    model = estimator(winner, spec, cfg["seed"])
    model.fit(design(final_train, spec), final_train.target.astype(int))
    p = model.predict_proba(design(test, spec))[:, 1]
    rule = historical_rule(final_train, test)
    measured = metrics(test.target, p, final_train.target.mean())
    simple = metrics(test.target, rule, final_train.target.mean())
    uncertainty = bootstrap_comparison(test.target.to_numpy(int), p, rule, cfg)
    useful = (uncertainty["capture_20_ci"][0] > .2
              and uncertainty["capture_difference_vs_rule_ci"][0] > 0
              and measured["brier_skill"] > 0)
    truth, predicted = calibration_curve(test.target.astype(int), p, n_bins=5, strategy="quantile")
    result = {**base, "status": "señal retrospectiva" if useful else "experimental",
              "winner": winner, "selection": "Mayor captura al 20% en validación; desempate por Brier y AUC.",
              "validation_year": int(val.score_year.iloc[0]), "test_year": int(test.score_year.iloc[0]),
              "selection_train_n": len(train), "final_train_n": len(final_train),
              "reference_prevalence": float(final_train.target.mean()),
              "selection_train_cohorts": sorted(train.cohorte.unique().astype(int).tolist()),
              "maximum_selection_training_label_ready_year": int(train.label_ready_year.max()),
              "maximum_validation_label_ready_year": int(val.label_ready_year.max()),
              "final_train_cohorts": sorted(final_train.cohorte.unique().astype(int).tolist()),
              "maximum_training_label_ready_year": int(final_train.label_ready_year.max()),
              "validation_candidates": candidates, "test": measured, "historical_rule": simple,
              "uncertainty": uncertainty,
              "calibration_bias": float(p.mean() - test.target.mean()),
              "calibration": [{"mean_prediction": float(a), "observed_fraction": float(b)}
                              for a, b in zip(predicted, truth)],
              "importance_validation": [{"variable": c, "auc_loss": float(v), "std": float(s)}
                                        for c, v, s in zip(xv.columns, importance.importances_mean,
                                                           importance.importances_std)],
              "capacity": [], "subgroups": []}
    for q in CAPACITIES:
        k = int(np.ceil(len(test) * q))
        c = capture(test.target, p, q)
        result["capacity"].append({"capacity": q, "review_slots": k, "capture": c,
                                   "precision": c * test.target.sum() / k,
                                   "rule_capture": capture(test.target, rule, q),
                                   "random_capture": k / len(test)})
    pred = test[["mrun", "cohorte", "score_year", "outcome_year", "nomb_sede", "nomb_carrera", "jornada", "target"]].copy()
    pred["risk_score"] = p
    pred["historical_rule"] = rule
    pred["model"] = winner
    pred["case"] = case
    pred["evaluation"] = "test temporal reservado"
    # El indicador de captura por subgrupo reprioriza dentro de ese subgrupo.
    for dimension in ("nomb_sede", "nomb_carrera"):
        for label, g in pred.groupby(dimension):
            if len(g) >= cfg["minimum_display_cell"]:
                result["subgroups"].append({"dimension": dimension, "group": label,
                                            **metrics(g.target, g.risk_score, final_train.target.mean())})
    return result, model, pred
