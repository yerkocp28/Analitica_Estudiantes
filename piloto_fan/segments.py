"""Segmentos aprendidos en cohortes antiguas, asignados a cohortes posteriores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["percentil_admision", "percentil_nem", "percentil_ranking", "duracion_nominal_anios"]
CATEGORICAL = ["jornada"]


def segment(data: pd.DataFrame, cfg: dict):
    # Carrera y sede se describen después. No queremos descubrir su organigrama.
    d = data.copy()
    years = sorted(d.cohorte.unique())
    if len(years) < 3:
        return {"status": "insuficiente", "reason": "Se requieren al menos tres cohortes."}, None, None
    fit_years = years[:-2]
    train = d[d.cohorte.isin(fit_years)]
    x = d[NUMERIC + CATEGORICAL].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
    for col in CATEGORICAL:
        x[col] = x[col].astype("string").fillna("Sin información").astype(str)
    prep = ColumnTransformer([
        ("academic", Pipeline([("missing", SimpleImputer(strategy="median", keep_empty_features=True)),
                                ("scale", StandardScaler())]), NUMERIC),
        ("schedule", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    a = prep.fit_transform(x.loc[train.index])
    all_x = prep.transform(x)
    candidates, eligible = [], []
    for k in cfg["segment_k"]:
        if k >= len(train):
            continue
        algorithms = {
            "K-means": KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]),
            "Mezcla gaussiana": GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                                                 reg_covar=1e-4, random_state=cfg["seed"]),
            "K-means bisectriz": BisectingKMeans(n_clusters=k, random_state=cfg["seed"]),
        }
        for name, model in algorithms.items():
            labels = model.fit_predict(a)
            counts = pd.Series(labels).value_counts()
            if len(counts) < 2:
                continue
            silhouette = float(silhouette_score(a, labels, sample_size=min(len(a), 1000),
                                                random_state=cfg["seed"]))
            admissible = (len(counts) == k and counts.min() >= cfg["segment_minimum"]
                          and counts.max() / len(a) <= cfg["segment_maximum_share"]
                          and getattr(model, "converged_", True))
            row = {"algorithm": name, "k": k, "silhouette": silhouette,
                   "smallest_segment": int(counts.min()), "largest_share": float(counts.max() / len(a)),
                   "admissible": bool(admissible)}
            candidates.append(row)
            if admissible:
                eligible.append((silhouette, -k, name, model))
    if not eligible:
        return {"status": "insuficiente", "candidates": candidates,
                "reason": "Ninguna partición satisface tamaño y concentración."}, None, None
    sil, minus_k, name, chosen = max(eligible, key=lambda a: (a[0], a[1]))
    # Numeración congelada por tamaño TRAIN; la cohorte nueva no cambia el significado.
    order = pd.Series(chosen.predict(a)).value_counts().index
    mapping = {int(k): int(i + 1) for i, k in enumerate(order)}
    d["segment"] = [mapping[int(k)] for k in chosen.predict(all_x)]
    profiles = []
    for (year, label), g in d.groupby(["cohorte", "segment"]):
        if len(g) < cfg["minimum_display_cell"]:
            continue
        row = {"cohort": int(year), "segment": int(label), "n": len(g),
               "share": float(len(g) / d.cohorte.eq(year).sum()),
               "observed_outcomes": int(g.target.notna().sum()),
               "continuity": float(1 - g.target.mean()) if g.target.notna().any() else None,
               "admission_coverage": float(g.percentil_admision.notna().mean())}
        for col in NUMERIC:
            row[col] = float(g[col].mean()) if g[col].notna().any() else None
        profiles.append(row)
    by_site = []
    for (year, site, label), g in d.groupby(["cohorte", "nomb_sede", "segment"]):
        if len(g) >= cfg["minimum_display_cell"]:
            by_site.append({"cohort": int(year), "site": site, "segment": int(label), "n": len(g)})
    train_shares = d[d.cohorte.isin(fit_years)].segment.value_counts(normalize=True)
    drift = []
    for year in years[-2:]:
        shares = d[d.cohorte.eq(year)].segment.value_counts(normalize=True)
        # Distancia total de variación entre distribuciones, no estabilidad causal.
        delta = sum(abs(shares.get(k, 0) - train_shares.get(k, 0)) for k in mapping.values()) / 2
        drift.append({"cohort": int(year), "total_variation": float(delta)})
    result = {"status": "exploratorio", "winner": name, "k": -minus_k, "silhouette": sil,
              "fit_cohorts": [int(y) for y in fit_years], "holdout_cohorts": [int(y) for y in years[-2:]],
              "features": NUMERIC + CATEGORICAL, "candidates": candidates,
              "profiles": profiles, "by_site": by_site, "distribution_shift": drift,
              "limitations": "La imputación puede influir en los grupos. No son niveles de riesgo ni necesidades diagnosticadas."}
    return result, {"preprocessor": prep, "model": chosen, "mapping": mapping}, d[["mrun", "cohorte", "segment"]]
