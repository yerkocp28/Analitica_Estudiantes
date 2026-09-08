"""Comparadores institucionales basados exclusivamente en perfiles de ingreso."""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from student_analytics.ingestion.retention import summarize_retention

SHARE_BLOCKS = {"areas": ["area"], "regiones": ["region"], "docencia": ["jornada", "modalidad"]}


def build_profiles(cohort: pd.DataFrame, year: int) -> pd.DataFrame:
    """Agregados sin MRUN exportable, sobre todas las inscripciones de ingreso."""
    cohort = cohort.copy()
    cohort["cod_inst"] = cohort.cod_inst.astype(str)
    grouped = cohort.groupby("cod_inst", sort=True)
    result = grouped.agg(nomb_inst=("nomb_inst", "first"),
                         cohorte_total=("cod_inst", "size"),
                         sedes=("cod_sede", "nunique"))
    result["log_cohorte"] = np.log1p(result.cohorte_total)
    result["log_sedes"] = np.log1p(result.sedes)
    for prefix, column in [("area", "area_conocimiento"), ("region", "region_sede"),
                           ("jornada", "jornada"), ("modalidad", "modalidad")]:
        values = cohort[column].astype("string").fillna("Sin información")
        counts = pd.crosstab(cohort.cod_inst, values).reindex(result.index, fill_value=0)
        shares = counts.div(result.cohorte_total, axis=0)
        result = result.join(shares.add_prefix(prefix + "::"))
        result[f"cobertura_{prefix}"] = values.ne("Sin información").groupby(cohort.cod_inst).mean()
    result.insert(0, "cohorte", year)
    return result.reset_index()


def profile_matrix(profiles: pd.DataFrame, weights: dict) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Raíz de proporciones; igual varianza total por bloque, pesos explícitos.

    No estandariza cada categoría por separado: una categoría rara no recibe
    un peso extremo. Lista positiva de variables excluye todos los resultados.
    """
    if (set(weights) != {"escala", *SHARE_BLOCKS}
            or any(not np.isfinite(w) or w <= 0 for w in weights.values())
            or not np.isclose(sum(weights.values()), 1)):
        raise ValueError("Los cuatro pesos de bloques deben ser positivos y sumar 1.")
    groups = {"escala": ["log_cohorte", "log_sedes"]}
    groups.update({block: sorted(c for c in profiles if any(c.startswith(p + "::") for p in prefixes))
                   for block, prefixes in SHARE_BLOCKS.items()})
    matrices = []
    for block, columns in groups.items():
        if not columns:
            raise ValueError(f"Faltan variables del bloque {block}")
        x = profiles[columns].fillna(0).to_numpy(dtype=float)
        if not np.isfinite(x).all() or (block != "escala" and (x < 0).any()):
            raise ValueError("Perfil institucional inválido")
        if block != "escala":
            x = np.sqrt(x)
        else:
            sd = x.std(axis=0)
            x = (x - x.mean(axis=0)) / np.where(sd > 1e-12, sd, 1)
        x = x - x.mean(axis=0)
        scale = np.sqrt(np.var(x, axis=0).sum())
        matrices.append(x / (scale if scale > 1e-12 else 1) * np.sqrt(weights[block]))
    return np.concatenate(matrices, axis=1), groups


def target_code(profiles: pd.DataFrame, name: str) -> str:
    def normalized(value):
        return "".join(c for c in unicodedata.normalize("NFKD", str(value).upper()) if not unicodedata.combining(c)).strip()
    ids = profiles.loc[profiles.nomb_inst.map(normalized).eq(normalized(name)), "cod_inst"].unique()
    if len(ids) != 1:
        raise ValueError("No se identificó una única Universidad Autónoma en la cohorte disponible.")
    return str(ids[0])


@dataclass
class PeerModel:
    universities: pd.DataFrame
    diagnostics: pd.DataFrame
    matrix: np.ndarray
    groups: dict[str, list[str]]
    algorithm: str
    k: int
    silhouette: float
    explained_variance: float


def fit_peers(profiles: pd.DataFrame, config: dict) -> PeerModel:
    """Compara tres algoritmos y selecciona silueta con tamaño mínimo de grupo."""
    p = profiles.loc[profiles.cohorte_total.ge(config["minimum_cohort"])].copy()
    p = p.sort_values("cod_inst").reset_index(drop=True)
    if len(p) < 2 * config["minimum_cluster"]:
        raise ValueError("Hay pocas universidades con tamaño suficiente para formar grupos comparables.")
    x, groups = profile_matrix(p, config["block_weights"])
    if np.linalg.norm(x) < 1e-10:
        raise ValueError("Los perfiles no presentan variación suficiente para formar grupos.")
    diagnostics, candidates = [], {}
    for k in config["k_values"]:
        if k >= len(p):
            continue
        models = {
            "K-means": KMeans(n_clusters=k, n_init=20, random_state=config["random_seed"]),
            "Jerárquico Ward": AgglomerativeClustering(n_clusters=k),
            "Mezcla gaussiana": GaussianMixture(n_components=k, covariance_type="diag", n_init=5,
                                               reg_covar=1e-4, random_state=config["random_seed"]),
        }
        for name, model in models.items():
            labels = model.fit_predict(x)
            counts = np.bincount(labels)
            distinct = len(np.unique(labels))
            score = silhouette_score(x, labels) if 1 < distinct < len(p) else float("nan")
            valid = (distinct == k and counts.min() >= config["minimum_cluster"]
                     and np.isfinite(score) and getattr(model, "converged_", True))
            diagnostics.append({"algoritmo": name, "k": k, "silueta": score,
                                "grupo_minimo": int(counts.min()), "grupo_maximo": int(counts.max()),
                                "admisible": bool(valid)})
            if valid:
                candidates[(name, k)] = labels
    table = pd.DataFrame(diagnostics).sort_values(["admisible", "silueta", "k"], ascending=[False, False, True])
    if not candidates:
        # Los vecinos siguen siendo interpretables aunque no exista una partición admisible.
        name, k, score = "Sin partición admisible", 0, float("nan")
        p["grupo"] = "Sin grupo"
    else:
        best = table.loc[table.admisible].iloc[0]
        name, k, score = best.algoritmo, int(best.k), float(best.silueta)
        p["grupo"] = [f"Grupo {v + 1}" for v in candidates[(name, k)]]
    pca = PCA(n_components=2)
    projection = pca.fit_transform(x)
    p["mapa_x"], p["mapa_y"] = projection[:, 0], projection[:, 1]
    return PeerModel(p, table, x, groups, name, k, score, float(pca.explained_variance_ratio_.sum()))


def neighbors(model: PeerModel, target: str) -> pd.DataFrame:
    positions = np.flatnonzero(model.universities.cod_inst.eq(target))
    if not len(positions):
        raise ValueError("La UA no cumple el tamaño mínimo de cohorte configurado.")
    index = positions[0]
    result = model.universities.copy()
    result["distancia"] = np.linalg.norm(model.matrix - model.matrix[index], axis=1)
    offset = 0
    for block, columns in model.groups.items():
        width = len(columns)
        result[f"distancia_{block}"] = np.linalg.norm(
            model.matrix[:, offset:offset + width] - model.matrix[index, offset:offset + width], axis=1)
        offset += width
    result["mismo_grupo_ua"] = result.grupo.eq(result.iloc[index].grupo) & (model.k > 0)
    return result.loc[~result.cod_inst.eq(target)].sort_values(["distancia", "cod_inst"])


def compare_outcomes(data: pd.DataFrame, year: int, target: str, peer_ids: list[str],
                     metric: str, area: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Referencias sin UA y estandarización por mezcla de áreas de la UA.

    La referencia ajustada se calcula solo en áreas comunes; la retención UA
    se recalcula en ese mismo soporte. Se informa la cobertura resultante.
    """
    selected = data.loc[data.cohorte.eq(year)].copy()
    if area is not None:
        selected = selected.loc[selected.area_conocimiento.eq(area)]
    summary = summarize_retention(selected, ["cod_inst", "nomb_inst"])
    peers = summary.loc[summary.cod_inst.isin(peer_ids) & ~summary.cod_inst.eq(target)]
    national = summary.loc[~summary.cod_inst.eq(target)]
    ua = summary.loc[summary.cod_inst.eq(target)]
    def rate(rows):
        return float(rows[metric].sum() / rows.n.sum()) if rows.n.sum() else float("nan")
    stats = {"ua": rate(ua), "ua_n": int(ua.n.sum()), "pares": rate(peers),
             "nacional": rate(national), "pares_n": int(peers.n.sum()),
             "pares_disponibles": int(peers.loc[peers.n.gt(0), "cod_inst"].nunique()),
             "ajustada": float("nan"), "ua_comun": float("nan"), "cobertura": 0.0}
    by_area = summarize_retention(selected, ["cod_inst", "area_conocimiento"])
    a = by_area.loc[by_area.cod_inst.eq(target)].set_index("area_conocimiento")
    b = by_area.loc[by_area.cod_inst.isin(peer_ids) & ~by_area.cod_inst.eq(target)]
    b = b.groupby("area_conocimiento")[["n", metric]].sum()
    common = a.loc[a.n.gt(0), ["n", metric]].join(b.loc[b.n.gt(0)], how="inner", rsuffix="_pares")
    if not common.empty and stats["ua_n"]:
        weights = common.n / common.n.sum()
        stats.update(ajustada=float((weights * common[f"{metric}_pares"] / common.n_pares).sum()),
                     ua_comun=float(common[metric].sum() / common.n.sum()),
                     cobertura=float(common.n.sum() / stats["ua_n"]))
    shown = summary.loc[summary.cod_inst.isin([target, *peer_ids])].copy()
    shown["retencion"] = shown[metric].div(shown.n.where(shown.n.gt(0)))
    shown["es_ua"] = shown.cod_inst.eq(target)
    shown["brecha_vs_ua_pp"] = 100 * (shown.retencion - stats["ua"])
    return shown, stats
