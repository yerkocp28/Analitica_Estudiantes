"""Fuentes existentes, abiertas solo para lectura; salidas en la carpeta del piloto."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def identifier(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.isin(["", "nan", "None", "<NA>", "0"]))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_entries(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    d["mrun"] = identifier(d.mrun)
    d["cod_inst"] = identifier(d.cod_inst)
    d = d[d.cod_inst.eq(cfg["institution_code"])].copy()
    audit = {"source_rows": len(d), "missing_identifier": int(d.mrun.isna().sum())}
    # Sin identidad no existe seguimiento: nunca se interpreta el cero heredado.
    d = d.dropna(subset=["mrun", "cohorte"]).drop_duplicates()
    first = d.groupby("mrun").cohorte.transform("min")
    audit["later_entries_excluded"] = int(d.cohorte.ne(first).sum())
    d = d[d.cohorte.eq(first)].copy()
    ambiguous = d.duplicated(["mrun", "cohorte"], keep=False)
    audit["ambiguous_first_entries_excluded"] = int(ambiguous.sum())
    d = d[~ambiguous].copy()
    scope = (d.area_conocimiento.astype("string").str.startswith(cfg["area_prefix"], na=False)
             & d.tipo_plan_carr.eq(cfg["regular_plan"])
             & d.nivel_carrera_1.astype("string").str.startswith(
                 cfg["professional_prefix"], na=False))
    d = d[scope].copy().sort_values(["cohorte", "mrun"]).reset_index(drop=True)
    audit["area_entries_before_catalog"] = len(d)
    catalog = cfg.get("faculty_catalog")
    audit["careers_excluded_pending_confirmation"] = {}
    if catalog:
        documented = d.nomb_carrera.isin(catalog)
        audit["careers_excluded_pending_confirmation"] = d.loc[~documented, "nomb_carrera"].value_counts().to_dict()
        d = d[documented].copy().reset_index(drop=True)
    for col in ("cohorte", "duracion_nominal_anios", "percentil_admision",
                "percentil_nem", "percentil_ranking"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    # Puntajes inconsistentes se desconocen, no se recortan a un extremo inventado.
    for col in ("percentil_admision", "percentil_nem", "percentil_ranking"):
        d[col] = d[col].where(d[col].between(0, 1))
    audit["pilot_entries"] = len(d)
    audit["distinct_people"] = int(d.mrun.nunique())
    audit["scope_confirmed_by_faculty"] = cfg["scope_confirmed_by_faculty"]
    audit["careers"] = d.nomb_carrera.value_counts().to_dict()
    audit["sites_unmapped"] = d.nomb_sede.value_counts().to_dict()
    return d, audit


def read_sources(root: Path, cfg: dict, progress=print):
    source = root / "data/results/ua_ingresantes.parquet"
    entries, audit = prepare_entries(pd.read_parquet(source), cfg)
    paths = [source]
    people = set(entries.mrun)
    history = []
    year_files = sorted((root / "data/interim/matricula").glob("matricula_*.parquet"))
    history_years = []
    for path in year_files:
        year = int(path.stem.split("_")[-1])
        d = pd.read_parquet(path, columns=["mrun", "cod_inst", "cod_carrera", "nomb_sede", "jornada"],
                            filters=[("cod_inst", "=", cfg["institution_code"])])
        d["mrun"] = identifier(d.mrun)
        d = d[d.mrun.isin(people)].copy()
        d["year"] = year
        history.append(d.drop_duplicates())
        history_years.append(year)
        paths.append(path)
    if not history:
        raise ValueError("Falta la historia de matrícula local.")
    progress(f"Historia de matrícula: {min(history_years)}–{max(history_years)}", flush=True)
    titles, title_years = [], []
    for folder in sorted((root / "data/external/titulados").glob("titulados_*")):
        csvs = list(folder.rglob("*.csv"))
        if not csvs:
            continue
        path = max(csvs, key=lambda p: p.stat().st_size)
        year = int(folder.name.split("_")[-1])
        # Conservar TODOS los títulos: un título previo no debe ocultar otro posterior.
        d = pd.read_csv(path, sep=";", encoding="utf-8", dtype="string",
                        usecols=["mrun", "cod_inst", "cod_carrera", "nivel_global"])
        d["mrun"] = identifier(d.mrun)
        d["cod_inst"] = identifier(d.cod_inst)
        d = d[d.cod_inst.eq(cfg["institution_code"]) & d.mrun.isin(people)
              & d.nivel_global.eq("Pregrado")].copy()
        d["year"] = year
        titles.append(d[["mrun", "year"]].drop_duplicates())
        title_years.append(year)
        paths.append(path)
    if not titles:
        raise ValueError("Faltan las bases locales de titulados.")
    audit["history_years"] = history_years
    audit["title_years"] = title_years
    return entries, pd.concat(history, ignore_index=True), pd.concat(titles, ignore_index=True), audit, paths


def continuity(entries: pd.DataFrame, history: pd.DataFrame, cfg: dict,
               history_years: list[int] | None = None) -> pd.DataFrame:
    d = entries[entries.cohorte.ge(cfg["admission_from"])].copy()
    observed_years = set(history_years if history_years is not None else history.year.unique())
    observed = set(zip(history.mrun, history.year))
    d["score_year"] = d.cohorte.astype(int)
    d["outcome_year"] = d.score_year + 1
    d["label_ready_year"] = d.outcome_year + cfg["label_lag_years"]
    d["target"] = [float((person, year) not in observed) if year in observed_years else np.nan
                   for person, year in zip(d.mrun, d.outcome_year)]
    return d


def graduation(entries: pd.DataFrame, history: pd.DataFrame, titles: pd.DataFrame,
               cfg: dict, history_years: list[int], title_years: list[int]):
    groups = {k: g for k, g in history.groupby("mrun", sort=False)}
    degrees = titles.groupby("mrun").year.apply(set).to_dict()
    available_h, available_t = set(history_years), set(title_years)
    rows, excluded = [], {"not_enrolled_at_landmark": 0, "already_graduated": 0,
                          "invalid_duration": 0, "missing_historical_years": 0}
    for _, row in entries.iterrows():
        entry = int(row.cohorte)
        cut = entry + cfg["landmark_years"]
        nominal = row.duracion_nominal_anios
        if pd.isna(nominal) or not 3 < nominal <= 8:
            excluded["invalid_duration"] += 1
            continue
        horizon = int(np.ceil(entry + nominal + cfg["graduation_grace_years"]))
        if not set(range(entry, cut + 1)).issubset(available_h) or not set(
                range(entry, cut + 1)).issubset(available_t):
            excluded["missing_historical_years"] += 1
            continue
        h = groups.get(row.mrun)
        h = h[h.year.between(entry, cut)] if h is not None else pd.DataFrame()
        if h.empty or cut not in set(h.year):
            excluded["not_enrolled_at_landmark"] += 1
            continue
        t = {y for y in degrees.get(row.mrun, set()) if y >= entry}
        if any(y <= cut for y in t):
            excluded["already_graduated"] += 1
            continue
        r = row.to_dict()
        r.update(score_year=cut, outcome_year=horizon,
                 label_ready_year=horizon + cfg["label_lag_years"],
                 years_enrolled=int(h.year.nunique()),
                 years_without_ua=int(cut - entry + 1 - h.year.nunique()),
                 career_codes_observed=int(h.cod_carrera.nunique()),
                 sites_observed=int(h.nomb_sede.nunique()),
                 schedules_observed=int(h.jornada.nunique()))
        # La ausencia de un archivo anual no se interpreta como ausencia de título.
        observable = set(range(cut + 1, horizon + 1)).issubset(available_t)
        r["target"] = float(not any(cut < y <= horizon for y in t)) if observable else np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    excluded["eligible_at_landmark"] = len(d)
    excluded["right_censored"] = int(d.target.isna().sum()) if len(d) else 0
    return d, excluded
