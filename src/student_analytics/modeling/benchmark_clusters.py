"""Lectura de los clusters del benchmark: qué distingue a cada grupo y dónde queda la UA.

Todo es descriptivo y se calcula sobre las universidades que entran al
agrupamiento. Nada de esto cambia los grupos ni los pares: solo los explica.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from student_analytics.modeling.benchmark import PeerModel

BLOCKS = {"escala": "Tamaño y sedes", "selectividad": "Selectividad", "areas": "Áreas",
          "regiones": "Regiones", "docencia": "Jornada y modalidad",
          "descriptiva": "No forma los grupos"}


def group_order(labels) -> list[str]:
    """«Grupo 10» va después de «Grupo 9», no después de «Grupo 1»."""
    return sorted(set(labels), key=lambda g: int(re.sub(r"\D", "", g) or 0))


def cluster_variables(model: PeerModel, code: str) -> list[dict]:
    """Las variables de cada bloque que forma los grupos, en su escala original.

    La distancia usa log(1+x) y raíz de proporciones; aquí se muestran sin
    transformar para que se puedan leer. Las regiones se resumen: el bloque usa
    las 16, pero se muestran la Metropolitana, el número de regiones y las
    regiones donde la UA tiene matrícula.
    """
    u = model.universities
    ua = u.loc[u.cod_inst.eq(code)]
    rows = []

    def add(column, label, fmt, block):
        if column in u and u[column].notna().any():
            rows.append({"variable": column, "etiqueta": label, "formato": fmt, "bloque": block})

    if "escala" in model.groups:
        add("cohorte_total", "Inscripciones de ingreso", ",", "escala")
        add("sedes", "Sedes", ",", "escala")
    if "selectividad" in model.groups:
        add("paes_promedio", "Puntaje PAES promedio", ".0f", "selectividad")
        add("paes_rango_intercuartil", "Dispersión PAES (p75 − p25)", ".0f", "selectividad")
    for column in model.groups.get("areas", []):
        name = column.split("::", 1)[1]
        if name != "Sin información" or u[column].max() >= .05:
            add(column, "Área · " + name, "%", "areas")
    if "regiones" in model.groups:
        add("region::Metropolitana", "Región Metropolitana", "%", "regiones")
        for column in model.groups["regiones"]:
            if column != "region::Metropolitana" and not ua.empty and float(ua[column].iloc[0]) >= .05:
                add(column, "Región · " + column.split("::", 1)[1], "%", "regiones")
        add("regiones_presencia", "Regiones con presencia", ",", "regiones")
    for column in model.groups.get("docencia", []):
        kind, name = column.split("::", 1)
        # La modalidad repite a la jornada (a distancia = no presencial); basta la presencial.
        keep = (kind == "jornada" and name not in ("Otra", "Sin información")) or name == "Presencial"
        if keep and u[column].max() >= .05:
            add(column, ("Jornada · " if kind == "jornada" else "Modalidad · ") + name, "%", "docencia")
    return rows


def percentile(series: pd.Series, value: float) -> float:
    """Porcentaje de las otras universidades que queda por debajo; los empates cuentan la mitad.

    0 es la menor y 100 la mayor. Es posición, no puntuación: en arancel o en
    duración estar arriba no es mejor.
    """
    s = series.dropna().astype(float)
    if len(s) < 2 or not np.isfinite(value):
        return float("nan")
    below, equal = float((s < value).sum()), float((s == value).sum())
    # La propia universidad está en la serie: se descuenta de los empates.
    return 100 * (below + .5 * max(equal - 1, 0)) / (len(s) - 1)


def cluster_profile(model: PeerModel, variables: list[dict], code: str,
                    frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mediana de cada variable por grupo y valor de la UA.

    `z` mide cuántas desviaciones estándar se aleja cada mediana del promedio
    de todas las universidades: es lo que colorea el mapa de calor.
    `frame` permite sumar columnas calculadas con el filtro, como la retención.
    """
    u = frame if frame is not None else model.universities
    rows = []
    for v in variables:
        column = u[v["variable"]].astype(float)
        mean, sd = column.mean(), column.std(ddof=0)

        def z(x):
            return float((x - mean) / sd) if np.isfinite(x) and sd > 1e-12 else 0.0

        ua = column[u.cod_inst.eq(code)]
        value = float(ua.iloc[0]) if len(ua) else float("nan")
        rows.append({**v, "columna": "Universidad Autónoma", "valor": value, "z": z(value),
                     "universidades": 1})
        for grupo in group_order(u.grupo):
            values = column[u.grupo.eq(grupo)].dropna()
            median = float(values.median()) if len(values) else float("nan")
            rows.append({**v, "columna": grupo, "valor": median, "z": z(median),
                         "universidades": int(len(values))})
    return pd.DataFrame(rows)


def positions(model: PeerModel, variables: list[dict], code: str,
              frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cada universidad en cada variable, como percentil entre las que tienen dato."""
    u = frame if frame is not None else model.universities
    rows = []
    for v in variables:
        column = u[v["variable"]].astype(float)
        for value, name, grupo, cod in zip(column, u.nomb_inst, u.grupo, u.cod_inst):
            if np.isfinite(value):
                rows.append({**v, "valor": value, "percentil": percentile(column, value),
                             "nomb_inst": name, "grupo": grupo, "es_ua": cod == code})
    return pd.DataFrame(rows)


def ua_summary(model: PeerModel, variables: list[dict], code: str,
               frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """La UA frente a su grupo y frente a todas, variable por variable."""
    u = frame if frame is not None else model.universities
    ua = u.loc[u.cod_inst.eq(code)]
    if ua.empty:
        return pd.DataFrame()
    own = u.grupo.eq(ua.grupo.iloc[0])
    rows = []
    for v in variables:
        column = u[v["variable"]].astype(float)
        value = float(ua[v["variable"]].iloc[0])
        scale = 100 if v["formato"] == "%" else 1
        suffix = " (%)" if scale == 100 else ""
        rows.append({"Bloque": BLOCKS[v["bloque"]], "Variable": v["etiqueta"] + suffix,
                     "UA": scale * value,
                     "Mediana del grupo UA": scale * float(column[own].median()),
                     "Mediana de todas": scale * float(column.median()),
                     "Percentil UA": round(percentile(column, value), 0),
                     "Universidades con dato": int(column.notna().sum())})
    return pd.DataFrame(rows)


def partition_agreement(model: PeerModel, code: str) -> tuple[pd.DataFrame, list[str]]:
    """¿Los otros algoritmos, con el mismo número de grupos, dejan a la UA con las mismas universidades?

    Devuelve una fila por algoritmo y el núcleo estable: las universidades que
    comparten grupo con la UA en los tres. El índice Rand ajustado compara la
    partición completa con la elegida (1 = idéntica, 0 = como al azar).
    """
    if not model.k or not model.partitions:
        return pd.DataFrame(), []
    u = model.universities
    index = int(np.flatnonzero(u.cod_inst.eq(code))[0])
    chosen = u.grupo.to_numpy()
    chosen_same = chosen == chosen[index]
    diagnostics = model.diagnostics.set_index(["algoritmo", "k"])
    rows, core = [], np.ones(len(u), dtype=bool)
    for (name, k), labels in model.partitions.items():
        if k != model.k:
            continue
        same = labels == labels[index]
        core &= same
        info = diagnostics.loc[(name, k)]
        rows.append({"Algoritmo": name + (" · elegido" if name == model.algorithm else ""),
                     "Admisible": "Sí" if info.admisible else "No",
                     "Silueta": float(info.silueta),
                     "Universidades en el grupo de la UA": int(same.sum()) - 1,
                     "También están en el grupo elegido": int((same & chosen_same).sum()) - 1,
                     "Concordancia con la elegida (ARI)": float(adjusted_rand_score(chosen, labels))})
    core[index] = False
    return pd.DataFrame(rows), sorted(u.nomb_inst[core].tolist())
