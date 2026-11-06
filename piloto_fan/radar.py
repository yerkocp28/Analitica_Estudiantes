"""Radar de inicio de semestre: prototipo retrospectivo sobre 2025-1 → 2025-2.

Cada dirección (carrera en una sede) recibe a sus estudiantes ordenados por el
riesgo de terminar el semestre siguiente con al menos una nota numérica bajo 4.
La lista se corta en la capacidad de revisión que fije la dirección y cada caso
trae las señales que más suben su riesgo.

TRES DECISIONES

1. **Ordena, no informa probabilidades.** El modelo está descalibrado —quedó
   declarado en el preregistro— y recalibrarlo no cambia el orden, que es lo
   único que usa una lista de capacidad fija. Mostrar un porcentaje invitaría a
   leerlo como certeza.
2. **Explica con señales, no con causas.** El efecto de una señal es cuánto
   bajaría el riesgo si esa señal tuviera su valor típico del entrenamiento,
   con todo lo demás igual. Sirve para cualquier modelo y se lee sin
   estadística: «promedio 4,3; típico 5,2».
3. **Es retrospectivo.** El último semestre del extracto ya terminó, así que
   muestra lo que habría recibido cada dirección al inicio de 2025-2 y qué pasó
   después. Usa el modelo guardado, sin reentrenar, y no exporta filas
   individuales.

Solo el caso académico: el de matrícula tiene 16 eventos en su prueba
(2025-1 → 2025-2 no cruza de año) y no se puede demostrar con estos datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dashboard import number
from .modelos_institucionales import NUMERIC
from .models import design

ORIGIN, OUTCOME = 202510, 202520
TRAIN_TERMS = [202410, 202420]
GROUP = ["career", "campus"]
NO_SIGNAL = "Sin señal académica dominante: pesa el contexto de carrera, sede y jornada"
SIGNALS = {
    "numeric_mean": ("Promedio", lambda v: number(v, 1)),
    "below4_fraction": ("Notas bajo 4", lambda v: number(100 * v, 0) + "%"),
    "numeric_sd": ("Dispersión de notas", lambda v: number(v, 1)),
    "numeric_coverage": ("Resultados con nota numérica", lambda v: number(100 * v, 0) + "%"),
    "observed_results": ("Asignaturas con resultado", lambda v: number(v, 0)),
}


def typical(dataset: pd.DataFrame) -> pd.Series:
    """Valores típicos de la población con la que se reentrenó el modelo final."""
    train = dataset.loc[dataset.source_term.isin(TRAIN_TERMS) & dataset.target.notna()]
    return train[NUMERIC].median()


def score(dataset: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Todos los estudiantes del semestre de origen, con resultado conocido o no.

    Quien no tiene resultado numérico en el semestre siguiente también entra a la
    lista: puede ser justamente alguien que se fue.
    """
    d = dataset.loc[dataset.source_term.eq(ORIGIN)].copy()
    d["risk"] = bundle["model"].predict_proba(design(d, bundle["spec"]))[:, 1]
    return d.reset_index(drop=True)


def effects(d: pd.DataFrame, bundle: dict, reference: pd.Series) -> pd.DataFrame:
    """Cuánto bajaría el riesgo si cada señal tuviera su valor típico."""
    out = {}
    for col in NUMERIC:
        x = d.copy()
        x[col] = reference[col]
        out[col] = d.risk.to_numpy() - bundle["model"].predict_proba(design(x, bundle["spec"]))[:, 1]
    return pd.DataFrame(out, index=d.index)


def explain(d: pd.DataFrame, effect: pd.DataFrame, reference: pd.Series,
            threshold: float = .02, top: int = 3) -> list[str]:
    texts = []
    for i, row in effect.iterrows():
        chosen = row[row >= threshold].sort_values(ascending=False).head(top).index
        parts = []
        for col in chosen:
            label, fmt = SIGNALS[col]
            value = d.at[i, col]
            parts.append(f"{label} {fmt(value) if pd.notna(value) else 'sin dato'} (típico {fmt(reference[col])})")
        texts.append(" · ".join(parts) or NO_SIGNAL)
    return texts


def radar(dataset: pd.DataFrame, bundle: dict, enrollment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lista completa del semestre de origen; `enrollment` aporta la ausencia en el semestre siguiente."""
    d = score(dataset, bundle)
    reference = typical(dataset)
    d["signals"] = explain(d, effects(d, bundle, reference), reference)
    absent = pd.Series(np.nan, index=d.index)
    if enrollment is not None:
        e = enrollment.loc[enrollment.source_term.eq(ORIGIN)].set_index("student_key").target
        absent = d.student_key.map(e)
    d["outcome"] = np.select(
        [d.target.eq(1), d.target.eq(0), absent.eq(1)],
        ["Al menos una nota bajo 4", "Sin notas bajo 4", "No se matriculó en 2025-2"],
        "Resultado incompleto o solo con códigos")
    return d


def prioritize(d: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Orden dentro de cada dirección y corte en su capacidad de revisión."""
    d = d.sort_values(["risk", "student_key"], ascending=[False, True], kind="mergesort").copy()
    d["priority"] = d.groupby(GROUP).cumcount() + 1
    size = d.groupby(GROUP).student_key.transform("size")
    d["listed"] = d.priority <= np.ceil(capacity * size)
    return d


def ranked_hits(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Aciertos esperados entre los k primeros, con empates repartidos al azar."""
    if k <= 0 or not len(y):
        return 0.
    threshold = np.sort(score)[-k]
    high, tied = score > threshold, score == threshold
    return float(y[high].sum() + (k - high.sum()) * y[tied].mean())


def retrospective(d: pd.DataFrame, capacity: float, minimum: int = 10) -> pd.DataFrame:
    """Qué habría encontrado cada lista, contada sobre quienes tienen resultado conocido.

    El corte se aplica a la población con resultado, igual que en la prueba del
    modelo. La regla simple ordena por la fracción previa de notas bajo 4 con la
    misma cantidad de cupos, y sus empates se reparten al azar.
    """
    rows = []
    for (career, campus), g in d.loc[d.target.notna()].groupby(GROUP, sort=True):
        if len(g) < minimum:
            continue
        y = g.target.to_numpy(float)
        k = int(np.ceil(capacity * len(g)))
        model, rule = ranked_hits(y, g.risk.to_numpy(), k), ranked_hits(y, g.below4_fraction.to_numpy(), k)
        rows.append({"Carrera": career, "Sede": campus, "Estudiantes con resultado": len(g),
                     "Cupos": k, "Aciertos del radar": model, "Aciertos de la regla simple": rule,
                     "Aciertos al azar": k * y.mean(), "Precisión del radar (%)": 100 * model / k,
                     "Frecuencia en la dirección (%)": 100 * y.mean(),
                     "Captura del radar (%)": 100 * model / y.sum() if y.sum() else np.nan})
    return pd.DataFrame(rows)
