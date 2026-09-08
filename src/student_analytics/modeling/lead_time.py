"""Poder predictivo vs anticipacion (doc maestro 68).

Responde la pregunta que define el valor operacional del proyecto y que el
informe ULagos dejo planteada:

    Si la direccion de carrera revisa al 10% / 20% de sus estudiantes en la
    semana N, que fraccion de los que van a reprobar alcanza a ver, y
    cuantas semanas quedan para hacer algo?

No basta con maximizar AUC. Un modelo mejor que llega en la semana 16 vale
menos que uno peor en la semana 5, porque en la semana 16 ya no hay margen
de intervencion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.builder import KEYS, build_features_at_week, feature_matrix
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class WeekResult:
    """Metricas de un punto de scoring."""

    week: int
    weeks_remaining: int
    n_train: int
    n_test: int
    prevalence: float
    auc: float
    brier: float
    lift_05: float
    lift_10: float
    lift_20: float

    def as_dict(self) -> dict:
        return asdict(self)


def capture_at(y_true: np.ndarray, score: np.ndarray, pct: float) -> float:
    """Fraccion de casos positivos capturada al priorizar el top `pct`.

    Es la metrica de focalizacion del benchmark ULagos: no cuantos aciertos
    tiene el modelo, sino cuantos de los que van a fallar quedan dentro de
    la lista que la direccion de carrera alcanza a revisar.
    """
    total = float(y_true.sum())
    if total == 0:
        return float("nan")
    n = max(1, int(round(len(score) * pct)))
    top = np.argsort(-score)[:n]
    return float(y_true[top].sum() / total)


def evaluate_lead_time(
    data: dict[str, pd.DataFrame],
    scoring_weeks: list[int],
    weeks_total: int,
    target: str = "failed",
    train_terms: list | None = None,
    test_terms: list | None = None,
    eligibility: callable | None = None,
    population: str = "fixed",
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Entrena y evalua un baseline en cada semana de scoring.

    La validacion es SIEMPRE temporal: los periodos antiguos entrenan y los
    recientes testean (doc maestro 27.1). Un split aleatorio mezclaria
    cohortes y produciria metricas optimistas que no se sostienen en
    produccion.

    `eligibility` filtra quien es un caso predecible; se usa para excluir a
    quienes ya se dieron de baja antes del scoring.

    `population` decide COMO se aplica ese filtro, y la eleccion cambia lo
    que la curva de lead time significa:

      "fixed"  (por defecto) — se fija una unica cohorte: quienes siguen
          matriculados en la ULTIMA semana de scoring, y se evalua a esa
          misma gente en todas las semanas. Es la comparacion honesta: el
          unico factor que varia entre semanas es cuanta informacion hay.

      "rolling" — se recalcula la elegibilidad en cada semana. Refleja mejor
          a quien se le hace scoring en produccion, pero la poblacion cambia
          semana a semana: los casos mas evidentes se van retirando, asi que
          la mejora de AUC mezcla "mas informacion" con "poblacion distinta"
          e infla la pendiente de la curva.

    Devuelve las metricas por semana y las predicciones de cada semana, para
    que la UI pueda mostrar el detalle sin reentrenar.
    """
    if population not in ("fixed", "rolling"):
        raise ValueError(f"population debe ser 'fixed' o 'rolling', no {population!r}")
    outcomes = data["outcomes"]
    terms = sorted(outcomes["term_id"].dropna().unique())
    if train_terms is None or test_terms is None:
        if len(terms) < 2:
            raise ValueError("Se necesitan al menos 2 periodos para validacion temporal")
        split = max(1, len(terms) - 2) if len(terms) > 2 else 1
        train_terms, test_terms = terms[:split], terms[split:]
    log.info("Validacion temporal: entrena %s, testea %s", train_terms, test_terms)

    # Cohorte fija: se resuelve UNA vez, en la ultima semana de scoring, y se
    # reutiliza en todas. Asi el n y la prevalencia son constantes y la curva
    # aisla el efecto de la informacion acumulada.
    cohorte_fija = None
    if eligibility is not None and population == "fixed":
        ancla = max(scoring_weeks)
        cohorte_fija = eligibility(outcomes, ancla)
        log.info("Cohorte fija anclada en la semana %s: %s casos (de %s)",
                 ancla, f"{len(cohorte_fija):,}", f"{len(outcomes):,}")

    rows: list[dict] = []
    preds: dict[int, pd.DataFrame] = {}

    for week in scoring_weeks:
        feats = build_features_at_week(data, week)
        if cohorte_fija is not None:
            elegibles = cohorte_fija
        elif eligibility is not None:
            elegibles = eligibility(outcomes, week)
        else:
            elegibles = outcomes
        df = feats.merge(elegibles[KEYS + [target]], on=KEYS, how="inner")

        tr = df[df["term_id"].isin(train_terms)]
        te = df[df["term_id"].isin(test_terms)]
        if tr.empty or te.empty or tr[target].nunique() < 2:
            log.warning("Semana %s sin datos suficientes, se omite", week)
            continue

        Xtr, Xte, cols = feature_matrix(tr, te)
        ytr = tr[target].to_numpy()
        yte = te[target].to_numpy()

        # Baseline obligatorio del doc maestro 26.1: regresion logistica.
        # Cualquier modelo mas complejo debe superarla para justificarse.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]

        rows.append(
            WeekResult(
                week=week,
                weeks_remaining=weeks_total - week,
                n_train=len(tr), n_test=len(te),
                prevalence=float(yte.mean()),
                auc=float(roc_auc_score(yte, p)),
                brier=float(brier_score_loss(yte, p)),
                lift_05=capture_at(yte, p, 0.05),
                lift_10=capture_at(yte, p, 0.10),
                lift_20=capture_at(yte, p, 0.20),
            ).as_dict()
        )

        detalle = te[KEYS].copy()
        detalle["risk_score"] = p
        detalle["y_true"] = yte
        detalle["week"] = week
        for c in ("campus", "career_code", "attendance_cum", "activity_mean",
                  "submission_rate", "mean_grade", "n_grades"):
            if c in te.columns:
                detalle[c] = te[c].to_numpy()
        preds[week] = detalle

        # Coeficientes del baseline: sirven como control de sanidad. Si el
        # signo de una variable contradice la teoria, casi siempre hay un
        # problema de construccion de features, no un hallazgo.
        coefs = pd.Series(model[-1].coef_[0], index=cols).sort_values()
        log.info("Semana %2d | AUC %.3f | top20%% captura %.1f%% | driver+: %s",
                 week, rows[-1]["auc"], 100 * rows[-1]["lift_20"], coefs.index[-1])

    return pd.DataFrame(rows), preds
