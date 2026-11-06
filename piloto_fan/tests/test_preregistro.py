import json

import numpy as np
import pandas as pd
import pytest

from piloto_fan.institucional import digest
from piloto_fan.preregistro import (
    LOCAL, OBJETIVO, ORIGEN, SELLOS, evaluar_predicciones, huella_contenido, puntuar,
)

N = 300
SIN_HISTORIA = 10  # k0..k9 no aparecen en 2026-1.


def predicciones(seed=0):
    rng = np.random.default_rng(seed)
    riesgo = rng.uniform(size=N)
    partes = []
    for caso in ("academico", "matricula"):
        partes.append(pd.DataFrame({
            "student_key": [f"k{i}" for i in range(N)], "source_term": ORIGEN, "outcome_term": OBJETIVO,
            "career": "Comercial", "campus": "Talca", "schedule": "Diurno", "case": caso,
            "risk_a": riesgo, "context_a": 0.5, "rule_a": rng.uniform(size=N),
            "risk_b": riesgo, "rule_b": rng.uniform(size=N),
        }))
    return pd.concat(partes, ignore_index=True)


def sello_de(pred):
    return {"predicciones": {"sha256_contenido": huella_contenido(pred)},
            "casos": {c: {"prevalencia_entrenamiento_a": .3, "prevalencia_entrenamiento_b": .3}
                      for c in ("academico", "matricula")}}


def observado(pred):
    """2026-1 sintético: nota bajo 4 si el riesgo sellado era alto; k0..k9 sin registro."""
    riesgo = pred.loc[pred.case.eq("academico")].set_index("student_key").risk_a
    courses = pd.DataFrame([
        {"student_key": k, "term": OBJETIVO, "nrc": 1, "grade_numeric": 3.0 if r > .7 else 5.5}
        for k, r in riesgo.items() if int(k[1:]) >= SIN_HISTORIA])
    return {"courses": courses, "enrollment": courses[["student_key", "term"]].drop_duplicates()}


def test_la_huella_no_depende_del_orden_y_detecta_cualquier_cambio():
    pred = predicciones()
    assert huella_contenido(pred.sample(frac=1, random_state=1)) == huella_contenido(pred)
    alterada = pred.copy()
    alterada.loc[0, "risk_a"] += 1e-6
    assert huella_contenido(alterada) != huella_contenido(pred)


def test_no_se_sella_si_el_resultado_ya_esta_en_los_datos():
    frames = {"enrollment": pd.DataFrame({"student_key": ["a"], "term": [OBJETIVO]}),
              "courses": pd.DataFrame({"student_key": ["a"], "term": [ORIGEN]})}
    with pytest.raises(ValueError, match="ya está en los datos"):
        puntuar(frames, {}, {})


def test_no_se_evalua_sin_el_semestre_objetivo():
    pred = predicciones()
    frames = observado(pred)
    frames["enrollment"] = frames["enrollment"].assign(term=ORIGEN)
    with pytest.raises(ValueError, match="todavía no traen"):
        evaluar_predicciones(pred, frames, sello_de(pred))


def test_predicciones_alteradas_despues_del_sello_no_se_evaluan():
    pred = predicciones()
    sello = sello_de(pred)
    pred.loc[pred.case.eq("academico"), "risk_a"] = 1 - pred.risk_a
    with pytest.raises(ValueError, match="no coinciden con el sello"):
        evaluar_predicciones(pred, observado(pred), sello)


def test_la_evaluacion_usa_las_etiquetas_del_entrenamiento():
    pred = predicciones()
    r = evaluar_predicciones(pred, observado(pred), sello_de(pred))
    # Sin historia en 2026-1: académico desconocido (no es éxito), matrícula = ausencia observada.
    assert r["academico"]["n_desconocido"] == SIN_HISTORIA
    assert r["matricula"]["n_desconocido"] == 0
    assert r["matricula"]["a"]["modelo"]["events"] == SIN_HISTORIA
    # El riesgo sellado separa el resultado académico; el de matrícula es independiente del evento.
    assert r["academico"]["estado"] == "señal prospectiva"
    assert r["matricula"]["estado"] == "sin señal prospectiva"


def test_el_sello_versionado_coincide_con_lo_local_y_no_trae_personas():
    if not (LOCAL / "latest.json").exists():
        pytest.skip("Requiere sellar con python -m piloto_fan.preregistro.")
    run = json.loads((LOCAL / "latest.json").read_text(encoding="utf-8"))["run_id"]
    texto = (SELLOS / f"{run}.json").read_text(encoding="utf-8")
    sello = json.loads(texto)
    archivo = LOCAL / run / "predicciones.parquet"
    assert digest(archivo) == sello["predicciones"]["sha256_archivo"]
    pred = pd.read_parquet(archivo)
    assert huella_contenido(pred) == sello["predicciones"]["sha256_contenido"]
    assert not any(k in texto for k in pred.student_key.unique())
    assert pred.source_term.eq(ORIGEN).all() and pred.outcome_term.eq(OBJETIVO).all()
