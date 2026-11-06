"""Predicciones selladas antes de conocer el resultado: python -m piloto_fan.preregistro

POR QUÉ EXISTE

Los modelos institucionales tienen tres transiciones y su prueba es una
reconstrucción retrospectiva: el resultado ya estaba en el archivo cuando se
eligió el modelo. Ninguna cantidad de cuidado convierte eso en una validación.
La única evaluación que no puede contaminarse es una cuyas predicciones
existían, con fecha y huella, antes de que el resultado llegara a los datos.

Este comando calcula el riesgo 2025-2 → 2026-1 y lo sella:

- las predicciones, una fila por persona, quedan en
  `artifacts/preregistro/<run_id>/`, ignorado por Git;
- su huella SHA-256, el resumen agregado y el protocolo de evaluación se
  escriben en `documentacion/preregistro/`, que SÍ se versiona. El commit que
  los agrega es el sello de tiempo.

Cuando la universidad entregue un extracto con 2026-1, se integra con
`python -m piloto_fan.institucional` y se evalúa con
`python -m piloto_fan.preregistro --evaluar`.

LO QUE NO HACE

No genera alertas. 2026-1 ya ocurrió, así que estas predicciones no sirven
para actuar sobre nadie: existen solo para ser evaluadas. No se muestran ni se
entregan, y el sello versionado no contiene ninguna fila de personas.

DOS VARIANTES, DOS PREGUNTAS

- A: el modelo evaluado en el informe, sin tocarlo. Responde si el informe
  generaliza a un semestre que no existía al sellar. Es la que decide.
- B: el mismo algoritmo e hiperparámetros reentrenado con las tres
  transiciones. Responde si conviene reentrenar con lo más reciente.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .institucional import HOME, ROOT, SOURCE, digest, load
from .modelos_institucionales import (
    CATEGORICAL, CFG, SPEC, TRANSITIONS, datasets, label, load_run, rule, snapshots, write,
)
from .models import bootstrap_comparison, design, estimator, metrics

ORIGEN, OBJETIVO = 202520, 202610
CASOS = {"academico": "academic_target", "matricula": "enrollment_target"}
CONTEXTUAL = {"numeric": [], "categorical": CATEGORICAL}
LOCAL = HOME / "artifacts" / "preregistro"
SELLOS = ROOT / "documentacion" / "preregistro"
COLUMNAS = ["student_key", "source_term", "outcome_term", "career", "campus", "schedule"]

PROTOCOLO = {
    "pregunta": "¿Conserva el modelo del informe su capacidad en un semestre que no existía al sellar?",
    "variantes": {
        "A": "Modelo del informe sin cambios, entrenado con 2024-1→2024-2 y 2024-2→2025-1. Decide el resultado.",
        "B": "Mismo algoritmo e hiperparámetros reentrenado con las tres transiciones. Informa si reentrenar ayuda.",
    },
    "etiquetas": {
        "academico": "Al menos una nota numérica bajo 4 en 2026-1. Negativo solo si todo lo observado es numérico "
                     "y ≥ 4; códigos o ausencia de historia dejan la etiqueta desconocida y fuera de la evaluación.",
        "matricula": "Ningún registro UA en 2026-1, considerando todos los estados y carreras del extracto.",
    },
    "definicion_etiquetas": "piloto_fan.modelos_institucionales.label, la misma del entrenamiento.",
    "reglas_de_comparacion": {
        "academico": "Fracción de notas numéricas bajo 4 en 2025-2.",
        "matricula": "Tasa histórica por carrera, contraída con 30 casos hacia el promedio.",
    },
    "metricas": [
        "Captura al 20% con desempate fraccional.",
        "AUC y precisión promedio.",
        "Brier y Brier skill frente a la prevalencia del entrenamiento de cada variante.",
        "IC 95% bootstrap por persona: 300 repeticiones, semilla 42.",
    ],
    "criterio": "Señal prospectiva si, en la variante A, el límite inferior del IC de la captura al 20% supera "
                "0,20 (azar), el límite inferior del IC de la diferencia contra la regla supera 0 y el Brier "
                "skill es positivo. Si no se cumple, se informa igual y el modelo no se promueve.",
    "invalidacion": [
        "La huella del archivo o del contenido no coincide con este sello.",
        "Cambió la clave de seudonimización (artifacts/institucional/.identity_key): los student_key no se unirían.",
        "El extracto 2026 redefine variables o estados del semestre de origen.",
        "Este protocolo se edita después del commit que lo sella.",
    ],
    "uso": "Las predicciones no se muestran ni se entregan y no generan alertas: 2026-1 ya ocurrió. "
           "Existen solo para ser evaluadas.",
}


def ultima(base: Path) -> Path:
    run = json.loads((base / "latest.json").read_text(encoding="utf-8"))["run_id"]
    if Path(run).name != run or "/" in run or "\\" in run:
        raise ValueError("Identificador de corrida inválido: " + run)
    return base / run


def huella_contenido(pred: pd.DataFrame) -> str:
    """SHA-256 de las predicciones, independiente del orden de filas y del formato de archivo."""
    ordenadas = pred.sort_values(["case", "student_key"]).reset_index(drop=True)
    texto = ordenadas.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def puntuar(frames: dict, modelos: dict, report: dict) -> tuple[pd.DataFrame, dict]:
    """Riesgo ORIGEN → OBJETIVO de cada caso, en las dos variantes y con sus reglas."""
    if OBJETIVO in set(frames["enrollment"].term) | set(frames["courses"].term):
        raise ValueError(f"{OBJETIVO} ya está en los datos: un preregistro exige no conocer el resultado.")
    origen = snapshots(frames)
    origen = origen.loc[origen.term.eq(ORIGEN)].rename(columns={"term": "source_term"})
    if origen.empty:
        raise ValueError(f"No hay estudiantes con notas numéricas en {ORIGEN}.")
    origen["outcome_term"] = OBJETIVO
    data, _, _ = datasets(frames)
    fuentes_a = [origen_t for origen_t, _ in TRANSITIONS[:2]]
    partes, resumen = [], {}
    for caso in CASOS:
        etiquetado = data[caso].dropna(subset=["target"])
        # A usa exactamente el entrenamiento final del informe: entrenamiento + validación.
        train_a = etiquetado.loc[etiquetado.source_term.isin(fuentes_a)]
        winner = report["cases"][caso]["winner"]
        modelo_b = estimator(winner, SPEC, CFG["seed"])
        modelo_b.fit(design(etiquetado, SPEC), etiquetado.target.astype(int))
        p = origen[COLUMNAS].copy()
        p["case"] = caso
        p["risk_a"] = modelos[caso]["model"].predict_proba(design(origen, SPEC))[:, 1]
        p["context_a"] = modelos[caso]["contextual_model"].predict_proba(design(origen, CONTEXTUAL))[:, 1]
        p["rule_a"] = rule(train_a, origen, caso)
        p["risk_b"] = modelo_b.predict_proba(design(origen, SPEC))[:, 1]
        p["rule_b"] = rule(etiquetado, origen, caso)
        partes.append(p)
        por_transicion = etiquetado.groupby(["source_term", "outcome_term"]).target.agg(["size", "mean"])
        # Las transiciones que salen del mismo semestre del año cruzan igual de año que la sellada.
        comparable = etiquetado.loc[etiquetado.source_term.mod(100).eq(ORIGEN % 100)]
        resumen[caso] = {
            "prevalencia_transicion_comparable": float(comparable.target.mean()) if len(comparable) else None,
            "algoritmo": winner, "n": len(p),
            "riesgo_medio_a": float(p.risk_a.mean()), "riesgo_medio_b": float(p.risk_b.mean()),
            "prevalencia_entrenamiento_a": float(train_a.target.mean()),
            "prevalencia_entrenamiento_b": float(etiquetado.target.mean()),
            "entrenamiento_a_n": len(train_a), "entrenamiento_b_n": len(etiquetado),
            "prevalencia_por_transicion": [
                {"origen": int(o), "objetivo": int(t), "n": int(r["size"]), "prevalencia": float(r["mean"])}
                for (o, t), r in por_transicion.iterrows()],
        }
    return pd.concat(partes, ignore_index=True), resumen


def advertencias(resumen: dict) -> list[str]:
    """Lo que ya se sabe que puede fallar, escrito antes de ver el resultado."""
    salida = []
    for caso, r in resumen.items():
        esperado = r["prevalencia_transicion_comparable"]
        if esperado is not None and abs(r["riesgo_medio_a"] - esperado) >= .05:
            sellado, previo = (f"{v:.1%}".replace(".", ",") for v in (r["riesgo_medio_a"], esperado))
            salida.append(
                f"{caso}: el riesgo medio sellado (A) es {sellado}, pero la transición que también cruza "
                f"de año tuvo {previo}. Se espera descalibración; el criterio principal es la captura al 20%, "
                f"que depende del orden de los estudiantes y no del nivel del riesgo.")
    return salida


def sellar() -> Path:
    integracion = ultima(HOME / "artifacts" / "institucional")
    frames, _, manifiesto_i = load(integracion)
    if SOURCE.exists() and digest(SOURCE) != manifiesto_i["source_sha256"]:
        raise ValueError("El Excel cambió: regenerar primero la integración institucional.")
    carpeta_m = ultima(HOME / "artifacts" / "institucional_modelos")
    report, _ = load_run(carpeta_m)  # Verifica las huellas de los modelos antes de cargarlos.
    if report["source_run_id"] != integracion.name:
        raise ValueError("Los modelos no salen de la integración vigente: reentrenar antes de sellar.")
    modelos = {caso: joblib.load(carpeta_m / f"{caso}_model.joblib") for caso in CASOS}
    with threadpool_limits(limits=1):
        pred, resumen = puntuar(frames, modelos, report)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = LOCAL / run_id
    out.mkdir(parents=True, exist_ok=False)
    pred.to_parquet(out / "predicciones.parquet", index=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    sello = {
        "tipo": "preregistro", "run_id": run_id, "creado_utc": now.isoformat(),
        "revision_git_al_sellar": revision,
        "transicion": {"origen": ORIGEN, "objetivo": OBJETIVO},
        "fuentes": {
            "integracion": {"run_id": integracion.name,
                            "manifest_sha256": digest(integracion / "manifest.json"),
                            "excel_sha256": manifiesto_i["source_sha256"]},
            "modelos": {"run_id": carpeta_m.name, "manifest_sha256": digest(carpeta_m / "manifest.json"),
                        "modelos_sha256": {c: digest(carpeta_m / f"{c}_model.joblib") for c in CASOS}},
        },
        "codigo_sha256": {n: digest(HOME / n) for n in ("preregistro.py", "modelos_institucionales.py", "models.py")},
        "predicciones": {
            "archivo": f"piloto_fan/artifacts/preregistro/{run_id}/predicciones.parquet",
            "versionado": False, "filas": len(pred),
            "sha256_archivo": digest(out / "predicciones.parquet"),
            "sha256_contenido": huella_contenido(pred),
        },
        "casos": resumen,
        "advertencias_previas": advertencias(resumen),
        "protocolo": PROTOCOLO,
    }
    write(out / "sello.json", sello)
    SELLOS.mkdir(parents=True, exist_ok=True)
    write(SELLOS / f"{run_id}.json", sello)
    puntero = LOCAL / ("latest." + uuid.uuid4().hex + ".tmp")
    write(puntero, {"run_id": run_id})
    os.replace(puntero, LOCAL / "latest.json")
    return SELLOS / f"{run_id}.json"


def evaluar_predicciones(pred: pd.DataFrame, frames: dict, sello: dict) -> dict:
    """Mide las predicciones selladas contra lo observado en OBJETIVO, sin reajustar nada."""
    if OBJETIVO not in set(frames["enrollment"].term):
        raise ValueError(f"Los datos todavía no traen {OBJETIVO}; no hay nada que evaluar.")
    if huella_contenido(pred) != sello["predicciones"]["sha256_contenido"]:
        raise ValueError("Las predicciones no coinciden con el sello: la evaluación no sería prospectiva.")
    observado = label(pred[["student_key"]].drop_duplicates(), frames, OBJETIVO).set_index("student_key")
    resultado = {}
    for caso, columna in CASOS.items():
        p = pred.loc[pred.case.eq(caso)].join(observado[columna].rename("target"), on="student_key")
        conocidos = p.dropna(subset=["target"])
        res = {"n_sellado": len(p), "n_evaluable": len(conocidos), "n_desconocido": len(p) - len(conocidos)}
        if conocidos.target.nunique() < 2:
            resultado[caso] = {**res, "estado": "no evaluable: el resultado tiene una sola clase"}
            continue
        y = conocidos.target.to_numpy(int)
        for v in ("a", "b"):
            previa = sello["casos"][caso][f"prevalencia_entrenamiento_{v}"]
            res[v] = {"modelo": metrics(y, conocidos[f"risk_{v}"], previa),
                      "regla": metrics(y, conocidos[f"rule_{v}"], previa),
                      "incertidumbre": bootstrap_comparison(y, conocidos[f"risk_{v}"].to_numpy(),
                                                            conocidos[f"rule_{v}"].to_numpy(), CFG)}
        res["contexto_a"] = metrics(y, conocidos.context_a, sello["casos"][caso]["prevalencia_entrenamiento_a"])
        a = res["a"]
        res["senal_prospectiva"] = bool(a["incertidumbre"]["capture_20_ci"][0] > .2
                                        and a["incertidumbre"]["capture_difference_vs_rule_ci"][0] > 0
                                        and (a["modelo"]["brier_skill"] or 0) > 0)
        res["estado"] = "señal prospectiva" if res["senal_prospectiva"] else "sin señal prospectiva"
        resultado[caso] = res
    return resultado


def evaluar() -> Path:
    carpeta = ultima(LOCAL)
    sello = json.loads((SELLOS / f"{carpeta.name}.json").read_text(encoding="utf-8"))  # El versionado manda.
    if digest(carpeta / "predicciones.parquet") != sello["predicciones"]["sha256_archivo"]:
        raise ValueError("El archivo de predicciones no coincide con el sello versionado.")
    frames, _, manifiesto = load(ultima(HOME / "artifacts" / "institucional"))
    resultado = evaluar_predicciones(pd.read_parquet(carpeta / "predicciones.parquet"), frames, sello)
    salida = SELLOS / f"{carpeta.name}.evaluacion.json"
    write(salida, {"run_id": carpeta.name, "evaluado_utc": datetime.now(timezone.utc).isoformat(),
                   "integracion_evaluada": manifiesto["run_id"], "protocolo": "el del sello, sin cambios",
                   "casos": resultado})
    return salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluar", action="store_true", help="evalúa el último sello contra 2026-1")
    destino = evaluar() if parser.parse_args().evaluar else sellar()
    print(f"-> {destino.relative_to(ROOT)}", flush=True)
