"""Exporta un resumen agregado de los modelos del piloto FAN.

    python scripts/export_pilot_summary.py

POR QUÉ ES UN PASO EXPLÍCITO Y NO AUTOMÁTICO

El piloto declara que **no actualiza por su cuenta el Streamlit publicado**, y
esa decisión se respeta: este script no lo llama nadie solo. Hay que correrlo a
propósito, cuando se quiere que la app principal muestre los resultados de una
corrida ya revisada.

QUÉ SE EXPORTA Y QUÉ NO

Solo **agregados**: nombre del caso, algoritmo ganador, métricas del test,
tamaños, veredicto y límites. Nunca predicciones, nunca personas, nunca el
identificador local del piloto. Lo que entra en `data/results/` puede terminar
en `deploy/data/`, así que la regla es la misma que para el resto: agregados
institucionales, jamás grano individual.

Lee el puntero `latest.json` de cada familia de corridas, que apunta a la
última ejecución terminada y verificada.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("export_pilot_summary")

PILOTO = REPO_ROOT / "piloto_fan" / "artifacts"
SALIDA = REPO_ROOT / "data" / "results" / "modelos_resumen.json"

# Metricas que se publican. Deliberadamente acotado: nada que permita
# reconstruir una prediccion individual.
METRICAS = ("n", "events", "prevalence", "auc", "average_precision",
            "brier", "brier_skill", "capture_20")

FAMILIAS = {
    "publico": {
        "carpeta": PILOTO,
        "fuente": "Datos abiertos Mineduc cruzados por MRUN",
        "etiquetas": {"continuidad": "No continuidad al año siguiente",
                      "titulacion": "No titulación en el horizonte"},
    },
    "institucional": {
        "carpeta": PILOTO / "institucional_modelos",
        "fuente": "Extracto institucional UA 2024-2025",
        "etiquetas": {"academico": "Resultado académico del semestre siguiente",
                      "matricula": "No presencia en matrícula del semestre siguiente"},
    },
}


def _ultima(carpeta: Path) -> Path | None:
    puntero = carpeta / "latest.json"
    if not puntero.exists():
        return None
    run = json.loads(puntero.read_text(encoding="utf-8")).get("run_id")
    destino = carpeta / run if run else None
    return destino if destino and (destino / "report.json").exists() else None


def _resumen_caso(nombre: str, caso: dict, meta: dict) -> dict:
    test = caso.get("test") or {}
    regla = caso.get("historical_rule") or {}
    fila = {
        "caso": nombre,
        "objetivo": meta["etiquetas"].get(nombre, nombre),
        "fuente": meta["fuente"],
        "estado": caso.get("status"),
        "algoritmo": caso.get("winner"),
    }
    fila.update({k: test[k] for k in METRICAS if k in test})
    # La regla historica es la referencia honesta: superar al azar no basta,
    # hay que superar a priorizar por la tasa historica de la carrera.
    for k in ("auc", "capture_20", "brier_skill"):
        if k in regla:
            fila[f"regla_{k}"] = regla[k]
    return fila


def main() -> int:
    casos, corridas = [], {}
    for familia, meta in FAMILIAS.items():
        base = _ultima(meta["carpeta"])
        if base is None:
            log.warning("%s: sin corrida verificada, se omite", familia)
            continue
        reporte = json.loads((base / "report.json").read_text(encoding="utf-8"))
        corridas[familia] = {"run_id": reporte.get("run_id", base.name),
                             "creado_utc": reporte.get("created_utc")}
        for nombre, caso in (reporte.get("cases") or {}).items():
            casos.append(_resumen_caso(nombre, caso, meta))
            log.info("%s/%s: %s (%s)", familia, nombre,
                     caso.get("status"), caso.get("winner"))

    if not casos:
        print("No hay corridas del piloto. Genera primero:\n"
              "  python -m piloto_fan.build\n"
              "  python -m piloto_fan.modelos_institucionales")
        return 1

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(
        {"generado_utc": datetime.now(timezone.utc).isoformat(),
         "corridas": corridas, "casos": casos},
        ensure_ascii=False, indent=2), encoding="utf-8")

    con_senial = [c for c in casos if c.get("estado") != "experimental"]
    print(f"{len(casos)} casos exportados · {len(con_senial)} con señal retrospectiva")
    for c in casos:
        auc = c.get("auc")
        print(f"  {c['caso']:14s} {str(c.get('estado')):22s} "
              f"AUC {auc:.3f}" if auc else f"  {c['caso']:14s} {c.get('estado')}")
    print(f"-> {SALIDA.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
