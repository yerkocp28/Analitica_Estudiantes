"""Archiva lo recuperable de cada corrida del piloto: python scripts/archive_pilot_runs.py

POR QUÉ

Las corridas del piloto pesan ~950 MB y están fuera del control de versiones,
con razón: contienen predicciones y datos individuales. Pero eso significa que
si se pierde el disco se pierde también la **trazabilidad** — qué modelo ganó,
con qué métricas, sobre qué partición y con qué huellas de las fuentes.

Este script versiona solo esa trazabilidad: unos pocos KB por corrida, sin una
sola fila de personas. Los modelos se pueden reentrenar; saber qué se decidió y
con qué evidencia, no.

QUÉ SE COPIA Y QUÉ NO

Se copian únicamente los JSON de metadatos: `report.json`, las *model cards*,
`feature_registry.json`, `manifest.json` y `config.json`. **Nunca** `.parquet`
ni `.joblib`: ahí viven las predicciones del test reservado y los conjuntos
derivados.

Además se aplica el mismo piso de celda que usa la interfaz del piloto: un
subgrupo con menos de `--minimo` personas se suprime antes de escribir. Hoy el
menor tiene 38 y no se suprimiría nada, pero el guardia tiene que existir antes
de que haga falta, no después.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("archive_pilot_runs")

PILOTO = REPO_ROOT / "piloto_fan" / "artifacts"
DESTINO = REPO_ROOT / "documentacion" / "corridas"

FAMILIAS = {
    "publico": PILOTO,
    "institucional": PILOTO / "institucional",
    "institucional_modelos": PILOTO / "institucional_modelos",
}
# Solo metadatos. Si aparece un JSON nuevo con grano individual, hay que
# agregarlo aqui a proposito; la lista blanca es la salvaguarda.
PERMITIDOS = {"report.json", "feature_registry.json", "manifest.json",
              "config.json", "quality.json", "pendientes.json"}
SUFIJOS_PERMITIDOS = ("_model_card.json",)

# Claves que contienen listas de subgrupos con conteos por celda.
CLAVES_SUBGRUPOS = ("subgroups", "subgrupos")


def _suprimir_celdas_chicas(valor, minimo: int, contador: list[int]):
    """Quita recursivamente los subgrupos bajo el mínimo de personas."""
    if isinstance(valor, dict):
        salida = {}
        for k, v in valor.items():
            if k in CLAVES_SUBGRUPOS and isinstance(v, list):
                grandes = [g for g in v
                           if not (isinstance(g, dict) and isinstance(g.get("n"), (int, float))
                                   and g["n"] < minimo)]
                contador[0] += len(v) - len(grandes)
                salida[k] = grandes
            else:
                salida[k] = _suprimir_celdas_chicas(v, minimo, contador)
        return salida
    if isinstance(valor, list):
        return [_suprimir_celdas_chicas(v, minimo, contador) for v in valor]
    return valor


def _archivos(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.glob("*.json")
                  if p.name in PERMITIDOS or p.name.endswith(SUFIJOS_PERMITIDOS))


def _corridas(carpeta: Path, todas: bool) -> list[Path]:
    if not carpeta.exists():
        return []
    if todas:
        return sorted(p for p in carpeta.iterdir()
                      if p.is_dir() and _archivos(p))
    puntero = carpeta / "latest.json"
    if not puntero.exists():
        return []
    run = json.loads(puntero.read_text(encoding="utf-8")).get("run_id")
    destino = carpeta / run if run else None
    return [destino] if destino and destino.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true",
                    help="archiva todas las corridas, no solo la última verificada")
    ap.add_argument("--minimo", type=int, default=10,
                    help="personas mínimas por subgrupo (default: el del piloto)")
    args = ap.parse_args()

    indice, suprimidos_total, bytes_total = [], 0, 0
    for familia, carpeta in FAMILIAS.items():
        for corrida in _corridas(carpeta, args.todas):
            archivos = _archivos(corrida)
            if not archivos:
                continue
            salida = DESTINO / familia / corrida.name
            salida.mkdir(parents=True, exist_ok=True)
            contador = [0]
            for origen in archivos:
                datos = json.loads(origen.read_text(encoding="utf-8"))
                limpio = _suprimir_celdas_chicas(datos, args.minimo, contador)
                texto = json.dumps(limpio, ensure_ascii=False, indent=2)
                (salida / origen.name).write_text(texto, encoding="utf-8")
                bytes_total += len(texto.encode("utf-8"))
            suprimidos_total += contador[0]
            indice.append({"familia": familia, "run_id": corrida.name,
                           "archivos": len(archivos),
                           "subgrupos_suprimidos": contador[0]})
            log.info("%s/%s: %d archivos%s", familia, corrida.name, len(archivos),
                     f", {contador[0]} subgrupos suprimidos" if contador[0] else "")

    if not indice:
        print("No hay corridas del piloto que archivar.\n"
              "  python -m piloto_fan.build\n"
              "  python -m piloto_fan.institucional\n"
              "  python -m piloto_fan.modelos_institucionales")
        return 1

    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / "indice.json").write_text(json.dumps(
        {"generado_utc": datetime.now(timezone.utc).isoformat(),
         "minimo_personas_subgrupo": args.minimo,
         "contenido": "Solo metadatos: reportes, model cards, registros de "
                      "variables y manifiestos. Sin predicciones ni personas.",
         "corridas": indice}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(indice)} corridas archivadas · {bytes_total / 1024:.0f} KB")
    if suprimidos_total:
        print(f"  {suprimidos_total} subgrupos suprimidos por tener menos de "
              f"{args.minimo} personas")
    for fila in indice:
        print(f"  {fila['familia']:22s} {fila['run_id']:26s} {fila['archivos']} archivos")
    print(f"-> {DESTINO.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
