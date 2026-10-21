"""Agregados publicables del extracto institucional UA.

    python scripts/export_institutional_aggregates.py

POR QUÉ EXISTE

El piloto FAN corre en local porque sus artefactos contienen resultados por
persona de 40.590 estudiantes, y eso no se versiona ni se publica. Pero el
equipo de desarrollo necesita ver los resultados sin instalar el proyecto.

Este script produce la versión publicable: **solo agregados**, sin
`student_key`, sin notas individuales, sin predicciones. Lo que sale de aquí
puede vivir en `deploy/data/` y llegar a una aplicación web.

LAS TRES REGLAS QUE SE APLICAN AL ESCRIBIR

1. **Piso de celda.** Ningún grupo con menos de `--minimo` personas se publica.
   Es el mismo mínimo que la interfaz del piloto ya aplica, y aquí se impone
   antes de escribir, no al mostrar.
2. **Sin identificadores.** Se cuenta `nunique()` sobre `student_key` y la
   columna se descarta. Ninguna salida la contiene.
3. **Lo que no se puede afirmar, no se nombra así.** La fracción de notas bajo
   4 se llama `bajo4` y no «reprobación»: mientras P01 siga sin respuesta no
   sabemos qué significan los 18.288 resultados codificados ni cuál es la
   definición institucional de aprobación.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("export_institutional_aggregates")

ORIGEN = REPO_ROOT / "piloto_fan" / "artifacts" / "institucional"
SALIDA = REPO_ROOT / "data" / "results" / "institucional_agregado"


def _ultima() -> Path | None:
    puntero = ORIGEN / "latest.json"
    if not puntero.exists():
        return None
    run = json.loads(puntero.read_text(encoding="utf-8")).get("run_id")
    destino = ORIGEN / run if run else None
    return destino if destino and destino.exists() else None


def _agregar(d: pd.DataFrame, por: list[str], minimo: int) -> pd.DataFrame:
    """Agrega por las dimensiones pedidas y suprime las celdas chicas."""
    g = d.groupby(por, dropna=False, observed=True)
    out = g.agg(
        personas=("student_key", "nunique"),
        registros=("student_key", "size"),
        con_nota=("grade_numeric", "count"),
        nota_media=("grade_numeric", "mean"),
        bajo4=("numeric_below4", "mean"),
    ).reset_index()
    out["cobertura_nota"] = out.con_nota / out.registros
    antes = len(out)
    out = out.loc[out.personas >= minimo].copy()
    suprimidos = antes - len(out)
    for c in ("nota_media", "bajo4", "cobertura_nota"):
        out[c] = out[c].round(4)
    out.attrs["suprimidos"] = suprimidos
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimo", type=int, default=10,
                    help="personas mínimas por celda publicada")
    ap.add_argument("--solo-fan", action="store_true",
                    help="restringe al alcance FAN propuesto")
    args = ap.parse_args()

    base = _ultima()
    if base is None:
        print("No hay corrida institucional. Genera antes:\n"
              "  python -m piloto_fan.institucional")
        return 1

    resultados = pd.read_parquet(base / "resultados_asignatura_periodo.parquet")
    mart = pd.read_parquet(base / "mart_estudiante_periodo.parquet")

    # El mart tiene una fila por persona-periodo con su carrera y campus; los
    # resultados vienen por persona-periodo-asignatura. Se unen por ahi.
    llaves = ["student_key", "term"]
    # `period`, `level` y `level_code` existen en LAS DOS tablas. Unir sin
    # mirar las sufijaba a period_x/period_y y los cortes no las encontraban:
    # del mart se toma solo lo que los resultados no traen ya.
    deseadas = ("career_code", "career", "campus_code", "campus", "schedule",
                "period", "year", "regime", "fan_proposed")
    contexto = [c for c in deseadas
                if c in mart.columns and c not in resultados.columns]
    d = resultados.merge(mart[llaves + contexto].drop_duplicates(llaves),
                         on=llaves, how="left")
    faltan = [c for c in deseadas if c not in d.columns]
    if faltan:
        log.warning("sin contexto para: %s", ", ".join(faltan))
    if args.solo_fan and "fan_proposed" in d.columns:
        d = d.loc[d.fan_proposed.fillna(False)]
        log.info("alcance FAN: %s registros", f"{len(d):,}")

    cortes = {
        "por_periodo": ["year", "period"],
        "por_carrera": ["career_code", "career"],
        "por_campus": ["campus_code", "campus"],
        "por_carrera_periodo": ["career_code", "career", "period"],
        "por_asignatura": ["course_code", "course_name"],
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    for viejo in SALIDA.glob("*.parquet"):
        viejo.unlink()

    resumen, total_suprimidos = [], 0
    for nombre, por in cortes.items():
        presentes = [c for c in por if c in d.columns]
        if not presentes:
            log.warning("%s: sin columnas disponibles, se omite", nombre)
            continue
        tabla = _agregar(d, presentes, args.minimo)
        # Control duro: ninguna salida puede llevar identificadores.
        assert "student_key" not in tabla.columns
        tabla.to_parquet(SALIDA / f"{nombre}.parquet", index=False)
        total_suprimidos += tabla.attrs["suprimidos"]
        resumen.append({"corte": nombre, "filas": len(tabla),
                        "celdas_suprimidas": tabla.attrs["suprimidos"]})
        log.info("%s: %d filas, %d celdas suprimidas", nombre, len(tabla),
                 tabla.attrs["suprimidos"])

    calidad = json.loads((base / "quality.json").read_text(encoding="utf-8"))
    (SALIDA / "meta.json").write_text(json.dumps({
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "corrida_origen": base.name,
        "minimo_personas_celda": args.minimo,
        "alcance": "FAN propuesto" if args.solo_fan else "todo el extracto",
        "personas_en_el_extracto": calidad.get("enrollment_people"),
        "registros_academicos": calidad.get("history_rows"),
        "periodos": calidad.get("periods"),
        "advertencia": ("La fracción bajo 4 es descriptiva sobre notas numéricas. "
                        "NO es la tasa oficial de reprobación: los resultados "
                        "codificados y la definición institucional de aprobación "
                        "siguen pendientes (P01, P02)."),
        "cortes": resumen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    peso = sum(f.stat().st_size for f in SALIDA.glob("*")) / 1e6
    print(f"{len(resumen)} cortes · {peso:.2f} MB · {total_suprimidos} celdas "
          f"suprimidas por tener menos de {args.minimo} personas")
    for r in resumen:
        print(f"  {r['corte']:24s} {r['filas']:>6,} filas")
    print(f"-> {SALIDA.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
