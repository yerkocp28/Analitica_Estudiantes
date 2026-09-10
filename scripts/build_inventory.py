"""Regenera los inventarios de variables y de campos crudos.

    python scripts/build_inventory.py

Se hacen por script y no a mano porque envejecen: la nota que decía que los
ratios usaban la cohorte de ingreso como denominador siguió ahí después de
que el código pasara a usar la matrícula total. Un inventario desactualizado
es peor que no tenerlo, porque se lee como si fuera cierto.

Salidas en documentacion/datos/:
  inventario_perfil_benchmark.csv        una fila por variable del perfil
  inventario_campos_fuentes_benchmark.csv  una fila por campo crudo de cada base
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_analytics.ingestion.cned import RESOURCE_LABELS  # noqa: E402
from student_analytics.ingestion.cohortes import COHORT_LABELS  # noqa: E402
from student_analytics.ingestion.paes import SELECTIVITY_LABELS  # noqa: E402
from student_analytics.modeling.benchmark import (NUMERIC_BLOCKS,  # noqa: E402
                                                  SHARE_BLOCKS)
from student_analytics.ui.benchmark import AXES  # noqa: E402

OUT = ROOT / "documentacion" / "datos"
PROFILES = ROOT / "data" / "results" / "benchmark_profiles.parquet"

# Advertencias que no se deducen del dato y hay que arrastrar con él.
NOTAS = {
    "arancel_mediano": "Registros en UF convertidos a CLP con factores anuales aproximados de config/benchmark.yml.",
    "matricula_mediana": "Registros en UF convertidos a CLP con factores anuales aproximados de config/benchmark.yml.",
    "carreras_acreditadas": "Proporción de inscripciones en carreras acreditadas; no proporción de carreras únicas.",
    "acreditacion_cned": "Foto del catálogo CNED, no una serie histórica por cohorte.",
    "docentes_jce": "Derivado: jornada completa + 0,5 media + 0,25 por hora.",
    "docentes_por_100_alumnos": "Denominador: MRUN únicos de toda la institución y todos los niveles, no la cohorte de ingreso.",
    "m2_construido_por_alumno": "Denominador: matrícula total institucional.",
    "pc_por_100_alumnos": "Denominador: matrícula total institucional.",
    "ejemplares_por_alumno": "Denominador: matrícula total institucional.",
    "cruch": "El CRUCH admite privadas desde la Ley 21.091; pertenecer a él ya no equivale a ser 'tradicional'.",
    "paes_promedio": "Promedio de Competencia Lectora y Matemática M1 de quienes rindieron; revisar paes_cobertura. NO comparable entre cohortes: la PDT (2021-2022) va de 150 a 850 y la PAES de 100 a 1000.",
    "paes_percentil_promedio": "Percentil nacional dentro de cada cohorte. Es la única variable de selectividad comparable a través del cambio de escala de 2023.",
    "paes_instrumento": "Prueba que rindió esa cohorte: PDT en 2021-2022, PAES desde 2023. Antes de 2021 no hay dato abierto (la PSU no se publica).",
    "paes_p25": "No comparable entre cohortes por el cambio de escala de 2023.",
    "paes_p75": "No comparable entre cohortes por el cambio de escala de 2023.",
    "paes_rango_intercuartil": "Dispersión en puntos de la prueba; no comparable entre cohortes por el cambio de escala de 2023.",
    "paes_cobertura": "Fracción de la cohorte con puntaje PAES. Donde es baja, los promedios describen una minoría.",
    "estudiantes_total": "MRUN únicos por universidad y año, todos los niveles; excluye registros sin MRUN.",
    "titulacion_oportuna": "Describe a la promoción que EGRESA, no a la cohorte del perfil. No ordena por calidad: correlaciona -0,08 con PAES y -0,19 con acreditación.",
    "titulacion_cohorte_anio": "Cohorte de ingreso a la que se refieren los indicadores de titulación longitudinal; es la misma para todas las universidades.",
    "titulacion_cohorte_carrera": "Tasa sobre ingresantes con horizonte observable, no sobre la cohorte completa.",
    "titulacion_cohorte_universidad": "Incluye a quien cambió de carrera dentro de la misma universidad.",
    "titulacion_cohorte_sistema": "Incluye traslados a otra universidad; sin este nivel, todo traslado se contaría como deserción.",
    "titulacion_cohorte_cobertura": "Fracción de la cohorte cuyo horizonte (nominal + holgura) cabe en los datos. Bajo 0,90 se anulan las tasas: quedarían descritas sólo por las carreras cortas.",
}

ORIGEN = {**{k: "CNED institucional" for k in RESOURCE_LABELS},
          **{k: "PAES × matrícula (MRUN)" for k in SELECTIVITY_LABELS},
          **{k: "Matrícula × titulados (MRUN, longitudinal)" for k in COHORT_LABELS},
          "titulacion_cohorte_anio": "Matrícula × titulados (MRUN, longitudinal)"}


def describir(columna: str) -> str:
    if columna in AXES:
        return AXES[columna][0]
    if columna in RESOURCE_LABELS:
        return RESOURCE_LABELS[columna]
    if columna in SELECTIVITY_LABELS:
        return SELECTIVITY_LABELS[columna]
    if columna in COHORT_LABELS:
        return COHORT_LABELS[columna]
    if columna == "titulacion_cohorte_anio":
        return "Cohorte de ingreso de referencia para la titulación longitudinal"
    for prefijo, etiqueta in [("area::", "Distribución por área"),
                              ("region::", "Distribución regional"),
                              ("jornada::", "Distribución por jornada"),
                              ("modalidad::", "Distribución por modalidad")]:
        if columna.startswith(prefijo):
            return f"{etiqueta}: {columna.split('::', 1)[1]} (proporción de inscripciones)"
    if columna.startswith("cobertura_"):
        return f"Fracción de registros con dato: {columna.removeprefix('cobertura_')}"
    return columna


def _columnas_del_clustering(d: pd.DataFrame) -> set[str]:
    """Qué variables entran hoy a la distancia, según la configuración.

    Se deriva de `block_weights` y no de una lista fija: cuando se activó el
    bloque de selectividad, una lista fija habría seguido informando los
    bloques antiguos sin que nada avisara.
    """
    pesos = yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
    activos = set((pesos or {}).get("block_weights") or {})
    columnas: set[str] = set()
    for bloque, variables in NUMERIC_BLOCKS.items():
        if bloque in activos:
            columnas |= {v for v in variables if v in d.columns}
    for bloque, prefijos in SHARE_BLOCKS.items():
        if bloque in activos:
            columnas |= {c for c in d.columns
                         if any(c.startswith(p + "::") for p in prefijos)}
    return columnas


def perfil() -> pd.DataFrame:
    d = pd.read_parquet(PROFILES)
    cohortes = sorted(d.cohorte.unique())
    del_clustering = _columnas_del_clustering(d)
    filas = []
    for c in d.columns:
        entra = c in del_clustering
        fila = {"variable": c, "descripcion": describir(c),
                "origen": ORIGEN.get(c, "Matrícula SIES / derivada"),
                "entra_clustering": bool(entra),
                "eje_disponible_app": c in AXES,
                "nota": NOTAS.get(c, "")}
        for año in cohortes:
            sub = d[d.cohorte.eq(año)]
            fila[f"universidades_con_dato_{año}"] = int(sub[c].notna().sum())
            fila[f"universidades_total_{año}"] = int(len(sub))
        filas.append(fila)
    return pd.DataFrame(filas)


def campos() -> pd.DataFrame:
    """Campos crudos de cada base descargada, sin leerlas enteras."""
    filas = []
    mineduc = ROOT / "data/external/mineduc"
    for carpeta in sorted(p for p in mineduc.glob("*") if p.is_dir()):
        for csv in sorted(carpeta.rglob("*.csv")):
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    head = pd.read_csv(csv, sep=";", encoding=enc, nrows=0)
                    if len(head.columns) > 2:
                        break
                except Exception:
                    continue
            else:
                continue
            filas += [{"fuente": "Mineduc", "base_hoja": carpeta.name,
                       "archivo": csv.name, "campo_original": col}
                      for col in head.columns]
    cned = ROOT / "data/external/cned/INDICES_Institucional_2005-2025.xlsx"
    if cned.exists():
        for hoja, tabla in pd.read_excel(cned, sheet_name=None, nrows=0).items():
            filas += [{"fuente": "CNED", "base_hoja": hoja, "archivo": cned.name,
                       "campo_original": col} for col in tabla.columns]
    return pd.DataFrame(filas)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PROFILES.exists():
        raise SystemExit("Faltan los perfiles: python scripts/build_benchmark.py")

    p = perfil()
    p.to_csv(OUT / "inventario_perfil_benchmark.csv", index=False, encoding="utf-8-sig")
    print(f"inventario_perfil_benchmark.csv       {len(p):4d} variables "
          f"({int(p.entra_clustering.sum())} al clustering, "
          f"{int(p.eje_disponible_app.sum())} como eje)")

    c = campos()
    c.to_csv(OUT / "inventario_campos_fuentes_benchmark.csv", index=False, encoding="utf-8-sig")
    print(f"inventario_campos_fuentes_benchmark.csv {len(c):4d} campos "
          f"en {c.base_hoja.nunique()} bases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
