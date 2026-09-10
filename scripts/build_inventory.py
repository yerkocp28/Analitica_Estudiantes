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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_analytics.ingestion.cned import RESOURCE_LABELS  # noqa: E402
from student_analytics.ingestion.paes import SELECTIVITY_LABELS  # noqa: E402
from student_analytics.modeling.benchmark import SHARE_BLOCKS  # noqa: E402
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
    "paes_promedio": "Promedio de Competencia Lectora y Matemática M1 de quienes rindieron; revisar paes_cobertura.",
    "paes_cobertura": "Fracción de la cohorte con puntaje PAES. Donde es baja, los promedios describen una minoría.",
    "estudiantes_total": "MRUN únicos por universidad y año, todos los niveles; excluye registros sin MRUN.",
}

ORIGEN = {**{k: "CNED institucional" for k in RESOURCE_LABELS},
          **{k: "PAES × matrícula (MRUN)" for k in SELECTIVITY_LABELS}}


def describir(columna: str) -> str:
    if columna in AXES:
        return AXES[columna][0]
    if columna in RESOURCE_LABELS:
        return RESOURCE_LABELS[columna]
    if columna in SELECTIVITY_LABELS:
        return SELECTIVITY_LABELS[columna]
    for prefijo, etiqueta in [("area::", "Distribución por área"),
                              ("region::", "Distribución regional"),
                              ("jornada::", "Distribución por jornada"),
                              ("modalidad::", "Distribución por modalidad")]:
        if columna.startswith(prefijo):
            return f"{etiqueta}: {columna.split('::', 1)[1]} (proporción de inscripciones)"
    if columna.startswith("cobertura_"):
        return f"Fracción de registros con dato: {columna.removeprefix('cobertura_')}"
    return columna


def perfil() -> pd.DataFrame:
    d = pd.read_parquet(PROFILES)
    cohortes = sorted(d.cohorte.unique())
    # Entran a la distancia solo escala y las distribuciones con "::".
    prefijos = tuple(p + "::" for lista in SHARE_BLOCKS.values() for p in lista)
    filas = []
    for c in d.columns:
        entra = c in ("log_cohorte", "log_sedes") or c.startswith(prefijos)
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
