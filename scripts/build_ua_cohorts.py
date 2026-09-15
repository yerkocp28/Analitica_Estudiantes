"""Dataset analítico de ingresantes reales de la UA: python scripts/build_ua_cohorts.py

Una fila por persona que ingresó a la Universidad Autónoma, con lo que se sabía
de ella **al matricularse** y con su resultado observado al año siguiente.

POR QUÉ IMPORTA

El proyecto se venía tratando como bloqueado por la falta de Banner y Canvas, y
lo está para el grano semanal. Pero la población del proyecto —los estudiantes
de la UA— sí está en datos abiertos: 19 cohortes, casi cien mil ingresantes,
con outcome observado. Eso permite segmentar y modelar sobre estudiantes
reales, no sintéticos.

Lo que NO permite es predecir reprobación de una asignatura ni desenganche:
para eso hace falta el comportamiento semanal dentro del curso.

COBERTURA DESIGUAL, DECLARADA

    matrícula        2007-2025   las 19 cohortes
    admisión         2021-2025   PAES/PDT; la PSU no es dato abierto
    socioeconómico   2023-2024   becas, SEP, asistencia de enseñanza media

Las features que no existen para una cohorte quedan en NaN, no en cero. El
alcance lo elige quien analiza, y `cobertura_*` deja explícito cuánto hay.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("build_ua_cohorts")

SLIM = REPO_ROOT / "data" / "interim" / "matricula"
MINEDUC = REPO_ROOT / "data" / "external" / "mineduc"
SALIDA = REPO_ROOT / "data" / "results" / "ua_ingresantes.parquet"
UA = "AUTONOMA"

COLUMNAS = ["mrun", "gen_alu", "anio_ing_carr_ori", "sem_ing_carr_ori",
            "cod_inst", "nomb_inst", "cod_sede", "nomb_sede", "cod_carrera",
            "nomb_carrera", "jornada", "modalidad", "tipo_plan_carr",
            "dur_total_carr", "region_sede", "nivel_carrera_1",
            "area_conocimiento", "valor_arancel", "formato_valores"]


def _cohorte(anio: int) -> pd.DataFrame | None:
    ruta = SLIM / f"matricula_{anio}.parquet"
    if not ruta.exists():
        return None
    d = pd.read_parquet(ruta, columns=[c for c in COLUMNAS])
    ua = d.loc[d.nomb_inst.str.contains(UA, na=False) & d.anio_ing_carr_ori.eq(anio)].copy()
    if ua.empty:
        return None
    ua["cohorte"] = anio
    ua["mrun"] = pd.to_numeric(ua.mrun, errors="coerce")
    # La nominal viene en semestres; el resto del proyecto razona en anios.
    ua["duracion_nominal_anios"] = ua.dur_total_carr / 2
    return ua.drop(columns=["dur_total_carr", "nomb_inst"])


def _continuidad(cohorte: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Los tres niveles anidados de continuidad, como en el resto del proyecto.

    `sistema` se busca en el índice de seguimiento, que NO está filtrado por
    tipo de institución: cambiarse a un CFT o a un IP no es deserción.
    """
    ruta = SLIM / f"seguimiento_{anio + 1}.parquet"
    if not ruta.exists():
        for col in ("sigue_carrera", "sigue_universidad", "sigue_sistema"):
            cohorte[col] = pd.NA
        return cohorte
    sig = pd.read_parquet(ruta, columns=["mrun", "cod_inst", "cod_carrera"])
    sig["mrun"] = pd.to_numeric(sig.mrun, errors="coerce")
    sig = sig.dropna(subset=["mrun"])
    for nombre, claves in [("sigue_carrera", ["mrun", "cod_inst", "cod_carrera"]),
                           ("sigue_universidad", ["mrun", "cod_inst"]),
                           ("sigue_sistema", ["mrun"])]:
        idx = pd.MultiIndex.from_frame(sig[claves].dropna().drop_duplicates())
        cohorte[nombre] = pd.MultiIndex.from_frame(cohorte[claves]).isin(idx).astype(int)
    return cohorte


def _admision(anio: int) -> pd.DataFrame | None:
    archivos = glob.glob(str(MINEDUC / f"paes_{anio}_puntajes" / "*.csv"))
    if not archivos:
        return None
    from student_analytics.ingestion.paes import load_scores
    d = load_scores(archivos[0])
    return d.rename(columns={"paes": "puntaje_admision", "nem": "puntaje_nem",
                             "ranking": "puntaje_ranking",
                             "percentil": "percentil_admision",
                             "instrumento": "prueba_admision"})


def _becas(anio: int) -> pd.DataFrame | None:
    archivos = glob.glob(str(MINEDUC / f"becas_{anio}" / "**" / "*.csv"), recursive=True)
    if not archivos:
        return None
    d = pd.read_csv(archivos[0], sep=";", encoding="utf-8", dtype=str, low_memory=False)
    d.columns = [c.upper() for c in d.columns]
    out = pd.DataFrame({"mrun": pd.to_numeric(d.get("MRUN"), errors="coerce")})
    for origen, destino in [("QUINTIL_INGRESO", "quintil_ingreso"),
                            ("DECIL_DFE", "decil_dfe")]:
        if origen in d.columns:
            out[destino] = pd.to_numeric(d[origen], errors="coerce").where(lambda s: s > 0)
    if "BENEFICIO_BECA_FSCU" in d.columns:
        out["tiene_beneficio"] = d.BENEFICIO_BECA_FSCU.notna().astype(int)
    return out.dropna(subset=["mrun"]).drop_duplicates("mrun")


def _sep(anio: int) -> pd.DataFrame | None:
    """Condición SEP del año ANTERIOR: es su último año de enseñanza media."""
    archivos = glob.glob(str(MINEDUC / f"sep_{anio - 1}" / "**" / "*.csv"), recursive=True)
    if not archivos:
        return None
    d = pd.read_csv(archivos[0], sep=";", encoding="latin-1", dtype=str, low_memory=False)
    d.columns = [c.upper() for c in d.columns]
    out = pd.DataFrame({"mrun": pd.to_numeric(d.get("MRUN"), errors="coerce")})
    for origen, destino in [("PRIORITARIO_ALU", "prioritario"),
                            ("PREFERENTE_ALU", "preferente"),
                            ("COD_DEPE2", "dependencia_colegio")]:
        if origen in d.columns:
            out[destino] = pd.to_numeric(d[origen], errors="coerce")
    return out.dropna(subset=["mrun"]).drop_duplicates("mrun")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=2007)
    ap.add_argument("--hasta", type=int, default=2025)
    args = ap.parse_args()

    piezas = []
    for anio in range(args.desde, args.hasta + 1):
        c = _cohorte(anio)
        if c is None:
            continue
        c = _continuidad(c, anio)
        for fuente, cargar in [("admision", _admision), ("becas", _becas), ("sep", _sep)]:
            extra = cargar(anio)
            if extra is not None:
                c = c.merge(extra, on="mrun", how="left")
        piezas.append(c)
        log.info("cohorte %s: %s ingresantes, %d columnas", anio, f"{len(c):,}", len(c.columns))

    if not piezas:
        raise SystemExit("Sin matrícula slim. Corre antes scripts/download_matricula.py")
    d = pd.concat(piezas, ignore_index=True)

    # EL NEM Y EL RANKING TIENEN EL MISMO CORTE DE ESCALA QUE LA PAES.
    #
    # En la PDT el NEM llegaba a 824 y en la PAES a 994; la media salta de 560
    # a 716 entre 2022 y 2023 sin que cambie nadie. Usar el puntaje crudo hace
    # que cualquier analisis agrupe por CALENDARIO creyendo que agrupa por
    # perfil: la primera segmentacion partia en dos un grupo identico en todo
    # lo demas, y la unica diferencia real era el anio de ingreso.
    #
    # El percentil dentro de la propia cohorte si cruza el corte, igual que
    # `percentil_admision` que ya venia calculado para el puntaje de admision.
    for origen, destino in [("puntaje_nem", "percentil_nem"),
                            ("puntaje_ranking", "percentil_ranking")]:
        if origen in d.columns:
            d[destino] = d.groupby("cohorte")[origen].rank(pct=True)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(SALIDA, index=False)

    print(f"\n{len(d):,} ingresantes UA · cohortes {d.cohorte.min()}–{d.cohorte.max()} "
          f"· {len(d.columns)} columnas")
    print(f"-> {SALIDA.relative_to(REPO_ROOT)}\n")
    print("Cobertura de las features que no existen en todas las cohortes:")
    for col in ("puntaje_admision", "puntaje_nem", "decil_dfe", "prioritario"):
        if col in d.columns:
            con = d[col].notna()
            anios = sorted(d.loc[con, "cohorte"].unique())
            print(f"  {col:20s} {con.mean():6.1%}  cohortes {anios[0]}–{anios[-1]}"
                  if len(anios) else f"  {col:20s} sin dato")
    print("\nResultado observado:")
    for col in ("sigue_carrera", "sigue_universidad", "sigue_sistema"):
        s = pd.to_numeric(d[col], errors="coerce")
        print(f"  {col:20s} {s.mean():6.1%}  ({s.notna().sum():,} con seguimiento)")
    print("\nSiguiente paso: python scripts/build_segments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
