"""Recursos institucionales desde la base INDICES del CNED.

Cubre dos de las limitaciones que el benchmark declaraba: cuerpo docente y
recursos institucionales. La base trae 2005-2025 por institución y sede, con
docentes por jornada y nivel de grado, inmuebles, laboratorios y bibliotecas.

Fuente: https://cned.cl/institucional/bases-de-datos/

DOS ADVERTENCIAS SOBRE ESTOS DATOS

1. Los códigos de institución del CNED NO son los de la matrícula del SIES
   (la Universidad Autónoma es 1037 en CNED y 31 en matrícula). La unión se
   hace por nombre normalizado, que calza 51 de 51 universidades del perfil.

2. Las bibliotecas están en dos hojas, 2005-2018 y 2019-2025, porque cambió
   el instrumento. Concatenarlas produciría un salto en 2019 que es
   metodológico, no real. Aquí se usa solo la hoja vigente.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

HOJA_INSTITUCION = "Univ IP CFT Vigentes"
HOJA_DOCENTES = "Docentes"
HOJA_INMUEBLES = "Inmuebles"
HOJA_LABORATORIOS = "Laboratorios y Talleres"
HOJA_BIBLIOTECAS = "Bibliotecas2019-2025"

# Etiquetas de las variables que se exponen aguas arriba.
RESOURCE_LABELS = {
    "anio_creacion": "Año de creación",
    "antiguedad": "Años desde la creación",
    "cruch": "Pertenece al CRUCH",
    "docentes_jce": "Docentes jornada completa equivalente",
    "docentes_por_100_alumnos": "Docentes JCE por 100 estudiantes",
    "share_doctorado": "Docentes con doctorado",
    "share_magister": "Docentes con magíster o doctorado",
    "share_jornada_completa": "Docentes con jornada completa",
    "share_docentes_mujeres": "Docentes mujeres",
    "m2_construido_por_alumno": "M² construidos por estudiante",
    "pc_por_100_alumnos": "PC para estudiantes por cada 100",
    "ejemplares_por_alumno": "Ejemplares de biblioteca por estudiante",
}


def normalize_name(value: str) -> str:
    """Nombre institucional comparable entre CNED y SIES.

    CNED abrevia "U." y escribe O`HIGGINS con acento grave; el SIES usa
    "UNIVERSIDAD" y apóstrofo. Sin unificar ambas cosas el cruce pierde
    instituciones sin avisar.
    """
    t = unicodedata.normalize("NFKD", str(value).upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    for signo in "'`´‘’":
        t = t.replace(signo, "")
    t = t.replace(".", " ").replace(",", " ").replace("-", " ")
    t = re.sub(r"\bU\b", "UNIVERSIDAD", t)
    t = re.sub(r"\bUNIV\b", "UNIVERSIDAD", t)
    t = re.sub(r"\bPONT\b", "PONTIFICIA", t)
    t = re.sub(r"\bDEL\b", "DE", t)
    for palabra in (" DE ", " LA ", " LAS ", " LOS ", " EL ", " Y "):
        t = t.replace(palabra, " ")
    return re.sub(r"\s+", " ", t).strip()


def _sum_columns(frame: pd.DataFrame, pattern: str) -> pd.Series:
    columns = [c for c in frame.columns if re.search(pattern, str(c), re.I)]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(columns))


def _closest_year(frame: pd.DataFrame, column: str, year: int) -> pd.DataFrame:
    """Solo el año solicitado: jamás se sustituye por otro sin evidencia."""
    years = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.assign(_anio=years).dropna(subset=["_anio"])
    if frame.empty:
        return frame
    return frame.loc[frame._anio.eq(year)]


def load_institutional(path: str | Path, year: int) -> pd.DataFrame:
    """Recursos por institución para la cohorte `year`, con clave de nombre."""
    path = Path(path)
    log.info("Leyendo INDICES institucional del CNED (%s)", path.name)
    hojas = pd.read_excel(path, sheet_name=None)

    inst = hojas[HOJA_INSTITUCION].copy()
    inst["clave"] = inst["Nombre institución"].map(normalize_name)
    acred = next((c for c in inst.columns if "Años Acreditación" in str(c)), None)
    base = pd.DataFrame({
        "clave": inst.clave,
        "cned_nombre": inst["Nombre institución"],
        "anio_creacion": pd.to_numeric(inst["Año Creación"], errors="coerce"),
        # La columna se llama "Tradicional" pero sus valores son del tipo
        # "(a) Universidades CRUCH" / "(b) Universidades Privadas": la palabra
        # "tradicional" no aparece en ningun valor.
        #
        # Y el nombre de la columna quedo obsoleto: desde la Ley 21.091 (2018)
        # el CRUCH admite privadas posteriores a 1981, y en 2019 entraron
        # Diego Portales, Alberto Hurtado y Los Andes. Pertenecer al CRUCH ya
        # no equivale a ser "tradicional", por eso la variable se llama
        # `cruch` y no `tradicional`.
        "cruch": inst["Tradicional"].astype(str).str.contains(
            "CRUCH", case=False, na=False).astype(int),
        "acreditacion_cned": pd.to_numeric(inst[acred], errors="coerce") if acred else np.nan,
    })
    base["antiguedad"] = year - base.anio_creacion
    if base.clave.duplicated().any():
        raise ValueError("Nombres institucionales CNED ambiguos tras normalizar")
    # Este atributo es una foto del catálogo, no una observación por cohorte.
    fecha = re.search(r"\(([^)]+)\)", str(acred))
    base["acreditacion_cned_fecha"] = fecha.group(1) if fecha else "Sin fecha"
    base["recursos_anio"] = year

    # --- Cuerpo docente -------------------------------------------------
    doc = _closest_year(hojas[HOJA_DOCENTES], "Año Proceso", year).copy()
    if not doc.empty:
        doc["clave"] = doc["Nombre Institución"].map(normalize_name)
        completa = _sum_columns(doc, r"^N.DocentesJornadaCompleta$")
        media = _sum_columns(doc, r"^N.DocentesJornadaMedia$")
        hora = _sum_columns(doc, r"^N.DocentesJornadaHora$")
        # Aproximación explícita del proyecto, no un JCE oficial del CNED.
        doc["_jce"] = completa + .5 * media + .25 * hora
        doc["_total"] = pd.to_numeric(doc.get("N°Docentes"), errors="coerce")
        doc["_doctor"] = _sum_columns(doc, r"^N.Doctor(Jornada)?(Hora|Media|Completa)$")
        doc["_magister"] = _sum_columns(doc, r"^N.Magister(Jornada)?(Hora|Media|Completa)$")
        doc["_completa"] = completa
        doc["_mujeres"] = pd.to_numeric(doc.get("N°DocentesMujeres"), errors="coerce")
        agg = doc.groupby("clave")[["_jce", "_total", "_doctor", "_magister",
                                    "_completa", "_mujeres"]].sum(min_count=1)
        observed = doc.groupby("clave")[agg.columns].count()
        complete = observed.eq(doc.groupby("clave").size(), axis=0)
        agg = agg.where(complete)
        total = agg._total.where(agg._total.gt(0))
        docentes = pd.DataFrame({
            "docentes_jce": agg._jce,
            "share_doctorado": agg._doctor.div(total),
            "share_magister": (agg._doctor + agg._magister).div(total),
            "share_jornada_completa": agg._completa.div(total),
            "share_docentes_mujeres": agg._mujeres.div(total),
        })
        base = base.merge(docentes, on="clave", how="left")

    # --- Infraestructura -------------------------------------------------
    inm = _closest_year(hojas[HOJA_INMUEBLES], "Año Información", year).copy()
    if not inm.empty:
        inm["clave"] = inm["Nombre Institución"].map(normalize_name)
        m2 = pd.to_numeric(inm.get("M2 Construido"), errors="coerce")
        base = base.merge(m2.groupby(inm.clave).sum(min_count=1).rename("m2_construido"),
                          on="clave", how="left")

    lab = _closest_year(hojas[HOJA_LABORATORIOS], "Año Proceso", year).copy()
    if not lab.empty:
        lab["clave"] = lab["Nombre Institución"].map(normalize_name)
        pcs = pd.to_numeric(lab.get("Nº de PC para alumnos"), errors="coerce")
        base = base.merge(pcs.groupby(lab.clave).sum(min_count=1).rename("pc_alumnos"),
                          on="clave", how="left")

    bib = hojas.get(HOJA_BIBLIOTECAS)
    if bib is not None:
        col_anio = next((c for c in bib.columns if "o" in str(c) and "roceso" in str(c)), None)
        bib = _closest_year(bib, col_anio, year).copy() if col_anio else bib.iloc[:0].copy()
        if not bib.empty:
            bib["clave"] = bib["Nombre Institución"].map(normalize_name)
            ejemplares = next((c for c in bib.columns if "jemplares" in str(c)), None)
            if ejemplares:
                base = base.merge(
                    pd.to_numeric(bib[ejemplares], errors="coerce")
                    .groupby(bib.clave).sum(min_count=1).rename("ejemplares"),
                    on="clave", how="left")

    log.info("  %s instituciones con recursos", f"{len(base):,}")
    return base


def attach_resources(profiles: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    """Une los recursos al perfil por nombre normalizado y calcula ratios.

    Los ratios usan estudiantes únicos identificados en toda la matrícula
    institucional del año (todos los niveles), nunca la cohorte de ingreso.
    Si ese denominador falta, la razón queda ausente.
    """
    out = profiles.copy()
    out["clave"] = out.nomb_inst.map(normalize_name)
    columnas = [c for c in resources.columns if c != "cned_nombre"]
    out = out.merge(resources[columnas], on="clave", how="left", validate="many_to_one")

    alumnos = out.get("estudiantes_total", pd.Series(np.nan, index=out.index))
    alumnos = alumnos.where(alumnos.gt(0))
    if "docentes_jce" in out:
        out["docentes_por_100_alumnos"] = 100 * out.docentes_jce.div(alumnos)
    if "m2_construido" in out:
        out["m2_construido_por_alumno"] = out.m2_construido.div(alumnos)
    if "pc_alumnos" in out:
        out["pc_por_100_alumnos"] = 100 * out.pc_alumnos.div(alumnos)
    if "ejemplares" in out:
        out["ejemplares_por_alumno"] = out.ejemplares.div(alumnos)
    return out.drop(columns=["clave"])
