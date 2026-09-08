"""Adaptador OULAD -> esquema canonico.

OULAD (Open University Learning Analytics Dataset) tiene ~32.000 estudiantes
reales de 7 modulos, con clicks diarios en el VLE, entregas, evaluaciones y
resultado final. Es el analogo publico mas cercano al grano
student x course x week que este proyecto quiere construir con Banner+Canvas.

Para que sirve aqui: valida la METODOLOGIA con datos reales antes de tener
accesos. El generador sintetico valida que el codigo corre; OULAD valida que
la prediccion temprana efectivamente funciona y cuanto se degrada al
adelantarla. Un modelo entrenado sobre datos que uno mismo genero no puede
responder eso.

Diferencias con la UA que hay que tener presentes al leer los resultados:
  - NO hay asistencia. OULAD es educacion a distancia: no existe el concepto.
    Es justamente la variable que ULagos identifico como clave, asi que estos
    resultados son un PISO, no un techo.
  - Escala de notas 0-100, no 1.0-7.0.
  - Modulos de ~9 meses, no semestres de 18 semanas.
  - Poblacion adulta a distancia del Reino Unido.

Fuente: https://analyse.kmi.open.ac.uk/open_dataset
Paper: Kuzilek, Hlosta & Zdrahal (2017), Scientific Data 4:170171.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

DAYS_PER_WEEK = 7

# code_presentation: el anio seguido de B (febrero) o J (octubre).
# El orden importa: la validacion temporal depende de el.
PRESENTATION_ORDER = {"2013B": 1, "2013J": 2, "2014B": 3, "2014J": 4}


def _to_week(days: pd.Series) -> pd.Series:
    """Dias desde el inicio del modulo -> numero de semana (1-indexado).

    OULAD admite dias negativos (actividad previa al inicio). Se colapsan a
    la semana 1 en vez de descartarse: es actividad real del estudiante.

    OULAD codifica los faltantes como "?", asi que varias columnas de fecha
    llegan como texto. La coercion es explicita para que un faltante quede
    como NA y no reviente ni, peor, se convierta silenciosamente en cero.
    """
    d = pd.to_numeric(days, errors="coerce")
    return (np.floor(d / DAYS_PER_WEEK) + 1).clip(lower=1).astype("Int64")


def _find_dir(root: Path) -> Path:
    """El zip de UCI a veces trae los CSV en una subcarpeta."""
    if (root / "studentInfo.csv").exists():
        return root
    for sub in root.rglob("studentInfo.csv"):
        return sub.parent
    raise FileNotFoundError(f"No se encontro studentInfo.csv bajo {root}")


def load_oulad(path: str | Path) -> dict[str, pd.DataFrame]:
    """Lee los CSV de OULAD y los mapea al esquema canonico.

    Devuelve las mismas tablas que el generador sintetico, salvo
    `attendance_weekly`, que no existe en esta fuente. El feature builder
    la trata como ausente y emite NaN, no ceros.
    """
    root = _find_dir(Path(path))
    log.info("Leyendo OULAD desde %s", root)

    info = pd.read_csv(root / "studentInfo.csv")
    reg = pd.read_csv(root / "studentRegistration.csv")
    vle = pd.read_csv(root / "studentVle.csv")
    sa = pd.read_csv(root / "studentAssessment.csv")
    assess = pd.read_csv(root / "assessments.csv")
    courses = pd.read_csv(root / "courses.csv")
    log.info("Filas leidas: info=%s vle=%s assessments=%s",
             f"{len(info):,}", f"{len(vle):,}", f"{len(sa):,}")

    # Mismo motivo: "?" convierte estas columnas en texto al leerlas.
    for df, cols in ((reg, ["date_registration", "date_unregistration"]),
                     (sa, ["score", "date_submitted"]),
                     (assess, ["date", "weight"]),
                     (vle, ["date", "sum_click"]),
                     (info, ["num_of_prev_attempts", "studied_credits"])):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    def keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rename(columns={"id_student": "student_id", "code_module": "course_id"})
        out["term_id"] = out["code_presentation"].map(PRESENTATION_ORDER)
        return out.drop(columns=["code_presentation"])

    info = keys(info)
    reg = keys(reg)
    vle = keys(vle)
    assess = keys(assess)
    courses = keys(courses)
    K = ["student_id", "course_id", "term_id"]

    # --- Dimensiones -------------------------------------------------
    student_master = (
        info.groupby("student_id", as_index=False)
        .agg(
            gender=("gender", "first"), region=("region", "first"),
            highest_education=("highest_education", "first"),
            imd_band=("imd_band", "first"), age_band=("age_band", "first"),
            disability=("disability", "first"),
            prior_attempts=("num_of_prev_attempts", "max"),
            credits_load=("studied_credits", "mean"),
        )
    )
    # OULAD no trae historial de aprobacion previo al dataset.
    student_master["historical_pass_rate"] = np.nan

    course_master = (
        courses.groupby("course_id", as_index=False)
        .agg(module_length=("module_presentation_length", "mean"))
    )
    course_master["level"] = np.nan

    enrollment = info[K].drop_duplicates().copy()
    enrollment["status"] = np.where(
        info.set_index(K).loc[enrollment.set_index(K).index, "final_result"]
        .isin(["Withdrawn"]).to_numpy(), "RE", "AP")

    # --- Actividad en el VLE -----------------------------------------
    vle["week"] = _to_week(vle["date"])
    activity = (
        vle.groupby(K + ["week"], as_index=False)
        .agg(page_views=("sum_click", "sum"), days_active=("date", "nunique"))
    )
    # Todos los modulos de OULAD usan el VLE de forma intensiva, a diferencia
    # de Canvas en la UA, donde la adopcion docente es heterogenea.
    activity["activity_reliable"] = True

    # --- Evaluaciones y entregas -------------------------------------
    # El examen final puede venir sin fecha: se ancla al cierre del modulo.
    assess = assess.merge(
        courses[["course_id", "term_id", "module_presentation_length"]],
        on=["course_id", "term_id"], how="left")
    assess["date"] = assess["date"].fillna(assess["module_presentation_length"])
    assess["due_week"] = _to_week(assess["date"])

    sa = sa.rename(columns={"id_student": "student_id"})
    sub = sa.merge(
        assess[["id_assessment", "course_id", "term_id", "due_week", "assessment_type",
                "weight"]],
        on="id_assessment", how="inner")
    sub["submit_week"] = _to_week(sub["date_submitted"])
    sub["is_late"] = (sub["submit_week"] > sub["due_week"]).astype(int)

    # Una fila en la semana de VENCIMIENTO (assigned) y, si la entrega llego
    # tarde, otra en la semana en que efectivamente llego. Asi, filtrar por
    # week <= N devuelve exactamente lo que se sabia en la semana N y no lo
    # que se supo despues.
    due_rows = pd.DataFrame({
        "student_id": sub["student_id"], "course_id": sub["course_id"],
        "term_id": sub["term_id"], "week": sub["due_week"],
        "assigned": 1,
        "submitted": (sub["is_late"] == 0).astype(int),
        "late": 0,
    })
    late_rows = pd.DataFrame({
        "student_id": sub.loc[sub["is_late"] == 1, "student_id"],
        "course_id": sub.loc[sub["is_late"] == 1, "course_id"],
        "term_id": sub.loc[sub["is_late"] == 1, "term_id"],
        "week": sub.loc[sub["is_late"] == 1, "submit_week"],
        "assigned": 0, "submitted": 1, "late": 1,
    })
    assignments = (
        pd.concat([due_rows, late_rows], ignore_index=True)
        .groupby(K + ["week"], as_index=False)
        .agg(assigned=("assigned", "sum"), submitted=("submitted", "sum"),
             late=("late", "sum"))
    )
    assignments["missing"] = (assignments["assigned"] - assignments["submitted"]).clip(lower=0)

    # --- Notas parciales ---------------------------------------------
    grades = sub[K + ["submit_week", "score", "weight"]].copy()
    grades = grades.rename(columns={"submit_week": "week", "score": "assessment_grade"})
    grades = grades.dropna(subset=["assessment_grade"])
    # OULAD no registra la fecha de CORRECCION, solo la de entrega. Se asume
    # disponibilidad inmediata, que es optimista: en la practica hay un
    # rezago de correccion. Los resultados tempranos aqui son, por tanto,
    # un limite superior de lo alcanzable.
    grades["available_from_week"] = grades["week"]
    grades = grades.sort_values(K + ["week"]).reset_index(drop=True)

    # --- Resultado ----------------------------------------------------
    outcomes = info[K + ["final_result"]].copy()
    outcomes["fail_by_grade"] = (outcomes["final_result"] == "Fail").astype(int)
    outcomes["withdrawn"] = (outcomes["final_result"] == "Withdrawn").astype(int)
    outcomes["failed"] = outcomes[["fail_by_grade", "withdrawn"]].max(axis=1)

    # Semana de baja: quien ya se retiro no es "predecible" mas adelante.
    reg["unreg_week"] = _to_week(reg["date_unregistration"])
    outcomes = outcomes.merge(reg[K + ["unreg_week"]], on=K, how="left")

    tables = {
        "student_master": student_master,
        "course_master": course_master,
        "enrollment": enrollment,
        "activity_weekly": activity,
        "assignments_weekly": assignments,
        "grades_weekly": grades,
        "outcomes": outcomes,
        # attendance_weekly: NO EXISTE en OULAD, deliberadamente ausente.
    }
    for name, df in tables.items():
        log.info("  %-20s %10s filas", name, f"{len(df):,}")
    return tables


def eligible_at_week(outcomes: pd.DataFrame, week: int) -> pd.DataFrame:
    """Estudiantes todavia matriculados al inicio de la semana `week`.

    Quien ya se dio de baja no es un caso a predecir: ya ocurrio. Incluirlo
    infla artificialmente las metricas, porque su ausencia total de
    actividad posterior "predice" un evento que ya sucedio.
    """
    aun = outcomes["unreg_week"].isna() | (outcomes["unreg_week"] > week)
    return outcomes[aun]
