"""Generador sintetico reproducible (doc maestro 18 y 71).

Orden causal, deliberado y no invertible:

    perfil latente -> comportamiento semanal -> agregados de cierre
                   -> riesgo latente -> resultado

El resultado NUNCA se sortea primero para despues fabricar comportamiento
que lo justifique. Si se hiciera al reves, cualquier modelo entrenado sobre
estos datos alcanzaria un AUC excelente y no se sabria si es porque el
pipeline funciona o porque el generador filtro la respuesta.

Consecuencia buscada: las features de la semana 3 son debilmente
informativas y las de la semana 12 fuertemente informativas, porque la
trayectoria todavia no termina de ocurrir. Ese gradiente es exactamente el
fenomeno que el proyecto necesita medir (doc maestro 68: predictive power
vs lead time) y el hallazgo central del benchmark ULagos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from .profiles import PROFILES

log = get_logger(__name__)

# Semanas del semestre en que se registran evaluaciones con nota.
ASSESSMENT_WEEKS = (4, 8, 12, 16)

DIFFICULTY_TIERS = ("quantitative_core", "disciplinary", "general")
# Efecto del curso sobre la nota final, en puntos de la escala 1-7.
TIER_GRADE_EFFECT = {"quantitative_core": -0.75, "disciplinary": -0.15, "general": 0.30}
# Multiplicador de actividad Canvas segun cuanto lo usa el docente.
ADOPTION_ACTIVITY = {"high": 1.0, "medium": 0.55, "low": 0.12}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Estandariza; devuelve ceros si la serie es constante."""
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / sd


def solve_intercept(score: np.ndarray, target_rate: float,
                    lo: float = -12.0, hi: float = 12.0, tol: float = 1e-5) -> float:
    """Encuentra b tal que mean(sigmoid(score + b)) == target_rate.

    Se resuelve por biseccion en vez de fijar un intercepto a mano. La
    consecuencia practica es que se puede cambiar cualquier peso del
    riesgo latente sin mover la tasa global de reprobacion: el ancla de
    calibracion manda sobre los coeficientes.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate debe estar en (0,1), recibido {target_rate}")
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = float(_sigmoid(score + mid).mean())
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_threshold(values: np.ndarray, target_rate: float) -> float:
    """Umbral que hace que una proporcion `target_rate` caiga por debajo."""
    return float(np.quantile(values, target_rate))


@dataclass
class SyntheticDataset:
    """Las ocho tablas del diccionario sintetico (doc maestro 19)."""

    student_master: pd.DataFrame
    course_master: pd.DataFrame
    enrollment: pd.DataFrame
    attendance_weekly: pd.DataFrame
    grades_weekly: pd.DataFrame
    canvas_weekly: pd.DataFrame
    assignments_weekly: pd.DataFrame
    outcomes: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "student_master": self.student_master,
            "course_master": self.course_master,
            "enrollment": self.enrollment,
            "attendance_weekly": self.attendance_weekly,
            "grades_weekly": self.grades_weekly,
            "canvas_weekly": self.canvas_weekly,
            "assignments_weekly": self.assignments_weekly,
            "outcomes": self.outcomes,
        }


class SyntheticGenerator:
    """Genera el dataset completo de forma determinista dado un seed."""

    def __init__(
        self,
        cfg: dict[str, Any],
        settings: dict[str, Any],
        n_students: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.academic = settings["academic"]
        self.rng = np.random.default_rng(seed if seed is not None else cfg["seed"])
        self.n_students = n_students or cfg["volume"]["n_students"]
        self.n_terms = cfg["volume"]["n_terms"]
        self.weeks = cfg["volume"]["weeks_per_term"]

    # ------------------------------------------------------------------
    # 1. Dimensiones
    # ------------------------------------------------------------------
    def build_students(self) -> pd.DataFrame:
        n = self.n_students
        rng = self.rng

        campus_names = list(self.cfg["campus"])
        campus_p = np.array([self.cfg["campus"][c] for c in campus_names])

        careers = self.cfg["careers"]
        career_codes = [c["code"] for c in careers]
        career_p = np.array([c["weight"] for c in careers])

        profile_names = list(self.cfg["profiles"])
        profile_p = np.array([self.cfg["profiles"][p]["prevalence"] for p in profile_names])
        profile = rng.choice(profile_names, size=n, p=profile_p)

        # Aptitud latente: se sortea del perfil. Es la unica variable que
        # ningun modelo podra observar directamente.
        ability = np.empty(n)
        hist_pass = np.empty(n)
        for name in profile_names:
            mask = profile == name
            k = int(mask.sum())
            if k == 0:
                continue
            prof = PROFILES[name]
            ability[mask] = rng.normal(prof.ability_mean, prof.ability_sd, k)
            hist_pass[mask] = np.clip(rng.normal(prof.historical_pass_rate, 0.08, k), 0.0, 1.0)
        ability = np.clip(ability, 1.0, 7.0)

        # Variables previas al ingreso. Correlacionan DEBILMENTE con la
        # aptitud a proposito: el hallazgo de ULagos es que NEM y PAES casi
        # no predicen la desercion. El generador reproduce esa debilidad
        # para que el pipeline la redescubra en vez de asumirla.
        admission_score = np.clip(
            500 + 55 * (0.35 * _zscore(ability) + 0.65 * rng.normal(0, 1, n)), 210, 1000
        ).round(0)
        hs_gpa = np.clip(
            5.3 + 0.55 * (0.40 * _zscore(ability) + 0.60 * rng.normal(0, 1, n)), 4.0, 7.0
        ).round(2)

        entry_term = rng.integers(1, self.n_terms + 1, size=n)

        return pd.DataFrame(
            {
                "student_id": [f"S{i:06d}" for i in range(n)],
                "cohort": 2020 + (entry_term - 1) // 2,
                "entry_term": entry_term,
                "campus": rng.choice(campus_names, size=n, p=campus_p),
                "faculty": "Administracion y Negocios",
                "career_code": rng.choice(career_codes, size=n, p=career_p),
                "sex": rng.choice(["F", "M"], size=n, p=[0.54, 0.46]),
                "funding_type": rng.choice(
                    ["gratuidad", "CAE", "beca_interna", "particular"],
                    size=n,
                    p=[0.42, 0.28, 0.12, 0.18],
                ),
                "admission_score": admission_score,
                "hs_gpa": hs_gpa,
                "historical_pass_rate": hist_pass.round(3),
                # Latentes: diagnostico del generador, NUNCA features.
                "_latent_profile": profile,
                "_latent_ability": ability.round(3),
            }
        )

    def build_courses(self) -> pd.DataFrame:
        rng = self.rng
        adoption = self.cfg["canvas_adoption"]
        adoption_levels = ["high", "medium", "low"]
        adoption_p = np.array([adoption[k] for k in adoption_levels])
        no_att = self.cfg["missingness"]["attendance_not_recorded_course_rate"]

        rows = []
        for career in self.cfg["careers"]:
            code = career["code"]
            for i in range(career["n_courses"]):
                # Los ramos cuantitativos se concentran en los primeros niveles.
                level = i // 3 + 1
                if level <= 2:
                    tier_p = [0.50, 0.30, 0.20]
                elif level <= 3:
                    tier_p = [0.25, 0.55, 0.20]
                else:
                    tier_p = [0.10, 0.70, 0.20]
                rows.append(
                    {
                        "course_id": f"{code}-{i:03d}",
                        "career_code": code,
                        "course_name": f"{code} Asignatura {i:02d}",
                        "level": level,
                        "credits": float(rng.choice([4, 5, 6, 8])),
                        "difficulty_tier": str(rng.choice(DIFFICULTY_TIERS, p=tier_p)),
                        # Adopcion docente de Canvas: propiedad del CURSO, no del
                        # estudiante. Sin esto, todo estudiante de un curso sin
                        # Canvas parece desenganchado (doc maestro 73).
                        "canvas_adoption": str(rng.choice(adoption_levels, p=adoption_p)),
                        # Cursos sin registro de asistencia en Banner.
                        "attendance_recorded": bool(rng.random() > no_att),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Inscripcion
    # ------------------------------------------------------------------
    def build_enrollment(self, students: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
        """Arma la canasta de asignaturas de cada estudiante por termino.

        Un estudiante aparece desde su termino de ingreso en adelante. La
        desercion se modela como ausencia en terminos posteriores, con
        probabilidad dependiente de su perfil, de modo que la retencion de
        primer anio se aproxime al ancla SIES de config/synthetic.yml.
        """
        rng = self.rng
        lo, hi = self.cfg["volume"]["courses_per_student_term"]
        by_career = {c: g["course_id"].to_numpy() for c, g in courses.groupby("career_code")}

        # Riesgo de abandono por perfil, anclado a la retencion objetivo.
        target_retention = self.cfg["calibration"]["retention_first_year"]
        base_dropout = 1.0 - target_retention
        prof_multiplier = {
            "A_high_performer": 0.15, "B_silent_high": 0.30,
            "C_struggling_engaged": 0.90, "D_academic_risk": 2.30,
            "E_disengaging": 2.60, "F_intermittent": 1.70,
            "G_recovering": 1.10,
        }
        mult = students["_latent_profile"].map(prof_multiplier).to_numpy()
        # Normalizar para que el promedio poblacional sea 1.0: asi la
        # retencion agregada queda igual al ancla SIES y los multiplicadores
        # solo redistribuyen el riesgo entre perfiles.
        mult = mult / float(mult.mean())
        p_dropout = np.clip(base_dropout * mult, 0.0, 0.85)

        # Primer termino que el estudiante YA NO cursa. La desercion ocurre
        # al CERRAR un semestre, no antes de cursarlo: si se aplica antes,
        # el estudiante nunca aparece inscrito y la retencion medida cae al
        # cuadrado de la tasa real.
        entry = students["entry_term"].to_numpy()
        last_term = np.full(len(students), self.n_terms + 1)
        for offset in range(self.n_terms):
            cursando = last_term > (entry + offset)
            drops = cursando & (rng.random(len(students)) < p_dropout)
            last_term = np.where(drops, entry + offset + 1, last_term)

        active = students[["student_id", "career_code"]].copy()
        active["entry_term"] = entry
        active["last_term"] = last_term

        rows: list[tuple[str, str, int]] = []
        for term in range(1, self.n_terms + 1):
            sel = active[(active["entry_term"] <= term) & (active["last_term"] > term)]
            if sel.empty:
                continue
            n_courses = rng.integers(lo, hi + 1, size=len(sel))
            for (sid, career), k in zip(
                sel[["student_id", "career_code"]].itertuples(index=False, name=None),
                n_courses,
            ):
                pool = by_career[career]
                chosen = rng.choice(pool, size=min(int(k), len(pool)), replace=False)
                rows.extend((sid, str(cid), term) for cid in chosen)

        return pd.DataFrame(rows, columns=["student_id", "course_id", "term_id"])

    # ------------------------------------------------------------------
    # 3. Trayectoria semanal
    # ------------------------------------------------------------------
    def _weekly_panel(self, enr: pd.DataFrame, students: pd.DataFrame,
                      courses: pd.DataFrame) -> pd.DataFrame:
        """Expande inscripciones a grano student x course x week.

        Cada metrica semanal es nivel_inicial + deriva*(avance) + ruido.
        La deriva es lo que separa al perfil E (se cae) del C (siempre bajo):
        mismo promedio al cierre, trayectoria opuesta.
        """
        rng = self.rng
        W = self.weeks

        base = enr.merge(
            students[["student_id", "_latent_profile"]], on="student_id", how="left"
        ).merge(
            courses[["course_id", "canvas_adoption", "attendance_recorded",
                     "difficulty_tier"]],
            on="course_id", how="left",
        )

        n_enr = len(base)
        weeks = np.arange(1, W + 1)
        # Progreso del semestre en [0, 1]: cuanto de la deriva ya ocurrio.
        progress = (weeks - 1) / (W - 1)

        panel = base.loc[base.index.repeat(W)].reset_index(drop=True)
        panel["week"] = np.tile(weeks, n_enr)
        prog = np.tile(progress, n_enr)

        # Parametros por fila segun el perfil del estudiante.
        prof_names = panel["_latent_profile"].to_numpy()
        att_start = np.empty(len(panel))
        att_drift = np.empty(len(panel))
        att_vol = np.empty(len(panel))
        can_start = np.empty(len(panel))
        can_drift = np.empty(len(panel))
        can_vol = np.empty(len(panel))
        sub_start = np.empty(len(panel))
        sub_drift = np.empty(len(panel))

        for name, prof in PROFILES.items():
            m = prof_names == name
            if not m.any():
                continue
            att_start[m], att_drift[m], att_vol[m] = (
                prof.attendance_start, prof.attendance_drift, prof.attendance_volatility)
            can_start[m], can_drift[m], can_vol[m] = (
                prof.canvas_start, prof.canvas_drift, prof.canvas_volatility)
            sub_start[m], sub_drift[m] = prof.submission_start, prof.submission_drift

        n = len(panel)
        attendance = np.clip(
            att_start + att_drift * prog + rng.normal(0, 1, n) * att_vol, 0.0, 1.0)
        canvas_rel = np.clip(
            can_start + can_drift * prog + rng.normal(0, 1, n) * can_vol, 0.0, 1.5)
        submission = np.clip(
            sub_start + sub_drift * prog + rng.normal(0, 1, n) * 0.07, 0.0, 1.0)

        panel["attendance_rate_week"] = attendance
        panel["submission_rate_week"] = submission
        # La actividad Canvas observada es comportamiento x adopcion docente.
        adoption_mult = panel["canvas_adoption"].map(ADOPTION_ACTIVITY).to_numpy()
        panel["_canvas_latent"] = canvas_rel
        panel["_canvas_observed"] = canvas_rel * adoption_mult
        return panel

    # ------------------------------------------------------------------
    # 3. Tablas de hechos semanales
    # ------------------------------------------------------------------
    def _facts_from_panel(self, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
        rng = self.rng
        keys = ["student_id", "course_id", "term_id", "week"]

        # --- Asistencia -------------------------------------------------
        sessions_held = np.full(len(panel), 2, dtype=int)
        attended = rng.binomial(sessions_held, panel["attendance_rate_week"].to_numpy())
        att = panel[keys].copy()
        att["sessions_held"] = sessions_held
        att["sessions_attended"] = attended
        att["attendance_rate"] = attended / sessions_held
        # Cursos sin registro en Banner: la fila existe pero viene vacia.
        # NULL explicito, jamas cero: un cero se lee como "no asistio".
        no_record = ~panel["attendance_recorded"].to_numpy()
        att.loc[no_record, ["sessions_held", "sessions_attended", "attendance_rate"]] = np.nan
        att["attendance_recorded"] = ~no_record

        # --- Canvas -----------------------------------------------------
        activity = panel["_canvas_observed"].to_numpy()
        canvas = panel[keys].copy()
        canvas["page_views"] = rng.poisson(np.clip(activity, 0, None) * 42)
        canvas["participations"] = rng.poisson(np.clip(activity, 0, None) * 6)
        canvas["days_active"] = np.minimum(
            rng.poisson(np.clip(activity, 0, None) * 3.2), 7)
        canvas["canvas_adoption"] = panel["canvas_adoption"].to_numpy()
        # En cursos de baja adopcion, la senal Canvas no es interpretable.
        # Se marca para que el feature builder la trate como no disponible
        # en vez de como desenganche (doc maestro 73).
        canvas["canvas_reliable"] = panel["canvas_adoption"].to_numpy() != "low"

        # --- Tareas -----------------------------------------------------
        # No todas las semanas tienen tareas.
        has_assignment = rng.random(len(panel)) < 0.55
        assigned = np.where(has_assignment, rng.integers(1, 3, len(panel)), 0)
        sub_rate = panel["submission_rate_week"].to_numpy()
        submitted = rng.binomial(assigned, sub_rate)
        # De lo entregado, parte llega atrasado; mas atraso si el
        # compromiso es bajo.
        late = rng.binomial(submitted, np.clip(0.45 * (1 - sub_rate), 0, 1))
        asg = panel[keys].copy()
        asg["assigned"] = assigned
        asg["submitted"] = submitted
        asg["late"] = late
        asg["missing"] = assigned - submitted

        return {"attendance_weekly": att, "canvas_weekly": canvas,
                "assignments_weekly": asg}

    def _weekly_grades(self, panel: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
        """Notas parciales en las semanas de evaluacion.

        Cada nota parcial mezcla aptitud latente, dificultad del curso y el
        comportamiento acumulado HASTA esa semana. Esto es lo que hace que
        las notas parciales tengan valor predictivo temprano -- justamente
        la variable que el informe ULagos identifico como faltante.
        """
        rng = self.rng
        ability = students.set_index("student_id")["_latent_ability"]

        ev = panel[panel["week"].isin(ASSESSMENT_WEEKS)].copy()
        ev["_ability"] = ev["student_id"].map(ability).to_numpy()
        ev["_tier_effect"] = ev["difficulty_tier"].map(TIER_GRADE_EFFECT).to_numpy()

        # Comportamiento acumulado hasta la semana de la evaluacion.
        panel_sorted = panel.sort_values(["student_id", "course_id", "term_id", "week"])
        grp = panel_sorted.groupby(["student_id", "course_id", "term_id"], sort=False)
        panel_sorted["_att_cum"] = grp["attendance_rate_week"].transform(
            lambda s: s.expanding().mean())
        panel_sorted["_sub_cum"] = grp["submission_rate_week"].transform(
            lambda s: s.expanding().mean())
        cum = panel_sorted.set_index(
            ["student_id", "course_id", "term_id", "week"])[["_att_cum", "_sub_cum"]]
        ev = ev.join(cum, on=["student_id", "course_id", "term_id", "week"])

        grade = (
            ev["_ability"].to_numpy()
            + ev["_tier_effect"].to_numpy()
            + 1.15 * (ev["_att_cum"].to_numpy() - 0.80)
            + 0.95 * (ev["_sub_cum"].to_numpy() - 0.80)
            + rng.normal(0, 0.55, len(ev))
        )
        out = ev[["student_id", "course_id", "term_id", "week"]].copy()
        out["assessment_grade"] = np.clip(grade, 1.0, 7.0).round(1)
        out["assessment_seq"] = out["week"].map(
            {w: i + 1 for i, w in enumerate(ASSESSMENT_WEEKS)})

        # Notas cargadas con atraso: existen pero no estaban disponibles
        # al momento del scoring de esa semana (doc maestro 74).
        late_rate = self.cfg["missingness"]["grade_late_entry_rate"]
        out["available_from_week"] = out["week"] + (
            rng.random(len(out)) < late_rate).astype(int) * 2
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Resultado: riesgo latente -> reprobacion
    # ------------------------------------------------------------------
    def _outcomes(self, panel: pd.DataFrame, grades: pd.DataFrame,
                  facts: dict[str, pd.DataFrame], students: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        rm = self.cfg["risk_model"]
        keys = ["student_id", "course_id", "term_id"]

        # Agregados de CIERRE de semestre (no disponibles durante el mismo).
        agg = panel.groupby(keys, sort=False).agg(
            final_attendance=("attendance_rate_week", "mean"),
            final_submission=("submission_rate_week", "mean"),
            canvas_latent=("_canvas_latent", "mean"),
        ).reset_index()

        asg = facts["assignments_weekly"].groupby(keys, sort=False).agg(
            assigned_total=("assigned", "sum"), missing_total=("missing", "sum")
        ).reset_index()
        agg = agg.merge(asg, on=keys, how="left")
        agg["missing_share"] = np.where(
            agg["assigned_total"] > 0, agg["missing_total"] / agg["assigned_total"], 0.0)

        gr = grades.groupby(keys, sort=False)["assessment_grade"].mean().reset_index()
        gr = gr.rename(columns={"assessment_grade": "mean_assessment_grade"})
        agg = agg.merge(gr, on=keys, how="left")

        agg = agg.merge(
            students[["student_id", "historical_pass_rate"]], on="student_id", how="left")
        agg = agg.merge(
            panel[keys + ["difficulty_tier"]].drop_duplicates(keys), on=keys, how="left")

        cal = self.cfg["calibration"]
        tier_diff = agg["difficulty_tier"].map(cal["difficulty_spread"]).to_numpy()

        # Riesgo latente segun los pesos configurables de config/synthetic.yml
        # (doc maestro 18.4). Cada termino esta orientado de modo que valores
        # mas altos = mas riesgo.
        signal = (
            rm["w_low_grade"] * _zscore(-agg["mean_assessment_grade"].to_numpy())
            + rm["w_attendance_drop"] * _zscore(-agg["final_attendance"].to_numpy())
            + rm["w_missing_assignments"] * _zscore(agg["missing_share"].to_numpy())
            + rm["w_canvas_drop"] * _zscore(-agg["canvas_latent"].to_numpy())
            + rm["w_historical_failure"] * _zscore(-agg["historical_pass_rate"].to_numpy())
            + rm["w_course_difficulty"] * tier_diff
        )
        noise = rng.normal(0, rm["noise_sd"], len(agg))
        raw = signal + noise

        # Reescalar a la dispersion objetivo. Sin esto el sigmoide se satura
        # y los perfiles de riesgo reprueban el 100% de las veces: el
        # problema se vuelve trivial y cualquier modelo parece excelente.
        raw_sd = float(np.std(raw))
        score = raw * (rm["latent_sd"] / raw_sd) if raw_sd > 1e-9 else raw

        # El intercepto se resuelve contra el ancla, no se fija a mano.
        intercept = solve_intercept(score, cal["course_fail_rate"])
        latent = score + intercept

        signal_share = float(np.var(signal)) / float(np.var(raw)) if np.var(raw) > 0 else 0.0
        log.info("Riesgo latente: sd=%.2f, intercepto=%.3f, senal/total=%.0f%%",
                 rm["latent_sd"], intercept, 100 * signal_share)

        p_fail = _sigmoid(latent)
        fail_by_grade = rng.binomial(1, p_fail)

        # Nota final coherente con el resultado sorteado: aprobados por
        # sobre 4.0, reprobados por debajo, ambos centrados en el
        # desempenio parcial observado.
        center = agg["mean_assessment_grade"].fillna(4.0).to_numpy()
        pass_line = self.academic["grade_pass"]
        final = np.where(
            fail_by_grade == 1,
            np.clip(np.minimum(center - 0.3, pass_line - 0.1)
                    - np.abs(rng.normal(0, 0.45, len(agg))), 1.0, pass_line - 0.1),
            np.clip(np.maximum(center, pass_line + 0.1)
                    + np.abs(rng.normal(0, 0.35, len(agg))), pass_line, 7.0),
        )

        out = agg[keys].copy()
        out["final_grade"] = np.round(final, 1)
        out["final_attendance"] = agg["final_attendance"].round(3)
        out["fail_by_grade"] = fail_by_grade.astype(int)

        # Segundo target, separado a proposito. Si la UA reprueba por
        # inasistencia, mezclar ambos hace que el modelo aprenda el
        # reglamento en vez del fenomeno academico.
        if self.academic["attendance_rule_enabled"]:
            att = agg["final_attendance"].to_numpy()
            # El umbral reglamentario nominal es settings.attendance_min_pass,
            # pero la tasa resultante depende de la distribucion simulada de
            # asistencia. Se ajusta el umbral efectivo al ancla para que la
            # causal reglamentaria sea minoritaria, como en la practica.
            thr = solve_threshold(att, cal["attendance_fail_rate"])
            thr = min(thr, self.academic["attendance_min_pass"])
            log.info("Umbral efectivo de asistencia: %.3f (nominal %.2f)",
                     thr, self.academic["attendance_min_pass"])
            out["fail_by_attendance"] = (att < thr).astype(int)
        else:
            out["fail_by_attendance"] = 0
        out["failed"] = ((out["fail_by_grade"] == 1) | (out["fail_by_attendance"] == 1)).astype(int)
        out["_p_fail_true"] = p_fail.round(4)
        return out

    # ------------------------------------------------------------------
    # 5. Orquestacion
    # ------------------------------------------------------------------
    def generate(self) -> SyntheticDataset:
        log.info("Generando %s estudiantes, %s terminos, %s semanas",
                 self.n_students, self.n_terms, self.weeks)

        students = self.build_students()
        courses = self.build_courses()
        enrollment = self.build_enrollment(students, courses)
        log.info("Inscripciones: %s filas", f"{len(enrollment):,}")

        panel = self._weekly_panel(enrollment, students, courses)
        log.info("Panel student x course x week: %s filas", f"{len(panel):,}")

        facts = self._facts_from_panel(panel)
        grades = self._weekly_grades(panel, students)
        outcomes = self._outcomes(panel, grades, facts, students)

        enrollment = enrollment.merge(
            outcomes[["student_id", "course_id", "term_id", "failed"]],
            on=["student_id", "course_id", "term_id"], how="left")
        enrollment["status"] = np.where(enrollment["failed"] == 1, "RE", "AP")
        enrollment = enrollment.drop(columns=["failed"])

        rate = outcomes["failed"].mean()
        log.info("Tasa de reprobacion a nivel asignatura: %.1f%%", 100 * rate)

        return SyntheticDataset(
            student_master=students,
            course_master=courses,
            enrollment=enrollment,
            attendance_weekly=facts["attendance_weekly"],
            grades_weekly=grades,
            canvas_weekly=facts["canvas_weekly"],
            assignments_weekly=facts["assignments_weekly"],
            outcomes=outcomes,
        )
