"""Perfiles latentes de estudiante (doc maestro 18.3).

Cada perfil describe una TRAYECTORIA semanal, no un promedio. Esa es la
diferencia que importa: un modelo entrenado sobre promedios no distingue
al perfil E (empieza bien y se cae) del perfil C (siempre bajo pero
constante), y esos dos casos requieren intervenciones opuestas.

El perfil es una variable LATENTE: se usa para generar comportamiento y
se guarda solo como etiqueta de diagnostico. Nunca debe entrar como
feature a un modelo -- en datos reales no existe.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """Parametros de la trayectoria semanal de un perfil.

    Los valores `*_start` son el nivel en la semana 1 y los `*_drift` el
    cambio total acumulado al llegar a la ultima semana del semestre.
    """

    name: str
    # Asistencia: proporcion 0-1.
    attendance_start: float
    attendance_drift: float
    attendance_volatility: float
    # Actividad Canvas: escala relativa, se convierte a page views.
    canvas_start: float
    canvas_drift: float
    canvas_volatility: float
    # Tasa de entrega de tareas: proporcion 0-1.
    submission_start: float
    submission_drift: float
    # Aptitud academica latente, en escala de notas 1-7.
    ability_mean: float
    ability_sd: float
    # Historia previa: tasa de aprobacion en semestres anteriores.
    historical_pass_rate: float


# Prevalencias en config/synthetic.yml; aqui solo la forma del comportamiento.
PROFILES: dict[str, Profile] = {
    # Asiste, entrega, rinde. Sin sobresaltos.
    "A_high_performer": Profile(
        name="A_high_performer",
        attendance_start=0.93, attendance_drift=-0.02, attendance_volatility=0.04,
        canvas_start=0.80, canvas_drift=0.00, canvas_volatility=0.12,
        submission_start=0.96, submission_drift=-0.01,
        ability_mean=5.9, ability_sd=0.45,
        historical_pass_rate=0.97,
    ),
    # Rinde alto con poca huella digital. El caso que mas falsos positivos
    # genera si el modelo se apoya demasiado en Canvas.
    "B_silent_high": Profile(
        name="B_silent_high",
        attendance_start=0.82, attendance_drift=-0.05, attendance_volatility=0.07,
        canvas_start=0.22, canvas_drift=-0.03, canvas_volatility=0.10,
        submission_start=0.88, submission_drift=-0.02,
        ability_mean=5.6, ability_sd=0.50,
        historical_pass_rate=0.94,
    ),
    # Se esfuerza y no le alcanza. Alta entrega, notas bajas.
    # El modelo debe distinguirlo de D: mismo riesgo, distinta causa.
    "C_struggling_engaged": Profile(
        name="C_struggling_engaged",
        attendance_start=0.88, attendance_drift=-0.03, attendance_volatility=0.05,
        canvas_start=0.78, canvas_drift=0.04, canvas_volatility=0.13,
        submission_start=0.92, submission_drift=-0.02,
        ability_mean=4.1, ability_sd=0.55,
        historical_pass_rate=0.72,
    ),
    # Riesgo academico sostenido, sin senal de quiebre.
    "D_academic_risk": Profile(
        name="D_academic_risk",
        attendance_start=0.81, attendance_drift=-0.12, attendance_volatility=0.09,
        canvas_start=0.42, canvas_drift=-0.06, canvas_volatility=0.15,
        submission_start=0.62, submission_drift=-0.08,
        ability_mean=3.6, ability_sd=0.60,
        historical_pass_rate=0.55,
    ),
    # Empieza bien y se desengancha. Es el caso de uso 2 del proyecto:
    # el unico perfil donde la deteccion temprana cambia el resultado.
    "E_disengaging": Profile(
        name="E_disengaging",
        attendance_start=0.90, attendance_drift=-0.44, attendance_volatility=0.07,
        canvas_start=0.72, canvas_drift=-0.60, canvas_volatility=0.12,
        submission_start=0.85, submission_drift=-0.70,
        ability_mean=4.6, ability_sd=0.65,
        historical_pass_rate=0.80,
    ),
    # Volatil: semanas activas y semanas desconectadas. Genera ruido y
    # castiga a los modelos que reaccionan a una sola semana mala.
    "F_intermittent": Profile(
        name="F_intermittent",
        attendance_start=0.82, attendance_drift=-0.12, attendance_volatility=0.24,
        canvas_start=0.55, canvas_drift=-0.05, canvas_volatility=0.30,
        submission_start=0.70, submission_drift=-0.10,
        ability_mean=4.3, ability_sd=0.70,
        historical_pass_rate=0.68,
    ),
    # Arranca mal y remonta. Existe para castigar modelos que solo miran
    # el nivel y no la tendencia: alertar aqui es un falso positivo.
    "G_recovering": Profile(
        name="G_recovering",
        attendance_start=0.66, attendance_drift=0.30, attendance_volatility=0.08,
        canvas_start=0.35, canvas_drift=0.45, canvas_volatility=0.14,
        submission_start=0.55, submission_drift=0.38,
        ability_mean=4.4, ability_sd=0.55,
        historical_pass_rate=0.63,
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Perfil desconocido: {name}. Validos: {sorted(PROFILES)}")
    return PROFILES[name]
