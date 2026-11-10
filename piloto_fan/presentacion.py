"""Recorrido para presentar los resultados a la FAN.

La portada cuenta la historia de afuera hacia adentro, en cuatro pasos:

1. **Posición frente a pares**: cómo está la FAN comparada con universidades
   parecidas (datos públicos).
2. **Mapa de asignaturas**: dónde se concentra el problema académico (extracto).
3. **Perfiles de ingreso**: quién llega a cada sede (datos públicos).
4. **Radar**: cómo se priorizaría el acompañamiento (extracto, retrospectivo).

Cada cifra de la portada se calcula con los mismos datos y funciones que su
vista de detalle: no hay números escritos a mano que puedan quedar viejos.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import yaml

from . import cohorte_nueva, radar
from .dashboard import number, site_mix
from .institucional import course_campus_map, select_data

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LANDING = "Resultados para la FAN"
POSITION = "1 · Posición frente a pares"
MAP = "2 · Mapa de asignaturas"
PROFILES = "3 · Perfiles de ingreso"
RADAR = "4 · Radar (prototipo)"
STORY = [LANDING, POSITION, MAP, PROFILES, RADAR]
BENCHMARK = "Benchmark completo"
AREA = "Administración y Comercio"
MATH = r"MATEM|C[ÁA]LCULO|[ÁA]LGEBRA|ESTAD[ÍI]ST"
FUTURE = pd.DataFrame([
    {"Si llega": "Extracto con 2026-1", "Permite": "Evaluar el preregistro sellado"},
    {"Si llega": "Extracto cada semestre", "Permite": "Usar el radar y mantener el mapa al día"},
    {"Si llega": "Códigos de resultado y calendario", "Permite": "Tasa oficial de reprobación"},
    {"Si llega": "Datos abiertos 2027", "Permite": "Continuidad de la cohorte 2026"},
    {"Si llega": "Registro de lo que hace cada dirección", "Permite": "Medir si el acompañamiento mejora la retención"},
])


def pct(x, digits=1):
    return "—" if x is None or pd.isna(x) else number(100 * x, digits) + "%"


def go(view):
    st.session_state["section"] = view


# ---------------------------------------------------------------- cálculos

@st.cache_data(show_spinner=False)
def position(signature: tuple) -> dict:
    from student_analytics.modeling.area_position import area_position
    config = yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / "deploy/data/retention_universities.parquet")
    profiles = pd.read_parquet(ROOT / "deploy/data/benchmark_profiles.parquet")
    return area_position(data, profiles, config, AREA)


def position_signature() -> tuple:
    files = [ROOT / "deploy/data/retention_universities.parquet", ROOT / "deploy/data/benchmark_profiles.parquet",
             ROOT / "config/benchmark.yml"]
    return tuple((p.name, p.stat().st_mtime_ns) for p in files)


def map_summary(h: pd.DataFrame, minimum: int) -> dict | None:
    table = course_campus_map(h, minimum)
    if table.empty:
        return None
    volume = table.groupby("Código de asignatura")["Notas bajo 4"].sum().sort_values(ascending=False)
    names = table.drop_duplicates("Código de asignatura").set_index("Código de asignatura")["Asignatura"]
    first = table.loc[table["Código de asignatura"].eq(volume.index[0])]
    low = first.loc[first["Notas numéricas bajo 4 (%)"].idxmin()]
    high = first.loc[first["Notas numéricas bajo 4 (%)"].idxmax()]
    return {"courses": len(volume), "top10_share": volume.head(10).sum() / volume.sum(),
            "top10_math": int(names.loc[volume.head(10).index].str.contains(MATH, regex=True).sum()),
            "course": names.loc[volume.index[0]],
            "low": (low["Sede de la asignatura"], low["Notas numéricas bajo 4 (%)"] / 100),
            "high": (high["Sede de la asignatura"], high["Notas numéricas bajo 4 (%)"] / 100),
            "spread": len(first) > 1}


def profile_data(frames: dict, run_id: str) -> tuple[pd.DataFrame, list[int]]:
    """Cohortes de la corrida con su perfil, más las que llegaron después."""
    data = frames["continuidad_dataset.parquet"].merge(frames["segmentacion_assignments.parquet"],
                                                       on=["mrun", "cohorte"], validate="one_to_one")
    later = cohorte_nueva.load(run_id)
    if later is None:
        return data, []
    return pd.concat([data, later], ignore_index=True), sorted(later.cohorte.unique().astype(int).tolist())


def profile_summary(data: pd.DataFrame, minimum: int) -> dict | None:
    year = int(data.cohorte.max())
    reference = [y for y in sorted(data.cohorte.unique().astype(int)) if y < year][-2:]
    if not reference:
        return None
    nem = data.groupby("segment").percentil_nem.mean()
    low, high = int(nem.idxmin()), int(nem.idxmax())
    known = data.loc[data.target.notna()]
    continuity = 1 - known.groupby("segment").target.mean()
    mix = site_mix(data, year, reference, minimum)
    lows = mix.loc[mix.Perfil.eq(f"Perfil {low}")].dropna(subset=["Participación (%)"])
    highs = mix.loc[mix.Perfil.eq(f"Perfil {high}")].dropna(subset=["Cambio (pp)"])
    if lows.empty or highs.empty:
        return None
    top = lows.loc[lows["Participación (%)"].idxmax()]
    rise = highs.loc[highs["Cambio (pp)"].idxmax()]
    return {"year": year, "reference": reference, "n": int(data.cohorte.eq(year).sum()),
            "low": low, "high": high, "low_continuity": continuity.get(low), "high_continuity": continuity.get(high),
            "top_site": top.Sede, "top_share": top["Participación (%)"] / 100,
            "rise_site": rise.Sede, "rise_pp": rise["Cambio (pp)"]}


def radar_summary(d: pd.DataFrame, capacity: float = .2) -> dict:
    table = radar.retrospective(radar.prioritize(d, capacity), capacity)
    return {"slots": table["Cupos"].sum(), "hits": table["Aciertos del radar"].sum(),
            "rule": table["Aciertos de la regla simple"].sum(), "random": table["Aciertos al azar"].sum(),
            "directions": len(table), "capacity": capacity}


# ---------------------------------------------------------------- vistas

def block(step, title, target, body):
    with st.container(border=True):
        st.subheader(f"{step} · {title}")
        body()
        st.button(f"Ver {title.lower()} →", on_click=go, args=(target,), key=f"go_{step}")


def render_landing(profiles_data: pd.DataFrame | None, minimum: int):
    from .institucional_ui import latest as institutional
    from .modelos_institucionales_ui import cached_radar, latest as models

    st.title("Resultados para la Facultad de Administración y Negocios")
    st.write("Cuatro productos construidos solo con lo que la universidad ya tiene: el extracto de Banner "
             "2024-1 a 2025-2 y los datos abiertos del Estado hasta 2026. El recorrido va de afuera hacia adentro: "
             "cómo está la FAN frente a universidades parecidas, dónde se concentra el problema académico, quién "
             "llega a cada sede y cómo se priorizaría el acompañamiento.")
    st.info("Nada aquí es una lista de estudiantes actuales ni una evaluación de docentes o sedes. Son "
            "descripciones y una prueba retrospectiva para decidir dónde mirar.")

    def p4():
        try:
            r = position(position_signature())
        except (OSError, ValueError, KeyError) as exc:
            st.info(f"No se pudo calcular la comparación con pares: {exc}")
            return
        a, b = st.columns([1, 2])
        a.metric(f"Sigue en la misma carrera · cohorte {r['year']}", pct(r["ua"]),
                 number(100 * (r["ua"] - r["peers_rate"]), 1) + " pp frente a pares", delta_color="off")
        b.write(f"El área de {AREA} retiene más que sus {len(r['peers'])} universidades pares "
                f"({pct(r['peers_rate'])}) y lo hizo en {r['ahead']} de las últimas {r['cohorts']} cohortes. "
                f"Pero queda bajo el promedio de la propia UA ({pct(r['whole_ua'])}): el margen de mejora "
                "está dentro de la universidad.")
        b.caption("Datos públicos Mineduc/SIES. Pares elegidos por perfil de ingreso, nunca por retención.")

    def p2():
        try:
            loaded = institutional(HOME)
        except (OSError, ValueError, KeyError) as exc:
            st.info(f"No se pudo verificar el extracto institucional: {exc}")
            return
        if loaded is None:
            st.info("Falta integrar el extracto institucional en este equipo.")
            return
        _, frames, _, _, cfg = loaded
        _, h, _ = select_data(frames, "fan_proposed")
        s = map_summary(h, cfg["alcance"]["minimo_personas_mostradas"])
        if s is None:
            st.info("No hay asignaturas con tamaño suficiente para publicar.")
            return
        a, b = st.columns([1, 2])
        a.metric("Notas bajo 4 en 10 asignaturas", pct(s["top10_share"], 0),
                 f"de {number(s['courses'])} asignaturas FAN", delta_color="off", delta_arrow="off")
        text = (f"{s['top10_math']} de esas 10 son matemáticas o estadística. ") if s["top10_math"] else ""
        if s["spread"]:
            text += (f"En {s['course'].capitalize()}, la que más notas bajo 4 reúne, van de "
                     f"{pct(s['low'][1], 0)} en {s['low'][0]} a {pct(s['high'][1], 0)} en {s['high'][0]}.")
        b.write(text)
        b.caption("Extracto 2024–2025, uso interno. Sirve para decidir dónde mirar, no para atribuir causas.")

    def p3():
        if profiles_data is None:
            st.info("Falta la corrida del piloto con datos públicos.")
            return
        s = profile_summary(profiles_data, minimum)
        if s is None:
            st.info("No hay cohortes suficientes para comparar perfiles.")
            return
        a, b = st.columns([1, 2])
        a.metric(f"Ingresantes FAN {s['year']}", number(s["n"]),
                 "perfiles congelados, sin reentrenar", delta_color="off", delta_arrow="off")
        b.write(f"En {s['top_site']}, {pct(s['top_share'], 0)} de los ingresantes {s['year']} tiene el perfil "
                f"de NEM y ranking más bajos (Perfil {s['low']}), cuya continuidad histórica es "
                f"{pct(s['low_continuity'])} frente a {pct(s['high_continuity'])} del Perfil {s['high']}. "
                f"{s['rise_site']} es la sede donde más creció el Perfil {s['high']} frente a "
                f"{'–'.join(map(str, s['reference']))} ({number(s['rise_pp'], 1)} pp).")
        b.caption("Datos públicos Mineduc/DEMRE. Los perfiles no son niveles de riesgo.")

    def p1():
        try:
            loaded = models(HOME)
        except (OSError, ValueError, KeyError) as exc:
            st.info(f"No se pudieron verificar los modelos institucionales: {exc}")
            return
        if loaded is None:
            st.info("Faltan los modelos institucionales en este equipo.")
            return
        folder, fingerprint, *_ = loaded
        s = radar_summary(cached_radar(folder, fingerprint))
        a, b = st.columns([1, 2])
        a.metric("Aciertos del radar en 2025-2", pct(s["hits"] / s["slots"], 0),
                 f"{number(s['hits'])} de {number(s['slots'])} listados", delta_color="off", delta_arrow="off")
        b.write(f"Revisando al {pct(s['capacity'], 0)} de cada dirección, el radar habría encontrado "
                f"{number(s['hits'])} estudiantes que terminaron con al menos una nota bajo 4. Una regla "
                f"simple habría encontrado {number(s['rule'])}, y el azar, unos {number(s['random'])}.")
        b.caption("Prueba retrospectiva: ese semestre ya terminó. Opera de verdad cuando llegue un extracto nuevo.")

    block(1, "Posición frente a pares", POSITION, p4)
    block(2, "Mapa de asignaturas", MAP, p2)
    block(3, "Perfiles de ingreso", PROFILES, p3)
    block(4, "Radar de inicio de semestre", RADAR, p1)
    st.subheader("Qué permitiría cada dato futuro")
    st.dataframe(FUTURE, hide_index=True, width="stretch")
    st.caption("San Miguel corresponde al campus El Llano. El alcance FAN es una propuesta documentada, "
               "pendiente de ratificación por la facultad.")


def render_position():
    from student_analytics.modeling.area_position import area_position_html
    st.title("Posición frente a universidades pares")
    st.caption(f"Área {AREA} · Datos públicos Mineduc/SIES, DEMRE y CNED · Se puede compartir sin restricciones")
    try:
        r = position(position_signature())
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"No se pudo calcular la comparación con pares: {exc}")
        return
    a, b, c = st.columns(3)
    a.metric(f"UA · cohorte {r['year']}", pct(r["ua"]))
    b.metric(f"{len(r['peers'])} pares", pct(r["peers_rate"]))
    c.metric("Resto del sistema", pct(r["system"]))
    st.markdown("\n".join("- " + f for f in r["findings"]))
    for title, table in r["sections"]:
        st.subheader(title)
        st.dataframe(table, hide_index=True, width="stretch")
    for note in r["notes"]:
        st.caption(note)
    a, b = st.columns(2)
    a.download_button("Descargar informe HTML", area_position_html(
        r, datetime.now(timezone.utc).strftime("%d/%m/%Y")), "informe_pares_fan.html", "text/html")
    b.button("Abrir el benchmark completo →", on_click=go, args=(BENCHMARK,), key="go_benchmark",
             help="Otras cohortes, áreas e indicadores, los clusters por variables y los modelos que los forman.")


def render_benchmark_full():
    """El benchmark interactivo de la app pública, leído de los datos públicos del repo."""
    from student_analytics.ui.benchmark import render_benchmark
    # Abre en el área de la vista 1 para que las cifras coincidan; las demás quedan a un clic.
    st.session_state.setdefault("bench_area", AREA)
    render_benchmark(ROOT / "deploy" / "data")


def render_map():
    from .institucional_ui import latest as institutional, render_course_map
    st.title("Mapa de asignaturas por sede")
    st.caption("Uso interno · Extracto universitario 2024–2025 · Alcance FAN propuesto")
    st.info("Muestra dónde se concentran las notas bajo 4 y si la misma asignatura rinde distinto según la sede "
            "donde se dicta. Es para decidir dónde mirar —refuerzos, ayudantías, criterios de evaluación—, "
            "no un juicio sobre docentes ni sedes.")
    try:
        loaded = institutional(HOME)
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"No se pudieron verificar los resultados institucionales: {exc}")
        return
    if loaded is None:
        st.info("Falta integrar el extracto institucional en este equipo.")
        st.code("python -m piloto_fan.institucional", language="bash")
        return
    run_id, frames, _, _, cfg = loaded
    available, _, _ = select_data(frames, "fan_proposed")
    a, b = st.columns(2)
    year = a.selectbox("Año", [None] + sorted(available.year.dropna().astype(int).unique().tolist()),
                       format_func=lambda v: "2024 y 2025" if v is None else str(v), key="map_year")
    schedule = b.selectbox("Jornada del plan", [None] + sorted(available.schedule.dropna().unique().tolist()),
                           format_func=lambda v: "Todas" if v is None else v, key="map_schedule")
    filters = {"year": year, "schedule": schedule}
    _, h, _ = select_data(frames, "fan_proposed", filters)
    minimum = cfg["alcance"]["minimo_personas_mostradas"]
    st.write(f"Cada celda es una asignatura en la sede donde se dictó, con al menos {minimum} personas con nota "
             "numérica. La sede es la de la asignatura, no la del plan del estudiante.")
    render_course_map(h, minimum, {"Ejecución institucional": run_id, "Alcance": "FAN propuesto",
                                   "Año": year or "2024 y 2025", "Jornada del plan": schedule or "Todas"})


def render_radar_view():
    from .modelos_institucionales_ui import latest as models, render_radar, reset
    st.title("Radar de inicio de semestre")
    try:
        loaded = models(HOME)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"No se pudieron verificar los modelos institucionales: {exc}")
        return
    if loaded is None:
        st.info("Genera los modelos con las tablas institucionales ya integradas.")
        st.code("python -m piloto_fan.modelos_institucionales")
        return
    folder, fingerprint, run_id, _, _ = loaded
    st.sidebar.subheader("Elegir una dirección")
    st.sidebar.button("Restablecer filtros del radar", on_click=reset)
    render_radar(folder, fingerprint, run_id)
