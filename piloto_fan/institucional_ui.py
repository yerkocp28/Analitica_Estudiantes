"""Vista local institucional, independiente de las corridas de datos públicos."""
from datetime import datetime
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .dashboard import number, percent
from .institucional import (
    CONFIG, FILES, course_campus_map, export_aggregate, grouped_results, load, read_config, select_data, summary,
)

# Rampa secuencial de un solo azul (100 → 650): la intensidad es la magnitud.
RAMP = ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"]
MARKS = {"Sobre el resto": " ▲", "Bajo el resto": " ▼"}
ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}

FILTERS = {
    "year": "Año observado", "regime": "Régimen", "term": "Período",
    "campus": "Campus del plan", "career": "Carrera del plan",
    "schedule": "Jornada", "plan_status": "Estado del plan",
}


@st.cache_data
def load_cached(folder, fingerprint):
    return load(Path(folder))


def reset():
    for col in FILTERS:
        st.session_state.pop("inst_" + col, None)
    st.session_state.pop("inst_scope", None)


def table_download(table, name, context):
    if table.empty:
        st.info("No hay grupos con al menos 10 personas para esta selección.")
        return
    formats = {}
    for col in table.select_dtypes(include="number"):
        digits = 2 if col == "Promedio numérico" else 1 if "(%)" in col or "(pp)" in col else 0
        formats[col] = lambda value, digits=digits: number(value, digits)
    st.dataframe(table.style.format(formats, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Descargar tabla filtrada", export_aggregate(table, context),
                       name + ".csv", "text/csv", key="inst_download_" + name)


def course_label(name):
    """«CÁLCULO II» → «Cálculo II»: tipo oración sin perder los números romanos."""
    words = str(name).capitalize().split(" ")
    return " ".join(w.upper() if w.upper() in ROMAN else w for w in words)


def course_heatmap(table, top):
    """Filas: asignaturas con más notas bajo 4; columnas: sede donde se dictaron."""
    volume = table.groupby("Código de asignatura")["Notas bajo 4"].sum().nlargest(top)
    d = table.loc[table["Código de asignatura"].isin(volume.index)].copy()
    d["Fila"] = d["Asignatura"].map(course_label) + " · " + d["Código de asignatura"]
    d["Texto"] = (d["Notas numéricas bajo 4 (%)"].round().astype(int).astype(str) + "%"
                  + d["Lectura"].map(MARKS).fillna(""))
    rows = (d.drop_duplicates("Código de asignatura").set_index("Código de asignatura")
            .loc[volume.index, "Fila"].tolist())
    top_value = max(50.0, float(d["Notas numéricas bajo 4 (%)"].max()))
    y = alt.Y("Fila:N", sort=rows, title=None, axis=alt.Axis(labelLimit=320))
    x = alt.X("Sede de la asignatura:N", title=None, sort="ascending",
              axis=alt.Axis(orient="top", labelAngle=0))
    tooltip = [alt.Tooltip("Asignatura:N"), alt.Tooltip("Código de asignatura:N"),
               alt.Tooltip("Sede de la asignatura:N", title="Sede"),
               alt.Tooltip("Personas con nota numérica:Q", format=","),
               alt.Tooltip("Notas numéricas bajo 4 (%):Q", title="Notas bajo 4 (%)", format=".1f"),
               alt.Tooltip("IC 95% inferior (%):Q", format=".1f"),
               alt.Tooltip("IC 95% superior (%):Q", format=".1f"),
               alt.Tooltip("Resto de sedes bajo 4 (%):Q", format=".1f"),
               alt.Tooltip("Lectura:N")]
    cells = alt.Chart(d).mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        x=x, y=y, tooltip=tooltip,
        color=alt.Color("Notas numéricas bajo 4 (%):Q", title="Notas bajo 4 (%)",
                        scale=alt.Scale(domain=[0, top_value], range=RAMP, clamp=True)))
    text = alt.Chart(d).mark_text(fontSize=12).encode(
        x=x, y=y, text="Texto:N", tooltip=tooltip,
        color=alt.condition(alt.datum["Notas numéricas bajo 4 (%)"] > .45*top_value,
                            alt.value("white"), alt.value("#0b0b0b")))
    return (cells + text).properties(height=26*len(rows))


def latest(home):
    """Integración institucional vigente, verificada; None si todavía no se generó."""
    base = Path(home) / "artifacts/institucional"
    pointer = base / "latest.json"
    if not pointer.exists():
        return None
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Identificador de ejecución inválido.")
    folder = base / run_id
    fingerprint = tuple((name, (folder/name).stat().st_mtime_ns, (folder/name).stat().st_size)
                        for name in [*FILES.values(), "quality.json", "pendientes.json", "manifest.json"])
    frames, quality, manifest = load_cached(str(folder), fingerprint)
    cfg = json.loads((folder/"pendientes.json").read_text(encoding="utf-8"))
    return run_id, frames, quality, manifest, cfg


def render_course_map(h, minimum, context, top_default=15):
    """Cuerpo del mapa de asignaturas por sede sobre los resultados `h` ya filtrados."""
    course_map = course_campus_map(h, minimum)
    if course_map.empty:
        st.info(f"No hay asignaturas con al menos {minimum} personas con nota numérica para esta selección.")
        return
    comparable = course_map.loc[course_map["Lectura"].ne("Única sede publicable")]
    a, b, c = st.columns(3)
    a.metric("Asignaturas comparables entre sedes", number(comparable["Código de asignatura"].nunique()))
    b.metric("Celdas claramente sobre el resto", number(comparable["Lectura"].eq("Sobre el resto").sum()))
    c.metric("Notas bajo 4 en el mapa", number(course_map["Notas bajo 4"].sum()))
    top = st.slider("Asignaturas con más notas bajo 4", 5, 40, top_default, step=5, key="inst_map_top")
    st.altair_chart(course_heatmap(course_map, top), width="stretch")
    st.caption("Filas ordenadas por cantidad de notas bajo 4, que es dónde el refuerzo alcanza a más "
               "estudiantes. ▲ / ▼: la sede está por sobre o por debajo del resto de las sedes en la "
               "misma asignatura y sus intervalos de 95% no se tocan; sin marca, la diferencia no es "
               "clara. Celda vacía: la asignatura no se dictó en esa sede o tuvo menos de "
               f"{minimum} personas.")
    st.warning("Una diferencia entre sedes no distingue docencia, criterios de evaluación ni perfil de "
               "ingreso de los estudiantes. Sirve para decidir dónde mirar, no para atribuir la causa. "
               "Las notas bajo 4 no son la tasa oficial de reprobación.")
    table_download(course_map, "institucional_mapa_asignaturas", context)


def render(home):
    st.title("Datos institucionales · Administración y Negocios")
    st.caption("Uso interno · Extracto universitario 2024–2025 · Análisis descriptivo por período")
    try:
        loaded = latest(home)
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"No se pudieron verificar los resultados institucionales: {exc}")
        return
    if loaded is None:
        st.info("La fuente institucional está documentada. Falta generar sus tablas analíticas locales.")
        st.code("python -m piloto_fan.institucional", language="bash")
        return
    run_id, frames, quality, manifest, cfg = loaded
    scopes = {"fan_proposed": "FAN propuesto · cuatro campus",
              "fan_label": "Todo lo etiquetado FAN · incluye Online e intercambio"}
    st.sidebar.subheader("Explorar datos institucionales")
    st.sidebar.button("Restablecer filtros institucionales", on_click=reset)
    scope = st.sidebar.selectbox("Alcance institucional", list(scopes),
                                format_func=scopes.get, key="inst_scope")
    available, _, _ = select_data(frames, scope)
    periods = dict(zip(available.term, available.period))
    selection = {}
    for col, label in FILTERS.items():
        options = [None] + sorted(available[col].dropna().unique().tolist())
        key = "inst_" + col
        if key not in st.session_state:
            st.session_state[key] = (
                max(options[1:]) if col == "year" and len(options) > 1 else
                "Semestral" if col == "regime" and "Semestral" in options else None)
        elif st.session_state[key] not in options:
            st.session_state[key] = None
        def label_value(value, column=col):
            if value is None:
                return "Todos"
            return str(periods.get(value, value)) if column == "term" else str(value)
        value = st.sidebar.selectbox(label, options, format_func=label_value, key=key)
        selection[col] = value
        if value is not None:
            available = available.loc[available[col].eq(value)]
    m, h, unresolved = select_data(frames, scope, selection)
    minimum = cfg["alcance"]["minimo_personas_mostradas"]
    view = summary(m, h)
    context = {"Ejecución institucional": run_id, "Alcance": scopes[scope],
               "Fecha de corte": "No informada"}
    context.update({label: ("Todos" if selection[c] is None else
                           periods.get(selection[c], selection[c]) if c == "term" else selection[c])
                    for c, label in FILTERS.items()})
    st.write("**Alcance provisional, pendiente de ratificación FAN.** "
             "Los estados seleccionados son los informados en el extracto; no definen por sí solos matrícula efectiva.")
    st.caption("Selección: " + " · ".join(f"{label}: {context[label]}" for label in FILTERS.values()))
    st.caption("Fecha de corte no informada · Procesado " +
               datetime.fromisoformat(manifest["created_utc"]).strftime("%d/%m/%Y") +
               " · Los filtros institucionales son independientes de los del piloto con datos públicos.")
    tabs = st.tabs(["Panorama", "Asignaturas", "Códigos de resultado", "Calidad y preguntas"])
    with tabs[0]:
        if m.empty:
            st.info("Sin registros para esta selección. No se sustituye por el total de la facultad.")
        elif view["people"] < minimum:
            st.info(f"Menos de {minimum} personas: se omiten los indicadores de este grupo.")
        else:
            a, b, c = st.columns(3)
            a.metric("Personas en matrícula / planes", number(view["people"]))
            b.metric("Registros de matrícula / plan", number(view["plan_rows"]))
            c.metric("Personas con resultados atribuibles",
                     number(view["history_people"]) if view["history_people"] >= minimum else "—")
            st.caption("Personas únicas en toda la selección; los registros de plan pueden repetirlas entre períodos. "
                       "Los resultados atribuibles requieren un único plan institucional en ese período.")
            enough = h.loc[h.grade_numeric.notna(), "student_key"].nunique() >= minimum
            a, b, c = st.columns(3)
            a.metric("Cobertura de notas numéricas", percent(view["numeric_coverage"]) if enough else "—")
            b.metric("Promedio simple de notas numéricas", number(view["numeric_mean"], 2) if enough else "—")
            c.metric("Notas numéricas bajo 4", percent(view["numeric_below4_rate"]) if enough else "—")
            st.caption("El denominador del último porcentaje son las notas numéricas de 1 a 7, no estudiantes ni todos "
                       "los resultados. No es una tasa oficial de reprobación. Sin ponderación por CREDIT_HOURS.")
            if not unresolved.empty:
                st.warning("La selección contiene resultados asociados a varios planes. Se conservan en la auditoría "
                           "y se excluyen de los indicadores académicos por carrera hasta confirmar su atribución.")
            if h.campus_mismatch.any():
                st.info("Hay asignaturas cuyo campus difiere del campus del plan. Los filtros usan el campus del plan.")
            if h.empty:
                st.info("No se observaron resultados académicos atribuibles. Esto no equivale a cero asignaturas inscritas.")
            st.subheader("Comparación de períodos")
            table_download(grouped_results(h, ["period"], minimum), "institucional_periodos", context)
            st.subheader("Comparación de carreras")
            table_download(grouped_results(h, ["career"], minimum), "institucional_carreras", context)
            st.subheader("Comparación de campus")
            table_download(grouped_results(h, ["campus"], minimum), "institucional_campus", context)
    with tabs[1]:
        st.write("Resultados por código y nombre original de asignatura, con un único plan institucional. "
                 "No se unen equivalencias curriculares supuestas.")
        table_download(grouped_results(h, ["course_code", "course_name"], minimum),
                       "institucional_asignaturas", context)
    with tabs[2]:
        st.write("Se conservan los códigos originales sin interpretarlos como nota cero ni como un mismo tipo de reprobación.")
        table_download(grouped_results(h.loc[h.result_code.notna()], ["result_code"], minimum),
                       "institucional_codigos", context)
        st.caption("Solo se muestran códigos con al menos 10 personas en la selección. Los códigos y notas ausentes "
                   "no participan en promedios ni en el porcentaje de notas numéricas bajo 4.")
    with tabs[3]:
        st.subheader("Preguntas pendientes y cómo seguimos trabajando")
        st.write("Estas preguntas no detienen el análisis descriptivo. Cada respuesta habilitará los indicadores "
                 "o modelos que hoy requieren esa definición.")
        # Las respuestas llegan sin que cambien los datos: el estado se lee del registro vigente, no de la
        # copia guardada con la corrida, que conserva lo pendiente al momento de integrar el extracto.
        questions = read_config() if CONFIG.exists() else cfg
        st.caption(f"Estado al {datetime.fromisoformat(questions['fecha']).strftime('%d/%m/%Y')}.")
        for question in questions["preguntas"]:
            with st.expander(f"{question['id']} · {question['pregunta']}"):
                st.write("**Estado:** " + question["estado"] + " · **Prioridad:** " + question["prioridad"])
                st.write("**Contraparte sugerida:** " + question["responsable_sugerido"])
                st.write("**Mientras tanto:** " + question["avanzamos_con"])
                st.write("**La respuesta habilita:** " + question["habilita"])
                st.write("**Límite actual:** " + question["restriccion_actual"])
                if question.get("detalle"):
                    st.markdown("**Detalle de la pregunta:**\n" + "\n".join("- " + d for d in question["detalle"]))
                answer = question.get("respuesta")
                if isinstance(answer, dict):
                    # Respuesta por tema: cada clave es un punto de la pregunta.
                    st.markdown(f"**Respuesta registrada** ({answer.get('via', 'vía no indicada')}, "
                                f"{answer.get('fecha', 'sin fecha')}):\n" + "\n".join(
                                    f"- **{key.replace('_', ' ').capitalize()}:** {value}"
                                    for key, value in answer.items() if key not in ("fecha", "via")))
                elif answer:
                    st.write("**Respuesta registrada:** " + answer)
                if question.get("verificado_contra_los_datos"):
                    st.write("**Verificado contra los datos:** " + question["verificado_contra_los_datos"])
                if question.get("pendiente_aun"):
                    st.markdown("**Sigue pendiente:**\n" + "\n".join("- " + p for p in question["pendiente_aun"]))
        question_text = "\n\n".join(
            f"{q['id']}. {q['pregunta']}\nResponsable sugerido: {q['responsable_sugerido']}\n"
            f"Estado: {q['estado']}\nMientras tanto: {q['avanzamos_con']}"
            + ("\nSigue pendiente: " + "; ".join(q["pendiente_aun"]) if q.get("pendiente_aun") else "")
            for q in questions["preguntas"])
        st.download_button("Descargar preguntas para la universidad", question_text.encode("utf-8-sig"),
                           "preguntas_fuente_institucional.txt", "text/plain")
        st.subheader("Auditoría de todo el extracto · sin filtros")
        controls = [
            ("Registros de historia", quality["history_rows"]),
            ("Registros de matrícula / plan", quality["enrollment_rows"]),
            ("Persona–período con varios planes", quality["multiple_plan_person_periods"]),
            ("Persona–período de matrícula sin historia observada", quality["enrollment_person_periods_without_history"]),
        ]
        controls += [("Historia: " + key, val) for key, val in quality["history_attribution"].items()]
        st.dataframe(pd.DataFrame(controls, columns=["Control general", "Registros"])
                     .style.format({"Registros": lambda v: number(v)}),
                     hide_index=True, width="stretch")
        st.write("**Aplicación al proyecto:** la vista **Modelos institucionales** presenta la segmentación académica "
                 "y dos experimentos predictivos entre períodos, cuando estén generados. Los modelos públicos de "
                 "continuidad y titulación conservan sus resultados. No se producen alertas individuales actuales.")
        st.caption("Fuente: " + manifest["source_file"] + " · Ejecución: " + run_id)
