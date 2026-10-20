"""Vista local institucional, independiente de las corridas de datos públicos."""
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .dashboard import number, percent
from .institucional import (
    FILES, export_aggregate, grouped_results, load, select_data, summary,
)

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
        digits = 2 if col == "Promedio numérico" else 1 if "(%)" in col else 0
        formats[col] = lambda value, digits=digits: number(value, digits)
    st.dataframe(table.style.format(formats, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Descargar tabla filtrada", export_aggregate(table, context),
                       name + ".csv", "text/csv", key="inst_download_" + name)


def render(home):
    st.title("Datos institucionales · Administración y Negocios")
    st.caption("Uso interno · Extracto universitario 2024–2025 · Análisis descriptivo por período")
    base = Path(home) / "artifacts/institucional"
    pointer = base / "latest.json"
    if not pointer.exists():
        st.info("La fuente institucional está documentada. Falta generar sus tablas analíticas locales.")
        st.code("python -m piloto_fan.institucional", language="bash")
        return
    try:
        run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
        if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("Identificador de ejecución inválido.")
        folder = base / run_id
        fingerprint = tuple((name, (folder/name).stat().st_mtime_ns, (folder/name).stat().st_size)
                            for name in [*FILES.values(), "quality.json", "pendientes.json", "manifest.json"])
        frames, quality, manifest = load_cached(str(folder), fingerprint)
        cfg = json.loads((folder/"pendientes.json").read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError) as exc:
        st.error(f"No se pudieron verificar los resultados institucionales: {exc}")
        return
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
        for question in cfg["preguntas"]:
            with st.expander(f"{question['id']} · {question['pregunta']}"):
                st.write("**Estado:** " + question["estado"] + " · **Prioridad:** " + question["prioridad"])
                st.write("**Contraparte sugerida:** " + question["responsable_sugerido"])
                st.write("**Mientras tanto:** " + question["avanzamos_con"])
                st.write("**La respuesta habilita:** " + question["habilita"])
                st.write("**Límite actual:** " + question["restriccion_actual"])
                if question.get("respuesta"):
                    st.write("**Respuesta registrada:** " + question["respuesta"])
        question_text = "\n\n".join(
            f"{q['id']}. {q['pregunta']}\nResponsable sugerido: {q['responsable_sugerido']}\n"
            f"Estado: {q['estado']}\nMientras tanto: {q['avanzamos_con']}" for q in cfg["preguntas"])
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
