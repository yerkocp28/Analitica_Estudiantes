"""Resultados agregados de los modelos institucionales; nunca publica predicciones individuales."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .dashboard import export_csv, number, percent
from .models import capture, metrics
from .modelos_institucionales import load_run

VARIABLES = {
    "observed_results": "Cantidad de resultados de asignaturas observados",
    "numeric_mean": "Promedio simple de notas numéricas",
    "numeric_sd": "Dispersión de notas numéricas",
    "numeric_coverage": "Proporción de resultados con nota numérica",
    "below4_fraction": "Fracción de notas numéricas bajo 4",
    "career_code": "Carrera del período de origen",
    "campus_code": "Campus del plan del período de origen",
    "schedule": "Jornada del período de origen",
}


@st.cache_data
def cached(folder, fingerprint):
    return load_run(folder)


def display_table(table):
    formats = {}
    for col in table.select_dtypes(include="number"):
        digits = (3 if col in ["AUC", "Brier", "Silueta"] else
                  2 if col == "Promedio numérico" else
                  1 if "(%)" in col or col == "Resultados observados por persona" else 0)
        formats[col] = lambda value, digits=digits: number(value, digits)
    st.dataframe(table.style.format(formats, na_rep="—"), hide_index=True, width="stretch")


def filters(frame):
    chosen = {}
    for col, label in [("campus", "Campus de origen"), ("career", "Carrera de origen"), ("schedule", "Jornada de origen")]:
        options = [None] + sorted(frame[col].dropna().unique())
        key = "im_" + col
        if st.session_state.get(key) not in options:
            st.session_state[key] = None
        value = st.sidebar.selectbox(label, options, format_func=lambda v: "Todos" if v is None else v, key=key)
        chosen[label] = value or "Todos"
        if value is not None:
            frame = frame.loc[frame[col].eq(value)]
    st.caption("Selección: " + " · ".join(f"{k}: {v}" for k,v in chosen.items()))
    return frame, chosen


def reset():
    for col in ["campus", "career", "schedule"]:
        st.session_state.pop("im_" + col, None)


def render(home):
    st.title("Modelos institucionales · FAN")
    st.warning("Prueba retrospectiva experimental. Se supone que las notas estaban disponibles al cierre del semestre "
               "de origen; falta confirmar sus fechas reales. No son alertas para estudiantes actuales.")
    base = Path(home)/"artifacts/institucional_modelos"
    pointer = base/"latest.json"
    if not pointer.exists():
        st.info("Genera los modelos con las tablas institucionales ya integradas.")
        st.code("python -m piloto_fan.modelos_institucionales")
        return
    try:
        run_id = json.loads(pointer.read_text())["run_id"]
        if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("Ejecución inválida.")
        folder = base/run_id
        fingerprint = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file())
        report, frames = cached(str(folder), fingerprint)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"No se pudieron verificar los modelos: {exc}")
        return
    latest_data = Path(home)/"artifacts/institucional/latest.json"
    if latest_data.exists() and json.loads(latest_data.read_text())["run_id"] != report["source_run_id"]:
        st.warning("La integración de datos cambió después de este entrenamiento. Se muestran los resultados "
                   "de la fuente anterior identificada en la trazabilidad.")
    choices = {"academico": "Resultado académico", "matricula": "Presencia en matrícula", "segmentos": "Perfiles académicos"}
    case = st.selectbox("Análisis institucional", list(choices), format_func=choices.get, key="im_case")
    st.sidebar.subheader("Explorar modelos institucionales")
    st.sidebar.button("Restablecer filtros de modelos", on_click=reset)
    st.sidebar.caption("Filtros independientes del tablero descriptivo y de los modelos con datos públicos.")
    if case == "segmentos":
        s = report["segments"]
        if "winner" not in s:
            st.info(s.get("reason", "Ninguna segmentación cumple las condiciones de tamaño y concentración."))
        else:
            all_d = frames["segmentacion_assignments"]
            term = st.selectbox("Semestre del perfil observado", sorted(all_d.term.unique()),
                                index=len(all_d.term.unique())-1, key="im_profile_term")
            d, selected = filters(all_d.loc[all_d.term.eq(term)].copy())
            st.write(s["interpretation"])
            st.caption(f"Modelo general: {s['winner']} · {s['k']} grupos · aprendido en {s['fit_term']} "
                       f"y aplicado sin reajuste a los semestres posteriores · silueta {number(s['silhouette'], 3)}.")
            rows = []
            for label, g in d.groupby("segment"):
                if len(g) >= 10:
                    rows.append({"Perfil": int(label), "Personas": len(g), "Participación (%)": 100*len(g)/len(d),
                                 "Promedio numérico": g.numeric_mean.mean(),
                                 "Resultados observados por persona": g.observed_results.mean(),
                                 "Notas bajo 4 por persona (%)": 100*g.below4_fraction.mean(),
                                 "Cobertura por persona (%)": 100*g.numeric_coverage.mean()})
            if rows:
                table = pd.DataFrame(rows)
                display_table(table)
                st.caption("Promedios entre personas; no equivalen a promediar todas las notas de la facultad. "
                           "Los grupos con menos de 10 personas se omiten sin cambiar el denominador total.")
                st.download_button("Descargar perfiles filtrados", export_csv(table, {}, {
                    "Ejecución": run_id, "Semestre": int(term), **selected}), "perfiles_academicos.csv", "text/csv")
            else:
                st.info("No hay perfiles con al menos 10 personas en esta selección.")
            with st.expander("Comparación general de segmentaciones · sin filtros"):
                candidates = pd.DataFrame(s["candidates"]).rename(columns={
                    "algorithm": "Algoritmo", "k": "Grupos", "silhouette": "Silueta",
                    "smallest": "Grupo menor", "largest_share": "Grupo mayor (%)", "admissible": "Admisible"})
                candidates["Grupo mayor (%)"] *= 100
                display_table(candidates)
    else:
        c = report["cases"][case]
        st.subheader(c["title"])
        if "test" not in c:
            st.info(c.get("reason", "Casos insuficientes."))
        else:
            st.caption(f"Ganador general: {c['winner']} · estado: experimental · origen 2025-1, resultado 2025-2. "
                       "Los filtros no reentrenan ni cambian el período de prueba.")
            for warning in c.get("diagnostics", []):
                st.warning("Diagnóstico general, sin filtros: " + warning)
            d, selected = filters(frames[case+"_predictions"].copy())
            if len(d) < 10:
                st.info("Menos de 10 personas evaluables en esta selección; se omiten las métricas.")
            else:
                m = metrics(d.target, d.risk_score, c["reference_prevalence"])
                a, b, e, f = st.columns(4)
                a.metric("Personas evaluadas", number(m["n"]))
                b.metric("Resultado observado", percent(m["prevalence"]))
                e.metric("AUC", number(m["auc"], 3))
                f.metric("Brier", number(m["brier"], 3))
                st.caption("AUC mide ordenamiento; Brier mide error de probabilidad (menor es mejor). "
                           "“—” indica que la métrica no se puede calcular.")
                if m["events"] < 30 or m["n"]-m["events"] < 30:
                    st.warning("Pocos eventos de una clase: la evaluación del grupo es inestable.")
                st.write(f"Predicción media: **{percent(m['mean_prediction'])}** · frecuencia observada: "
                         f"**{percent(m['prevalence'])}**. No interpretar las probabilidades como calibradas.")
                capacity = st.select_slider("Capacidad de revisión (%)", [5, 10, 20, 30, 50],
                                            value=20, key="im_capacity")
                q = capacity/100
                slots = int(np.ceil(len(d)*q))
                cap = capture(d.target, d.risk_score, q)
                a, b, e = st.columns(3)
                a.metric("Cupos históricos", number(slots))
                b.metric("Captura del modelo", percent(cap))
                e.metric("Precisión de la prioridad", percent(
                    cap*d.target.sum()/slots if d.target.sum() else 0))
                st.caption("La prioridad se recalcula dentro del subgrupo. Captura mide qué fracción de los eventos "
                           "se encuentra; precisión mide cuántas personas priorizadas presentan el evento. No mide impacto.")
                rows = []
                for name, score in [("Modelo con notas", d.risk_score), ("Regla simple", d.rule_score),
                                    ("Solo carrera, campus y jornada", d.contextual_score)]:
                    v = metrics(d.target, score, c["reference_prevalence"])
                    rows.append({"Referencia": name, "AUC": v["auc"],
                                 "Captura a capacidad seleccionada (%)": 100*capture(d.target, score, q)})
                table = pd.DataFrame(rows)
                display_table(table)
                st.caption("Regla: " + c["rule"] + ". El modelo contextual usa el mismo algoritmo sin variables académicas. "
                           "No es una comparación con el modelo público, que utiliza otra población.")
                st.download_button("Descargar comparación filtrada", export_csv(table, {}, {
                    "Ejecución": run_id, "Caso": c["title"], "Origen": 202510, "Resultado": 202520,
                    "Capacidad (%)": capacity, **selected}), "comparacion_modelos_institucionales.csv", "text/csv")
            with st.expander("Selección e incertidumbre general · sin filtros"):
                st.write("Entrenamiento: 2024-1→2024-2. Validación: 2024-2→2025-1. "
                         "Selección por captura al 20%, desempate Brier/AUC; reajuste con los dos períodos anteriores. "
                         "Prueba reservada: 2025-1→2025-2.")
                table = pd.DataFrame(c["validation_candidates"])[["algorithm", "auc", "brier", "capture_20"]].rename(
                    columns={"algorithm": "Algoritmo", "auc": "AUC", "brier": "Brier", "capture_20": "Captura al 20% (%)"})
                table["Captura al 20% (%)"] *= 100
                display_table(table)
                ci = c["uncertainty"]
                st.write("Captura general al 20%, intervalo bootstrap 95%: " +
                         " – ".join(percent(v) for v in ci["capture_20_ci"]) + ".")
                st.write("Diferencia frente a la regla, intervalo 95%: " +
                         " – ".join(number(100*v, 1) for v in ci["capture_difference_vs_rule_ci"]) + " puntos porcentuales.")
                st.caption(ci["interpretation"] + " La referencia al 20% es fija, aunque cambie el deslizador.")
                st.write("Frecuencia del evento: entrenamiento " + percent(c.get("selection_training_prevalence")) +
                         ", validación " + percent(c.get("validation_prevalence")) + ", prueba " + percent(c["test"]["prevalence"]) + ".")
    with st.expander("Variables, cobertura, supuestos y trazabilidad"):
        st.write("Variables del semestre de origen:")
        for label in VARIABLES.values():
            st.write("• " + label)
        st.caption("La segmentación usa solo las cinco variables numéricas. No utiliza carrera, campus ni jornada.")
        display_table(pd.DataFrame(report["coverage"]).rename(columns={
            "source_term": "Semestre de origen", "outcome_term": "Semestre de resultado",
            "base_people": "Personas de base", "academic_known": "Resultado académico conocido",
            "academic_unknown": "Resultado académico desconocido", "enrollment_known": "Presencia evaluable"}))
        for item in report["limitations"]:
            st.write("• " + item)
        st.caption("Ejecución de modelos: " + run_id + " · integración fuente: " + report["source_run_id"])
        st.download_button("Descargar informe general de modelos", json.dumps(report, ensure_ascii=False, indent=2),
                           "modelos_institucionales_general.json", "application/json")
