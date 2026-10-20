"""Streamlit FAN independiente: streamlit run piloto_fan/app.py --server.port 8502."""
from datetime import datetime
import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piloto_fan.dashboard import DIMENSIONS, display_name, export_csv, filtered, filtered_case, number, percent, profiles, series
from piloto_fan.data import digest
from piloto_fan.reporting import executive_html

HOME = Path(__file__).resolve().parent
RED = "#ae1935"
st.set_page_config(page_title="Administración y Negocios | Piloto UA", page_icon="🎓", layout="wide")
st.markdown("""<style>.stMainBlockContainer{max-width:1400px;padding-top:2rem}
div[data-testid="stMetric"]{border:1px solid #eee;border-top:3px solid #ae1935;border-radius:8px;padding:14px}
</style>""", unsafe_allow_html=True)
logo = ROOT / "documentacion/assets/logo-ua.png"
if logo.exists():
    st.sidebar.image(str(logo), width=140)
st.sidebar.title("Administración y Negocios")
st.sidebar.caption("Universidad Autónoma de Chile · Piloto de trayectorias")


@st.cache_data
def load_run(run_id, fingerprint):
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Identificador de ejecución inválido.")
    folder = HOME / "artifacts" / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    names = ["report.json", "poblacion_dataset.parquet", "continuidad_dataset.parquet"]
    names += [f"{key}_predictions.parquet" for key, value in report["cases"].items() if "test" in value]
    if "winner" in report["segments"]:
        names.append("segmentacion_assignments.parquet")
    for name in names:
        if name not in manifest["artifacts_sha256"] or digest(folder/name) != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Artefacto ausente o alterado: {name}. Ejecuta python -m piloto_fan.build")
    return report, {name: pd.read_parquet(folder/name) for name in names if name.endswith(".parquet")}


section = st.sidebar.radio("Vista", ["Resumen", "Datos institucionales FAN", "Modelos institucionales", "Perfiles de ingreso",
                                     "Continuidad", "Titulación", "Calidad y método", "Demostración", "Benchmark"])
if section == "Datos institucionales FAN":
    from piloto_fan.institucional_ui import render
    render(HOME)
    st.stop()
if section == "Modelos institucionales":
    from piloto_fan.modelos_institucionales_ui import render
    render(HOME)
    st.stop()

pointer = HOME / "artifacts/latest.json"
if not pointer.exists():
    st.title("Facultad de Administración y Negocios")
    st.info("Genera los resultados del piloto con las fuentes locales existentes.")
    st.code("python -m piloto_fan.build", language="bash")
    st.stop()
try:
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    folder = HOME / "artifacts" / run_id
    fingerprint = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in folder.iterdir() if p.is_file())
    report, frames = load_run(run_id, fingerprint)
except (OSError, KeyError, ValueError) as exc:
    st.error(f"No se pudo verificar el piloto: {exc}")
    st.stop()

cfg = report["config"]
minimum = cfg["minimum_display_cell"]
population = frames["poblacion_dataset.parquet"]


def clear_filters():
    for key in DIMENSIONS:
        st.session_state["filter_" + key] = None


selection = {}
if section in ["Resumen", "Perfiles de ingreso", "Continuidad", "Titulación", "Calidad y método"]:
    st.sidebar.subheader("Explorar la facultad")
    st.sidebar.button("Restablecer filtros", on_click=clear_filters)
    available = population
    for col, label in DIMENSIONS.items():
        options = [None] + sorted(available[col].dropna().unique().tolist())
        key = "filter_" + col
        if st.session_state.get(key) not in options:
            st.session_state[key] = None
        value = st.sidebar.selectbox(label, options, format_func=display_name, key=key)
        selection[col] = value
        available = filtered(available, {col: value})
    st.sidebar.caption("Sede, carrera y jornada al ingreso. Los filtros no reentrenan modelos.")
population_view = filtered(population, selection)
st.sidebar.caption("Datos reales · actualización " + datetime.fromisoformat(report["created_utc"]).strftime("%d/%m/%Y"))
with st.sidebar.expander("Informe general y trazabilidad"):
    st.caption("Descargas de todo el piloto, sin filtros de pantalla.")
    st.download_button("Informe general HTML", executive_html(report), "piloto_fan_general.html", "text/html")
    st.download_button("Resultados generales JSON", json.dumps(report, ensure_ascii=False, indent=2), "piloto_fan_general.json", "application/json")
    st.caption("Ejecución " + run_id)


def context_line():
    st.caption("Selección: " + " · ".join(f"{DIMENSIONS[k]}: {display_name(v)}" for k, v in selection.items()))


def show_table(table):
    if table.empty:
        st.info("Sin filas publicables para esta selección.")
        return
    formats = {}
    for col in table.select_dtypes(include="number"):
        if "Cohorte" in col or "Año" in col:
            formats[col] = "{:.0f}"
        elif any(word in col for word in ["Estudiantes", "Cupos", "Casos", "Seguimientos", "Grupos", "Mínimo"]):
            formats[col] = "{:,.0f}"
        else:
            formats[col] = "{:,.3f}" if col in ["AUC", "Brier", "Silueta", "Pérdida de AUC", "Desviación"] else "{:,.1f}"
    st.dataframe(table.style.format(formats, na_rep="—", decimal=",", thousands="."), hide_index=True, width="stretch")


def download(table, label, name, **context):
    if not table.empty:
        st.download_button(label, export_csv(table, selection, {"Ejecución": run_id, **context}), name, "text/csv", key=name)


def year_choice(data):
    years = sorted(data.cohorte.unique().astype(int).tolist())
    if not years:
        st.info("Sin cohortes para esta selección.")
        st.stop()
    if st.session_state.get("cohort") not in years:
        st.session_state["cohort"] = years[-1]
    return st.selectbox("Cohorte de ingreso", years, key="cohort")


def case_view(key, title):
    case = report["cases"][key]
    st.title(title)
    context_line()
    st.write(case["features"]["target"])
    if "test" not in case:
        st.warning(case["reason"])
        return
    pred = filtered(frames[f"{key}_predictions.parquet"], selection)
    st.caption(f"Modelo general: {case['winner']} · estado general: {case['status']}. Selección en "
               f"{case['validation_year']}; evaluación posterior en {case['test_year']}.")
    if pred.empty:
        st.info("La selección no tiene casos en el año reservado para evaluar este modelo. Cambia los filtros; no se reemplaza por el total de la facultad.")
        return
    years = str(int(pred.cohorte.min())) if pred.cohorte.nunique() == 1 else f"{int(pred.cohorte.min())}–{int(pred.cohorte.max())}"
    st.write(f"**Cohorte de ingreso: {years} · "
             f"año de evaluación: {case['test_year']} · seguimiento hasta {int(pred.outcome_year.max())}.**")
    view = filtered_case(pred, case, cfg)
    if view is None:
        st.info(f"Menos de {minimum} casos. Se omiten las métricas del grupo por tamaño insuficiente.")
        return
    m = view["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Casos evaluados en la selección", number(m["n"]))
    b.metric("Resultado adverso observado", percent(m["prevalence"]))
    c.metric("AUC de la selección", number(m["auc"], 3))
    d.metric("Brier de la selección", number(m["brier"], 3))
    st.caption("AUC: ordenamiento (0,5 equivale al azar). Brier: error de probabilidad, menor es mejor. “—” significa no calculable.")
    if m["auc"] is None:
        st.warning("Este grupo contiene una sola clase de resultado; no permite calcular AUC.")
    if len(pred) < cfg["minimum_evaluation"] or view["uncertainty"] is None:
        st.warning("Subgrupo pequeño o con pocos eventos: lectura exploratoria, sin conclusión firme sobre desempeño.")
    if abs(m["mean_prediction"] - m["prevalence"]) >= .05:
        st.warning(f"Calibración a revisar: probabilidad media {percent(m['mean_prediction'])} frente a frecuencia real "
                   f"{percent(m['prevalence'])}. Estas probabilidades no están habilitadas para uso operativo.")
    st.subheader("Capacidad de revisión dentro de la selección")
    q = st.select_slider("Capacidad de revisión (%)", [5, 10, 20, 30, 50], value=20, key=f"capacity_{key}")
    capacities = view["capacity"]
    row = capacities[capacities["Capacidad (%)"].eq(q)].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Cupos históricos de revisión", number(row["Cupos"]))
    difference = row["Captura modelo (%)"]-row["Captura regla (%)"]
    b.metric("Casos adversos capturados", percent(row["Captura modelo (%)"]/100),
             number(difference, 1) + " pp frente a regla" if pd.notna(difference) else None)
    c.metric("Precisión de la prioridad", percent(row["Precisión (%)"]/100))
    st.caption("Se recalcula la prioridad dentro del grupo filtrado. Captura: fracción de todos los casos adversos encontrada. "
               "Precisión: fracción de casos adversos dentro de la prioridad. No mide el efecto de contactar personas.")
    long = capacities.melt(id_vars="Capacidad (%)", value_vars=["Captura modelo (%)", "Captura regla (%)", "Captura azar (%)"], var_name="Referencia", value_name="Captura (%)")
    long["Detalle"] = long["Captura (%)"].map(lambda v: percent(v/100))
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Capacidad (%):Q", title="Capacidad de revisión (%)", axis=alt.Axis(format=".0f")),
        y=alt.Y("Captura (%):Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(format=".0f")),
        color="Referencia:N", tooltip=["Referencia", "Detalle", "Capacidad (%)"]), width="stretch")
    ci = view["uncertainty"]
    if ci:
        lo, hi = ci["capture_20_ci"]
        dlo, dhi = ci["capture_difference_vs_rule_ci"]
        st.write(f"**Referencia fija al 20% para esta selección:** intervalo bootstrap 95% de captura "
                 f"{percent(lo)}–{percent(hi)}; ventaja sobre la regla entre {number(dlo*100, 1)} y {number(dhi*100, 1)} pp.")
        st.caption(ci["interpretation"])
    download(capacities, "Descargar comparación filtrada", f"{key}_seleccion.csv", Caso=title,
             **{"Año de evaluación": case["test_year"], "Cohorte de ingreso": int(pred.cohorte.min()), "Casos evaluados": len(pred), "Estado general del modelo": case["status"]})
    with st.expander("Calibración de la selección"):
        points = alt.Chart(view["calibration"]).mark_line(point=True, color=RED).encode(
            x=alt.X("Predicción:Q", title="Probabilidad promedio (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Frecuencia observada:Q", title="Frecuencia real (%)", scale=alt.Scale(domain=[0, 100])))
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
        st.altair_chart(points + diagonal, width="stretch")
        st.caption("La diagonal representa calibración ideal. Una buena AUC no garantiza probabilidades calibradas.")
    with st.expander("Selección de algoritmos y variables: evaluación general sin filtros"):
        table = pd.DataFrame(case["validation_candidates"])[["algorithm", "n", "auc", "brier", "capture_20"]]
        table["capture_20"] *= 100
        show_table(table.rename(columns={"algorithm": "Algoritmo", "n": "Casos", "auc": "AUC", "brier": "Brier", "capture_20": "Captura al 20% (%)"}))
        st.caption("Validación general: el algoritmo no cambia al seleccionar una sede o carrera.")
        st.write(case["features"]["timing"])
        importance = pd.DataFrame(case["importance_validation"]).sort_values("auc_loss", ascending=False)
        labels = {"percentil_admision": "Percentil de admisión", "percentil_nem": "Percentil NEM", "percentil_ranking": "Percentil ranking",
                  "duracion_nominal_anios": "Duración nominal", "nomb_carrera": "Carrera al ingreso", "nomb_sede": "Sede al ingreso", "jornada": "Jornada al ingreso",
                  "years_enrolled": "Años matriculados UA", "years_without_ua": "Años sin matrícula UA", "career_codes_observed": "Códigos de carrera observados",
                  "sites_observed": "Sedes observadas", "schedules_observed": "Jornadas observadas"}
        importance["variable"] = importance.variable.map(lambda x: labels.get(x, x))
        show_table(importance.rename(columns={"variable": "Variable", "auc_loss": "Pérdida de AUC", "std": "Desviación"}))
        st.caption("Importancia global por permutación en validación; no identifica causas individuales.")


if section == "Resumen":
    st.title("Facultad de Administración y Negocios")
    st.write("Comprender a quienes ingresan, revisar continuidad y explorar trayectorias de titulación.")
    context_line()
    year = year_choice(population_view)
    cohort = population_view[population_view.cohorte.eq(year)]
    a, b, c = st.columns(3)
    a.metric("Ingresantes de la selección", number(len(cohort)))
    b.metric("Continuidad UA al año siguiente", percent(1-cohort.target.mean()) if cohort.target.notna().sum() >= minimum else "—")
    c.metric("Cobertura de puntaje de admisión", percent(cohort.percentil_admision.notna().mean()) if len(cohort) >= minimum else "—")
    st.caption(f"Cohorte {year}, seguimiento en {year+1}. Primer ingreso observado a UA, planes regulares y carreras documentadas del catálogo FAN.")
    if len(cohort) < minimum:
        st.info(f"Menos de {minimum} casos: no se publican tasas para esta selección.")
    st.subheader("Evolución histórica de la selección")
    table = series(population_view, minimum)
    st.altair_chart(alt.Chart(table).mark_line(point=True, color=RED).encode(
        x=alt.X("Cohorte de ingreso:O", title="Cohorte de ingreso"),
        y=alt.Y("Continuidad UA (%):Q", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Cohorte de ingreso", "Estudiantes", "Seguimientos disponibles"]), width="stretch")
    with st.expander("Tabla histórica y descarga"):
        show_table(table)
        download(table, "Descargar evolución filtrada", "evolucion_fan.csv", Indicador="Continuidad UA al año siguiente")
    st.subheader("Estado general de los modelos de la facultad")
    for key, case in report["cases"].items():
        if "test" in case:
            st.write(f"**{key.title()}: {case['status']}** · captura al 20% {percent(case['test']['capture_20'])} "
                     f"frente a {percent(case['historical_rule']['capture_20'])} de la regla. Evaluación {case['test_year']}.")
    st.caption("Referencias generales de sus propias cohortes de evaluación. Abra Continuidad o Titulación para evaluar el grupo filtrado.")
    st.info("Recorrido sugerido: elija sede y carrera → revise una cohorte → explore Perfiles de ingreso → "
            "compare la capacidad de revisión en Continuidad → consulte Calidad y método antes de usar el resultado.")
    st.caption("Catálogo documentado, pendiente de ratificación institucional. Sin Canvas/Banner ni alertas individuales activadas.")

elif section == "Perfiles de ingreso":
    st.title("Perfiles de ingreso de la facultad")
    context_line()
    s = report["segments"]
    if "winner" not in s:
        st.warning(s["reason"])
        st.stop()
    data = frames["continuidad_dataset.parquet"].merge(frames["segmentacion_assignments.parquet"], on=["mrun", "cohorte"], validate="one_to_one")
    data = filtered(data, selection)
    year = year_choice(data)
    data = data[data.cohorte.eq(year)]
    st.caption(f"Aprendizaje: {', '.join(map(str, s['fit_cohorts']))}; asignación sin reajuste: "
               f"{', '.join(map(str, s['holdout_cohorts']))}. {s['winner']}; silueta {number(s['silhouette'], 3)}.")
    table = profiles(data, minimum)
    omitted = len(data) - (int(table.Estudiantes.sum()) if not table.empty else 0)
    if omitted:
        st.info(f"{number(omitted)} casos en celdas menores a {minimum}: se omite su detalle. Las participaciones "
                "conservan toda la selección como denominador y pueden sumar menos de 100%.")
    if not table.empty:
        st.bar_chart(table.set_index("Perfil")["Estudiantes"], color=RED)
        show_table(table)
        download(table, "Descargar perfiles filtrados", "perfiles_fan.csv", **{"Cohorte de ingreso": year})
    else:
        st.info("No hay perfiles con tamaño suficiente para publicar en esta selección.")
    st.caption("No son niveles de riesgo. Carrera, sede y continuidad no definen los grupos. NEM/ranking usan percentiles "
               "dentro de la cohorte UA; admisión usa su referencia nacional. Ausente no equivale a cero.")
    with st.expander("Comparación general de alternativas de segmentación"):
        candidates = pd.DataFrame(s["candidates"]).rename(columns={"algorithm": "Algoritmo", "k": "Grupos", "silhouette": "Silueta", "smallest_segment": "Mínimo de estudiantes", "largest_share": "Concentración (%)", "admissible": "Admisible"})
        candidates["Concentración (%)"] *= 100
        show_table(candidates)

elif section == "Continuidad":
    case_view("continuidad", "Continuidad al año siguiente")

elif section == "Titulación":
    case_view("titulacion", "Titulación en un horizonte definido")
    st.caption("Matriculados UA en ingreso + 3, sin título UA observado desde el ingreso hasta ese corte. "
               "Horizonte: ingreso + duración nominal + 2 años. Es una población que permanecía matriculada, no todos los ingresantes.")

elif section == "Calidad y método":
    st.title("Alcance, calidad y método")
    context_line()
    st.subheader("Catálogo de carreras de Administración y Negocios")
    catalog = pd.DataFrame([{"Carrera": name, "Evidencia": status, "Estudiantes históricos": report["quality"]["careers"].get(name, 0)} for name, status in cfg["faculty_catalog"].items()])
    show_table(catalog)
    st.caption("Catálogo general. Las carreras históricas se conservan separadas; la documentación no reemplaza la ratificación del alcance por FAN.")
    for i, source in enumerate(cfg["faculty_sources"]):
        st.link_button("Fuente institucional " + str(i+1), source)
    st.caption("El Llano se muestra como San Miguel / El Llano según su ubicación institucional. “SEDE SANTIAGO” histórica sigue separada.")
    st.link_button("Ubicación oficial de El Llano", cfg["campus_label_source"])
    excluded = report["quality"].get("careers_excluded_pending_confirmation", {})
    with st.expander("Carreras del área excluidas mientras se confirma su pertenencia histórica"):
        show_table(pd.DataFrame([{"Carrera": k, "Estudiantes": v} for k, v in excluded.items()]))
    st.subheader("Cobertura de la selección")
    rows = []
    for year, group in population_view.groupby("cohorte"):
        if len(group) >= minimum:
            rows.append({"Cohorte de ingreso": year, "Estudiantes": len(group), "Admisión disponible (%)": 100*group.percentil_admision.notna().mean(),
                         "NEM disponible (%)": 100*group.percentil_nem.notna().mean(), "Ranking disponible (%)": 100*group.percentil_ranking.notna().mean()})
    coverage = pd.DataFrame(rows)
    show_table(coverage)
    download(coverage, "Descargar cobertura filtrada", "cobertura_fan.csv")
    with st.expander("Depuración general y seguimiento de titulación: sin filtros"):
        q = report["quality"]
        show_table(pd.DataFrame([{"Control": label, "Casos": q[key]} for key, label in [
            ("source_rows", "Fuente UA original"), ("missing_identifier", "Sin identificador: excluidos"),
            ("later_entries_excluded", "Ingresos posteriores excluidos"), ("ambiguous_first_entries_excluded", "Primeras entradas ambiguas"),
            ("area_entries_before_catalog", "Área Administración antes de catálogo"), ("pilot_entries", "Catálogo FAN documentado")]]))
        labels = {"not_enrolled_at_landmark": "Sin matrícula en el corte", "already_graduated": "Ya titulados en el corte", "invalid_duration": "Duración fuera del alcance",
                  "missing_historical_years": "Corte aún no observable o historia incompleta", "eligible_at_landmark": "Elegibles en el corte", "right_censored": "Horizonte aún no observable"}
        show_table(pd.DataFrame([{"Control": labels[k], "Casos": v} for k, v in q["graduation"].items()]))
    for item in report["assumptions"]:
        st.write("• " + item)
    st.write("**Uso retrospectivo:** los filtros recalculan resultados del grupo elegido, sin reentrenar. "
             "La comparación de algoritmos y el estado del modelo son generales. Celdas pequeñas y métricas no calculables se señalan explícitamente.")

elif section == "Demostración":
    st.title("Prueba del flujo de revisión")
    st.warning("Seis casos completamente ficticios. No participan en modelos ni métricas.")
    scenarios = {"DEMO-01": "Verificar antecedentes incompletos de ingreso.", "DEMO-02": "Distinguir interrupción y posterior reingreso.",
                 "DEMO-03": "Confirmar si un cambio de código corresponde a cambio real de carrera.", "DEMO-04": "Reconocer un horizonte de titulación todavía abierto.",
                 "DEMO-05": "Comprobar fechas de corte y seguimiento.", "DEMO-06": "Reconocer un subgrupo demasiado pequeño para concluir."}
    chosen = st.selectbox("Caso ficticio", list(scenarios))
    st.write(scenarios[chosen])
    st.checkbox("Revisión de antecedentes realizada", key="review_"+chosen)
    st.text_area("Observación del evaluador (sin datos personales)", key="note_"+chosen)
    st.caption("Solo durante la sesión del navegador. No registra contactos ni intervenciones.")

elif section == "Benchmark":
    st.title("Benchmark como complemento de la facultad")
    st.write("La herramienta institucional permite comparar la UA con universidades similares. Para contextualizar FAN, "
             "seleccione Administración y Comercio en esa aplicación; ese agregado no equivale necesariamente a una facultad.")
    st.link_button("Abrir benchmark institucional", "https://esv4ahpqwzywqytgyavzym.streamlit.app/")
    st.caption("Aplicación independiente: los filtros no se transfieren. Las comparaciones son descriptivas.")
