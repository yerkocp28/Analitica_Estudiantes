"""Sección complementaria de comparación universitaria con datos públicos."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from student_analytics.ingestion.retention import summarize_retention

# Paleta categorica validada, en su orden fijo (el mismo del cockpit). El orden
# es el mecanismo de separacion para daltonismo, no una eleccion estetica.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Techo de series en el grafico de lineas. Mas alla de ocho hay que ciclar
# colores, y con 51 universidades la leyenda deja de ser legible.
MAX_SERIES = 8


@st.cache_data
def load_retention(path: str, modified: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def render_retention(results_dir: Path) -> None:
    st.title("Retención universitaria")
    st.caption("Sección complementaria · Matrícula pública Mineduc/SIES · Continuidad al año siguiente")
    path = results_dir / "retention_universities.parquet"
    if not path.exists():
        st.info("Aún no se han preparado los resultados de retención universitaria.")
        st.code("python scripts/build_retention.py", language="bash")
        return
    data = load_retention(str(path), path.stat().st_mtime)
    if data.empty:
        st.info("No hay cohortes universitarias disponibles.")
        return
    a, b = st.columns(2)
    year = a.selectbox("Cohorte de ingreso", sorted(data.cohorte.unique()),
                       index=len(data.cohorte.unique()) - 1)
    area = b.selectbox("Área de conocimiento", ["Todas"] + sorted(data.area_conocimiento.unique()))
    filtered = data[data.cohorte.eq(year)]
    history = data
    if area != "Todas":
        filtered = filtered[filtered.area_conocimiento.eq(area)]
        history = history[history.area_conocimiento.eq(area)]
    labels = {"retencion_misma_carrera": "Misma carrera y universidad",
              "retencion_misma_universidad": "Misma universidad",
              "retencion_sistema": "Cualquier institución del sistema"}
    metric = st.selectbox("Definición de continuidad", list(labels), format_func=labels.get)
    minimum = st.number_input("Mínimo de inscripciones con seguimiento por universidad", min_value=1, value=30)
    summary = summarize_retention(filtered, ["cod_inst", "nomb_inst"])
    national_n = int(summary.n.sum())
    numerator = metric.removeprefix("retencion_")
    national_num = summary[numerator].sum()
    benchmark = national_num / national_n if national_n else float("nan")
    # Cada universidad se compara contra el sistema SIN ella misma. Incluirla
    # en su propia referencia atenua la brecha, y mas cuanto mas grande es la
    # institucion. Es el mismo criterio que usa la seccion de benchmark.
    rest_n = national_n - summary.n
    summary["referencia_resto"] = (national_num - summary[numerator]).div(rest_n.where(rest_n.gt(0)))
    eligible = summary[summary.n.ge(minimum)]
    options = sorted(eligible.nomb_inst.unique())
    selected = st.multiselect("Universidades a comparar", options, default=options,
                              help="Puedes quitar universidades para comparar un grupo específico.")
    shown = eligible[eligible.nomb_inst.isin(selected)].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Universidades seleccionadas", len(shown))
    c2.metric("Inscripciones con seguimiento", f"{int(shown.n.sum()):,}")
    c3.metric("Referencia nacional" + (" del área" if area != "Todas" else ""), f"{benchmark:.1%}")
    st.caption(f"Cohorte {year} → matrícula {year + 1}. Referencia ponderada por inscripciones; "
               f"incluye todas las universidades{' del área' if area != 'Todas' else ''}, sin aplicar "
               "selección ni mínimo de tamaño. La brecha de cada universidad se calcula contra el sistema "
               "excluyéndola a ella misma.")
    if shown.empty:
        st.info("Selecciona universidades o reduce el mínimo de inscripciones.")
        return
    shown["brecha_pp"] = 100 * (shown[metric] - shown["referencia_resto"])
    chart = alt.Chart(shown).mark_bar(color="#2a78d6").encode(
        y=alt.Y("nomb_inst:N", sort="-x", title=None),
        x=alt.X(f"{metric}:Q", title=labels[metric], scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("nomb_inst:N", title="Universidad"), alt.Tooltip("n:Q", title="Inscripciones"),
                 alt.Tooltip(f"{metric}:Q", title="Retención", format=".1%"),
                 alt.Tooltip("brecha_pp:Q", title="Brecha vs resto del sistema (pp)", format="+.1f")])
    rule = alt.Chart(pd.DataFrame({"referencia": [benchmark]})).mark_rule(
        color="#555555", strokeDash=[5, 4]).encode(x="referencia:Q")
    st.altair_chart((chart + rule).properties(height=max(230, len(shown) * 23)), width="stretch")
    st.caption("Línea discontinua: referencia nacional del área seleccionada.")
    display = shown[["nomb_inst", "n", *labels, "brecha_pp", "sin_mrun"]].copy()
    for col in labels:
        display[col] *= 100
    display = display.rename(columns={"nomb_inst": "Universidad", "n": "Inscripciones",
        **{k: f"{v} (%)" for k, v in labels.items()},
        "brecha_pp": "Brecha vs resto (pp)", "sin_mrun": "Sin MRUN"})
    display = display.round(1)
    st.dataframe(display, hide_index=True, width="stretch")
    st.download_button("Descargar comparación (CSV)", display.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"retencion_universidades_{year}.csv", mime="text/csv")
    st.subheader("Comparación entre cohortes")
    trend = summarize_retention(history[history.nomb_inst.isin(selected)], ["cohorte", "cod_inst", "nomb_inst"])
    trend = trend[trend.n.ge(minimum)]
    # Con muchas universidades seleccionadas el gráfico se vuelve ilegible, así
    # que se muestran las de mayor volumen y se dice cuántas quedaron fuera.
    universes = trend.groupby("nomb_inst").n.sum().sort_values(ascending=False)
    visible = universes.head(MAX_SERIES).index
    omitted = len(universes) - len(visible)
    trend_shown = trend[trend.nomb_inst.isin(visible)]
    st.altair_chart(alt.Chart(trend_shown).mark_line(size=2, point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("cohorte:O", title="Cohorte de ingreso"),
        y=alt.Y(f"{metric}:Q", title=labels[metric], axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("nomb_inst:N", title="Universidad", scale=alt.Scale(range=CATEGORICAL)),
        tooltip=["nomb_inst", "cohorte", "n", alt.Tooltip(f"{metric}:Q", format=".1%")]), width="stretch")
    if omitted > 0:
        st.caption(f"Se grafican las {len(visible)} universidades con más inscripciones de la selección; "
                   f"quedan {omitted} fuera del gráfico. La tabla de arriba incluye a todas. "
                   "Reduce la selección para ver otras en la comparación entre cohortes.")
    if trend_shown.cohorte.nunique() < 3:
        st.caption("Con dos cohortes esto describe un cambio entre dos puntos, no una tendencia.")
    st.subheader("Detalle por sede y carrera")
    institution = st.selectbox("Universidad para explorar", sorted(shown.nomb_inst.unique()))
    detail = summarize_retention(filtered[filtered.nomb_inst.eq(institution)], ["nomb_sede", "nomb_carrera"])
    detail = detail.rename(columns={"nomb_sede": "Sede", "nomb_carrera": "Carrera",
                                    "n": "Inscripciones", "sin_mrun": "Sin MRUN"})
    detail["Retención (%)"] = (100 * detail[metric]).round(1)
    st.dataframe(detail[["Sede", "Carrera", "Inscripciones", "Retención (%)", "Sin MRUN"]],
                 hide_index=True, width="stretch")
    with st.expander("Cómo interpretar estos resultados", expanded=True):
        st.markdown(
            "La cohorte incluye inscripciones de pregrado con año de ingreso a la carrera igual al año seleccionado. "
            "Una persona inscrita en varias carreras puede contar más de una vez. Se eliminan duplicados exactos. "
            "Los registros sin MRUN se informan aparte y se excluyen del denominador.\n\n"
            "**Misma carrera:** coincide MRUN, institución y código de carrera. **Misma universidad:** "
            "coincide MRUN e institución, aunque cambie de carrera. **Sistema:** aparece en cualquier "
            "institución al año siguiente, incluidos IP y CFT. No aparecer no prueba abandono definitivo.\n\n"
            "Son cálculos propios sobre bases públicas, sin ajuste por composición del estudiantado; "
            "las diferencias no constituyen un ranking de calidad ni un efecto causal de la universidad. "
            "Los grupos pequeños tienen porcentajes más variables. Cambios de códigos de carrera pueden afectar la continuidad medida.")
