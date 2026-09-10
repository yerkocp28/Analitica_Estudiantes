"""Early Warning Cockpit — Student Analytics UA.

    streamlit run src/student_analytics/ui/app.py

Principio de diseno (doc maestro 2.4): la plataforma no debe mostrar solo
una probabilidad. Debe responder a quien revisar, con que urgencia, por que,
y — sobre todo — cuanto alcanza a cubrir la direccion de carrera con la
capacidad que realmente tiene.

Por eso la pieza central no es el AUC sino la CURVA DE CAPACIDAD: si puedo
revisar a 40 estudiantes esta semana, a que fraccion de los que van a
reprobar llego. Es la metrica del benchmark ULagos y la unica que se traduce
en una decision.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = Path(os.environ.get("STUDENT_ANALYTICS_RESULTS_DIR", REPO_ROOT / "data" / "results"))

# Paleta de estado: reservada, nunca reutilizada como color de serie.
# Cada banda va SIEMPRE con icono y etiqueta, nunca solo con color.
BANDS = [
    ("Critico", "#d03b3b", "●"),
    ("Serio", "#ec835a", "◐"),
    ("Atencion", "#fab219", "○"),
    ("Sin alerta", "#0ca30c", "—"),
]
BAND_ORDER = [b[0] for b in BANDS]
BAND_COLORS = [b[1] for b in BANDS]
BAND_ICONS = dict((b[0], b[2]) for b in BANDS)

SEQ_BLUE = "#2a78d6"
INK_MUTED = "#8a8880"

# Paleta categorica en su orden fijo y validado. El ORDEN es el mecanismo
# de separacion para daltonismo, no una eleccion estetica: nunca ciclar ni
# reordenar. Un estudiante cursa 4-6 asignaturas, dentro del rango que
# aprueba las comprobaciones de pares adyacentes para lineas.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# El icono de la pestania es el escudo de la Universidad Autonoma, recortado del
# logo institucional: el logo completo lleva el nombre en dos lineas y a 16 px
# el texto es ilegible. Si el archivo no esta —una copia parcial del repo, por
# ejemplo— se cae al emoji en vez de romper el arranque.
FAVICON = REPO_ROOT / "documentacion" / "assets" / "favicon-ua.png"
st.set_page_config(page_title="Student Analytics UA", layout="wide",
                   page_icon=str(FAVICON) if FAVICON.is_file() else "\U0001f393")


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_results(source: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    m = RESULTS_DIR / f"metrics_{source}.parquet"
    p = RESULTS_DIR / f"predictions_{source}.parquet"
    if not (m.exists() and p.exists()):
        return None
    return pd.read_parquet(m), pd.read_parquet(p)


def assign_bands(df: pd.DataFrame, cuts=(0.05, 0.15, 0.30)) -> pd.DataFrame:
    """Bandas por percentil del score dentro de la semana.

    Los cortes son PROVISORIOS. El doc maestro (65) es explicito en que el
    umbral de alto riesgo no debe fijarse sin datos reales ni sin conocer la
    capacidad de intervencion de cada direccion de carrera. Se exponen aqui
    como parametro, no como decision tomada.
    """
    out = df.copy()
    r = out["risk_score"].rank(pct=True, ascending=False)
    out["band"] = np.select(
        [r <= cuts[0], r <= cuts[1], r <= cuts[2]],
        [BAND_ORDER[0], BAND_ORDER[1], BAND_ORDER[2]],
        default=BAND_ORDER[3],
    )
    return out


def capacity_curve(y: np.ndarray, score: np.ndarray, points: int = 60) -> pd.DataFrame:
    """Fraccion de reprobaciones capturada segun cuantos se prioricen."""
    order = np.argsort(-score)
    hits = np.cumsum(y[order])
    total = max(1.0, float(y.sum()))
    n = len(score)
    idx = np.unique(np.linspace(1, n, points).astype(int))
    return pd.DataFrame({
        "revisados": idx / n,
        "capturados": hits[idx - 1] / total,
        "n_revisados": idx,
    })


# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
st.sidebar.image(str(REPO_ROOT / "documentacion" / "assets" / "logo-ua.png"),
                 width="stretch")
st.sidebar.title("Student Analytics UA")
st.sidebar.caption("Proyecto de analítica estudiantil · Benchmark institucional complementario")
with st.sidebar.expander("Acerca de esta herramienta"):
    st.write("Datos públicos para comparar instituciones y una demostración de alerta temprana. "
             "Sin acceso a Banner/Canvas de la UA. La comparación institucional no predice riesgo individual.")
    st.link_button("Documentación del proyecto", "https://github.com/yerkocp28/Analitica_Estudiantes")

section = st.sidebar.radio("Sección", ["Hallazgos", "Benchmark UA", "Alerta temprana",
                                       "Retención universitaria"])
if section == "Hallazgos":
    from student_analytics.ui.hallazgos import render_hallazgos

    render_hallazgos(RESULTS_DIR)
    st.stop()
if section == "Benchmark UA":
    from student_analytics.ui.benchmark import render_benchmark

    render_benchmark(RESULTS_DIR)
    st.stop()
if section == "Retención universitaria":
    from student_analytics.ui.retention import render_retention

    render_retention(RESULTS_DIR)
    st.stop()

available = [s for s in ("synthetic", "oulad") if load_results(s) is not None]
if not available:
    st.error(
        "No hay resultados todavia. Genera los datos y corre la evaluacion:\n\n"
        "```\npython scripts/generate_synthetic.py\n"
        "python scripts/validate_lead_time.py --source synthetic --out data/results\n```"
    )
    st.stop()

LABELS = {"synthetic": "Sintetico (forma UA)", "oulad": "OULAD (datos reales, sin asistencia)"}
source = st.sidebar.radio("Fuente de datos", available,
                          format_func=lambda s: LABELS.get(s, s))
metrics, preds = load_results(source)

if source == "oulad":
    st.sidebar.info(
        "OULAD es educacion a distancia del Reino Unido y **no tiene asistencia**. "
        "Sirve para validar la metodologia con datos reales, no para estimar cifras UA."
    )
else:
    st.sidebar.warning(
        "Datos **sinteticos** calibrados contra anclas SIES. Ninguna cifra de esta "
        "vista es un hallazgo sobre estudiantes reales."
    )

weeks = sorted(preds["week"].unique())
week = st.sidebar.select_slider("Semana de scoring", options=weeks, value=weeks[len(weeks) // 2])

view = preds[preds["week"] == week].copy()

for col, label in (("campus", "Sede"), ("career_code", "Carrera")):
    if col in view.columns and view[col].notna().any():
        opts = sorted(view[col].dropna().unique())
        sel = st.sidebar.multiselect(label, opts, default=opts)
        view = view[view[col].isin(sel)]

st.sidebar.markdown("---")
capacity = st.sidebar.slider(
    "Capacidad de revision (% de estudiantes)", 1, 50, 20, step=1,
    help="Cuantos estudiantes alcanza a contactar la direccion de carrera esta semana.",
) / 100

cut_critical = st.sidebar.slider("Corte banda critica (%)", 1, 15, 5) / 100

if view.empty:
    st.warning("Ningun estudiante con los filtros seleccionados.")
    st.stop()

view = assign_bands(view, cuts=(cut_critical, 0.15, 0.30))

tab_cockpit, tab_360, tab_modelo = st.tabs(
    ["Cockpit de alerta temprana", "Estudiante 360", "Desempeno del modelo"])


# ----------------------------------------------------------------------
# Cockpit
# ----------------------------------------------------------------------
with tab_cockpit:
    st.caption("Demostración retrospectiva. Cada caso corresponde a estudiante × asignatura × período; "
               "una persona puede aparecer en varios casos. La captura se calcula con resultados ya observados.")
    y = view["y_true"].to_numpy()
    s = view["risk_score"].to_numpy()
    curve = capacity_curve(y, s)
    n_rev = max(1, int(round(len(view) * capacity)))
    captured = float(y[np.argsort(-s)[:n_rev]].sum() / max(1, y.sum()))

    fila = metrics[metrics["week"] == week]
    auc = float(fila["auc"].iloc[0]) if not fila.empty else float("nan")
    total_weeks = int(week + fila["weeks_remaining"].iloc[0]) if not fila.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos a revisar", f"{n_rev:,}", f"{capacity:.0%} de {len(view):,}")
    c2.metric("Reprobaciones capturadas", f"{captured:.0%}",
              f"{captured / capacity:.1f}x sobre azar")
    c3.metric("Semanas para intervenir",
              f"{fila['weeks_remaining'].iloc[0]:.0f}" if not fila.empty else "—",
              f"semana {week} de {total_weeks}" if total_weeks else None)
    c4.metric("AUC del baseline", f"{auc:.3f}" if auc == auc else "—",
              f"prevalencia {y.mean():.0%}", delta_color="off")

    st.markdown("### Cuanto alcanzo a cubrir con la capacidad que tengo")
    st.caption(
        "Cada punto responde: si priorizo ese porcentaje de estudiantes, a que "
        "fraccion de los que efectivamente van a reprobar llego. La linea gris es "
        "el resultado de elegir al azar."
    )

    base = alt.Chart(curve).encode(
        x=alt.X("revisados:Q", title="Estudiantes priorizados",
                axis=alt.Axis(format="%", grid=False)),
        y=alt.Y("capturados:Q", title="Reprobaciones capturadas",
                axis=alt.Axis(format="%")),
    )
    azar = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], size=2, color=INK_MUTED).encode(x="x:Q", y="y:Q")
    linea = base.mark_line(size=2, color=SEQ_BLUE)
    punto = alt.Chart(pd.DataFrame({
        "revisados": [capacity], "capturados": [captured],
    })).mark_point(size=140, filled=True, color=SEQ_BLUE,
                   stroke="#fcfcfb", strokeWidth=2).encode(
        x="revisados:Q", y="capturados:Q",
        tooltip=[alt.Tooltip("revisados:Q", title="Priorizados", format=".0%"),
                 alt.Tooltip("capturados:Q", title="Capturados", format=".0%")])
    hover = base.mark_point(size=90, opacity=0).encode(
        tooltip=[alt.Tooltip("revisados:Q", title="Priorizados", format=".1%"),
                 alt.Tooltip("capturados:Q", title="Capturados", format=".1%"),
                 alt.Tooltip("n_revisados:Q", title="N estudiantes")])
    st.altair_chart((azar + linea + punto + hover).properties(height=320),
                    width="stretch")

    st.markdown("### Distribucion por banda de riesgo")
    conteo = (view.groupby("band").size().reindex(BAND_ORDER).fillna(0)
              .rename("n").reset_index())
    conteo["etiqueta"] = conteo["band"].map(BAND_ICONS) + "  " + conteo["band"]
    barras = alt.Chart(conteo).mark_bar(cornerRadiusEnd=4, size=22).encode(
        y=alt.Y("etiqueta:N", sort=[BAND_ICONS[b] + "  " + b for b in BAND_ORDER],
                title=None),
        x=alt.X("n:Q", title="Estudiantes", axis=alt.Axis(grid=False)),
        color=alt.Color("band:N", scale=alt.Scale(domain=BAND_ORDER, range=BAND_COLORS),
                        legend=None),
        tooltip=[alt.Tooltip("band:N", title="Banda"),
                 alt.Tooltip("n:Q", title="Estudiantes")],
    )
    texto = barras.mark_text(align="left", dx=6, color="#52514e").encode(text="n:Q")
    st.altair_chart((barras + texto).properties(height=150), width="stretch")

    st.markdown(f"### Lista priorizada — top {n_rev:,}")
    cols = [c for c in ("student_id", "course_id", "campus", "career_code", "band",
                        "risk_score", "attendance_cum", "submission_rate",
                        "mean_grade", "n_grades", "activity_mean")
            if c in view.columns]
    tabla = view.nlargest(n_rev, "risk_score")[cols].copy()
    for proportion in ("attendance_cum", "submission_rate"):
        if proportion in tabla:
            tabla[proportion] *= 100
    tabla.insert(0, "", tabla["band"].map(BAND_ICONS))
    st.dataframe(
        tabla, width="stretch", hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Riesgo", format="%.2f", min_value=0.0, max_value=1.0),
            "attendance_cum": st.column_config.NumberColumn("Asistencia", format="%.0f%%"),
            "submission_rate": st.column_config.NumberColumn("Entregas", format="%.0f%%"),
            "mean_grade": st.column_config.NumberColumn("Nota media", format="%.1f"),
            "n_grades": st.column_config.NumberColumn("N notas", format="%d"),
            "activity_mean": st.column_config.NumberColumn("Actividad", format="%.0f"),
            "band": st.column_config.TextColumn("Banda"),
        },
    )
    st.caption(
        "El modelo prioriza; no decide. La revision y cualquier accion son de la "
        "direccion de carrera (Ley 21.719: derecho a intervencion humana y a "
        "explicacion en decisiones automatizadas)."
    )


# ----------------------------------------------------------------------
# Estudiante 360
# ----------------------------------------------------------------------
with tab_360:
    st.markdown("### Ficha del estudiante")
    top_ids = view.nlargest(200, "risk_score")["student_id"].unique().tolist()
    sid = st.selectbox("Estudiante (ordenados por riesgo)", top_ids)

    ficha = preds[(preds["student_id"] == sid)].sort_values(["week", "course_id"])
    actual = ficha[ficha["week"] == week]

    if actual.empty:
        st.info("Sin registros para este estudiante en la semana seleccionada.")
    else:
        banda = assign_bands(view, cuts=(cut_critical, 0.15, 0.30))
        banda = banda[banda["student_id"] == sid]["band"].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Banda", f"{BAND_ICONS[banda]}  {banda}")
        m2.metric("Riesgo maximo", f"{actual['risk_score'].max():.2f}")
        m3.metric("Asignaturas en riesgo",
                  f"{(actual['risk_score'] > actual['risk_score'].median()).sum()}"
                  f" de {len(actual)}")

        st.markdown("#### Evolucion del riesgo durante el periodo")
        st.caption("Lo que importa no es el nivel sino la pendiente: un riesgo alto "
                   "y estable es distinto de uno que se esta acelerando.")
        traj = alt.Chart(ficha).mark_line(size=2, point=alt.OverlayMarkDef(size=60)).encode(
            x=alt.X("week:O", title="Semana de scoring"),
            y=alt.Y("risk_score:Q", title="Riesgo", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("course_id:N", title="Asignatura",
                            scale=alt.Scale(range=CATEGORICAL)),
            tooltip=["course_id", "week", alt.Tooltip("risk_score:Q", format=".2f")],
        ).properties(height=280)
        st.altair_chart(traj, width="stretch")

        st.markdown("#### Senales por asignatura en esta semana")
        drivers = [c for c in ("course_id", "risk_score", "attendance_cum",
                               "submission_rate", "mean_grade", "n_grades",
                               "activity_mean") if c in actual.columns]
        st.dataframe(actual[drivers].sort_values("risk_score", ascending=False),
                     width="stretch", hide_index=True)
        st.caption(
            "Estas son las senales que entraron al modelo, no una explicacion causal. "
            "El doc maestro (91) es explicito: correlacion observada, no causa."
        )


# ----------------------------------------------------------------------
# Desempeno del modelo
# ----------------------------------------------------------------------
with tab_modelo:
    st.markdown("### Poder predictivo vs anticipacion")
    st.caption(
        "El criterio del proyecto no es maximizar AUC: es predecir lo bastante "
        "bien, lo bastante temprano, para que todavia quede tiempo de actuar. "
        "Un modelo mejor en la semana 16 vale menos que uno peor en la semana 5."
    )

    largo = metrics.melt(
        id_vars=["week", "weeks_remaining"],
        value_vars=["lift_05", "lift_10", "lift_20"],
        var_name="capacidad", value_name="captura")
    largo["capacidad"] = largo["capacidad"].map(
        {"lift_05": "Top 5%", "lift_10": "Top 10%", "lift_20": "Top 20%"})

    # Tres series categoricas en los tres primeros slots validados.
    lift_chart = alt.Chart(largo).mark_line(
        size=2, point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("week:O", title="Semana de scoring"),
        y=alt.Y("captura:Q", title="Reprobaciones capturadas",
                axis=alt.Axis(format="%")),
        color=alt.Color("capacidad:N", title="Capacidad de revision",
                        scale=alt.Scale(domain=["Top 5%", "Top 10%", "Top 20%"],
                                        range=["#2a78d6", "#eb6834", "#1baf7a"])),
        tooltip=["capacidad", "week",
                 alt.Tooltip("captura:Q", format=".1%"),
                 alt.Tooltip("weeks_remaining:Q", title="Semanas restantes")],
    ).properties(height=300)
    st.altair_chart(lift_chart, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Discriminacion (AUC)")
        auc_chart = alt.Chart(metrics).mark_line(
            size=2, color=SEQ_BLUE, point=alt.OverlayMarkDef(size=70)).encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("auc:Q", title="AUC", scale=alt.Scale(domain=[0.5, 1.0])),
            tooltip=["week", alt.Tooltip("auc:Q", format=".3f")],
        ).properties(height=240)
        azar_auc = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            strokeDash=[4, 4], size=2, color=INK_MUTED).encode(y="y:Q")
        st.altair_chart(auc_chart + azar_auc, width="stretch")
        st.caption("0,5 = azar. La linea punteada marca ese piso.")

    with c2:
        st.markdown("#### Calibracion (Brier, menor es mejor)")
        brier_chart = alt.Chart(metrics).mark_line(
            size=2, color=SEQ_BLUE, point=alt.OverlayMarkDef(size=70)).encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("brier:Q", title="Brier score"),
            tooltip=["week", alt.Tooltip("brier:Q", format=".3f")],
        ).properties(height=240)
        st.altair_chart(brier_chart, width="stretch")
        st.caption("Mide si las probabilidades son creibles, no solo si ordenan bien.")

    st.markdown("#### Tabla de resultados")
    tabla_m = metrics.copy()
    tabla_m.columns = ["Semana", "Semanas restantes", "N train", "N test",
                       "Prevalencia", "AUC", "Brier", "Top 5%", "Top 10%", "Top 20%"]
    for proportion in ("Prevalencia", "Top 5%", "Top 10%", "Top 20%"):
        tabla_m[proportion] *= 100
    st.dataframe(tabla_m, width="stretch", hide_index=True,
                 column_config={
                     c: st.column_config.NumberColumn(c, format="%.1f%%")
                     for c in ("Prevalencia", "Top 5%", "Top 10%", "Top 20%")})
    st.caption(
        "Validacion temporal: los periodos antiguos entrenan y los recientes "
        "testean. Nunca split aleatorio (doc maestro 27.1)."
    )
