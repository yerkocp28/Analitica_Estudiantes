"""Vista publicable del extracto institucional de la UA.

QUÉ ES Y QUÉ NO

El piloto FAN corre en local porque sus artefactos tienen resultados por
persona de 40.590 estudiantes. Esta vista es la versión publicable de esos
mismos datos: **solo agregados**, producidos por
`scripts/export_institutional_aggregates.py`, sin identificadores y con un piso
de diez personas por celda aplicado al escribir, no al mostrar.

Sirve para que el equipo de desarrollo vea los resultados sin instalar el
proyecto ni tocar el extracto. **No reemplaza al piloto**: el detalle por
asignatura con filtros dependientes, la auditoría completa y los modelos
institucionales siguen viviendo ahí, en local.

LO QUE NO SE PUEDE AFIRMAR TODAVÍA

La fracción de notas bajo 4 es descriptiva sobre notas numéricas. **No es la
tasa de reprobación**: hay 18.288 resultados en código institucional cuyo
significado sigue pendiente de confirmación, y la definición institucional de
aprobación tampoco está acordada. Mientras P01 y P02 no se respondan, esta
vista describe lo observado y no lo bautiza como indicador oficial.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8880"
CARPETA = "institucional_agregado"

ETIQUETAS = {
    "personas": "Personas", "registros": "Registros", "con_nota": "Con nota numérica",
    "nota_media": "Nota media", "bajo4": "Bajo 4", "cobertura_nota": "Cobertura de nota",
    "career": "Carrera", "career_code": "Código", "campus": "Campus",
    "campus_code": "Sede", "period": "Período", "year": "Año",
    "course_name": "Asignatura", "course_code": "Código asignatura",
}


@st.cache_data(show_spinner=False)
def _cargar(carpeta: str, firma: tuple) -> tuple[dict[str, pd.DataFrame], dict]:
    base = Path(carpeta)
    tablas = {p.stem: pd.read_parquet(p) for p in sorted(base.glob("*.parquet"))}
    meta_path = base / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return tablas, meta


def _firma(base: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in base.glob("*")))


def _tabla(d: pd.DataFrame, orden: str | None = None) -> None:
    v = d.rename(columns=ETIQUETAS)
    if orden and ETIQUETAS.get(orden, orden) in v.columns:
        v = v.sort_values(ETIQUETAS.get(orden, orden), ascending=False)
    st.dataframe(v, hide_index=True, width="stretch", column_config={
        "Nota media": st.column_config.NumberColumn("Nota media", format="%.2f"),
        "Bajo 4": st.column_config.NumberColumn("Bajo 4", format="%.1f%%"),
        "Cobertura de nota": st.column_config.NumberColumn(
            "Cobertura de nota", format="%.1f%%"),
    })


def render_institucional(results_dir: Path) -> None:
    st.title("Datos institucionales UA")
    st.caption("Extracto de matrícula e historia académica · Agregados publicables · "
               "Sin identificadores, mínimo 10 personas por celda")

    base = Path(results_dir) / CARPETA
    if not base.exists() or not any(base.glob("*.parquet")):
        st.info("No hay agregados institucionales publicados. Para generarlos desde "
                "una corrida del piloto ya verificada:")
        st.code("python -m piloto_fan.institucional\n"
                "python scripts/export_institutional_aggregates.py", language="bash")
        return

    tablas, meta = _cargar(str(base), _firma(base))

    a, b, c = st.columns(3)
    a.metric("Personas en el extracto",
             f"{meta.get('personas_en_el_extracto', 0):,}".replace(",", "."),
             "matrícula 2024–2025", delta_color="off")
    b.metric("Registros académicos",
             f"{meta.get('registros_academicos', 0):,}".replace(",", "."),
             "historia por asignatura", delta_color="off")
    periodos = meta.get("periodos") or []
    c.metric("Períodos cubiertos", f"{len(periodos)}",
             "incluye verano y trimestres", delta_color="off")

    st.warning(
        "**La fracción bajo 4 no es la tasa de reprobación.** Es descriptiva sobre "
        "notas numéricas. Hay 18.288 resultados en código institucional cuyo "
        "significado sigue pendiente de confirmación con Registro Académico, y la "
        "definición institucional de aprobación tampoco está acordada.")

    pestanas = st.tabs(["Por período", "Por carrera", "Por campus", "Por asignatura",
                        "Qué falta definir"])

    with pestanas[0]:
        d = tablas.get("por_periodo")
        if d is None or d.empty:
            st.info("Sin corte por período.")
        else:
            st.markdown("**Volumen y resultado observado en cada período.** Los "
                        "períodos de verano y trimestrales tienen volúmenes mucho "
                        "menores que los semestrales; conviene no compararlos de frente.")
            ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("period:N", title=None, sort=None,
                        axis=alt.Axis(labelAngle=-40)),
                y=alt.Y("personas:Q", title="Personas"),
                color=alt.value(AZUL),
                tooltip=[alt.Tooltip("period:N", title="Período"),
                         alt.Tooltip("personas:Q", title="Personas", format=","),
                         alt.Tooltip("nota_media:Q", title="Nota media", format=".2f")])
            st.altair_chart(ch.properties(height=260), width="stretch")
            _tabla(d, "personas")

    with pestanas[1]:
        d = tablas.get("por_carrera")
        if d is None or d.empty:
            st.info("Sin corte por carrera.")
        else:
            st.markdown("**Carreras ordenadas por volumen.** La nota media y la "
                        "fracción bajo 4 describen el período completo del extracto.")
            _tabla(d, "personas")
            serie = tablas.get("por_carrera_periodo")
            if serie is not None and not serie.empty:
                st.markdown("**Evolución por período**, para las carreras con dato "
                            "en más de un período.")
                top = d.nlargest(8, "personas").career.tolist()
                sub = serie[serie.career.isin(top)]
                ch = alt.Chart(sub).mark_line(size=2, point=True).encode(
                    x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=-40)),
                    y=alt.Y("nota_media:Q", title="Nota media",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("career:N", title="Carrera"),
                    tooltip=["career", "period",
                             alt.Tooltip("nota_media:Q", format=".2f"),
                             alt.Tooltip("personas:Q", format=",")])
                st.altair_chart(ch.properties(height=300), width="stretch")

    with pestanas[2]:
        d = tablas.get("por_campus")
        if d is None or d.empty:
            st.info("Sin corte por campus.")
        else:
            st.markdown("**Campus.** La atribución a campus usa el plan del "
                        "estudiante en el período; 1.048 registros tienen un campus "
                        "de asignatura distinto al del plan y su interpretación "
                        "está pendiente.")
            _tabla(d, "personas")

    with pestanas[3]:
        d = tablas.get("por_asignatura")
        if d is None or d.empty:
            st.info("Sin corte por asignatura.")
        else:
            st.markdown(f"**{len(d):,} asignaturas** con al menos diez personas. "
                        "Las que tienen menos no se publican.".replace(",", "."))
            minimo = st.slider("Personas mínimas", 10, 200, 30, step=10,
                               key="inst_min_personas")
            sub = d[d.personas >= minimo]
            st.caption(f"{len(sub):,} asignaturas con {minimo} o más personas"
                       .replace(",", "."))
            if not sub.empty:
                ch = alt.Chart(sub).mark_circle(size=55, opacity=.55).encode(
                    x=alt.X("personas:Q", title="Personas", scale=alt.Scale(type="log")),
                    y=alt.Y("bajo4:Q", title="Fracción de notas bajo 4",
                            axis=alt.Axis(format="%")),
                    color=alt.value(NARANJA),
                    tooltip=[alt.Tooltip("course_name:N", title="Asignatura"),
                             alt.Tooltip("personas:Q", format=","),
                             alt.Tooltip("nota_media:Q", format=".2f"),
                             alt.Tooltip("bajo4:Q", format=".1%")])
                st.altair_chart(ch.properties(height=320), width="stretch")
                st.caption("Eje horizontal en escala logarítmica: el tamaño de las "
                           "asignaturas varía en dos órdenes de magnitud.")
                _tabla(sub.nlargest(40, "personas"), "personas")

    with pestanas[4]:
        st.markdown(
            "Estos agregados describen lo observado en el extracto. **Varias "
            "definiciones siguen pendientes con la universidad**, y hasta "
            "resolverlas hay cosas que no pueden afirmarse:")
        st.dataframe(pd.DataFrame([
            {"Pendiente": "Fecha de corte y filtros de la extracción",
             "Mientras tanto": "La ausencia en el archivo no se lee como deserción "
                               "ni los registros como alumnos activos actuales"},
            {"Pendiente": "Significado de los resultados codificados (RI, RR, RM…)",
             "Mientras tanto": "La fracción bajo 4 es descriptiva, no la tasa "
                               "oficial de reprobación"},
            {"Pendiente": "Unidad de CREDIT_HOURS y equivalencia con SCT",
             "Mientras tanto": "No se ponderan notas por carga"},
            {"Pendiente": "Alcance de carreras y campus aprobado por la facultad",
             "Mientras tanto": "El catálogo es una propuesta documentada, no una "
                               "ratificación"},
        ]), hide_index=True, width="stretch")
        if meta:
            st.caption(f"Corrida de origen: {meta.get('corrida_origen', '—')} · "
                       f"alcance: {meta.get('alcance', '—')} · mínimo por celda: "
                       f"{meta.get('minimo_personas_celda', '—')} personas")

    st.divider()
    st.caption(
        "Esta vista muestra únicamente agregados. El detalle por persona, los "
        "modelos institucionales y la auditoría completa viven en el piloto FAN, "
        "que corre en local y no se publica.")
