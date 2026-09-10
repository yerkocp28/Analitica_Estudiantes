"""Vista de hallazgos: lo que el proyecto encontró, en una sola pantalla.

Las otras secciones son herramientas de exploración: sirven para preguntar.
Ésta responde. Reúne los resultados que ya están establecidos —el argumento
central, la posición real de la UA, y las trampas de datos que cambiaron un
resultado— para que no haya que reconstruirlos filtrando.

Ninguna cifra está escrita aquí. Todas salen de `modeling/findings.py`, que
las calcula desde los parquet, para que el dashboard y el informe
metodológico no puedan divergir.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from student_analytics.modeling import findings as F

# Los mismos colores del resto de la app, en su orden fijo y validado.
AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
GRIS = "#8a8880"
ROJO = "#d03b3b"


@st.cache_data(show_spinner=False)
def _perfiles(path: str, modificado: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _pp(x: float) -> str:
    return f"{x:+.1f} pp" if np.isfinite(x) else "—"


# ----------------------------------------------------------------------
def _argumento(results: Path) -> None:
    st.subheader("El argumento del proyecto")
    d = F.argumento_central(results)
    if d is None or len(d) < 2:
        st.info("Faltan los resultados del piso predictivo o de OULAD. "
                "Corre `scripts/analyze_mineduc.py` y `scripts/validate_lead_time.py`.")
        return
    previo, conducta = d.iloc[0], d.iloc[1]
    brecha = conducta.auc - previo.auc

    a, b, c = st.columns(3)
    a.metric("Todo el expediente previo al ingreso", f"AUC {previo.auc:.3f}",
             f"n = {previo.n:,}" if previo.n else None, delta_color="off")
    b.metric(f"Conducta en la {conducta.momento.lower()}", f"AUC {conducta.auc:.3f}",
             f"n = {conducta.n:,}" if conducta.n else None, delta_color="off")
    c.metric("Diferencia", f"{brecha:+.3f}", conducta.anticipacion, delta_color="off")

    st.markdown(
        f"Todo lo que el Estado de Chile sabe de un estudiante **antes de que pise la "
        f"universidad** —PAES, NEM, ranking, dependencia del colegio, asistencia mensual "
        f"de cuarto medio, decil de ingreso oficial, becas— alcanza **AUC {previo.auc:.3f}** "
        f"sobre {previo.n:,} estudiantes. Un baseline que solo mira comportamiento dentro "
        f"del curso llega a **{conducta.auc:.3f}** en la {conducta.momento.lower()}, "
        f"**sin asistencia y sin ningún dato previo**.")
    st.markdown(
        "**Por eso el proyecto pide acceso a Banner y Canvas.** El valor no está en lo que "
        "ya se sabe del estudiante al matricularse: está en lo que hace las primeras semanas.")

    comp = pd.DataFrame({
        "fuente": ["Expediente previo al ingreso", f"Conducta, {conducta.momento.lower()}"],
        "auc": [previo.auc, conducta.auc],
    })
    barras = alt.Chart(comp).mark_bar(cornerRadiusEnd=4, size=34).encode(
        y=alt.Y("fuente:N", title=None, sort=None),
        x=alt.X("auc:Q", title="AUC", scale=alt.Scale(domain=[0.5, 0.75]),
                axis=alt.Axis(grid=False)),
        color=alt.Color("fuente:N", scale=alt.Scale(
            domain=comp.fuente.tolist(), range=[GRIS, AZUL]), legend=None),
        tooltip=[alt.Tooltip("fuente:N", title="Información"),
                 alt.Tooltip("auc:Q", format=".3f")])
    texto = barras.mark_text(align="left", dx=6, color="#52514e").encode(
        text=alt.Text("auc:Q", format=".3f"))
    azar = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        strokeDash=[4, 4], color=GRIS).encode(x="x:Q")
    st.altair_chart((azar + barras + texto).properties(height=130), width="stretch")
    st.caption("El eje parte en 0,5 porque ese es el azar. La validación del piso es "
               "temporal: la cohorte antigua entrena y la reciente testea.")


# ----------------------------------------------------------------------
def _retencion(results: Path) -> None:
    st.subheader("Retención: la UA frente al sistema")
    serie = F.serie_retencion(results)
    if serie is None:
        st.info("Faltan los agregados de retención. Corre `scripts/build_retention.py`.")
        return

    ultima = serie.iloc[-1]
    a, b, c = st.columns(3)
    a.metric(f"Continuidad UA · cohorte {int(ultima.cohorte)}",
             f"{ultima.retencion_misma_universidad_ua:.1%}",
             _pp(ultima.brecha_universidad_pp) + " vs sistema")
    b.metric("Resto del sistema universitario",
             f"{ultima.retencion_misma_universidad_sistema:.1%}",
             f"{int(ultima.n_sistema):,} inscripciones", delta_color="off")
    c.metric("Cohortes con serie", f"{len(serie)}",
             f"{int(serie.cohorte.min())}–{int(serie.cohorte.max())}", delta_color="off")

    nivel = st.radio("Nivel de continuidad", ["misma_universidad", "misma_carrera", "sistema"],
                     format_func={"misma_universidad": "En su universidad",
                                  "misma_carrera": "En la misma carrera",
                                  "sistema": "En cualquier institución"}.get,
                     horizontal=True, key="hall_nivel")

    largo = serie.melt(id_vars=["cohorte", "carrera_comparable"],
                       value_vars=[f"retencion_{nivel}_ua", f"retencion_{nivel}_sistema"],
                       var_name="serie", value_name="retencion")
    largo["serie"] = largo.serie.str.endswith("_ua").map(
        {True: "Universidad Autónoma", False: "Resto del sistema"})
    # La continuidad DE CARRERA no se puede calcular donde falta el codigo:
    # se corta la linea en vez de dibujar una caida que no ocurrio.
    if nivel == "misma_carrera":
        largo = largo[largo.carrera_comparable]

    linea = alt.Chart(largo).mark_line(size=2, point=alt.OverlayMarkDef(size=45)).encode(
        x=alt.X("cohorte:O", title="Cohorte de ingreso"),
        y=alt.Y("retencion:Q", title="Continuidad al año siguiente",
                axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(
            domain=["Universidad Autónoma", "Resto del sistema"], range=[AZUL, GRIS])),
        tooltip=["serie", "cohorte", alt.Tooltip("retencion:Q", format=".1%")])
    st.altair_chart(linea.properties(height=300), width="stretch")

    if nivel == "misma_carrera":
        st.warning(f"Las cohortes {', '.join(str(c) for c in F.COHORTES_SIN_CODIGO)} "
                   "**no aparecen** en este nivel: falta el código de carrera en una quinta "
                   "parte de esa matrícula, así que la continuidad de carrera se hundiría "
                   "sin que nadie hubiera desertado. Los otros dos niveles no dependen de "
                   "ese campo y sí cubren la serie completa.")

    st.markdown("**La brecha entre sedes es el hallazgo que cambia el diseño del modelo.**")
    sedes = F.retencion_por_sede(results, area="Administración y Comercio")
    if sedes is None or sedes.empty:
        sedes = F.retencion_por_sede(results)
    if sedes is not None and not sedes.empty:
        rango = 100 * (sedes.retencion_misma_carrera.max() - sedes.retencion_misma_carrera.min())
        barras = alt.Chart(sedes).mark_bar(cornerRadiusEnd=4, size=26).encode(
            y=alt.Y("nomb_sede:N", title=None, sort="-x"),
            x=alt.X("retencion_misma_carrera:Q", title="Continuidad en la misma carrera",
                    axis=alt.Axis(format="%", grid=False), scale=alt.Scale(zero=False)),
            color=alt.value(AZUL),
            tooltip=[alt.Tooltip("nomb_sede:N", title="Sede"),
                     alt.Tooltip("n:Q", title="Inscripciones"),
                     alt.Tooltip("retencion_misma_carrera:Q", format=".1%")])
        etiqueta = barras.mark_text(align="left", dx=6, color="#52514e").encode(
            text=alt.Text("retencion_misma_carrera:Q", format=".1%"))
        st.altair_chart((barras + etiqueta).properties(height=30 + 34 * len(sedes)),
                        width="stretch")
        st.markdown(
            f"Hay **{rango:.0f} puntos porcentuales** entre la sede con mayor y menor "
            f"continuidad, en la cohorte {int(sedes.cohorte.iloc[0])}. Con esos tamaños no es "
            "ruido: la sede debería entrar al modelo como variable, no quedarse como filtro "
            "del tablero.")


# ----------------------------------------------------------------------
def _selectividad(perfiles: pd.DataFrame) -> None:
    st.subheader("Selectividad de admisión: la UA es del montón, y estable")
    d = F.posicion_selectividad(perfiles)
    if d is None:
        st.info("Los perfiles no traen selectividad. Corre `scripts/download_paes.py` "
                "y `scripts/build_benchmark.py`.")
        return

    a, b, c = st.columns(3)
    ult = d.iloc[-1]
    a.metric(f"Percentil nacional de la UA · {int(ult.cohorte)}",
             f"{ult.ua_percentil:.0%}",
             f"supera a {ult.posicion_ua:.0%} de las {ult.universidades} universidades",
             delta_color="off")
    b.metric("Rango en todas las cohortes",
             f"{d.ua_percentil.min():.0%} – {d.ua_percentil.max():.0%}",
             f"{len(d)} cohortes con dato", delta_color="off")
    c.metric("Cobertura PAES de la cohorte UA", f"{ult.ua_cobertura:.0%}",
             "el resto entra por otras vías", delta_color="off")

    st.markdown(
        "La UA **no recibe estudiantes mejores ni peores que el promedio** del sistema, y esa "
        "posición no se mueve. Es un dato importante para el proyecto: sus resultados de "
        "continuidad y titulación no pueden atribuirse al perfil de entrada.")

    izq, der = st.columns(2)
    with izq:
        st.markdown("**Percentil nacional · comparable entre años**")
        largo = d.melt(id_vars=["cohorte", "instrumento"],
                       value_vars=["ua_percentil", "sistema_percentil_mediano"],
                       var_name="serie", value_name="valor")
        largo["serie"] = largo.serie.map({"ua_percentil": "Universidad Autónoma",
                                          "sistema_percentil_mediano": "Mediana del sistema"})
        ch = alt.Chart(largo).mark_line(size=2, point=alt.OverlayMarkDef(size=55)).encode(
            x=alt.X("cohorte:O", title="Cohorte"),
            y=alt.Y("valor:Q", title="Percentil nacional", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0.3, 0.8])),
            color=alt.Color("serie:N", title=None, scale=alt.Scale(
                domain=["Universidad Autónoma", "Mediana del sistema"], range=[AZUL, GRIS])),
            tooltip=["serie", "cohorte", alt.Tooltip("valor:Q", format=".1%")])
        st.altair_chart(ch.properties(height=260), width="stretch")
    with der:
        st.markdown("**Puntaje crudo · NO comparable entre años**")
        largo2 = d.melt(id_vars=["cohorte", "instrumento"],
                        value_vars=["ua_puntaje", "sistema_puntaje_medio"],
                        var_name="serie", value_name="valor")
        largo2["serie"] = largo2.serie.map({"ua_puntaje": "Universidad Autónoma",
                                            "sistema_puntaje_medio": "Media del sistema"})
        ch2 = alt.Chart(largo2).mark_line(size=2, point=alt.OverlayMarkDef(size=55)).encode(
            x=alt.X("cohorte:O", title="Cohorte"),
            y=alt.Y("valor:Q", title="Puntaje", scale=alt.Scale(zero=False)),
            color=alt.Color("serie:N", title=None, scale=alt.Scale(
                domain=["Universidad Autónoma", "Media del sistema"], range=[NARANJA, GRIS])),
            strokeDash=alt.StrokeDash("instrumento:N", title="Prueba"),
            tooltip=["serie", "cohorte", "instrumento",
                     alt.Tooltip("valor:Q", format=".0f")])
        st.altair_chart(ch2.properties(height=260), width="stretch")

    st.error(
        "**El salto del gráfico derecho no es un cambio real.** Las cohortes 2021 y 2022 "
        "rindieron la Prueba de Transición (150–850, media 500) y desde 2023 rige la PAES "
        "(100–1000, media cercana a 610). El puntaje crudo no cruza ese corte; el percentil "
        "de la izquierda sí. Antes de 2021 no hay dato: **la PSU no está publicada como "
        "microdato abierto** y hay que pedirla al DEMRE.")

    st.dataframe(
        d.assign(**{
            "Cohorte": d.cohorte, "Prueba": d.instrumento,
            "Puntaje UA": d.ua_puntaje.round(0), "Media del sistema": d.sistema_puntaje_medio.round(0),
            "Percentil UA": d.ua_percentil, "Posición en el ranking": d.posicion_ua,
            "Universidades": d.universidades, "Cobertura UA": d.ua_cobertura,
        })[["Cohorte", "Prueba", "Puntaje UA", "Media del sistema", "Percentil UA",
            "Posición en el ranking", "Universidades", "Cobertura UA"]],
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(c, format="%.0f%%")
                       for c in ("Percentil UA", "Posición en el ranking", "Cobertura UA")})


# ----------------------------------------------------------------------
def _titulacion(perfiles: pd.DataFrame) -> None:
    st.subheader("Titulación: dos indicadores que apuntan al revés")
    d = F.contraste_titulacion(perfiles)
    if d is None:
        st.info("Los perfiles no traen titulación. Corre `scripts/build_cohorts.py` "
                "y `scripts/build_benchmark.py`.")
        return
    cohorte = d.attrs.get("cohorte")

    st.markdown(
        "Sobre **las mismas universidades**, la titulación transversal correlaciona en "
        "sentido contrario a la titulación por cohorte. No es ruido: miden cosas distintas "
        "y solo una puede leerse como resultado institucional.")

    refs = [c for c in F.REFERENCIAS if c in d.columns]
    orden = d.indicador.tolist()
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GRIS, size=1).encode(x="x:Q")
    # Un grafico por referencia en vez de una faceta: Altair no deja poner
    # capas dentro de una faceta, y la linea del cero es imprescindible para
    # leer el signo de un solo vistazo.
    for ref, columna in zip(refs, st.columns(len(refs))):
        sub = d[["indicador", "tipo", ref]].dropna().rename(columns={ref: "correlacion"})
        with columna:
            st.markdown(f"**{F.REFERENCIAS[ref]}**")
            barras = alt.Chart(sub).mark_bar(cornerRadiusEnd=3, size=20).encode(
                y=alt.Y("indicador:N", title=None, sort=orden,
                        axis=alt.Axis(labelLimit=260)),
                x=alt.X("correlacion:Q", title="Correlación",
                        scale=alt.Scale(domain=[-0.6, 0.6]), axis=alt.Axis(grid=False)),
                color=alt.Color("tipo:N", title="Tipo de indicador", scale=alt.Scale(
                    domain=["Transversal", "Por cohorte"], range=[NARANJA, AZUL])),
                tooltip=["indicador", alt.Tooltip("correlacion:Q", format="+.2f")])
            etiqueta = barras.mark_text(
                align=alt.expr("datum.correlacion < 0 ? 'right' : 'left'"),
                dx=alt.expr("datum.correlacion < 0 ? -5 : 5"), color="#52514e").encode(
                text=alt.Text("correlacion:Q", format="+.2f"))
            st.altair_chart((cero + barras + etiqueta).properties(height=150),
                            width="stretch")

    st.markdown(
        "**La transversal mide velocidad, no logro.** Cuenta cuánto se demoró quien *sí* se "
        "tituló, y eso lo manda la mezcla de carreras: una universidad con mucha ingeniería "
        "aparece abajo porque esos programas se alargan, no porque enseñe peor. Por eso "
        "correlaciona **negativo** con selectividad y con acreditación.")
    st.markdown(
        "**La de cohorte mide cuántos llegan.** Toma a los que entraron y los busca en las "
        "bases de titulados por MRUN. Correlaciona **positivo** con ambas referencias, que es "
        "como debería comportarse un resultado.")
    st.info("**Consecuencia práctica:** la titulación oportuna dejaba a la UA en percentil 94. "
            "Ese número no debe presentarse como logro. El indicador por cohorte es el que "
            "sostiene una afirmación sobre desempeño institucional.")

    tabla = d.copy()
    ren = {"indicador": "Indicador", **{k: v for k, v in F.REFERENCIAS.items() if k in d.columns}}
    st.dataframe(tabla[list(ren)].rename(columns=ren), hide_index=True, width="stretch",
                 column_config={v: st.column_config.NumberColumn(v, format="%+.2f")
                                for k, v in ren.items() if k != "indicador"})
    if cohorte:
        st.caption(f"Correlaciones sobre la cohorte {cohorte} del perfil. La titulación por "
                   "cohorte se refiere a la última cohorte de ingreso con horizonte completo, "
                   "que es anterior: una cohorte reciente todavía no se ha titulado.")


# ----------------------------------------------------------------------
def _trampas(results: Path, perfiles: pd.DataFrame | None) -> None:
    st.subheader("Trampas de datos que cambiaron un resultado")
    st.markdown(
        "Esta es la parte del trabajo que no se ve en ningún gráfico y es la que sostiene "
        "todo lo demás. Cada fila es un error que los datos inducían y que habría producido "
        "una conclusión falsa con toda naturalidad.")
    d = F.trampas_datos(results, perfiles)
    fuentes = st.multiselect("Filtrar por fuente", sorted(d.fuente.unique()),
                             default=[], key="hall_trampas")
    if fuentes:
        d = d[d.fuente.isin(fuentes)]
    st.dataframe(d.rename(columns={"fuente": "Fuente", "trampa": "Qué trae el dato",
                                   "efecto": "Qué pasa si no se corrige",
                                   "estado": "Estado"}),
                 hide_index=True, width="stretch")
    st.caption("«Límite permanente» significa que no hay corrección posible con datos "
               "abiertos: es una restricción de la fuente, no del pipeline.")


def _fuentes(results: Path, perfiles: pd.DataFrame | None) -> None:
    st.subheader("Qué aportó cada fuente")
    d = F.aporte_fuentes(results, perfiles)
    st.dataframe(d.rename(columns={"fuente": "Fuente", "grano": "Grano",
                                   "cobertura": "Cobertura", "aporta": "Qué aporta",
                                   "variables": "Variables en el perfil"}),
                 hide_index=True, width="stretch")
    st.caption("El conteo de variables sale del perfil generado, no de una lista escrita "
               "a mano: si se agrega una fuente, la cifra se mueve sola.")


# ----------------------------------------------------------------------
def render_hallazgos(results_dir: Path) -> None:
    st.title("Hallazgos")
    st.caption("Lo que el proyecto encontró · Todas las cifras se calculan desde los "
               "artefactos, ninguna está escrita a mano")

    perfiles_path = results_dir / "benchmark_profiles.parquet"
    perfiles = (_perfiles(str(perfiles_path), perfiles_path.stat().st_mtime)
                if perfiles_path.exists() else None)

    _argumento(results_dir)
    st.divider()

    tabs = st.tabs(["Retención", "Selectividad de admisión", "Titulación",
                    "Trampas de datos", "Fuentes"])
    with tabs[0]:
        _retencion(results_dir)
    with tabs[1]:
        if perfiles is None:
            st.info("Faltan los perfiles del benchmark.")
        else:
            _selectividad(perfiles)
    with tabs[2]:
        if perfiles is None:
            st.info("Faltan los perfiles del benchmark.")
        else:
            _titulacion(perfiles)
    with tabs[3]:
        _trampas(results_dir, perfiles)
    with tabs[4]:
        _fuentes(results_dir, perfiles)

    st.divider()
    st.caption(
        "Ninguna cifra de esta vista proviene de datos internos de la Universidad Autónoma. "
        "Todo sale de fuentes públicas del Mineduc, el CNED y OULAD. Las comparaciones entre "
        "instituciones son descriptivas y no establecen causalidad.")
