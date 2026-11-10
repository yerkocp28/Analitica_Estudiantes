"""Benchmark institucional centrado en la UA, con pares explicables."""
from pathlib import Path
import hashlib
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import sklearn
from sklearn.metrics import adjusted_rand_score

from student_analytics.modeling.benchmark import compare_outcomes, fit_peers, neighbors, target_code
from student_analytics.modeling.benchmark_clusters import (
    BLOCKS, cluster_profile, cluster_variables, group_order, partition_agreement, percentile, positions, ua_summary,
)
from student_analytics.modeling.benchmark_reporting import sensitivity, continuity_breakdown, executive_html

UA_RED = "#E2211C"
BLUE = "#2a78d6"
GREY = "#b0b0b0"
INK = "#6B6560"
# Colores de los grupos: validados entre todos los pares, aunque con
# deuteranopia algunos quedan en la banda 6-8, así que cada grupo lleva
# además su propia forma. No hay rojo: el rojo es la UA, que va como estrella.
CLUSTER_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#52514e"]
CLUSTER_SHAPES = ["circle", "square", "triangle-up", "diamond", "triangle-down", "cross"]
STAR = ("M0,-1L0.225,-0.309L0.951,-0.309L0.363,0.118L0.588,0.809L0,0.382L-0.588,0.809"
        "L-0.363,0.118L-0.951,-0.309L-0.225,-0.309Z")
# Van de cien a decenas de miles: en escala lineal todo queda apretado a la izquierda.
LOG_AXES = {"cohorte_total", "estudiantes_total", "titulados_total", "tamano_por_sede"}
# Lo que se puede sumar al perfil de los grupos sin que forme parte de ellos.
EXTRA = ["retencion", "arancel_mediano", "acreditacion_anios", "docentes_por_100_alumnos", "share_doctorado",
         "paes_percentil_promedio", "titulacion_oportuna", "titulacion_cohorte_universidad"]
LABELS = {"misma_carrera": "Misma carrera y universidad", "misma_universidad": "Misma universidad",
          "sistema": "Cualquier institución del sistema"}

# Ejes disponibles en el mapa de posicionamiento: etiqueta y formato de eje.
# `retencion` se calcula con el filtro vigente; el resto viene del perfil.
AXES = {
    "retencion": ("Continuidad al año siguiente", "%"),
    "cohorte_total": ("Inscripciones de ingreso", ","),
    "sedes": ("Número de sedes", ","),
    "tamano_por_sede": ("Inscripciones por sede", ","),
    "arancel_mediano": ("Arancel anual mediano (CLP)", "~s"),
    "matricula_mediana": ("Matrícula anual mediana (CLP)", "~s"),
    "acreditacion_anios": ("Años de acreditación institucional", ","),
    "carreras_acreditadas": ("Inscripciones en carreras acreditadas", "%"),
    "duracion_media": ("Duración nominal media (semestres)", ".1f"),
    "ingreso_no_regular": ("Ingreso por vías no regulares", "%"),
    "ingreso_pace": ("Ingreso vía PACE", "%"),
    "concentracion_areas": ("Concentración de áreas (HHI)", ".2f"),
    "regiones_presencia": ("Regiones con presencia", ","),
    "distancia": ("Distancia al perfil de la UA", ".2f"),
    # Recursos institucionales del CNED. Aparecen solo si la base esta
    # descargada: `disponibles` filtra por columnas con dato.
    "docentes_por_100_alumnos": ("Docentes JCE estimados por 100 estudiantes totales", ".1f"),
    "share_doctorado": ("Docentes con doctorado", "%"),
    "share_magister": ("Docentes con magíster o doctorado", "%"),
    "share_jornada_completa": ("Docentes con jornada completa", "%"),
    "share_docentes_mujeres": ("Docentes mujeres", "%"),
    "m2_construido_por_alumno": ("M² construidos por estudiante", ".1f"),
    "pc_por_100_alumnos": ("PC para estudiantes por cada 100", ".1f"),
    "ejemplares_por_alumno": ("Ejemplares de biblioteca por estudiante", ".1f"),
    "anio_creacion": ("Año de creación", "d"),
    "acreditacion_cned": ("Acreditación CNED · foto del catálogo 2025", ","),
    "estudiantes_total": ("Estudiantes identificados · toda la institución", ","),
    # Selectividad de admisión, cruzando PAES por MRUN. `paes_cobertura` no es
    # un indicador de calidad sino la advertencia que acompaña a los demás:
    # donde la cobertura es baja, el promedio describe a una minoría de la
    # cohorte y no al perfil de ingreso de la institución.
    "paes_promedio": ("Puntaje PAES promedio de la cohorte", ".0f"),
    "paes_p25": ("PAES percentil 25 de la cohorte", ".0f"),
    "paes_p75": ("PAES percentil 75 de la cohorte", ".0f"),
    "paes_rango_intercuartil": ("Dispersión PAES de la cohorte (p75 − p25)", ".0f"),
    "nem_promedio": ("Puntaje NEM promedio", ".0f"),
    "ranking_promedio": ("Puntaje ranking promedio", ".0f"),
    "paes_cobertura": ("Cohorte con puntaje PAES · cobertura", "%"),
    # El puntaje crudo NO se puede comparar entre cohortes: 2021-2022 rindieron
    # la PDT (150-850) y desde 2023 es la PAES (100-1000). El percentil nacional
    # sí cruza ese corte, y es el eje correcto para mirar la serie.
    "paes_percentil_promedio": ("Percentil nacional promedio de la cohorte · comparable entre años", "%"),
    # Titulacion de la PROMOCION QUE EGRESA, no de la cohorte de ingreso. Las
    # etiquetas lo dicen para que nadie las lea como tasa de titulacion de
    # quienes entraron ese anio: de esos todavia no se titula nadie.
    "titulados_total": ("Titulados de pregrado en el año", ","),
    "titulados_por_100_estudiantes": ("Titulados por 100 estudiantes · flujo de salida", ".1f"),
    "titulacion_duracion_mediana": ("Años hasta titularse · promoción que egresa", ".1f"),
    "titulacion_sobreduracion": ("Sobreduración mediana · promoción que egresa", ".1f"),
    "titulacion_oportuna": ("Se tituló dentro de la duración nominal", "%"),
    "titulacion_oportuna_holgada": ("Se tituló dentro de la nominal más un año", "%"),
    "titulacion_cobertura": ("Titulados con duración calculable · cobertura", "%"),
    # Titulacion POR COHORTE DE INGRESO: seguimiento longitudinal por MRUN.
    # Esta si es una tasa —de los que entraron, cuantos se titularon— pero
    # de una cohorte ANTIGUA, la ultima cuyo plazo alcanzo a cumplirse. Los
    # tres niveles estan anidados: carrera <= universidad <= sistema, y la
    # brecha entre ellos es cambio de carrera y traslado, no desercion.
    "titulacion_cohorte_carrera": ("Se tituló de la carrera que empezó · cohorte", "%"),
    "titulacion_cohorte_universidad": ("Se tituló en su universidad de ingreso · cohorte", "%"),
    "titulacion_cohorte_sistema": ("Se tituló en alguna universidad · cohorte", "%"),
    "titulacion_cohorte_anios": ("Años medianos hasta el título · cohorte", ".1f"),
    "titulacion_cohorte_n": ("Ingresantes con horizonte observable · cohorte", ","),
}


@st.cache_data(show_spinner=False)
def load_benchmark(folder: str, signatures: tuple) -> tuple:
    root = Path(folder)
    outcomes = pd.read_parquet(root / "retention_universities.parquet")
    profiles = pd.read_parquet(root / "benchmark_profiles.parquet")
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for filename, field in [("retention_universities.parquet", "retencion_sha256"),
                            ("benchmark_profiles.parquet", "perfiles_sha256")]:
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError("Los perfiles y resultados se generaron en momentos distintos. Regenera el benchmark.")
    return outcomes, profiles, manifest


@st.cache_data(show_spinner=False)
def cached_model(profiles: pd.DataFrame, config: dict):
    return fit_peers(profiles, config)


def percent(value: float) -> str:
    return f"{value:.1%}" if np.isfinite(value) else "Sin datos"


def short_name(name: str) -> str:
    """Nombre compacto para rotular puntos sin tapar el grafico."""
    text = str(name).title()
    for viejo, nuevo in [("Universidad ", "U. "), ("Pontificia U. ", "P. U. "),
                         ("U. De ", "U. "), (" De Chile", ""), (" De ", " de "),
                         (" Del ", " del "), (" La ", " la "), (" Y ", " y ")]:
        text = text.replace(viejo, nuevo)
    return text if len(text) <= 26 else text[:25] + "…"


def positioning_frame(model, nearest, data, year, metric, area, code, peer_ids):
    """Une perfil, distancia y resultado para el mapa de posicionamiento."""
    from student_analytics.ingestion.retention import summarize_retention

    selected = data.loc[data.cohorte.eq(year)]
    if area is not None:
        selected = selected.loc[selected.area_conocimiento.eq(area)]
    resultado = summarize_retention(selected, ["cod_inst"])
    resultado["retencion"] = resultado[metric].div(resultado.n.where(resultado.n.gt(0)))

    frame = model.universities.merge(
        resultado[["cod_inst", "n", "retencion"]], on="cod_inst", how="left")
    frame = frame.merge(nearest[["cod_inst", "distancia"]], on="cod_inst", how="left")
    # La UA no aparece en `nearest` (se excluye a sí misma): su distancia es 0.
    frame.loc[frame.cod_inst.eq(code), "distancia"] = 0.0
    frame["rol"] = np.where(frame.cod_inst.eq(code), "Universidad Autónoma",
                            np.where(frame.cod_inst.isin(peer_ids), "Par seleccionado", "Otras universidades"))
    return frame


def scatter(frame: pd.DataFrame, x: str, y: str, quadrants: bool = True):
    """Dispersión con la UA destacada y medianas como líneas de referencia.

    Mismo código de color que el mapa PCA: rojo la UA, azul los pares, gris el
    resto. Sólo la UA y los pares llevan etiqueta directa; poner el nombre de
    las 51 universidades haría el gráfico ilegible.
    """
    usable = frame.loc[frame[x].notna() & frame[y].notna()].copy()
    if usable.empty:
        return None, 0
    usable["etiqueta"] = usable.nomb_inst.map(short_name)
    x_title, x_fmt = AXES[x]
    y_title, y_fmt = AXES[y]
    orden = ["Universidad Autónoma", "Par seleccionado", "Otras universidades"]

    # El eje x se ensancha por la derecha: las etiquetas van a la derecha del
    # punto y sin este margen la ultima se sale de la lamina.
    lo, hi = float(usable[x].min()), float(usable[x].max())
    span = (hi - lo) or (abs(hi) or 1)
    dominio_x = [lo - .04 * span, hi + .22 * span]

    base = alt.Chart(usable).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=x_fmt),
                scale=alt.Scale(domain=dominio_x, zero=False, nice=False, clamp=True)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_fmt),
                scale=alt.Scale(zero=False, nice=True)))
    # Una sola leyenda. La forma repite la informacion del color a proposito
    # -- asi la distincion no depende solo del tono -- pero su leyenda se
    # oculta para no duplicar la del color. El tamanio jerarquiza: la UA y sus
    # pares por delante de las otras 45.
    puntos = base.mark_point(filled=True, stroke="#ffffff", strokeWidth=1.2).encode(
        color=alt.Color("rol:N", title=None, sort=orden,
                        scale=alt.Scale(domain=orden, range=[UA_RED, BLUE, GREY]),
                        legend=alt.Legend(orient="top", symbolSize=110)),
        shape=alt.Shape("rol:N", sort=orden, legend=None,
                        scale=alt.Scale(domain=orden, range=["circle", "square", "triangle"])),
        size=alt.Size("rol:N", sort=orden, legend=None,
                      scale=alt.Scale(domain=orden, range=[240, 170, 70])),
        opacity=alt.Opacity("rol:N", sort=orden, legend=None,
                            scale=alt.Scale(domain=orden, range=[1, .95, .5])),
        tooltip=[alt.Tooltip("nomb_inst:N", title="Universidad"),
                 alt.Tooltip("rol:N", title="Rol"),
                 alt.Tooltip("grupo:N", title="Grupo"),
                 alt.Tooltip(f"{x}:Q", title=x_title, format=x_fmt),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=y_fmt),
                 alt.Tooltip("cohorte_total:Q", title="Inscripciones", format=",")])
    capas = []
    if quadrants:
        medianas = pd.DataFrame({"vx": [usable[x].median()], "vy": [usable[y].median()]})
        capas += [
            alt.Chart(medianas).mark_rule(strokeDash=[4, 4], size=1, color=INK).encode(x="vx:Q"),
            alt.Chart(medianas).mark_rule(strokeDash=[4, 4], size=1, color=INK).encode(y="vy:Q"),
        ]
    capas.append(puntos)
    # La UA se vuelve a dibujar encima: con aranceles y retenciones parecidos
    # su punto quedaba tapado por el de un par, y es el sujeto del grafico.
    ua = usable.loc[usable.rol.eq("Universidad Autónoma")]
    if not ua.empty:
        capas.append(alt.Chart(ua).mark_point(
            filled=True, shape="circle", size=240, color=UA_RED,
            stroke="#ffffff", strokeWidth=2).encode(x=f"{x}:Q", y=f"{y}:Q"))
    etiquetados = usable.loc[usable.rol.ne("Otras universidades")].copy()
    if not etiquetados.empty:
        # Nombres cortos y alternancia arriba/abajo: con seis instituciones en
        # un rango estrecho, los rotulos completos se pisaban entre si.
        etiquetados = etiquetados.sort_values(y).reset_index(drop=True)
        # `dy` es propiedad de la marca, no canal de codificacion, asi que la
        # alternancia se hace con dos capas en vez de una columna de desvio.
        for resto, desvio in ((0, -11), (1, 13)):
            grupo = etiquetados.loc[etiquetados.index % 2 == resto]
            if grupo.empty:
                continue
            capas.append(alt.Chart(grupo).mark_text(
                align="left", dx=11, dy=desvio, fontSize=10, fontWeight=600,
                color=INK).encode(x=f"{x}:Q", y=f"{y}:Q", text="etiqueta:N"))
    return alt.layer(*capas).properties(height=430), len(usable)


def gap(a: float, b: float) -> str | None:
    return f"UA {(a - b) * 100:+.1f} pp" if np.isfinite(a) and np.isfinite(b) else None


def as_text(value: float, fmt: str) -> str:
    """El valor con el mismo formato que su eje, para rotular celdas y tooltips."""
    if not np.isfinite(value):
        return "—"
    if fmt == "%":
        return f"{value:.0%}"
    if fmt == "~s":
        return f"{value / 1e6:.1f} M" if abs(value) >= 1e6 else f"{value:,.0f}"
    if fmt in (".1f", ".2f"):
        return f"{value:{fmt}}"
    return f"{value:,.0f}"


def group_encoding(frame: pd.DataFrame, shown: list[str], ua_shown: str, focus: bool):
    """Color y forma por grupo; con `focus`, solo el grupo de la UA frente al resto."""
    if focus:
        rest = "Resto de los grupos"
        frame = frame.assign(clave=np.where(frame.grupo_vista.eq(ua_shown), ua_shown, rest))
        domain = [ua_shown, rest]
        colors = [CLUSTER_COLORS[shown.index(ua_shown)], GREY]
        shapes = [CLUSTER_SHAPES[shown.index(ua_shown)], "circle"]
    else:
        frame = frame.assign(clave=frame.grupo_vista)
        domain, colors, shapes = shown, CLUSTER_COLORS[:len(shown)], CLUSTER_SHAPES[:len(shown)]
    # Color y forma con el mismo título se funden en una sola leyenda.
    color = alt.Color("clave:N", title="Grupo", sort=domain, scale=alt.Scale(domain=domain, range=colors),
                      legend=alt.Legend(orient="top", symbolSize=110))
    shape = alt.Shape("clave:N", title="Grupo", sort=domain, scale=alt.Scale(domain=domain, range=shapes))
    return frame, color, shape


def ua_star(data: pd.DataFrame, x, y, tooltip, label: bool = True):
    """La UA encima de todo, como estrella roja con nombre: es el sujeto de cada gráfico."""
    layers = [alt.Chart(data).mark_point(shape=STAR, filled=True, size=460, color=UA_RED, opacity=1,
                                         stroke="#1a1a1a", strokeWidth=1).encode(x=x, y=y, tooltip=tooltip)]
    if label:
        layers.append(alt.Chart(data).mark_text(align="left", dx=14, fontSize=11, fontWeight=700, color="#1a1a1a")
                      .encode(x=x, y=y, text=alt.value("Universidad Autónoma")))
    return layers


def cluster_scatter(frame, x, y, labels, shown, ua_shown, code, focus=False, names=True):
    """Universidades por dos variables, coloreadas por grupo, con la UA destacada."""
    usable = frame.loc[frame[x].notna() & frame[y].notna()].copy()
    if usable.empty:
        return None, 0
    usable, color, shape = group_encoding(usable, shown, ua_shown, focus)
    usable["etiqueta"] = usable.nomb_inst.map(short_name)

    def channel(kind, var):
        title, fmt = labels[var]
        log = var in LOG_AXES and usable[var].gt(0).all()
        spec = dict(title=title + (" · escala logarítmica" if log else ""), axis=alt.Axis(format=fmt),
                    scale=alt.Scale(type="log") if log else alt.Scale(zero=False, nice=True))
        return (alt.X if kind == "x" else alt.Y)(f"{var}:Q", **spec)

    ex, ey = channel("x", x), channel("y", y)
    tooltip = [alt.Tooltip("nomb_inst:N", title="Universidad"), alt.Tooltip("grupo_vista:N", title="Grupo"),
               alt.Tooltip(f"{x}:Q", title=labels[x][0], format=labels[x][1]),
               alt.Tooltip(f"{y}:Q", title=labels[y][0], format=labels[y][1]),
               alt.Tooltip("cohorte_total:Q", title="Inscripciones", format=",")]
    medians = pd.DataFrame({"vx": [usable[x].median()], "vy": [usable[y].median()]})
    layers = [alt.Chart(medians).mark_rule(strokeDash=[4, 4], size=1, color=INK).encode(x=alt.X("vx:Q")),
              alt.Chart(medians).mark_rule(strokeDash=[4, 4], size=1, color=INK).encode(y=alt.Y("vy:Q")),
              alt.Chart(usable).mark_point(filled=True, size=110, opacity=.85, stroke="#ffffff", strokeWidth=1)
              .encode(x=ex, y=ey, color=color, shape=shape, tooltip=tooltip)]
    if names:
        mates = usable.loc[usable.grupo_vista.eq(ua_shown) & usable.cod_inst.ne(code)]
        mates = mates.sort_values(y).reset_index(drop=True)
        for rest, offset in ((0, -10), (1, 12)):
            part = mates.loc[mates.index % 2 == rest]
            if not part.empty:
                layers.append(alt.Chart(part).mark_text(align="left", dx=9, dy=offset, fontSize=10, color=INK)
                              .encode(x=ex, y=ey, text="etiqueta:N"))
    ua = usable.loc[usable.cod_inst.eq(code)]
    if not ua.empty:
        layers += ua_star(ua, ex, ey, tooltip)
    return alt.layer(*layers).properties(height=460), len(usable)


def percentile_strip(pos, rows, shown, ua_shown, focus):
    """Una fila por variable: cada universidad en su percentil y la UA como estrella."""
    pos = pos.assign(texto=[as_text(v, f) for v, f in zip(pos.valor, pos.formato)])
    pos, color, shape = group_encoding(pos, shown, ua_shown, focus)
    x = alt.X("percentil:Q", scale=alt.Scale(domain=[0, 100]),
              title="Percentil entre las universidades · 0 la menor, 100 la mayor")
    y = alt.Y("etiqueta:N", sort=rows, title=None, axis=alt.Axis(labelLimit=300, grid=True, gridColor="#eeeeee"))
    tooltip = [alt.Tooltip("nomb_inst:N", title="Universidad"), alt.Tooltip("grupo_vista:N", title="Grupo"),
               alt.Tooltip("etiqueta:N", title="Variable"), alt.Tooltip("texto:N", title="Valor"),
               alt.Tooltip("percentil:Q", title="Percentil", format=".0f")]
    middle = alt.Chart(pd.DataFrame({"m": [50]})).mark_rule(strokeDash=[4, 4], color=INK).encode(x="m:Q")
    dots = alt.Chart(pos.loc[~pos.es_ua]).mark_point(filled=True, size=55, opacity=.7, stroke="#ffffff",
                                                     strokeWidth=.6).encode(x=x, y=y, color=color, shape=shape,
                                                                            tooltip=tooltip)
    layers = [middle, dots] + ua_star(pos.loc[pos.es_ua], x, y, tooltip, label=False)
    return alt.layer(*layers).properties(height=30 * len(rows) + 40)


def profile_heatmap(profile, rows, columns):
    """Mediana de cada grupo y valor de la UA; el color dice cuán lejos están del promedio de todas."""
    p = profile.assign(texto=[as_text(v, f) for v, f in zip(profile.valor, profile.formato)],
                       bloque_nombre=profile.bloque.map(BLOCKS))
    base = alt.Chart(p).encode(
        x=alt.X("columna:N", sort=columns, title=None,
              axis=alt.Axis(orient="top", labelAngle=0, labelLimit=170, labelOverlap=False)),
        y=alt.Y("etiqueta:N", sort=rows, title=None, axis=alt.Axis(labelLimit=300)))
    cells = base.mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("z:Q", title="Desviaciones respecto del promedio",
                        scale=alt.Scale(scheme="blueorange", domain=[-2, 2], clamp=True),
                        legend=alt.Legend(orient="bottom", gradientLength=260, titleLimit=320)),
        tooltip=[alt.Tooltip("columna:N", title="Columna"), alt.Tooltip("bloque_nombre:N", title="Bloque"),
                 alt.Tooltip("etiqueta:N", title="Variable"), alt.Tooltip("texto:N", title="Mediana o valor"),
                 alt.Tooltip("z:Q", title="Desviaciones", format="+.2f"),
                 alt.Tooltip("universidades:Q", title="Universidades con dato")])
    text = base.mark_text(fontSize=11).encode(
        text="texto:N", color=alt.condition("abs(datum.z) > 1.3", alt.value("#ffffff"), alt.value("#1a1a1a")))
    return (cells + text).properties(height=28 * len(rows) + 30)


def render_clusters(model, frame, code, year, area_name):
    st.subheader("¿En qué grupo queda la UA y qué distingue a cada grupo?")
    if not model.k:
        st.warning("En esta cohorte ninguna partición cumple los requisitos de tamaño y concentración, así que no "
                   "hay grupos que mostrar. Los pares por cercanía y la selección manual siguen disponibles.")
        return
    groups = group_order(frame.grupo)
    ua_group = frame.loc[frame.cod_inst.eq(code), "grupo"].iloc[0]
    names = {g: f"{g} (UA)" if g == ua_group else g for g in groups}
    shown, ua_shown = [names[g] for g in groups], names[ua_group]
    frame = frame.assign(grupo_vista=frame.grupo.map(names))
    sizes = frame.grupo.value_counts()

    a, b, c = st.columns(3)
    a.metric("Grupo de la UA", ua_group)
    a.caption(f"Con {sizes[ua_group] - 1} universidades más.")
    b.metric("Grupos en la cohorte", model.k)
    b.caption(" · ".join(f"{g}: {sizes[g]}" for g in groups) + f" · {len(frame)} universidades en total.")
    c.metric("Modelo que forma los grupos", model.algorithm)
    c.caption(f"Silueta {model.silhouette:.3f}. El detalle está en «Modelos de agrupamiento».")
    mates = frame.loc[frame.grupo.eq(ua_group) & frame.cod_inst.ne(code)].sort_values("distancia")
    st.markdown("**Comparten grupo con la UA**, de la más cercana a la más lejana en perfil: "
                + ", ".join(mates.nomb_inst.map(short_name)) + ".")
    st.caption(f"Cohorte de ingreso {year}. Los grupos se forman con el perfil de ingreso de la cohorte completa: "
               "tamaño y sedes, selectividad (si hay PAES), áreas, regiones, jornada y modalidad. La continuidad, "
               "los costos y los recursos no participan. Los números de grupo son etiquetas, no un ranking.")

    variables = cluster_variables(model, code)
    left, right = st.columns([3, 2])
    extra = left.toggle("Sumar variables que no forman los grupos: continuidad, arancel, acreditación, "
                        "docentes, titulación", key="bench_cluster_extra")
    focus = right.radio("Colores", ["Todos los grupos", "Grupo de la UA frente al resto"], horizontal=True,
                        key="bench_cluster_focus") != "Todos los grupos"
    if extra:
        variables += [{"variable": k, "etiqueta": AXES[k][0], "formato": AXES[k][1], "bloque": "descriptiva"}
                      for k in EXTRA if k in frame and frame[k].notna().any()]
        st.caption(f"La continuidad usa el indicador y el área elegidos arriba ({area_name}); el resto describe "
                   "a toda la institución.")
    rows = [v["etiqueta"] for v in variables]

    st.markdown("#### Los grupos en dos variables")
    labels = {**AXES, **{v["variable"]: (v["etiqueta"], v["formato"]) for v in variables},
              "mapa_x": ("Componente 1 del perfil (PCA)", ".2f"), "mapa_y": ("Componente 2 del perfil (PCA)", ".2f")}
    options = list(dict.fromkeys([v["variable"] for v in variables] + ["mapa_x", "mapa_y"]
                                 + [k for k in AXES if k in frame and frame[k].notna().any()]))
    default_x = "paes_promedio" if "paes_promedio" in options else "cohorte_total"
    ex, ey, en = st.columns([2, 2, 1])
    x_var = ex.selectbox("Eje horizontal", options, index=options.index(default_x),
                         format_func=lambda k: labels[k][0], key="bench_cluster_x")
    y_var = ey.selectbox("Eje vertical", options, index=options.index("cohorte_total"),
                         format_func=lambda k: labels[k][0], key="bench_cluster_y")
    show_names = en.checkbox("Nombres del grupo UA", value=True, key="bench_cluster_names")
    chart, count = cluster_scatter(frame, x_var, y_var, labels, shown, ua_shown, code, focus, show_names)
    if chart is None:
        st.info("No hay universidades con dato en ambas variables.")
    else:
        st.altair_chart(chart, width="stretch")
        missing = len(frame) - count
        st.caption("Cada punto es una universidad; color y forma indican su grupo. Las líneas punteadas son las "
                   "medianas de todas las universidades con dato."
                   + (f" {missing} universidades quedan fuera por dato ausente." if missing else "")
                   + " Los grupos se forman con todas las variables a la vez: dos ejes muestran solo una parte "
                   "de la separación. Los componentes PCA resumen el perfil completo en dos dimensiones.")
        if frame.loc[frame.cod_inst.eq(code), [x_var, y_var]].isna().any(axis=None):
            st.warning("La UA no tiene dato en alguno de los ejes; su estrella no aparece.")

    st.markdown("#### ¿Dónde está la UA en cada variable?")
    pos = positions(model, variables, code, frame).assign(grupo_vista=lambda d: d.grupo.map(names))
    st.altair_chart(percentile_strip(pos, rows, shown, ua_shown, focus), width="stretch")
    st.caption("Cada fila es una variable y cada punto una universidad, ubicada en su percentil: qué porcentaje "
               "de las demás queda por debajo. La estrella roja es la UA; la línea punteada, la mitad. Es posición, "
               "no puntuación: en tamaño, arancel o una área, estar a la derecha no es mejor ni peor.")

    st.markdown("#### ¿Qué distingue a cada grupo?")
    profile = cluster_profile(model, variables, code, frame)
    profile["columna"] = profile.columna.map(lambda c: names.get(c, c))
    profile["columna"] = profile.columna.replace("Universidad Autónoma", "UA")
    columns = ["UA", ua_shown] + [s for s in shown if s != ua_shown]
    st.altair_chart(profile_heatmap(profile, rows, columns), width="stretch")
    st.caption("Mediana de cada grupo y valor de la UA. Naranjo: sobre el promedio de todas las universidades; "
               "azul: bajo el promedio; gris: cerca. La primera columna permite leer a la UA contra su grupo y "
               "contra los demás.")

    st.markdown("#### La UA frente a su grupo y frente a todas")
    st.dataframe(ua_summary(model, variables, code, frame).round(1), hide_index=True, width="stretch")
    export = frame[["cohorte", "nomb_inst", "grupo", "distancia"]].copy()
    export["es_ua"] = frame.cod_inst.eq(code)
    for v in variables:
        export[v["etiqueta"] + (" (%)" if v["formato"] == "%" else "")] = (
            frame[v["variable"]] * (100 if v["formato"] == "%" else 1))
    export = export.rename(columns={"cohorte": "Cohorte", "nomb_inst": "Universidad", "grupo": "Grupo",
                                    "distancia": "Distancia al perfil UA", "es_ua": "Es la UA"})
    export["orden"] = export.Grupo.map({g: i for i, g in enumerate(groups)})
    export = export.sort_values(["orden", "Distancia al perfil UA"]).drop(columns="orden").round(3)
    with st.expander(f"Universidades de cada grupo · {len(frame)} en la cohorte {year}"):
        st.dataframe(export, hide_index=True, width="stretch")
    st.download_button("Descargar grupos y variables (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"benchmark_clusters_{year}.csv", mime="text/csv")


def silhouette_chart(diagnostics: pd.DataFrame, algorithm: str, k: int):
    """Silueta de cada algoritmo según el número de grupos; la elegida va marcada."""
    d = diagnostics.copy()
    d["estado"] = np.where(d.admisible, "Admisible", "Descartada")
    d["concentracion_pct"] = d.concentracion
    algorithms = ["K-means", "Jerárquico Ward", "Mezcla gaussiana"]
    base = alt.Chart(d).encode(
        x=alt.X("k:O", title="Número de grupos", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("silueta:Q", title="Silueta · mayor es más separado", scale=alt.Scale(zero=False, nice=True)),
        color=alt.Color("algoritmo:N", title=None, sort=algorithms,
                        scale=alt.Scale(domain=algorithms, range=["#2a78d6", "#eb6834", "#1baf7a"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("algoritmo:N", title="Algoritmo"), alt.Tooltip("k:O", title="Grupos"),
                 alt.Tooltip("silueta:Q", title="Silueta", format=".3f"), alt.Tooltip("estado:N", title="Estado"),
                 alt.Tooltip("grupo_minimo:Q", title="Grupo menor"),
                 alt.Tooltip("concentracion_pct:Q", title="Grupo mayor", format=".0%")])
    layers = [base.mark_line(strokeWidth=2),
              base.transform_filter("datum.admisible").mark_point(filled=True, size=90, opacity=1),
              base.transform_filter("!datum.admisible").mark_point(filled=False, size=90, strokeWidth=1.6,
                                                                   fill="#ffffff", opacity=1)]
    chosen = d.loc[d.algoritmo.eq(algorithm) & d.k.eq(k)]
    if not chosen.empty:
        layers += [alt.Chart(chosen).mark_point(size=480, filled=False, color="#1a1a1a", strokeWidth=2)
                   .encode(x="k:O", y="silueta:Q"),
                   alt.Chart(chosen).mark_text(dy=-22, fontWeight=700, color="#1a1a1a")
                   .encode(x="k:O", y="silueta:Q", text=alt.value("Elegido"))]
    return alt.layer(*layers).properties(height=320)


def render_models(model, config, code, year, years, profiles, nearest, number):
    st.subheader("¿Qué modelo forma los grupos y cuán firme es la agrupación?")
    st.markdown(
        f"Se prueban tres algoritmos con {', '.join(map(str, config['k_values']))} grupos, sobre el mismo perfil "
        "de ingreso y la misma distancia:\n"
        "- **K-means**: forma grupos alrededor de centros; cada universidad va al centro más cercano.\n"
        "- **Jerárquico Ward**: parte con cada universidad sola y va uniendo las más parecidas, de a pares.\n"
        "- **Mezcla gaussiana**: supone que cada grupo es una nube con su propia dispersión.\n\n"
        f"Una solución es **admisible** si ningún grupo tiene menos de {config['minimum_cluster']} universidades "
        f"y ninguno concentra más del {config['maximum_group_share']:.0%}. Entre las admisibles gana la de mayor "
        "silueta, que mide cuánto se parece cada universidad a su grupo frente al grupo vecino.")
    if not model.k:
        st.warning("Ninguna partición cumple los requisitos de tamaño y concentración. "
                   "Se mantienen disponibles los vecinos por perfil y la selección manual.")
    else:
        grupo_ua = model.universities.loc[model.universities.cod_inst.eq(code), "grupo"].iloc[0]
        peso_ua = float(model.universities.grupo.eq(grupo_ua).mean())
        st.write(f"Alternativa seleccionada: **{model.algorithm}, {model.k} grupos**, "
                 f"silueta **{model.silhouette:.3f}**. El grupo de la UA reúne el "
                 f"**{peso_ua:.0%}** de las universidades incluidas.")
        if model.silhouette < .25:
            st.warning("La separación entre grupos es débil (silueta < 0,25, umbral orientativo). "
                       "Interpreta el cluster como exploratorio y revisa las distancias de los pares.")
    st.altair_chart(silhouette_chart(model.diagnostics, model.algorithm, model.k), width="stretch")
    st.caption("Punto lleno: solución admisible; punto vacío: descartada por tamaño o concentración. "
               "El círculo negro marca la elegida.")
    st.dataframe(model.diagnostics.rename(columns={
        "algoritmo": "Algoritmo", "k": "Grupos", "silueta": "Silueta", "grupo_minimo": "Grupo menor",
        "grupo_maximo": "Grupo mayor", "concentracion": "Concentración del grupo mayor",
        "admisible": "Admisible"}).round(3), hide_index=True, width="stretch")
    st.caption("Por qué el límite de concentración: la silueta premia las particiones que aíslan unas pocas "
               "instituciones atípicas y dejan al resto en un único grupo. Esa solución separa bien pero no sirve "
               "para comparar, porque el grupo de la UA terminaría conteniendo casi todas las universidades.")
    if model.k:
        groups = group_order(model.universities.grupo)
        sizes = model.universities.grupo.value_counts().reindex(groups).rename_axis("grupo").reset_index(name="n")
        sizes["vista"] = [f"{g} (UA)" if g == grupo_ua else g for g in sizes.grupo]
        bars = alt.Chart(sizes).encode(
            y=alt.Y("vista:N", sort=list(sizes.vista), title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("n:Q", title="Universidades"),
            tooltip=[alt.Tooltip("vista:N", title="Grupo"), alt.Tooltip("n:Q", title="Universidades")])
        st.markdown("**Tamaño de cada grupo**")
        st.altair_chart((bars.mark_bar(cornerRadiusEnd=4, height=18).encode(
            color=alt.Color("vista:N", legend=None, sort=list(sizes.vista),
                            scale=alt.Scale(domain=list(sizes.vista), range=CLUSTER_COLORS[:len(sizes)])))
            + bars.mark_text(align="left", dx=4, color="#1a1a1a").encode(text="n:Q"))
            .properties(height=34 * len(sizes)), width="stretch")

        st.markdown("**¿Otro algoritmo dejaría a la UA con las mismas universidades?**")
        agreement, core = partition_agreement(model, code)
        st.dataframe(agreement.round(3), hide_index=True, width="stretch")
        st.caption(f"Los tres algoritmos con {model.k} grupos. «También están en el grupo elegido» cuenta cuántas "
                   "de las universidades que acompañan a la UA en ese algoritmo la acompañan también en la "
                   "solución elegida. El índice Rand ajustado compara las particiones completas: 1 es idéntica; "
                   "0, como al azar.")
        if core:
            st.success(f"**Núcleo estable:** {len(core)} "
                       + ("universidad comparte" if len(core) == 1 else "universidades comparten")
                       + " grupo con la UA en los tres algoritmos: " + ", ".join(map(short_name, core)) + ".")
        else:
            st.info("Ninguna universidad comparte grupo con la UA en los tres algoritmos: el grupo depende del "
                    "algoritmo. Para elegir pares conviene la cercanía por perfil, que no depende de los grupos.")

    st.markdown("**Estabilidad entre cohortes**")
    others = [y for y in years if y != year]
    if others:
        previous = min(others, key=lambda y: abs(y - year))
        try:
            other_model = cached_model(profiles.loc[profiles.cohorte.eq(previous)], config)
            previous_neighbors = neighbors(other_model, code)
            now = set(nearest.head(int(number)).cod_inst)
            then = set(previous_neighbors.head(int(number)).cod_inst)
            overlap = len(now & then)
            st.info(f"Estabilidad de vecinos: {overlap} de los {int(number)} pares más cercanos de {year} "
                    f"también están entre los más cercanos de {previous}. "
                    "Se recalculan perfiles y escalas en cada cohorte; esta medida no valida causalidad.")
            # Los bloques activos NO son los mismos en todos los anios: la
            # selectividad solo existe desde 2021. Una caida de estabilidad
            # en ese borde puede ser el cambio de bloques y no un cambio
            # real del sistema, asi que hay que decirlo antes del numero.
            distintos = set(model.groups) ^ set(other_model.groups)
            if distintos:
                st.warning(f"Las dos cohortes no se comparan con los mismos bloques: "
                           f"{', '.join(sorted(distintos))} solo está disponible en una de "
                           "ellas. Parte de la diferencia de estabilidad es ese cambio de "
                           "variables, no un cambio en el sistema universitario.")
            common = model.universities[["cod_inst", "grupo"]].merge(
                other_model.universities[["cod_inst", "grupo"]], on="cod_inst", suffixes=("_actual", "_otra"))
            if model.k and other_model.k and len(common) > 1:
                ari = adjusted_rand_score(common.grupo_actual, common.grupo_otra)
                st.caption(f"Estabilidad de la partición completa: índice Rand ajustado {ari:.3f} "
                           f"sobre {len(common)} universidades comunes. 1 indica agrupamientos idénticos; "
                           "0, concordancia similar al azar. Los algoritmos y k se eligen por separado en cada cohorte.")
        except ValueError as exc:
            st.caption(f"Estabilidad temporal no disponible: {exc}")


def render_benchmark(results_dir: Path) -> None:
    st.title("Universidad Autónoma · Benchmark")
    st.markdown("**¿Cómo se sitúa la UA frente a universidades de perfil similar?**")
    st.caption("Cohortes de ingreso · Datos públicos Mineduc/SIES · Comparación descriptiva de continuidad al año siguiente")
    with st.expander("Cómo usar este complemento del proyecto"):
        st.markdown("**1. Compara la continuidad** de la UA y revisa el tamaño de las cohortes. "
                    "**2. Examina los pares** y comprueba si la brecha cambia al ampliar el grupo. "
                    "**3. Explora áreas y recursos** para formular preguntas de gestión. "
                    "El benchmark aporta contexto institucional; el cockpit de alerta temprana es una "
                    "demostración separada de predicción por estudiante y asignatura, con datos sintéticos y OULAD.")
    files = [results_dir / name for name in ["retention_universities.parquet", "benchmark_profiles.parquet", "benchmark_manifest.json"]]
    if not all(p.exists() for p in files):
        st.info("Aún no están preparados los perfiles para identificar universidades comparables.")
        st.code("python scripts/build_retention.py\npython scripts/build_benchmark.py", language="bash")
        return
    config_path = Path(__file__).resolve().parents[3] / "config/benchmark.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        data, profiles, manifest = load_benchmark(
            str(results_dir), tuple(p.stat().st_mtime_ns for p in files))
    except ValueError as exc:
        st.warning(str(exc))
        st.code("python scripts/build_benchmark.py", language="bash")
        return
    if profiles.empty or data.empty:
        st.info("No hay cohortes disponibles para comparar.")
        return
    c1, c2, c3 = st.columns([1, 2, 2])
    years = sorted(set(profiles.cohorte) & set(data.cohorte))
    if not years:
        st.info("Los perfiles y los resultados no tienen cohortes en común.")
        return
    year = c1.selectbox("Cohorte de ingreso", years, index=len(years) - 1, key="bench_year")
    metric = c2.selectbox("Continuidad que se compara", list(LABELS), format_func=LABELS.get, key="bench_metric")
    area_name = c3.selectbox("Área del resultado", ["Todas las áreas"] + sorted(data.loc[data.cohorte.eq(year), "area_conocimiento"].unique()), key="bench_area")
    area = None if area_name == "Todas las áreas" else area_name
    try:
        cohort = profiles.loc[profiles.cohorte.eq(year)].copy()
        code = target_code(cohort, config["target_name"])
        with st.spinner("Identificando perfiles comparables…"):
            model = cached_model(cohort, config)
            nearest = neighbors(model, code)
    except ValueError as exc:
        st.info(str(exc))
        return
    # La selección se resuelve por perfiles de la cohorte completa, antes de mirar resultados.
    selection, count = st.columns([3, 1])
    mode = selection.radio("Grupo de comparación", ["Más cercanas por perfil", "Mismo cluster que la UA", "Selección manual"],
                           horizontal=True, key="bench_mode")
    number = count.number_input("Número de pares", min_value=1, max_value=len(nearest),
                                value=min(config["nearest_peers"], len(nearest)), disabled=mode != "Más cercanas por perfil",
                                key="bench_count")
    if mode == "Mismo cluster que la UA":
        peer_ids = nearest.loc[nearest.mismo_grupo_ua, "cod_inst"].tolist()
    elif mode == "Selección manual":
        names = nearest.set_index("cod_inst").nomb_inst.to_dict()
        peer_ids = st.multiselect("Universidades pares (UA permanece como referencia)", nearest.cod_inst.tolist(),
                                  default=nearest.head(int(number)).cod_inst.tolist(), format_func=names.get,
                                  key=f"bench_manual_{year}")
    else:
        peer_ids = nearest.head(int(number)).cod_inst.tolist()
    if not peer_ids:
        st.info("No hay pares seleccionados. Elige universidades manualmente o utiliza las más cercanas por perfil.")
        return
    shown, stats = compare_outcomes(data, int(year), code, peer_ids, metric, area)
    st.caption(f"Ingreso {year} → matrícula {year + 1} · {len(peer_ids)} pares seleccionados · "
               "Los pares se identifican con el perfil completo de ingreso; cambiar el área o la continuidad no cambia su selección.")
    if stats["ua_n"] == 0:
        st.info("La UA no tiene inscripciones con seguimiento en esta área y cohorte. Selecciona otra área.")
        return
    if stats["pares_disponibles"] < len(peer_ids):
        st.warning(f"Solo {stats['pares_disponibles']} de {len(peer_ids)} pares tienen resultados en esta área. "
                   "Las referencias usan únicamente los pares con denominador disponible.")
    # En 2007 y 2008 falta `cod_carrera` en una quinta parte de la matricula.
    # Sin codigo la fila no puede calzar a nivel de carrera aunque la persona
    # haya seguido en la misma, asi que ese indicador aparece hundido sin que
    # nadie haya desertado. Los niveles de universidad y sistema no dependen
    # de ese campo y si son comparables en esos anios.
    if metric == "misma_carrera" and "con_cod_carrera" in data.columns:
        cohorte = data.loc[data.cohorte.eq(int(year))]
        total = cohorte.n.sum()
        cobertura = cohorte.con_cod_carrera.sum() / total if total else 1.0
        if cobertura < 0.95:
            st.warning(f"En la cohorte {year} solo el {cobertura:.0%} de las inscripciones trae "
                       "código de carrera, así que la continuidad **de carrera** aparece más baja "
                       "de lo que fue: sin código la fila no puede calzar aunque el estudiante "
                       "haya seguido en la misma carrera. Usa continuidad en la misma universidad "
                       "o en el sistema, que no dependen de ese campo.")
    a, b, c, d = st.columns(4)
    a.metric("Continuidad UA", percent(stats["ua"]))
    a.caption(f"{stats['ua_n']:,} inscripciones con seguimiento")
    b.metric("Pares · promedio ponderado", percent(stats["pares"]), gap(stats["ua"], stats["pares"]), delta_color="off")
    c.metric("Resto del sistema universitario", percent(stats["nacional"]), gap(stats["ua"], stats["nacional"]), delta_color="off")
    d.metric("Pares · misma mezcla de áreas UA", percent(stats["ajustada"]), gap(stats["ua_comun"], stats["ajustada"]), delta_color="off")
    st.caption("pp = puntos porcentuales. Las referencias de pares y del sistema excluyen a la UA. "
               f"El ajuste por áreas cubre {stats['cobertura']:.1%} de las inscripciones UA del filtro; "
               "su brecha usa la retención UA de esas mismas áreas comunes.")
    small = shown.loc[shown.n.between(1, 29), "nomb_inst"].tolist()
    if small:
        st.warning("Comparaciones de tamaño reducido (menos de 30 inscripciones): " + "; ".join(small)
                   + ". Interpreta sus porcentajes con el denominador a la vista.")
    scenarios = sensitivity(data, int(year), code, nearest.cod_inst.tolist(), metric, area)
    frame = positioning_frame(model, nearest, data, int(year), metric, area, code, peer_ids)
    position, clusters, mapa, similar, models, evolution, methodology = st.tabs(
        ["Posición de la UA", "Clusters por variables", "Mapa de posicionamiento", "Por qué son comparables",
         "Modelos de agrupamiento", "Evolución y áreas", "Método y cobertura"])
    with position:
        st.subheader("La UA frente a sus pares")
        finite = shown.loc[shown.retencion.notna()].copy()
        finite["referencia"] = np.where(finite.es_ua, "Universidad Autónoma", "Universidad par")
        chart = alt.Chart(finite).mark_bar().encode(
            y=alt.Y("nomb_inst:N", title=None, sort="-x", axis=alt.Axis(labelLimit=340, labelFontSize=12)),
            x=alt.X("retencion:Q", title=LABELS[metric], scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("referencia:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Universidad Autónoma", "Universidad par"], range=[UA_RED, BLUE])),
            tooltip=[alt.Tooltip("nomb_inst:N", title="Universidad"), alt.Tooltip("n:Q", title="Inscripciones"),
                     alt.Tooltip("retencion:Q", title="Continuidad", format=".1%")])
        if np.isfinite(stats["pares"]):
            rule = alt.Chart(pd.DataFrame({"promedio": [stats["pares"]]})).mark_rule(
                color="#555555", strokeDash=[5, 4]).encode(x="promedio:Q")
            chart = chart + rule
        st.altair_chart(chart.properties(height=max(260, 35 * len(shown))), width="stretch")
        st.caption("Rojo e identificación por nombre: UA. Línea discontinua: promedio ponderado de los pares disponibles.")
        if np.isfinite(stats["pares"]):
            delta = (stats["ua"] - stats["pares"]) * 100
            st.markdown(f"La continuidad de la UA está **{abs(delta):.1f} puntos porcentuales "
                        f"{'sobre' if delta >= 0 else 'bajo'} sus pares** en este filtro. "
                        "La diferencia describe estas cohortes; no mide un efecto atribuible a la universidad.")
        export = shown.sort_values("retencion", ascending=False)[["cod_inst", "nomb_inst", "n", "sin_mrun", "retencion", "brecha_vs_ua_pp"]].copy()
        export["retencion"] *= 100
        export.insert(0, "cohorte", year)
        export["area"] = area_name
        export["definicion"] = LABELS[metric]
        export["seleccion_pares"] = mode
        export["referencia_pares_pct"] = 100 * stats["pares"]
        export = export.rename(columns={"nomb_inst": "Universidad", "n": "Inscripciones", "sin_mrun": "Sin MRUN",
                                        "retencion": "Continuidad (%)", "brecha_vs_ua_pp": "Brecha vs UA (pp)"}).round(2)
        st.dataframe(export[["Universidad", "Inscripciones", "Sin MRUN", "Continuidad (%)", "Brecha vs UA (pp)"]],
                     hide_index=True, width="stretch")
        st.download_button("Descargar benchmark con contexto", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"benchmark_ua_{year}_{metric}.csv", mime="text/csv")
        st.download_button("Descargar informe ejecutivo (HTML)",
                           executive_html(int(year), area_name, LABELS[metric], mode, stats,
                                          export[["Universidad", "Inscripciones", "Sin MRUN", "Continuidad (%)", "Brecha vs UA (pp)"]],
                                          scenarios, manifest["generado_utc"]),
                           file_name=f"informe_benchmark_ua_{year}.html", mime="text/html")
        st.subheader("Permanencia y movilidad al año siguiente")
        transitions = continuity_breakdown(shown)
        states = ["Misma carrera", "Otra carrera en la institución", "Otra institución", "Sin matrícula observada"]
        st.altair_chart(alt.Chart(transitions).mark_bar().encode(
            y=alt.Y("Referencia:N", title=None),
            x=alt.X("Proporción:Q", title="Distribución de la cohorte", axis=alt.Axis(format="%")),
            color=alt.Color("Estado:N", sort=states, scale=alt.Scale(domain=states,
                range=["#2a78d6", "#1baf7a", "#eda100", "#8a8880"]), legend=alt.Legend(orient="bottom")),
            order="Orden:Q", tooltip=["Referencia", "Estado", "Inscripciones", alt.Tooltip("Proporción:Q", format=".1%")]
        ).properties(height=140), width="stretch")
        st.caption("Estados excluyentes por inscripción de origen. Un cambio de carrera o institución no es abandono. "
                   "Esta descomposición muestra todos los estados, cualquiera sea el indicador elegido arriba.")
        st.subheader("¿La brecha depende de escoger cinco pares?")
        st.dataframe(scenarios.round(2), hide_index=True, width="stretch")
        st.caption("Escenarios con los 3, 5, 8 y 10 vecinos más cercanos; conservan la cohorte, el área y el indicador. "
                   "Reemplazan la selección manual o de cluster solo en esta tabla. No son intervalos de confianza.")
    with clusters:
        render_clusters(model, frame, code, int(year), area_name)
    with models:
        render_models(model, config, code, year, years, profiles, nearest, number)
    with mapa:
        st.subheader("¿Dónde se sitúa la UA entre las universidades chilenas?")
        st.caption("Cada punto es una universidad. Las líneas punteadas marcan la mediana de las universidades con dato en ambos ejes "
                   "y dividen el gráfico en cuadrantes. Sólo la UA y sus pares llevan nombre.")
        st.caption("El filtro de área afecta la continuidad; los costos, recursos y distribuciones describen el perfil completo "
                   "o toda la institución. No son recursos exclusivos del área seleccionada.")
        disponibles = [k for k in AXES if k in frame.columns and frame[k].notna().any()]
        ex, ey = st.columns(2)
        x_var = ex.selectbox("Eje horizontal", disponibles, index=disponibles.index("arancel_mediano")
                             if "arancel_mediano" in disponibles else 0,
                             format_func=lambda k: AXES[k][0], key="bench_x")
        y_var = ey.selectbox("Eje vertical", disponibles, index=disponibles.index("retencion")
                             if "retencion" in disponibles else 0,
                             format_func=lambda k: AXES[k][0], key="bench_y")
        grafico, n_puntos = scatter(frame, x_var, y_var)
        if x_var == y_var:
            st.info("Elegiste el mismo indicador en ambos ejes; cambia uno para realizar una comparación útil.")
        if grafico is None:
            st.info("No hay universidades con datos en ambas variables para este filtro.")
        else:
            st.altair_chart(grafico, width="stretch")
            faltan = len(frame) - n_puntos
            if faltan:
                st.caption(f"{n_puntos} universidades con dato en ambos ejes; {faltan} quedan fuera por dato ausente.")
            target_row = frame.loc[frame.cod_inst.eq(code)]
            if target_row[[x_var, y_var]].isna().any(axis=None):
                st.warning("La UA no tiene dato en alguno de los ejes; su punto no aparece en este gráfico.")

        st.subheader("¿Las universidades parecidas obtienen resultados parecidos?")
        st.caption("Distancia al perfil de la UA frente a continuidad. La asociación describe esta cohorte; "
                   "no mide cuánto explica el perfil ni identifica un efecto causal.")
        rel, _ = scatter(frame, "distancia", "retencion", quadrants=False)
        if rel is None:
            st.info("Sin datos suficientes para relacionar perfil y resultado.")
        else:
            st.altair_chart(rel, width="stretch")
            pares = frame.loc[frame.distancia.notna() & frame.retencion.notna() & frame.distancia.gt(0)]
            if len(pares) > 5 and pares.distancia.nunique() > 1 and pares.retencion.nunique() > 1:
                # La lectura se deriva del valor observado. Un texto fijo se
                # vuelve falso en cuanto cambian los datos o el filtro.
                r = float(np.corrcoef(pares.distancia, pares.retencion)[0, 1])
                rs = float(pares.distancia.corr(pares.retencion, method="spearman"))
                fuerza = ("prácticamente nula" if abs(r) < .2 else
                          "débil" if abs(r) < .4 else
                          "moderada" if abs(r) < .6 else "apreciable")
                sentido = ("las universidades de perfil más distinto a la UA tienden a mostrar "
                           "menor continuidad" if r < 0 else
                           "las universidades de perfil más distinto a la UA tienden a mostrar "
                           "mayor continuidad")
                st.markdown(
                    f"Correlación entre distancia de perfil y continuidad: **{r:+.2f}** "
                    f"(Spearman {rs:+.2f}) sobre {len(pares)} universidades: relación **{fuerza}**. "
                    + (f"Con ese signo, {sentido}. " if abs(r) >= .2 else
                       "En esta cohorte no se observa una asociación lineal clara con la distancia a la UA. "))
                st.caption("La distancia se mide desde la UA, así que esto describe su vecindario y no una "
                           "regularidad del sistema. Tampoco implica causalidad: perfil y resultado pueden "
                           "compartir causas que no están en el modelo, como selectividad de admisión o "
                           "composición socioeconómica del estudiantado.")

        st.subheader("Posición relativa de la UA")
        filas = []
        for clave in disponibles:
            serie = frame[clave].dropna()
            valor = frame.loc[frame.cod_inst.eq(code), clave]
            if serie.empty or valor.empty or not np.isfinite(valor.iloc[0]):
                continue
            v = float(valor.iloc[0])
            multiplier = 100 if AXES[clave][1] == "%" else 1
            filas.append({"Variable": AXES[clave][0] + (" (%)" if multiplier == 100 else ""), "UA": multiplier * v,
                          "Mediana de universidades con dato": multiplier * float(serie.median()),
                          "Percentil UA": round(percentile(serie, v), 0),
                          "Universidades con dato": int(serie.size)})
        if filas:
            st.dataframe(pd.DataFrame(filas).round(2), hide_index=True, width="stretch")
            st.caption("El percentil indica qué porcentaje de las demás universidades queda por debajo de la UA en "
                       "cada variable; los empates cuentan la mitad. No es una puntuación: en arancel o duración, "
                       "estar arriba no es mejor.")

    with similar:
        st.subheader("Similitud institucional explicada")
        bloques_activos = [b for b in model.groups]
        st.markdown("La cercanía combina **tamaño de cohorte y sedes, mezcla de áreas, presencia regional, "
                    "jornada y modalidad**, y —cuando hay datos PAES— **el nivel y la dispersión de los "
                    "puntajes de ingreso**. Todos los bloques pesan igual. "
                    "La retención no participa en la distancia ni en los clusters.")
        if "selectividad" in bloques_activos:
            st.info("**La selectividad participa en la distancia.** Los pares no son solo universidades que "
                    "ofrecen carreras parecidas, sino que además reciben estudiantes con puntajes parecidos. "
                    "Eso excluye del grupo a instituciones de oferta similar pero admisión mucho más o menos "
                    "selectiva. Se configura en `block_weights` de config/benchmark.yml.")
        st.caption("Se describe el perfil de las cohortes de ingreso a carrera, no la totalidad de la universidad. "
                   "Los recursos institucionales y el cuerpo docente (base INDICES del CNED) y la selectividad "
                   "de admisión (PAES cruzada por MRUN) están disponibles como variables descriptivas en el "
                   "mapa de posicionamiento. Los recursos y el cuerpo docente **no participan** en la "
                   "distancia; la selectividad sí, si está activada en la configuración. "
                   "La selectividad describe solo a quienes rindieron la PAES: revisa su cobertura antes de "
                   "interpretarla. Sigue sin incorporarse actividad de investigación.")
        peers = nearest.loc[nearest.cod_inst.isin(peer_ids)]
        table = peers[["nomb_inst", "cohorte_total", "sedes", "grupo", "distancia", "mismo_grupo_ua"]].rename(columns={
            "nomb_inst": "Universidad", "cohorte_total": "Tamaño de cohorte", "sedes": "Sedes",
            "grupo": "Cluster", "distancia": "Distancia a UA", "mismo_grupo_ua": "Mismo cluster UA"})
        st.dataframe(table.round(3), hide_index=True, width="stretch")
        st.caption("Menor distancia implica mayor similitud en las variables utilizadas. No es una probabilidad ni un porcentaje de similitud.")
        ua_profile = model.universities.loc[model.universities.cod_inst.eq(code)].iloc[0]
        st.caption(f"Perfil UA: {int(ua_profile.cohorte_total):,} inscripciones de ingreso y {int(ua_profile.sedes)} sedes.")
        st.markdown("**¿En qué se parecen y en qué difieren?**")
        dimensions = peers[["nomb_inst", *[f"distancia_{b}" for b in model.groups]]].melt(
            id_vars="nomb_inst", var_name="bloque", value_name="distancia")
        dimensions["bloque"] = dimensions.bloque.map(
            {"distancia_escala": "Tamaño y sedes", "distancia_areas": "Áreas",
             "distancia_regiones": "Regiones", "distancia_docencia": "Jornada y modalidad",
             "distancia_selectividad": "Selectividad"}).fillna(dimensions.bloque)
        st.altair_chart(alt.Chart(dimensions).mark_rect().encode(
            x=alt.X("bloque:N", title=None), y=alt.Y("nomb_inst:N", title=None, axis=alt.Axis(labelLimit=340)),
            color=alt.Color("distancia:Q", title="Distancia", scale=alt.Scale(scheme="blues", domainMin=0)),
            tooltip=["nomb_inst", "bloque", alt.Tooltip("distancia:Q", format=".3f")]), width="stretch")
        block = st.selectbox("Distribución que quieres contrastar", ["area", "region", "jornada", "modalidad"],
                             format_func={"area": "Áreas de conocimiento", "region": "Regiones", "jornada": "Jornadas", "modalidad": "Modalidades"}.get,
                             key="bench_profile_block")
        columns = [col for col in profiles if col.startswith(block + "::")]
        comparison = pd.DataFrame({"Categoría": [col.split("::", 1)[1] for col in columns],
            "Universidad Autónoma": ua_profile[columns].astype(float).to_numpy(),
            "Pares": np.average(peers[columns].to_numpy(dtype=float), axis=0, weights=peers.cohorte_total)})
        comparison = comparison.melt(id_vars="Categoría", var_name="Referencia", value_name="Proporción")
        st.altair_chart(alt.Chart(comparison).mark_bar().encode(
            y=alt.Y("Categoría:N", title=None, axis=alt.Axis(labelLimit=280)), x=alt.X("Proporción:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            yOffset="Referencia:N", color=alt.Color("Referencia:N", scale=alt.Scale(domain=["Universidad Autónoma", "Pares"], range=[UA_RED, BLUE])),
            tooltip=["Categoría", "Referencia", alt.Tooltip("Proporción:Q", format=".1%")]), width="stretch")
        st.caption("Distribución de pares ponderada por sus inscripciones de ingreso. Incluye registros sin MRUN.")
        st.subheader("Mapa de perfiles universitarios")
        mapped = model.universities.copy()
        mapped["seleccion"] = np.where(mapped.cod_inst.eq(code), "UA", np.where(mapped.cod_inst.isin(peer_ids), "Par seleccionado", "Otras"))
        chart = alt.Chart(mapped).mark_point(filled=True, size=100, opacity=.8).encode(
            x=alt.X("mapa_x:Q", title="Componente 1"), y=alt.Y("mapa_y:Q", title="Componente 2"),
            color=alt.Color("seleccion:N", title=None, scale=alt.Scale(domain=["UA", "Par seleccionado", "Otras"], range=[UA_RED, BLUE, "#b0b0b0"])),
            shape=alt.Shape("seleccion:N", title=None),
            tooltip=["nomb_inst", "grupo", "cohorte_total", "sedes"])
        label = alt.Chart(mapped.loc[mapped.cod_inst.eq(code)]).mark_text(text="UA", dy=-14, fontWeight="bold", color=UA_RED).encode(x="mapa_x:Q", y="mapa_y:Q")
        st.altair_chart((chart + label).properties(height=360), width="stretch")
        st.caption(f"Proyección PCA: representa {model.explained_variance:.1%} de la variación del perfil. "
                   "La selección utiliza todas las dimensiones; las distancias del mapa son una aproximación.")
    with evolution:
        st.subheader("¿La brecha se mantiene entre cohortes?")
        rows = []
        for y in years:
            _, s = compare_outcomes(data, int(y), code, peer_ids, metric, area)
            for name, value in [("Universidad Autónoma", s["ua"]), ("Pares seleccionados", s["pares"]), ("Resto del sistema", s["nacional"])]:
                rows.append({"cohorte": str(y), "referencia": name, "retencion": value,
                             "pares_disponibles": s["pares_disponibles"]})
        history = pd.DataFrame(rows)
        st.altair_chart(alt.Chart(history).mark_line(point=True).encode(
            x=alt.X("cohorte:O", title="Cohorte de ingreso"),
            y=alt.Y("retencion:Q", title="Continuidad", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("referencia:N", title=None, scale=alt.Scale(domain=["Universidad Autónoma", "Pares seleccionados", "Resto del sistema"], range=[UA_RED, BLUE, "#777777"])),
            tooltip=["cohorte", "referencia", "pares_disponibles", alt.Tooltip("retencion:Q", format=".1%")]), width="stretch")
        st.caption("Se mantiene el mismo listado de pares seleccionado arriba para todas las cohortes. "
                   "El peso de cada par depende de sus inscripciones en cada año; los datos ausentes no se sustituyen por cero. "
                   "Dos cohortes permiten una primera comparación, no establecer una tendencia de largo plazo.")
        st.subheader("Áreas que explican el resultado agregado")
        area_rows = []
        for label in sorted(data.loc[data.cohorte.eq(year) & data.cod_inst.eq(code), "area_conocimiento"].unique()):
            _, s = compare_outcomes(data, int(year), code, peer_ids, metric, label)
            area_rows.append({"Área": label, "Inscripciones UA": s["ua_n"], "UA (%)": 100 * s["ua"],
                              "Pares (%)": 100 * s["pares"], "Brecha UA (pp)": 100 * (s["ua"] - s["pares"]),
                              "Pares disponibles": s["pares_disponibles"]})
        st.dataframe(pd.DataFrame(area_rows).round(1), hide_index=True, width="stretch")
        st.caption("Desglose de todas las áreas UA de la cohorte, independientemente del filtro superior. "
                   "Una celda vacía indica falta de denominador comparable.")
    with methodology:
        st.caption("La calidad del agrupamiento, la comparación de algoritmos y la estabilidad entre cohortes "
                   "están en la pestaña «Modelos de agrupamiento».")
        st.subheader("Cobertura y trazabilidad")
        resource_fields = [c for c in ["nomb_inst", "estudiantes_total", "matriculas_sin_mrun", "recursos_anio",
                                       "docentes_por_100_alumnos", "share_doctorado", "m2_construido_por_alumno",
                                       "pc_por_100_alumnos", "ejemplares_por_alumno", "acreditacion_cned_fecha"] if c in cohort]
        st.markdown("**Recursos institucionales.** Las razones usan estudiantes únicos identificados en toda la matrícula "
                    "del año, incluidos todos los niveles. Se excluyen MRUN ausentes del denominador y se informa su cantidad. "
                    "Los recursos se toman solo del mismo año; no se sustituyen por observaciones futuras. "
                    "El JCE es una aproximación del proyecto: completa + 0,5 × media + 0,25 × por hora. "
                    "La acreditación CNED es una foto del catálogo, con fecha explícita, y no una serie histórica.")
        st.dataframe(cohort[resource_fields].rename(columns={"nomb_inst": "Universidad"}), hide_index=True, width="stretch")
        st.caption("Los aranceles están ponderados por inscripciones de ingreso; montos en UF se convierten "
                   "con factores anuales aproximados. Los recursos institucionales no intervienen en el clustering.")
        st.write(f"Universidades con perfil en la cohorte: **{len(cohort)}**. "
                 f"Incluidas en el agrupamiento: **{len(model.universities)}**, "
                 f"con al menos {config['minimum_cohort']} inscripciones de ingreso.")
        coverage = model.universities[["nomb_inst", "cohorte_total", "cobertura_area", "cobertura_region", "cobertura_jornada", "cobertura_modalidad"]].copy()
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.caption("Coberturas expresadas entre 0 y 1. La categoría «Sin información» se mantiene explícita en el perfil.")
        st.markdown("**Cómo se construye la distancia.** Tamaño y sedes se transforman con log(1+x). "
                    "Las distribuciones usan raíz cuadrada de las proporciones; cada bloque se normaliza por su "
                    "varianza total y recibe el peso declarado en la configuración. La selectividad, si "
                    "participa, se estandariza como el bloque de escala; un dato ausente se imputa con la "
                    "mediana y no con cero, para no inventar una universidad de puntaje mínimo. "
                    "Las universidades pesan igual al ajustar los clusters. "
                    "Cambiar estas variables o pesos puede cambiar los pares.")
        st.markdown("**Selección de la partición.** Se exige un tamaño mínimo de grupo y que ninguno concentre "
                    f"más del {config['maximum_group_share']:.0%} de las universidades. Solo entre las soluciones "
                    "que cumplen ambas condiciones se compara la silueta. Los vecinos más cercanos por perfil no "
                    "dependen de esta elección: se calculan sobre la distancia, no sobre los clusters.")
        st.markdown("**Titulación.** Describe a la promoción que EGRESA ese año, no a la cohorte que "
                    "ingresa: quien se titula en 2024 entró alrededor de 2017-2019, y de la cohorte 2024 "
                    "todavía no se ha titulado nadie. Por eso no es una tasa de titulación —no dice qué "
                    "proporción de quienes entraron llegará a titularse— sino cuánto se demoraron quienes "
                    "sí lo hicieron. Se excluyen los planes de continuidad, cuya duración no es comparable "
                    "porque reconocen estudios previos, y el valor centinela 1900 del año de ingreso, que "
                    "afecta al 10% de los registros.")
        st.markdown("**La titulación oportuna no ordena por calidad.** En estos datos correlaciona −0,08 con "
                    "el puntaje PAES de ingreso y −0,19 con los años de acreditación: depende sobre todo de "
                    "la mezcla de carreras. Las universidades con fuerte peso de ingeniería aparecen abajo "
                    "porque esos programas se alargan, no porque enseñen peor.")
        st.markdown("**Los puntajes de admisión cambian de escala en 2023.** Las cohortes 2021 y 2022 "
                    "rindieron la Prueba de Transición, que iba de 150 a 850 con media 500; desde 2023 "
                    "la PAES va de 100 a 1000 con media cercana a 610. El puntaje crudo **no es "
                    "comparable** entre ambos lados: un promedio institucional salta unos cien puntos "
                    "sin que haya cambiado nada real. Para mirar la serie hay que usar el **percentil "
                    "nacional**, que sí cruza el corte. Los pares no se ven afectados: cada cohorte se "
                    "estandariza y se agrupa por separado. Antes de 2021 no hay dato — la PSU no está "
                    "publicada como microdato abierto — y en esas cohortes el bloque de selectividad "
                    "se omite y su peso se reparte entre los demás.")
        st.markdown("**Titulación por cohorte de ingreso.** Es la contraparte longitudinal de lo anterior y"
                    "sí es una tasa: se toma la cohorte que ingresó en un año y se la busca en las bases de "
                    "titulados de los años siguientes cruzando por MRUN, que es un identificador enmascarado "
                    "pero estable entre bases. Se mide en tres niveles anidados —se tituló de la carrera que "
                    "empezó, se tituló en su universidad de ingreso, se tituló en alguna universidad—. La "
                    "brecha entre el primero y el segundo es cambio de carrera; entre el segundo y el "
                    "tercero, traslado a otra institución. Sin el tercer nivel, todo traslado se contaría "
                    "como fracaso.")
        st.markdown("**Por qué la cohorte de referencia es antigua.** Una cohorte solo puede evaluarse cuando "
                    "su plazo se cumplió, y una carrera de cinco años con dos de holgura necesita siete años "
                    "de datos posteriores. De la cohorte del perfil todavía no se titula nadie, así que se "
                    "muestra la última cohorte con horizonte completo y la app indica de qué año viene. Cada "
                    "estudiante entra al denominador solo si su propio horizonte —su duración nominal más la "
                    "holgura— cabe en los datos disponibles; quien no alcanza a ser observado ese tiempo "
                    "queda fuera del denominador en vez de contarse como si hubiera desertado. Sin esa "
                    "corrección por censura, las cohortes recientes parecen catastróficas por el solo hecho "
                    "de ser recientes.")
        st.markdown("**Denominadores.** Son inscripciones de ingreso a carrera de pregrado universitario, "
                    "no necesariamente personas que ingresan por primera vez a educación superior. Una persona puede "
                    "contar en varias carreras. Se eliminan duplicados de las columnas canónicas. La retención excluye "
                    "registros sin MRUN; los perfiles sí los incluyen. Se compara el mismo código de carrera e institución, "
                    "o solo institución, o cualquier matrícula del sistema, según el indicador elegido.")
        st.markdown("**Ajuste por áreas.** Aplica las proporciones de áreas de la UA a las tasas de los pares. "
                    "Se limita a áreas con datos en ambos lados y muestra su cobertura. Es un ajuste de composición "
                    "académica, no un ajuste completo por riesgo: no controla selección de estudiantes ni condiciones socioeconómicas. "
                    "Las brechas no deben interpretarse como calidad causal ni como deserción definitiva. "
                    "No se muestran intervalos muestrales porque son registros administrativos; persisten errores de registro, "
                    "cobertura y cambios de codificación.")
        st.markdown("Referencia técnica: [evaluación de clusters y silueta — scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation).")
        st.caption(f"Agregados generados: {manifest['generado_utc']}. "
                   "Se comprueba que perfiles y retención correspondan al mismo conjunto de artefactos.")
        provenance = {"datos": manifest, "configuracion": config, "cohorte": int(year), "metrica": metric,
                      "area": area_name, "seleccion": mode, "ua": code, "pares": peer_ids,
                      "variables_por_bloque": model.groups,
                      "sklearn_version": sklearn.__version__,
                      "modelo": {"algoritmo": model.algorithm, "k": model.k,
                                 "silueta": model.silhouette if np.isfinite(model.silhouette) else None}}
        st.download_button("Descargar ficha metodológica (JSON)", json.dumps(provenance, ensure_ascii=False, indent=2),
                           file_name=f"benchmark_ua_metodologia_{year}.json", mime="application/json")
