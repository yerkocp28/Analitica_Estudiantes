"""Posición de un área de la UA frente a sus universidades pares.

Es el cálculo del informe P4 de la FAN. Vive aquí, y no en el script, porque lo
usan dos salidas: `scripts/informe_pares_fan.py` (HTML para circular) y la vista
del piloto FAN. Ambas deben decir exactamente lo mismo.

- Los pares se eligen por el perfil completo de ingreso de la universidad
  (tamaño, sedes, áreas, regiones, jornada y selectividad), nunca por su
  retención: si se eligieran por resultado, la comparación sería circular.
- Los pares quedan fijos en la serie: son los de la cohorte más reciente, para
  que la evolución no mezcle cambios de la UA con cambios del grupo.
- Es descriptivo. Una brecha no es un ranking de calidad ni explica su causa.
"""
from __future__ import annotations

import pandas as pd

from .benchmark import compare_outcomes, fit_peers, neighbors, target_code
from .benchmark_reporting import report_html, sensitivity

LEVELS = {"misma_carrera": "Sigue en la misma carrera",
          "misma_universidad": "Sigue en la misma universidad",
          "sistema": "Sigue en alguna institución del sistema"}
# Perfil institucional para acreditación: toda la universidad, no solo el área.
PROFILE = {
    "paes_percentil_promedio": ("Percentil nacional PAES promedio de la cohorte", "pct"),
    "arancel_mediano": ("Arancel anual mediano", "clp"),
    "acreditacion_anios": ("Años de acreditación institucional", "num"),
    "titulacion_oportuna": ("Titulados dentro de la duración nominal", "pct"),
    "titulacion_cohorte_universidad": ("Cohorte titulada en su universidad de ingreso", "pct"),
    "docentes_por_100_alumnos": ("Docentes JCE por 100 estudiantes", "dec"),
    "share_doctorado": ("Docentes con doctorado", "pct"),
    "share_jornada_completa": ("Docentes con jornada completa", "pct"),
}


def pct(x):
    return "Sin datos" if pd.isna(x) else f"{100 * x:.1f}".replace(".", ",") + "%"


def pp(x):
    return "Sin datos" if pd.isna(x) else f"{100 * x:+.1f}".replace(".", ",") + " pp"


def value(x, kind):
    if pd.isna(x):
        return "Sin datos"
    if kind == "pct":
        return pct(x)
    if kind == "clp":
        return "$" + f"{x:,.0f}".replace(",", ".")
    return f"{x:.1f}".replace(".", ",") if kind == "dec" else f"{x:.0f}"


def count(x):
    return f"{int(x):,}".replace(",", ".")


def name(series: pd.Series) -> pd.Series:
    """Las bases vienen en mayúsculas y sin tildes; se recuperan solo las minúsculas."""
    return series.str.title().str.replace(" De ", " de ").str.replace(" Del ", " del ")


def area_position(data: pd.DataFrame, profiles: pd.DataFrame, config: dict, area: str,
                  cohorts: int = 5) -> dict:
    year = int(data.loc[data.area_conocimiento.eq(area), "cohorte"].max())
    profile = profiles.loc[profiles.cohorte.eq(year)]
    model = fit_peers(profile, config)
    ua = target_code(model.universities, config["target_name"])
    nearest = neighbors(model, ua)
    peers = nearest.cod_inst.head(config["nearest_peers"]).tolist()
    peer_names = name(nearest.nomb_inst.head(config["nearest_peers"])).tolist()

    levels, stats = [], {}
    for metric, label in LEVELS.items():
        _, s = compare_outcomes(data, year, ua, peers, metric, area)
        stats[metric] = s
        levels.append({"Indicador": label, "Universidad Autónoma": pct(s["ua"]),
                       f"Pares ({s['pares_disponibles']})": pct(s["pares"]),
                       "Resto del sistema": pct(s["nacional"]),
                       "Diferencia con los pares": pp(s["ua"] - s["pares"])})

    series, ahead = [], 0
    for y in range(year - cohorts + 1, year + 1):
        _, s = compare_outcomes(data, y, ua, peers, "misma_carrera", area)
        ahead += s["ua"] > s["pares"]
        series.append({"Cohorte": f"{y} → {y + 1}", "Ingresantes UA del área": count(s["ua_n"]),
                       "Universidad Autónoma": pct(s["ua"]), "Pares": pct(s["pares"]),
                       "Resto del sistema": pct(s["nacional"]), "Diferencia con los pares": pp(s["ua"] - s["pares"])})

    compared, _ = compare_outcomes(data, year, ua, peers, "misma_carrera", area)
    compared = compared.sort_values(["es_ua", "retencion"], ascending=[False, False])
    universities = pd.DataFrame({
        "Universidad": name(compared.nomb_inst),
        "Ingresantes del área": compared.n.map(count),
        "Sigue en la misma carrera": compared.retencion.map(pct),
        "Diferencia con la UA": compared.brecha_vs_ua_pp.map(
            lambda v: "—" if pd.isna(v) or v == 0 else f"{v:+.1f}".replace(".", ",") + " pp")})

    scenarios = sensitivity(data, year, ua, nearest.cod_inst.tolist(), "misma_carrera", area)
    gaps = scenarios["Brecha UA (pp)"] / 100
    sensitivity_table = pd.DataFrame({
        "Pares": scenarios["Pares seleccionados"],
        "Ingresantes del área en los pares": scenarios["Inscripciones pares"].map(count),
        "Pares: sigue en la misma carrera": (scenarios["Continuidad pares (%)"] / 100).map(pct),
        "Diferencia de la UA": gaps.map(pp)})

    ua_row = profile.loc[profile.cod_inst.eq(ua)].iloc[0]
    group = profile.loc[profile.cod_inst.isin(peers)]
    shown = [(col, label, kind) for col, (label, kind) in PROFILE.items()
             if col in profile.columns and group[col].notna().any()]
    institutional = pd.DataFrame([{
        "Indicador": label, "Universidad Autónoma": value(ua_row[col], kind),
        "Mediana de los pares": value(group[col].median(), kind),
        "Rango de los pares": f"{value(group[col].min(), kind)} – {value(group[col].max(), kind)}"}
        for col, label, kind in shown])
    # Para acreditación importa lo que queda fuera de todo el rango de los pares.
    outside = [f"{label}: UA {value(ua_row[col], kind)}, "
               f"{'bajo' if ua_row[col] < group[col].min() else 'sobre'} el rango de sus pares "
               f"({value(group[col].min(), kind)} – {value(group[col].max(), kind)})."
               for col, label, kind in shown
               if pd.notna(ua_row[col]) and not group[col].min() <= ua_row[col] <= group[col].max()]

    s = stats["misma_carrera"]
    whole_ua = compare_outcomes(data, year, ua, peers, "misma_carrera", None)[1]["ua"]
    internal = (f"Aun así, el área queda {pp(whole_ua - s['ua'])[1:]} bajo el promedio de la propia UA "
                f"({pct(whole_ua)}): el margen de mejora está dentro de la universidad, no en la comparación externa."
                if s["ua"] < whole_ua else
                f"El área también supera el promedio de la propia UA ({pct(whole_ua)}).")
    findings = [
        f"En la cohorte {year}, el {pct(s['ua'])} de quienes ingresaron al área de {area} en la UA siguió en la "
        f"misma carrera al año siguiente: {pp(s['ua'] - s['pares'])} frente a sus {len(peers)} universidades "
        f"pares ({pct(s['pares'])}) y {pp(s['ua'] - s['nacional'])} frente al resto del sistema ({pct(s['nacional'])}).",
        f"La UA superó a sus pares en {ahead} de las últimas {cohorts} cohortes, con los mismos pares en toda la serie.",
        f"El resultado no depende de cuántos pares se elijan: con 3 a 10, la diferencia va de "
        f"{pp(gaps.min())} a {pp(gaps.max())}.",
        internal,
        *outside,
    ]
    notes = [
        "Pares: las universidades más parecidas a la UA por tamaño, sedes, áreas, regiones, jornada y "
        "selectividad de su cohorte de ingreso. No se eligen por retención. Pares de la cohorte "
        f"{year}: {', '.join(peer_names)}.",
        "Continuidad: inscripciones de ingreso a una carrera del área con MRUN disponible, observadas en la "
        "matrícula del año siguiente. Los tres niveles están anidados; la diferencia entre ellos es cambio de "
        "carrera o de institución, no deserción. Una persona puede aparecer en más de una carrera.",
        "Las referencias de pares y del sistema excluyen a la UA y ponderan por inscripciones. Las brechas son "
        "descriptivas: no son un ranking de calidad ni explican su causa.",
        "Fuentes: matrícula pública Mineduc/SIES, admisión DEMRE y recursos institucionales CNED, "
        "en los agregados versionados de deploy/data.",
    ]
    return {
        "area": area, "year": year, "cohorts": cohorts, "peers": peer_names,
        "ua": s["ua"], "peers_rate": s["pares"], "system": s["nacional"], "whole_ua": whole_ua,
        "ahead": ahead, "gap_range": (float(gaps.min()), float(gaps.max())), "outside": outside,
        "findings": findings, "notes": notes,
        "sections": [("Continuidad al año siguiente · tres niveles", pd.DataFrame(levels)),
                     ("Evolución con los mismos pares", pd.DataFrame(series)),
                     ("Universidades comparadas", universities),
                     ("Sensibilidad al número de pares", sensitivity_table),
                     ("Perfil institucional frente a los pares · toda la universidad", institutional)],
    }


def area_position_html(result: dict, generated: str) -> str:
    return report_html(
        "Facultad de Administración y Negocios · Posición frente a universidades pares",
        f"Universidad Autónoma de Chile · Área {result['area']} · Cohortes "
        f"{result['year'] - result['cohorts'] + 1}–{result['year']} · Datos públicos",
        result["findings"], result["sections"], result["notes"], generated)
