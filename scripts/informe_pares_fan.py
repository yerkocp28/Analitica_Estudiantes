"""Posición de la FAN frente a universidades pares: python scripts/informe_pares_fan.py

Producto P4 de la propuesta. Compara la continuidad al año siguiente de quienes
ingresan al área de Administración y Comercio en la UA con la de sus cinco
universidades más parecidas por perfil de ingreso, y con el resto del sistema.

Usa solo los agregados públicos versionados en `deploy/data/` (Mineduc/SIES y
CNED), así que el informe puede circular sin restricciones de privacidad.

QUÉ ES Y QUÉ NO

- Los pares se eligen por el perfil completo de ingreso de la universidad
  (tamaño, sedes, áreas, regiones, jornada y selectividad), nunca por su
  retención: si se eligieran por resultado, la comparación sería circular.
- Los pares quedan fijos en la serie: son los de la cohorte más reciente, para
  que la evolución no mezcle cambios de la UA con cambios del grupo.
- Es descriptivo. Una brecha no es un ranking de calidad ni explica su causa.

Escribe documentacion/informe_pares_fan.html.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_analytics.modeling.benchmark import compare_outcomes, fit_peers, neighbors, target_code  # noqa: E402
from student_analytics.modeling.benchmark_reporting import report_html, sensitivity  # noqa: E402

DATOS = ROOT / "deploy" / "data"
SALIDA = ROOT / "documentacion" / "informe_pares_fan.html"
AREA = "Administración y Comercio"
COHORTES = 5
NIVELES = {"misma_carrera": "Sigue en la misma carrera",
           "misma_universidad": "Sigue en la misma universidad",
           "sistema": "Sigue en alguna institución del sistema"}
# Perfil institucional para acreditación: toda la universidad, no solo el área.
PERFIL = {
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


def formato(x, tipo):
    if pd.isna(x):
        return "Sin datos"
    if tipo == "pct":
        return pct(x)
    if tipo == "clp":
        return "$" + f"{x:,.0f}".replace(",", ".")
    return f"{x:.1f}".replace(".", ",") if tipo == "dec" else f"{x:.0f}"


def nombre(serie: pd.Series) -> pd.Series:
    """Las bases vienen en mayúsculas y sin tildes; se recuperan solo las minúsculas."""
    return serie.str.title().str.replace(" De ", " de ").str.replace(" Del ", " del ")


def main() -> int:
    config = yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
    datos = pd.read_parquet(DATOS / "retention_universities.parquet")
    perfiles = pd.read_parquet(DATOS / "benchmark_profiles.parquet")
    anio = int(datos.loc[datos.area_conocimiento.eq(AREA), "cohorte"].max())
    perfil = perfiles.loc[perfiles.cohorte.eq(anio)]
    modelo = fit_peers(perfil, config)
    ua = target_code(modelo.universities, config["target_name"])
    cercanas = neighbors(modelo, ua)
    pares = cercanas.cod_inst.head(config["nearest_peers"]).tolist()
    nombres = nombre(cercanas.nomb_inst.head(config["nearest_peers"])).tolist()

    niveles, stats = [], {}
    for metrica, etiqueta in NIVELES.items():
        _, s = compare_outcomes(datos, anio, ua, pares, metrica, AREA)
        stats[metrica] = s
        niveles.append({"Indicador": etiqueta, "Universidad Autónoma": pct(s["ua"]),
                        f"Pares ({s['pares_disponibles']})": pct(s["pares"]),
                        "Resto del sistema": pct(s["nacional"]),
                        "Diferencia con los pares": pp(s["ua"] - s["pares"])})

    serie, ventaja = [], 0
    for y in range(anio - COHORTES + 1, anio + 1):
        _, s = compare_outcomes(datos, y, ua, pares, "misma_carrera", AREA)
        ventaja += s["ua"] > s["pares"]
        serie.append({"Cohorte": f"{y} → {y + 1}", "Ingresantes UA del área": f"{s['ua_n']:,}".replace(",", "."),
                      "Universidad Autónoma": pct(s["ua"]), "Pares": pct(s["pares"]),
                      "Resto del sistema": pct(s["nacional"]), "Diferencia con los pares": pp(s["ua"] - s["pares"])})

    comparadas, _ = compare_outcomes(datos, anio, ua, pares, "misma_carrera", AREA)
    comparadas = comparadas.sort_values(["es_ua", "retencion"], ascending=[False, False])
    universidades = pd.DataFrame({
        "Universidad": nombre(comparadas.nomb_inst),
        "Ingresantes del área": comparadas.n.map(lambda v: f"{int(v):,}".replace(",", ".")),
        "Sigue en la misma carrera": comparadas.retencion.map(pct),
        "Diferencia con la UA": comparadas.brecha_vs_ua_pp.map(
            lambda v: "—" if pd.isna(v) or v == 0 else f"{v:+.1f}".replace(".", ",") + " pp")})

    escenarios = sensitivity(datos, anio, ua, cercanas.cod_inst.tolist(), "misma_carrera", AREA)
    brechas = escenarios["Brecha UA (pp)"]
    sensibilidad = pd.DataFrame({
        "Pares": escenarios["Pares seleccionados"],
        "Ingresantes del área en los pares": escenarios["Inscripciones pares"].map(lambda v: f"{int(v):,}".replace(",", ".")),
        "Pares: sigue en la misma carrera": (escenarios["Continuidad pares (%)"] / 100).map(pct),
        "Diferencia de la UA": (brechas / 100).map(pp)})

    fila_ua = perfil.loc[perfil.cod_inst.eq(ua)].iloc[0]
    grupo = perfil.loc[perfil.cod_inst.isin(pares)]
    institucional = pd.DataFrame([{
        "Indicador": etiqueta, "Universidad Autónoma": formato(fila_ua[col], tipo),
        "Mediana de los pares": formato(grupo[col].median(), tipo),
        "Rango de los pares": f"{formato(grupo[col].min(), tipo)} – {formato(grupo[col].max(), tipo)}"}
        for col, (etiqueta, tipo) in PERFIL.items() if col in perfil.columns and grupo[col].notna().any()])
    # Para acreditación importa lo que queda fuera de todo el rango de los pares.
    fuera = [f"{etiqueta}: UA {formato(fila_ua[col], tipo)}, "
             f"{'bajo' if fila_ua[col] < grupo[col].min() else 'sobre'} el rango de sus pares "
             f"({formato(grupo[col].min(), tipo)} – {formato(grupo[col].max(), tipo)})."
             for col, (etiqueta, tipo) in PERFIL.items()
             if col in perfil.columns and pd.notna(fila_ua[col]) and grupo[col].notna().any()
             and not grupo[col].min() <= fila_ua[col] <= grupo[col].max()]

    s = stats["misma_carrera"]
    toda_ua = compare_outcomes(datos, anio, ua, pares, "misma_carrera", None)[1]["ua"]
    interna = (f"Aun así, el área queda {pp(toda_ua - s['ua'])[1:]} bajo el promedio de la propia UA "
               f"({pct(toda_ua)}): el margen de mejora está dentro de la universidad, no en la comparación externa."
               if s["ua"] < toda_ua else
               f"El área también supera el promedio de la propia UA ({pct(toda_ua)}).")
    hallazgos = [
        f"En la cohorte {anio}, el {pct(s['ua'])} de quienes ingresaron al área de {AREA} en la UA siguió en la "
        f"misma carrera al año siguiente: {pp(s['ua'] - s['pares'])} frente a sus {len(pares)} universidades "
        f"pares ({pct(s['pares'])}) y {pp(s['ua'] - s['nacional'])} frente al resto del sistema ({pct(s['nacional'])}).",
        f"La UA superó a sus pares en {ventaja} de las últimas {COHORTES} cohortes, con los mismos pares en toda la serie.",
        f"El resultado no depende de cuántos pares se elijan: con 3 a 10, la diferencia va de "
        f"{pp(brechas.min() / 100)} a {pp(brechas.max() / 100)}.",
        interna,
        *fuera,
    ]
    secciones = [
        ("Continuidad al año siguiente · tres niveles", pd.DataFrame(niveles)),
        ("Evolución con los mismos pares", pd.DataFrame(serie)),
        ("Universidades comparadas", universidades),
        ("Sensibilidad al número de pares", sensibilidad),
        ("Perfil institucional frente a los pares · toda la universidad", institucional),
    ]
    notas = [
        "Pares: las universidades más parecidas a la UA por tamaño, sedes, áreas, regiones, jornada y "
        "selectividad de su cohorte de ingreso. No se eligen por retención. Pares de la cohorte "
        f"{anio}: {', '.join(nombres)}.",
        "Continuidad: inscripciones de ingreso a una carrera del área con MRUN disponible, observadas en la "
        "matrícula del año siguiente. Los tres niveles están anidados; la diferencia entre ellos es cambio de "
        "carrera o de institución, no deserción. Una persona puede aparecer en más de una carrera.",
        "Las referencias de pares y del sistema excluyen a la UA y ponderan por inscripciones. Las brechas son "
        "descriptivas: no son un ranking de calidad ni explican su causa.",
        "Fuentes: matrícula pública Mineduc/SIES, admisión DEMRE y recursos institucionales CNED, "
        "en los agregados versionados de deploy/data.",
    ]
    html = report_html(
        "Facultad de Administración y Negocios · Posición frente a universidades pares",
        f"Universidad Autónoma de Chile · Área {AREA} · Cohortes {anio - COHORTES + 1}–{anio} · Datos públicos",
        hallazgos, secciones, notas, datetime.now(timezone.utc).strftime("%d/%m/%Y"))
    SALIDA.write_text(html, encoding="utf-8")
    print("\n".join("· " + h for h in hallazgos))
    print(f"-> {SALIDA.relative_to(ROOT) if SALIDA.is_relative_to(ROOT) else SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
