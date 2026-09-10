"""Lecturas descriptivas y exportación autónoma para gestión universitaria."""
from html import escape

import numpy as np
import pandas as pd

from .benchmark import compare_outcomes


def sensitivity(data, year, target, ordered_peers, metric, area=None):
    """Varía el número de vecinos manteniendo distancia, cohorte e indicador."""
    rows = []
    for k in sorted({min(n, len(ordered_peers)) for n in (3, 5, 8, 10)} - {0}):
        _, stats = compare_outcomes(data, year, target, ordered_peers[:k], metric, area)
        rows.append({"Pares seleccionados": k, "Pares con datos": stats["pares_disponibles"],
                     "Inscripciones pares": stats["pares_n"], "Continuidad pares (%)": 100 * stats["pares"],
                     "Brecha UA (pp)": 100 * (stats["ua"] - stats["pares"]),
                     "Brecha ajustada UA (pp)": 100 * (stats["ua_comun"] - stats["ajustada"]),
                     "Cobertura ajuste (%)": 100 * stats["cobertura"]})
    return pd.DataFrame(rows)


def continuity_breakdown(shown):
    """Estados jerárquicos de continuidad: no implican abandono definitivo."""
    rows = []
    for is_ua, label in [(True, "Universidad Autónoma"), (False, "Pares")]:
        group = shown.loc[shown.es_ua.eq(is_ua)]
        n = group.n.sum()
        if not n:
            continue
        program, institution, system = [group[c].sum() for c in ("misma_carrera", "misma_universidad", "sistema")]
        if not 0 <= program <= institution <= system <= n:
            raise ValueError("Los estados de continuidad no son consistentes")
        for order, (state, count) in enumerate([
            ("Misma carrera", program), ("Otra carrera en la institución", institution - program),
            ("Otra institución", system - institution), ("Sin matrícula observada", n - system),
        ]):
            rows.append({"Referencia": label, "Estado": state, "Proporción": count / n,
                         "Inscripciones": int(count), "Orden": order})
    return pd.DataFrame(rows)


def executive_html(year, area, indicator, mode, stats, table, scenarios, generated):
    """HTML imprimible sin dependencias ni scripts; etiquetas siempre escapadas."""
    def pct(x):
        return f"{x:.1%}" if np.isfinite(x) else "Sin datos"
    facts = [("Continuidad UA", pct(stats["ua"])), ("Referencia de pares", pct(stats["pares"])),
             ("Resto del sistema", pct(stats["nacional"])), ("Pares ajustados por áreas", pct(stats["ajustada"])),
             ("UA en las mismas áreas comunes", pct(stats["ua_comun"])),
             ("Cobertura del ajuste", pct(stats["cobertura"]))]
    cards = "".join(f"<li><strong>{escape(k)}</strong>: {escape(v)}</li>" for k, v in facts)
    return f"""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark UA · {int(year)}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#3d3935;max-width:1100px;margin:40px auto;padding:0 24px}}
h1{{border-top:6px solid #e2211c;padding-top:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f4f4f4}}
@media print{{body{{margin:0;font-size:12px}}table{{font-size:10px}}}}</style>
<h1>Universidad Autónoma · Benchmark institucional</h1>
<p>Complemento del proyecto de Analítica de Estudiantes. Comparación descriptiva con datos públicos.</p>
<p><strong>Cohorte:</strong> {int(year)} → {int(year)+1}<br>
<strong>Área:</strong> {escape(area)}<br><strong>Indicador:</strong> {escape(indicator)}<br>
<strong>Selección de pares:</strong> {escape(mode)}<br><strong>Datos generados:</strong> {escape(generated)}</p>
<h2>Lectura ejecutiva</h2><ul>{cards}</ul>
<p>Las referencias excluyen a la UA y ponderan por inscripciones. La referencia ajustada aplica la mezcla
de áreas de la UA a los pares; su comparación utiliza únicamente áreas comunes.</p>
<h2>Universidades comparadas</h2>{table.to_html(index=False, escape=True, na_rep='Sin datos', border=0)}
<h2>Sensibilidad al número de vecinos</h2>{scenarios.round(2).to_html(index=False, escape=True, na_rep='Sin datos', border=0)}
<p>Estos escenarios recalculan los vecinos más cercanos, aunque arriba se haya elegido un grupo manual o cluster.
No son intervalos de confianza ni una prueba causal.</p>
<h2>Alcance</h2><p>El denominador de continuidad son inscripciones de ingreso a carrera con MRUN disponible.
Una persona puede aparecer en varias carreras. No observar matrícula al año siguiente no prueba abandono definitivo.
La selección automática considera tamaño, sedes, áreas, regiones, jornada y modalidad; no utiliza retención.
Las brechas no equivalen a un ranking de calidad ni predicen el riesgo de estudiantes individuales.</p>
<p>Fuentes: matrícula pública Mineduc/SIES; recursos institucionales CNED. Para los detalles de variables,
pesos y códigos institucionales, conserva también la ficha metodológica JSON descargable en la aplicación.</p></html>"""
