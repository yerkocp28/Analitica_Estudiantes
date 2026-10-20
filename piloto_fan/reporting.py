"""Informe agregado exportable, sin identificadores ni predicciones individuales."""
from html import escape

import pandas as pd
from .dashboard import number, percent


def executive_html(report: dict) -> str:
    parts = ["<!doctype html><html lang='es'><meta charset='utf-8'>",
             "<title>Piloto FAN · Informe de resultados</title><style>",
             "body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17202a}",
             "h1,h2{color:#a71930}table{border-collapse:collapse;width:100%;font-size:13px}",
             "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}li{margin:8px 0}",
             "@media print{body{margin:0}h2{break-after:avoid}table{break-inside:avoid}}</style>",
             "<h1>Facultad de Administración y Negocios · Piloto de trayectorias</h1>",
             f"<p>Ejecución: {escape(report['run_id'])} · {escape(report['created_utc'])}</p>",
             "<p>Informe general sin filtros. Evaluación retrospectiva con datos públicos reales de la UA. Catálogo documentado, pendiente de ratificación académica.</p>"]
    if report["config"].get("faculty_catalog"):
        parts.append("<h2>Catálogo de carreras incluido</h2>")
        parts.append(pd.DataFrame([{"Carrera": k, "Evidencia": v} for k, v in report["config"]["faculty_catalog"].items()]).to_html(index=False, escape=True))
    for key, case in report["cases"].items():
        parts.append(f"<h2>{escape(key.title())}: {escape(case['status'])}</h2>")
        if "test" not in case:
            parts.append(f"<p>{escape(case.get('reason', 'Sin evaluación suficiente.'))}</p>")
            continue
        parts.append(f"<p>{escape(case['features']['target'])}. Modelo: {escape(case['winner'])}. "
                     f"Validación: {case['validation_year']}; test: {case['test_year']}.</p>")
        rows = []
        for label, result in [("Modelo seleccionado", case["test"]), ("Regla histórica por carrera", case["historical_rule"])]:
            rows.append({"Referencia": label, "Casos": number(result["n"]), "AUC": number(result["auc"], 3),
                         "Captura al 20%": percent(result['capture_20']), "Brier": number(result["brier"], 4)})
        parts.append(pd.DataFrame(rows).to_html(index=False, escape=True))
        parts.append("<p>El desempeño no mide el efecto de contactar estudiantes ni demuestra reducción del abandono.</p>")
    segments = report["segments"]
    parts.append("<h2>Segmentación</h2>")
    if "winner" in segments:
        parts.append(f"<p>{escape(segments['winner'])}, {segments['k']} grupos; "
                     f"silueta {number(segments['silhouette'], 3)}. Perfiles exploratorios, no diagnósticos individuales.</p>")
    parts.append("<h2>Condiciones de interpretación</h2><ul>")
    parts.extend(f"<li>{escape(line)}</li>" for line in report["assumptions"])
    parts.append("</ul><h2>Validación institucional pendiente</h2><p>Confirmar carreras y sedes, "
                 "acordar casos de uso y evaluar la herramienta con las direcciones. "
                 "El acceso VRA y las alertas operativas requieren una etapa institucional posterior.</p></html>")
    return "".join(parts)
