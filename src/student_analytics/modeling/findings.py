"""Hallazgos del proyecto, calculados una sola vez desde los artefactos.

POR QUÉ ESTE MÓDULO EXISTE

Los mismos hallazgos se cuentan en tres lugares: el informe metodológico, el
dashboard y los mensajes de commit. Si cada uno recalculara por su cuenta —o
peor, llevara la cifra escrita a mano— divergirían en cuanto cambiara una
base, y la versión escrita a mano se leería como si siguiera siendo cierta.

Aquí cada hallazgo es una función que devuelve un DataFrame desde los parquet
de `data/results`. El informe los consume a través de CSV que genera
`scripts/build_findings.py`; el dashboard llama estas mismas funciones. Ninguna
cifra se escribe dos veces.

Las funciones devuelven `None` cuando falta su insumo, en vez de fallar: el
proyecto se arma por partes y la vista tiene que funcionar con lo que haya.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.retention import summarize_retention
from ..logging_setup import get_logger

log = get_logger(__name__)

UA = "AUTONOMA"
# Cohortes en que falta el codigo de carrera en buena parte de las
# inscripciones de ingreso. Medido sobre la COHORTE, que es el denominador de
# la serie de retencion: 24,5% en 2007 y 17,7% en 2008, contra 2,2% en 2009.
# (Sobre toda la educacion superior las cifras son 28,5% y 20,9%; conviene no
# mezclarlas, que es de donde salio una inconsistencia en la documentacion.)
COHORTES_SIN_CODIGO = (2007, 2008)


def _es_ua(df: pd.DataFrame) -> pd.Series:
    return df.nomb_inst.astype("string").str.contains(UA, na=False)


def _leer(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


# ----------------------------------------------------------------------
# 1. El argumento central del proyecto
# ----------------------------------------------------------------------
def argumento_central(results: Path, ablacion: Path | None = None) -> pd.DataFrame | None:
    """Piso predictivo previo al ingreso frente al baseline conductual.

    Es el hallazgo que justifica el proyecto entero: todo lo que el Estado
    sabe del estudiante ANTES de que entre rinde menos que unas pocas semanas
    de comportamiento observado dentro del curso.

    El piso se toma del modelo COMPLETO de la ablación —expediente de
    admisión más asistencia escolar más socioeconómico— y no del modelo base,
    porque la comparación honesta es contra todo lo disponible, no contra una
    versión recortada.
    """
    nacional = _leer(results / "piso_preingreso_nacional.parquet")
    oulad = _leer(results / "metrics_oulad.parquet")
    # Se busca primero junto a los resultados, para que el paquete de
    # despliegue pueda llevarlo, y luego en la documentacion del repo.
    candidatos = [ablacion] if ablacion else []
    candidatos += [results / "ablacion_piso.csv",
                   results.parent.parent / "documentacion/datos/ablacion_piso.csv"]
    abl = next((pd.read_csv(c) for c in candidatos if c and Path(c).exists()), None)

    filas = []
    if abl is not None and not abl.empty:
        # "D. Base + ambos" es el expediente completo. Si cambian las
        # etiquetas, se toma el de mayor AUC, que es el mismo criterio.
        completo = abl.loc[abl.auc.idxmax()]
        n = None
        if nacional is not None and not nacional.empty:
            univ = nacional[nacional.poblacion.astype(str).str.contains("universidad", na=False)]
            if not univ.empty:
                n = int(univ.iloc[0].n)
        filas.append({
            "momento": "Antes de entrar a la universidad",
            "detalle": "Expediente completo: PAES, NEM, ranking, colegio, asistencia de "
                       "enseñanza media, decil de ingreso, becas",
            "auc": float(completo.auc),
            "n": n,
            "anticipacion": "Antes del primer día de clases",
        })
    if oulad is None and not filas:
        return None
    if oulad is not None and not oulad.empty:
        temprana = oulad.loc[oulad.week.le(4)].sort_values("week")
        if not temprana.empty:
            f = temprana.iloc[-1]
            filas.append({
                "momento": f"Semana {int(f.week)} del curso",
                "detalle": "Solo comportamiento observado en la plataforma. "
                           "Sin asistencia, sin datos previos al ingreso",
                "auc": float(f.auc),
                "n": int(f.get("n_test", 0)) if pd.notna(f.get("n_test")) else None,
                "anticipacion": f"Quedan {f.weeks_remaining:.0f} semanas para actuar",
            })
    return pd.DataFrame(filas) if filas else None


# ----------------------------------------------------------------------
# 2. Retención: la serie larga y la brecha entre sedes
# ----------------------------------------------------------------------
def serie_retencion(results: Path) -> pd.DataFrame | None:
    """Continuidad de la UA frente al resto del sistema, cohorte a cohorte.

    La referencia del sistema EXCLUYE a la UA, para que una institución
    grande no atenúe su propia comparación.
    """
    data = _leer(results / "retention_universities.parquet")
    if data is None or data.empty:
        return None
    ua, resto = data[_es_ua(data)], data[~_es_ua(data)]
    if ua.empty:
        return None
    a = summarize_retention(ua, ["cohorte"])
    b = summarize_retention(resto, ["cohorte"])
    out = a.merge(b, on="cohorte", suffixes=("_ua", "_sistema"))
    out["brecha_carrera_pp"] = 100 * (out.retencion_misma_carrera_ua
                                      - out.retencion_misma_carrera_sistema)
    out["brecha_universidad_pp"] = 100 * (out.retencion_misma_universidad_ua
                                          - out.retencion_misma_universidad_sistema)
    # Sin codigo de carrera la continuidad DE CARRERA no puede calzar aunque
    # la persona haya seguido; se marca para que el grafico la corte.
    out["carrera_comparable"] = ~out.cohorte.isin(COHORTES_SIN_CODIGO)
    if "cobertura_cod_carrera_ua" in out:
        out["cobertura_cod_carrera"] = out.cobertura_cod_carrera_ua
    return out


def retencion_por_sede(results: Path, cohorte: int | None = None,
                       area: str | None = None) -> pd.DataFrame | None:
    """Brecha entre sedes de la UA: el hallazgo que cambia el diseño del modelo."""
    data = _leer(results / "retention_universities.parquet")
    if data is None or data.empty:
        return None
    ua = data[_es_ua(data)]
    if ua.empty:
        return None
    cohorte = int(cohorte if cohorte is not None else ua.cohorte.max())
    ua = ua[ua.cohorte.eq(cohorte)]
    if area:
        ua = ua[ua.area_conocimiento.eq(area)]
    if ua.empty:
        return None
    out = summarize_retention(ua, ["nomb_sede"]).sort_values(
        "retencion_misma_carrera", ascending=False)
    out.insert(0, "cohorte", cohorte)
    return out


# ----------------------------------------------------------------------
# 3. Selectividad de admisión
# ----------------------------------------------------------------------
def posicion_selectividad(profiles: pd.DataFrame) -> pd.DataFrame | None:
    """Dónde queda la UA en selectividad, cohorte a cohorte.

    Se informa el percentil nacional además del puntaje porque **el puntaje
    crudo no cruza el cambio de escala de 2023**: la PDT iba de 150 a 850 y
    la PAES va de 100 a 1000.
    """
    if "paes_percentil_promedio" not in profiles.columns:
        return None
    d = profiles.dropna(subset=["paes_percentil_promedio"])
    if d.empty:
        return None
    filas = []
    for anio, g in d.groupby("cohorte"):
        u = g[_es_ua(g)]
        if u.empty:
            continue
        u = u.iloc[0]
        filas.append({
            "cohorte": int(anio),
            "instrumento": u.get("paes_instrumento", "—"),
            "ua_puntaje": float(u.paes_promedio) if pd.notna(u.paes_promedio) else np.nan,
            "sistema_puntaje_medio": float(g.paes_promedio.mean()),
            "ua_percentil": float(u.paes_percentil_promedio),
            "sistema_percentil_mediano": float(g.paes_percentil_promedio.median()),
            "posicion_ua": float((g.paes_percentil_promedio
                                  < u.paes_percentil_promedio).mean()),
            "universidades": int(len(g)),
            "ua_cobertura": float(u.paes_cobertura) if pd.notna(u.paes_cobertura) else np.nan,
        })
    return pd.DataFrame(filas) if filas else None


# ----------------------------------------------------------------------
# 4. Los dos indicadores de titulación no miden lo mismo
# ----------------------------------------------------------------------
CONTRASTE = {
    "titulacion_oportuna": "Transversal · se tituló dentro del plazo nominal",
    "titulacion_cohorte_carrera": "Por cohorte · se tituló de la carrera que empezó",
    "titulacion_cohorte_universidad": "Por cohorte · se tituló en su universidad",
    "titulacion_cohorte_sistema": "Por cohorte · se tituló en alguna universidad",
}
REFERENCIAS = {
    "paes_percentil_promedio": "Selectividad de admisión (percentil)",
    "acreditacion_anios": "Años de acreditación institucional",
}


def contraste_titulacion(profiles: pd.DataFrame,
                         cohorte: int | None = None) -> pd.DataFrame | None:
    """Correlación de cada indicador de titulación con selectividad y acreditación.

    El resultado es el hallazgo: los dos indicadores apuntan en direcciones
    OPUESTAS sobre las mismas universidades. La versión transversal mide
    velocidad entre quienes se titularon y la manda la mezcla de carreras; la
    de cohorte mide cuántos llegan, y se comporta como un resultado.
    """
    if profiles.empty:
        return None
    cohorte = int(cohorte if cohorte is not None else profiles.cohorte.max())
    d = profiles[profiles.cohorte.eq(cohorte)]
    filas = []
    for var, etiqueta in CONTRASTE.items():
        if var not in d.columns:
            continue
        fila = {"indicador": etiqueta, "variable": var,
                "tipo": "Transversal" if var == "titulacion_oportuna" else "Por cohorte"}
        util = False
        for ref, ref_etiq in REFERENCIAS.items():
            if ref not in d.columns:
                continue
            s = d.dropna(subset=[var, ref])
            if len(s) < 15:
                continue
            fila[ref] = float(s[var].corr(s[ref]))
            fila[f"n_{ref}"] = int(len(s))
            util = True
        if util:
            filas.append(fila)
    if not filas:
        return None
    out = pd.DataFrame(filas)
    out.attrs["cohorte"] = cohorte
    return out


# ----------------------------------------------------------------------
# 5. Las trampas de datos que cambiaron un resultado
# ----------------------------------------------------------------------
def trampas_datos(results: Path, profiles: pd.DataFrame | None) -> pd.DataFrame:
    """Cada trampa encontrada, con el efecto medido de no corregirla.

    Van juntas y con número porque son la parte del trabajo que no se ve en
    ningún gráfico y es la que sostiene todo lo demás.
    """
    filas = [
        {"fuente": "PAES / PDT", "trampa": "El valor 0 significa «no rindió», no «obtuvo cero»",
         "efecto": "Tratarlo como puntaje hunde todas las medias institucionales",
         "estado": "Corregido"},
        {"fuente": "PAES / PDT", "trampa": "Cambio de escala en 2023 (150–850 → 100–1000)",
         "efecto": "El promedio institucional salta ~90 puntos entre 2022 y 2023 sin cambio real",
         "estado": "Corregido con percentil nacional"},
        {"fuente": "Admisión", "trampa": "La PSU (2004–2020) no está publicada",
         "efecto": "14 de 19 cohortes no pueden medir selectividad",
         "estado": "Límite permanente"},
        {"fuente": "Matrícula", "trampa": "Falta cod_carrera en 24,5% de la cohorte 2007 y 17,7% de la 2008",
         "efecto": "La continuidad de carrera aparece en 53,9% en vez de ~72% sin que nadie deserte",
         "estado": "Detectado y advertido"},
        {"fuente": "Matrícula", "trampa": "anio_ing_carr_ori usa 9999 y 9995 como centinela",
         "efecto": "Cohortes contaminadas con años imposibles",
         "estado": "Corregido"},
        {"fuente": "Titulados", "trampa": "anio_ing_carr_ori usa 1900 como centinela, en 10,06% de los casos",
         "efecto": "La duración media pasa de 5 a 17,4 años, con máximos de 124",
         "estado": "Corregido"},
        {"fuente": "Titulados", "trampa": "Los planes de continuidad reconocen estudios previos",
         "efecto": "Sobreduración 3,52 años contra 0,58 del plan regular",
         "estado": "Corregido"},
        {"fuente": "Titulados", "trampa": "La duración nominal viene en SEMESTRES",
         "efecto": "Sin dividir por dos, toda sobreduración sale negativa por cinco años",
         "estado": "Corregido"},
        {"fuente": "Longitudinal", "trampa": "La censura por derecha también sesga por institución",
         "efecto": "La UA aparecía primera del sistema en 2018 con 70,4% y percentil 98",
         "estado": "Corregido con piso de cobertura"},
        {"fuente": "Longitudinal", "trampa": "Filtrar por año después de agregar por clave",
         "efecto": "Un título previo anulaba al estudiante entero; aparecía sistema < universidad",
         "estado": "Corregido"},
        {"fuente": "Matrícula", "trampa": "valor_arancel mezcla pesos y UF en la misma columna",
         "efecto": "7.638 filas en UF (mediana 215) junto a 145.862 en pesos (mediana 4.986.900)",
         "estado": "Corregido"},
        {"fuente": "Becas", "trampa": "La base de asignaciones no contiene el CAE",
         "efecto": "La variable salía constante en cero; la administra la Comisión Ingresa",
         "estado": "Reemplazado por FSCU y beca de arancel"},
        {"fuente": "CNED", "trampa": "«Tradicional» marca pertenencia al CRUCH, no antigüedad",
         "efecto": "Desde la Ley 21.091 el CRUCH admite privadas; DP, UAH y U. Andes entraron en 2019",
         "estado": "Corregido"},
        {"fuente": "Clustering", "trampa": "La silueta elige siempre la partición que aísla atípicos",
         "efecto": "k=2 dejaba al 92% de las universidades —la UA incluida— en un solo grupo",
         "estado": "Corregido con techo de concentración"},
    ]
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# 6. Qué aportó cada fuente
# ----------------------------------------------------------------------
def aporte_fuentes(results: Path, profiles: pd.DataFrame | None) -> pd.DataFrame:
    """Inventario vivo: años cubiertos y variables que dejó cada fuente."""
    ret = _leer(results / "retention_universities.parquet")
    coh = _leer(results / "cohort_completion.parquet")

    def _rango(d, col="cohorte"):
        if d is None or d.empty:
            return "—"
        return f"{int(d[col].min())}–{int(d[col].max())}"

    def _cuenta(prefijos):
        if profiles is None:
            return 0
        return sum(1 for c in profiles.columns
                   if any(c.startswith(p) for p in prefijos))

    filas = [
        {"fuente": "Matrícula en Educación Superior (SIES)", "grano": "Individual (MRUN)",
         "cobertura": "2007–2026", "aporta": "Cohortes de ingreso, continuidad, perfil de oferta",
         "variables": _cuenta(("area::", "region::", "jornada::", "modalidad::",
                               "cohorte_", "arancel", "sedes"))},
        {"fuente": "Titulados en Educación Superior", "grano": "Individual (MRUN)",
         "cobertura": "2007–2025", "aporta": "Titulación transversal y por cohorte",
         "variables": _cuenta(("titulacion_", "titulados_"))},
        {"fuente": "Pruebas de admisión (PDT y PAES)", "grano": "Individual (MRUN)",
         "cobertura": "2021–2026", "aporta": "Selectividad de admisión",
         "variables": _cuenta(("paes_", "nem_", "ranking_"))},
        {"fuente": "CNED INDICES institucional", "grano": "Institución × año",
         "cobertura": "2005–2025", "aporta": "Cuerpo docente, infraestructura, acreditación",
         "variables": _cuenta(("docentes_", "share_", "m2_", "pc_", "ejemplares_",
                               "acreditacion_", "anio_creacion", "cruch"))},
        {"fuente": "Asistencia escolar, SEP y becas", "grano": "Individual (MRUN)",
         "cobertura": "2022–2024", "aporta": "Piso predictivo previo al ingreso",
         "variables": 0},
        {"fuente": "OULAD (Open University)", "grano": "Estudiante × módulo × semana",
         "cobertura": "28.785 estudiantes", "aporta": "Validar la anticipación con conducta real",
         "variables": 0},
    ]
    out = pd.DataFrame(filas)
    out.attrs["retencion"] = _rango(ret)
    out.attrs["cohortes"] = _rango(coh)
    return out
