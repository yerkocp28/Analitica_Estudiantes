"""Segmentación de ingresantes de la Universidad Autónoma.

Es el entregable del mes 3, y se construye sobre estudiantes **reales**: las
19 cohortes de ingreso de la UA están en datos abiertos, con su resultado
observado. No hace falta Banner para esto.

TRES REGLAS QUE VIENEN DE ERRORES YA COMETIDOS EN ESTE PROYECTO

1. **El resultado no define el segmento.** Solo entran variables conocidas al
   matricularse. Si la continuidad entrara, describir la retención de cada
   segmento sería circular, y el segmento no serviría para decidir nada el día
   que llega una cohorte nueva —que es el único día en que sirve.

2. **La silueta sola elige siempre la partición que aísla atípicos.** En el
   benchmark institucional eso dejaba al 92% de las universidades en un solo
   grupo: limpio de medir, inútil para actuar. Aquí hay techo de concentración
   y tamaño mínimo, y la silueta decide solo entre las soluciones admisibles.

3. **Un segmento que no cambia lo que hace una dirección de carrera es un
   segmento inútil**, por buena que sea su silueta. Por eso `perfilar` describe
   cada grupo por lo que implica —dónde está, qué estudia, con qué puntaje
   entra, cuánta continuidad tiene— y no por su centroide en el espacio
   transformado.

SOBRE LA COMPARACIÓN DE ALGORITMOS

El documento maestro exige al menos tres alternativas. Se comparan tres
familias genuinamente distintas, no tres variantes de lo mismo: centroides
(K-means), modelo probabilístico con asignación blanda (mezcla gaussiana) y
divisivo jerárquico (K-means bisectriz). Aglomerativo queda fuera por costo
cuadrático: con 98.812 filas no es viable y muestrear hasta que lo sea
cambiaría lo que se compara.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from ..logging_setup import get_logger

log = get_logger(__name__)

# Variables por bloque. Las categoricas se expanden a indicadoras y las
# numericas se estandarizan; cada bloque se normaliza por su varianza total.
BLOQUES_CATEGORICOS = {
    "estructura": ["area_conocimiento", "jornada", "modalidad"],
    # `territorio` existe como bloque posible pero NO esta activo por omision:
    # ver la explicacion en config/segmentation.yml. La sede es descriptor.
    "territorio": ["nomb_sede", "region_sede"],
}
BLOQUES_NUMERICOS = {
    # PERCENTILES, NO PUNTAJES CRUDOS. El NEM y el ranking cambiaron de escala
    # entre la PDT y la PAES igual que el puntaje de admision: el maximo pasa
    # de 824 a 994 y la media de 560 a 716 entre 2022 y 2023. Con el crudo, la
    # segmentacion agrupa por CALENDARIO creyendo que agrupa por perfil.
    "academico": ["percentil_admision", "percentil_nem", "percentil_ranking"],
    "estructura": ["duracion_nominal_anios"],
}

# Jamas pueden entrar al espacio de segmentacion: son el resultado.
PROHIBIDAS = {"sigue_carrera", "sigue_universidad", "sigue_sistema"}


@dataclass
class ModeloSegmentos:
    etiquetas: pd.Series
    algoritmo: str
    k: int
    silueta: float
    diagnostico: pd.DataFrame
    columnas: dict[str, list[str]]


def _bloque_categorico(d: pd.DataFrame, columnas: list[str]) -> tuple[np.ndarray, list[str]]:
    # Una variable constante no separa a nadie y solo diluye el bloque: en la
    # UA, `modalidad` es 100% Presencial. Se descarta en vez de arrastrarla.
    presentes = [c for c in columnas
                 if c in d.columns and d[c].astype("string").nunique(dropna=False) > 1]
    if not presentes:
        return np.empty((len(d), 0)), []
    ind = pd.get_dummies(d[presentes].astype("string").fillna("Sin información"),
                         prefix=presentes, prefix_sep="::", dtype=float)
    return ind.to_numpy(), list(ind.columns)


def _bloque_numerico(d: pd.DataFrame, columnas: list[str]) -> tuple[np.ndarray, list[str]]:
    presentes = [c for c in columnas
                 if c in d.columns and d[c].notna().any()]
    if not presentes:
        return np.empty((len(d), 0)), []
    x = d[presentes].astype(float)
    # Ausente se imputa con la mediana: deja a esa persona en el centro en vez
    # de en un extremo inventado. Ausencia no es cero.
    x = x.fillna(x.median()).to_numpy()
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), presentes


def matriz_segmentacion(d: pd.DataFrame,
                        pesos: dict[str, float]) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Espacio de segmentación: bloques con igual peso, sin el resultado.

    Cada bloque se divide por la raíz de su varianza total, de modo que un
    bloque con muchas categorías no domine por tener más columnas.
    """
    filtradas = PROHIBIDAS & set(d.columns)
    if filtradas:
        log.debug("Columnas de resultado excluidas del espacio: %s", sorted(filtradas))

    if not pesos or min(pesos.values()) <= 0 or not np.isclose(sum(pesos.values()), 1):
        raise ValueError("Los pesos de bloques deben ser positivos y sumar 1.")

    piezas, usadas = [], {}
    for bloque, peso in pesos.items():
        partes, nombres = [], []
        x, cols = _bloque_categorico(d, BLOQUES_CATEGORICOS.get(bloque, []))
        if cols:
            partes.append(x)
            nombres += cols
        x, cols = _bloque_numerico(d, BLOQUES_NUMERICOS.get(bloque, []))
        if cols:
            partes.append(x)
            nombres += cols
        if not partes:
            log.warning("Bloque '%s' sin variables con dato; se omite y su peso se reparte",
                        bloque)
            continue
        bloque_x = np.hstack(partes)
        bloque_x = bloque_x - bloque_x.mean(axis=0)
        escala = np.sqrt(np.var(bloque_x, axis=0).sum())
        piezas.append((bloque, bloque_x / (escala if escala > 1e-12 else 1), peso))
        usadas[bloque] = nombres

    if not piezas:
        raise ValueError("Ningún bloque tiene variables disponibles.")
    total = sum(p for _, _, p in piezas)
    matriz = np.hstack([x * np.sqrt(p / total) for _, x, p in piezas])
    return matriz, usadas


def _candidatos(x: np.ndarray, k: int, semilla: int) -> dict:
    """Tres familias distintas, no tres variantes de la misma."""
    return {
        "K-means": KMeans(n_clusters=k, n_init=10, random_state=semilla),
        "Mezcla gaussiana": GaussianMixture(n_components=k, covariance_type="diag",
                                            n_init=3, reg_covar=1e-4,
                                            random_state=semilla),
        "K-means bisectriz": BisectingKMeans(n_clusters=k, random_state=semilla),
    }


def segmentar(d: pd.DataFrame, config: dict) -> ModeloSegmentos:
    """Compara algoritmos y k, y elige entre los admisibles.

    Admisible significa: ningún segmento bajo el mínimo accionable y ninguno
    que concentre más de la fracción configurada. Entre los que cumplen, gana
    la silueta; ante empate, el k menor.
    """
    matriz, columnas = matriz_segmentacion(d, config["block_weights"])
    semilla = config.get("random_seed", 42)

    # Con decenas de miles de filas y varios algoritmos, ajustar sobre una
    # muestra no cambia la estructura y hace viable comparar. La asignacion
    # final se hace sobre todas las filas.
    tope = int(config.get("muestra_ajuste", 20000))
    rng = np.random.default_rng(semilla)
    idx = (rng.choice(len(matriz), tope, replace=False)
           if len(matriz) > tope else np.arange(len(matriz)))
    ajuste = matriz[idx]

    filas, mejores = [], {}
    for k in config["k_values"]:
        for nombre, modelo in _candidatos(ajuste, k, semilla).items():
            try:
                etiquetas = modelo.fit_predict(ajuste)
            except Exception as exc:                      # modelo que no converge
                log.debug("%s k=%d no ajustó: %s", nombre, k, exc)
                continue
            cuentas = pd.Series(etiquetas).value_counts()
            if len(cuentas) != k:
                continue
            concentracion = cuentas.max() / len(etiquetas)
            minimo = cuentas.min() / len(etiquetas) * len(d)
            # La silueta sobre una submuestra: es O(n^2) y aqui solo sirve
            # para comparar candidatos entre si.
            sub = idx[:5000] if len(idx) > 5000 else idx
            silueta = float(silhouette_score(matriz[sub],
                                             modelo.predict(matriz[sub])
                                             if hasattr(modelo, "predict")
                                             else etiquetas[:len(sub)]))
            admisible = (minimo >= config["minimum_segment"]
                         and concentracion <= config["maximum_segment_share"])
            filas.append({"algoritmo": nombre, "k": k, "silueta": round(silueta, 4),
                          "concentracion": round(float(concentracion), 4),
                          "segmento_menor": int(minimo), "admisible": admisible})
            if admisible:
                mejores[(nombre, k)] = (silueta, modelo)

    diagnostico = pd.DataFrame(filas).sort_values(
        ["admisible", "silueta"], ascending=[False, False]).reset_index(drop=True)
    if not mejores:
        raise ValueError(
            "Ninguna partición cumple el tamaño mínimo y el techo de concentración. "
            "Revisa `minimum_segment` y `maximum_segment_share` en config/segmentation.yml.")

    (algoritmo, k), (silueta, modelo) = max(
        mejores.items(), key=lambda kv: (kv[1][0], -kv[0][1]))
    etiquetas = pd.Series(modelo.predict(matriz), index=d.index, name="segmento")
    # Se renumeran por tamanio: el segmento 1 es siempre el mas grande, para
    # que los nombres no cambien de significado entre corridas.
    orden = etiquetas.value_counts().index
    etiquetas = etiquetas.map({v: i + 1 for i, v in enumerate(orden)})
    log.info("Segmentación: %s con k=%d, silueta %.3f", algoritmo, k, silueta)
    return ModeloSegmentos(etiquetas, algoritmo, k, silueta, diagnostico, columnas)


def perfilar(d: pd.DataFrame, etiquetas: pd.Series) -> pd.DataFrame:
    """Describe cada segmento por lo que implica, no por su centroide.

    La continuidad aparece aquí como **descripción**, nunca como insumo: el
    segmento ya se definió sin ella.
    """
    x = d.copy()
    x["segmento"] = etiquetas.to_numpy()
    filas = []
    for seg, g in x.groupby("segmento", sort=True):
        fila = {"segmento": int(seg), "n": len(g), "share": len(g) / len(x)}
        for col, etiqueta in [("nomb_sede", "sede_principal"),
                              ("area_conocimiento", "area_principal"),
                              ("jornada", "jornada_principal")]:
            if col in g.columns and g[col].notna().any():
                moda = g[col].astype("string").mode()
                fila[etiqueta] = moda.iloc[0] if len(moda) else "—"
                fila[f"{etiqueta}_share"] = float(g[col].astype("string")
                                                  .eq(fila[etiqueta]).mean())
        for col in ("percentil_admision", "puntaje_admision", "puntaje_nem",
                    "duracion_nominal_anios", "decil_dfe"):
            if col in g.columns and g[col].notna().any():
                fila[col] = float(g[col].mean())
        for col in ("sigue_carrera", "sigue_universidad", "sigue_sistema"):
            if col in g.columns:
                fila[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        filas.append(fila)
    perfil = pd.DataFrame(filas)
    # Brecha de continuidad frente al conjunto: es lo que dice si el segmento
    # importa para actuar o solo es distinto.
    if "sigue_carrera" in perfil.columns:
        base = pd.to_numeric(x.sigue_carrera, errors="coerce").mean()
        perfil["brecha_carrera_pp"] = 100 * (perfil.sigue_carrera - base)
    return perfil.sort_values("n", ascending=False).reset_index(drop=True)


def estabilidad(d: pd.DataFrame, etiquetas: pd.Series,
                por: str = "cohorte") -> pd.DataFrame:
    """Cómo se reparte cada cohorte entre los segmentos.

    Un segmento que solo existe en una cohorte describe un año, no un tipo de
    estudiante, y no sirve para recibir a la cohorte siguiente.
    """
    x = pd.crosstab(d[por], etiquetas.to_numpy(), normalize="index")
    x.columns = [f"segmento_{c}" for c in x.columns]
    return x.reset_index()
