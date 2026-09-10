"""Seguimiento longitudinal de cohortes: titulación por cohorte de ingreso.

Esta es la versión **B** de titulación, y responde la pregunta que la versión
transversal de `titulados.py` no puede responder: de quienes ingresaron en el
año Y, ¿qué proporción llegó a titularse? La transversal describe a la
promoción que egresa —cuánto se demoró quien sí se tituló—; ésta describe a la
cohorte que entra.

CÓMO SE ENLAZA

El MRUN es un identificador enmascarado pero **estable entre bases y años**.
Se toma la cohorte de ingreso en la matrícula del año Y y se busca a cada
persona en las bases de titulados de Y en adelante. Se mide en tres niveles,
los mismos de la retención, porque la diferencia entre ellos es informativa:

  misma_carrera      se tituló de lo que empezó, en la misma institución;
  misma_universidad  se tituló en la misma institución, de cualquier carrera;
  sistema            se tituló en cualquier institución del país.

La brecha entre el primero y el segundo mide cambio interno de carrera; la
brecha entre el segundo y el tercero mide transferencia a otra universidad.
Sin el tercer nivel, todo traslado se contabiliza como fracaso.

EL PROBLEMA CENTRAL: CENSURA POR DERECHA

Una cohorte sólo puede observarse durante los años que hay de datos después
de ella. La cohorte 2007 tiene 18 años de seguimiento; la 2022, tres. Si se
comparan sin corregir, las cohortes recientes parecen catastróficas por el
puro hecho de ser recientes.

Aquí se corrige exigiendo un **horizonte observable por estudiante**: cada
uno aporta al denominador sólo si `Y + nominal + holgura` cae dentro de los
años con datos. Una carrera de 5 años con holgura 2 necesita datos hasta
Y+7. Quien no alcanza a ser observado ese tiempo no entra en el
denominador, en vez de entrar como si hubiera desertado.

TRAMPAS DE ESTAS BASES (verificadas, no supuestas)

  - `anio_ing_carr_ori` trae 9999 y 9995 como centinelas y ~3,4% de nulos.
    Al definir la cohorte por igualdad con el año, quedan fuera solos.
  - `dur_total_carr` viene en SEMESTRES; la duración real se cuenta en años.
  - Una persona con dos carreras aparece dos veces: el denominador cuenta
    inscripciones, y el nivel `sistema` se deduplica por MRUN.
  - Los planes especiales y de continuidad reconocen estudios previos, así
    que su duración no es comparable con la de un plan regular.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

# Los mismos tres niveles que usa la retención, para que ambas series se lean
# en la misma escala.
NIVELES = {
    "misma_carrera": ["mrun", "cod_inst", "cod_carrera"],
    "misma_universidad": ["mrun", "cod_inst"],
    "sistema": ["mrun"],
}

DIMENSIONES = ["cod_inst", "nomb_inst", "area_conocimiento"]

COHORT_LABELS = {
    "titulacion_cohorte_carrera": "Titulados de su carrera de ingreso",
    "titulacion_cohorte_universidad": "Titulados en su universidad de ingreso",
    "titulacion_cohorte_sistema": "Titulados en cualquier universidad",
    "titulacion_cohorte_n": "Ingresantes con horizonte observable",
    "titulacion_cohorte_anios": "Años medianos hasta el título",
    "titulacion_cohorte_cobertura": "Cohorte alcanzada a observar",
}


def cargar_cohorte(parquet: str | Path, anio: int,
                   solo_regular: bool = True) -> pd.DataFrame:
    """Ingresantes de pregrado universitario del año `anio`.

    La cohorte se define por `anio_ing_carr_ori`, no por `cat_periodo`: quien
    aparece en la matrícula del año Y habiendo ingresado en Y-3 es un alumno
    antiguo, no un ingresante.
    """
    cols = ["mrun", "cod_inst", "nomb_inst", "cod_carrera", "nomb_carrera",
            "area_conocimiento", "anio_ing_carr_ori", "dur_total_carr",
            "tipo_plan_carr", "gen_alu", "nomb_sede"]
    df = pd.read_parquet(parquet, columns=[c for c in cols if c])
    df = df.loc[df.anio_ing_carr_ori.eq(anio)].copy()
    if solo_regular and "tipo_plan_carr" in df:
        df = df.loc[df.tipo_plan_carr.astype("string").eq("Plan Regular")]
    df = df.drop_duplicates()
    # La nominal viene en semestres; todo el resto del cálculo es en años.
    df["nominal_anios"] = df.dur_total_carr / 2
    df["cohorte"] = anio
    return df


def indice_titulacion(rutas: dict[int, str | Path]) -> pd.DataFrame:
    """Tabla (mrun, cod_inst, cod_carrera, anio_titulo) de todos los años.

    Se queda con el PRIMER título de cada combinación: a quien se titula dos
    veces de lo mismo le interesa la primera vez.
    """
    cols = ["mrun", "cod_inst", "cod_carrera", "cat_periodo",
            "nivel_global", "tipo_inst_1"]
    partes = []
    for anio in sorted(rutas):
        df = pd.read_csv(rutas[anio], sep=";", encoding="utf-8", dtype=str,
                         usecols=lambda c: c in cols, low_memory=False)
        df = df.loc[df.nivel_global.eq("Pregrado")
                    & df.tipo_inst_1.eq("Universidades")]
        df = df[["mrun", "cod_inst", "cod_carrera"]].copy()
        df["anio_titulo"] = anio
        partes.append(df.dropna(subset=["mrun"]).drop_duplicates())
        log.info("  titulados %s: %s", anio, f"{len(partes[-1]):,}")
    indice = pd.concat(partes, ignore_index=True)
    return indice.sort_values("anio_titulo").drop_duplicates(
        subset=["mrun", "cod_inst", "cod_carrera"], keep="first")


def seguir_cohorte(cohorte: pd.DataFrame, indice: pd.DataFrame, anio: int,
                   ultimo_anio: int, holgura: int = 2) -> pd.DataFrame:
    """Marca, por ingresante, si se tituló en cada nivel y en qué año.

    `holgura` son los años de gracia sobre la duración nominal que se
    conceden antes de considerar cerrado el caso. Con holgura 2, una carrera
    de 5 años se juzga a los 7.
    """
    out = cohorte.copy()
    out["mrun"] = out.mrun.astype("string")
    out["cod_inst"] = out.cod_inst.astype("string")
    out["cod_carrera"] = out.cod_carrera.astype("string")

    idx = indice.copy()
    for c in ("mrun", "cod_inst", "cod_carrera"):
        idx[c] = idx[c].astype("string")
    idx["anio_titulo"] = pd.to_numeric(idx.anio_titulo, errors="coerce")

    # Los titulos anteriores al ingreso —o del mismo anio— son titulos PREVIOS
    # de otra carrera, no el resultado de esta cohorte. Hay que descartarlos
    # ANTES de tomar el minimo por clave: si se toma el minimo primero y se
    # filtra despues, un titulo viejo anula al estudiante entero y esconde el
    # titulo posterior que si corresponde. Como el nivel `sistema` agrega mas
    # titulos por persona, es el que mas se envenena, y las tasas dejan de
    # estar anidadas: aparece sistema < universidad, que es imposible.
    idx = idx.loc[idx.anio_titulo.ge(anio + 1)]

    for nombre, claves in NIVELES.items():
        # El primer titulo valido por clave: quien cambia de carrera dentro de
        # la misma universidad aporta su titulacion mas temprana.
        primero = (idx.dropna(subset=claves)
                   .groupby(claves, observed=True).anio_titulo.min()
                   .rename(f"anio_{nombre}").reset_index())
        out = out.merge(primero, on=claves, how="left")
        out[f"anios_{nombre}"] = out[f"anio_{nombre}"] - anio
        out[nombre] = out[f"anios_{nombre}"].notna().astype(int)

    # Censura: sólo cuenta en el denominador quien alcanzó a ser observado
    # el tiempo que su propia carrera exige.
    horizonte = anio + out.nominal_anios + holgura
    out["observable"] = (out.nominal_anios.gt(0) & horizonte.le(ultimo_anio)).astype(int)
    out["horizonte"] = horizonte

    # Titulacion dentro del plazo: la nominal mas la holgura, no "alguna vez".
    for nombre in NIVELES:
        dentro = out[f"anios_{nombre}"].le(out.nominal_anios + holgura)
        out[f"{nombre}_en_plazo"] = (dentro & out.observable.eq(1)).fillna(False).astype(int)
    return out


def agregar_por_institucion(seguimiento: pd.DataFrame, minimo: int = 30,
                            min_cobertura: float = 0.90) -> pd.DataFrame:
    """Tasas de titulación por institución y cohorte, sólo sobre observables.

    `min_cobertura` es la fracción de los ingresantes de esa institución que
    debe alcanzar a observarse. Hace falta ADEMÁS del filtro por estudiante
    porque la censura no golpea a todos por igual: cuando quedan pocos años
    de seguimiento, las carreras largas caen primero, y una universidad con
    mucha medicina o ingeniería queda descrita sólo por sus carreras cortas.
    Su tasa entonces sube, pero no porque titule mejor. Sin este piso, la
    misma institución aparenta mejorar varios puntos en la última cohorte y
    empeorar cuando el dato se completa.
    """
    obs = seguimiento.loc[seguimiento.observable.eq(1)].copy()
    for col in DIMENSIONES:
        obs[col] = obs[col].astype("string").fillna("Sin información")

    cobertura = (seguimiento.groupby(["cohorte", "cod_inst"], observed=True)
                 .observable.mean().rename("titulacion_cohorte_cobertura")
                 .reset_index())

    cuentas = {"titulacion_cohorte_n": ("observable", "sum")}
    for nombre in NIVELES:
        cuentas[f"n_{nombre}"] = (f"{nombre}_en_plazo", "sum")
    g = obs.groupby(["cohorte", "cod_inst", "nomb_inst"], observed=True)
    res = g.agg(**cuentas).reset_index().merge(
        cobertura, on=["cohorte", "cod_inst"], how="left")

    sufijo = {"misma_carrera": "carrera", "misma_universidad": "universidad",
              "sistema": "sistema"}
    valido = (res.titulacion_cohorte_n.ge(minimo)
              & res.titulacion_cohorte_cobertura.ge(min_cobertura))
    n = res.titulacion_cohorte_n.where(valido)
    for nombre, corto in sufijo.items():
        res[f"titulacion_cohorte_{corto}"] = res[f"n_{nombre}"].div(n)

    # Años medianos hasta el título, entre quienes efectivamente se titularon
    # en su universidad. Describe la velocidad, no la proporción.
    tit = obs.loc[obs.misma_universidad_en_plazo.eq(1)]
    medianos = (tit.groupby(["cohorte", "cod_inst"], observed=True)
                .anios_misma_universidad.median()
                .rename("titulacion_cohorte_anios").reset_index())
    res = res.merge(medianos, on=["cohorte", "cod_inst"], how="left")
    res.loc[~valido, "titulacion_cohorte_anios"] = np.nan
    return res


def adjuntar_cohorte(perfiles: pd.DataFrame, cohortes: pd.DataFrame,
                     min_universidades: float = 0.80) -> pd.DataFrame:
    """Une la titulación por cohorte al perfil, con UNA cohorte común.

    La cohorte del perfil (2023/2024) todavía no puede observarse, así que se
    trae la última que sí completó su horizonte y se deja explícito de qué año
    viene.

    Esa cohorte tiene que ser **la misma para todas** las universidades. Si se
    tomara por institución la más reciente que tenga dato, la más nueva
    quedaría representada por las pocas que alcanzan a cerrarla —justo las de
    carreras cortas— y el benchmark compararía a unas en un año contra otras
    en otro. `min_universidades` exige que la cohorte elegida tenga tasa
    utilizable en esa fracción del sistema.
    """
    if cohortes.empty:
        return perfiles.copy()
    con_dato = cohortes.dropna(subset=["titulacion_cohorte_universidad"])
    if con_dato.empty:
        return perfiles.copy()
    cobertura = (con_dato.groupby("cohorte", observed=True).size()
                 / cohortes.groupby("cohorte", observed=True).size())
    elegibles = cobertura.loc[cobertura.ge(min_universidades)]
    elegida = int(elegibles.index.max() if len(elegibles) else cobertura.index.max())
    ultima = con_dato.loc[con_dato.cohorte.eq(elegida)].copy()
    ultima = ultima.rename(columns={"cohorte": "titulacion_cohorte_anio"})
    cols = ["cod_inst", "titulacion_cohorte_anio", "titulacion_cohorte_n",
            "titulacion_cohorte_carrera", "titulacion_cohorte_universidad",
            "titulacion_cohorte_sistema", "titulacion_cohorte_anios"]
    out = perfiles.copy()
    out["cod_inst"] = out.cod_inst.astype(str)
    ultima = ultima[[c for c in cols if c in ultima.columns]].copy()
    ultima["cod_inst"] = ultima.cod_inst.astype(str)
    return out.merge(ultima, on="cod_inst", how="left")
