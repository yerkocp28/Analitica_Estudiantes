"""Validación de adaptadores contra `config/data_contracts.yml`.

POR QUÉ EXISTE

Todo el pipeline aguas abajo —marts, features, modelos, tablero— consume un
diccionario de DataFrames en esquema canónico, no una fuente. Hoy ese
diccionario lo producen el generador sintético y el adaptador de OULAD; mañana
lo producirá uno de Banner y otro de Canvas.

Sin una verificación, un adaptador nuevo puede devolver un esquema **parecido
pero mal** —otra columna clave, otra semana base, un grano que no es único— y
eso no falla: se propaga. Un grano no único abre los joins en silencio y el
error aparece semanas después como "resultados raros".

Escribir este validador ya encontró dos derivas reales en el propio contrato:
el nombre `canvas_weekly` que ningún adaptador usaba, y un grano semanal sin
`term_id` que producía 295.884 duplicados.

QUÉ ES ERROR Y QUÉ ES HUECO

- **Error**: el adaptador dice algo falso sobre sus datos. Grano no único,
  clave nula, tipo incompatible, dataset desconocido. Rompe el pipeline y la
  validación falla.
- **Hueco**: el adaptador no entrega un campo que el contrato declara. No es un
  error —OULAD no tiene asistencia y nunca la tendrá— pero se reporta, y el
  test lo fija por adaptador para que no crezca sin que nadie lo note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ..logging_setup import get_logger

log = get_logger(__name__)

CONTRATO = Path(__file__).resolve().parents[3] / "config" / "data_contracts.yml"

# Tipo declarado -> comprobación sobre la serie. Deliberadamente laxo con los
# enteros: un conteo puede llegar como float si trae ausentes, y eso es
# correcto —ausencia no es cero— así que no debe fallar la validación.
VERIFICADORES = {
    "str": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "int": lambda s: pd.api.types.is_numeric_dtype(s),
    "float": lambda s: pd.api.types.is_numeric_dtype(s),
    "bool": lambda s: pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s),
    "date": lambda s: (pd.api.types.is_datetime64_any_dtype(s)
                       or pd.api.types.is_string_dtype(s)
                       or pd.api.types.is_object_dtype(s)),
}


@dataclass
class Informe:
    """Resultado de validar un adaptador."""

    adaptador: str
    errores: list[str] = field(default_factory=list)
    huecos: dict[str, list[str]] = field(default_factory=dict)
    ausentes: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    filas: dict[str, int] = field(default_factory=dict)

    @property
    def valido(self) -> bool:
        return not self.errores

    def resumen(self) -> str:
        partes = [f"{self.adaptador}: {'OK' if self.valido else 'FALLA'}",
                  f"{len(self.filas)} datasets, {sum(self.filas.values()):,} filas"]
        if self.ausentes:
            partes.append(f"sin entregar: {', '.join(sorted(self.ausentes))}")
        if self.extra:
            partes.append(f"fuera del contrato: {', '.join(sorted(self.extra))}")
        campos = sum(len(v) for v in self.huecos.values())
        if campos:
            partes.append(f"{campos} campos declarados sin dato")
        return " | ".join(partes)


def cargar_contrato(ruta: str | Path | None = None) -> dict:
    """Los datasets declarados, tal como los define el YAML."""
    ruta = Path(ruta) if ruta else CONTRATO
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))["datasets"]


def grano(declarado) -> list[str]:
    """El grano siempre como lista, venga como escalar o como lista."""
    return [declarado] if isinstance(declarado, str) else list(declarado)


def validar_dataset(nombre: str, df: pd.DataFrame, spec: dict) -> tuple[list[str], list[str]]:
    """Errores y huecos de un dataset. Ver el módulo para la distinción."""
    errores, huecos = [], []

    claves = grano(spec.get("grain", []))
    faltan_claves = [c for c in claves if c not in df.columns]
    if faltan_claves:
        errores.append(f"{nombre}: faltan columnas del grano {faltan_claves}")
    elif claves:
        # El grano es la propiedad mas importante: si no es unico, cualquier
        # join aguas abajo multiplica filas sin avisar.
        duplicados = int(df.duplicated(subset=claves).sum())
        if duplicados:
            errores.append(f"{nombre}: el grano {claves} no es único "
                           f"({duplicados:,} filas duplicadas de {len(df):,})")
        nulos = int(df[claves].isna().any(axis=1).sum())
        if nulos:
            errores.append(f"{nombre}: {nulos:,} filas con clave nula en {claves}")

    for campo, meta in (spec.get("fields") or {}).items():
        if campo not in df.columns:
            huecos.append(campo)
            continue
        tipo = (meta or {}).get("type")
        verificador = VERIFICADORES.get(tipo)
        if verificador and not df[campo].empty and not verificador(df[campo]):
            errores.append(f"{nombre}.{campo}: se declaró {tipo} y llegó "
                           f"{df[campo].dtype}")
    return errores, huecos


def validar_adaptador(datos: dict[str, pd.DataFrame], adaptador: str = "adaptador",
                      contrato: dict | None = None) -> Informe:
    """Valida el diccionario que entrega un adaptador contra el contrato.

    No exige que estén todos los datasets: un adaptador puede entregar un
    subconjunto y eso se reporta en `ausentes`. Lo que sí exige es que **lo
    que entrega sea cierto**.
    """
    contrato = contrato if contrato is not None else cargar_contrato()
    informe = Informe(adaptador=adaptador)

    for nombre in sorted(set(datos) - set(contrato)):
        informe.extra.append(nombre)
    informe.ausentes = sorted(set(contrato) - set(datos))

    for nombre, spec in contrato.items():
        df = datos.get(nombre)
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            informe.errores.append(f"{nombre}: se esperaba un DataFrame y llegó "
                                   f"{type(df).__name__}")
            continue
        informe.filas[nombre] = len(df)
        errores, huecos = validar_dataset(nombre, df, spec)
        informe.errores += errores
        if huecos:
            informe.huecos[nombre] = huecos

    log.info("%s", informe.resumen())
    return informe
