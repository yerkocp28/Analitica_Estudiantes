"""Contrato de los agregados institucionales que recibe la aplicación web."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CUTS = {
    "por_periodo": ["year", "period"],
    "por_carrera": ["career_code", "career"],
    "por_campus": ["campus_code", "campus"],
    "por_carrera_periodo": ["career_code", "career", "period"],
    "por_asignatura": ["course_code", "course_name"],
}
NUMERIC_MEASURES = ["personas_con_nota", "con_nota", "nota_media", "bajo4", "cobertura_nota"]
MEASURES = ["personas", "registros", *NUMERIC_MEASURES]
FILES = tuple(f"{name}.parquet" for name in CUTS) + ("meta.json",)


def validate_table(table: pd.DataFrame, name: str, minimum: int) -> None:
    if type(minimum) is not int or minimum < 10:
        raise ValueError("El mínimo de publicación debe ser un entero de al menos 10 personas.")
    if set(table) != set(CUTS[name] + MEASURES):
        raise ValueError(f"Columnas no autorizadas o ausentes en {name}.")
    if table.duplicated(CUTS[name]).any():
        raise ValueError(f"Celdas duplicadas en {name}.")
    if not table.personas.ge(minimum).all() or not table.registros.ge(table.personas).all():
        raise ValueError(f"Conteos inválidos o celdas pequeñas en {name}.")
    hidden = table[NUMERIC_MEASURES].isna().all(axis=1)
    visible = table.loc[~hidden]
    valid = (
        visible[NUMERIC_MEASURES].notna().all(axis=1)
        & visible.personas_con_nota.between(minimum, visible.personas)
        & visible.con_nota.between(visible.personas_con_nota, visible.registros)
        & visible.nota_media.between(1, 7)
        & visible.bajo4.between(0, 1)
        & visible.cobertura_nota.between(0, 1)
        & (visible.cobertura_nota - visible.con_nota / visible.registros).abs().le(.000051)
    )
    if not valid.all():
        raise ValueError(f"Estadísticas numéricas inválidas o insuficientes personas con nota en {name}.")
    for col in ["personas", "registros", "personas_con_nota", "con_nota"]:
        if not table[col].dropna().mod(1).eq(0).all():
            raise ValueError(f"Conteos no enteros en {name}.")


def load_publication(folder: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Rechaza archivos adicionales, columnas individuales y huellas alteradas."""
    folder = Path(folder)
    if {p.name for p in folder.iterdir()} != set(FILES):
        raise ValueError("La carpeta institucional debe contener solo los cinco agregados y meta.json.")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    tables = {}
    for name in CUTS:
        filename = name + ".parquet"
        path = folder / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("archivos_sha256", {}).get(filename):
            raise ValueError(f"Huella institucional ausente o alterada: {filename}.")
        table = pd.read_parquet(path)
        validate_table(table, name, meta.get("minimo_personas_celda"))
        tables[name] = table
    return tables, meta
