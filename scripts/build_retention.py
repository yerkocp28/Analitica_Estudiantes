"""Genera agregados públicos para Streamlit: python scripts/build_retention.py

Con la matrícula histórica descargada (`scripts/download_matricula.py`) esto
produce **una transición por cada par de años consecutivos** —2007→2008 hasta
la última disponible— en vez de las dos que daban los tres años sueltos.

Lee la capa slim en parquet, no los CSV crudos: son ~40 MB por año en vez de
900 MB, y la máquina no tiene RAM para lo segundo. Si la capa slim no existe
cae de vuelta a los CSV, para no romper una instalación a medio armar.

El seguimiento del año siguiente usa `seguimiento_YYYY.parquet`, que NO está
filtrado por tipo de institución: la retención "sistema" significa seguir en
educación superior, así que quien se cambia a un CFT o a un IP no es una
deserción.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from student_analytics.ingestion.retention import aggregate_retention  # noqa: E402

SLIM = ROOT / "data" / "interim" / "matricula"
CRUDO = ROOT / "data" / "external" / "mineduc"
COLS = ["mrun", "anio_ing_carr_ori", "nivel_global", "tipo_inst_1", "cod_inst",
        "nomb_inst", "nomb_sede", "cod_carrera", "nomb_carrera", "area_conocimiento"]
CLAVES = ["mrun", "cod_inst", "cod_carrera"]


def _anios_slim() -> dict[int, Path]:
    return {int(p.stem.split("_")[-1]): p for p in sorted(SLIM.glob("matricula_*.parquet"))}


def _anios_crudos() -> dict[int, Path]:
    salida = {}
    for carpeta in sorted(CRUDO.glob("matricula_*")):
        csvs = sorted(carpeta.rglob("*.csv"))
        if csvs:
            salida[int(carpeta.name.split("_")[-1])] = max(
                csvs, key=lambda p: p.stat().st_size)
    return salida


def _cohorte(fuente: Path, anio: int) -> pd.DataFrame:
    if fuente.suffix == ".parquet":
        d = pd.read_parquet(fuente, columns=COLS)
        return d.loc[d.anio_ing_carr_ori.eq(anio)]
    dtype = {c: "string" for c in COLS if c != "anio_ing_carr_ori"}
    trozos = pd.read_csv(fuente, sep=";", encoding="utf-8", usecols=COLS,
                         dtype=dtype, chunksize=200_000, low_memory=False)
    return pd.concat([t.loc[t.anio_ing_carr_ori.eq(anio)
                            & t.nivel_global.eq("Pregrado")
                            & t.tipo_inst_1.eq("Universidades")] for t in trozos])


def _siguiente(anio: int, crudos: dict[int, Path]) -> pd.DataFrame:
    """Quiénes están matriculados el año siguiente, en toda la educación superior."""
    indice = SLIM / f"seguimiento_{anio}.parquet"
    if indice.exists():
        return pd.read_parquet(indice, columns=CLAVES)
    if anio in crudos:
        return pd.read_csv(crudos[anio], sep=";", encoding="utf-8",
                           usecols=CLAVES, dtype={c: "string" for c in CLAVES})
    d = pd.read_parquet(SLIM / f"matricula_{anio}.parquet", columns=CLAVES)
    print(f"    aviso: {anio} sin índice de seguimiento; "
          f"la retención de sistema queda acotada a universidades")
    return d


def main() -> None:
    slim, crudos = _anios_slim(), _anios_crudos()
    fuentes = slim or crudos
    if not fuentes:
        raise SystemExit("Faltan datos de matrícula. Corre scripts/download_matricula.py")
    print(f"Matrícula disponible: {min(fuentes)}–{max(fuentes)} ({len(fuentes)} años)"
          f"{'  [slim]' if slim else '  [CSV crudo]'}")

    resultados = []
    for anio in sorted(fuentes):
        if anio + 1 not in fuentes and not (SLIM / f"seguimiento_{anio + 1}.parquet").exists():
            continue
        print(f"Procesando cohorte {anio} -> {anio + 1}", flush=True)
        actual = _cohorte(fuentes[anio], anio)
        if actual.empty:
            print("    sin ingresantes, se omite")
            continue
        resultados.append(aggregate_retention(actual, _siguiente(anio + 1, crudos), anio))

    if not resultados:
        raise SystemExit("Se necesitan al menos dos años consecutivos de matrícula.")
    final = pd.concat(resultados, ignore_index=True)

    # La continuidad de CARRERA depende de que exista `cod_carrera` en ambos
    # anios. Donde falta mucho, la serie cae sin que nadie haya desertado.
    if "con_cod_carrera" in final:
        cob = (final.groupby("cohorte").con_cod_carrera.sum()
               / final.groupby("cohorte").n.sum())
        malas = cob.loc[cob.lt(0.95)]
        if len(malas):
            print("\nAVISO: cobertura baja de cod_carrera en estas cohortes. La "
                  "continuidad de CARRERA no es comparable ahi;\n       usa el nivel "
                  "de universidad o de sistema, que no dependen de ese campo.")
            for anio, v in malas.items():
                print(f"       cohorte {int(anio)}: {v:.1%} de las inscripciones con código")

    out = ROOT / "data" / "results" / "retention_universities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out, index=False)
    print(f"\n{len(resultados)} transiciones guardadas en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
