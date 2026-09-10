"""Descarga la matrícula histórica de educación superior (Mineduc, 2007–hoy).

    python scripts/download_matricula.py                 # todo lo que falte
    python scripts/download_matricula.py 2007 2008       # años específicos
    python scripts/download_matricula.py --purgar-csv    # borra el CSV crudo

Fuente: https://datosabiertos.mineduc.cl/matricula-en-educacion-superior/

POR QUÉ HAY UNA CAPA SLIM EN PARQUET

Cada año pesa ~900 MB de CSV y la máquina tiene ~5 GB de RAM libre. Seguir a
una cohorte exige abrir *todos* los años a la vez, así que leer los CSV
completos no es viable. Al terminar cada descarga se escribe un parquet con
las columnas de trayectoria (`COLUMNAS`) filtrado a pregrado universitario:
~40 MB por año, columnar y comprimido. Ese parquet es lo que consume
`build_cohorts.py`; el CSV crudo se conserva por si más adelante hacen falta
otras columnas, y `--purgar-csv` lo elimina si aprieta el disco.

QUÉ HABILITA

  - 19 transiciones de retención (2007→2008 … 2025→2026) en vez de dos;
  - titulación POR COHORTE de ingreso, cruzando MRUN contra la base de
    titulados: qué proporción de los que entraron en el año Y llegó a
    titularse, que es la pregunta que la versión transversal no responde.

El MRUN es un identificador enmascarado pero estable entre bases y años, que
es exactamente lo que hace posible el seguimiento longitudinal.
"""
from __future__ import annotations

import argparse
import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "mineduc"
SLIM = REPO_ROOT / "data" / "interim" / "matricula"
PAGINA = "https://datosabiertos.mineduc.cl/matricula-en-educacion-superior/"
SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# Columnas de trayectoria. Las viejas no siempre las traen todas: se piden con
# un callable, no con una lista, para que falten sin romper la lectura.
COLUMNAS = [
    "cat_periodo", "mrun", "gen_alu", "anio_ing_carr_ori", "sem_ing_carr_ori",
    "tipo_inst_1", "cod_inst", "nomb_inst", "cod_sede", "nomb_sede",
    "cod_carrera", "nomb_carrera", "jornada", "modalidad", "tipo_plan_carr",
    "dur_total_carr", "region_sede", "nivel_global", "nivel_carrera_1",
    "area_conocimiento", "valor_arancel", "formato_valores",
]
NUMERICAS = ["cat_periodo", "anio_ing_carr_ori", "sem_ing_carr_ori",
             "dur_total_carr", "valor_arancel"]
TROZO = 400_000


def _contexto() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def catalogo() -> dict[int, str]:
    """Año -> URL, leído de la página en vez de fijado en el código.

    Las rutas del Mineduc incluyen el mes de publicación y cambian cuando
    republican un año; hardcodearlas las rompe en silencio.
    """
    req = urllib.request.Request(PAGINA, headers=HDRS)
    with urllib.request.urlopen(req, timeout=120, context=_contexto()) as r:
        html = r.read().decode("utf-8", errors="replace")
    urls = sorted(set(re.findall(
        r'href="(https://datosabiertos\.mineduc\.cl/wp-content/uploads/'
        r'[^"]+\.(?:rar|zip|csv|7z))"', html)))
    salida: dict[int, str] = {}
    for u in urls:
        m = re.search(r"(20\d{2})", u.split("/")[-1])
        if m:
            salida[int(m.group(1))] = u
    return salida


def _descargar(url: str, destino: Path) -> None:
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=1800, context=_contexto()) as r, \
            destino.open("wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        hecho = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            hecho += len(chunk)
            if total:
                sys.stdout.write(f"\r    {hecho / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    print()


def compactar(csv: Path, anio: int, destino: Path,
              seguimiento: Path | None = None) -> Path:
    """Escribe los dos parquet de un año en una sola lectura del CSV.

    `destino` es la capa slim: pregrado universitario con las columnas de
    trayectoria, que es la cohorte a analizar.

    `seguimiento` es un índice mucho más chico (MRUN, institución, carrera y
    tipo) **sin filtrar por nivel ni tipo de institución**. Existe porque la
    retención a nivel de sistema significa "sigue en educación superior", no
    "sigue en alguna universidad": si el año siguiente se buscara sólo en el
    parquet slim, quien se cambia a un CFT o a un IP se contaría como
    deserción y la tasa quedaría sesgada hacia abajo.

    El CSV entra en bloques porque un año completo no cabe cómodo en memoria
    junto con lo demás.
    """
    dtype = {c: "string" for c in COLUMNAS if c not in NUMERICAS}
    trozos, indices = [], []
    for bloque in pd.read_csv(csv, sep=";", encoding="utf-8", low_memory=False,
                              usecols=lambda c: c in COLUMNAS, dtype=dtype,
                              chunksize=TROZO, on_bad_lines="warn"):
        for c in NUMERICAS:
            if c in bloque:
                bloque[c] = pd.to_numeric(bloque[c], errors="coerce")
        if seguimiento is not None:
            claves = [c for c in ("mrun", "cod_inst", "cod_carrera",
                                  "tipo_inst_1", "nivel_global") if c in bloque]
            indices.append(bloque[claves].dropna(subset=["mrun"]).drop_duplicates())
        filtro = pd.Series(True, index=bloque.index)
        if "nivel_global" in bloque:
            filtro &= bloque.nivel_global.eq("Pregrado")
        if "tipo_inst_1" in bloque:
            filtro &= bloque.tipo_inst_1.eq("Universidades")
        trozos.append(bloque.loc[filtro])
    df = pd.concat(trozos, ignore_index=True)

    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        print(f"    columnas ausentes en {anio}: {', '.join(faltan)}")
    if "cat_periodo" not in df.columns:
        df["cat_periodo"] = anio          # los años viejos no siempre la traen

    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    print(f"    slim {destino.name:28s} {len(df):>9,} filas  "
          f"{destino.stat().st_size / 1e6:6.1f} MB")

    if seguimiento is not None and indices:
        idx = pd.concat(indices, ignore_index=True).drop_duplicates()
        idx.to_parquet(seguimiento, index=False)
        print(f"    segu {seguimiento.name:28s} {len(idx):>9,} filas  "
              f"{seguimiento.stat().st_size / 1e6:6.1f} MB")
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anios", nargs="*", type=int, help="años a descargar")
    ap.add_argument("--desde", type=int, default=2007)
    ap.add_argument("--hasta", type=int, default=2026)
    ap.add_argument("--purgar-csv", action="store_true",
                    help="borra el CSV crudo tras compactar")
    ap.add_argument("--solo-compactar", action="store_true",
                    help="no descarga: rehace el parquet de lo ya bajado")
    args = ap.parse_args()

    disponibles = catalogo()
    print(f"Publicados por Mineduc: {min(disponibles)}–{max(disponibles)} "
          f"({len(disponibles)} años)")
    pedidos = args.anios or [a for a in sorted(disponibles)
                             if args.desde <= a <= args.hasta]
    SLIM.mkdir(parents=True, exist_ok=True)

    for anio in pedidos:
        carpeta = DEST / f"matricula_{anio}"
        parquet = SLIM / f"matricula_{anio}.parquet"
        indice = SLIM / f"seguimiento_{anio}.parquet"
        if parquet.exists() and indice.exists() and not args.solo_compactar:
            print(f"{anio}: slim ya existe, se omite")
            continue

        csvs = sorted(carpeta.rglob("*.csv")) if carpeta.exists() else []
        if not csvs and not args.solo_compactar:
            if anio not in disponibles:
                print(f"{anio}: no publicado")
                continue
            url = disponibles[anio]
            carpeta.mkdir(parents=True, exist_ok=True)
            archivo = DEST / url.split("/")[-1]
            print(f"{anio}: {url.split('/')[-1]}")
            try:
                _descargar(url, archivo)
            except Exception as exc:
                print(f"    FALLO: {type(exc).__name__}: {exc}")
                archivo.unlink(missing_ok=True)
                continue
            if not SEVENZIP.exists():
                print(f"    ERROR: falta 7-Zip en {SEVENZIP}")
                return 1
            subprocess.run([str(SEVENZIP), "x", "-y", f"-o{carpeta}", str(archivo)],
                           capture_output=True)
            archivo.unlink(missing_ok=True)
            csvs = sorted(carpeta.rglob("*.csv"))

        if not csvs:
            print(f"{anio}: sin CSV tras extraer")
            continue
        # El .rar trae a veces el diccionario en otro CSV; el de matricula es
        # siempre el mas grande con diferencia.
        csv = max(csvs, key=lambda p: p.stat().st_size)
        try:
            compactar(csv, anio, parquet, seguimiento=indice)
        except Exception as exc:
            print(f"    FALLO al compactar {anio}: {type(exc).__name__}: {exc}")
            parquet.unlink(missing_ok=True)
            indice.unlink(missing_ok=True)
            continue
        if args.purgar_csv:
            csv.unlink(missing_ok=True)
            print(f"    CSV crudo eliminado")

    hechos = sorted(SLIM.glob("matricula_*.parquet"))
    print(f"\n{len(hechos)} años slim en {SLIM}")
    if hechos:
        print(f"Siguiente paso: python scripts/build_cohorts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
