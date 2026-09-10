"""Descarga los puntajes de admisión a la educación superior (Mineduc / DEMRE).

    python scripts/download_paes.py              # todos los años publicados
    python scripts/download_paes.py 2021 2022    # años específicos

Fuente: https://datosabiertos.mineduc.cl/pruebas-de-admision-a-la-educacion-superior/

QUÉ HAY Y QUÉ NO

El portal publica **desde 2021**, y con tres nombres distintos para la misma
cosa según el año: "Prueba de Transición Universitaria" (2021-2022), "Prueba
de Acceso a la Educación Superior" (2023) y "PAES" (2024 en adelante). Por eso
el catálogo se lee de la página y se reconoce por patrón, en vez de fijar las
rutas en el código.

**La PSU (2004-2020) no está en datos abiertos.** No es un descuido de este
script: la página de admisión no la publica y no aparece en ninguna otra
sección del portal. Quien la necesite debe pedirla al DEMRE por su proceso de
acceso a datos de investigación. En la práctica esto pone un piso duro: la
selectividad de admisión sólo puede medirse desde la cohorte 2021, y las
cohortes anteriores del benchmark se comparan sin ella.

ADVERTENCIA DE ESCALA

Los puntajes **no son comparables entre 2022 y 2023**. La PDT usaba el rango
150-850 y la PAES pasó a 100-1000, además de cambiar las pruebas que la
componen. Cualquier serie que mezcle ambos lados del corte sin estandarizar
está inventando una tendencia. `ingestion/paes.py` calcula el percentil dentro
de cada cohorte justamente para poder cruzar el corte.
"""
from __future__ import annotations

import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "mineduc"
PAGINA = ("https://datosabiertos.mineduc.cl/"
          "pruebas-de-admision-a-la-educacion-superior/")
SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# Los tres nombres que ha tenido el archivo de puntajes, y el separador que
# usa cada uno (unos con guion, otros con guion bajo).
PATRON = re.compile(
    r"(?:Prueba[-_]de[-_]Transicion[-_]Universitaria"
    r"|Prueba[-_]de[-_]Acceso[-_]a[-_]la[-_]Educacion[-_]Superior"
    r"|PAES)"
    r"[-_](20\d{2})[-_]Inscritos[-_]Puntajes", re.IGNORECASE)


def _contexto() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def catalogo() -> dict[int, str]:
    """Año de admisión -> URL del archivo de inscritos y puntajes."""
    req = urllib.request.Request(PAGINA, headers=HDRS)
    with urllib.request.urlopen(req, timeout=120, context=_contexto()) as r:
        html = r.read().decode("utf-8", errors="replace")
    urls = sorted(set(re.findall(
        r'href="(https://datosabiertos\.mineduc\.cl/wp-content/uploads/'
        r'[^"]+\.(?:rar|zip|csv|7z))"', html)))
    salida: dict[int, str] = {}
    for u in urls:
        m = PATRON.search(u.split("/")[-1])
        if m:
            salida[int(m.group(1))] = u
    return salida


def main() -> int:
    pedidos = [int(a) for a in sys.argv[1:]]
    disponibles = catalogo()
    if not disponibles:
        print("No se encontraron archivos de puntajes en la página.")
        return 1
    print(f"Puntajes publicados: {min(disponibles)}–{max(disponibles)} "
          f"({len(disponibles)} años)")
    print("La PSU (2004-2020) no está en datos abiertos; hay que pedirla al DEMRE.")
    DEST.mkdir(parents=True, exist_ok=True)

    for anio in (pedidos or sorted(disponibles)):
        if anio not in disponibles:
            print(f"{anio}: no publicado")
            continue
        # Se conserva el nombre de carpeta que ya espera build_benchmark.py.
        carpeta = DEST / f"paes_{anio}_puntajes"
        if carpeta.exists() and any(carpeta.rglob("*.csv")):
            print(f"{anio}: ya descargado")
            continue
        url = disponibles[anio]
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = DEST / url.split("/")[-1]
        print(f"{anio}: {url.split('/')[-1]}")
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=1800, context=_contexto()) as r, \
                    archivo.open("wb") as fh:
                total = int(r.headers.get("Content-Length", 0))
                hecho = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    hecho += len(chunk)
                    if total:
                        sys.stdout.write(f"\r    {hecho / 1e6:6.1f} / {total / 1e6:.1f} MB")
                        sys.stdout.flush()
            print()
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
        for f in sorted(carpeta.rglob("*.csv")):
            print(f"    {f.name:55s} {f.stat().st_size / 1e6:7.1f} MB")

    hechos = sorted(p.name.split("_")[1] for p in DEST.glob("paes_*_puntajes")
                    if any(p.rglob("*.csv")))
    print(f"\n{len(hechos)} años con puntajes: {', '.join(hechos)}")
    print("Siguiente paso: python scripts/build_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
