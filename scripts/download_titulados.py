"""Descarga las bases de Titulados en Educación Superior (Mineduc).

    python scripts/download_titulados.py            # los años de las cohortes del perfil
    python scripts/download_titulados.py 2020 2021  # años específicos

Fuente: https://datosabiertos.mineduc.cl/titulados-en-educacion-superior/
Publicados 2007–2025. Cada año pesa ~6 MB comprimido y ~170 MB descomprimido.

ADVERTENCIA DE INTERPRETACIÓN

Quien se titula en 2024 ingresó alrededor de 2017–2019, así que esta base
describe una promoción distinta de la cohorte de ingreso que retrata el
perfil del benchmark. Sirve para indicadores transversales de la promoción
que egresa —duración real, sobreduración, titulación oportuna— y NO para
estimar qué proporción de la cohorte 2024 llegará a titularse. Eso último
exige matrícula histórica y seguimiento por cohorte.
"""
from __future__ import annotations

import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "titulados"
PAGINA = "https://datosabiertos.mineduc.cl/titulados-en-educacion-superior/"
SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def _contexto() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def catalogo() -> dict[int, str]:
    """Año -> URL, leído de la página en vez de hardcodeado.

    Las rutas del Mineduc incluyen el mes de publicación y cambian cuando
    republican un año; fijarlas en el código las rompe en silencio.
    """
    req = urllib.request.Request(PAGINA, headers=HDRS)
    with urllib.request.urlopen(req, timeout=120, context=_contexto()) as r:
        html = r.read().decode("utf-8", errors="replace")
    urls = sorted(set(re.findall(
        r'href="(https://datosabiertos\.mineduc\.cl/wp-content/uploads/'
        r'[^"]+\.(?:rar|zip|csv))"', html)))
    salida = {}
    for u in urls:
        m = re.search(r"(20\d{2})", u.split("/")[-1])
        if m:
            salida[int(m.group(1))] = u
    return salida


def anios_del_perfil() -> list[int]:
    """Cohortes presentes en el perfil del benchmark, si ya está generado."""
    perfiles = REPO_ROOT / "data/results/benchmark_profiles.parquet"
    if not perfiles.exists():
        return []
    import pandas as pd
    return sorted(int(a) for a in pd.read_parquet(perfiles).cohorte.unique())


def main() -> int:
    pedidos = [int(a) for a in sys.argv[1:]] or anios_del_perfil()
    if not pedidos:
        print("Sin años que descargar. Genera antes el perfil o indica años.")
        return 1

    disponibles = catalogo()
    print(f"Publicados por Mineduc: {min(disponibles)}–{max(disponibles)} "
          f"({len(disponibles)} años)")
    DEST.mkdir(parents=True, exist_ok=True)

    for anio in pedidos:
        if anio not in disponibles:
            print(f"{anio}: no publicado")
            continue
        carpeta = DEST / f"titulados_{anio}"
        if carpeta.exists() and any(carpeta.rglob("*.csv")):
            print(f"{anio}: ya descargado")
            continue
        url = disponibles[anio]
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = DEST / url.split("/")[-1]
        print(f"{anio}: {url.split('/')[-1]}")
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=900, context=_contexto()) as r, \
                    archivo.open("wb") as fh:
                total = int(r.headers.get("Content-Length", 0))
                hecho = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    hecho += len(chunk)
                    if total:
                        sys.stdout.write(f"\r    {hecho / 1e6:5.1f} / {total / 1e6:.1f} MB")
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

    print(f"\nListo en {DEST}\nSiguiente paso: python scripts/build_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
