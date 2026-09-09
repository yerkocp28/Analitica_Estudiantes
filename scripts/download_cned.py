"""Descarga las bases INDICES del Consejo Nacional de Educación.

    python scripts/download_cned.py

La base institucional (3,3 MB) cubre 2005-2025 y trae por institución y sede
el cuerpo docente por jornada y nivel de grado, inmuebles, laboratorios y
bibliotecas. Es lo que permite incorporar recursos institucionales al
benchmark, que hasta ahora se declaraban fuera de alcance.

Fuente: https://cned.cl/institucional/bases-de-datos/
Los datos los reportan las propias instituciones al CNED.
"""
from __future__ import annotations

import ssl
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "cned"

FILES = {
    # Docentes, inmuebles, laboratorios y bibliotecas por institución y sede.
    "INDICES_Institucional_2005-2025.xlsx":
        "https://cned.cl/wp-content/uploads/2025/12/INDICES_Institucional_2005-2025.xlsx",
    # Matrícula por programa desde 2005: permite extender el benchmark más
    # allá de las dos cohortes que dan las bases del SIES descargadas.
    "Base-INDICES-Matricula-2005-2026.xlsx":
        "https://cned.cl/wp-content/uploads/2026/08/Base-INDICES-Matricula-2005-2026.xlsx",
}

HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def main() -> int:
    ap_only = sys.argv[1:] or None
    DEST.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for nombre, url in FILES.items():
        if ap_only and nombre not in ap_only:
            continue
        destino = DEST / nombre
        if destino.exists():
            print(f"{nombre}: ya existe ({destino.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"{nombre}\n    {url}")
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=900, context=ctx) as r, \
                    destino.open("wb") as fh:
                total = int(r.headers.get("Content-Length", 0))
                hecho = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    hecho += len(chunk)
                    if total:
                        sys.stdout.write(f"\r    {hecho / 1e6:6.1f} / {total / 1e6:.1f} MB")
                        sys.stdout.flush()
            print(f"\n    {destino.stat().st_size / 1e6:.1f} MB")
        except Exception as exc:
            print(f"    FALLO: {type(exc).__name__}: {exc}")
            destino.unlink(missing_ok=True)

    print(f"\nListo en {DEST}")
    print("Siguiente paso: python scripts/build_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
