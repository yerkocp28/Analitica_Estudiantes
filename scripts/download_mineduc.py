"""Descarga bases abiertas de Mineduc / SIES.

    python scripts/download_mineduc.py

Estas bases estan desagregadas a nivel individual y usan **MRUN**, un
identificador ficticio y estable que permite seguir la trayectoria de una
persona entre bases y anios (prekinder -> doctorado). Chile es de los pocos
paises OCDE que publica esto abiertamente.

Para que sirven en este proyecto (y para que NO):

  SI  - calibrar el generador sintetico con tasas chilenas reales, en vez
        de los [SUPUESTO] que hoy tiene config/synthetic.yml;
  SI  - medir el PISO predictivo de las variables previas al ingreso a
        escala nacional (replicar ULagos con 20 anios en vez de 2 cohortes);
  SI  - benchmark de la UA contra el sistema;
  NO  - features de estudiantes UA. El MRUN esta enmascarado y no existe
        tabla de equivalencia con el RUT que tiene Banner. No hay forma de
        unir estas bases a un estudiante concreto de la universidad.

Licencia: datos abiertos de gobierno. Fuente: datosabiertos.mineduc.cl
"""
from __future__ import annotations

import argparse
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "mineduc"

BASE = "https://datosabiertos.mineduc.cl/wp-content/uploads"

# Solo lo necesario para el analisis; el catalogo completo es mucho mayor.
FILES = {
    # Tres anios consecutivos de matricula permiten dos transiciones
    # (2023->2024 y 2024->2025) y por tanto validacion TEMPORAL: la cohorte
    # antigua entrena y la reciente testea. Con un solo anio solo cabe un
    # split aleatorio, que mezcla cohortes y es optimista.
    "matricula_2023": f"{BASE}/2026/09/Matricula-Ed-Superior-2023.rar",
    "matricula_2024": f"{BASE}/2026/09/Matricula-Ed-Superior-2024.rar",
    "matricula_2025": f"{BASE}/2026/09/Matricula-Ed-Superior-2025.rar",
    # En 2023 la prueba aun se llamaba "de Acceso a la Educacion Superior".
    "paes_2023_puntajes": (f"{BASE}/2023/05/Prueba-de-Acceso-a-la-Educacion"
                           "-Superior-2023-Inscritos-Puntajes-1.rar"),
    "paes_2024_puntajes": f"{BASE}/2025/10/PAES-2024-Inscritos-Puntajes.rar",
}

SEVENZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def _download(url: str, path: Path) -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=900, context=ctx) as r, path.open("wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    print()


def _extract(archive: Path, into: Path) -> bool:
    """7-Zip maneja rar y zip. Sin el, no se puede seguir en Windows."""
    into.mkdir(parents=True, exist_ok=True)
    if not SEVENZIP.exists():
        print(f"    ERROR: no se encontro 7-Zip en {SEVENZIP}")
        print("    Instalar con: winget install 7zip.7zip")
        return False
    res = subprocess.run([str(SEVENZIP), "x", "-y", f"-o{into}", str(archive)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ERROR al extraer: {res.stderr[-400:]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FILES),
                    help="Descargar solo estos")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    targets = {k: FILES[k] for k in (args.only or FILES)}
    DEST.mkdir(parents=True, exist_ok=True)

    for name, url in targets.items():
        into = DEST / name
        if into.exists() and any(into.iterdir()):
            print(f"{name}: ya existe, se omite")
            continue
        print(f"{name}\n    {url}")
        archive = DEST / f"{name}{Path(url).suffix}"
        try:
            _download(url, archive)
        except Exception as exc:
            print(f"    FALLO en la descarga: {type(exc).__name__}: {exc}")
            continue
        if _extract(archive, into):
            for f in sorted(into.rglob("*")):
                if f.is_file():
                    print(f"    {f.name:55s} {f.stat().st_size / 1e6:8.1f} MB")
            if not args.keep_archives:
                archive.unlink(missing_ok=True)

    print(f"\nListo en {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
