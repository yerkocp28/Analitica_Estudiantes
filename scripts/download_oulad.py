"""Descarga OULAD (Open University Learning Analytics Dataset).

    python scripts/download_oulad.py

~45 MB comprimidos, ~440 MB descomprimidos (studentVle.csv es el grueso).
Se descarga desde el mirror de UCI, que sirve el zip de forma directa; el
sitio original de la Open University responde con una pagina HTML.

Licencia: CC-BY 4.0. Cita:
Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). Open University Learning
Analytics dataset. Scientific Data 4, 170171.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/349/"
       "open+university+learning+analytics+dataset.zip")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "external" / "oulad"
ZIP_PATH = DEST.parent / "oulad.zip"

EXPECTED = ["studentInfo.csv", "studentVle.csv", "studentAssessment.csv",
            "assessments.csv", "courses.csv", "studentRegistration.csv"]


def _progress(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, block * block_size * 100 // total)
    sys.stdout.write(f"\r  descargando... {pct:3d}%")
    sys.stdout.flush()


def main() -> int:
    if all((DEST / f).exists() for f in EXPECTED):
        print(f"OULAD ya esta en {DEST}")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Descargando OULAD desde UCI\n  {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH, _progress)
    print(f"\n  {ZIP_PATH.stat().st_size / 1e6:.1f} MB")

    if not zipfile.is_zipfile(ZIP_PATH):
        print("ERROR: la descarga no es un zip (probablemente una pagina HTML).")
        print("Descarga manual: https://analyse.kmi.open.ac.uk/open_dataset")
        return 1

    print("Descomprimiendo...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DEST)

    faltan = [f for f in EXPECTED if not (DEST / f).exists()]
    if faltan:
        print(f"ADVERTENCIA: faltan archivos esperados: {faltan}")
        return 1

    print(f"\nListo en {DEST}")
    for f in EXPECTED:
        print(f"  {f:28s} {(DEST / f).stat().st_size / 1e6:8.1f} MB")
    print("\nSiguiente paso:")
    print("  python scripts/validate_lead_time.py --source oulad --out data/results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
