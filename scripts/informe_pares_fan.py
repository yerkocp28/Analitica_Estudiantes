"""Posición de la FAN frente a universidades pares: python scripts/informe_pares_fan.py

Producto P4 de la propuesta. Compara la continuidad al año siguiente de quienes
ingresan al área de Administración y Comercio en la UA con la de sus cinco
universidades más parecidas por perfil de ingreso, y con el resto del sistema.

Usa solo los agregados públicos versionados en `deploy/data/` (Mineduc/SIES y
CNED), así que el informe puede circular sin restricciones de privacidad. El
cálculo está en `student_analytics.modeling.area_position`, que también usa la
vista del piloto FAN.

Escribe documentacion/informe_pares_fan.html.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from student_analytics.modeling.area_position import area_position, area_position_html  # noqa: E402

DATOS = ROOT / "deploy" / "data"
SALIDA = ROOT / "documentacion" / "informe_pares_fan.html"
AREA = "Administración y Comercio"


def main() -> int:
    config = yaml.safe_load((ROOT / "config/benchmark.yml").read_text(encoding="utf-8"))
    resultado = area_position(pd.read_parquet(DATOS / "retention_universities.parquet"),
                              pd.read_parquet(DATOS / "benchmark_profiles.parquet"), config, AREA)
    SALIDA.write_text(area_position_html(resultado, datetime.now(timezone.utc).strftime("%d/%m/%Y")),
                      encoding="utf-8")
    print("\n".join("· " + h for h in resultado["findings"]))
    print(f"-> {SALIDA.relative_to(ROOT) if SALIDA.is_relative_to(ROOT) else SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
