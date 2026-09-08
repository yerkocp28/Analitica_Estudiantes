"""Carga de configuracion YAML.

Toda constante del proyecto vive en config/*.yml, nunca en el codigo.
Esto permite reejecutar el pipeline con otra calibracion sin tocar Python
(doc maestro 76: reproducibilidad).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Lee un archivo de config/ por nombre (con o sin extension)."""
    path = CONFIG_DIR / (name if name.endswith(".yml") else f"{name}.yml")
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Settings:
    """Configuracion general resuelta a rutas absolutas."""

    raw: dict[str, Any]

    @classmethod
    def load(cls) -> "Settings":
        return cls(raw=load_yaml("settings"))

    @property
    def academic(self) -> dict[str, Any]:
        return self.raw["academic"]

    def path(self, key: str) -> Path:
        """Resuelve una ruta declarada en settings.paths contra el repo."""
        return REPO_ROOT / self.raw["paths"][key]


def load_synthetic_config() -> dict[str, Any]:
    """Configuracion del generador sintetico, validada minimamente."""
    cfg = load_yaml("synthetic")
    _validate_synthetic(cfg)
    return cfg


def _validate_synthetic(cfg: dict[str, Any]) -> None:
    """Falla temprano si la calibracion es internamente inconsistente."""
    tol = 1e-9

    campus_total = sum(cfg["campus"].values())
    if abs(campus_total - 1.0) > tol:
        raise ValueError(f"campus debe sumar 1.0, suma {campus_total}")

    prof_total = sum(p["prevalence"] for p in cfg["profiles"].values())
    if abs(prof_total - 1.0) > tol:
        raise ValueError(f"profiles.prevalence debe sumar 1.0, suma {prof_total}")

    career_total = sum(c["weight"] for c in cfg["careers"])
    if abs(career_total - 1.0) > tol:
        raise ValueError(f"careers.weight debe sumar 1.0, suma {career_total}")

    adoption_total = sum(
        cfg["canvas_adoption"][k] for k in ("high", "medium", "low")
    )
    if abs(adoption_total - 1.0) > tol:
        raise ValueError(f"canvas_adoption debe sumar 1.0, suma {adoption_total}")
