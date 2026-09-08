"""Logging estructurado y unico para todo el paquete."""
from __future__ import annotations

import logging

from .config import Settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado segun settings.logging."""
    global _CONFIGURED
    if not _CONFIGURED:
        cfg = Settings.load().raw["logging"]
        logging.basicConfig(level=cfg["level"], format=cfg["format"])
        _CONFIGURED = True
    return logging.getLogger(name)
