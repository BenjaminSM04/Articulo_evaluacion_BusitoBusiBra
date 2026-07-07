"""Logging sencillo a consola y (opcionalmente) a archivo."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "busdg", log_dir: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado.

    Args:
        name: nombre del logger (evita duplicar handlers si se reutiliza).
        log_dir: si se indica, además escribe un archivo de log con timestamp.
        level: nivel de logging.
    """
    logger = logging.getLogger(name)
    if logger.handlers:            # ya configurado: no duplicar handlers
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"log_{stamp}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
