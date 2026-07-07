"""Download BUSI and BUS-BRA into ``data/raw/`` via the Kaggle API.

Requires a configured Kaggle token (see data/README.md). If the data already exists the download
is skipped. Manual download is always possible (URLs in data/README.md); this module only
automates the common path.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from ..config import Config


def _kaggle_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _download_kaggle_dataset(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] kaggle datasets download -d {slug} -> {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )
    # Some mirrors leave a leftover zip; unzip and remove if present.
    for zf in dest.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink(missing_ok=True)


def _looks_populated(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.png"))


def download_dataset(cfg: Config, key: str, force: bool = False) -> Path:
    """Download a single dataset by config key ('busi' or 'bus_bra')."""
    ds = cfg.datasets[key]
    raw_dir = cfg.resolve(ds["raw_dir"])
    if _looks_populated(raw_dir) and not force:
        print(f"[download] {ds['name']} ya presente en {raw_dir} (omitido).")
        return raw_dir
    if not _kaggle_available():
        raise RuntimeError(
            "La CLI de Kaggle no está disponible o no está configurada. "
            "Instala 'kaggle', coloca kaggle.json (ver data/README.md), o descarga manualmente."
        )
    _download_kaggle_dataset(ds["kaggle_slug"], raw_dir)
    if not _looks_populated(raw_dir):
        raise RuntimeError(f"Descarga de {ds['name']} terminó pero no se hallaron PNG en {raw_dir}.")
    print(f"[download] {ds['name']} listo en {raw_dir}.")
    return raw_dir


def download_all(cfg: Config, only: str | None = None, force: bool = False) -> None:
    keys = [only] if only else list(cfg.datasets.keys())
    for key in keys:
        download_dataset(cfg, key, force=force)
