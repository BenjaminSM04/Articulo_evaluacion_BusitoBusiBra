"""Paso 1 — Descargar BUSI y BUS-BRA a data/raw/.

Uso:
    python scripts/01_download_data.py --config config/config.yaml
    python scripts/01_download_data.py --only busi          # solo un dataset
    python scripts/01_download_data.py --force               # re-descargar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.download import download_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga de datasets")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--only", choices=["busi", "bus_bra"], default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_all(cfg, only=args.only, force=args.force)


if __name__ == "__main__":
    main()
