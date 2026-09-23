"""Paso 2 — Preprocesar BUS-BRA privado antes de construir las particiones.

Uso:
    python scripts/02_preprocess.py --config config/config_publication_v3_5seed.yaml

BUSI se cura por separado con scripts/10_curate_busi.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.preprocessing import preprocess_all  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocesamiento")
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", choices=["bus_bra"], default="bus_bra")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    preprocess_all(cfg, only=args.only)


if __name__ == "__main__":
    main()
