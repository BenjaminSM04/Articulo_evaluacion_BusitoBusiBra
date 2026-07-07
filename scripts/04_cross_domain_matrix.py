"""Paso 4 — Matriz de generalización cruzada (entrenar en X, evaluar en Y).

Construye la matriz N×N de AUC (diagonal = intra-dominio; fuera de diagonal = cross-domain),
opcionalmente aplicando una técnica de adaptación de dominio, y guarda el heatmap.

Uso:
    python scripts/04_cross_domain_matrix.py --arch resnet50 --adaptation none
    python scripts/04_cross_domain_matrix.py --arch resnet50 --adaptation finetune
    python scripts/04_cross_domain_matrix.py --arch resnet50 --adaptation dann
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.evaluation.cross_domain import run_generalization_matrix  # noqa: E402
from src.utils.reporting import plot_cross_domain_matrix  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Matriz de generalización cruzada")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--adaptation", default=None,
                    choices=["none", "finetune", "dann", "self_training"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    result = run_generalization_matrix(cfg, arch=args.arch, adaptation=args.adaptation)

    domains = result["domains"]
    matrix = np.array(result["auc_matrix"])
    fig_path = cfg.path("figures") / f"matrix_{result['arch']}_{result['adaptation']}.png"
    plot_cross_domain_matrix(matrix, domains, domains, fig_path, metric_name="AUC")
    print(f"[ok] heatmap -> {fig_path}")
    for s, g in result["gaps"].items():
        print(f"  {s}: intra={g['intra_auc']:.3f}  cross={g['mean_cross_auc']:.3f}  "
              f"gap={g['mean_gap']:.3f}")


if __name__ == "__main__":
    main()
