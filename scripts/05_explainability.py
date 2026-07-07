"""Paso 5 — Explicabilidad (Grad-CAM y SHAP) sobre un modelo entrenado.

Carga un checkpoint y genera explicaciones sobre un dataset (típicamente el target, para inspeccionar
el comportamiento cross-domain). Compara visualmente si el modelo mira la lesión o artefactos.

Uso:
    python scripts/05_explainability.py --checkpoint results/models/baseline_busi_resnet50_fold0.pt \
        --dataset bus_bra --method both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.datasets import load_manifest  # noqa: E402
from src.explainability import run_gradcam, run_shap  # noqa: E402
from src.models.architectures import build_model  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Explicabilidad (XAI)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["busi", "bus_bra"])
    ap.add_argument("--method", default="both", choices=["gradcam", "shap", "both"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    if "arch" in ckpt:
        cfg.model["architecture"] = ckpt["arch"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device).eval()

    df = load_manifest(cfg, args.dataset)
    root = cfg._root
    tag = f"{Path(args.checkpoint).stem}_on_{args.dataset}"
    out_dir = cfg.path("figures") / "explainability"

    if args.method in ("gradcam", "both"):
        run_gradcam(model, cfg, df, root, out_dir, tag=tag)
    if args.method in ("shap", "both"):
        run_shap(model, cfg, df, root, out_dir, tag=tag)
    print("[ok] explicaciones generadas.")


if __name__ == "__main__":
    main()
