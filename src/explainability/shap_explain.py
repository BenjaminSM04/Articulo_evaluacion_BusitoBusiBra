"""SHAP attributions for a sample of cases (pixel-level feature importance).

Uses ``shap.GradientExplainer`` (robust for modern CNNs) with a background drawn from the dataset.
SHAP on images is compute-heavy, so keep ``explainability.shap.n_eval`` small. Grad-CAM is the
primary qualitative tool; SHAP complements it with signed pixel attributions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import UltrasoundDataset
from ..training.transforms import build_eval_transforms
from .gradcam import _ClassLogitsOnly, _denormalize


def _stack_batch(df, root, cfg, n: int) -> torch.Tensor:
    """Load the first ``n`` images of a manifest as a normalized tensor batch."""
    eval_tf = build_eval_transforms(cfg)
    ds = UltrasoundDataset(df.iloc[:n], root, eval_tf)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    x, _, _ = next(iter(loader))
    return x


def run_shap(model, cfg: Config, df, root: Path, out_dir: Path, tag: str = "") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    model.to(device).eval()
    wrapped = _ClassLogitsOnly(model).to(device)

    bg = _stack_batch(df, root, cfg, cfg.explainability.shap.background_size).to(device)
    sample = _stack_batch(df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True),
                          root, cfg, cfg.explainability.shap.n_eval).to(device)

    explainer = shap.GradientExplainer(wrapped, bg)
    shap_values = explainer.shap_values(sample)   # list (per class) of arrays (N,C,H,W)

    # Convert tensors to HWC images in [0,1] for shap.image_plot.
    sample_imgs = np.stack([_denormalize(sample[i].detach(), cfg) for i in range(sample.size(0))])
    if isinstance(shap_values, list):
        sv = [np.transpose(s, (0, 2, 3, 1)) for s in shap_values]
    else:
        sv = np.transpose(shap_values, (0, 2, 3, 1))

    import matplotlib.pyplot as plt
    shap.image_plot(sv, sample_imgs, show=False)
    out_path = out_dir / f"shap_{tag or 'samples'}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] -> {out_path}")
    return out_path
