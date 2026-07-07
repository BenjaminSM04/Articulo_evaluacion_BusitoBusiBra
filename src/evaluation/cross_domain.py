"""Cross-domain generalization matrix.

For each *source* dataset we train one model on a stratified train/val split and keep a held-out
test split. Then we fill an N×N matrix where:

* diagonal  (train==eval) = intra-domain performance on the source's held-out test set;
* off-diag  (train!=eval) = performance of the source model on the *entire* target dataset,
  optionally after a domain-adaptation step (none | finetune | dann | self_training).

The primary cell value is AUC (with bootstrap CI); full metrics per cell are also saved.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import (
    UltrasoundDataset,
    build_dataloaders,
    class_weights,
    load_manifest,
    make_cv_splits,
)
from ..models.architectures import build_model
from ..training.domain_adaptation import dann_train, finetune_on_target, self_training
from ..training.train import train_model
from ..training.transforms import build_eval_transforms, build_train_transforms
from ..utils.reporting import save_json
from .metrics import auc_metric, bootstrap_ci, compute_metrics, run_inference


def _evaluate_on(model, df: pd.DataFrame, root, cfg: Config, exclude: np.ndarray | None = None):
    """Run inference on a manifest (optionally excluding indices) and return metrics + CI."""
    if exclude is not None and len(exclude):
        df = df.drop(index=exclude).reset_index(drop=True)
    loader = DataLoader(UltrasoundDataset(df, root, build_eval_transforms(cfg)),
                        batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)
    y_true, y_prob, _ = run_inference(model, loader, cfg.device, cfg.training.mixed_precision)
    metrics = compute_metrics(y_true, y_prob)
    if cfg.evaluation.bootstrap_ci:
        lo, hi = bootstrap_ci(y_true, y_prob, auc_metric,
                              cfg.evaluation.n_bootstrap, cfg.evaluation.ci_level, cfg.seed)
        metrics["auc_ci"] = [lo, hi]
    return metrics


def run_generalization_matrix(cfg: Config, arch: str | None = None,
                              adaptation: str | None = None) -> dict:
    if arch:
        cfg.model["architecture"] = arch
    method = adaptation or cfg.domain_adaptation.method
    arch = cfg.model.architecture
    keys = list(cfg.datasets.keys())
    root = cfg._root
    train_tf, eval_tf = build_train_transforms(cfg), build_eval_transforms(cfg)
    manifests = {k: load_manifest(cfg, k) for k in keys}

    # --- 1) Train one baseline model per source domain (+ keep its test split) ---
    source_models: dict[str, torch.nn.Module] = {}
    diag_metrics: dict[str, dict] = {}
    for s in keys:
        df = manifests[s]
        split = make_cv_splits(df, cfg)[0]   # single representative fold for the matrix
        loaders = build_dataloaders(df, root, train_tf, eval_tf, cfg, split)
        model = build_model(cfg)
        ckpt = cfg.path("models") / f"baseline_{s}_{arch}.pt"
        print(f"\n=== Entrenando baseline en {s} ({arch}) ===")
        out = train_model(model, loaders, cfg, class_weights(df.iloc[split["train_idx"]]),
                          device=cfg.device, ckpt_path=ckpt)
        source_models[s] = out["model"]
        y_true, y_prob, _ = run_inference(out["model"], loaders["test"], cfg.device,
                                          cfg.training.mixed_precision)
        diag = compute_metrics(y_true, y_prob)
        if cfg.evaluation.bootstrap_ci:
            diag["auc_ci"] = list(bootstrap_ci(y_true, y_prob, auc_metric,
                                               cfg.evaluation.n_bootstrap,
                                               cfg.evaluation.ci_level, cfg.seed))
        diag_metrics[s] = diag

    # --- 2) Fill the matrix ---
    n = len(keys)
    matrix = np.full((n, n), np.nan)
    full: dict[str, dict] = {}
    for i, s in enumerate(keys):
        for j, t in enumerate(keys):
            if s == t:
                m = diag_metrics[s]
            else:
                print(f"\n=== Cross-domain {s} -> {t}  (adaptación: {method}) ===")
                df_t = manifests[t]
                exclude = None
                if method == "none":
                    model_eval = source_models[s]
                elif method == "finetune":
                    model_eval, exclude = finetune_on_target(
                        copy.deepcopy(source_models[s]), df_t, root, cfg)
                elif method == "dann":
                    model_eval = dann_train(manifests[s], df_t, root, cfg)
                elif method == "self_training":
                    model_eval = self_training(copy.deepcopy(source_models[s]),
                                               manifests[s], df_t, root, cfg)
                else:
                    raise ValueError(f"Método de adaptación desconocido: {method}")
                m = _evaluate_on(model_eval, df_t, root, cfg, exclude)
            matrix[i, j] = m["auc"]
            full[f"{s}__to__{t}"] = m
            print(f"  {s} -> {t}: AUC={m['auc']:.3f}  sens={m['sensitivity']:.3f}  "
                  f"spec={m['specificity']:.3f}")

    # --- 3) Generalization gap (intra - cross) per source ---
    gaps = {}
    for i, s in enumerate(keys):
        intra = matrix[i, i]
        cross = [matrix[i, j] for j in range(n) if j != i]
        gaps[s] = {"intra_auc": float(intra),
                   "mean_cross_auc": float(np.nanmean(cross)) if cross else float("nan"),
                   "mean_gap": float(intra - np.nanmean(cross)) if cross else float("nan")}

    result = {"arch": arch, "adaptation": method, "domains": keys,
              "auc_matrix": matrix.tolist(), "metrics": full, "gaps": gaps}
    out_path = cfg.path("reports") / f"cross_domain_{arch}_{method}.json"
    save_json(result, out_path)
    print(f"\n[matrix] resultados -> {out_path}")
    return result
