"""Experimento D.2 — Adaptación por alineación de características: Deep CORAL o MMD.

Fuente: BUSI etiquetado. Objetivo: BUS-BRA sin etiquetas. Se añade a la pérdida de
clasificación un término que acerca las distribuciones de features source/target
(CORAL = diferencia de covarianzas; MMD = discrepancia media máxima multi-kernel).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ..datasets import build_index
from ..evaluation.evaluate_external import evaluate_and_report
from ..models import build_classifier
from ..utils.logger import get_logger
from ..utils.seed import set_seed
from .train_baseline import _save_summary
from .trainer_utils import (
    compute_class_weights,
    make_internal_splits,
    make_target_splits,
    train_domain_adaptation,
)


def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: distancia entre las covarianzas de source y target."""
    d = fs.size(1)
    fs_c = fs - fs.mean(0, keepdim=True)
    ft_c = ft - ft.mean(0, keepdim=True)
    cov_s = fs_c.t() @ fs_c / max(1, fs.size(0) - 1)
    cov_t = ft_c.t() @ ft_c / max(1, ft.size(0) - 1)
    return (cov_s - cov_t).pow(2).sum() / (4 * d * d)


def mmd_loss(fs: torch.Tensor, ft: torch.Tensor, sigmas) -> torch.Tensor:
    """MMD con kernel RBF multi-escala."""
    def rbf(a, b):
        aa = a.pow(2).sum(1, keepdim=True)
        bb = b.pow(2).sum(1, keepdim=True)
        dist = aa - 2 * a @ b.t() + bb.t()
        k = 0.0
        for s in sigmas:
            k = k + torch.exp(-dist / (2 * s ** 2))
        return k
    return rbf(fs, fs).mean() + rbf(ft, ft).mean() - 2 * rbf(fs, ft).mean()


def _make_align_loss(cfg: dict):
    da = cfg["domain_adaptation"]
    method = da.get("method", "coral")
    weight = da.get("align_weight", 1.0)
    warm = da.get("align_warmup_epochs", 5)
    sigmas = da.get("mmd_sigmas", [1, 2, 4, 8, 16])
    epochs = cfg["train"]["epochs"]

    def batch_loss(model, batch_s, batch_t, device, progress, class_weight):
        xs, ys, _ = batch_s
        xt, _, _ = batch_t
        xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
        logits_s, fs = model(xs, return_features=True)
        ft = model.features(xt)
        loss_cls = F.cross_entropy(logits_s, ys, weight=class_weight)
        align = mmd_loss(fs, ft, sigmas) if method == "mmd" else coral_loss(fs, ft)
        # Warmup del peso de alineación (estabiliza el inicio).
        cur_epoch = progress * epochs
        w = weight * min(1.0, cur_epoch / max(1, warm))
        return loss_cls + w * align

    return batch_loss


def train_coral(cfg: dict, logger=None) -> dict:
    method = cfg["domain_adaptation"].get("method", "coral")
    logger = logger or get_logger(method, cfg["output"]["results_dir"])
    device = cfg["device"]
    results_dir = cfg["output"]["results_dir"]
    backbone = cfg["model"]["backbone"]

    df_s = build_index("busi", cfg, logger)
    df_t = build_index("busbra", cfg, logger)
    records = []

    for seed in cfg["train"]["seeds"]:
        set_seed(seed)
        s = make_internal_splits(df_s, cfg, "busi", seed)
        df_s_tr, df_s_val = s["folds"][0]
        t = make_target_splits(df_t, cfg, seed)
        df_t_adapt, df_t_test = t["adapt"], t["test"]

        model = build_classifier(cfg).to(device)
        cw = (compute_class_weights(df_s_tr["label_idx"].values, device)
              if cfg["train"]["class_weights"] == "balanced" else None)
        ckpt = os.path.join(results_dir, backbone, f"seed{seed}", "best.pt")
        logger.info("[seed %d] %s | source=%d adapt(target)=%d test(target)=%d",
                    seed, method.upper(), len(df_s_tr), len(df_t_adapt), len(df_t_test))

        train_domain_adaptation(model, df_s_tr, df_s_val, df_t_adapt, cfg, device, logger,
                                ckpt, _make_align_loss(cfg), cw)
        metrics = evaluate_and_report(
            model, df_t_test, cfg, device,
            out_dir=os.path.join(results_dir, backbone, f"seed{seed}", "eval_target"),
            tag=f"busbra_{method}", logger=logger)
        metrics.update({"seed": seed, "method": method, "backbone": backbone})
        records.append(metrics)

    _save_summary(records, results_dir, logger)
    return {"records": records}
