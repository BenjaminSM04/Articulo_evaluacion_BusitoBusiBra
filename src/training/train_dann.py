"""Experimento D.1 — DANN (adaptación de dominio adversarial NO supervisada).

Fuente: BUSI etiquetado. Objetivo: BUS-BRA sin etiquetas (pool de adaptación).
Evaluación: test de BUS-BRA reservado por paciente. Selección de modelo por AUC de
validación del source (no se usan etiquetas del target durante el entrenamiento).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ..datasets import build_index
from ..evaluation.evaluate_external import evaluate_and_report
from ..models import DANN
from ..models.dann import dann_lambda
from ..utils.logger import get_logger
from ..utils.seed import set_seed
from .train_baseline import _save_summary
from .trainer_utils import (
    compute_class_weights,
    make_internal_splits,
    make_target_splits,
    train_domain_adaptation,
)


def _make_dann_loss(cfg: dict):
    da = cfg["domain_adaptation"]
    use_sched = da.get("lambda_schedule", True)
    gamma = da.get("lambda_gamma", 10.0)
    max_lambda = da.get("lambda_grl", 1.0)

    def batch_loss(model, batch_s, batch_t, device, progress, class_weight):
        xs, ys, _ = batch_s
        xt, _, _ = batch_t
        xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
        lambd = dann_lambda(progress, 1.0, gamma, max_lambda) if use_sched else max_lambda
        cls_logits, dom_s = model(xs, lambd)
        _, dom_t = model(xt, lambd)
        loss_cls = F.cross_entropy(cls_logits, ys, weight=class_weight)
        d_s = torch.zeros(xs.size(0), dtype=torch.long, device=device)
        d_t = torch.ones(xt.size(0), dtype=torch.long, device=device)
        loss_dom = F.cross_entropy(dom_s, d_s) + F.cross_entropy(dom_t, d_t)
        return loss_cls + loss_dom

    return batch_loss


def train_dann(cfg: dict, logger=None) -> dict:
    logger = logger or get_logger("dann", cfg["output"]["results_dir"])
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

        model = DANN(cfg).to(device)
        cw = (compute_class_weights(df_s_tr["label_idx"].values, device)
              if cfg["train"]["class_weights"] == "balanced" else None)
        ckpt = os.path.join(results_dir, backbone, f"seed{seed}", "best.pt")
        logger.info("[seed %d] DANN | source=%d val=%d adapt(target s/etiqueta)=%d test(target)=%d",
                    seed, len(df_s_tr), len(df_s_val), len(df_t_adapt), len(df_t_test))

        train_domain_adaptation(model, df_s_tr, df_s_val, df_t_adapt, cfg, device, logger,
                                ckpt, _make_dann_loss(cfg), cw)
        metrics = evaluate_and_report(
            model, df_t_test, cfg, device,
            out_dir=os.path.join(results_dir, backbone, f"seed{seed}", "eval_target"),
            tag="busbra_dann", logger=logger)
        metrics.update({"seed": seed, "method": "dann", "backbone": backbone})
        records.append(metrics)

    _save_summary(records, results_dir, logger)
    return {"records": records}
