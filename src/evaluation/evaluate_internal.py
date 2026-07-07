"""Evaluación interna standalone: un checkpoint sobre el test reservado de BUSI.

(El baseline ya evalúa el test interno; esta función permite reevaluar un checkpoint
existente sin reentrenar.)
"""
from __future__ import annotations

import os

import torch

from ..datasets import build_index
from ..models import build_classifier
from ..utils.logger import get_logger
from .evaluate_external import evaluate_and_report


def run_internal_eval(cfg: dict, logger=None) -> dict:
    logger = logger or get_logger("internal_eval", cfg["output"]["results_dir"])
    device = cfg["device"]
    model = build_classifier(cfg).to(device)

    ckpt = cfg.get("checkpoint")
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state
                              else state, strict=False)
    else:
        logger.warning("Sin checkpoint válido: se evalúa un modelo no entrenado (solo prueba).")

    df = build_index("busi", cfg, logger)
    from ..training.trainer_utils import make_internal_splits   # import diferido (evita ciclo)
    df_test = make_internal_splits(df, cfg, "busi", cfg["seed"])["test"]
    return evaluate_and_report(model, df_test, cfg, device,
                               out_dir=cfg["output"]["results_dir"],
                               tag="busi_internal", logger=logger)
