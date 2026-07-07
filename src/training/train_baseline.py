"""Experimento A — Baseline interno: entrenar y evaluar en BUSI.

Multi-semilla (media +/- sd). Selección de modelo por AUC de validación del source.
Guarda el mejor checkpoint (para la validación externa del Experimento B) y las
métricas de test interno con IC por bootstrap.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..datasets import build_index
from ..evaluation.evaluate_external import evaluate_and_report
from ..models import build_classifier
from ..utils.logger import get_logger
from ..utils.seed import set_seed
from .trainer_utils import (
    compute_class_weights,
    fit_classifier,
    load_checkpoint,
    make_internal_splits,
    make_target_splits,
    sample_few_shot,
)


def train_baseline(cfg: dict, logger=None) -> dict:
    logger = logger or get_logger("baseline", cfg["output"]["results_dir"])
    device = cfg["device"]
    results_dir = cfg["output"]["results_dir"]
    backbone = cfg["model"]["backbone"]

    df = build_index("busi", cfg, logger)
    records: list[dict] = []
    best_overall = {"val_auc": -np.inf, "ckpt": None}

    for seed in cfg["train"]["seeds"]:
        set_seed(seed)
        splits = make_internal_splits(df, cfg, dataset_name="busi", seed=seed)
        df_test = splits["test"]
        for fold_i, (df_tr, df_val) in enumerate(splits["folds"]):
            seed_dir = os.path.join(results_dir, backbone, f"seed{seed}")
            ckpt = (os.path.join(seed_dir, "best.pt") if len(splits["folds"]) == 1
                    else os.path.join(seed_dir, f"fold{fold_i}", "best.pt"))

            model = build_classifier(cfg).to(device)
            class_weight = (compute_class_weights(df_tr["label_idx"].values, device)
                            if cfg["train"]["class_weights"] == "balanced" else None)
            logger.info("[seed %d | fold %d] entrenando %s ...", seed, fold_i, backbone)
            val_auc = fit_classifier(model, df_tr, df_val, cfg, device, logger, ckpt, class_weight)

            # Evaluación en el TEST interno reservado (con gráficos + IC bootstrap).
            metrics = evaluate_and_report(
                model, df_test, cfg, device,
                out_dir=os.path.join(seed_dir, f"eval_test_fold{fold_i}"),
                tag="busi_internal", logger=logger)
            metrics.update({"seed": seed, "fold": fold_i, "val_auc": val_auc,
                            "backbone": backbone})
            records.append(metrics)
            if val_auc > best_overall["val_auc"]:
                best_overall = {"val_auc": val_auc, "ckpt": ckpt}

    _save_summary(records, results_dir, logger)
    logger.info("Baseline listo. Mejor checkpoint: %s", best_overall["ckpt"])
    return {"best_checkpoint": best_overall["ckpt"], "records": records}


def _save_summary(records: list[dict], results_dir: str, logger) -> None:
    """Guarda tabla por corrida y resumen agregado (media +/- sd)."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_dir, "metrics_all_runs.csv"), index=False)
    numeric = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
               "precision", "f1", "auc", "pr_auc", "brier", "ece"]
    summary = {}
    for col in numeric:
        if col in df.columns:
            summary[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if logger:
        logger.info("Resumen -> %s", os.path.join(results_dir, "summary.json"))


def _split_labeled(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Divide un conjunto etiquetado pequeño en train/val (por paciente si es posible)."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    y = df["label_idx"].values
    if df["patient_id"].notna().all() and df["patient_id"].nunique() > 1:
        sp = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr, va = next(sp.split(df, y, df["patient_id"].values))
    else:
        sp = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr, va = next(sp.split(df, y))
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)


def train_finetune(cfg: dict, logger=None) -> dict:
    """Experimento E — Fine-tuning controlado con pocas etiquetas de BUS-BRA.

    Parte de un checkpoint entrenado en BUSI (``cfg['checkpoint']``) y lo ajusta con
    5/10/20 % de BUS-BRA etiquetado (muestreado POR PACIENTE), evaluando en el test
    target reservado (disjunto del subconjunto etiquetado). Comparable con la DA
    no supervisada (D).
    """
    logger = logger or get_logger("finetune", cfg["output"]["results_dir"])
    device = cfg["device"]
    results_dir = cfg["output"]["results_dir"]
    backbone = cfg["model"]["backbone"]

    df_t = build_index("busbra", cfg, logger)
    fractions = cfg["split"].get("finetune_label_fractions") \
        or [cfg["split"].get("finetune_label_fraction", 0.0)]
    fractions = [f for f in fractions if f and f > 0] or [0.05, 0.10, 0.20]

    records = []
    for seed in cfg["train"]["seeds"]:
        set_seed(seed)
        target = make_target_splits(df_t, cfg, seed)
        pool, df_test = target["adapt"], target["test"]
        for frac in fractions:
            df_lab = sample_few_shot(pool, frac, seed)
            df_tr, df_val = _split_labeled(df_lab, val_frac=0.2, seed=seed)
            pct = int(round(frac * 100))

            model = build_classifier(cfg).to(device)
            if cfg.get("checkpoint"):
                load_checkpoint(model, cfg["checkpoint"], device)   # init desde BUSI
            class_weight = (compute_class_weights(df_tr["label_idx"].values, device)
                            if cfg["train"]["class_weights"] == "balanced" else None)
            ckpt = os.path.join(results_dir, backbone, f"seed{seed}", f"frac{pct}", "best.pt")
            logger.info("[seed %d | %d%%] fine-tuning: etiquetadas=%d (train=%d val=%d) test=%d",
                        seed, pct, len(df_lab), len(df_tr), len(df_val), len(df_test))
            fit_classifier(model, df_tr, df_val, cfg, device, logger, ckpt, class_weight)

            metrics = evaluate_and_report(
                model, df_test, cfg, device,
                out_dir=os.path.join(results_dir, backbone, f"seed{seed}", f"frac{pct}", "eval_target"),
                tag=f"busbra_ft{pct}", logger=logger)
            metrics.update({"seed": seed, "method": f"finetune_{pct}pct",
                            "label_fraction": frac, "backbone": backbone})
            records.append(metrics)

    _save_summary(records, results_dir, logger)
    return {"records": records}
