"""Evaluación y reportes (usado por todos los experimentos).

``evaluate_and_report`` es el núcleo reutilizable: corre inferencia, calcula el panel
de métricas con IC por bootstrap y guarda predicciones, JSON de métricas y gráficos
(ROC, PR, matriz de confusión, calibración).

``run_external_eval`` es el Experimento B: carga un checkpoint entrenado en BUSI y lo
evalúa sobre el test reservado de BUS-BRA (mismo split por paciente que la DA, para que
B/D/E sean comparables sobre el mismo conjunto de prueba).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ..datasets import UltrasoundDataset, build_index, build_transforms  # noqa: E402
from ..models import build_classifier  # noqa: E402
from ..utils.logger import get_logger  # noqa: E402
from ..utils.seed import seed_worker  # noqa: E402
from .calibration import plot_reliability  # noqa: E402
from .metrics import (  # noqa: E402
    auc_metric,
    bootstrap_ci,
    compute_metrics,
    pr_auc_metric,
    pr_points,
    roc_points,
    run_inference,
)


def _make_eval_loader(df, cfg) -> DataLoader:
    tf = build_transforms(cfg, train=False)
    ds = UltrasoundDataset(df, tf, roi_mode=cfg["data"]["roi_mode"],
                           roi_margin=cfg["data"]["roi_margin"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                      num_workers=cfg["train"]["num_workers"], worker_init_fn=seed_worker,
                      pin_memory=("cuda" in str(cfg["device"])))


def _plot_roc(y_true, y_prob, path, tag):
    fpr, tpr = roc_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_metric(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("1 - especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title(f"ROC — {tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_pr(y_true, y_prob, path, tag):
    recall, precision = pr_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"PR-AUC={pr_auc_metric(y_true, y_prob):.3f}")
    ax.set_xlabel("Recall (sensibilidad)")
    ax.set_ylabel("Precisión")
    ax.set_title(f"Precision-Recall — {tag}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_cm(cm, path, tag):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["benigno", "maligno"]); ax.set_yticklabels(["benigno", "maligno"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión — {tag}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_and_report(model, df, cfg, device, out_dir, tag="eval", logger=None) -> dict:
    """Evalúa un modelo sobre ``df`` y guarda métricas + gráficos. Devuelve el dict de métricas."""
    os.makedirs(out_dir, exist_ok=True)
    amp = cfg["train"].get("mixed_precision", True) and "cuda" in str(device)
    loader = _make_eval_loader(df, cfg)
    y_true, y_prob, idx = run_inference(model, loader, device, amp)

    metrics = compute_metrics(y_true, y_prob, cfg["eval"].get("threshold", 0.5))
    if cfg["eval"].get("bootstrap_ci", True):
        nb = cfg["eval"]["n_bootstrap"]
        lv = cfg["eval"]["ci_level"]
        metrics["auc_ci"] = bootstrap_ci(y_true, y_prob, auc_metric, nb, lv)
        metrics["pr_auc_ci"] = bootstrap_ci(y_true, y_prob, pr_auc_metric, nb, lv)

    pd.DataFrame({"index": idx, "y_true": y_true, "y_prob": y_prob}) \
        .to_csv(os.path.join(out_dir, f"{tag}_predictions.csv"), index=False)
    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    if len(np.unique(y_true)) > 1:
        _plot_roc(y_true, y_prob, os.path.join(out_dir, f"{tag}_roc.png"), tag)
        _plot_pr(y_true, y_prob, os.path.join(out_dir, f"{tag}_pr.png"), tag)
        plot_reliability(y_true, y_prob, os.path.join(out_dir, f"{tag}_calibration.png"))
    _plot_cm(metrics["confusion_matrix"], os.path.join(out_dir, f"{tag}_cm.png"), tag)

    if logger:
        logger.info("[%s] AUC=%.3f PR-AUC=%.3f sens=%.3f spec=%.3f bAcc=%.3f (n=%d)",
                    tag, metrics["auc"], metrics["pr_auc"], metrics["sensitivity"],
                    metrics["specificity"], metrics["balanced_accuracy"], metrics["n"])
    return metrics


def run_external_eval(cfg: dict, logger=None) -> dict:
    """Experimento B — BUSI -> BUS-BRA sin reentrenar (mide la caída por domain shift)."""
    logger = logger or get_logger("external_eval", cfg["output"]["results_dir"])
    device = cfg["device"]
    ckpt = cfg.get("checkpoint")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(
            "Falta el checkpoint del baseline BUSI (Experimento A). "
            "Ejecuta primero baseline_busi.yaml o pasa --checkpoint. Ver README.")

    model = build_classifier(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state,
                          strict=False)

    df_t = build_index("busbra", cfg, logger)
    from ..training.trainer_utils import make_target_splits   # import diferido (evita ciclo)
    df_test = make_target_splits(df_t, cfg, cfg["seed"])["test"]
    logger.info("Evaluación externa sobre test target reservado: n=%d", len(df_test))
    return evaluate_and_report(model, df_test, cfg, device,
                               out_dir=cfg["output"]["results_dir"],
                               tag="busi_to_busbra", logger=logger)
