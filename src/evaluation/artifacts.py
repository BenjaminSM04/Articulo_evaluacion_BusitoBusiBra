"""Evaluation artifact writer for article-scale experiments.

This module is intentionally independent from the numbered scripts so baseline, external,
adaptation, fine-tuning, and tests can all persist the same artifact contract.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import UltrasoundDataset
from ..training.transforms import build_eval_transforms
from ..utils.reporting import load_json, plot_confusion_matrix, plot_roc, save_json
from .calibration import plot_reliability
from .metrics import compute_metrics, pr_auc_metric, pr_points, roc_points, run_inference

ARTICLE_FIGURE_DIRS = (
    "confusion_matrices",
    "roc_curves",
    "pr_curves",
    "calibration_curves",
    "gradcam",
)

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auc",
    "pr_auc",
    "brier",
    "ece",
]


def safe_id(value: str) -> str:
    """Filesystem-safe experiment identifier."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "experiment"


def ensure_article_dirs(cfg: Config) -> dict[str, Path]:
    """Create and return the final article result directory layout."""
    results = cfg.path("results")
    dirs: dict[str, Path] = {
        "results": results,
        "metrics": results / "metrics",
        "checkpoints": results / "checkpoints",
        "logs": results / "logs",
        "report": results / "report",
        "figures": results / "figures",
    }
    for name in ARTICLE_FIGURE_DIRS:
        dirs[name] = dirs["figures"] / name
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    recall, precision = pr_points(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc_metric(y_true, y_prob):.3f}")
    ax.set_xlabel("Recall / sensibilidad")
    ax.set_ylabel("Precisión")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model_to_artifacts(
    model,
    df: pd.DataFrame,
    root: Path,
    cfg: Config,
    experiment_id: str,
    dirs: dict[str, Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run inference and persist metrics, predictions, and standard figures."""
    dirs = dirs or ensure_article_dirs(cfg)
    experiment_id = safe_id(experiment_id)
    metadata = metadata or {}

    loader = DataLoader(
        UltrasoundDataset(df.reset_index(drop=True), root, build_eval_transforms(cfg)),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory="cuda" in str(cfg.device),
        drop_last=False,
        persistent_workers=cfg.training.num_workers > 0,
    )
    y_true, y_prob, idx = run_inference(
        model,
        loader,
        cfg.device,
        cfg.training.mixed_precision,
    )
    metrics = compute_metrics(y_true, y_prob)
    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        **metadata,
        **metrics,
    }

    pred_path = dirs["metrics"] / f"{experiment_id}_predictions.csv"
    pd.DataFrame({"index": idx, "y_true": y_true, "y_prob_malignant": y_prob}).to_csv(
        pred_path, index=False
    )
    record["predictions_path"] = str(pred_path)

    metrics_path = dirs["metrics"] / f"{experiment_id}_metrics.json"
    save_json(record, metrics_path)
    record["metrics_path"] = str(metrics_path)

    cm = np.asarray(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        ["benign", "malignant"],
        dirs["confusion_matrices"] / f"{experiment_id}_cm.png",
        title=f"Matriz de confusión - {experiment_id}",
    )
    if len(np.unique(y_true)) > 1:
        fpr, tpr = roc_points(y_true, y_prob)
        plot_roc(
            fpr,
            tpr,
            metrics["auc"],
            dirs["roc_curves"] / f"{experiment_id}_roc.png",
            title=f"ROC - {experiment_id}",
        )
        _plot_pr_curve(
            y_true,
            y_prob,
            dirs["pr_curves"] / f"{experiment_id}_pr.png",
            title=f"Precision-Recall - {experiment_id}",
        )
        plot_reliability(
            y_true,
            y_prob,
            str(dirs["calibration_curves"] / f"{experiment_id}_calibration.png"),
            title=f"Calibración - {experiment_id}",
        )
    return record


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON metric record CSV-friendly without losing key scalar metrics."""
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key == "confusion_matrix":
            flat[key] = str(value)
        elif isinstance(value, (list, tuple, dict)):
            flat[key] = str(value)
        else:
            flat[key] = value
    return flat


def load_metric_records(metrics_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(metrics_dir).glob("*_metrics.json")):
        records.append(load_json(path))
    return records
