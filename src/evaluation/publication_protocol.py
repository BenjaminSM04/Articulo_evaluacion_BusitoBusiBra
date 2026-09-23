"""Locked inference, calibration, and patient-level utilities for protocol v2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import UltrasoundDataset
from ..training.publication_protocol import load_publication_model
from ..training.transforms import build_eval_transforms
from .metrics import compute_metrics


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    nonnegative = x >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


@torch.no_grad()
def infer_logits(
    checkpoint_path: str | Path,
    df: pd.DataFrame,
    root: Path,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Infer two-class logits and retain stable sample/patient identifiers."""
    required = {"sample_id", "image_path", "label", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Inference manifest is missing {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        raise ValueError("Inference manifest has duplicated sample IDs.")

    model, checkpoint = load_publication_model(checkpoint_path, cfg)
    frame = df.reset_index(drop=True).copy()
    workers = int(cfg.training.num_workers)
    loader = DataLoader(
        UltrasoundDataset(frame, root, build_eval_transforms(cfg)),
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory="cuda" in str(cfg.device),
        persistent_workers=workers > 0,
    )
    logits_0, logits_1, indices = [], [], []
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"
    model.eval()
    for x, _, idx in loader:
        x = x.to(cfg.device, non_blocking=True)
        with torch.autocast(
            device_type=device_type,
            enabled=bool(cfg.training.mixed_precision and device_type == "cuda"),
        ):
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
        logits = logits.float().cpu().numpy()
        logits_0.append(logits[:, 0])
        logits_1.append(logits[:, 1])
        indices.append(np.asarray(idx, dtype=int))

    idx = np.concatenate(indices)
    if sorted(idx.tolist()) != list(range(len(frame))):
        raise RuntimeError("Inference did not return each manifest row exactly once.")
    order = np.argsort(idx)
    metadata_columns = [
        col
        for col in (
            "sample_id",
            "patient_id",
            "image_path",
            "label",
            "label_idx",
            "birads",
            "original_path",
        )
        if col in frame.columns
    ]
    out = frame.loc[:, metadata_columns].iloc[idx[order]].reset_index(drop=True)
    out["logit_benign"] = np.concatenate(logits_0)[order]
    out["logit_malignant"] = np.concatenate(logits_1)[order]
    out["logit_difference"] = out["logit_malignant"] - out["logit_benign"]
    out["probability_raw"] = sigmoid(out["logit_difference"].to_numpy())
    return out, checkpoint


def aggregate_patients(image_predictions: pd.DataFrame) -> pd.DataFrame:
    """Average malignant logit differences within patient (pre-specified primary rule)."""
    if "patient_id" not in image_predictions.columns:
        raise ValueError("Patient aggregation requires patient_id.")
    if image_predictions["patient_id"].astype(str).str.len().eq(0).any():
        raise ValueError("Patient aggregation found a missing patient_id.")
    inconsistent = image_predictions.groupby("patient_id")["label_idx"].nunique()
    if inconsistent.gt(1).any():
        raise ValueError("At least one patient has inconsistent outcome labels.")
    aggregations: dict[str, tuple[str, str]] = {
        "label_idx": ("label_idx", "first"),
        "label": ("label", "first"),
        "logit_difference": ("logit_difference", "mean"),
        "n_images": ("sample_id", "size"),
    }
    if "birads" in image_predictions.columns:
        aggregations["birads"] = ("birads", "first")
    out = (
        image_predictions.groupby("patient_id", as_index=False)
        .agg(**aggregations)
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    out["probability_raw"] = sigmoid(out["logit_difference"].to_numpy())
    return out


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit one positive temperature by minimising binary NLL on calibration patients."""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(logits) != len(y_true) or len(logits) == 0:
        raise ValueError("Temperature scaling requires aligned non-empty arrays.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Temperature scaling requires both outcome classes.")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probability = np.clip(sigmoid(logits / temperature), 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, probability, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimisation failed: {result.message}")
    return float(math.exp(float(result.x)))


def select_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose a finite Youden threshold on calibration data only."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Youden threshold requires both outcome classes.")
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = np.where(finite, tpr - fpr, -np.inf)
    threshold = float(thresholds[int(np.argmax(score))])
    return float(np.clip(threshold, 0.0, 1.0))


def risk_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for malignant *risk*, not top-label confidence."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (y_prob >= edges[index]) & (y_prob <= edges[index + 1])
        else:
            selected = (y_prob >= edges[index]) & (y_prob < edges[index + 1])
        if not selected.any():
            continue
        value += selected.mean() * abs(y_true[selected].mean() - y_prob[selected].mean())
    return float(value)


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope using a logistic recalibration model."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    try:
        # ``C=np.inf`` is sklearn's non-penalised formulation without the deprecated
        # ``penalty=None`` argument (equivalent for the lbfgs solver).
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10_000)
        model.fit(logit, y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan")


def prediction_metrics(
    table: pd.DataFrame,
    *,
    probability_column: str,
    threshold: float,
) -> dict[str, float]:
    y_true = table["label_idx"].to_numpy(dtype=int)
    y_prob = table[probability_column].to_numpy(dtype=float)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["risk_ece"] = risk_ece(y_true, y_prob)
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    return metrics


def calibrate_from_patient_table(patient_calibration: pd.DataFrame) -> dict[str, float]:
    y_true = patient_calibration["label_idx"].to_numpy(dtype=int)
    logits = patient_calibration["logit_difference"].to_numpy(dtype=float)
    temperature = fit_temperature(logits, y_true)
    calibrated = sigmoid(logits / temperature)
    threshold = select_youden_threshold(y_true, calibrated)
    raw_threshold = select_youden_threshold(y_true, sigmoid(logits))
    return {
        "temperature": float(temperature),
        "threshold_calibrated": float(threshold),
        "threshold_raw": float(raw_threshold),
    }


def apply_temperature(table: pd.DataFrame, temperature: float) -> pd.DataFrame:
    out = table.copy()
    out["probability_calibrated"] = sigmoid(
        out["logit_difference"].to_numpy(dtype=float) / float(temperature)
    )
    return out
