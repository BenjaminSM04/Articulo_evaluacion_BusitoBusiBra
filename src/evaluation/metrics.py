"""Métricas de clasificación, inferencia, bootstrap y test de DeLong.

Convención: la clase POSITIVA es 'maligno' (índice 1), porque la prioridad clínica
es no perder cánceres (sensibilidad = recall de maligno).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from .calibration import expected_calibration_error


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader, device: str = "cuda", amp: bool = True):
    """Devuelve (y_true, y_prob, indices). y_prob = P(clase = maligno)."""
    model.eval()
    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    ys, ps, idxs = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=dev_type, enabled=amp and dev_type == "cuda"):
            out = model(x)
            if isinstance(out, tuple):          # DANN -> (class_logits, domain_logits)
                out = out[0]
            prob = torch.softmax(out.float(), dim=1)[:, 1]
        ys.append(np.asarray(y))
        ps.append(prob.cpu().numpy())
        idxs.append(np.asarray(idx))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idxs)


def auc_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return roc_auc_score(y_true, y_prob)


def pr_auc_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return average_precision_score(y_true, y_prob)


def find_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Umbral que maximiza el índice de Youden (tpr - fpr) en el conjunto dado."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def binary_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean Bernoulli negative log likelihood with finite endpoint clipping."""

    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("NLL requiere vectores de igual longitud")
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("NLL requiere etiquetas binarias no vacias")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("NLL requiere probabilidades finitas entre 0 y 1")
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Panel completo de métricas para clasificación binaria benigno/maligno."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:                          # una sola clase presente
        auc = pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "nll": binary_nll(y_true, y_prob),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(tp + fn),
        "confusion_matrix": cm.tolist(),
    }


def roc_points(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def pr_points(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return recall, precision


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int = 1000, level: float = 0.95, seed: int = 42):
    """IC por bootstrap percentílico para una métrica escalar."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        sample = rng.integers(0, n, n)
        if len(np.unique(y_true[sample])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[sample], y_prob[sample]))
        except ValueError:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    alpha = (1 - level) / 2
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)))


# --------------------------------------------------------------------------- #
# Test de DeLong para comparar dos AUC correlacionados (mismo conjunto de test).
# Implementación fastDeLong (Sun & Xu, 2014). Devuelve (auc1, auc2, p-value).
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    """Compara AUC(prob_a) vs AUC(prob_b) sobre el MISMO y_true. Devuelve (auc_a, auc_b, p)."""
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind="mergesort")
    m = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))       # test bilateral, sin scipy
    return float(aucs[0]), float(aucs[1]), float(p)
