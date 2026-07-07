"""Calibración: ECE, curva de fiabilidad, Brier y temperature scaling.

Bajo domain shift es habitual que el modelo esté MAL calibrado en el objetivo; por eso
se reportan ECE y Brier sobre el target y se ofrece temperature scaling (ajustado en
validación, NUNCA en el test).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE con confianza = max(p, 1-p) y binning uniforme."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(brier_score_loss(y_true, np.asarray(y_prob, dtype=float)))


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Devuelve (confianza_media, frecuencia_observada, conteo) por bin, para P(maligno)."""
    p = np.asarray(y_prob, dtype=float)
    y = np.asarray(y_true).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        xs.append(float(p[mask].mean()))
        ys.append(float(y[mask].mean()))
        ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_reliability(y_true, y_prob, path: str, title: str = "Curva de calibración",
                     n_bins: int = 10) -> None:
    xs, ys, _ = reliability_curve(y_true, y_prob, n_bins)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfecta")
    ax.plot(xs, ys, "o-", label="Modelo")
    ax.set_xlabel("Confianza media predicha")
    ax.set_ylabel("Frecuencia observada de maligno")
    ax.set_title(f"{title} (ECE={expected_calibration_error(y_true, y_prob):.3f})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_temperature(logits, labels, max_iter: int = 100) -> float:
    """Ajusta un escalar de temperatura minimizando la NLL (Guo et al., 2017).

    logits: tensor/array (N, 2); labels: (N,). Devuelve T > 0. Ajustar en VALIDACIÓN.
    """
    import torch
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)     # optimiza log(T) para T>0
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())
