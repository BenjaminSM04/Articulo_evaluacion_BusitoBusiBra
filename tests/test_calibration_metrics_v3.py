"""Probability quality metrics used by the v3 review controls."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import binary_nll, compute_metrics


def test_binary_nll_matches_hand_calculation_and_metric_panel() -> None:
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.6, 0.3])
    expected = -np.mean(np.log([0.9, 0.8, 0.6, 0.7]))

    assert binary_nll(labels, probabilities) == pytest.approx(expected)
    metrics = compute_metrics(labels, probabilities)
    assert metrics["nll"] == pytest.approx(expected)
    assert metrics["brier"] == pytest.approx(np.mean((labels - probabilities) ** 2))
    assert np.isfinite(metrics["ece"])


def test_binary_nll_clips_extreme_probabilities_and_rejects_invalid_inputs() -> None:
    assert np.isfinite(binary_nll(np.array([1, 0]), np.array([0.0, 1.0])))
    with pytest.raises(ValueError, match="probabilidades"):
        binary_nll(np.array([0, 1]), np.array([-0.1, 0.8]))
    with pytest.raises(ValueError, match="etiquetas"):
        binary_nll(np.array([0, 2]), np.array([0.1, 0.8]))
    with pytest.raises(ValueError, match="longitud"):
        binary_nll(np.array([0]), np.array([0.1, 0.8]))
