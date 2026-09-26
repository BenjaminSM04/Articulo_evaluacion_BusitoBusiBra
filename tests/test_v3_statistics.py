"""Synthetic tests for the isolated v3 threshold and comparison statistics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.publication_statistics import (
    PRIMARY_COMPARISON_METHODS,
    SECONDARY_COMPARISON_METHODS,
    compare_v3_method_families,
    describe_hyperparameter_sensitivity,
    threshold_sensitivity_panel,
)


def test_threshold_panel_evaluates_all_metrics_on_same_evaluation_cohort():
    evaluation_labels = np.array([0, 0, 1, 1])
    evaluation_probabilities = np.array([0.2, 0.8, 0.6, 0.9])
    source_val = (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    calibration = (np.array([0, 0, 1, 1]), np.array([0.2, 0.4, 0.55, 0.65]))

    rows = threshold_sensitivity_panel(
        evaluation_labels,
        evaluation_probabilities,
        source_val=source_val,
        calibration=calibration,
    )

    assert [row["threshold_source"] for row in rows] == [
        "fixed_0.5",
        "source_val",
        "caller_provided_calibration",
    ]
    assert all(row["evaluation_n"] == 4 for row in rows)
    assert all(row["nll"] == pytest.approx(rows[0]["nll"]) for row in rows)
    assert all(row["brier"] == pytest.approx(rows[0]["brier"]) for row in rows)
    assert all(row["ece"] == pytest.approx(rows[0]["ece"]) for row in rows)
    assert rows[0]["threshold"] == 0.5
    assert rows[1]["threshold"] == pytest.approx(0.8)
    assert rows[2]["threshold"] == pytest.approx(0.55)
    assert rows[0]["sensitivity"] == pytest.approx(1.0)
    assert rows[0]["specificity"] == pytest.approx(0.5)


def test_threshold_panel_omits_calibration_youden_without_explicit_data():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    rows = threshold_sensitivity_panel(
        labels,
        probabilities,
        source_val=(labels, probabilities),
    )

    assert [row["threshold_source"] for row in rows] == ["fixed_0.5", "source_val"]


def test_comparison_families_have_separate_six_test_holm_corrections():
    labels = np.array([0] * 12 + [1] * 12)
    baseline = np.concatenate((np.linspace(0.2, 0.6, 12), np.linspace(0.4, 0.8, 12)))
    stronger = np.concatenate((np.linspace(0.01, 0.2, 12), np.linspace(0.8, 0.99, 12)))
    predictions = {}
    for architecture in ("resnet18", "efficientnet_b0"):
        predictions[(architecture, "source_only_matched")] = (labels, baseline)
        for method in (*PRIMARY_COMPARISON_METHODS, *SECONDARY_COMPARISON_METHODS):
            predictions[(architecture, method)] = (labels, stronger)

    rows = compare_v3_method_families(predictions)

    assert len(rows) == 12
    for family in ("primary", "secondary"):
        family_rows = [row for row in rows if row["family"] == family]
        assert len(family_rows) == 6
        assert {row["n_holm_comparisons"] for row in family_rows} == {6}
        assert all(row["p_holm"] <= min(1.0, 6 * row["p_value"]) for row in family_rows)
    assert {row["family"] for row in rows} == {"primary", "secondary"}


def test_comparison_rejects_unaligned_labels():
    labels = np.array([0, 0, 1, 1])
    predictions = {
        ("resnet18", "source_only_matched"): (labels, np.array([0.1, 0.2, 0.8, 0.9])),
    }
    for method in (*PRIMARY_COMPARISON_METHODS, *SECONDARY_COMPARISON_METHODS):
        method_labels = np.array([0, 1, 0, 1]) if method == "dann" else labels
        predictions[("resnet18", method)] = (
            method_labels,
            np.array([0.1, 0.2, 0.8, 0.9]),
        )
        predictions[("efficientnet_b0", method)] = (labels, np.array([0.1, 0.2, 0.8, 0.9]))
    predictions[("efficientnet_b0", "source_only_matched")] = (
        labels,
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    with pytest.raises(ValueError, match="aligned labels"):
        compare_v3_method_families(
            predictions,
            architectures=("resnet18", "efficientnet_b0"),
        )


def test_hyperparameter_sensitivity_is_descriptive_without_adjusted_p_values():
    rows = describe_hyperparameter_sensitivity(
        [
            {"parameter": "lambda", "value": 0.1, "sensitivity": 0.8},
            {"parameter": "lambda", "value": 0.5, "sensitivity": 0.9},
        ]
    )

    assert rows[0]["value"] == 0.1
    assert rows[1]["value"] == 0.5
    assert all("p_holm" not in row and "p_adjusted" not in row for row in rows)
