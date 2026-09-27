"""End-to-end synthetic contract for the frozen v3 prediction matrix."""

from __future__ import annotations

import importlib
import importlib.util
import json

import numpy as np
import pandas as pd

from src.training.experiment_matrix import build_experiment_matrix

ARCHITECTURES = ("resnet18", "efficientnet_b0")
SEEDS = (17, 42, 73, 101, 202)
COHORTS = (
    "source_val",
    "source_test",
    "target_calibration_images",
    "target_calibration_patients",
    "target_test_images",
    "target_test_patients",
)


def _write_matrix(root):
    jobs = build_experiment_matrix(architectures=ARCHITECTURES, seeds=SEEDS)
    index = pd.DataFrame(
        {
            "experiment_id": job.experiment_id,
            "family": job.family,
            "arch": job.arch,
            "seed": job.seed,
            "method": job.method,
            "hyperparameter": job.hyperparameter,
            "hyperparameter_value": job.value,
        }
        for job in jobs
    )
    for cohort in COHORTS:
        (root / cohort).mkdir(parents=True)
    for job_number, job in enumerate(jobs):
        for cohort_number, cohort in enumerate(COHORTS):
            source = cohort.startswith("source_")
            labels = np.array([0] * 12 + [1] * 12)
            rng = np.random.default_rng(9000 + job_number * 7 + cohort_number)
            logits = (labels * 2 - 1) * (0.55 + (job_number % 7) * 0.015)
            logits = logits + rng.normal(0, 0.5, len(labels))
            table = pd.DataFrame(
                {
                    "sample_id" if source or cohort.endswith("_images") else "patient_id": [
                        f"{cohort}_{n:03d}" for n in range(len(labels))
                    ],
                    "label": np.where(labels == 0, "benign", "malignant"),
                    "label_idx": labels,
                    "logit_difference": logits,
                    "arch": job.arch,
                    "method": job.method,
                    "seed": job.seed,
                    "finalization_fingerprint_sha256": "frozen-synthetic-run",
                }
            )
            if not source and cohort.endswith("_images"):
                table["patient_id"] = [
                    f"{cohort.replace('_images', '_patients')}_{n:03d}" for n in range(len(labels))
                ]
            table.to_csv(root / cohort / f"{job.experiment_id}.csv", index=False)
    return index


def test_v3_analysis_consumes_all_160_variants_and_separates_families(tmp_path):
    assert importlib.util.find_spec("src.evaluation.publication_statistics_v3") is not None
    v3 = importlib.import_module("src.evaluation.publication_statistics_v3")
    assert callable(getattr(v3, "analyze_v3_publication_predictions", None))
    predictions = tmp_path / "predictions"
    index = _write_matrix(predictions)

    outputs = v3.analyze_v3_publication_predictions(
        predictions,
        tmp_path / "analysis",
        checkpoint_index=index,
        architectures=ARCHITECTURES,
        seeds=SEEDS,
        n_bootstrap=20,
        confidence_level=0.95,
        bootstrap_seed=99,
        expected_finalization_fingerprint="frozen-synthetic-run",
    )

    variants = pd.read_csv(outputs["all_variant_metrics"])
    families = pd.read_csv(outputs["v3_method_families"])
    sensitivity = pd.read_csv(outputs["v3_hyperparameter_sensitivity"])
    thresholds = pd.read_csv(outputs["v3_threshold_sensitivity"])
    assert len(variants) == 160
    assert variants["experiment_id"].is_unique
    assert {"nll_raw", "nll_calibrated"}.issubset(variants)
    assert len(families) == 12
    assert families.groupby("family").size().to_dict() == {"primary": 6, "secondary": 6}
    assert set(families["n_holm_comparisons"]) == {6}
    assert np.isfinite(families[["ci_low", "ci_high", "bootstrap_p_value"]]).all().all()
    assert len(sensitivity) == 45
    assert set(sensitivity["value"]) == {0.5, 1.0, 2.0}
    assert not {"p_value", "p_holm"} & set(sensitivity)
    assert len(thresholds) == 60
    assert set(thresholds["threshold_source"]) == {
        "fixed_0.5",
        "source_val",
        "caller_provided_calibration",
    }
    assert set(
        zip(
            thresholds["threshold_source"],
            thresholds["selection_cohort"],
            thresholds["selection_unit"],
            strict=True,
        )
    ) == {
        ("fixed_0.5", "none", "none"),
        ("source_val", "source_val", "image"),
        ("caller_provided_calibration", "target_calibration", "patient"),
    }
    assert set(variants.loc[variants["family"].eq("intensity"), "method"]) == {
        "intensity_source_direct",
        "intensity",
    }
    report = outputs["markdown"].read_text(encoding="utf-8")
    metadata = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert "v2" not in report.lower()
    assert metadata["analysis"]["protocol_version"] == "v3"
    assert "two independent" in metadata["analysis"]["multiplicity"].lower()
    assert len(metadata["v3_method_families"]) == 12


def test_v3_analysis_rejects_incomplete_prediction_inventory(tmp_path):
    assert importlib.util.find_spec("src.evaluation.publication_statistics_v3") is not None
    v3 = importlib.import_module("src.evaluation.publication_statistics_v3")
    assert callable(getattr(v3, "analyze_v3_publication_predictions", None))
    predictions = tmp_path / "predictions"
    index = _write_matrix(predictions)
    missing = predictions / "target_test_patients" / "main_resnet18_seed17_dann.csv"
    missing.unlink()

    try:
        v3.analyze_v3_publication_predictions(
            predictions,
            tmp_path / "analysis",
            checkpoint_index=index,
            architectures=ARCHITECTURES,
            seeds=SEEDS,
            n_bootstrap=20,
            confidence_level=0.95,
            bootstrap_seed=99,
            expected_finalization_fingerprint="frozen-synthetic-run",
        )
    except FileNotFoundError as exc:
        assert "main_resnet18_seed17_dann.csv" in str(exc)
    else:
        raise AssertionError("Incomplete v3 prediction inventory was accepted")
