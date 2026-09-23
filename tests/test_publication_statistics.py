"""Synthetic tests for publication post-inference analysis across seed counts."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.publication_statistics import (
    COMPARATOR_METHOD,
    ENSEMBLE_METHODS,
    UDA_METHODS,
    align_seed_predictions,
    align_source_seed_predictions,
    analyze_publication_predictions,
    bootstrap_metric_distributions,
    build_confirmatory_ensemble,
    confirmatory_metric_values,
    holm_adjust,
    paired_delong_test,
    stratified_patient_bootstrap_indices,
    validate_locked_artifacts,
)


def _patient_table(
    patient_ids,
    labels,
    logits,
    *,
    architecture="resnet18",
    method=COMPARATOR_METHOD,
    seed=17,
):
    labels = np.asarray(labels, dtype=int)
    return pd.DataFrame(
        {
            "patient_id": [str(value) for value in patient_ids],
            "label": np.where(labels == 1, "malignant", "benign"),
            "label_idx": labels,
            "n_images": 1,
            "logit_difference": np.asarray(logits, dtype=float),
            "arch": architecture,
            "method": method,
            "seed": seed,
            "finalization_fingerprint_sha256": "test-fingerprint",
        }
    )


def _source_table(
    sample_ids,
    labels,
    logits,
    *,
    architecture="resnet18",
    method=COMPARATOR_METHOD,
    seed=17,
):
    labels = np.asarray(labels, dtype=int)
    return pd.DataFrame(
        {
            "sample_id": [str(value) for value in sample_ids],
            "label": np.where(labels == 1, "malignant", "benign"),
            "label_idx": labels,
            "logit_difference": np.asarray(logits, dtype=float),
            "arch": architecture,
            "method": method,
            "seed": seed,
            "finalization_fingerprint_sha256": "test-fingerprint",
        }
    )


def test_seed_alignment_uses_patient_ids_and_means_logits():
    seeds = (17, 42, 73)
    patients = ["p3", "p1", "p2", "p4"]
    labels = np.array([1, 0, 0, 1])
    calibration_tables = {}
    test_tables = {}
    for column, seed in enumerate(seeds):
        calibration_tables[seed] = _patient_table(
            patients,
            labels,
            np.arange(4) + column,
            seed=seed,
        ).sample(frac=1.0, random_state=seed)
        test_tables[seed] = _patient_table(
            [f"t{value}" for value in range(4)],
            labels,
            np.arange(4) + 0.5 * column,
            seed=seed,
        ).sample(frac=1.0, random_state=seed + 1)

    calibration = align_seed_predictions(calibration_tables, seeds=seeds)
    test = align_seed_predictions(test_tables, seeds=seeds)
    ensemble = build_confirmatory_ensemble(
        calibration,
        test,
        architecture="resnet18",
        method=COMPARATOR_METHOD,
    )

    assert calibration.patients["patient_id"].tolist() == sorted(patients)
    assert ensemble.test["logit_difference"].to_numpy() == pytest.approx(test.logits.mean(axis=1))
    assert ensemble.temperature > 0
    assert 0 <= ensemble.threshold <= 1


def test_alignment_rejects_label_disagreement():
    seeds = (17, 42, 73)
    tables = {seed: _patient_table(["p1", "p2"], [0, 1], [-1, 1], seed=seed) for seed in seeds}
    tables[73].loc[tables[73]["patient_id"].eq("p2"), ["label", "label_idx"]] = [
        "benign",
        0,
    ]
    with pytest.raises(ValueError, match="inconsistent"):
        align_seed_predictions(tables, seeds=seeds)


def test_source_alignment_uses_sample_id_not_row_order():
    seeds = (17, 42, 73)
    tables = {
        seed: _source_table(
            ["s3", "s1", "s2", "s4"],
            [1, 0, 0, 1],
            np.arange(4) + index,
            seed=seed,
        ).sample(frac=1.0, random_state=seed)
        for index, seed in enumerate(seeds)
    }
    aligned = align_source_seed_predictions(tables, seeds=seeds)
    assert aligned.images["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert aligned.logits.shape == (4, 3)


def test_shared_bootstrap_is_stratified_and_deterministic():
    labels = np.array([0] * 7 + [1] * 3)
    first = stratified_patient_bootstrap_indices(labels, n_bootstrap=100, seed=123)
    second = stratified_patient_bootstrap_indices(labels, n_bootstrap=100, seed=123)
    assert np.array_equal(first, second)
    assert first.shape == (100, 10)
    for row in first:
        sampled = labels[row]
        assert int((sampled == 0).sum()) == 7
        assert int((sampled == 1).sum()) == 3


def test_vectorized_bootstrap_matches_direct_metric_calculation():
    labels = np.array([0, 0, 0, 1, 1])
    raw = np.array([0.10, 0.35, 0.35, 0.60, 0.90])
    calibrated = np.array([0.05, 0.30, 0.30, 0.70, 0.95])
    plan = stratified_patient_bootstrap_indices(labels, n_bootstrap=12, seed=8)
    observed = bootstrap_metric_distributions(
        labels,
        raw,
        calibrated,
        0.5,
        plan,
    )
    for replicate, indices in enumerate(plan):
        expected = confirmatory_metric_values(
            labels[indices],
            raw[indices],
            calibrated[indices],
            0.5,
        )
        for metric, value in expected.items():
            assert observed[metric][replicate] == pytest.approx(value)


def test_delong_and_holm_detect_stronger_patient_scores():
    labels = np.array([0] * 30 + [1] * 30)
    comparator = np.concatenate((np.linspace(0.20, 0.80, 30), np.linspace(0.40, 0.90, 30)))
    method = np.concatenate((np.linspace(0.01, 0.25, 30), np.linspace(0.75, 0.99, 30)))
    result = paired_delong_test(labels, method, comparator)
    assert result["auc_method"] > result["auc_comparator"]
    assert result["delta_auc"] > 0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_locked_artifact_inventory_detects_missing_changed_and_escaping_files(tmp_path):
    artifact = tmp_path / "results" / "predictions" / "frozen.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen prediction\n")
    artifact_hashes = {
        "results/predictions/frozen.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()
    }
    inventory_hash = hashlib.sha256(
        json.dumps(
            artifact_hashes,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validate_locked_artifacts(
        tmp_path,
        artifact_hashes,
        expected_manifest_sha256=inventory_hash,
    )

    artifact.write_bytes(b"changed prediction\n")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_locked_artifacts(
            tmp_path,
            artifact_hashes,
            expected_manifest_sha256=inventory_hash,
        )
    artifact.unlink()
    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        validate_locked_artifacts(
            tmp_path,
            artifact_hashes,
            expected_manifest_sha256=inventory_hash,
        )

    escaping_hashes = {"../outside.csv": "0" * 64}
    escaping_inventory_hash = hashlib.sha256(
        json.dumps(
            escaping_hashes,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="escapes the project root"):
        validate_locked_artifacts(
            tmp_path,
            escaping_hashes,
            expected_manifest_sha256=escaping_inventory_hash,
        )


@pytest.mark.parametrize(
    ("seeds", "ensemble_variant", "ensemble_estimand"),
    [
        (
            (17, 42, 73),
            "three_seed_logit_ensemble",
            "mean_logit_difference_across_three_seeds",
        ),
        (
            (17, 42, 73, 101, 202),
            "5_seed_logit_ensemble",
            "mean_logit_difference_across_5_seeds",
        ),
    ],
)
def test_end_to_end_synthetic_analysis_writes_all_tables(
    tmp_path, seeds, ensemble_variant, ensemble_estimand
):
    prediction_root = tmp_path / "predictions"
    output_dir = tmp_path / "analysis"
    architecture = "resnet18"
    methods = (*ENSEMBLE_METHODS, "finetune_5pct")
    rng = np.random.default_rng(2026)
    calibration_labels = np.array([0] * 12 + [1] * 12)
    test_labels = np.array([0] * 20 + [1] * 20)

    for cohort, labels in (
        ("target_calibration_patients", calibration_labels),
        ("target_test_patients", test_labels),
    ):
        directory = prediction_root / cohort
        directory.mkdir(parents=True)
        patient_ids = [f"{cohort}_{index:03d}" for index in range(len(labels))]
        for method_index, method in enumerate(methods):
            separation = 0.7 + 0.25 * method_index
            for seed_index, seed in enumerate(seeds):
                logits = (
                    (2 * labels - 1) * separation
                    + 0.05 * seed_index
                    + rng.normal(0.0, 0.55, len(labels))
                )
                table = _patient_table(
                    patient_ids,
                    labels,
                    logits,
                    architecture=architecture,
                    method=method,
                    seed=seed,
                ).sample(frac=1.0, random_state=seed + method_index)
                table.to_csv(
                    directory / f"{architecture}_{method}_seed{seed}.csv",
                    index=False,
                )

    source_val_labels = np.array([0] * 12 + [1] * 8)
    source_test_labels = np.array([0] * 18 + [1] * 12)
    for cohort, labels in (
        ("source_val", source_val_labels),
        ("source_test", source_test_labels),
    ):
        directory = prediction_root / cohort
        directory.mkdir(parents=True)
        sample_ids = [f"{cohort}_{index:03d}" for index in range(len(labels))]
        for method_index, method in enumerate(ENSEMBLE_METHODS):
            for seed_index, seed in enumerate(seeds):
                logits = (
                    (2 * labels - 1) * (0.8 + 0.15 * method_index)
                    + 0.05 * seed_index
                    + rng.normal(0.0, 0.55, len(labels))
                )
                table = _source_table(
                    sample_ids,
                    labels,
                    logits,
                    architecture=architecture,
                    method=method,
                    seed=seed,
                ).sample(frac=1.0, random_state=seed + method_index)
                table.to_csv(
                    directory / f"{architecture}_{method}_seed{seed}.csv",
                    index=False,
                )

    outputs = analyze_publication_predictions(
        prediction_root,
        output_dir,
        architectures=[architecture],
        methods=methods,
        seeds=seeds,
        n_bootstrap=50,
        confidence_level=0.95,
        bootstrap_seed=99,
        require_six_uda_comparisons=False,
        expected_finalization_fingerprint="test-fingerprint",
    )
    assert all(path.exists() for path in outputs.values())

    per_seed = pd.read_csv(outputs["per_seed_metrics"])
    ensemble = pd.read_csv(outputs["ensemble_metrics"])
    source_ensemble = pd.read_csv(outputs["source_ensemble_metrics"])
    paired = pd.read_csv(outputs["paired_differences"])
    delong = pd.read_csv(outputs["delong_holm"])
    calibration = pd.read_csv(outputs["calibration"])
    diagnostics = pd.read_csv(outputs["target_calibration_diagnostics"])
    finetune_replicas = pd.read_csv(outputs["finetune_replica_metrics"])
    finetune_summary = pd.read_csv(outputs["finetune_summary"])
    assert len(per_seed) == len(methods) * len(seeds)
    assert len(ensemble) == len(ENSEMBLE_METHODS) * 10
    assert len(source_ensemble) == len(ENSEMBLE_METHODS) * 7
    assert len(paired) == len(UDA_METHODS) * 10
    assert len(delong) == len(UDA_METHODS)
    assert len(calibration) == len(methods) * len(seeds) + len(ENSEMBLE_METHODS)
    assert len(diagnostics) == len(ENSEMBLE_METHODS)
    assert set(delong["method"]) == set(UDA_METHODS)
    assert np.isfinite(ensemble[["estimate", "ci_low", "ci_high"]]).all().all()
    assert not (ensemble["method"] == "finetune_5pct").any()
    ft_rows = per_seed[per_seed["method"].eq("finetune_5pct")]
    assert set(ft_rows["replicate_variation"]) == {"training_cohort_and_seed"}
    assert not ft_rows["logit_ensemble_reported"].any()
    assert len(finetune_replicas) == len(seeds)
    assert set(finetune_summary["method"]) == {"finetune_5pct"}
    assert set(finetune_summary["n_seeds"]) == {len(seeds)}
    assert not list((output_dir / "ensemble_predictions").rglob("*finetune*.csv"))
    assert set(calibration[calibration["seed"].isna()]["variant"]) == {ensemble_variant}
    target_predictions = pd.read_csv(
        output_dir
        / "ensemble_predictions"
        / "target"
        / f"{architecture}_{ENSEMBLE_METHODS[0]}_target_test_patients.csv"
    )
    source_predictions = pd.read_csv(
        output_dir
        / "ensemble_predictions"
        / "source"
        / f"{architecture}_{ENSEMBLE_METHODS[0]}_source_test_images.csv"
    )
    assert set(target_predictions["variant"]) == {ensemble_variant}
    assert set(source_predictions["variant"]) == {ensemble_variant}
    metadata = json.loads(outputs["json"].read_text(encoding="utf-8"))["analysis"]
    assert metadata["seeds"] == list(seeds)
    assert metadata["ensemble"] == ensemble_estimand
    assert metadata["bootstrap_intervals_for"] == ensemble_variant

    plan = pd.read_json(outputs["bootstrap_plan"])
    assert int(plan.loc["n_bootstrap", "target_patient"]) == 50
    assert bool(
        plan.loc[
            "shared_across_all_valid_ensembles_and_metrics",
            "target_patient",
        ]
    )
    assert math.isclose(float(plan.loc["n_positive", "target_patient"]), 20)

    mismatched_path = (
        prediction_root
        / "target_calibration_patients"
        / f"{architecture}_{ENSEMBLE_METHODS[0]}_seed{seeds[0]}.csv"
    )
    mismatched = pd.read_csv(mismatched_path)
    mismatched["finalization_fingerprint_sha256"] = "different-run"
    mismatched.to_csv(mismatched_path, index=False)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        analyze_publication_predictions(
            prediction_root,
            tmp_path / "analysis_mismatch",
            architectures=[architecture],
            methods=methods,
            seeds=seeds,
            n_bootstrap=5,
            confidence_level=0.95,
            bootstrap_seed=99,
            require_six_uda_comparisons=False,
            expected_finalization_fingerprint="test-fingerprint",
        )
