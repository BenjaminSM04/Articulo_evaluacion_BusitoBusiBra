"""Patient-level statistics for locked publication prediction cohorts.

This module consumes only the frozen patient-level prediction CSVs produced by
``scripts/12_finalize_publication_inference.py``. It never loads images or checkpoints.

For source/control/UDA methods, the ensemble estimator is the mean ``logit_difference`` across
the pre-specified seeds (three in v2, at least five in the new run). Fine-tuning replicas are not
ensemble-compatible because both their
labelled training cohort and seed vary; they are reported separately and summarised by mean and
standard deviation. Temperature and the Youden operating point are fitted on target-calibration
patients, then frozen before target-test metrics are calculated. All target-test bootstrap
intervals and paired differences use one shared, stratified set of patient resamples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .metrics import binary_nll
from .publication_protocol import (
    calibration_intercept_slope,
    fit_temperature,
    risk_ece,
    select_youden_threshold,
    sigmoid,
)

CONFIRMATORY_METRICS = (
    "auc_raw",
    "pr_auc_raw",
    "brier_raw",
    "brier_calibrated",
    "risk_ece_raw",
    "risk_ece_calibrated",
    "sensitivity_calibrated",
    "specificity_calibrated",
    "balanced_accuracy_calibrated",
    "f1_calibrated",
)
SOURCE_METRICS = (
    "auc_raw",
    "pr_auc_raw",
    "brier_raw",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "f1",
)
UDA_METHODS = ("dann", "coral", "mmd")
COMPARATOR_METHOD = "source_only_matched"
ENSEMBLE_METHODS = (
    "source_direct",
    COMPARATOR_METHOD,
    *UDA_METHODS,
)
PRIMARY_COMPARISON_METHODS = ("dann", "coral", "mmd")
SECONDARY_COMPARISON_METHODS = ("adabn", "intensity", "roi")
V3_ARCHITECTURES = ("resnet18", "efficientnet_b0")


def _canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_artifacts(
    root: str | Path,
    artifact_hashes: Any,
    *,
    expected_manifest_sha256: Any,
) -> None:
    """Verify the complete final-inference artifact inventory from its immutable lock."""
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Final inference lock lacks a non-empty artifact hash inventory.")
    if not isinstance(expected_manifest_sha256, str) or not expected_manifest_sha256.strip():
        raise ValueError("Final inference lock lacks the artifact inventory hash.")
    observed_manifest_sha256 = _canonical_record_sha256(artifact_hashes)
    if observed_manifest_sha256 != expected_manifest_sha256.strip():
        raise ValueError("Final inference artifact inventory does not match its locked hash.")

    resolved_root = Path(root).resolve()
    for relative, expected_sha256 in sorted(artifact_hashes.items()):
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Final inference artifact inventory contains an invalid path.")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
        ):
            raise ValueError(f"Final inference artifact {relative!r} has an invalid SHA-256.")
        raw_path = Path(relative.replace("\\", "/"))
        if raw_path.is_absolute() or raw_path.drive:
            raise ValueError(f"Locked artifact path must be project-relative: {relative!r}.")
        artifact_path = (resolved_root / raw_path).resolve()
        try:
            artifact_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Locked artifact path escapes the project root: {relative!r}."
            ) from exc
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Locked final-inference artifact is missing: {artifact_path}")
        if _sha256_file(artifact_path) != expected_sha256.lower():
            raise ValueError(f"Locked final-inference artifact changed: {relative}.")


@dataclass(frozen=True)
class AlignedSeedPredictions:
    """Canonical patient table and one aligned logit column per seed."""

    patients: pd.DataFrame
    logits: np.ndarray
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class AlignedSourcePredictions:
    """Canonical source-image table and one aligned logit column per seed."""

    images: pd.DataFrame
    logits: np.ndarray
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ConfirmatoryEnsemble:
    """Frozen calibration and target-test results for one architecture/method."""

    architecture: str
    method: str
    calibration: pd.DataFrame
    test: pd.DataFrame
    temperature: float
    threshold: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class SourceEnsemble:
    """Source ensemble evaluated by image with a validation-only threshold."""

    architecture: str
    method: str
    validation: pd.DataFrame
    test: pd.DataFrame
    threshold: float
    metrics: dict[str, float]


def _validate_patient_table(
    table: pd.DataFrame,
    *,
    source: str,
    expected_architecture: str | None = None,
    expected_method: str | None = None,
    expected_seed: int | None = None,
) -> pd.DataFrame:
    required = {"patient_id", "label", "label_idx", "logit_difference"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns {sorted(missing)}.")

    out = table.copy()
    if out["patient_id"].isna().any():
        raise ValueError(f"{source} contains a missing patient_id.")
    out["patient_id"] = out["patient_id"].astype(str)
    if out["patient_id"].str.strip().str.len().eq(0).any():
        raise ValueError(f"{source} contains an empty patient_id.")
    if out["patient_id"].duplicated().any():
        duplicated = out.loc[out["patient_id"].duplicated(), "patient_id"].iloc[0]
        raise ValueError(f"{source} contains duplicate patient_id={duplicated}.")

    out["label_idx"] = pd.to_numeric(out["label_idx"], errors="raise").astype(int)
    if not set(out["label_idx"]).issubset({0, 1}):
        raise ValueError(f"{source} contains a label_idx outside {{0, 1}}.")
    if out["label"].isna().any():
        raise ValueError(f"{source} contains a missing label.")
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    if out["label"].str.len().eq(0).any():
        raise ValueError(f"{source} contains an empty label.")
    expected_labels = out["label_idx"].map({0: "benign", 1: "malignant"})
    if not out["label"].equals(expected_labels):
        raise ValueError(f"{source} contains inconsistent label/label_idx values.")
    out["logit_difference"] = pd.to_numeric(out["logit_difference"], errors="raise").astype(float)
    if not np.isfinite(out["logit_difference"]).all():
        raise ValueError(f"{source} contains a non-finite logit_difference.")

    expected_metadata = {
        "arch": expected_architecture,
        "method": expected_method,
        "seed": expected_seed,
    }
    for column, expected in expected_metadata.items():
        if expected is None or column not in out.columns:
            continue
        values = out[column].dropna().astype(str).unique()
        if column == "seed":
            matches = len(values) == 1 and int(float(values[0])) == int(expected)
        else:
            matches = len(values) == 1 and values[0] == str(expected)
        if not matches:
            raise ValueError(
                f"{source} metadata {column} does not match expected value {expected}."
            )

    return out.sort_values("patient_id").reset_index(drop=True)


def read_patient_prediction_csv(
    path: str | Path,
    *,
    architecture: str,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """Read and validate one patient prediction file without coercing patient identifiers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, dtype={"patient_id": str, "label": str})
    return _validate_patient_table(
        table,
        source=str(path),
        expected_architecture=architecture,
        expected_method=method,
        expected_seed=seed,
    )


def _validate_source_image_table(
    table: pd.DataFrame,
    *,
    source: str,
    expected_architecture: str | None = None,
    expected_method: str | None = None,
    expected_seed: int | None = None,
) -> pd.DataFrame:
    required = {"sample_id", "label", "label_idx", "logit_difference"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns {sorted(missing)}.")
    out = table.copy()
    if out["sample_id"].isna().any():
        raise ValueError(f"{source} contains a missing sample_id.")
    out["sample_id"] = out["sample_id"].astype(str)
    if out["sample_id"].str.strip().str.len().eq(0).any():
        raise ValueError(f"{source} contains an empty sample_id.")
    if out["sample_id"].duplicated().any():
        duplicated = out.loc[out["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"{source} contains duplicate sample_id={duplicated}.")
    out["label_idx"] = pd.to_numeric(out["label_idx"], errors="raise").astype(int)
    if not set(out["label_idx"]).issubset({0, 1}):
        raise ValueError(f"{source} contains a label_idx outside {{0, 1}}.")
    if out["label"].isna().any():
        raise ValueError(f"{source} contains a missing label.")
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    expected_labels = out["label_idx"].map({0: "benign", 1: "malignant"})
    if not out["label"].equals(expected_labels):
        raise ValueError(f"{source} contains inconsistent label/label_idx values.")
    out["logit_difference"] = pd.to_numeric(out["logit_difference"], errors="raise").astype(float)
    if not np.isfinite(out["logit_difference"]).all():
        raise ValueError(f"{source} contains a non-finite logit_difference.")

    expected_metadata = {
        "arch": expected_architecture,
        "method": expected_method,
        "seed": expected_seed,
    }
    for column, expected in expected_metadata.items():
        if expected is None or column not in out.columns:
            continue
        values = out[column].dropna().astype(str).unique()
        if column == "seed":
            matches = len(values) == 1 and int(float(values[0])) == int(expected)
        else:
            matches = len(values) == 1 and values[0] == str(expected)
        if not matches:
            raise ValueError(
                f"{source} metadata {column} does not match expected value {expected}."
            )
    return out.sort_values("sample_id").reset_index(drop=True)


def read_source_prediction_csv(
    path: str | Path,
    *,
    architecture: str,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """Read one frozen source image-level prediction file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, dtype={"sample_id": str, "label": str})
    return _validate_source_image_table(
        table,
        source=str(path),
        expected_architecture=architecture,
        expected_method=method,
        expected_seed=seed,
    )


def _validated_ensemble_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if (len(ordered_seeds) != 3 and len(ordered_seeds) < 5) or len(
        set(ordered_seeds)
    ) != len(ordered_seeds):
        raise ValueError("An ensemble requires three historical or at least five distinct seeds.")
    return ordered_seeds


def _ensemble_variant(seeds: Sequence[int]) -> str:
    if len(seeds) == 3:
        return "three_seed_logit_ensemble"
    return f"{len(seeds)}_seed_logit_ensemble"


def _ensemble_estimand(seeds: Sequence[int]) -> str:
    if len(seeds) == 3:
        return "mean_logit_difference_across_three_seeds"
    return f"mean_logit_difference_across_{len(seeds)}_seeds"


def align_seed_predictions(
    tables: Mapping[int, pd.DataFrame],
    *,
    seeds: Sequence[int],
    source: str = "prediction tables",
) -> AlignedSeedPredictions:
    """Align seed predictions by patient ID and require identical outcomes.

    Rows may arrive in a different order, but every seed must contain exactly the same patients,
    numeric labels, textual labels, and (when present) number of images.
    """
    ordered_seeds = _validated_ensemble_seeds(seeds)
    if set(tables) != set(ordered_seeds):
        raise ValueError(
            f"{source} seeds do not match the frozen set: "
            f"expected={ordered_seeds}, observed={sorted(tables)}."
        )

    validated = {
        seed: _validate_patient_table(tables[seed], source=f"{source}/seed={seed}")
        for seed in ordered_seeds
    }
    first = validated[ordered_seeds[0]].set_index("patient_id", drop=False)
    patient_order = first.index
    metadata_columns = [
        column
        for column in ("patient_id", "label", "label_idx", "n_images", "birads")
        if column in first.columns
    ]
    patients = first.loc[:, metadata_columns].reset_index(drop=True)
    logits: list[np.ndarray] = []

    for seed in ordered_seeds:
        current = validated[seed].set_index("patient_id", drop=False)
        if set(current.index) != set(patient_order):
            missing = sorted(set(patient_order) - set(current.index))
            extra = sorted(set(current.index) - set(patient_order))
            raise ValueError(
                f"{source}/seed={seed} patient mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
        current = current.loc[patient_order]
        for column in ("label", "label_idx"):
            if not np.array_equal(current[column].to_numpy(), first[column].to_numpy()):
                raise ValueError(f"{source}/seed={seed} has inconsistent {column}.")
        if "n_images" in first.columns and "n_images" in current.columns:
            if not np.array_equal(current["n_images"].to_numpy(), first["n_images"].to_numpy()):
                raise ValueError(f"{source}/seed={seed} has inconsistent n_images.")
        logits.append(current["logit_difference"].to_numpy(dtype=float))

    return AlignedSeedPredictions(
        patients=patients,
        logits=np.column_stack(logits),
        seeds=ordered_seeds,
    )


def align_source_seed_predictions(
    tables: Mapping[int, pd.DataFrame],
    *,
    seeds: Sequence[int],
    source: str = "source prediction tables",
) -> AlignedSourcePredictions:
    """Align source-image prediction tables by stable sample ID and outcome."""
    ordered_seeds = _validated_ensemble_seeds(seeds)
    if set(tables) != set(ordered_seeds):
        raise ValueError(
            f"{source} seeds do not match the frozen set: "
            f"expected={ordered_seeds}, observed={sorted(tables)}."
        )
    validated = {
        seed: _validate_source_image_table(tables[seed], source=f"{source}/seed={seed}")
        for seed in ordered_seeds
    }
    first = validated[ordered_seeds[0]].set_index("sample_id", drop=False)
    sample_order = first.index
    metadata_columns = [
        column
        for column in ("sample_id", "label", "label_idx", "image_path")
        if column in first.columns
    ]
    images = first.loc[:, metadata_columns].reset_index(drop=True)
    logits: list[np.ndarray] = []
    for seed in ordered_seeds:
        current = validated[seed].set_index("sample_id", drop=False)
        if set(current.index) != set(sample_order):
            missing = sorted(set(sample_order) - set(current.index))
            extra = sorted(set(current.index) - set(sample_order))
            raise ValueError(
                f"{source}/seed={seed} sample mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
        current = current.loc[sample_order]
        for column in ("label", "label_idx"):
            if not np.array_equal(current[column].to_numpy(), first[column].to_numpy()):
                raise ValueError(f"{source}/seed={seed} has inconsistent {column}.")
        logits.append(current["logit_difference"].to_numpy(dtype=float))
    return AlignedSourcePredictions(
        images=images,
        logits=np.column_stack(logits),
        seeds=ordered_seeds,
    )


def _assert_same_cohort(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    source: str,
) -> None:
    for column in ("patient_id", "label", "label_idx"):
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(f"{source} does not align with the frozen cohort on {column}.")


def confirmatory_metric_values(
    y_true: np.ndarray,
    probability_raw: np.ndarray,
    probability_calibrated: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate pre-specified raw-risk and calibrated operating-point metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability_raw = np.asarray(probability_raw, dtype=float)
    probability_calibrated = np.asarray(probability_calibrated, dtype=float)
    if not (y_true.shape == probability_raw.shape == probability_calibrated.shape):
        raise ValueError("Metric arrays are not aligned.")
    if len(y_true) == 0 or len(np.unique(y_true)) != 2:
        raise ValueError("Confirmatory metrics require a non-empty two-class cohort.")
    if not (
        np.isfinite(probability_raw).all()
        and np.isfinite(probability_calibrated).all()
        and np.isfinite(threshold)
    ):
        raise ValueError("Metric inputs must be finite.")

    predicted = (probability_calibrated >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "auc_raw": float(roc_auc_score(y_true, probability_raw)),
        "pr_auc_raw": float(average_precision_score(y_true, probability_raw)),
        "brier_raw": float(brier_score_loss(y_true, probability_raw)),
        "brier_calibrated": float(brier_score_loss(y_true, probability_calibrated)),
        "risk_ece_raw": float(risk_ece(y_true, probability_raw)),
        "risk_ece_calibrated": float(risk_ece(y_true, probability_calibrated)),
        "sensitivity_calibrated": float(sensitivity),
        "specificity_calibrated": float(specificity),
        "balanced_accuracy_calibrated": float(balanced_accuracy_score(y_true, predicted)),
        "f1_calibrated": float(f1_score(y_true, predicted, zero_division=0)),
    }


def source_metric_values(
    y_true: np.ndarray,
    probability_raw: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate internal source-test metrics at a source-validation threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probability_raw = np.asarray(probability_raw, dtype=float)
    if y_true.shape != probability_raw.shape:
        raise ValueError("Source metric arrays are not aligned.")
    if len(y_true) == 0 or len(np.unique(y_true)) != 2:
        raise ValueError("Source metrics require a non-empty two-class cohort.")
    predicted = (probability_raw >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "auc_raw": float(roc_auc_score(y_true, probability_raw)),
        "pr_auc_raw": float(average_precision_score(y_true, probability_raw)),
        "brier_raw": float(brier_score_loss(y_true, probability_raw)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def fit_and_apply_calibration(
    calibration_logits: np.ndarray,
    calibration_labels: np.ndarray,
    test_logits: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit temperature and Youden only on calibration patients, then transform test logits."""
    calibration_logits = np.asarray(calibration_logits, dtype=float)
    calibration_labels = np.asarray(calibration_labels, dtype=int)
    test_logits = np.asarray(test_logits, dtype=float)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibration_probability = sigmoid(calibration_logits / temperature)
    threshold = select_youden_threshold(calibration_labels, calibration_probability)
    test_probability_raw = sigmoid(test_logits)
    test_probability_calibrated = sigmoid(test_logits / temperature)
    return (
        float(temperature),
        float(threshold),
        calibration_probability,
        test_probability_raw,
        test_probability_calibrated,
    )


def build_confirmatory_ensemble(
    calibration: AlignedSeedPredictions,
    test: AlignedSeedPredictions,
    *,
    architecture: str,
    method: str,
) -> ConfirmatoryEnsemble:
    """Average seed logits and freeze calibration before evaluating target test."""
    if calibration.seeds != test.seeds:
        raise ValueError("Calibration and test seeds differ.")
    calibration_logits = calibration.logits.mean(axis=1)
    test_logits = test.logits.mean(axis=1)
    calibration_labels = calibration.patients["label_idx"].to_numpy(dtype=int)
    test_labels = test.patients["label_idx"].to_numpy(dtype=int)
    temperature, threshold, cal_probability, raw_probability, calibrated_probability = (
        fit_and_apply_calibration(
            calibration_logits,
            calibration_labels,
            test_logits,
        )
    )

    calibration_table = calibration.patients.copy()
    calibration_table["logit_difference"] = calibration_logits
    calibration_table["probability_raw"] = sigmoid(calibration_logits)
    calibration_table["probability_calibrated"] = cal_probability
    test_table = test.patients.copy()
    test_table["logit_difference"] = test_logits
    test_table["probability_raw"] = raw_probability
    test_table["probability_calibrated"] = calibrated_probability
    for table, cohort in (
        (calibration_table, "target_calibration_patients"),
        (test_table, "target_test_patients"),
    ):
        table["arch"] = architecture
        table["method"] = method
        table["variant"] = _ensemble_variant(calibration.seeds)
        table["cohort"] = cohort
        table["temperature"] = temperature
        table["threshold_calibrated"] = threshold

    metrics = confirmatory_metric_values(
        test_labels,
        raw_probability,
        calibrated_probability,
        threshold,
    )
    return ConfirmatoryEnsemble(
        architecture=architecture,
        method=method,
        calibration=calibration_table,
        test=test_table,
        temperature=temperature,
        threshold=threshold,
        metrics=metrics,
    )


def build_source_ensemble(
    validation: AlignedSourcePredictions,
    test: AlignedSourcePredictions,
    *,
    architecture: str,
    method: str,
) -> SourceEnsemble:
    """Average seed logits and select the operating point only on source validation."""
    if validation.seeds != test.seeds:
        raise ValueError("Source validation and test seeds differ.")
    validation_logits = validation.logits.mean(axis=1)
    test_logits = test.logits.mean(axis=1)
    validation_probability = sigmoid(validation_logits)
    test_probability = sigmoid(test_logits)
    validation_labels = validation.images["label_idx"].to_numpy(dtype=int)
    test_labels = test.images["label_idx"].to_numpy(dtype=int)
    threshold = select_youden_threshold(validation_labels, validation_probability)

    validation_table = validation.images.copy()
    validation_table["logit_difference"] = validation_logits
    validation_table["probability_raw"] = validation_probability
    test_table = test.images.copy()
    test_table["logit_difference"] = test_logits
    test_table["probability_raw"] = test_probability
    for table, cohort in (
        (validation_table, "source_val_images"),
        (test_table, "source_test_images"),
    ):
        table["arch"] = architecture
        table["method"] = method
        table["variant"] = _ensemble_variant(validation.seeds)
        table["cohort"] = cohort
        table["threshold_source_val"] = threshold

    return SourceEnsemble(
        architecture=architecture,
        method=method,
        validation=validation_table,
        test=test_table,
        threshold=float(threshold),
        metrics=source_metric_values(test_labels, test_probability, threshold),
    )


def stratified_patient_bootstrap_indices(
    y_true: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 20260723,
) -> np.ndarray:
    """Create shared patient resamples with fixed class counts in every replicate."""
    y_true = np.asarray(y_true, dtype=int)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    classes = np.unique(y_true)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("Stratified bootstrap requires both binary outcome classes.")
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(int(seed))
    negative_draws = rng.choice(negative, size=(int(n_bootstrap), len(negative)), replace=True)
    positive_draws = rng.choice(positive, size=(int(n_bootstrap), len(positive)), replace=True)
    return np.concatenate((negative_draws, positive_draws), axis=1).astype(np.int32, copy=False)


def _bootstrap_risk_ece(
    sampled_labels: np.ndarray,
    sampled_probability: np.ndarray,
    *,
    bins: int = 10,
) -> np.ndarray:
    """Vectorised risk ECE for a matrix of bootstrap samples."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    n_replicates, n_observations = sampled_labels.shape
    values = np.zeros(n_replicates, dtype=float)
    for index in range(bins):
        if index == bins - 1:
            selected = (sampled_probability >= edges[index]) & (
                sampled_probability <= edges[index + 1]
            )
        else:
            selected = (sampled_probability >= edges[index]) & (
                sampled_probability < edges[index + 1]
            )
        counts = selected.sum(axis=1)
        nonempty = counts > 0
        observed = np.divide(
            (sampled_labels * selected).sum(axis=1),
            counts,
            out=np.zeros(n_replicates, dtype=float),
            where=nonempty,
        )
        predicted = np.divide(
            (sampled_probability * selected).sum(axis=1),
            counts,
            out=np.zeros(n_replicates, dtype=float),
            where=nonempty,
        )
        values += counts / n_observations * np.abs(observed - predicted)
    return values


def bootstrap_metric_distributions(
    y_true: np.ndarray,
    probability_raw: np.ndarray,
    probability_calibrated: np.ndarray,
    threshold: float,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate all metrics on an already-frozen shared bootstrap plan."""
    y_true = np.asarray(y_true, dtype=int)
    probability_raw = np.asarray(probability_raw, dtype=float)
    probability_calibrated = np.asarray(probability_calibrated, dtype=float)
    bootstrap_indices = np.asarray(bootstrap_indices, dtype=int)
    if bootstrap_indices.ndim != 2 or bootstrap_indices.shape[1] != len(y_true):
        raise ValueError("Bootstrap plan does not match the patient cohort.")
    if bootstrap_indices.size and (
        bootstrap_indices.min() < 0 or bootstrap_indices.max() >= len(y_true)
    ):
        raise ValueError("Bootstrap plan contains an out-of-range patient index.")

    sampled_labels = y_true[bootstrap_indices]
    sampled_raw = probability_raw[bootstrap_indices]
    sampled_calibrated = probability_calibrated[bootstrap_indices]
    predicted = sampled_calibrated >= float(threshold)
    positive = sampled_labels == 1
    negative = ~positive
    n_positive = positive.sum(axis=1)
    n_negative = negative.sum(axis=1)
    if np.any(n_positive == 0) or np.any(n_negative == 0):
        raise ValueError("Every stratified bootstrap replicate must contain both classes.")

    true_positive = (predicted & positive).sum(axis=1)
    false_positive = (predicted & negative).sum(axis=1)
    false_negative = ((~predicted) & positive).sum(axis=1)
    true_negative = ((~predicted) & negative).sum(axis=1)

    # Rank-sum AUC is vectorised across bootstrap rows. ``method='average'`` is important
    # because resampling necessarily creates tied copies of the same patient score.
    ranks = stats.rankdata(sampled_raw, axis=1, method="average")
    positive_rank_sum = np.where(positive, ranks, 0.0).sum(axis=1)
    auc = (positive_rank_sum - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative)

    # Average precision stays per row so tied threshold groups follow sklearn's exact
    # definition. The remaining metrics are fully vectorised.
    pr_auc = np.fromiter(
        (
            average_precision_score(sampled_labels[index], sampled_raw[index])
            for index in range(len(bootstrap_indices))
        ),
        dtype=float,
        count=len(bootstrap_indices),
    )
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.zeros(len(f1_denominator), dtype=float),
        where=f1_denominator != 0,
    )
    return {
        "auc_raw": auc.astype(float, copy=False),
        "pr_auc_raw": pr_auc,
        "brier_raw": np.mean((sampled_raw - sampled_labels) ** 2, axis=1),
        "brier_calibrated": np.mean((sampled_calibrated - sampled_labels) ** 2, axis=1),
        "risk_ece_raw": _bootstrap_risk_ece(sampled_labels, sampled_raw),
        "risk_ece_calibrated": _bootstrap_risk_ece(sampled_labels, sampled_calibrated),
        "sensitivity_calibrated": sensitivity.astype(float, copy=False),
        "specificity_calibrated": specificity.astype(float, copy=False),
        "balanced_accuracy_calibrated": 0.5 * (sensitivity + specificity),
        "f1_calibrated": f1,
    }


def bootstrap_source_metric_distributions(
    y_true: np.ndarray,
    probability_raw: np.ndarray,
    threshold: float,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Source-image bootstrap metrics using the threshold frozen on source validation."""
    target_named = bootstrap_metric_distributions(
        y_true,
        probability_raw,
        probability_raw,
        threshold,
        bootstrap_indices,
    )
    return {
        "auc_raw": target_named["auc_raw"],
        "pr_auc_raw": target_named["pr_auc_raw"],
        "brier_raw": target_named["brier_raw"],
        "sensitivity": target_named["sensitivity_calibrated"],
        "specificity": target_named["specificity_calibrated"],
        "balanced_accuracy": target_named["balanced_accuracy_calibrated"],
        "f1": target_named["f1_calibrated"],
    }


def percentile_interval(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval over finite bootstrap estimates."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def paired_bootstrap_pvalue(differences: np.ndarray) -> float:
    """Two-sided paired bootstrap sign p-value with a finite-sample correction."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")
    nonpositive = (np.count_nonzero(differences <= 0.0) + 1.0) / (len(differences) + 1.0)
    nonnegative = (np.count_nonzero(differences >= 0.0) + 1.0) / (len(differences) + 1.0)
    return float(min(1.0, 2.0 * min(nonpositive, nonnegative)))


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    out = np.empty(len(values), dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int):
    n_negative = predictions_sorted.shape[1] - n_positive
    positive = predictions_sorted[:, :n_positive]
    negative = predictions_sorted[:, n_positive:]
    tx = np.asarray([_midrank(row) for row in positive])
    ty = np.asarray([_midrank(row) for row in negative])
    tz = np.asarray([_midrank(row) for row in predictions_sorted])
    aucs = (
        tz[:, :n_positive].sum(axis=1) / n_positive / n_negative
        - (n_positive + 1.0) / 2.0 / n_negative
    )
    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    covariance = np.cov(v01) / n_positive + np.cov(v10) / n_negative
    return aucs, np.atleast_2d(covariance)


def paired_delong_test(
    y_true: np.ndarray,
    probability_method: np.ndarray,
    probability_comparator: np.ndarray,
) -> dict[str, float]:
    """Paired DeLong test over independent patient-level observations."""
    y_true = np.asarray(y_true, dtype=int)
    probability_method = np.asarray(probability_method, dtype=float)
    probability_comparator = np.asarray(probability_comparator, dtype=float)
    if not (y_true.shape == probability_method.shape == probability_comparator.shape):
        raise ValueError("DeLong inputs are not aligned.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("DeLong requires both outcome classes.")

    order = np.argsort(-y_true, kind="mergesort")
    n_positive = int(y_true.sum())
    predictions = np.vstack((probability_method, probability_comparator))[:, order]
    aucs, covariance = _fast_delong(predictions, n_positive)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    difference = float(aucs[0] - aucs[1])
    if variance <= np.finfo(float).eps:
        if math.isclose(difference, 0.0, abs_tol=1e-15):
            z_value, p_value = 0.0, 1.0
        else:
            z_value = math.copysign(float("inf"), difference)
            p_value = 0.0
    else:
        z_value = difference / math.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    return {
        "auc_method": float(aucs[0]),
        "auc_comparator": float(aucs[1]),
        "delta_auc": difference,
        "variance": variance,
        "z_value": float(z_value),
        "p_value": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving the original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite_positions = np.flatnonzero(np.isfinite(p_values))
    if len(finite_positions) == 0:
        return adjusted
    order = finite_positions[np.argsort(p_values[finite_positions], kind="mergesort")]
    running = 0.0
    total = len(order)
    for rank, position in enumerate(order):
        candidate = (total - rank) * p_values[position]
        running = max(running, candidate)
        adjusted[position] = min(1.0, running)
    return adjusted


def _validate_v3_binary_predictions(
    y_true: Sequence[int], y_prob: Sequence[float], *, source: str
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(f"{source} requires aligned one-dimensional labels and probabilities.")
    if not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{source} requires non-empty binary labels.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{source} requires finite probabilities between zero and one.")
    return labels.astype(int), probabilities


def threshold_sensitivity_panel(
    y_true_evaluation: Sequence[int],
    y_prob_evaluation: Sequence[float],
    *,
    source_val: tuple[Sequence[int], Sequence[float]],
    calibration: tuple[Sequence[int], Sequence[float]] | None = None,
    ece_bins: int = 10,
) -> list[dict[str, Any]]:
    """Evaluate fixed and validation-selected thresholds on one unchanged cohort.

    ``source_val`` must contain source validation labels/probabilities. The optional
    ``calibration`` pair is used only when explicitly supplied by the caller. NLL,
    Brier and ECE always describe the evaluation probabilities, independent of threshold.
    """
    labels, probabilities = _validate_v3_binary_predictions(
        y_true_evaluation,
        y_prob_evaluation,
        source="Evaluation cohort",
    )
    if ece_bins < 1:
        raise ValueError("ECE bin count must be positive.")
    selection_sets = [("fixed_0.5", None, None, 0.5)]
    validation_labels, validation_probabilities = _validate_v3_binary_predictions(
        source_val[0], source_val[1], source="source_val Youden selection"
    )
    if len(np.unique(validation_labels)) != 2:
        raise ValueError("Youden selection requires both outcome classes in source_val.")
    selection_sets.append(
        (
            "source_val",
            validation_labels,
            validation_probabilities,
            select_youden_threshold(validation_labels, validation_probabilities),
        )
    )
    if calibration is not None:
        calibration_labels, calibration_probabilities = _validate_v3_binary_predictions(
            calibration[0], calibration[1], source="caller-provided calibration Youden selection"
        )
        if len(np.unique(calibration_labels)) != 2:
            raise ValueError("Youden selection requires both outcome classes in calibration.")
        selection_sets.append(
            (
                "caller_provided_calibration",
                calibration_labels,
                calibration_probabilities,
                select_youden_threshold(calibration_labels, calibration_probabilities),
            )
        )

    brier = float(brier_score_loss(labels, probabilities))
    nll = binary_nll(labels, probabilities)
    ece = risk_ece(labels, probabilities, bins=ece_bins)
    rows: list[dict[str, Any]] = []
    for source, _, _, threshold in selection_sets:
        predicted = probabilities >= threshold
        positives = labels == 1
        negatives = labels == 0
        tp = int(np.count_nonzero(predicted & positives))
        tn = int(np.count_nonzero(~predicted & negatives))
        rows.append(
            {
                "threshold_source": source,
                "threshold": float(threshold),
                "sensitivity": float(tp / positives.sum()) if positives.any() else float("nan"),
                "specificity": float(tn / negatives.sum()) if negatives.any() else float("nan"),
                "nll": nll,
                "brier": brier,
                "ece": ece,
                "evaluation_n": int(len(labels)),
                "evaluation_n_positive": int(positives.sum()),
            }
        )
    return rows


def describe_hyperparameter_sensitivity(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy sensitivity-analysis observations as descriptive rows without p-values."""
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if "parameter" not in observation or "value" not in observation:
            raise ValueError("Sensitivity observations require parameter and value fields.")
        rows.append(
            {
                key: value
                for key, value in observation.items()
                if key not in {"p_value", "p_adjusted", "p_holm", "reject_holm_0_05"}
            }
        )
    return rows


def compare_v3_method_families(
    predictions: Mapping[tuple[str, str], tuple[Sequence[int], Sequence[float]]],
    *,
    architectures: Sequence[str] = V3_ARCHITECTURES,
) -> list[dict[str, Any]]:
    """Compare each v3 method with its matched baseline using family-wise Holm tests.

    Inputs are aligned evaluation-cohort labels/probabilities keyed by architecture and
    method. Primary and secondary DeLong p-values are adjusted in independent families.
    """
    architecture_values = tuple(architectures)
    if len(architecture_values) != 2 or len(set(architecture_values)) != 2:
        raise ValueError("v3 comparison requires exactly two distinct architectures.")
    family_methods = (
        ("primary", PRIMARY_COMPARISON_METHODS),
        ("secondary", SECONDARY_COMPARISON_METHODS),
    )
    rows: list[dict[str, Any]] = []
    for family, methods in family_methods:
        start = len(rows)
        for architecture in architecture_values:
            comparator_key = (architecture, COMPARATOR_METHOD)
            if comparator_key not in predictions:
                raise ValueError(f"Missing v3 prediction entry {comparator_key}.")
            comparator_labels, comparator_probabilities = _validate_v3_binary_predictions(
                *predictions[comparator_key], source=f"{architecture}/{COMPARATOR_METHOD}"
            )
            for method in methods:
                key = (architecture, method)
                if key not in predictions:
                    raise ValueError(f"Missing v3 prediction entry {key}.")
                labels, probabilities = _validate_v3_binary_predictions(
                    *predictions[key], source=f"{architecture}/{method}"
                )
                if not np.array_equal(labels, comparator_labels):
                    raise ValueError(
                        f"{architecture}/{method} and {COMPARATOR_METHOD} require aligned labels."
                    )
                comparison = paired_delong_test(
                    comparator_labels,
                    probabilities,
                    comparator_probabilities,
                )
                rows.append(
                    {
                        "family": family,
                        "arch": architecture,
                        "method": method,
                        "comparator": COMPARATOR_METHOD,
                        "n_evaluation": int(len(labels)),
                        **comparison,
                    }
                )
        family_rows = rows[start:]
        adjusted = holm_adjust([row["p_value"] for row in family_rows])
        for row, p_holm in zip(family_rows, adjusted, strict=True):
            row["p_holm"] = float(p_holm)
            row["reject_holm_0_05"] = bool(p_holm < 0.05)
            row["n_holm_comparisons"] = len(family_rows)
    return rows


def _load_cohort_seed_tables(
    prediction_root: Path,
    cohort: str,
    *,
    architecture: str,
    method: str,
    seeds: Sequence[int],
) -> dict[int, pd.DataFrame]:
    directory = prediction_root / cohort
    return {
        int(seed): read_patient_prediction_csv(
            directory / f"{architecture}_{method}_seed{int(seed)}.csv",
            architecture=architecture,
            method=method,
            seed=int(seed),
        )
        for seed in seeds
    }


def _load_source_seed_tables(
    prediction_root: Path,
    cohort: str,
    *,
    architecture: str,
    method: str,
    seeds: Sequence[int],
) -> dict[int, pd.DataFrame]:
    directory = prediction_root / cohort
    return {
        int(seed): read_source_prediction_csv(
            directory / f"{architecture}_{method}_seed{int(seed)}.csv",
            architecture=architecture,
            method=method,
            seed=int(seed),
        )
        for seed in seeds
    }


def _assert_prediction_fingerprint(
    tables: Mapping[int, pd.DataFrame],
    *,
    expected: str | None,
    source: str,
) -> None:
    """Require every input table to belong to the locked final-inference run."""
    if expected is None:
        return
    column = "finalization_fingerprint_sha256"
    for seed, table in tables.items():
        if column not in table.columns:
            raise ValueError(f"{source}/seed={seed} lacks {column}; provenance cannot be verified.")
        if table[column].isna().any():
            raise ValueError(f"{source}/seed={seed} contains a missing finalization fingerprint.")
        values = table[column].astype(str).str.strip().unique()
        if len(values) != 1 or values[0] != expected:
            raise ValueError(
                f"{source}/seed={seed} finalization fingerprint does not match the lock."
            )


def _seed_summary(per_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (architecture, method), group in per_seed_metrics.groupby(["arch", "method"], sort=True):
        for metric in CONFIRMATORY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "arch": architecture,
                    "method": method,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "replicate_variation": (
                        "training_cohort_and_seed"
                        if str(method).startswith("finetune_")
                        else "seed"
                    ),
                    "logit_ensemble_reported": method in ENSEMBLE_METHODS,
                }
            )
    return pd.DataFrame(rows)


def _dataframe_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records, mapping non-finite floats to null."""
    return json.loads(table.to_json(orient="records", double_precision=15))


def _write_markdown_report(
    path: Path,
    *,
    n_patients: int,
    n_positive: int,
    seeds: Sequence[int],
    n_bootstrap: int,
    seed_summary: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    delong: pd.DataFrame,
    paired_differences: pd.DataFrame,
    calibration: pd.DataFrame,
    target_calibration_diagnostics: pd.DataFrame,
    source_ensemble_metrics: pd.DataFrame,
) -> None:
    seed_auc = seed_summary[seed_summary["metric"].eq("auc_raw")].copy()
    seed_auc["media ± DE"] = seed_auc.apply(
        lambda row: f"{row['mean']:.3f} ± {row['sd']:.3f}", axis=1
    )
    target_display = ensemble_metrics[
        ensemble_metrics["metric"].isin(
            (
                "auc_raw",
                "pr_auc_raw",
                "brier_raw",
                "brier_calibrated",
                "risk_ece_raw",
                "risk_ece_calibrated",
            )
        )
    ].copy()
    target_display["estimación (IC95%)"] = target_display.apply(
        lambda row: (f"{row['estimate']:.3f} " f"({row['ci_low']:.3f}–{row['ci_high']:.3f})"),
        axis=1,
    )
    source_display = source_ensemble_metrics[
        source_ensemble_metrics["metric"].isin(("auc_raw", "pr_auc_raw", "brier_raw"))
    ].copy()
    source_display["estimación (IC95%)"] = source_display.apply(
        lambda row: (f"{row['estimate']:.3f} " f"({row['ci_low']:.3f}–{row['ci_high']:.3f})"),
        axis=1,
    )
    paired_auc = paired_differences[paired_differences["metric"].eq("auc_raw")].copy()
    finetune_seed_auc = seed_auc[seed_auc["method"].str.startswith("finetune_")]

    sections = [
        "# Análisis confirmatorio de publicación v2",
        "",
        (
            f"Unidad primaria: paciente. Test congelado: {n_patients} pacientes "
            f"({n_positive} malignos). Ensamble: media de `logit_difference` de las "
            f"semillas {list(seeds)}."
        ),
        (
            f"Los parámetros de temperatura y Youden se ajustaron exclusivamente con "
            f"`target_calibration_patients`. Todos los IC y diferencias del conjunto objetivo "
            f"usan los mismos {n_bootstrap} remuestreos estratificados por paciente."
        ),
        (
            "Los ensambles se limitan a `source_direct`, `source_only_matched`, DANN, CORAL y "
            "MMD. Fine-tuning no se ensambla: cada réplica varía simultáneamente la cohorte "
            "etiquetada y la semilla, por lo que solo se resume como media ± DE."
        ),
        "",
        "## AUC por semilla",
        "",
        seed_auc[["arch", "method", "media ± DE", "replicate_variation"]].to_markdown(index=False),
        "",
        "## Fine-tuning: réplicas, sin ensamble",
        "",
        finetune_seed_auc[["arch", "method", "media ± DE", "replicate_variation"]].to_markdown(
            index=False
        ),
        "",
        "## Métricas objetivo de ensambles válidos",
        "",
        target_display[["arch", "method", "metric", "estimación (IC95%)"]].to_markdown(index=False),
        "",
        "## Rendimiento interno fuente por imagen",
        "",
        (
            "El umbral se seleccionó con el ensamble sobre `source_val` y se congeló "
            "antes de evaluar `source_test`. Los IC usan bootstrap estratificado por imagen."
        ),
        "",
        source_display[["arch", "method", "metric", "estimación (IC95%)"]].to_markdown(index=False),
        "",
        "## DeLong pareado y corrección de Holm",
        "",
        delong[
            [
                "arch",
                "method",
                "comparator",
                "delta_auc",
                "p_value",
                "p_holm",
                "reject_holm_0_05",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Diferencias pareadas de AUC por bootstrap",
        "",
        paired_auc[
            [
                "arch",
                "method",
                "comparator",
                "estimate_difference",
                "ci_low",
                "ci_high",
                "bootstrap_p_value",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Calibración congelada",
        "",
        calibration[
            ["arch", "method", "variant", "temperature", "threshold_calibrated"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Diagnóstico descriptivo de calibración en test objetivo",
        "",
        target_calibration_diagnostics.to_markdown(index=False, floatfmt=".4f"),
        "",
        (
            "Las diferencias se expresan como método menos comparador. Para Brier, un valor "
            "negativo favorece al método; para ECE, un valor negativo también indica menor error "
            "de calibración; para las demás métricas, un valor positivo lo favorece."
        ),
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def analyze_publication_predictions(
    prediction_root: str | Path,
    output_dir: str | Path,
    *,
    architectures: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260723,
    require_six_uda_comparisons: bool = True,
    expected_finalization_fingerprint: str | None = None,
) -> dict[str, Path]:
    """Run and persist the full confirmatory post-inference analysis."""
    prediction_root = Path(prediction_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_prediction_dir = output_dir / "ensemble_predictions"
    ensemble_prediction_dir.mkdir(parents=True, exist_ok=True)

    architectures = tuple(str(value) for value in architectures)
    methods = tuple(str(value) for value in methods)
    seeds = _validated_ensemble_seeds(seeds)
    if expected_finalization_fingerprint is not None:
        expected_finalization_fingerprint = str(expected_finalization_fingerprint).strip()
        if not expected_finalization_fingerprint:
            raise ValueError("Expected finalization fingerprint cannot be empty.")
    if COMPARATOR_METHOD not in methods:
        raise ValueError(f"Missing comparator method {COMPARATOR_METHOD}.")
    missing_uda = set(UDA_METHODS) - set(methods)
    if missing_uda:
        raise ValueError(f"Missing UDA methods: {sorted(missing_uda)}.")
    expected_comparisons = len(architectures) * len(UDA_METHODS)
    if require_six_uda_comparisons and expected_comparisons != 6:
        raise ValueError(
            f"Frozen protocol requires six UDA comparisons, got {expected_comparisons}."
        )

    missing_ensemble_methods = set(ENSEMBLE_METHODS) - set(methods)
    if missing_ensemble_methods:
        raise ValueError(
            f"Missing methods required for valid ensembles: {sorted(missing_ensemble_methods)}."
        )
    target_ensemble_prediction_dir = ensemble_prediction_dir / "target"
    source_ensemble_prediction_dir = ensemble_prediction_dir / "source"
    target_ensemble_prediction_dir.mkdir(parents=True, exist_ok=True)
    source_ensemble_prediction_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_diagnostic_rows: list[dict[str, Any]] = []
    ensembles: dict[tuple[str, str], ConfirmatoryEnsemble] = {}
    source_ensembles: dict[tuple[str, str], SourceEnsemble] = {}
    reference_calibration: pd.DataFrame | None = None
    reference_test: pd.DataFrame | None = None
    reference_source_val: pd.DataFrame | None = None
    reference_source_test: pd.DataFrame | None = None

    # Target per-seed metrics are reported for every method. Only methods whose runs
    # estimate the same intervention are allowed to form a logit ensemble.
    for architecture in architectures:
        for method in methods:
            calibration_tables = _load_cohort_seed_tables(
                prediction_root,
                "target_calibration_patients",
                architecture=architecture,
                method=method,
                seeds=seeds,
            )
            _assert_prediction_fingerprint(
                calibration_tables,
                expected=expected_finalization_fingerprint,
                source=f"{architecture}/{method}/target_calibration",
            )
            calibration = align_seed_predictions(
                calibration_tables,
                seeds=seeds,
                source=f"{architecture}/{method}/target_calibration",
            )
            test_tables = _load_cohort_seed_tables(
                prediction_root,
                "target_test_patients",
                architecture=architecture,
                method=method,
                seeds=seeds,
            )
            _assert_prediction_fingerprint(
                test_tables,
                expected=expected_finalization_fingerprint,
                source=f"{architecture}/{method}/target_test",
            )
            test = align_seed_predictions(
                test_tables,
                seeds=seeds,
                source=f"{architecture}/{method}/target_test",
            )
            if reference_calibration is None:
                reference_calibration = calibration.patients
                reference_test = test.patients
                if set(reference_calibration["patient_id"]) & set(reference_test["patient_id"]):
                    raise ValueError("Calibration and test cohorts share a patient.")
            else:
                _assert_same_cohort(
                    reference_calibration,
                    calibration.patients,
                    source=f"{architecture}/{method}/target_calibration",
                )
                _assert_same_cohort(
                    reference_test,
                    test.patients,
                    source=f"{architecture}/{method}/target_test",
                )

            calibration_labels = calibration.patients["label_idx"].to_numpy(dtype=int)
            test_labels = test.patients["label_idx"].to_numpy(dtype=int)
            for column, seed in enumerate(seeds):
                (
                    temperature,
                    threshold,
                    _,
                    probability_raw,
                    probability_calibrated,
                ) = fit_and_apply_calibration(
                    calibration.logits[:, column],
                    calibration_labels,
                    test.logits[:, column],
                )
                per_seed_rows.append(
                    {
                        "arch": architecture,
                        "method": method,
                        "seed": int(seed),
                        "replicate_variation": (
                            "training_cohort_and_seed" if method.startswith("finetune_") else "seed"
                        ),
                        "logit_ensemble_reported": method in ENSEMBLE_METHODS,
                        "n_test_patients": int(len(test_labels)),
                        "n_test_positive": int(test_labels.sum()),
                        **confirmatory_metric_values(
                            test_labels,
                            probability_raw,
                            probability_calibrated,
                            threshold,
                        ),
                    }
                )
                calibration_rows.append(
                    {
                        "arch": architecture,
                        "method": method,
                        "variant": f"seed_{seed}",
                        "seed": int(seed),
                        "temperature": temperature,
                        "threshold_calibrated": threshold,
                        "n_calibration_patients": int(len(calibration_labels)),
                    }
                )

            if method not in ENSEMBLE_METHODS:
                continue
            ensemble = build_confirmatory_ensemble(
                calibration,
                test,
                architecture=architecture,
                method=method,
            )
            ensembles[(architecture, method)] = ensemble
            calibration_rows.append(
                {
                    "arch": architecture,
                    "method": method,
                    "variant": _ensemble_variant(seeds),
                    "seed": "",
                    "temperature": ensemble.temperature,
                    "threshold_calibrated": ensemble.threshold,
                    "n_calibration_patients": int(len(ensemble.calibration)),
                }
            )
            raw_intercept, raw_slope = calibration_intercept_slope(
                test_labels,
                ensemble.test["probability_raw"].to_numpy(dtype=float),
            )
            calibrated_intercept, calibrated_slope = calibration_intercept_slope(
                test_labels,
                ensemble.test["probability_calibrated"].to_numpy(dtype=float),
            )
            calibration_diagnostic_rows.append(
                {
                    "arch": architecture,
                    "method": method,
                    "n_test_patients": int(len(test_labels)),
                    "raw_intercept": raw_intercept,
                    "raw_slope": raw_slope,
                    "calibrated_intercept": calibrated_intercept,
                    "calibrated_slope": calibrated_slope,
                    "brier_raw": ensemble.metrics["brier_raw"],
                    "brier_calibrated": ensemble.metrics["brier_calibrated"],
                    "risk_ece_raw": ensemble.metrics["risk_ece_raw"],
                    "risk_ece_calibrated": ensemble.metrics["risk_ece_calibrated"],
                    "diagnostic_scope": "descriptive_target_test",
                }
            )
            ensemble.calibration.to_csv(
                target_ensemble_prediction_dir
                / f"{architecture}_{method}_target_calibration_patients.csv",
                index=False,
            )
            ensemble.test.to_csv(
                target_ensemble_prediction_dir
                / f"{architecture}_{method}_target_test_patients.csv",
                index=False,
            )

    # Internal source performance is an image-level analysis. Fine-tuning is intentionally
    # excluded because its replicas use different labelled target cohorts.
    for architecture in architectures:
        for method in ENSEMBLE_METHODS:
            source_val_tables = _load_source_seed_tables(
                prediction_root,
                "source_val",
                architecture=architecture,
                method=method,
                seeds=seeds,
            )
            _assert_prediction_fingerprint(
                source_val_tables,
                expected=expected_finalization_fingerprint,
                source=f"{architecture}/{method}/source_val",
            )
            source_val = align_source_seed_predictions(
                source_val_tables,
                seeds=seeds,
                source=f"{architecture}/{method}/source_val",
            )
            source_test_tables = _load_source_seed_tables(
                prediction_root,
                "source_test",
                architecture=architecture,
                method=method,
                seeds=seeds,
            )
            _assert_prediction_fingerprint(
                source_test_tables,
                expected=expected_finalization_fingerprint,
                source=f"{architecture}/{method}/source_test",
            )
            source_test = align_source_seed_predictions(
                source_test_tables,
                seeds=seeds,
                source=f"{architecture}/{method}/source_test",
            )
            if reference_source_val is None:
                reference_source_val = source_val.images
                reference_source_test = source_test.images
                if set(reference_source_val["sample_id"]) & set(reference_source_test["sample_id"]):
                    raise ValueError("Source validation and test share a sample.")
            else:
                for reference, candidate, cohort in (
                    (reference_source_val, source_val.images, "source_val"),
                    (reference_source_test, source_test.images, "source_test"),
                ):
                    for column in ("sample_id", "label", "label_idx"):
                        if not np.array_equal(
                            reference[column].to_numpy(),
                            candidate[column].to_numpy(),
                        ):
                            raise ValueError(
                                f"{architecture}/{method}/{cohort} does not align on {column}."
                            )
            source_ensemble = build_source_ensemble(
                source_val,
                source_test,
                architecture=architecture,
                method=method,
            )
            source_ensembles[(architecture, method)] = source_ensemble
            source_ensemble.validation.to_csv(
                source_ensemble_prediction_dir / f"{architecture}_{method}_source_val_images.csv",
                index=False,
            )
            source_ensemble.test.to_csv(
                source_ensemble_prediction_dir / f"{architecture}_{method}_source_test_images.csv",
                index=False,
            )

    if (
        reference_test is None
        or reference_calibration is None
        or reference_source_test is None
        or reference_source_val is None
    ):
        raise ValueError("No complete prediction matrix was analysed.")
    test_labels = reference_test["label_idx"].to_numpy(dtype=int)
    target_bootstrap_indices = stratified_patient_bootstrap_indices(
        test_labels,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    distributions: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    ensemble_metric_rows: list[dict[str, Any]] = []
    for key, ensemble in ensembles.items():
        probability_raw = ensemble.test["probability_raw"].to_numpy(dtype=float)
        probability_calibrated = ensemble.test["probability_calibrated"].to_numpy(dtype=float)
        distribution = bootstrap_metric_distributions(
            test_labels,
            probability_raw,
            probability_calibrated,
            ensemble.threshold,
            target_bootstrap_indices,
        )
        distributions[key] = distribution
        for metric in CONFIRMATORY_METRICS:
            lower, upper = percentile_interval(
                distribution[metric],
                confidence_level=confidence_level,
            )
            ensemble_metric_rows.append(
                {
                    "arch": key[0],
                    "method": key[1],
                    "metric": metric,
                    "estimate": ensemble.metrics[metric],
                    "ci_low": lower,
                    "ci_high": upper,
                    "confidence_level": confidence_level,
                    "n_bootstrap": int(n_bootstrap),
                    "n_test_patients": int(len(test_labels)),
                    "n_test_positive": int(test_labels.sum()),
                    "probability_scale": (
                        "raw" if metric.endswith("_raw") else "temperature_scaled"
                    ),
                    "unit": "patient",
                }
            )

    source_test_labels = reference_source_test["label_idx"].to_numpy(dtype=int)
    source_bootstrap_indices = stratified_patient_bootstrap_indices(
        source_test_labels,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    source_metric_rows: list[dict[str, Any]] = []
    for key, ensemble in source_ensembles.items():
        probability = ensemble.test["probability_raw"].to_numpy(dtype=float)
        source_distribution = bootstrap_source_metric_distributions(
            source_test_labels,
            probability,
            ensemble.threshold,
            source_bootstrap_indices,
        )
        for metric in SOURCE_METRICS:
            lower, upper = percentile_interval(
                source_distribution[metric],
                confidence_level=confidence_level,
            )
            source_metric_rows.append(
                {
                    "arch": key[0],
                    "method": key[1],
                    "metric": metric,
                    "estimate": ensemble.metrics[metric],
                    "ci_low": lower,
                    "ci_high": upper,
                    "threshold_selected_on_source_val": ensemble.threshold,
                    "confidence_level": confidence_level,
                    "n_bootstrap": int(n_bootstrap),
                    "n_test_images": int(len(source_test_labels)),
                    "n_test_positive": int(source_test_labels.sum()),
                    "unit": "image",
                }
            )

    paired_rows: list[dict[str, Any]] = []
    delong_rows: list[dict[str, Any]] = []
    for architecture in architectures:
        comparator = ensembles[(architecture, COMPARATOR_METHOD)]
        comparator_distribution = distributions[(architecture, COMPARATOR_METHOD)]
        comparator_probability = comparator.test["probability_raw"].to_numpy(dtype=float)
        for method in UDA_METHODS:
            ensemble = ensembles[(architecture, method)]
            method_distribution = distributions[(architecture, method)]
            for metric in CONFIRMATORY_METRICS:
                difference_distribution = (
                    method_distribution[metric] - comparator_distribution[metric]
                )
                lower, upper = percentile_interval(
                    difference_distribution,
                    confidence_level=confidence_level,
                )
                paired_rows.append(
                    {
                        "arch": architecture,
                        "method": method,
                        "comparator": COMPARATOR_METHOD,
                        "metric": metric,
                        "estimate_method": ensemble.metrics[metric],
                        "estimate_comparator": comparator.metrics[metric],
                        "estimate_difference": (
                            ensemble.metrics[metric] - comparator.metrics[metric]
                        ),
                        "ci_low": lower,
                        "ci_high": upper,
                        "bootstrap_p_value": paired_bootstrap_pvalue(difference_distribution),
                        "n_bootstrap": int(n_bootstrap),
                        "higher_is_better": not metric.startswith(("brier_", "risk_ece_")),
                        "unit": "patient",
                    }
                )

            delong = paired_delong_test(
                test_labels,
                ensemble.test["probability_raw"].to_numpy(dtype=float),
                comparator_probability,
            )
            delong_rows.append(
                {
                    "arch": architecture,
                    "method": method,
                    "comparator": COMPARATOR_METHOD,
                    "n_patients": int(len(test_labels)),
                    "n_positive": int(test_labels.sum()),
                    "unit": "patient",
                    **delong,
                }
            )

    delong_table = pd.DataFrame(delong_rows)
    delong_table["p_holm"] = holm_adjust(delong_table["p_value"].to_numpy())
    delong_table["reject_holm_0_05"] = delong_table["p_holm"] < 0.05
    delong_table["n_holm_comparisons"] = int(len(delong_table))
    if require_six_uda_comparisons and len(delong_table) != 6:
        raise RuntimeError(f"Expected six DeLong comparisons, wrote {len(delong_table)}.")

    per_seed_table = pd.DataFrame(per_seed_rows).sort_values(["arch", "method", "seed"])
    seed_summary = _seed_summary(per_seed_table)
    finetune_replica_metrics = per_seed_table[
        per_seed_table["method"].str.startswith("finetune_")
    ].copy()
    finetune_summary = seed_summary[seed_summary["method"].str.startswith("finetune_")].copy()
    ensemble_metrics = pd.DataFrame(ensemble_metric_rows).sort_values(["arch", "method", "metric"])
    source_ensemble_metrics = pd.DataFrame(source_metric_rows).sort_values(
        ["arch", "method", "metric"]
    )
    paired_differences = pd.DataFrame(paired_rows).sort_values(["arch", "method", "metric"])
    calibration_table = pd.DataFrame(calibration_rows).sort_values(["arch", "method", "variant"])
    target_calibration_diagnostics = pd.DataFrame(calibration_diagnostic_rows).sort_values(
        ["arch", "method"]
    )

    output_paths = {
        "per_seed_metrics": output_dir / "per_seed_metrics.csv",
        "per_seed_summary": output_dir / "per_seed_summary.csv",
        "finetune_replica_metrics": output_dir / "finetune_replica_metrics.csv",
        "finetune_summary": output_dir / "finetune_summary.csv",
        "ensemble_metrics": output_dir / "ensemble_metrics.csv",
        "source_ensemble_metrics": output_dir / "source_ensemble_metrics.csv",
        "paired_differences": output_dir / "paired_bootstrap_differences.csv",
        "delong_holm": output_dir / "delong_holm.csv",
        "calibration": output_dir / "calibration_parameters.csv",
        "target_calibration_diagnostics": (output_dir / "target_calibration_diagnostics.csv"),
        "bootstrap_plan": output_dir / "bootstrap_plan.json",
        "json": output_dir / "confirmatory_analysis.json",
        "markdown": output_dir / "confirmatory_analysis.md",
    }
    tables = {
        "per_seed_metrics": per_seed_table,
        "per_seed_summary": seed_summary,
        "finetune_replica_metrics": finetune_replica_metrics,
        "finetune_summary": finetune_summary,
        "ensemble_metrics": ensemble_metrics,
        "source_ensemble_metrics": source_ensemble_metrics,
        "paired_differences": paired_differences,
        "delong_holm": delong_table,
        "calibration": calibration_table,
        "target_calibration_diagnostics": target_calibration_diagnostics,
    }
    for key, table in tables.items():
        table.to_csv(output_paths[key], index=False)

    patient_hash = hashlib.sha256(
        "\n".join(reference_test["patient_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    target_bootstrap_hash = hashlib.sha256(target_bootstrap_indices.tobytes()).hexdigest()
    source_sample_hash = hashlib.sha256(
        "\n".join(reference_source_test["sample_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    source_bootstrap_hash = hashlib.sha256(source_bootstrap_indices.tobytes()).hexdigest()
    bootstrap_plan = {
        "target_patient": {
            "unit": "patient",
            "stratified_by": "label_idx",
            "shared_across_all_valid_ensembles_and_metrics": True,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "n_observations": int(len(test_labels)),
            "n_negative": int((test_labels == 0).sum()),
            "n_positive": int((test_labels == 1).sum()),
            "observation_order_sha256": patient_hash,
            "bootstrap_indices_sha256": target_bootstrap_hash,
        },
        "source_image": {
            "unit": "image",
            "stratified_by": "label_idx",
            "shared_across_all_valid_ensembles_and_metrics": True,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "n_observations": int(len(source_test_labels)),
            "n_negative": int((source_test_labels == 0).sum()),
            "n_positive": int((source_test_labels == 1).sum()),
            "observation_order_sha256": source_sample_hash,
            "bootstrap_indices_sha256": source_bootstrap_hash,
        },
    }
    output_paths["bootstrap_plan"].write_text(
        json.dumps(bootstrap_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = {
        "analysis": {
            "unit": "patient",
            "ensemble": _ensemble_estimand(seeds),
            "ensemble_methods": list(ENSEMBLE_METHODS),
            "finetune_estimand": (
                "replicate metrics and mean_sd only; training cohort and seed both vary"
            ),
            "seeds": list(seeds),
            "temperature_and_youden_fit_cohort": "target_calibration_patients",
            "test_cohort": "target_test_patients",
            "bootstrap_intervals_for": _ensemble_variant(seeds),
            "pr_auc_definition": "average_precision",
            "confidence_level": float(confidence_level),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "multiplicity": "Holm across six patient-level paired DeLong AUC tests",
            "source_internal_unit": "image",
            "source_threshold_fit_cohort": "source_val",
            "source_evaluation_cohort": "source_test",
        },
        "bootstrap_plan": bootstrap_plan,
        **{key: _dataframe_records(table) for key, table in tables.items()},
    }
    output_paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_report(
        output_paths["markdown"],
        n_patients=len(test_labels),
        n_positive=int(test_labels.sum()),
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        seed_summary=seed_summary,
        ensemble_metrics=ensemble_metrics,
        delong=delong_table,
        paired_differences=paired_differences,
        calibration=calibration_table,
        target_calibration_diagnostics=target_calibration_diagnostics,
        source_ensemble_metrics=source_ensemble_metrics,
    )
    return output_paths
