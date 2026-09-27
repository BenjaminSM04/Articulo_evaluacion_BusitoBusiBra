"""Locked v3 analysis over the complete 160-variant experiment registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.training.experiment_matrix import Experiment, build_experiment_matrix

from .publication_statistics import (
    ENSEMBLE_METHODS,
    _assert_prediction_fingerprint,
    align_seed_predictions,
    align_source_seed_predictions,
    analyze_publication_predictions,
    bootstrap_metric_distributions,
    build_confirmatory_ensemble,
    compare_v3_method_families,
    confirmatory_metric_values,
    fit_and_apply_calibration,
    paired_bootstrap_pvalue,
    percentile_interval,
    read_patient_prediction_csv,
    read_source_prediction_csv,
    sigmoid,
    stratified_patient_bootstrap_indices,
    threshold_sensitivity_panel,
)

COHORTS = (
    "source_val",
    "source_test",
    "target_calibration_images",
    "target_calibration_patients",
    "target_test_images",
    "target_test_patients",
)
SECONDARY_VARIANTS = (
    ("adabn", "adabn", "adabn"),
    ("intensity_source_direct", "intensity", "source_direct"),
    ("intensity", "intensity", "source_only_matched"),
    ("roi_source_direct", "roi", "source_direct"),
    ("roi", "roi", "source_only_matched"),
)


def _validate_index_and_inventory(
    prediction_root: Path,
    checkpoint_index: pd.DataFrame,
    *,
    architectures: Sequence[str],
    seeds: Sequence[int],
) -> tuple[Experiment, ...]:
    jobs = build_experiment_matrix(architectures=architectures, seeds=seeds)
    if len(jobs) != 160:
        raise ValueError("v3 requires exactly 160 experiment identities")
    required = {"experiment_id", "family", "arch", "seed", "method"}
    if not required.issubset(checkpoint_index):
        raise ValueError(f"v3 checkpoint index lacks {sorted(required - set(checkpoint_index))}")
    rows = checkpoint_index.set_index("experiment_id")
    if not rows.index.is_unique:
        raise ValueError("v3 checkpoint index contains duplicate identities")
    if set(rows.index.astype(str)) != {job.experiment_id for job in jobs}:
        raise ValueError("v3 checkpoint index does not match the frozen 160 identities")
    for job in jobs:
        row = rows.loc[job.experiment_id]
        if (
            str(row["family"]) != job.family
            or str(row["arch"]) != job.arch
            or int(row["seed"]) != job.seed
            or str(row["method"]) != job.method
        ):
            raise ValueError(f"v3 checkpoint descriptor changed: {job.experiment_id}")
    expected = {f"{job.experiment_id}.csv" for job in jobs}
    for cohort in COHORTS:
        directory = prediction_root / cohort
        for name in expected:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        observed = {path.name for path in directory.glob("*.csv")}
        if observed != expected:
            raise ValueError(f"v3 {cohort} has unexpected prediction files")
    return jobs


def _patient_tables(
    root: Path,
    jobs: dict[int, Experiment],
    *,
    architecture: str,
    seeds: Sequence[int],
    expected_fingerprint: str,
) -> tuple[Any, Any]:
    by_cohort: dict[str, Any] = {}
    for cohort in ("target_calibration_patients", "target_test_patients"):
        tables = {
            seed: read_patient_prediction_csv(
                root / cohort / f"{jobs[seed].experiment_id}.csv",
                architecture=architecture,
                method=jobs[seed].method,
                seed=seed,
            )
            for seed in seeds
        }
        _assert_prediction_fingerprint(
            tables, expected=expected_fingerprint, source=f"{architecture}/{cohort}"
        )
        by_cohort[cohort] = align_seed_predictions(
            tables, seeds=seeds, source=f"{architecture}/{cohort}"
        )
    return by_cohort["target_calibration_patients"], by_cohort["target_test_patients"]


def _source_validation(
    root: Path,
    jobs: dict[int, Experiment],
    *,
    architecture: str,
    seeds: Sequence[int],
    expected_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray]:
    tables = {
        seed: read_source_prediction_csv(
            root / "source_val" / f"{jobs[seed].experiment_id}.csv",
            architecture=architecture,
            method=jobs[seed].method,
            seed=seed,
        )
        for seed in seeds
    }
    _assert_prediction_fingerprint(tables, expected=expected_fingerprint, source="source_val")
    aligned = align_source_seed_predictions(tables, seeds=seeds, source="source_val")
    return aligned.images["label_idx"].to_numpy(dtype=int), sigmoid(aligned.logits.mean(axis=1))


def _per_seed_rows(
    calibration: Any,
    test: Any,
    jobs: dict[int, Experiment],
    *,
    architecture: str,
    alias: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    calibration_labels = calibration.patients["label_idx"].to_numpy(dtype=int)
    test_labels = test.patients["label_idx"].to_numpy(dtype=int)
    rows = []
    for column, seed in enumerate(seeds):
        _, threshold, _, raw, calibrated = fit_and_apply_calibration(
            calibration.logits[:, column], calibration_labels, test.logits[:, column]
        )
        rows.append(
            {
                "experiment_id": jobs[seed].experiment_id,
                "family": jobs[seed].family,
                "arch": architecture,
                "method": alias,
                "physical_method": jobs[seed].method,
                "seed": seed,
                "replicate_variation": "training_cohort_and_seed"
                if alias.startswith("finetune_")
                else "seed",
                **confirmatory_metric_values(test_labels, raw, calibrated, threshold),
            }
        )
    return rows


def analyze_v3_publication_predictions(
    prediction_root: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_index: pd.DataFrame,
    architectures: Sequence[str],
    seeds: Sequence[int],
    n_bootstrap: int,
    confidence_level: float,
    bootstrap_seed: int,
    expected_finalization_fingerprint: str,
) -> dict[str, Path]:
    """Analyze every frozen variant while keeping the v2 analysis API intact."""
    prediction_root = Path(prediction_root)
    output_dir = Path(output_dir)
    architectures = tuple(architectures)
    seeds = tuple(int(seed) for seed in seeds)
    jobs = _validate_index_and_inventory(
        prediction_root, checkpoint_index, architectures=architectures, seeds=seeds
    )
    by_key = {(job.family, job.arch, job.seed, job.method, job.value): job for job in jobs}
    main_by_key = {(job.arch, job.method, job.seed): job for job in jobs if job.family == "main"}
    main_methods = (
        "source_direct",
        "source_only_matched",
        "dann",
        "coral",
        "mmd",
        "finetune_5pct",
        "finetune_10pct",
        "finetune_20pct",
    )
    main_jobs = {
        (arch, method, seed): main_by_key[(arch, method, seed)]
        for arch in architectures
        for method in main_methods
        for seed in seeds
    }
    outputs = analyze_publication_predictions(
        prediction_root,
        output_dir,
        architectures=architectures,
        methods=main_methods,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        require_six_uda_comparisons=True,
        expected_finalization_fingerprint=expected_finalization_fingerprint,
        prediction_names={key: job.experiment_id for key, job in main_jobs.items()},
    )
    main_rows = pd.read_csv(outputs["per_seed_metrics"])
    variant_rows: list[dict[str, Any]] = []
    for row in main_rows.to_dict("records"):
        job = main_jobs[(row["arch"], row["method"], int(row["seed"]))]
        variant_rows.append({"experiment_id": job.experiment_id, "family": "main", **row})

    target_dir = output_dir / "ensemble_predictions" / "target"
    target_ensembles: dict[tuple[str, str], Any] = {}
    source_validation: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for arch in architectures:
        for method in ENSEMBLE_METHODS:
            target_ensembles[(arch, method)] = (
                pd.read_csv(target_dir / f"{arch}_{method}_target_calibration_patients.csv"),
                pd.read_csv(target_dir / f"{arch}_{method}_target_test_patients.csv"),
            )
            source_validation[(arch, method)] = _source_validation(
                prediction_root,
                {seed: main_jobs[(arch, method, seed)] for seed in seeds},
                architecture=arch,
                seeds=seeds,
                expected_fingerprint=expected_finalization_fingerprint,
            )
        for alias, family, physical_method in SECONDARY_VARIANTS:
            selected = {seed: by_key[(family, arch, seed, physical_method, None)] for seed in seeds}
            calibration, test = _patient_tables(
                prediction_root,
                selected,
                architecture=arch,
                seeds=seeds,
                expected_fingerprint=expected_finalization_fingerprint,
            )
            variant_rows.extend(
                _per_seed_rows(
                    calibration, test, selected, architecture=arch, alias=alias, seeds=seeds
                )
            )
            ensemble = build_confirmatory_ensemble(
                calibration, test, architecture=arch, method=alias
            )
            ensemble.calibration.to_csv(
                target_dir / f"{arch}_{alias}_target_calibration_patients.csv", index=False
            )
            ensemble.test.to_csv(
                target_dir / f"{arch}_{alias}_target_test_patients.csv", index=False
            )
            target_ensembles[(arch, alias)] = (ensemble.calibration, ensemble.test)
            source_validation[(arch, alias)] = _source_validation(
                prediction_root,
                selected,
                architecture=arch,
                seeds=seeds,
                expected_fingerprint=expected_finalization_fingerprint,
            )

    sensitivity_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.family != "sensitivity":
            continue
        calibration = read_patient_prediction_csv(
            prediction_root / "target_calibration_patients" / f"{job.experiment_id}.csv",
            architecture=job.arch,
            method=job.method,
            seed=job.seed,
        )
        test = read_patient_prediction_csv(
            prediction_root / "target_test_patients" / f"{job.experiment_id}.csv",
            architecture=job.arch,
            method=job.method,
            seed=job.seed,
        )
        for cohort, table in (("calibration", calibration), ("test", test)):
            _assert_prediction_fingerprint(
                {job.seed: table},
                expected=expected_finalization_fingerprint,
                source=f"{job.experiment_id}/{cohort}",
            )
        calibration_labels = calibration["label_idx"].to_numpy(dtype=int)
        test_labels = test["label_idx"].to_numpy(dtype=int)
        _, threshold, _, raw, calibrated = fit_and_apply_calibration(
            calibration["logit_difference"].to_numpy(dtype=float),
            calibration_labels,
            test["logit_difference"].to_numpy(dtype=float),
        )
        row = {
            "experiment_id": job.experiment_id,
            "family": job.family,
            "arch": job.arch,
            "method": job.method,
            "physical_method": job.method,
            "seed": job.seed,
            "replicate_variation": "seed",
            **confirmatory_metric_values(test_labels, raw, calibrated, threshold),
        }
        variant_rows.append(row)
        sensitivity_rows.append({"parameter": job.hyperparameter, "value": float(job.value), **row})
    for row in variant_rows:
        if (
            row["family"] == "main"
            and row["arch"] == "resnet18"
            and row["method"] in ("dann", "coral", "mmd")
        ):
            parameter = {"dann": "dann_lambda", "coral": "align_weight", "mmd": "mmd_sigma_scale"}[
                row["method"]
            ]
            sensitivity_rows.append({"parameter": parameter, "value": 1.0, **row})

    comparison_inputs = {
        key: (
            tables[1]["label_idx"].to_numpy(dtype=int),
            tables[1]["probability_raw"].to_numpy(dtype=float),
        )
        for key, tables in target_ensembles.items()
    }
    family_rows = compare_v3_method_families(comparison_inputs, architectures=architectures)
    reference_labels = comparison_inputs[(architectures[0], "source_only_matched")][0]
    bootstrap_indices = stratified_patient_bootstrap_indices(
        reference_labels, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    auc_distributions: dict[tuple[str, str], np.ndarray] = {}
    for family_row in family_rows:
        for method in (family_row["method"], family_row["comparator"]):
            key = (family_row["arch"], method)
            if key in auc_distributions:
                continue
            labels, probability = comparison_inputs[key]
            if not np.array_equal(labels, reference_labels):
                raise ValueError(f"v3 bootstrap cohort is not aligned for {key}")
            auc_distributions[key] = bootstrap_metric_distributions(
                labels, probability, probability, 0.5, bootstrap_indices
            )["auc_raw"]
        difference = (
            auc_distributions[(family_row["arch"], family_row["method"])]
            - (auc_distributions[(family_row["arch"], family_row["comparator"])])
        )
        family_row["ci_low"], family_row["ci_high"] = percentile_interval(
            difference, confidence_level=confidence_level
        )
        family_row["bootstrap_p_value"] = paired_bootstrap_pvalue(difference)
        family_row["n_bootstrap"] = n_bootstrap
    threshold_rows: list[dict[str, Any]] = []
    for (arch, method), (calibration, test) in target_ensembles.items():
        source_labels, source_probabilities = source_validation[(arch, method)]
        panel = threshold_sensitivity_panel(
            test["label_idx"].to_numpy(dtype=int),
            test["probability_raw"].to_numpy(dtype=float),
            source_val=(source_labels, source_probabilities),
            calibration=(
                calibration["label_idx"].to_numpy(dtype=int),
                calibration["probability_raw"].to_numpy(dtype=float),
            ),
        )
        selection_metadata = {
            "fixed_0.5": ("none", "none"),
            "source_val": ("source_val", "image"),
            "caller_provided_calibration": ("target_calibration", "patient"),
        }
        for row in panel:
            selection_cohort, selection_unit = selection_metadata[row["threshold_source"]]
            threshold_rows.append(
                {
                    "arch": arch,
                    "method": method,
                    "selection_cohort": selection_cohort,
                    "selection_unit": selection_unit,
                    **row,
                }
            )

    all_variants = pd.DataFrame(variant_rows).sort_values("experiment_id")
    if len(all_variants) != 160 or not all_variants["experiment_id"].is_unique:
        raise RuntimeError("v3 analysis did not cover exactly 160 distinct variants")
    additional = {
        "all_variant_metrics": output_dir / "all_variant_metrics.csv",
        "v3_method_families": output_dir / "v3_method_families.csv",
        "v3_hyperparameter_sensitivity": output_dir / "v3_hyperparameter_sensitivity.csv",
        "v3_threshold_sensitivity": output_dir / "v3_threshold_sensitivity.csv",
    }
    all_variants.to_csv(additional["all_variant_metrics"], index=False)
    pd.DataFrame(family_rows).to_csv(additional["v3_method_families"], index=False)
    pd.DataFrame(sensitivity_rows).sort_values(["parameter", "value", "seed"]).to_csv(
        additional["v3_hyperparameter_sensitivity"], index=False
    )
    pd.DataFrame(threshold_rows).sort_values(["arch", "method", "threshold_source"]).to_csv(
        additional["v3_threshold_sensitivity"], index=False
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    payload["analysis"]["protocol_version"] = "v3"
    payload["analysis"]["historical_test_exposure"] = True
    payload["analysis"]["primary_holm_family"] = "six UDA versus matched source-only"
    payload["analysis"]["secondary_holm_family"] = (
        "six AdaBN/intensity/ROI versus matched source-only"
    )
    payload["analysis"]["multiplicity"] = (
        "Two independent six-comparison Holm families: primary UDA methods and "
        "secondary AdaBN/intensity/ROI controls, each versus matched source-only."
    )
    payload["v3_method_families"] = pd.DataFrame(family_rows).to_dict("records")
    payload["v3_hyperparameter_sensitivity"] = pd.DataFrame(sensitivity_rows).to_dict("records")
    outputs["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# V3 locked historical-test reanalysis",
        "",
        "The target test had prior exploratory exposure; inferential results are descriptive.",
        "Five-seed dispersion and patient-bootstrap intervals are distinct quantities.",
        "Primary and secondary Holm families each contain six comparisons.",
        "",
        "## Primary and secondary comparisons",
        pd.DataFrame(family_rows)[
            ["family", "arch", "method", "delta_auc", "ci_low", "ci_high", "p_holm"]
        ].to_markdown(index=False),
        "",
        "## Hyperparameter sensitivity",
        "The sensitivity panel is descriptive and has no adjusted p values.",
    ]
    outputs["markdown"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return {**outputs, **additional}
