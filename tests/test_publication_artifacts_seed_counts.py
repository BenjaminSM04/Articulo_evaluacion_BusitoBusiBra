"""Regression tests for publication artifacts built from variable seed counts."""

from __future__ import annotations

import importlib
import json

import pandas as pd

artifacts = importlib.import_module("scripts.14_generate_publication_artifacts")


def test_v3_cohort_flow_uses_frozen_split_and_exclusion_counts(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "source_split_metadata.json").write_text(
        json.dumps(
            {
                "source_rows_manifest": 386,
                "source_rows_kept": 358,
                "source_exclusions": {
                    "duplicate_group_member": 9,
                    "objection_axilla": 14,
                    "objection_needle": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    cohorts = pd.DataFrame(
        [
            {
                "cohort": "Source training",
                "images": 229,
                "benign_images": 135,
                "malignant_images": 94,
                "patients": "NA",
            },
            {
                "cohort": "Source validation",
                "images": 57,
                "benign_images": 34,
                "malignant_images": 23,
                "patients": "NA",
            },
            {
                "cohort": "Source internal test",
                "images": 72,
                "benign_images": 43,
                "malignant_images": 29,
                "patients": "NA",
            },
            {
                "cohort": "Target adaptation",
                "images": 946,
                "benign_images": 642,
                "malignant_images": 304,
                "patients": 532,
            },
            {
                "cohort": "Target calibration",
                "images": 193,
                "benign_images": 131,
                "malignant_images": 62,
                "patients": 106,
            },
            {
                "cohort": "Historical external test",
                "images": 736,
                "benign_images": 495,
                "malignant_images": 241,
                "patients": 426,
            },
        ]
    )

    labels = artifacts._cohort_flow_labels(tmp_path, cohorts)

    assert "386" in labels["source_original"]
    assert "28 excluded" in labels["source_reviewed"]
    assert "358" in labels["source_reviewed"]
    assert "229" in labels["source_train"]
    assert "57" in labels["source_val"]
    assert "72" in labels["source_test"]
    assert "426 patients" in labels["target_test"]
    figure_paths = artifacts.figure_cohort_flow(tmp_path, cohorts, tmp_path / "figures")
    assert len(figure_paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)


def test_v3_supplement_tables_include_all_controls_and_sensitivity():
    tables = {
        "v3_method_families": pd.DataFrame(
            [
                {"family": family, "arch": arch, "method": f"m{index}", "p_holm": 0.5}
                for family in ("primary", "secondary")
                for arch in ("resnet18", "efficientnet_b0")
                for index in range(3)
            ]
        ),
        "v3_hyperparameter_sensitivity": pd.DataFrame(
            [
                {"parameter": parameter, "value": value, "seed": seed, "auc_raw": 0.7}
                for parameter in ("dann_lambda", "align_weight", "mmd_sigma_scale")
                for value in (0.5, 1.0, 2.0)
                for seed in (17, 42, 73, 101, 202)
            ]
        ),
        "v3_threshold_sensitivity": pd.DataFrame(
            [
                {"arch": arch, "method": method, "threshold_source": source}
                for arch in ("resnet18", "efficientnet_b0")
                for method in range(10)
                for source in ("fixed_0.5", "source_youden", "calibration_youden")
            ]
        ),
        "all_variant_metrics": pd.DataFrame(
            [{"experiment_id": f"v{i}", "nll_raw": 0.3, "nll_calibrated": 0.2} for i in range(160)]
        ),
    }

    extra = artifacts._v3_supplement_tables(tables)

    assert len(artifacts._replica_metrics_table(tables)) == 160
    assert set(("nll_raw", "nll_calibrated")).issubset(
        artifacts._replica_metrics_table(tables).columns
    )
    assert len(extra) == 3
    assert len(extra["tabla_s7_controles.csv"]) == 6
    assert len(extra["tabla_s8_sensibilidad.csv"]) == 45
    assert len(extra["tabla_s9_umbrales.csv"]) == 60


def test_finetune_assignment_table_includes_five_seed_files(tmp_path):
    seeds = (17, 42, 73, 101, 202)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for seed in seeds:
        rows = [
            {
                "selected": True,
                "budget": budget,
                "role": role,
                "patient_id": f"patient-{seed}-{budget}-{role}",
                "label_idx": role == "val",
            }
            for budget in ("05pct", "10pct", "20pct")
            for role in ("train", "val")
        ]
        pd.DataFrame(rows).to_csv(split_dir / f"finetune_assignments_seed{seed}.csv", index=False)

    table = artifacts.table_finetune_assignments(tmp_path, seeds=seeds)

    assert len(table) == 30
    assert set(table["seed"]) == set(seeds)
    assert set(table["patients"]) == {1}


def test_finetune_figure_accepts_five_replicas_per_method(tmp_path):
    seeds = (17, 42, 73, 101, 202)
    methods = ("finetune_5pct", "finetune_10pct", "finetune_20pct")
    architectures = ("resnet18", "efficientnet_b0")
    replicas = pd.DataFrame(
        [
            {"arch": arch, "method": method, "seed": seed, "auc_raw": 0.55 + 0.02 * index}
            for arch in architectures
            for method in methods
            for index, seed in enumerate(seeds)
        ]
    )
    summary = pd.DataFrame(
        [
            {"arch": arch, "method": method, "metric": "auc_raw", "mean": 0.59, "sd": 0.03}
            for arch in architectures
            for method in methods
        ]
    )

    paths = artifacts.figure_finetune(replicas, summary, tmp_path)

    assert {path.suffix for path in paths} == {".png", ".jpg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
