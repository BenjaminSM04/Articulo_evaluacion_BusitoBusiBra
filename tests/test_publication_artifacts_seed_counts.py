"""Regression tests for publication artifacts built from variable seed counts."""

from __future__ import annotations

import importlib

import pandas as pd


artifacts = importlib.import_module("scripts.14_generate_publication_artifacts")


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
