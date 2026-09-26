"""Synthetic preflight checks for the locked v3 finalizer; no private manifests are read."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import Config
from src.data.publication_splits import canonical_assignment_hash
from src.training.experiment_matrix import build_experiment_matrix
from src.training.publication_protocol import sha256_file


def _finalizer():
    path = Path(__file__).resolve().parents[1] / "scripts/12_finalize_publication_inference.py"
    spec = importlib.util.spec_from_file_location("publication_finalizer_v3_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/11_run_publication_experiments.py"
    spec = importlib.util.spec_from_file_location("publication_runner_v3_compat_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(tmp_path: Path) -> Config:
    cfg = Config(
        {
            "publication": {
                "protocol_version": "3.1-source-cohort-review-5seed",
                "architectures": ["resnet18", "efficientnet_b0"],
                "seeds": [17, 42, 73, 101, 202],
                "adaptation": {
                    "mmd_sigmas": [1, 2, 4],
                    "dann_lambda": 1.0,
                    "align_weight": 1.0,
                },
            },
            "paths": {"results": "results/v3"},
        }
    )
    cfg._root = tmp_path
    return cfg


def _complete_index(tmp_path: Path) -> pd.DataFrame:
    runner = _runner()
    cfg = _cfg(tmp_path)
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"),
        seeds=(17, 42, 73, 101, 202),
    )
    rows = []
    for job in jobs:
        rows.append(
            {
                "experiment_id": job.experiment_id,
                "family": job.family,
                "stage": job.stage,
                "arch": job.arch,
                "seed": job.seed,
                "method": job.method,
                "preprocessing": job.preprocessing,
                "hyperparameter": job.hyperparameter,
                "hyperparameter_value": job.value,
                "parent_id": job.parent_id,
                "parent_checkpoint_sha256": "parent-digest" if job.parent_id else "",
                "variant_config_sha256": runner._variant_config_hash(
                    runner._config_for_job(cfg, job), job
                ),
                "checkpoint_path": (
                    f"results/v3/checkpoints/v3/{job.experiment_id}.pt"
                ),
                "checkpoint_sha256": "checkpoint-digest",
            }
        )
    return pd.DataFrame(rows)


def test_v3_index_accepts_exactly_160_unique_variants(tmp_path):
    finalizer = _finalizer()
    index = _complete_index(tmp_path)
    jobs = finalizer._validate_v3_index(index, _cfg(tmp_path), tmp_path)
    assert len(jobs) == 160
    assert sum(job.trainable for job in jobs.values()) == 150
    assert sum(not job.trainable for job in jobs.values()) == 10


def test_v3_index_rejects_changed_variant_config_hash(tmp_path):
    finalizer = _finalizer()
    index = _complete_index(tmp_path)
    index.loc[0, "variant_config_sha256"] = "altered"
    with pytest.raises(ValueError, match="variant config"):
        finalizer._validate_v3_index(index, _cfg(tmp_path), tmp_path)


def test_v3_index_rejects_nearby_but_different_hyperparameter(tmp_path):
    finalizer = _finalizer()
    index = _complete_index(tmp_path)
    row = index["hyperparameter_value"].first_valid_index()
    index.loc[row, "hyperparameter_value"] = float(
        index.loc[row, "hyperparameter_value"]
    ) + 0.0000001
    with pytest.raises(ValueError, match="hyperparameter"):
        finalizer._validate_v3_index(index, _cfg(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "change",
    (
        lambda frame: frame.iloc[:-1].copy(),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.assign(family=["wrong", *frame.family.iloc[1:]]),
        lambda frame: frame.assign(
            checkpoint_path=["other.pt", *frame.checkpoint_path.iloc[1:]]
        ),
    ),
)
def test_v3_index_rejects_incomplete_duplicate_or_misidentified_variant(tmp_path, change):
    finalizer = _finalizer()
    with pytest.raises(ValueError):
        finalizer._validate_v3_index(change(_complete_index(tmp_path)), _cfg(tmp_path), tmp_path)


def test_v3_lineage_rejects_parent_digest_mismatch(tmp_path):
    finalizer = _finalizer()
    parent_path = tmp_path / "parent.pt"
    child_path = tmp_path / "child.pt"
    common = {key: "frozen" for key in finalizer.TRAINING_PROVENANCE_KEYS}
    common["protocol_version"] = "3.1-source-cohort-review-5seed"
    parent_id = "main_resnet18_seed17_source_direct"
    child_id = "main_resnet18_seed17_source_only_matched"
    parent_provenance = {
        **common,
        "experiment_id": parent_id,
        "family": "main",
        "preprocessing": "none",
        "hyperparameter": "",
        "hyperparameter_value": None,
        "parent_id": "",
        "parent_checkpoint_sha256": "",
        "variant_config_sha256": "config-parent",
    }
    torch.save(
        {
            "arch": "resnet18", "seed": 17, "method": "source_direct",
            "model_type": "baseline", "provenance": parent_provenance,
        },
        parent_path,
    )
    digest = sha256_file(parent_path)
    child_provenance = {
        **common,
        "experiment_id": child_id,
        "family": "main",
        "preprocessing": "none",
        "hyperparameter": "",
        "hyperparameter_value": None,
        "parent_id": parent_id,
        "parent_checkpoint_sha256": digest,
        "initial_checkpoint_sha256": digest,
        "variant_config_sha256": "config-child",
    }
    torch.save(
        {
            "arch": "resnet18", "seed": 17, "method": "source_only_matched",
            "model_type": "baseline", "provenance": child_provenance,
        },
        child_path,
    )
    index = pd.DataFrame(
        [
            {
                "experiment_id": parent_id, "family": "main", "arch": "resnet18",
                "seed": 17, "method": "source_direct", "preprocessing": "none",
                "hyperparameter": "", "hyperparameter_value": None, "parent_id": "",
                "parent_checkpoint_sha256": "", "variant_config_sha256": "config-parent",
                "checkpoint_path": str(parent_path), "checkpoint_sha256": digest,
            },
            {
                "experiment_id": child_id, "family": "main", "arch": "resnet18",
                "seed": 17, "method": "source_only_matched", "preprocessing": "none",
                "hyperparameter": "", "hyperparameter_value": None, "parent_id": parent_id,
                "parent_checkpoint_sha256": digest, "variant_config_sha256": "config-child",
                "checkpoint_path": str(child_path), "checkpoint_sha256": sha256_file(child_path),
            },
        ]
    )
    finalizer._verify_v3_checkpoint_lineage(index, common, tmp_path)
    index.loc[1, "parent_checkpoint_sha256"] = "altered"
    with pytest.raises(ValueError, match="parent"):
        finalizer._verify_v3_checkpoint_lineage(index, common, tmp_path)


def test_v3_requires_written_ethics_basis_before_test_marker(tmp_path, monkeypatch):
    finalizer = _finalizer()
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(finalizer, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["finalize", "--config", "synthetic.yaml", "--unlock-test"],
    )
    with pytest.raises(SystemExit, match="ethics|ética|etica"):
        finalizer.main()
    assert not (cfg.path("results") / "TEST_ACCESS_STARTED.json").exists()
    assert not (cfg.path("results") / "logs" / "TRAINING_ACTIVE.lock").exists()


def test_v3_ethics_basis_is_hashed_and_empty_file_is_rejected(tmp_path):
    finalizer = _finalizer()
    evidence = tmp_path / "ethics-basis.md"
    evidence.write_text(
        "Secondary analysis of public, de-identified BUSI and BUS-BRA data.",
        encoding="utf-8",
    )
    assert finalizer._ethics_basis_sha256(evidence) == sha256_file(evidence)
    evidence.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty|vacío|vacio"):
        finalizer._ethics_basis_sha256(evidence)


def test_v3_uses_separate_source_split_hash_registry(tmp_path):
    finalizer = _finalizer()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "source_assignment_hashes.json").write_text("{}", encoding="utf-8")
    (split_dir / "source_split_metadata.json").write_text("{}", encoding="utf-8")
    (split_dir / "assignment_hashes.json").write_text("old", encoding="utf-8")
    assignment, metadata = finalizer._split_provenance_files(
        "3.1-source-cohort-review-5seed", split_dir
    )
    assert assignment.name == "source_assignment_hashes.json"
    assert metadata.name == "source_split_metadata.json"
    assert json.loads(assignment.read_text(encoding="utf-8")) == {}


def test_v3_target_assignments_match_registry_and_locked_manifests(tmp_path):
    finalizer = _finalizer()
    splits = tmp_path / "splits"
    splits.mkdir()
    assignments = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "patient_id": ["p1", "p2", "p3"],
            "partition": ["target_adapt", "target_calibration", "target_test"],
        }
    )
    assignments.to_csv(splits / "target_assignments.csv", index=False)
    expected_hash = canonical_assignment_hash(
        assignments, ["sample_id", "patient_id", "partition"]
    )
    (splits / "assignment_hashes.json").write_text(
        json.dumps({"target_assignments": expected_hash}), encoding="utf-8"
    )
    frames = tuple(
        assignments.loc[assignments.partition.eq(partition)].copy()
        for partition in ("target_adapt", "target_calibration", "target_test")
    )
    assert finalizer._validate_v3_target_assignments(splits, *frames) == expected_hash

    changed_test = frames[2].copy()
    changed_test.loc[:, "sample_id"] = "different"
    with pytest.raises(ValueError, match="target assignment"):
        finalizer._validate_v3_target_assignments(
            splits, frames[0], frames[1], changed_test
        )

    assignments.loc[2, "partition"] = "target_adapt"
    assignments.to_csv(splits / "target_assignments.csv", index=False)
    with pytest.raises(ValueError, match="target assignment"):
        finalizer._validate_v3_target_assignments(splits, *frames)


def test_roi_mask_fingerprint_covers_every_mask_and_rejects_missing_path(tmp_path):
    finalizer = _finalizer()
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "one.png").write_bytes(b"first synthetic mask")
    (masks / "two.png").write_bytes(b"second synthetic mask")
    frame = pd.DataFrame({"mask_path": ["masks/one.png|masks/two.png"]})
    expected = hashlib.sha256(
        (sha256_file(masks / "one.png") + sha256_file(masks / "two.png")).encode("ascii")
    ).hexdigest()
    assert finalizer._roi_mask_sha256(tmp_path, (frame,)) == expected
    with pytest.raises(ValueError, match="ROI"):
        finalizer._roi_mask_sha256(
            tmp_path, (pd.DataFrame({"mask_path": ["masks/absent.png"]}),)
        )
