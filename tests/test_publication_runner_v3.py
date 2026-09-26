"""Pruebas sintéticas del plan y de las barreras del runner v3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import Config
from src.data.datasets import UltrasoundDataset
from src.training.experiment_matrix import build_experiment_matrix

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publication_runner_v3", ROOT / "scripts/11_run_publication_experiments.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_plan_only_is_read_only_and_filters_canonical_jobs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n  results: output\npublication:\n"
        "  protocol_version: 3.0-review-5seed\n"
        "  architectures: [resnet18, efficientnet_b0]\n"
        "  seeds: [17, 42, 73, 101, 202]\n"
        "  cohort_version: reviewed_related_image_group_v3\n"
        "  controls:\n"
        "    intensity: {method: global_rgb_percentile, low: 1, high: 99}\n"
        "    roi: {method: union_masks, margin_fraction: 0.1}\n"
        "    adabn: {batch_size: 16, order_by: sample_id, parent_method: source_only_matched}\n"
        "  sensitivity:\n"
        "    dann_lambda: [0.5, 2.0]\n"
        "    align_weight: [0.5, 2.0]\n"
        "    mmd_sigma_scale: [0.5, 2.0]\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/11_run_publication_experiments.py"),
         "--config", str(config), "--plan-only", "--pilot", "--stage", "controls"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["jobs"]) == 5
    assert {job["stage"] for job in plan["jobs"]} == {"controls"}
    assert not (tmp_path / "output").exists()


def test_plan_rejects_unfrozen_control_contract() -> None:
    cfg = Config({"publication": {"protocol_version": "3.0-review-5seed",
        "architectures": ["resnet18", "efficientnet_b0"],
        "seeds": [17, 42, 73, 101, 202],
        "cohort_version": "reviewed_related_image_group_v3"}})
    with pytest.raises(ValueError, match="controls"):
        runner._v3_plan(cfg, stage="all", pilot=False)


def test_stage_requires_parent_or_selects_it_in_same_run() -> None:
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )
    selected = runner._select_v3_jobs(jobs, stage="sensitivity", pilot=False)
    assert len(selected) == 6
    assert all(job.parent_id == "main_resnet18_seed17_source_direct" for job in selected)


def test_job_config_is_independent_and_sets_sensitivity() -> None:
    cfg = Config({"publication": {"adaptation": {"dann_lambda": 1.0,
        "align_weight": 1.0, "mmd_sigmas": [1, 2, 4]}}})
    job = next(job for job in build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    ) if job.hyperparameter == "dann_lambda")
    changed = runner._config_for_job(cfg, job)
    assert changed.publication.adaptation.dann_lambda == 0.5
    assert cfg.publication.adaptation.dann_lambda == 1.0
    assert changed.publication.adaptation is not cfg.publication.adaptation


def test_variant_fingerprint_changes_for_controls_and_sensitivity() -> None:
    cfg = Config({"publication": {"adaptation": {"dann_lambda": 1.0,
        "align_weight": 1.0, "mmd_sigmas": [1, 2, 4]}}})
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )
    base = next(job for job in jobs if job.experiment_id == "main_resnet18_seed17_source_direct")
    intensity = next(
        job for job in jobs if job.experiment_id == "intensity_resnet18_seed17_source_direct"
    )
    sensitivity = next(job for job in jobs if job.hyperparameter == "dann_lambda")
    assert len({runner._variant_config_hash(_config, job) for _config, job in (
        (runner._config_for_job(cfg, base), base),
        (runner._config_for_job(cfg, intensity), intensity),
        (runner._config_for_job(cfg, sensitivity), sensitivity),
    )}) == 3


def test_pilot_gate_rejects_insufficient_disk_and_slow_projection() -> None:
    with pytest.raises(RuntimeError, match="disco"):
        runner._check_v3_resources(free_disk_gb=19.9, peak_vram_gb=7.0)
    with pytest.raises(RuntimeError, match="VRAM"):
        runner._check_v3_resources(free_disk_gb=20.0, peak_vram_gb=8.1)
    with pytest.raises(RuntimeError, match="16"):
        runner._check_pilot_projection(pilot_hours=2.1, pilot_jobs=20, total_jobs=160)


def test_projection_includes_blind_inference_and_elapsed_work() -> None:
    with pytest.raises(RuntimeError, match="16"):
        runner._check_pilot_projection(
            pilot_hours=1.5, pilot_inference_hours=0.4, pilot_jobs=20,
            total_jobs=160, consumed_hours=3.0, completed_jobs=20,
        )


def test_source_split_gate_checks_public_audit_and_disjoint_groups(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("group,decision\na,keep\n", encoding="utf-8")
    source_input = tmp_path / "curated.csv"
    source_input.write_text("sample_id\ns1\n", encoding="utf-8")
    cfg = Config({"paths": {"results": "."}, "publication": {"split_seed": 20260723,
        "cohort_version": "reviewed_related_image_group_v3",
        "source_group_audit": "audit.csv"},
        "datasets": {"busi": {"manifest": "curated.csv"}}})
    cfg._root = tmp_path
    train = pd.DataFrame({"sample_id": [f"tr{i}" for i in range(229)],
                          "pawlowska_group_id": [f"g{i}" for i in range(229)]})
    val = pd.DataFrame({"sample_id": [f"va{i}" for i in range(57)],
                        "pawlowska_group_id": [f"v{i}" for i in range(57)]})
    splits = tmp_path / "splits"
    splits.mkdir()
    train_path = splits / "source_train_manifest.csv"
    val_path = splits / "source_val_manifest.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    hashes = {
        "source_group_audit": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "source_manifest_input": hashlib.sha256(source_input.read_bytes()).hexdigest(),
    }
    metadata = {"hashes": hashes, "split_seed": 20260723,
                "source_split_unit": "reviewed_related_image_group_v3",
                "source_partition_counts": {"source_train": 229, "source_val": 57,
                                            "source_test": 72},
                "source_train_manifest_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "source_val_manifest_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest()}
    runner._validate_v3_source_split(cfg, tmp_path, train, val, metadata, hashes)
    source_input.write_text("sample_id\nchanged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Curated BUSI|input|manifiesto fuente"):
        runner._validate_v3_source_split(cfg, tmp_path, train, val, metadata, hashes)
    source_input.write_text("sample_id\ns1\n", encoding="utf-8")
    val.loc[0, "pawlowska_group_id"] = "g0"
    with pytest.raises(ValueError, match="grupo"):
        runner._validate_v3_source_split(cfg, tmp_path, train, val, metadata, hashes)
    val.loc[0, "pawlowska_group_id"] = "v0"
    audit.write_text("group,decision\na,drop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="auditoría"):
        runner._validate_v3_source_split(cfg, tmp_path, train, val, metadata, hashes)
    audit.write_text("group,decision\na,keep\n", encoding="utf-8")
    train_path.write_bytes(train_path.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="huella"):
        runner._validate_v3_source_split(cfg, tmp_path, train, val, metadata, hashes)


def test_wall_gate_reserves_longest_observed_job() -> None:
    assert runner._should_pause_before_job(17.4, 18.0, [0.2, 0.5])
    assert not runner._should_pause_before_job(16.0, 18.0, [0.2, 0.5])


def test_v3_requires_clean_identified_git_commit() -> None:
    with pytest.raises(RuntimeError, match="commit"):
        runner._assert_clean_git_state({"git_commit": "", "git_dirty_at_run": None})
    with pytest.raises(RuntimeError, match="limpio"):
        runner._assert_clean_git_state({"git_commit": "a" * 40, "git_dirty_at_run": True})


def test_v3_fingerprint_includes_source_split_generator() -> None:
    _, files = runner._training_code_fingerprint(ROOT, protocol_version="3.0-review-5seed")
    assert "src/data/publication_splits.py" in files


def test_v3_imagenet_weight_fingerprint_uses_local_timm_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import huggingface_hub

    first = tmp_path / "resnet" / "snapshots" / "synthetic-revision" / "model.safetensors"
    second = tmp_path / "efficientnet" / "snapshots" / "synthetic-revision" / "model.safetensors"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"synthetic resnet weights")
    monkeypatch.setattr(runner, "V3_IMAGENET_WEIGHTS", {
        "resnet18": {
            "model_id": "timm/resnet18.a1_in1k", "revision": "synthetic-revision",
            "filename": "model.safetensors",
            "sha256": hashlib.sha256(b"synthetic resnet weights").hexdigest(),
        },
        "efficientnet_b0": {
            "model_id": "timm/efficientnet_b0.ra_in1k", "revision": "synthetic-revision",
            "filename": "model.safetensors",
            "sha256": hashlib.sha256(b"synthetic efficientnet weights").hexdigest(),
        },
    })

    def cached(*, repo_id, filename, local_files_only):
        assert filename == "model.safetensors"
        assert local_files_only is True
        path = first if repo_id == "timm/resnet18.a1_in1k" else second
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    with pytest.raises(FileNotFoundError, match="efficientnet"):
        runner._imagenet_weights_sha256()
    second.write_bytes(b"synthetic efficientnet weights")
    records, paths = runner._imagenet_weights_sha256()
    assert paths == {"resnet18": first, "efficientnet_b0": second}
    assert records == {
        "resnet18": {
            "model_id": "timm/resnet18.a1_in1k",
            "revision": "synthetic-revision", "filename": "model.safetensors",
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        },
        "efficientnet_b0": {
            "model_id": "timm/efficientnet_b0.ra_in1k",
            "revision": "synthetic-revision", "filename": "model.safetensors",
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
        },
    }
    import timm.models._hub as timm_hub

    original_download = timm_hub.hf_hub_download
    try:
        runner._pin_v3_imagenet_weights(records, paths)
        assert timm_hub.hf_hub_download(
            "timm/resnet18.a1_in1k", filename="model.safetensors"
        ) == str(first)
        with pytest.raises(RuntimeError, match="fuera del protocolo"):
            timm_hub.hf_hub_download("other/model", filename="model.safetensors")
        first.write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="cambiaron"):
            timm_hub.hf_hub_download(
                "timm/resnet18.a1_in1k", filename="model.safetensors"
            )
    finally:
        timm_hub.hf_hub_download = original_download


def test_v3_missing_weights_leave_no_frozen_run_state(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    splits = results / "splits"
    splits.mkdir(parents=True)
    (splits / "source_assignment_hashes.json").write_text("{}", encoding="utf-8")
    (splits / "source_split_metadata.json").write_text("{}", encoding="utf-8")
    for partition in ("source_train", "source_val", "target_adapt"):
        (splits / f"{partition}_manifest.csv").write_text(
            "sample_id\nsynthetic\n", encoding="utf-8"
        )
    cfg = Config({"paths": {"results": "results"}, "publication": {
        "protocol_version": "3.1-review", "architectures": ["resnet18", "efficientnet_b0"],
        "seeds": [17, 42, 73, 101, 202],
    }})
    cfg._root = tmp_path
    monkeypatch.setattr(runner, "_validate_v3_contract", lambda cfg: None)
    monkeypatch.setattr(runner, "_git_state", lambda root: {
        "git_commit": "a" * 40, "git_dirty_at_run": False,
    })
    monkeypatch.setattr(runner, "_preflight_v3_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_check_v3_resources", lambda **k: None)
    monkeypatch.setattr(runner, "_read_manifest", lambda path, **k: pd.DataFrame({
        "sample_id": [path.stem], "patient_id": [path.stem],
    }))
    monkeypatch.setattr(runner, "_image_fingerprint", lambda *a, **k: (
        pd.DataFrame(), "images", "image-csv",
    ))
    monkeypatch.setattr(runner, "_validate_v3_source_split", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_effective_config_text", lambda cfg: "frozen config\n")
    monkeypatch.setattr(runner, "_training_code_fingerprint", lambda *a, **k: (
        "code", {},
    ))
    monkeypatch.setattr(runner, "_runtime_fingerprint", lambda: ("runtime", {}))

    def missing_weights():
        raise FileNotFoundError("synthetic timm cache missing")

    monkeypatch.setattr(runner, "_imagenet_weights_sha256", missing_weights)
    args = SimpleNamespace(pilot=True, stage="all", resume=False, max_wall_hours=18)
    with pytest.raises(FileNotFoundError, match="timm cache missing"):
        runner._run_v3(cfg, args)
    assert not (results / "effective_config.yaml").exists()
    assert not (results / "logs" / "training_provenance.json").exists()

    monkeypatch.setattr(runner, "_imagenet_weights_sha256", lambda: ({}, {}))
    monkeypatch.setattr(runner, "_pin_v3_imagenet_weights", lambda *a: None)
    args.pilot = False
    with pytest.raises(RuntimeError, match="piloto"):
        runner._run_v3(cfg, args)
    assert not (results / "effective_config.yaml").exists()

    args.full_no_pilot = True
    monkeypatch.setattr(runner, "_imagenet_weights_sha256", missing_weights)
    with pytest.raises(FileNotFoundError, match="timm cache missing"):
        runner._run_v3(cfg, args)
    assert not (results / "effective_config.yaml").exists()
    args.full_no_pilot = False
    monkeypatch.setattr(runner, "_imagenet_weights_sha256", lambda: ({}, {}))

    import timm.models._hub as timm_hub

    original_download = timm_hub.hf_hub_download

    def install_synthetic_pin(*_args):
        timm_hub.hf_hub_download = lambda *a, **k: "synthetic"
        return original_download

    monkeypatch.setattr(runner, "_pin_v3_imagenet_weights", install_synthetic_pin)
    args.pilot = True
    args.resume = True
    with pytest.raises(RuntimeError, match="configuración diferente"):
        runner._run_v3(cfg, args)
    assert timm_hub.hf_hub_download is original_download


def test_v3_resume_rejects_changed_imagenet_weight_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    provenance = {key: "frozen" for key in runner.RESUME_PROVENANCE_KEYS}
    provenance["imagenet_weights_sha256"] = {"resnet18": {"sha256": "old"}}
    torch.save({
        "arch": "resnet18", "seed": 17, "method": "source_direct",
        "model_type": "baseline", "provenance": provenance,
    }, checkpoint)
    runner._assert_resume_checkpoint(
        checkpoint, arch="resnet18", seed=17, method="source_direct",
        model_type="baseline", expected_provenance=provenance,
    )
    changed = {**provenance, "imagenet_weights_sha256": {"resnet18": {"sha256": "new"}}}
    with pytest.raises(RuntimeError, match="imagenet_weights_sha256"):
        runner._assert_resume_checkpoint(
            checkpoint, arch="resnet18", seed=17, method="source_direct",
            model_type="baseline", expected_provenance=changed,
        )


def test_resume_counts_time_of_interrupted_job(tmp_path: Path) -> None:
    elapsed = tmp_path / "elapsed.json"
    session = tmp_path / "session.json"
    elapsed.write_text('{"hours": 1.0}', encoding="utf-8")
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    session.write_text(json.dumps({"base_hours": 1.0,
        "started_at_utc": started.isoformat(), "active": True}), encoding="utf-8")
    assert runner._recover_v3_hours(elapsed, session) >= 2.9


def test_controls_preflight_rejects_missing_adabn_parent(tmp_path: Path) -> None:
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )
    controls = runner._select_v3_jobs(jobs, stage="controls", pilot=False)
    with pytest.raises(RuntimeError, match="main_resnet18_seed17_source_only_matched"):
        runner._preflight_v3_dependencies(controls, tmp_path)


def test_controls_preflight_requires_indexed_external_parent(tmp_path: Path) -> None:
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )
    controls = tuple(
        job for job in jobs if job.experiment_id == "adabn_resnet18_seed17"
    )
    checkpoints = tmp_path / "checkpoints"
    parent_id = "main_resnet18_seed17_source_only_matched"
    parent = checkpoints / "v3" / f"{parent_id}.pt"
    parent.parent.mkdir(parents=True)
    frozen = {key: "frozen" for key in runner.RESUME_PROVENANCE_KEYS}
    frozen["protocol_version"] = "3.1-review"
    torch.save({
        "arch": "resnet18", "seed": 17, "method": "source_only_matched",
        "model_type": "baseline", "provenance": {
            **frozen, "experiment_id": parent_id,
        },
    }, parent)
    with pytest.raises(RuntimeError, match="índice|index|indexado"):
        runner._preflight_v3_dependencies(controls, checkpoints)
    index = checkpoints / "checkpoint_index.csv"
    row = {
        "experiment_id": parent_id, "arch": "resnet18", "seed": 17,
        "method": "source_only_matched", "checkpoint_sha256": runner.sha256_file(parent),
    }
    pd.DataFrame([row]).to_csv(index, index=False)
    with pytest.raises(RuntimeError, match="procedencia|Procedencia"):
        runner._preflight_v3_dependencies(
            controls, checkpoints, protocol_version="3.1-review"
        )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "training_provenance.json").write_text(json.dumps(frozen), encoding="utf-8")
    runner._preflight_v3_dependencies(
        controls, checkpoints, protocol_version="3.1-review"
    )
    parent.write_bytes(parent.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="SHA"):
        runner._preflight_v3_dependencies(
            controls, checkpoints, protocol_version="3.1-review"
        )
    row["checkpoint_sha256"] = runner.sha256_file(parent)
    pd.DataFrame([row]).to_csv(index, index=False)
    with pytest.raises(RuntimeError, match="Procedencia"):
        runner._preflight_v3_dependencies(
            controls, checkpoints, protocol_version="3.2-other"
        )


def test_resume_can_reindex_valid_checkpoint_after_interruption(tmp_path: Path) -> None:
    job = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )[0]
    output = tmp_path / "checkpoint.pt"
    output.write_bytes(b"completed checkpoint")
    row = runner._v3_index_row(job, output, tmp_path, {
        "parent_checkpoint_sha256": "", "variant_config_sha256": "a" * 64,
    })
    assert row["experiment_id"] == job.experiment_id
    assert row["checkpoint_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_dataset_applies_roi_before_transform_and_rejects_missing_mask(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 7:12] = 255
    second_mask = np.zeros((20, 30), dtype=np.uint8)
    second_mask[10:14, 15:18] = 255
    cv2.imwrite(str(tmp_path / "image.png"), image)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    cv2.imwrite(str(tmp_path / "mask2.png"), second_mask)
    seen: list[tuple[int, int]] = []

    def transform(*, image):
        seen.append(image.shape[:2])
        return {"image": image}

    frame = pd.DataFrame([{"image_path": "image.png", "mask_path": "mask.png",
                           "mask_paths": "mask.png|mask2.png", "label_idx": 0}])
    dataset = UltrasoundDataset(frame, tmp_path, transform, preprocessing="roi")
    dataset[0]
    assert seen == [(11, 15)]
    single = frame.copy()
    single.loc[0, "mask_paths"] = ""
    seen.clear()
    UltrasoundDataset(single, tmp_path, transform, preprocessing="roi")[0]
    assert seen == [(7, 7)]
    old_hash = runner._roi_masks_hash(tmp_path, (frame,))
    cv2.imwrite(str(tmp_path / "mask2.png"), np.full((20, 30), 255, dtype=np.uint8))
    assert runner._roi_masks_hash(tmp_path, (frame,)) != old_hash
    frame.loc[0, "mask_paths"] = "missing.png"
    with pytest.raises(FileNotFoundError):
        UltrasoundDataset(frame, tmp_path, preprocessing="roi")[0]
    cv2.imwrite(str(tmp_path.parent / "outside-mask.png"), mask)
    frame.loc[0, "mask_paths"] = "../outside-mask.png"
    with pytest.raises(ValueError, match="fuera"):
        UltrasoundDataset(frame, tmp_path, preprocessing="roi")[0]
