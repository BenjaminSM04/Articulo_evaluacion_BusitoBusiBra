"""Pruebas sintéticas del staging privado; nunca acceden a datasets reales."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "15_stage_v3_private_development.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage_v3_private_development", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _png(path: Path, *, shape: tuple[int, int] = (4, 5), empty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros(shape, dtype=np.uint8)
    if not empty:
        pixels[1, 1] = 255
    assert cv2.imwrite(str(path), pixels)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "original"
    clone = tmp_path / "clone"
    for rel in (
        "data/raw/busi/train.png",
        "data/raw/busi/val.png",
        "data/raw/busi/train-mask.png",
        "data/raw/busi/train-mask-2.png",
        "data/raw/busi/val-mask.png",
        "data/raw/bra/adapt.png",
        "data/raw/bra/adapt-mask.png",
    ):
        _png(source / rel)
    for rel, payload in {
        "data/raw/busi/held-out.png": b"NEVER COPY SOURCE TEST",
        "data/raw/bra/calibration.png": b"NEVER COPY CALIBRATION",
        "data/raw/bra/test.png": b"NEVER COPY TARGET TEST",
    }.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    license_path = source / "data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes(b"Synthetic BUS-BRA license notice\n")
    _csv(
        source / "data/processed/busi_curated_manifest.csv",
        [{"sample_id": "all-source", "image_path": "data/raw/busi/held-out.png"}],
    )
    copied_manifest = clone / "data/processed/busi_curated_manifest.csv"
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    copied_manifest.write_bytes((source / "data/processed/busi_curated_manifest.csv").read_bytes())
    source_splits = clone / "results/publication_v3_5seed/splits"
    _csv(
        source_splits / "source_train_manifest.csv",
        [
            {
                "sample_id": "s1",
                "image_path": "data/raw/busi/train.png",
                "mask_paths": "data/raw/busi/train-mask.png|data/raw/busi/train-mask-2.png",
                "partition": "source_train",
            }
        ],
    )
    _csv(
        source_splits / "source_val_manifest.csv",
        [
            {
                "sample_id": "s2",
                "image_path": "data/raw/busi/val.png",
                "mask_paths": "",
                "mask_path": "data/raw/busi/val-mask.png",
                "partition": "source_val",
            }
        ],
    )
    historical = source / "results/publication_v3_5seed/splits"
    _csv(
        historical / "target_adapt_manifest.csv",
        [
            {
                "sample_id": "t1",
                "patient_id": "p1",
                "label_idx": "1",
                "image_path": "data/raw/bra/adapt.png",
                "mask_path": "data/raw/bra/adapt-mask.png",
                "partition": "target_adapt",
            }
        ],
    )
    _csv(
        historical / "finetune_assignments_seed17.csv",
        [
            {
                "sample_id": "t1",
                "patient_id": "p1",
                "label_idx": "1",
                "budget_fraction": str(budget),
                "selected": "True",
                "role": "train",
                "selection_seed": "17",
            }
            for budget in (0.05, 0.10, 0.20)
        ],
    )
    for name in ("source_test_manifest", "target_calibration_manifest", "target_test_manifest"):
        (historical / f"{name}.csv").write_bytes(b"FORBIDDEN SENTINEL\xff")
    source_metadata = {
        "hashes": {
            "source_manifest_input": hashlib.sha256(
                (source / "data/processed/busi_curated_manifest.csv").read_bytes()
            ).hexdigest()
        },
        **{
            f"{partition}_manifest_sha256": hashlib.sha256(
                (source_splits / f"{partition}_manifest.csv").read_bytes()
            ).hexdigest()
            for partition in ("source_train", "source_val")
        },
    }
    (source_splits / "source_split_metadata.json").write_text(
        json.dumps(source_metadata), encoding="utf-8"
    )
    return source, clone


def _trust_fixture(stage, source: Path, *, seeds: tuple[int, ...] = (17,)) -> None:
    stage.DEFAULT_SEEDS = seeds
    stage.CURATED_MANIFEST_SHA256 = hashlib.sha256(
        (source / "data/processed/busi_curated_manifest.csv").read_bytes()
    ).hexdigest()
    stage.BUS_BRA_LICENSE_SHA256 = hashlib.sha256(
        (source / "data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt").read_bytes()
    ).hexdigest()
    stage.TARGET_ADAPT_SHA256 = hashlib.sha256(
        (source / "results/publication_v3_5seed/splits/target_adapt_manifest.csv").read_bytes()
    ).hexdigest()
    stage.FINETUNE_SHA256 = {
        seed: hashlib.sha256(
            (
                source / f"results/publication_v3_5seed/splits/finetune_assignments_seed{seed}.csv"
            ).read_bytes()
        ).hexdigest()
        for seed in seeds
    }


def _stage(stage, source: Path, clone: Path, *, seeds: tuple[int, ...] = (17,)):
    _trust_fixture(stage, source, seeds=seeds)
    return stage.stage_development(source, clone, seeds=seeds)


def _refresh_source_metadata(clone: Path) -> None:
    split_dir = clone / "results/publication_v3_5seed/splits"
    metadata_path = split_dir / "source_split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for partition in ("source_train", "source_val"):
        metadata[f"{partition}_manifest_sha256"] = hashlib.sha256(
            (split_dir / f"{partition}_manifest.csv").read_bytes()
        ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_prepare_manifest_only_copies_manifest(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    _trust_fixture(stage, source)
    (clone / "data/processed/busi_curated_manifest.csv").unlink()
    stage.prepare_manifest(source, clone)
    assert (clone / "data/processed/busi_curated_manifest.csv").read_bytes() == (
        source / "data/processed/busi_curated_manifest.csv"
    ).read_bytes()
    assert not (clone / "data/raw/busi/held-out.png").exists()
    assert not (clone / "data/raw/busi/train.png").exists()


def test_prepare_rejects_unapproved_curated_manifest(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    (clone / "data/processed/busi_curated_manifest.csv").unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        stage.prepare_manifest(source, clone)
    assert not (clone / "data/processed/busi_curated_manifest.csv").exists()


def test_stage_rejects_partial_seed_list(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    with pytest.raises(ValueError, match="semillas|seeds"):
        _module().stage_development(source, clone, seeds=(17,))
    assert not (clone / "data/raw/busi/train.png").exists()


def test_stage_rejects_unapproved_license_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    _trust_fixture(stage, source)
    stage.BUS_BRA_LICENSE_SHA256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        stage.stage_development(source, clone, seeds=(17,))
    assert not (clone / "data/raw/busi/train.png").exists()


def test_stage_copies_allowlisted_pixels_and_assignments_only(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    report = _stage(stage, source, clone)
    for rel in (
        "data/raw/busi/train.png",
        "data/raw/busi/val.png",
        "data/raw/busi/train-mask.png",
        "data/raw/busi/train-mask-2.png",
        "data/raw/busi/val-mask.png",
        "data/raw/bra/adapt.png",
        "data/raw/bra/adapt-mask.png",
        "data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt",
        "results/publication_v3_5seed/splits/target_adapt_manifest.csv",
        "results/publication_v3_5seed/splits/finetune_assignments_seed17.csv",
    ):
        assert (clone / rel).read_bytes() == (source / rel).read_bytes()
    assert report["copied_files"] == 10
    inventory = clone / "results/publication_v3_5seed/logs/private_staging_sha256.json"
    payload = inventory.read_bytes()
    assert report["report_sha256"] == hashlib.sha256(payload).hexdigest()
    assert json.loads(payload)["sha256"]["data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt"] == (
        hashlib.sha256(
            (source / "data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt").read_bytes()
        ).hexdigest()
    )
    assert "sha256" not in report
    for rel in (
        "data/raw/busi/held-out.png",
        "data/raw/bra/calibration.png",
        "data/raw/bra/test.png",
        "results/publication_v3_5seed/splits/source_test_manifest.csv",
        "results/publication_v3_5seed/splits/target_calibration_manifest.csv",
        "results/publication_v3_5seed/splits/target_test_manifest.csv",
    ):
        assert not (clone / rel).exists()


@pytest.mark.parametrize("bad", ["../outside.png", "C:/outside.png", "//server/share.png"])
def test_rejects_escape_paths_before_copy(tmp_path: Path, bad: str) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        clone / "results/publication_v3_5seed/splits/source_val_manifest.csv",
        [{"sample_id": "s2", "image_path": bad, "mask_paths": "", "partition": "source_val"}],
    )
    _refresh_source_metadata(clone)
    with pytest.raises(ValueError, match="ruta|path|proyecto"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


def test_rejects_existing_destination_with_different_bytes(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    destination = clone / "data/raw/busi/train.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"inconsistent")
    with pytest.raises(ValueError, match="destino|SHA"):
        _stage(_module(), source, clone)
    assert destination.read_bytes() == b"inconsistent"


def test_rejects_inconsistent_finetune_sample_ids(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        source / "results/publication_v3_5seed/splits/finetune_assignments_seed17.csv",
        [
            {
                "sample_id": "held-out", "patient_id": "p2", "label_idx": "1",
                "budget_fraction": str(budget), "selected": "True", "role": "train",
                "selection_seed": "17",
            }
            for budget in (0.05, 0.10, 0.20)
        ],
    )
    with pytest.raises(ValueError, match="sample_id ajeno"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/bra/adapt.png").exists()


def test_rejects_same_image_in_train_and_validation(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        clone / "results/publication_v3_5seed/splits/source_val_manifest.csv",
        [
            {
                "sample_id": "s2",
                "image_path": "data/raw/busi/train.png",
                "mask_paths": "",
                "partition": "source_val",
            }
        ],
    )
    _refresh_source_metadata(clone)
    with pytest.raises(ValueError, match="source_train|source_val"):
        _stage(_module(), source, clone)


def test_rejects_incomplete_finetune_assignments(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        source / "results/publication_v3_5seed/splits/target_adapt_manifest.csv",
        [
            {
                "sample_id": "t1",
                "patient_id": "p1",
                "label_idx": "1",
                "image_path": "data/raw/bra/adapt.png",
                "mask_path": "data/raw/bra/adapt-mask.png",
                "partition": "target_adapt",
            },
            {
                "sample_id": "t2",
                "patient_id": "p2",
                "label_idx": "1",
                "image_path": "data/raw/bra/adapt.png",
                "mask_path": "data/raw/bra/adapt-mask.png",
                "partition": "target_adapt",
            },
        ],
    )
    with pytest.raises(ValueError, match="finetune incompleto"):
        _stage(_module(), source, clone)


def test_source_mask_path_is_copied_when_plural_column_is_blank(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        clone / "results/publication_v3_5seed/splits/source_val_manifest.csv",
        [
            {
                "sample_id": "s2",
                "image_path": "data/raw/busi/val.png",
                "mask_paths": "",
                "mask_path": "data/raw/busi/val-mask.png",
                "partition": "source_val",
            }
        ],
    )
    _stage(_module(), source, clone)
    assert (clone / "data/raw/busi/val-mask.png").read_bytes() == (
        source / "data/raw/busi/val-mask.png"
    ).read_bytes()


@pytest.mark.parametrize(
    "mask_rel", ["data/raw/busi/train-mask.png", "data/raw/bra/adapt-mask.png"]
)
def test_rejects_misaligned_roi_mask_before_copy(tmp_path: Path, mask_rel: str) -> None:
    source, clone = _fixture(tmp_path)
    _png(source / mask_rel, shape=(3, 5))
    with pytest.raises(ValueError, match="máscara|dimensiones|ROI"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()
    assert not (clone / "results/publication_v3_5seed/logs/private_staging_sha256.json").exists()


@pytest.mark.parametrize(
    "mask_rel", ["data/raw/busi/train-mask.png", "data/raw/bra/adapt-mask.png"]
)
def test_rejects_empty_roi_mask_before_copy(tmp_path: Path, mask_rel: str) -> None:
    source, clone = _fixture(tmp_path)
    _png(source / mask_rel, empty=True)
    with pytest.raises(ValueError, match="máscara|vacía|ROI"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


def test_rejects_undecodable_development_image_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    (source / "data/raw/busi/val.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="imagen|decodificar"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


@pytest.mark.parametrize("partition", ["source_val", "target_adapt"])
def test_rejects_missing_roi_mask_before_copy(tmp_path: Path, partition: str) -> None:
    source, clone = _fixture(tmp_path)
    manifest_root = clone if partition == "source_val" else source
    relative = f"results/publication_v3_5seed/splits/{partition}_manifest.csv"
    row = {
        "sample_id": "s2" if partition == "source_val" else "t1",
        "patient_id": "p1" if partition == "target_adapt" else "",
        "label_idx": "1" if partition == "target_adapt" else "",
        "image_path": "data/raw/busi/val.png"
        if partition == "source_val"
        else "data/raw/bra/adapt.png",
        "mask_path": "",
        "partition": partition,
    }
    _csv(manifest_root / relative, [row])
    if partition == "source_val":
        _refresh_source_metadata(clone)
    with pytest.raises(ValueError, match="máscara|ROI"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


def test_rejects_target_image_path_shared_with_source(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        source / "results/publication_v3_5seed/splits/target_adapt_manifest.csv",
        [
            {
                "sample_id": "t1",
                "image_path": "data/raw/busi/train.png",
                "mask_path": "data/raw/bra/adapt-mask.png",
                "partition": "target_adapt",
            }
        ],
    )
    with pytest.raises(ValueError, match="source|target_adapt"):
        _stage(_module(), source, clone)


def test_rejects_changed_source_manifest_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    _csv(
        clone / "results/publication_v3_5seed/splits/source_train_manifest.csv",
        [
            {
                "sample_id": "s1",
                "image_path": "data/raw/busi/held-out.png",
                "mask_path": "data/raw/busi/train-mask.png",
                "partition": "source_train",
            }
        ],
    )
    with pytest.raises(ValueError, match="SHA-256|huella"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/held-out.png").exists()


def test_rejects_changed_curated_manifest_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    (clone / "data/processed/busi_curated_manifest.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256|huella"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


def test_rejects_source_metadata_with_unrelated_curated_hash(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    metadata_path = clone / "results/publication_v3_5seed/splits/source_split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hashes"]["source_manifest_input"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()


def test_rejects_unapproved_target_adapt_hash_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    _trust_fixture(stage, source)
    stage.TARGET_ADAPT_SHA256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256|huella"):
        stage.stage_development(source, clone, seeds=(17,))
    assert not (clone / "data/raw/bra/adapt.png").exists()


def test_rejects_assignment_with_wrong_seed_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    assignment = source / "results/publication_v3_5seed/splits/finetune_assignments_seed17.csv"
    rows = list(csv.DictReader(assignment.open(encoding="utf-8", newline="")))
    rows[0]["selection_seed"] = "42"
    _csv(assignment, rows)
    with pytest.raises(ValueError, match="semilla|selection_seed"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/bra/adapt.png").exists()


def test_rejects_unapproved_finetune_hash_before_copy(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    _trust_fixture(stage, source)
    stage.FINETUNE_SHA256 = {17: "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        stage.stage_development(source, clone, seeds=(17,))
    assert not (
        clone / "results/publication_v3_5seed/splits/finetune_assignments_seed17.csv"
    ).exists()


def test_rejects_duplicate_finetune_sample_within_budget(tmp_path: Path) -> None:
    source, clone = _fixture(tmp_path)
    assignment = source / "results/publication_v3_5seed/splits/finetune_assignments_seed17.csv"
    with assignment.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _csv(assignment, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicado"):
        _stage(_module(), source, clone)
    assert not (clone / "data/raw/bra/adapt.png").exists()


def test_rejects_source_changed_after_preflight(tmp_path: Path, monkeypatch) -> None:
    source, clone = _fixture(tmp_path)
    stage = _module()
    original_copy = stage._checked_copy
    changed = source / "data/raw/busi/train.png"

    def change_at_copy(source_root, clone_root, relative, *, expected_hash):
        if relative.as_posix() == "data/raw/busi/train.png":
            _png(changed, shape=(4, 5), empty=True)
        return original_copy(source_root, clone_root, relative, expected_hash=expected_hash)

    monkeypatch.setattr(stage, "_checked_copy", change_at_copy)
    with pytest.raises(ValueError, match="SHA-256|huella"):
        _stage(stage, source, clone)
    assert not (clone / "data/raw/busi/train.png").exists()
