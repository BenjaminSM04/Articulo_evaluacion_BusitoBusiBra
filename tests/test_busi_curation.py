"""Unit tests for the strict BUSI curation source and file audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from src.busi_curation import (
    CURATED_BUSI_MAPPING_COMMIT,
    CURATED_BUSI_MAPPING_SHA256,
    CURATED_BUSI_SOURCE_DOI,
    DATASET_SOURCE_DOI,
    PAWLOWSKA_SOURCE_DOI,
    BusiCurationError,
    BusiMember,
    CuratedMappingEntry,
    decide_binary_curation,
    load_comment_groups,
    load_curated_mapping,
    parse_annotation_tags,
    parse_braced_ampersand_list,
    resolve_curated_file_metadata,
    resolve_file_metadata,
    validate_curated_file_inventory,
    validate_curated_mapping,
    validate_curated_selection,
    validate_source_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMENT_LIST = REPO_ROOT / "resources" / "busi_curation" / "dataset_comment_list.csv"
OFFICIAL_MAPPING = REPO_ROOT / "resources" / "busi_curation" / "mapping_curated_BUSI.csv"


def test_group_and_annotation_parsers_preserve_source_order():
    assert parse_braced_ampersand_list("{42&131&488}", field="ID", source_row=2) == (
        "42",
        "131",
        "488",
    )
    assert parse_braced_ampersand_list(
        "{benign (42)&benign (131)&malignant (51)}",
        field="Filename",
        source_row=2,
    ) == ("benign (42)", "benign (131)", "malignant (51)")
    assert parse_annotation_tags("measurement&doppler") == (
        "measurement",
        "doppler",
    )


def test_group_parser_rejects_unbalanced_or_empty_items():
    with pytest.raises(BusiCurationError, match="llaves"):
        parse_braced_ampersand_list("1&2", field="ID", source_row=9)
    with pytest.raises(BusiCurationError, match="elemento vacío"):
        parse_braced_ampersand_list("{1&&2}", field="ID", source_row=9)


def test_published_comment_list_has_exact_coverage_and_expected_selection():
    groups = load_comment_groups(COMMENT_LIST)
    source_counts = validate_source_coverage(groups)
    decisions = decide_binary_curation(groups)
    curated_counts = validate_curated_selection(decisions)
    kept = [decision for decision in decisions if decision.decision == "keep"]

    assert len(groups) == 587
    assert sum(len(group.members) for group in groups) == 780
    assert source_counts == {"benign": 437, "malignant": 210, "normal": 133}
    assert len(kept) == 457
    assert curated_counts == {"benign": 296, "malignant": 161}
    assert all(decision.member.label != "normal" for decision in kept)
    assert all(decision.member == decision.group.members[0] for decision in kept)
    assert all(not decision.group.objection for decision in kept)

    decisions_by_reason = Counter(
        reason for decision in decisions for reason in decision.decision_reasons
    )
    assert decisions_by_reason["objection_axilla"] > 0
    assert decisions_by_reason["objection_needle"] > 0
    assert decisions_by_reason["objection_multiclass"] > 0
    assert decisions_by_reason["duplicate_group_member"] > 0


def test_official_mapping_has_frozen_provenance_and_binary_composition():
    entries = load_curated_mapping(OFFICIAL_MAPPING)
    class_counts = validate_curated_mapping(entries, mapping_path=OFFICIAL_MAPPING)
    binary_entries = [entry for entry in entries if entry.label != "normal"]

    assert len(entries) == 450
    assert class_counts == {"benign": 222, "malignant": 164, "normal": 64}
    assert len(binary_entries) == 386
    assert Counter(entry.label for entry in binary_entries) == {
        "benign": 222,
        "malignant": 164,
    }
    assert CURATED_BUSI_MAPPING_SHA256 == hashlib.sha256(OFFICIAL_MAPPING.read_bytes()).hexdigest()
    assert CURATED_BUSI_MAPPING_COMMIT == "54687ba5f2cf4378d36fc5d762bd15c6068d2bd4"


def test_resolve_file_metadata_points_to_all_existing_masks_and_hashes(tmp_path):
    class_dir = tmp_path / "data" / "raw" / "BUSI" / "benign"
    class_dir.mkdir(parents=True)
    image = class_dir / "benign (1).png"
    primary_mask = class_dir / "benign (1)_mask.png"
    second_mask = class_dir / "benign (1)_mask_1.png"
    image.write_bytes(b"raw-image")
    primary_mask.write_bytes(b"primary-mask")
    second_mask.write_bytes(b"second-mask")

    metadata = resolve_file_metadata(
        BusiMember(
            image_id=1,
            filename="benign (1)",
            label="benign",
            class_number=1,
        ),
        raw_root=tmp_path / "data" / "raw" / "BUSI",
        project_root=tmp_path,
    )

    assert metadata.image_path == "data/raw/BUSI/benign/benign (1).png"
    assert metadata.mask_path == "data/raw/BUSI/benign/benign (1)_mask.png"
    assert metadata.mask_paths == (
        "data/raw/BUSI/benign/benign (1)_mask.png",
        "data/raw/BUSI/benign/benign (1)_mask_1.png",
    )
    assert metadata.image_sha256 == hashlib.sha256(b"raw-image").hexdigest()
    assert metadata.mask_sha256s == (
        hashlib.sha256(b"primary-mask").hexdigest(),
        hashlib.sha256(b"second-mask").hexdigest(),
    )
    assert PAWLOWSKA_SOURCE_DOI == "10.1016/j.dib.2023.109247"
    assert DATASET_SOURCE_DOI == "10.1016/j.dib.2019.104863"


def test_official_curated_inventory_and_hashes_are_one_to_one(tmp_path):
    curated_root = tmp_path / "data" / "raw" / "curated_busi_v1" / "Curated BUSI"
    images = curated_root / "images"
    masks = curated_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    image = images / "benign_id_20.png"
    mask = masks / "benign_id_20_mask.png"
    image.write_bytes(b"official-image")
    mask.write_bytes(b"official-mask")
    entry = CuratedMappingEntry(source_row=2, label="benign", class_number=20)

    assert validate_curated_file_inventory([entry], curated_root=curated_root) == {
        "images": 1,
        "masks": 1,
    }
    metadata = resolve_curated_file_metadata(
        entry,
        curated_root=curated_root,
        project_root=tmp_path,
    )

    assert metadata.image_path.endswith("curated_busi_v1/Curated BUSI/images/benign_id_20.png")
    assert metadata.mask_path.endswith("curated_busi_v1/Curated BUSI/masks/benign_id_20_mask.png")
    assert metadata.image_sha256 == hashlib.sha256(b"official-image").hexdigest()
    assert metadata.mask_sha256s == (hashlib.sha256(b"official-mask").hexdigest(),)
    assert CURATED_BUSI_SOURCE_DOI == "10.5281/zenodo.19047974"
