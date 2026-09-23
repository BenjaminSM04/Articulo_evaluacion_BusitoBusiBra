"""Tests for the frozen, publication-facing BUSI/BUS-BRA partitions."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.data.publication_splits import (
    SOURCE_PARTITIONS,
    TARGET_PARTITIONS,
    add_stable_sample_ids,
    build_publication_splits,
    canonical_assignment_hash,
    create_publication_split_files,
    make_busbra_target_assignments,
    make_busi_group_disjoint_source_assignments,
    make_busi_source_assignments,
    normalise_manifest_path_columns,
    select_runtime_image_paths,
    verify_assignment_hash,
)


def _busi_manifest() -> pd.DataFrame:
    rows = []
    for label, name, count in ((0, "benign", 222), (1, "malignant", 164)):
        for image_id in range(1, count + 1):
            path = f"data/raw/BUSI/{name}/{name} ({image_id}).png"
            rows.append(
                {
                    "image_path": path.replace("raw", "processed"),
                    "original_path": path,
                    "dataset": "busi",
                    "image_id": image_id,
                    "label": name,
                    "label_idx": label,
                    "patient_id": "",
                }
            )
    return pd.DataFrame(rows)


def _target_and_historical_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 180 patients, with a class-stratified historical 80/20 split."""

    rows = []
    test_patients = {
        *(f"b{index:03d}" for index in range(24)),
        *(f"m{index:03d}" for index in range(12)),
    }
    for label, prefix, count in ((0, "b", 120), (1, "m", 60)):
        label_name = "benign" if label == 0 else "malignant"
        for patient_number in range(count):
            patient_id = f"{prefix}{patient_number:03d}"
            n_images = 2 if patient_number % 3 else 1
            for view in range(n_images):
                path = f"data/raw/BUS-BRA/{patient_id}_view{view}.png"
                rows.append(
                    {
                        "image_path": path.replace("raw", "processed"),
                        "original_path": path,
                        "dataset": "bus_bra",
                        "label": label_name,
                        "label_idx": label,
                        "patient_id": patient_id,
                    }
                )
    target = pd.DataFrame(rows)
    historical = pd.DataFrame(
        {
            "partition": [
                "test" if patient in test_patients else "adapt" for patient in target["patient_id"]
            ],
            "index": range(len(target)),
        }
    )
    return target, historical


def _bundle(seeds=(101, 202)):
    target, historical = _target_and_historical_split()
    return build_publication_splits(
        _busi_manifest(),
        target,
        historical,
        split_seed=17,
        finetune_seeds=seeds,
    )


def test_runtime_pixels_replace_preprocessed_path_without_losing_provenance():
    target, _ = _target_and_historical_split()
    selected = select_runtime_image_paths(target, path_column="original_path")

    assert selected["image_path"].tolist() == target["original_path"].tolist()
    assert selected["processed_image_path"].tolist() == target["image_path"].tolist()
    assert target["image_path"].str.contains("processed").all()
    with pytest.raises(KeyError):
        select_runtime_image_paths(target, path_column="missing_pixels")


def test_manifest_paths_are_serialised_portably():
    frame = pd.DataFrame(
        {
            "image_path": [r"data\processed\image.png"],
            "original_path": [r"data\raw\image.png"],
        }
    )
    portable = normalise_manifest_path_columns(frame)
    assert portable.loc[0, "image_path"] == "data/processed/image.png"
    assert portable.loc[0, "original_path"] == "data/raw/image.png"


def test_busi_source_split_is_64_16_20_stratified_and_row_order_independent():
    manifest = _busi_manifest()
    source, assignments = make_busi_source_assignments(manifest, seed=17)
    counts = source["partition"].value_counts().to_dict()

    assert counts == {"source_train": 247, "source_val": 62, "source_test": 77}
    assert set(source["split_unit"]) == {"image_after_curated_busi"}
    assert source["sample_id"].is_unique
    assert len(assignments) == 386
    for partition in SOURCE_PARTITIONS:
        assert set(source.loc[source["partition"].eq(partition), "label_idx"]) == {0, 1}

    shuffled = manifest.sample(frac=1, random_state=99).reset_index(drop=True)
    _, shuffled_assignments = make_busi_source_assignments(shuffled, seed=17)
    expected = assignments.set_index("sample_id")["partition"].sort_index()
    observed = shuffled_assignments.set_index("sample_id")["partition"].sort_index()
    pd.testing.assert_series_equal(expected, observed)


def test_busi_source_split_rejects_non_curated_manifest():
    with pytest.raises(ValueError, match="386"):
        make_busi_source_assignments(_busi_manifest().iloc[:-1], seed=17)


def _busi_group_audit() -> pd.DataFrame:
    manifest = _busi_manifest()
    audit = manifest[["label", "image_id"]].copy()
    audit["group_id"] = [
        f"{label}-{image_id}"
        for label, image_id in audit[["label", "image_id"]].itertuples(index=False)
    ]
    benign_related = audit["label"].eq("benign") & audit["image_id"].isin([1, 2, 3])
    malignant_related = audit["label"].eq("malignant") & audit["image_id"].isin([1, 2])
    audit.loc[benign_related, "group_id"] = "benign-related"
    audit.loc[malignant_related, "group_id"] = "malignant-related"
    return audit


def test_group_disjoint_source_split_preserves_exact_counts_and_is_order_independent():
    manifest = _busi_manifest()
    audit = _busi_group_audit()
    source, assignments = make_busi_group_disjoint_source_assignments(
        manifest, audit, seed=17
    )
    assert source["partition"].value_counts().to_dict() == {
        "source_train": 247, "source_val": 62, "source_test": 77
    }
    assert set(source["split_unit"]) == {"pawlowska_group_after_curated_busi"}
    assert source.groupby("pawlowska_group_id")["partition"].nunique().max() == 1
    assert len(assignments) == 386

    shuffled, shuffled_assignments = make_busi_group_disjoint_source_assignments(
        manifest.sample(frac=1, random_state=2),
        audit.sample(frac=1, random_state=3),
        seed=17,
    )
    pd.testing.assert_series_equal(
        assignments.set_index("sample_id")["partition"].sort_index(),
        shuffled_assignments.set_index("sample_id")["partition"].sort_index(),
    )
    assert shuffled.groupby("pawlowska_group_id")["partition"].nunique().max() == 1


def test_group_disjoint_source_split_requires_a_complete_unique_crosswalk():
    manifest = _busi_manifest()
    audit = _busi_group_audit()
    with pytest.raises(ValueError, match="complete"):
        make_busi_group_disjoint_source_assignments(manifest, audit.iloc[:-1], seed=17)
    with pytest.raises(ValueError, match="unique"):
        make_busi_group_disjoint_source_assignments(
            manifest, pd.concat([audit, audit.iloc[[0]]]), seed=17
        )


def test_review_split_builder_uses_group_audit_and_preserves_target(tmp_path):
    target, historical = _target_and_historical_split()
    plain = build_publication_splits(
        _busi_manifest(), target, historical, split_seed=17, finetune_seeds=(101, 202)
    )
    reviewed = create_publication_split_files(
        {"busi": _busi_manifest(), "bus_bra": target},
        {"seed": 17, "publication_splits": {"split_seed": 17, "seeds": [101, 202]}},
        tmp_path / "reviewed_splits",
        historical_target_split=historical,
        source_group_audit=_busi_group_audit(),
    )
    assert reviewed.source_manifest.groupby("pawlowska_group_id")["partition"].nunique().max() == 1
    assert reviewed.hashes["source_assignments"] != plain.hashes["source_assignments"]
    assert reviewed.hashes["target_assignments"] == plain.hashes["target_assignments"]
    assert reviewed.metadata["source_split_unit"] == "pawlowska_group_after_curated_busi"


def test_target_split_preserves_historical_test_and_has_zero_patient_overlap():
    target, historical = _target_and_historical_split()
    target_with_ids = add_stable_sample_ids(target, "bus_bra")
    historical_test_indices = historical.loc[historical["partition"].eq("test"), "index"]
    expected_test_ids = set(target_with_ids.iloc[historical_test_indices]["sample_id"])

    split_target, assignments = make_busbra_target_assignments(
        target,
        historical.sample(frac=1, random_state=8),
        seed=17,
    )
    observed_test_ids = set(
        split_target.loc[split_target["partition"].eq("target_test"), "sample_id"]
    )
    assert observed_test_ids == expected_test_ids
    assert set(assignments["partition"]) == set(TARGET_PARTITIONS)
    assert assignments["sample_id"].is_unique
    assert len(assignments) == len(target)

    patient_sets = {
        partition: set(split_target.loc[split_target["partition"].eq(partition), "patient_id"])
        for partition in TARGET_PARTITIONS
    }
    for index, left in enumerate(TARGET_PARTITIONS):
        for right in TARGET_PARTITIONS[index + 1 :]:
            assert patient_sets[left].isdisjoint(patient_sets[right])

    historical_adapt_patients = target.loc[
        historical["partition"].eq("adapt"), "patient_id"
    ].nunique()
    expected_calibration = math.floor(historical_adapt_patients / 6 + 0.5)
    assert len(patient_sets["target_calibration"]) == expected_calibration
    assert len(patient_sets["target_adapt"]) == historical_adapt_patients - expected_calibration
    for partition in TARGET_PARTITIONS:
        labels = split_target.loc[split_target["partition"].eq(partition), "label_idx"]
        assert set(labels) == {0, 1}


def test_target_split_rejects_historical_patient_leakage():
    target, historical = _target_and_historical_split()
    patient = target["patient_id"].value_counts().loc[lambda counts: counts.ge(2)].index[0]
    patient_indices = target.index[target["patient_id"].eq(patient)].tolist()
    assert len(patient_indices) >= 2
    historical.loc[historical["index"].eq(patient_indices[0]), "partition"] = "adapt"
    historical.loc[historical["index"].eq(patient_indices[1]), "partition"] = "test"

    with pytest.raises(ValueError, match="Fuga histórica"):
        make_busbra_target_assignments(target, historical, seed=17)


def test_finetune_cohorts_are_patient_level_and_train_val_are_both_nested():
    bundle = _bundle(seeds=(101, 202))
    target_adapt = bundle.target_manifest.loc[
        bundle.target_manifest["partition"].eq("target_adapt")
    ]
    patient_label = target_adapt.groupby("patient_id")["label_idx"].first().to_dict()

    for seed, assignments in bundle.finetune_assignments.items():
        assert set(assignments["selection_seed"]) == {seed}
        previous_selected: set[str] = set()
        previous_roles = {"train": set(), "val": set()}
        for budget in (0.05, 0.10, 0.20):
            subset = assignments.loc[assignments["budget_fraction"].eq(budget)]
            assert set(subset["sample_id"]) == set(target_adapt["sample_id"])
            patient_state = subset.groupby("patient_id").agg(
                selected=("selected", "first"),
                selected_nunique=("selected", "nunique"),
                role=("role", "first"),
                role_nunique=("role", "nunique"),
            )
            assert patient_state["selected_nunique"].max() == 1
            assert patient_state["role_nunique"].max() == 1

            selected = set(patient_state.index[patient_state["selected"]])
            assert previous_selected.issubset(selected)
            previous_selected = selected
            for role in ("train", "val"):
                role_patients = set(
                    patient_state.index[patient_state["selected"] & patient_state["role"].eq(role)]
                )
                assert previous_roles[role].issubset(role_patients)
                assert {patient_label[patient] for patient in role_patients} == {0, 1}
                previous_roles[role] = role_patients

            # Every selected patient contributes all of their images.
            selected_rows = subset.loc[subset["selected"]]
            expected_images = target_adapt.loc[
                target_adapt["patient_id"].isin(selected), "sample_id"
            ]
            assert set(selected_rows["sample_id"]) == set(expected_images)

    first_seed_patients = set(
        bundle.finetune_assignments[101].loc[
            bundle.finetune_assignments[101]["selected"]
            & bundle.finetune_assignments[101]["budget_fraction"].eq(0.20),
            "patient_id",
        ]
    )
    second_seed_patients = set(
        bundle.finetune_assignments[202].loc[
            bundle.finetune_assignments[202]["selected"]
            & bundle.finetune_assignments[202]["budget_fraction"].eq(0.20),
            "patient_id",
        ]
    )
    assert first_seed_patients != second_seed_patients


def test_assignment_hash_is_order_independent_and_detects_tampering():
    source = _bundle(seeds=(101,)).source_assignments
    columns = ["sample_id", "partition"]
    expected = canonical_assignment_hash(source, columns)
    shuffled = source.sample(frac=1, random_state=3)

    assert canonical_assignment_hash(shuffled, columns) == expected
    verify_assignment_hash(shuffled, expected, columns)

    tampered = source.copy()
    tampered.loc[tampered.index[0], "partition"] = "source_test"
    with pytest.raises(ValueError, match="Hash de asignación inválido"):
        verify_assignment_hash(tampered, expected, columns)


def test_runner_entrypoint_writes_assignments_and_isolated_manifests(tmp_path):
    target, historical = _target_and_historical_split()
    output = tmp_path / "results" / "publication_v2" / "splits"
    config = {
        "seed": 17,
        "publication_splits": {
            "split_seed": 17,
            "seeds": [101, 202],
        },
    }
    bundle = create_publication_split_files(
        {"busi": _busi_manifest(), "bus_bra": target},
        config,
        output,
        historical_target_split=historical,
    )

    expected_files = {
        "source_assignments.csv",
        "target_assignments.csv",
        "finetune_assignments_seed101.csv",
        "finetune_assignments_seed202.csv",
        "source_train_manifest.csv",
        "source_val_manifest.csv",
        "source_test_manifest.csv",
        "target_adapt_manifest.csv",
        "target_calibration_manifest.csv",
        "target_test_manifest.csv",
        "assignment_hashes.json",
        "split_metadata.json",
    }
    assert expected_files.issubset({path.name for path in output.iterdir()})

    source_parts = {
        partition: pd.read_csv(output / f"{partition}_manifest.csv")
        for partition in SOURCE_PARTITIONS
    }
    assert {key: len(value) for key, value in source_parts.items()} == {
        "source_train": 247,
        "source_val": 62,
        "source_test": 77,
    }
    source_ids = [set(frame["sample_id"]) for frame in source_parts.values()]
    assert not source_ids[0] & source_ids[1]
    assert not source_ids[0] & source_ids[2]
    assert not source_ids[1] & source_ids[2]

    target_parts = {
        partition: pd.read_csv(
            output / f"{partition}_manifest.csv",
            dtype={"patient_id": str},
        )
        for partition in TARGET_PARTITIONS
    }
    target_patients = [set(frame["patient_id"]) for frame in target_parts.values()]
    assert not target_patients[0] & target_patients[1]
    assert not target_patients[0] & target_patients[2]
    assert not target_patients[1] & target_patients[2]

    written_hashes = json.loads((output / "assignment_hashes.json").read_text())
    assert written_hashes == bundle.hashes
    verify_assignment_hash(
        pd.read_csv(output / "source_assignments.csv"),
        written_hashes["source_assignments"],
        ["sample_id", "partition"],
    )


def test_repeated_runner_calls_produce_identical_hashes(tmp_path):
    target, historical = _target_and_historical_split()
    manifests = {"busi": _busi_manifest(), "bus_bra": target}
    config = {"seed": 17, "publication_splits": {"seeds": [101]}}

    first = create_publication_split_files(
        manifests,
        config,
        tmp_path / "first",
        historical_target_split=historical,
    )
    second = create_publication_split_files(
        {
            "busi": manifests["busi"].sample(frac=1, random_state=1),
            # Target order cannot change because historical indices refer to this manifest.
            "bus_bra": manifests["bus_bra"],
        },
        config,
        tmp_path / "second",
        historical_target_split=historical.sample(frac=1, random_state=2),
    )
    assert first.hashes == second.hashes
