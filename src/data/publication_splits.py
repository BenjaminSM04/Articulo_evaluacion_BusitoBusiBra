"""Reproducible, leakage-resistant data splits for the publication protocol.

This module is deliberately independent from training code.  It consumes the
already-curated BUSI manifest, the BUS-BRA manifest, and the historical
BUS-BRA adaptation/test index file.  It then:

* creates image-level BUSI source train/validation/test partitions (BUSI does
  not publish patient identifiers, so these are never described as
  patient-level partitions);
* preserves the historical BUS-BRA target test partition and verifies that it
  is patient-disjoint from the historical adaptation pool;
* divides that adaptation pool into patient-level target adaptation and
  calibration partitions; and
* creates nested, patient-level fine-tuning cohorts for configurable seeds and
  label budgets.

All stochastic operations start from stable, sorted identifiers and use local
NumPy generators.  Consequently, assignments do not depend on input row order
or on random state elsewhere in the process.  Canonical SHA-256 hashes make
the assignments auditable after they are written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SOURCE_PARTITIONS = ("source_train", "source_val", "source_test")
TARGET_PARTITIONS = ("target_adapt", "target_calibration", "target_test")
DEFAULT_BUDGETS = (0.05, 0.10, 0.20)
DEFAULT_BUSI_CLASS_COUNTS = {0: 222, 1: 164}
V3_BUSI_CLASS_COUNTS = {0: 212, 1: 146}
V3_BUSI_EXCLUSION_COUNTS = {
    "duplicate_group_member": 9,
    "objection_axilla": 14,
    "objection_needle": 5,
}
V3_EXPLICIT_SOURCE_GROUPS = (
    ("benign", 121, 102),
    ("malignant", 621, 639),
    ("malignant", 644, 576),
)

FrameLike = pd.DataFrame | str | Path


@dataclass(frozen=True)
class PublicationSplitBundle:
    """In-memory representation of every publication split artifact."""

    source_manifest: pd.DataFrame
    target_manifest: pd.DataFrame
    source_assignments: pd.DataFrame
    target_assignments: pd.DataFrame
    finetune_assignments: dict[int, pd.DataFrame]
    hashes: dict[str, str]
    metadata: dict[str, Any]


def _read_frame(value: FrameLike, *, patient_ids: bool = False) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    dtype = {"patient_id": "string"} if patient_ids else None
    return pd.read_csv(Path(value), dtype=dtype)


def _normalise_patient_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _normalise_identity(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.lower()


def select_runtime_image_paths(
    manifest: pd.DataFrame,
    *,
    path_column: str,
) -> pd.DataFrame:
    """Use one declared pixel source while preserving the former processed path.

    Publication v2 applies the same runtime resize to original BUSI and BUS-BRA
    pixels.  The historical BUS-BRA manifest points ``image_path`` at a 224x224
    preprocessing artifact, so the configured ``original_path`` must replace it
    before manifests are frozen.
    """

    if path_column == "image_path":
        return manifest.copy()
    if path_column not in manifest.columns:
        raise KeyError(f"El manifest no contiene la ruta de píxeles {path_column!r}.")
    selected = manifest[path_column].fillna("").astype(str).str.strip()
    if selected.eq("").any():
        rows = selected.index[selected.eq("")].tolist()[:5]
        raise ValueError(f"{path_column} contiene rutas vacías en filas {rows}.")
    out = manifest.copy()
    out["processed_image_path"] = out["image_path"]
    out["image_path"] = selected
    return out


def normalise_manifest_path_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    """Store relative manifest paths with POSIX separators for portable packages."""
    out = manifest.copy()
    for column in ("image_path", "original_path", "processed_image_path", "mask_path"):
        if column in out.columns:
            out[column] = (
                out[column].fillna("").astype(str).str.strip().str.replace("\\", "/", regex=False)
            )
    return out


def add_stable_sample_ids(
    manifest: pd.DataFrame,
    dataset: str,
    identity_columns: Sequence[str] = ("original_path", "image_path", "id"),
) -> pd.DataFrame:
    """Return a manifest with a unique, row-order-independent ``sample_id``.

    Existing non-empty sample IDs are preserved.  Otherwise, the first
    non-empty value among ``identity_columns`` is normalised and hashed.
    """

    df = manifest.copy()
    if "sample_id" in df.columns:
        ids = df["sample_id"].fillna("").astype(str).str.strip()
        if ids.ne("").all():
            if ids.duplicated().any():
                duplicates = ids[ids.duplicated(keep=False)].unique()[:5]
                raise ValueError(f"sample_id duplicado: {duplicates.tolist()}")
            df["sample_id"] = ids
            return df

    available = [column for column in identity_columns if column in df.columns]
    if not available:
        raise KeyError(
            "No se puede construir sample_id: falta sample_id y ninguna columna "
            f"de identidad está disponible ({list(identity_columns)})."
        )

    sample_ids: list[str] = []
    for row_number, row in df.iterrows():
        identity = ""
        identity_column = ""
        for column in available:
            value = row[column]
            if not pd.isna(value) and str(value).strip():
                identity = _normalise_identity(value)
                identity_column = column
                break
        if not identity:
            raise ValueError(f"Fila {row_number}: no tiene una identidad estable.")
        material = f"{dataset.lower()}|{identity_column}|{identity}".encode("utf-8")
        sample_ids.append(f"{dataset.lower()}:{hashlib.sha256(material).hexdigest()}")

    if pd.Series(sample_ids).duplicated().any():
        raise ValueError(
            "Las columnas de identidad producen sample_id duplicados; "
            "el manifest no identifica cada imagen de forma única."
        )
    df["sample_id"] = sample_ids
    return df


def _seed_for(seed: int, namespace: str) -> int:
    material = f"{int(seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _round_fraction(total: int, fraction: float) -> int:
    """Round half up, avoiding Python's banker rounding for split sizes."""

    return int(math.floor(total * float(fraction) + 0.5))


def _allocate_stratified_counts(
    class_counts: Mapping[int, int],
    holdout_total: int,
) -> dict[int, int]:
    """Allocate an exact holdout size proportionally while retaining each class."""

    labels = sorted(int(label) for label in class_counts)
    counts = {int(label): int(class_counts[label]) for label in labels}
    total = sum(counts.values())
    if total <= 0 or not 0 < holdout_total < total:
        raise ValueError("El tamaño holdout debe estar entre 1 y N-1.")
    if any(count < 2 for count in counts.values()):
        raise ValueError("Cada clase necesita al menos dos unidades para un split estratificado.")

    minimum = {label: 1 for label in labels}
    maximum = {label: counts[label] - 1 for label in labels}
    if holdout_total < sum(minimum.values()) or holdout_total > sum(maximum.values()):
        raise ValueError(
            "El tamaño holdout no permite conservar todas las clases en ambas particiones."
        )

    raw = {label: counts[label] * holdout_total / total for label in labels}
    allocation = {
        label: min(maximum[label], max(minimum[label], int(math.floor(raw[label]))))
        for label in labels
    }

    while sum(allocation.values()) < holdout_total:
        candidates = [label for label in labels if allocation[label] < maximum[label]]
        if not candidates:
            raise RuntimeError("No se pudo completar la asignación estratificada.")
        chosen = max(
            candidates,
            key=lambda label: (raw[label] - allocation[label], counts[label], -label),
        )
        allocation[chosen] += 1

    while sum(allocation.values()) > holdout_total:
        candidates = [label for label in labels if allocation[label] > minimum[label]]
        if not candidates:
            raise RuntimeError("No se pudo reducir la asignación estratificada.")
        chosen = min(
            candidates,
            key=lambda label: (raw[label] - allocation[label], counts[label], label),
        )
        allocation[chosen] -= 1

    return allocation


def _stratified_holdout_ids(
    units: pd.DataFrame,
    *,
    id_column: str,
    label_column: str,
    holdout_total: int,
    seed: int,
    namespace: str,
) -> tuple[set[str], set[str]]:
    """Split unique units into retained/holdout sets with exact class counts."""

    required = {id_column, label_column}
    if not required.issubset(units.columns):
        raise KeyError(f"Faltan columnas para split estratificado: {sorted(required - set(units))}")
    if units[id_column].astype(str).duplicated().any():
        raise ValueError(f"{id_column} debe identificar una única unidad de partición.")

    work = units[[id_column, label_column]].copy()
    work[id_column] = work[id_column].astype(str)
    work[label_column] = pd.to_numeric(work[label_column], errors="raise").astype(int)
    class_counts = work[label_column].value_counts().sort_index().to_dict()
    allocation = _allocate_stratified_counts(class_counts, holdout_total)

    retained: set[str] = set()
    holdout: set[str] = set()
    for label in sorted(allocation):
        ids = sorted(work.loc[work[label_column].eq(label), id_column].tolist())
        rng = np.random.default_rng(_seed_for(seed, f"{namespace}|class={label}"))
        ordered = [ids[index] for index in rng.permutation(len(ids))]
        n_holdout = allocation[label]
        holdout.update(ordered[:n_holdout])
        retained.update(ordered[n_holdout:])
    return retained, holdout


def _validate_binary_labels(df: pd.DataFrame, *, context: str) -> None:
    if "label_idx" not in df.columns:
        raise KeyError(f"{context}: falta label_idx.")
    labels = set(pd.to_numeric(df["label_idx"], errors="raise").astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"{context}: se esperaban las clases {{0, 1}}, se obtuvo {labels}.")


def make_busi_source_assignments(
    busi_manifest: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float = 0.20,
    val_fraction_of_remainder: float = 0.20,
    expected_rows: int = 386,
    expected_class_counts: Mapping[int, int] | None = DEFAULT_BUSI_CLASS_COUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create BUSI source train/val/test splits without claiming patient grouping."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1.")
    if not 0 < val_fraction_of_remainder < 1:
        raise ValueError("val_fraction_of_remainder debe estar entre 0 y 1.")

    source = add_stable_sample_ids(busi_manifest, "busi")
    source["label_idx"] = pd.to_numeric(source["label_idx"], errors="raise").astype(int)
    _validate_binary_labels(source, context="BUSI")
    if len(source) != int(expected_rows):
        raise ValueError(
            f"BUSI curado debe contener {expected_rows} filas; se encontraron {len(source)}."
        )
    if expected_class_counts is not None:
        observed = source["label_idx"].value_counts().sort_index().to_dict()
        expected = {int(label): int(count) for label, count in expected_class_counts.items()}
        if observed != expected:
            raise ValueError(
                f"Distribución BUSI curada inesperada: {observed}; se esperaba {expected}."
            )

    units = source[["sample_id", "label_idx"]].sort_values("sample_id").reset_index(drop=True)
    test_n = _round_fraction(len(units), test_fraction)
    remainder_ids, test_ids = _stratified_holdout_ids(
        units,
        id_column="sample_id",
        label_column="label_idx",
        holdout_total=test_n,
        seed=seed,
        namespace="busi_source_test",
    )
    remainder = units.loc[units["sample_id"].isin(remainder_ids)].reset_index(drop=True)
    val_n = _round_fraction(len(remainder), val_fraction_of_remainder)
    train_ids, val_ids = _stratified_holdout_ids(
        remainder,
        id_column="sample_id",
        label_column="label_idx",
        holdout_total=val_n,
        seed=seed,
        namespace="busi_source_val",
    )

    partition_of = {
        **{sample_id: "source_train" for sample_id in train_ids},
        **{sample_id: "source_val" for sample_id in val_ids},
        **{sample_id: "source_test" for sample_id in test_ids},
    }
    source["partition"] = source["sample_id"].map(partition_of)
    source["split_unit"] = "image_after_curated_busi"
    source["split_seed"] = int(seed)
    assignments = (
        source[["sample_id", "partition", "label_idx", "split_unit", "split_seed"]]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    verify_busi_source_assignments(
        source,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        expected_rows=expected_rows,
    )
    return source, assignments


def make_busi_group_disjoint_source_assignments(
    busi_manifest: pd.DataFrame,
    pawlowska_audit: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float = 0.20,
    val_fraction_of_remainder: float = 0.20,
    expected_rows: int = 386,
    expected_class_counts: Mapping[int, int] | None = DEFAULT_BUSI_CLASS_COUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep documented related-image groups together without calling them patients.

    Start from the deterministic image-stratified allocation, put each related
    group in its modal partition, then exchange singleton groups of the same
    class to recover the original exact class/partition quotas. No performance
    result is consulted when choosing a partition.
    """

    source, _ = make_busi_source_assignments(
        busi_manifest,
        seed=seed,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        expected_rows=expected_rows,
        expected_class_counts=expected_class_counts,
    )
    required = {"label", "image_id", "group_id"}
    if not required.issubset(pawlowska_audit.columns) or not {"label", "image_id"}.issubset(source):
        raise ValueError("Pawlowska crosswalk requires label, image_id and group_id.")
    audit = pawlowska_audit[["label", "image_id", "group_id"]].copy()
    audit["image_id"] = pd.to_numeric(audit["image_id"], errors="raise").astype(int)
    if audit.duplicated(["label", "image_id"]).any():
        raise ValueError("Pawlowska crosswalk keys must be unique.")
    audit["group_id"] = audit["group_id"].fillna("").astype(str).str.strip()
    if audit["group_id"].eq("").any():
        raise ValueError("Pawlowska crosswalk group_id must be nonempty.")
    source["image_id"] = pd.to_numeric(source["image_id"], errors="raise").astype(int)
    source = source.merge(
        audit.rename(columns={"group_id": "pawlowska_group_id"}),
        on=["label", "image_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if source["pawlowska_group_id"].isna().any():
        raise ValueError("Pawlowska crosswalk must be complete for every curated image.")
    mixed = source.groupby("pawlowska_group_id")["label_idx"].nunique()
    if mixed.gt(1).any():
        raise ValueError("Pawlowska crosswalk contains a group with mixed classes.")

    target_quotas = source.groupby(["label_idx", "partition"]).size().to_dict()
    group_sizes = source.groupby("pawlowska_group_id").size().to_dict()
    for group_id in sorted(group_id for group_id, size in group_sizes.items() if size > 1):
        rows = source.loc[source["pawlowska_group_id"].eq(group_id)]
        votes = rows["partition"].value_counts().to_dict()
        chosen = min(
            SOURCE_PARTITIONS,
            key=lambda partition: (
                -votes.get(partition, 0),
                _seed_for(seed, f"pawlowska_group|{group_id}|{partition}"),
            ),
        )
        source.loc[rows.index, "partition"] = chosen

    for label in sorted(source["label_idx"].unique()):
        while True:
            counts = source.loc[source["label_idx"].eq(label), "partition"].value_counts().to_dict()
            excess = [
                partition for partition in SOURCE_PARTITIONS
                if counts.get(partition, 0) > target_quotas[(label, partition)]
            ]
            deficit = [
                partition for partition in SOURCE_PARTITIONS
                if counts.get(partition, 0) < target_quotas[(label, partition)]
            ]
            if not excess and not deficit:
                break
            if not excess or not deficit:
                raise RuntimeError("Pawlowska group rebalance has inconsistent quotas.")
            donor, recipient = excess[0], deficit[0]
            eligible = source.loc[
                source["label_idx"].eq(label)
                & source["partition"].eq(donor)
                & source["pawlowska_group_id"].map(group_sizes).eq(1),
                ["sample_id"],
            ]
            if eligible.empty:
                raise ValueError("Insufficient singleton BUSI groups to rebalance exact quotas.")
            chosen_id = min(
                eligible["sample_id"].astype(str),
                key=lambda sample_id: _seed_for(
                    seed, f"pawlowska_rebalance|{label}|{sample_id}|{recipient}"
                ),
            )
            source.loc[source["sample_id"].eq(chosen_id), "partition"] = recipient

    source["split_unit"] = "pawlowska_group_after_curated_busi"
    if source.groupby("pawlowska_group_id")["partition"].nunique().gt(1).any():
        raise RuntimeError("A related BUSI group crosses source partitions.")
    verify_busi_source_assignments(
        source,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        expected_rows=expected_rows,
        expected_split_unit="pawlowska_group_after_curated_busi",
    )
    assignments = (
        source[["sample_id", "partition", "label_idx", "split_unit", "split_seed"]]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    return source, assignments


def _normalise_historical_partition(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"adapt", "adaptation", "target_adapt", "historical_adapt"}:
        return "historical_adapt"
    if text in {"test", "target_test"}:
        return "target_test"
    raise ValueError(f"Partición histórica no reconocida: {value!r}")


def _patient_table(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if "patient_id" not in df.columns:
        raise KeyError(f"{context}: falta patient_id.")
    work = df.copy()
    work["patient_id"] = work["patient_id"].map(_normalise_patient_id)
    if work["patient_id"].eq("").any():
        raise ValueError(f"{context}: hay patient_id vacíos.")
    work["label_idx"] = pd.to_numeric(work["label_idx"], errors="raise").astype(int)
    mixed = work.groupby("patient_id")["label_idx"].nunique()
    if mixed.gt(1).any():
        patients = mixed[mixed.gt(1)].index.astype(str).tolist()[:5]
        raise ValueError(f"{context}: pacientes con etiquetas contradictorias: {patients}")
    return (
        work.groupby("patient_id", as_index=False)
        .agg(label_idx=("label_idx", "first"), n_images=("sample_id", "size"))
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def make_busbra_target_assignments(
    target_manifest: pd.DataFrame,
    historical_split: pd.DataFrame,
    *,
    seed: int,
    calibration_fraction_of_historical_adapt: float = 1 / 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the historical test and split its adaptation pool by patient."""

    if not 0 < calibration_fraction_of_historical_adapt < 1:
        raise ValueError("La fracción de calibración debe estar entre 0 y 1.")

    target = add_stable_sample_ids(target_manifest, "bus_bra")
    target["patient_id"] = target["patient_id"].map(_normalise_patient_id)
    target["label_idx"] = pd.to_numeric(target["label_idx"], errors="raise").astype(int)
    _validate_binary_labels(target, context="BUS-BRA")
    _patient_table(target, context="BUS-BRA")

    split = historical_split.copy()
    if not {"partition", "index"}.issubset(split.columns):
        raise KeyError("El split histórico requiere las columnas partition e index.")
    numeric_index = pd.to_numeric(split["index"], errors="raise")
    if not np.equal(numeric_index, np.floor(numeric_index)).all():
        raise ValueError("El split histórico contiene índices no enteros.")
    split["index"] = numeric_index.astype(int)
    if split["index"].duplicated().any():
        raise ValueError("El split histórico contiene índices duplicados.")
    expected_indices = set(range(len(target)))
    observed_indices = set(split["index"].tolist())
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)[:5]
        extra = sorted(observed_indices - expected_indices)[:5]
        raise ValueError(
            "El split histórico no cubre exactamente el manifest. "
            f"Faltan={missing}, sobran={extra}."
        )
    split["historical_partition"] = split["partition"].map(_normalise_historical_partition)
    if set(split["historical_partition"]) != {"historical_adapt", "target_test"}:
        raise ValueError("El split histórico debe contener adapt y test.")

    historical_by_index = split.set_index("index")["historical_partition"]
    target["historical_partition"] = [
        historical_by_index.loc[index] for index in range(len(target))
    ]
    adapt_pool = target.loc[target["historical_partition"].eq("historical_adapt")].copy()
    test = target.loc[target["historical_partition"].eq("target_test")].copy()

    adapt_patients = _patient_table(adapt_pool, context="BUS-BRA historical adapt")
    test_patients = set(test["patient_id"])
    overlap = set(adapt_patients["patient_id"]) & test_patients
    if overlap:
        raise ValueError(f"Fuga histórica adapt/test a nivel paciente: {sorted(overlap)[:5]}")

    calibration_n = _round_fraction(len(adapt_patients), calibration_fraction_of_historical_adapt)
    target_adapt_patients, calibration_patients = _stratified_holdout_ids(
        adapt_patients,
        id_column="patient_id",
        label_column="label_idx",
        holdout_total=calibration_n,
        seed=seed,
        namespace="busbra_target_calibration",
    )
    partition_of_patient = {
        **{patient: "target_adapt" for patient in target_adapt_patients},
        **{patient: "target_calibration" for patient in calibration_patients},
        **{patient: "target_test" for patient in test_patients},
    }
    target["partition"] = target["patient_id"].map(partition_of_patient)
    target["split_unit"] = "patient"
    target["split_seed"] = int(seed)
    target = target.drop(columns=["historical_partition"])
    assignments = (
        target[["sample_id", "patient_id", "partition", "label_idx", "split_unit", "split_seed"]]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    verify_target_assignments(target)
    historical_test_indices = sorted(
        split.loc[split["historical_partition"].eq("target_test"), "index"].tolist()
    )
    expected_test_ids = set(target.iloc[historical_test_indices]["sample_id"])
    observed_test_ids = set(target.loc[target["partition"].eq("target_test"), "sample_id"])
    if expected_test_ids != observed_test_ids:
        raise RuntimeError("No se pudo reconstruir íntegramente el test histórico.")
    return target, assignments


def _ordered_patients_by_class(
    patients: pd.DataFrame,
    *,
    seed: int,
) -> dict[int, list[str]]:
    ordered: dict[int, list[str]] = {}
    for label in sorted(patients["label_idx"].unique()):
        ids = sorted(
            patients.loc[patients["label_idx"].eq(label), "patient_id"].astype(str).tolist()
        )
        rng = np.random.default_rng(_seed_for(seed, f"finetune_order|class={int(label)}"))
        ordered[int(label)] = [ids[index] for index in rng.permutation(len(ids))]
    return ordered


def _nested_roles(n_patients: int, val_fraction: float) -> list[str]:
    """Assign fixed roles whose train and validation prefixes are both nested."""

    if not 0 < val_fraction < 0.5:
        raise ValueError("finetune_val_fraction debe estar entre 0 y 0.5.")
    roles: list[str] = []
    n_val = 0
    for prefix_size in range(1, n_patients + 1):
        if prefix_size == 1:
            desired_val = 0
        else:
            desired_val = max(1, _round_fraction(prefix_size, val_fraction))
            desired_val = min(prefix_size - 1, desired_val)
        if desired_val > n_val:
            roles.append("val")
            n_val += 1
        else:
            roles.append("train")
    return roles


def _budget_name(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}pct"


def make_finetune_assignments(
    target_manifest: pd.DataFrame,
    *,
    seeds: Sequence[int],
    budget_fractions: Sequence[float] = DEFAULT_BUDGETS,
    val_fraction: float = 0.20,
) -> dict[int, pd.DataFrame]:
    """Create nested patient cohorts and expand them to target-adapt samples."""

    target_adapt = target_manifest.loc[target_manifest["partition"].eq("target_adapt")].copy()
    patients = _patient_table(target_adapt, context="target_adapt")
    budgets = tuple(sorted({float(value) for value in budget_fractions}))
    if not budgets or any(not 0 < value <= 1 for value in budgets):
        raise ValueError("Los presupuestos deben ser fracciones únicas en (0, 1].")
    clean_seeds = tuple(int(seed) for seed in seeds)
    if not clean_seeds or len(set(clean_seeds)) != len(clean_seeds):
        raise ValueError("seeds debe contener enteros únicos y al menos una semilla.")

    result: dict[int, pd.DataFrame] = {}
    base_columns = ["sample_id", "patient_id", "label_idx"]
    for seed in clean_seeds:
        ordered = _ordered_patients_by_class(patients, seed=seed)
        role_of: dict[str, str] = {}
        for patient_ids in ordered.values():
            roles = _nested_roles(len(patient_ids), val_fraction)
            role_of.update(dict(zip(patient_ids, roles, strict=True)))

        frames: list[pd.DataFrame] = []
        for budget in budgets:
            selected: set[str] = set()
            for label, patient_ids in ordered.items():
                selected_n = max(2, _round_fraction(len(patient_ids), budget))
                selected_n = min(len(patient_ids), selected_n)
                if selected_n < 2:
                    raise ValueError(
                        f"Clase {label}: el presupuesto {budget} no permite train y val."
                    )
                selected.update(patient_ids[:selected_n])

            frame = target_adapt[base_columns].copy()
            frame["budget_fraction"] = float(budget)
            frame["budget"] = _budget_name(budget)
            frame["selected"] = frame["patient_id"].isin(selected)
            frame["role"] = frame["patient_id"].map(role_of).where(frame["selected"], "")
            frame["selection_seed"] = int(seed)
            frames.append(frame)

        assignments = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["budget_fraction", "sample_id"])
            .reset_index(drop=True)
        )
        verify_finetune_assignments(
            assignments,
            target_adapt,
            budget_fractions=budgets,
        )
        result[seed] = assignments
    return result


def canonical_assignment_hash(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """Compute an order-independent SHA-256 over selected assignment columns."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"No se pueden hashear columnas ausentes: {sorted(missing)}")
    canonical = frame[list(columns)].copy()
    for column in columns:
        if pd.api.types.is_bool_dtype(canonical[column]):
            canonical[column] = canonical[column].map({True: "true", False: "false"})
        elif pd.api.types.is_numeric_dtype(canonical[column]):
            canonical[column] = canonical[column].map(
                lambda value: "" if pd.isna(value) else format(float(value), ".12g")
            )
        else:
            canonical[column] = canonical[column].fillna("").astype(str)
    canonical = canonical.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    records = canonical.to_dict(orient="records")
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_assignment_hash(
    frame: pd.DataFrame,
    expected_hash: str,
    columns: Sequence[str],
) -> None:
    observed = canonical_assignment_hash(frame, columns)
    if observed != str(expected_hash):
        raise ValueError(f"Hash de asignación inválido: {observed} != {expected_hash}")


def verify_busi_source_assignments(
    source_manifest: pd.DataFrame,
    *,
    test_fraction: float,
    val_fraction_of_remainder: float,
    expected_rows: int,
    expected_split_unit: str = "image_after_curated_busi",
) -> None:
    required = {"sample_id", "partition", "label_idx", "split_unit"}
    if not required.issubset(source_manifest.columns):
        raise KeyError(f"BUSI assignments: faltan {sorted(required - set(source_manifest))}")
    if len(source_manifest) != expected_rows:
        raise ValueError("BUSI assignments no conserva todas las filas.")
    if source_manifest["sample_id"].duplicated().any():
        raise ValueError("BUSI assignments contiene sample_id duplicados.")
    if set(source_manifest["partition"]) != set(SOURCE_PARTITIONS):
        raise ValueError("BUSI assignments no contiene las tres particiones source.")
    if set(source_manifest["split_unit"]) != {expected_split_unit}:
        raise ValueError("Unidad de partición fuente BUSI inesperada.")

    test_n = _round_fraction(expected_rows, test_fraction)
    val_n = _round_fraction(expected_rows - test_n, val_fraction_of_remainder)
    expected_sizes = {
        "source_train": expected_rows - test_n - val_n,
        "source_val": val_n,
        "source_test": test_n,
    }
    observed_sizes = source_manifest["partition"].value_counts().to_dict()
    if observed_sizes != expected_sizes:
        raise ValueError(f"Tamaños source inesperados: {observed_sizes} != {expected_sizes}")
    for partition in SOURCE_PARTITIONS:
        labels = set(source_manifest.loc[source_manifest["partition"].eq(partition), "label_idx"])
        if labels != {0, 1}:
            raise ValueError(f"{partition} no contiene ambas clases.")


def verify_target_assignments(target_manifest: pd.DataFrame) -> None:
    required = {"sample_id", "patient_id", "partition", "label_idx", "split_unit"}
    if not required.issubset(target_manifest.columns):
        raise KeyError(f"Target assignments: faltan {sorted(required - set(target_manifest))}")
    if target_manifest["sample_id"].duplicated().any():
        raise ValueError("Target assignments contiene sample_id duplicados.")
    if set(target_manifest["partition"]) != set(TARGET_PARTITIONS):
        raise ValueError("Target assignments no contiene las tres particiones target.")
    if set(target_manifest["split_unit"]) != {"patient"}:
        raise ValueError("Las particiones target deben ser por paciente.")

    covered: set[str] = set()
    patient_sets: dict[str, set[str]] = {}
    for partition in TARGET_PARTITIONS:
        subset = target_manifest.loc[target_manifest["partition"].eq(partition)]
        labels = set(subset["label_idx"])
        if labels != {0, 1}:
            raise ValueError(f"{partition} no contiene ambas clases.")
        sample_ids = set(subset["sample_id"])
        if covered & sample_ids:
            raise ValueError("Una imagen target aparece en más de una partición.")
        covered.update(sample_ids)
        patient_sets[partition] = set(subset["patient_id"])

    for index, left in enumerate(TARGET_PARTITIONS):
        for right in TARGET_PARTITIONS[index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise ValueError(f"Fuga de pacientes entre {left} y {right}: {sorted(overlap)[:5]}")
    _patient_table(target_manifest, context="target assignments")


def verify_finetune_assignments(
    assignments: pd.DataFrame,
    target_adapt_manifest: pd.DataFrame,
    *,
    budget_fractions: Sequence[float],
) -> None:
    required = {
        "sample_id",
        "patient_id",
        "label_idx",
        "budget_fraction",
        "role",
        "selected",
        "selection_seed",
    }
    if not required.issubset(assignments.columns):
        raise KeyError(f"Fine-tune assignments: faltan {sorted(required - set(assignments))}")
    budgets = tuple(sorted(float(value) for value in budget_fractions))
    if set(assignments["budget_fraction"].astype(float)) != set(budgets):
        raise ValueError("Fine-tune assignments no contiene todos los presupuestos.")

    expected_samples = set(target_adapt_manifest["sample_id"])
    patient_labels = (
        target_adapt_manifest.groupby("patient_id")["label_idx"].first().astype(int).to_dict()
    )
    previous_by_role: dict[str, set[str]] = {"train": set(), "val": set()}
    previous_selected: set[str] = set()
    for budget in budgets:
        subset = assignments.loc[
            np.isclose(assignments["budget_fraction"].astype(float), budget)
        ].copy()
        if set(subset["sample_id"]) != expected_samples or subset["sample_id"].duplicated().any():
            raise ValueError(f"El presupuesto {budget} no cubre target_adapt exactamente una vez.")
        unselected_roles = subset.loc[~subset["selected"].astype(bool), "role"].fillna("")
        if unselected_roles.astype(str).str.len().gt(0).any():
            raise ValueError("Una muestra no seleccionada tiene rol train/val.")

        patient_state = subset.groupby("patient_id", as_index=False).agg(
            selected_nunique=("selected", "nunique"),
            selected=("selected", "first"),
            role_nunique=("role", "nunique"),
            role=("role", "first"),
        )
        if (
            patient_state["selected_nunique"].gt(1).any()
            or patient_state["role_nunique"].gt(1).any()
        ):
            raise ValueError("Las imágenes de un paciente no comparten selección y rol.")
        selected_patients = set(patient_state.loc[patient_state["selected"], "patient_id"])
        if not previous_selected.issubset(selected_patients):
            raise ValueError("Las cohortes de fine-tuning no son anidadas.")
        previous_selected = selected_patients

        for role in ("train", "val"):
            role_patients = set(
                patient_state.loc[
                    patient_state["selected"] & patient_state["role"].eq(role), "patient_id"
                ]
            )
            if not previous_by_role[role].issubset(role_patients):
                raise ValueError(f"Las cohortes {role} no son anidadas.")
            previous_by_role[role] = role_patients
            labels = {patient_labels[patient] for patient in role_patients}
            if labels != {0, 1}:
                raise ValueError(f"El rol {role} del presupuesto {budget} no tiene ambas clases.")


def _compute_hashes(
    source_assignments: pd.DataFrame,
    target_assignments: pd.DataFrame,
    finetune_assignments: Mapping[int, pd.DataFrame],
) -> dict[str, str]:
    hashes = {
        "source_assignments": canonical_assignment_hash(
            source_assignments, ["sample_id", "partition"]
        ),
        "target_assignments": canonical_assignment_hash(
            target_assignments, ["sample_id", "patient_id", "partition"]
        ),
    }
    for seed, assignments in sorted(finetune_assignments.items()):
        hashes[f"finetune_assignments_seed{seed}"] = canonical_assignment_hash(
            assignments,
            [
                "sample_id",
                "patient_id",
                "budget_fraction",
                "role",
                "selected",
                "selection_seed",
            ],
        )
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hashes["all_assignments"] = hashlib.sha256(payload).hexdigest()
    return hashes


def build_publication_splits(
    busi_manifest: pd.DataFrame,
    target_manifest: pd.DataFrame,
    historical_target_split: pd.DataFrame,
    *,
    split_seed: int,
    finetune_seeds: Sequence[int],
    source_test_fraction: float = 0.20,
    source_val_fraction_of_remainder: float = 0.20,
    calibration_fraction_of_historical_adapt: float = 1 / 6,
    budget_fractions: Sequence[float] = DEFAULT_BUDGETS,
    finetune_val_fraction: float = 0.20,
    expected_busi_rows: int = 386,
    expected_busi_class_counts: Mapping[int, int] | None = DEFAULT_BUSI_CLASS_COUNTS,
    source_group_audit: FrameLike | None = None,
) -> PublicationSplitBundle:
    """Build and verify every split without writing to disk."""

    source_split = (
        make_busi_source_assignments
        if source_group_audit is None
        else make_busi_group_disjoint_source_assignments
    )
    source_args = () if source_group_audit is None else (_read_frame(source_group_audit),)
    source, source_assignments = source_split(
        busi_manifest,
        *source_args,
        seed=split_seed,
        test_fraction=source_test_fraction,
        val_fraction_of_remainder=source_val_fraction_of_remainder,
        expected_rows=expected_busi_rows,
        expected_class_counts=expected_busi_class_counts,
    )
    target, target_assignments = make_busbra_target_assignments(
        target_manifest,
        historical_target_split,
        seed=split_seed,
        calibration_fraction_of_historical_adapt=calibration_fraction_of_historical_adapt,
    )
    finetune = make_finetune_assignments(
        target,
        seeds=finetune_seeds,
        budget_fractions=budget_fractions,
        val_fraction=finetune_val_fraction,
    )
    hashes = _compute_hashes(source_assignments, target_assignments, finetune)
    metadata = {
        "split_seed": int(split_seed),
        "finetune_seeds": [int(seed) for seed in finetune_seeds],
        "source_test_fraction": float(source_test_fraction),
        "source_val_fraction_of_remainder": float(source_val_fraction_of_remainder),
        "calibration_fraction_of_historical_adapt": float(calibration_fraction_of_historical_adapt),
        "budget_fractions": [float(value) for value in sorted(budget_fractions)],
        "finetune_val_fraction": float(finetune_val_fraction),
        "source_partition_counts": source["partition"].value_counts().sort_index().to_dict(),
        "target_image_counts": target["partition"].value_counts().sort_index().to_dict(),
        "target_patient_counts": (
            target[["patient_id", "partition"]]
            .drop_duplicates()
            .groupby("partition")
            .size()
            .sort_index()
            .to_dict()
        ),
    }
    if source_group_audit is not None:
        metadata["source_split_unit"] = "pawlowska_group_after_curated_busi"
        metadata["source_related_groups"] = int(source["pawlowska_group_id"].nunique())
        if not isinstance(source_group_audit, pd.DataFrame):
            metadata["source_group_audit_sha256"] = hashlib.sha256(
                Path(source_group_audit).read_bytes()
            ).hexdigest()
    return PublicationSplitBundle(
        source_manifest=source,
        target_manifest=target,
        source_assignments=source_assignments,
        target_assignments=target_assignments,
        finetune_assignments=finetune,
        hashes=hashes,
        metadata=metadata,
    )


def _config_value(config: Any, path: Sequence[str], default: Any) -> Any:
    current = config
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            try:
                current = getattr(current, key)
            except (AttributeError, KeyError):
                return default
    return current


def _default_historical_split_path(config: Any) -> Path:
    root = getattr(config, "_root", None)
    if root is not None:
        historical = Path(root) / "results" / "metrics" / "bus_bra_adaptation_test_split.csv"
        if historical.exists():
            return historical
    if config is not None and callable(getattr(config, "path", None)):
        return Path(config.path("results")) / "metrics" / "bus_bra_adaptation_test_split.csv"
    results = _config_value(config, ("paths", "results"), None)
    if results is not None:
        root = Path(getattr(config, "_root", "."))
        return root / str(results) / "metrics" / "bus_bra_adaptation_test_split.csv"
    raise ValueError(
        "Debe proporcionarse historical_target_split o un config con la ruta de results."
    )


def persist_publication_splits(
    bundle: PublicationSplitBundle,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write runner-facing assignments, isolated manifests, hashes and metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["source_assignments"] = output / "source_assignments.csv"
    bundle.source_assignments.to_csv(paths["source_assignments"], index=False)
    paths["target_assignments"] = output / "target_assignments.csv"
    bundle.target_assignments.to_csv(paths["target_assignments"], index=False)

    for seed, assignments in sorted(bundle.finetune_assignments.items()):
        key = f"finetune_assignments_seed{seed}"
        paths[key] = output / f"{key}.csv"
        assignments.to_csv(paths[key], index=False)

    for partition in SOURCE_PARTITIONS:
        key = f"{partition}_manifest"
        paths[key] = output / f"{key}.csv"
        subset = bundle.source_manifest.loc[
            bundle.source_manifest["partition"].eq(partition)
        ].copy()
        subset.to_csv(paths[key], index=False)

    for partition in TARGET_PARTITIONS:
        key = f"{partition}_manifest"
        paths[key] = output / f"{key}.csv"
        subset = bundle.target_manifest.loc[
            bundle.target_manifest["partition"].eq(partition)
        ].copy()
        subset.to_csv(paths[key], index=False)

    paths["assignment_hashes"] = output / "assignment_hashes.json"
    paths["assignment_hashes"].write_text(
        json.dumps(bundle.hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["split_metadata"] = output / "split_metadata.json"
    paths["split_metadata"].write_text(
        json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _source_input_sha256(value: FrameLike) -> str:
    if isinstance(value, (str, Path)):
        return hashlib.sha256(Path(value).read_bytes()).hexdigest()
    payload = value.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_publication_source_only_files(
    source_manifest: FrameLike,
    source_group_audit: FrameLike,
    output_dir: str | Path,
    *,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Build v3 source partitions from public source metadata only.

    This entry point intentionally has no target or historical-split argument,
    so it cannot read or persist target data. The source manifest is the local
    386-row Curated BUSI manifest; the group/decision crosswalk is public.
    """

    source = normalise_manifest_path_columns(_read_frame(source_manifest))
    audit = _read_frame(source_group_audit)
    if len(source) != 386:
        raise ValueError(f"El manifiesto BUSI fuente debe tener 386 filas; tiene {len(source)}.")
    required_source = {"label", "label_idx"}
    required_audit = {"label", "group_id", "decision", "decision_reason"}
    if not required_source.issubset(source.columns):
        missing = sorted(required_source - set(source))
        raise ValueError(f"Manifiesto fuente incompleto: faltan {missing}.")
    if not required_audit.issubset(audit.columns):
        missing = sorted(required_audit - set(audit))
        raise ValueError(f"Auditoría fuente incompleta: faltan {missing}.")

    key_pairs = [
        ("image_id", "image_id"),
        ("class_id", "class_id"),
    ]
    available_keys = [
        pair for pair in key_pairs if pair[0] in source.columns and pair[1] in audit.columns
    ]
    if not available_keys:
        raise ValueError("El cruce fuente requiere image_id global o class_id.")

    audit = audit.copy()
    audit["label"] = audit["label"].astype(str).str.strip().str.lower()
    audit["group_id"] = audit["group_id"].fillna("").astype(str).str.strip()
    if audit["group_id"].eq("").any():
        raise ValueError("La auditoría fuente contiene group_id vacío.")
    for _, audit_key in available_keys:
        audit[audit_key] = pd.to_numeric(audit[audit_key], errors="raise").astype(int)
        if audit.duplicated(["label", audit_key]).any():
            raise ValueError(f"Las claves de auditoría ({audit_key}, label) deben ser unique.")

    source = source.copy()
    source["label"] = source["label"].astype(str).str.strip().str.lower()
    for source_key, _ in available_keys:
        source[source_key] = pd.to_numeric(source[source_key], errors="raise").astype(int)
        if source.duplicated(["label", source_key]).any():
            raise ValueError(f"Las claves del manifiesto ({source_key}, label) deben ser unique.")

    audit_by_key = {
        audit_key: audit.set_index(["label", audit_key], drop=False)
        for _, audit_key in available_keys
    }

    def _audit_identity(record: Mapping[str, Any]) -> tuple[str, int]:
        identifier = record.get("image_id", record.get("class_id"))
        return str(record["label"]), int(identifier)

    matched_records: list[dict[str, Any] | None] = []
    for row in source.to_dict("records"):
        matches: list[dict[str, Any]] = []
        for source_key, audit_key in available_keys:
            key = (row["label"], int(row[source_key]))
            index = audit_by_key[audit_key].index
            if key in index:
                selected = audit_by_key[audit_key].loc[key]
                if isinstance(selected, pd.DataFrame):
                    matches.append(selected.iloc[0].to_dict())
                else:
                    matches.append(selected.to_dict())
        if matches and any(
            _audit_identity(item) != _audit_identity(matches[0])
            for item in matches[1:]
        ):
            raise ValueError(
                "Cruce fuente ambiguo: image_id y class_id identifican filas distintas."
            )
        matched_records.append(matches[0] if matches else None)
    if any(record is None for record in matched_records):
        raise ValueError("Cruce de auditoría fuente incompleto para el manifiesto de 386 imágenes.")

    crosswalk = pd.DataFrame(matched_records).reset_index(drop=True)
    crosswalk_key = "image_id" if "image_id" in crosswalk else "class_id"
    if crosswalk.duplicated(["label", crosswalk_key]).any():
        raise ValueError(f"El cruce fuente no es unique por {crosswalk_key}.")
    source = source.reset_index(drop=True)
    source["curation_decision"] = crosswalk["decision"].astype(str)
    source["curation_decision_reason"] = crosswalk["decision_reason"].astype(str)
    source["pawlowska_group_id"] = crosswalk["group_id"].astype(str)
    source["audit_image_id"] = (
        pd.to_numeric(crosswalk["image_id"], errors="raise").astype(int)
        if "image_id" in crosswalk
        else source.get("image_id", source.get("class_id"))
    )
    source["image_id"] = source["audit_image_id"]

    excluded = source.loc[~source["curation_decision"].eq("keep")]
    observed_exclusions = excluded["curation_decision_reason"].value_counts().to_dict()
    if observed_exclusions != V3_BUSI_EXCLUSION_COUNTS:
        raise ValueError(
            f"Exclusiones BUSI v3 inesperadas: {observed_exclusions}; "
            f"se esperaban {V3_BUSI_EXCLUSION_COUNTS}."
        )
    source = source.loc[source["curation_decision"].eq("keep")].copy().reset_index(drop=True)
    source["label_idx"] = pd.to_numeric(source["label_idx"], errors="raise").astype(int)
    observed_classes = source["label_idx"].value_counts().sort_index().to_dict()
    if observed_classes != V3_BUSI_CLASS_COUNTS:
        raise ValueError(
            f"Cohorte BUSI v3 inesperada: {observed_classes}; se esperaba {V3_BUSI_CLASS_COUNTS}."
        )

    # Apply only the three prespecified same-class pairs. Cross-class screening
    # pairs remain distinct even if a future audit row uses a shared annotation.
    explicit_ids: dict[tuple[str, int], str] = {}
    for label, left, right in V3_EXPLICIT_SOURCE_GROUPS:
        group_id = f"manual-{label}-{left}-{right}"
        explicit_ids[(label, left)] = group_id
        explicit_ids[(label, right)] = group_id
    source["pawlowska_group_id"] = [
        explicit_ids.get((str(row.label), int(row.audit_image_id)), row.pawlowska_group_id)
        for row in source.itertuples(index=False)
    ]

    filtered_audit = source[["label", "audit_image_id", "pawlowska_group_id"]].rename(
        columns={"audit_image_id": "image_id", "pawlowska_group_id": "group_id"}
    )
    source = source.drop(columns=["pawlowska_group_id"])
    source, assignments = make_busi_group_disjoint_source_assignments(
        source,
        filtered_audit,
        seed=seed,
        test_fraction=0.20,
        val_fraction_of_remainder=0.20,
        expected_rows=358,
        expected_class_counts=V3_BUSI_CLASS_COUNTS,
    )
    source["split_unit"] = "reviewed_related_image_group_v3"
    assignments["split_unit"] = "reviewed_related_image_group_v3"
    source["split_seed"] = int(seed)
    assignments["split_seed"] = int(seed)
    expected_counts = {"source_train": 229, "source_val": 57, "source_test": 72}
    observed_counts = source["partition"].value_counts().sort_index().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Conteos de partición fuente v3 inesperados: {observed_counts}.")

    hashes = {
        "source_assignments": canonical_assignment_hash(
            assignments, ["sample_id", "partition"]
        ),
        "source_manifest_input": _source_input_sha256(source_manifest),
        "source_group_audit": _source_input_sha256(source_group_audit),
    }
    metadata: dict[str, Any] = {
        "protocol": "publication_v3_source_only",
        "split_seed": int(seed),
        "source_rows_manifest": 386,
        "source_rows_kept": 358,
        "source_class_counts": {"benign": 212, "malignant": 146},
        "source_exclusions": observed_exclusions,
        "source_partition_counts": observed_counts,
        "source_split_unit": "reviewed_related_image_group_v3",
        "source_group_count": int(source["pawlowska_group_id"].nunique()),
        "explicit_source_groups": [list(values) for values in V3_EXPLICIT_SOURCE_GROUPS],
        "hashes": hashes,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["source_assignments"] = output / "source_assignments.csv"
    assignments.to_csv(paths["source_assignments"], index=False)
    for partition in SOURCE_PARTITIONS:
        key = f"{partition}_manifest"
        paths[key] = output / f"{key}.csv"
        source.loc[source["partition"].eq(partition)].to_csv(paths[key], index=False)
        if partition in ("source_train", "source_val"):
            metadata[f"{key}_sha256"] = hashlib.sha256(paths[key].read_bytes()).hexdigest()
    paths["source_assignment_hashes"] = output / "source_assignment_hashes.json"
    paths["source_assignment_hashes"].write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["source_split_metadata"] = output / "source_split_metadata.json"
    paths["source_split_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "source_manifest": source,
        "source_assignments": assignments,
        "hashes": hashes,
        "metadata": metadata,
        "paths": paths,
    }


def create_publication_split_files(
    manifests: Mapping[str, FrameLike],
    config: Any,
    output_dir: str | Path,
    *,
    historical_target_split: FrameLike | None = None,
    seeds: Sequence[int] | None = None,
    source_key: str = "busi",
    target_key: str = "bus_bra",
    source_test_fraction: float = 0.20,
    source_val_fraction_of_remainder: float = 0.20,
    calibration_fraction_of_historical_adapt: float = 1 / 6,
    budget_fractions: Sequence[float] = DEFAULT_BUDGETS,
    finetune_val_fraction: float = 0.20,
    source_group_audit: FrameLike | None = None,
) -> PublicationSplitBundle:
    """Main runner integration point: build, verify, and write all split files.

    ``output_dir`` should normally be
    ``results/publication_v2/splits``.  Seed lists may be supplied directly or
    through ``config.publication_splits.seeds``.  If neither exists, the
    project's top-level ``config.seed`` is used.
    """

    if source_key not in manifests or target_key not in manifests:
        raise KeyError(f"manifests debe contener {source_key!r} y {target_key!r}.")
    source = normalise_manifest_path_columns(_read_frame(manifests[source_key]))
    target = normalise_manifest_path_columns(_read_frame(manifests[target_key], patient_ids=True))
    target_runtime_pixels = str(
        _config_value(config, ("datasets", target_key, "runtime_pixels"), "image_path")
    )
    target = select_runtime_image_paths(target, path_column=target_runtime_pixels)
    historical_value = (
        historical_target_split
        if historical_target_split is not None
        else _default_historical_split_path(config)
    )
    historical = _read_frame(historical_value)

    split_seed = int(
        _config_value(
            config,
            ("publication_splits", "split_seed"),
            _config_value(
                config,
                ("publication", "split_seed"),
                _config_value(config, ("seed",), 42),
            ),
        )
    )
    if seeds is None:
        configured = _config_value(config, ("publication_splits", "seeds"), None)
        if configured is None:
            configured = _config_value(config, ("publication", "seeds"), None)
        seeds = configured if configured is not None else [split_seed]

    bundle = build_publication_splits(
        source,
        target,
        historical,
        split_seed=split_seed,
        finetune_seeds=seeds,
        source_test_fraction=source_test_fraction,
        source_val_fraction_of_remainder=source_val_fraction_of_remainder,
        calibration_fraction_of_historical_adapt=calibration_fraction_of_historical_adapt,
        budget_fractions=budget_fractions,
        finetune_val_fraction=finetune_val_fraction,
        source_group_audit=source_group_audit,
    )
    bundle.metadata["source_runtime_pixels"] = "image_path"
    bundle.metadata["target_runtime_pixels"] = target_runtime_pixels
    bundle.metadata["common_runtime_resize"] = True
    persist_publication_splits(bundle, output_dir)
    return bundle


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crear las particiones reproducibles del protocolo seleccionado."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--historical-target-split", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Genera únicamente particiones, manifiestos y huellas de la fuente BUSI v3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI used before any training command for the selected protocol."""

    from ..config import load_config

    args = build_cli_parser().parse_args(argv)
    cfg = load_config(args.config)
    source_manifest = args.source_manifest or cfg.resolve(cfg.datasets.busi.manifest)
    output_dir = args.output_dir or (cfg.path("results") / "splits")
    publication = cfg.publication
    source_group_audit = publication.get("source_group_audit")
    if source_group_audit is not None:
        source_group_audit = cfg.resolve(str(source_group_audit))

    if args.source_only:
        result = create_publication_source_only_files(
            source_manifest,
            source_group_audit,
            output_dir,
            seed=int(publication.split_seed),
        )
        print(f"[splits] artefactos fuente v3 escritos en {Path(output_dir).resolve()}")
        print(json.dumps(result["metadata"], indent=2, sort_keys=True))
        print("[splits] SHA-256")
        for name, digest in sorted(result["hashes"].items()):
            print(f"  {name}: {digest}")
        return 0

    target_manifest = args.target_manifest or cfg.resolve(cfg.datasets.bus_bra.manifest)
    historical_split = args.historical_target_split or (
        cfg._root / "results" / "metrics" / "bus_bra_adaptation_test_split.csv"
    )
    bundle = create_publication_split_files(
        {"busi": source_manifest, "bus_bra": target_manifest},
        cfg,
        output_dir,
        historical_target_split=historical_split,
        seeds=args.seeds,
        source_test_fraction=float(publication.source_test_fraction),
        source_val_fraction_of_remainder=float(publication.source_val_fraction_of_remainder),
        calibration_fraction_of_historical_adapt=float(
            publication.target_calibration_fraction_of_adapt_pool
        ),
        budget_fractions=tuple(float(value) for value in publication.finetune_fractions),
        source_group_audit=source_group_audit,
    )
    print(f"[splits] escritos en {Path(output_dir).resolve()}")
    print(json.dumps(bundle.metadata, indent=2, sort_keys=True))
    print("[splits] SHA-256")
    for name, digest in sorted(bundle.hashes.items()):
        print(f"  {name}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
