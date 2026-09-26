"""Unlock and run the single final inference for the selected protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.data.datasets import roi_mask_paths  # noqa: E402
from src.data.publication_splits import canonical_assignment_hash  # noqa: E402
from src.evaluation.publication_protocol import (  # noqa: E402
    aggregate_patients,
    apply_temperature,
    calibrate_from_patient_table,
    infer_logits,
    prediction_metrics,
    select_youden_threshold,
)
from src.training.experiment_matrix import build_experiment_matrix  # noqa: E402
from src.training.publication_protocol import sha256_file  # noqa: E402

FINALIZATION_CODE_FILES = (
    "scripts/12_finalize_publication_inference.py",
    "scripts/13_analyze_publication_results.py",
    "src/config.py",
    "src/data/datasets.py",
    "src/evaluation/calibration.py",
    "src/evaluation/metrics.py",
    "src/evaluation/publication_protocol.py",
    "src/evaluation/publication_statistics.py",
    "src/models/architectures.py",
    "src/training/publication_protocol.py",
    "src/training/control_preprocessing.py",
    "src/training/experiment_matrix.py",
    "src/training/transforms.py",
    "requirements.txt",
)
TRAINING_PROVENANCE_KEYS = (
    "protocol_version",
    "effective_config_sha256",
    "source_train_sha256",
    "source_val_sha256",
    "target_adapt_sha256",
    "training_code_sha256",
    "training_images_sha256",
    "training_image_hash_manifest_sha256",
    "split_assignment_hashes_sha256",
    "split_metadata_sha256",
    "runtime_fingerprint_sha256",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _code_fingerprint(root: Path) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in sorted(FINALIZATION_CODE_FILES):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing finalization provenance input: {path}")
        file_hash = sha256_file(path)
        hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), hashes


def _runtime_fingerprint() -> str:
    import albumentations
    import scipy
    import sklearn
    import timm

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "albumentations": albumentations.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "timm": timm.__version__,
    }
    return _fingerprint_sha256(runtime)


def _fingerprint_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _acquire_finalization_transition(results: Path) -> Path:
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = logs / "TRAINING_ACTIVE.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Training or another finalizer is active according to {lock}.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {"pid": os.getpid(), "started_at": _now(), "mode": "finalization_transition"},
            stream,
        )
    return lock


def _release_transition(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def _run_with_finalization_lock(results: Path, action: Callable[[], Any]) -> Any:
    """Keep the training/finalization exclusion lock through artifact completion."""
    lock = _acquire_finalization_transition(results)
    try:
        return action()
    finally:
        _release_transition(lock)


def _freeze_test_access(
    results: Path,
    preliminary_fingerprint: dict[str, Any],
    *,
    resume: bool,
) -> str:
    fingerprint_sha256 = _fingerprint_sha256(preliminary_fingerprint)
    marker = results / "TEST_ACCESS_STARTED.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior.get("test_access_fingerprint_sha256") != fingerprint_sha256:
            raise RuntimeError("Cannot resume: preliminary test-access fingerprint changed.")
        if not resume:
            raise RuntimeError(
                f"Test access was already initiated ({marker}). Use --resume only for "
                "the identical frozen run."
            )
        return fingerprint_sha256
    record = {
        "started_at": _now(),
        "status": "irreversibly_frozen_before_test_read",
        "test_access_fingerprint_sha256": fingerprint_sha256,
        **preliminary_fingerprint,
    }
    try:
        with open(marker, "x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
    except FileExistsError as exc:
        raise RuntimeError("A concurrent process initiated test access.") from exc
    return fingerprint_sha256


def _plain_config(value: Any, *, top_level: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _plain_config(item)
            for key, item in value.items()
            if not (top_level and str(key).startswith("_"))
        }
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _semantic_config_sha256(cfg) -> str:
    return _fingerprint_sha256(_plain_config(cfg, top_level=True))


def _portable_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_artifact(path: str | Path, root: Path) -> Path:
    value = Path(str(path).replace("\\", "/"))
    return (value if value.is_absolute() else root / value).resolve()


def _image_fingerprint(
    root: Path,
    cohorts: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str, str]:
    rows: list[dict[str, str]] = []
    resolved_root = root.resolve()
    for cohort, frame in cohorts.items():
        for row in frame[["sample_id", "image_path"]].itertuples(index=False):
            raw_path = Path(str(row.image_path).replace("\\", "/"))
            path = raw_path if raw_path.is_absolute() else resolved_root / raw_path
            path = path.resolve()
            try:
                relative = path.relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Image path escapes the project root: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "cohort": cohort,
                    "sample_id": str(row.sample_id),
                    "image_path": relative,
                    "image_sha256": sha256_file(path),
                }
            )
    table = pd.DataFrame(rows).sort_values(["cohort", "sample_id"]).reset_index(drop=True)
    digest = hashlib.sha256()
    for row in table.itertuples(index=False):
        digest.update(
            f"{row.cohort}\0{row.sample_id}\0{row.image_path}\0{row.image_sha256}\n".encode("utf-8")
        )
    csv_text = table.to_csv(index=False, lineterminator="\n")
    return table, digest.hexdigest(), hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _roi_mask_sha256(root: Path, frames: tuple[pd.DataFrame, ...]) -> str:
    """Hash every ROI mask byte in manifest order, matching the training protocol."""
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for frame in frames:
        for _, row in frame.iterrows():
            for part in roi_mask_paths(row):
                path = (resolved_root / part).resolve()
                if not path.is_relative_to(resolved_root) or not path.is_file():
                    raise ValueError(f"ROI mask missing or outside the project: {part}")
                digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _validate_training_code(root: Path, provenance: dict[str, Any]) -> None:
    expected = provenance.get("training_code_files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Training provenance lacks per-file code hashes.")
    changed = [
        relative
        for relative, digest in expected.items()
        if not (root / relative).exists() or sha256_file(root / relative) != digest
    ]
    if changed:
        raise RuntimeError(
            "Training code changed after checkpoint creation: " + ", ".join(sorted(changed))
        )


def _freeze_or_validate_finalization(
    results: Path,
    fingerprint: dict[str, Any],
    *,
    resume: bool,
) -> str:
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    started_path = results / "FINAL_INFERENCE_STARTED.json"
    completed_path = results / "FINAL_INFERENCE_LOCK.json"

    for path in (completed_path, started_path):
        if not path.exists():
            continue
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("finalization_fingerprint_sha256") != fingerprint_sha256:
            raise RuntimeError(
                f"Cannot mix final inference protocols: fingerprint differs from {path}."
            )
        if not resume:
            raise RuntimeError(
                f"Final inference was already started ({path}). Use --resume only to "
                "continue the identical frozen run."
            )

    if not started_path.exists():
        record = {
            "started_at": _now(),
            "status": "started",
            "finalization_fingerprint_sha256": fingerprint_sha256,
            **fingerprint,
        }
        try:
            with open(started_path, "x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False)
        except FileExistsError as exc:
            raise RuntimeError("A concurrent final-inference process acquired the lock.") from exc
    return fingerprint_sha256


def _read_manifest(path: Path, *, require_patient: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    dtype = {"sample_id": str}
    if require_patient:
        dtype["patient_id"] = str
    df = pd.read_csv(path, dtype=dtype).fillna("")
    required = {"sample_id", "image_path", "label", "label_idx"}
    if require_patient:
        required.add("patient_id")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate sample IDs.")
    if require_patient and df["patient_id"].astype(str).str.len().eq(0).any():
        raise ValueError(f"{path} has missing patient IDs.")
    return df.reset_index(drop=True)


def _assert_disjoint(label: str, frames: dict[str, pd.DataFrame], column: str) -> None:
    keys = list(frames)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = set(frames[left][column].astype(str)) & set(frames[right][column].astype(str))
            if overlap:
                raise ValueError(f"{label} leakage between {left}/{right}: {sorted(overlap)[:5]}")


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _verify_checkpoint_lineage(
    index: pd.DataFrame,
    training_provenance: dict[str, Any],
    splits: Path,
) -> None:
    source_hashes = {
        (row.arch, int(row.seed)): row.checkpoint_sha256
        for row in index[index["method"].eq("source_direct")].itertuples()
    }
    for row in index.itertuples():
        state = torch.load(row.checkpoint_path, map_location="cpu", weights_only=False)
        expected_top_level = {
            "arch": str(row.arch),
            "seed": int(row.seed),
            "method": str(row.method),
            "model_type": "dann" if row.method == "dann" else "baseline",
        }
        for key, expected in expected_top_level.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"Checkpoint/index mismatch for {row.arch}/{row.method}/seed={row.seed}: "
                    f"{key}={state.get(key)!r}, expected {expected!r}."
                )
        provenance = state.get("provenance", {})
        for key in TRAINING_PROVENANCE_KEYS:
            if provenance.get(key) != training_provenance.get(key):
                raise ValueError(
                    f"Checkpoint provenance mismatch for {row.arch}/{row.method}/"
                    f"seed={row.seed}: {key}."
                )
        if row.method != "source_direct":
            expected_source = source_hashes.get((row.arch, int(row.seed)))
            observed_source = provenance.get("initial_checkpoint_sha256")
            if expected_source is None or observed_source != expected_source:
                raise ValueError(
                    f"Broken source lineage for {row.arch}/{row.method}/seed={row.seed}"
                )
        if str(row.method).startswith("finetune_"):
            assignment = splits / f"finetune_assignments_seed{int(row.seed)}.csv"
            if provenance.get("budget_assignment_sha256") != sha256_file(assignment):
                raise ValueError(
                    f"Fine-tuning assignment changed for {row.arch}/{row.method}/"
                    f"seed={row.seed}."
                )


def _field(value: Any) -> str:
    """Turn CSV blanks into empty strings without accepting literal missing values."""
    return "" if pd.isna(value) else str(value)


def _validate_v3_index(index: pd.DataFrame, cfg: Config, root: Path) -> dict[str, Any]:
    """Require exactly the frozen 160 v3 identities and their immutable descriptors."""
    required = {
        "experiment_id", "family", "stage", "arch", "seed", "method",
        "preprocessing", "hyperparameter", "hyperparameter_value", "parent_id",
        "parent_checkpoint_sha256", "variant_config_sha256", "checkpoint_path",
        "checkpoint_sha256",
    }
    missing_columns = required - set(index.columns)
    if missing_columns:
        raise ValueError(f"v3 checkpoint index lacks columns {sorted(missing_columns)}")
    jobs = build_experiment_matrix(
        architectures=tuple(cfg.publication.architectures),
        seeds=tuple(int(seed) for seed in cfg.publication.seeds),
    )
    expected = {job.experiment_id: job for job in jobs}
    observed = index["experiment_id"].astype(str).tolist()
    if len(index) != 160 or len(set(observed)) != 160 or set(observed) != set(expected):
        raise ValueError("v3 checkpoint index must contain exactly 160 unique frozen variants")
    if sum(job.trainable for job in jobs) != 150 or sum(not job.trainable for job in jobs) != 10:
        raise ValueError("v3 registry must contain 150 trained and 10 AdaBN variants")
    result_path = cfg.path("results")
    for row in index.itertuples(index=False):
        job = expected[str(row.experiment_id)]
        for key in (
            "family", "stage", "arch", "method", "preprocessing", "hyperparameter", "parent_id"
        ):
            if _field(getattr(row, key)) != str(getattr(job, key)):
                raise ValueError(f"v3 index descriptor mismatch for {job.experiment_id}: {key}")
        if int(row.seed) != job.seed:
            raise ValueError(f"v3 index seed mismatch for {job.experiment_id}")
        value = row.hyperparameter_value
        if (job.value is None and _field(value)) or (
            job.value is not None and (not _field(value) or float(value) != float(job.value))
        ):
            raise ValueError(f"v3 index hyperparameter mismatch for {job.experiment_id}")
        expected_path = (result_path / "checkpoints" / "v3" / f"{job.experiment_id}.pt").resolve()
        actual_path = _resolve_artifact(row.checkpoint_path, root)
        if actual_path != expected_path:
            raise ValueError(f"v3 index checkpoint path mismatch for {job.experiment_id}")
        if not _field(row.variant_config_sha256) or not _field(row.checkpoint_sha256):
            raise ValueError(f"v3 index lacks digests for {job.experiment_id}")
        variant_cfg = _config_for_v3_job(cfg, job)
        expected_variant_sha256 = _fingerprint_sha256(
            {
                "config": _plain_config(variant_cfg, top_level=True),
                "preprocessing": job.preprocessing,
            }
        )
        if _field(row.variant_config_sha256) != expected_variant_sha256:
            raise ValueError(f"v3 variant config hash mismatch for {job.experiment_id}")
    return expected


def _verify_v3_checkpoint_lineage(
    index: pd.DataFrame,
    training_provenance: dict[str, Any],
    splits: Path,
    *,
    source_roi_masks_sha256: str | None = None,
) -> None:
    """Verify every checkpoint's identity, frozen run, variant, and parent bytes."""
    by_id = {str(row.experiment_id): row for row in index.itertuples(index=False)}
    for row in index.itertuples(index=False):
        state = torch.load(row.checkpoint_path, map_location="cpu", weights_only=False)
        expected_top_level = {
            "arch": str(row.arch),
            "seed": int(row.seed),
            "method": str(row.method),
            "model_type": "dann" if row.method == "dann" else "baseline",
        }
        for key, expected in expected_top_level.items():
            if state.get(key) != expected:
                raise ValueError(f"v3 checkpoint identity mismatch for {row.experiment_id}: {key}")
        provenance = state.get("provenance", {})
        for key in TRAINING_PROVENANCE_KEYS:
            if provenance.get(key) != training_provenance.get(key):
                raise ValueError(f"v3 provenance mismatch for {row.experiment_id}: {key}")
        for key in (
            "experiment_id", "family", "preprocessing", "hyperparameter", "parent_id",
            "parent_checkpoint_sha256", "variant_config_sha256",
        ):
            if _field(provenance.get(key)) != _field(getattr(row, key)):
                raise ValueError(f"v3 variant provenance mismatch for {row.experiment_id}: {key}")
        stored_value = provenance.get("hyperparameter_value")
        indexed_value = row.hyperparameter_value
        if (_field(stored_value) or _field(indexed_value)) and (
            not _field(stored_value)
            or not _field(indexed_value)
            or float(stored_value) != float(indexed_value)
        ):
            raise ValueError(f"v3 hyperparameter provenance mismatch for {row.experiment_id}")
        parent_id = _field(row.parent_id)
        if parent_id:
            parent = by_id.get(parent_id)
            if parent is None or _field(row.parent_checkpoint_sha256) != _field(
                parent.checkpoint_sha256
            ):
                raise ValueError(f"v3 parent digest mismatch for {row.experiment_id}")
            if provenance.get("initial_checkpoint_sha256") != parent.checkpoint_sha256:
                raise ValueError(f"v3 initial parent mismatch for {row.experiment_id}")
        elif _field(row.parent_checkpoint_sha256):
            raise ValueError(
                f"v3 root checkpoint has unexpected parent digest: {row.experiment_id}"
            )
        if row.method.startswith("finetune_"):
            assignment = splits / f"finetune_assignments_seed{int(row.seed)}.csv"
            if provenance.get("budget_assignment_sha256") != sha256_file(assignment):
                raise ValueError(f"v3 fine-tuning assignment changed for {row.experiment_id}")
        if row.preprocessing == "roi" and not provenance.get("roi_masks_sha256"):
            raise ValueError(f"v3 ROI mask provenance missing for {row.experiment_id}")
        if row.preprocessing == "roi" and source_roi_masks_sha256 is not None:
            if provenance["roi_masks_sha256"] != source_roi_masks_sha256:
                raise ValueError(f"v3 ROI mask bytes changed for {row.experiment_id}")


def _split_provenance_files(protocol_version: str, splits: Path) -> tuple[Path, Path]:
    if str(protocol_version).startswith("3."):
        return splits / "source_assignment_hashes.json", splits / "source_split_metadata.json"
    return splits / "assignment_hashes.json", splits / "split_metadata.json"


def _validate_v3_target_assignments(
    splits: Path,
    target_adapt: pd.DataFrame,
    target_calibration: pd.DataFrame,
    target_test: pd.DataFrame,
) -> str:
    """Bind the historical target registry to all three locked manifests."""
    columns = ["sample_id", "patient_id", "partition"]
    assignments = pd.read_csv(splits / "target_assignments.csv", dtype=str).fillna("")
    registry = json.loads((splits / "assignment_hashes.json").read_text(encoding="utf-8"))
    combined = pd.concat(
        (target_adapt, target_calibration, target_test), ignore_index=True
    )
    if assignments["sample_id"].duplicated().any() or combined["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample in historical target assignment")
    for frame, partition in (
        (target_adapt, "target_adapt"),
        (target_calibration, "target_calibration"),
        (target_test, "target_test"),
    ):
        if not frame["partition"].eq(partition).all():
            raise ValueError(f"Unexpected target assignment partition: {partition}")
    assignment_hash = canonical_assignment_hash(assignments, columns)
    if (
        registry.get("target_assignments") != assignment_hash
        or canonical_assignment_hash(combined, columns) != assignment_hash
    ):
        raise ValueError("Historical target assignment differs from locked manifests or registry")
    return assignment_hash


def _ethics_basis_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Written ethics basis is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Written ethics basis is empty.")
    return sha256_file(path)


def _config_for_v3_job(cfg: Config, job: Any) -> Config:
    """Recreate the exact per-variant input pipeline used during v3 training."""
    clone = Config(copy.deepcopy(dict(cfg)))
    clone._root = cfg._root
    clone._config_path = cfg.get("_config_path", "")
    clone._control_preprocessing = job.preprocessing
    if job.family == "sensitivity":
        adaptation = clone.publication.adaptation
        if job.hyperparameter == "mmd_sigma_scale":
            adaptation.mmd_sigmas = [
                float(value) * float(job.value) for value in adaptation.mmd_sigmas
            ]
        else:
            adaptation[job.hyperparameter] = float(job.value)
    return clone


def _expected_index(cfg) -> set[tuple[str, int, str]]:
    methods = (
        ["source_direct"]
        + [str(v) for v in cfg.publication.methods]
        + [f"finetune_{int(round(float(v) * 100))}pct" for v in cfg.publication.finetune_fractions]
    )
    return {
        (str(arch), int(seed), method)
        for arch in cfg.publication.architectures
        for seed in cfg.publication.seeds
        for method in methods
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked final inference for publication protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help="Required acknowledgement that training and checkpoint selection are frozen.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ethics-basis",
        type=Path,
        help=(
            "Written basis explaining why no additional review was conducted for the "
            "secondary analysis of public de-identified data."
        ),
    )
    args = parser.parse_args()
    if not args.unlock_test:
        raise SystemExit(
            "Refusing to load target_test. Re-run with --unlock-test only after training is frozen."
        )

    requested_cfg = load_config(args.config, create_dirs=False)
    if str(requested_cfg.publication.protocol_version).startswith("3."):
        if args.ethics_basis is None:
            raise SystemExit(
                "Refusing v3 test access without a written ethics basis. "
                "Provide --ethics-basis with the documented secondary-analysis rationale."
            )
        args.ethics_basis_sha256 = _ethics_basis_sha256(args.ethics_basis)
    root = requested_cfg._root
    results = requested_cfg.path("results")
    _run_with_finalization_lock(
        results,
        lambda: _finalize_locked(args, requested_cfg, root, results),
    )


def _finalize_locked(
    args: argparse.Namespace, requested_cfg: Config, root: Path, results: Path
) -> None:
    effective_config_path = results / "effective_config.yaml"
    training_provenance_path = results / "logs" / "training_provenance.json"
    if not effective_config_path.exists() or not training_provenance_path.exists():
        raise FileNotFoundError("Frozen effective config or training provenance is missing.")
    cfg = load_config(effective_config_path)
    if cfg.path("results") != results:
        raise ValueError("Frozen effective config points to a different results directory.")
    if _semantic_config_sha256(requested_cfg) != _semantic_config_sha256(cfg):
        raise ValueError(
            "Requested config differs semantically from the effective training config."
        )
    v3 = str(cfg.publication.protocol_version).startswith("3.")
    if v3:
        current_ethics_sha256 = _ethics_basis_sha256(args.ethics_basis)
        if current_ethics_sha256 != args.ethics_basis_sha256:
            raise RuntimeError("Written ethics basis changed during finalization.")
    training_provenance = json.loads(training_provenance_path.read_text(encoding="utf-8"))
    effective_config_sha256 = sha256_file(effective_config_path)
    if training_provenance.get("effective_config_sha256") != effective_config_sha256:
        raise ValueError("Effective config hash does not match training provenance.")
    _validate_training_code(root, training_provenance)
    if training_provenance.get("runtime_fingerprint_sha256") != _runtime_fingerprint():
        raise RuntimeError("Runtime environment changed after checkpoint creation.")
    splits = results / "splits"
    checkpoint_index_path = results / "checkpoints" / "checkpoint_index.csv"
    if not checkpoint_index_path.exists():
        raise FileNotFoundError("Missing checkpoint_index.csv; training is incomplete.")
    index = pd.read_csv(checkpoint_index_path)
    required_columns = {
        "arch",
        "seed",
        "method",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    if required_columns - set(index.columns):
        raise ValueError("Checkpoint index is incomplete.")
    v3_jobs: dict[str, Any] = {}
    if v3:
        v3_jobs = _validate_v3_index(index, cfg, root)
    else:
        if index.duplicated(["arch", "seed", "method"]).any():
            raise ValueError("Checkpoint index contains duplicate experiment identities.")
        observed = {(str(row.arch), int(row.seed), str(row.method)) for row in index.itertuples()}
        missing = _expected_index(cfg) - observed
        extra = observed - _expected_index(cfg)
        if missing or extra:
            raise ValueError(
                f"Checkpoint matrix does not match frozen config; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
    index["checkpoint_path"] = index["checkpoint_path"].map(
        lambda value: str(
            (
                Path(value)
                if Path(value).is_absolute()
                else (root / Path(str(value).replace("\\", "/")))
            ).resolve()
        )
    )
    for row in index.itertuples():
        path = Path(row.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(row.checkpoint_sha256):
            raise ValueError(f"Checkpoint digest mismatch: {path}")
    source_train = source_val = None
    if v3:
        # Development manifests and ROI masks are safe to verify before test access.
        source_train = _read_manifest(splits / "source_train_manifest.csv", require_patient=False)
        source_val = _read_manifest(splits / "source_val_manifest.csv", require_patient=False)
        source_roi_masks_sha256 = _roi_mask_sha256(root, (source_train, source_val))
        _verify_v3_checkpoint_lineage(
            index,
            training_provenance,
            splits,
            source_roi_masks_sha256=source_roi_masks_sha256,
        )
    else:
        _verify_checkpoint_lineage(index, training_provenance, splits)
    finalization_code_sha256, finalization_code_files = _code_fingerprint(root)
    preliminary_fingerprint = {
        "config_sha256": sha256_file(args.config),
        "effective_config_sha256": effective_config_sha256,
        "semantic_config_sha256": _semantic_config_sha256(cfg),
        "training_provenance_sha256": sha256_file(training_provenance_path),
        "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
        "finalization_code_sha256": finalization_code_sha256,
        "finalization_code_files": finalization_code_files,
        "n_experiments": int(len(index)),
    }
    if v3:
        preliminary_fingerprint["ethics_basis_sha256"] = args.ethics_basis_sha256
    test_access_fingerprint_sha256 = _freeze_test_access(
        results,
        preliminary_fingerprint,
        resume=args.resume,
    )
    manifest_paths = {
        "source_train_manifest_sha256": splits / "source_train_manifest.csv",
        "source_val_manifest_sha256": splits / "source_val_manifest.csv",
        "source_test_manifest_sha256": splits / "source_test_manifest.csv",
        "target_adapt_manifest_sha256": splits / "target_adapt_manifest.csv",
        "target_calibration_manifest_sha256": splits / "target_calibration_manifest.csv",
        "target_test_manifest_sha256": splits / "target_test_manifest.csv",
    }
    for path in manifest_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if not v3:
        source_train = _read_manifest(splits / "source_train_manifest.csv", require_patient=False)
        source_val = _read_manifest(splits / "source_val_manifest.csv", require_patient=False)
    source_test = _read_manifest(splits / "source_test_manifest.csv", require_patient=False)
    target_adapt = _read_manifest(splits / "target_adapt_manifest.csv", require_patient=True)
    target_calibration = _read_manifest(
        splits / "target_calibration_manifest.csv",
        require_patient=True,
    )
    target_test = _read_manifest(splits / "target_test_manifest.csv", require_patient=True)
    _assert_disjoint(
        "source sample",
        {"train": source_train, "val": source_val, "test": source_test},
        "sample_id",
    )
    _assert_disjoint(
        "target patient",
        {
            "adapt": target_adapt,
            "calibration": target_calibration,
            "test": target_test,
        },
        "patient_id",
    )
    if len(target_test) != 736 or target_test["patient_id"].nunique() != 426:
        raise ValueError("Frozen BUS-BRA test changed; expected 736 images from 426 patients.")

    manifest_hashes = {key: sha256_file(path) for key, path in manifest_paths.items()}
    target_assignment_hash = (
        _validate_v3_target_assignments(
            splits, target_adapt, target_calibration, target_test
        )
        if v3 else None
    )
    assignment_path, metadata_path = _split_provenance_files(
        str(cfg.publication.protocol_version), splits
    )
    if training_provenance.get("split_assignment_hashes_sha256") != sha256_file(
        assignment_path
    ):
        raise ValueError("Split assignment hash registry changed after training.")
    if training_provenance.get("split_metadata_sha256") != sha256_file(
        metadata_path
    ):
        raise ValueError("Split metadata changed after training.")
    for key in ("source_train_sha256", "source_val_sha256", "target_adapt_sha256"):
        manifest_key = key.replace("_sha256", "_manifest_sha256")
        if training_provenance.get(key) != manifest_hashes[manifest_key]:
            raise ValueError(f"Training manifest changed after checkpoint creation: {key}.")
    _, training_images_sha256, training_image_csv_sha256 = _image_fingerprint(
        root,
        {
            "source_train": source_train,
            "source_val": source_val,
            "target_adapt": target_adapt,
        },
    )
    if training_images_sha256 != training_provenance.get(
        "training_images_sha256"
    ) or training_image_csv_sha256 != training_provenance.get(
        "training_image_hash_manifest_sha256"
    ):
        raise ValueError("Training image bytes changed after checkpoint creation.")
    inference_image_table, inference_images_sha256, inference_image_csv_sha256 = _image_fingerprint(
        root,
        {
            "source_val": source_val,
            "source_test": source_test,
            "target_calibration": target_calibration,
            "target_test": target_test,
        },
    )
    inference_roi_masks_sha256 = (
        _roi_mask_sha256(root, (source_val, source_test, target_calibration, target_test))
        if v3 else None
    )
    fingerprint = {
        "config_sha256": sha256_file(args.config),
        "effective_config_sha256": effective_config_sha256,
        "semantic_config_sha256": _semantic_config_sha256(cfg),
        "training_provenance_sha256": sha256_file(training_provenance_path),
        "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
        **manifest_hashes,
        "training_images_sha256": training_images_sha256,
        "inference_images_sha256": inference_images_sha256,
        "inference_image_hash_manifest_sha256": inference_image_csv_sha256,
        "test_access_fingerprint_sha256": test_access_fingerprint_sha256,
        "finalization_code_sha256": finalization_code_sha256,
        "finalization_code_files": finalization_code_files,
        "n_experiments": int(len(index)),
    }
    if v3:
        fingerprint["ethics_basis_sha256"] = args.ethics_basis_sha256
        fingerprint["inference_roi_masks_sha256"] = inference_roi_masks_sha256
        fingerprint["target_assignments_sha256"] = target_assignment_hash
    finalization_fingerprint_sha256 = _freeze_or_validate_finalization(
        results,
        fingerprint,
        resume=args.resume,
    )
    inference_image_table.to_csv(
        results / "FINAL_INFERENCE_IMAGE_HASHES.csv",
        index=False,
        lineterminator="\n",
    )

    prediction_root = results / "predictions"
    metrics_root = results / "metrics"
    calibration_root = results / "calibration"
    for path in (prediction_root, metrics_root, calibration_root):
        path.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    ordering = ["experiment_id"] if v3 else ["arch", "method", "seed"]
    for row in index.sort_values(ordering).itertuples():
        experiment_id = (
            str(row.experiment_id) if v3 else f"{row.arch}_{row.method}_seed{int(row.seed)}"
        )
        inference_cfg = _config_for_v3_job(cfg, v3_jobs[experiment_id]) if v3 else cfg
        metrics_path = metrics_root / f"{experiment_id}.json"
        if args.resume and metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("checkpoint_sha256") != row.checkpoint_sha256:
                raise ValueError(f"Cannot resume {experiment_id}: checkpoint changed.")
            if existing.get("finalization_fingerprint_sha256") != finalization_fingerprint_sha256:
                raise ValueError(
                    f"Cannot resume {experiment_id}: finalization fingerprint changed."
                )
            prediction_hashes = existing.get("prediction_sha256", {})
            prediction_paths = existing.get("prediction_paths", {})
            if set(prediction_hashes) != set(prediction_paths):
                raise ValueError(f"Cannot resume {experiment_id}: prediction index is incomplete.")
            for key, saved_path in prediction_paths.items():
                artifact = _resolve_artifact(saved_path, root)
                if not artifact.is_file() or sha256_file(artifact) != prediction_hashes[key]:
                    raise ValueError(
                        f"Cannot resume {experiment_id}: prediction artifact changed ({key})."
                    )
            metric_rows.extend(existing["metric_records"])
            calibration_rows.append(existing["calibration"])
            print(f"[skip] {experiment_id}")
            continue

        print(f"[infer] {experiment_id}: source validation/test")
        source_val_images, _ = infer_logits(
            row.checkpoint_path,
            source_val,
            root,
            inference_cfg,
        )
        source_threshold = select_youden_threshold(
            source_val_images["label_idx"].to_numpy(dtype=int),
            source_val_images["probability_raw"].to_numpy(dtype=float),
        )
        source_test_images, _ = infer_logits(
            row.checkpoint_path,
            source_test,
            root,
            inference_cfg,
        )
        source_metrics = prediction_metrics(
            source_test_images,
            probability_column="probability_raw",
            threshold=source_threshold,
        )

        print(f"[infer] {experiment_id}: target calibration")
        target_cal_images, _ = infer_logits(
            row.checkpoint_path,
            target_calibration,
            root,
            inference_cfg,
        )
        target_cal_patients = aggregate_patients(target_cal_images)
        calibration = calibrate_from_patient_table(target_cal_patients)
        target_cal_images = apply_temperature(target_cal_images, calibration["temperature"])
        target_cal_patients = apply_temperature(
            target_cal_patients,
            calibration["temperature"],
        )

        # This is the only point where target_test is passed to inference.
        print(f"[infer] {experiment_id}: target test (locked)")
        target_test_images, _ = infer_logits(
            row.checkpoint_path,
            target_test,
            root,
            inference_cfg,
        )
        target_test_patients = aggregate_patients(target_test_images)
        target_test_images = apply_temperature(
            target_test_images,
            calibration["temperature"],
        )
        target_test_patients = apply_temperature(
            target_test_patients,
            calibration["temperature"],
        )
        target_raw_metrics = prediction_metrics(
            target_test_patients,
            probability_column="probability_raw",
            threshold=calibration["threshold_raw"],
        )
        target_calibrated_metrics = prediction_metrics(
            target_test_patients,
            probability_column="probability_calibrated",
            threshold=calibration["threshold_calibrated"],
        )

        metadata = {
            "experiment_id": experiment_id,
            "arch": str(row.arch),
            "method": str(row.method),
            "seed": int(row.seed),
            "checkpoint_path": str(Path(row.checkpoint_path).resolve()),
            "checkpoint_sha256": str(row.checkpoint_sha256),
            "finalization_fingerprint_sha256": finalization_fingerprint_sha256,
        }
        if v3:
            metadata.update(
                {
                    "family": str(row.family),
                    "preprocessing": str(row.preprocessing),
                    "hyperparameter": _field(row.hyperparameter),
                    "hyperparameter_value": (
                        None if not _field(row.hyperparameter_value)
                        else float(row.hyperparameter_value)
                    ),
                    "parent_id": _field(row.parent_id),
                }
            )
        for table in (
            source_val_images,
            source_test_images,
            target_cal_images,
            target_cal_patients,
            target_test_images,
            target_test_patients,
        ):
            for key, value in metadata.items():
                table[key] = value

        output_paths = {
            "source_val_images": prediction_root / "source_val" / f"{experiment_id}.csv",
            "source_test_images": prediction_root / "source_test" / f"{experiment_id}.csv",
            "target_calibration_images": (
                prediction_root / "target_calibration_images" / f"{experiment_id}.csv"
            ),
            "target_calibration_patients": (
                prediction_root / "target_calibration_patients" / f"{experiment_id}.csv"
            ),
            "target_test_images": (prediction_root / "target_test_images" / f"{experiment_id}.csv"),
            "target_test_patients": (
                prediction_root / "target_test_patients" / f"{experiment_id}.csv"
            ),
        }
        tables = {
            "source_val_images": source_val_images,
            "source_test_images": source_test_images,
            "target_calibration_images": target_cal_images,
            "target_calibration_patients": target_cal_patients,
            "target_test_images": target_test_images,
            "target_test_patients": target_test_patients,
        }
        for key, path in output_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tables[key].to_csv(path, index=False)
        prediction_hashes = {key: sha256_file(path) for key, path in output_paths.items()}

        records = []
        for scope, values, threshold in (
            ("source_test_image_raw", source_metrics, source_threshold),
            (
                "target_test_patient_raw",
                target_raw_metrics,
                calibration["threshold_raw"],
            ),
            (
                "target_test_patient_calibrated",
                target_calibrated_metrics,
                calibration["threshold_calibrated"],
            ),
        ):
            records.append(
                {
                    **metadata,
                    "scope": scope,
                    "threshold_selected_outside_test": float(threshold),
                    **_jsonable(values),
                }
            )
        calibration_record = {
            **metadata,
            **{key: float(value) for key, value in calibration.items()},
            "n_calibration_patients": int(len(target_cal_patients)),
            "n_calibration_images": int(len(target_cal_images)),
        }
        payload = {
            **metadata,
            "metric_records": records,
            "calibration": calibration_record,
            "prediction_paths": {
                key: _portable_path(path, root) for key, path in output_paths.items()
            },
            "prediction_sha256": prediction_hashes,
        }
        metrics_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metric_rows.extend(records)
        calibration_rows.append(calibration_record)

    pd.DataFrame(metric_rows).to_csv(metrics_root / "per_seed_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        calibration_root / "temperature_and_thresholds.csv",
        index=False,
    )
    _, ending_inference_images_sha256, ending_image_csv_sha256 = _image_fingerprint(
        root,
        {
            "source_val": source_val,
            "source_test": source_test,
            "target_calibration": target_calibration,
            "target_test": target_test,
        },
    )
    if (
        ending_inference_images_sha256 != inference_images_sha256
        or ending_image_csv_sha256 != inference_image_csv_sha256
    ):
        raise RuntimeError("At least one inference image changed during final evaluation.")
    if v3 and _roi_mask_sha256(
        root, (source_val, source_test, target_calibration, target_test)
    ) != inference_roi_masks_sha256:
        raise RuntimeError("At least one ROI mask changed during final evaluation.")

    artifact_paths = sorted(prediction_root.rglob("*.csv")) + sorted(metrics_root.rglob("*.json"))
    artifact_paths += [
        metrics_root / "per_seed_metrics.csv",
        calibration_root / "temperature_and_thresholds.csv",
        results / "FINAL_INFERENCE_IMAGE_HASHES.csv",
    ]
    artifact_hashes = {
        _portable_path(path, root): sha256_file(path) for path in sorted(set(artifact_paths))
    }
    lock_record = {
        "unlocked": True,
        "status": "completed",
        "completed_at": _now(),
        "finalization_fingerprint_sha256": finalization_fingerprint_sha256,
        **fingerprint,
        "target_test_images": int(len(target_test)),
        "target_test_patients": int(target_test["patient_id"].nunique()),
        "artifact_hashes": artifact_hashes,
        "artifact_hashes_sha256": _fingerprint_sha256(artifact_hashes),
    }
    (results / "FINAL_INFERENCE_LOCK.json").write_text(
        json.dumps(lock_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] final inference completed for {len(index)} frozen checkpoints")


if __name__ == "__main__":
    main()
