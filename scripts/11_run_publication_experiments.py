"""Run the training-only stage of an explicitly selected publication protocol.

This entry point never opens ``source_test_manifest.csv``,
``target_calibration_manifest.csv`` or ``target_test_manifest.csv``.  Its complete input surface is
limited to source train/validation, unlabelled target-adapt, and the labelled fine-tuning budgets.
The locked sets are handled later by ``12_finalize_publication_inference.py``.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.training.publication_protocol import (  # noqa: E402
    sha256_file,
    train_finetune_checkpoint,
    train_matched_adaptation,
    train_source_checkpoint,
)

TRAINING_CODE_FILES = (
    "docs/PROTOCOL_PUBLICATION_V2.md",
    "scripts/11_run_publication_experiments.py",
    "src/config.py",
    "src/data/datasets.py",
    "src/evaluation/calibration.py",
    "src/evaluation/metrics.py",
    "src/models/architectures.py",
    "src/training/domain_adaptation.py",
    "src/training/publication_protocol.py",
    "src/training/transforms.py",
    "src/utils/seed.py",
    "requirements.txt",
)
RESUME_PROVENANCE_KEYS = (
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


def _read_manifest(path: Path, *, patient: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared split {path}. After protocol approval, run "
            "python -m src.data.publication_splits --config <config> first."
        )
    dtype = {"sample_id": str}
    if patient:
        dtype["patient_id"] = str
    df = pd.read_csv(path, dtype=dtype).fillna("")
    required = {"sample_id", "image_path", "label", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        dup = df.loc[df["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"{path} has duplicated sample_id={dup}")
    return df.reset_index(drop=True)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty_at_run": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "", "git_dirty_at_run": None}


def _training_code_fingerprint(
    root: Path, *, protocol_version: str = "2.0"
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    digest = hashlib.sha256()
    protocol_document = (
        "docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md"
        if str(protocol_version).startswith("3.")
        else "docs/PROTOCOL_PUBLICATION_V2.md"
    )
    code_files = (protocol_document,) + tuple(
        relative
        for relative in TRAINING_CODE_FILES
        if relative != "docs/PROTOCOL_PUBLICATION_V2.md"
    )
    for relative in sorted(code_files):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing training provenance input: {path}")
        file_hash = sha256_file(path)
        hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), hashes


def _runtime_fingerprint() -> tuple[str, dict[str, Any]]:
    import albumentations
    import scipy
    import sklearn
    import timm
    import torch

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
    serialized = json.dumps(
        runtime,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), runtime


def _portable_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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


def _effective_config_text(cfg) -> str:
    return yaml.safe_dump(
        _plain_config(cfg, top_level=True),
        sort_keys=False,
        allow_unicode=True,
    )


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
    if table.duplicated(["cohort", "sample_id"]).any():
        raise ValueError("Image fingerprint contains duplicate cohort/sample identities.")
    digest = hashlib.sha256()
    for row in table.itertuples(index=False):
        digest.update(
            f"{row.cohort}\0{row.sample_id}\0{row.image_path}\0{row.image_sha256}\n".encode("utf-8")
        )
    csv_text = table.to_csv(index=False, lineterminator="\n")
    return table, digest.hexdigest(), hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _acquire_training_lock(results: Path) -> Path:
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    final_markers = (
        "TEST_ACCESS_STARTED.json",
        "FINAL_INFERENCE_STARTED.json",
        "FINAL_INFERENCE_LOCK.json",
    )
    for marker in final_markers:
        if (results / marker).exists():
            raise RuntimeError(f"Training is permanently frozen because {results / marker} exists.")
    lock = logs / "TRAINING_ACTIVE.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another training process may be active ({lock}). "
            "Verify the recorded PID before removing a stale lock."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": _now()}, stream)
    # Close the check/create race with finalization. Both processes use this same
    # exclusive lock while transitioning protocol state.
    for marker in final_markers:
        if (results / marker).exists():
            lock.unlink(missing_ok=True)
            raise RuntimeError(f"Training is permanently frozen because {results / marker} exists.")
    return lock


def _release_training_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def _load_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _record_step(
    status_path: Path,
    status_rows: list[dict[str, Any]],
    *,
    experiment_id: str,
    metadata: dict[str, Any],
    fn,
) -> Any:
    row = {
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": _now(),
        "ended_at": "",
        "error": "",
        **metadata,
    }
    status_rows.append(row)
    _write_status(status_path, status_rows)
    try:
        result = fn()
        row["status"] = "success"
        return result
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        raise
    finally:
        row["ended_at"] = _now()
        _write_status(status_path, status_rows)


def _assert_resume_checkpoint(
    checkpoint: Path,
    *,
    arch: str,
    seed: int,
    method: str,
    model_type: str,
    expected_provenance: dict[str, Any],
    source_checkpoint: Path | None = None,
) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_top_level = {
        "arch": arch,
        "seed": int(seed),
        "method": method,
        "model_type": model_type,
    }
    for key, expected in expected_top_level.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"Stale checkpoint {checkpoint}: {key}={state.get(key)!r}, expected {expected!r}."
            )
    provenance = state.get("provenance", {})
    for key in RESUME_PROVENANCE_KEYS:
        expected = expected_provenance[key]
        if provenance.get(key) != expected:
            raise RuntimeError(f"Stale checkpoint {checkpoint}: provenance {key} changed.")
    for key in ("budget_assignment_sha256", "budget_fraction"):
        if key in expected_provenance and provenance.get(key) != expected_provenance[key]:
            raise RuntimeError(f"Stale checkpoint {checkpoint}: provenance {key} changed.")
    if source_checkpoint is not None:
        expected_source = sha256_file(source_checkpoint)
        if provenance.get("initial_checkpoint_sha256") != expected_source:
            raise RuntimeError(f"{checkpoint} does not derive from the current source checkpoint.")


def _validate_budget(
    target_adapt: pd.DataFrame,
    assignment_path: Path,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not assignment_path.exists():
        raise FileNotFoundError(f"Missing fine-tuning assignment: {assignment_path}")
    assignment = pd.read_csv(
        assignment_path,
        dtype={"sample_id": str, "patient_id": str},
    ).fillna("")
    required = {"sample_id", "budget_fraction", "role", "selected"}
    missing = required - set(assignment.columns)
    if missing:
        raise ValueError(f"{assignment_path} is missing {sorted(missing)}")
    budget = assignment[
        np.isclose(assignment["budget_fraction"].astype(float), float(fraction))
    ].copy()
    selected = budget["selected"].map(
        lambda value: (
            value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() == "true"
        )
    )
    budget = budget[selected].copy()
    if budget.empty:
        raise ValueError(f"No rows for budget={fraction} in {assignment_path}")
    if budget["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated sample IDs for budget={fraction} in {assignment_path}")
    unknown = set(budget["sample_id"]) - set(target_adapt["sample_id"])
    if unknown:
        raise ValueError(f"Budget contains samples outside target_adapt: {sorted(unknown)[:5]}")
    merged = budget[["sample_id", "role"]].merge(
        target_adapt,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    train = merged[merged["role"].eq("train")].drop(columns=["role"]).reset_index(drop=True)
    val = merged[merged["role"].eq("val")].drop(columns=["role"]).reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(f"Budget={fraction} requires non-empty train and val roles.")
    train_patients = set(train["patient_id"].astype(str))
    val_patients = set(val["patient_id"].astype(str))
    if train_patients & val_patients:
        raise ValueError(f"Patient leakage in budget={fraction}")
    summary = {
        "budget_assignment_path": str(assignment_path.resolve()),
        "budget_assignment_sha256": sha256_file(assignment_path),
        "budget_fraction": float(fraction),
        "n_selected_images": int(len(merged)),
        "n_selected_patients": int(merged["patient_id"].astype(str).nunique()),
    }
    return train, val, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train publication protocol v2 without test access"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg._root
    results = cfg.path("results")
    training_lock = _acquire_training_lock(results)
    atexit.register(_release_training_lock, training_lock)
    splits = results / "splits"
    checkpoints = results / "checkpoints"
    status_path = results / "logs" / "training_status.csv"
    status_rows = _load_status(status_path) if args.resume else []

    # Deliberately load only training-stage manifests.
    source_train_path = splits / "source_train_manifest.csv"
    source_val_path = splits / "source_val_manifest.csv"
    target_adapt_path = splits / "target_adapt_manifest.csv"
    source_train = _read_manifest(source_train_path)
    source_val = _read_manifest(source_val_path)
    target_adapt = _read_manifest(target_adapt_path, patient=True)
    image_hash_table, training_images_sha256, image_hash_csv_sha256 = _image_fingerprint(
        root,
        {
            "source_train": source_train,
            "source_val": source_val,
            "target_adapt": target_adapt,
        },
    )

    if set(source_train["sample_id"]) & set(source_val["sample_id"]):
        raise ValueError("Source train/validation overlap.")
    if target_adapt["patient_id"].astype(str).str.len().eq(0).any():
        raise ValueError("target_adapt contains missing patient identifiers.")
    if target_adapt.groupby("patient_id")["label_idx"].nunique().gt(1).any():
        raise ValueError("A target-adapt patient has inconsistent labels.")

    architectures = args.architectures or list(cfg.publication.architectures)
    seeds = args.seeds or [int(v) for v in cfg.publication.seeds]
    methods = [str(v) for v in cfg.publication.methods]
    fractions = [float(v) for v in cfg.publication.finetune_fractions]

    effective_config_path = results / "effective_config.yaml"
    effective_config_text = _effective_config_text(cfg)
    effective_config_candidate_sha256 = hashlib.sha256(
        effective_config_text.encode("utf-8")
    ).hexdigest()
    if args.resume:
        if not effective_config_path.exists():
            raise RuntimeError("Cannot resume without the original effective_config.yaml.")
        if sha256_file(effective_config_path) != effective_config_candidate_sha256:
            raise RuntimeError(
                "Cannot resume: requested config differs from the frozen effective config."
            )
    else:
        effective_config_path.write_bytes(effective_config_text.encode("utf-8"))
    training_code_sha256, training_code_files = _training_code_fingerprint(
        root, protocol_version=str(cfg.publication.protocol_version)
    )
    runtime_fingerprint_sha256, runtime = _runtime_fingerprint()
    base_provenance = {
        "protocol_version": str(cfg.publication.protocol_version),
        "config_path": str(Path(args.config).resolve()),
        "effective_config_path": str(effective_config_path.resolve()),
        "effective_config_sha256": sha256_file(effective_config_path),
        "source_train_sha256": sha256_file(source_train_path),
        "source_val_sha256": sha256_file(source_val_path),
        "target_adapt_sha256": sha256_file(target_adapt_path),
        "split_assignment_hashes_sha256": sha256_file(splits / "assignment_hashes.json"),
        "split_metadata_sha256": sha256_file(splits / "split_metadata.json"),
        "training_code_sha256": training_code_sha256,
        "training_code_files": training_code_files,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "runtime": runtime,
        "training_images_sha256": training_images_sha256,
        "training_image_hash_manifest_sha256": image_hash_csv_sha256,
        "n_training_images_hashed": int(len(image_hash_table)),
        **_git_state(root),
    }
    (results / "logs").mkdir(parents=True, exist_ok=True)
    provenance_path = results / "logs" / "training_provenance.json"
    if args.resume and provenance_path.exists():
        prior = json.loads(provenance_path.read_text(encoding="utf-8"))
        changed = [
            key for key in RESUME_PROVENANCE_KEYS if prior.get(key) != base_provenance.get(key)
        ]
        if changed:
            raise RuntimeError(
                "Cannot resume a different training protocol; changed provenance: "
                + ", ".join(changed)
            )
    image_hash_table.to_csv(
        results / "logs" / "training_image_hashes.csv",
        index=False,
        lineterminator="\n",
    )
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(base_provenance, fh, indent=2, ensure_ascii=False)

    checkpoint_rows: list[dict[str, Any]] = []
    for arch in architectures:
        for seed in seeds:
            source_path = checkpoints / "source" / f"{arch}_seed{seed}.pt"
            source_id = f"source_{arch}_seed{seed}"
            if source_path.exists() and args.resume:
                _assert_resume_checkpoint(
                    source_path,
                    arch=arch,
                    seed=seed,
                    method="source_direct",
                    model_type="baseline",
                    expected_provenance=base_provenance,
                )
                print(f"[skip] {source_id}")
            else:
                _record_step(
                    status_path,
                    status_rows,
                    experiment_id=source_id,
                    metadata={"phase": "source", "arch": arch, "seed": seed},
                    fn=lambda arch=arch, seed=seed, source_path=source_path: (
                        train_source_checkpoint(
                            source_train,
                            source_val,
                            root,
                            cfg,
                            arch=arch,
                            seed=seed,
                            checkpoint_path=source_path,
                            provenance=base_provenance,
                        )
                    ),
                )
            checkpoint_rows.append(
                {
                    "arch": arch,
                    "seed": seed,
                    "method": "source_direct",
                    "checkpoint_path": _portable_path(source_path, root),
                    "checkpoint_sha256": sha256_file(source_path),
                }
            )
            if args.source_only:
                continue

            for method in methods:
                output = checkpoints / "adaptation" / f"{arch}_{method}_seed{seed}.pt"
                exp_id = f"{method}_{arch}_seed{seed}"
                if output.exists() and args.resume:
                    _assert_resume_checkpoint(
                        output,
                        arch=arch,
                        seed=seed,
                        method=method,
                        model_type="dann" if method == "dann" else "baseline",
                        expected_provenance=base_provenance,
                        source_checkpoint=source_path,
                    )
                    print(f"[skip] {exp_id}")
                else:
                    _record_step(
                        status_path,
                        status_rows,
                        experiment_id=exp_id,
                        metadata={
                            "phase": "adaptation",
                            "arch": arch,
                            "seed": seed,
                            "method": method,
                        },
                        fn=partial(
                            train_matched_adaptation,
                            source_path,
                            source_train,
                            source_val,
                            target_adapt,
                            root,
                            cfg,
                            arch=arch,
                            seed=seed,
                            method=method,
                            checkpoint_path=output,
                            provenance=base_provenance,
                        ),
                    )
                checkpoint_rows.append(
                    {
                        "arch": arch,
                        "seed": seed,
                        "method": method,
                        "checkpoint_path": _portable_path(output, root),
                        "checkpoint_sha256": sha256_file(output),
                    }
                )

            assignment_path = splits / f"finetune_assignments_seed{seed}.csv"
            for fraction in fractions:
                target_train, target_val, budget_meta = _validate_budget(
                    target_adapt,
                    assignment_path,
                    fraction,
                )
                method = f"finetune_{int(round(100 * fraction))}pct"
                output = checkpoints / "finetune" / f"{arch}_{method}_seed{seed}.pt"
                exp_id = f"{method}_{arch}_seed{seed}"
                if output.exists() and args.resume:
                    _assert_resume_checkpoint(
                        output,
                        arch=arch,
                        seed=seed,
                        method=method,
                        model_type="baseline",
                        expected_provenance={**base_provenance, **budget_meta},
                        source_checkpoint=source_path,
                    )
                    print(f"[skip] {exp_id}")
                else:
                    _record_step(
                        status_path,
                        status_rows,
                        experiment_id=exp_id,
                        metadata={
                            "phase": "finetune",
                            "arch": arch,
                            "seed": seed,
                            "method": method,
                            "fraction": fraction,
                        },
                        fn=partial(
                            train_finetune_checkpoint,
                            source_path,
                            target_train,
                            target_val,
                            root,
                            cfg,
                            arch=arch,
                            seed=seed,
                            fraction=fraction,
                            checkpoint_path=output,
                            provenance={**base_provenance, **budget_meta},
                        ),
                    )
                checkpoint_rows.append(
                    {
                        "arch": arch,
                        "seed": seed,
                        "method": method,
                        "checkpoint_path": _portable_path(output, root),
                        "checkpoint_sha256": sha256_file(output),
                    }
                )

    index = pd.DataFrame(checkpoint_rows).drop_duplicates(
        subset=["arch", "seed", "method"],
        keep="last",
    )
    index.to_csv(checkpoints / "checkpoint_index.csv", index=False)
    _, ending_images_sha256, ending_image_csv_sha256 = _image_fingerprint(
        root,
        {
            "source_train": source_train,
            "source_val": source_val,
            "target_adapt": target_adapt,
        },
    )
    if (
        ending_images_sha256 != training_images_sha256
        or ending_image_csv_sha256 != image_hash_csv_sha256
    ):
        raise RuntimeError("At least one training image changed during the experiment.")
    print(f"[done] {len(index)} checkpoints indexed at {checkpoints / 'checkpoint_index.csv'}")
    _release_training_lock(training_lock)
    atexit.unregister(_release_training_lock)


if __name__ == "__main__":
    main()
