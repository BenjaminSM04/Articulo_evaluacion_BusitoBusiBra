"""Run the training-only stage of an explicitly selected publication protocol.

This entry point never opens ``source_test_manifest.csv``,
``target_calibration_manifest.csv`` or ``target_test_manifest.csv``.  Its complete input surface is
limited to source train/validation, unlabelled target-adapt, and the labelled fine-tuning budgets.
The locked sets are handled later by ``12_finalize_publication_inference.py``.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.data.datasets import roi_mask_paths  # noqa: E402
from src.evaluation.metrics import run_inference  # noqa: E402
from src.training.adabn import order_target_adapt, refresh_batchnorm_statistics  # noqa: E402
from src.training.experiment_matrix import (  # noqa: E402
    SENSITIVITY_VALUES,
    build_experiment_matrix,
    pilot_experiments,
)
from src.training.publication_protocol import (  # noqa: E402
    _loader,
    load_publication_model,
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
V3_IMAGENET_WEIGHTS = {
    "resnet18": {
        "model_id": "timm/resnet18.a1_in1k",
        "revision": "491b427b45c94c7fb0e78b5474cc919aff584bbf",
        "filename": "model.safetensors",
        "sha256": "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
    },
    "efficientnet_b0": {
        "model_id": "timm/efficientnet_b0.ra_in1k",
        "revision": "1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a",
        "filename": "model.safetensors",
        "sha256": "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd",
    },
}


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
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}",
                 "status", "--porcelain"],
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
    if str(protocol_version).startswith("3."):
        code_files += (
            "src/data/publication_splits.py",
            "src/training/experiment_matrix.py",
            "src/training/control_preprocessing.py",
            "src/training/adabn.py",
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
    started = time.monotonic()
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
        row["duration_hours"] = (time.monotonic() - started) / 3600
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
    for key in (
        "experiment_id", "family", "preprocessing", "hyperparameter",
        "hyperparameter_value", "parent_id", "parent_checkpoint_sha256",
        "roi_masks_sha256", "variant_config_sha256", "imagenet_weights_sha256",
    ):
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


def _select_v3_jobs(jobs, *, stage: str, pilot: bool):
    selected = pilot_experiments(jobs) if pilot else tuple(jobs)
    return tuple(job for job in selected if stage == "all" or job.stage == stage)


def _config_for_job(cfg: Config, job) -> Config:
    clone = Config(copy.deepcopy(dict(cfg)))
    clone._root = cfg._root if "_root" in cfg else Path(__file__).resolve().parents[1]
    clone._config_path = cfg.get("_config_path", "")
    clone._control_preprocessing = job.preprocessing
    if job.family == "sensitivity":
        adaptation = clone.publication.adaptation
        if job.hyperparameter == "mmd_sigma_scale":
            adaptation.mmd_sigmas = [float(v) * float(job.value) for v in adaptation.mmd_sigmas]
        else:
            adaptation[job.hyperparameter] = float(job.value)
    return clone


def _variant_config_hash(cfg: Config, job) -> str:
    payload = {
        "config": _plain_config(cfg, top_level=True),
        "preprocessing": job.preprocessing,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_v3_resources(*, free_disk_gb: float, peak_vram_gb: float) -> None:
    if free_disk_gb < 20:
        raise RuntimeError("El disco libre debe ser al menos 20 GB antes de continuar.")
    if peak_vram_gb > 8:
        raise RuntimeError("La VRAM pico excede el límite de 8 GB.")


def _imagenet_weights_sha256() -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    """Resolve the exact timm HF snapshots locally before starting v3."""
    import huggingface_hub
    import timm
    from timm.models._builder import _resolve_pretrained_source

    records: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for arch, frozen in V3_IMAGENET_WEIGHTS.items():
        source, model_id = _resolve_pretrained_source(timm.get_pretrained_cfg(arch).to_dict())
        if source != "hf-hub" or model_id != frozen["model_id"]:
            raise RuntimeError(f"Fuente ImageNet timm inesperada para {arch}: {source}/{model_id}")
        try:
            path = Path(huggingface_hub.hf_hub_download(
                repo_id=model_id, filename="model.safetensors", local_files_only=True,
            ))
        except Exception as exc:
            raise FileNotFoundError(f"Faltan pesos ImageNet timm en cache: {model_id}") from exc
        if not path.is_file() or path.parent.parent.name != "snapshots":
            raise FileNotFoundError(f"Snapshot ImageNet timm inválido: {path}")
        record = {
            "model_id": model_id,
            "revision": path.parent.name,
            "filename": "model.safetensors",
            "sha256": sha256_file(path),
        }
        if record != frozen:
            raise RuntimeError(f"Snapshot ImageNet timm difiere del protocolo v3: {arch}")
        records[arch] = record
        paths[arch] = path
    return records, paths


def _pin_v3_imagenet_weights(
    records: dict[str, dict[str, str]], paths: dict[str, Path]
) -> Any:
    """Make timm load only the verified local snapshots during this v3 process."""
    import timm.models._hub as timm_hub

    by_id = {record["model_id"]: (record, paths[arch]) for arch, record in records.items()}
    for record, path in by_id.values():
        if path.parent.name != record["revision"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Pesos ImageNet cambiaron antes de cargar: {path}")

    def pinned_download(repo_id, *, filename, **kwargs):
        entry = by_id.get(repo_id)
        if entry is None or filename != "model.safetensors":
            raise RuntimeError(
                f"Solicitud de pesos ImageNet fuera del protocolo: {repo_id}/{filename}"
            )
        record, path = entry
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Pesos ImageNet cambiaron durante la corrida: {path}")
        return str(path)

    original_download = timm_hub.hf_hub_download
    timm_hub.hf_hub_download = pinned_download
    return original_download


def _assert_clean_git_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("git_commit"):
        raise RuntimeError("No se pudo identificar el commit de Git antes del piloto v3.")
    if state.get("git_dirty_at_run") is not False:
        raise RuntimeError("El repositorio debe estar limpio antes del piloto v3.")
    return state


def _check_pilot_projection(
    *, pilot_hours: float, pilot_jobs: int, total_jobs: int,
    pilot_inference_hours: float = 0.0, consumed_hours: float = 0.0,
    completed_jobs: int = 0,
) -> float:
    if pilot_jobs <= 0 or pilot_hours <= 0:
        raise ValueError("El piloto debe tener trabajos y duración positivos.")
    if not 0 <= completed_jobs <= total_jobs or consumed_hours < 0 or pilot_inference_hours < 0:
        raise ValueError("Tiempo o número de trabajos proyectados inválidos.")
    per_job_hours = (pilot_hours + pilot_inference_hours) / pilot_jobs
    projection = consumed_hours + per_job_hours * (total_jobs - completed_jobs) * 1.1
    if projection > 16:
        raise RuntimeError(f"Proyección de {projection:.2f} h supera 16 h; quedan 2 h de reserva.")
    return projection


def _v3_checkpoint_path(checkpoints: Path, job) -> Path:
    return checkpoints / "v3" / f"{job.experiment_id}.pt"


def _roi_masks_hash(root: Path, frames: tuple[pd.DataFrame, ...]) -> str:
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for frame in frames:
        for _, row in frame.iterrows():
            for part in roi_mask_paths(row):
                path = (root / part.strip()).resolve()
                if not path.is_relative_to(resolved_root):
                    raise ValueError("Máscara ROI fuera del proyecto.")
                digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _validate_v3_source_split(
    cfg: Config, root: Path, source_train: pd.DataFrame, source_val: pd.DataFrame,
    metadata: dict[str, Any], hashes: dict[str, str],
) -> None:
    if metadata.get("hashes") != hashes:
        raise ValueError("Los hashes fuente no coinciden con su metadata.")
    if metadata.get("split_seed") != int(cfg.publication.split_seed):
        raise ValueError("La semilla de partición fuente difiere del protocolo.")
    if metadata.get("source_split_unit") != cfg.publication.cohort_version:
        raise ValueError("La cohorte fuente difiere de cohort_version.")
    counts = {"source_train": 229, "source_val": 57, "source_test": 72}
    if metadata.get("source_partition_counts") != counts:
        raise ValueError("Los conteos de partición fuente v3 difieren del protocolo.")
    if len(source_train) != 229 or len(source_val) != 57:
        raise ValueError("Los manifiestos train/val no coinciden con los conteos fuente.")
    split_dir = cfg.path("results") / "splits"
    for name in ("source_train", "source_val"):
        key = f"{name}_manifest_sha256"
        path = split_dir / f"{name}_manifest.csv"
        if metadata.get(key) != sha256_file(path):
            raise ValueError(f"La huella del manifiesto {name} difiere de la partición fuente.")
    audit_path = (root / str(cfg.publication.source_group_audit)).resolve()
    if not audit_path.is_relative_to(root.resolve()):
        raise ValueError("La auditoría fuente está fuera del proyecto.")
    if hashes.get("source_group_audit") != sha256_file(audit_path):
        raise ValueError("Cambió la auditoría pública de grupos fuente.")
    source_input_path = (root / str(cfg.datasets.busi.manifest)).resolve()
    if not source_input_path.is_relative_to(root.resolve()):
        raise ValueError("El manifiesto Curated BUSI está fuera del proyecto.")
    if hashes.get("source_manifest_input") != sha256_file(source_input_path):
        raise ValueError("Cambió el manifiesto fuente Curated BUSI.")
    for frame in (source_train, source_val):
        if "pawlowska_group_id" not in frame.columns:
            raise ValueError("Falta pawlowska_group_id en el manifiesto fuente.")
        if frame.pawlowska_group_id.astype(str).str.strip().eq("").any():
            raise ValueError("Existe un grupo fuente vacío.")
    if set(source_train.sample_id) & set(source_val.sample_id):
        raise ValueError("Source train/validation overlap.")
    if set(source_train.pawlowska_group_id) & set(source_val.pawlowska_group_id):
        raise ValueError("Existe un grupo relacionado en train y validation.")


def _should_pause_before_job(
    elapsed_hours: float, limit_hours: float, observed_job_hours: list[float]
) -> bool:
    estimate = max(observed_job_hours, default=0.0) * 1.25
    return elapsed_hours + estimate >= limit_hours


def _preflight_v3_dependencies(
    jobs, checkpoints: Path, *, protocol_version: str | None = None
) -> None:
    """Reject an incomplete stage before starting any of its training jobs."""
    import torch

    scheduled: set[str] = set()
    index_rows: list[dict[str, Any]] | None = None
    frozen_run: dict[str, Any] | None = None
    for job in jobs:
        if job.experiment_id in scheduled:
            raise RuntimeError(f"Trabajo v3 duplicado: {job.experiment_id}")
        if job.parent_id and job.parent_id not in scheduled:
            parent = checkpoints / "v3" / f"{job.parent_id}.pt"
            if not parent.is_file():
                raise RuntimeError(
                    f"Falta padre {job.parent_id} para {job.experiment_id}; "
                    "ejecute primero la etapa principal."
                )
            if index_rows is None:
                index_rows = _load_status(checkpoints / "checkpoint_index.csv")
            matching = [
                row for row in index_rows if str(row.get("experiment_id")) == job.parent_id
            ]
            if len(matching) != 1:
                raise RuntimeError(f"Padre externo {job.parent_id} no tiene fila única en índice.")
            indexed = matching[0]
            if (
                indexed.get("arch") != job.arch
                or int(indexed["seed"]) != int(job.seed)
                or not str(job.parent_id).endswith(f"_{indexed.get('method')}")
            ):
                raise RuntimeError(
                    f"Identidad del padre externo {job.parent_id} difiere del índice."
                )
            if indexed.get("checkpoint_sha256") != sha256_file(parent):
                raise RuntimeError(f"SHA del padre externo {job.parent_id} difiere del índice.")
            state = torch.load(parent, map_location="cpu", weights_only=False)
            expected = {
                "arch": indexed.get("arch"),
                "seed": int(indexed["seed"]),
                "method": indexed.get("method"),
            }
            if any(state.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    f"Identidad del padre externo {job.parent_id} difiere del índice."
                )
            parent_provenance = state.get("provenance", {})
            if parent_provenance.get("experiment_id") != job.parent_id or (
                protocol_version is not None
                and parent_provenance.get("protocol_version") != protocol_version
            ):
                raise RuntimeError(f"Procedencia del padre externo {job.parent_id} difiere.")
            if protocol_version is not None:
                if frozen_run is None:
                    provenance_path = checkpoints.parent / "logs" / "training_provenance.json"
                    if not provenance_path.is_file():
                        raise RuntimeError("Falta procedencia de la corrida para el padre externo.")
                    frozen_run = json.loads(provenance_path.read_text(encoding="utf-8"))
                keys = (*RESUME_PROVENANCE_KEYS, "imagenet_weights_sha256")
                if frozen_run.get("protocol_version") != protocol_version or any(
                    parent_provenance.get(key) != frozen_run.get(key) for key in keys
                ):
                    raise RuntimeError(f"Procedencia del padre externo {job.parent_id} difiere.")
        scheduled.add(job.experiment_id)


def _v3_index_row(job, output: Path, root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the same immutable index row after a post-save interruption."""
    return {
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
        "parent_checkpoint_sha256": provenance["parent_checkpoint_sha256"],
        "variant_config_sha256": provenance["variant_config_sha256"],
        "checkpoint_path": _portable_path(output, root),
        "checkpoint_sha256": sha256_file(output),
    }


def _write_v3_json(path: Path, record: dict[str, Any]) -> None:
    """Replace a small state file atomically so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _recover_v3_hours(elapsed_path: Path, session_path: Path) -> float:
    """Conservatively charge an interrupted active session to the wall budget."""
    recorded = 0.0
    if elapsed_path.is_file():
        recorded = float(json.loads(elapsed_path.read_text(encoding="utf-8"))["hours"])
    if recorded < 0 or not np.isfinite(recorded):
        raise ValueError("Horas v3 registradas inválidas")
    if not session_path.is_file():
        return recorded
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not session.get("active"):
        return recorded
    started = datetime.fromisoformat(str(session["started_at_utc"]))
    if started.tzinfo is None:
        raise ValueError("La sesión v3 activa carece de zona horaria")
    base = float(session["base_hours"])
    if base < 0 or not np.isfinite(base):
        raise ValueError("Horas base v3 inválidas")
    wall_hours = max(0.0, (datetime.now(timezone.utc) - started).total_seconds() / 3600)
    return max(recorded, base + wall_hours)


def _v3_plan(cfg: Config, *, stage: str, pilot: bool) -> dict[str, Any]:
    _validate_v3_contract(cfg)
    jobs = build_experiment_matrix(
        architectures=tuple(cfg.publication.architectures),
        seeds=tuple(int(seed) for seed in cfg.publication.seeds),
    )
    selected = _select_v3_jobs(jobs, stage=stage, pilot=pilot)
    return {
        "protocol_version": str(cfg.publication.protocol_version),
        "stage": stage,
        "pilot": pilot,
        "total_jobs": len(jobs),
        "jobs": [{**asdict(job), "stage": job.stage} for job in selected],
    }


def _validate_v3_contract(cfg: Config) -> None:
    publication = cfg.publication
    if publication.get("cohort_version") != "reviewed_related_image_group_v3":
        raise ValueError("v3 requiere cohort_version reviewed_related_image_group_v3")
    controls = publication.get("controls")
    if not controls:
        raise ValueError("v3 requiere publication.controls congelado")
    expected_controls = {
        "intensity": {"method": "global_rgb_percentile", "low": 1, "high": 99},
        "roi": {"method": "union_masks", "margin_fraction": 0.1},
        "adabn": {"batch_size": 16, "order_by": "sample_id",
                  "parent_method": "source_only_matched"},
    }
    if _plain_config(controls) != expected_controls:
        raise ValueError("publication.controls difiere del protocolo v3 congelado")
    expected_sensitivity = {
        name: [0.5, 2.0] for name in {entry[1] for entry in SENSITIVITY_VALUES}
    }
    if _plain_config(publication.get("sensitivity")) != expected_sensitivity:
        raise ValueError("publication.sensitivity difiere del protocolo v3 congelado")
    if list(publication.architectures) != ["resnet18", "efficientnet_b0"] or list(
        publication.seeds
    ) != [17, 42, 73, 101, 202]:
        raise ValueError("v3 requiere dos arquitecturas y cinco semillas canónicas")


def _run_v3(cfg: Config, args) -> None:
    import torch

    _validate_v3_contract(cfg)
    root = cfg._root
    git_state = _assert_clean_git_state(_git_state(root))
    results = cfg.path("results")
    checkpoints = results / "checkpoints"
    splits = results / "splits"
    if not args.pilot and not (results / "logs" / "v3_pilot_gate.json").is_file():
        raise RuntimeError("Falta el piloto v3 y su gate de proyección.")
    if args.pilot and args.stage != "all":
        raise ValueError("El piloto ejecutable requiere --stage all para completar 20 trabajos.")
    jobs = _select_v3_jobs(
        build_experiment_matrix(
            architectures=tuple(cfg.publication.architectures),
            seeds=tuple(int(seed) for seed in cfg.publication.seeds),
        ),
        stage=args.stage,
        pilot=args.pilot,
    )
    _preflight_v3_dependencies(
        jobs, checkpoints, protocol_version=str(cfg.publication.protocol_version)
    )
    free_disk_gb = shutil.disk_usage(results).free / 1024**3
    _check_v3_resources(free_disk_gb=free_disk_gb, peak_vram_gb=0)
    lock = _acquire_training_lock(results)
    atexit.register(_release_training_lock, lock)
    started: float | None = None
    previous_hours = 0.0
    elapsed_path = results / "logs" / "v3_elapsed.json"
    session_path = results / "logs" / "v3_session.json"
    indexed: dict[str, dict[str, Any]] = {}
    original_timm_download = None
    try:
        source_train_path = splits / "source_train_manifest.csv"
        source_val_path = splits / "source_val_manifest.csv"
        target_adapt_path = splits / "target_adapt_manifest.csv"
        source_train = _read_manifest(source_train_path)
        source_val = _read_manifest(source_val_path)
        target_adapt = _read_manifest(target_adapt_path, patient=True)
        if set(source_train.sample_id) & set(source_val.sample_id):
            raise ValueError("Source train/validation overlap.")
        if target_adapt.patient_id.astype(str).str.len().eq(0).any():
            raise ValueError("target_adapt contiene patient_id vacío.")
        images, image_hash, image_csv_hash = _image_fingerprint(root, {
            "source_train": source_train, "source_val": source_val,
            "target_adapt": target_adapt,
        })
        source_hashes_path = splits / "source_assignment_hashes.json"
        source_metadata_path = splits / "source_split_metadata.json"
        source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        _validate_v3_source_split(
            cfg, root, source_train, source_val, source_metadata, source_hashes
        )
        config_path = results / "effective_config.yaml"
        config_text = _effective_config_text(cfg)
        config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        code_hash, code_files = _training_code_fingerprint(
            root, protocol_version=str(cfg.publication.protocol_version)
        )
        runtime_hash, runtime = _runtime_fingerprint()
        imagenet_weights_sha256, imagenet_weight_paths = _imagenet_weights_sha256()
        original_timm_download = _pin_v3_imagenet_weights(
            imagenet_weights_sha256, imagenet_weight_paths
        )
        if args.resume:
            if not config_path.exists() or sha256_file(config_path) != config_hash:
                raise RuntimeError("No se puede reanudar con configuración diferente.")
        elif config_path.exists():
            raise RuntimeError("Ya existe una corrida v3; use --resume para continuar.")
        else:
            config_path.write_text(config_text, encoding="utf-8")
        provenance = {
            "protocol_version": str(cfg.publication.protocol_version),
            "effective_config_sha256": config_hash,
            "source_train_sha256": sha256_file(source_train_path),
            "source_val_sha256": sha256_file(source_val_path),
            "target_adapt_sha256": sha256_file(target_adapt_path),
            "split_assignment_hashes_sha256": sha256_file(source_hashes_path),
            "split_metadata_sha256": sha256_file(source_metadata_path),
            "training_code_sha256": code_hash,
            "training_code_files": code_files,
            "runtime_fingerprint_sha256": runtime_hash,
            "runtime": runtime,
            "imagenet_weights_sha256": imagenet_weights_sha256,
            "training_images_sha256": image_hash,
            "training_image_hash_manifest_sha256": image_csv_hash,
            "n_training_images_hashed": len(images),
            **git_state,
        }
        logs = results / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        provenance_path = logs / "training_provenance.json"
        if args.resume:
            if not provenance_path.exists():
                raise RuntimeError("Falta la procedencia de la corrida v3.")
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            changed = [key for key in RESUME_PROVENANCE_KEYS
                       if previous.get(key) != provenance.get(key)]
            if previous.get("imagenet_weights_sha256") != imagenet_weights_sha256:
                changed.append("imagenet_weights_sha256")
            if changed:
                raise RuntimeError(f"Procedencia de reanudación distinta: {changed}")
        else:
            provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            images.to_csv(logs / "training_image_hashes.csv", index=False)

        index_path = checkpoints / "checkpoint_index.csv"
        index_rows = _load_status(index_path) if args.resume else []
        indexed = {str(row["experiment_id"]): row for row in index_rows}
        status_path = logs / "training_status.csv"
        status_rows = _load_status(status_path) if args.resume else []
        if args.resume:
            previous_hours = _recover_v3_hours(elapsed_path, session_path)
            _write_v3_json(elapsed_path, {
                "hours": previous_hours, "completed_jobs": len(indexed)
            })
            _write_v3_json(session_path, {"active": False, "hours": previous_hours})
        if not args.pilot:
            pilot_path = logs / "v3_pilot_gate.json"
            if not pilot_path.exists():
                raise RuntimeError("Falta el piloto v3 y su gate de proyección.")
            pilot_record = json.loads(pilot_path.read_text(encoding="utf-8"))
            if pilot_record["provenance_sha256"] != sha256_file(provenance_path):
                raise RuntimeError("El gate piloto pertenece a otra procedencia.")
            if pilot_record.get("target_adapt_roi_masks_sha256") != _roi_masks_hash(
                root, (target_adapt,)
            ):
                raise RuntimeError("Las máscaras ROI de target_adapt cambiaron desde el piloto.")
            _check_pilot_projection(
                pilot_hours=float(pilot_record["pilot_hours"]),
                pilot_inference_hours=float(pilot_record["pilot_inference_hours"]),
                pilot_jobs=int(pilot_record["pilot_jobs"]), total_jobs=160,
                consumed_hours=previous_hours, completed_jobs=len(indexed),
            )
        started = time.monotonic()
        _write_v3_json(session_path, {
            "active": True,
            "base_hours": previous_hours,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        for job in jobs:
            elapsed_hours = previous_hours + (time.monotonic() - started) / 3600
            observed_hours = [
                float(row["duration_hours"]) for row in status_rows
                if row.get("status") == "success" and row.get("duration_hours") not in ("", None)
            ]
            if _should_pause_before_job(elapsed_hours, args.max_wall_hours, observed_hours):
                print(f"[pausa] Tiempo acumulado {elapsed_hours:.2f} h; reanudar con --resume")
                break
            _check_v3_resources(
                free_disk_gb=shutil.disk_usage(results).free / 1024**3,
                peak_vram_gb=(torch.cuda.max_memory_allocated() / 1024**3
                              if torch.cuda.is_available() else 0),
            )
            output = _v3_checkpoint_path(checkpoints, job)
            parent_path = None
            if job.parent_id:
                parent_path = checkpoints / "v3" / f"{job.parent_id}.pt"
                if not parent_path.is_file():
                    raise RuntimeError(f"Falta padre {job.parent_id} para {job.experiment_id}")
            job_cfg = _config_for_job(cfg, job)
            job_provenance = {
                **provenance, "experiment_id": job.experiment_id,
                "family": job.family, "preprocessing": job.preprocessing,
                "hyperparameter": job.hyperparameter, "hyperparameter_value": job.value,
                "parent_id": job.parent_id,
                "parent_checkpoint_sha256": sha256_file(parent_path) if parent_path else "",
                "variant_config_sha256": _variant_config_hash(job_cfg, job),
            }
            if job.preprocessing == "roi":
                job_provenance["roi_masks_sha256"] = _roi_masks_hash(
                    root, (source_train, source_val)
                )
            if job.method.startswith("finetune_"):
                assignment = splits / f"finetune_assignments_seed{job.seed}.csv"
                target_train, target_val, budget = _validate_budget(
                    target_adapt, assignment, float(job.value)
                )
                job_provenance.update(budget)
            target_train = target_train if job.method.startswith("finetune_") else None
            target_val = target_val if job.method.startswith("finetune_") else None
            if output.exists():
                if not args.resume:
                    raise RuntimeError(f"Checkpoint existente: {output}; use --resume")
                _assert_resume_checkpoint(
                    output, arch=job.arch, seed=job.seed, method=job.method,
                    model_type="dann" if job.method == "dann" else "baseline",
                    expected_provenance=job_provenance,
                    source_checkpoint=parent_path if job.method != "adabn" else None,
                )
                checksum = sha256_file(output)
                if job.experiment_id not in indexed:
                    indexed[job.experiment_id] = _v3_index_row(
                        job, output, root, job_provenance
                    )
                    pd.DataFrame(indexed.values()).to_csv(index_path, index=False)
                    print(f"[reindex] {job.experiment_id}")
                elif indexed[job.experiment_id].get("checkpoint_sha256") != checksum:
                    raise RuntimeError(f"Índice alterado para {job.experiment_id}")
                print(f"[skip] {job.experiment_id}")
                continue

            def execute(
                job=job, job_cfg=job_cfg, output=output, parent_path=parent_path,
                job_provenance=job_provenance,
                target_train=target_train, target_val=target_val,
            ):
                if job.method == "source_direct":
                    return train_source_checkpoint(
                        source_train, source_val, root, job_cfg, arch=job.arch,
                        seed=job.seed, checkpoint_path=output, provenance=job_provenance,
                    )
                if job.method == "adabn":
                    model, state = load_publication_model(parent_path, job_cfg)
                    ordered = order_target_adapt(target_adapt)
                    job_cfg.training.batch_size = int(cfg.publication.controls.adabn.batch_size)
                    loader = _loader(ordered, root, job_cfg, train=False, seed=job.seed,
                                     label_blind=True)
                    count = refresh_batchnorm_statistics(model, loader, device=job_cfg.device)
                    state.update({
                        "model_state": model.state_dict(), "method": "adabn",
                        "provenance": {
                            **job_provenance, "adabn_n_images": count,
                            "initial_checkpoint_sha256": sha256_file(parent_path),
                        },
                    })
                    output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state, output)
                    return output
                if job.method.startswith("finetune_"):
                    return train_finetune_checkpoint(
                        parent_path, target_train, target_val, root, job_cfg,
                        arch=job.arch, seed=job.seed, fraction=float(job.value),
                        checkpoint_path=output, provenance=job_provenance,
                    )
                return train_matched_adaptation(
                    parent_path, source_train, source_val, target_adapt, root, job_cfg,
                    arch=job.arch, seed=job.seed, method=job.method,
                    checkpoint_path=output, provenance=job_provenance,
                )

            _record_step(
                status_path, status_rows, experiment_id=job.experiment_id,
                metadata={"stage": job.stage, "family": job.family, "parent_id": job.parent_id,
                          "arch": job.arch, "seed": job.seed, "method": job.method},
                fn=execute,
            )
            indexed[job.experiment_id] = _v3_index_row(job, output, root, job_provenance)
            pd.DataFrame(indexed.values()).to_csv(index_path, index=False)
            elapsed_hours = previous_hours + (time.monotonic() - started) / 3600
            _write_v3_json(elapsed_path, {
                "hours": elapsed_hours, "completed_jobs": len(indexed)
            })
            print(f"[v3] {job.experiment_id}: {elapsed_hours:.2f} h acumuladas")
        pilot_gate_path = logs / "v3_pilot_gate.json"
        if (
            args.pilot
            and all(job.experiment_id in indexed for job in jobs)
            and not pilot_gate_path.exists()
        ):
            pilot_hours = previous_hours + (time.monotonic() - started) / 3600
            inference_started = time.monotonic()
            ordered_adapt = order_target_adapt(target_adapt)
            roi_target_hash = ""
            for job in jobs:
                job_cfg = _config_for_job(cfg, job)
                if job.preprocessing == "roi":
                    roi_target_hash = _roi_masks_hash(root, (target_adapt,))
                model, _ = load_publication_model(_v3_checkpoint_path(checkpoints, job), job_cfg)
                loader = _loader(
                    ordered_adapt, root, job_cfg, train=False, seed=job.seed,
                    label_blind=True,
                )
                run_inference(model, loader, job_cfg.device, job_cfg.training.mixed_precision)
            inference_hours = (time.monotonic() - inference_started) / 3600
            consumed_hours = pilot_hours + inference_hours
            _check_v3_resources(
                free_disk_gb=shutil.disk_usage(results).free / 1024**3,
                peak_vram_gb=(torch.cuda.max_memory_allocated() / 1024**3
                              if torch.cuda.is_available() else 0),
            )
            projected = _check_pilot_projection(
                pilot_hours=pilot_hours, pilot_inference_hours=inference_hours,
                pilot_jobs=len(jobs), total_jobs=160,
                consumed_hours=consumed_hours, completed_jobs=len(indexed),
            )
            pilot_gate_path.write_text(json.dumps({
                "pilot_hours": pilot_hours, "pilot_jobs": len(jobs),
                "pilot_inference_hours": inference_hours,
                "target_adapt_roi_masks_sha256": roi_target_hash,
                "projected_hours": projected,
                "provenance_sha256": sha256_file(provenance_path),
            }, indent=2), encoding="utf-8")
            _write_v3_json(elapsed_path, {
                "hours": consumed_hours, "completed_jobs": len(indexed)
            })
        _, ending_hash, ending_csv_hash = _image_fingerprint(root, {
            "source_train": source_train, "source_val": source_val,
            "target_adapt": target_adapt,
        })
        if (ending_hash, ending_csv_hash) != (image_hash, image_csv_hash):
            raise RuntimeError("Las imágenes de entrenamiento cambiaron durante la corrida.")
        print(f"[v3] {len(indexed)}/160 checkpoints indexados")
    finally:
        try:
            if started is not None:
                charged_hours = previous_hours + (time.monotonic() - started) / 3600
                _write_v3_json(elapsed_path, {
                    "hours": charged_hours, "completed_jobs": len(indexed)
                })
                _write_v3_json(session_path, {
                    "active": False, "hours": charged_hours
                })
        finally:
            if original_timm_download is not None:
                import timm.models._hub as timm_hub

                timm_hub.hf_hub_download = original_timm_download
            _release_training_lock(lock)
            atexit.unregister(_release_training_lock)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train publication protocol v2 without test access"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--stage", choices=("main", "controls", "sensitivity", "all"),
                        default="all")
    parser.add_argument("--max-wall-hours", type=float, default=18.0)
    args = parser.parse_args()

    cfg = load_config(args.config, create_dirs=not args.plan_only)
    is_v3 = str(cfg.publication.protocol_version).startswith("3.")
    if args.plan_only:
        if not is_v3:
            parser.error("--plan-only requiere protocolo v3")
        print(json.dumps(_v3_plan(cfg, stage=args.stage, pilot=args.pilot), indent=2))
        return
    if is_v3:
        if args.architectures or args.seeds or args.source_only:
            parser.error(
                "v3 usa la matriz canónica; no admite --architectures, --seeds ni --source-only"
            )
        if not 0 < args.max_wall_hours <= 18:
            parser.error("--max-wall-hours debe estar entre 0 y 18")
        _run_v3(cfg, args)
        return
    if args.pilot or args.stage != "all" or args.max_wall_hours != 18.0:
        parser.error("--pilot, --stage y --max-wall-hours requieren protocolo v3")
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
