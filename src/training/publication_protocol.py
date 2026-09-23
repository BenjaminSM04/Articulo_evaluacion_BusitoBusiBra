"""Training primitives for the locked publication protocol.

The functions in this module deliberately do not know where the target test set lives.  The
training entry point receives only source train/validation and the *unlabelled* target-adaptation
pool.  Final inference is implemented in a separate evaluation module so a test row cannot enter
checkpoint selection accidentally.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import UltrasoundDataset, class_weights
from ..evaluation.metrics import compute_metrics, run_inference
from ..models.architectures import BaselineClassifier, DANNModel
from ..utils.seed import seed_everything, seed_worker
from .domain_adaptation import coral_loss, mmd_loss
from .transforms import build_eval_transforms, build_train_transforms


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loader(
    df: pd.DataFrame,
    root: Path,
    cfg: Config,
    *,
    train: bool,
    seed: int,
    label_blind: bool = False,
) -> DataLoader:
    frame = df.reset_index(drop=True).copy()
    if label_blind:
        # UltrasoundDataset requires a numeric field, but UDA code must not receive a target
        # diagnosis.  -1 is never consumed by any target loss.
        frame["label_idx"] = -1
        frame["label"] = "unlabelled"
    transform = build_train_transforms(cfg) if train else build_eval_transforms(cfg)
    # Albumentations 2 owns an RNG independent from NumPy's global RNG. Seed it explicitly;
    # worker processes receive distinct deterministic seeds again in ``seed_worker``.
    if callable(getattr(transform, "set_random_seed", None)):
        transform.set_random_seed(int(seed))
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(cfg.training.num_workers)
    return DataLoader(
        UltrasoundDataset(
            frame,
            root,
            transform,
        ),
        batch_size=int(cfg.training.batch_size),
        shuffle=train,
        drop_last=False,
        num_workers=workers,
        pin_memory="cuda" in str(cfg.device),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if train else None,
        generator=generator,
    )


def _amp_scaler(cfg: Config):
    enabled = bool(cfg.training.mixed_precision and "cuda" in str(cfg.device))
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _autocast(cfg: Config, scaler):
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"
    return torch.autocast(device_type=device_type, enabled=scaler.is_enabled())


@torch.no_grad()
def _validation_loss(model: nn.Module, loader: DataLoader, cfg: Config) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"
    for x, y, _ in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        with torch.autocast(
            device_type=device_type,
            enabled=bool(cfg.training.mixed_precision and device_type == "cuda"),
        ):
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, y)
        total += float(loss.item()) * len(y)
        count += len(y)
    return total / max(1, count)


def _make_baseline(arch: str, cfg: Config, *, pretrained: bool) -> BaselineClassifier:
    return BaselineClassifier(
        arch=arch,
        num_classes=int(cfg.model.num_classes),
        pretrained=pretrained,
        dropout=float(cfg.model.dropout),
    )


def _make_dann_from_baseline(
    baseline_state: dict[str, torch.Tensor],
    arch: str,
    cfg: Config,
) -> DANNModel:
    baseline = _make_baseline(arch, cfg, pretrained=False)
    baseline.load_state_dict(baseline_state, strict=True)
    model = DANNModel(
        arch=arch,
        num_classes=int(cfg.model.num_classes),
        pretrained=False,
        dropout=float(cfg.model.dropout),
    )
    model.backbone.load_state_dict(baseline.backbone.state_dict(), strict=True)
    model.label_head.load_state_dict(baseline.head.state_dict(), strict=True)
    return model


def _cosine_scheduler(optimizer, epochs: int, warmup_epochs: int = 0):
    """Epoch-level linear warm-up followed by cosine decay."""
    epochs = int(epochs)
    warmup_epochs = int(warmup_epochs)
    if epochs <= 0 or not 0 <= warmup_epochs < epochs:
        raise ValueError("Scheduler requires epochs > 0 and 0 <= warmup_epochs < epochs.")

    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Keep source BatchNorm running statistics fixed while retaining affine gradients."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _patient_validation_auc(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
) -> float:
    """Compute validation AUC after the pre-specified mean-logit patient aggregation."""
    frame = loader.dataset.df
    if "patient_id" not in frame.columns:
        raise ValueError("Patient-level validation requires patient_id.")
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"
    with torch.no_grad():
        for images, y_true, indices in loader:
            images = images.to(cfg.device, non_blocking=True)
            with torch.autocast(
                device_type=device_type,
                enabled=bool(cfg.training.mixed_precision and device_type == "cuda"),
            ):
                output = model(images)
                output = output[0] if isinstance(output, tuple) else output
            labels.append(np.asarray(y_true, dtype=int))
            logits.append((output[:, 1] - output[:, 0]).float().cpu().numpy())
            row_indices.append(np.asarray(indices, dtype=int))
    y_true = np.concatenate(labels)
    logit_difference = np.concatenate(logits)
    indices = np.concatenate(row_indices)
    predictions = pd.DataFrame(
        {
            "patient_id": frame.iloc[indices]["patient_id"].astype(str).to_numpy(),
            "label_idx": y_true,
            "logit_difference": logit_difference,
        }
    )
    inconsistent = predictions.groupby("patient_id")["label_idx"].nunique()
    if inconsistent.gt(1).any():
        raise ValueError("Fine-tuning validation contains inconsistent patient labels.")
    patients = predictions.groupby("patient_id", as_index=False).agg(
        label_idx=("label_idx", "first"),
        logit_difference=("logit_difference", "mean"),
    )
    patient_probability = 1.0 / (1.0 + np.exp(-patients["logit_difference"].to_numpy()))
    return float(compute_metrics(patients["label_idx"].to_numpy(), patient_probability)["auc"])


def _save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_type: str,
    arch: str,
    seed: int,
    method: str,
    best_epoch: int,
    best_val_auc: float,
    history: list[dict[str, float]],
    provenance: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": model_type,
            "arch": arch,
            "seed": int(seed),
            "method": method,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "history": history,
            "provenance": provenance,
        },
        path,
    )
    return path


def train_source_checkpoint(
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    root: Path,
    cfg: Config,
    *,
    arch: str,
    seed: int,
    checkpoint_path: str | Path,
    provenance: dict[str, Any],
) -> Path:
    """Train one ImageNet-initialised source checkpoint and select on source validation only."""
    seed_everything(seed)
    model = _make_baseline(arch, cfg, pretrained=True).to(cfg.device)
    epochs = int(cfg.publication.source.epochs)
    lr = float(cfg.publication.source.lr)
    train_loader = _loader(source_train, root, cfg, train=True, seed=seed)
    val_loader = _loader(source_val, root, cfg, train=False, seed=seed)
    criterion = nn.CrossEntropyLoss(weight=class_weights(source_train).to(cfg.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = _cosine_scheduler(
        optimizer,
        epochs,
        warmup_epochs=int(cfg.publication.source.lr_warmup_epochs),
    )
    scaler = _amp_scaler(cfg)
    best_auc, best_epoch, best_state = -math.inf, -1, None
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        running, seen = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, scaler):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(y)
            seen += len(y)
        scheduler.step()

        y_true, y_prob, _ = run_inference(
            model,
            val_loader,
            cfg.device,
            cfg.training.mixed_precision,
        )
        val_auc = float(compute_metrics(y_true, y_prob)["auc"])
        row = {
            "epoch": float(epoch + 1),
            "train_loss": running / max(1, seen),
            "val_loss": _validation_loss(model, val_loader, cfg),
            "val_auc": val_auc,
            "lr": epoch_lr,
        }
        history.append(row)
        print(
            f"[source {arch} seed={seed}] {epoch + 1:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Source training did not produce a selectable checkpoint.")
    model.load_state_dict(best_state, strict=True)
    return _save_checkpoint(
        checkpoint_path,
        model,
        model_type="baseline",
        arch=arch,
        seed=seed,
        method="source_direct",
        best_epoch=best_epoch,
        best_val_auc=best_auc,
        history=history,
        provenance=provenance,
    )


def load_publication_model(
    checkpoint_path: str | Path,
    cfg: Config,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load baseline or DANN checkpoints without silently dropping classification heads."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = str(state["arch"])
    model_type = str(state.get("model_type", "baseline"))
    if model_type == "dann":
        model: nn.Module = DANNModel(
            arch=arch,
            num_classes=int(cfg.model.num_classes),
            pretrained=False,
            dropout=float(cfg.model.dropout),
        )
    elif model_type == "baseline":
        model = _make_baseline(arch, cfg, pretrained=False)
    else:
        raise ValueError(f"Unknown publication checkpoint model_type={model_type!r}")
    model.load_state_dict(state["model_state"], strict=True)
    model.to(cfg.device).eval()
    return model, state


def _baseline_state_from_source(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source.get("model_type") != "baseline" or source.get("method") != "source_direct":
        raise ValueError(f"{path} is not a source baseline checkpoint.")
    return source["model_state"], source


def train_matched_adaptation(
    source_checkpoint: str | Path,
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    target_adapt: pd.DataFrame,
    root: Path,
    cfg: Config,
    *,
    arch: str,
    seed: int,
    method: str,
    checkpoint_path: str | Path,
    provenance: dict[str, Any],
) -> Path:
    """Train a matched source-only/DANN/CORAL/MMD continuation.

    Every method consumes exactly one pass over the same source loader per epoch.  The target
    loader is label-blind and cycled as needed.  All epochs are executed; the retained state is
    chosen solely by source-validation AUC.
    """
    allowed = {"source_only_matched", "dann", "coral", "mmd"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    seed_everything(seed)
    baseline_state, source_meta = _baseline_state_from_source(source_checkpoint)
    source_hash = sha256_file(source_checkpoint)
    if method == "dann":
        model: nn.Module = _make_dann_from_baseline(baseline_state, arch, cfg)
        model_type = "dann"
    else:
        model = _make_baseline(arch, cfg, pretrained=False)
        model.load_state_dict(baseline_state, strict=True)
        model_type = "baseline"
    model.to(cfg.device)

    adapt_cfg = cfg.publication.adaptation
    epochs = int(adapt_cfg.epochs)
    src_loader = _loader(source_train, root, cfg, train=True, seed=seed)
    val_loader = _loader(source_val, root, cfg, train=False, seed=seed)
    tgt_loader = None
    if method != "source_only_matched":
        tgt_loader = _loader(
            target_adapt,
            root,
            cfg,
            train=True,
            seed=seed + 100_000,
            label_blind=True,
        )

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights(source_train).to(cfg.device))
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(adapt_cfg.lr),
        weight_decay=float(adapt_cfg.weight_decay),
    )
    scheduler = _cosine_scheduler(
        optimizer,
        epochs,
        warmup_epochs=int(adapt_cfg.lr_warmup_epochs),
    )
    scaler = _amp_scaler(cfg)
    best_auc, best_epoch, best_state = -math.inf, -1, None
    history: list[dict[str, float]] = []
    total_steps = max(1, epochs * len(src_loader))

    for epoch in range(epochs):
        model.train()
        if bool(adapt_cfg.freeze_batchnorm_running_stats):
            _freeze_batchnorm_running_stats(model)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        target_iter = iter(tgt_loader) if tgt_loader is not None else None
        running, running_cls, running_aux, seen = 0.0, 0.0, 0.0, 0
        for step, (xs, ys, _) in enumerate(src_loader):
            xs = xs.to(cfg.device, non_blocking=True)
            ys = ys.to(cfg.device, non_blocking=True)
            xt = None
            if target_iter is not None:
                try:
                    xt, _, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(tgt_loader)
                    xt, _, _ = next(target_iter)
                xt = xt.to(cfg.device, non_blocking=True)

            global_step = epoch * len(src_loader) + step
            progress = global_step / max(1, total_steps - 1)
            warm = min(
                1.0,
                (epoch + step / max(1, len(src_loader)))
                / max(1.0, float(adapt_cfg.align_warmup_epochs)),
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, scaler):
                if method == "dann":
                    lambda_ = float(adapt_cfg.dann_lambda) * (
                        2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
                    )
                    logits_s, domain_s = model(xs, lambda_)
                    _, domain_t = model(xt, lambda_)
                    loss_cls = cls_criterion(logits_s, ys)
                    zeros = torch.zeros(len(xs), dtype=torch.long, device=cfg.device)
                    ones = torch.ones(len(xt), dtype=torch.long, device=cfg.device)
                    loss_aux = 0.5 * (
                        domain_criterion(domain_s, zeros) + domain_criterion(domain_t, ones)
                    )
                    loss = loss_cls + float(adapt_cfg.domain_weight) * loss_aux
                else:
                    features_s = model.features(xs)
                    logits_s = model.head(features_s)
                    loss_cls = cls_criterion(logits_s, ys)
                    if method == "source_only_matched":
                        loss_aux = torch.zeros((), device=cfg.device)
                    else:
                        features_t = model.features(xt)
                        if method == "coral":
                            loss_aux = coral_loss(features_s, features_t)
                        else:
                            loss_aux = mmd_loss(
                                features_s,
                                features_t,
                                tuple(float(v) for v in adapt_cfg.mmd_sigmas),
                            )
                    loss = loss_cls + float(adapt_cfg.align_weight) * warm * loss_aux
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = len(ys)
            running += float(loss.item()) * n
            running_cls += float(loss_cls.item()) * n
            running_aux += float(loss_aux.item()) * n
            seen += n
        scheduler.step()

        y_true, y_prob, _ = run_inference(
            model,
            val_loader,
            cfg.device,
            cfg.training.mixed_precision,
        )
        val_auc = float(compute_metrics(y_true, y_prob)["auc"])
        row = {
            "epoch": float(epoch + 1),
            "train_loss": running / max(1, seen),
            "classification_loss": running_cls / max(1, seen),
            "auxiliary_loss": running_aux / max(1, seen),
            "val_loss": _validation_loss(model, val_loader, cfg),
            "val_auc": val_auc,
            "source_steps": float(len(src_loader)),
            "lr": epoch_lr,
        }
        history.append(row)
        print(
            f"[{method} {arch} seed={seed}] {epoch + 1:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"{method} did not produce a selectable checkpoint.")
    model.load_state_dict(best_state, strict=True)
    provenance = {
        **provenance,
        "initial_checkpoint": str(Path(source_checkpoint).resolve()),
        "initial_checkpoint_sha256": source_hash,
        "initial_best_val_auc": float(source_meta["best_val_auc"]),
        "source_steps_per_epoch": int(len(src_loader)),
        "epochs_executed": epochs,
        "batchnorm_running_stats_frozen": bool(adapt_cfg.freeze_batchnorm_running_stats),
    }
    return _save_checkpoint(
        checkpoint_path,
        model,
        model_type=model_type,
        arch=arch,
        seed=seed,
        method=method,
        best_epoch=best_epoch,
        best_val_auc=best_auc,
        history=history,
        provenance=provenance,
    )


def train_finetune_checkpoint(
    source_checkpoint: str | Path,
    target_train: pd.DataFrame,
    target_val: pd.DataFrame,
    root: Path,
    cfg: Config,
    *,
    arch: str,
    seed: int,
    fraction: float,
    checkpoint_path: str | Path,
    provenance: dict[str, Any],
) -> Path:
    """Fine-tune from the same source checkpoint on a patient-disjoint labelled budget."""
    if target_train.empty or target_val.empty:
        raise ValueError("Fine-tuning train and validation cohorts must both be non-empty.")
    overlap = set(target_train["patient_id"].astype(str)) & set(
        target_val["patient_id"].astype(str)
    )
    if overlap:
        raise ValueError(f"Patient leakage in fine-tuning split: {sorted(overlap)[:5]}")

    seed_everything(seed)
    baseline_state, source_meta = _baseline_state_from_source(source_checkpoint)
    model = _make_baseline(arch, cfg, pretrained=False)
    model.load_state_dict(baseline_state, strict=True)
    model.to(cfg.device)

    ft_cfg = cfg.publication.finetune
    epochs = int(ft_cfg.epochs)
    train_loader = _loader(target_train, root, cfg, train=True, seed=seed + 200_000)
    val_loader = _loader(target_val, root, cfg, train=False, seed=seed)
    criterion = nn.CrossEntropyLoss(weight=class_weights(target_train).to(cfg.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ft_cfg.lr),
        weight_decay=float(ft_cfg.weight_decay),
    )
    scheduler = _cosine_scheduler(
        optimizer,
        epochs,
        warmup_epochs=int(ft_cfg.lr_warmup_epochs),
    )
    scaler = _amp_scaler(cfg)
    selected_auc, selected_epoch = float("nan"), -1
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        running, seen = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, scaler):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(y)
            seen += len(y)
        scheduler.step()

        val_auc = _patient_validation_auc(model, val_loader, cfg)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": running / max(1, seen),
            "val_loss": _validation_loss(model, val_loader, cfg),
            "val_auc": val_auc,
            "lr": epoch_lr,
        }
        history.append(row)
        print(
            f"[finetune {fraction:.2f} {arch} seed={seed}] {epoch + 1:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} val_auc={val_auc:.4f}"
        )
        # Epoch count is frozen a priori; tiny patient validation sets are monitoring-only.
        selected_auc = val_auc
        selected_epoch = epoch + 1

    if selected_epoch != epochs:
        raise RuntimeError("Fine-tuning did not complete its pre-specified fixed epoch count.")
    provenance = {
        **provenance,
        "initial_checkpoint": str(Path(source_checkpoint).resolve()),
        "initial_checkpoint_sha256": sha256_file(source_checkpoint),
        "initial_best_val_auc": float(source_meta["best_val_auc"]),
        "label_fraction": float(fraction),
        "n_train_images": int(len(target_train)),
        "n_val_images": int(len(target_val)),
        "n_train_patients": int(target_train["patient_id"].astype(str).nunique()),
        "n_val_patients": int(target_val["patient_id"].astype(str).nunique()),
        "validation_unit": "patient",
        "patient_aggregation": "mean_logit",
        "checkpoint_selection_rule": str(ft_cfg.selection_rule),
        "epochs_executed": epochs,
    }
    return _save_checkpoint(
        checkpoint_path,
        model,
        model_type="baseline",
        arch=arch,
        seed=seed,
        method=f"finetune_{int(round(100 * fraction))}pct",
        best_epoch=selected_epoch,
        best_val_auc=selected_auc,
        history=history,
        provenance=provenance,
    )
