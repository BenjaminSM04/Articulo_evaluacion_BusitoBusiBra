"""Supervised training loop with AMP, warmup+cosine schedule and early stopping.

Returns the model with best-validation weights restored, plus the training history. The monitored
validation metric (default AUC) is what early stopping and checkpoint selection use.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..evaluation.metrics import compute_metrics, run_inference


def _build_optimizer(model: nn.Module, cfg: Config):
    t = cfg.training
    params = [p for p in model.parameters() if p.requires_grad]
    if t.optimizer == "sgd":
        return torch.optim.SGD(params, lr=t.lr, momentum=0.9, weight_decay=t.weight_decay)
    return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)


def _build_scheduler(optimizer, cfg: Config):
    t = cfg.training
    if t.scheduler == "none":
        return None, False
    if t.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5), True

    warmup, total = t.warmup_epochs, t.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), False


def train_model(model: nn.Module, loaders: dict, cfg: Config,
                class_weights: torch.Tensor | None = None,
                device: str | None = None, ckpt_path: str | Path | None = None) -> dict:
    device = device or cfg.device
    model.to(device)
    t = cfg.training

    weight = class_weights.to(device) if (class_weights is not None
                                          and t.class_weights == "balanced") else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    scheduler, step_on_metric = _build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=t.mixed_precision and "cuda" in str(device))

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_metric, best_epoch, best_state = -np.inf, -1, None
    patience = t.early_stopping_patience

    for epoch in range(t.epochs):
        # ---- train ----
        model.train()
        running = 0.0
        for x, y, _ in tqdm(loaders["train"], desc=f"época {epoch+1}/{t.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                enabled=scaler.is_enabled()):
                out = model(x)
                out = out[0] if isinstance(out, tuple) else out
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        train_loss = running / len(loaders["train"].dataset)

        # ---- validate ----
        y_true, y_prob, _ = run_inference(model, loaders["val"], device, amp=t.mixed_precision)
        val_metrics = compute_metrics(y_true, y_prob)
        monitored = val_metrics[t.early_stopping_metric]
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_metrics["auc"])

        if scheduler is not None:
            scheduler.step(monitored) if step_on_metric else scheduler.step()

        print(f"  época {epoch+1}: train_loss={train_loss:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  val_acc={val_metrics['accuracy']:.4f}")

        # ---- checkpoint / early stopping ----
        if monitored > best_metric:
            best_metric, best_epoch = monitored, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  early stopping en época {epoch+1} (mejor={best_metric:.4f} "
                  f"en época {best_epoch+1}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "arch": cfg.model.architecture,
                    "best_metric": best_metric, "best_epoch": best_epoch}, ckpt_path)

    return {"model": model, "history": history,
            "best_metric": best_metric, "best_epoch": best_epoch}
