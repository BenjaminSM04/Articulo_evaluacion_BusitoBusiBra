"""Domain-adaptation strategies for the source->target setting.

Three methods, matching the proposal:

* ``finetune``       — supervised fine-tuning on a small labeled subset of the target (few-shot DA).
* ``dann``           — unsupervised adversarial alignment (Ganin et al., 2016): source labels +
                       unlabeled target, with a Gradient Reversal Layer.
* ``self_training``  — pseudo-label high-confidence target samples with the source model, then
                       retrain; repeat for several rounds.

Each function returns the adapted model. They are intentionally small and composable so they can
be swapped via ``config.domain_adaptation.method``.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import UltrasoundDataset, class_weights
from ..evaluation.metrics import run_inference
from ..models.architectures import DANNModel, build_model
from ..utils.seed import seed_worker
from .train import train_model
from .transforms import build_eval_transforms, build_train_transforms


# ---------------------------------------------------------------------------
# 1) Supervised fine-tuning on labeled target (few-shot)
# ---------------------------------------------------------------------------
def finetune_on_target(model: nn.Module, target_df: pd.DataFrame, root, cfg: Config):
    """Fine-tune a source-trained model on a labeled target subset.

    Uses ``cfg.domain_adaptation.finetune.n_target_labeled`` samples per class for adaptation.
    Returns (adapted_model, used_indices) so those indices can be excluded from evaluation.
    """
    ft = cfg.domain_adaptation.finetune
    n = ft.n_target_labeled
    if n <= 0:
        print("[finetune] n_target_labeled=0 -> escenario no supervisado; se omite fine-tuning.")
        return model, np.array([], dtype=int)

    rng = np.random.default_rng(cfg.seed)
    used = []
    for cls in (0, 1):
        cls_idx = target_df.index[target_df["label_idx"] == cls].to_numpy()
        used.extend(rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False))
    used = np.array(used, dtype=int)

    if ft.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Build a tiny train/val loader pair from the labeled target subset.
    sub = target_df.loc[used].reset_index(drop=True)
    val_n = max(1, int(0.2 * len(sub)))
    val_idx = np.arange(len(sub))[:val_n]
    train_idx = np.arange(len(sub))[val_n:]
    train_tf, eval_tf = build_train_transforms(cfg), build_eval_transforms(cfg)
    loaders = {
        "train": DataLoader(UltrasoundDataset(sub.iloc[train_idx], root, train_tf),
                            batch_size=cfg.training.batch_size, shuffle=True,
                            num_workers=cfg.training.num_workers, worker_init_fn=seed_worker),
        "val": DataLoader(UltrasoundDataset(sub.iloc[val_idx], root, eval_tf),
                          batch_size=cfg.training.batch_size, shuffle=False,
                          num_workers=cfg.training.num_workers),
    }
    cfg_ft = copy.deepcopy(cfg)
    cfg_ft.training["epochs"] = ft.epochs
    cfg_ft.training["lr"] = ft.lr
    out = train_model(model, loaders, cfg_ft, class_weights(sub), device=cfg.device)
    return out["model"], used


def _stratified_fraction_indices(df: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction debe estar en (0, 1].")
    rng = np.random.default_rng(seed)
    used: list[int] = []
    for cls in (0, 1):
        cls_idx = df.index[df["label_idx"] == cls].to_numpy()
        if len(cls_idx) == 0:
            continue
        n = max(1, int(math.ceil(len(cls_idx) * float(fraction))))
        used.extend(rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False).tolist())
    return np.asarray(sorted(used), dtype=int)


def _small_supervised_split(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Stable train/val split for very small few-shot subsets."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in (0, 1):
        cls_pos = np.flatnonzero(df["label_idx"].to_numpy() == cls)
        rng.shuffle(cls_pos)
        if len(cls_pos) <= 2:
            train_idx.extend(cls_pos.tolist())
            val_idx.extend(cls_pos.tolist())
            continue
        n_val = max(1, int(round(0.2 * len(cls_pos))))
        val_idx.extend(cls_pos[:n_val].tolist())
        train_idx.extend(cls_pos[n_val:].tolist())
    if not train_idx:
        train_idx = list(range(len(df)))
    if not val_idx:
        val_idx = train_idx.copy()
    return {
        "train_idx": np.asarray(sorted(set(train_idx)), dtype=int),
        "val_idx": np.asarray(sorted(set(val_idx)), dtype=int),
    }


def finetune_on_target_fraction(
    model: nn.Module,
    target_df: pd.DataFrame,
    root,
    cfg: Config,
    fraction: float,
):
    """Fine-tune on a stratified fraction of the target adaptation pool.

    The caller must pass only the target pool allowed for adaptation, not the reserved target test
    set. Returns (adapted_model, used_indices_in_target_df).
    """
    used = _stratified_fraction_indices(target_df, fraction, cfg.seed)
    sub = target_df.loc[used].reset_index(drop=True)

    if cfg.domain_adaptation.finetune.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    split = _small_supervised_split(sub, cfg.seed)
    train_tf, eval_tf = build_train_transforms(cfg), build_eval_transforms(cfg)
    loaders = {
        "train": DataLoader(
            UltrasoundDataset(sub.iloc[split["train_idx"]], root, train_tf),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=cfg.training.num_workers > 0,
        ),
        "val": DataLoader(
            UltrasoundDataset(sub.iloc[split["val_idx"]], root, eval_tf),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
        ),
    }
    cfg_ft = copy.deepcopy(cfg)
    cfg_ft.training["epochs"] = cfg.domain_adaptation.finetune.epochs
    cfg_ft.training["lr"] = cfg.domain_adaptation.finetune.lr
    out = train_model(model, loaders, cfg_ft, class_weights(sub.iloc[split["train_idx"]]),
                      device=cfg.device)
    return out["model"], used


# ---------------------------------------------------------------------------
# 2) DANN — unsupervised adversarial domain adaptation
# ---------------------------------------------------------------------------
def dann_train(source_df: pd.DataFrame, target_df: pd.DataFrame, root, cfg: Config) -> nn.Module:
    """Train a DANN model: source supervised + domain-adversarial on unlabeled target."""
    device = cfg.device
    da = cfg.domain_adaptation.dann
    model = DANNModel(arch=cfg.model.architecture, num_classes=cfg.model.num_classes,
                      pretrained=cfg.model.pretrained, dropout=cfg.model.dropout).to(device)

    train_tf = build_train_transforms(cfg)
    src_loader = DataLoader(UltrasoundDataset(source_df, root, train_tf),
                            batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.training.num_workers, worker_init_fn=seed_worker)
    tgt_loader = DataLoader(UltrasoundDataset(target_df, root, train_tf),
                            batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.training.num_workers, worker_init_fn=seed_worker)

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights(source_df).to(device))
    dom_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.training.mixed_precision and "cuda" in str(device)
    )

    total_epochs = cfg.training.epochs
    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    for epoch in range(total_epochs):
        model.train()
        for step, ((xs, ys, _), (xt, _, _)) in enumerate(
                zip(src_loader, itertools.cycle(tgt_loader))):
            if step >= steps_per_epoch:
                break
            # lambda schedule (Ganin): ramps 0 -> lambda_grl over training
            if da.lambda_schedule:
                p = (epoch * steps_per_epoch + step) / (total_epochs * steps_per_epoch)
                lambda_ = da.lambda_grl * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            else:
                lambda_ = da.lambda_grl

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            d_src = torch.zeros(xs.size(0), dtype=torch.long, device=device)
            d_tgt = torch.ones(xt.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                enabled=scaler.is_enabled()):
                cls_logits, dom_logits_s = model(xs, lambda_)
                _, dom_logits_t = model(xt, lambda_)
                loss = (cls_criterion(cls_logits, ys)
                        + dom_criterion(dom_logits_s, d_src)
                        + dom_criterion(dom_logits_t, d_tgt))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"[dann] época {epoch+1}/{total_epochs} (lambda final={lambda_:.3f})")
    return model


# ---------------------------------------------------------------------------
# 3) Feature alignment — CORAL / MMD
# ---------------------------------------------------------------------------
def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: squared distance between source and target covariances."""
    d = fs.size(1)
    fs_c = fs - fs.mean(0, keepdim=True)
    ft_c = ft - ft.mean(0, keepdim=True)
    cov_s = fs_c.t() @ fs_c / max(1, fs.size(0) - 1)
    cov_t = ft_c.t() @ ft_c / max(1, ft.size(0) - 1)
    return (cov_s - cov_t).pow(2).sum() / (4 * d * d)


def mmd_loss(fs: torch.Tensor, ft: torch.Tensor, sigmas=(1, 2, 4, 8, 16)) -> torch.Tensor:
    """Multi-kernel RBF maximum mean discrepancy."""
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aa = a.pow(2).sum(1, keepdim=True)
        bb = b.pow(2).sum(1, keepdim=True)
        dist = aa - 2 * a @ b.t() + bb.t()
        kernel = torch.zeros_like(dist)
        for sigma in sigmas:
            kernel = kernel + torch.exp(-dist / (2 * float(sigma) ** 2))
        return kernel

    return rbf(fs, fs).mean() + rbf(ft, ft).mean() - 2 * rbf(fs, ft).mean()


def feature_alignment_train(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    root,
    cfg: Config,
    method: str = "coral",
) -> nn.Module:
    """Train source-supervised model with unsupervised target feature alignment."""
    method = method.lower()
    if method not in {"coral", "mmd"}:
        raise ValueError("method debe ser 'coral' o 'mmd'.")

    device = cfg.device
    da_cfg = cfg.domain_adaptation
    align_weight = float(da_cfg.get("align_weight", 1.0))
    align_warmup_epochs = int(da_cfg.get("align_warmup_epochs", 5))
    mmd_sigmas = da_cfg.get("mmd_sigmas", [1, 2, 4, 8, 16])

    model = build_model(cfg).to(device)
    train_tf = build_train_transforms(cfg)
    src_loader = DataLoader(
        UltrasoundDataset(source_df, root, train_tf),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.training.num_workers > 0,
    )
    tgt_loader = DataLoader(
        UltrasoundDataset(target_df, root, train_tf),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.training.num_workers > 0,
    )
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights(source_df).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.training.mixed_precision and "cuda" in str(device)
    )

    steps_per_epoch = max(1, min(len(src_loader), len(tgt_loader)))
    for epoch in range(cfg.training.epochs):
        model.train()
        for step, ((xs, ys, _), (xt, _, _)) in enumerate(
            zip(src_loader, itertools.cycle(tgt_loader))
        ):
            if step >= steps_per_epoch:
                break
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            progress_epoch = epoch + step / max(1, steps_per_epoch)
            warm = min(1.0, progress_epoch / max(1, align_warmup_epochs))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if "cuda" in str(device) else "cpu",
                enabled=scaler.is_enabled(),
            ):
                logits_s = model(xs)
                fs = model.features(xs)
                ft = model.features(xt)
                align = coral_loss(fs, ft) if method == "coral" else mmd_loss(fs, ft, mmd_sigmas)
                loss = cls_criterion(logits_s, ys) + align_weight * warm * align
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(
            f"[{method}] época {epoch+1}/{cfg.training.epochs} "
            f"(align_w={align_weight * warm:.3f})"
        )
    return model


# ---------------------------------------------------------------------------
# 4) Self-training with pseudo-labels
# ---------------------------------------------------------------------------
def self_training(model: nn.Module, source_df: pd.DataFrame, target_df: pd.DataFrame,
                  root, cfg: Config) -> nn.Module:
    """Iteratively pseudo-label confident target samples and retrain on source + pseudo-labels."""
    st = cfg.domain_adaptation.self_training
    eval_tf = build_eval_transforms(cfg)
    cur_model = model

    for rnd in range(st.n_rounds):
        loader = DataLoader(UltrasoundDataset(target_df, root, eval_tf),
                            batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers)
        _, prob, idx = run_inference(cur_model, loader, cfg.device, cfg.training.mixed_precision)
        confident = (prob >= st.confidence_threshold) | (prob <= 1 - st.confidence_threshold)
        if confident.sum() == 0:
            print(f"[self-training] ronda {rnd+1}: 0 muestras confiables; se detiene.")
            break
        pseudo = target_df.iloc[idx[confident]].copy()
        pseudo["label_idx"] = (prob[confident] >= 0.5).astype(int)
        pseudo["label"] = pseudo["label_idx"].map({0: "benign", 1: "malignant"})
        combined = pd.concat([source_df, pseudo], ignore_index=True)
        print(f"[self-training] ronda {rnd+1}: +{len(pseudo)} pseudo-etiquetas "
              f"(total entrenamiento={len(combined)}).")

        # Retrain from scratch on the augmented set with a small internal val split.
        from sklearn.model_selection import train_test_split

        from ..models.architectures import build_model
        tr, va = train_test_split(np.arange(len(combined)), test_size=0.15,
                                  random_state=cfg.seed, stratify=combined["label_idx"])
        train_tf = build_train_transforms(cfg)
        loaders = {
            "train": DataLoader(UltrasoundDataset(combined.iloc[tr], root, train_tf),
                                batch_size=cfg.training.batch_size, shuffle=True,
                                num_workers=cfg.training.num_workers, worker_init_fn=seed_worker),
            "val": DataLoader(UltrasoundDataset(combined.iloc[va], root, eval_tf),
                              batch_size=cfg.training.batch_size, shuffle=False,
                              num_workers=cfg.training.num_workers),
        }
        cur_model = build_model(cfg).to(cfg.device)
        cur_model = train_model(cur_model, loaders, cfg, class_weights(combined),
                                device=cfg.device)["model"]
    return cur_model
