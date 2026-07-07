"""Utilidades de entrenamiento compartidas: splits por paciente (sin fuga de datos),
optimizador/scheduler con warmup, pesos por clase, early stopping, DataLoaders
reproducibles, bucle de época y checkpoints.

Reglas anti-fuga que implementa este módulo:
  * Split INTERNO (BUSI): estratificado; agrupado por paciente solo si el dataset
    tiene patient_id (BUSI no; BUS-BRA sí).
  * Split del OBJETIVO (BUS-BRA) para DA: por paciente en 'adapt' (sin etiquetas)
    y 'test' reservado; el test NUNCA se usa para adaptar.
  * Few-shot: el subconjunto etiquetado se muestrea del pool de adaptación (disjunto
    del test) y por paciente.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..datasets import UltrasoundDataset, build_transforms
from ..evaluation.metrics import auc_metric, run_inference
from ..utils.config import resolve_group_by_patient
from ..utils.seed import seed_worker


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def build_loader(df: pd.DataFrame, cfg: dict, train: bool, shuffle: bool | None = None,
                 reference=None) -> DataLoader:
    """Crea un DataLoader reproducible sobre un sub-DataFrame de índice."""
    tf = build_transforms(cfg, train=train, reference=reference)
    ds = UltrasoundDataset(df, tf, roi_mode=cfg["data"]["roi_mode"],
                           roi_margin=cfg["data"]["roi_margin"])
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train if shuffle is None else shuffle,
        num_workers=cfg["train"]["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=("cuda" in str(cfg["device"])),
        drop_last=False,
    )


# --------------------------------------------------------------------------- #
# Splits (por paciente cuando corresponde)
# --------------------------------------------------------------------------- #
def _has_patient(df: pd.DataFrame) -> bool:
    return df["patient_id"].notna().all() and df["patient_id"].nunique() > 1


def make_internal_splits(df: pd.DataFrame, cfg: dict, dataset_name: str = "busi",
                         seed: int = 42) -> dict:
    """Divide el dataset interno en test + (folds de) train/val.

    Devuelve {'test': df_test, 'folds': [(df_tr, df_val), ...], 'grouped': bool}.
    """
    from sklearn.model_selection import (
        GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit,
    )
    grouped = resolve_group_by_patient(cfg, dataset_name) and _has_patient(df)
    y = df["label_idx"].values
    groups = df["patient_id"].values
    test_frac = cfg["split"]["internal_test_fraction"]
    val_frac = cfg["split"]["internal_val_fraction"]
    n_folds = int(cfg["split"].get("n_folds", 0))

    if test_frac and test_frac > 0:
        if grouped:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            trv_idx, test_idx = next(splitter.split(df, y, groups))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            trv_idx, test_idx = next(splitter.split(df, y))
    else:
        trv_idx, test_idx = np.arange(len(df)), np.array([], dtype=int)

    df_trv = df.iloc[trv_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    y2, g2 = df_trv["label_idx"].values, df_trv["patient_id"].values

    folds = []
    if n_folds and n_folds > 1:
        if grouped:
            kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            iterator = kf.split(df_trv, y2, g2)
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            iterator = kf.split(df_trv, y2)
        for tr, va in iterator:
            folds.append((df_trv.iloc[tr].reset_index(drop=True),
                          df_trv.iloc[va].reset_index(drop=True)))
    else:
        if grouped:
            sp = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr, va = next(sp.split(df_trv, y2, g2))
        else:
            sp = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr, va = next(sp.split(df_trv, y2))
        folds.append((df_trv.iloc[tr].reset_index(drop=True),
                      df_trv.iloc[va].reset_index(drop=True)))
    return {"test": df_test, "folds": folds, "grouped": grouped}


def make_target_splits(df_target: pd.DataFrame, cfg: dict, seed: int = 42) -> dict:
    """Divide BUS-BRA por paciente en 'adapt' (sin etiquetas) y 'test' reservado."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    frac = cfg["split"]["target_test_fraction"]
    y, g = df_target["label_idx"].values, df_target["patient_id"].values
    if _has_patient(df_target):
        sp = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        adapt_idx, test_idx = next(sp.split(df_target, y, g))
    else:
        sp = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        adapt_idx, test_idx = next(sp.split(df_target, y))
    return {"adapt": df_target.iloc[adapt_idx].reset_index(drop=True),
            "test": df_target.iloc[test_idx].reset_index(drop=True)}


def sample_few_shot(df_pool: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Muestrea (por paciente si es posible) una fracción etiquetada del pool objetivo."""
    if fraction <= 0:
        return df_pool.iloc[0:0]
    rng = np.random.default_rng(seed)
    if _has_patient(df_pool):
        patients = df_pool["patient_id"].unique()
        k = max(1, int(round(len(patients) * fraction)))
        chosen = set(rng.choice(patients, size=k, replace=False))
        return df_pool[df_pool["patient_id"].isin(chosen)].reset_index(drop=True)
    k = max(1, int(round(len(df_pool) * fraction)))
    idx = rng.choice(len(df_pool), size=k, replace=False)
    return df_pool.iloc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Optimizador, scheduler, pesos de clase, early stopping
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, cfg: dict):
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=0.9,
                               weight_decay=t["weight_decay"], nesterov=True)
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, cfg: dict):
    """Devuelve (scheduler, needs_metric). Cosine con warmup por época, o plateau."""
    t = cfg["train"]
    epochs, warm = t["epochs"], t.get("warmup_epochs", 0)
    if t["scheduler"] == "cosine":
        def fn(epoch):
            if epoch < warm:
                return (epoch + 1) / max(1, warm)
            p = (epoch - warm) / max(1, epochs - warm)
            return 0.5 * (1.0 + math.cos(math.pi * p))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn), False
    if t["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3), True
    return None, False


def compute_class_weights(labels, device) -> torch.Tensor:
    """Pesos inversos a la frecuencia (balanced) para CrossEntropy."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    w = torch.ones(2, dtype=torch.float32)
    for c, val in zip(classes, weights):
        w[int(c)] = float(val)
    return w.to(device)


class EarlyStopping:
    """Detiene el entrenamiento si la métrica de validación deja de mejorar."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Devuelve True si la métrica mejoró en este paso."""
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# --------------------------------------------------------------------------- #
# Bucle de entrenamiento supervisado + checkpoints
# --------------------------------------------------------------------------- #
def _amp_enabled(cfg: dict) -> bool:
    return bool(cfg["train"].get("mixed_precision", True)) and "cuda" in str(cfg["device"])


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp: bool) -> float:
    model.train()
    running = 0.0
    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, enabled=amp):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / max(1, len(loader.dataset))


def save_checkpoint(model: nn.Module, cfg: dict, path: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "extra": extra or {}}, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return ckpt if isinstance(ckpt, dict) else {}


def fit_classifier(model, df_train, df_val, cfg, device, logger, ckpt_path,
                   class_weight=None) -> float:
    """Entrena un clasificador con early stopping sobre el AUC de validación (source).

    Guarda el MEJOR checkpoint y lo recarga al final. Devuelve el mejor AUC de val.
    """
    train_loader = build_loader(df_train, cfg, train=True)
    val_loader = build_loader(df_val, cfg, train=False, shuffle=False)
    optimizer = build_optimizer(model, cfg)
    scheduler, needs_metric = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    amp = _amp_enabled(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper = EarlyStopping(cfg["train"]["early_stopping_patience"], mode="max")
    best_auc = -math.inf

    for epoch in range(cfg["train"]["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
        y_true, y_prob, _ = run_inference(model, val_loader, device, amp)
        try:
            val_auc = auc_metric(y_true, y_prob)
        except ValueError:
            val_auc = float("nan")
        if scheduler is not None:
            scheduler.step(val_auc) if needs_metric else scheduler.step()
        improved = stopper.step(val_auc if not math.isnan(val_auc) else -math.inf)
        if improved:
            best_auc = val_auc
            save_checkpoint(model, cfg, ckpt_path, extra={"epoch": epoch, "val_auc": val_auc})
        if logger:
            logger.info("epoch %02d | loss %.4f | val_auc %.4f%s",
                        epoch, loss, val_auc, "  *" if improved else "")
        if stopper.should_stop:
            if logger:
                logger.info("early stopping en epoch %d (mejor val_auc=%.4f)", epoch, best_auc)
            break

    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path, device)
    return best_auc


def train_domain_adaptation(model, df_s_tr, df_s_val, df_t_adapt, cfg, device, logger,
                            ckpt_path, batch_loss_fn, class_weight=None) -> float:
    """Bucle genérico de adaptación de dominio: source etiquetado + target SIN etiquetas.

    ``batch_loss_fn(model, batch_s, batch_t, device, progress, class_weight) -> loss``
    define la pérdida específica del método (DANN, CORAL, MMD). La SELECCIÓN de modelo
    usa el AUC de validación del SOURCE (nunca etiquetas del target). Guarda el mejor
    clasificador (para DANN, el submódulo ``classifier``, compatible con evaluación).
    """
    import itertools
    src_loader = build_loader(df_s_tr, cfg, train=True)
    val_loader = build_loader(df_s_val, cfg, train=False, shuffle=False)
    tgt_loader = build_loader(df_t_adapt, cfg, train=True, shuffle=True)

    optimizer = build_optimizer(model, cfg)
    scheduler, needs_metric = build_scheduler(optimizer, cfg)
    amp = _amp_enabled(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper = EarlyStopping(cfg["train"]["early_stopping_patience"], mode="max")
    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    to_save = getattr(model, "classifier", model)   # DANN -> guardar clasificador interno
    best_auc = -math.inf

    steps = max(len(src_loader), len(tgt_loader))
    total = cfg["train"]["epochs"] * steps
    gstep = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        src_it, tgt_it = itertools.cycle(src_loader), itertools.cycle(tgt_loader)
        running = 0.0
        for _ in range(steps):
            batch_s, batch_t = next(src_it), next(tgt_it)
            progress = gstep / max(1, total)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev_type, enabled=amp):
                loss = batch_loss_fn(model, batch_s, batch_t, device, progress, class_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            gstep += 1
        y_true, y_prob, _ = run_inference(model, val_loader, device, amp)
        try:
            val_auc = auc_metric(y_true, y_prob)
        except ValueError:
            val_auc = float("nan")
        if scheduler is not None:
            scheduler.step(val_auc) if needs_metric else scheduler.step()
        improved = stopper.step(val_auc if not math.isnan(val_auc) else -math.inf)
        if improved:
            best_auc = val_auc
            save_checkpoint(to_save, cfg, ckpt_path, extra={"epoch": epoch, "val_auc": val_auc})
        if logger:
            logger.info("epoch %02d | da_loss %.4f | src_val_auc %.4f%s",
                        epoch, running / max(1, steps), val_auc, "  *" if improved else "")
        if stopper.should_stop:
            break

    if os.path.exists(ckpt_path):
        load_checkpoint(to_save, ckpt_path, device)
    return best_auc
