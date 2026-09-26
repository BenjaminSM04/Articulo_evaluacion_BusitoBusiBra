"""PyTorch dataset, cross-validation splitting and dataloader construction.

Everything operates on *manifests* (CSV produced by preprocessing). Splits are grouped by patient
when patient IDs are available (BUS-BRA) to prevent leakage; BUSI has no patient IDs, so its
splits are image-level and stratified — this limitation is documented in the methodology.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..training.control_preprocessing import crop_union_roi, percentile_normalize_rgb
from ..utils.seed import seed_worker

OPTIONAL_MANIFEST_COLUMNS = ("mask_path", "bbox")


def roi_mask_paths(row: pd.Series) -> list[str]:
    """Resolve la lista declarada de máscaras; el campo plural conserva uniones BUSI."""
    plural = row.get("mask_paths", "")
    raw = plural if not pd.isna(plural) and str(plural).strip() else row.get("mask_path", "")
    if pd.isna(raw):
        raw = ""
    paths = [part.strip() for part in str(raw).split("|")]
    if not paths or any(not part for part in paths):
        raise ValueError(f"ROI sin máscaras para {row.get('image_path', '')}")
    return paths


def load_manifest(cfg: Config, key: str) -> pd.DataFrame:
    """Read a dataset manifest produced by preprocessing."""
    path = cfg.path("data_processed") / f"{key}_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta el manifest {path}. Ejecuta scripts/02_preprocess.py.")
    df = pd.read_csv(path, dtype={"patient_id": str, "birads": str}).fillna("")
    for col in OPTIONAL_MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.reset_index(drop=True)


class UltrasoundDataset(Dataset):
    """Reads images listed in a manifest DataFrame and applies an albumentations transform."""

    def __init__(self, df: pd.DataFrame, root: Path, transform=None, *, preprocessing="none"):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        if preprocessing not in {"none", "intensity", "roi"}:
            raise ValueError(f"Control desconocido: {preprocessing}")
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.root / row["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer {row['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.preprocessing == "intensity":
            img = percentile_normalize_rgb(img)
        elif self.preprocessing == "roi":
            masks = []
            for part in roi_mask_paths(row):
                path = (self.root / part).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise ValueError(f"Máscara ROI fuera del proyecto: {part}")
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"No se pudo leer mascara ROI {path}")
                masks.append(mask)
            img = crop_union_roi(img, masks)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        label = int(row["label_idx"])
        return img, label, idx


# ---------------------------------------------------------------------------
# Cross-validation splitting
# ---------------------------------------------------------------------------
def make_cv_splits(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """Build k folds. Each fold is a dict with train_idx / val_idx / test_idx (positional)."""
    n_folds = cfg.cross_validation.n_folds
    seed = cfg.seed
    y = df["label_idx"].to_numpy()
    groups = df["patient_id"].to_numpy()
    has_groups = (
        cfg.cross_validation.group_by_patient
        and (df["patient_id"].astype(str).str.len() > 0).all()
    )

    idx_all = np.arange(len(df))
    if has_groups:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(idx_all, y, groups)
    else:
        if cfg.cross_validation.group_by_patient:
            print("[cv] aviso: sin IDs de paciente (p.ej. BUSI); se usa split a nivel imagen.")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(idx_all, y)

    folds = []
    for trainval_idx, test_idx in fold_iter:
        # Carve a validation set out of train (for early stopping). Group-aware when possible.
        strat = y[trainval_idx]
        if has_groups:
            g = groups[trainval_idx]
            # split groups, not rows, to keep patients intact
            uniq = np.unique(g)
            tr_g, val_g = train_test_split(uniq, test_size=cfg.cross_validation.val_fraction,
                                           random_state=seed)
            train_idx = trainval_idx[np.isin(g, tr_g)]
            val_idx = trainval_idx[np.isin(g, val_g)]
        else:
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=cfg.cross_validation.val_fraction,
                random_state=seed, stratify=strat)
        folds.append({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    return folds


def make_target_adaptation_split(
    df: pd.DataFrame,
    cfg: Config,
    test_fraction: float = 0.40,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Reserve a fixed target-domain test set, grouped by patient when possible.

    Returns positional indices in ``df``. The ``adapt_idx`` partition is the only target data
    allowed for unsupervised adaptation or supervised few-shot fine-tuning; ``test_idx`` remains
    untouched until final evaluation.
    """
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1.")

    seed = cfg.seed if seed is None else seed
    y = df["label_idx"].to_numpy()
    patient = df.get("patient_id", pd.Series([""] * len(df))).astype(str)
    has_groups = patient.str.len().gt(0).all()

    if has_groups:
        group_df = (
            pd.DataFrame({"patient_id": patient, "label_idx": y})
            .groupby("patient_id", as_index=False)["label_idx"]
            .max()
        )
        groups = group_df["patient_id"].to_numpy()
        labels = group_df["label_idx"].to_numpy()
        try:
            adapt_groups, test_groups = train_test_split(
                groups,
                test_size=test_fraction,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            adapt_groups, test_groups = train_test_split(
                groups,
                test_size=test_fraction,
                random_state=seed,
                stratify=None,
            )
        adapt_idx = np.flatnonzero(patient.isin(adapt_groups).to_numpy())
        test_idx = np.flatnonzero(patient.isin(test_groups).to_numpy())
    else:
        idx = np.arange(len(df))
        try:
            adapt_idx, test_idx = train_test_split(
                idx,
                test_size=test_fraction,
                random_state=seed,
                stratify=y,
            )
        except ValueError:
            adapt_idx, test_idx = train_test_split(
                idx,
                test_size=test_fraction,
                random_state=seed,
                stratify=None,
            )

    return {
        "adapt_idx": np.asarray(sorted(adapt_idx), dtype=int),
        "test_idx": np.asarray(sorted(test_idx), dtype=int),
    }


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(df: pd.DataFrame, root: Path, train_tf, eval_tf, cfg: Config,
                      split: dict) -> dict[str, DataLoader]:
    """Create train/val/test loaders from a fold split dict."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    bs = cfg.training.batch_size
    nw = cfg.training.num_workers

    def loader(indices, transform, shuffle):
        ds = UltrasoundDataset(df.iloc[indices], root, transform)
        # persistent_workers evita re-spawnear los workers en cada época (crítico en
        # Windows, donde el spawn cuesta ~30 s/época y domina el tiempo de entrenamiento).
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                          pin_memory=True, worker_init_fn=seed_worker, generator=g,
                          drop_last=False, persistent_workers=nw > 0)

    return {
        "train": loader(split["train_idx"], train_tf, True),
        "val": loader(split["val_idx"], eval_tf, False),
        "test": loader(split["test_idx"], eval_tf, False),
    }


def class_weights(df: pd.DataFrame, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced benign/malignant."""
    counts = df["label_idx"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0).to_numpy().astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
