"""Clase Dataset común a BUSI y BUS-BRA y utilidades de E/S de imágenes.

Ambos conjuntos se representan con el MISMO esquema de índice (un DataFrame con
columnas homogéneas), de modo que el resto del pipeline es agnóstico al dataset:

    image_path : ruta absoluta a la imagen
    mask_path  : ruta a la máscara de lesión (o None)
    label      : 'benign' | 'malignant'
    label_idx  : 0 | 1
    patient_id : identificador de paciente (o None si el dataset no lo provee)
    dataset    : 'busi' | 'busbra'
    domain     : 0 (source) | 1 (target)   [se asigna al construir loaders]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

LABEL_MAP = {"benign": 0, "malignant": 1}
INDEX_COLUMNS = ["image_path", "mask_path", "label", "label_idx", "patient_id", "dataset"]


def load_rgb(path: str) -> np.ndarray:
    """Carga una imagen como RGB uint8 (H,W,3), replicando canal si es monocroma."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_mask(path: str) -> np.ndarray:
    """Carga una máscara binaria (H,W) uint8 {0,1}."""
    with Image.open(path) as im:
        m = np.array(im.convert("L"))
    return (m > 127).astype(np.uint8)


def crop_to_mask(img_rgb: np.ndarray, mask: np.ndarray, margin: float = 0.10) -> np.ndarray:
    """Recorta la imagen a la caja de la lesión con margen relativo (ROI ORÁCULO).

    Nota: usa la máscara ground-truth, que en despliegue real no existe. Es un límite
    superior/diagnóstico, no un pipeline desplegable.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img_rgb
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, x1 = max(0, x0 - mx), min(w, x1 + mx)
    y0, y1 = max(0, y0 - my), min(h, y1 + my)
    return img_rgb[y0:y1, x0:x1]


class UltrasoundDataset(Dataset):
    """Envuelve un DataFrame de índice para el DataLoader.

    __getitem__ devuelve ``(tensor, label_idx, row_index)``; el índice permite
    trazar cada predicción de vuelta a su fila (útil para Grad-CAM y análisis).
    """

    def __init__(self, df: pd.DataFrame, transform, roi_mode: str = "full",
                 roi_margin: float = 0.10):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.roi_mode = roi_mode
        self.roi_margin = roi_margin

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = load_rgb(row["image_path"])
        if self.roi_mode == "oracle_roi" and isinstance(row.get("mask_path"), str):
            try:
                img = crop_to_mask(img, load_mask(row["mask_path"]), self.roi_margin)
            except (FileNotFoundError, OSError):
                pass
        tensor = self.transform(img)
        return tensor, int(row["label_idx"]), i
