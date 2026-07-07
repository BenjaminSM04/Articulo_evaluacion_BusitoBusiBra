"""Transforms de imagen: resize, normalización ImageNet, augmentation y opciones
de preprocesamiento que también funcionan como ablaciones de dominio (CLAHE y
histogram matching).

La ecografía es monocroma: se replica a 3 canales para usar backbones ImageNet.
El augmentation se aplica SOLO en entrenamiento; se evita el volteo vertical y los
recortes agresivos porque alteran la semántica clínica de la imagen.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

try:                              # OpenCV es opcional (solo para CLAHE)
    import cv2
except ImportError:               # pragma: no cover
    cv2 = None

try:                              # scikit-image es opcional (histogram matching)
    from skimage.exposure import match_histograms
except ImportError:               # pragma: no cover
    match_histograms = None


class AddGaussianNoise:
    """Ruido gaussiano leve sobre el tensor (aug. realista para ecografía)."""

    def __init__(self, std: float = 0.02):
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE sobre el canal de luminancia. Requiere OpenCV."""
    if cv2 is None:
        raise ImportError("CLAHE requiere opencv-python. Instálalo o pon clahe: false.")
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def match_histogram(img_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """Iguala el histograma de la imagen al de una referencia (del SOURCE).

    Advertencia metodológica: el histogram matching target->source ES una forma de
    adaptación de dominio. Úsese como ABLATION explícita, nunca oculto en el baseline.
    """
    if match_histograms is None:
        raise ImportError("histogram_matching requiere scikit-image.")
    matched = match_histograms(img_rgb, reference_rgb, channel_axis=-1)
    return matched.astype(np.uint8)


class UltrasoundTransform:
    """Pipeline configurable. Entrada: imagen RGB uint8 (H,W,3). Salida: tensor (3,H,W).

    Args:
        cfg: configuración completa (usa cfg['data'] y cfg['augmentation']).
        train: si True aplica augmentation.
        reference: imagen de referencia (RGB uint8) para histogram matching.
    """

    def __init__(self, cfg: dict, train: bool, reference: np.ndarray | None = None):
        self.cfg = cfg
        self.train = train
        self.reference = reference
        size = int(cfg["data"]["image_size"])

        aug: list = []
        if train:
            a = cfg["augmentation"]
            aug = [
                T.RandomHorizontalFlip(p=a["horizontal_flip"]),
                T.RandomRotation(degrees=a["rotation_deg"]),
                T.ColorJitter(brightness=a["brightness_contrast"],
                              contrast=a["brightness_contrast"]),
            ]
        self.pil_pipeline = T.Compose([T.Resize((size, size)), *aug, T.ToTensor()])
        self.noise = AddGaussianNoise(cfg["augmentation"]["gaussian_noise_std"] if train else 0.0)
        self.normalize = T.Normalize(cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"])

    def __call__(self, img_rgb: np.ndarray) -> torch.Tensor:
        img = img_rgb
        if self.cfg["data"].get("clahe"):
            img = apply_clahe(img)
        if self.cfg["data"].get("histogram_matching") and self.reference is not None:
            img = match_histogram(img, self.reference)
        tensor = self.pil_pipeline(Image.fromarray(img))
        tensor = self.noise(tensor)
        return self.normalize(tensor)


def build_transforms(cfg: dict, train: bool, reference: np.ndarray | None = None) -> UltrasoundTransform:
    """Fábrica de transforms (azúcar sintáctico)."""
    return UltrasoundTransform(cfg, train=train, reference=reference)
