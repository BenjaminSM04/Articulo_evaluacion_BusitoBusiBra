"""Controles puros de intensidad y región de interés para imágenes RGB."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np


def percentile_normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Normaliza canales RGB uint8 con los percentiles espaciales 1 y 99."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe tener forma HxWx3 (RGB).")
    if image.dtype != np.uint8:
        raise ValueError("La imagen RGB debe tener dtype uint8.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("La imagen RGB no puede estar vacía.")

    limits = np.percentile(image, (1, 99), axis=(0, 1))
    lower, upper = limits
    spans = upper - lower
    normalized = np.zeros(image.shape, dtype=np.uint8)
    valid_channels = spans > 0
    if np.any(valid_channels):
        scaled = (
            (image[:, :, valid_channels].astype(np.float64) - lower[valid_channels])
            / spans[valid_channels]
            * 255.0
        )
        normalized[:, :, valid_channels] = np.clip(scaled, 0, 255).astype(np.uint8)
    return normalized


def crop_union_roi(image: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    """Recorta la unión de máscaras 2D con margen del 10 por ciento por eje."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe tener forma HxWx3 (RGB).")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("La imagen RGB no puede estar vacía.")
    if masks is None or len(masks) == 0:
        raise ValueError("Se requiere al menos una máscara.")

    height, width = image.shape[:2]
    foreground = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError("Cada máscara debe ser un arreglo 2D alineado.")
        if mask.shape != (height, width):
            raise ValueError("Las dimensiones de cada máscara deben coincidir con la imagen.")
        foreground |= mask > 0

    coordinates = np.argwhere(foreground)
    if coordinates.size == 0:
        raise ValueError("La unión de máscaras está vacía.")

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    box_height = int(y_max - y_min + 1)
    box_width = int(x_max - x_min + 1)
    margin_y = ceil(0.1 * box_height)
    margin_x = ceil(0.1 * box_width)
    top = max(0, int(y_min) - margin_y)
    bottom = min(height, int(y_max) + 1 + margin_y)
    left = max(0, int(x_min) - margin_x)
    right = min(width, int(x_max) + 1 + margin_x)
    return image[top:bottom, left:right].copy()
