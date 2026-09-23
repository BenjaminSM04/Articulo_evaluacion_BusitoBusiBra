"""Albumentations transforms for training and evaluation.

Augmentation is deliberately conservative: ultrasound has a clinically meaningful orientation, so
we avoid vertical flips and aggressive crops that would alter anatomy. Normalization uses ImageNet
statistics to match the pretrained backbones.

``augmentation.gaussian_noise_var`` is expressed as pixel-intensity variance on the 0--255 scale,
matching the configuration used by the original experiments. Albumentations 2.x instead expects
``GaussNoise.std_range`` as a fraction of the maximum pixel value, so the conversion is
``sqrt(variance) / 255``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import Config


def gaussian_variance_to_std_range(
    variance_range: Sequence[float],
    max_pixel_value: float = 255.0,
) -> tuple[float, float]:
    """Convert pixel-space variances to Albumentations 2.x fractional standard deviations.

    Args:
        variance_range: Inclusive ``(min_variance, max_variance)`` in squared pixel-intensity
            units. For uint8 images, the largest meaningful variance is ``255**2``.
        max_pixel_value: Maximum representable image value before normalization.

    Raises:
        ValueError: If the range is malformed, non-finite, negative, reversed, or outside the
            representable pixel range.
    """
    if len(variance_range) != 2:
        raise ValueError("gaussian_noise_var debe contener exactamente dos valores.")

    low, high = (float(value) for value in variance_range)
    max_pixel_value = float(max_pixel_value)
    if not all(math.isfinite(value) for value in (low, high, max_pixel_value)):
        raise ValueError("gaussian_noise_var y max_pixel_value deben ser finitos.")
    if max_pixel_value <= 0:
        raise ValueError("max_pixel_value debe ser mayor que cero.")
    if low < 0 or high < low:
        raise ValueError("gaussian_noise_var debe cumplir 0 <= mínimo <= máximo.")
    if high > max_pixel_value**2:
        raise ValueError(
            "gaussian_noise_var excede la varianza representable por el rango de píxeles."
        )

    return math.sqrt(low) / max_pixel_value, math.sqrt(high) / max_pixel_value


def build_train_transforms(cfg: Config) -> A.Compose:
    p = cfg.preprocessing
    a = cfg.augmentation
    noise_std_range = gaussian_variance_to_std_range(a.gaussian_noise_var)
    return A.Compose(
        [
            A.Resize(p.image_size, p.image_size),
            A.HorizontalFlip(p=a.horizontal_flip),
            A.Rotate(limit=a.rotation_limit, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=a.brightness_contrast, contrast_limit=a.brightness_contrast, p=0.5
            ),
            A.GaussNoise(std_range=noise_std_range, mean_range=(0.0, 0.0), p=0.3),
            A.Normalize(mean=p.normalize_mean, std=p.normalize_std),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(cfg: Config) -> A.Compose:
    p = cfg.preprocessing
    return A.Compose(
        [
            A.Resize(p.image_size, p.image_size),
            A.Normalize(mean=p.normalize_mean, std=p.normalize_std),
            ToTensorV2(),
        ]
    )
