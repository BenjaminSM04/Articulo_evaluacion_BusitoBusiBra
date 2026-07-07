"""Albumentations transforms for training and evaluation.

Augmentation is deliberately conservative: ultrasound has a clinically meaningful orientation, so
we avoid vertical flips and aggressive crops that would alter anatomy. Normalization uses ImageNet
statistics to match the pretrained backbones.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import Config


def build_train_transforms(cfg: Config) -> A.Compose:
    p = cfg.preprocessing
    a = cfg.augmentation
    return A.Compose([
        A.Resize(p.image_size, p.image_size),
        A.HorizontalFlip(p=a.horizontal_flip),
        A.Rotate(limit=a.rotation_limit, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=a.brightness_contrast,
                                   contrast_limit=a.brightness_contrast, p=0.5),
        A.GaussNoise(var_limit=tuple(a.gaussian_noise_var), p=0.3),
        A.Normalize(mean=p.normalize_mean, std=p.normalize_std),
        ToTensorV2(),
    ])


def build_eval_transforms(cfg: Config) -> A.Compose:
    p = cfg.preprocessing
    return A.Compose([
        A.Resize(p.image_size, p.image_size),
        A.Normalize(mean=p.normalize_mean, std=p.normalize_std),
        ToTensorV2(),
    ])
