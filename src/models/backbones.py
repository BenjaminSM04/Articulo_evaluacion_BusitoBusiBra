"""Backbones preentrenados en ImageNet, adaptados como extractores de características.

Se devuelve el backbone SIN su cabeza de clasificación (produce el vector de features
tras el global pooling) junto con la dimensión de features y un getter de la última
capa convolucional (necesaria para Grad-CAM).
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "efficientnet_b0", "densenet121")


def _build(name_fn, weights_enum, pretrained: bool):
    """Carga con la API nueva de torchvision (weights=) y cae a la antigua (pretrained=)."""
    try:
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        return name_fn(weights=weights)
    except Exception:  # torchvision antiguo
        return name_fn(pretrained=pretrained)


def get_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int, Callable]:
    """Devuelve (backbone, feat_dim, target_layer_getter).

    backbone(x) -> tensor (B, feat_dim) tras global average pooling.
    target_layer_getter(backbone) -> última capa conv (para Grad-CAM).
    """
    name = name.lower()
    if name == "resnet18":
        net = _build(models.resnet18, getattr(models, "ResNet18_Weights", None), pretrained)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim, (lambda m: m.layer4[-1])

    if name == "efficientnet_b0":
        net = _build(models.efficientnet_b0, getattr(models, "EfficientNet_B0_Weights", None),
                     pretrained)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim, (lambda m: m.features[-1])

    if name == "densenet121":
        net = _build(models.densenet121, getattr(models, "DenseNet121_Weights", None), pretrained)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        # densenet.forward NO incluye pooling con classifier=Identity; se añade abajo.
        net = _DenseNetWrapper(net)
        return net, feat_dim, (lambda m: m.net.features.norm5)

    raise ValueError(f"Backbone no soportado: {name!r}. Usa uno de {SUPPORTED}.")


class _DenseNetWrapper(nn.Module):
    """DenseNet con classifier=Identity requiere ReLU + global pooling explícitos."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        import torch.nn.functional as F
        feats = self.net.features(x)
        out = F.relu(feats, inplace=True)
        out = self.pool(out).flatten(1)
        return out
