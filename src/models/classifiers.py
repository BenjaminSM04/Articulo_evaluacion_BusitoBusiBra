"""Clasificador binario benigno/maligno = backbone + cabeza (dropout + lineal)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import get_backbone


class BreastClassifier(nn.Module):
    """Extractor de características + cabeza de clasificación a 2 clases."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone, self.feat_dim, self._target_getter = get_backbone(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, num_classes))

    def forward(self, x, return_features: bool = False):
        feat = self.backbone(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits

    def features(self, x) -> torch.Tensor:
        """Vector de características tras el global pooling (para CORAL/MMD)."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Última capa convolucional (para Grad-CAM)."""
        return self._target_getter(self.backbone)


def build_classifier(cfg: dict) -> BreastClassifier:
    """Construye el clasificador desde la configuración."""
    m = cfg["model"]
    return BreastClassifier(
        backbone=m["backbone"],
        pretrained=m.get("pretrained", True),
        num_classes=m.get("num_classes", 2),
        dropout=m.get("dropout", 0.3),
    )
