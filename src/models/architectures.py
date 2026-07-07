"""Network architectures.

Backbones come from ``timm`` (ImageNet-pretrained), so swapping ResNet-50 for EfficientNet-B0 is
a one-line config change. Two model types are provided:

* ``BaselineClassifier`` — backbone + classification head. Used for all baselines and fine-tuning.
* ``DANNModel`` — adds a domain classifier behind a Gradient Reversal Layer for adversarial
  domain adaptation (Ganin et al., 2016).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.autograd import Function

from ..config import Config


# ---------------------------------------------------------------------------
# Baseline classifier
# ---------------------------------------------------------------------------
class BaselineClassifier(nn.Module):
    def __init__(self, arch: str = "resnet50", num_classes: int = 2,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        # num_classes=0 -> backbone returns a pooled feature vector (no head)
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                           global_pool="avg")
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
        )
        self.arch = arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer + DANN
# ---------------------------------------------------------------------------
class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class DANNModel(nn.Module):
    """Domain-Adversarial Neural Network: shared backbone, label head, domain head."""

    def __init__(self, arch: str = "resnet50", num_classes: int = 2,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                           global_pool="avg")
        f = self.backbone.num_features
        self.num_features = f
        self.label_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(f, num_classes))
        self.domain_head = nn.Sequential(
            nn.Linear(f, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 2)
        )
        self.arch = arch

    def forward(self, x: torch.Tensor, lambda_: float = 0.0):
        feat = self.backbone(x)
        class_logits = self.label_head(feat)
        domain_logits = self.domain_head(grad_reverse(feat, lambda_))
        return class_logits, domain_logits


# ---------------------------------------------------------------------------
# Factory + Grad-CAM helper
# ---------------------------------------------------------------------------
def build_model(cfg: Config, dann: bool = False) -> nn.Module:
    m = cfg.model
    cls = DANNModel if dann else BaselineClassifier
    return cls(arch=m.architecture, num_classes=m.num_classes,
               pretrained=m.pretrained, dropout=m.dropout)


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer of the backbone for Grad-CAM.

    Works for the ResNet and EfficientNet families exposed by timm. For other backbones, the last
    Conv2d found by traversal is used as a fallback.
    """
    backbone = getattr(model, "backbone", model)
    arch = getattr(model, "arch", "")
    if "resnet" in arch:
        return backbone.layer4[-1]
    if "efficientnet" in arch:
        return backbone.conv_head if hasattr(backbone, "conv_head") else backbone.blocks[-1]
    last_conv = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No se encontró capa convolucional para Grad-CAM.")
    return last_conv
