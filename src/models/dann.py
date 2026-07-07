"""DANN: Domain-Adversarial Neural Network (Ganin et al., 2016).

Un extractor de características compartido alimenta (a) la cabeza de tarea
(benigno/maligno) y (b) una cabeza de dominio conectada por una capa de inversión
de gradiente (GRL). Durante el retropropagado el gradiente de la cabeza de dominio
se multiplica por -lambda, empujando al extractor a producir características
indistinguibles entre dominios (source=BUSI, target=BUS-BRA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from .classifiers import build_classifier


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Capa de inversión de gradiente (GRL)."""
    return _GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Clasifica el dominio (source vs target) a partir de las características."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class DANN(nn.Module):
    """Envuelve un BreastClassifier y añade la cabeza de dominio con GRL."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.classifier = build_classifier(cfg)
        self.domain_head = DomainDiscriminator(self.classifier.feat_dim)

    def forward(self, x, lambd: float = 0.0):
        logits, feat = self.classifier(x, return_features=True)
        domain_logits = self.domain_head(grad_reverse(feat, lambd))
        return logits, domain_logits

    def features(self, x) -> torch.Tensor:
        return self.classifier.features(x)

    def get_target_layer(self) -> nn.Module:
        return self.classifier.get_target_layer()


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Programación creciente de lambda: 2/(1+exp(-gamma*p)) - 1  (Ganin et al., 2016)."""
    import math
    p = step / max(1, total_steps)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
