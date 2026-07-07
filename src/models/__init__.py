"""Subpaquete de modelos: backbones, clasificador binario y DANN."""
from .backbones import SUPPORTED, get_backbone  # noqa: F401
from .classifiers import BreastClassifier, build_classifier  # noqa: F401
from .dann import DANN, DomainDiscriminator, dann_lambda, grad_reverse  # noqa: F401
