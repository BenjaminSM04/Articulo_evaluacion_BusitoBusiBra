"""Paquete del proyecto: generalización entre poblaciones en ecografía mamaria.

Estructura:
    src.datasets       Carga de BUSI / BUS-BRA, transforms y splits por paciente.
    src.models         Backbones (ResNet-18, EfficientNet-B0, DenseNet-121), DANN.
    src.training       Bucles de entrenamiento (baseline, DANN, CORAL/MMD).
    src.evaluation     Métricas, calibración y evaluación interna/externa.
    src.explainability Grad-CAM y análisis de localización de la atención.
    src.utils          Semillas, logging y configuración.
    src.main           Punto de entrada CLI (dispatcher por experimento).
"""

__version__ = "1.0.0"
