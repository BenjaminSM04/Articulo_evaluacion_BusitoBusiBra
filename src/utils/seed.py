"""Reproducibilidad: fija todas las semillas y hace cuDNN determinista.

El determinismo importa porque el objetivo del estudio es comparar rendimiento entre
dominios; el ruido entre corridas no debe confundirse con un efecto de domain shift.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Fija las semillas de Python, NumPy y PyTorch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only evita fallos duros en operaciones sin variante determinista.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


# Alias retrocompatible con el código previo del proyecto.
seed_everything = set_seed


def seed_worker(worker_id: int) -> None:
    """Semilla por worker del DataLoader (usar como ``worker_init_fn``)."""
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
