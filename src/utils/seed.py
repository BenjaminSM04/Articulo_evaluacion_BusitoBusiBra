"""Reproducibilidad: fija semillas y solicita algoritmos deterministas.

El determinismo importa porque el objetivo del estudio es comparar rendimiento entre
dominios; el ruido entre corridas no debe confundirse con un efecto de domain shift.

Limitaciones:

* ``CUBLAS_WORKSPACE_CONFIG`` debe establecerse antes de crear el primer contexto CUDA. Esta
  función lo configura antes de importar/sembrar PyTorch y avisa si CUDA ya estaba inicializado.
* ``PYTHONHASHSEED`` solo controla plenamente el hash de un intérprete iniciado con esa variable;
  asignarlo aquí sí se propaga a procesos hijos, pero no reconstruye el hash del proceso actual.
* ``warn_only=True`` permite continuar si PyTorch no ofrece una implementación determinista.
  Reproducibilidad bit a bit también depende de hardware, drivers, versiones y operaciones de
  terceros.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Fija las semillas de Python, NumPy y PyTorch (CPU + CUDA).

    Debe llamarse al comienzo del proceso, antes de construir modelos o mover tensores a CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    previous_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic:
        # PyTorch/CuBLAS reads this setting when the first CUDA context is initialized.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        cuda_was_initialized = torch.cuda.is_initialized()
        if (
            deterministic
            and cuda_was_initialized
            and previous_cublas_config != CUBLAS_WORKSPACE_CONFIG
        ):
            warnings.warn(
                "CUDA ya estaba inicializado antes de fijar CUBLAS_WORKSPACE_CONFIG; "
                "reinicia el proceso para garantizar que CuBLAS use la configuración determinista.",
                RuntimeWarning,
                stacklevel=2,
            )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # No oculta operaciones sin variante determinista: PyTorch emitirá una advertencia.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


# Alias retrocompatible con el código previo del proyecto.
seed_everything = set_seed


def seed_worker(worker_id: int) -> None:
    """Semilla por worker del DataLoader (usar como ``worker_init_fn``)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    transform = getattr(worker_info.dataset, "transform", None) if worker_info else None
    if callable(getattr(transform, "set_random_seed", None)):
        # Albumentations 2 does not consume NumPy's global RNG.
        transform.set_random_seed(int(worker_seed))
