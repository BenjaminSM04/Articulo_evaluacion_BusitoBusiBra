"""Subpaquete de datos: índices de BUSI/BUS-BRA, Dataset común y transforms."""
from .base import (  # noqa: F401
    INDEX_COLUMNS,
    LABEL_MAP,
    UltrasoundDataset,
    crop_to_mask,
    load_mask,
    load_rgb,
)
from .busbra_dataset import build_busbra_index  # noqa: F401
from .busi_dataset import build_busi_index  # noqa: F401
from .transforms import UltrasoundTransform, build_transforms  # noqa: F401


def build_index(name: str, cfg: dict, logger=None):
    """Devuelve el DataFrame de índice del dataset indicado ('busi' | 'busbra')."""
    key = name.lower().replace("-", "").replace("_", "")
    if key == "busi":
        return build_busi_index(cfg, logger)
    if key == "busbra":
        return build_busbra_index(cfg, logger)
    raise ValueError(f"Dataset desconocido: {name!r} (usa 'busi' o 'busbra').")
