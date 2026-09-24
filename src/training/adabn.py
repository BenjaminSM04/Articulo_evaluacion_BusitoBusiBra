"""Label-blind BatchNorm recalibration using the frozen target adaptation pool."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import torch
from torch import nn


def order_target_adapt(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by stable sample identity and remove labels from the adaptation pass."""

    if "sample_id" not in frame.columns:
        raise KeyError("target_adapt requiere sample_id")
    ids = frame["sample_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("target_adapt contiene sample_id vacio o duplicado")
    ordered = frame.copy()
    ordered["sample_id"] = ids
    ordered = ordered.sort_values("sample_id", kind="stable").reset_index(drop=True)
    ordered["label_idx"] = -1
    ordered["label"] = "unlabelled"
    return ordered


def refresh_batchnorm_statistics(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, ...]],
    *,
    device: str | torch.device,
) -> int:
    """Reset BN and accumulate one ordered pass with dropout disabled and no gradients.

    The caller must provide a non-shuffled loader of ``order_target_adapt`` with
    batch size 16. The final partial batch is retained.
    """

    batchnorms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batchnorms:
        raise ValueError("AdaBN requiere al menos una capa BatchNorm")

    model.eval()
    for module in batchnorms:
        if not module.track_running_stats:
            raise ValueError("AdaBN requiere BatchNorm con running stats")
        module.reset_running_stats()
        module.momentum = None
        module.train()

    seen = 0
    try:
        with torch.no_grad():
            for batch in batches:
                images = batch[0].to(device)
                if images.shape[0] == 0:
                    continue
                model(images)
                seen += int(images.shape[0])
    finally:
        model.eval()
    if seen == 0:
        raise ValueError("El loader de AdaBN esta vacio")
    return seen
