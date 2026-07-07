"""Cuantificación de la localización de Grad-CAM contra la máscara de lesión.

Responde a la pregunta central del estudio: ¿el modelo mira la LESIÓN o mira
artefactos (texto, calipers, sombras)? Y ¿se mantiene el foco al cambiar de población?

Métricas (cam y mask deben tener el mismo tamaño HxW):
  * energy_in_mask : fracción de la energía del mapa que cae dentro de la máscara.
  * pointing_game  : 1 si el píxel de máxima activación cae dentro de la máscara.
  * iou_threshold  : IoU entre el CAM umbralizado y la máscara.
"""
from __future__ import annotations

import numpy as np


def energy_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    total = float(cam.sum())
    if total <= 0:
        return float("nan")
    return float(cam[mask > 0].sum() / total)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    idx = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return float(mask[idx] > 0)


def iou_threshold(cam: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    cam_bin = (cam >= thr).astype(np.uint8)
    m = (mask > 0).astype(np.uint8)
    inter = int((cam_bin & m).sum())
    union = int((cam_bin | m).sum())
    return float(inter / union) if union > 0 else float("nan")


def summarize(records: list[dict]) -> dict:
    """Agrega una lista de dicts con claves energy/pointing/iou (media, ignora NaN)."""
    def _mean(key):
        vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": len(records),
        "energy_in_mask_mean": _mean("energy"),
        "pointing_game_rate": _mean("pointing"),
        "iou_mean": _mean("iou"),
    }
