"""Subpaquete de explicabilidad: Grad-CAM/Grad-CAM++ y localización de la lesión."""
from .gradcam import GradCAM, run_gradcam  # noqa: F401
from .lesion_attention_analysis import (  # noqa: F401
    energy_in_mask,
    iou_threshold,
    pointing_game,
    summarize,
)
from .shap_explain import run_shap  # noqa: F401
