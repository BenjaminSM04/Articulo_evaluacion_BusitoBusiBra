"""Carga y validación de configuración por YAML.

Todos los scripts leen un único YAML de ``configs/``. Los campos ausentes se
completan con ``DEFAULT_CONFIG`` mediante *deep-merge*, de modo que cada YAML solo
necesita declarar lo específico del experimento.
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Valores por defecto (única fuente de verdad del esquema de configuración).
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "baseline",      # baseline | external_eval | dann | coral | mmd | gradcam
    "seed": 42,
    "device": "cuda",              # cuda | cpu (si no hay GPU se degrada a cpu)

    "data": {
        "busi_root": "data/BUSI",
        "busbra_root": "data/BUS-BRA",
        "busbra_csv": None,        # None = autodetectar el CSV de metadatos
        "image_size": 224,
        "classes": ["benign", "malignant"],   # 'normal' se EXCLUYE del estudio
        "positive_class": "malignant",
        "to_rgb": True,            # ecografía monocroma -> replicar a 3 canales
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "busi_dedup": True,        # deduplicación pHash de BUSI (evita fuga de datos)
        "dedup_hamming": 5,
        "roi_mode": "full",        # full | oracle_roi (recorte con máscara GT = oráculo)
        "roi_margin": 0.10,
        "clahe": False,
        "histogram_matching": False,
    },

    "augmentation": {
        "horizontal_flip": 0.5,
        "rotation_deg": 15,
        "brightness_contrast": 0.2,
        "gaussian_noise_std": 0.02,
    },

    "split": {
        "internal_val_fraction": 0.15,
        "internal_test_fraction": 0.15,
        "n_folds": 0,              # 0 = split único; >1 = validación cruzada
        "group_by_patient": "auto",  # auto | true | false
        "target_test_fraction": 0.40,  # test target reservado (por paciente)
        "finetune_label_fraction": 0.0,  # 0.0 | 0.05 | 0.10 | 0.20
    },

    "model": {
        "backbone": "resnet18",    # resnet18 | efficientnet_b0 | densenet121
        "pretrained": True,
        "dropout": 0.3,
        "num_classes": 2,
    },

    "train": {
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",      # adamw | sgd
        "scheduler": "cosine",     # cosine | plateau | none
        "warmup_epochs": 3,
        "early_stopping_metric": "auc",
        "early_stopping_patience": 10,
        "class_weights": "balanced",  # balanced | none
        "mixed_precision": True,
        "num_workers": 4,
        "seeds": [42],             # multi-semilla: media +/- sd
    },

    "domain_adaptation": {
        "method": "none",          # none | dann | coral | mmd
        "lambda_grl": 1.0,
        "lambda_schedule": True,
        "lambda_gamma": 10.0,
        "align_weight": 1.0,
        "align_warmup_epochs": 5,
        "mmd_kernel": "rbf",
        "mmd_sigmas": [1, 2, 4, 8, 16],
    },

    "eval": {
        "threshold": 0.5,
        "threshold_strategy": "fixed",  # fixed | youden_source
        "bootstrap_ci": True,
        "n_bootstrap": 1000,
        "ci_level": 0.95,
    },

    "explainability": {
        "method": "gradcam",       # gradcam | gradcampp
        "eval_domain": "both",     # busi | busbra | both
        "n_samples": 24,
        "cam_threshold": 0.5,
        "target_layer": "auto",
        "save_overlays": True,
    },

    "checkpoint": None,            # ruta a un modelo entrenado (eval / gradcam / init)
    "output": {"results_dir": "results/experiment"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona ``override`` sobre ``base`` de forma recursiva (devuelve copia)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str) -> dict:
    """Lee un YAML y lo fusiona con ``DEFAULT_CONFIG``.

    Además resuelve el dispositivo: si se pide ``cuda`` pero no hay GPU, degrada a ``cpu``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)

    # Degradar a CPU si no hay GPU disponible.
    if cfg.get("device") == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                cfg["device"] = "cpu"
        except ImportError:
            cfg["device"] = "cpu"

    # results_dir por defecto según el nombre del experimento.
    if cfg["output"].get("results_dir") in (None, "results/experiment"):
        cfg["output"]["results_dir"] = f"results/{cfg['experiment']}"
    return cfg


def save_config(cfg: dict, path: str) -> None:
    """Guarda la configuración efectiva (para trazabilidad/reproducibilidad)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Acceso por clave con puntos, p. ej. ``get(cfg, 'train.lr', 1e-4)``."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_group_by_patient(cfg: dict, dataset_name: str) -> bool:
    """Resuelve la política 'auto': BUS-BRA sí (tiene ID de paciente); BUSI no."""
    policy = cfg["split"].get("group_by_patient", "auto")
    if policy == "auto":
        return dataset_name.lower() in ("busbra", "bus-bra", "bus_bra")
    return bool(policy)
