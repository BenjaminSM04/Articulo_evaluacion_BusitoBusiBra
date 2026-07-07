"""Subpaquete de entrenamiento: baseline, fine-tuning y adaptación de dominio."""
from .train_baseline import train_baseline, train_finetune  # noqa: F401
from .train_coral import coral_loss, mmd_loss, train_coral  # noqa: F401
from .train_dann import train_dann  # noqa: F401
from .trainer_utils import (  # noqa: F401
    build_loader,
    fit_classifier,
    make_internal_splits,
    make_target_splits,
    sample_few_shot,
    train_domain_adaptation,
)
