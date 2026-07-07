"""Subpaquete de evaluación: métricas, calibración y reportes interno/externo."""
from .calibration import (  # noqa: F401
    brier_score,
    expected_calibration_error,
    fit_temperature,
    plot_reliability,
    reliability_curve,
)
from .metrics import (  # noqa: F401
    auc_metric,
    bootstrap_ci,
    compute_metrics,
    delong_roc_test,
    find_threshold_youden,
    pr_auc_metric,
    pr_points,
    roc_points,
    run_inference,
)
from .evaluate_external import evaluate_and_report, run_external_eval  # noqa: F401
from .evaluate_internal import run_internal_eval  # noqa: F401
