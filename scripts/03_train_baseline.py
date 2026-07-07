"""Paso 3 — Entrenar un baseline intra-dominio con validación cruzada.

Entrena ``n_folds`` modelos sobre un dataset, evalúa cada fold en su test y reporta media ± desv.
Guarda checkpoints por fold, curvas de entrenamiento (fold 0) y un JSON con métricas agregadas.

Uso:
    python scripts/03_train_baseline.py --dataset busi    --arch resnet50
    python scripts/03_train_baseline.py --dataset bus_bra --arch efficientnet_b0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.datasets import (  # noqa: E402
    build_dataloaders,
    class_weights,
    load_manifest,
    make_cv_splits,
)
from src.evaluation.metrics import (  # noqa: E402
    auc_metric,
    bootstrap_ci,
    compute_metrics,
    run_inference,
)
from src.models.architectures import build_model  # noqa: E402
from src.training.train import train_model  # noqa: E402
from src.training.transforms import build_eval_transforms, build_train_transforms  # noqa: E402
from src.utils.reporting import plot_training_curves, save_json  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento baseline (CV)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dataset", required=True, choices=["busi", "bus_bra"])
    ap.add_argument("--arch", default=None, help="Sobrescribe model.architecture")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg.model["architecture"] = args.arch
    arch = cfg.model.architecture
    seed_everything(cfg.seed)

    df = load_manifest(cfg, args.dataset)
    root = cfg._root
    train_tf, eval_tf = build_train_transforms(cfg), build_eval_transforms(cfg)
    folds = make_cv_splits(df, cfg)

    fold_metrics = []
    for k, split in enumerate(folds):
        print(f"\n========== {args.dataset} | {arch} | fold {k+1}/{len(folds)} ==========")
        loaders = build_dataloaders(df, root, train_tf, eval_tf, cfg, split)
        model = build_model(cfg)
        ckpt = cfg.path("models") / f"baseline_{args.dataset}_{arch}_fold{k}.pt"
        out = train_model(model, loaders, cfg, class_weights(df.iloc[split["train_idx"]]),
                          device=cfg.device, ckpt_path=ckpt)
        if k == 0:
            plot_training_curves(out["history"],
                                 cfg.path("figures") / f"curves_{args.dataset}_{arch}.png")
        y_true, y_prob, _ = run_inference(out["model"], loaders["test"], cfg.device,
                                          cfg.training.mixed_precision)
        m = compute_metrics(y_true, y_prob)
        m["auc_ci"] = list(bootstrap_ci(y_true, y_prob, auc_metric,
                                        cfg.evaluation.n_bootstrap, cfg.evaluation.ci_level,
                                        cfg.seed))
        fold_metrics.append(m)
        print(f"  fold {k+1}: AUC={m['auc']:.3f}  acc={m['accuracy']:.3f}  "
              f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}")

    # Aggregate mean ± std across folds
    keys = ["accuracy", "sensitivity", "specificity", "auc", "f1", "balanced_accuracy"]
    summary = {k: {"mean": float(np.nanmean([m[k] for m in fold_metrics])),
                   "std": float(np.nanstd([m[k] for m in fold_metrics]))} for k in keys}
    result = {"dataset": args.dataset, "arch": arch, "n_folds": len(folds),
              "summary": summary, "folds": fold_metrics}
    out_path = cfg.path("reports") / f"baseline_{args.dataset}_{arch}.json"
    save_json(result, out_path)
    cfg.dump(cfg.path("reports") / f"config_used_baseline_{args.dataset}_{arch}.yaml")

    print(f"\n=== RESUMEN {args.dataset} ({arch}) ===")
    for k in keys:
        print(f"  {k:18s}: {summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}")
    print(f"[ok] -> {out_path}")


if __name__ == "__main__":
    main()
