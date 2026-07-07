"""Run the full BUSI -> BUS-BRA article experiment suite.

The script is deliberately conservative: every experiment is wrapped in a status/log entry so one
failure is documented and the remaining experiments can continue.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.data.datasets import (  # noqa: E402
    build_dataloaders,
    class_weights,
    load_manifest,
    make_cv_splits,
    make_target_adaptation_split,
)
from src.data.preprocessing import preprocess_all  # noqa: E402
from src.evaluation.artifacts import (  # noqa: E402
    ensure_article_dirs,
    evaluate_model_to_artifacts,
    flatten_record,
    load_metric_records,
    safe_id,
)
from src.explainability import run_gradcam  # noqa: E402
from src.models.architectures import build_model  # noqa: E402
from src.training.domain_adaptation import (  # noqa: E402
    dann_train,
    feature_alignment_train,
    finetune_on_target_fraction,
    self_training,
)
from src.training.train import train_model  # noqa: E402
from src.training.transforms import build_eval_transforms, build_train_transforms  # noqa: E402
from src.utils.article_report import (  # noqa: E402
    compute_drop_table,
    compute_recovery_table,
    write_article_reports,
    write_environment_report,
)  # noqa: E402
from src.utils.reporting import plot_training_curves, save_json  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_status(status_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(path, index=False)


def _load_status_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = pd.read_csv(path).fillna("").to_dict("records")
    ended_at = _now()
    for row in rows:
        if row.get("status") == "running":
            row["status"] = "failed"
            row["ended_at"] = row.get("ended_at") or ended_at
            row["error"] = row.get("error") or "interrupted_before_resume"
    return rows


def _has_success(status_rows: list[dict[str, Any]], name: str) -> bool:
    experiment_id = safe_id(name)
    return any(
        row.get("experiment_id") == experiment_id and row.get("status") == "success"
        for row in status_rows
    )


def _run_step(
    name: str,
    phase: str,
    dirs: dict[str, Path],
    status_rows: list[dict[str, Any]],
    fn: Callable[[], Any],
    resume: bool = False,
    **metadata,
) -> Any | None:
    if resume and _has_success(status_rows, name):
        print(f"[skip] {name} (success previo)")
        return None

    started = _now()
    log_path = dirs["logs"] / f"{safe_id(name)}.log"
    row = {
        "experiment_id": safe_id(name),
        "phase": phase,
        "status": "running",
        "started_at": started,
        "ended_at": "",
        "log_path": str(log_path),
        "error": "",
        **metadata,
    }
    status_rows.append(row)
    _write_status(status_rows, dirs["metrics"] / "experiment_status.csv")
    print(f"[start] {name}")
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
                result = fn()
        row["status"] = "success"
        print(f"[ok] {name}")
        return result
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        with open(log_path, "a", encoding="utf-8") as log_fh:
            traceback.print_exc(file=log_fh)
        print(f"[failed] {name}: {exc}")
        return None
    finally:
        row["ended_at"] = _now()
        _write_status(status_rows, dirs["metrics"] / "experiment_status.csv")


def _needs_preprocess(cfg: Config, keys: list[str]) -> bool:
    required = {"mask_path", "bbox"}
    for key in keys:
        path = cfg.path("data_processed") / f"{key}_manifest.csv"
        if not path.exists():
            return True
        cols = set(pd.read_csv(path, nrows=1).columns)
        if not required.issubset(cols):
            return True
    return False


def _dataset_summary(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in manifests.items():
        counts = df["label"].value_counts().to_dict()
        patient = df.get("patient_id", pd.Series([""] * len(df))).astype(str)
        rows.append({
            "dataset": key,
            "n_images": int(len(df)),
            "benign": int(counts.get("benign", 0)),
            "malignant": int(counts.get("malignant", 0)),
            "normal": int(counts.get("normal", 0)),
            "n_patients": int(patient[patient.str.len() > 0].nunique()) if len(df) else 0,
            "with_masks": int(
                df.get("mask_path", pd.Series([""] * len(df))).astype(str).str.len().gt(0).sum()
            ),
            "with_bbox": int(
                df.get("bbox", pd.Series([""] * len(df))).astype(str).str.len().gt(0).sum()
            ),
        })
    return pd.DataFrame(rows)


def _load_model_from_checkpoint(cfg: Config, arch: str, ckpt_path: Path):
    cfg.model["architecture"] = arch
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state, strict=False)
    model.to(cfg.device).eval()
    return model


def _checkpoint_candidates(
    cfg: Config,
    dirs: dict[str, Path],
    dataset: str,
    arch: str,
    fold: int,
) -> list[Path]:
    name = f"baseline_{dataset}_{arch}_fold{fold}.pt"
    return [
        dirs["checkpoints"] / name,
        cfg.path("models") / name,
    ]


def _find_valid_checkpoint(
    cfg: Config,
    dirs: dict[str, Path],
    dataset: str,
    arch: str,
    fold: int,
) -> Path | None:
    for candidate in _checkpoint_candidates(cfg, dirs, dataset, arch, fold):
        if not candidate.exists():
            continue
        try:
            state = torch.load(candidate, map_location="cpu")
            if isinstance(state, dict) and state.get("arch", arch) not in {arch, None}:
                continue
            if "model_state" not in state and not all(isinstance(k, str) for k in state.keys()):
                continue
            return candidate
        except Exception:
            continue
    return None


def _persist_checkpoint_copy(src: Path, dst: Path) -> Path:
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        state = torch.load(src, map_location="cpu")
        torch.save(state, dst)
    return dst


def _write_all_metrics(records: list[dict[str, Any]], dirs: dict[str, Path]) -> pd.DataFrame:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        flat = flatten_record(record)
        experiment_id = str(flat.get("experiment_id", len(deduped)))
        deduped[experiment_id] = flat
    df = pd.DataFrame(deduped.values())
    path = dirs["metrics"] / "all_metrics.csv"
    df.to_csv(path, index=False)
    return df


def train_or_reuse_baseline(
    cfg: Config,
    dirs: dict[str, Path],
    dataset: str,
    arch: str,
    resume: bool,
) -> list[dict[str, Any]]:
    cfg.model["architecture"] = arch
    df = load_manifest(cfg, dataset)
    root = cfg._root
    train_tf, eval_tf = build_train_transforms(cfg), build_eval_transforms(cfg)
    folds = make_cv_splits(df, cfg)
    records: list[dict[str, Any]] = []

    for k, split in enumerate(folds):
        exp_id = f"baseline_internal_{dataset}_{arch}_fold{k}"
        article_ckpt = dirs["checkpoints"] / f"baseline_{dataset}_{arch}_fold{k}.pt"
        ckpt = _find_valid_checkpoint(cfg, dirs, dataset, arch, k) if resume else None
        if ckpt is not None:
            ckpt = _persist_checkpoint_copy(ckpt, article_ckpt)
            model = _load_model_from_checkpoint(cfg, arch, ckpt)
        else:
            loaders = build_dataloaders(df, root, train_tf, eval_tf, cfg, split)
            model = build_model(cfg)
            out = train_model(
                model,
                loaders,
                cfg,
                class_weights(df.iloc[split["train_idx"]]),
                device=cfg.device,
                ckpt_path=article_ckpt,
            )
            model = out["model"]
            if k == 0:
                plot_training_curves(
                    out["history"],
                    dirs["figures"] / f"training_curves_{dataset}_{arch}.png",
                )
            ckpt = article_ckpt

        rec = evaluate_model_to_artifacts(
            model,
            df.iloc[split["test_idx"]],
            root,
            cfg,
            exp_id,
            dirs,
            metadata={
                "phase": "baseline_internal",
                "dataset": dataset,
                "arch": arch,
                "fold": k,
                "checkpoint_path": str(ckpt),
            },
        )
        records.append(rec)

    summary_path = dirs["metrics"] / f"baseline_{dataset}_{arch}.json"
    save_json({"dataset": dataset, "arch": arch, "folds": records}, summary_path)
    return records


def evaluate_external(
    cfg: Config,
    dirs: dict[str, Path],
    source: str,
    target: str,
    arch: str,
    target_df: pd.DataFrame,
    target_test_idx: np.ndarray,
    resume: bool,
) -> list[dict[str, Any]]:
    cfg.model["architecture"] = arch
    records: list[dict[str, Any]] = []
    folds = make_cv_splits(load_manifest(cfg, source), cfg)
    for k, _ in enumerate(folds):
        ckpt = _find_valid_checkpoint(cfg, dirs, source, arch, k) if resume else None
        if ckpt is None:
            ckpt = dirs["checkpoints"] / f"baseline_{source}_{arch}_fold{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Falta checkpoint source para evaluación externa: {ckpt}")
        model = _load_model_from_checkpoint(cfg, arch, ckpt)
        for phase, subset in (
            ("external_all", target_df),
            ("external_test", target_df.iloc[target_test_idx]),
        ):
            exp_id = f"{phase}_{source}_to_{target}_{arch}_fold{k}"
            rec = evaluate_model_to_artifacts(
                model,
                subset,
                cfg._root,
                cfg,
                exp_id,
                dirs,
                metadata={
                    "phase": phase,
                    "source": source,
                    "target": target,
                    "arch": arch,
                    "fold": k,
                    "method": "none",
                    "checkpoint_path": str(ckpt),
                },
            )
            records.append(rec)
    return records


def _load_source_fold0(cfg: Config, dirs: dict[str, Path], source: str, arch: str):
    ckpt = _find_valid_checkpoint(cfg, dirs, source, arch, 0)
    if ckpt is None:
        ckpt = dirs["checkpoints"] / f"baseline_{source}_{arch}_fold0.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Falta checkpoint fold0 para {source}/{arch}: {ckpt}")
    return _load_model_from_checkpoint(cfg, arch, ckpt), ckpt


def run_adaptation(
    cfg: Config,
    dirs: dict[str, Path],
    source: str,
    target: str,
    arch: str,
    method: str,
    source_df: pd.DataFrame,
    target_adapt_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
) -> dict[str, Any]:
    cfg.model["architecture"] = arch
    root = cfg._root
    if method == "none":
        model, source_ckpt = _load_source_fold0(cfg, dirs, source, arch)
        ckpt = source_ckpt
    elif method == "dann":
        model = dann_train(source_df, target_adapt_df, root, cfg)
        ckpt = dirs["checkpoints"] / f"adapt_{source}_to_{target}_{arch}_{method}.pt"
        torch.save({"model_state": model.state_dict(), "arch": arch, "method": method}, ckpt)
    elif method in {"coral", "mmd"}:
        model = feature_alignment_train(source_df, target_adapt_df, root, cfg, method=method)
        ckpt = dirs["checkpoints"] / f"adapt_{source}_to_{target}_{arch}_{method}.pt"
        torch.save({"model_state": model.state_dict(), "arch": arch, "method": method}, ckpt)
    elif method == "self_training":
        model, _ = _load_source_fold0(cfg, dirs, source, arch)
        model = self_training(model, source_df, target_adapt_df, root, cfg)
        ckpt = dirs["checkpoints"] / f"adapt_{source}_to_{target}_{arch}_{method}.pt"
        torch.save({"model_state": model.state_dict(), "arch": arch, "method": method}, ckpt)
    else:
        raise ValueError(f"Método de adaptación no soportado: {method}")

    return evaluate_model_to_artifacts(
        model,
        target_test_df,
        root,
        cfg,
        f"adaptation_{source}_to_{target}_{arch}_{method}",
        dirs,
        metadata={
            "phase": "adaptation",
            "source": source,
            "target": target,
            "arch": arch,
            "method": method,
            "checkpoint_path": str(ckpt),
        },
    )


def run_finetuning(
    cfg: Config,
    dirs: dict[str, Path],
    source: str,
    target: str,
    arch: str,
    fraction: float,
    target_adapt_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
) -> dict[str, Any]:
    cfg.model["architecture"] = arch
    model, _ = _load_source_fold0(cfg, dirs, source, arch)
    model, used = finetune_on_target_fraction(model, target_adapt_df, cfg._root, cfg, fraction)
    method = f"finetune_{int(round(fraction * 100))}pct"
    ckpt = dirs["checkpoints"] / f"{source}_to_{target}_{arch}_{method}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "arch": arch,
        "method": method,
        "target_fraction": fraction,
        "target_used_indices": used.tolist(),
    }, ckpt)
    return evaluate_model_to_artifacts(
        model,
        target_test_df,
        cfg._root,
        cfg,
        f"finetuning_{source}_to_{target}_{arch}_{method}",
        dirs,
        metadata={
            "phase": "finetuning",
            "source": source,
            "target": target,
            "arch": arch,
            "method": method,
            "target_fraction": fraction,
            "target_labeled_n": int(len(used)),
            "checkpoint_path": str(ckpt),
        },
    )


def _best_record(
    records: list[dict[str, Any]],
    phases: set[str],
    exclude_method: set[str] | None = None,
) -> dict[str, Any] | None:
    exclude_method = exclude_method or set()
    candidates = [
        r for r in records
        if r.get("phase") in phases
        and r.get("checkpoint_path")
        and Path(str(r.get("checkpoint_path"))).exists()
        and r.get("method", "") not in exclude_method
        and pd.notna(r.get("auc", np.nan))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: float(r.get("auc", float("-inf"))), reverse=True)[0]


def run_gradcam_for_best(
    cfg: Config,
    dirs: dict[str, Path],
    target_test_df: pd.DataFrame,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    selected = {
        "baseline_external": _best_record(
            records,
            {"external_test", "adaptation"},
            {"dann", "coral", "mmd", "self_training"},
        ),
        "best_adapted": _best_record(records, {"adaptation", "finetuning"}, {"none"}),
    }
    for label, rec in selected.items():
        if rec is None:
            continue
        arch = str(rec["arch"])
        model = _load_model_from_checkpoint(cfg, arch, Path(str(rec["checkpoint_path"])))
        out = run_gradcam(
            model,
            cfg,
            target_test_df,
            cfg._root,
            dirs["gradcam"],
            tag=f"{label}_{arch}_{rec.get('method', 'none')}",
        )
        summary = out.get("summary", {})
        rows.append({
            "model_role": label,
            "arch": arch,
            "method": rec.get("method", "none"),
            "auc": rec.get("auc", np.nan),
            "energy_in_mask_mean": summary.get("energy_in_mask_mean", np.nan),
            "pointing_game_rate": summary.get("pointing_game_rate", np.nan),
            "iou_mean": summary.get("iou_mean", np.nan),
            "grid_path": out.get("grid_path", ""),
            "localization_csv": out.get("localization_csv", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(dirs["metrics"] / "gradcam_summary.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Corrida científica BUSI -> BUS-BRA")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", default="busi")
    ap.add_argument("--target", default="bus_bra")
    ap.add_argument("--architectures", nargs="+",
                    default=["resnet50", "resnet18", "efficientnet_b0", "densenet121"])
    ap.add_argument("--adaptations", nargs="+",
                    default=["none", "dann", "coral", "mmd", "self_training"])
    ap.add_argument("--finetune-fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--target-test-fraction", type=float, default=0.40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-gradcam", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    dirs = ensure_article_dirs(cfg)
    status_path = dirs["metrics"] / "experiment_status.csv"
    status_rows = _load_status_rows(status_path) if args.resume else []
    if status_rows:
        _write_status(status_rows, status_path)
    metric_records = load_metric_records(dirs["metrics"]) if args.resume else []

    write_environment_report(dirs["logs"] / "environment.md", {
        "config": args.config,
        "source": args.source,
        "target": args.target,
    })

    if not args.skip_preprocess and _needs_preprocess(cfg, [args.source, args.target]):
        _run_step(
            "preprocess_required_manifests",
            "preprocess",
            dirs,
            status_rows,
            lambda: preprocess_all(cfg),
            resume=args.resume,
            source=args.source,
            target=args.target,
        )

    manifests = {
        args.source: load_manifest(cfg, args.source),
        args.target: load_manifest(cfg, args.target),
    }
    dataset_summary = _dataset_summary(manifests)
    dataset_summary.to_csv(dirs["metrics"] / "dataset_summary.csv", index=False)
    if (manifests[args.source]["label"].astype(str).str.lower() == "normal").any():
        raise RuntimeError("El manifest de BUSI contiene clase normal en la comparación principal.")

    split = make_target_adaptation_split(
        manifests[args.target],
        cfg,
        test_fraction=args.target_test_fraction,
        seed=cfg.seed,
    )
    pd.DataFrame({
        "partition": ["adapt"] * len(split["adapt_idx"]) + ["test"] * len(split["test_idx"]),
        "index": np.concatenate([split["adapt_idx"], split["test_idx"]]),
    }).to_csv(dirs["metrics"] / f"{args.target}_adaptation_test_split.csv", index=False)
    target_adapt_df = manifests[args.target].iloc[split["adapt_idx"]].reset_index(drop=True)
    target_test_df = manifests[args.target].iloc[split["test_idx"]].reset_index(drop=True)

    for arch in args.architectures:
        for dataset in (args.source, args.target):
            result = _run_step(
                f"baseline_{dataset}_{arch}",
                "baseline_internal",
                dirs,
                status_rows,
                lambda dataset=dataset, arch=arch: train_or_reuse_baseline(
                    copy.deepcopy(cfg), dirs, dataset, arch, args.resume
                ),
                resume=args.resume,
                dataset=dataset,
                arch=arch,
            )
            if result:
                metric_records.extend(result)
                _write_all_metrics(metric_records, dirs)

        result = _run_step(
            f"external_{args.source}_to_{args.target}_{arch}",
            "external_validation",
            dirs,
            status_rows,
            lambda arch=arch: evaluate_external(
                copy.deepcopy(cfg),
                dirs,
                args.source,
                args.target,
                arch,
                manifests[args.target],
                split["test_idx"],
                args.resume,
            ),
            resume=args.resume,
            source=args.source,
            target=args.target,
            arch=arch,
        )
        if result:
            metric_records.extend(result)
            _write_all_metrics(metric_records, dirs)

        for method in args.adaptations:
            result = _run_step(
                f"adaptation_{args.source}_to_{args.target}_{arch}_{method}",
                "adaptation",
                dirs,
                status_rows,
                lambda arch=arch, method=method: run_adaptation(
                    copy.deepcopy(cfg),
                    dirs,
                    args.source,
                    args.target,
                    arch,
                    method,
                    manifests[args.source],
                    target_adapt_df,
                    target_test_df,
                ),
                resume=args.resume,
                source=args.source,
                target=args.target,
                arch=arch,
                method=method,
            )
            if result:
                metric_records.append(result)
                _write_all_metrics(metric_records, dirs)

        for fraction in args.finetune_fractions:
            result = _run_step(
                f"finetuning_{args.source}_to_{args.target}_{arch}_{int(round(fraction * 100))}pct",
                "finetuning",
                dirs,
                status_rows,
                lambda arch=arch, fraction=fraction: run_finetuning(
                    copy.deepcopy(cfg),
                    dirs,
                    args.source,
                    args.target,
                    arch,
                    fraction,
                    target_adapt_df,
                    target_test_df,
                ),
                resume=args.resume,
                source=args.source,
                target=args.target,
                arch=arch,
                fraction=fraction,
            )
            if result:
                metric_records.append(result)
                _write_all_metrics(metric_records, dirs)

    metrics_df = _write_all_metrics(metric_records, dirs)
    drop_df = compute_drop_table(metrics_df, args.source, args.target)
    recovery_df = compute_recovery_table(metrics_df, args.source, args.target)
    drop_df.to_csv(dirs["metrics"] / "performance_drop.csv", index=False)
    recovery_df.to_csv(dirs["metrics"] / "recovery_table.csv", index=False)

    gradcam_df = pd.DataFrame()
    if not args.skip_gradcam:
        result = _run_step(
            "gradcam_best_models",
            "explainability",
            dirs,
            status_rows,
            lambda: run_gradcam_for_best(copy.deepcopy(cfg), dirs, target_test_df, metric_records),
            resume=args.resume,
        )
        if result is not None:
            gradcam_df = result

    status_df = pd.DataFrame(status_rows)
    outputs = write_article_reports(
        dirs["report"],
        dataset_summary,
        metrics_df,
        drop_df,
        recovery_df,
        status_df,
        gradcam_df,
    )
    print("[done] corrida finalizada")
    for key, value in outputs.items():
        print(f"  {key}: {value if value else 'NA'}")


if __name__ == "__main__":
    main()
