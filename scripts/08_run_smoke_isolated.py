"""Run a synthetic smoke test in a temporary directory.

Unlike scripts/run_smoke.bat, this does not write to project-level data/raw, data/processed, or
results. It creates an absolute-path temporary config and runs the active pipeline against it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test aislado con datos sintéticos")
    ap.add_argument("--keep", action="store_true", help="No borra el directorio temporal al salir")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load((root / "config" / "config_smoke.yaml").read_text(encoding="utf-8"))

    tmp_ctx = tempfile.TemporaryDirectory(prefix="busgen_smoke_")
    tmp = Path(tmp_ctx.name)
    data_raw = tmp / "data" / "raw"
    data_processed = tmp / "data" / "processed"
    results = tmp / "results"

    base_cfg["paths"] = {
        "data_raw": str(data_raw),
        "data_processed": str(data_processed),
        "results": str(results),
        "models": str(results / "models"),
        "figures": str(results / "figures"),
        "reports": str(results / "reports"),
    }
    base_cfg["datasets"]["busi"]["raw_dir"] = str(data_raw / "BUSI")
    base_cfg["datasets"]["bus_bra"]["raw_dir"] = str(data_raw / "BUS-BRA")
    base_cfg["model"]["pretrained"] = False
    base_cfg["training"]["epochs"] = 1
    base_cfg["training"]["num_workers"] = 0
    base_cfg["evaluation"]["n_bootstrap"] = 20

    cfg_path = tmp / "config_smoke_isolated.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    commands = [
        [sys.executable, "scripts/00_make_synthetic_data.py", "--config", str(cfg_path)],
        [sys.executable, "scripts/02_preprocess.py", "--config", str(cfg_path)],
        [
            sys.executable,
            "scripts/03_train_baseline.py",
            "--config",
            str(cfg_path),
            "--dataset",
            "busi",
        ],
        [
            sys.executable,
            "scripts/04_cross_domain_matrix.py",
            "--config",
            str(cfg_path),
            "--adaptation",
            "none",
        ],
        [sys.executable, "scripts/06_generate_report.py", "--config", str(cfg_path)],
    ]
    try:
        for cmd in commands:
            print("[smoke]", " ".join(cmd))
            subprocess.check_call(cmd, cwd=root)
        print(f"[ok] smoke aislado completado en {tmp}")
        if args.keep:
            tmp_ctx._finalizer.detach()  # noqa: SLF001 - intentional for --keep
    finally:
        if not args.keep:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
