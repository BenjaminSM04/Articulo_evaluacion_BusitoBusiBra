"""Analyze frozen predictions for the explicitly selected publication protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.evaluation.publication_statistics import (  # noqa: E402
    analyze_publication_predictions,
    validate_locked_artifacts,
)


def _frozen_methods(cfg) -> list[str]:
    return (
        ["source_direct"]
        + [str(value) for value in cfg.publication.methods]
        + [
            f"finetune_{int(round(float(value) * 100))}pct"
            for value in cfg.publication.finetune_fractions
        ]
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_analysis_code(root: Path, lock: dict) -> None:
    expected = lock.get("finalization_code_files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Final inference lock lacks frozen analysis-code hashes.")
    aggregate = hashlib.sha256()
    for relative in sorted(expected):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen analysis dependency is missing: {path}")
        observed = _sha256_file(path)
        if observed != expected[relative]:
            raise ValueError(f"Analysis code changed after test access: {relative}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(observed.encode("ascii"))
        aggregate.update(b"\n")
    if aggregate.hexdigest() != lock.get("finalization_code_sha256"):
        raise ValueError("Aggregate analysis-code fingerprint does not match the lock.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level confirmatory statistics after locked final inference"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    requested_cfg = load_config(args.config)
    results = requested_cfg.path("results")
    lock_path = results / "FINAL_INFERENCE_LOCK.json"
    if not lock_path.exists():
        raise FileNotFoundError(
            "Missing FINAL_INFERENCE_LOCK.json. Run script 12 with the frozen checkpoints first."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("unlocked") or lock.get("status") != "completed":
        raise ValueError("Final inference lock is not marked as unlocked/completed.")
    finalization_fingerprint = str(lock.get("finalization_fingerprint_sha256", "")).strip()
    if not finalization_fingerprint:
        raise ValueError("Final inference lock lacks its finalization fingerprint.")
    observed_config_hash = _sha256_file(args.config)
    if lock.get("config_sha256") != observed_config_hash:
        raise ValueError(
            "The publication config changed after final inference; refusing post-hoc analysis."
        )
    effective_config_path = results / "effective_config.yaml"
    if not effective_config_path.is_file():
        raise FileNotFoundError("Frozen effective_config.yaml is missing.")
    observed_effective_config_hash = _sha256_file(effective_config_path)
    if lock.get("effective_config_sha256") != observed_effective_config_hash:
        raise ValueError("Frozen effective config does not match the final inference lock.")
    cfg = load_config(effective_config_path)
    if cfg.path("results") != results:
        raise ValueError("Frozen effective config points to a different results directory.")
    _validate_frozen_analysis_code(cfg._root, lock)

    validate_locked_artifacts(
        cfg._root,
        lock.get("artifact_hashes"),
        expected_manifest_sha256=lock.get("artifact_hashes_sha256"),
    )

    architectures = [str(value) for value in cfg.publication.architectures]
    seeds = [int(value) for value in cfg.publication.seeds]
    methods = _frozen_methods(cfg)
    expected_experiments = len(architectures) * len(seeds) * len(methods)
    if int(lock.get("n_experiments", -1)) != expected_experiments:
        raise ValueError(
            "Final inference lock does not match the configured experiment matrix: "
            f"expected {expected_experiments}, observed {lock.get('n_experiments')}."
        )
    if len(architectures) * 3 != 6:
        raise ValueError("Protocol v2 requires two architectures and six UDA comparisons.")
    if int(cfg.evaluation.n_bootstrap) != 5000:
        raise ValueError("The confirmatory protocol requires exactly 5000 bootstrap replicates.")
    if not abs(float(cfg.evaluation.ci_level) - 0.95) < 1e-12:
        raise ValueError("The confirmatory protocol requires 95% confidence intervals.")
    if int(lock.get("target_test_patients", -1)) != 426:
        raise ValueError("The locked target test must contain exactly 426 patients.")

    output_paths = analyze_publication_predictions(
        results / "predictions",
        results / "analysis",
        architectures=architectures,
        methods=methods,
        seeds=seeds,
        n_bootstrap=int(cfg.evaluation.n_bootstrap),
        confidence_level=float(cfg.evaluation.ci_level),
        bootstrap_seed=int(cfg.publication.split_seed),
        require_six_uda_comparisons=True,
        expected_finalization_fingerprint=finalization_fingerprint,
    )
    print("[done] confirmatory analysis written:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
