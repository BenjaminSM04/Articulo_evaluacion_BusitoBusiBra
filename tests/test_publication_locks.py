"""Atomic lock and irreversible test-access checks for publication protocol v2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_and_finalization_share_one_atomic_transition_lock(tmp_path):
    train = _load_script("publication_train_lock_test", "11_run_publication_experiments.py")
    finalize = _load_script(
        "publication_finalize_lock_test",
        "12_finalize_publication_inference.py",
    )

    training_lock = train._acquire_training_lock(tmp_path)
    with pytest.raises(RuntimeError, match="active"):
        finalize._acquire_finalization_transition(tmp_path)
    train._release_training_lock(training_lock)

    transition_lock = finalize._acquire_finalization_transition(tmp_path)
    with pytest.raises(RuntimeError, match="active"):
        train._acquire_training_lock(tmp_path)
    finalize._release_transition(transition_lock)


def test_finalization_lock_covers_the_whole_action_and_cleans_up(tmp_path):
    finalize = _load_script(
        "publication_finalize_lifetime_test", "12_finalize_publication_inference.py"
    )

    def run_inference():
        assert (tmp_path / "logs" / "TRAINING_ACTIVE.lock").is_file()
        with pytest.raises(RuntimeError, match="active"):
            finalize._acquire_finalization_transition(tmp_path)
        return "completed"

    assert finalize._run_with_finalization_lock(tmp_path, run_inference) == "completed"
    assert not (tmp_path / "logs" / "TRAINING_ACTIVE.lock").exists()

    def interrupted_inference():
        assert (tmp_path / "logs" / "TRAINING_ACTIVE.lock").is_file()
        raise ValueError("interrupted")

    with pytest.raises(ValueError, match="interrupted"):
        finalize._run_with_finalization_lock(tmp_path, interrupted_inference)
    assert not (tmp_path / "logs" / "TRAINING_ACTIVE.lock").exists()


def test_test_access_marker_is_irreversible_and_fingerprint_bound(tmp_path):
    train = _load_script("publication_train_marker_test", "11_run_publication_experiments.py")
    finalize = _load_script(
        "publication_finalize_marker_test",
        "12_finalize_publication_inference.py",
    )
    fingerprint = {"config": "abc", "checkpoints": "def"}

    observed = finalize._freeze_test_access(tmp_path, fingerprint, resume=False)
    assert (tmp_path / "TEST_ACCESS_STARTED.json").is_file()
    assert observed
    with pytest.raises(RuntimeError, match="already initiated"):
        finalize._freeze_test_access(tmp_path, fingerprint, resume=False)
    assert finalize._freeze_test_access(tmp_path, fingerprint, resume=True) == observed
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        finalize._freeze_test_access(
            tmp_path,
            {**fingerprint, "checkpoints": "changed"},
            resume=True,
        )
    with pytest.raises(RuntimeError, match="permanently frozen"):
        train._acquire_training_lock(tmp_path)


def test_training_fingerprint_selects_the_protocol_for_each_version():
    train = _load_script(
        "publication_train_protocol_fingerprint", "11_run_publication_experiments.py"
    )
    root = Path(__file__).resolve().parents[1]
    _, old = train._training_code_fingerprint(root, protocol_version="2.0")
    _, review = train._training_code_fingerprint(root, protocol_version="3.0-review-5seed")
    assert "docs/PROTOCOL_PUBLICATION_V2.md" in old
    assert "docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md" in review
    assert "docs/PROTOCOL_PUBLICATION_V2.md" not in review
