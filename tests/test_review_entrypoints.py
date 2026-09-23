"""Safety checks for a code-only checkout without local clinical data."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("command", "required_option"),
    [
        (["scripts/02_preprocess.py"], "--config"),
        (["scripts/11_run_publication_experiments.py"], "--config"),
        (["scripts/12_finalize_publication_inference.py"], "--config"),
        (["scripts/13_analyze_publication_results.py"], "--config"),
        (["scripts/14_generate_publication_artifacts.py"], "--results"),
    ],
)
def test_review_commands_require_explicit_input(command: list[str], required_option: str) -> None:
    """No default may silently target the historical v2 data or final test."""
    process = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 2
    assert "required" in process.stderr.lower()
    assert required_option in process.stderr


def test_config_loader_requires_an_explicit_protocol() -> None:
    with pytest.raises(TypeError):
        load_config()


def test_split_cli_requires_an_explicit_protocol() -> None:
    # Parsing alone must reject an unspecified config, before opening any data.
    module = runpy.run_path(str(ROOT / "src/data/publication_splits.py"))
    with pytest.raises(SystemExit) as error:
        module["build_cli_parser"]().parse_args([])
    assert error.value.code == 2


def test_legacy_busi_preprocessor_is_not_a_review_entrypoint() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/02_preprocess.py", "--config", "absent.yaml", "--only", "busi"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 2
    assert "busi" in process.stderr


def test_missing_review_split_points_to_existing_split_cli(tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/11_run_publication_experiments.py"))
    with pytest.raises(FileNotFoundError) as error:
        module["_read_manifest"](tmp_path / "source_train_manifest.csv")
    assert "python -m src.data.publication_splits" in str(error.value)
