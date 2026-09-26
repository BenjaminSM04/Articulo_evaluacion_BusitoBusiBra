"""Load and validate the explicitly selected YAML protocol configuration.

This module loads a chosen configuration into a dot-accessible object, resolves all paths
relative to the project root, and creates the output directories. Requiring a path prevents
accidentally selecting the historical v2 protocol for a future review run.

Implementation note: nested dicts are converted to ``DotDict`` **in place** at load time, so that
both ``cfg.model.architecture`` and ``cfg.model["architecture"] = x`` operate on the same object.
A naive "wrap-on-access" version would return a fresh copy each time and silently drop overrides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Dictionary with attribute access. Nested dicts/lists are wrapped recursively at init."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, DotDict):
            return value
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [DotDict._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)


def _to_plain(obj: Any) -> Any:
    """Recursively convert DotDict back to plain dict/list (for YAML serialization)."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def project_root() -> Path:
    """Return the repository root (two levels up from this file: src/ -> root)."""
    return Path(__file__).resolve().parents[1]


class Config(DotDict):
    """Configuration object loaded from YAML with path-resolution helpers."""

    @classmethod
    def load(cls, path: str | Path, *, create_dirs: bool = True) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        cfg = cls(raw)
        cfg._root = project_root()
        cfg._config_path = str(path.resolve())
        if create_dirs:
            cfg._make_dirs()
        return cfg

    # -- path helpers ---------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a path from the ``paths`` block against the project root."""
        return (self._root / self["paths"][key]).resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve any relative path against the project root."""
        return (self._root / relative).resolve()

    def _make_dirs(self) -> None:
        for key in ("data_raw", "data_processed", "results", "models", "figures", "reports"):
            self.path(key).mkdir(parents=True, exist_ok=True)

    def dump(self, dst: str | Path) -> None:
        """Persist the effective config next to results for provenance."""
        plain = {k: _to_plain(v) for k, v in self.items() if not k.startswith("_")}
        with open(dst, "w", encoding="utf-8") as fh:
            yaml.safe_dump(plain, fh, sort_keys=False, allow_unicode=True)


def load_config(path: str | Path, *, create_dirs: bool = True) -> Config:
    """Load an explicitly selected protocol; never select a historical default."""
    return Config.load(path, create_dirs=create_dirs)
