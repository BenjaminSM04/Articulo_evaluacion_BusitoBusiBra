#!/usr/bin/env python
"""Create official Curated BUSI and Pawłowska sensitivity metadata artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.busi_curation import curate_busi  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the primary Curated BUSI v1 cohort (386 binary images) and "
            "the Pawłowska sensitivity cohort (457 binary images). Both sources, "
            "local file inventories, and SHA-256 provenance are validated. The "
            "official IDs select original BUSI pixels; the Zenodo copy is audit-only."
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPO_ROOT / "resources" / "busi_curation" / "mapping_curated_BUSI.csv",
        help="Official Curated BUSI v1 class/id mapping.",
    )
    parser.add_argument(
        "--curated-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "curated_busi_v1" / "Curated BUSI",
        help="Curated BUSI v1 copy used only to verify images/ and masks/ coverage.",
    )
    parser.add_argument(
        "--comment-list",
        type=Path,
        default=REPO_ROOT / "resources" / "busi_curation" / "dataset_comment_list.csv",
        help="Semicolon-delimited source audit CSV.",
    )
    parser.add_argument(
        "--original-raw-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "BUSI",
        help="Original BUSI pixels used by both classification manifests.",
    )
    parser.add_argument(
        "--primary-manifest-out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "busi_curated_manifest.csv",
        help="Pipeline output path for the official 386-row binary manifest.",
    )
    parser.add_argument(
        "--primary-publication-manifest-out",
        type=Path,
        default=(REPO_ROOT / "resources" / "busi_curation" / "busi_curated_official_manifest.csv"),
        help="Publishable copy of the official primary manifest.",
    )
    parser.add_argument(
        "--primary-audit-out",
        type=Path,
        default=(REPO_ROOT / "resources" / "busi_curation" / "busi_curated_official_audit.csv"),
        help="Output path for the 450-row official Curated BUSI audit.",
    )
    parser.add_argument(
        "--sensitivity-manifest-out",
        type=Path,
        default=(
            REPO_ROOT / "resources" / "busi_curation" / "busi_pawlowska_sensitivity_manifest.csv"
        ),
        help="Output path for the 457-row Pawłowska sensitivity manifest.",
    )
    parser.add_argument(
        "--sensitivity-audit-out",
        type=Path,
        default=(
            REPO_ROOT / "resources" / "busi_curation" / "busi_pawlowska_sensitivity_audit.csv"
        ),
        help="Output path for the 780-row Pawłowska image-level audit.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to make every output path portable and relative.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = curate_busi(
        mapping=args.mapping,
        curated_root=args.curated_root,
        comment_list=args.comment_list,
        original_raw_root=args.original_raw_root,
        project_root=args.project_root,
        primary_manifest_out=args.primary_manifest_out,
        primary_publication_manifest_out=args.primary_publication_manifest_out,
        primary_audit_out=args.primary_audit_out,
        sensitivity_manifest_out=args.sensitivity_manifest_out,
        sensitivity_audit_out=args.sensitivity_audit_out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
