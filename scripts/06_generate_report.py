"""Paso 6 — Consolidar resultados en tablas y un resumen Markdown.

Lee todos los JSON de results/reports/ (baselines y matrices cross-domain) y genera:
    results/reports/SUMMARY.md   (resumen legible con tablas)
    results/reports/baselines.csv
    results/reports/cross_domain.csv

Uso:
    python scripts/06_generate_report.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.utils.reporting import (  # noqa: E402
    git_commit,
    load_json,
    save_latex_table,
    save_matrix_latex,
    timestamp,
)


def _collect_baselines(reports: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(reports.glob("baseline_*.json")):
        d = load_json(f)
        row = {"dataset": d["dataset"], "arch": d["arch"], "n_folds": d["n_folds"]}
        for k, v in d["summary"].items():
            row[k] = f"{v['mean']:.3f} ± {v['std']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_cross_domain(reports: Path) -> tuple[pd.DataFrame, list[str], list[dict]]:
    rows, blocks, records = [], [], []
    for f in sorted(reports.glob("cross_domain_*.json")):
        d = load_json(f)
        records.append(d)
        domains = d["domains"]
        mat = d["auc_matrix"]
        blocks.append(f"\n### {d['arch']} — adaptación: `{d['adaptation']}`\n")
        header = "| train ↓ \\ eval → | " + " | ".join(domains) + " |"
        sep = "|" + "---|" * (len(domains) + 1)
        blocks.append(header)
        blocks.append(sep)
        for i, s in enumerate(domains):
            cells = " | ".join(f"{mat[i][j]:.3f}" for j in range(len(domains)))
            blocks.append(f"| **{s}** | {cells} |")
        for s, g in d["gaps"].items():
            blocks.append(f"\n- `{s}`: intra={g['intra_auc']:.3f}, "
                          f"cross={g['mean_cross_auc']:.3f}, **gap={g['mean_gap']:.3f}**")
            rows.append({"arch": d["arch"], "adaptation": d["adaptation"], "source": s,
                         "intra_auc": g["intra_auc"], "mean_cross_auc": g["mean_cross_auc"],
                         "mean_gap": g["mean_gap"]})
    return pd.DataFrame(rows), blocks, records


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte consolidado")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    reports = cfg.path("reports")

    base_df = _collect_baselines(reports)
    cross_df, cross_blocks, cross_records = _collect_cross_domain(reports)
    if not base_df.empty:
        base_df.to_csv(reports / "baselines.csv", index=False)
    if not cross_df.empty:
        cross_df.to_csv(reports / "cross_domain.csv", index=False)

    # Fragmentos LaTeX (andamiaje del manuscrito): se autollenan con estos números.
    if not base_df.empty:
        save_latex_table(
            base_df, reports / "table_baselines.tex",
            caption=("Baselines intra-dominio: media $\\pm$ desviación sobre los folds "
                     "de validación cruzada."),
            label="tab:baselines",
            header_map={"dataset": "Dataset", "arch": "Arquitectura", "n_folds": "Folds",
                        "accuracy": "Acc.", "sensitivity": "Sens.", "specificity": "Espec.",
                        "auc": "AUC", "f1": "F1", "balanced_accuracy": "Bal.\\ acc."},
            column_format="llccccccc")
    if cross_records:
        save_matrix_latex(cross_records, reports / "table_matrix.tex")

    lines = [f"# Resumen de resultados\n",
             f"_Generado: {timestamp()} | commit: `{git_commit()}`_\n",
             "## Baselines intra-dominio (media ± desv. sobre folds)\n"]
    lines.append(base_df.to_markdown(index=False) if not base_df.empty
                 else "_Sin resultados de baseline todavía._")
    lines.append("\n## Generalización cruzada (AUC)\n")
    lines.extend(cross_blocks if cross_blocks else ["_Sin matrices cross-domain todavía._"])

    summary = reports / "SUMMARY.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] -> {summary}")


if __name__ == "__main__":
    main()
