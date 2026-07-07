"""Reporting helpers: provenance, JSON I/O and standard figures.

Everything an experiment produces (metrics, the effective config, the git commit) is written to
disk so a run can be traced back exactly. Figures use a consistent style.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# JSON I/O (numpy-safe)
# ---------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=_NpEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, classes: list[str], path: str | Path,
                          title: str = "Matriz de confusión") -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: str | Path,
             title: str = "Curva ROC") -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("1 - Especificidad (FPR)")
    ax.set_ylabel("Sensibilidad (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_domain_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                             path: str | Path, metric_name: str = "AUC") -> None:
    """Heatmap of the generalization matrix (rows = train domain, cols = eval domain)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", vmin=0.5, vmax=1.0,
                xticklabels=col_labels, yticklabels=row_labels, ax=ax,
                cbar_kws={"label": metric_name})
    ax.set_xlabel("Evaluado en")
    ax.set_ylabel("Entrenado en")
    ax.set_title(f"Matriz de generalización cruzada ({metric_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict[str, list[float]], path: str | Path) -> None:
    """Plot loss and the monitored metric across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Pérdida")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    if "val_auc" in history:
        axes[1].plot(history["val_auc"], label="val AUC", color="tab:green")
    axes[1].set_title("AUC (validación)")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tablas LaTeX (andamiaje del manuscrito: se autollenan al ejecutar el pipeline)
# ---------------------------------------------------------------------------
# Estas funciones emiten fragmentos .tex (booktabs) listos para \input en
# Latex Doc/main.tex. Se regeneran en cada corrida, de modo que las tablas de
# Resultados del artículo reflejan siempre los números reales, sin copiar a mano.
def _latex_cell(value: Any) -> str:
    """Escapa el contenido controlado de nuestras celdas (números, nombres, '±')."""
    s = str(value)
    s = s.replace("±", r"$\pm$")     # media ± desv.
    s = s.replace("_", r"\_")        # p. ej. bus_bra
    return s


def save_latex_table(df, path: str | Path, caption: str, label: str,
                     header_map: dict[str, str] | None = None,
                     column_format: str | None = None) -> None:
    """Escribe un DataFrame como tabla LaTeX booktabs a ``path``."""
    cols = list(df.columns)
    headers = [(header_map or {}).get(c, c) for c in cols]
    colfmt = column_format or ("l" * len(cols))
    lines = [
        "% Autogenerado por scripts/06_generate_report.py — no editar a mano.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule",
        " & ".join(_latex_cell(h) for h in headers) + r" \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(" & ".join(_latex_cell(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_matrix_latex(cross_records: list[dict], path: str | Path,
                      caption: str | None = None, label: str = "tab:matrix") -> None:
    """Emite la matriz de generalización cruzada (AUC) sin adaptación como tabla LaTeX.

    ``cross_records`` es la lista de JSON de ``cross_domain_*.json`` ya cargados.
    Se elige la condición ``adaptation == 'none'`` (la caída de dominio de referencia).
    """
    rec = next((r for r in cross_records if r.get("adaptation") == "none"), None)
    rec = rec or (cross_records[0] if cross_records else None)
    if rec is None:
        return
    domains, mat = rec["domains"], rec["auc_matrix"]
    caption = caption or ("Matriz de generalización cruzada (AUC). Filas: dominio de "
                          "entrenamiento; columnas: dominio de evaluación.")
    lines = [
        "% Autogenerado por scripts/06_generate_report.py — no editar a mano.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{l{'c' * len(domains)}}}", r"\toprule",
        r"\textbf{Entrena $\downarrow$ / Evalúa $\rightarrow$} & "
        + " & ".join(rf"\textbf{{{_latex_cell(d)}}}" for d in domains) + r" \\",
        r"\midrule",
    ]
    for i, s in enumerate(domains):
        cells = " & ".join(f"{mat[i][j]:.3f}" for j in range(len(domains)))
        lines.append(rf"\textbf{{{_latex_cell(s)}}} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
