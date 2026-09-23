"""Generate publication tables and figures from explicitly selected locked results.

This script is intentionally downstream from the frozen confirmatory analysis. It never loads
checkpoints or images and does not recompute inferential statistics. Every plotted value is read
from the CSV artifacts written by ``scripts/13_analyze_publication_results.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    }
)

METHOD_LABELS = {
    "source_direct": "Fuente directa",
    "source_only_matched": "Control fuente emparejado",
    "dann": "DANN",
    "coral": "CORAL",
    "mmd": "MMD",
    "finetune_5pct": "Ajuste fino 5 %",
    "finetune_10pct": "Ajuste fino 10 %",
    "finetune_20pct": "Ajuste fino 20 %",
}
ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
}
ARCH_COLORS = {
    "resnet18": "#0072B2",
    "efficientnet_b0": "#D55E00",
}
ENSEMBLE_METHODS = (
    "source_direct",
    "source_only_matched",
    "dann",
    "coral",
    "mmd",
)
UDA_METHODS = ("dann", "coral", "mmd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required publication artifact is missing: {path}")
    return path


def _load_locked_inputs(results: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    lock_path = _require(results / "FINAL_INFERENCE_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "completed" or not lock.get("unlocked"):
        raise ValueError("Final inference lock is not completed/unlocked.")
    if int(lock.get("target_test_patients", -1)) != 426:
        raise ValueError("Publication figures require the frozen 426-patient target test.")

    analysis = results / "analysis"
    names = (
        "ensemble_metrics",
        "per_seed_metrics",
        "per_seed_summary",
        "finetune_replica_metrics",
        "finetune_summary",
        "paired_bootstrap_differences",
        "delong_holm",
        "calibration_parameters",
        "target_calibration_diagnostics",
        "source_ensemble_metrics",
    )
    tables = {name: pd.read_csv(_require(analysis / f"{name}.csv")) for name in names}

    observed_architectures = set(tables["ensemble_metrics"]["arch"].astype(str))
    if observed_architectures != set(ARCH_LABELS):
        raise ValueError(
            "Expected exactly ResNet-18 and EfficientNet-B0 analysis rows; got "
            f"{sorted(observed_architectures)}."
        )
    observed_methods = set(tables["ensemble_metrics"]["method"].astype(str))
    if observed_methods != set(ENSEMBLE_METHODS):
        raise ValueError(
            f"Expected ensemble methods {list(ENSEMBLE_METHODS)}, got {sorted(observed_methods)}."
        )
    return lock, tables


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    jpg = stem.with_suffix(".jpg")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(jpg, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Explicitly persist 600-dpi metadata and high JPEG quality.
    with Image.open(png) as image:
        image.save(png, dpi=(600, 600), optimize=True)
    with Image.open(jpg) as image:
        image.convert("RGB").save(jpg, quality=95, dpi=(600, 600), subsampling=0)
    return [png, jpg]


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def figure_cohort_flow(output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Flujo de cohortes y particiones del estudio",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.245,
        0.915,
        "DOMINIO FUENTE",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#0072B2",
    )
    ax.text(
        0.75,
        0.915,
        "DOMINIO OBJETIVO",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#D55E00",
    )

    # Connectors are drawn first so that every line remains behind the boxes.
    arrow = dict(arrowstyle="-|>", lw=1.15, color="#666666", mutation_scale=10)
    ax.annotate("", xy=(0.245, 0.72), xytext=(0.245, 0.77), arrowprops=arrow)

    # BUSI is split into three mutually exclusive image-level partitions.
    ax.plot([0.245, 0.245, 0.035, 0.035], [0.59, 0.555, 0.555, 0.235], color="#777777", lw=1.05)
    for y in (0.47, 0.35, 0.23):
        ax.annotate("", xy=(0.06, y), xytext=(0.035, y), arrowprops=arrow)

    # BUS-BRA is split into a historical development pool and a locked test set.
    ax.annotate("", xy=(0.75, 0.72), xytext=(0.75, 0.77), arrowprops=arrow)
    ax.plot([0.75, 0.98, 0.98], [0.745, 0.745, 0.23], color="#777777", lw=1.05)
    ax.annotate("", xy=(0.94, 0.23), xytext=(0.98, 0.23), arrowprops=arrow)

    # The development pool is partitioned by patient into adaptation and calibration.
    ax.plot([0.75, 0.75], [0.59, 0.555], color="#777777", lw=1.05)
    ax.plot([0.6225, 0.8525], [0.555, 0.555], color="#777777", lw=1.05)
    ax.annotate("", xy=(0.6225, 0.52), xytext=(0.6225, 0.555), arrowprops=arrow)
    ax.annotate("", xy=(0.8525, 0.52), xytext=(0.8525, 0.555), arrowprops=arrow)

    _box(
        ax,
        (0.05, 0.77),
        0.39,
        0.09,
        "Curated BUSI v1.0\n450 imágenes",
        facecolor="#E8F1FA",
        edgecolor="#0072B2",
        fontsize=8.5,
    )
    _box(
        ax,
        (0.55, 0.77),
        0.40,
        0.09,
        "BUS-BRA\n1875 imágenes · 1064 pacientes",
        facecolor="#FCEFE8",
        edgecolor="#D55E00",
        fontsize=8.5,
    )
    _box(
        ax,
        (0.05, 0.59),
        0.39,
        0.11,
        "Cohorte binaria elegible\n386 imágenes · B/M: 222/164",
        facecolor="#E8F1FA",
        edgecolor="#0072B2",
        fontsize=8.5,
    )
    _box(
        ax,
        (0.06, 0.43),
        0.38,
        0.08,
        "Entrenamiento fuente\n247 imágenes · B/M: 142/105",
        facecolor="#F4F8FC",
        edgecolor="#0072B2",
        fontsize=8,
    )
    _box(
        ax,
        (0.06, 0.31),
        0.38,
        0.08,
        "Validación fuente\n62 imágenes · B/M: 36/26",
        facecolor="#F4F8FC",
        edgecolor="#0072B2",
        fontsize=8,
    )
    _box(
        ax,
        (0.06, 0.19),
        0.38,
        0.08,
        "Prueba interna\n77 imágenes · B/M: 44/33",
        facecolor="#F4F8FC",
        edgecolor="#0072B2",
        fontsize=8,
    )
    _box(
        ax,
        (0.55, 0.59),
        0.40,
        0.11,
        "Desarrollo histórico\n1139 imágenes · 638 pacientes\nB/M (imágenes): 773/366",
        facecolor="#FCF6F2",
        edgecolor="#D55E00",
        fontsize=8,
    )
    _box(
        ax,
        (0.52, 0.39),
        0.205,
        0.13,
        "Adaptación\n946 imágenes\n532 pacientes\nB/M: 642/304",
        facecolor="#FCF6F2",
        edgecolor="#D55E00",
        fontsize=8,
    )
    _box(
        ax,
        (0.75, 0.39),
        0.205,
        0.13,
        "Calibración\n193 imágenes\n106 pacientes\nB/M: 131/62",
        facecolor="#FCF6F2",
        edgecolor="#D55E00",
        fontsize=8,
    )
    _box(
        ax,
        (0.58, 0.17),
        0.36,
        0.12,
        "Prueba externa histórica bloqueada\n"
        "736 imágenes · 426 pacientes\n"
        "B/M (imágenes): 495/241",
        facecolor="#FFF3CD",
        edgecolor="#9C6B00",
        fontsize=8,
    )

    ax.text(
        0.5,
        0.045,
        "B/M = benignas/malignas. En BUS-BRA, B/M corresponde a imágenes; "
        "las particiones no comparten pacientes.",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )
    return _save_figure(fig, output_dir / "figura_1_flujo_cohortes")


def figure_target_auc(ensemble: pd.DataFrame, output_dir: Path) -> list[Path]:
    data = ensemble[ensemble["metric"].eq("auc_raw")].copy()
    data["method"] = pd.Categorical(data["method"], ENSEMBLE_METHODS, ordered=True)
    data["arch"] = pd.Categorical(data["arch"], list(ARCH_LABELS), ordered=True)
    data = data.sort_values(["arch", "method"]).reset_index(drop=True)
    if len(data) != 10:
        raise ValueError(f"Expected 10 target AUC rows, found {len(data)}.")

    labels: list[str] = []
    y_positions: list[float] = []
    cursor = 0.0
    for architecture in ARCH_LABELS:
        subset = data[data["arch"].eq(architecture)]
        for method in ENSEMBLE_METHODS:
            row = subset[subset["method"].eq(method)]
            if len(row) != 1:
                raise ValueError(f"Missing AUC row for {architecture}/{method}.")
            labels.append(f"{ARCH_LABELS[architecture]} · {METHOD_LABELS[method]}")
            y_positions.append(cursor)
            cursor += 1.0
        cursor += 0.65

    fig, ax = plt.subplots(figsize=(8.3, 5.8))
    for y, (_, row) in zip(y_positions, data.iterrows(), strict=True):
        color = ARCH_COLORS[str(row["arch"])]
        estimate = float(row["estimate"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        ax.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5.5,
            linewidth=1.4,
        )
        ax.text(high + 0.008, y, f"{estimate:.3f} ({low:.3f}–{high:.3f})", va="center", fontsize=8)

    all_low = float(data["ci_low"].min())
    all_high = float(data["ci_high"].max())
    left = max(0.0, min(0.48, all_low - 0.06))
    right = min(1.18, max(1.0, all_high + 0.24))
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(left, right)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_xlabel("ROC-AUC por paciente (IC del 95 %)")
    ax.set_title("Discriminación en la prueba externa histórica")
    ax.text(
        0.0,
        -0.16,
        "IC: 5000 remuestreos bootstrap estratificados por paciente. "
        "La línea discontinua indica discriminación aleatoria.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figura_2_auc_externo")


def figure_uda_differences(
    paired: pd.DataFrame,
    delong: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    data = paired[paired["metric"].eq("auc_raw")].copy()
    tests = delong[["arch", "method", "p_holm", "reject_holm_0_05"]].drop_duplicates(
        ["arch", "method"]
    )
    data = data.merge(tests, on=["arch", "method"], how="left", validate="one_to_one")
    data["method"] = pd.Categorical(data["method"], UDA_METHODS, ordered=True)
    data["arch"] = pd.Categorical(data["arch"], list(ARCH_LABELS), ordered=True)
    data = data.sort_values(["arch", "method"]).reset_index(drop=True)
    if len(data) != 6:
        raise ValueError(f"Expected six paired UDA AUC differences, found {len(data)}.")

    labels = [
        f"{ARCH_LABELS[str(row.arch)]} · {METHOD_LABELS[str(row.method)]}"
        for row in data.itertuples()
    ]
    y = np.arange(len(data), dtype=float)
    y[3:] += 0.55
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for ypos, row in zip(y, data.itertuples(), strict=True):
        estimate = float(row.estimate_difference)
        low = float(row.ci_low)
        high = float(row.ci_high)
        color = ARCH_COLORS[str(row.arch)]
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5.5,
            linewidth=1.4,
        )
    span = max(abs(float(data["ci_low"].min())), abs(float(data["ci_high"].max())))
    limit = max(0.04, span + 0.005)
    ax.set_xlim(-limit, limit)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_xlabel("Diferencia de ROC-AUC (método − control fuente emparejado)")
    ax.set_title("Comparaciones no supervisadas emparejadas")
    ax.text(
        0.0,
        -0.20,
        "IC por bootstrap pareado. Las estimaciones exactas y los valores p de DeLong "
        "corregidos mediante Holm se presentan en la Tabla 3.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figura_3_diferencias_uda")


def _finetune_fraction(method: str) -> int:
    match = re.fullmatch(r"finetune_(\d+)pct", str(method))
    if match is None:
        raise ValueError(f"Unexpected fine-tuning method name: {method}")
    return int(match.group(1))


def figure_finetune(
    replicas: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    data = replicas[replicas["method"].astype(str).str.startswith("finetune_")].copy()
    data["budget"] = data["method"].map(_finetune_fraction)
    means = summary[
        summary["metric"].eq("auc_raw") & summary["method"].astype(str).str.startswith("finetune_")
    ].copy()
    means["budget"] = means["method"].map(_finetune_fraction)
    seed_values = sorted(data["seed"].astype(int).unique())
    replicate_counts = data.groupby(["arch", "method"])["seed"].nunique()
    if (
        len(means) != 6
        or len(data) != 6 * len(seed_values)
        or len(replicate_counts) != 6
        or not replicate_counts.eq(len(seed_values)).all()
    ):
        raise ValueError(
            f"Expected six fine-tuning summaries and one replica per method/seed; "
            f"found {len(data)} replicas and {len(means)} summaries."
        )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    offsets = {"resnet18": -0.55, "efficientnet_b0": 0.55}
    marker_shapes = ("o", "s", "^", "D", "P", "X", "v")
    markers = {
        int(seed): marker_shapes[index % len(marker_shapes)]
        for index, seed in enumerate(seed_values)
    }
    for architecture in ARCH_LABELS:
        color = ARCH_COLORS[architecture]
        subset = data[data["arch"].eq(architecture)]
        for seed in sorted(subset["seed"].astype(int).unique()):
            seed_rows = subset[subset["seed"].astype(int).eq(seed)].sort_values("budget")
            ax.scatter(
                seed_rows["budget"] + offsets[architecture],
                seed_rows["auc_raw"],
                marker=markers[int(seed)],
                s=34,
                color=color,
                alpha=0.5,
                linewidths=0,
            )
        mean_rows = means[means["arch"].eq(architecture)].sort_values("budget")
        ax.errorbar(
            mean_rows["budget"] + offsets[architecture],
            mean_rows["mean"],
            yerr=mean_rows["sd"],
            fmt="-o",
            color=color,
            capsize=4,
            linewidth=1.8,
            markersize=5.5,
            label=ARCH_LABELS[architecture],
        )
    ax.set_xticks([5, 10, 20], ["5 %", "10 %", "20 %"])
    ax.set_xlim(2, 23)
    ax.set_ylim(
        max(0.0, float(data["auc_raw"].min()) - 0.03),
        min(1.0, float(data["auc_raw"].max()) + 0.03),
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_xlabel("Presupuesto de pacientes objetivo etiquetados")
    ax.set_ylabel("ROC-AUC por paciente")
    ax.set_title("Ajuste fino: réplicas y media ± DE")
    ax.legend(frameon=False)
    seed_list = ", ".join(str(seed) for seed in seed_values[:-1]) + f" y {seed_values[-1]}"
    ax.text(
        0.0,
        -0.18,
        f"Los puntos tenues son réplicas (semillas {seed_list}). Cada réplica usa una "
        "cohorte anidada distinta; no se formó un ensamble.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figura_4_ajuste_fino")


def _reliability_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for index in range(bins):
        if index == bins - 1:
            selected = (y_prob >= edges[index]) & (y_prob <= edges[index + 1])
        else:
            selected = (y_prob >= edges[index]) & (y_prob < edges[index + 1])
        if not selected.any():
            continue
        predicted.append(float(y_prob[selected].mean()))
        observed.append(float(y_true[selected].mean()))
        counts.append(int(selected.sum()))
    return np.asarray(predicted), np.asarray(observed), np.asarray(counts)


def figure_reliability(results: Path, output_dir: Path) -> list[Path]:
    prediction_dir = results / "analysis" / "ensemble_predictions" / "target"
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.2), sharex=True, sharey=True)
    for row_index, architecture in enumerate(ARCH_LABELS):
        for column_index, method in enumerate(ENSEMBLE_METHODS):
            ax = axes[row_index, column_index]
            table = pd.read_csv(
                _require(prediction_dir / f"{architecture}_{method}_target_test_patients.csv")
            )
            y_true = table["label_idx"].to_numpy(dtype=int)
            for column, label, color, marker in (
                ("probability_raw", "Cruda", "#999999", "o"),
                ("probability_calibrated", "Calibrada", ARCH_COLORS[architecture], "s"),
            ):
                predicted, observed, _ = _reliability_points(
                    y_true,
                    table[column].to_numpy(dtype=float),
                )
                ax.plot(predicted, observed, marker=marker, color=color, linewidth=1, label=label)
            ax.plot([0, 1], [0, 1], "--", color="#444444", linewidth=0.8)
            ax.set_title(
                f"{ARCH_LABELS[architecture]}\n{METHOD_LABELS[method]}",
                fontsize=8.5,
            )
            ax.grid(color="#E5E5E5", linewidth=0.5)
            if column_index == 0:
                ax.set_ylabel("Frecuencia observada")
            if row_index == 1:
                ax.set_xlabel("Riesgo predicho")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Curvas de fiabilidad descriptivas en la prueba objetivo (10 intervalos)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.015,
        "Cada punto representa un intervalo no vacío. La calibración se ajustó exclusivamente "
        "en la cohorte de calibración separada.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    return _save_figure(fig, output_dir / "suplementaria_1_curvas_fiabilidad")


def _patient_count(frame: pd.DataFrame, label_idx: int | None = None) -> int | str:
    if "patient_id" not in frame.columns:
        return "NA"
    values = frame
    if label_idx is not None:
        values = values[values["label_idx"].astype(int).eq(label_idx)]
    identifiers = values["patient_id"].dropna().astype(str)
    identifiers = identifiers[identifiers.str.strip().ne("")]
    return int(identifiers.nunique()) if len(identifiers) else "NA"


def table_cohorts(results: Path) -> pd.DataFrame:
    specifications = (
        ("Curated BUSI", "Entrenamiento fuente", "source_train_manifest.csv"),
        ("Curated BUSI", "Ajuste fuente", "source_val_manifest.csv"),
        ("Curated BUSI", "Prueba interna", "source_test_manifest.csv"),
        ("BUS-BRA", "Adaptación", "target_adapt_manifest.csv"),
        ("BUS-BRA", "Calibración", "target_calibration_manifest.csv"),
        ("BUS-BRA", "Prueba externa histórica", "target_test_manifest.csv"),
    )
    rows: list[dict[str, object]] = []
    for dataset, cohort, filename in specifications:
        frame = pd.read_csv(_require(results / "splits" / filename))
        rows.append(
            {
                "dataset": dataset,
                "cohort": cohort,
                "images": int(len(frame)),
                "benign_images": int(frame["label_idx"].astype(int).eq(0).sum()),
                "malignant_images": int(frame["label_idx"].astype(int).eq(1).sum()),
                "patients": _patient_count(frame),
                "benign_patients": _patient_count(frame, 0),
                "malignant_patients": _patient_count(frame, 1),
            }
        )
    return pd.DataFrame(rows)


def table_finetune_assignments(results: Path, *, seeds: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        frame = pd.read_csv(_require(results / "splits" / f"finetune_assignments_seed{seed}.csv"))
        selected = frame[frame["selected"].astype(str).str.lower().eq("true")].copy()
        for budget in ("05pct", "10pct", "20pct"):
            for role in ("train", "val"):
                subset = selected[
                    selected["budget"].astype(str).eq(budget)
                    & selected["role"].astype(str).eq(role)
                ]
                rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "role": role,
                        "patients": int(subset["patient_id"].astype(str).nunique()),
                        "images": int(len(subset)),
                        "benign_patients": int(
                            subset[subset["label_idx"].astype(int).eq(0)]["patient_id"]
                            .astype(str)
                            .nunique()
                        ),
                        "malignant_patients": int(
                            subset[subset["label_idx"].astype(int).eq(1)]["patient_id"]
                            .astype(str)
                            .nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def table_target_confusion_matrices(results: Path) -> pd.DataFrame:
    prediction_dir = results / "analysis" / "ensemble_predictions" / "target"
    rows: list[dict[str, object]] = []
    for architecture in ARCH_LABELS:
        for method in ENSEMBLE_METHODS:
            path = _require(prediction_dir / f"{architecture}_{method}_target_test_patients.csv")
            frame = pd.read_csv(path)
            required = {"label_idx", "probability_calibrated", "threshold_calibrated"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing confusion-matrix columns: {sorted(missing)}")
            labels = frame["label_idx"].to_numpy(dtype=int)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"{path} contains non-binary labels.")
            thresholds = frame["threshold_calibrated"].to_numpy(dtype=float)
            if not np.isfinite(thresholds).all() or not np.allclose(
                thresholds,
                thresholds[0],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{path} does not contain one finite frozen threshold.")
            probabilities = frame["probability_calibrated"].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all():
                raise ValueError(f"{path} contains non-finite calibrated probabilities.")
            predictions = (probabilities >= thresholds[0]).astype(int)
            tn = int(((labels == 0) & (predictions == 0)).sum())
            fp = int(((labels == 0) & (predictions == 1)).sum())
            fn = int(((labels == 1) & (predictions == 0)).sum())
            tp = int(((labels == 1) & (predictions == 1)).sum())
            if tn + fp + fn + tp != len(frame):
                raise ValueError(f"{path} produced an incomplete confusion matrix.")
            rows.append(
                {
                    "arch": architecture,
                    "method": method,
                    "n_test_patients": int(len(frame)),
                    "threshold_calibrated": float(thresholds[0]),
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                }
            )
    return pd.DataFrame(rows)


def _format_interval(row: pd.Series) -> str:
    return f"{float(row['estimate']):.3f} ({float(row['ci_low']):.3f}–{float(row['ci_high']):.3f})"


def build_tables(
    results: Path,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    cohorts = table_cohorts(results)
    seeds = sorted(tables["per_seed_metrics"]["seed"].astype(int).unique())
    assignments = table_finetune_assignments(results, seeds=seeds)
    confusion_matrices = table_target_confusion_matrices(results)
    table_map: dict[str, pd.DataFrame] = {
        "tabla_1_cohortes.csv": cohorts,
        "tabla_s1_asignaciones_ajuste_fino.csv": assignments,
        "tabla_s6_matrices_confusion.csv": confusion_matrices,
    }

    target_auc = tables["ensemble_metrics"][
        tables["ensemble_metrics"]["metric"].eq("auc_raw")
    ].copy()
    seed_auc = tables["per_seed_summary"][
        tables["per_seed_summary"]["metric"].eq("auc_raw")
        & tables["per_seed_summary"]["method"].isin(ENSEMBLE_METHODS)
    ][["arch", "method", "mean", "sd"]]
    target_auc = target_auc.merge(
        seed_auc,
        on=["arch", "method"],
        how="left",
        validate="one_to_one",
    )
    target_auc["auc_ensemble_ic95"] = target_auc.apply(_format_interval, axis=1)
    target_auc["auc_seed_mean_sd"] = target_auc.apply(
        lambda row: f"{float(row['mean']):.3f} ± {float(row['sd']):.3f}",
        axis=1,
    )
    table_map["tabla_2_auc_objetivo.csv"] = target_auc[
        [
            "arch",
            "method",
            "n_test_patients",
            "n_test_positive",
            "auc_ensemble_ic95",
            "auc_seed_mean_sd",
        ]
    ]

    uda = tables["paired_bootstrap_differences"][
        tables["paired_bootstrap_differences"]["metric"].eq("auc_raw")
    ].copy()
    delong = tables["delong_holm"][["arch", "method", "p_value", "p_holm", "reject_holm_0_05"]]
    uda = uda.merge(delong, on=["arch", "method"], how="left", validate="one_to_one")
    uda["delta_auc_ic95"] = uda.apply(
        lambda row: (
            f"{float(row['estimate_difference']):+.3f} "
            f"({float(row['ci_low']):+.3f}–{float(row['ci_high']):+.3f})"
        ),
        axis=1,
    )
    table_map["tabla_3_diferencias_uda.csv"] = uda[
        [
            "arch",
            "method",
            "comparator",
            "delta_auc_ic95",
            "bootstrap_p_value",
            "p_value",
            "p_holm",
            "reject_holm_0_05",
        ]
    ]

    fine = tables["finetune_summary"][tables["finetune_summary"]["metric"].eq("auc_raw")].copy()
    fine["auc_mean_sd"] = fine.apply(
        lambda row: f"{float(row['mean']):.3f} ± {float(row['sd']):.3f}",
        axis=1,
    )
    table_map["tabla_4_ajuste_fino.csv"] = fine[
        ["arch", "method", "n_seeds", "auc_mean_sd", "replicate_variation"]
    ]

    table_map["tabla_s2_metricas_objetivo_completas.csv"] = tables["ensemble_metrics"]
    table_map["tabla_s3_metricas_fuente.csv"] = tables["source_ensemble_metrics"]
    table_map["tabla_s4_metricas_por_replica.csv"] = tables["per_seed_metrics"]
    table_map["tabla_s5_calibracion.csv"] = tables["target_calibration_diagnostics"]

    for filename, table in table_map.items():
        path = output_dir / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    return written


def _all_source_files(results: Path) -> Iterable[Path]:
    yield results / "FINAL_INFERENCE_LOCK.json"
    yield results / "effective_config.yaml"
    for directory in (results / "analysis", results / "splits"):
        for path in sorted(directory.glob("*.csv")):
            yield path
        for path in sorted(directory.glob("*.json")):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-resolution publication figures and editable tables"
    )
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    results = Path(args.results).resolve()
    _, tables = _load_locked_inputs(results)
    figure_dir = results / "figures"
    table_dir = results / "publication_tables"

    written: list[Path] = []
    written.extend(figure_cohort_flow(figure_dir))
    written.extend(figure_target_auc(tables["ensemble_metrics"], figure_dir))
    written.extend(
        figure_uda_differences(
            tables["paired_bootstrap_differences"],
            tables["delong_holm"],
            figure_dir,
        )
    )
    written.extend(
        figure_finetune(
            tables["finetune_replica_metrics"],
            tables["finetune_summary"],
            figure_dir,
        )
    )
    written.extend(figure_reliability(results, figure_dir))
    written.extend(build_tables(results, tables, table_dir))

    sources = {
        str(path.relative_to(results)).replace("\\", "/"): sha256_file(path)
        for path in _all_source_files(results)
        if path.is_file()
    }
    artifacts = {
        str(path.relative_to(results)).replace("\\", "/"): sha256_file(path) for path in written
    }
    manifest = {
        "generator": "scripts/14_generate_publication_artifacts.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "source_files": sources,
        "artifacts": artifacts,
    }
    manifest_path = results / "PUBLICATION_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] generated {len(written)} publication artifacts")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
