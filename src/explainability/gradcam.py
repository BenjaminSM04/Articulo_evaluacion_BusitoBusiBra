"""Grad-CAM / Grad-CAM++ for the active BUSI -> BUS-BRA pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ..config import Config
from ..models.architectures import get_gradcam_target_layer
from ..training.transforms import build_eval_transforms
from .lesion_attention_analysis import energy_in_mask, iou_threshold, pointing_game, summarize

MASK_SEPARATOR = "|"


class _ClassLogitsOnly(torch.nn.Module):
    """Wrapper used by SHAP and Grad-CAM when a model returns auxiliary heads."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


class GradCAM:
    """Small, dependency-free Grad-CAM implementation for a single image batch."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        mode: str = "gradcam",
    ):
        self.model = model
        self.mode = mode
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._save_activation)
        self._h2 = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def _weights(self) -> torch.Tensor:
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM requires a completed forward/backward pass.")
        grads = self.gradients
        if self.mode == "gradcampp":
            g2 = grads.pow(2)
            g3 = g2 * grads
            sum_act = self.activations.sum(dim=(2, 3), keepdim=True)
            alpha = g2 / (2 * g2 + sum_act * g3 + 1e-8)
            return (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        return grads.mean(dim=(2, 3), keepdim=True)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x)
            out = out[0] if isinstance(out, tuple) else out
            if class_idx is None:
                class_idx = int(out.argmax(dim=1)[0].item())
            out[0, class_idx].backward()
        if self.activations is None:
            raise RuntimeError("No se capturaron activaciones para Grad-CAM.")
        cam = torch.relu((self._weights() * self.activations).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def _safe_tag(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))
    return value.strip("_") or "samples"


def _denormalize(tensor: torch.Tensor, cfg: Config) -> np.ndarray:
    """Convert a normalized CHW tensor back to HWC float image in [0, 1]."""
    arr = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.asarray(cfg.preprocessing.normalize_mean, dtype=np.float32)
    std = np.asarray(cfg.preprocessing.normalize_std, dtype=np.float32)
    arr = arr * std + mean
    return np.clip(arr, 0.0, 1.0)


def _read_rgb(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer imagen para Grad-CAM: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _split_mask_cell(cell: object, root: Path) -> list[Path]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    paths = []
    for part in str(cell).split(MASK_SEPARATOR):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        paths.append(p if p.is_absolute() else root / p)
    return paths


def _derive_mask_paths(row: pd.Series, root: Path) -> list[Path]:
    paths = _split_mask_cell(row.get("mask_path", ""), root)
    if paths:
        return paths

    original = row.get("original_path", "")
    if not original:
        return []
    original_path = Path(original)
    original_path = original_path if original_path.is_absolute() else root / original_path

    dataset = str(row.get("dataset", "")).lower()
    if dataset == "busi":
        return sorted(original_path.parent.glob(f"{original_path.stem}_mask*.png"))
    if dataset in {"bus_bra", "busbra"}:
        stem = original_path.stem.lower()
        mask_stem = f"mask_{stem.removeprefix('bus_')}"
        busbra_root = root / "data" / "raw" / "BUS-BRA"
        return sorted(busbra_root.rglob(f"{mask_stem}.png")) if busbra_root.exists() else []
    return []


def _load_combined_mask(row: pd.Series, root: Path, size: int) -> np.ndarray | None:
    masks = []
    for path in _derive_mask_paths(row, root):
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 0)
    if not masks:
        return None
    return np.logical_or.reduce(masks).astype(np.uint8)


def _sample_balanced(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    per_class = max(1, n // 2)
    sampled = []
    for lab in (0, 1):
        sub = df[df["label_idx"] == lab]
        if len(sub):
            take = min(per_class, len(sub))
            sampled.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    out = pd.concat(sampled) if sampled else df.head(0)
    if len(out) < n:
        remaining = df.drop(out.index, errors="ignore")
        if len(remaining):
            extra = remaining.iloc[
                rng.choice(
                    len(remaining),
                    size=min(n - len(out), len(remaining)),
                    replace=False,
                )
            ]
            out = pd.concat([out, extra])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def _overlay(rgb01: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.5 * rgb01 + 0.5 * heat, 0.0, 1.0)


def _save_grid(images: list[np.ndarray], titles: list[str], path: Path) -> None:
    if not images:
        return
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_gradcam(
    model: torch.nn.Module,
    cfg: Config,
    df: pd.DataFrame,
    root: Path,
    out_dir: Path,
    tag: str = "",
    mode: str = "gradcam",
) -> dict:
    """Generate Grad-CAM overlays and lesion-localization metrics when masks exist."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_tag(tag)
    size = int(cfg.preprocessing.image_size)
    n_samples = int(cfg.explainability.n_samples)
    sample = _sample_balanced(df, n_samples, cfg.seed)
    transform = build_eval_transforms(cfg)

    model.to(cfg.device).eval()
    cam_engine = GradCAM(model, get_gradcam_target_layer(model), mode=mode)
    overlays: list[np.ndarray] = []
    titles: list[str] = []
    loc_records: list[dict] = []

    try:
        for i, row in sample.iterrows():
            img_path = Path(row["image_path"])
            img_path = img_path if img_path.is_absolute() else Path(root) / img_path
            rgb = _read_rgb(img_path, size)
            x = transform(image=rgb)["image"].unsqueeze(0).to(cfg.device)
            class_idx = int(row.get("label_idx", 1))
            cam = cam_engine(x, class_idx=class_idx)
            cam = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
            overlay = _overlay(rgb.astype(np.float32) / 255.0, cam)
            label = str(row.get("label", class_idx))
            pred_title = f"{label} | {Path(row['image_path']).name}"
            overlays.append(overlay)
            titles.append(pred_title)

            single_path = out_dir / f"gradcam_{tag}_{i:03d}_{label}.png"
            plt.imsave(single_path, overlay)

            mask = _load_combined_mask(row, Path(root), size)
            if mask is not None:
                loc_records.append({
                    "sample_index": int(i),
                    "image_path": str(row["image_path"]),
                    "label": label,
                    "energy_in_mask": energy_in_mask(cam, mask),
                    "pointing_game": pointing_game(cam, mask),
                    "iou": iou_threshold(cam, mask, 0.5),
                    "overlay_path": str(single_path),
                })
    finally:
        cam_engine.remove()

    grid_path = out_dir / f"gradcam_{tag}_grid.png"
    _save_grid(overlays, titles, grid_path)
    loc_csv = out_dir / f"gradcam_{tag}_localization.csv"
    summary_path = out_dir / f"gradcam_{tag}_summary.json"

    summary = {}
    if loc_records:
        pd.DataFrame(loc_records).to_csv(loc_csv, index=False)
        records_for_summary = [
            {"energy": r["energy_in_mask"], "pointing": r["pointing_game"], "iou": r["iou"]}
            for r in loc_records
        ]
        summary = summarize(records_for_summary)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "tag": tag,
            "n_samples": int(len(sample)),
            "grid_path": str(grid_path),
            "localization_csv": str(loc_csv) if loc_records else "",
            "summary": summary,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[gradcam] -> {grid_path}")
    return {
        "grid_path": str(grid_path),
        "localization_csv": str(loc_csv) if loc_records else "",
        "summary_path": str(summary_path),
        "summary": summary,
    }
