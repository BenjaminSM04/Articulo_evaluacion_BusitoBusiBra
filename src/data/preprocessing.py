"""Preprocessing: deduplicate, resize, unify labels and build manifest CSVs.

The pipeline writes, for each dataset, a manifest CSV with one row per usable image:

    image_path, dataset, label, label_idx, patient_id, birads, original_path

Manifests are the interface between preprocessing and everything downstream — training and
evaluation only ever read manifests, never the raw folders.

Deduplication is mandatory for BUSI (~235 known duplicates). Near-duplicates are detected with
perceptual hashing (pHash) and the dropped files are logged to ``dedup_report.csv``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
import pandas as pd
from PIL import Image

from ..config import Config

LABEL_TO_IDX = {"benign": 0, "malignant": 1}
MASK_SEPARATOR = "|"


def _relative(cfg: Config, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(cfg._root))
    except ValueError:
        return str(path)


def _join_mask_paths(cfg: Config, paths: list[Path]) -> str:
    return MASK_SEPARATOR.join(_relative(cfg, p) for p in sorted(paths))


def _busi_mask_paths(image_path: Path) -> list[Path]:
    """Return all BUSI masks associated with an image, including multi-mask lesions."""
    return sorted(image_path.parent.glob(f"{image_path.stem}_mask*.png"))


def _busbra_mask_key(mask_path: Path) -> str:
    stem = mask_path.stem.lower()
    if stem.startswith("mask_"):
        return f"bus_{stem.removeprefix('mask_')}"
    return stem


# ---------------------------------------------------------------------------
# Deduplication (perceptual hashing)
# ---------------------------------------------------------------------------
def perceptual_hash(path: Path, hash_size: int = 16) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img.convert("L"), hash_size=hash_size)


def find_duplicates(paths: list[Path], hash_size: int = 16, max_distance: int = 5):
    """Return (kept_paths, dropped_records).

    Greedy: iterate in sorted order, keep an image unless it is within ``max_distance`` Hamming
    distance of an already-kept image. O(n^2) which is fine for these dataset sizes (<2k images).
    """
    hashes: dict[Path, imagehash.ImageHash] = {}
    for p in sorted(paths):
        try:
            hashes[p] = perceptual_hash(p, hash_size)
        except Exception as exc:  # corrupt file
            print(f"[dedup] no se pudo leer {p}: {exc}")

    kept: list[Path] = []
    kept_hashes: list[tuple[Path, imagehash.ImageHash]] = []
    dropped: list[dict] = []
    for p, h in hashes.items():
        match = next(((kp, kh) for kp, kh in kept_hashes if (h - kh) <= max_distance), None)
        if match is None:
            kept.append(p)
            kept_hashes.append((p, h))
        else:
            dropped.append({"dropped": str(p), "duplicate_of": str(match[0]),
                            "distance": int(h - match[1])})
    return kept, dropped


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------
def resize_and_save(src: Path, dst: Path, size: int, to_rgb: bool = True) -> None:
    """Resize keeping it square (cv2 INTER_AREA) and save as PNG."""
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if not to_rgb else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {src}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


# ---------------------------------------------------------------------------
# Manifest builders (one per dataset, because the layouts differ)
# ---------------------------------------------------------------------------
def _busi_raw_images(raw_dir: Path) -> list[tuple[Path, str]]:
    """List (image_path, label) for BUSI, excluding masks and the 'normal' class handling.

    BUSI layout: <raw>/{benign,malignant,normal}/<class> (N).png  and  ..._mask.png
    """
    items: list[tuple[Path, str]] = []
    for label in ("benign", "malignant"):
        folder = raw_dir / label
        if not folder.exists():
            print(f"[busi] aviso: no existe carpeta {folder}")
            continue
        for img in folder.glob("*.png"):
            if "_mask" in img.stem:          # excluir máscaras de segmentación
                continue
            items.append((img, label))
    return items


def build_busi_manifest(cfg: Config) -> pd.DataFrame:
    raw_dir = cfg.resolve(cfg.datasets.busi["raw_dir"])
    out_dir = cfg.path("data_processed") / "busi"
    cfg.path("data_processed").mkdir(parents=True, exist_ok=True)
    items = _busi_raw_images(raw_dir)
    paths = [p for p, _ in items]
    label_of = {p: lab for p, lab in items}

    rows = []
    if cfg.preprocessing.deduplicate:
        kept, dropped = find_duplicates(paths, cfg.preprocessing.dedup_hash_size,
                                        cfg.preprocessing.dedup_max_distance)
        pd.DataFrame(dropped).to_csv(cfg.path("data_processed") / "dedup_report.csv", index=False)
        print(f"[busi] deduplicación: {len(paths)} -> {len(kept)} "
              f"({len(dropped)} duplicados eliminados; ver dedup_report.csv)")
    else:
        kept = paths

    for p in kept:
        label = label_of[p]
        dst = out_dir / label / p.name
        resize_and_save(p, dst, cfg.preprocessing.image_size, cfg.preprocessing.to_rgb)
        rows.append({
            "image_path": _relative(cfg, dst),
            "dataset": "busi",
            "label": label,
            "label_idx": LABEL_TO_IDX[label],
            "patient_id": "",          # BUSI no provee ID de paciente -> split a nivel imagen
            "birads": "",
            "original_path": _relative(cfg, p),
            "mask_path": _join_mask_paths(cfg, _busi_mask_paths(p)),
            "bbox": "",
        })
    return pd.DataFrame(rows)


def build_bus_bra_manifest(cfg: Config) -> pd.DataFrame:
    """Build BUS-BRA manifest from its metadata CSV.

    Column names in the official CSV may vary slightly across releases, so we match them
    case-insensitively. Verify against your downloaded ``bus_data.csv`` if this raises.
    """
    raw_dir = cfg.resolve(cfg.datasets.bus_bra["raw_dir"])
    out_dir = cfg.path("data_processed") / "bus_bra"
    meta_name = cfg.datasets.bus_bra.get("metadata_csv", "bus_data.csv")

    csv_path = next(raw_dir.rglob(meta_name), None) or next(raw_dir.rglob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"No se encontró el CSV de metadatos de BUS-BRA en {raw_dir}.")
    meta = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in meta.columns}

    def col(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    id_col = col("id", "image", "filename")
    path_col = col("pathology", "class", "label", "diagnosis")
    case_col = col("case", "patient", "patient_id", "subject")
    birads_col = col("birads", "bi-rads", "bi_rads")
    bbox_col = col("bbox", "bounding_box", "box")
    if id_col is None or path_col is None:
        raise KeyError(f"Columnas no reconocidas en {csv_path.name}: {list(meta.columns)}. "
                       "Ajusta build_bus_bra_manifest a tu versión del dataset.")

    # Index available image files by stem for robust matching.
    files = {f.stem.lower(): f for f in raw_dir.rglob("*.png") if "mask" not in f.stem.lower()}
    masks = {_busbra_mask_key(f): f for f in raw_dir.rglob("*.png")
             if f.stem.lower().startswith("mask_")}

    rows = []
    for _, r in meta.iterrows():
        raw_label = str(r[path_col]).strip().lower()
        if raw_label not in LABEL_TO_IDX:          # ignora valores fuera de benign/malignant
            continue
        stem = str(r[id_col]).strip().lower().replace(".png", "")
        src = files.get(stem)
        if src is None:                            # imagen referenciada en CSV pero ausente
            continue
        dst = out_dir / raw_label / src.name
        resize_and_save(src, dst, cfg.preprocessing.image_size, cfg.preprocessing.to_rgb)
        rows.append({
            "image_path": _relative(cfg, dst),
            "dataset": "bus_bra",
            "label": raw_label,
            "label_idx": LABEL_TO_IDX[raw_label],
            "patient_id": str(r[case_col]) if case_col else "",
            "birads": str(r[birads_col]) if birads_col else "",
            "original_path": _relative(cfg, src),
            "mask_path": _relative(cfg, masks.get(stem)),
            "bbox": str(r[bbox_col]) if bbox_col else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def preprocess_dataset(cfg: Config, key: str) -> Path:
    builders = {"busi": build_busi_manifest, "bus_bra": build_bus_bra_manifest}
    if key not in builders:
        raise ValueError(f"Dataset desconocido: {key}")
    df = builders[key](cfg)
    manifest = cfg.path("data_processed") / f"{key}_manifest.csv"
    df.to_csv(manifest, index=False)
    print(f"[{key}] manifest: {len(df)} imágenes -> {manifest}")
    print(df["label"].value_counts().to_string())
    return manifest


def preprocess_all(cfg: Config, only: str | None = None) -> None:
    keys = [only] if only else list(cfg.datasets.keys())
    for key in keys:
        preprocess_dataset(cfg, key)
