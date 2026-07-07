"""Índice del dataset BUSI (Egipto).

Estructura esperada (Kaggle: aryashah2k/breast-ultrasound-images-dataset):

    data/BUSI/
        benign/     benign (1).png, benign (1)_mask.png, ...
        malignant/  malignant (1).png, malignant (1)_mask.png, ...
        normal/     (se EXCLUYE del estudio)

Puntos metodológicos:
  * Se excluye la clase 'normal' (tarea binaria benigno/maligno comparable con BUS-BRA).
  * BUSI NO provee identificador de paciente -> patient_id = None (split a nivel imagen).
  * Deduplicación pHash OBLIGATORIA: BUSI tiene ~235 duplicados; sin quitarlos hay fuga
    de datos que infla las métricas internas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import INDEX_COLUMNS, LABEL_MAP

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _is_mask(name: str) -> bool:
    return "_mask" in name.lower()


def _find_mask(img_path: Path) -> str | None:
    """Busca la máscara asociada a una imagen BUSI."""
    cand = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")
    if cand.exists():
        return str(cand)
    matches = sorted(img_path.parent.glob(f"{img_path.stem}_mask*"))
    return str(matches[0]) if matches else None


def deduplicate(df: pd.DataFrame, hamming: int = 5, logger=None) -> tuple[pd.DataFrame, list]:
    """Elimina duplicados por perceptual hashing (distancia de Hamming <= umbral).

    Devuelve (df_sin_duplicados, lista_de_rutas_descartadas). Si 'imagehash' no está
    instalado, avisa y no deduplica (¡instálalo para un estudio riguroso!).
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        if logger:
            logger.warning("imagehash no instalado: se OMITE la deduplicación de BUSI "
                           "(riesgo de fuga de datos). Instala 'imagehash'.")
        return df, []

    hashes: list = []
    keep: list = []
    dropped: list = []
    for idx, row in df.iterrows():
        with Image.open(row["image_path"]) as im:
            h = imagehash.phash(im.convert("L"))
        if any((h - kh) <= hamming for kh in hashes):
            dropped.append(row["image_path"])
        else:
            hashes.append(h)
            keep.append(idx)
    if logger:
        logger.info("Deduplicación BUSI: %d imágenes, %d duplicadas descartadas, %d conservadas",
                    len(df), len(dropped), len(keep))
    return df.loc[keep].reset_index(drop=True), dropped


def build_busi_index(cfg: dict, logger=None) -> pd.DataFrame:
    """Construye el DataFrame de índice de BUSI (benigno/maligno, deduplicado)."""
    root = Path(cfg["data"]["busi_root"])
    classes = [c for c in cfg["data"]["classes"] if c in LABEL_MAP]  # excluye 'normal'
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Descarga BUSI y colócalo en data/BUSI/ "
            f"(carpetas benign/ malignant/). Ver README.")

    rows = []
    for cls in classes:
        img_paths = [
            p for p in root.rglob("*")
            if p.suffix.lower() in IMG_EXTS
            and p.parent.name.lower() == cls.lower()
            and not _is_mask(p.name)
        ]
        for p in sorted(img_paths):
            rows.append({
                "image_path": str(p),
                "mask_path": _find_mask(p),
                "label": cls,
                "label_idx": LABEL_MAP[cls],
                "patient_id": None,          # BUSI no tiene ID de paciente
                "dataset": "busi",
            })

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df.empty:
        raise FileNotFoundError(
            f"No se encontraron imágenes benignas/malignas en {root}. Revisa la estructura.")

    if cfg["data"].get("busi_dedup", True):
        df, dropped = deduplicate(df, int(cfg["data"].get("dedup_hamming", 5)), logger)
        # Reporte de deduplicación para trazabilidad.
        out = Path(cfg["output"]["results_dir"]) / "busi_dedup_report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"dropped_duplicate": dropped}).to_csv(out, index=False)

    if logger:
        logger.info("BUSI: %d imágenes (%s)", len(df),
                    df["label"].value_counts().to_dict())
    return df
