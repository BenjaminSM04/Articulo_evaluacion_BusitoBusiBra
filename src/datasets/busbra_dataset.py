"""Índice del dataset BUS-BRA (Brasil).

Estructura esperada (Kaggle: orvile/bus-bra-a-breast-ultrasound-dataset; Zenodo 8231412):

    data/BUS-BRA/
        Images/     bus_0001-l.png, ...
        Masks/      mask_bus_0001-l.png, ...
        bus_data.csv   (metadatos: ID, Case, Pathology, BIRADS, Device, ...)

Puntos metodológicos:
  * BUS-BRA SÍ provee identificador de paciente (columna 'Case') -> split POR PACIENTE.
  * Etiqueta de la columna 'Pathology' (benign/malignant); confirmada por biopsia.
  * Máscaras disponibles -> se usan para métricas de localización de Grad-CAM.

El parser es robusto a variaciones de nombres de columnas/carpetas; si algo no se
detecta, lanza un error claro con instrucciones.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import INDEX_COLUMNS, LABEL_MAP

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _autodetect_csv(root: Path) -> Path:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontró CSV de metadatos en {root}. Ver README.")
    # Prioriza un CSV cuyo nombre sugiera metadatos.
    for c in csvs:
        if any(k in c.name.lower() for k in ("bus", "data", "meta", "label")):
            return c
    return csvs[0]


def _pick_column(columns: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in columns:
            return columns[cand]
    # coincidencia parcial
    for low, orig in columns.items():
        if any(cand in low for cand in candidates):
            return orig
    return None


def _resolve_file(root: Path, image_id: str, kind: str) -> str | None:
    """Localiza la imagen o máscara del caso ``image_id``.

    kind: 'image' | 'mask'. Prueba subcarpetas y prefijos habituales.
    """
    stem = Path(str(image_id)).stem
    subdirs = (["Images", "images", "IMAGES", "."] if kind == "image"
               else ["Masks", "masks", "MASKS", "."])
    prefixes = [""] if kind == "image" else ["mask_", "", "Mask_"]
    for sub in subdirs:
        base = root / sub if sub != "." else root
        if not base.exists():
            continue
        for pref in prefixes:
            for ext in IMG_EXTS:
                cand = base / f"{pref}{stem}{ext}"
                if cand.exists():
                    return str(cand)
    # último recurso: búsqueda recursiva
    hits = [p for p in root.rglob(f"*{stem}*") if p.suffix.lower() in IMG_EXTS
            and (("mask" in p.name.lower()) == (kind == "mask"))]
    return str(hits[0]) if hits else None


def _derive_patient(image_id: str) -> str:
    """Deriva un ID de paciente del nombre del caso (fallback si no hay columna 'Case')."""
    stem = Path(str(image_id)).stem
    # bus_0001-l -> 0001 ; quita sufijo de lado y no-dígitos
    digits = "".join(ch for ch in stem.split("-")[0] if ch.isdigit())
    return digits or stem


def build_busbra_index(cfg: dict, logger=None) -> pd.DataFrame:
    """Construye el DataFrame de índice de BUS-BRA (benigno/maligno, con patient_id)."""
    root = Path(cfg["data"]["busbra_root"])
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Descarga BUS-BRA y colócalo en data/BUS-BRA/. Ver README.")

    csv_cfg = cfg["data"].get("busbra_csv")
    csv_path = (root / csv_cfg) if csv_cfg else _autodetect_csv(root)
    meta = pd.read_csv(csv_path)
    columns = {c.lower().strip(): c for c in meta.columns}

    id_col = _pick_column(columns, ["id", "image", "filename", "file", "name"])
    path_col = _pick_column(columns, ["pathology", "label", "class", "diagnosis"])
    case_col = _pick_column(columns, ["case", "patient", "subject"])
    if id_col is None or path_col is None:
        raise KeyError(
            f"No pude identificar columnas de ID/Patología en {csv_path.name}. "
            f"Columnas: {list(meta.columns)}. Ajusta data.busbra_csv o renómbralas.")

    rows = []
    skipped = 0
    for _, r in meta.iterrows():
        label = str(r[path_col]).strip().lower()
        if label not in LABEL_MAP:            # descarta 'normal' u otros
            skipped += 1
            continue
        image_id = r[id_col]
        image_path = _resolve_file(root, image_id, "image")
        if image_path is None:
            skipped += 1
            continue
        patient = str(r[case_col]) if case_col else _derive_patient(image_id)
        rows.append({
            "image_path": image_path,
            "mask_path": _resolve_file(root, image_id, "mask"),
            "label": label,
            "label_idx": LABEL_MAP[label],
            "patient_id": patient,
            "dataset": "busbra",
        })

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df.empty:
        raise FileNotFoundError(
            f"No se pudieron enlazar imágenes de BUS-BRA en {root}. Revisa la estructura/CSV.")
    if logger:
        logger.info("BUS-BRA: %d imágenes, %d pacientes, %d descartadas (%s)",
                    len(df), df["patient_id"].nunique(), skipped,
                    df["label"].value_counts().to_dict())
    return df
