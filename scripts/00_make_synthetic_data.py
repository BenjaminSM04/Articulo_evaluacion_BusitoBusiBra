"""Paso 0 (opcional) — Generar datos SINTÉTICOS con la estructura de BUSI y BUS-BRA.

Permite validar TODO el pipeline (preprocesamiento -> entrenamiento -> matriz cross-domain ->
XAI -> reporte) **sin descargar nada de Kaggle y sin GPU**. Las imágenes son ruido con un "blob"
gaussiano: en las malignas el blob es más grande y brillante, de modo que existe una señal
aprendible y las métricas no son degeneradas. NO sustituye a los datos reales; es solo para
verificar que la maquinaria funciona en tu equipo.

Uso:
    python scripts/00_make_synthetic_data.py --config config/config.yaml
    # luego:  python scripts/02_preprocess.py  (omite el 01_download)

Tras esto puedes correr el flujo completo en modo rápido editando config.yaml
(p.ej. model.pretrained=false, training.epochs=1, device=cpu) o usando scripts/run_smoke.* .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _make_image(size: int, malignant: bool, rng: np.random.Generator) -> np.ndarray:
    """Random speckle background + a Gaussian blob. Malignant -> bigger/brighter blob."""
    img = rng.normal(40, 12, size=(size, size)).clip(0, 255)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
    radius = rng.uniform(size * 0.18, size * 0.28) if malignant else rng.uniform(size * 0.08, size * 0.15)
    peak = rng.uniform(150, 220) if malignant else rng.uniform(80, 130)
    blob = peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)))
    return (img + blob).clip(0, 255).astype(np.uint8)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def make_busi(raw_dir: Path, rng: np.random.Generator, n_benign=40, n_malignant=30, n_normal=8,
              n_duplicates=6) -> None:
    """BUSI layout: {benign,malignant,normal}/<class> (i).png + máscaras + duplicados."""
    specs = {"benign": (n_benign, False), "malignant": (n_malignant, True), "normal": (n_normal, False)}
    first_benign = None
    for label, (n, malignant) in specs.items():
        for i in range(1, n + 1):
            size = int(rng.integers(450, 560))
            arr = _make_image(size, malignant, rng)
            img_path = raw_dir / label / f"{label} ({i}).png"
            _save(arr, img_path)
            # máscara dummy (se ignora en clasificación pero imita el dataset real)
            mask = (arr > 120).astype(np.uint8) * 255
            _save(mask, raw_dir / label / f"{label} ({i})_mask.png")
            if label == "benign" and i == 1:
                first_benign = arr.copy()
    # Inyecta duplicados exactos de la primera benigna para probar la deduplicación.
    for d in range(n_benign + 1, n_benign + 1 + n_duplicates):
        _save(first_benign, raw_dir / "benign" / f"benign ({d}).png")
        m = (first_benign > 120).astype(np.uint8) * 255
        _save(m, raw_dir / "benign" / f"benign ({d})_mask.png")
    print(f"[synthetic] BUSI -> {raw_dir} (incluye {n_duplicates} duplicados para test de dedup)")


def make_bus_bra(raw_dir: Path, rng: np.random.Generator, n_patients=30) -> None:
    """BUS-BRA layout: Images/bus_XXXX-l.png + bus_data.csv (ID, Case, Pathology, BIRADS)."""
    rows = []
    img_dir = raw_dir / "Images"
    idx = 1
    for case in range(1, n_patients + 1):
        malignant = bool(rng.integers(0, 2))
        label = "malignant" if malignant else "benign"
        for _ in range(int(rng.integers(1, 3))):       # 1-2 imágenes por paciente
            stem = f"bus_{idx:04d}-l"
            size = int(rng.integers(450, 560))
            _save(_make_image(size, malignant, rng), img_dir / f"{stem}.png")
            rows.append({"ID": stem, "Case": f"case_{case:03d}", "Pathology": label,
                         "BIRADS": int(rng.integers(2, 6)), "Device": "synthetic"})
            idx += 1
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw_dir / "bus_data.csv", index=False)
    print(f"[synthetic] BUS-BRA -> {raw_dir} ({len(rows)} imágenes, {n_patients} pacientes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generar datos sintéticos para validar el pipeline")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    make_busi(cfg.resolve(cfg.datasets.busi["raw_dir"]), rng)
    make_bus_bra(cfg.resolve(cfg.datasets.bus_bra["raw_dir"]), rng)
    print("\n[ok] Datos sintéticos creados. Siguiente: python scripts/02_preprocess.py")


if __name__ == "__main__":
    main()
