"""Pruebas de humo: verifican que el grafo de imports y las piezas clave funcionan sin GPU ni datos.

Ejecuta:  pytest -q
No entrenan nada pesado; usan tensores aleatorios y un mini-DataFrame sintético.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image


def test_config_loads():
    from src.config import load_config
    cfg = load_config("config/config.yaml")
    assert cfg.model.num_classes == 2
    assert cfg.preprocessing.image_size == 224


def test_build_model_forward():
    from src.config import load_config
    from src.models.architectures import build_model
    cfg = load_config("config/config.yaml")
    cfg.model["pretrained"] = False          # evita descargar pesos en CI
    cfg.model["architecture"] = "resnet50"
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)


def test_dann_forward_returns_two_heads():
    from src.config import load_config
    from src.models.architectures import build_model
    cfg = load_config("config/config.yaml")
    cfg.model["pretrained"] = False
    model = build_model(cfg, dann=True)
    cls, dom = model(torch.randn(2, 3, 224, 224), lambda_=0.5)
    assert cls.shape == (2, 2) and dom.shape == (2, 2)


def test_metrics_basic():
    from src.evaluation.metrics import compute_metrics
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert 0.99 <= m["auc"] <= 1.0


def test_cv_splits_no_patient_leakage():
    """With patient IDs, no patient should appear in more than one split partition."""
    from src.config import load_config
    from src.data.datasets import make_cv_splits
    cfg = load_config("config/config.yaml")
    n = 200
    df = pd.DataFrame({
        "label_idx": np.random.randint(0, 2, n),
        "patient_id": [f"p{i // 2}" for i in range(n)],   # 2 imágenes por paciente
    })
    folds = make_cv_splits(df, cfg)
    for split in folds:
        groups = {name: set(df.iloc[split[name]]["patient_id"]) for name in
                  ("train_idx", "val_idx", "test_idx")}
        assert groups["train_idx"].isdisjoint(groups["test_idx"])
        assert groups["val_idx"].isdisjoint(groups["test_idx"])


def test_busi_manifest_includes_mask_and_bbox_columns(tmp_path):
    from src.config import load_config
    from src.data.preprocessing import build_busi_manifest

    cfg = load_config("config/config_smoke.yaml")
    cfg.paths["data_processed"] = str(tmp_path / "processed")
    cfg.datasets.busi["raw_dir"] = str(tmp_path / "raw" / "BUSI")
    raw = tmp_path / "raw" / "BUSI" / "benign"
    raw.mkdir(parents=True)
    arr = np.full((16, 16), 80, dtype=np.uint8)
    Image.fromarray(arr).save(raw / "benign (1).png")
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(raw / "benign (1)_mask.png")

    df = build_busi_manifest(cfg)
    assert {"mask_path", "bbox"}.issubset(df.columns)
    assert df.loc[0, "mask_path"].endswith("benign (1)_mask.png")
    assert df.loc[0, "bbox"] == ""


def test_target_adaptation_split_has_no_patient_leakage():
    from src.config import load_config
    from src.data.datasets import make_target_adaptation_split

    cfg = load_config("config/config.yaml")
    rows = []
    for i in range(40):
        rows.append({"patient_id": f"p{i}", "label_idx": i % 2})
        rows.append({"patient_id": f"p{i}", "label_idx": i % 2})
    df = pd.DataFrame(rows)
    split = make_target_adaptation_split(df, cfg, test_fraction=0.25)
    adapt_patients = set(df.iloc[split["adapt_idx"]]["patient_id"])
    test_patients = set(df.iloc[split["test_idx"]]["patient_id"])
    assert adapt_patients.isdisjoint(test_patients)
    assert len(split["test_idx"]) > 0


def test_finetune_fraction_selects_stratified_subset():
    from src.training.domain_adaptation import _stratified_fraction_indices

    df = pd.DataFrame({"label_idx": [0] * 100 + [1] * 40})
    used = _stratified_fraction_indices(df, 0.10, seed=42)
    selected = df.loc[used]
    assert len(selected[selected["label_idx"] == 0]) == 10
    assert len(selected[selected["label_idx"] == 1]) == 4


def test_drop_table_calculates_absolute_and_percent_drop():
    from src.utils.article_report import compute_drop_table

    metrics = pd.DataFrame([
        {"phase": "baseline_internal", "dataset": "busi", "arch": "resnet18", "auc": 0.9},
        {
            "phase": "external_all", "source": "busi", "target": "bus_bra",
            "arch": "resnet18", "auc": 0.72,
        },
    ])
    drop = compute_drop_table(metrics, "busi", "bus_bra")
    auc_drop = drop[drop["metric"] == "auc"].iloc[0]
    assert np.isclose(auc_drop["absolute_drop"], 0.18)
    assert np.isclose(auc_drop["percent_drop"], 20.0)


def test_article_report_generates_markdown_with_na(tmp_path):
    from src.utils.article_report import write_article_reports

    outputs = write_article_reports(
        tmp_path,
        dataset_summary=pd.DataFrame([{"dataset": "busi", "n_images": 2}]),
        metrics_df=pd.DataFrame(),
        drop_df=pd.DataFrame(),
        recovery_df=pd.DataFrame(),
        status_df=pd.DataFrame([{"experiment_id": "x", "status": "failed"}]),
    )
    assert outputs["markdown"].exists()
    assert outputs["article_materials"].exists()
    assert "NA" in outputs["markdown"].read_text(encoding="utf-8")


def test_gradcam_runs_minimal(tmp_path):
    from src.config import load_config
    from src.explainability import run_gradcam

    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)
            self.arch = "tiny"

        def forward(self, x):
            feat = torch.relu(self.conv(x))
            return self.fc(self.pool(feat).flatten(1))

    cfg = load_config("config/config_smoke.yaml")
    cfg.device = "cpu"
    cfg.preprocessing["image_size"] = 32
    cfg.explainability["n_samples"] = 1
    cfg.training["mixed_precision"] = False

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 180
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    Image.fromarray(img).save(tmp_path / "img.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    df = pd.DataFrame([{
        "image_path": "img.png",
        "mask_path": "mask.png",
        "dataset": "busi",
        "label": "malignant",
        "label_idx": 1,
        "original_path": "",
    }])
    out = run_gradcam(TinyCNN(), cfg, df, tmp_path, tmp_path / "gradcam", tag="test")
    assert out["grid_path"]
    assert (tmp_path / "gradcam" / "gradcam_test_grid.png").exists()
