from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import log_loss

from src.config import Config
from src.evaluation import publication_protocol as inference_module
from src.evaluation.publication_protocol import (
    aggregate_patients,
    apply_temperature,
    calibrate_from_patient_table,
    fit_temperature,
    risk_ece,
    sigmoid,
)
from src.training.publication_protocol import (
    _cosine_scheduler,
    _freeze_batchnorm_running_stats,
)


def _image_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a1", "a2", "b1", "b2"],
            "patient_id": ["a", "a", "b", "b"],
            "label": ["benign", "benign", "malignant", "malignant"],
            "label_idx": [0, 0, 1, 1],
            "logit_difference": [-2.0, 0.0, 0.0, 2.0],
        }
    )


def test_patient_aggregation_uses_mean_logit() -> None:
    patients = aggregate_patients(_image_predictions())
    assert patients["patient_id"].tolist() == ["a", "b"]
    assert patients["logit_difference"].tolist() == [-1.0, 1.0]
    assert patients["n_images"].tolist() == [2, 2]
    np.testing.assert_allclose(
        patients["probability_raw"].to_numpy(),
        sigmoid(np.array([-1.0, 1.0])),
    )


def test_patient_aggregation_rejects_mixed_labels() -> None:
    frame = _image_predictions()
    frame.loc[1, "label_idx"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_patients(frame)


def test_temperature_scaling_improves_overconfident_nll() -> None:
    logits = np.array([-8.0, -6.0, -4.0, 4.0, 6.0, 8.0])
    labels = np.array([0, 0, 1, 0, 1, 1])
    temperature = fit_temperature(logits, labels)
    before = log_loss(labels, sigmoid(logits), labels=[0, 1])
    after = log_loss(labels, sigmoid(logits / temperature), labels=[0, 1])
    assert temperature > 1.0
    assert after < before


def test_calibration_is_fit_on_patient_table_and_applied_without_relabelling() -> None:
    patients = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(8)],
            "label_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "logit_difference": [-3.0, -2.0, -1.0, 0.5, -0.5, 1.0, 2.0, 3.0],
        }
    )
    calibration = calibrate_from_patient_table(patients)
    transformed = apply_temperature(patients, calibration["temperature"])
    assert transformed["label_idx"].tolist() == patients["label_idx"].tolist()
    assert transformed["probability_calibrated"].between(0, 1).all()
    assert 0 <= calibration["threshold_calibrated"] <= 1


def test_risk_ece_is_zero_for_exact_binwise_risks() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert risk_ece(labels, probabilities, bins=10) == pytest.approx(0.0)


def test_scheduler_applies_declared_warmup_and_cosine_decay() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.3)
    scheduler = _cosine_scheduler(optimizer, epochs=6, warmup_epochs=3)

    observed = []
    for _ in range(6):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert observed[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert observed[-1] < observed[-2] < observed[-3]


def test_freezing_batchnorm_preserves_running_statistics() -> None:
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    model.train()
    _freeze_batchnorm_running_stats(model)
    batchnorm = model[0]
    before = batchnorm.running_mean.clone()
    model(torch.full((8, 3), 5.0)).sum().backward()

    assert torch.equal(batchnorm.running_mean, before)
    assert batchnorm.weight.grad is not None


@pytest.mark.parametrize("preprocessing", ["none", "intensity", "roi"])
def test_inference_uses_checkpoint_preprocessing(monkeypatch, tmp_path, preprocessing) -> None:
    observed = []

    class SyntheticDataset:
        def __init__(self, frame, root, transform, *, preprocessing):
            observed.append(preprocessing)
            self.frame = frame

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            return torch.zeros(3, 4, 4), 0, index

    class ConstantModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(len(images), 2)

    monkeypatch.setattr(inference_module, "UltrasoundDataset", SyntheticDataset)
    monkeypatch.setattr(inference_module, "build_eval_transforms", lambda cfg: None)
    monkeypatch.setattr(
        inference_module,
        "load_publication_model",
        lambda checkpoint_path, cfg: (ConstantModel(), {}),
    )
    cfg = Config({
        "device": "cpu",
        "training": {"num_workers": 0, "batch_size": 2, "mixed_precision": False},
    })
    cfg._control_preprocessing = preprocessing
    frame = pd.DataFrame({
        "sample_id": ["synthetic-1"], "image_path": ["unused.png"],
        "label": ["benign"], "label_idx": [0],
    })

    output, _ = inference_module.infer_logits(tmp_path / "unused.pt", frame, tmp_path, cfg)

    assert observed == [preprocessing]
    assert output.sample_id.tolist() == ["synthetic-1"]
