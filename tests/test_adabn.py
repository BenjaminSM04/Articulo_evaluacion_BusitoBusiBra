"""AdaBN is a label-blind, deterministic pass over target adaptation images."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from src.training.adabn import order_target_adapt, refresh_batchnorm_statistics


def test_target_order_is_stable_and_rejects_duplicate_ids() -> None:
    frame = pd.DataFrame({"sample_id": ["z", "a", "m"], "label_idx": [1, 0, 1]})
    ordered = order_target_adapt(frame)
    assert ordered["sample_id"].tolist() == ["a", "m", "z"]
    assert ordered["label_idx"].tolist() == [-1, -1, -1]
    assert ordered["label"].tolist() == ["unlabelled"] * 3
    with pytest.raises(ValueError, match="sample_id"):
        order_target_adapt(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(p=0.95)
        self.seen_training_states: list[tuple[bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_training_states.append((self.bn.training, self.dropout.training))
        return self.dropout(self.bn(x))


def test_adabn_resets_stats_uses_cumulative_batches_without_gradients() -> None:
    model = _Probe()
    model.train()
    model.bn.running_mean.fill_(100.0)
    model.bn.running_var.fill_(9.0)
    model.bn.num_batches_tracked.fill_(7)
    inputs = [
        (torch.tensor([[1.0, 3.0], [3.0, 5.0]]),),
        (torch.tensor([[5.0, 7.0], [7.0, 9.0]]),),
    ]

    count = refresh_batchnorm_statistics(model, inputs, device="cpu")

    assert count == 4
    assert model.seen_training_states == [(True, False), (True, False)]
    assert model.bn.num_batches_tracked.item() == 2
    torch.testing.assert_close(model.bn.running_mean, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(model.bn.running_var, torch.tensor([2.0, 2.0]))
    assert model.training is False
    assert model.bn.training is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_adabn_rejects_empty_loader_and_model_without_batchnorm() -> None:
    with pytest.raises(ValueError, match="BatchNorm"):
        refresh_batchnorm_statistics(nn.Linear(2, 2), [(torch.ones(2, 2),)], device="cpu")
    with pytest.raises(ValueError, match="vacio"):
        refresh_batchnorm_statistics(_Probe(), [], device="cpu")
