"""Hook contracts for all-layer frozen relation capture."""

from __future__ import annotations

import torch
from torch import nn

from analysis.relational_selector.layer_capture import LayerRelationCollector


class _Block(nn.Module):
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tokens + 0.1, torch.empty(0, device=tokens.device)


class _TwelveBlockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(12))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens, _ = block(tokens)
        return tokens


def test_capture_has_all_12_raw_relation_shapes_and_no_grad() -> None:
    model = _TwelveBlockModel()
    collector = LayerRelationCollector(model)
    tokens = torch.randn(2, 804, 384, requires_grad=True)
    labels = torch.zeros(2, 20)
    labels[0, [1, 3]] = 1
    labels[1, 5] = 1
    collector.set_positive_labels(labels)
    _ = model(tokens)
    records = collector.consume()
    collector.close()
    assert len(records) == 12
    for record in records:
        assert record.raw.shape == (2, 20, 784)
        assert record.residual.shape == (2, 20, 784)
        assert record.common.shape == (2, 784)
        assert record.positive_common.shape == (2, 784)
        assert record.mean_class.shape == (2, 384)
        assert not record.raw.requires_grad
        assert torch.isfinite(record.positive_common[0]).all()
        assert torch.isnan(record.positive_common[1]).all()
