"""Memory-bounded post-block relation capture for frozen Task A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .scores import raw_dot_scores


@dataclass
class LayerRelations:
    """Only relation maps and class means; raw patch tokens are never cached."""

    raw: torch.Tensor  # [B,C,P] float32
    residual: torch.Tensor  # [B,C,P] float32
    common: torch.Tensor  # [B,P] float32
    positive_common: torch.Tensor  # [B,P], NaN for single-label images
    mean_class: torch.Tensor  # [B,D] float32


class LayerRelationCollector:
    """Observe native post-block tokens and immediately reduce them to Task A maps.

    The collector intentionally has no semantic-mask argument.  Image-level
    positive labels are optional and are used only for the preregistered G+
    diagnostic; they cannot affect raw, residual, or selector relations.
    """

    def __init__(self, model: torch.nn.Module, *, num_classes: int = 20, patch_count: int = 784, width: int = 384) -> None:
        self.num_classes = int(num_classes)
        self.patch_count = int(patch_count)
        self.width = int(width)
        self._labels: Optional[torch.Tensor] = None
        self.records: list[LayerRelations] = []
        self._handles = [block.register_forward_hook(self._hook) for block in model.blocks]

    def set_positive_labels(self, labels: torch.Tensor) -> None:
        if labels.ndim != 2 or labels.shape[1] != self.num_classes:
            raise ValueError("positive labels must be [B, num_classes]")
        self._labels = labels.detach().to(dtype=torch.bool)

    def clear(self) -> None:
        self.records.clear()
        self._labels = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("MCTformer+ block hook expected (tokens, attention)")
        tokens = output[0]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError("MCTformer+ block output tokens must be rank 3")
        expected = self.num_classes + self.patch_count
        if tokens.shape[1:] != (expected, self.width):
            raise RuntimeError(
                "native token contract expected [B,804,384], got "
                f"{tuple(tokens.shape)}"
            )
        classes = tokens[:, : self.num_classes].float()
        patches = tokens[:, self.num_classes :].float()
        raw = raw_dot_scores(classes, patches)
        mean_class = classes.mean(dim=1)
        common = torch.einsum("bd,bpd->bp", mean_class, patches) / math.sqrt(self.width)
        residual = raw - common[:, None, :]
        if self._labels is None or self._labels.shape[0] != tokens.shape[0]:
            raise RuntimeError("positive labels must be set before the frozen forward")
        labels = self._labels.to(device=tokens.device)
        positive_count = labels.sum(dim=1)
        # G+ is diagnostic only; make unavailable single-label rows explicit.
        positive_weights = labels.float()
        positive_mean = torch.einsum("bc,bcd->bd", positive_weights, classes)
        positive_mean = positive_mean / positive_count.clamp_min(1).float().unsqueeze(1)
        positive_common = torch.einsum("bd,bpd->bp", positive_mean, patches) / math.sqrt(self.width)
        positive_common = positive_common.masked_fill((positive_count < 2).unsqueeze(1), float("nan"))
        if not (torch.isfinite(raw).all() and torch.isfinite(residual).all() and torch.isfinite(common).all()):
            raise RuntimeError("Task A relation capture produced a non-finite value")
        self.records.append(
            LayerRelations(
                raw=raw.detach(),
                residual=residual.detach(),
                common=common.detach(),
                positive_common=positive_common.detach(),
                mean_class=mean_class.detach(),
            )
        )

    def consume(self) -> list[LayerRelations]:
        if len(self.records) != 12:
            raise RuntimeError(f"expected all 12 post-block captures, got {len(self.records)}")
        records = self.records
        self.records = []
        self._labels = None
        return records
