"""Numerical helpers for Experiment 1 class-specific patch scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def class_specific_patch_score(
    class_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return float32 cosine scores with shape ``[B, C, P]``.

    The two inputs must be post-block representations with shapes ``[B, C, D]``
    and ``[B, P, D]``.  Converting to float32 here keeps the score definition
    independent of any mixed-precision setting used by the host forward pass.
    """

    if class_tokens.ndim != 3 or patch_tokens.ndim != 3:
        raise ValueError(
            "class_tokens and patch_tokens must both be rank-3 tensors; "
            f"got {tuple(class_tokens.shape)} and {tuple(patch_tokens.shape)}"
        )
    if class_tokens.shape[0] != patch_tokens.shape[0]:
        raise ValueError("class and patch tensors must have the same batch size")
    if class_tokens.shape[-1] != patch_tokens.shape[-1]:
        raise ValueError("class and patch tensors must have the same embedding width")
    if class_tokens.shape[1] < 1 or patch_tokens.shape[1] < 1:
        raise ValueError("class and patch tensors must each contain at least one token")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    class_unit = F.normalize(class_tokens.float(), p=2, dim=-1, eps=eps)
    patch_unit = F.normalize(patch_tokens.float(), p=2, dim=-1, eps=eps)
    return torch.einsum("bcd,bpd->bcp", class_unit, patch_unit)


def _pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"expected one or two patch-size values, got {values}")
    return values


def infer_patch_grid(
    image_shape: Sequence[int],
    patch_size: Union[int, Sequence[int]],
    num_patches: int,
) -> Tuple[int, int]:
    """Infer ``(grid_h, grid_w)`` without assuming a square image."""

    if len(image_shape) < 2:
        raise ValueError(f"image_shape must contain H and W, got {image_shape}")
    height, width = int(image_shape[-2]), int(image_shape[-1])
    patch_h, patch_w = _pair(patch_size)
    if min(height, width, patch_h, patch_w, int(num_patches)) <= 0:
        raise ValueError("image, patch, and token dimensions must be positive")
    if height % patch_h or width % patch_w:
        raise ValueError(
            f"image {(height, width)} is not divisible by patch size {(patch_h, patch_w)}"
        )
    grid_h, grid_w = height // patch_h, width // patch_w
    if grid_h * grid_w != int(num_patches):
        raise ValueError(
            f"grid {(grid_h, grid_w)} implies {grid_h * grid_w} patches, "
            f"but the model produced {num_patches}"
        )
    return grid_h, grid_w


@dataclass
class _LayerAccumulator:
    finite_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    chunks: list[np.ndarray] = field(default_factory=list)

    def add(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        self.nan_count += int(np.isnan(flat).sum())
        self.inf_count += int(np.isinf(flat).sum())
        finite = flat[np.isfinite(flat)]
        if not finite.size:
            return
        finite64 = finite.astype(np.float64, copy=False)
        self.finite_count += int(finite.size)
        self.total += float(finite64.sum(dtype=np.float64))
        self.total_square += float(np.square(finite64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.chunks.append(finite.copy())

    def finish(self) -> dict:
        if not self.finite_count:
            return {
                "score_min": None,
                "score_max": None,
                "score_mean": None,
                "score_std": None,
                "score_q05": None,
                "score_q50": None,
                "score_q95": None,
                "finite_count": 0,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
            }
        mean = self.total / self.finite_count
        variance = max(0.0, self.total_square / self.finite_count - mean * mean)
        values = np.concatenate(self.chunks)
        quantiles = np.quantile(values, (0.05, 0.50, 0.95))
        self.chunks.clear()
        return {
            "score_min": self.minimum,
            "score_max": self.maximum,
            "score_mean": mean,
            "score_std": math.sqrt(variance),
            "score_q05": float(quantiles[0]),
            "score_q50": float(quantiles[1]),
            "score_q95": float(quantiles[2]),
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


class LayerScoreSummary:
    """Accumulate exact sanity statistics over saved positive-class maps."""

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self.depth = int(depth)
        self.layers = [_LayerAccumulator() for _ in range(self.depth)]
        self.num_images = 0
        self.num_positive_class_maps = 0

    def add_image(self, positive_scores: np.ndarray) -> None:
        scores = np.asarray(positive_scores, dtype=np.float32)
        if scores.ndim != 3 or scores.shape[0] != self.depth:
            raise ValueError(
                f"expected [L, K, P] with L={self.depth}, got {scores.shape}"
            )
        self.num_images += 1
        self.num_positive_class_maps += int(scores.shape[1])
        for layer, values in zip(self.layers, scores):
            layer.add(values)

    def finish(self, model_name: str) -> list[dict[str, object]]:
        rows = []
        for layer_index, accumulator in enumerate(self.layers, start=1):
            row: dict[str, object] = {
                "model": model_name,
                "layer": layer_index,
                "num_images": self.num_images,
                "num_positive_class_maps": self.num_positive_class_maps,
            }
            row.update(accumulator.finish())
            rows.append(row)
        return rows
