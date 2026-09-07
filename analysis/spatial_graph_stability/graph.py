"""Local graph construction and Experiment-2-compatible patch GT labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from analysis.lazy_assignment.experiment2.patch_regions import (
    PATCH_LABEL_ID_TO_COLUMN,
    VOC_VOID_ID,
    patch_label_counts,
)


GRAPH_NAMES = ("B0_uniform", "B1_spatial", "B2_feature", "B3_spatial_feature")
PATCH_LABEL_MIXED = -1
PATCH_LABEL_VOID = -2


@dataclass(frozen=True)
class LocalEdges:
    source: np.ndarray
    target: np.ndarray
    squared_distance: np.ndarray
    grid_size: tuple[int, int]

    @property
    def count(self) -> int:
        return int(self.source.size)


def _grid_pair(grid_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(grid_size, (int, np.integer)):
        pair = (int(grid_size), int(grid_size))
    else:
        pair = tuple(int(value) for value in grid_size)
        if len(pair) != 2:
            raise ValueError(f"grid_size must contain two values, got {pair}")
    if min(pair) < 1:
        raise ValueError(f"grid dimensions must be positive, got {pair}")
    return pair


def local_edge_index(grid_size: int | Sequence[int]) -> LocalEdges:
    """Return every undirected 8-neighbor edge exactly once.

    The four forward directions are right, down, down-right, and down-left.
    """

    height, width = _grid_pair(grid_size)
    source: list[int] = []
    target: list[int] = []
    distance: list[int] = []
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(height):
        for column in range(width):
            first = row * width + column
            for delta_row, delta_column in directions:
                other_row = row + delta_row
                other_column = column + delta_column
                if 0 <= other_row < height and 0 <= other_column < width:
                    source.append(first)
                    target.append(other_row * width + other_column)
                    distance.append(delta_row * delta_row + delta_column * delta_column)
    result = LocalEdges(
        source=np.asarray(source, dtype=np.int64),
        target=np.asarray(target, dtype=np.int64),
        squared_distance=np.asarray(distance, dtype=np.uint8),
        grid_size=(height, width),
    )
    expected = height * (width - 1) + (height - 1) * width
    expected += 2 * (height - 1) * (width - 1)
    if result.count != expected:
        raise RuntimeError(f"constructed {result.count} local edges, expected {expected}")
    if np.any(result.source >= result.target):
        # Down-left edges can have a larger source flat index than target. They
        # are still unique undirected edges, so canonicalize every pair.
        lower = np.minimum(result.source, result.target)
        upper = np.maximum(result.source, result.target)
        result = LocalEdges(lower, upper, result.squared_distance, result.grid_size)
    pairs = np.stack((result.source, result.target), axis=-1)
    if len(np.unique(pairs, axis=0)) != result.count:
        raise RuntimeError("local edge construction produced duplicate pairs")
    return result


def construct_local_graphs(
    patch_tokens: torch.Tensor,
    edges: LocalEdges,
    *,
    sigma_s: float = 1.0,
) -> Mapping[str, torch.Tensor]:
    """Construct B0--B3 without accepting or consulting semantic labels."""

    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must have shape [B,N,D]")
    if patch_tokens.shape[1] != edges.grid_size[0] * edges.grid_size[1]:
        raise ValueError("patch-token count does not match the local edge grid")
    sigma_s = float(sigma_s)
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError("sigma_s must be finite and positive")

    device = patch_tokens.device
    source = torch.as_tensor(edges.source, device=device, dtype=torch.long)
    target = torch.as_tensor(edges.target, device=device, dtype=torch.long)
    distance = torch.as_tensor(
        edges.squared_distance, device=device, dtype=patch_tokens.dtype
    )
    uniform = torch.ones(
        (patch_tokens.shape[0], edges.count),
        device=device,
        dtype=patch_tokens.dtype,
    )
    spatial = torch.exp(-distance / (2.0 * sigma_s * sigma_s)).unsqueeze(0)
    spatial = spatial.expand_as(uniform)

    unit = F.normalize(patch_tokens, p=2, dim=-1, eps=1e-12)
    forward = (unit[:, source] * unit[:, target]).sum(dim=-1)
    reverse = (unit[:, target] * unit[:, source]).sum(dim=-1)
    # Explicit symmetrization documents the graph contract, even though cosine
    # and the spatial Gaussian are analytically symmetric.
    feature = (((forward + reverse) * 0.5 + 1.0) * 0.5).clamp(0.0, 1.0)
    joint = spatial * feature
    graphs = {
        "B0_uniform": uniform,
        "B1_spatial": spatial,
        "B2_feature": feature,
        "B3_spatial_feature": joint,
    }
    for name, weights in graphs.items():
        if weights.shape != (patch_tokens.shape[0], edges.count):
            raise RuntimeError(f"{name} has unexpected shape {weights.shape}")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise RuntimeError(f"{name} contains invalid graph weights")
    return graphs


def dense_symmetric_adjacency(weights: torch.Tensor, edges: LocalEdges) -> torch.Tensor:
    if weights.ndim != 2 or weights.shape[1] != edges.count:
        raise ValueError("weights must have shape [B,E]")
    nodes = edges.grid_size[0] * edges.grid_size[1]
    adjacency = weights.new_zeros((weights.shape[0], nodes, nodes))
    source = torch.as_tensor(edges.source, device=weights.device, dtype=torch.long)
    target = torch.as_tensor(edges.target, device=weights.device, dtype=torch.long)
    adjacency[:, source, target] = weights
    adjacency[:, target, source] = weights
    return adjacency


def semantic_patch_labels(
    mask: np.ndarray | torch.Tensor,
    *,
    patch_size: int | Sequence[int] = 16,
    semantic_majority: float = 0.5,
    minimum_valid_fraction: float = 0.5,
) -> np.ndarray:
    """Assign 0..20, mixed, or void using Experiment 2's exact conventions."""

    semantic_majority = float(semantic_majority)
    minimum_valid_fraction = float(minimum_valid_fraction)
    if not 0.0 < semantic_majority <= 1.0:
        raise ValueError("semantic_majority must be in (0,1]")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0,1]")
    counts = patch_label_counts(mask, patch_size=patch_size).astype(np.int64)
    patch_area = counts.sum(axis=-1)
    if not np.all(patch_area == patch_area[0]):
        raise RuntimeError("patch areas are inconsistent")
    void = counts[:, PATCH_LABEL_ID_TO_COLUMN[VOC_VOID_ID]]
    valid = patch_area - void
    labels = np.full(len(counts), PATCH_LABEL_MIXED, dtype=np.int8)
    sufficiently_valid = valid / patch_area >= minimum_valid_fraction
    labels[~sufficiently_valid] = PATCH_LABEL_VOID

    semantic_counts = counts[:, :21]
    safe_valid = np.maximum(valid, 1)[:, None]
    fractions = semantic_counts / safe_valid
    maximum = fractions.max(axis=-1)
    ties = np.isclose(fractions, maximum[:, None], rtol=0.0, atol=0.0).sum(axis=-1)
    owned = sufficiently_valid & (maximum >= semantic_majority) & (ties == 1)
    labels[owned] = fractions.argmax(axis=-1)[owned].astype(np.int8)

    if isinstance(mask, torch.Tensor):
        height, width = tuple(int(value) for value in mask.shape)
    else:
        height, width = np.asarray(mask).shape
    if isinstance(patch_size, (int, np.integer)):
        patch_h = patch_w = int(patch_size)
    else:
        patch_h, patch_w = (int(value) for value in patch_size)
    return labels.reshape(height // patch_h, width // patch_w)


def boundary_patch_mask(labels: np.ndarray, edges: LocalEdges) -> np.ndarray:
    flat = np.asarray(labels).reshape(-1)
    if flat.size != edges.grid_size[0] * edges.grid_size[1]:
        raise ValueError("label map and edge grid differ")
    valid = flat >= 0
    cross = (
        valid[edges.source]
        & valid[edges.target]
        & (flat[edges.source] != flat[edges.target])
    )
    boundary = np.zeros(flat.size, dtype=bool)
    boundary[edges.source[cross]] = True
    boundary[edges.target[cross]] = True
    return boundary.reshape(edges.grid_size)
