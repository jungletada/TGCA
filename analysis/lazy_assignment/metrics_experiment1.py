"""Pure representation-map metrics for Experiment 1 result analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
ENTROPY_TEMPERATURES = (0.05, 0.10, 0.20)
TOPK_RATIOS = (0.05, 0.10, 0.20)


def _flat_float64(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        raise ValueError("score map must contain at least one patch")
    if not np.isfinite(array).all():
        raise ValueError("score map contains NaN or Inf")
    return array


def topk_count(num_patches: int, ratio: float) -> int:
    if num_patches < 1:
        raise ValueError("num_patches must be positive")
    if not 0 < ratio <= 1:
        raise ValueError(f"top-k ratio must be in (0, 1], got {ratio}")
    return min(num_patches, max(1, int(math.ceil(num_patches * ratio))))


def topk_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    """Return an exact-size deterministic top-k mask.

    Stable sorting makes tie handling reproducible: lower flattened patch indices
    win ties.  This avoids threshold masks whose size changes when scores tie.
    """

    flat = _flat_float64(values)
    count = topk_count(flat.size, ratio)
    order = np.argsort(-flat, kind="stable")
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(np.asarray(values).shape)


def bottomk_mean(values: np.ndarray, ratio: float) -> float:
    flat = _flat_float64(values)
    count = topk_count(flat.size, ratio)
    order = np.argsort(flat, kind="stable")
    return float(flat[order[:count]].mean())


def topk_mean(values: np.ndarray, ratio: float) -> float:
    flat = _flat_float64(values)
    mask = topk_mask(flat, ratio).reshape(-1)
    return float(flat[mask].mean())


def normalized_spatial_entropy(values: np.ndarray, temperature: float) -> float:
    """Entropy of a fixed-temperature softmax over cosine scores.

    This is an auxiliary concentration metric, not attention entropy.
    """

    flat = _flat_float64(values)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = flat / float(temperature)
    logits -= logits.max()
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
    return float(entropy / math.log(flat.size)) if flat.size > 1 else 0.0


def total_variation(values: np.ndarray, grid_h: int, grid_w: int) -> float:
    flat = _flat_float64(values)
    if grid_h < 1 or grid_w < 1 or grid_h * grid_w != flat.size:
        raise ValueError("grid shape does not match score-map size")
    grid = flat.reshape(grid_h, grid_w)
    horizontal = np.abs(np.diff(grid, axis=1)).reshape(-1)
    vertical = np.abs(np.diff(grid, axis=0)).reshape(-1)
    edges = np.concatenate((horizontal, vertical))
    return float(edges.mean()) if edges.size else 0.0


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = _flat_float64(left)
    right = _flat_float64(right)
    if left.size != right.size:
        raise ValueError("correlation inputs must have equal size")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(left, right)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = _flat_float64(left)
    right = _flat_float64(right)
    if left.size != right.size:
        raise ValueError("correlation inputs must have equal size")
    return _pearson(rankdata(left, method="average"), rankdata(right, method="average"))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = _flat_float64(left)
    right = _flat_float64(right)
    if left.size != right.size:
        raise ValueError("cosine inputs must have equal size")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def neighbor_correlations(
    values: np.ndarray, grid_h: int, grid_w: int
) -> tuple[float, float]:
    flat = _flat_float64(values)
    if grid_h < 1 or grid_w < 1 or grid_h * grid_w != flat.size:
        raise ValueError("grid shape does not match score-map size")
    grid = flat.reshape(grid_h, grid_w)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    if grid_w > 1:
        left_parts.append(grid[:, :-1].reshape(-1))
        right_parts.append(grid[:, 1:].reshape(-1))
    if grid_h > 1:
        left_parts.append(grid[:-1, :].reshape(-1))
        right_parts.append(grid[1:, :].reshape(-1))
    if not left_parts:
        return float("nan"), float("nan")
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    return pearson_correlation(left, right), spearman_correlation(left, right)


def component_metrics(mask: np.ndarray) -> tuple[int, float]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    selected = int(binary.sum())
    if selected == 0:
        return 0, 0.0
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, count = ndimage.label(binary, structure=structure)
    sizes = np.bincount(labels.reshape(-1))[1:]
    largest_fraction = float(sizes.max() / selected) if sizes.size else 0.0
    return int(count), largest_fraction


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool).reshape(-1)
    right = np.asarray(right, dtype=bool).reshape(-1)
    if left.size != right.size:
        raise ValueError("Jaccard masks must have equal size")
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def topk_jaccard(left: np.ndarray, right: np.ndarray, ratio: float) -> float:
    return jaccard(topk_mask(left, ratio), topk_mask(right, ratio))


def score_map_metrics(
    values: np.ndarray,
    grid_h: int,
    grid_w: int,
    entropy_temperatures: Iterable[float] = ENTROPY_TEMPERATURES,
) -> dict[str, float | int]:
    flat = _flat_float64(values)
    if grid_h * grid_w != flat.size:
        raise ValueError("grid shape does not match score-map size")
    quantiles = np.quantile(flat, QUANTILE_LEVELS)
    result: dict[str, float | int] = {
        "num_patches": int(flat.size),
        "score_min": float(flat.min()),
        "score_max": float(flat.max()),
        "score_mean": float(flat.mean()),
        "score_std": float(flat.std(ddof=0)),
        "positive_score_fraction": float(np.mean(flat > 0)),
        "negative_score_fraction": float(np.mean(flat < 0)),
        "top_01_mean": topk_mean(flat, 0.01),
        "top_05_mean": topk_mean(flat, 0.05),
        "top_10_mean": topk_mean(flat, 0.10),
        "bottom_10_mean": bottomk_mean(flat, 0.10),
        "dynamic_range": float(flat.max() - flat.min()),
    }
    for level, value in zip(QUANTILE_LEVELS, quantiles):
        result[f"score_q{int(round(level * 100)):02d}"] = float(value)
    result["iqr"] = float(result["score_q75"] - result["score_q25"])
    result["upper_tail_gap"] = float(result["score_q95"] - result["score_q50"])
    result["top10_concentration"] = float(
        result["top_10_mean"] - result["score_q50"]
    )
    for temperature in entropy_temperatures:
        suffix = f"{int(round(float(temperature) * 1000)):03d}"
        result[f"spatial_entropy_tau_{suffix}"] = normalized_spatial_entropy(
            flat, float(temperature)
        )
    result["total_variation"] = total_variation(flat, grid_h, grid_w)
    neighbor_pearson, neighbor_spearman = neighbor_correlations(flat, grid_h, grid_w)
    result["neighbor_pearson"] = neighbor_pearson
    result["neighbor_spearman"] = neighbor_spearman
    top10 = topk_mask(flat, 0.10).reshape(grid_h, grid_w)
    components, largest = component_metrics(top10)
    result["num_components_top10"] = components
    result["largest_component_fraction_top10"] = largest
    return result

