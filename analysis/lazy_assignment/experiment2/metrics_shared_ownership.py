"""Shared high-score support ownership metrics for positive class pairs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from analysis.lazy_assignment.experiment2.metrics_region import (
    jaccard,
    overlap_coefficient,
    stable_topk_mask,
)


PAIR_REGION_NAMES = (
    "target_a",
    "target_b",
    "other_fg",
    "background",
    "mixed",
    "void",
)


def _pair_masks(
    labels: np.ndarray, values: Mapping[str, object] | None = None
) -> dict[str, np.ndarray]:
    array = np.asarray(labels).reshape(-1)
    if values is None:
        values = {
            "target_a": 0,
            "target_b": 1,
            "other_fg": 2,
            "background": 3,
            "mixed": 4,
            "void": 5,
        }
    missing = set(PAIR_REGION_NAMES).difference(values)
    if missing:
        raise ValueError(f"missing pair-region values: {sorted(missing)}")
    return {name: array == values[name] for name in PAIR_REGION_NAMES}


def shared_support_mask(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    ratio: float,
    eligible: np.ndarray,
) -> np.ndarray:
    return np.logical_and(
        stable_topk_mask(scores_a, ratio, eligible),
        stable_topk_mask(scores_b, ratio, eligible),
    )


def shared_support_metrics(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pair_regions: np.ndarray,
    *,
    ratio: float,
    previous_scores_a: np.ndarray | None = None,
    previous_scores_b: np.ndarray | None = None,
    region_values: Mapping[str, object] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float | int]:
    """Describe current and newly introduced shared top-tail supports."""

    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    masks = _pair_masks(pair_regions, region_values)
    if a.size != b.size or a.size != masks["void"].size:
        raise ValueError("scores and pair-region map must have equal size")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores contain NaN or Inf")
    eligible = ~masks["void"]
    shared = shared_support_mask(a, b, ratio, eligible).reshape(-1)
    shared_size = int(shared.sum())
    valid_size = int(eligible.sum())

    result: dict[str, float | int] = {
        "topk_ratio": float(ratio),
        "shared_set_size": shared_size,
        "topk_jaccard": jaccard(
            stable_topk_mask(a, ratio, eligible),
            stable_topk_mask(b, ratio, eligible),
        ),
        "topk_overlap_coefficient": overlap_coefficient(
            stable_topk_mask(a, ratio, eligible),
            stable_topk_mask(b, ratio, eligible),
        ),
    }
    for name in PAIR_REGION_NAMES[:-1]:
        numerator = int(np.logical_and(shared, masks[name]).sum())
        result[f"shared_{name}_fraction"] = (
            float(numerator / shared_size) if shared_size else float("nan")
        )
        area = float(masks[name].sum() / valid_size) if valid_size else float("nan")
        result[f"shared_{name}_enrichment"] = (
            float(result[f"shared_{name}_fraction"]) / (area + epsilon)
            if shared_size and area > 0.0
            else float("nan")
        )
    result["shared_mixed_void_fraction"] = (
        float(np.logical_and(shared, masks["mixed"]).sum() / shared_size)
        if shared_size
        else float("nan")
    )

    if (previous_scores_a is None) != (previous_scores_b is None):
        raise ValueError("both previous score maps must be supplied together")
    if previous_scores_a is None:
        new_shared = shared.copy()
        has_previous = False
    else:
        previous = shared_support_mask(
            previous_scores_a, previous_scores_b, ratio, eligible
        ).reshape(-1)
        new_shared = np.logical_and(shared, ~previous)
        has_previous = True
    new_size = int(new_shared.sum())
    result["has_previous_layer"] = int(has_previous)
    result["new_shared_from_previous_layer"] = new_size
    for name in PAIR_REGION_NAMES[:-1]:
        result[f"new_shared_{name}_fraction"] = (
            float(np.logical_and(new_shared, masks[name]).sum() / new_size)
            if new_size
            else float("nan")
        )
    result["new_shared_mixed_void_fraction"] = (
        float(np.logical_and(new_shared, masks["mixed"]).sum() / new_size)
        if new_size
        else float("nan")
    )
    return result


def pairwise_cosine(tokens: np.ndarray) -> np.ndarray:
    """Return the symmetric cosine matrix for ``[K,D]`` class tokens."""

    array = np.asarray(tokens, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("tokens must have shape [K,D]")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    unit = np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)
    return unit @ unit.T
