"""Metrics for comparing LaST channel-selection vote maps across bases."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import rankdata


def vote_counts(indices: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Aggregate ``[B,K,D]`` channel selections into ``[B,N]`` vote maps."""

    if indices.ndim != 3:
        raise ValueError(f"indices must have shape [B,K,D], got {indices.shape}")
    if num_patches < 1:
        raise ValueError("num_patches must be positive")
    if torch.any(indices < 0) or torch.any(indices >= num_patches):
        raise ValueError("selected patch index is out of range")
    flattened = indices.reshape(indices.shape[0], -1)
    votes = torch.zeros(
        (indices.shape[0], num_patches),
        device=indices.device,
        dtype=torch.int64,
    )
    votes.scatter_add_(1, flattened, torch.ones_like(flattened, dtype=torch.int64))
    return votes


def _stable_top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1:
        raise ValueError("top-fraction input must be a vector")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    count = min(len(vector), max(1, int(math.ceil(len(vector) * fraction))))
    # lexsort uses the final key as primary: descending value, then ascending
    # patch index for deterministic resolution of the many zero-vote ties.
    order = np.lexsort((np.arange(len(vector)), -vector.astype(np.float64)))
    return np.sort(order[:count])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = rankdata(first, method="average")
    second_rank = rankdata(second, method="average")
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def vote_entropy(votes: np.ndarray) -> tuple[float, float]:
    values = np.asarray(votes, dtype=np.float64)
    total = values.sum()
    if values.ndim != 1 or total <= 0:
        raise ValueError("votes must be a non-empty vector with positive mass")
    probabilities = values[values > 0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    return entropy, normalized


def compare_vote_maps(
    identity: np.ndarray,
    transformed: np.ndarray,
    *,
    top_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    first = np.asarray(identity)
    second = np.asarray(transformed)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("vote maps must have the same [B,N] shape")
    batch = first.shape[0]
    spearman = np.full(batch, np.nan, dtype=np.float64)
    jaccard = np.full(batch, np.nan, dtype=np.float64)
    normalized_l1 = np.full(batch, np.nan, dtype=np.float64)
    entropy = np.full(batch, np.nan, dtype=np.float64)
    entropy_normalized = np.full(batch, np.nan, dtype=np.float64)
    for index in range(batch):
        spearman[index] = _spearman(first[index], second[index])
        first_top = _stable_top_fraction(first[index], top_fraction)
        second_top = _stable_top_fraction(second[index], top_fraction)
        intersection = np.intersect1d(first_top, second_top, assume_unique=True).size
        union = np.union1d(first_top, second_top).size
        jaccard[index] = intersection / union if union else float("nan")
        first_probability = first[index] / first[index].sum()
        second_probability = second[index] / second[index].sum()
        normalized_l1[index] = np.abs(first_probability - second_probability).sum()
        entropy[index], entropy_normalized[index] = vote_entropy(second[index])
    return {
        "vote_spearman": spearman,
        "top10_jaccard": jaccard,
        "normalized_l1": normalized_l1,
        "vote_entropy_nats": entropy,
        "vote_entropy_normalized": entropy_normalized,
    }


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return {
            "count": int(vector.size),
            "finite_count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q05": None,
            "q95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(vector.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def logit_comparison(identity: np.ndarray, transformed: np.ndarray) -> dict[str, np.ndarray]:
    first = np.asarray(identity, dtype=np.float64)
    second = np.asarray(transformed, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("logits must share shape [B,C]")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.full(first.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    top1 = first.argmax(axis=1) == second.argmax(axis=1)
    top5_first = np.argpartition(first, -5, axis=1)[:, -5:]
    top5_second = np.argpartition(second, -5, axis=1)[:, -5:]
    overlap = np.asarray(
        [len(np.intersect1d(a, b, assume_unique=False)) / 5 for a, b in zip(top5_first, top5_second)],
        dtype=np.float64,
    )
    exact = np.asarray(
        [set(a.tolist()) == set(b.tolist()) for a, b in zip(top5_first, top5_second)],
        dtype=np.float64,
    )
    return {
        "logit_cosine": cosine,
        "logit_l1": np.abs(first - second).sum(axis=1),
        "logit_l1_mean": np.abs(first - second).mean(axis=1),
        "logit_l2": np.linalg.norm(first - second, axis=1),
        "top1_agreement": top1.astype(np.float64),
        "top5_overlap_fraction": overlap,
        "top5_exact_set_agreement": exact,
    }


def channel_selection_agreement(
    identity_indices: np.ndarray,
    transformed_indices: np.ndarray,
    correspondence: np.ndarray,
) -> np.ndarray:
    first = np.asarray(identity_indices)
    second = np.asarray(transformed_indices)
    mapping = np.asarray(correspondence, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != len(mapping):
        raise ValueError("selection arrays/correspondence have incompatible shapes")
    return (first[:, :, mapping] == second).mean(axis=(1, 2))


def semantic_vote_composition(votes: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    vote = np.asarray(votes, dtype=np.float64).reshape(-1)
    semantic = np.asarray(labels).reshape(-1)
    if vote.shape != semantic.shape:
        raise ValueError("vote and semantic patch maps must have the same shape")
    masks = {
        "foreground": semantic > 0,
        "background": semantic == 0,
        "mixed": semantic == -1,
        "void": semantic == -2,
    }
    total = vote.sum()
    valid_vote = vote[masks["foreground"] | masks["background"]].sum()
    valid_area = np.count_nonzero(masks["foreground"] | masks["background"])
    foreground_area = np.count_nonzero(masks["foreground"])
    result: dict[str, float] = {}
    for name, mask in masks.items():
        result[f"{name}_vote"] = float(vote[mask].sum())
        result[f"{name}_vote_fraction_all"] = float(vote[mask].sum() / total)
        result[f"{name}_patch_count"] = float(np.count_nonzero(mask))
    result["foreground_vote_fraction_valid"] = (
        float(result["foreground_vote"] / valid_vote) if valid_vote > 0 else float("nan")
    )
    result["background_vote_fraction_valid"] = (
        float(result["background_vote"] / valid_vote) if valid_vote > 0 else float("nan")
    )
    foreground_area_fraction = foreground_area / valid_area if valid_area else float("nan")
    result["foreground_area_fraction_valid"] = float(foreground_area_fraction)
    result["foreground_vote_enrichment"] = (
        float(result["foreground_vote_fraction_valid"] / foreground_area_fraction)
        if foreground_area_fraction > 0
        else float("nan")
    )
    top = _stable_top_fraction(vote, 0.1)
    for name, mask in masks.items():
        result[f"top10_{name}_fraction"] = float(np.count_nonzero(mask[top]) / len(top))
    return result
