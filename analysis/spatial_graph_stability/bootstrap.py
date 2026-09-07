"""Whole-image clustered paired bootstrap utilities for graph diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ClusterDraws:
    image_ids: tuple[str, ...]
    repeats: int
    seed: int
    multiplicities: np.ndarray


def make_cluster_draws(
    image_ids: list[str] | tuple[str, ...],
    *,
    repeats: int,
    seed: int,
) -> ClusterDraws:
    ids = tuple(str(value) for value in image_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("image cluster IDs must be non-empty and unique")
    if repeats < 1 or seed < 0:
        raise ValueError("bootstrap repeats must be positive and seed non-negative")
    probabilities = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    multiplicities = generator.multinomial(
        len(ids), probabilities, size=int(repeats)
    ).astype(np.uint16)
    if not np.all(multiplicities.sum(axis=1) == len(ids)):
        raise RuntimeError("cluster bootstrap draw has the wrong image count")
    return ClusterDraws(ids, int(repeats), int(seed), multiplicities)


def percentile_interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return None, None, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(finite.size)


def ratio_estimate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    top = float(np.asarray(numerator, dtype=np.float64).sum())
    bottom = float(np.asarray(denominator, dtype=np.float64).sum())
    return top / bottom if bottom > 0 else None


def ratio_bootstrap(
    draws: ClusterDraws,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    top = np.asarray(numerator, dtype=np.float64).reshape(-1)
    bottom = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if len(top) != len(draws.image_ids) or top.shape != bottom.shape:
        raise ValueError("one numerator/denominator value is required per image")
    multiplicities = draws.multiplicities.astype(np.float64, copy=False)
    sampled_top = multiplicities @ top
    sampled_bottom = multiplicities @ bottom
    return np.divide(
        sampled_top,
        sampled_bottom,
        out=np.full(draws.repeats, np.nan, dtype=np.float64),
        where=sampled_bottom > 0,
    )


def exact_rank_metrics(target: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    labels = np.asarray(target, dtype=bool).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None, None
    return (
        float(roc_auc_score(labels.astype(np.uint8), values)),
        float(average_precision_score(labels.astype(np.uint8), values)),
    )


def rank_histograms_by_image(
    image_index: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    *,
    num_images: int,
    bins: int,
    score_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(image_index, dtype=np.int64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if not (images.shape == labels.shape == values.shape):
        raise ValueError("image, target, and score arrays must have equal length")
    if num_images < 1 or bins < 2 or np.any(images < 0) or np.any(images >= num_images):
        raise ValueError("invalid histogram image count, bin count, or image index")
    low, high = (float(value) for value in score_range)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("score_range must be finite and increasing")
    finite = np.isfinite(values)
    images = images[finite]
    labels = labels[finite]
    clipped = np.clip(values[finite], low, high)
    indices = np.floor((clipped - low) / (high - low) * (bins - 1)).astype(np.int64)
    flattened = images * (2 * bins) + labels.astype(np.int64) * bins + indices
    counts = np.bincount(flattened, minlength=num_images * 2 * bins)
    counts = counts.reshape(num_images, 2, bins).astype(np.uint32)
    return counts[:, 1], counts[:, 0]


def _rank_metrics_from_histogram(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.shape != neg.shape or pos.ndim != 2:
        raise ValueError("positive/negative histograms must share [draw,bins] shape")
    pos_total = pos.sum(axis=1)
    neg_total = neg.sum(axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    auc_numerator = np.sum(pos * (neg_below + 0.5 * neg), axis=1)
    auc = np.divide(
        auc_numerator,
        pos_total * neg_total,
        out=np.full(len(pos), np.nan, dtype=np.float64),
        where=(pos_total > 0) & (neg_total > 0),
    )
    pos_desc = pos[:, ::-1]
    neg_desc = neg[:, ::-1]
    true_positive = np.cumsum(pos_desc, axis=1)
    false_positive = np.cumsum(neg_desc, axis=1)
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    ap = np.divide(
        np.sum(precision * pos_desc, axis=1),
        pos_total,
        out=np.full(len(pos), np.nan, dtype=np.float64),
        where=pos_total > 0,
    )
    ap[neg_total <= 0] = np.nan
    return auc, ap


def histogram_point_metrics(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> tuple[float | None, float | None]:
    positive = np.asarray(positive_histogram).sum(axis=0, keepdims=True)
    negative = np.asarray(negative_histogram).sum(axis=0, keepdims=True)
    auc, ap = _rank_metrics_from_histogram(positive, negative)
    return (
        float(auc[0]) if np.isfinite(auc[0]) else None,
        float(ap[0]) if np.isfinite(ap[0]) else None,
    )


def histogram_rank_bootstrap(
    draws: ClusterDraws,
    positive_histogram: np.ndarray,
    negative_histogram: np.ndarray,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive_histogram, dtype=np.uint32)
    negative = np.asarray(negative_histogram, dtype=np.uint32)
    if positive.shape != negative.shape or positive.shape[0] != len(draws.image_ids):
        raise ValueError("histograms require one equally-shaped row per image cluster")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA histogram bootstrap requested but unavailable")
    multiplicities = torch.from_numpy(draws.multiplicities.astype(np.float32)).to(requested)
    histograms = torch.from_numpy(
        np.concatenate((positive, negative), axis=1).astype(np.float32)
    ).to(requested)
    previous_tf32 = None
    if requested.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sampled = multiplicities @ histograms
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    bins = positive.shape[1]
    sampled_positive = sampled[:, :bins]
    sampled_negative = sampled[:, bins:]
    pos_total = sampled_positive.sum(dim=1)
    neg_total = sampled_negative.sum(dim=1)
    neg_below = torch.cumsum(sampled_negative, dim=1) - sampled_negative
    auc = (sampled_positive * (neg_below + 0.5 * sampled_negative)).sum(dim=1)
    auc = auc / (pos_total * neg_total).clamp_min(1.0)
    auc[(pos_total <= 0) | (neg_total <= 0)] = torch.nan

    pos_desc = sampled_positive.flip(1)
    neg_desc = sampled_negative.flip(1)
    true_positive = torch.cumsum(pos_desc, dim=1)
    false_positive = torch.cumsum(neg_desc, dim=1)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    ap = (precision * pos_desc).sum(dim=1) / pos_total.clamp_min(1.0)
    ap[(pos_total <= 0) | (neg_total <= 0)] = torch.nan
    return auc.cpu().double().numpy(), ap.cpu().double().numpy()
