"""Pure semantic-region metrics for Experiment 2.

The functions in this module operate on one spatial map at a time.  They do
not read model outputs or VOC files, which makes the ranking, undefined-case,
and signed-AUC contracts independently testable.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


TOPK_RATIOS = (0.05, 0.10, 0.20)
REGION_NAMES = ("target", "other_fg", "background", "mixed", "void")


def _flat(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("a spatial map must contain at least one patch")
    if not np.isfinite(array).all():
        raise ValueError("a spatial map contains NaN or Inf")
    return array


def _regions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError("region labels must contain at least one patch")
    return array


def stable_topk_mask(
    values: np.ndarray, ratio: float, eligible: np.ndarray | None = None
) -> np.ndarray:
    """Select an exact-size top-k set with stable flattened-index tie breaks."""

    scores = _flat(values)
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if eligible is None:
        eligible_flat = np.ones(scores.size, dtype=bool)
    else:
        eligible_flat = np.asarray(eligible, dtype=bool).reshape(-1)
        if eligible_flat.size != scores.size:
            raise ValueError("eligible mask and score map must have equal size")
    indices = np.flatnonzero(eligible_flat)
    if not indices.size:
        return np.zeros(scores.size, dtype=bool).reshape(np.asarray(values).shape)
    count = min(indices.size, max(1, int(math.ceil(indices.size * float(ratio)))))
    order = np.argsort(-scores[indices], kind="stable")
    selected = np.zeros(scores.size, dtype=bool)
    selected[indices[order[:count]]] = True
    return selected.reshape(np.asarray(values).shape)


def overlap_coefficient(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool).reshape(-1)
    b = np.asarray(right, dtype=bool).reshape(-1)
    if a.size != b.size:
        raise ValueError("overlap masks must have equal size")
    denominator = min(int(a.sum()), int(b.sum()))
    if denominator == 0:
        return 1.0 if not (a.any() or b.any()) else 0.0
    return float(np.logical_and(a, b).sum() / denominator)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool).reshape(-1)
    b = np.asarray(right, dtype=bool).reshape(-1)
    if a.size != b.size:
        raise ValueError("Jaccard masks must have equal size")
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _binary_discrimination(
    scores: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> tuple[float, float]:
    include = np.logical_or(positive, negative)
    labels = positive[include].astype(np.int8)
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan"), float("nan")
    selected_scores = scores[include]
    return (
        float(roc_auc_score(labels, selected_scores)),
        float(average_precision_score(labels, selected_scores)),
    )


def _region_summary(scores: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = scores[mask]
    if selected.size == 0:
        return {key: float("nan") for key in ("mean", "median", "q90", "q95")}
    return {
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "q90": float(np.quantile(selected, 0.90)),
        "q95": float(np.quantile(selected, 0.95)),
    }


def spatial_total_variation(values: np.ndarray, grid_h: int, grid_w: int) -> float:
    array = _flat(values)
    if int(grid_h) * int(grid_w) != array.size:
        raise ValueError("grid shape does not match map size")
    grid = array.reshape(int(grid_h), int(grid_w))
    edges: list[np.ndarray] = []
    if grid_w > 1:
        edges.append(np.abs(np.diff(grid, axis=1)).reshape(-1))
    if grid_h > 1:
        edges.append(np.abs(np.diff(grid, axis=0)).reshape(-1))
    return float(np.concatenate(edges).mean()) if edges else 0.0


def zscore_spatial_entropy(values: np.ndarray) -> float:
    """Normalized entropy of softmax(z-score(map)); shape-only control."""

    array = _flat(values)
    if array.size == 1:
        return 0.0
    std = float(array.std(ddof=0))
    if std <= 1e-12:
        return 1.0
    logits = (array - array.mean()) / std
    logits -= logits.max()
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
    return float(entropy / math.log(array.size))


def _coerce_region_masks(
    region_labels: np.ndarray,
    region_values: Mapping[str, object] | None,
) -> dict[str, np.ndarray]:
    labels = _regions(region_labels)
    if region_values is None:
        # Canonical numeric contract used by patch_regions.py.
        region_values = {
            "target": 0,
            "other_fg": 1,
            "background": 2,
            "mixed": 3,
            "void": 4,
        }
    missing = set(REGION_NAMES).difference(region_values)
    if missing:
        raise ValueError(f"missing region values: {sorted(missing)}")
    return {name: labels == region_values[name] for name in REGION_NAMES}


def region_map_metrics(
    values: np.ndarray,
    region_labels: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    nonnegative_mass: bool = False,
    region_values: Mapping[str, object] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float | int | str | bool]:
    """Compute the pre-registered ownership metrics for one map.

    Void patches are excluded from ranking and area denominators.  Mixed
    patches remain valid but are not included in either binary AUC contrast.
    Undefined contrasts are represented by NaN and never imputed or flipped.
    """

    scores = _flat(values)
    labels = _regions(region_labels)
    if labels.size != scores.size or int(grid_h) * int(grid_w) != scores.size:
        raise ValueError("score, regions, and grid geometry do not agree")
    masks = _coerce_region_masks(labels, region_values)
    valid = ~masks["void"]
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("map has no valid patches")
    valid_scores = scores[valid]
    degenerate = bool(np.ptp(valid_scores) <= epsilon)
    all_zero = bool(np.max(np.abs(valid_scores)) <= epsilon)
    result: dict[str, float | int | str | bool] = {
        "num_target": int(masks["target"].sum()),
        "num_other_fg": int(masks["other_fg"].sum()),
        "num_bg": int(masks["background"].sum()),
        "num_mixed": int(masks["mixed"].sum()),
        "num_void": int(masks["void"].sum()),
        "num_valid": valid_count,
        "has_target_region": bool(masks["target"].any()),
        "degenerate_map": degenerate,
        "all_zero_map": all_zero,
        "score_mean": float(valid_scores.mean()),
        "score_std": float(valid_scores.std(ddof=0)),
    }
    quantiles = np.quantile(valid_scores, (0.25, 0.50, 0.75, 0.90, 0.95))
    iqr = float(quantiles[2] - quantiles[0])
    upper_tail = float(quantiles[4] - quantiles[1])
    score_std = float(result["score_std"])
    result.update(
        {
            "score_q25": float(quantiles[0]),
            "score_median": float(quantiles[1]),
            "score_q75": float(quantiles[2]),
            "score_q90": float(quantiles[3]),
            "score_q95": float(quantiles[4]),
            "upper_tail_gap": upper_tail,
            "upper_tail_over_std": upper_tail / (score_std + epsilon),
            "upper_tail_over_iqr": upper_tail / (iqr + epsilon),
            "total_variation": spatial_total_variation(scores, grid_h, grid_w),
            "total_variation_over_std": spatial_total_variation(scores, grid_h, grid_w)
            / (score_std + epsilon),
            "zscore_spatial_entropy": zscore_spatial_entropy(valid_scores),
        }
    )

    if nonnegative_mass and float(valid_scores.min()) < -epsilon:
        raise ValueError("mass metrics require a nonnegative map")
    if nonnegative_mass:
        mass_sum = float(valid_scores.sum())
        result["conditional_bg_mass"] = (
            float(scores[masks["background"]].sum() / mass_sum)
            if mass_sum > epsilon
            else float("nan")
        )
    else:
        result["conditional_bg_mass"] = float("nan")

    if nonnegative_mass and all_zero:
        result["top1_region"] = "degenerate"
        result["target_hit"] = result["other_fg_hit"] = False
        result["background_hit"] = result["mixed_hit"] = False
    else:
        valid_indices = np.flatnonzero(valid)
        top_index = int(valid_indices[np.argmax(scores[valid_indices])])
        top_region = next(name for name in REGION_NAMES[:-1] if masks[name][top_index])
        result["top1_region"] = top_region
        for name in REGION_NAMES[:-1]:
            key = "background_hit" if name == "background" else f"{name}_hit"
            result[key] = bool(name == top_region)

    area_fraction = {
        name: float(masks[name].sum() / valid_count) for name in REGION_NAMES[:-1]
    }
    for ratio in TOPK_RATIOS:
        suffix = f"{int(round(100 * ratio)):02d}"
        selected = stable_topk_mask(scores, ratio, valid).reshape(-1)
        count = int(selected.sum())
        for name in REGION_NAMES[:-1]:
            fraction = float(np.logical_and(selected, masks[name]).sum() / count)
            output_name = "bg" if name == "background" else name
            result[f"{output_name}_top{suffix}_fraction"] = fraction
            result[f"{output_name}_tail_enrich_{suffix}"] = (
                fraction / (area_fraction[name] + epsilon)
                if area_fraction[name] > 0.0
                else float("nan")
            )

    auc_bg, ap_bg = _binary_discrimination(scores, masks["target"], masks["background"])
    auc_other, ap_other = _binary_discrimination(
        scores, masks["target"], masks["other_fg"]
    )
    result.update(
        {
            "auc_target_bg": auc_bg,
            "ap_target_bg": ap_bg,
            "auc_target_other": auc_other,
            "ap_target_other": ap_other,
            "orientation_target_bg": (
                float(np.sign(auc_bg - 0.5)) if np.isfinite(auc_bg) else float("nan")
            ),
            "separability_target_bg": (
                float(2.0 * abs(auc_bg - 0.5)) if np.isfinite(auc_bg) else float("nan")
            ),
        }
    )
    summaries = {
        name: _region_summary(scores, masks[name]) for name in REGION_NAMES[:3]
    }
    for name, summary in summaries.items():
        output_name = "bg" if name == "background" else name
        result.update({f"{output_name}_{key}": value for key, value in summary.items()})
    result["target_bg_mean_margin"] = (
        float(summaries["target"]["mean"] - summaries["background"]["mean"])
        if np.isfinite(summaries["target"]["mean"])
        and np.isfinite(summaries["background"]["mean"])
        else float("nan")
    )
    result["target_other_mean_margin"] = (
        float(summaries["target"]["mean"] - summaries["other_fg"]["mean"])
        if np.isfinite(summaries["target"]["mean"])
        and np.isfinite(summaries["other_fg"]["mean"])
        else float("nan")
    )
    return result


def spatial_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation without importing scipy in per-map hot loops."""

    a = _flat(left)
    b = _flat(right)
    if a.size != b.size:
        raise ValueError("correlation maps must have equal size")
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return float("nan")
    # Stable average ranks for ties.
    from scipy.stats import rankdata

    ar = rankdata(a, method="average")
    br = rankdata(b, method="average")
    return float(np.corrcoef(ar, br)[0, 1])


def map_overlap_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    ratio: float = 0.10,
    eligible: np.ndarray | None = None,
) -> dict[str, float]:
    left_top = stable_topk_mask(left, ratio, eligible)
    right_top = stable_topk_mask(right, ratio, eligible)
    return {
        "spearman": spatial_spearman(left, right),
        "topk_jaccard": jaccard(left_top, right_top),
        "topk_overlap_coefficient": overlap_coefficient(left_top, right_top),
    }
