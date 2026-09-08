"""GT-aware evaluation of already-computed frozen Task A relation maps."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np

from analysis.lazy_assignment.experiment2.metrics_region import jaccard, spatial_spearman, stable_topk_mask
from analysis.lazy_assignment.experiment2.patch_regions import (
    REGION_BACKGROUND,
    REGION_OTHER_FOREGROUND,
    REGION_TARGET,
    REGION_VOID,
    assign_patch_regions,
)


TOP_RATIOS = (0.01, 0.05, 0.10)


def label_stratum(label_count: int) -> str:
    if label_count == 1:
        return "single_label"
    if label_count == 2:
        return "exactly_2_labels"
    if label_count >= 3:
        return "3plus_labels"
    raise ValueError("every VOC image must have at least one positive class")


def global_patch_regions(mask: np.ndarray) -> np.ndarray:
    """Return target/other/bg/mixed/void codes collapsed to any-FG semantics."""

    # The union of target and other foreground is every VOC foreground ID;
    # the choice of target ID therefore cannot change the requested FG-vs-BG
    # diagnostic.
    return np.asarray(assign_patch_regions(mask, target_class_id=0)["region_codes"]).reshape(-1)


def _pearson(left: np.ndarray, right: np.ndarray, eligible: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)[eligible]
    b = np.asarray(right, dtype=np.float64).reshape(-1)[eligible]
    if len(a) < 2 or np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r_squared(target: np.ndarray, predictor: np.ndarray, eligible: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64).reshape(-1)[eligible]
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)[eligible]
    if len(y) < 2 or np.ptp(y) <= 1e-12 or np.ptp(x) <= 1e-12:
        return float("nan")
    design = np.column_stack((x, np.ones_like(x)))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(1.0 - np.square(residual).sum() / np.square(y - y.mean()).sum())


def _pair_jaccard(left: np.ndarray, right: np.ndarray, ratio: float, eligible: np.ndarray) -> float:
    return jaccard(
        stable_topk_mask(left, ratio, eligible),
        stable_topk_mask(right, ratio, eligible),
    )


def evaluate_layer_relations(
    *,
    image_id: str,
    image_index: int,
    labels: np.ndarray,
    regions: np.ndarray,
    layer: int,
    raw: np.ndarray,
    residual: np.ndarray,
    common: np.ndarray,
    positive_common: np.ndarray,
) -> dict[str, object]:
    """One compact image/layer Task A row after the relation capture is frozen."""

    active = np.flatnonzero(np.asarray(labels, dtype=np.float32) > 0)
    count = int(len(active))
    if not count:
        raise ValueError("no positive class label")
    codes = np.asarray(regions).reshape(-1)
    valid = codes != REGION_VOID
    foreground = np.logical_or(codes == REGION_TARGET, codes == REGION_OTHER_FOREGROUND)
    background = codes == REGION_BACKGROUND
    raw = np.asarray(raw, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    common = np.asarray(common, dtype=np.float32).reshape(-1)
    if raw.shape != residual.shape or raw.shape[0] != 20 or raw.shape[1] != codes.size:
        raise ValueError("Task A score and patch-region geometry mismatch")
    row: dict[str, object] = {
        "image_id": image_id,
        "image_index": int(image_index),
        "layer": int(layer),
        "label_count": count,
        "label_stratum": label_stratum(count),
        "common_fg_top05": float(np.logical_and(stable_topk_mask(common, 0.05, valid).reshape(-1), foreground).sum() / max(1, stable_topk_mask(common, 0.05, valid).sum())),
        "common_bg_top05": float(np.logical_and(stable_topk_mask(common, 0.05, valid).reshape(-1), background).sum() / max(1, stable_topk_mask(common, 0.05, valid).sum())),
        "common_positive_spearman": float("nan"),
        "common_positive_jaccard_top05": float("nan"),
    }
    if count >= 2 and np.isfinite(positive_common).all():
        row["common_positive_spearman"] = spatial_spearman(common[valid], np.asarray(positive_common).reshape(-1)[valid])
        row["common_positive_jaccard_top05"] = _pair_jaccard(common, positive_common, 0.05, valid)
    r2 = [_r_squared(raw[class_id], common, valid) for class_id in active]
    row["common_r2"] = float(np.nanmean(r2)) if np.isfinite(r2).any() else float("nan")
    if count < 2:
        for prefix in ("raw", "residual"):
            row[f"{prefix}_pair_corr"] = float("nan")
            for ratio in TOP_RATIOS:
                row[f"{prefix}_pair_jaccard_top{int(ratio * 100):02d}"] = float("nan")
        return row
    values: dict[str, list[float]] = {
        "raw_pair_corr": [], "residual_pair_corr": [],
        **{f"raw_pair_jaccard_top{int(ratio * 100):02d}": [] for ratio in TOP_RATIOS},
        **{f"residual_pair_jaccard_top{int(ratio * 100):02d}": [] for ratio in TOP_RATIOS},
    }
    for first, second in itertools.combinations(active.tolist(), 2):
        values["raw_pair_corr"].append(_pearson(raw[first], raw[second], valid))
        values["residual_pair_corr"].append(_pearson(residual[first], residual[second], valid))
        for ratio in TOP_RATIOS:
            suffix = f"top{int(ratio * 100):02d}"
            values[f"raw_pair_jaccard_{suffix}"].append(_pair_jaccard(raw[first], raw[second], ratio, valid))
            values[f"residual_pair_jaccard_{suffix}"].append(_pair_jaccard(residual[first], residual[second], ratio, valid))
    for key, metric_values in values.items():
        finite = np.asarray(metric_values, dtype=float)
        row[key] = float(np.nanmean(finite)) if np.isfinite(finite).any() else float("nan")
    return row
