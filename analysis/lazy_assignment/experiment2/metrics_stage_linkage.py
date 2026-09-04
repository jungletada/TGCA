"""Feature-to-attention-to-CAM map linkage and transition metrics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from analysis.lazy_assignment.experiment2.metrics_region import (
    map_overlap_metrics,
    stable_topk_mask,
)


def stage_transition_metrics(
    source: np.ndarray,
    destination: np.ndarray,
    region_labels: np.ndarray,
    *,
    ratio: float = 0.10,
    region_values: Mapping[str, object] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float | int]:
    """Quantify semantic survival, introduction, and removal for X -> Y."""

    if region_values is None:
        region_values = {
            "target": 0,
            "other_fg": 1,
            "background": 2,
            "mixed": 3,
            "void": 4,
        }
    labels = np.asarray(region_labels).reshape(-1)
    src = np.asarray(source, dtype=np.float64).reshape(-1)
    dst = np.asarray(destination, dtype=np.float64).reshape(-1)
    if src.size != dst.size or src.size != labels.size:
        raise ValueError("source, destination, and region map sizes differ")
    eligible = labels != region_values["void"]
    src_top = stable_topk_mask(src, ratio, eligible).reshape(-1)
    dst_top = stable_topk_mask(dst, ratio, eligible).reshape(-1)
    introduced = np.logical_and(dst_top, ~src_top)
    removed = np.logical_and(src_top, ~dst_top)
    common = np.logical_and(src_top, dst_top)
    result: dict[str, float | int] = {
        "topk_ratio": float(ratio),
        "source_topk_size": int(src_top.sum()),
        "destination_topk_size": int(dst_top.sum()),
        "common_topk_size": int(common.sum()),
        "introduced_size": int(introduced.sum()),
        "removed_size": int(removed.sum()),
    }
    result.update(map_overlap_metrics(src, dst, ratio=ratio, eligible=eligible))
    names = ("target", "other_fg", "background")
    for name in names:
        region = labels == region_values[name]
        source_region = np.logical_and(src_top, region)
        destination_region = np.logical_and(dst_top, region)
        source_count = int(source_region.sum())
        destination_count = int(destination_region.sum())
        result[f"survive_{name}"] = (
            float(
                np.logical_and(source_region, dst_top).sum() / (source_count + epsilon)
            )
            if source_count
            else float("nan")
        )
        result[f"destination_retained_{name}"] = (
            float(
                np.logical_and(destination_region, src_top).sum()
                / (destination_count + epsilon)
            )
            if destination_count
            else float("nan")
        )
        result[f"introduced_{name}_fraction"] = float(
            np.logical_and(introduced, region).sum() / (int(dst_top.sum()) + epsilon)
        )
        result[f"removed_{name}_fraction"] = float(
            np.logical_and(removed, region).sum() / (int(src_top.sum()) + epsilon)
        )
    return result
