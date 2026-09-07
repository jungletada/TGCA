"""Semantic and conditional diagnostics shared by Phase C and Phase D.

All map-level rows retain ``image_id``.  Aggregate uncertainty is calculated
only by resampling these image clusters; no patch or image-class pair is ever
used as an independent bootstrap unit.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import (
    iter_all_and_label_strata,
    summarize_clustered,
)
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES
from analysis.lazy_assignment.experiment2.metrics_region import (
    TOPK_RATIOS,
    jaccard,
    region_map_metrics,
    spatial_spearman,
    stable_topk_mask,
    zscore_spatial_entropy,
)

from .bootstrap import ClusterDraws, make_cluster_draws, percentile_interval, ratio_bootstrap
from .graph import LocalEdges, PATCH_LABEL_MIXED, PATCH_LABEL_VOID, boundary_patch_mask


PRIMARY_REGION_METRICS = (
    "target_hit",
    "other_fg_hit",
    "background_hit",
    "target_top05_fraction",
    "other_fg_top05_fraction",
    "bg_top05_fraction",
    "target_tail_enrich_05",
    "other_fg_tail_enrich_05",
    "bg_tail_enrich_05",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "target_tail_enrich_10",
    "other_fg_tail_enrich_10",
    "bg_tail_enrich_10",
    "target_top20_fraction",
    "other_fg_top20_fraction",
    "bg_top20_fraction",
    "target_tail_enrich_20",
    "other_fg_tail_enrich_20",
    "bg_tail_enrich_20",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "target_mean",
    "other_fg_mean",
    "bg_mean",
    "target_median",
    "other_fg_median",
    "bg_median",
)


def label_stratum(count: int) -> str:
    if int(count) == 1:
        return "single_label"
    if int(count) == 2:
        return "exactly_2_labels"
    if int(count) >= 3:
        return "3plus_labels"
    raise ValueError("a positive class map must come from a labelled image")


def class_region_labels(semantic_labels: np.ndarray, class_id: int) -> np.ndarray:
    """Map Phase-B semantic labels to Experiment-2 region labels 0..4."""

    labels = np.asarray(semantic_labels, dtype=np.int16).reshape(-1)
    if not 0 <= int(class_id) < 20:
        raise ValueError("class_id must be in VOC foreground range 0..19")
    target_label = int(class_id) + 1
    result = np.full(labels.shape, 3, dtype=np.int8)  # mixed by default
    result[labels == 0] = 2
    result[(labels > 0) & (labels != target_label)] = 1
    result[labels == target_label] = 0
    result[labels == PATCH_LABEL_MIXED] = 3
    result[labels == PATCH_LABEL_VOID] = 4
    return result


def valid_patch_mask(semantic_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(semantic_labels).reshape(-1)
    return labels != PATCH_LABEL_VOID


def _finite_auc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(positive | negative, dtype=bool)
    truth = np.asarray(positive[selected], dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)[selected]
    if truth.size == 0 or truth.min() == truth.max() or not np.isfinite(values).all():
        return float("nan"), float("nan")
    return float(roc_auc_score(truth, values)), float(average_precision_score(truth, values))


def map_region_rows(
    *,
    image_ids: Sequence[str],
    positive_labels: np.ndarray,
    semantic_labels: np.ndarray,
    image_label_counts: np.ndarray,
    score_maps: Mapping[str, np.ndarray],
    graph: str,
    lambda_value: float,
) -> list[dict[str, object]]:
    """Compute Experiment-2-compatible ownership diagnostics for positive maps."""

    positive = np.asarray(positive_labels, dtype=bool)
    semantics = np.asarray(semantic_labels)
    count = np.asarray(image_label_counts, dtype=np.int64)
    if positive.shape != (len(image_ids), 20) or semantics.shape != (len(image_ids), 784):
        raise ValueError("positive labels or semantic labels have incorrect Phase C-D shape")
    rows: list[dict[str, object]] = []
    for score_name, maps in score_maps.items():
        array = np.asarray(maps, dtype=np.float64)
        if array.shape != (len(image_ids), 20, 784) or not np.isfinite(array).all():
            raise ValueError(f"{score_name} maps must be finite [I,20,784]")
        for image_index, class_id in zip(*np.nonzero(positive)):
            regions = class_region_labels(semantics[image_index], int(class_id))
            metrics = region_map_metrics(
                array[image_index, class_id], regions, grid_h=28, grid_w=28,
                nonnegative_mass=bool(np.min(array[image_index, class_id]) >= -1e-12),
            )
            rows.append(
                {
                    "image_id": str(image_ids[image_index]),
                    "image_index": int(image_index),
                    "class_id": int(class_id),
                    "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "num_positive_classes": int(count[image_index]),
                    "label_stratum": label_stratum(int(count[image_index])),
                    "graph": graph,
                    "lambda": float(lambda_value),
                    "score": str(score_name),
                    **metrics,
                }
            )
    return rows


def summarize_region_rows(
    rows: Sequence[Mapping[str, object]], *, repeats: int, seed: int
) -> list[dict[str, object]]:
    """Return micro, equal-class macro, and per-class clustered summaries."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    outputs: list[dict[str, object]] = []
    group_columns = ("graph", "lambda", "score")
    for keys, subset in frame.groupby(list(group_columns), sort=True):
        identity = dict(zip(group_columns, keys))
        for stratum, scoped in iter_all_and_label_strata(subset):
            outputs.extend(
                summarize_clustered(
                    scoped,
                    value_cols=PRIMARY_REGION_METRICS,
                    identity={**identity, "scope": "all_positive_classes", "stratum": stratum},
                    repeats=repeats,
                    seed=seed,
                )
            )
        for class_id, class_frame in subset.groupby("class_id", sort=True):
            outputs.extend(
                summarize_clustered(
                    class_frame,
                    value_cols=PRIMARY_REGION_METRICS,
                    identity={
                        **identity,
                        "scope": "per_class",
                        "stratum": "all",
                        "class_id": int(class_id),
                        "class_name": VOC_CLASS_NAMES[int(class_id)],
                    },
                    repeats=repeats,
                    seed=seed,
                    include_macro_class=False,
                )
            )
    return outputs


def region_distribution_rows(
    *,
    image_ids: Sequence[str],
    positive_labels: np.ndarray,
    semantic_labels: np.ndarray,
    image_label_counts: np.ndarray,
    scores: np.ndarray,
    graph: str,
    lambda_value: float,
    edges: LocalEdges,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Per-map region/interior distributions and target interior-boundary deltas."""

    rows: list[dict[str, object]] = []
    positive = np.asarray(positive_labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    for image_index, class_id in zip(*np.nonzero(positive)):
        semantic = np.asarray(semantic_labels[image_index]).reshape(-1)
        boundary = boundary_patch_mask(semantic.reshape(28, 28), edges).reshape(-1)
        regions = class_region_labels(semantic, int(class_id))
        base = {
            "image_id": str(image_ids[image_index]),
            "image_index": int(image_index),
            "class_id": int(class_id),
            "class_name": VOC_CLASS_NAMES[int(class_id)],
            "num_positive_classes": int(image_label_counts[image_index]),
            "label_stratum": label_stratum(int(image_label_counts[image_index])),
            "graph": graph,
            "lambda": float(lambda_value),
        }
        for region, value in (("target", 0), ("other_fg", 1), ("background", 2)):
            for location, mask in (
                ("interior", (regions == value) & ~boundary),
                ("boundary", (regions == value) & boundary),
            ):
                selected = values[image_index, class_id][mask]
                rows.append(
                    {
                        **base,
                        "region": region,
                        "location": location,
                        "num_patches": int(selected.size),
                        "mean_stability": float(selected.mean()) if selected.size else float("nan"),
                        "median_stability": float(np.median(selected)) if selected.size else float("nan"),
                        "q25_stability": float(np.quantile(selected, 0.25)) if selected.size else float("nan"),
                        "q75_stability": float(np.quantile(selected, 0.75)) if selected.size else float("nan"),
                    }
                )
    frame = pd.DataFrame(rows)
    differences: list[dict[str, object]] = []
    if not frame.empty:
        wide = frame.pivot_table(
            index=["image_id", "image_index", "class_id", "class_name", "num_positive_classes", "label_stratum", "graph", "lambda", "region"],
            columns="location", values="mean_stability", aggfunc="first",
        ).reset_index()
        for _, row in wide.iterrows():
            if np.isfinite(row.get("interior", np.nan)) and np.isfinite(row.get("boundary", np.nan)):
                differences.append(
                    {
                        **{name: row[name] for name in wide.columns if name not in ("interior", "boundary")},
                        "metric": "interior_minus_boundary_mean_stability",
                        "estimate": float(row["interior"] - row["boundary"]),
                    }
                )
    return rows, differences


def summarize_simple_rows(
    rows: Sequence[Mapping[str, object]], *, value_cols: Sequence[str], repeats: int, seed: int,
    group_columns: Sequence[str] = ("graph", "lambda"),
    per_class: bool = True,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    result: list[dict[str, object]] = []
    for keys, subset in frame.groupby(list(group_columns), sort=True):
        identity = dict(zip(group_columns, keys))
        for stratum, scoped in iter_all_and_label_strata(subset):
            result.extend(summarize_clustered(
                scoped, value_cols=value_cols,
                identity={**identity, "scope": "all_positive_classes", "stratum": stratum},
                repeats=repeats, seed=seed,
            ))
        if per_class and "class_id" in subset:
            for class_id, class_frame in subset.groupby("class_id", sort=True):
                result.extend(summarize_clustered(
                    class_frame, value_cols=value_cols,
                    identity={**identity, "scope": "per_class", "stratum": "all", "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)]},
                    repeats=repeats, seed=seed, include_macro_class=False,
                ))
    return result


def class_overlap_rows(
    *, image_ids: Sequence[str], positive_labels: np.ndarray, semantic_labels: np.ndarray,
    image_label_counts: np.ndarray, scores: np.ndarray, graph: str, lambda_value: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positive = np.asarray(positive_labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    for image_index in range(len(image_ids)):
        classes = np.flatnonzero(positive[image_index])
        if len(classes) < 2:
            continue
        valid = valid_patch_mask(semantic_labels[image_index])
        semantic = np.asarray(semantic_labels[image_index]).reshape(-1)
        for first, second in itertools.combinations(classes.tolist(), 2):
            first_map, second_map = values[image_index, first], values[image_index, second]
            first_top10 = stable_topk_mask(first_map, 0.10, valid)
            second_top10 = stable_topk_mask(second_map, 0.10, valid)
            shared = first_top10 & second_top10
            shared_count = int(shared.sum())
            shared_target = np.isin(semantic, (first + 1, second + 1))
            rows.append({
                "image_id": str(image_ids[image_index]), "image_index": int(image_index),
                "class_id": int(first), "class_name": VOC_CLASS_NAMES[int(first)],
                "other_class_id": int(second), "other_class_name": VOC_CLASS_NAMES[int(second)],
                "num_positive_classes": int(image_label_counts[image_index]),
                "label_stratum": label_stratum(int(image_label_counts[image_index])),
                "graph": graph, "lambda": float(lambda_value),
                "stability_spearman": spatial_spearman(first_map[valid], second_map[valid]),
                "top10_jaccard": jaccard(first_top10, second_top10),
                "top20_jaccard": jaccard(
                    stable_topk_mask(first_map, 0.20, valid), stable_topk_mask(second_map, 0.20, valid)
                ),
                "shared_count": shared_count,
                "shared_target_either_fraction": float((shared & shared_target).sum() / shared_count) if shared_count else float("nan"),
                "shared_background_fraction": float((shared & (semantic == 0)).sum() / shared_count) if shared_count else float("nan"),
                "shared_other_foreground_fraction": float((shared & (semantic > 0) & ~shared_target).sum() / shared_count) if shared_count else float("nan"),
            })
    return rows


def negative_class_rows(
    *, image_ids: Sequence[str], positive_labels: np.ndarray, semantic_labels: np.ndarray,
    image_label_counts: np.ndarray, scores: np.ndarray, graph: str, lambda_value: float,
    score_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positive = np.asarray(positive_labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    for image_index in range(len(image_ids)):
        valid = valid_patch_mask(semantic_labels[image_index])
        for class_id in range(20):
            local = values[image_index, class_id][valid]
            top = stable_topk_mask(values[image_index, class_id], 0.10, valid)
            rows.append({
                "image_id": str(image_ids[image_index]), "image_index": int(image_index),
                "class_id": class_id, "class_name": VOC_CLASS_NAMES[class_id],
                "num_positive_classes": int(image_label_counts[image_index]),
                "label_stratum": label_stratum(int(image_label_counts[image_index])),
                "presence": "positive" if positive[image_index, class_id] else "absent",
                "graph": graph, "lambda": float(lambda_value), "score": score_name,
                "mean_score": float(local.mean()), "top10_mean_score": float(values[image_index, class_id][top].mean()),
                "max_score": float(local.max()), "spatial_entropy": zscore_spatial_entropy(local),
                "high_score_fraction": float((local >= 0.8).mean()),
            })
    return rows


def boundary_distances(semantic_labels: np.ndarray, edges: LocalEdges) -> np.ndarray:
    """Patch-grid distance to a semantic boundary; void is retained as NaN."""

    labels = np.asarray(semantic_labels)
    result = np.full(labels.shape, np.nan, dtype=np.float32)
    for image_index, row in enumerate(labels):
        boundary = boundary_patch_mask(row.reshape(28, 28), edges)
        valid = row.reshape(28, 28) != PATCH_LABEL_VOID
        if valid.any() and boundary.any():
            distance = distance_transform_edt(~boundary)
            result[image_index] = np.where(valid, distance, np.nan).reshape(-1)
        elif valid.any():
            result[image_index] = np.where(valid, 0.0, np.nan).reshape(-1)
    return result


def local_map_variance(values: np.ndarray, edges: LocalEdges) -> np.ndarray:
    """Unweighted mean squared change to each 8-neighbor, [I,C,N]."""

    maps = np.asarray(values, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[-1] != 784:
        raise ValueError("maps must have shape [I,C,784]")
    output = np.zeros_like(maps, dtype=np.float64)
    counts = np.zeros(784, dtype=np.float64)
    for source, target in zip(edges.source, edges.target):
        delta = np.square(maps[:, :, source] - maps[:, :, target])
        output[:, :, source] += delta
        output[:, :, target] += delta
        counts[source] += 1
        counts[target] += 1
    return output / counts[None, None, :]


def correlation_rows(
    *, image_ids: Sequence[str], positive_labels: np.ndarray, semantic_labels: np.ndarray,
    image_label_counts: np.ndarray, stability: np.ndarray, raw_maps: np.ndarray,
    node_degree: np.ndarray, boundary_distance: np.ndarray, local_variance: np.ndarray,
    graph: str, lambda_value: float, edges: LocalEdges,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pos = np.asarray(positive_labels, dtype=bool)
    stability = np.asarray(stability, dtype=np.float64)
    raw = np.asarray(raw_maps, dtype=np.float64)
    for image_index, class_id in zip(*np.nonzero(pos)):
        semantic = semantic_labels[image_index]
        valid = valid_patch_mask(semantic)
        boundary = boundary_patch_mask(semantic.reshape(28, 28), edges).reshape(-1)
        quantities = {
            "raw_response": raw[image_index, class_id],
            "relu_response": np.maximum(raw[image_index, class_id], 0.0),
            "boundary_distance": boundary_distance[image_index],
            "local_graph_degree": node_degree[image_index],
            "local_class_map_variance": local_variance[image_index, class_id],
        }
        for stratum, mask in (("all_valid", valid), ("semantic_interior", valid & ~boundary), ("semantic_boundary", valid & boundary)):
            if int(mask.sum()) < 3:
                continue
            for name, quantity in quantities.items():
                finite = mask & np.isfinite(quantity)
                if int(finite.sum()) < 3:
                    continue
                rows.append({
                    "image_id": str(image_ids[image_index]), "image_index": int(image_index),
                    "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "num_positive_classes": int(image_label_counts[image_index]),
                    "label_stratum": label_stratum(int(image_label_counts[image_index])),
                    "graph": graph, "lambda": float(lambda_value), "stratum": stratum,
                    "quantity": name,
                    "spearman_stability_vs_quantity": spatial_spearman(stability[image_index, class_id][finite], quantity[finite]),
                })
    return rows


def relevance_maps(raw_maps: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    raw = np.asarray(raw_maps, dtype=np.float64)
    relu = np.maximum(raw, 0.0)
    maxima = relu.max(axis=-1, keepdims=True)
    return np.divide(relu, maxima, out=np.zeros_like(relu), where=maxima > float(epsilon))


def within_map_quintiles(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Stable five equal-count bins over eligible patches; ineligible = -1."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    result = np.full(scores.size, -1, dtype=np.int8)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return result
    order = np.argsort(-scores[indices], kind="stable")
    ranks_desc = np.empty(indices.size, dtype=np.int64)
    ranks_desc[order] = np.arange(indices.size)
    # q=4 is highest, q=0 is lowest despite descending ranks.
    result[indices] = (4 - np.floor(5 * ranks_desc / indices.size)).astype(np.int8)
    return result


def ratio_summary(
    frame: pd.DataFrame,
    *, image_ids: Sequence[str], numerator: str, denominator: str,
    identity: Mapping[str, object], repeats: int, seed: int,
) -> dict[str, object]:
    """A ratio with whole-image multinomial bootstrap and no patch resampling."""

    indexed = frame.groupby("image_id", sort=False)[[numerator, denominator]].sum()
    aligned = indexed.reindex(list(map(str, image_ids)), fill_value=0.0)
    draws: ClusterDraws = make_cluster_draws(list(map(str, image_ids)), repeats=repeats, seed=seed)
    numerator_values = aligned[numerator].to_numpy(dtype=np.float64)
    denominator_values = aligned[denominator].to_numpy(dtype=np.float64)
    samples = ratio_bootstrap(draws, numerator_values, denominator_values)
    low, high, finite = percentile_interval(samples)
    top = float(numerator_values.sum())
    bottom = float(denominator_values.sum())
    return {
        **identity,
        "estimate": top / bottom if bottom > 0 else float("nan"),
        "ci_low": low if low is not None else float("nan"),
        "ci_high": high if high is not None else float("nan"),
        "num_images": int((denominator_values > 0).sum()),
        "num_images_total": len(image_ids),
        "bootstrap_repeats": repeats,
        "bootstrap_valid_repeats": finite,
        "bootstrap_seed": seed,
        "bootstrap_unit": "whole image",
    }


def paired_ratio_difference(
    *, image_ids: Sequence[str], first_numerator: np.ndarray, first_denominator: np.ndarray,
    second_numerator: np.ndarray, second_denominator: np.ndarray, repeats: int, seed: int,
) -> tuple[float, float, float, int]:
    """Paired image-cluster CI for ratio(first) - ratio(second)."""

    draws = make_cluster_draws(list(map(str, image_ids)), repeats=repeats, seed=seed)
    first = ratio_bootstrap(draws, first_numerator, first_denominator)
    second = ratio_bootstrap(draws, second_numerator, second_denominator)
    delta = first - second
    low, high, finite = percentile_interval(delta)
    point_first = first_numerator.sum() / first_denominator.sum() if first_denominator.sum() else float("nan")
    point_second = second_numerator.sum() / second_denominator.sum() if second_denominator.sum() else float("nan")
    return float(point_first - point_second), float(low if low is not None else np.nan), float(high if high is not None else np.nan), finite


def score_pair_overlap(
    first: np.ndarray, second: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    return {
        "spearman": spatial_spearman(first[eligible], second[eligible]),
        "top10_jaccard": jaccard(
            stable_topk_mask(first, 0.10, eligible), stable_topk_mask(second, 0.10, eligible)
        ),
    }
