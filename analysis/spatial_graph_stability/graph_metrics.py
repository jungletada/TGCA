"""Semantic metrics and clustered inference for frozen local patch graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES

from .bootstrap import (
    ClusterDraws,
    exact_rank_metrics,
    histogram_point_metrics,
    histogram_rank_bootstrap,
    make_cluster_draws,
    percentile_interval,
    rank_histograms_by_image,
    ratio_bootstrap,
    ratio_estimate,
)
from .graph import GRAPH_NAMES, LocalEdges


RATIO_METRICS = (
    "same_semantic_purity",
    "semantic_boundary_leakage",
    "foreground_only_purity",
    "foreground_background_leakage",
)
RANK_METRICS = (
    "edge_auroc",
    "edge_auprc",
    "foreground_only_edge_auroc",
    "foreground_only_edge_auprc",
)
ALL_METRICS = RATIO_METRICS + RANK_METRICS


def image_strata(label_counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(label_counts, dtype=np.int64)
    return {
        "all": np.arange(len(counts), dtype=np.int64),
        "single_label": np.flatnonzero(counts == 1),
        "two_label": np.flatnonzero(counts == 2),
        "three_plus_label": np.flatnonzero(counts >= 3),
    }


def _scope_masks(
    first: np.ndarray,
    second: np.ndarray,
    class_id: int | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    valid = (first >= 0) & (second >= 0)
    same = valid & (first == second)
    foreground = valid & (first > 0) & (second > 0)
    foreground_same = foreground & (first == second)
    foreground_background = valid & (
        ((first == 0) & (second > 0)) | ((second == 0) & (first > 0))
    )
    if class_id is None:
        scope = valid
        fg_scope = foreground
        positive = same
        fg_positive = foreground_same
    else:
        semantic_id = int(class_id) + 1
        incident = (first == semantic_id) | (second == semantic_id)
        scope = valid & incident
        positive = (first == semantic_id) & (second == semantic_id)
        fg_scope = scope & (first > 0) & (second > 0)
        fg_positive = positive
        foreground_background = scope & (
            ((first == semantic_id) & (second == 0))
            | ((second == semantic_id) & (first == 0))
        )
    ratio = {
        "same_semantic_purity": (positive, scope),
        "semantic_boundary_leakage": (scope & ~positive, scope),
        "foreground_only_purity": (fg_positive, fg_scope),
        "foreground_background_leakage": (foreground_background, scope),
    }
    rank = {
        "edge": (scope, positive),
        "foreground_only_edge": (fg_scope, fg_positive),
    }
    return ratio, rank


def _result_row(
    *,
    aggregation: str,
    stratum: str,
    graph: str,
    metric: str,
    estimate: float | None,
    bootstrap_values: np.ndarray,
    images: int,
    edges: int,
    repeats: int,
    seed: int,
    point_method: str,
    ci_method: str,
    histogram_estimate: float | None = None,
) -> dict[str, object]:
    low, high, finite = percentile_interval(bootstrap_values)
    approximation_error = (
        abs(estimate - histogram_estimate)
        if estimate is not None and histogram_estimate is not None
        else None
    )
    return {
        "aggregation": aggregation,
        "stratum": stratum,
        "graph": graph,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_finite_draws": finite,
        "n_images": images,
        "n_edges": edges,
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "point_method": point_method,
        "ci_method": ci_method,
        "histogram_point_estimate": histogram_estimate,
        "histogram_abs_error": approximation_error,
    }


def _evaluate_scope(
    *,
    labels: np.ndarray,
    weights: Mapping[str, np.ndarray],
    edges: LocalEdges,
    image_ids: list[str],
    draws: ClusterDraws,
    stratum: str,
    aggregation: str,
    class_id: int | None,
    bins: int,
    bootstrap_device: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    first = labels[:, edges.source]
    second = labels[:, edges.target]
    ratio_masks, rank_masks = _scope_masks(first, second, class_id)
    rows: list[dict[str, object]] = []
    bootstraps: dict[tuple[str, str], np.ndarray] = {}
    for graph in GRAPH_NAMES:
        graph_weight = np.asarray(weights[graph], dtype=np.float64)
        for metric, (numerator_mask, denominator_mask) in ratio_masks.items():
            numerator = np.sum(graph_weight * numerator_mask, axis=1)
            denominator = np.sum(graph_weight * denominator_mask, axis=1)
            bootstrap = ratio_bootstrap(draws, numerator, denominator)
            estimate = ratio_estimate(numerator, denominator)
            count = int(np.count_nonzero(denominator_mask))
            rows.append(
                _result_row(
                    aggregation=aggregation,
                    stratum=stratum,
                    graph=graph,
                    metric=metric,
                    estimate=estimate,
                    bootstrap_values=bootstrap,
                    images=len(image_ids),
                    edges=count,
                    repeats=draws.repeats,
                    seed=draws.seed,
                    point_method="exact weighted ratio of pooled local edges",
                    ci_method="whole-image clustered percentile bootstrap",
                )
            )
            bootstraps[(graph, metric)] = bootstrap

        for prefix, (sample_mask, positive_mask) in rank_masks.items():
            image_grid = np.broadcast_to(
                np.arange(len(image_ids), dtype=np.int64)[:, None], sample_mask.shape
            )
            sampled_images = image_grid[sample_mask]
            sampled_targets = positive_mask[sample_mask]
            sampled_scores = graph_weight[sample_mask]
            exact_auc, exact_ap = exact_rank_metrics(sampled_targets, sampled_scores)
            positive_hist, negative_hist = rank_histograms_by_image(
                sampled_images,
                sampled_targets,
                sampled_scores,
                num_images=len(image_ids),
                bins=bins,
                score_range=(0.0, 1.0),
            )
            histogram_auc, histogram_ap = histogram_point_metrics(
                positive_hist, negative_hist
            )
            bootstrap_auc, bootstrap_ap = histogram_rank_bootstrap(
                draws,
                positive_hist,
                negative_hist,
                device=bootstrap_device,
            )
            for suffix, estimate, histogram_estimate, bootstrap in (
                ("auroc", exact_auc, histogram_auc, bootstrap_auc),
                ("auprc", exact_ap, histogram_ap, bootstrap_ap),
            ):
                metric = f"{prefix}_{suffix}"
                rows.append(
                    _result_row(
                        aggregation=aggregation,
                        stratum=stratum,
                        graph=graph,
                        metric=metric,
                        estimate=estimate,
                        bootstrap_values=bootstrap,
                        images=len(image_ids),
                        edges=int(sample_mask.sum()),
                        repeats=draws.repeats,
                        seed=draws.seed,
                        point_method="exact pooled-edge sklearn rank metric",
                        ci_method=(
                            f"whole-image clustered percentile bootstrap from {bins}-bin "
                            "per-image sufficient histograms"
                        ),
                        histogram_estimate=histogram_estimate,
                    )
                )
                bootstraps[(graph, metric)] = bootstrap
    return rows, bootstraps


def _macro_rows(
    per_class_rows: list[dict[str, object]],
    per_class_bootstrap: Mapping[tuple[int, str, str], np.ndarray],
    *,
    stratum: str,
    image_count: int,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    row_lookup = {
        (int(row["class_id"]), str(row["graph"]), str(row["metric"])): row
        for row in per_class_rows
        if row["stratum"] == stratum
    }
    rows: list[dict[str, object]] = []
    bootstraps: dict[tuple[str, str], np.ndarray] = {}
    for graph in GRAPH_NAMES:
        for metric in ALL_METRICS:
            estimates = [
                row_lookup[(class_id, graph, metric)]["estimate"]
                for class_id in range(20)
                if row_lookup.get((class_id, graph, metric), {}).get("estimate") is not None
            ]
            arrays = [
                per_class_bootstrap[(class_id, graph, metric)]
                for class_id in range(20)
                if (class_id, graph, metric) in per_class_bootstrap
            ]
            if arrays:
                with np.errstate(invalid="ignore"):
                    bootstrap = np.nanmean(np.stack(arrays, axis=0), axis=0)
            else:
                bootstrap = np.full(repeats, np.nan, dtype=np.float64)
            estimate = float(np.mean(estimates)) if estimates else None
            rows.append(
                _result_row(
                    aggregation="macro_class",
                    stratum=stratum,
                    graph=graph,
                    metric=metric,
                    estimate=estimate,
                    bootstrap_values=bootstrap,
                    images=image_count,
                    edges=sum(
                        int(row_lookup.get((class_id, graph, metric), {}).get("n_edges", 0))
                        for class_id in range(20)
                    ),
                    repeats=repeats,
                    seed=seed,
                    point_method="unweighted mean over finite foreground-class estimates",
                    ci_method="paired whole-image cluster bootstrap then macro-class mean",
                )
            )
            bootstraps[(graph, metric)] = bootstrap
    return rows, bootstraps


def evaluate_graph_metrics(
    *,
    image_ids: list[str],
    image_label_counts: np.ndarray,
    semantic_labels: np.ndarray,
    weights: Mapping[str, np.ndarray],
    edges: LocalEdges,
    repeats: int,
    seed: int,
    bins: int,
    bootstrap_device: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    if semantic_labels.shape != (len(image_ids), edges.grid_size[0] * edges.grid_size[1]):
        raise ValueError("semantic-label cache shape is invalid")
    if set(weights) != set(GRAPH_NAMES):
        raise ValueError("graph cache does not contain exactly B0--B3")
    for value in weights.values():
        if value.shape != (len(image_ids), edges.count):
            raise ValueError("graph-weight cache shape is invalid")
    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    draw_values: dict[tuple[str, str, str, str], np.ndarray] = {}
    strata = image_strata(image_label_counts)
    for stratum, selected in strata.items():
        selected_ids = [image_ids[index] for index in selected]
        draws = make_cluster_draws(selected_ids, repeats=repeats, seed=seed)
        subset_labels = semantic_labels[selected]
        subset_weights = {name: value[selected] for name, value in weights.items()}
        rows, bootstraps = _evaluate_scope(
            labels=subset_labels,
            weights=subset_weights,
            edges=edges,
            image_ids=selected_ids,
            draws=draws,
            stratum=stratum,
            aggregation="micro",
            class_id=None,
            bins=bins,
            bootstrap_device=bootstrap_device,
        )
        metric_rows.extend(rows)
        for (graph, metric), values in bootstraps.items():
            draw_values[("micro", stratum, graph, metric)] = values

        # Full per-class and macro-class inference is required overall. Image
        # label-count strata are reported as micro estimates to avoid unstable
        # empty class-by-stratum cells while retaining image-clustered inference.
        if stratum != "all":
            continue
        class_bootstrap: dict[tuple[int, str, str], np.ndarray] = {}
        for class_id, class_name in enumerate(VOC_CLASS_NAMES):
            class_rows, class_draws = _evaluate_scope(
                labels=subset_labels,
                weights=subset_weights,
                edges=edges,
                image_ids=selected_ids,
                draws=draws,
                stratum=stratum,
                aggregation="class",
                class_id=class_id,
                bins=bins,
                bootstrap_device=bootstrap_device,
            )
            for row in class_rows:
                row["class_id"] = class_id
                row["class_name"] = class_name
                per_class_rows.append(row)
            for (graph, metric), values in class_draws.items():
                class_bootstrap[(class_id, graph, metric)] = values
        macro_rows, macro_draws = _macro_rows(
            per_class_rows,
            class_bootstrap,
            stratum="all",
            image_count=len(selected_ids),
            repeats=repeats,
            seed=seed,
        )
        metric_rows.extend(macro_rows)
        for (graph, metric), values in macro_draws.items():
            draw_values[("macro_class", "all", graph, metric)] = values

    lookup = {
        (str(row["aggregation"]), str(row["stratum"]), str(row["graph"]), str(row["metric"])): row
        for row in metric_rows
    }
    delta_rows: list[dict[str, object]] = []
    comparisons = (
        ("B3-B1", "B3_spatial_feature", "B1_spatial"),
        ("B3-B0", "B3_spatial_feature", "B0_uniform"),
        ("B3-B2", "B3_spatial_feature", "B2_feature"),
    )
    scopes = sorted({(key[0], key[1]) for key in draw_values})
    for aggregation, stratum in scopes:
        for label, first_graph, second_graph in comparisons:
            for metric in ALL_METRICS:
                first_key = (aggregation, stratum, first_graph, metric)
                second_key = (aggregation, stratum, second_graph, metric)
                if first_key not in draw_values or second_key not in draw_values:
                    continue
                difference = draw_values[first_key] - draw_values[second_key]
                low, high, finite = percentile_interval(difference)
                first_estimate = lookup[first_key]["estimate"]
                second_estimate = lookup[second_key]["estimate"]
                estimate = (
                    float(first_estimate) - float(second_estimate)
                    if first_estimate is not None and second_estimate is not None
                    else None
                )
                finite_values = difference[np.isfinite(difference)]
                delta_rows.append(
                    {
                        "aggregation": aggregation,
                        "stratum": stratum,
                        "comparison": label,
                        "first_graph": first_graph,
                        "second_graph": second_graph,
                        "metric": metric,
                        "estimate_delta": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "probability_delta_gt_zero": (
                            float(np.mean(finite_values > 0)) if finite_values.size else None
                        ),
                        "bootstrap_finite_draws": finite,
                        "bootstrap_repeats": repeats,
                        "bootstrap_seed": seed,
                        "bootstrap_unit": "whole image",
                        "paired": True,
                    }
                )
    histogram_errors = [
        float(row["histogram_abs_error"])
        for row in [*metric_rows, *per_class_rows]
        if row.get("histogram_abs_error") is not None
    ]
    audit = {
        "bootstrap_unit": "whole image",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "confidence_interval": "95% percentile",
        "paired_graph_draws": True,
        "histogram_bins": bins,
        "rank_point_estimates": "exact pooled-edge sklearn",
        "rank_ci_method": "per-image fixed-bin sufficient histograms",
        "maximum_full_sample_histogram_abs_error": max(histogram_errors, default=0.0),
        "image_counts_by_stratum": {key: int(len(value)) for key, value in strata.items()},
    }
    return metric_rows, per_class_rows, delta_rows, audit


def evaluate_boundary_metrics(
    *,
    image_ids: list[str],
    semantic_labels: np.ndarray,
    weights: Mapping[str, np.ndarray],
    edges: LocalEdges,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    first = semantic_labels[:, edges.source]
    second = semantic_labels[:, edges.target]
    valid = (first >= 0) & (second >= 0)
    cross = valid & (first != second)
    fg_bg = valid & (
        ((first == 0) & (second > 0)) | ((second == 0) & (first > 0))
    )
    boundary_nodes = np.zeros_like(semantic_labels, dtype=bool)
    for image_index in range(len(image_ids)):
        boundary_nodes[image_index, edges.source[cross[image_index]]] = True
        boundary_nodes[image_index, edges.target[cross[image_index]]] = True
    near = boundary_nodes[:, edges.source] | boundary_nodes[:, edges.target]
    categories = {
        "same_semantic_interior": valid & (first == second) & ~near,
        "same_semantic_boundary_near": valid & (first == second) & near,
        "cross_semantic_boundary": cross,
        "foreground_background_boundary": fg_bg,
    }
    draws = make_cluster_draws(image_ids, repeats=repeats, seed=seed)
    rows: list[dict[str, object]] = []
    for graph in GRAPH_NAMES:
        graph_weights = np.asarray(weights[graph], dtype=np.float64)
        for category, mask in categories.items():
            numerator = np.sum(graph_weights * mask, axis=1)
            denominator = mask.sum(axis=1).astype(np.float64)
            bootstrap = ratio_bootstrap(draws, numerator, denominator)
            low, high, finite = percentile_interval(bootstrap)
            values = graph_weights[mask]
            rows.append(
                {
                    "graph": graph,
                    "edge_category": category,
                    "edge_count": int(values.size),
                    "image_count": len(image_ids),
                    "mean_weight": float(values.mean()) if values.size else None,
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "median_weight": float(np.median(values)) if values.size else None,
                    "q10_weight": float(np.quantile(values, 0.1)) if values.size else None,
                    "q90_weight": float(np.quantile(values, 0.9)) if values.size else None,
                    "bootstrap_finite_draws": finite,
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                    "bootstrap_unit": "whole image",
                }
            )
    return rows
