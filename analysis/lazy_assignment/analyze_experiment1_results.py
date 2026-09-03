#!/usr/bin/env python3
"""Analyze Experiment 1 canonical tables with image-clustered bootstrap CIs."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.lazy_assignment.bootstrap import (
    BootstrapEstimate,
    cluster_bootstrap_macro_class_means,
    cluster_bootstrap_means,
    cluster_standardized_effect,
    derived_seed,
)
from analysis.lazy_assignment.experiment1_analysis_common import (
    AnalysisLog,
    MODEL_ORDER,
    VOC_CLASS_NAMES,
    json_dump,
    timestamp,
)


SUMMARY_METRICS = OrderedDict(
    (
        ("mean_score", "score_mean"),
        ("median_score", "score_q50"),
        ("mean_max_score", "score_max"),
        ("mean_q95", "score_q95"),
        ("mean_upper_tail_gap", "upper_tail_gap"),
        ("mean_score_std", "score_std"),
        ("mean_top10_concentration", "top10_concentration"),
        ("mean_spatial_entropy", "spatial_entropy_tau_100"),
        ("mean_spatial_entropy_tau_050", "spatial_entropy_tau_050"),
        ("mean_spatial_entropy_tau_200", "spatial_entropy_tau_200"),
        ("mean_total_variation", "total_variation"),
        ("mean_neighbor_pearson", "neighbor_pearson"),
        ("mean_neighbor_spearman", "neighbor_spearman"),
        ("mean_num_components_top10", "num_components_top10"),
        (
            "mean_largest_component_fraction",
            "largest_component_fraction_top10",
        ),
    )
)
DELTA_METRICS = OrderedDict(
    (
        ("max_score", "score_max"),
        ("q95", "score_q95"),
        ("upper_tail_gap", "upper_tail_gap"),
        ("spatial_entropy", "spatial_entropy_tau_100"),
        ("total_variation", "total_variation"),
        (
            "top10_largest_component_fraction",
            "largest_component_fraction_top10",
        ),
    )
)
CLASS_METRICS = OrderedDict(
    (
        ("mean_max_score", "score_max"),
        ("mean_q95", "score_q95"),
        ("mean_upper_tail_gap", "upper_tail_gap"),
        ("mean_spatial_entropy", "spatial_entropy_tau_100"),
        ("mean_total_variation", "total_variation"),
        ("mean_top10_concentration", "top10_concentration"),
        (
            "mean_top10_largest_component",
            "largest_component_fraction_top10",
        ),
    )
)
PAIRED_MAP_METRICS = OrderedDict(
    (
        ("score_mean", "score_mean"),
        ("score_max", "score_max"),
        ("q95", "score_q95"),
        ("upper_tail_gap", "upper_tail_gap"),
        ("score_std", "score_std"),
        ("spatial_entropy", "spatial_entropy_tau_100"),
        ("total_variation", "total_variation"),
        (
            "largest_component_fraction",
            "largest_component_fraction_top10",
        ),
    )
)
PRIMARY_METRICS = {
    "upper_tail_gap",
    "spatial_entropy",
    "top10_class_map_jaccard",
    "consecutive_layer_spearman",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--topk-ratios", default="0.05,0.10,0.20")
    parser.add_argument("--entropy-temperatures", default="0.05,0.10,0.20")
    return parser.parse_args()


def add_estimates(
    row: dict[str, object],
    estimates: dict[str, BootstrapEstimate],
    output_to_source: OrderedDict[str, str],
) -> None:
    for output_name, source_name in output_to_source.items():
        estimate = estimates[source_name]
        row[output_name] = estimate.estimate
        row[f"{output_name}_ci_low"] = estimate.ci_low
        row[f"{output_name}_ci_high"] = estimate.ci_high


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")
    reloaded = pd.read_csv(path)
    if len(reloaded) != len(frame) or list(reloaded.columns) != list(frame.columns):
        raise RuntimeError(f"CSV round-trip mismatch: {path}")


def layerwise_summaries(
    maps: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    counts: dict[str, int] = {}
    source_metrics = list(SUMMARY_METRICS.values())
    for model in MODEL_ORDER:
        rows: list[dict[str, object]] = []
        for layer in range(1, 13):
            subset = maps[(maps["model"] == model) & (maps["layer"] == layer)]
            micro = cluster_bootstrap_means(
                subset,
                "image_id",
                source_metrics,
                repeats,
                derived_seed(seed, "layer-summary", model, layer, "micro"),
            )
            macro = cluster_bootstrap_macro_class_means(
                subset,
                "image_id",
                "class_id",
                source_metrics,
                repeats,
                derived_seed(seed, "layer-summary", model, layer, "macro"),
            )
            for aggregation, estimates in (("micro", micro), ("macro_class", macro)):
                row: dict[str, object] = {
                    "model": model,
                    "aggregation": aggregation,
                    "layer": layer,
                    "block_index": layer - 1,
                    "num_images": subset["image_id"].nunique(),
                    "num_image_class_pairs": len(subset),
                    "num_classes": subset["class_id"].nunique(),
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                }
                add_estimates(row, estimates, SUMMARY_METRICS)
                rows.append(row)
        frame = pd.DataFrame(rows)
        filename = f"layerwise_summary_{model}.csv"
        write_csv(frame, tables / filename)
        counts[filename] = len(frame)
        log(f"Wrote {filename}: {len(frame)} rows")
    return counts


def add_layer_deltas(maps: pd.DataFrame) -> pd.DataFrame:
    result = maps.sort_values(
        ["model", "image_id", "class_id", "layer"], ignore_index=True
    ).copy()
    group = result.groupby(["model", "image_id", "class_id"], sort=False)
    for metric in DELTA_METRICS.values():
        result[f"{metric}_from_layer1"] = result[metric] - group[metric].transform(
            "first"
        )
        result[f"{metric}_from_previous"] = result[metric] - group[metric].shift(1)
    return result


def layerwise_deltas(
    maps: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    deltas = add_layer_deltas(maps)
    counts: dict[str, int] = {}
    for model in MODEL_ORDER:
        rows: list[dict[str, object]] = []
        for layer in range(1, 13):
            base_subset = deltas[(deltas["model"] == model) & (deltas["layer"] == layer)]
            for delta_type in ("from_layer1", "from_previous"):
                if layer == 1 and delta_type == "from_previous":
                    continue
                source_cols = [
                    f"{source_metric}_{delta_type}"
                    for source_metric in DELTA_METRICS.values()
                ]
                for aggregation in ("micro", "macro_class"):
                    local_seed = derived_seed(
                        seed,
                        "layer-delta",
                        model,
                        layer,
                        delta_type,
                        aggregation,
                    )
                    if aggregation == "micro":
                        estimates = cluster_bootstrap_means(
                            base_subset,
                            "image_id",
                            source_cols,
                            repeats,
                            local_seed,
                        )
                    else:
                        estimates = cluster_bootstrap_macro_class_means(
                            base_subset,
                            "image_id",
                            "class_id",
                            source_cols,
                            repeats,
                            local_seed,
                        )
                    for output_metric, source_metric in DELTA_METRICS.items():
                        estimate = estimates[f"{source_metric}_{delta_type}"]
                        rows.append(
                            {
                                "model": model,
                                "aggregation": aggregation,
                                "delta_reference": delta_type,
                                "layer": layer,
                                "block_index": layer - 1,
                                "metric": output_metric,
                                "num_images": base_subset["image_id"].nunique(),
                                "num_image_class_pairs": len(base_subset),
                                "mean_delta": estimate.estimate,
                                "ci_low": estimate.ci_low,
                                "ci_high": estimate.ci_high,
                                "bootstrap_repeats": repeats,
                                "bootstrap_seed": seed,
                            }
                        )
        frame = pd.DataFrame(rows)
        filename = f"layerwise_delta_{model}.csv"
        write_csv(frame, tables / filename)
        counts[filename] = len(frame)
        log(f"Wrote {filename}: {len(frame)} rows")
    return counts


def rank_stability_tables(
    ranks: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    metrics = OrderedDict(
        (
            ("mean_consecutive_layer_spearman", "consecutive_layer_spearman"),
            ("mean_layer1_to_layer_spearman", "layer1_to_layer_spearman"),
            (
                "mean_consecutive_layer_top10_jaccard",
                "consecutive_layer_top10_jaccard",
            ),
            (
                "mean_layer1_to_layer_top10_jaccard",
                "layer1_to_layer_top10_jaccard",
            ),
        )
    )
    counts: dict[str, int] = {}
    for model in MODEL_ORDER:
        rows: list[dict[str, object]] = []
        for layer in range(1, 13):
            subset = ranks[(ranks["model"] == model) & (ranks["layer"] == layer)]
            row: dict[str, object] = {
                "model": model,
                "layer": layer,
                "block_index": layer - 1,
                "num_images": subset["image_id"].nunique(),
                "num_image_class_pairs": len(subset),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            }
            finite_cols = [
                source
                for source in metrics.values()
                if np.isfinite(subset[source].to_numpy(dtype=float)).any()
            ]
            estimates = cluster_bootstrap_means(
                subset,
                "image_id",
                finite_cols,
                repeats,
                derived_seed(seed, "rank", model, layer),
            )
            for output_name, source_name in metrics.items():
                if source_name not in estimates:
                    row[output_name] = np.nan
                    row[f"{output_name}_ci_low"] = np.nan
                    row[f"{output_name}_ci_high"] = np.nan
                else:
                    estimate = estimates[source_name]
                    row[output_name] = estimate.estimate
                    row[f"{output_name}_ci_low"] = estimate.ci_low
                    row[f"{output_name}_ci_high"] = estimate.ci_high
            rows.append(row)
        frame = pd.DataFrame(rows)
        filename = f"layer_rank_stability_{model}.csv"
        write_csv(frame, tables / filename)
        counts[filename] = len(frame)
        log(f"Wrote {filename}: {len(frame)} rows")
    return counts


def diversity_table(
    pairs: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    metrics = OrderedDict(
        (
            ("pairwise_spearman", "pairwise_class_spearman"),
            ("pairwise_cosine", "pairwise_class_cosine"),
            ("top05_jaccard", "top05_class_map_jaccard"),
            ("top10_jaccard", "top10_class_map_jaccard"),
            ("top20_jaccard", "top20_class_map_jaccard"),
        )
    )
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for layer in range(1, 13):
            subset = pairs[(pairs["model"] == model) & (pairs["layer"] == layer)]
            estimates = cluster_bootstrap_means(
                subset,
                "image_id",
                list(metrics.values()),
                repeats,
                derived_seed(seed, "diversity", model, layer),
            )
            row: dict[str, object] = {
                "model": model,
                "layer": layer,
                "block_index": layer - 1,
                "num_multilabel_images": subset["image_id"].nunique(),
                "num_class_pairs": len(subset),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
            }
            for output_name, source_name in metrics.items():
                estimate = estimates[source_name]
                row[f"{output_name}_mean"] = estimate.estimate
                row[f"{output_name}_ci_low"] = estimate.ci_low
                row[f"{output_name}_ci_high"] = estimate.ci_high
            rows.append(row)
    frame = pd.DataFrame(rows)
    filename = "class_map_diversity_by_layer.csv"
    write_csv(frame, tables / filename)
    log(f"Wrote {filename}: {len(frame)} rows")
    return {filename: len(frame)}


def classwise_tables(
    maps: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    source_metrics = list(CLASS_METRICS.values())
    for class_id, class_name in enumerate(VOC_CLASS_NAMES):
        for layer in range(1, 13):
            model_subsets: dict[str, pd.DataFrame] = {}
            for model in MODEL_ORDER:
                subset = maps[
                    (maps["model"] == model)
                    & (maps["class_id"] == class_id)
                    & (maps["layer"] == layer)
                ]
                model_subsets[model] = subset
                estimates = cluster_bootstrap_means(
                    subset,
                    "image_id",
                    source_metrics,
                    repeats,
                    derived_seed(seed, "classwise", model, class_id, layer),
                )
                row: dict[str, object] = {
                    "model": model,
                    "class_id": class_id,
                    "class_name": class_name,
                    "layer": layer,
                    "block_index": layer - 1,
                    "num_images": subset["image_id"].nunique(),
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                }
                add_estimates(row, estimates, CLASS_METRICS)
                summary_rows.append(row)

            left = model_subsets["mctformer"]
            right = model_subsets["mctformer_plus"]
            merge_cols = ["image_id", "class_id", "layer"]
            paired = left[merge_cols + source_metrics].merge(
                right[merge_cols + source_metrics],
                on=merge_cols,
                suffixes=("_mctformer", "_mctformer_plus"),
                validate="one_to_one",
            )
            delta_cols: list[str] = []
            for source_metric in source_metrics:
                delta = f"{source_metric}_delta"
                paired[delta] = (
                    paired[f"{source_metric}_mctformer_plus"]
                    - paired[f"{source_metric}_mctformer"]
                )
                delta_cols.append(delta)
            estimates = cluster_bootstrap_means(
                paired,
                "image_id",
                delta_cols,
                repeats,
                derived_seed(seed, "classwise-paired", class_id, layer),
            )
            for output_name, source_metric in CLASS_METRICS.items():
                delta = f"{source_metric}_delta"
                estimate = estimates[delta]
                paired_rows.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "layer": layer,
                        "block_index": layer - 1,
                        "metric": output_name.removeprefix("mean_"),
                        "num_images": paired["image_id"].nunique(),
                        "paired_delta": estimate.estimate,
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                        "standardized_effect": cluster_standardized_effect(
                            paired, "image_id", delta
                        ),
                        "bootstrap_repeats": repeats,
                        "bootstrap_seed": seed,
                    }
                )
        log(f"Completed class-wise inference for {class_id:02d} {class_name}")
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    write_csv(summary, tables / "classwise_summary.csv")
    write_csv(paired, tables / "classwise_paired_delta.csv")
    log(f"Wrote classwise_summary.csv: {len(summary)} rows")
    log(f"Wrote classwise_paired_delta.csv: {len(paired)} rows")
    return {
        "classwise_summary.csv": len(summary),
        "classwise_paired_delta.csv": len(paired),
    }


def paired_table(
    paired_maps: pd.DataFrame,
    ranks: pd.DataFrame,
    diversity_pairs: pd.DataFrame,
    tables: Path,
    repeats: int,
    seed: int,
    log: AnalysisLog,
) -> dict[str, int]:
    rows: list[dict[str, object]] = []

    def append_rows(
        frame: pd.DataFrame,
        layer: int,
        metric_map: OrderedDict[str, str],
        unit_type: str,
    ) -> None:
        delta_columns = [f"{source}_delta" for source in metric_map.values()]
        estimates = cluster_bootstrap_means(
            frame,
            "image_id",
            delta_columns,
            repeats,
            derived_seed(seed, "paired", unit_type, layer),
        )
        for metric, source in metric_map.items():
            delta_col = f"{source}_delta"
            estimate = estimates[delta_col]
            rows.append(
                {
                    "metric": metric,
                    "metric_family": unit_type,
                    "primary_metric": metric in PRIMARY_METRICS,
                    "layer": layer,
                    "block_index": layer - 1,
                    "n_images": frame["image_id"].nunique(),
                    "n_image_class_pairs": len(frame),
                    "unit_type": unit_type,
                    "mctformer_mean": frame[f"{source}_mctformer"].mean(),
                    "mctformer_plus_mean": frame[f"{source}_mctformer_plus"].mean(),
                    "paired_delta": estimate.estimate,
                    "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high,
                    "standardized_effect": cluster_standardized_effect(
                        frame, "image_id", delta_col
                    ),
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                }
            )

    for layer in range(1, 13):
        subset = paired_maps[paired_maps["layer"] == layer].copy()
        for source in PAIRED_MAP_METRICS.values():
            subset[f"{source}_delta"] = subset[f"{source}_delta"]
        append_rows(subset, layer, PAIRED_MAP_METRICS, "image_class_map")

    diversity_metrics = OrderedDict(
        (
            ("class_map_pairwise_spearman", "pairwise_class_spearman"),
            ("top10_class_map_jaccard", "top10_class_map_jaccard"),
        )
    )
    diversity_keys = ["image_id", "class_id_a", "class_id_b", "layer"]
    left = diversity_pairs[diversity_pairs["model"] == "mctformer"]
    right = diversity_pairs[diversity_pairs["model"] == "mctformer_plus"]
    diversity_sources = list(diversity_metrics.values())
    diversity_paired = left[diversity_keys + diversity_sources].merge(
        right[diversity_keys + diversity_sources],
        on=diversity_keys,
        suffixes=("_mctformer", "_mctformer_plus"),
        validate="one_to_one",
    )
    for source in diversity_sources:
        diversity_paired[f"{source}_delta"] = (
            diversity_paired[f"{source}_mctformer_plus"]
            - diversity_paired[f"{source}_mctformer"]
        )
    for layer in range(1, 13):
        append_rows(
            diversity_paired[diversity_paired["layer"] == layer],
            layer,
            diversity_metrics,
            "within_image_class_pair",
        )

    rank_metrics = OrderedDict(
        (
            ("consecutive_layer_spearman", "consecutive_layer_spearman"),
            (
                "consecutive_layer_top10_jaccard",
                "consecutive_layer_top10_jaccard",
            ),
        )
    )
    rank_keys = ["image_id", "class_id", "layer"]
    rank_sources = list(rank_metrics.values())
    left_rank = ranks[ranks["model"] == "mctformer"]
    right_rank = ranks[ranks["model"] == "mctformer_plus"]
    rank_paired = left_rank[rank_keys + rank_sources].merge(
        right_rank[rank_keys + rank_sources],
        on=rank_keys,
        suffixes=("_mctformer", "_mctformer_plus"),
        validate="one_to_one",
    )
    for source in rank_sources:
        rank_paired[f"{source}_delta"] = (
            rank_paired[f"{source}_mctformer_plus"]
            - rank_paired[f"{source}_mctformer"]
        )
    for layer in range(2, 13):
        append_rows(
            rank_paired[rank_paired["layer"] == layer],
            layer,
            rank_metrics,
            "image_class_layer_transition",
        )

    frame = pd.DataFrame(rows).sort_values(["metric_family", "metric", "layer"])
    filename = "mctformer_vs_plus_paired.csv"
    write_csv(frame, tables / filename)
    log(f"Wrote {filename}: {len(frame)} rows")
    return {filename: len(frame)}


def single_multilabel_table(
    maps: pd.DataFrame, tables: Path, repeats: int, seed: int, log: AnalysisLog
) -> dict[str, int]:
    metric_map = OrderedDict(
        (
            ("mean_q95", "score_q95"),
            ("mean_upper_tail_gap", "upper_tail_gap"),
            ("mean_spatial_entropy", "spatial_entropy_tau_100"),
            ("mean_total_variation", "total_variation"),
            (
                "mean_largest_component_fraction",
                "largest_component_fraction_top10",
            ),
        )
    )
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for layer in range(1, 13):
            for is_multilabel, label in ((False, "single_label"), (True, "multi_label")):
                subset = maps[
                    (maps["model"] == model)
                    & (maps["layer"] == layer)
                    & (maps["is_multilabel"] == is_multilabel)
                ]
                estimates = cluster_bootstrap_means(
                    subset,
                    "image_id",
                    list(metric_map.values()),
                    repeats,
                    derived_seed(seed, "label-cardinality", model, layer, label),
                )
                row: dict[str, object] = {
                    "model": model,
                    "image_group": label,
                    "layer": layer,
                    "block_index": layer - 1,
                    "num_images": subset["image_id"].nunique(),
                    "num_image_class_pairs": len(subset),
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                }
                add_estimates(row, estimates, metric_map)
                rows.append(row)
    frame = pd.DataFrame(rows)
    filename = "single_vs_multilabel_summary.csv"
    write_csv(frame, tables / filename)
    log(f"Wrote {filename}: {len(frame)} rows")
    return {filename: len(frame)}


def analyze(args: argparse.Namespace) -> dict[str, object]:
    canonical = args.canonical_dir.resolve()
    output = args.output_dir.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=False)
    log = AnalysisLog(output / "analysis.log")
    if args.bootstrap_repeats != 5000:
        log(f"WARNING: final guide specifies 5000 repeats; received {args.bootstrap_repeats}")
    if args.bootstrap_seed != 20260901:
        log(f"WARNING: prespecified seed is 20260901; received {args.bootstrap_seed}")
    if args.topk_ratios != "0.05,0.10,0.20":
        raise ValueError("canonical top-k ratios are fixed at 0.05,0.10,0.20")
    if args.entropy_temperatures != "0.05,0.10,0.20":
        raise ValueError("canonical entropy temperatures are fixed at 0.05,0.10,0.20")

    maps = pd.read_parquet(canonical / "per_image_class_layer.parquet")
    paired_maps = pd.read_parquet(canonical / "per_pair_layer.parquet")
    ranks = pd.read_parquet(canonical / "rank_stability.parquet")
    diversity_pairs = pd.read_parquet(canonical / "multiclass_pair_layer.parquet")
    log(
        f"Loaded canonical tables: maps={len(maps)}, paired={len(paired_maps)}, "
        f"rank={len(ranks)}, multiclass_pairs={len(diversity_pairs)}"
    )

    row_counts: dict[str, int] = {}
    row_counts.update(
        layerwise_summaries(
            maps, tables, args.bootstrap_repeats, args.bootstrap_seed, log
        )
    )
    row_counts.update(
        layerwise_deltas(maps, tables, args.bootstrap_repeats, args.bootstrap_seed, log)
    )
    row_counts.update(
        rank_stability_tables(
            ranks, tables, args.bootstrap_repeats, args.bootstrap_seed, log
        )
    )
    row_counts.update(
        diversity_table(
            diversity_pairs,
            tables,
            args.bootstrap_repeats,
            args.bootstrap_seed,
            log,
        )
    )
    row_counts.update(
        single_multilabel_table(
            maps, tables, args.bootstrap_repeats, args.bootstrap_seed, log
        )
    )
    row_counts.update(
        paired_table(
            paired_maps,
            ranks,
            diversity_pairs,
            tables,
            args.bootstrap_repeats,
            args.bootstrap_seed,
            log,
        )
    )
    row_counts.update(
        classwise_tables(maps, tables, args.bootstrap_repeats, args.bootstrap_seed, log)
    )

    metadata: dict[str, object] = {
        "generated_at": timestamp(),
        "canonical_dir": str(canonical),
        "bootstrap": {
            "unit": "image_id cluster",
            "paired_definition": "sample images with replacement and retain every image-class/class-pair row; deltas are MCTformer+ minus MCTformer",
            "confidence_interval": "2.5th and 97.5th percentiles",
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "patches_treated_as_independent": False,
            "image_class_pairs_treated_as_independent_for_ci": False,
        },
        "primary_metrics": sorted(PRIMARY_METRICS),
        "multiple_comparison_policy": "primary metrics prespecified; other layer-by-metric intervals are descriptive and are not selected post hoc",
        "row_counts": row_counts,
        "segmentation_ground_truth_loaded": False,
    }
    json_dump(tables / "analysis_metadata.json", metadata)
    log("All table analyses complete")
    return metadata


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
