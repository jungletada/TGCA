#!/usr/bin/env python3
"""Create deterministic summary plots from Experiment 1 analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {"mctformer": "MCTformer", "mctformer_plus": "MCTformer+"}
MODEL_COLORS = {"mctformer": "#3568A8", "mctformer_plus": "#D45B37"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xlabel("Layer")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(1, 13))
    axis.grid(alpha=0.2, linewidth=0.7)


def plot_line_ci(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    label: str,
    color: str,
    ci_stem: str | None = None,
) -> None:
    ci_stem = ci_stem or value
    x = frame["layer"].to_numpy()
    y = frame[value].to_numpy()
    low = frame[f"{ci_stem}_ci_low"].to_numpy()
    high = frame[f"{ci_stem}_ci_high"].to_numpy()
    axis.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, label=label, color=color)
    axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_layer_summaries(tables: Path, output: Path) -> None:
    summaries = {
        model: pd.read_csv(tables / f"layerwise_summary_{model}.csv").query(
            "aggregation == 'micro'"
        )
        for model in MODEL_LABELS
    }
    panels = (
        ("mean_score", "Mean cosine score"),
        ("mean_max_score", "Mean map maximum"),
        ("mean_q95", "Mean map q95"),
        ("mean_upper_tail_gap", "Mean q95 − median"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, ylabel) in zip(axes.reshape(-1), panels):
        for model, frame in summaries.items():
            plot_line_ci(
                axis,
                frame,
                metric,
                MODEL_LABELS[model],
                MODEL_COLORS[model],
                ci_stem=metric.removesuffix("_mean"),
            )
        style_axis(axis, ylabel)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Layer-wise representation score scale and upper tail (micro mean, 95% image-cluster CI)")
    save_figure(figure, output / "layerwise_score_scale_and_tail.png")

    panels = (
        ("mean_top10_concentration", "Top-10% mean − median"),
        ("mean_spatial_entropy", "Spatial softmax entropy (τ=0.10)"),
        ("mean_total_variation", "Total variation"),
        (
            "mean_largest_component_fraction",
            "Largest top-10% component fraction",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, ylabel) in zip(axes.reshape(-1), panels):
        for model, frame in summaries.items():
            plot_line_ci(
                axis,
                frame,
                metric,
                MODEL_LABELS[model],
                MODEL_COLORS[model],
                ci_stem=metric.removesuffix("_mean"),
            )
        style_axis(axis, ylabel)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Concentration and spatial structure (representation maps only)")
    save_figure(figure, output / "layerwise_concentration_and_structure.png")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    entropy_panels = (
        ("mean_spatial_entropy_tau_050", "τ=0.05"),
        ("mean_spatial_entropy", "τ=0.10"),
        ("mean_spatial_entropy_tau_200", "τ=0.20"),
    )
    for axis, (metric, title) in zip(axes, entropy_panels):
        for model, frame in summaries.items():
            plot_line_ci(
                axis,
                frame,
                metric,
                MODEL_LABELS[model],
                MODEL_COLORS[model],
            )
        style_axis(axis, "Normalized spatial entropy")
        axis.set_title(title)
    axes[0].legend(frameon=False)
    figure.suptitle("Fixed-temperature entropy sensitivity (not attention entropy)")
    save_figure(figure, output / "spatial_entropy_temperature_sensitivity.png")


def plot_rank_and_diversity(tables: Path, output: Path) -> None:
    rank_frames = {
        model: pd.read_csv(tables / f"layer_rank_stability_{model}.csv")
        for model in MODEL_LABELS
    }
    panels = (
        ("mean_consecutive_layer_spearman", "Consecutive-layer Spearman"),
        ("mean_layer1_to_layer_spearman", "Layer-1-to-layer Spearman"),
        (
            "mean_consecutive_layer_top10_jaccard",
            "Consecutive top-10% Jaccard",
        ),
        (
            "mean_layer1_to_layer_top10_jaccard",
            "Layer-1-to-layer top-10% Jaccard",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, ylabel) in zip(axes.reshape(-1), panels):
        for model, frame in rank_frames.items():
            finite = frame[np.isfinite(frame[metric])]
            plot_line_ci(
                axis,
                finite,
                metric,
                MODEL_LABELS[model],
                MODEL_COLORS[model],
            )
        style_axis(axis, ylabel)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Cross-layer spatial rank stability (95% image-cluster CI)")
    save_figure(figure, output / "layer_rank_stability.png")

    diversity = pd.read_csv(tables / "class_map_diversity_by_layer.csv")
    panels = (
        ("pairwise_spearman_mean", "Within-image class-pair Spearman"),
        ("pairwise_cosine_mean", "Within-image class-pair cosine"),
        ("top10_jaccard_mean", "Within-image class-pair top-10% Jaccard"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for axis, (metric, ylabel) in zip(axes, panels):
        for model in MODEL_LABELS:
            frame = diversity[diversity["model"] == model]
            plot_line_ci(
                axis,
                frame,
                metric,
                MODEL_LABELS[model],
                MODEL_COLORS[model],
                ci_stem=metric.removesuffix("_mean"),
            )
        style_axis(axis, ylabel)
    axes[0].legend(frameon=False)
    figure.suptitle("Multi-class map diversity on multi-label images")
    save_figure(figure, output / "class_map_diversity_by_layer.png")


def plot_paired(tables: Path, output: Path) -> None:
    paired = pd.read_csv(tables / "mctformer_vs_plus_paired.csv")
    panels = (
        ("upper_tail_gap", "Δ upper-tail gap"),
        ("spatial_entropy", "Δ spatial entropy (τ=0.10)"),
        ("top10_class_map_jaccard", "Δ class-map top-10% Jaccard"),
        ("consecutive_layer_spearman", "Δ consecutive-layer Spearman"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, ylabel) in zip(axes.reshape(-1), panels):
        frame = paired[paired["metric"] == metric]
        axis.axhline(0.0, color="#333333", linewidth=0.9)
        axis.plot(
            frame["layer"],
            frame["paired_delta"],
            marker="o",
            markersize=3.5,
            color="#6A3D9A",
        )
        axis.fill_between(
            frame["layer"],
            frame["ci_low"],
            frame["ci_high"],
            color="#6A3D9A",
            alpha=0.18,
            linewidth=0,
        )
        style_axis(axis, ylabel)
    figure.suptitle("MCTformer+ − MCTformer paired deltas (95% image-cluster CI)")
    save_figure(figure, output / "paired_primary_metric_deltas.png")

    classwise = pd.read_csv(tables / "classwise_paired_delta.csv")
    classwise = classwise[
        (classwise["layer"] == 12)
        & (classwise["metric"].isin(["upper_tail_gap", "spatial_entropy"]))
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    class_order = list(
        classwise[classwise["metric"] == "upper_tail_gap"]
        .sort_values("paired_delta")["class_name"]
    )
    y = np.arange(len(class_order))
    for axis, metric, title in zip(
        axes,
        ("upper_tail_gap", "spatial_entropy"),
        ("Upper-tail gap", "Spatial entropy (τ=0.10)"),
    ):
        frame = classwise[classwise["metric"] == metric].set_index("class_name").loc[
            class_order
        ]
        error = np.vstack(
            (
                frame["paired_delta"] - frame["ci_low"],
                frame["ci_high"] - frame["paired_delta"],
            )
        )
        axis.errorbar(
            frame["paired_delta"],
            y,
            xerr=error,
            fmt="o",
            markersize=4,
            color="#6A3D9A",
            ecolor="#999999",
            capsize=2,
        )
        axis.axvline(0.0, color="#333333", linewidth=0.9)
        axis.set_title(title)
        axis.set_xlabel("Paired delta (MCTformer+ − MCTformer)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, class_order)
    figure.suptitle("Layer-12 class-wise paired representation differences")
    save_figure(figure, output / "classwise_layer12_paired_deltas.png")


def main() -> None:
    args = parse_args()
    analysis = args.analysis_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = analysis / "tables"
    plot_layer_summaries(tables, output)
    plot_rank_and_diversity(tables, output)
    plot_paired(tables, output)
    files = sorted(output.glob("*.png"))
    if len(files) != 7 or any(path.stat().st_size == 0 for path in files):
        raise RuntimeError(f"expected seven non-empty plots, found {files}")


if __name__ == "__main__":
    main()
