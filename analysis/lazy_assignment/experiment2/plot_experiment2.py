#!/usr/bin/env python3
"""Create the thirteen pre-registered Experiment 2 diagnostic figures.

Only aggregate CSV tables are read. Missing files or slices are visibly
annotated in their corresponding panel; they are never replaced by synthetic
or default values.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from analysis.lazy_assignment.experiment2.common import sha256_file  # noqa: E402


TABLE_FILES = (
    "layerwise_region_metrics.csv",
    "cam_stage_region_metrics.csv",
    "stage_transition_metrics.csv",
    "shared_support_ownership.csv",
    "new_shared_support_l9_l12.csv",
    "last_three_aggregation_analysis.csv",
    "classwise_results.csv",
    "paired_model_deltas.csv",
    "probe_validity_raw_norm_qk_attn.csv",
    "class_token_similarity_vs_map_overlap.csv",
    "classification_stratified_results.csv",
)

PLOT_FILES = (
    "feature_region_metrics_by_layer.png",
    "attention_region_metrics_by_layer.png",
    "feature_vs_attention_c_pim.png",
    "feature_vs_attention_bg_tail_enrichment.png",
    "shared_support_ownership_by_layer.png",
    "new_shared_support_l9_l12.png",
    "last3_attention_aggregation_analysis.png",
    "cam_stage_background_leakage.png",
    "stage_transition_background_survival.png",
    "target_retention_vs_bg_removal.png",
    "class_token_similarity_vs_map_overlap.png",
    "probe_validity_raw_norm_qk_attn.png",
    "classwise_l12_semantic_ownership.png",
)

MODEL_LABELS = {"mctformer": "MCTformer", "mctformer_plus": "MCTformer+"}
MODEL_COLORS = {"mctformer": "#4477AA", "mctformer_plus": "#CC6677"}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def load_tables(tables_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}
    for filename in TABLE_FILES:
        path = tables_dir / filename
        if not path.is_file():
            frames[filename] = pd.DataFrame()
            notes[filename] = f"Missing table: {filename}"
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, ValueError) as error:
            frames[filename] = pd.DataFrame()
            notes[filename] = f"Unreadable table: {filename} ({error})"
            continue
        if list(frame.columns) == ["status"] and frame["status"].eq("no_rows").all():
            frames[filename] = pd.DataFrame()
            notes[filename] = f"No rows: {filename}"
        else:
            frames[filename] = frame
    return frames, notes


def _preferred(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, expected in (("label_stratum", "all"), ("aggregation", "micro")):
        if column in result:
            result = result[result[column].astype(str) == expected]
    for column, expected in (("rho", 0.5), ("topk_ratio", 0.10)):
        if column in result:
            numeric = pd.to_numeric(result[column], errors="coerce")
            if np.isfinite(numeric).any():
                result = result[np.isclose(numeric, expected, atol=1e-9)]
    return result


def _filter(frame: pd.DataFrame, **values: object) -> pd.DataFrame:
    result = frame
    for column, expected in values.items():
        if column not in result:
            return result.iloc[0:0]
        if isinstance(expected, (set, tuple, list)):
            result = result[result[column].isin(expected)]
        elif callable(expected):
            result = result[expected(result[column])]
        else:
            result = result[result[column] == expected]
    return result


def _layer_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"(?:^|[^0-9])L?(\d+)(?:$|[^0-9])", str(value))
        return float(match.group(1)) if match else float("nan")


def _annotate_missing(axis: plt.Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#666666",
        fontsize=9,
        wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _line_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    x_column: str,
    group_columns: Sequence[str] = ("model",),
    reference: Optional[float] = None,
) -> None:
    if frame.empty or "metric" not in frame or "estimate" not in frame:
        _annotate_missing(axis, f"No matching rows\n{metric}")
        axis.set_title(title)
        return
    subset = frame[frame["metric"].astype(str) == metric].copy()
    subset["estimate"] = pd.to_numeric(subset["estimate"], errors="coerce")
    subset = subset[np.isfinite(subset["estimate"])]
    if x_column not in subset or subset.empty:
        _annotate_missing(axis, f"No matching rows\n{metric}")
        axis.set_title(title)
        return
    numeric_x = subset[x_column].map(_layer_value)
    use_numeric = np.isfinite(numeric_x).all()
    if use_numeric:
        subset["_x"] = numeric_x
        tick_labels = None
    else:
        categories = list(dict.fromkeys(subset[x_column].astype(str).tolist()))
        lookup = {value: index for index, value in enumerate(categories)}
        subset["_x"] = subset[x_column].astype(str).map(lookup).astype(float)
        tick_labels = categories
    existing_groups = [column for column in group_columns if column in subset]
    if len(existing_groups) == 1:
        grouped = subset.groupby(existing_groups[0], sort=True, dropna=False)
    elif existing_groups:
        grouped = subset.groupby(existing_groups, sort=True, dropna=False)
    else:
        grouped = [((), subset)]
    drew = False
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("_x", kind="stable")
        label_parts = []
        color = None
        for column, value in zip(existing_groups, keys):
            if column == "model":
                label_parts.append(MODEL_LABELS.get(str(value), str(value)))
                color = MODEL_COLORS.get(str(value))
            else:
                label_parts.append(str(value))
        label = " / ".join(label_parts) if label_parts else metric
        yerr = None
        if {"ci_low", "ci_high"}.issubset(group.columns):
            low = pd.to_numeric(group["ci_low"], errors="coerce").to_numpy(float)
            high = pd.to_numeric(group["ci_high"], errors="coerce").to_numpy(float)
            estimate = group["estimate"].to_numpy(float)
            if np.isfinite(low).all() and np.isfinite(high).all():
                yerr = np.vstack(
                    (np.maximum(0.0, estimate - low), np.maximum(0.0, high - estimate))
                )
        axis.errorbar(
            group["_x"],
            group["estimate"],
            yerr=yerr,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            capsize=2,
            label=label,
            color=color,
        )
        drew = True
    if not drew:
        _annotate_missing(axis, f"No finite rows\n{metric}")
    elif tick_labels is not None:
        axis.set_xticks(range(len(tick_labels)), tick_labels, rotation=30, ha="right")
    if reference is not None:
        axis.axhline(reference, color="#888888", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.set_xlabel(x_column.replace("_", " "))


def _finish(figure: plt.Figure, axes: np.ndarray, path: Path, dpi: int) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for axis in np.asarray(axes).reshape(-1):
        local_handles, local_labels = axis.get_legend_handles_labels()
        for handle, label in zip(local_handles, local_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8
        )
        figure.subplots_adjust(bottom=0.19)
    figure.tight_layout(rect=(0, 0.08 if handles else 0, 1, 1))
    figure.savefig(
        path, dpi=dpi, bbox_inches="tight", metadata={"Software": "TGCA Experiment 2"}
    )
    plt.close(figure)


def _multi_metric_plot(
    frame: pd.DataFrame,
    specifications: Sequence[tuple[str, str, Optional[float]]],
    *,
    path: Path,
    dpi: int,
    x_column: str,
    group_columns: Sequence[str],
) -> None:
    figure, axes = plt.subplots(
        1, len(specifications), figsize=(5.0 * len(specifications), 4.0), squeeze=False
    )
    for axis, (metric, title, reference) in zip(axes[0], specifications):
        _line_panel(
            axis,
            frame,
            metric=metric,
            title=title,
            x_column=x_column,
            group_columns=group_columns,
            reference=reference,
        )
    _finish(figure, axes, path, dpi)


def generate_plots(
    tables_dir: Path,
    output_dir: Path,
    dpi: int = 180,
    command: Optional[str] = None,
) -> dict[str, object]:
    tables_dir = tables_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite plot directory: {output_dir}")
    output_dir.mkdir(parents=True)
    frames, notes = load_tables(tables_dir)
    layer = _preferred(frames["layerwise_region_metrics.csv"])
    cam = _preferred(frames["cam_stage_region_metrics.csv"])
    transition = _preferred(frames["stage_transition_metrics.csv"])
    shared = _preferred(frames["shared_support_ownership.csv"])
    new_shared = _preferred(frames["new_shared_support_l9_l12.csv"])
    last3 = _preferred(frames["last_three_aggregation_analysis.csv"])
    classwise = _preferred(frames["classwise_results.csv"])
    probe = _preferred(frames["probe_validity_raw_norm_qk_attn.csv"])
    token = _preferred(frames["class_token_similarity_vs_map_overlap.csv"])
    if "metric" in token:
        token = token.copy()
        token["metric"] = token["metric"].replace(
            {
                "feature_top10_jaccard": "feature_post_top10_jaccard",
                "attn_top10_jaccard": "attn_c2p_top10_jaccard",
            }
        )

    _multi_metric_plot(
        _filter(layer, signal="feature_post"),
        (
            ("target_hit", "Feature C-PiM", None),
            ("bg_tail_enrich_10", "Feature BG enrichment@10%", 1.0),
            ("auc_target_bg", "Feature target-vs-BG AUROC", 0.5),
        ),
        path=output_dir / PLOT_FILES[0],
        dpi=dpi,
        x_column="layer",
        group_columns=("model",),
    )
    _multi_metric_plot(
        _filter(layer, signal={"attn", "attn_c2p_conditional"}),
        (
            ("target_hit", "Attention C-PiM", None),
            ("bg_tail_enrich_10", "Attention BG enrichment@10%", 1.0),
            ("conditional_bg_mass", "Conditional attention BG mass", None),
        ),
        path=output_dir / PLOT_FILES[1],
        dpi=dpi,
        x_column="layer",
        group_columns=("model",),
    )
    _multi_metric_plot(
        _filter(
            layer,
            signal={
                "feature_post",
                "feature_norm",
                "qk",
                "qk_mean",
                "attn",
                "attn_c2p_conditional",
            },
        ),
        (("target_hit", "C-PiM across probes", None),),
        path=output_dir / PLOT_FILES[2],
        dpi=dpi,
        x_column="layer",
        group_columns=("model", "signal"),
    )
    _multi_metric_plot(
        _filter(
            layer,
            signal={
                "feature_post",
                "feature_norm",
                "qk",
                "qk_mean",
                "attn",
                "attn_c2p_conditional",
            },
        ),
        (("bg_tail_enrich_10", "BG enrichment across probes", 1.0),),
        path=output_dir / PLOT_FILES[3],
        dpi=dpi,
        x_column="layer",
        group_columns=("model", "signal"),
    )
    _multi_metric_plot(
        _filter(shared, signal="feature_post"),
        (
            ("shared_target_a_fraction", "Shared target A", None),
            ("shared_target_b_fraction", "Shared target B", None),
            ("shared_other_fg_fraction", "Shared other foreground", None),
            ("shared_background_fraction", "Shared background", None),
        ),
        path=output_dir / PLOT_FILES[4],
        dpi=dpi,
        x_column="layer_or_stage",
        group_columns=("model",),
    )
    if "layer_or_stage" in new_shared:
        layer_number = new_shared["layer_or_stage"].map(_layer_value)
        new_shared = new_shared[layer_number.isin((10.0, 11.0, 12.0))]
    _multi_metric_plot(
        new_shared,
        (
            ("new_shared_target_a_fraction", "New shared target A", None),
            ("new_shared_target_b_fraction", "New shared target B", None),
            ("new_shared_other_fg_fraction", "New shared other FG", None),
            ("new_shared_background_fraction", "New shared background", None),
        ),
        path=output_dir / PLOT_FILES[5],
        dpi=dpi,
        x_column="layer_or_stage",
        group_columns=("model", "signal"),
    )
    if not last3.empty:
        last3 = last3.copy()
        signal = (
            last3["signal"].astype(str)
            if "signal" in last3
            else pd.Series("", index=last3.index)
        )
        stage = (
            last3["stage"].astype(str)
            if "stage" in last3
            else pd.Series("", index=last3.index)
        )
        last3["source"] = signal.mask(signal.isin(("", "nan")), stage)
        if "layer" in last3:
            layer_number = last3["layer"].map(_layer_value)
            single_layer = signal.eq("attn_c2p_conditional") & np.isfinite(layer_number)
            last3.loc[single_layer, "source"] = layer_number[single_layer].map(
                lambda value: f"attention L{int(value)}"
            )
    _multi_metric_plot(
        last3,
        (
            ("target_hit", "Last-three C-PiM", None),
            ("bg_tail_enrich_10", "Last-three BG enrichment", 1.0),
            ("conditional_bg_mass", "Last-three conditional BG mass", None),
        ),
        path=output_dir / PLOT_FILES[6],
        dpi=dpi,
        x_column="source",
        group_columns=("model",),
    )
    _multi_metric_plot(
        cam,
        (
            ("conditional_bg_mass", "CAM-stage conditional BG mass", None),
            ("target_hit", "CAM-stage C-PiM", None),
            ("bg_tail_enrich_10", "CAM-stage BG enrichment", 1.0),
        ),
        path=output_dir / PLOT_FILES[7],
        dpi=dpi,
        x_column="stage",
        group_columns=("model",),
    )
    _multi_metric_plot(
        transition,
        (
            ("survive_background", "Background top-tail survival", None),
            ("introduced_background_fraction", "Introduced background", None),
        ),
        path=output_dir / PLOT_FILES[8],
        dpi=dpi,
        x_column="transition",
        group_columns=("model",),
    )
    _multi_metric_plot(
        transition,
        (
            ("survive_target", "Target retention", None),
            ("removed_background_fraction", "Background removal", None),
        ),
        path=output_dir / PLOT_FILES[9],
        dpi=dpi,
        x_column="transition",
        group_columns=("model",),
    )
    _multi_metric_plot(
        token,
        (
            ("class_token_cosine", "Positive class-token cosine", None),
            ("feature_post_top10_jaccard", "Feature top-10% overlap", None),
            ("attn_c2p_top10_jaccard", "Attention top-10% overlap", None),
        ),
        path=output_dir / PLOT_FILES[10],
        dpi=dpi,
        x_column="layer",
        group_columns=("model",),
    )
    _multi_metric_plot(
        probe,
        (
            ("auc_target_bg", "Probe target-vs-BG AUROC", 0.5),
            ("target_hit", "Probe C-PiM", None),
            ("bg_tail_enrich_10", "Probe BG enrichment", 1.0),
        ),
        path=output_dir / PLOT_FILES[11],
        dpi=dpi,
        x_column="layer",
        group_columns=("model", "signal"),
    )
    if "layer" in classwise:
        classwise = classwise[
            np.isclose(pd.to_numeric(classwise["layer"], errors="coerce"), 12)
        ]
    classwise = _filter(classwise, signal="feature_post")
    _multi_metric_plot(
        classwise,
        (
            ("target_hit", "Class-wise L12 C-PiM", None),
            ("bg_tail_enrich_10", "Class-wise L12 BG enrichment", 1.0),
        ),
        path=output_dir / PLOT_FILES[12],
        dpi=dpi,
        x_column="class_id",
        group_columns=("model",),
    )
    # Replace numeric class ticks in the final figure is intentionally avoided:
    # table class IDs remain the unambiguous source of truth.
    result = {
        "status": "complete",
        "tables_dir": str(tables_dir),
        "input_table_sha256": {
            filename: sha256_file(tables_dir / filename)
            for filename in TABLE_FILES
            if (tables_dir / filename).is_file()
        },
        "output_dir": str(output_dir),
        "plots": [str(output_dir / filename) for filename in PLOT_FILES],
        "plot_sha256": {
            filename: sha256_file(output_dir / filename) for filename in PLOT_FILES
        },
        "missing_or_empty_tables": notes,
        "invented_values": False,
        "command": command,
    }
    (output_dir / "plot_metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    result = generate_plots(
        args.tables_dir,
        args.output_dir,
        args.dpi,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
