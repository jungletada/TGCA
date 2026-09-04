#!/usr/bin/env python3
"""Generate evidence-bounded Experiment 2 reports from completed tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from analysis.lazy_assignment.experiment2.common import (  # noqa: E402
    VOC_CLASS_NAMES,
    timestamp,
)
from analysis.lazy_assignment.experiment2.delivery_validation import (  # noqa: E402
    verify_visual_deliverables,
)
from analysis.lazy_assignment.experiment2.plot_experiment2 import (  # noqa: E402
    PLOT_FILES,
    TABLE_FILES as PLOT_INPUT_TABLE_FILES,
)
from analysis.lazy_assignment.experiment2.select_experiment2_examples import (  # noqa: E402
    NEW_CATEGORIES,
)


TABLE_FILES = (
    "layerwise_region_metrics.csv",
    "cam_stage_region_metrics.csv",
    "target_visible_region_metrics.csv",
    "target_visible_classwise_results.csv",
    "stage_transition_metrics.csv",
    "stage_transition_classwise_results.csv",
    "shared_support_ownership.csv",
    "shared_support_class_marginals.csv",
    "new_shared_support_l9_l12.csv",
    "last_three_aggregation_analysis.csv",
    "classwise_results.csv",
    "paired_model_deltas.csv",
    "probe_validity_raw_norm_qk_attn.csv",
    "patch_norm_joint_control.csv",
    "class_token_similarity_vs_map_overlap.csv",
    "classification_stratified_results.csv",
    "classification_conditioned_classwise_results.csv",
    "class_pair_focal_classification_stratified_results.csv",
    "class_pair_joint_classification_stratified_results.csv",
    "multiclass_map_diversity.csv",
    "qk_head_region_summary.csv",
    "qk_head_classwise_results.csv",
    "per_image_class_failure_patterns.csv",
    "failure_pattern_summary.csv",
    "feature_attention_cam_linkage.csv",
    "priority_layer_results.csv",
    "class_token_map_overlap_association.csv",
    "class_token_map_overlap_endpoint_association.csv",
    "class_pair_macro_class_results.csv",
    "class_pair_classwise_results.csv",
    "checkpoint_classification_performance.csv",
    "raw_final_cam_miou.csv",
)

CASE_DESCRIPTIONS = {
    "A": "feature, attention, and final CAM all show background enrichment",
    "B": "feature background enrichment is filtered before final localization",
    "C": "attention routing is the first measured background-enriched level",
    "D": "final propagation is the first measured background-enriched level",
    "E": "late shared support is dominated by foreground ownership/collision",
    "F": "late shared support is majority background",
    "G": "raw post-block cosine reverses while QK/attention/CAM geometry succeeds",
    "UNRESOLVED": "no Case A–G condition is separated from its pre-specified reference by image-clustered 95% intervals",
}
DECISION_RULE_VERSION = "exp2-operational-v1-locked-before-full-run-2026-09-04"
DECISION_PRECEDENCE = ("G", "F", "E", "A", "B", "C", "D")


@dataclass(frozen=True)
class MetricRow:
    table: str
    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    identity: dict[str, object]
    num_images: Optional[int]
    num_rows: Optional[int]
    bootstrap_valid_fraction: Optional[float]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument(
        "--immutability-dir",
        type=Path,
        required=True,
        help="completed post-pipeline immutable-input verification directory",
    )
    parser.add_argument("--examples-dir", type=Path)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _json_or_empty(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256_or_unavailable(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for filename in TABLE_FILES:
        path = directory / filename
        try:
            frame = (
                pd.read_csv(path, low_memory=False)
                if path.is_file()
                else pd.DataFrame()
            )
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if list(frame.columns) == ["status"]:
            frame = pd.DataFrame()
        result[filename] = frame
    return result


def _layer_number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"L?(\d+)", str(value))
        return float(match.group(1)) if match else float("nan")


def _preferred(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in (("label_stratum", "all"), ("aggregation", "micro")):
        if column in result:
            result = result[result[column].astype(str) == value]
    for column, value in (("rho", 0.5), ("topk_ratio", 0.1)):
        if column in result:
            numeric = pd.to_numeric(result[column], errors="coerce")
            if np.isfinite(numeric).any():
                # Concatenated products such as paired_model_deltas contain
                # metrics for which rho/top-k is genuinely not applicable.
                # Retain those NaN identities while selecting the registered
                # value wherever the dimension exists.
                result = result[numeric.isna() | np.isclose(numeric, value, atol=1e-9)]
    return result


def metric_row(
    frame: pd.DataFrame,
    table: str,
    metric: str,
    *,
    model: Optional[str] = None,
    signal: Optional[object] = None,
    stage: Optional[str] = None,
    layer: Optional[int] = None,
    extra_filters: Optional[dict[str, object]] = None,
) -> Optional[MetricRow]:
    subset = _preferred(frame)
    filters = {"model": model, "signal": signal, "stage": stage}
    filters.update(extra_filters or {})
    for column, value in filters.items():
        if value is not None:
            if column not in subset:
                return None
            if isinstance(value, (set, tuple, list)):
                subset = subset[
                    subset[column].astype(str).isin([str(item) for item in value])
                ]
            else:
                subset = subset[subset[column].astype(str) == str(value)]
    if layer is not None:
        layer_column = next(
            (name for name in ("layer", "layer_or_stage") if name in subset), None
        )
        if layer_column is None:
            return None
        layer_values = pd.to_numeric(
            subset[layer_column].map(_layer_number), errors="coerce"
        ).to_numpy(dtype=float)
        subset = subset[np.isclose(layer_values, float(layer))]
    if "metric" not in subset or "estimate" not in subset:
        return None
    subset = subset[subset["metric"].astype(str) == metric].copy()
    subset["estimate"] = pd.to_numeric(subset["estimate"], errors="coerce")
    subset = subset[np.isfinite(subset["estimate"])]
    if subset.empty:
        return None
    # A duplicated identity would make prose selection ambiguous. Stable order
    # makes the selected row reproducible while preserving the ambiguity note in
    # the table inventory.
    row = subset.sort_index(kind="stable").iloc[0]
    low = float(pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0])
    high = float(
        pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
    )
    identity = {
        key: row[key]
        for key in (
            "model",
            "signal",
            "stage",
            "layer",
            "layer_or_stage",
            "rho",
            "topk_ratio",
            "label_stratum",
            "aggregation",
            "transition",
            "new_shared_transition",
            "delta",
            "source_table",
            "classification_subset",
        )
        if key in row and pd.notna(row[key])
    }

    def optional_int(name: str) -> Optional[int]:
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        return int(value) if pd.notna(value) else None

    valid_fraction = pd.to_numeric(
        pd.Series([row.get("bootstrap_valid_fraction")]), errors="coerce"
    ).iloc[0]
    return MetricRow(
        table,
        metric,
        float(row["estimate"]),
        low,
        high,
        identity,
        optional_int("num_images"),
        optional_int("num_rows"),
        float(valid_fraction) if pd.notna(valid_fraction) else None,
    )


def _format_metric(row: Optional[MetricRow]) -> str:
    if row is None:
        return "unavailable"
    if math.isfinite(row.ci_low) and math.isfinite(row.ci_high):
        return f"{row.estimate:.4f} (95% CI {row.ci_low:.4f}, {row.ci_high:.4f})"
    return f"{row.estimate:.4f} (CI unavailable)"


def _fact(label: str, row: Optional[MetricRow]) -> str:
    if row is None:
        return f"[Unsupported] {label}: the required table slice is unavailable."
    return f"[Fact] {label}: **{_format_metric(row)}** (`{row.table}`)."


def _inference(label: str, row: Optional[MetricRow], reference: float) -> str:
    if row is None or not (math.isfinite(row.ci_low) and math.isfinite(row.ci_high)):
        return f"[Unsupported] {label}: no finite image-clustered confidence interval is available."
    if row.ci_low > reference:
        relation = "above"
    elif row.ci_high < reference:
        relation = "below"
    else:
        return (
            f"[Unsupported] {label}: the 95% interval {_format_metric(row)} crosses "
            f"the registered reference {reference:.2f}."
        )
    return (
        f"[Statistical inference] {label}: the image-clustered 95% interval is "
        f"entirely {relation} {reference:.2f}."
    )


def _compact_metric(row: Optional[MetricRow]) -> str:
    if row is None:
        return "—"
    interval = (
        f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
        if math.isfinite(row.ci_low) and math.isfinite(row.ci_high)
        else f"{row.estimate:.3f} [CI NA]"
    )
    denominator = (
        f"; n={row.num_images} images/{row.num_rows} rows"
        if row.num_images is not None and row.num_rows is not None
        else ""
    )
    return interval + denominator


def _layer_focus_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | Signal | Layer | C-PiM | BG enrich@10% | target-vs-BG AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    models = ("mctformer", "mctformer_plus")
    signals = ("feature_post", "feature_norm", "qk_mean", "attn_c2p_conditional")
    for model in models:
        for signal in signals:
            for layer in (4, 5, 9, 10, 11, 12):
                rows = [
                    metric_row(
                        frame,
                        "layerwise_region_metrics.csv",
                        metric,
                        model=model,
                        signal=signal,
                        layer=layer,
                    )
                    for metric in ("target_hit", "bg_tail_enrich_10", "auc_target_bg")
                ]
                if not any(row is not None for row in rows):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            model,
                            signal,
                            str(layer),
                            *(_compact_metric(row) for row in rows),
                        )
                    )
                    + " |"
                )
    return "\n".join(lines)


def _new_shared_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Signal | Transition | new target-A | new target-B | new pair-target | new dominant-target | new other-FG | new BG |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (
        "new_shared_target_a_fraction",
        "new_shared_target_b_fraction",
        "new_shared_pair_target_fraction",
        "new_shared_dominant_target_fraction",
        "new_shared_other_fg_fraction",
        "new_shared_background_fraction",
    )
    for signal in ("feature_post", "feature_norm", "qk_mean", "attn_c2p_conditional"):
        for transition in ("L9_to_L10", "L10_to_L11", "L11_to_L12"):
            rows = [
                metric_row(
                    frame,
                    "new_shared_support_l9_l12.csv",
                    metric,
                    model="mctformer_plus",
                    signal=signal,
                    extra_filters={"new_shared_transition": transition},
                )
                for metric in metrics
            ]
            if any(row is not None for row in rows):
                lines.append(
                    "| "
                    + " | ".join(
                        (signal, transition, *(_compact_metric(row) for row in rows))
                    )
                    + " |"
                )
    return "\n".join(lines)


def _shared_layer_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Signal | Layer | target-A | target-B | pair-target | dominant-target | other-FG | BG | BG enrichment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (
        "shared_target_a_fraction",
        "shared_target_b_fraction",
        "shared_pair_target_fraction",
        "shared_dominant_target_fraction",
        "shared_other_fg_fraction",
        "shared_background_fraction",
        "shared_background_enrichment",
    )
    specifications = [
        (signal, str(layer), layer)
        for signal in (
            "feature_post",
            "feature_norm",
            "qk_mean",
            "attn_c2p_conditional",
        )
        for layer in (4, 5, 9, 10, 11, 12)
    ]
    specifications.extend(
        (
            ("attn_official_conditional", "official_last3", None),
            ("attn_mid3_conditional", "mid3_L4-L6", None),
        )
    )
    for signal, label, layer in specifications:
        rows = [
            metric_row(
                frame,
                "shared_support_ownership.csv",
                metric,
                model="mctformer_plus",
                signal=signal,
                layer=layer,
            )
            for metric in metrics
        ]
        if any(row is not None for row in rows):
            lines.append(
                "| "
                + " | ".join((signal, label, *(_compact_metric(row) for row in rows)))
                + " |"
            )
    return "\n".join(lines)


def _cam_stage_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | CAM stage | C-PiM | conditional BG mass | BG enrich@10% | target-vs-BG AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metrics = (
        "target_hit",
        "conditional_bg_mass",
        "bg_tail_enrich_10",
        "auc_target_bg",
    )
    for model in ("mctformer", "mctformer_plus"):
        for stage in ("patch_cam", "c2p_cam", "final_cam"):
            rows = [
                metric_row(
                    frame,
                    "cam_stage_region_metrics.csv",
                    metric,
                    model=model,
                    stage=stage,
                )
                for metric in metrics
            ]
            if any(row is not None for row in rows):
                lines.append(
                    "| "
                    + " | ".join(
                        (model, stage, *(_compact_metric(row) for row in rows))
                    )
                    + " |"
                )
    return "\n".join(lines)


def _linkage_table(frame: pd.DataFrame) -> str:
    specifications = (
        ("feature_norm→QK L12", "feature_norm_to_qk", 12),
        ("QK→attention L12", "qk_to_attn", 12),
        ("post-feature→attention L12", "feature_post_to_attn", 12),
        ("feature L12→patch CAM", "feature_l12_to_patch_cam", 12),
        ("official attention→C2P CAM", "official_attn_to_c2p_cam", 0),
        ("C2P CAM→final CAM", "c2p_cam_to_final_cam", 0),
    )
    metrics = (
        "spearman",
        "topk_jaccard",
        "survive_target",
        "survive_background",
        "introduced_background_fraction",
        "removed_background_fraction",
    )
    lines = [
        "| Model | Transition | Spearman | top-10% Jaccard | target survival | BG survival | BG introduced | BG removed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("mctformer", "mctformer_plus"):
        for label, transition, layer in specifications:
            rows = [
                metric_row(
                    frame,
                    "stage_transition_metrics.csv",
                    metric,
                    model=model,
                    layer=layer,
                    extra_filters={"transition": transition},
                )
                for metric in metrics
            ]
            if any(row is not None for row in rows):
                lines.append(
                    "| "
                    + " | ".join(
                        (model, label, *(_compact_metric(row) for row in rows))
                    )
                    + " |"
                )
    return "\n".join(lines)


def _transition_classwise_delta_table(frame: pd.DataFrame) -> str:
    """Render compact descriptive class-wise extrema for paired transitions."""

    required = {
        "class_id",
        "transition",
        "metric",
        "estimate",
        "aggregation",
        "label_stratum",
    }
    if not required.issubset(frame.columns):
        return "[Unsupported] Class-wise transition rows are unavailable."
    selected = _preferred(frame)
    if "model_or_delta" in selected:
        selected = selected[
            selected["model_or_delta"].astype(str) == "mctformer_plus_minus_mctformer"
        ]
    elif "delta" in selected:
        selected = selected[
            selected["delta"].astype(str) == "mctformer_plus_minus_mctformer"
        ]
    else:
        return "[Unsupported] Paired class-wise transition deltas are unavailable."
    specifications = (
        ("post-feature→attention L12", "feature_post_to_attn", 12),
        ("official attention→C2P CAM", "official_attn_to_c2p_cam", 0),
        ("C2P CAM→final CAM", "c2p_cam_to_final_cam", 0),
    )
    metrics = (
        "survive_background",
        "introduced_background_fraction",
        "removed_background_fraction",
    )
    lines = [
        "| Transition | Metric | Classes | Lowest focal-class Δ (+ − base) | Highest focal-class Δ (+ − base) |",
        "|---|---|---:|---|---|",
    ]
    for label, transition_name, layer in specifications:
        for metric in metrics:
            local = selected[
                (selected["transition"].astype(str) == transition_name)
                & (selected["metric"].astype(str) == metric)
            ].copy()
            if "layer" in local:
                layers = pd.to_numeric(
                    local["layer"].map(_layer_number), errors="coerce"
                ).to_numpy(dtype=float)
                local = local[np.isclose(layers, float(layer))]
            local["estimate"] = pd.to_numeric(local["estimate"], errors="coerce")
            local = local[np.isfinite(local["estimate"])]
            if local.empty:
                continue
            local = local.sort_values(["estimate", "class_id"], kind="stable")

            def class_cell(row: pd.Series) -> str:
                class_id = int(row["class_id"])
                class_name = (
                    VOC_CLASS_NAMES[class_id]
                    if 0 <= class_id < len(VOC_CLASS_NAMES)
                    else f"class_{class_id}"
                )
                return f"{class_name}: {_format_metric_row_series(row)}"

            lines.append(
                f"| {label} | {metric} | {local['class_id'].nunique()} | "
                f"{class_cell(local.iloc[0])} | {class_cell(local.iloc[-1])} |"
            )
    if len(lines) == 2:
        return "[Unsupported] Paired class-wise transition deltas are unavailable."
    return "\n".join(lines)


def _last_three_table(frame: pd.DataFrame) -> str:
    specifications = (
        ("A_c2p L10", "attn_c2p_conditional", None, 10),
        ("A_c2p L11", "attn_c2p_conditional", None, 11),
        ("A_c2p L12", "attn_c2p_conditional", None, 12),
        ("A_c2p native last3", "attn_official_conditional", None, None),
        ("A_c2p mid3 L4–L6", "attn_mid3_conditional", None, None),
        ("CAM with L10 A_c2p", None, "diagnostic_c2p_cam_l10", None),
        ("CAM with L11 A_c2p", None, "diagnostic_c2p_cam_l11", None),
        ("CAM with L12 A_c2p", None, "diagnostic_c2p_cam_l12", None),
        # The single-layer and mid3 diagnostic maps have already undergone the
        # native all-layer A_p2p propagation.  Compare them with the equally
        # propagated native endpoint, not the pre-propagation c2p_cam stage.
        ("CAM with native last3", None, "final_cam", None),
        ("CAM with mid3", None, "diagnostic_c2p_cam_mid3", None),
    )
    lines = [
        "| MCTformer+ diagnostic source | C-PiM | BG enrich@10% | target-vs-BG AUROC | conditional BG mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, signal, stage, layer in specifications:
        rows = [
            metric_row(
                frame,
                "last_three_aggregation_analysis.csv",
                metric,
                model="mctformer_plus",
                signal=signal,
                stage=stage,
                layer=layer,
            )
            for metric in (
                "target_hit",
                "bg_tail_enrich_10",
                "auc_target_bg",
                "conditional_bg_mass",
            )
        ]
        if any(row is not None for row in rows):
            lines.append(
                "| "
                + " | ".join((label, *(_compact_metric(row) for row in rows)))
                + " |"
            )
    return "\n".join(lines)


def _failure_pattern_table(frame: pd.DataFrame) -> str:
    labels = {
        "type_a_representation_filtered": "Type A: representation filtered",
        "type_b_attention_routing": "Type B: attention routing",
        "type_c_patch_head": "Type C: patch head",
        "type_d_propagation_amplification": "Type D: propagation",
        "type_e_full_pipeline": "Type E: full pipeline",
        "unclassified_pattern": "Unclassified by A–E",
    }
    lines = [
        "| Model | Full-set image-class pattern | Prevalence |",
        "|---|---|---:|",
    ]
    for model in ("mctformer", "mctformer_plus"):
        for metric, label in labels.items():
            row = metric_row(
                frame,
                "failure_pattern_summary.csv",
                metric,
                model=model,
            )
            if row is not None:
                lines.append(f"| {model} | {label} | {_compact_metric(row)} |")
    return "\n".join(lines)


def _paired_focus_table(frame: pd.DataFrame) -> str:
    specifications = (
        (
            "L12 feature target-vs-BG AUROC",
            "auc_target_bg",
            {"source_table": "layer_signal", "signal": "feature_post", "layer": 12},
        ),
        (
            "L12 attention target-vs-BG AUROC",
            "auc_target_bg",
            {
                "source_table": "layer_signal",
                "signal": "attn_c2p_conditional",
                "layer": 12,
            },
        ),
        (
            "Final-CAM target-vs-BG AUROC",
            "auc_target_bg",
            {"source_table": "cam_stage", "stage": "final_cam"},
        ),
        (
            "L12 feature shared-BG fraction",
            "shared_background_fraction",
            {
                "source_table": "shared_support",
                "signal": "feature_post",
                "layer_or_stage": "L12",
            },
        ),
        (
            "L11→L12 new shared-BG fraction",
            "new_shared_background_fraction",
            {
                "source_table": "shared_support",
                "signal": "feature_post",
                "new_shared_transition": "L11_to_L12",
            },
        ),
        (
            "C2P→final background survival",
            "survive_background",
            {
                "source_table": "stage_transition",
                "transition": "c2p_cam_to_final_cam",
                "layer": 0,
            },
        ),
    )
    lines = [
        "| Paired quantity | MCTformer+ − MCTformer |",
        "|---|---:|",
    ]
    for label, metric, filters in specifications:
        layer = filters.pop("layer", None)
        signal = filters.pop("signal", None)
        stage = filters.pop("stage", None)
        row = metric_row(
            frame,
            "paired_model_deltas.csv",
            metric,
            signal=signal,
            stage=stage,
            layer=layer,
            extra_filters=filters,
        )
        lines.append(f"| {label} | {_compact_metric(row)} |")
    return "\n".join(lines)


def _label_strata_table(frame: pd.DataFrame) -> str:
    subset = frame.copy()
    required = {
        "model",
        "signal",
        "layer",
        "rho",
        "aggregation",
        "label_stratum",
        "metric",
    }
    if not required.issubset(subset.columns):
        return "[Unsupported] Label-stratified layer table is unavailable."
    subset = subset[
        (subset["model"] == "mctformer_plus")
        & subset["signal"].isin(("feature_post", "attn_c2p_conditional"))
        & (pd.to_numeric(subset["layer"], errors="coerce") == 12)
        & np.isclose(pd.to_numeric(subset["rho"], errors="coerce"), 0.5)
        & (subset["aggregation"] == "micro")
        & subset["metric"].isin(("bg_tail_enrich_10", "auc_target_bg"))
    ]
    lines = [
        "| Signal | Label stratum | BG enrich@10% | target-vs-BG AUROC |",
        "|---|---|---:|---:|",
    ]
    for signal in ("feature_post", "attn_c2p_conditional"):
        for stratum in ("all", "single_label", "exactly_2_labels", "3plus_labels"):
            local = subset[
                (subset["signal"] == signal) & (subset["label_stratum"] == stratum)
            ]
            cells = []
            for metric in ("bg_tail_enrich_10", "auc_target_bg"):
                row = local[local["metric"] == metric]
                cells.append(
                    "—"
                    if row.empty
                    else _format_metric_row_series(row.sort_index().iloc[0])
                )
            lines.append(f"| {signal} | {stratum} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def _format_metric_row_series(row: pd.Series) -> str:
    estimate = float(
        pd.to_numeric(pd.Series([row.get("estimate")]), errors="coerce").iloc[0]
    )
    low = float(pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0])
    high = float(
        pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
    )
    images = pd.to_numeric(pd.Series([row.get("num_images")]), errors="coerce").iloc[0]
    text = (
        f"{estimate:.3f} [{low:.3f}, {high:.3f}]"
        if all(math.isfinite(value) for value in (estimate, low, high))
        else "CI unavailable"
    )
    return text + (f"; n={int(images)} images" if pd.notna(images) else "")


def _qk_head_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | L12 head | target mean QK | BG mean QK |",
        "|---|---:|---:|---:|",
    ]
    for model in ("mctformer", "mctformer_plus"):
        for head in range(6):
            target = metric_row(
                frame,
                "qk_head_region_summary.csv",
                f"qk_head{head}_target_mean",
                model=model,
                layer=12,
            )
            background = metric_row(
                frame,
                "qk_head_region_summary.csv",
                f"qk_head{head}_bg_mean",
                model=model,
                layer=12,
            )
            if target is not None or background is not None:
                lines.append(
                    f"| {model} | {head} | {_compact_metric(target)} | "
                    f"{_compact_metric(background)} |"
                )
    return "\n".join(lines)


def _patch_norm_joint_table(frame: pd.DataFrame) -> str:
    required = {
        "model_or_delta",
        "layer",
        "rho",
        "label_stratum",
        "aggregation",
        "aggregation_scope",
        "metric",
    }
    if frame.empty or not required.issubset(frame.columns):
        return "[Unsupported] Joint feature-score/patch-norm controls are unavailable."
    selected = frame[
        frame["layer"].isin((4, 5, 9, 10, 11, 12))
        & np.isclose(pd.to_numeric(frame["rho"], errors="coerce"), 0.5)
        & (frame["label_stratum"].astype(str) == "all")
        & (frame["aggregation"].astype(str) == "micro")
        & (frame["aggregation_scope"].astype(str) == "overall")
    ]
    metric_labels = {
        "post_cosine_patch_l2norm_pearson_bg": "Pearson(cosine,norm), BG",
        "feature_top10_bg_patch_l2norm_enrichment_vs_bg": (
            "mean norm(top10∩BG) / mean norm(BG)"
        ),
        "feature_top10_bg_below_valid_l2norm_median_fraction": (
            "P(norm≤valid median | top10∩BG)"
        ),
        "feature_top10_bg_above_valid_l2norm_q75_fraction": (
            "P(norm≥valid q75 | top10∩BG)"
        ),
    }
    lines = [
        "| Layer | Quantity | MCTformer+ | Paired Δ (MCTformer+−MCTformer) |",
        "|---:|---|---:|---:|",
    ]
    for layer in (4, 5, 9, 10, 11, 12):
        for metric, label in metric_labels.items():
            local = selected[
                (pd.to_numeric(selected["layer"], errors="coerce") == layer)
                & (selected["metric"].astype(str) == metric)
            ]
            plus = local[local["model_or_delta"].astype(str) == "mctformer_plus"]
            delta = local[
                local["model_or_delta"].astype(str) == "mctformer_plus_minus_mctformer"
            ]
            if plus.empty and delta.empty:
                continue
            lines.append(
                f"| {layer} | {label} | "
                f"{'—' if plus.empty else _format_metric_row_series(plus.sort_index().iloc[0])} | "
                f"{'—' if delta.empty else _format_metric_row_series(delta.sort_index().iloc[0])} |"
            )
    if len(lines) == 2:
        return "[Unsupported] Joint feature-score/patch-norm priority-layer rows are unavailable."
    return "\n".join(lines)


def _aggregation_coverage_table(
    classification_classwise: pd.DataFrame,
    target_visible_classwise: pd.DataFrame,
    qk_head_classwise: pd.DataFrame,
) -> str:
    """Inventory the Experiment 2 section-17 class-wise control coverage."""

    specifications = (
        (
            "Classification-conditioned layer/CAM",
            classification_classwise,
            "N/A: model-specific status",
        ),
        (
            "Target-visible layer/CAM",
            target_visible_classwise,
            "exact common image_id/class_id",
        ),
        (
            "QK head controls",
            qk_head_classwise,
            "exact common image_id/class_id",
        ),
    )
    lines = [
        "| Control | Per-model rows M / M+ | Paired delta rows | Classes | Label strata | Comparison |",
        "|---|---:|---:|---:|---|---|",
    ]
    stratum_order = (
        "all",
        "single_label",
        "exactly_2_labels",
        "3plus_labels",
    )
    for label, frame, comparison in specifications:
        if frame.empty or "model_or_delta" not in frame:
            model_counts = (0, 0)
            paired_rows = 0
            class_count = 0
            strata = "none"
        else:
            identities = frame["model_or_delta"].astype(str)
            model_counts = tuple(
                int((identities == model).sum())
                for model in ("mctformer", "mctformer_plus")
            )
            paired_rows = int((identities == "mctformer_plus_minus_mctformer").sum())
            class_count = (
                int(frame["class_id"].dropna().nunique()) if "class_id" in frame else 0
            )
            present = (
                set(frame["label_stratum"].dropna().astype(str))
                if "label_stratum" in frame
                else set()
            )
            strata = (
                ", ".join(stratum for stratum in stratum_order if stratum in present)
                or "none"
            )
        lines.append(
            f"| {label} | {model_counts[0]} / {model_counts[1]} | "
            f"{paired_rows} | {class_count} | {strata} | {comparison} |"
        )
    return "\n".join(lines)


def _classwise_extremes_table(frame: pd.DataFrame) -> str:
    required = {
        "class_id",
        "source_table",
        "model",
        "signal",
        "layer",
        "rho",
        "label_stratum",
        "aggregation",
        "metric",
        "estimate",
    }
    if not required.issubset(frame.columns):
        return "[Unsupported] Class-wise focus rows are unavailable."
    selected = frame[
        (frame["source_table"] == "layer_signal")
        & (frame["model"] == "mctformer_plus")
        & frame["signal"].isin(("feature_post", "attn_c2p_conditional"))
        & (pd.to_numeric(frame["layer"], errors="coerce") == 12)
        & np.isclose(pd.to_numeric(frame["rho"], errors="coerce"), 0.5)
        & (frame["label_stratum"] == "all")
        & (frame["aggregation"] == "micro")
        & (frame["metric"] == "auc_target_bg")
    ].copy()
    selected["estimate"] = pd.to_numeric(selected["estimate"], errors="coerce")
    selected = selected[np.isfinite(selected["estimate"])]
    lines = [
        "| Signal | Tail | Class | target-vs-BG AUROC |",
        "|---|---|---|---:|",
    ]
    for signal in ("feature_post", "attn_c2p_conditional"):
        local = selected[selected["signal"] == signal].sort_values(
            ["estimate", "class_id"], kind="stable"
        )
        chosen = [("lowest", row) for _, row in local.head(3).iterrows()]
        chosen.extend(("highest", row) for _, row in local.tail(3).iterrows())
        for tail, row in chosen:
            class_id = int(row["class_id"])
            class_name = (
                VOC_CLASS_NAMES[class_id]
                if 0 <= class_id < len(VOC_CLASS_NAMES)
                else str(class_id)
            )
            lines.append(
                f"| {signal} | {tail} | {class_id}: {class_name} | "
                f"{_format_metric_row_series(row)} |"
            )
    return "\n".join(lines)


def _association_table(frame: pd.DataFrame) -> str:
    required = {"model", "layer", "label_stratum", "aggregation", "metric"}
    if not required.issubset(frame.columns):
        return "[Unsupported] Token/map association rows are unavailable."
    selected = frame[
        (frame["model"] == "mctformer_plus")
        & frame["layer"].isin((4, 5, 9, 10, 11, 12))
        & (frame["label_stratum"] == "all")
        & frame["aggregation"].isin(("micro_pair_pearson", "image_mean_pearson"))
    ]
    metric_labels = {
        "pearson_class_token_cosine_vs_feature_post_top10_jaccard": "feature",
        "pearson_class_token_cosine_vs_qk_mean_top10_jaccard": "QK",
        "pearson_class_token_cosine_vs_attn_c2p_top10_jaccard": "attention",
    }
    lines = [
        "| Layer | Estimand | Map family | Pearson(token cosine, top-10% overlap) |",
        "|---:|---|---|---:|",
    ]
    for layer in (4, 5, 9, 10, 11, 12):
        for aggregation, estimand in (
            ("micro_pair_pearson", "pair-row micro; image bootstrap"),
            ("image_mean_pearson", "per-image pair mean"),
        ):
            for metric, label in metric_labels.items():
                local = selected[
                    (selected["layer"] == layer)
                    & (selected["metric"] == metric)
                    & (selected["aggregation"] == aggregation)
                ]
                if not local.empty:
                    lines.append(
                        f"| {layer} | {estimand} | {label} | "
                        f"{_format_metric_row_series(local.sort_index().iloc[0])} |"
                    )
    return "\n".join(lines)


def _endpoint_association_table(frame: pd.DataFrame) -> str:
    required = {
        "model",
        "layer",
        "label_stratum",
        "aggregation",
        "metric",
    }
    if not required.issubset(frame.columns):
        return "[Unsupported] Endpoint class-wise association rows are unavailable."
    selected = frame[
        (frame["model"].astype(str) == "mctformer_plus")
        & frame["layer"].isin((4, 5, 9, 10, 11, 12))
        & (frame["label_stratum"].astype(str) == "all")
        & (frame["aggregation"].astype(str) == "macro_class_pearson")
    ]
    metric_labels = {
        "pearson_class_token_cosine_vs_feature_post_top10_jaccard": "feature",
        "pearson_class_token_cosine_vs_qk_mean_top10_jaccard": "QK",
        "pearson_class_token_cosine_vs_attn_c2p_top10_jaccard": "attention",
    }
    lines = [
        "| Layer | Map family | Equal-class mean Pearson | Finite/observed focal classes |",
        "|---:|---|---:|---:|",
    ]
    for layer in (4, 5, 9, 10, 11, 12):
        for metric, label in metric_labels.items():
            local = selected[
                (selected["layer"] == layer) & (selected["metric"] == metric)
            ]
            if local.empty:
                continue
            row = local.sort_index().iloc[0]
            finite = pd.to_numeric(
                pd.Series([row.get("num_classes")]), errors="coerce"
            ).iloc[0]
            total = pd.to_numeric(
                pd.Series([row.get("num_classes_total")]), errors="coerce"
            ).iloc[0]
            denominator = (
                "—"
                if pd.isna(finite) or pd.isna(total)
                else f"{int(finite)}/{int(total)}"
            )
            lines.append(
                f"| {layer} | {label} | {_format_metric_row_series(row)} | "
                f"{denominator} |"
            )
    if len(lines) == 2:
        return "[Unsupported] Equal-class endpoint association rows are unavailable."
    return "\n".join(lines)


def _multiclass_diversity_table(frame: pd.DataFrame) -> str:
    lines = [
        "| MCTformer+ signal | Layer | mean positive-class-pair top-10% Jaccard |",
        "|---|---:|---:|",
    ]
    for signal in ("feature_post", "feature_norm", "qk_mean", "attn_c2p_conditional"):
        for layer in (4, 5, 9, 10, 11, 12):
            row = metric_row(
                frame,
                "multiclass_map_diversity.csv",
                "topk_jaccard",
                model="mctformer_plus",
                signal=signal,
                layer=layer,
            )
            if row is not None:
                lines.append(f"| {signal} | {layer} | {_compact_metric(row)} |")
    return "\n".join(lines)


def _classification_table(frame: pd.DataFrame) -> str:
    required = {"classification_subset", "signal"}
    if not required.issubset(frame.columns):
        return "[Unsupported] Classification-stratified rows are unavailable."
    statuses = (
        "both_positive",
        "class_only_positive",
        "patch_only_positive",
        "neither_positive",
        "either_negative",
    )
    lines = [
        "| Classification status | Signal/stage | BG enrich@10% | target-vs-BG AUROC |",
        "|---|---|---:|---:|",
    ]
    for status in statuses:
        for signal, layer in (
            ("feature_post", 12),
            ("attn_c2p_conditional", 12),
            ("final_cam", -1),
        ):
            filters = {"classification_subset": status}
            rows = [
                metric_row(
                    frame,
                    "classification_stratified_results.csv",
                    metric,
                    model="mctformer_plus",
                    signal=signal,
                    layer=layer,
                    extra_filters=filters,
                )
                for metric in ("bg_tail_enrich_10", "auc_target_bg")
            ]
            if any(row is not None for row in rows):
                lines.append(
                    f"| {status} | {signal} | {_compact_metric(rows[0])} | "
                    f"{_compact_metric(rows[1])} |"
                )
    return "\n".join(lines)


def _class_pair_classification_control_tables(
    focal: pd.DataFrame, joint: pd.DataFrame
) -> str:
    """Render focal-endpoint and unordered-pair classification controls."""

    focal_statuses = (
        "both_positive",
        "class_only_positive",
        "patch_only_positive",
        "neither_positive",
        "either_negative",
    )
    joint_statuses = (
        "both_classes_both_positive",
        "either_class_negative",
    )
    focal_specs = (
        (
            "shared feature support",
            "shared_own_target_fraction",
            {"source_table": "shared_support", "signal": "feature_post"},
        ),
        (
            "shared feature support",
            "shared_background_fraction",
            {"source_table": "shared_support", "signal": "feature_post"},
        ),
        (
            "feature map diversity",
            "topk_jaccard",
            {
                "source_table": "multiclass_map_diversity",
                "signal": "feature_post",
            },
        ),
        (
            "class-token pair",
            "class_token_cosine",
            {"source_table": "class_token_map_overlap"},
        ),
    )
    joint_specs = (
        (
            "shared feature support",
            "shared_pair_target_fraction",
            {"source_table": "shared_support", "signal": "feature_post"},
        ),
        (
            "shared feature support",
            "shared_background_fraction",
            {"source_table": "shared_support", "signal": "feature_post"},
        ),
        (
            "feature map diversity",
            "topk_jaccard",
            {
                "source_table": "multiclass_map_diversity",
                "signal": "feature_post",
            },
        ),
        (
            "class-token pair",
            "class_token_cosine",
            {"source_table": "class_token_map_overlap"},
        ),
    )

    lines = [
        "### Pair controls by classification status",
        "",
        "[Structural N/A] Positive-class-pair analyses have no single-label "
        "stratum. Machine-readable rows retain `all`, `exactly_2_labels`, and "
        "`3plus_labels` wherever samples exist.",
        "",
        "Focal endpoint (the focal class status; `either_negative` is the union "
        "of the other three non-`both_positive` states):",
        "",
        "| Focal status | Pair family | MCTformer+ L12 micro quantity | Estimate |",
        "|---|---|---|---:|",
    ]
    rendered_rows = 0
    for status in focal_statuses:
        for family, metric, filters in focal_specs:
            row = metric_row(
                focal,
                "class_pair_focal_classification_stratified_results.csv",
                metric,
                model="mctformer_plus",
                layer=12,
                extra_filters={**filters, "classification_subset": status},
            )
            if row is not None:
                lines.append(
                    f"| {status} | {family} | {metric} | {_compact_metric(row)} |"
                )
                rendered_rows += 1

    lines.extend(
        [
            "",
            "Unordered pair (both positive classes jointly pass both heads versus "
            "at least one class failing either head):",
            "",
            "| Pair status | Pair family | MCTformer+ L12 micro quantity | Estimate |",
            "|---|---|---|---:|",
        ]
    )
    for status in joint_statuses:
        for family, metric, filters in joint_specs:
            row = metric_row(
                joint,
                "class_pair_joint_classification_stratified_results.csv",
                metric,
                model="mctformer_plus",
                layer=12,
                extra_filters={**filters, "classification_subset": status},
            )
            if row is not None:
                lines.append(
                    f"| {status} | {family} | {metric} | {_compact_metric(row)} |"
                )
                rendered_rows += 1
    if rendered_rows == 0:
        return "[Unsupported] Pair classification-control rows are unavailable."
    return "\n".join(lines)


def _matched_checkpoint_evaluation_table(
    classification: pd.DataFrame, raw_cam: pd.DataFrame
) -> str:
    """Render matched checkpoint metrics and paired deltas by image stratum."""

    required = {
        "model_or_delta",
        "label_stratum",
        "aggregation",
        "metric",
        "estimate",
    }
    if not required.issubset(classification.columns) or not required.issubset(
        raw_cam.columns
    ):
        return "[Unsupported] Matched checkpoint mAP/raw-CAM mIoU rows are unavailable."
    lines = [
        "| Evaluation | Logit/CAM source | Label stratum | MCTformer | MCTformer+ | Paired Δ (+ − base) |",
        "|---|---|---|---:|---:|---:|",
    ]
    series_order = (
        "mctformer",
        "mctformer_plus",
        "mctformer_plus_minus_mctformer",
    )
    specifications = (
        (
            classification,
            "classification mAP",
            "class_token",
            "mean_average_precision",
            {"logit_source": "class_token"},
        ),
        (
            classification,
            "classification mAP",
            "patch_head",
            "mean_average_precision",
            {"logit_source": "patch_head"},
        ),
        (
            raw_cam,
            "raw final-CAM mIoU",
            "final_cam @ BG=0.45",
            "mean_intersection_over_union",
            {},
        ),
    )
    for frame, label, source, metric, filters in specifications:
        for stratum in ("all", "single_label", "exactly_2_labels", "3plus_labels"):
            subset = frame[
                (frame["metric"].astype(str) == metric)
                & (frame["aggregation"].astype(str) == "macro_class")
                & (frame["label_stratum"].astype(str) == stratum)
            ]
            for column, value in filters.items():
                if column not in subset:
                    subset = subset.iloc[0:0]
                    break
                subset = subset[subset[column].astype(str) == value]
            cells: list[str] = []
            for series in series_order:
                local = subset[subset["model_or_delta"].astype(str) == series]
                cells.append(
                    "—"
                    if local.empty
                    else _format_metric_row_series(local.sort_index().iloc[0])
                )
            if any(cell != "—" for cell in cells):
                lines.append(
                    f"| {label} | {source} | {stratum} | {cells[0]} | "
                    f"{cells[1]} | {cells[2]} |"
                )
    return "\n".join(lines)


def _class_pair_macro_table(
    shared_marginals: pd.DataFrame, pair_macro: pd.DataFrame
) -> str:
    """Render equal-class macro summaries after two-endpoint pair expansion."""

    lines = [
        "| Pair family | MCTformer+ L12 equal-class macro quantity | Estimate |",
        "|---|---|---:|",
    ]
    specifications = (
        (
            shared_marginals,
            "shared feature support",
            "shared_own_target_fraction",
            {"signal": "feature_post", "layer_or_stage": 12},
        ),
        (
            shared_marginals,
            "shared feature support",
            "shared_partner_target_fraction",
            {"signal": "feature_post", "layer_or_stage": 12},
        ),
        (
            shared_marginals,
            "shared feature support",
            "shared_background_fraction",
            {"signal": "feature_post", "layer_or_stage": 12},
        ),
        (
            pair_macro,
            "feature map diversity",
            "topk_jaccard",
            {
                "source_table": "multiclass_map_diversity",
                "signal": "feature_post",
                "layer": 12,
            },
        ),
        (
            pair_macro,
            "class-token pair",
            "class_token_cosine",
            {"source_table": "class_token_map_overlap", "layer": 12},
        ),
    )
    for frame, family, metric, filters in specifications:
        required = {"model", "metric", "aggregation", "label_stratum"}
        if not required.issubset(frame.columns):
            continue
        subset = frame[
            (frame["model"].astype(str) == "mctformer_plus")
            & (frame["metric"].astype(str) == metric)
            & (frame["aggregation"].astype(str) == "macro_class")
            & (frame["label_stratum"].astype(str) == "all")
        ]
        for column, value in filters.items():
            if column not in subset:
                subset = subset.iloc[0:0]
                break
            if column in {"layer", "layer_or_stage"}:
                numeric = pd.to_numeric(
                    subset[column].map(_layer_number), errors="coerce"
                )
                subset = subset[np.isclose(numeric, float(value))]
            else:
                subset = subset[subset[column].astype(str) == str(value)]
        if "rho" in subset:
            subset = subset[
                np.isclose(pd.to_numeric(subset["rho"], errors="coerce"), 0.5)
            ]
        if "topk_ratio" in subset:
            subset = subset[
                np.isclose(pd.to_numeric(subset["topk_ratio"], errors="coerce"), 0.1)
            ]
        if not subset.empty:
            lines.append(
                f"| {family} | {metric} | "
                f"{_format_metric_row_series(subset.sort_index().iloc[0])} |"
            )
    if len(lines) == 2:
        return "[Unsupported] Endpoint-aware pair macro-class rows are unavailable."
    return "\n".join(lines)


def _checkpoint_classwise_extremes_table(
    classification: pd.DataFrame, raw_cam: pd.DataFrame
) -> str:
    """Show compact class-wise tails for matched output-level diagnostics."""

    lines = [
        "| Evaluation/source | Tail | Class | Class-wise score |",
        "|---|---|---|---:|",
    ]
    specifications = (
        (
            classification,
            "classification AP / class token",
            "average_precision",
            {"logit_source": "class_token"},
        ),
        (
            classification,
            "classification AP / patch head",
            "average_precision",
            {"logit_source": "patch_head"},
        ),
        (raw_cam, "raw final-CAM IoU", "intersection_over_union", {}),
    )
    for frame, label, metric, filters in specifications:
        required = {
            "model_or_delta",
            "label_stratum",
            "aggregation",
            "metric",
            "class_id",
            "estimate",
        }
        if not required.issubset(frame.columns):
            continue
        subset = frame[
            (frame["model_or_delta"].astype(str) == "mctformer_plus")
            & (frame["label_stratum"].astype(str) == "all")
            & (frame["aggregation"].astype(str) == "classwise")
            & (frame["metric"].astype(str) == metric)
        ].copy()
        for column, value in filters.items():
            if column not in subset:
                subset = subset.iloc[0:0]
                break
            subset = subset[subset[column].astype(str) == value]
        subset["estimate"] = pd.to_numeric(subset.get("estimate"), errors="coerce")
        subset = subset[np.isfinite(subset["estimate"])].sort_values(
            ["estimate", "class_id"], kind="stable"
        )
        chosen = [("lowest", row) for _, row in subset.head(2).iterrows()]
        chosen.extend(("highest", row) for _, row in subset.tail(2).iterrows())
        for tail, row in chosen:
            class_id = int(row["class_id"])
            if label.startswith("raw"):
                class_name = (
                    "background" if class_id == 0 else VOC_CLASS_NAMES[class_id - 1]
                )
            else:
                class_name = VOC_CLASS_NAMES[class_id]
            lines.append(
                f"| {label} | {tail} | {class_id}: {class_name} | "
                f"{_format_metric_row_series(row)} |"
            )
    if len(lines) == 2:
        return "[Unsupported] Matched output-level class-wise rows are unavailable."
    return "\n".join(lines)


def _source_lines(
    source_metadata: dict,
    *,
    audit_dir: Path,
    canonical_metadata_path: Optional[Path],
    canonical_metadata: dict,
) -> list[str]:
    sources = source_metadata.get("sources", {})
    lines: list[str] = []
    for model in ("mctformer", "mctformer_plus"):
        value = sources.get(model, {})
        checkpoint = value.get("checkpoint", {})
        if value:
            lines.append(
                f"- [Fact] `{model}` result root: `{value.get('result_root')}`; "
                f"checkpoint: `{checkpoint.get('path')}`; SHA-256: `{checkpoint.get('sha256')}`; "
                f"CLI model name: `{value.get('model_cli_name')}`."
            )
        else:
            lines.append(
                f"- [Unsupported] `{model}` provenance is absent from source metadata."
            )
    paired = source_metadata.get("paired_analysis_root")
    lines.append(
        f"- [Fact] Paired Experiment 1 analysis root: `{paired}`."
        if paired
        else "- [Unsupported] Paired Experiment 1 root is absent."
    )
    source_metadata_path = audit_dir / "source_metadata.json"
    immutable_manifest_path = audit_dir / "file_manifest_before.csv"
    if source_metadata_path.is_file():
        lines.append(
            f"- [Fact] Input-audit metadata: `{source_metadata_path}`; SHA-256: `{_sha256_or_unavailable(source_metadata_path)}`.",
        )
    else:
        lines.append(
            f"- [Unsupported] Input-audit metadata is missing: `{source_metadata_path}`."
        )
    if immutable_manifest_path.is_file():
        lines.append(
            f"- [Fact] Immutable input manifest: `{immutable_manifest_path}`; SHA-256: `{_sha256_or_unavailable(immutable_manifest_path)}`; rows: `{source_metadata.get('before_manifest', {}).get('row_count', 'unavailable')}`.",
        )
    else:
        lines.append(
            f"- [Unsupported] Immutable input manifest is missing: `{immutable_manifest_path}`."
        )
    if canonical_metadata_path is not None:
        lines.append(
            f"- [Fact] Canonical metadata: `{canonical_metadata_path}`; SHA-256: `{_sha256_or_unavailable(canonical_metadata_path)}`; source immutability verified: `{canonical_metadata.get('source_immutability_verified', 'unavailable')}`."
        )
        for model, record in canonical_metadata.get(
            "source_tree_before_after", {}
        ).items():
            lines.append(
                f"- [Fact] `{model}` signal root: `{canonical_metadata.get('source_roots', {}).get(model, 'unavailable')}`; immutable tree SHA-256: `{record.get('tree_sha256_before', 'unavailable')}`."
            )
        for model, root_value in canonical_metadata.get("source_roots", {}).items():
            root = Path(str(root_value))
            metadata_path = root / "metadata.json"
            completion_path = root / "completion.json"
            completion = _json_or_empty(completion_path)
            if not completion:
                lines.append(
                    f"- [Unsupported] `{model}` signal completion metadata is missing: `{completion_path}`."
                )
                continue
            lines.append(
                f"- [Fact] `{model}` signal metadata/completion: `{metadata_path}` (`{_sha256_or_unavailable(metadata_path)}`) / `{completion_path}` (`{_sha256_or_unavailable(completion_path)}`); run kind/images: `{completion.get('run_kind')}`/`{completion.get('num_images')}`; Experiment-1 feature max-|Δ|: `{completion.get('experiment1_feature_post_max_abs_diff')}`; QK→attention max-|Δ|: `{completion.get('qk_attention_max_abs_diff')}`; native-CAM max-|Δ|: `{completion.get('native_cam_max_abs_diff')}`."
            )
    return lines


def _table_inventory(tables: dict[str, pd.DataFrame], tables_dir: Path) -> str:
    lines = ["| Table | Rows | SHA-256 | Path |", "|---|---:|---|---|"]
    for filename in TABLE_FILES:
        path = tables_dir / filename
        lines.append(
            f"| `{filename}` | {len(tables[filename])} | `{_sha256_or_unavailable(path)}` | `{path}` |"
        )
    return "\n".join(lines)


def _verify_execution_provenance(
    analysis_root: Path, analysis_metadata: dict
) -> dict[str, dict[str, str]]:
    records = analysis_metadata.get("provenance_files")
    if not isinstance(records, dict):
        raise RuntimeError("analysis metadata has no hashed provenance-file inventory")
    expected = {
        "analysis_log": ("Analysis log", analysis_root / "analysis.log"),
        "exact_commands": (
            "Full command ledger",
            analysis_root.parent / "exact_commands.sh",
        ),
        "pipeline_metadata": (
            "Pipeline metadata",
            analysis_root.parent / "pipeline_metadata.json",
        ),
    }
    verified: dict[str, dict[str, str]] = {}
    for name, (label, expected_path) in expected.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"analysis provenance inventory lacks {name}")
        recorded_path = Path(str(record.get("path", ""))).expanduser().resolve()
        expected_path = expected_path.resolve()
        if recorded_path != expected_path:
            raise RuntimeError(
                f"analysis provenance path mismatch for {name}: "
                f"{recorded_path} != {expected_path}"
            )
        if not expected_path.is_file():
            raise RuntimeError(
                f"required Experiment 2 provenance artifact is missing: {expected_path}"
            )
        actual_sha256 = _sha256_or_unavailable(expected_path)
        if record.get("sha256") != actual_sha256:
            raise RuntimeError(
                f"analysis provenance SHA-256 mismatch for {name}: "
                f"{actual_sha256} != {record.get('sha256')}"
            )
        verified[name] = {
            "label": label,
            "path": str(expected_path),
            "sha256": actual_sha256,
        }
    return verified


def _execution_provenance_lines(
    analysis_metadata: dict,
    canonical_metadata: dict,
    verified: dict[str, dict[str, str]],
) -> list[str]:
    lines = [
        f"- [Fact] Analysis command: `{analysis_metadata.get('command', 'unavailable')}`.",
        f"- [Fact] Canonical-build command: `{canonical_metadata.get('command', 'unavailable')}`.",
    ]
    for name in ("analysis_log", "exact_commands", "pipeline_metadata"):
        record = verified[name]
        lines.append(
            f"- [Fact] {record['label']}: `{record['path']}`; SHA-256: "
            f"`{record['sha256']}`."
        )
    return lines


def _canonical_provenance(
    source_metadata: dict, analysis_metadata: dict, audit_dir: Path
) -> tuple[Path, dict]:
    value = analysis_metadata.get("canonical_dir")
    if not isinstance(value, str) or not value:
        raise RuntimeError("analysis metadata does not identify its canonical input")
    path = Path(value).expanduser().resolve() / "canonical_metadata.json"
    metadata = _json_or_empty(path)
    if not metadata:
        raise RuntimeError(f"canonical metadata is missing or invalid: {path}")
    if metadata.get("status") != "complete":
        raise RuntimeError(f"canonical run is not complete: {path}")
    expected = source_metadata.get("dataset", {}).get("num_images")
    observed = metadata.get("num_manifest_images_per_model")
    if int(expected or -1) != 1449:
        raise RuntimeError(
            f"full VOC audit must contain 1,449 images, got {expected!r}"
        )
    if int(observed or -1) != int(expected):
        raise RuntimeError(
            "refusing to turn smoke/incomplete tables into scientific conclusions: "
            f"canonical has {observed} images/model, audit expects {expected}"
        )
    canonical_verification = analysis_metadata.get("canonical_verification")
    if not isinstance(canonical_verification, dict):
        raise RuntimeError("analysis metadata has no canonical verification snapshot")
    recorded_metadata_path = (
        Path(str(canonical_verification.get("metadata_path", "")))
        .expanduser()
        .resolve()
    )
    actual_metadata_sha256 = _sha256_or_unavailable(path)
    if (
        recorded_metadata_path != path
        or canonical_verification.get("metadata_sha256") != actual_metadata_sha256
    ):
        raise RuntimeError(
            "canonical metadata changed or was relinked after analysis: "
            f"path={recorded_metadata_path} expected={path}; "
            f"sha256={actual_metadata_sha256} recorded="
            f"{canonical_verification.get('metadata_sha256')}"
        )
    if metadata.get("source_immutability_verified") is not True:
        raise RuntimeError("canonical source-tree immutability was not verified")
    if metadata.get("source_manifests_exact_match") is not True:
        raise RuntimeError("canonical signal manifests were not an exact paired match")
    source_roots = metadata.get("source_roots")
    if not isinstance(source_roots, dict) or set(source_roots) != {
        "mctformer",
        "mctformer_plus",
    }:
        raise RuntimeError("canonical metadata lacks the two exact signal roots")
    audit_metadata_path = (audit_dir / "source_metadata.json").resolve()
    audit_metadata_sha256 = _sha256_or_unavailable(audit_metadata_path)
    expected_dataset = source_metadata.get("dataset", {})
    signal_commits: set[str] = set()
    expected_transform = (
        "bicubic short-side Resize(512) -> CenterCrop(448) -> ToTensor -> "
        "ImageNet Normalize; matched nearest-neighbor semantic-mask geometry"
    )
    for model, root_value in source_roots.items():
        root = Path(str(root_value)).expanduser().resolve()
        completion_path = root / "completion.json"
        signal_metadata_path = root / "metadata.json"
        completion = _json_or_empty(completion_path)
        signal_metadata = _json_or_empty(signal_metadata_path)
        if not completion:
            raise RuntimeError(
                f"signal completion metadata is missing: {completion_path}"
            )
        if not signal_metadata:
            raise RuntimeError(
                f"signal run metadata is missing: {signal_metadata_path}"
            )
        if (
            completion.get("status") != "complete"
            or completion.get("run_kind") != "full"
            or int(completion.get("num_images", -1)) != 1449
            or signal_metadata.get("status") != "complete"
            or signal_metadata.get("run_kind") != "full"
            or int(signal_metadata.get("processed_images", -1)) != 1449
            or signal_metadata.get("model") != model
        ):
            raise RuntimeError(
                "refusing incomplete/non-full signal outputs: "
                f"{model} status/kind/images={completion.get('status')}/"
                f"{completion.get('run_kind')}/{completion.get('num_images')}"
            )
        recorded_audit = (
            Path(str(signal_metadata.get("source_metadata", ""))).expanduser().resolve()
        )
        if (
            recorded_audit != audit_metadata_path
            or signal_metadata.get("source_metadata_sha256") != audit_metadata_sha256
        ):
            raise RuntimeError(f"{model} signal run is not linked to this input audit")
        source = source_metadata.get("sources", {}).get(model, {})
        expected_checkpoint = source.get("checkpoint", {})
        recorded_checkpoint = signal_metadata.get("checkpoint", {})
        if (
            Path(str(signal_metadata.get("experiment1_result_root", "")))
            .expanduser()
            .resolve()
            != Path(str(source.get("result_root", ""))).expanduser().resolve()
            or Path(str(recorded_checkpoint.get("path", ""))).expanduser().resolve()
            != Path(str(expected_checkpoint.get("path", ""))).expanduser().resolve()
            or recorded_checkpoint.get("sha256") != expected_checkpoint.get("sha256")
        ):
            raise RuntimeError(
                f"{model} signal provenance disagrees with audited Experiment 1 inputs"
            )
        signal_dataset = signal_metadata.get("dataset", {})
        for key in ("voc_root", "list_path"):
            if (
                Path(str(signal_dataset.get(key, ""))).expanduser().resolve()
                != Path(str(expected_dataset.get(key, ""))).expanduser().resolve()
            ):
                raise RuntimeError(f"{model} signal dataset {key} disagrees with audit")
        if (
            int(signal_dataset.get("input_size", -1)) != 448
            or int(signal_dataset.get("patch_size", -1)) != 16
            or int(signal_dataset.get("expected_images", -1)) != 1449
            or signal_dataset.get("transform") != expected_transform
        ):
            raise RuntimeError(f"{model} signal geometry/transform contract is invalid")

        tolerances = {
            "experiment1_feature_post_max_abs_diff": (1e-6, False),
            "qk_attention_max_abs_diff": (1e-6, False),
            "native_cam_max_abs_diff": (1e-6, False),
            "attention_row_sum_max_abs_error": (5e-6, True),
            "conditional_attention_row_sum_max_abs_error": (5e-6, True),
        }
        for key, (limit, inclusive) in tolerances.items():
            try:
                value = float(signal_metadata[key])
                completion_value = float(completion[key])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{model} signal metadata lacks finite {key}"
                ) from error
            valid = math.isfinite(value) and (
                value <= limit if inclusive else value < limit
            )
            if not valid or value != completion_value:
                raise RuntimeError(
                    f"{model} failed or inconsistently recorded numerical check {key}: "
                    f"metadata={value}, completion={completion_value}, limit={limit}"
                )

        git = signal_metadata.get("git", {})
        tracked = git.get("runtime_source_tracked", {})
        source_hashes = git.get("runtime_source_sha256", {})
        commit = str(git.get("commit", ""))
        required_runtime = {
            "analysis/lazy_assignment/experiment2/run_experiment2_signals.py",
            "analysis/lazy_assignment/experiment2/evaluation_metrics.py",
        }
        if (
            not commit
            or git.get("tracked_dirty") is not False
            or not isinstance(tracked, dict)
            or not tracked
            or not all(value is True for value in tracked.values())
            or not required_runtime.issubset(tracked)
            or not isinstance(source_hashes, dict)
            or set(source_hashes) != set(tracked)
        ):
            raise RuntimeError(f"{model} signal run lacks clean tracked Git provenance")
        for relative, digest in source_hashes.items():
            runtime_path = _REPOSITORY_ROOT / relative
            if _sha256_or_unavailable(runtime_path) != digest:
                raise RuntimeError(
                    f"{model} runtime source no longer matches recorded hash: {relative}"
                )
        signal_commits.add(commit)

        tree = metadata.get("source_tree_before_after", {}).get(model, {})
        if (
            int(tree.get("num_files", -1)) < 4
            or not tree.get("tree_sha256_before")
            or tree.get("tree_sha256_before") != tree.get("tree_sha256_after")
        ):
            raise RuntimeError(f"{model} canonical source-tree verification is invalid")
    if len(signal_commits) != 1:
        raise RuntimeError(
            "paired signal roots were generated from different Git commits"
        )
    return path, metadata


def _verify_audit_and_immutability(
    audit_dir: Path, immutability_dir: Path
) -> tuple[dict, dict]:
    source_path = audit_dir / "source_metadata.json"
    source = _json_or_empty(source_path)
    if source.get("integrity_passed") is not True:
        raise RuntimeError(f"input audit did not pass: {source_path}")
    if int(source.get("dataset", {}).get("num_images", -1)) != 1449:
        raise RuntimeError("input audit is not the full 1,449-image VOC val set")
    before = audit_dir / "file_manifest_before.csv"
    if not before.is_file():
        raise RuntimeError(f"immutable input baseline is missing: {before}")
    expected_rows = int(source.get("before_manifest", {}).get("row_count", -1))
    if expected_rows < 1 or len(pd.read_csv(before)) != expected_rows:
        raise RuntimeError("immutable input baseline row count disagrees with audit")

    verification_path = immutability_dir / "immutability_verification.json"
    verification = _json_or_empty(verification_path)
    if (
        verification.get("status") != "complete"
        or verification.get("integrity_passed") is not True
        or int(verification.get("files_checked", -1)) != expected_rows
        or int(verification.get("missing_files", -1)) != 0
        or int(verification.get("size_changed_files", -1)) != 0
        or int(verification.get("sha256_changed_files", -1)) != 0
    ):
        raise RuntimeError(
            f"post-pipeline immutable-input verification did not pass: {verification_path}"
        )
    recorded_before = Path(str(verification.get("before_manifest", ""))).resolve()
    if recorded_before != before.resolve():
        raise RuntimeError(
            "immutability verification used a different baseline manifest"
        )
    after = immutability_dir / "file_manifest_after.csv"
    if not after.is_file() or len(pd.read_csv(after)) != expected_rows:
        raise RuntimeError("post-pipeline immutable manifest is missing or incomplete")
    return source, verification


def _verify_full_analysis_outputs(
    analysis_metadata: dict,
    tables_dir: Path,
) -> None:
    if analysis_metadata.get("status") != "complete":
        raise RuntimeError("analysis metadata is not complete")
    bootstrap = analysis_metadata.get("bootstrap", {})
    if int(bootstrap.get("repeats", -1)) != 5000:
        raise RuntimeError(
            "scientific report requires exactly 5,000 image-clustered bootstrap "
            f"repeats, got {bootstrap.get('repeats')!r}"
        )
    files = analysis_metadata.get("output_files")
    if not isinstance(files, dict):
        raise RuntimeError("analysis metadata has no hashed output-file inventory")
    for filename in TABLE_FILES:
        record = files.get(filename)
        path = tables_dir / filename
        if not isinstance(record, dict) or not path.is_file():
            raise RuntimeError(
                f"analysis output inventory is incomplete for {filename}"
            )
        actual = _sha256_or_unavailable(path)
        if record.get("sha256") != actual:
            raise RuntimeError(
                f"analysis table SHA-256 mismatch for {filename}: {actual} != "
                f"{record.get('sha256')}"
            )


def _ci_above(row: Optional[MetricRow], reference: float) -> bool:
    return row is not None and math.isfinite(row.ci_low) and row.ci_low > reference


def _ci_below(row: Optional[MetricRow], reference: float) -> bool:
    return row is not None and math.isfinite(row.ci_high) and row.ci_high < reference


def decide_case(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    layer = tables["layerwise_region_metrics.csv"]
    cam = tables["cam_stage_region_metrics.csv"]
    shared = tables["shared_support_ownership.csv"]
    feature_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    attention_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    qk_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal=("qk_mean", "qk"),
        layer=12,
    )
    final_auc = metric_row(
        cam,
        "cam_stage_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        stage="final_cam",
    )
    evidence: dict[str, Optional[MetricRow]] = {
        "feature_l12_auc_target_bg": feature_auc,
        "qk_l12_auc_target_bg": qk_auc,
        "attention_l12_auc_target_bg": attention_auc,
        "final_cam_auc_target_bg": final_auc,
    }
    evaluations: dict[str, dict[str, object]] = {
        "G": {
            "satisfied": _ci_below(feature_auc, 0.5)
            and all(_ci_above(row, 0.5) for row in (qk_auc, attention_auc, final_auc)),
            "rule": (
                "feature_post L12 AUROC CI < 0.5 AND QK, attention, and "
                "final-CAM AUROC CIs > 0.5"
            ),
        }
    }

    ownership = {
        name: metric_row(
            shared,
            "shared_support_ownership.csv",
            metric,
            model="mctformer_plus",
            signal="feature_post",
            layer=12,
        )
        for name, metric in (
            ("target_a", "shared_target_a_fraction"),
            ("target_b", "shared_target_b_fraction"),
            ("other_fg", "shared_other_fg_fraction"),
            ("background", "shared_background_fraction"),
        )
    }
    dominant_target = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_dominant_target_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    pair_target = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_pair_target_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    evidence.update({f"shared_{key}": value for key, value in ownership.items()})
    evidence.update(
        {
            "shared_pair_target": pair_target,
            "shared_dominant_target": dominant_target,
        }
    )
    complete_ownership = all(value is not None for value in ownership.values())
    evaluations.update(
        {
            "F": {
                "satisfied": complete_ownership
                and _ci_above(ownership["background"], 0.5),
                "rule": (
                    "MCTformer+ feature_post L12 shared-background fraction 95% "
                    "CI > 0.5 at rho=0.5/top10%, pair-micro/all-images"
                ),
            },
            "E": {
                "satisfied": complete_ownership and _ci_above(dominant_target, 0.5),
                "rule": (
                    "MCTformer+ feature_post L12 order-invariant per-pair "
                    "dominant-target fraction 95% CI > 0.5"
                ),
            },
        }
    )

    feature_bg = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "bg_tail_enrich_10",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    attention_bg = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "bg_tail_enrich_10",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    final_bg = metric_row(
        cam,
        "cam_stage_region_metrics.csv",
        "bg_tail_enrich_10",
        model="mctformer_plus",
        stage="final_cam",
    )
    evidence.update(
        {
            "feature_l12_bg_enrichment": feature_bg,
            "attention_l12_bg_enrichment": attention_bg,
            "final_cam_bg_enrichment": final_bg,
        }
    )
    complete_pipeline = all(
        row is not None for row in (feature_bg, attention_bg, final_bg)
    )
    above = (False, False, False)
    below = (False, False, False)
    if complete_pipeline:
        assert (
            feature_bg is not None and attention_bg is not None and final_bg is not None
        )
        above = tuple(
            _ci_above(row, 1.0) for row in (feature_bg, attention_bg, final_bg)
        )
        below = tuple(
            _ci_below(row, 1.0) for row in (feature_bg, attention_bg, final_bg)
        )
    evaluations.update(
        {
            "A": {
                "satisfied": complete_pipeline and above == (True, True, True),
                "rule": "feature, conditional-attention, and final-CAM BG-enrichment CIs all > 1",
            },
            "B": {
                "satisfied": complete_pipeline and above[0] and below[1] and below[2],
                "rule": "feature BG-enrichment CI > 1 AND attention/final-CAM CIs < 1",
            },
            "C": {
                "satisfied": complete_pipeline and below[0] and above[1],
                "rule": "feature BG-enrichment CI < 1 AND attention CI > 1",
            },
            "D": {
                "satisfied": complete_pipeline and below[0] and below[1] and above[2],
                "rule": "feature/attention BG-enrichment CIs < 1 AND final-CAM CI > 1",
            },
        }
    )
    satisfied = [case for case in DECISION_PRECEDENCE if evaluations[case]["satisfied"]]
    primary = satisfied[0] if satisfied else "UNRESOLVED"
    primary_rule = (
        str(evaluations[primary]["rule"])
        if primary != "UNRESOLVED"
        else (
            "none of the pre-full-run operational Case A–G tests had every "
            "required 95% confidence interval on the required side of its reference"
        )
    )
    return {
        "case": primary,
        "primary_case": primary,
        "satisfied_cases": satisfied,
        "description": CASE_DESCRIPTIONS[primary],
        "rule": primary_rule,
        "case_evaluations": evaluations,
        "precedence": list(DECISION_PRECEDENCE),
        "rule_version": DECISION_RULE_VERSION,
        "evidence": evidence,
    }


def _decision_markdown(decision: dict[str, object], bootstrap: dict) -> str:
    case = str(decision["case"])
    evidence = decision["evidence"]
    lines = [
        "# Next Experiment Decision",
        "",
        f"Generated: `{timestamp()}`",
        "",
        f"## Selected Case: {case}",
        "",
        f"[Fact] The pre-full-run operational rule selected **Case {case}** as primary: {decision['description']}.",
        "",
        f"[Fact] Rule: `{decision['rule']}`.",
        "",
        f"[Fact] All independently satisfied cases: `{decision['satisfied_cases']}`; precedence: `{decision['precedence']}`; rule version: `{decision['rule_version']}`. Coexisting cases are disclosed rather than hidden by the primary-case choice.",
        "",
        "## Metric evidence",
        "",
        "| Quantity | Estimate and image-clustered 95% CI | Source |",
        "|---|---:|---|",
    ]
    for name, row in evidence.items():
        if row is not None:
            lines.append(f"| {name} | {_format_metric(row)} | `{row.table}` |")
    lines.extend(
        [
            "",
            "## Statistical scope",
            "",
            f"[Fact] Bootstrap unit: `{bootstrap.get('unit', 'image_id cluster')}`; repeats: `{bootstrap.get('repeats', 'unavailable')}`; seed: `{bootstrap.get('seed', 'unavailable')}`. Multiple patches and classes from one image were not treated as independent.",
            "",
            "[Interpretation candidate] This case identifies the next causal question; it does not establish a mechanism by itself.",
            "",
            "[Unsupported] No intervention, training change, or proposed solution is selected or implemented by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_reports(
    analysis_root: Path,
    output_dir: Path,
    *,
    tables_dir: Optional[Path] = None,
    audit_dir: Optional[Path] = None,
    immutability_dir: Optional[Path] = None,
    examples_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    render_dir: Optional[Path] = None,
    command: Optional[str] = None,
) -> dict[str, object]:
    analysis_root = analysis_root.resolve()
    tables_dir = (tables_dir or analysis_root / "tables").resolve()
    audit_dir = (audit_dir or analysis_root / "audit").resolve()
    immutability_dir = (
        immutability_dir or analysis_root.parent / "integrity" / "final"
    ).resolve()
    examples_dir = (examples_dir or analysis_root / "examples").resolve()
    plots_dir = (plots_dir or analysis_root / "plots").resolve()
    render_dir = (render_dir or analysis_root / "rendered_examples").resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output_dir}")
    tables = _load_tables(tables_dir)
    source_metadata, immutability_metadata = _verify_audit_and_immutability(
        audit_dir, immutability_dir
    )
    analysis_metadata = _json_or_empty(analysis_root / "analysis_metadata.json")
    if not analysis_metadata:
        analysis_metadata = _json_or_empty(tables_dir.parent / "analysis_metadata.json")
    canonical_metadata_path, canonical_metadata = _canonical_provenance(
        source_metadata, analysis_metadata, audit_dir
    )
    _verify_full_analysis_outputs(analysis_metadata, tables_dir)
    execution_provenance = _verify_execution_provenance(
        analysis_root, analysis_metadata
    )
    selection_metadata = _json_or_empty(examples_dir / "selection_metadata.json")
    visual_validation = verify_visual_deliverables(
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        examples_dir=examples_dir,
        render_dir=render_dir,
        canonical_metadata_path=canonical_metadata_path,
        source_metadata_path=audit_dir / "source_metadata.json",
        required_plot_files=PLOT_FILES,
        required_plot_input_files=PLOT_INPUT_TABLE_FILES,
        required_categories=NEW_CATEGORIES,
    )
    decision = decide_case(tables)
    bootstrap = analysis_metadata.get("bootstrap", {})

    layer = tables["layerwise_region_metrics.csv"]
    cam = tables["cam_stage_region_metrics.csv"]
    visible = tables["target_visible_region_metrics.csv"]
    target_visible_classwise = tables["target_visible_classwise_results.csv"]
    shared = tables["shared_support_ownership.csv"]
    shared_marginals = tables["shared_support_class_marginals.csv"]
    new_shared = tables["new_shared_support_l9_l12.csv"]
    transition = tables["stage_transition_metrics.csv"]
    transition_classwise = tables["stage_transition_classwise_results.csv"]
    last3 = tables["last_three_aggregation_analysis.csv"]
    paired = tables["paired_model_deltas.csv"]
    token = tables["class_token_similarity_vs_map_overlap.csv"]
    qk_heads = tables["qk_head_region_summary.csv"]
    patch_norm_joint = tables["patch_norm_joint_control.csv"]
    qk_head_classwise = tables["qk_head_classwise_results.csv"]
    failure_summary = tables["failure_pattern_summary.csv"]
    classwise = tables["classwise_results.csv"]
    association = tables["class_token_map_overlap_association.csv"]
    endpoint_association = tables["class_token_map_overlap_endpoint_association.csv"]
    diversity = tables["multiclass_map_diversity.csv"]
    classification = tables["classification_stratified_results.csv"]
    classification_classwise = tables[
        "classification_conditioned_classwise_results.csv"
    ]
    pair_focal_classification = tables[
        "class_pair_focal_classification_stratified_results.csv"
    ]
    pair_joint_classification = tables[
        "class_pair_joint_classification_stratified_results.csv"
    ]
    pair_macro = tables["class_pair_macro_class_results.csv"]
    pair_classwise = tables["class_pair_classwise_results.csv"]
    checkpoint_classification = tables["checkpoint_classification_performance.csv"]
    raw_cam_miou = tables["raw_final_cam_miou.csv"]

    feature_target = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "target_hit",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    feature_bg = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "bg_tail_enrich_10",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    feature_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    attention_target = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "target_hit",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    attention_bg = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "bg_tail_enrich_10",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    attention_mass = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "conditional_bg_mass",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    shared_bg = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_background_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    shared_a = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_target_a_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    shared_b = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_target_b_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    shared_pair_target = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_pair_target_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    shared_dominant_target = metric_row(
        shared,
        "shared_support_ownership.csv",
        "shared_dominant_target_fraction",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    token_cosine = metric_row(
        token,
        "class_token_similarity_vs_map_overlap.csv",
        "class_token_cosine",
        model="mctformer_plus",
        layer=12,
    )
    patch_bg = metric_row(
        cam,
        "cam_stage_region_metrics.csv",
        "conditional_bg_mass",
        model="mctformer_plus",
        stage="patch_cam",
    )
    c2p_bg = metric_row(
        cam,
        "cam_stage_region_metrics.csv",
        "conditional_bg_mass",
        model="mctformer_plus",
        stage="c2p_cam",
    )
    final_bg = metric_row(
        cam,
        "cam_stage_region_metrics.csv",
        "conditional_bg_mass",
        model="mctformer_plus",
        stage="final_cam",
    )
    norm_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal="feature_norm",
        layer=12,
    )
    qk_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal=("qk_mean", "qk"),
        layer=12,
    )
    attn_auc = metric_row(
        layer,
        "layerwise_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    visible_feature_auc = metric_row(
        visible,
        "target_visible_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal="feature_post",
        layer=12,
    )
    visible_attn_auc = metric_row(
        visible,
        "target_visible_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        signal=("attn_c2p_conditional", "attn"),
        layer=12,
    )
    visible_final_auc = metric_row(
        visible,
        "target_visible_region_metrics.csv",
        "auc_target_bg",
        model="mctformer_plus",
        stage="final_cam",
    )

    dataset = source_metadata.get("dataset", {})
    gt = source_metadata.get("gt_summary", {})
    report_lines = [
        "# Experiment 2 Semantic Ownership Report",
        "",
        f"Generated: `{timestamp()}`",
        "",
        "## 1. Data and Integrity",
        "",
        *(
            _source_lines(
                source_metadata,
                audit_dir=audit_dir,
                canonical_metadata_path=canonical_metadata_path,
                canonical_metadata=canonical_metadata,
            )
        ),
        *_execution_provenance_lines(
            analysis_metadata, canonical_metadata, execution_provenance
        ),
        f"- [Fact] Post-pipeline immutable-input verification: `{immutability_dir / 'immutability_verification.json'}`; files checked: `{immutability_metadata.get('files_checked')}`; missing/size-changed/hash-changed: `{immutability_metadata.get('missing_files')}/{immutability_metadata.get('size_changed_files')}/{immutability_metadata.get('sha256_changed_files')}`.",
        f"- [Fact] Visual-delivery verification: `{visual_validation['plots']['plot_count']}` hashed pre-specified plots, `{visual_validation['render']['rendered_panels']}` rendered Experiment 2 panels, and `{visual_validation['render']['fixed_links']}` immutable Experiment 1 links; render source NPZ files were unchanged.",
        f"- [Structural N/A] New example categories with no rule-eligible full-set candidate: `{visual_validation['selection']['missing_new_categories']}`. Empty categories are disclosed rather than filled by manual selection.",
        "",
        f"- [Fact] Dataset root: `{dataset.get('voc_root', 'unavailable')}`; list: `{dataset.get('list_path', 'unavailable')}`; labels: `{dataset.get('labels_path', 'unavailable')}`; images: `{dataset.get('num_images', 'unavailable')}`; input/patch size: `{dataset.get('input_size', 'unavailable')}/{dataset.get('patch_size', 'unavailable')}`.",
        f"- [Fact] Input audit integrity: `{source_metadata.get('integrity_passed', 'unavailable')}`. Raw mask/ImageLabel mismatches: `{gt.get('raw_mask_image_label_mismatch_count', 'unavailable')}`.",
        f"- [Fact] Exactly `{gt.get('positive_pairs_with_no_target_pixels_after_crop', 'unavailable')}` positive pairs have no target pixels after the matched crop; `{gt.get('positive_pairs_without_target_dominant_patch_rho05', 'unavailable')}` / `{gt.get('positive_pairs_without_target_dominant_patch_rho07', 'unavailable')}` have no target-dominant patch at rho=0.5 / 0.7. These are analysis strata, not corrupt samples.",
        "- [Fact] Primary region estimates retain all positive image-class pairs. `target_visible_region_metrics.csv`, when present, is the boundary control that excludes only pairs lacking a target-dominant patch at the corresponding rho; it does not replace the primary analysis.",
        "",
        _table_inventory(tables, tables_dir),
        "",
        "## 2. Signals and Exact Native Pipelines",
        "",
        "[Fact] `feature_post_l` is post-block/pre-final-LayerNorm cosine. `feature_norm_l`, QK energy, and `A_c2p_l` are computed from block-l pre-attention inputs. Therefore `feature_post_l ↔ attention_l` has a within-block timing offset; `feature_norm_l ↔ QK_l ↔ attention_l` is the same-stage control.",
        "",
        "[Fact] Native MCTformer uses the head-mean sum of L10–L12 class-to-patch attention, multiplies it with ReLU patch CAM, sums all-layer patch affinity, and propagates the refined CAM. Native MCTformer+ uses the head-mean average of L10–L12, `sqrt(A_c2p * patch_CAM)`, then the all-layer patch-affinity sum.",
        "",
        "[Fact] `feature_final_norm` is an analysis-only probe control. It is not an extra native CAM stage and is not evidence of a modified model.",
        "",
        "## 3. GT Patch Region Definition",
        "",
        "[Fact] Mask IDs use 0=background, 1…20=foreground, 255=void; image class c maps to mask ID c+1. RGB and mask share scalar short-side resize and center crop; mask interpolation is nearest-neighbor. Patches with <50% valid pixels are void, otherwise target/other-FG/background requires dominant fraction rho=0.5 (rho=0.7 sensitivity), with remaining patches mixed.",
        "",
        "[Fact] Target-empty contrasts are retained as NA rather than imputed: target-vs-BG/other AUROC and AUPRC are undefined when either comparison set is absent.",
        "",
        "## 4. Feature-level Semantic Ownership",
        "",
        _fact("MCTformer+ L12 feature C-PiM", feature_target),
        _fact("MCTformer+ L12 feature BG-tail enrichment@10%", feature_bg),
        _inference(
            "L12 feature BG enrichment relative to area-matched reference",
            feature_bg,
            1.0,
        ),
        _fact("MCTformer+ L12 signed target-vs-BG AUROC", feature_auc),
        "",
        "[Fact] The following focus table reports the pre-registered L4/L5 and L9–L12 slices; the underlying layer table and diagnostic plots retain all 12 layers. Brackets are image-clustered 95% CIs, and each metric cell gives its own finite denominator.",
        "",
        _layer_focus_table(layer),
        "",
        "[Interpretation candidate] Feature-level ownership describes representation alignment only; it does not establish that the class token reads or causally uses those patches.",
        "",
        "## 5. Attention-level Semantic Ownership",
        "",
        _fact("MCTformer+ L12 attention C-PiM", attention_target),
        _fact("MCTformer+ L12 attention BG-tail enrichment@10%", attention_bg),
        _fact("MCTformer+ L12 conditional attention BG mass", attention_mass),
        "",
        "[Fact] Conditional attention mass renormalizes over patch keys. It is distinct from the raw class-query patch-group mass and does not change top-k ranking.",
        "",
        "## 6. Shared Top-Tail Ownership",
        "",
        _fact("MCTformer+ L12 shared-support background fraction", shared_bg),
        _fact("MCTformer+ L12 shared-support target-A fraction", shared_a),
        _fact("MCTformer+ L12 shared-support target-B fraction", shared_b),
        _fact(
            "MCTformer+ L12 shared-support pair-target fraction",
            shared_pair_target,
        ),
        _fact(
            "MCTformer+ L12 order-invariant dominant-target fraction",
            shared_dominant_target,
        ),
        f"[Fact] Transition-focused table `new_shared_support_l9_l12.csv` contains **{len(new_shared)} rows**, covering measured new shared patches for L9→L10, L10→L11, and L11→L12 slices present in canonical data.",
        "",
        _shared_layer_table(shared),
        "",
        "[Fact] Highest-priority ownership of patches newly entering the shared set:",
        "",
        _new_shared_table(new_shared),
        "",
        _fact("MCTformer+ L12 positive class-token cosine", token_cosine),
        "",
        "[Fact] Multi-class map-diversity trajectory (unordered positive class pairs; micro pair mean with image-clustered CI):",
        "",
        _multiclass_diversity_table(diversity),
        "",
        "[Interpretation candidate] Jointly reading token cosine and map overlap can distinguish token collapse from shared-patch attraction, but association alone is not causal.",
        "",
        "## 7. Official Last-Three Attention Analysis",
        "",
        f"[Fact] `last_three_aggregation_analysis.csv` contains **{len(last3)} aggregate rows** for available L10/L11/L12/native-last3/mid3 signal and CAM controls.",
        "",
        _last_three_table(last3),
        "",
        "[Unsupported] A diagnosis-only mid3 or single-layer CAM is not an alternative method and is not a tuned performance claim.",
        "",
        "## 8. Patch CAM → C2P CAM → Final CAM",
        "",
        _fact("MCTformer+ patch-CAM conditional BG mass", patch_bg),
        _fact("MCTformer+ class-attention-CAM conditional BG mass", c2p_bg),
        _fact("MCTformer+ final propagated-CAM conditional BG mass", final_bg),
        "",
        _cam_stage_table(cam),
        "",
        "[Fact] These are exact native patch, class-attention-refined, and A_p2p-propagated stages only if the recorded native-equivalence tests pass; this report does not substitute visual similarity for numerical equivalence.",
        "",
        "## 9. Feature–Attention–CAM Linkage",
        "",
        f"[Fact] `stage_transition_metrics.csv` contains **{len(transition)} image-clustered aggregate rows** for survival, introduction, removal, rank correlation, and top-tail overlap.",
        "",
        _linkage_table(transition),
        "",
        f"[Fact] `stage_transition_classwise_results.csv` contains **{len(transition_classwise)} rows** covering per-model and exact-common-key paired MCTformer+ minus MCTformer survival/introduction/removal estimates by focal class. All four label-count strata are retained when samples exist, and every confidence interval resamples whole images.",
        "",
        "[Fact] The compact table below shows descriptive all-image class-wise paired-delta extrema; it is not a multiple-testing-adjusted ranking. Complete per-model, paired, target, other-foreground, and background rows remain machine-readable:",
        "",
        _transition_classwise_delta_table(transition_classwise),
        "",
        "[Fact] Full-set image-class patterns below use the pre-registered `BG-TailEnrich@10% > 1` threshold. Flags are non-exclusive, so their prevalences need not sum to one.",
        "",
        _failure_pattern_table(failure_summary),
        "",
        "[Fact] Same-layer raw post-block cosine and attention are temporally offset. Same-stage conclusions must prioritize pre-attention normalized feature → QK → softmax attention; post-block feature → attention is descriptive only.",
        "",
        "[Interpretation candidate] Background survival/introduction/removal localizes where a measured ownership pattern changes in the pipeline; it does not identify why it changes.",
        "",
        "## 10. Probe Validity: Raw Cosine vs Norm/QK/Attention",
        "",
        _fact("MCTformer+ L12 raw post-block target-vs-BG AUROC", feature_auc),
        _fact(
            "MCTformer+ L12 pre-attention normalized-feature target-vs-BG AUROC",
            norm_auc,
        ),
        _fact("MCTformer+ L12 QK-energy target-vs-BG AUROC", qk_auc),
        _fact("MCTformer+ L12 attention target-vs-BG AUROC", attn_auc),
        "",
        "[Fact] Head-wise QK region means preserve heterogeneity hidden by the six-head mean:",
        "",
        _qk_head_table(qk_heads),
        "",
        "[Fact] The joint post-block cosine/patch-L2-norm control uses within-image thresholds fixed before the full run. It reports score–norm correlation and whether high-score background patches concentrate below the valid-patch norm median or above its q75, together with exact-common-image/class paired model deltas:",
        "",
        _patch_norm_joint_table(patch_norm_joint),
        "",
        "[Unsupported] Patch-norm concentration cannot by itself identify a register-style artifact or a semantic shortcut; these rows only discriminate high-norm versus low-norm association patterns.",
        "",
        "[Fact] Token/map association reports both the positive-class-pair micro estimand and a per-image-pair-mean sensitivity estimand. Both resample whole image clusters; class-pair rows are never bootstrap units.",
        "",
        _association_table(association),
        "",
        "[Fact] Endpoint-expanded token/map associations additionally report "
        "within-focal-class Pearson estimates and an equal-class macro estimand. "
        "Every macro bootstrap draw recomputes the class correlations before "
        "averaging finite classes equally:",
        "",
        _endpoint_association_table(endpoint_association),
        "",
        "[Unsupported] Raw cosine sign or magnitude is not a calibrated class probability and must not be interpreted across classes without these geometry controls.",
        "",
        "## 11. MCTformer vs MCTformer+",
        "",
        f"[Fact] `paired_model_deltas.csv` contains **{len(paired)} rows**; every delta is registered as MCTformer+ minus MCTformer on common keys.",
        "",
        _paired_focus_table(paired),
        "",
        f"[Fact] Bootstrap metadata: unit=`{bootstrap.get('unit', 'unavailable')}`, repeats=`{bootstrap.get('repeats', 'unavailable')}`, seed=`{bootstrap.get('seed', 'unavailable')}`, CI=`{bootstrap.get('ci', 'unavailable')}`. Patches and multiple classes from one image are not independent units.",
        "",
        "[Fact] Statistical decision rule: a paired difference is treated as resolved only when its image-clustered 95% CI excludes zero; table point estimates alone are not sufficient.",
        "",
        "## 12. Class-wise and Multi-label Analysis",
        "",
        f"[Fact] Primary layer/CAM class-wise rows: **{len(classwise)}**; endpoint-aware shared-support rows: **{len(shared_marginals)}**; pair macro rows: **{len(pair_macro)}**; pair class-wise/per-model-and-paired rows: **{len(pair_classwise)}**; classification-stratified ownership rows: **{len(classification)}**; pair focal-status rows: **{len(pair_focal_classification)}**; pair joint-status rows: **{len(pair_joint_classification)}**. Aggregates retain all/single-label/exactly-2/3+ strata when those samples exist.",
        "",
        "[Fact] `classwise_results.csv` covers every canonical layer signal and CAM stage with the full region-metric family, including top-5/top-20 composition and enrichment. It contains both per-model within-class estimates and exact-common-`image_id,class_id` paired MCTformer+ minus MCTformer deltas; confidence intervals resample whole images.",
        "",
        f"[Fact] Automatic example selection retained `{selection_metadata.get('experiment1_fixed_rows_retained', 'unavailable')}` fixed Experiment 1 rows and generated `{selection_metadata.get('total_rows', 'unavailable')}` total manifest rows without manual cherry-picking.",
        "",
        _label_strata_table(layer),
        "",
        "[Fact] Classification-correctness control (class-token and patch-head logits use the fixed native zero threshold):",
        "",
        _classification_table(classification),
        "",
        "[Fact] Section-17 class-wise aggregation coverage is materialized in "
        "`classification_conditioned_classwise_results.csv`, "
        "`target_visible_classwise_results.csv`, and "
        "`qk_head_classwise_results.csv`. Per-model rows retain available "
        "label-count strata and image-cluster confidence intervals; target-visible "
        "and QK paired deltas use only exact common image_id/class_id keys:",
        "",
        _aggregation_coverage_table(
            classification_classwise,
            target_visible_classwise,
            qk_head_classwise,
        ),
        "",
        "[Fact] Classification status is model-specific, so status-conditioned "
        "ownership summaries are reported separately per model and are not "
        "misrepresented as common-key paired model deltas:",
        "",
        _class_pair_classification_control_tables(
            pair_focal_classification, pair_joint_classification
        ),
        "",
        "[Fact] Matched frozen-checkpoint output diagnostics use all 20 class logits and the exact native final CAM. Both models and every paired delta reuse the same whole-image bootstrap draws within an analysis family:",
        "",
        _matched_checkpoint_evaluation_table(checkpoint_classification, raw_cam_miou),
        "",
        "[Fact] Raw final-CAM mIoU is a fixed diagnostic on the deterministic transformed 448×448 crop: native final CAM, bilinear upsampling (`align_corners=False`), per-active-class min-max normalization, GT-positive class gating, fixed background threshold 0.45, and void exclusion. It is not a full-image multi-scale or downstream-segmentation result.",
        "",
        "[Fact] Equal-class macro results below expand every unordered positive class pair to both focal endpoints before image-cluster resampling; this prevents the canonical `class_a < class_b` ordering from defining class weights:",
        "",
        _class_pair_macro_table(shared_marginals, pair_macro),
        "",
        "[Fact] Lowest/highest class-wise L12 target-vs-BG AUROC slices (descriptive multiplicity-aware diagnostics, not separately thresholded discoveries):",
        "",
        _classwise_extremes_table(classwise),
        "",
        "[Fact] Lowest/highest matched checkpoint output scores by class (descriptive tails; all class-wise rows remain in the machine-readable tables):",
        "",
        _checkpoint_classwise_extremes_table(checkpoint_classification, raw_cam_miou),
        "",
        "[Interpretation candidate] Class-wise heterogeneity and classification-correctness strata constrain generality; neither should be collapsed into a single universal class-semantic claim.",
        "",
        _fact(
            "Target-visible control: MCTformer+ L12 feature target-vs-BG AUROC",
            visible_feature_auc,
        ),
        _fact(
            "Target-visible control: MCTformer+ L12 attention target-vs-BG AUROC",
            visible_attn_auc,
        ),
        _fact(
            "Target-visible control: MCTformer+ final-CAM target-vs-BG AUROC",
            visible_final_auc,
        ),
        "",
        "## 13. What the Results Support",
        "",
        "[Fact] The tables quantify semantic-region ownership at representation, attention-routing, and exact native CAM stages, including shared supports and layer/stage transitions.",
        "",
        f"[Interpretation candidate] The pre-full-run operational rule selects Case {decision['case']} ({decision['description']}) as primary. Independently satisfied cases are `{decision['satisfied_cases']}` under precedence `{decision['precedence']}`; this is a hypothesis-selection result, not a causal result.",
        "",
        "## 14. What the Results Do Not Support",
        "",
        "- [Unsupported] The results do not prove a causal shortcut, training-time failure mechanism, or classification dependence on any patch.",
        "- [Unsupported] Attention weights are not treated as causal explanations.",
        "- [Unsupported] CAM-region mass and the fixed transformed-crop raw-CAM mIoU diagnostic are not equivalent to downstream segmentation mIoU.",
        "- [Unsupported] Center-cropped target-invisible pairs cannot support target-vs-region discrimination claims and are not silently counted as ordinary negatives.",
        "- [Unsupported] Fixed-threshold raw-cosine interpretations, method improvements, retraining benefits, and solution claims are outside this experiment.",
        "",
        "## 15. Decision for the Next Causal Experiment",
        "",
        f"[Fact] Selected primary **Case {decision['case']}** by rule version `{decision['rule_version']}`: `{decision['rule']}`. This exact CI rule and precedence were fixed in code before the full 1,449-image signal run; the source plan supplied qualitative cases but did not specify mutual exclusion or precedence.",
        "",
        "[Interpretation candidate] The selected case defines the next causal question to test under a separately preregistered intervention; it does not license a mechanism claim now.",
        "",
        "[Unsupported] No model change, new loss, token, attention modification, training, or proposed method is authorized by this report.",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md"
    decision_path = output_dir / "NEXT_EXPERIMENT_DECISION.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    decision_path.write_text(_decision_markdown(decision, bootstrap), encoding="utf-8")
    result = {
        "status": "complete",
        "report": str(report_path),
        "decision": str(decision_path),
        "report_sha256": _sha256_or_unavailable(report_path),
        "decision_sha256": _sha256_or_unavailable(decision_path),
        "selected_case": decision["case"],
        "satisfied_cases": decision["satisfied_cases"],
        "decision_rule_version": decision["rule_version"],
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "claims_are_table_derived": True,
        "model_execution": False,
        "immutable_inputs_verified": True,
        "immutability_verification": str(
            immutability_dir / "immutability_verification.json"
        ),
        "immutability_files_checked": immutability_metadata.get("files_checked"),
        "visual_deliverables": visual_validation,
        "command": command,
    }
    metadata_path = output_dir / "report_metadata.json"
    metadata_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["metadata"] = str(metadata_path)
    result["metadata_sha256"] = _sha256_or_unavailable(metadata_path)
    return result


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    result = generate_reports(
        args.analysis_root,
        args.output_dir,
        tables_dir=args.tables_dir,
        audit_dir=args.audit_dir,
        immutability_dir=args.immutability_dir,
        examples_dir=args.examples_dir,
        plots_dir=args.plots_dir,
        render_dir=args.render_dir,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
