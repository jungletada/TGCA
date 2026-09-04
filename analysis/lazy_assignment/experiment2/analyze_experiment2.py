#!/usr/bin/env python3
"""Aggregate Experiment 2 canonical tables with image-clustered bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.bootstrap import derived_seed  # noqa: E402
from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import (  # noqa: E402
    iter_all_and_label_strata,
    paired_model_frame,
    summarize_clustered_pair_correlations,
    summarize_clustered_macro_class_correlations,
    summarize_clustered,
    summarize_image_mean_correlations,
)
from analysis.lazy_assignment.experiment2.pairwise_class_analysis import (  # noqa: E402
    add_order_invariant_shared_metrics,
    expand_shared_pairs_to_endpoints,
    expand_symmetric_pairs_to_endpoints,
)
from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    paired_cam_iou_bootstrap,
    paired_classification_bootstrap,
)


BOOTSTRAP_SEED = 20260901
MODEL_ORDER = ("mctformer", "mctformer_plus")
PATCH_NORM_JOINT_VALUE_COLUMNS = (
    "post_cosine_patch_l2norm_pearson_valid",
    "post_cosine_patch_l2norm_pearson_bg",
    "feature_top10_bg_patch_l2norm_mean",
    "feature_top10_bg_patch_l2norm_enrichment_vs_bg",
    "feature_top10_bg_below_valid_l2norm_median_fraction",
    "feature_top10_bg_above_valid_l2norm_q75_fraction",
    "feature_top10_bg_patch_count",
)
REGION_VALUE_COLUMNS = (
    "target_hit",
    "other_fg_hit",
    "background_hit",
    "mixed_hit",
    "degenerate_map",
    "all_zero_map",
    "target_top05_fraction",
    "other_fg_top05_fraction",
    "bg_top05_fraction",
    "mixed_top05_fraction",
    "target_tail_enrich_05",
    "other_fg_tail_enrich_05",
    "bg_tail_enrich_05",
    "mixed_tail_enrich_05",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "mixed_top10_fraction",
    "target_tail_enrich_10",
    "other_fg_tail_enrich_10",
    "bg_tail_enrich_10",
    "mixed_tail_enrich_10",
    "target_top20_fraction",
    "other_fg_top20_fraction",
    "bg_top20_fraction",
    "mixed_top20_fraction",
    "target_tail_enrich_20",
    "other_fg_tail_enrich_20",
    "bg_tail_enrich_20",
    "mixed_tail_enrich_20",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "orientation_target_bg",
    "separability_target_bg",
    "conditional_bg_mass",
    "score_mean",
    "score_std",
    "score_q25",
    "score_median",
    "score_q75",
    "score_q90",
    "score_q95",
    "upper_tail_gap",
    "target_bg_mean_margin",
    "target_other_mean_margin",
    "upper_tail_over_std",
    "upper_tail_over_iqr",
    "total_variation",
    "total_variation_over_std",
    "zscore_spatial_entropy",
    "target_mean",
    "target_median",
    "target_q90",
    "target_q95",
    "other_fg_mean",
    "other_fg_median",
    "other_fg_q90",
    "other_fg_q95",
    "bg_mean",
    "bg_median",
    "bg_q90",
    "bg_q95",
    "attn_patch_mass",
    *PATCH_NORM_JOINT_VALUE_COLUMNS,
)
CORE_REGION_VALUE_COLUMNS = (
    "target_hit",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "target_tail_enrich_10",
    "other_fg_tail_enrich_10",
    "bg_tail_enrich_10",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "conditional_bg_mass",
    "target_bg_mean_margin",
    "target_other_mean_margin",
    "attn_patch_mass",
)
SHARED_VALUE_COLUMNS = (
    "shared_set_size",
    "topk_jaccard",
    "topk_overlap_coefficient",
    "shared_target_a_fraction",
    "shared_target_b_fraction",
    "shared_other_fg_fraction",
    "shared_background_fraction",
    "shared_mixed_void_fraction",
    "shared_target_a_enrichment",
    "shared_target_b_enrichment",
    "shared_other_fg_enrichment",
    "shared_background_enrichment",
    "shared_mixed_enrichment",
    "has_previous_layer",
    "new_shared_from_previous_layer",
    "new_shared_target_a_fraction",
    "new_shared_target_b_fraction",
    "new_shared_other_fg_fraction",
    "new_shared_background_fraction",
    "new_shared_mixed_void_fraction",
    "shared_pair_target_fraction",
    "shared_dominant_target_fraction",
    "new_shared_pair_target_fraction",
    "new_shared_dominant_target_fraction",
)
SHARED_FOCAL_VALUE_COLUMNS = (
    "shared_own_target_fraction",
    "shared_partner_target_fraction",
    "shared_other_fg_fraction",
    "shared_background_fraction",
    "shared_mixed_void_fraction",
    "shared_own_target_enrichment",
    "shared_partner_target_enrichment",
    "shared_other_fg_enrichment",
    "shared_background_enrichment",
    "shared_pair_target_fraction",
    "shared_dominant_target_fraction",
    "new_shared_own_target_fraction",
    "new_shared_partner_target_fraction",
    "new_shared_other_fg_fraction",
    "new_shared_background_fraction",
    "new_shared_mixed_void_fraction",
    "new_shared_pair_target_fraction",
    "new_shared_dominant_target_fraction",
)
TRANSITION_VALUE_COLUMNS = (
    "spearman",
    "topk_jaccard",
    "topk_overlap_coefficient",
    "source_topk_size",
    "destination_topk_size",
    "common_topk_size",
    "introduced_size",
    "removed_size",
    "survive_target",
    "survive_other_fg",
    "survive_background",
    "destination_retained_target",
    "destination_retained_other_fg",
    "destination_retained_background",
    "introduced_target_fraction",
    "introduced_other_fg_fraction",
    "introduced_background_fraction",
    "removed_target_fraction",
    "removed_other_fg_fraction",
    "removed_background_fraction",
)
TRANSITION_CLASSWISE_VALUE_COLUMNS = (
    "survive_target",
    "survive_other_fg",
    "survive_background",
    "destination_retained_target",
    "destination_retained_other_fg",
    "destination_retained_background",
    "introduced_target_fraction",
    "introduced_other_fg_fraction",
    "introduced_background_fraction",
    "removed_target_fraction",
    "removed_other_fg_fraction",
    "removed_background_fraction",
)
PAIR_DIVERSITY_VALUE_COLUMNS = (
    "class_token_cosine",
    "spearman",
    "topk_jaccard",
    "topk_overlap_coefficient",
)
TOKEN_VALUE_COLUMNS = (
    "class_token_cosine",
    "feature_post_spearman",
    "feature_post_top05_jaccard",
    "feature_post_top05_overlap_coefficient",
    "feature_post_top10_jaccard",
    "feature_post_top10_overlap_coefficient",
    "feature_post_top20_jaccard",
    "feature_post_top20_overlap_coefficient",
    "attn_c2p_spearman",
    "attn_c2p_top10_jaccard",
    "attn_c2p_top10_overlap_coefficient",
    "qk_mean_spearman",
    "qk_mean_top10_jaccard",
    "qk_mean_top10_overlap_coefficient",
)
QK_HEAD_VALUE_COLUMNS = tuple(
    f"qk_head{head}_{region}_mean"
    for head in range(6)
    for region in ("target", "other_fg", "bg")
)
FAILURE_PATTERN_COLUMNS = (
    "type_a_representation_filtered",
    "type_b_attention_routing",
    "type_c_patch_head",
    "type_d_propagation_amplification",
    "type_e_full_pipeline",
    "unclassified_pattern",
)
CLASSIFICATION_STATUSES = (
    "both_positive",
    "class_only_positive",
    "patch_only_positive",
    "neither_positive",
)
FOCAL_CLASSIFICATION_SUBSETS = (*CLASSIFICATION_STATUSES, "either_negative")
PAIR_CLASSIFICATION_SUBSETS = (
    "both_classes_both_positive",
    "either_class_negative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="permit canonical signal roots marked smoke (never for final reports)",
    )
    args = parser.parse_args()
    if args.bootstrap_repeats < 1:
        parser.error("--bootstrap-repeats must be positive")
    return args


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(path, index=False, float_format="%.10g")
    observed_rows = 0
    observed_columns: list[str] | None = None
    for chunk in pd.read_csv(path, low_memory=False, chunksize=100_000):
        observed_rows += len(chunk)
        if observed_columns is None:
            observed_columns = list(chunk.columns)
    if observed_rows != len(frame) or observed_columns != list(frame.columns):
        raise RuntimeError(f"CSV round-trip mismatch: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": len(frame.columns),
        "chunked_roundtrip_verified": True,
    }


def _available(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in frame and np.isfinite(frame[column].to_numpy(dtype=float)).any()
    ]


def _identity_groups(
    frame: pd.DataFrame, columns: Sequence[str]
) -> Iterable[tuple[tuple[object, ...], pd.DataFrame]]:
    if not columns:
        yield (), frame
        return
    grouper: object = columns[0] if len(columns) == 1 else list(columns)
    yield from frame.groupby(grouper, sort=True, dropna=False)


def _clustered_table(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    label_strata: bool = True,
    macro_class: bool = True,
    class_col: str = "class_id",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in _identity_groups(frame, group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_cols, keys))
        strata = iter_all_and_label_strata(group) if label_strata else (("all", group),)
        for stratum, subset in strata:
            if subset.empty:
                continue
            rows.extend(
                summarize_clustered(
                    subset,
                    value_cols=_available(subset, value_cols),
                    identity={**identity, "label_stratum": stratum},
                    repeats=repeats,
                    seed=derived_seed(seed, "summary", stratum),
                    class_col=class_col,
                    include_macro_class=macro_class,
                )
            )
    return pd.DataFrame(rows)


def _paired_table(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    label_strata: bool = True,
    macro_class: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    non_model_groups = [column for column in group_cols if column != "model"]
    for keys, group in _identity_groups(frame, non_model_groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(non_model_groups, keys))
        strata = iter_all_and_label_strata(group) if label_strata else (("all", group),)
        for stratum, subset in strata:
            values = _available(subset, value_cols)
            if not values or set(subset["model"].unique()) != set(MODEL_ORDER):
                continue
            paired = paired_model_frame(
                subset,
                key_cols=key_cols,
                value_cols=values,
            )
            if paired.empty:
                continue
            rows.extend(
                summarize_clustered(
                    paired,
                    value_cols=values,
                    identity={
                        **identity,
                        "label_stratum": stratum,
                        "delta": "mctformer_plus_minus_mctformer",
                    },
                    repeats=repeats,
                    seed=derived_seed(seed, "paired", stratum),
                    include_macro_class=macro_class,
                )
            )
    return pd.DataFrame(rows)


def _token_overlap_associations(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    overlap_columns = (
        "feature_post_top10_jaccard",
        "attn_c2p_top10_jaccard",
        "qk_mean_top10_jaccard",
    )
    for keys, group in _identity_groups(frame, ("model", "layer")):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(("model", "layer"), keys))
        for stratum, subset in iter_all_and_label_strata(group):
            if subset.empty:
                continue
            rows.extend(
                summarize_clustered_pair_correlations(
                    subset,
                    x_col="class_token_cosine",
                    y_cols=overlap_columns,
                    identity={**identity, "label_stratum": stratum},
                    repeats=repeats,
                    seed=derived_seed(seed, "pair-association", stratum),
                )
            )
            rows.extend(
                summarize_image_mean_correlations(
                    subset,
                    x_col="class_token_cosine",
                    y_cols=overlap_columns,
                    identity={**identity, "label_stratum": stratum},
                    repeats=repeats,
                    seed=derived_seed(seed, "association", stratum),
                )
            )
    return pd.DataFrame(rows)


def _token_endpoint_associations(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Focal-class and equal-class token/map Pearson associations."""

    if frame.empty:
        return pd.DataFrame()
    required = {"model", "layer", "class_id", "image_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"token endpoint association misses columns: {missing}")
    overlap_columns = (
        "feature_post_top10_jaccard",
        "attn_c2p_top10_jaccard",
        "qk_mean_top10_jaccard",
    )
    rows: list[dict[str, object]] = []
    for keys, group in _identity_groups(frame, ("model", "layer")):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(("model", "layer"), keys))
        for stratum, subset in iter_all_and_label_strata(group):
            if subset.empty:
                continue
            stratum_seed = derived_seed(seed, "endpoint-association", stratum)
            rows.extend(
                summarize_clustered_macro_class_correlations(
                    subset,
                    x_col="class_token_cosine",
                    y_cols=overlap_columns,
                    identity={**identity, "label_stratum": stratum},
                    repeats=repeats,
                    seed=derived_seed(stratum_seed, "macro-class"),
                    class_col="class_id",
                )
            )
            for class_id, class_subset in subset.groupby("class_id", sort=True):
                class_rows = summarize_clustered_pair_correlations(
                    class_subset,
                    x_col="class_token_cosine",
                    y_cols=overlap_columns,
                    identity={
                        **identity,
                        "label_stratum": stratum,
                        "class_id": int(class_id),
                    },
                    repeats=repeats,
                    seed=derived_seed(stratum_seed, "class", int(class_id)),
                )
                for row in class_rows:
                    row["aggregation"] = "classwise_pair_pearson"
                    row["association_unit"] = (
                        "within-focal-class positive-pair rows with image-cluster "
                        "resampling"
                    )
                rows.extend(class_rows)
    return pd.DataFrame(rows)


def _classification_subset_expansion(
    frame: pd.DataFrame, *, scope: str
) -> pd.DataFrame:
    """Expand classification controls without changing the bootstrap unit.

    ``image_class`` and ``focal_endpoint`` use the focal class' native
    four-way class-token/patch-head status.  ``either_negative`` is their
    prespecified union: every status except ``both_positive``.  At unordered
    pair scope, both endpoints must be ``both_positive`` to enter
    ``both_classes_both_positive``; its complement is
    ``either_class_negative``.

    Expansion duplicates rows only across reported subsets.  Each subset is
    subsequently summarized with whole ``image_id`` clusters, never with the
    duplicated rows, endpoints, or pairs as bootstrap units.
    """

    if scope not in {"image_class", "focal_endpoint", "unordered_pair"}:
        raise ValueError(f"unknown classification-control scope: {scope}")
    status_columns = (
        ("classification_status_a", "classification_status_b")
        if scope == "unordered_pair"
        else ("classification_status",)
    )
    missing = [column for column in status_columns if column not in frame]
    if missing:
        if frame.empty:
            result = frame.copy()
            result["classification_subset"] = pd.Series(dtype="string")
            return result
        raise ValueError(
            f"{scope} classification control misses required columns: {missing}"
        )

    for column in status_columns:
        status = frame[column]
        if bool(status.isna().any()):
            raise ValueError(f"{column} contains missing classification status")
        unexpected = sorted(set(status.astype(str)) - set(CLASSIFICATION_STATUSES))
        if unexpected:
            raise ValueError(
                f"{column} contains unexpected classification status: {unexpected}"
            )

    parts: list[pd.DataFrame] = []
    if scope in {"image_class", "focal_endpoint"}:
        status = frame["classification_status"].astype(str)
        for subset_name in FOCAL_CLASSIFICATION_SUBSETS:
            mask = (
                status.ne("both_positive")
                if subset_name == "either_negative"
                else status.eq(subset_name)
            )
            if bool(mask.any()):
                parts.append(frame.loc[mask].assign(classification_subset=subset_name))
    else:
        both_positive = frame["classification_status_a"].astype(str).eq(
            "both_positive"
        ) & frame["classification_status_b"].astype(str).eq("both_positive")
        for subset_name in PAIR_CLASSIFICATION_SUBSETS:
            mask = (
                both_positive
                if subset_name == "both_classes_both_positive"
                else ~both_positive
            )
            if bool(mask.any()):
                parts.append(frame.loc[mask].assign(classification_subset=subset_name))
    if not parts:
        result = frame.iloc[0:0].copy()
        result["classification_subset"] = pd.Series(dtype="string")
        return result
    return pd.concat(parts, ignore_index=True, sort=False)


def _classification_stratified_summary(
    frame: pd.DataFrame,
    *,
    scope: str,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    macro_class: bool,
    class_col: str = "class_id",
) -> pd.DataFrame:
    expanded = _classification_subset_expansion(frame, scope=scope)
    if expanded.empty:
        return pd.DataFrame()
    table = _clustered_table(
        expanded,
        group_cols=("classification_subset", *group_cols),
        value_cols=value_cols,
        repeats=repeats,
        seed=derived_seed(seed, "classification", scope),
        label_strata=True,
        macro_class=macro_class,
        class_col=class_col,
    )
    if not table.empty:
        table.insert(0, "classification_scope", scope)
    return table


def _classification_control(
    frame: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    return _classification_stratified_summary(
        frame,
        scope="image_class",
        group_cols=("model", "signal", "layer", "rho"),
        value_cols=CORE_REGION_VALUE_COLUMNS,
        repeats=repeats,
        seed=seed,
        macro_class=True,
    )


def _single_classwise_coverage_summary(
    frame: pd.DataFrame,
    *,
    source_table: str,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    paired_comparison: bool,
    key_cols: Sequence[str] = ("image_id", "class_id"),
) -> pd.DataFrame:
    """Add explicit within-class summaries and optional exact-key deltas.

    Every confidence interval resamples whole ``image_id`` clusters.  The
    paired branch first intersects ``key_cols`` within each non-model metric
    identity, then computes MCTformer+ minus MCTformer.  It is intentionally
    disabled for model-specific classification-status conditioning.
    """

    if frame.empty:
        return pd.DataFrame()
    required = {"model", "image_id", "class_id", *group_cols, *key_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_table} class-wise input misses columns: {missing}")
    if "model" not in group_cols:
        raise ValueError("group_cols must contain model")
    parts: list[pd.DataFrame] = []
    for class_id, subset in frame.groupby("class_id", sort=True):
        local_seed = derived_seed(seed, source_table, int(class_id))
        per_model = _clustered_table(
            subset,
            group_cols=group_cols,
            value_cols=value_cols,
            repeats=repeats,
            seed=derived_seed(local_seed, "per-model"),
            label_strata=True,
            macro_class=False,
        )
        if not per_model.empty:
            per_model.insert(0, "class_id", int(class_id))
            per_model.insert(1, "source_table", source_table)
            per_model["model_or_delta"] = per_model["model"].astype(str)
            per_model["aggregation_scope"] = "within_class"
            per_model["comparison_policy"] = (
                "per_model_not_paired"
                if paired_comparison
                else "not_applicable_model_specific_conditioning"
            )
            parts.append(per_model)

        if not paired_comparison:
            continue
        paired = _paired_table(
            subset,
            group_cols=group_cols,
            key_cols=key_cols,
            value_cols=value_cols,
            repeats=repeats,
            seed=derived_seed(local_seed, "paired"),
            label_strata=True,
            macro_class=False,
        )
        if not paired.empty:
            paired.insert(0, "class_id", int(class_id))
            paired.insert(1, "source_table", source_table)
            paired["model_or_delta"] = "mctformer_plus_minus_mctformer"
            paired["aggregation_scope"] = "within_class"
            paired["comparison_policy"] = "exact_common_key_paired"
            paired["paired_key_columns"] = ",".join(key_cols)
            parts.append(paired)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _classification_conditioned_classwise_control(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Per-model class-wise controls under native classification status.

    Status is a model output and therefore defines different subsets for the
    two checkpoints.  Cross-model paired deltas are structurally inapplicable;
    the returned rows state that policy explicitly rather than comparing
    mismatched conditioned populations.
    """

    expanded = _classification_subset_expansion(frame, scope="image_class")
    return _single_classwise_coverage_summary(
        expanded,
        source_table="classification_conditioned",
        group_cols=(
            "model",
            "control_source",
            "classification_subset",
            "signal",
            "layer",
            "rho",
        ),
        value_cols=CORE_REGION_VALUE_COLUMNS,
        repeats=repeats,
        seed=seed,
        paired_comparison=False,
    )


def _primary_classwise_summary(
    layer: pd.DataFrame,
    cam: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Full per-model and paired region metrics within every focal class."""

    # Section 17 applies to every canonical layer signal and CAM stage, including
    # the relative-score, active-softmax, patch-norm, mid3, and diagnostic controls.
    classwise_layer = layer
    classwise_cam = cam
    parts = [
        _single_classwise_coverage_summary(
            classwise_layer,
            source_table="layer_signal",
            group_cols=("model", "signal", "layer", "rho"),
            value_cols=REGION_VALUE_COLUMNS,
            repeats=repeats,
            seed=derived_seed(seed, "layer-signal"),
            paired_comparison=True,
        ),
        _single_classwise_coverage_summary(
            classwise_cam,
            source_table="cam_stage",
            group_cols=("model", "stage", "rho"),
            value_cols=REGION_VALUE_COLUMNS,
            repeats=repeats,
            seed=derived_seed(seed, "cam-stage"),
            paired_comparison=True,
        ),
    ]
    nonempty = [frame for frame in parts if not frame.empty]
    return (
        pd.concat(nonempty, ignore_index=True, sort=False)
        if nonempty
        else pd.DataFrame()
    )


def _class_pair_classwise_summary(
    frame: pd.DataFrame,
    *,
    source_table: str,
    group_cols: Sequence[str],
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Per-model and paired endpoint summaries for every focal class.

    Pair rows have already been expanded to both focal endpoints.  Deltas are
    formed only on exact common endpoint keys and are always
    MCTformer+ minus MCTformer.  ``_paired_table`` resamples ``image_id``
    clusters, so multiple partner classes in one image remain dependent.
    """

    if frame.empty:
        return pd.DataFrame()
    required = {"class_id", "image_id", "partner_class_id", "model"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source_table} endpoint table misses class-wise columns: {missing}"
        )
    parts: list[pd.DataFrame] = []
    for class_id, subset in frame.groupby("class_id", sort=True):
        local_seed = derived_seed(seed, source_table, int(class_id))
        per_model = _clustered_table(
            subset,
            group_cols=group_cols,
            value_cols=value_cols,
            repeats=repeats,
            seed=derived_seed(local_seed, "per-model"),
            label_strata=True,
            macro_class=False,
        )
        if not per_model.empty:
            per_model.insert(0, "class_id", int(class_id))
            per_model.insert(1, "source_table", source_table)
            per_model["model_or_delta"] = per_model["model"].astype(str)
            parts.append(per_model)

        paired = _paired_table(
            subset,
            group_cols=group_cols,
            key_cols=key_cols,
            value_cols=value_cols,
            repeats=repeats,
            seed=derived_seed(local_seed, "paired"),
            label_strata=True,
            macro_class=False,
        )
        if not paired.empty:
            paired.insert(0, "class_id", int(class_id))
            paired.insert(1, "source_table", source_table)
            paired["model_or_delta"] = "mctformer_plus_minus_mctformer"
            parts.append(paired)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _stage_transition_classwise_summary(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Per-model and exact-common-key paired transition metrics by class."""

    if frame.empty:
        return pd.DataFrame()
    group_cols = ("model", "transition", "layer", "rho", "topk_ratio")
    required = {"image_id", "class_id", *group_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stage transition table misses class-wise columns: {missing}")
    parts: list[pd.DataFrame] = []
    for class_id, subset in frame.groupby("class_id", sort=True):
        local_seed = derived_seed(seed, "stage-transition", int(class_id))
        per_model = _clustered_table(
            subset,
            group_cols=group_cols,
            value_cols=TRANSITION_CLASSWISE_VALUE_COLUMNS,
            repeats=repeats,
            seed=derived_seed(local_seed, "per-model"),
            label_strata=True,
            macro_class=False,
        )
        if not per_model.empty:
            per_model.insert(0, "class_id", int(class_id))
            per_model.insert(1, "source_table", "stage_transition")
            per_model["model_or_delta"] = per_model["model"].astype(str)
            parts.append(per_model)

        paired = _paired_table(
            subset,
            group_cols=group_cols,
            key_cols=("image_id", "class_id"),
            value_cols=TRANSITION_CLASSWISE_VALUE_COLUMNS,
            repeats=repeats,
            seed=derived_seed(local_seed, "paired"),
            label_strata=True,
            macro_class=False,
        )
        if not paired.empty:
            paired.insert(0, "class_id", int(class_id))
            paired.insert(1, "source_table", "stage_transition")
            paired["model_or_delta"] = "mctformer_plus_minus_mctformer"
            parts.append(paired)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _paired_checkpoint_evaluations(
    classification: pd.DataFrame,
    cam_confusion: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_keys = ["image_id", "class_id"]
    if classification.duplicated(["model", *classification_keys]).any():
        raise RuntimeError("classification canonical keys are not unique")
    models = {}
    label_reference: pd.DataFrame | None = None
    for model in MODEL_ORDER:
        subset = classification[classification["model"] == model]
        target = subset.pivot(index="image_id", columns="class_id", values="target")
        target = target.sort_index().sort_index(axis=1)
        if list(target.columns) != list(range(20)):
            raise RuntimeError(
                f"{model} classification table does not contain 20 classes"
            )
        if label_reference is None:
            label_reference = target
        elif not target.equals(label_reference):
            raise RuntimeError("paired model image-level labels do not match")
        models[model] = {
            "class_token": subset.pivot(
                index="image_id", columns="class_id", values="class_logit"
            )
            .reindex(index=target.index, columns=target.columns)
            .to_numpy(dtype=np.float64),
            "patch_head": subset.pivot(
                index="image_id", columns="class_id", values="patch_class_logit"
            )
            .reindex(index=target.index, columns=target.columns)
            .to_numpy(dtype=np.float64),
        }
    assert label_reference is not None
    image_ids = label_reference.index.astype(str).tolist()
    labels = label_reference.to_numpy(dtype=np.uint8)
    classification_rows: list[dict[str, object]] = []
    for source in ("class_token", "patch_head"):
        classification_rows.extend(
            paired_classification_bootstrap(
                image_ids,
                labels,
                models["mctformer"][source],
                models["mctformer_plus"][source],
                repeats=repeats,
                seed=derived_seed(seed, "checkpoint-classification", source),
                logit_source=source,
            )
        )

    if cam_confusion.duplicated(
        ["model", "image_id", "gt_class_id", "pred_class_id"]
    ).any():
        raise RuntimeError("raw-CAM confusion canonical keys are not unique")
    confusion_models = {}
    expected_columns = pd.MultiIndex.from_product([range(21), range(21)])
    for model in MODEL_ORDER:
        subset = cam_confusion[cam_confusion["model"] == model]
        pivot = subset.pivot(
            index="image_id",
            columns=["gt_class_id", "pred_class_id"],
            values="pixel_count",
        ).sort_index()
        pivot = pivot.reindex(index=label_reference.index, columns=expected_columns)
        if pivot.isna().any().any():
            raise RuntimeError(f"{model} raw-CAM confusion table is incomplete")
        confusion_models[model] = pivot.to_numpy(dtype=np.int64).reshape(-1, 21, 21)
    cam_rows = paired_cam_iou_bootstrap(
        image_ids,
        labels,
        confusion_models["mctformer"],
        confusion_models["mctformer_plus"],
        repeats=repeats,
        seed=derived_seed(seed, "raw-final-cam-miou"),
    )
    return pd.DataFrame(classification_rows), pd.DataFrame(cam_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_canonical(
    canonical: Path,
    paths: dict[str, Path],
    *,
    allow_smoke: bool,
) -> dict[str, object]:
    metadata_path = canonical / "canonical_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing canonical metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("canonical metadata is not complete")
    if metadata.get("source_immutability_verified") is not True:
        raise RuntimeError("canonical source immutability was not verified")
    if metadata.get("source_manifests_exact_match") is not True:
        raise RuntimeError("canonical source manifests did not match")
    table_records = metadata.get("tables")
    if not isinstance(table_records, dict):
        raise RuntimeError("canonical metadata has no table records")
    verified_tables: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical {name} table: {path}")
        record = table_records.get(path.stem)
        if not isinstance(record, dict):
            raise RuntimeError(f"canonical metadata has no record for {path.stem}")
        expected_hash = record.get("sha256")
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"canonical table SHA-256 mismatch for {path.name}: "
                f"{actual_hash} != {expected_hash}"
            )
        if record.get("roundtrip_verified") is not True:
            raise RuntimeError(f"canonical round-trip was not verified for {path.name}")
        verified_tables[name] = {
            "path": str(path),
            "sha256": actual_hash,
            "expected_rows": int(record["rows"]),
        }

    expected_images = int(metadata.get("num_manifest_images_per_model", 0))
    if expected_images < 1:
        raise RuntimeError("canonical metadata has invalid image count")
    source_roots = metadata.get("source_roots")
    if not isinstance(source_roots, dict) or set(source_roots) != set(MODEL_ORDER):
        raise RuntimeError("canonical metadata does not identify both signal roots")
    signal_runs: dict[str, dict[str, object]] = {}
    for model, root_value in source_roots.items():
        signal_root = Path(str(root_value)).resolve()
        completion_path = signal_root / "completion.json"
        signal_metadata_path = signal_root / "metadata.json"
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        if not signal_metadata_path.is_file():
            raise FileNotFoundError(signal_metadata_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        signal_metadata = json.loads(signal_metadata_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            raise RuntimeError(f"{model} signal run is not complete")
        run_kind = completion.get("run_kind")
        if run_kind != "full" and not allow_smoke:
            raise RuntimeError(
                f"refusing non-full canonical input for analysis: {model} "
                f"run_kind={run_kind!r}; use --allow-smoke only for tests"
            )
        if int(completion.get("num_images", -1)) != expected_images:
            raise RuntimeError(f"{model} signal image count disagrees with canonical")
        if (
            signal_metadata.get("status") != "complete"
            or signal_metadata.get("model") != model
            or signal_metadata.get("run_kind") != run_kind
            or int(signal_metadata.get("processed_images", -1)) != expected_images
        ):
            raise RuntimeError(f"{model} signal metadata/completion identity disagrees")
        source_metadata_path = (
            Path(str(signal_metadata.get("source_metadata", ""))).expanduser().resolve()
        )
        if not source_metadata_path.is_file() or _sha256(
            source_metadata_path
        ) != signal_metadata.get("source_metadata_sha256"):
            raise RuntimeError(f"{model} signal input-audit linkage is invalid")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        source = source_metadata.get("sources", {}).get(model, {})
        checkpoint = signal_metadata.get("checkpoint", {})
        audited_checkpoint = source.get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("path", ""))).expanduser().resolve()
        if (
            source_metadata.get("integrity_passed") is not True
            or checkpoint_path
            != Path(str(audited_checkpoint.get("path", ""))).expanduser().resolve()
            or checkpoint.get("sha256") != audited_checkpoint.get("sha256")
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint.get("sha256")
        ):
            raise RuntimeError(f"{model} signal checkpoint/audit provenance is invalid")
        signal_dataset = signal_metadata.get("dataset", {})
        if (
            int(signal_dataset.get("input_size", -1)) != 448
            or int(signal_dataset.get("patch_size", -1)) != 16
            or "nearest-neighbor semantic-mask geometry"
            not in str(signal_dataset.get("transform", ""))
        ):
            raise RuntimeError(f"{model} signal geometry contract is invalid")
        tolerance_checks = (
            ("experiment1_feature_post_max_abs_diff", 1e-6, False),
            ("qk_attention_max_abs_diff", 1e-6, False),
            ("native_cam_max_abs_diff", 1e-6, False),
            ("attention_row_sum_max_abs_error", 5e-6, True),
            ("conditional_attention_row_sum_max_abs_error", 5e-6, True),
        )
        for key, limit, inclusive in tolerance_checks:
            try:
                value = float(signal_metadata[key])
                completion_value = float(completion[key])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"{model} lacks numerical check {key}") from error
            if (
                not np.isfinite(value)
                or (value > limit if inclusive else value >= limit)
                or value != completion_value
            ):
                raise RuntimeError(f"{model} failed numerical check {key}: {value}")
        git = signal_metadata.get("git", {})
        tracked = git.get("runtime_source_tracked", {})
        if run_kind == "full" and (
            git.get("tracked_dirty") is not False
            or not isinstance(tracked, dict)
            or not tracked
            or not all(value is True for value in tracked.values())
        ):
            raise RuntimeError(f"{model} full signal run lacks clean tracked source")
        signal_runs[model] = {
            "root": str(signal_root),
            "metadata": str(signal_metadata_path),
            "metadata_sha256": _sha256(signal_metadata_path),
            "completion": str(completion_path),
            "completion_sha256": _sha256(completion_path),
            "run_kind": run_kind,
            "num_images": int(completion["num_images"]),
        }
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "num_images_per_model": expected_images,
        "allow_smoke": bool(allow_smoke),
        "tables": verified_tables,
        "signal_runs": signal_runs,
    }


def _failure_pattern_frame(layer: pd.DataFrame, cam: pd.DataFrame) -> pd.DataFrame:
    """Classify every complete image-class tuple using registered >1 enrichment.

    Pattern flags are intentionally non-exclusive: a row can expose more than
    one measured failure location.  The raw per-pair table is retained so the
    prevalence summary can always be audited back to its inputs.
    """

    keys = [
        "model",
        "image_id",
        "class_id",
        "class_name",
        "num_positive_classes",
        "classification_status",
        "rho",
    ]
    layer_subset = layer[
        (pd.to_numeric(layer["layer"], errors="coerce") == 12)
        & layer["signal"].isin(("feature_post", "attn_c2p_conditional"))
    ][[*keys, "signal", "bg_tail_enrich_10"]]
    layer_wide = layer_subset.pivot(
        index=keys, columns="signal", values="bg_tail_enrich_10"
    ).reset_index()
    layer_wide = layer_wide.rename(
        columns={
            "feature_post": "feature_bg_enrichment",
            "attn_c2p_conditional": "attention_bg_enrichment",
        }
    )
    cam_subset = cam[cam["stage"].isin(("patch_cam", "c2p_cam", "final_cam"))][
        [*keys, "stage", "bg_tail_enrich_10"]
    ]
    cam_wide = cam_subset.pivot(
        index=keys, columns="stage", values="bg_tail_enrich_10"
    ).reset_index()
    cam_wide = cam_wide.rename(
        columns={
            "patch_cam": "patch_cam_bg_enrichment",
            "c2p_cam": "c2p_cam_bg_enrichment",
            "final_cam": "final_cam_bg_enrichment",
        }
    )
    result = layer_wide.merge(cam_wide, on=keys, how="inner", validate="one_to_one")
    required = (
        "feature_bg_enrichment",
        "attention_bg_enrichment",
        "patch_cam_bg_enrichment",
        "c2p_cam_bg_enrichment",
        "final_cam_bg_enrichment",
    )
    complete = np.logical_and.reduce(
        [
            np.isfinite(pd.to_numeric(result[column], errors="coerce"))
            for column in required
        ]
    )
    feature_high = result["feature_bg_enrichment"] > 1.0
    attention_high = result["attention_bg_enrichment"] > 1.0
    patch_high = result["patch_cam_bg_enrichment"] > 1.0
    c2p_high = result["c2p_cam_bg_enrichment"] > 1.0
    final_high = result["final_cam_bg_enrichment"] > 1.0
    result["complete_signal_tuple"] = complete
    result["type_a_representation_filtered"] = (
        complete & feature_high & ~attention_high & ~final_high
    )
    result["type_b_attention_routing"] = complete & ~feature_high & attention_high
    result["type_c_patch_head"] = (
        complete & ~feature_high & ~attention_high & patch_high
    )
    result["type_d_propagation_amplification"] = complete & ~c2p_high & final_high
    result["type_e_full_pipeline"] = (
        complete & feature_high & attention_high & final_high
    )
    active = result[list(FAILURE_PATTERN_COLUMNS[:-1])].sum(axis=1)
    result["unclassified_pattern"] = complete & (active == 0)
    result["num_active_patterns"] = active.astype(int)
    result["pattern_threshold"] = "BG-TailEnrich@10% > 1"
    return result


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    canonical = args.canonical_dir.resolve()
    output = args.output_dir.resolve()
    external_provenance_paths = {
        "exact_commands": output.parent / "exact_commands.sh",
        "pipeline_metadata": output.parent / "pipeline_metadata.json",
    }
    external_provenance_sha256: dict[str, str] = {}
    for name, path in external_provenance_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"required Experiment 2 provenance artifact is missing: {path}"
            )
        if path.stat().st_size == 0:
            raise RuntimeError(
                f"required Experiment 2 provenance artifact is empty: {path}"
            )
        external_provenance_sha256[name] = _sha256(path)
    paths = {
        "layer": canonical / "per_image_class_layer_signal.parquet",
        "cam": canonical / "per_image_class_cam_stage.parquet",
        "transition": canonical / "per_image_class_stage_transition.parquet",
        "pair": canonical / "per_multilabel_class_pair_layer_signal.parquet",
        "shared": canonical / "per_shared_patch_ownership.parquet",
        "token": canonical / "per_class_token_pair_layer.parquet",
        "classification": canonical / "per_image_classification.parquet",
        "cam_confusion": canonical / "per_image_cam_confusion.parquet",
        "source": canonical / "source_index.parquet",
    }
    canonical_verification = _verify_canonical(
        canonical, paths, allow_smoke=bool(args.allow_smoke)
    )
    output.mkdir(parents=True, exist_ok=False)
    tables_out = output / "tables"
    tables_out.mkdir()
    log_path = output / "analysis.log"
    command = shlex.join([sys.executable, *sys.argv])
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log("canonical metadata, table hashes, and signal completion state verified")
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name, frame in frames.items():
        expected_rows = canonical_verification["tables"][name]["expected_rows"]
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"canonical row-count mismatch for {name}: {len(frame)} != {expected_rows}"
            )
    layer = frames["layer"]
    cam = frames["cam"]

    products: dict[str, pd.DataFrame] = {}
    products["layerwise_region_metrics.csv"] = _clustered_table(
        layer,
        group_cols=("model", "signal", "layer", "rho"),
        value_cols=REGION_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    products["cam_stage_region_metrics.csv"] = _clustered_table(
        cam,
        group_cols=("model", "stage", "rho"),
        value_cols=REGION_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "cam"),
    )
    feature_patch_norm = layer[layer["signal"].astype(str) == "feature_post"]
    patch_norm_per_model = _clustered_table(
        feature_patch_norm,
        group_cols=("model", "layer", "rho"),
        value_cols=PATCH_NORM_JOINT_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "feature-patch-norm"),
    )
    if not patch_norm_per_model.empty:
        patch_norm_per_model["model_or_delta"] = patch_norm_per_model["model"].astype(
            str
        )
        patch_norm_per_model["comparison_policy"] = "per_model_not_paired"
        patch_norm_per_model["aggregation_scope"] = "overall"
    patch_norm_paired = _paired_table(
        feature_patch_norm,
        group_cols=("model", "layer", "rho"),
        key_cols=("image_id", "class_id"),
        value_cols=PATCH_NORM_JOINT_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "paired-feature-patch-norm"),
    )
    if not patch_norm_paired.empty:
        patch_norm_paired["model_or_delta"] = "mctformer_plus_minus_mctformer"
        patch_norm_paired["comparison_policy"] = "exact_common_key_paired"
        patch_norm_paired["paired_key_columns"] = "image_id,class_id"
        patch_norm_paired["aggregation_scope"] = "overall"
    patch_norm_classwise = _single_classwise_coverage_summary(
        feature_patch_norm,
        source_table="patch_norm_joint_control",
        group_cols=("model", "layer", "rho"),
        value_cols=PATCH_NORM_JOINT_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "classwise-feature-patch-norm"),
        paired_comparison=True,
    )
    products["patch_norm_joint_control.csv"] = pd.concat(
        [patch_norm_per_model, patch_norm_paired, patch_norm_classwise],
        ignore_index=True,
        sort=False,
    )
    log("completed primary layer-wise and CAM-stage bootstrap summaries")
    visible_layer = layer[layer["has_target_region"].astype(bool)]
    visible_cam = cam[cam["has_target_region"].astype(bool)]
    products["target_visible_region_metrics.csv"] = pd.concat(
        [
            _clustered_table(
                visible_layer,
                group_cols=("model", "signal", "layer", "rho"),
                value_cols=CORE_REGION_VALUE_COLUMNS,
                repeats=args.bootstrap_repeats,
                seed=derived_seed(args.bootstrap_seed, "visible-layer"),
            ).assign(table="layer_signal"),
            _clustered_table(
                visible_cam,
                group_cols=("model", "stage", "rho"),
                value_cols=CORE_REGION_VALUE_COLUMNS,
                repeats=args.bootstrap_repeats,
                seed=derived_seed(args.bootstrap_seed, "visible-cam"),
            ).assign(table="cam_stage"),
        ],
        ignore_index=True,
        sort=False,
    )
    target_visible_classwise_parts = [
        _single_classwise_coverage_summary(
            visible_layer,
            source_table="target_visible_layer_signal",
            group_cols=("model", "signal", "layer", "rho"),
            value_cols=CORE_REGION_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "visible-layer-classwise"),
            paired_comparison=True,
        ),
        _single_classwise_coverage_summary(
            visible_cam,
            source_table="target_visible_cam_stage",
            group_cols=("model", "stage", "rho"),
            value_cols=CORE_REGION_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "visible-cam-classwise"),
            paired_comparison=True,
        ),
    ]
    target_visible_classwise_nonempty = [
        frame for frame in target_visible_classwise_parts if not frame.empty
    ]
    products["target_visible_classwise_results.csv"] = (
        pd.concat(
            target_visible_classwise_nonempty,
            ignore_index=True,
            sort=False,
        )
        if target_visible_classwise_nonempty
        else pd.DataFrame()
    )
    products["stage_transition_metrics.csv"] = _clustered_table(
        frames["transition"],
        group_cols=("model", "transition", "layer", "rho", "topk_ratio"),
        value_cols=TRANSITION_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "transition"),
    )
    products["stage_transition_classwise_results.csv"] = (
        _stage_transition_classwise_summary(
            frames["transition"],
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "transition-classwise"),
        )
    )
    log(
        "completed target-visible, aggregate transition, and per-model/paired "
        "class-wise transition summaries"
    )
    shared_input = add_order_invariant_shared_metrics(frames["shared"])
    shared_endpoints = expand_shared_pairs_to_endpoints(frames["shared"])
    # A tiny smoke prefix may contain only single-label images, in which case
    # the canonical pair table is correctly empty.  Supply the analysis-only
    # columns so that this structural N/A case yields explicit empty products
    # instead of being mistaken for a schema failure.
    if shared_input.empty:
        for column in (
            "has_previous_layer",
            "new_shared_transition",
            *SHARED_VALUE_COLUMNS,
        ):
            if column not in shared_input:
                shared_input[column] = pd.Series(dtype="object")
    if shared_endpoints.empty:
        for column in (
            "has_previous_layer",
            "new_shared_transition",
            "class_id",
            "partner_class_id",
            *SHARED_FOCAL_VALUE_COLUMNS,
        ):
            if column not in shared_endpoints:
                shared_endpoints[column] = pd.Series(dtype="object")
    no_previous = ~shared_input["has_previous_layer"].astype(bool)
    no_previous_endpoints = ~shared_endpoints["has_previous_layer"].astype(bool)
    for column in SHARED_VALUE_COLUMNS:
        if column.startswith("new_shared_") and column in shared_input:
            shared_input.loc[no_previous, column] = np.nan
    for column in SHARED_FOCAL_VALUE_COLUMNS:
        if column.startswith("new_shared_") and column in shared_endpoints:
            shared_endpoints.loc[no_previous_endpoints, column] = np.nan
    products["shared_support_ownership.csv"] = _clustered_table(
        shared_input,
        group_cols=(
            "model",
            "signal",
            "layer_or_stage",
            "new_shared_transition",
            "rho",
            "topk_ratio",
        ),
        value_cols=SHARED_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "shared"),
        macro_class=False,
    )
    shared = products["shared_support_ownership.csv"]
    if not shared.empty:
        transition_names = {"L9_to_L10", "L10_to_L11", "L11_to_L12"}
        products["new_shared_support_l9_l12.csv"] = shared[
            shared["new_shared_transition"].isin(transition_names)
            & shared["metric"].astype(str).str.startswith("new_shared_")
        ].copy()
    else:
        products["new_shared_support_l9_l12.csv"] = pd.DataFrame()

    products["multiclass_map_diversity.csv"] = _clustered_table(
        frames["pair"],
        group_cols=("model", "signal", "layer", "topk_ratio"),
        value_cols=PAIR_DIVERSITY_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "pair-diversity"),
        macro_class=False,
    )

    shared_group_cols = (
        "model",
        "signal",
        "layer_or_stage",
        "new_shared_transition",
        "rho",
        "topk_ratio",
    )
    products["shared_support_class_marginals.csv"] = _clustered_table(
        shared_endpoints,
        group_cols=shared_group_cols,
        value_cols=SHARED_FOCAL_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "shared-class-marginals"),
        macro_class=True,
        class_col="class_id",
    )
    pair_endpoints = expand_symmetric_pairs_to_endpoints(frames["pair"])
    token_endpoints = expand_symmetric_pairs_to_endpoints(frames["token"])
    pair_macro_parts = [
        _clustered_table(
            pair_endpoints,
            group_cols=("model", "signal", "layer", "topk_ratio"),
            value_cols=PAIR_DIVERSITY_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "pair-class-marginals"),
            macro_class=True,
            class_col="class_id",
        ).assign(source_table="multiclass_map_diversity"),
        _clustered_table(
            token_endpoints,
            group_cols=("model", "layer"),
            value_cols=TOKEN_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "token-class-marginals"),
            macro_class=True,
            class_col="class_id",
        ).assign(source_table="class_token_map_overlap"),
    ]
    nonempty_pair_macro = [frame for frame in pair_macro_parts if not frame.empty]
    products["class_pair_macro_class_results.csv"] = (
        pd.concat(nonempty_pair_macro, ignore_index=True, sort=False)
        if nonempty_pair_macro
        else pd.DataFrame()
    )

    focal_classification_parts: list[pd.DataFrame] = []
    pair_classification_parts: list[pd.DataFrame] = []
    for (
        source_name,
        focal_frame,
        unordered_frame,
        group_cols,
        focal_value_cols,
        unordered_value_cols,
    ) in (
        (
            "shared_support",
            shared_endpoints,
            shared_input,
            shared_group_cols,
            SHARED_FOCAL_VALUE_COLUMNS,
            SHARED_VALUE_COLUMNS,
        ),
        (
            "multiclass_map_diversity",
            pair_endpoints,
            frames["pair"],
            ("model", "signal", "layer", "topk_ratio"),
            PAIR_DIVERSITY_VALUE_COLUMNS,
            PAIR_DIVERSITY_VALUE_COLUMNS,
        ),
        (
            "class_token_map_overlap",
            token_endpoints,
            frames["token"],
            ("model", "layer"),
            TOKEN_VALUE_COLUMNS,
            TOKEN_VALUE_COLUMNS,
        ),
    ):
        focal_table = _classification_stratified_summary(
            focal_frame,
            scope="focal_endpoint",
            group_cols=group_cols,
            value_cols=focal_value_cols,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(
                args.bootstrap_seed, "pair-focal-classification", source_name
            ),
            macro_class=True,
            class_col="class_id",
        )
        if not focal_table.empty:
            focal_table.insert(0, "source_table", source_name)
            focal_classification_parts.append(focal_table)
        unordered_table = _classification_stratified_summary(
            unordered_frame,
            scope="unordered_pair",
            group_cols=group_cols,
            value_cols=unordered_value_cols,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(
                args.bootstrap_seed, "pair-joint-classification", source_name
            ),
            macro_class=False,
        )
        if not unordered_table.empty:
            unordered_table.insert(0, "source_table", source_name)
            pair_classification_parts.append(unordered_table)
    products["class_pair_focal_classification_stratified_results.csv"] = (
        pd.concat(focal_classification_parts, ignore_index=True, sort=False)
        if focal_classification_parts
        else pd.DataFrame()
    )
    products["class_pair_joint_classification_stratified_results.csv"] = (
        pd.concat(pair_classification_parts, ignore_index=True, sort=False)
        if pair_classification_parts
        else pd.DataFrame()
    )

    pair_classwise_parts = []
    for source_name, source_frame, group_cols, key_cols, value_cols in (
        (
            "shared_support",
            shared_endpoints,
            shared_group_cols,
            ("image_id", "class_id", "partner_class_id"),
            SHARED_FOCAL_VALUE_COLUMNS,
        ),
        (
            "multiclass_map_diversity",
            pair_endpoints,
            ("model", "signal", "layer", "topk_ratio"),
            ("image_id", "class_id", "partner_class_id"),
            PAIR_DIVERSITY_VALUE_COLUMNS,
        ),
        (
            "class_token_map_overlap",
            token_endpoints,
            ("model", "layer"),
            ("image_id", "class_id", "partner_class_id"),
            TOKEN_VALUE_COLUMNS,
        ),
    ):
        table = _class_pair_classwise_summary(
            source_frame,
            source_table=source_name,
            group_cols=group_cols,
            key_cols=key_cols,
            value_cols=value_cols,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "pair-classwise"),
        )
        if not table.empty:
            pair_classwise_parts.append(table)
    products["class_pair_classwise_results.csv"] = (
        pd.concat(pair_classwise_parts, ignore_index=True, sort=False)
        if pair_classwise_parts
        else pd.DataFrame()
    )
    qk_head_input = layer[layer["signal"] == "qk_mean"]
    products["qk_head_region_summary.csv"] = _clustered_table(
        qk_head_input,
        group_cols=("model", "layer", "rho"),
        value_cols=QK_HEAD_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "qk-head-regions"),
    )
    products["qk_head_classwise_results.csv"] = _single_classwise_coverage_summary(
        qk_head_input,
        source_table="qk_head_control",
        group_cols=("model", "signal", "layer", "rho"),
        value_cols=QK_HEAD_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "qk-head-classwise"),
        paired_comparison=True,
    )
    log(
        "completed shared support, multi-class diversity, endpoint/pair "
        "classification controls, paired class-wise deltas, and aggregate/class-wise "
        "head-wise QK summaries"
    )

    failure_patterns = _failure_pattern_frame(layer, cam)
    products["per_image_class_failure_patterns.csv"] = failure_patterns
    products["failure_pattern_summary.csv"] = _clustered_table(
        failure_patterns[failure_patterns["complete_signal_tuple"]],
        group_cols=("model", "rho"),
        value_cols=FAILURE_PATTERN_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "failure-patterns"),
    )

    products["feature_attention_cam_linkage.csv"] = products[
        "stage_transition_metrics.csv"
    ].copy()
    classification_layer = layer[
        layer["signal"].isin(
            ("feature_post", "feature_norm", "qk_mean", "attn_c2p_conditional")
        )
    ]
    classification_cam = cam[cam["stage"].isin(("patch_cam", "c2p_cam", "final_cam"))]
    classification_control_input = pd.concat(
        [
            classification_layer.assign(
                stage="layer_signal", control_source="layer_signal"
            ),
            classification_cam.rename(columns={"stage": "signal"}).assign(
                layer=-1, control_source="cam_stage"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    products["classification_stratified_results.csv"] = _classification_control(
        classification_control_input,
        args.bootstrap_repeats,
        derived_seed(args.bootstrap_seed, "classification"),
    )
    products["classification_conditioned_classwise_results.csv"] = (
        _classification_conditioned_classwise_control(
            classification_control_input,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "classification-classwise"),
        )
    )
    (
        products["checkpoint_classification_performance.csv"],
        products["raw_final_cam_miou.csv"],
    ) = _paired_checkpoint_evaluations(
        frames["classification"],
        frames["cam_confusion"],
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "checkpoint-evaluation"),
    )
    log(
        "completed failure-pattern, aggregate/class-wise classification-stratified, "
        "matched classification-mAP, and fixed-threshold raw-CAM summaries"
    )

    priority_signals = {
        "feature_post",
        "feature_norm",
        "feature_final_norm",
        "feature_post_relative",
        "feature_post_active_softmax",
        "patch_norm",
        "qk_mean",
        "attn_c2p_conditional",
    }
    products["probe_validity_raw_norm_qk_attn.csv"] = products[
        "layerwise_region_metrics.csv"
    ][lambda value: value["signal"].isin(priority_signals)].copy()
    products["priority_layer_results.csv"] = products["layerwise_region_metrics.csv"][
        lambda value: (
            value["layer"].isin((1, 4, 5, 8, 9, 10, 11, 12))
            | value["signal"].isin(
                (
                    "feature_final_norm",
                    "attn_official_conditional",
                    "attn_mid3_conditional",
                )
            )
        )
        & value["metric"].isin(CORE_REGION_VALUE_COLUMNS)
    ].copy()
    aggregate_attention_names = {
        "attn_official_conditional",
        "attn_mid3_conditional",
    }
    diagnostic_cam_names = {
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
        # Diagnostic single-layer/mid3 maps include the native A_p2p
        # propagation, so their matched native endpoint is final_cam.
        "final_cam",
    }
    products["last_three_aggregation_analysis.csv"] = pd.concat(
        [
            products["layerwise_region_metrics.csv"][
                (
                    (
                        products["layerwise_region_metrics.csv"]["signal"]
                        == "attn_c2p_conditional"
                    )
                    & products["layerwise_region_metrics.csv"]["layer"].isin(
                        [10, 11, 12]
                    )
                )
                | products["layerwise_region_metrics.csv"]["signal"].isin(
                    aggregate_attention_names
                )
            ],
            products["cam_stage_region_metrics.csv"][
                products["cam_stage_region_metrics.csv"]["stage"].isin(
                    diagnostic_cam_names
                )
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    products["classwise_results.csv"] = _primary_classwise_summary(
        layer,
        cam,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "primary-classwise"),
    )
    log(
        "completed full-metric per-model and exact-common-key paired class-wise "
        "summaries"
    )

    paired_parts = [
        _paired_table(
            layer,
            group_cols=("model", "signal", "layer", "rho"),
            key_cols=("image_id", "class_id"),
            value_cols=REGION_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-layer"),
        ).assign(source_table="layer_signal"),
        _paired_table(
            cam,
            group_cols=("model", "stage", "rho"),
            key_cols=("image_id", "class_id"),
            value_cols=REGION_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-cam"),
        ).assign(source_table="cam_stage"),
        _paired_table(
            frames["transition"],
            group_cols=("model", "transition", "layer", "rho", "topk_ratio"),
            key_cols=("image_id", "class_id"),
            value_cols=TRANSITION_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-transition"),
        ).assign(source_table="stage_transition"),
        _paired_table(
            shared_input,
            group_cols=(
                "model",
                "signal",
                "layer_or_stage",
                "new_shared_transition",
                "rho",
                "topk_ratio",
            ),
            key_cols=("image_id", "class_a", "class_b"),
            value_cols=SHARED_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-shared"),
            macro_class=False,
        ).assign(source_table="shared_support"),
        _paired_table(
            shared_endpoints,
            group_cols=("model", *shared_group_cols[1:]),
            key_cols=("image_id", "class_id", "partner_class_id"),
            value_cols=SHARED_FOCAL_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-shared-class-marginals"),
            macro_class=True,
        ).assign(source_table="shared_support_focal_class"),
        _paired_table(
            frames["pair"],
            group_cols=("model", "signal", "layer", "topk_ratio"),
            key_cols=("image_id", "class_a", "class_b"),
            value_cols=PAIR_DIVERSITY_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-pair-diversity"),
            macro_class=False,
        ).assign(source_table="multiclass_map_diversity"),
        _paired_table(
            pair_endpoints,
            group_cols=("model", "signal", "layer", "topk_ratio"),
            key_cols=("image_id", "class_id", "partner_class_id"),
            value_cols=PAIR_DIVERSITY_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-pair-class-marginals"),
            macro_class=True,
        ).assign(source_table="multiclass_map_diversity_focal_class"),
        _paired_table(
            frames["token"],
            group_cols=("model", "layer"),
            key_cols=("image_id", "class_a", "class_b"),
            value_cols=TOKEN_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-token"),
            macro_class=False,
        ).assign(source_table="class_token_map_overlap"),
        _paired_table(
            token_endpoints,
            group_cols=("model", "layer"),
            key_cols=("image_id", "class_id", "partner_class_id"),
            value_cols=TOKEN_VALUE_COLUMNS,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(args.bootstrap_seed, "paired-token-class-marginals"),
            macro_class=True,
        ).assign(source_table="class_token_map_overlap_focal_class"),
    ]
    products["paired_model_deltas.csv"] = pd.concat(
        [frame for frame in paired_parts if not frame.empty],
        ignore_index=True,
        sort=False,
    )
    log("completed common-key paired model summaries")

    products["class_token_similarity_vs_map_overlap.csv"] = _clustered_table(
        frames["token"],
        group_cols=("model", "layer"),
        value_cols=TOKEN_VALUE_COLUMNS,
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "token"),
        macro_class=False,
    )
    products["class_token_map_overlap_association.csv"] = _token_overlap_associations(
        frames["token"],
        repeats=args.bootstrap_repeats,
        seed=derived_seed(args.bootstrap_seed, "token-overlap-association"),
    )
    products["class_token_map_overlap_endpoint_association.csv"] = (
        _token_endpoint_associations(
            token_endpoints,
            repeats=args.bootstrap_repeats,
            seed=derived_seed(
                args.bootstrap_seed, "token-overlap-endpoint-association"
            ),
        )
    )
    log(
        "completed class-token/map overlap summaries and unordered-pair, "
        "focal-class, and equal-class associations"
    )

    counts: dict[str, int] = {}
    output_files: dict[str, dict[str, object]] = {}
    for filename, frame in products.items():
        if frame.empty and len(frame.columns) == 0:
            # Preserve an explicit, readable empty result instead of emitting a
            # zero-byte file that pandas cannot reload.
            frame = pd.DataFrame({"status": ["no_rows"]})
        output_files[filename] = _write_csv(frame, tables_out / filename)
        counts[filename] = len(frame)
        log(f"wrote {filename}: {len(frame)} rows")

    for name, path in external_provenance_paths.items():
        if not path.is_file() or _sha256(path) != external_provenance_sha256[name]:
            raise RuntimeError(
                f"Experiment 2 provenance artifact changed during analysis: {path}"
            )
    provenance_files = {
        "analysis_log": {
            "path": str(log_path.resolve()),
            "sha256": _sha256(log_path),
        },
        **{
            name: {
                "path": str(path.resolve()),
                "sha256": external_provenance_sha256[name],
            }
            for name, path in external_provenance_paths.items()
        },
    }

    metadata = {
        "status": "complete",
        "analysis": "experiment2_semantic_ownership",
        "command": command,
        "canonical_dir": str(canonical),
        "canonical_verification": canonical_verification,
        "bootstrap": {
            "unit": "image_id cluster",
            "paired_delta": "mctformer_plus_minus_mctformer",
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "ci": "95% percentile",
            "patches_or_image_class_pairs_treated_independent": False,
            "same_image_draw_reused_within_analysis_family": True,
            "finite_denominators_reported_per_metric": True,
            "valid_bootstrap_replicates_reported_per_metric": True,
        },
        "failure_pattern_definition": {
            "threshold": "BG-TailEnrich@10% > 1",
            "flags_are_nonexclusive": True,
            "type_a": "feature high, attention not high, final CAM not high",
            "type_b": "feature not high, attention high",
            "type_c": "feature and attention not high, patch CAM high",
            "type_d": "c2p CAM not high, final CAM high",
            "type_e": "feature, attention, and final CAM high",
        },
        "class_pair_macro_policy": (
            "each unordered pair is expanded to both focal endpoints; focal-class "
            "marginals, equal-class macro means, class-wise results, and label-count "
            "strata use image-cluster resampling"
        ),
        "classification_control_policy": {
            "native_statuses": list(CLASSIFICATION_STATUSES),
            "focal_union": (
                "either_negative means the focal class is not both_positive; "
                "reported for image-class and endpoint-expanded pair analyses"
            ),
            "unordered_pair_partition": list(PAIR_CLASSIFICATION_SUBSETS),
            "unordered_pair_definition": (
                "both_classes_both_positive requires both endpoints to be "
                "both_positive; either_class_negative is its complement"
            ),
            "model_comparison": (
                "classification-conditioned summaries are per-model because status "
                "is model-specific; no cross-model conditioned delta is reported"
            ),
            "bootstrap_unit": "image_id cluster",
        },
        "aggregation_coverage_policy": {
            "controls": [
                "classification-conditioned layer and CAM",
                "target-visible layer and CAM",
                "QK head region metrics",
            ],
            "per_model": "within-class summaries for every available VOC class",
            "classification_conditioned_delta": (
                "not applicable because native classification status is model-specific"
            ),
            "target_visible_and_qk_delta": (
                "MCTformer+ minus MCTformer on exact common image_id/class_id keys "
                "within metric identity"
            ),
            "label_strata": [
                "all",
                "single_label",
                "exactly_2_labels",
                "3plus_labels",
            ],
            "bootstrap_unit": "image_id cluster",
        },
        "patch_norm_joint_control_policy": {
            "scope": "post-block cosine and same-layer post-block patch-token L2 norm",
            "low_norm": "at or below the within-image valid-patch norm median",
            "high_norm": "at or above the within-image valid-patch norm q75",
            "top_tail": "stable top 10% feature-post cosine among non-void patches",
            "paired_delta": (
                "MCTformer+ minus MCTformer on exact common image_id/class_id keys"
            ),
            "claim_boundary": (
                "norm-concentration diagnostic; does not establish a register token "
                "or semantic shortcut"
            ),
        },
        "primary_classwise_policy": (
            "full REGION_VALUE_COLUMNS for primary layer/CAM signals; per-model "
            "within-class summaries and MCTformer+ minus MCTformer deltas on exact "
            "common image_id/class_id keys; image-cluster bootstrap with all four "
            "label-count strata"
        ),
        "class_pair_classwise_delta_policy": (
            "endpoint-expanded focal class; exact common image_id/class_id/"
            "partner_class_id keys within each signal/layer identity; "
            "MCTformer+ minus MCTformer; image-cluster paired bootstrap"
        ),
        "stage_transition_classwise_policy": (
            "per-model and MCTformer+ minus MCTformer survival/introduction/"
            "removal summaries by class; paired deltas use exact common "
            "image_id/class_id keys within transition/layer/rho/top-k identity; "
            "all confidence intervals resample image_id clusters"
        ),
        "token_map_association_policy": {
            "retained_estimands": [
                "unordered positive-pair micro Pearson",
                "per-image pair-mean Pearson sensitivity",
            ],
            "endpoint_estimands": [
                "within-focal-class Pearson",
                "equal-class mean of within-focal-class Pearson",
            ],
            "macro_bootstrap": (
                "sample whole images, recompute every focal-class correlation, "
                "then average finite class correlations equally"
            ),
            "degenerate_class_policy": (
                "exclude undefined within-class correlations per metric/replicate; "
                "report finite and total class denominators"
            ),
        },
        "runtime_source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("bootstrap_experiment2.py").resolve(),
                Path(__file__).with_name("evaluation_metrics.py").resolve(),
                Path(__file__).with_name("pairwise_class_analysis.py").resolve(),
            )
        },
        "matched_checkpoint_evaluation": {
            "classification": (
                "all 20 VOC foreground logits; class-token and native patch-head "
                "AP/mAP; identical image-cluster bootstrap draws across models"
            ),
            "raw_final_cam": (
                "native final CAM, single-scale 448 deterministic crop, bilinear "
                "upsampling with align_corners=False, per-active-class min-max "
                "normalization, GT-positive class gating, fixed background "
                "threshold 0.45, void ignored"
            ),
            "scope": (
                "matched transformed-crop diagnostic, not downstream segmentation "
                "training or full-image multi-scale evaluation"
            ),
        },
        "canonical_rows": {name: len(frame) for name, frame in frames.items()},
        "output_rows": counts,
        "output_files": output_files,
        "provenance_files": provenance_files,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
