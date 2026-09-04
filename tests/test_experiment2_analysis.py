import pandas as pd

from analysis.lazy_assignment.experiment2.analyze_experiment2 import (
    _class_pair_classwise_summary,
    _classification_control,
    _classification_stratified_summary,
    _clustered_table,
    _failure_pattern_frame,
    _paired_table,
    _stage_transition_classwise_summary,
    _token_endpoint_associations,
)
from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import (
    summarize_clustered_macro_class_correlations,
)


def _frame():
    rows = []
    for model, shift in (("mctformer", 0.0), ("mctformer_plus", 0.2)):
        for image, count in (("a", 1), ("b", 2), ("c", 3)):
            rows.append(
                {
                    "model": model,
                    "image_id": image,
                    "class_id": 0,
                    "num_positive_classes": count,
                    "signal": "feature_post",
                    "layer": 12,
                    "rho": 0.5,
                    "target_hit": shift + (image != "a"),
                }
            )
    return pd.DataFrame(rows)


def test_clustered_summary_contains_all_requested_label_strata():
    result = _clustered_table(
        _frame(),
        group_cols=("model", "signal", "layer", "rho"),
        value_cols=("target_hit",),
        repeats=50,
        seed=7,
    )
    assert set(result["label_stratum"]) == {
        "all",
        "single_label",
        "exactly_2_labels",
        "3plus_labels",
    }
    assert set(result["aggregation"]) == {"micro", "macro_class"}


def test_paired_summary_is_plus_minus_base_and_image_clustered():
    result = _paired_table(
        _frame(),
        group_cols=("model", "signal", "layer", "rho"),
        key_cols=("image_id", "class_id"),
        value_cols=("target_hit",),
        repeats=100,
        seed=7,
    )
    row = result[
        (result["label_stratum"] == "all") & (result["aggregation"] == "micro")
    ].iloc[0]
    assert abs(row["estimate"] - 0.2) < 1e-12
    assert row["num_images"] == 3
    assert row["delta"] == "mctformer_plus_minus_mctformer"


def test_failure_patterns_are_explicit_nonexclusive_full_tuple_flags():
    base = {
        "model": "mctformer_plus",
        "image_id": "a",
        "class_id": 0,
        "class_name": "aeroplane",
        "num_positive_classes": 1,
        "classification_status": "both_positive",
        "rho": 0.5,
    }
    layer = pd.DataFrame(
        [
            {**base, "layer": 12, "signal": "feature_post", "bg_tail_enrich_10": 1.4},
            {
                **base,
                "layer": 12,
                "signal": "attn_c2p_conditional",
                "bg_tail_enrich_10": 1.3,
            },
        ]
    )
    cam = pd.DataFrame(
        [
            {**base, "stage": "patch_cam", "bg_tail_enrich_10": 1.2},
            {**base, "stage": "c2p_cam", "bg_tail_enrich_10": 0.8},
            {**base, "stage": "final_cam", "bg_tail_enrich_10": 1.1},
        ]
    )
    result = _failure_pattern_frame(layer, cam).iloc[0]
    assert bool(result["type_e_full_pipeline"])
    assert bool(result["type_d_propagation_amplification"])
    assert result["num_active_patterns"] == 2


def test_single_class_control_adds_either_negative_union():
    statuses = (
        "both_positive",
        "class_only_positive",
        "patch_only_positive",
        "neither_positive",
    )
    frame = pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": f"image_{index}",
                "class_id": 0,
                "num_positive_classes": 1,
                "signal": "feature_post",
                "layer": 12,
                "rho": 0.5,
                "classification_status": status,
                "target_hit": float(index == 0),
            }
            for index, status in enumerate(statuses)
        ]
    )
    result = _classification_control(frame, repeats=30, seed=9)
    assert set(result["classification_subset"]) == {*statuses, "either_negative"}
    either = result[
        (result["classification_subset"] == "either_negative")
        & (result["label_stratum"] == "all")
        & (result["aggregation"] == "micro")
        & (result["metric"] == "target_hit")
    ].iloc[0]
    assert either["estimate"] == 0.0
    assert either["num_images"] == 3
    assert either["num_rows"] == 3
    assert either["classification_scope"] == "image_class"


def test_pair_classification_controls_cover_focal_and_joint_partitions():
    statuses = (
        "both_positive",
        "class_only_positive",
        "patch_only_positive",
        "neither_positive",
    )
    focal = pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": f"focal_{index}",
                "class_id": index,
                "num_positive_classes": 2 if index < 2 else 3,
                "signal": "feature_post",
                "layer": 12,
                "topk_ratio": 0.1,
                "classification_status": status,
                "topk_jaccard": float(index),
            }
            for index, status in enumerate(statuses)
        ]
    )
    focal_result = _classification_stratified_summary(
        focal,
        scope="focal_endpoint",
        group_cols=("model", "signal", "layer", "topk_ratio"),
        value_cols=("topk_jaccard",),
        repeats=20,
        seed=11,
        macro_class=True,
    )
    assert set(focal_result["classification_subset"]) == {
        *statuses,
        "either_negative",
    }
    assert set(focal_result["label_stratum"]) == {
        "all",
        "exactly_2_labels",
        "3plus_labels",
    }
    assert "single_label" not in set(focal_result["label_stratum"])

    unordered = pd.DataFrame(
        [
            {
                "model": "mctformer_plus",
                "image_id": "pair_both",
                "num_positive_classes": 2,
                "signal": "feature_post",
                "layer": 12,
                "topk_ratio": 0.1,
                "classification_status_a": "both_positive",
                "classification_status_b": "both_positive",
                "topk_jaccard": 0.2,
            },
            {
                "model": "mctformer_plus",
                "image_id": "pair_negative",
                "num_positive_classes": 3,
                "signal": "feature_post",
                "layer": 12,
                "topk_ratio": 0.1,
                "classification_status_a": "both_positive",
                "classification_status_b": "class_only_positive",
                "topk_jaccard": 0.8,
            },
        ]
    )
    joint_result = _classification_stratified_summary(
        unordered,
        scope="unordered_pair",
        group_cols=("model", "signal", "layer", "topk_ratio"),
        value_cols=("topk_jaccard",),
        repeats=20,
        seed=12,
        macro_class=False,
    )
    assert set(joint_result["classification_subset"]) == {
        "both_classes_both_positive",
        "either_class_negative",
    }
    assert set(joint_result["label_stratum"]) == {
        "all",
        "exactly_2_labels",
        "3plus_labels",
    }


def test_pair_classwise_delta_is_plus_minus_base_common_key_and_image_clustered():
    rows = []
    common = {
        "mctformer": ((1, 0.1), (2, 0.2)),
        "mctformer_plus": ((1, 0.4), (2, 0.5)),
    }
    for model, partner_values in common.items():
        for partner, value in partner_values:
            rows.append(
                {
                    "model": model,
                    "image_id": "common_image",
                    "class_id": 0,
                    "partner_class_id": partner,
                    "num_positive_classes": 2,
                    "signal": "feature_post",
                    "layer": 12,
                    "topk_ratio": 0.1,
                    "topk_jaccard": value,
                }
            )
    rows.extend(
        [
            {
                "model": "mctformer",
                "image_id": "baseline_only",
                "class_id": 0,
                "partner_class_id": 1,
                "num_positive_classes": 2,
                "signal": "feature_post",
                "layer": 12,
                "topk_ratio": 0.1,
                "topk_jaccard": 9.0,
            },
            {
                "model": "mctformer_plus",
                "image_id": "plus_only",
                "class_id": 0,
                "partner_class_id": 1,
                "num_positive_classes": 2,
                "signal": "feature_post",
                "layer": 12,
                "topk_ratio": 0.1,
                "topk_jaccard": -9.0,
            },
        ]
    )
    result = _class_pair_classwise_summary(
        pd.DataFrame(rows),
        source_table="multiclass_map_diversity",
        group_cols=("model", "signal", "layer", "topk_ratio"),
        key_cols=("image_id", "class_id", "partner_class_id"),
        value_cols=("topk_jaccard",),
        repeats=30,
        seed=13,
    )
    delta = result[
        (result["model_or_delta"] == "mctformer_plus_minus_mctformer")
        & (result["label_stratum"] == "all")
        & (result["aggregation"] == "micro")
        & (result["metric"] == "topk_jaccard")
    ].iloc[0]
    assert abs(delta["estimate"] - 0.3) < 1e-12
    assert delta["num_rows"] == 2
    assert delta["num_images"] == 1
    assert delta["delta"] == "mctformer_plus_minus_mctformer"


def test_transition_classwise_per_model_and_common_key_paired_strata():
    rows = []
    for model, shift in (("mctformer", 0.0), ("mctformer_plus", 0.2)):
        for image_id, count, baseline in (
            ("single", 1, 0.1),
            ("two", 2, 0.2),
            ("three", 3, 0.3),
        ):
            rows.append(
                {
                    "model": model,
                    "image_id": image_id,
                    "class_id": 0,
                    "num_positive_classes": count,
                    "transition": "feature_post_to_attn",
                    "layer": 12,
                    "rho": 0.5,
                    "topk_ratio": 0.1,
                    "survive_background": baseline + shift,
                    "introduced_background_fraction": 0.4 + shift,
                    "removed_background_fraction": 0.5 + shift,
                }
            )
    # These extreme rows occur in only one model and must not enter the delta.
    rows.extend(
        [
            {
                "model": "mctformer",
                "image_id": "baseline_only",
                "class_id": 0,
                "num_positive_classes": 1,
                "transition": "feature_post_to_attn",
                "layer": 12,
                "rho": 0.5,
                "topk_ratio": 0.1,
                "survive_background": 9.0,
                "introduced_background_fraction": 9.0,
                "removed_background_fraction": 9.0,
            },
            {
                "model": "mctformer_plus",
                "image_id": "plus_only",
                "class_id": 0,
                "num_positive_classes": 1,
                "transition": "feature_post_to_attn",
                "layer": 12,
                "rho": 0.5,
                "topk_ratio": 0.1,
                "survive_background": -9.0,
                "introduced_background_fraction": -9.0,
                "removed_background_fraction": -9.0,
            },
        ]
    )
    result = _stage_transition_classwise_summary(
        pd.DataFrame(rows), repeats=40, seed=16
    )
    assert set(result["model_or_delta"].dropna()) == {
        "mctformer",
        "mctformer_plus",
        "mctformer_plus_minus_mctformer",
    }
    delta = result[
        (result["model_or_delta"] == "mctformer_plus_minus_mctformer")
        & (result["metric"] == "survive_background")
        & (result["aggregation"] == "micro")
    ]
    assert set(delta["label_stratum"]) == {
        "all",
        "single_label",
        "exactly_2_labels",
        "3plus_labels",
    }
    all_row = delta[delta["label_stratum"] == "all"].iloc[0]
    assert abs(all_row["estimate"] - 0.2) < 1e-12
    assert all_row["num_rows"] == 3
    assert all_row["num_images"] == 3
    assert all_row["delta"] == "mctformer_plus_minus_mctformer"
    assert all_row["source_table"] == "stage_transition"


def test_macro_class_correlation_equal_weights_and_omits_degenerate_class():
    rows = []
    for image_index, x in enumerate((0.0, 1.0, 2.0)):
        for class_id, y in ((0, x), (1, 2.0 - x), (2, 1.0)):
            rows.append(
                {
                    "image_id": f"image_{image_index}",
                    "class_id": class_id,
                    "x": x,
                    "y": y,
                }
            )
    result = summarize_clustered_macro_class_correlations(
        pd.DataFrame(rows),
        x_col="x",
        y_cols=("y",),
        identity={"model": "mctformer_plus", "layer": 12},
        repeats=100,
        seed=14,
    )
    row = result[0]
    assert abs(row["estimate"]) < 1e-12
    assert row["num_classes"] == 2
    assert row["num_classes_total"] == 3
    assert row["num_images"] == 3
    assert row["num_rows"] == 9
    assert 0 < row["bootstrap_valid_repeats"] <= 100


def test_endpoint_associations_are_order_invariant_clustered_and_stratified():
    rows = []
    for stratum_prefix, count in (("two", 2), ("three", 3)):
        for image_index, x in enumerate((0.0, 1.0, 2.0)):
            image_id = f"{stratum_prefix}_{image_index}"
            # Two partners for focal class 0 deliberately share one image
            # cluster; they must not become independent bootstrap units.
            for partner in (1, 2):
                rows.append(
                    {
                        "model": "mctformer_plus",
                        "image_id": image_id,
                        "class_id": 0,
                        "partner_class_id": partner,
                        "num_positive_classes": count,
                        "layer": 12,
                        "class_token_cosine": x,
                        "feature_post_top10_jaccard": x,
                    }
                )
            rows.append(
                {
                    "model": "mctformer_plus",
                    "image_id": image_id,
                    "class_id": 1,
                    "partner_class_id": 0,
                    "num_positive_classes": count,
                    "layer": 12,
                    "class_token_cosine": x,
                    "feature_post_top10_jaccard": 2.0 - x,
                }
            )
    frame = pd.DataFrame(rows)
    forward = _token_endpoint_associations(frame, repeats=80, seed=15)
    reverse = _token_endpoint_associations(
        frame.iloc[::-1].reset_index(drop=True), repeats=80, seed=15
    )
    assert set(forward["label_stratum"]) == {
        "all",
        "exactly_2_labels",
        "3plus_labels",
    }
    assert "single_label" not in set(forward["label_stratum"])
    class_zero = forward[
        (forward["aggregation"] == "classwise_pair_pearson")
        & (forward["class_id"] == 0)
        & (forward["label_stratum"] == "all")
    ].iloc[0]
    assert class_zero["num_images"] == 6
    assert class_zero["num_rows"] == 12
    keys = ["aggregation", "label_stratum", "metric", "class_id"]
    left = forward.sort_values(keys, na_position="last").reset_index(drop=True)
    right = reverse.sort_values(keys, na_position="last").reset_index(drop=True)
    pd.testing.assert_series_equal(left["estimate"], right["estimate"])
    pd.testing.assert_series_equal(left["ci_low"], right["ci_low"])
    pd.testing.assert_series_equal(left["ci_high"], right["ci_high"])
