import pandas as pd

from analysis.lazy_assignment.experiment2.analyze_experiment2 import (
    _classification_conditioned_classwise_control,
    _primary_classwise_summary,
    _single_classwise_coverage_summary,
)


def _coverage_frame() -> pd.DataFrame:
    rows = []
    common = {
        "mctformer": (("single", 1, 0.1), ("two", 2, 0.2)),
        "mctformer_plus": (("single", 1, 0.4), ("two", 2, 0.5)),
    }
    for model, values in common.items():
        for image_id, label_count, value in values:
            rows.append(
                {
                    "model": model,
                    "image_id": image_id,
                    "class_id": 0,
                    "num_positive_classes": label_count,
                    "signal": "qk_mean",
                    "layer": 12,
                    "rho": 0.5,
                    "qk_head0_bg_mean": value,
                }
            )
    rows.extend(
        [
            {
                "model": "mctformer",
                "image_id": "baseline_only",
                "class_id": 0,
                "num_positive_classes": 3,
                "signal": "qk_mean",
                "layer": 12,
                "rho": 0.5,
                "qk_head0_bg_mean": 9.0,
            },
            {
                "model": "mctformer_plus",
                "image_id": "plus_only",
                "class_id": 0,
                "num_positive_classes": 3,
                "signal": "qk_mean",
                "layer": 12,
                "rho": 0.5,
                "qk_head0_bg_mean": -9.0,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_single_classwise_coverage_pairs_only_exact_common_image_class_keys():
    result = _single_classwise_coverage_summary(
        _coverage_frame(),
        source_table="qk_head_control",
        group_cols=("model", "signal", "layer", "rho"),
        value_cols=("qk_head0_bg_mean",),
        repeats=40,
        seed=101,
        paired_comparison=True,
    )

    delta = result[
        (result["model_or_delta"] == "mctformer_plus_minus_mctformer")
        & (result["label_stratum"] == "all")
        & (result["metric"] == "qk_head0_bg_mean")
    ].iloc[0]
    assert abs(delta["estimate"] - 0.3) < 1e-12
    assert delta["num_rows"] == 2
    assert delta["num_images"] == 2
    assert delta["class_id"] == 0
    assert delta["bootstrap_repeats"] == 40
    assert delta["paired_key_columns"] == "image_id,class_id"
    assert delta["comparison_policy"] == "exact_common_key_paired"
    per_model = result[result["model_or_delta"] == "mctformer"]
    assert set(per_model["comparison_policy"]) == {"per_model_not_paired"}
    assert set(result["label_stratum"]) == {
        "all",
        "single_label",
        "exactly_2_labels",
        "3plus_labels",
    }


def test_classification_conditioned_classwise_is_per_model_structural_na():
    rows = []
    statuses = {
        "mctformer": ("both_positive", "both_positive"),
        "mctformer_plus": ("class_only_positive", "both_positive"),
    }
    for model, model_statuses in statuses.items():
        for index, status in enumerate(model_statuses):
            rows.append(
                {
                    "model": model,
                    "image_id": f"image_{index}",
                    "class_id": 4,
                    "num_positive_classes": index + 1,
                    "control_source": "layer_signal",
                    "signal": "feature_post",
                    "layer": 12,
                    "rho": 0.5,
                    "classification_status": status,
                    "target_hit": float(index),
                }
            )

    result = _classification_conditioned_classwise_control(
        pd.DataFrame(rows), repeats=30, seed=202
    )

    assert set(result["model_or_delta"]) == {"mctformer", "mctformer_plus"}
    assert set(result["class_id"]) == {4}
    assert set(result["classification_subset"]) == {
        "both_positive",
        "class_only_positive",
        "either_negative",
    }
    assert set(result["comparison_policy"]) == {
        "not_applicable_model_specific_conditioning"
    }
    assert set(result["aggregation_scope"]) == {"within_class"}
    assert "delta" not in result.columns


def test_classwise_coverage_rejects_pairing_without_model_identity():
    try:
        _single_classwise_coverage_summary(
            _coverage_frame(),
            source_table="bad",
            group_cols=("signal", "layer", "rho"),
            value_cols=("qk_head0_bg_mean",),
            repeats=10,
            seed=303,
            paired_comparison=True,
        )
    except ValueError as error:
        assert "group_cols must contain model" in str(error)
    else:
        raise AssertionError("missing model identity must be rejected")


def test_primary_classwise_has_full_topk_metrics_and_exact_paired_keys():
    layer = _coverage_frame().rename(
        columns={"qk_head0_bg_mean": "target_top05_fraction"}
    )
    layer["bg_tail_enrich_20"] = layer["target_top05_fraction"] + 1.0
    cam = layer.rename(columns={"signal": "stage"}).copy()
    cam["stage"] = "final_cam"

    result = _primary_classwise_summary(layer, cam, repeats=40, seed=404)
    assert {"target_top05_fraction", "bg_tail_enrich_20"}.issubset(
        set(result["metric"])
    )
    assert set(result["source_table"]) == {"layer_signal", "cam_stage"}
    paired = result[
        (result["model_or_delta"] == "mctformer_plus_minus_mctformer")
        & (result["source_table"] == "layer_signal")
        & (result["label_stratum"] == "all")
        & (result["metric"] == "target_top05_fraction")
    ].iloc[0]
    assert abs(paired["estimate"] - 0.3) < 1e-12
    assert paired["num_images"] == 2
    assert paired["num_rows"] == 2
    assert paired["paired_key_columns"] == "image_id,class_id"
    assert paired["comparison_policy"] == "exact_common_key_paired"
