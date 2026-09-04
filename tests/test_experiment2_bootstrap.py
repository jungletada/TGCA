import numpy as np
import pandas as pd
import pytest

from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import (
    _finite_interval,
    _finite_intervals,
    add_label_stratum,
    paired_model_frame,
    summarize_clustered,
    summarize_clustered_pair_correlations,
    summarize_image_mean_correlations,
)


def test_vectorized_finite_intervals_exactly_match_scalar_definition():
    values = np.asarray(
        [
            [0.0, np.nan, np.nan, -np.inf, 7.0],
            [1.0, 5.0, np.inf, 3.0, 7.0],
            [2.0, np.nan, -np.inf, np.inf, 7.0],
            [100.0, 9.0, np.nan, 4.0, 7.0],
        ],
        dtype=np.float64,
    )

    lows, highs, valid_counts = _finite_intervals(values)
    scalar = [_finite_interval(values[:, index]) for index in range(values.shape[1])]

    np.testing.assert_array_equal(valid_counts, [4, 2, 0, 2, 4])
    for index, (expected_low, expected_high, expected_count) in enumerate(scalar):
        if expected_count:
            assert lows[index] == expected_low
            assert highs[index] == expected_high
        else:
            assert np.isnan(lows[index])
            assert np.isnan(highs[index])
        assert valid_counts[index] == expected_count


def test_vectorized_finite_intervals_rejects_non_matrix_input():
    with pytest.raises(ValueError, match=r"\[repeat, metric\]"):
        _finite_intervals(np.asarray([1.0, 2.0]))


def test_paired_model_frame_uses_only_common_keys_and_plus_minus_base():
    frame = pd.DataFrame(
        [
            {"model": "mctformer", "image_id": "a", "class_id": 1, "x": 1.0},
            {"model": "mctformer_plus", "image_id": "a", "class_id": 1, "x": 3.0},
            {"model": "mctformer", "image_id": "b", "class_id": 1, "x": 9.0},
        ]
    )
    paired = paired_model_frame(
        frame, key_cols=("image_id", "class_id"), value_cols=("x",)
    )
    assert len(paired) == 1
    assert paired.iloc[0]["x"] == 2.0


def test_paired_model_frame_converts_boolean_metrics_to_numeric_deltas():
    frame = pd.DataFrame(
        [
            {"model": "mctformer", "image_id": "a", "class_id": 1, "target_hit": False},
            {
                "model": "mctformer_plus",
                "image_id": "a",
                "class_id": 1,
                "target_hit": True,
            },
        ]
    )
    paired = paired_model_frame(
        frame,
        key_cols=("image_id", "class_id"),
        value_cols=("target_hit",),
    )
    assert paired.iloc[0]["target_hit"] == 1.0


def test_label_strata_are_pre_registered():
    frame = add_label_stratum(pd.DataFrame({"num_positive_classes": [1, 2, 3, 7]}))
    assert frame["label_stratum"].tolist() == [
        "single_label",
        "exactly_2_labels",
        "3plus_labels",
        "3plus_labels",
    ]


def test_cluster_summary_resamples_images_not_rows():
    # Image a has two rows but counts as one sampled cluster; this test checks
    # the reported sampling unit and deterministic point estimate.
    frame = pd.DataFrame(
        [
            {"image_id": "a", "class_id": 0, "x": 0.0},
            {"image_id": "a", "class_id": 1, "x": 2.0},
            {"image_id": "b", "class_id": 0, "x": 4.0},
        ]
    )
    rows = summarize_clustered(
        frame,
        value_cols=("x",),
        identity={"group": "g"},
        repeats=100,
        seed=123,
        include_macro_class=False,
    )
    assert len(rows) == 1
    assert rows[0]["estimate"] == pytest.approx(2.0)
    assert rows[0]["num_images"] == 2
    assert rows[0]["num_rows"] == 3


def test_cluster_summary_reports_metric_specific_finite_denominators():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "class_id": 0, "x": 1.0},
            {"image_id": "a", "class_id": 1, "x": float("nan")},
            {"image_id": "b", "class_id": 0, "x": float("nan")},
        ]
    )
    row = summarize_clustered(
        frame,
        value_cols=("x",),
        identity={"group": "finite"},
        repeats=200,
        seed=91,
        include_macro_class=False,
    )[0]
    assert row["num_images"] == 1
    assert row["num_images_total"] == 2
    assert row["num_rows"] == 1
    assert row["num_rows_total"] == 3
    assert 0 < row["bootstrap_valid_repeats"] < 200
    assert row["bootstrap_seed"] != row["bootstrap_base_seed"]


def test_image_mean_correlation_collapses_class_pairs_before_bootstrap():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "x": 0.0, "y": 0.0},
            {"image_id": "a", "x": 2.0, "y": 2.0},
            {"image_id": "b", "x": 2.0, "y": 2.0},
            {"image_id": "c", "x": 3.0, "y": 3.0},
        ]
    )
    rows = summarize_image_mean_correlations(
        frame,
        x_col="x",
        y_cols=("y",),
        identity={"layer": 12},
        repeats=100,
        seed=7,
    )
    assert len(rows) == 1
    assert rows[0]["estimate"] == pytest.approx(1.0)
    assert rows[0]["num_images"] == 3
    assert rows[0]["association_unit"].startswith("per-image mean")


def test_pair_correlation_keeps_pair_estimand_but_bootstraps_images():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "x": 0.0, "y": 0.0},
            {"image_id": "a", "x": 1.0, "y": 1.0},
            {"image_id": "b", "x": 2.0, "y": 2.0},
            {"image_id": "c", "x": 3.0, "y": 3.0},
        ]
    )
    rows = summarize_clustered_pair_correlations(
        frame,
        x_col="x",
        y_cols=("y",),
        identity={"layer": 12},
        repeats=100,
        seed=7,
    )
    assert len(rows) == 1
    assert rows[0]["estimate"] == pytest.approx(1.0)
    assert rows[0]["aggregation"] == "micro_pair_pearson"
    assert rows[0]["num_images"] == 3
    assert rows[0]["num_rows"] == 4
    assert "image-cluster" in rows[0]["association_unit"]
