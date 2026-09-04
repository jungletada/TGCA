from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (
    DEFAULT_BOOTSTRAP_REPEATS,
    image_multinomial_draws,
    multiplicities_for_rows,
    paired_clustered_auc_ap_summary,
    paired_clustered_mean_summary,
    paired_confusion_metric_summary,
    summarize_clustered_means,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    cam_metrics_from_confusion,
)


def _record(frame, *, aggregation, metric, series=None):
    selected = frame[
        (frame["aggregation"] == aggregation) & (frame["metric"] == metric)
    ]
    if series is not None:
        selected = selected[selected["series"] == series]
    assert len(selected) == 1
    return selected.iloc[0]


def test_default_is_exactly_5000_repeats():
    assert DEFAULT_BOOTSTRAP_REPEATS == 5000


def test_whole_image_draws_are_deterministic_order_independent_multinomials():
    first = image_multinomial_draws(["b", "a", "b", "c"], repeats=64, seed=71)
    second = image_multinomial_draws(["c", "b", "a"], repeats=64, seed=71)
    assert first.image_ids == ("a", "b", "c")
    np.testing.assert_array_equal(first.multiplicities, second.multiplicities)
    np.testing.assert_array_equal(first.multiplicities.sum(axis=1), 3)
    assert not first.multiplicities.flags.writeable


def test_duplicate_rows_from_one_image_always_receive_identical_weight():
    draws = image_multinomial_draws(["a", "b"], repeats=80, seed=9)
    weights = multiplicities_for_rows(draws, ["a", "a", "b", "a"])
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, 0], weights[:, 3])
    assert np.any(weights[:, 0] != weights[:, 2])


def test_row_micro_and_macro_class_means_use_image_clusters():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "class_id": 0, "value": 0.0},
            {"image_id": "a", "class_id": 1, "value": 10.0},
            {"image_id": "b", "class_id": 0, "value": 2.0},
        ]
    )
    draws = image_multinomial_draws(frame.image_id.tolist(), repeats=100, seed=13)
    summary = summarize_clustered_means(frame, value_cols=("value",), draws=draws)
    micro = _record(summary, aggregation="micro", metric="value")
    macro = _record(summary, aggregation="macro_class", metric="value")
    assert micro["estimate"] == pytest.approx(4.0)
    assert macro["estimate"] == pytest.approx(5.5)
    assert micro["num_images"] == 2
    assert micro["num_rows"] == 3
    assert micro["bootstrap_unit"] == "image"


def test_finite_bootstrap_denominator_excludes_draws_without_finite_image():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "class_id": 0, "value": 1.0},
            {"image_id": "b", "class_id": 0, "value": np.nan},
        ]
    )
    summary = summarize_clustered_means(
        frame,
        value_cols=("value",),
        repeats=400,
        seed=111,
        include_macro_class=False,
    )
    row = summary.iloc[0]
    assert row["num_images"] == 1
    assert row["num_images_total"] == 2
    assert 0 < row["bootstrap_valid_repeats"] < 400
    assert row["bootstrap_valid_fraction"] == pytest.approx(
        row["bootstrap_valid_repeats"] / 400
    )


def test_paired_row_summary_reuses_draws_and_has_exact_zero_delta():
    records = []
    for system in ("B0", "B1"):
        records.extend(
            [
                {"system": system, "image_id": "a", "class_id": 0, "x": 1.0},
                {"system": system, "image_id": "a", "class_id": 1, "x": 3.0},
                {"system": system, "image_id": "b", "class_id": 0, "x": 8.0},
            ]
        )
    summary = paired_clustered_mean_summary(
        pd.DataFrame(records),
        system_col="system",
        baseline="B0",
        comparison="B1",
        key_cols=("image_id", "class_id"),
        value_cols=("x",),
        repeats=200,
        seed=17,
    )
    delta = summary[summary["paired_delta"]]
    assert set(delta["aggregation"]) == {"micro", "macro_class"}
    assert np.all(delta["estimate"] == 0.0)
    assert np.all(delta["ci_low"] == 0.0)
    assert np.all(delta["ci_high"] == 0.0)
    assert np.all(delta["bootstrap_valid_repeats"] == 200)


def _confusions():
    first = np.zeros((21, 21), dtype=np.int64)
    first[0, 0] = 20
    first[0, 1] = 2
    first[1, 0] = 4
    first[1, 1] = 10
    second = np.zeros((21, 21), dtype=np.int64)
    second[0, 0] = 15
    second[0, 1] = 3
    second[1, 0] = 2
    second[1, 1] = 12
    return np.stack((first, second))


def test_paired_confusion_uses_nonlinear_aggregate_metrics_and_zero_delta():
    confusions = _confusions()
    summary = paired_confusion_metric_summary(
        ["b", "a"],
        confusions,
        confusions.copy(),
        baseline_name="B0",
        comparison_name="B1",
        repeats=160,
        seed=29,
    )
    expected = cam_metrics_from_confusion(confusions.sum(axis=0))
    for metric in (
        "mean_iou",
        "binary_foreground_precision",
        "binary_foreground_recall",
        "semantic_correct_foreground_precision",
        "semantic_correct_foreground_recall",
    ):
        baseline = summary[(summary.series == "B0") & (summary.metric == metric)].iloc[
            0
        ]
        assert baseline.estimate == pytest.approx(expected[metric])
        delta = summary[
            (summary.series == "B1_minus_B0") & (summary.metric == metric)
        ].iloc[0]
        assert delta.estimate == pytest.approx(0.0)
        assert delta.ci_low == pytest.approx(0.0)
        assert delta.ci_high == pytest.approx(0.0)
        assert delta.bootstrap_valid_repeats == 160


def _auc_inputs(duplicate_a: int = 1):
    base_rows = [
        ("a", 0, 1, 0.9, 0.8),
        ("a", 0, 0, 0.1, 0.2),
        ("a", 0, 1, 0.7, 0.6),
        ("a", 0, 0, 0.3, 0.4),
    ]
    rows = base_rows * duplicate_a + [
        ("b", 0, 1, 0.2, 0.2),
        ("b", 0, 0, 0.8, 0.8),
        ("b", 0, 1, 0.4, 0.4),
        ("b", 0, 0, 0.6, 0.6),
        ("b", 1, 1, 0.9, 0.9),
        ("b", 1, 0, 0.1, 0.1),
    ]
    columns = tuple(zip(*rows))
    return columns


def test_clustered_auc_ap_has_exact_paired_zero_delta():
    image_ids, class_ids, labels, scores, _ = _auc_inputs()
    summary = paired_clustered_auc_ap_summary(
        image_ids,
        class_ids,
        labels,
        scores,
        scores,
        baseline_name="raw",
        comparison_name="copy",
        repeats=180,
        seed=37,
    )
    delta = summary[summary["paired_delta"]]
    assert set(delta.metric) == {"auroc", "average_precision"}
    assert set(delta.aggregation) == {"micro", "macro_class"}
    assert np.all(delta.estimate == 0.0)
    assert np.all(delta.ci_low == 0.0)
    assert np.all(delta.ci_high == 0.0)
    assert np.all(delta.bootstrap_valid_repeats == 180)


def test_uniform_patch_duplication_within_one_image_does_not_create_clusters():
    ordinary = _auc_inputs(duplicate_a=1)
    duplicated = _auc_inputs(duplicate_a=7)
    first = paired_clustered_auc_ap_summary(
        *ordinary,
        baseline_name="raw",
        comparison_name="other",
        repeats=220,
        seed=43,
    ).sort_values(["metric", "aggregation", "series"])
    second = paired_clustered_auc_ap_summary(
        *duplicated,
        baseline_name="raw",
        comparison_name="other",
        repeats=220,
        seed=43,
    ).sort_values(["metric", "aggregation", "series"])
    assert set(first.num_images_total) == {2}
    assert set(second.num_images_total) == {2}
    np.testing.assert_allclose(first.estimate, second.estimate, rtol=0, atol=0)
    np.testing.assert_allclose(first.ci_low, second.ci_low, rtol=0, atol=0)
    np.testing.assert_allclose(first.ci_high, second.ci_high, rtol=0, atol=0)


def test_auc_ap_reports_finite_replicate_denominators_for_missing_contrast():
    summary = paired_clustered_auc_ap_summary(
        ["a", "a", "b", "b"],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [0.9, 0.1, 0.8, 0.7],
        [0.9, 0.1, 0.8, 0.7],
        repeats=300,
        seed=51,
    )
    row = summary[
        (summary.metric == "auroc")
        & (summary.series == "baseline")
        & (summary.aggregation == "micro")
    ].iloc[0]
    assert row.num_images == 1
    assert 0 < row.bootstrap_valid_repeats < 300
