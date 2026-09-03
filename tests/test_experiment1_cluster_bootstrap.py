"""Tests that Experiment 1 uncertainty is clustered at image level."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.lazy_assignment.bootstrap import (
    cluster_bootstrap_macro_class_means,
    cluster_bootstrap_means,
    cluster_standardized_effect,
    draw_cluster_counts,
)


def _unequal_cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a", "a", "a", "b"],
            "class_id": [0, 1, 2, 0],
            "value": [0.0, 0.0, 0.0, 4.0],
        }
    )


def test_bootstrap_resamples_whole_images_and_is_reproducible() -> None:
    frame = _unequal_cluster_frame()
    first = cluster_bootstrap_means(
        frame, "image_id", ["value"], repeats=2000, seed=17
    )["value"]
    second = cluster_bootstrap_means(
        frame, "image_id", ["value"], repeats=2000, seed=17
    )["value"]

    # The point estimate remains pair-equal (micro), while uncertainty draws
    # complete image clusters instead of pretending the four rows are independent.
    assert first.estimate == pytest.approx(1.0)
    assert first == second
    assert first.n_clusters == 2
    assert first.n_rows == 4
    assert first.ci_low <= first.estimate <= first.ci_high


def test_cluster_draw_counts_sum_to_number_of_images() -> None:
    draws = draw_cluster_counts(num_clusters=7, repeats=25, seed=3)
    repeated = draw_cluster_counts(num_clusters=7, repeats=25, seed=3)

    assert draws.shape == (25, 7)
    assert np.all(draws.sum(axis=1) == 7)
    assert np.array_equal(draws, repeated)

    # A single sampled count is applied to every class/model row from that image.
    # Consequently a paired delta equals the difference of model estimates formed
    # with the exact same bootstrap image indices.
    left_image_means = np.linspace(-0.5, 0.5, 7)
    right_image_means = np.linspace(0.5, 1.5, 7)
    paired_delta = draws @ (right_image_means - left_image_means) / 7
    same_draw_difference = (
        draws @ right_image_means / 7 - draws @ left_image_means / 7
    )
    assert np.allclose(paired_delta, same_draw_difference)


def test_macro_class_point_estimate_gives_classes_equal_weight() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "class_id": [0, 0, 1],
            "value": [0.0, 2.0, 9.0],
        }
    )
    result = cluster_bootstrap_macro_class_means(
        frame,
        "image_id",
        "class_id",
        ["value"],
        repeats=1000,
        seed=29,
    )["value"]

    # class 0 mean=1 and class 1 mean=9, hence equal-class macro mean=5.
    assert result.estimate == pytest.approx(5.0)
    assert result.n_clusters == 3


def test_standardized_paired_effect_is_computed_from_image_deltas() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b", "c"],
            "delta": [1.0, 3.0, 2.0, 4.0, 5.0],
        }
    )
    image_deltas = np.asarray([2.0, 3.0, 5.0])
    expected = image_deltas.mean() / image_deltas.std(ddof=1)

    assert cluster_standardized_effect(frame, "image_id", "delta") == pytest.approx(
        expected
    )
