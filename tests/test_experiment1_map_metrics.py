"""Deterministic checks for Experiment 1 representation-map metrics."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.lazy_assignment.metrics_experiment1 import (
    component_metrics,
    score_map_metrics,
    spearman_correlation,
    topk_count,
    topk_jaccard,
    topk_mask,
)


def test_constant_map_has_zero_scale_and_spatial_variation() -> None:
    values = np.full((4, 5), 0.25, dtype=np.float32)
    metrics = score_map_metrics(values, 4, 5)

    assert metrics["score_min"] == pytest.approx(0.25)
    assert metrics["score_max"] == pytest.approx(0.25)
    assert metrics["score_std"] == pytest.approx(0.0)
    assert metrics["dynamic_range"] == pytest.approx(0.0)
    assert metrics["upper_tail_gap"] == pytest.approx(0.0)
    assert metrics["top10_concentration"] == pytest.approx(0.0)
    assert metrics["total_variation"] == pytest.approx(0.0)
    assert metrics["spatial_entropy_tau_100"] == pytest.approx(1.0)
    assert np.isnan(metrics["neighbor_pearson"])
    assert np.isnan(metrics["neighbor_spearman"])


def test_single_sharp_peak_has_low_entropy_and_one_component() -> None:
    values = np.zeros((10, 10), dtype=np.float64)
    values[4, 4] = 1.0
    metrics = score_map_metrics(values, 10, 10)

    assert metrics["score_max"] == pytest.approx(1.0)
    assert metrics["score_q95"] == pytest.approx(0.0)
    assert metrics["top10_concentration"] == pytest.approx(0.1)
    assert metrics["spatial_entropy_tau_050"] < metrics["spatial_entropy_tau_100"]
    assert metrics["spatial_entropy_tau_100"] < metrics["spatial_entropy_tau_200"]
    assert metrics["num_components_top10"] >= 1


def test_two_disconnected_components_are_counted_with_four_neighbors() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[4, 4] = True

    count, largest_fraction = component_metrics(mask)

    assert count == 2
    assert largest_fraction == pytest.approx(2 / 3)


def test_monotone_gradient_has_expected_quantiles_and_neighbor_order() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4) / 15.0
    metrics = score_map_metrics(values, 4, 4)

    assert metrics["score_q50"] == pytest.approx(0.5)
    assert metrics["score_q95"] == pytest.approx(0.95)
    assert metrics["upper_tail_gap"] == pytest.approx(0.45)
    assert metrics["neighbor_pearson"] > 0.8
    assert metrics["neighbor_spearman"] > 0.8
    assert spearman_correlation(values, values) == pytest.approx(1.0)
    assert spearman_correlation(values, -values) == pytest.approx(-1.0)


def test_topk_uses_ceil_and_stable_flat_index_tie_breaking() -> None:
    values = np.ones((3, 3), dtype=np.float64)
    mask = topk_mask(values, 0.20)

    assert topk_count(9, 0.20) == 2
    assert mask.sum() == 2
    assert mask.reshape(-1).tolist() == [True, True, False, False, False, False, False, False, False]
    assert topk_jaccard(values, values, 0.20) == pytest.approx(1.0)


def test_invalid_map_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        score_map_metrics(np.asarray([0.0, np.nan]), 1, 2)
    with pytest.raises(ValueError, match="grid shape"):
        score_map_metrics(np.zeros(4), 1, 3)
