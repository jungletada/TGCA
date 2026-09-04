from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.lazy_assignment.experiment2.native_cam_stages import (
    decompose_native_cam_reduced,
)
from analysis.lazy_assignment.experiment3 import cam_layer_intervention as cam_readout


def _layer_values(batch: int = 1, classes: int = 1, patches: int = 2) -> torch.Tensor:
    return torch.stack(
        [torch.full((batch, classes, patches), float(layer)) for layer in range(1, 13)]
    )


@pytest.mark.parametrize(
    ("host", "variant", "expected"),
    [
        ("mctformer", "B0", 10.0 + 11.0 + 12.0),
        ("mctformer", "B4", 10.0 + 11.0),
        ("mctformer_plus", "B0", 11.0),
        ("MCTformer+", "l10_l11", 10.5),
    ],
)
def test_b0_b4_use_exact_host_specific_aggregation(host, variant, expected):
    result = cam_readout.aggregate_raw_c2p(host, _layer_values(), variant)
    torch.testing.assert_close(
        result, torch.full_like(result, expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("host", ["mctformer", "mctformer_plus"])
def test_b0_exactly_matches_experiment2_native_decomposition(host):
    generator = torch.Generator().manual_seed(309)
    c2p = torch.rand(12, 2, 3, 4, generator=generator)
    patch_cam = torch.rand(2, 3, 2, 2, generator=generator)
    p2p = torch.rand(2, 4, 4, generator=generator)
    reference = decompose_native_cam_reduced(host, patch_cam, c2p, p2p, num_classes=3)

    result = cam_readout.construct_cam_readout(host, patch_cam, c2p, p2p, "B0")

    torch.testing.assert_close(
        result.raw_c2p, reference["official_c2p_flat"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.preprop_cam, reference["class_attention_cam"], rtol=0, atol=0
    )
    torch.testing.assert_close(result.final_cam, reference["final_cam"], rtol=0, atol=0)


def test_plus_b4_applies_sqrt_after_layer_mean():
    layers = torch.zeros(12, 1, 1, 2)
    layers[9] = torch.tensor([[[1.0, 9.0]]])
    layers[10] = torch.tensor([[[7.0, 23.0]]])
    patch = torch.tensor([[[4.0, 9.0]]])
    identity = torch.eye(2).unsqueeze(0)

    result = cam_readout.construct_cam_readout(
        "mctformer_plus", patch, layers, identity, "B4"
    )

    expected_attention = torch.tensor([[[4.0, 16.0]]])
    expected = torch.sqrt(expected_attention * patch)
    torch.testing.assert_close(result.raw_c2p, expected_attention, rtol=0, atol=0)
    torch.testing.assert_close(result.preprop_cam, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.final_cam, expected, rtol=0, atol=0)


def test_p2p_uses_query_by_key_orientation_and_preserves_grid_shape():
    layers = torch.zeros(12, 1, 1, 2)
    layers[9] = torch.tensor([[[2.0, 3.0]]])
    patch = torch.tensor([[[[5.0, 7.0]]]])
    p2p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    result = cam_readout.construct_cam_readout("mctformer", patch, layers, p2p, "B1")

    preprop = torch.tensor([10.0, 21.0])
    expected = torch.tensor([[[[52.0, 114.0]]]])
    torch.testing.assert_close(result.preprop_cam.flatten(), preprop, rtol=0, atol=0)
    torch.testing.assert_close(result.final_cam, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda layers: layers[:11], "exactly 12"),
        (
            lambda layers: layers.__setitem__((0, 0, 0, 0), float("nan")) or layers,
            "NaN or Inf",
        ),
        (
            lambda layers: layers.__setitem__((0, 0, 0, 0), -1.0) or layers,
            "non-negative",
        ),
    ],
)
def test_raw_c2p_guards_shape_finite_and_nonnegative(mutation, message):
    layers = mutation(_layer_values())
    with pytest.raises(ValueError, match=message):
        cam_readout.aggregate_raw_c2p("mctformer", layers, "B0")


def test_cam_readout_rejects_incompatible_patch_and_p2p_shapes():
    layers = _layer_values(patches=3)
    patch = torch.ones(1, 1, 2)
    p2p = torch.eye(2).unsqueeze(0)
    with pytest.raises(ValueError, match="shapes disagree"):
        cam_readout.construct_cam_readout("mctformer", patch, layers, p2p, "B0")

    layers = _layer_values(patches=2)
    with pytest.raises(ValueError, match="patch_to_patch_sum must have shape"):
        cam_readout.construct_cam_readout(
            "mctformer", patch, layers, torch.ones(1, 2, 3), "B0"
        )


def test_threshold_grid_is_exact_inclusive_41_point_contract():
    grid = cam_readout.cam_threshold_grid()
    assert grid.dtype == np.float64
    assert len(grid) == 41
    assert grid[0] == 0.20
    assert grid[25] == 0.45
    assert grid[-1] == 0.60
    np.testing.assert_allclose(np.diff(grid), 0.01, rtol=0, atol=2e-16)


def test_threshold_tie_resolves_to_prepended_background(monkeypatch):
    normalized = np.zeros((1, 448, 448), dtype=np.float32)
    normalized[0, 0, 0] = np.float32(0.45)
    normalized[0, 0, 1] = np.nextafter(np.float32(0.45), np.float32(1.0))
    monkeypatch.setattr(
        cam_readout,
        "upsample_and_normalize_active_cams",
        lambda _cam: normalized,
    )

    prediction = cam_readout.raw_cam_prediction_at_threshold(
        np.zeros((1, 28, 28), dtype=np.float32), [6], 0.45
    )

    assert prediction[0, 0] == 0
    assert prediction[0, 1] == 7


def test_confusion_metrics_distinguish_binary_and_semantic_foreground():
    confusion = np.zeros((21, 21), dtype=np.int64)
    confusion[0, 0] = 80
    confusion[0, 1] = 10
    confusion[1, 0] = 20
    confusion[1, 1] = 30
    confusion[1, 2] = 5
    confusion[2, 1] = 7
    confusion[2, 2] = 28

    result = cam_readout.cam_metrics_from_confusion(confusion)

    assert result["binary_foreground_precision"] == pytest.approx(70 / 80)
    assert result["binary_foreground_recall"] == pytest.approx(70 / 90)
    assert result["semantic_correct_foreground_precision"] == pytest.approx(58 / 80)
    assert result["semantic_correct_foreground_recall"] == pytest.approx(58 / 90)
    assert result["per_class_precision"][1] == pytest.approx(30 / 47)
    assert result["per_class_recall"][1] == pytest.approx(30 / 55)
    expected_iou = np.asarray([80 / 110, 30 / 72, 28 / 40])
    np.testing.assert_allclose(result["per_class_iou"][:3], expected_iou)
    assert np.isnan(result["per_class_iou"][3:]).all()
    assert result["mean_iou"] == pytest.approx(expected_iou.mean())


def test_confusion_metrics_return_nan_for_undefined_foreground_denominators():
    confusion = np.zeros((21, 21), dtype=np.int64)
    confusion[0, 0] = 10
    result = cam_readout.cam_metrics_from_confusion(confusion)
    assert np.isnan(result["binary_foreground_precision"])
    assert np.isnan(result["binary_foreground_recall"])
    assert np.isnan(result["semantic_correct_foreground_precision"])
    assert np.isnan(result["semantic_correct_foreground_recall"])


def test_native_best_anchor_uses_lowest_threshold_on_exact_tie():
    thresholds = np.asarray([0.20, 0.21, 0.22, 0.23])
    native = np.asarray([0.5, 0.7, 0.7, np.nan])
    anchor = cam_readout.native_best_threshold_anchor(thresholds, native)

    assert anchor.index == 1
    assert anchor.threshold == pytest.approx(0.21)
    assert anchor.native_metric == pytest.approx(0.7)
    sampled = cam_readout.values_at_native_anchor(
        {"B0": native, "B1": [0.6, 0.8, 0.9, 0.1]}, anchor
    )
    assert sampled == {"B0": pytest.approx(0.7), "B1": pytest.approx(0.8)}


def test_native_best_anchor_rejects_nonmonotonic_grid_and_all_nan_curve():
    with pytest.raises(ValueError, match="strictly increasing"):
        cam_readout.native_best_threshold_anchor([0.2, 0.2], [0.4, 0.5])
    with pytest.raises(ValueError, match="finite value"):
        cam_readout.native_best_threshold_anchor([0.2, 0.3], [np.nan, np.nan])
