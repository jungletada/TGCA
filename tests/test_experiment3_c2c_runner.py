from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    cam_threshold_grid,
)
from analysis.lazy_assignment.experiment3.run_c2c_intervention import (
    SIGNAL_KEYS,
    _head_region_means,
    _threshold_confusions,
    _validate_payload,
)


def test_head_region_means_keeps_original_class_id_axis():
    values = torch.zeros(3, 2, 20, 784)
    for class_id in range(20):
        values[:, :, class_id] = float(class_id)
    positive = np.asarray([2, 7], dtype=np.int64)
    regions = np.zeros((2, 784), dtype=np.int8)
    regions[:, :10] = 0
    regions[:, 10:20] = 1
    regions[:, 20:] = 2

    result = _head_region_means(values, positive, regions)

    assert result.shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(result[:, :, 0], np.full((3, 2, 3), 2.0))
    np.testing.assert_array_equal(result[:, :, 1], np.full((3, 2, 3), 7.0))
    with pytest.raises(ValueError, match=r"\[3,H,20,784\]"):
        _head_region_means(values[:, :, positive], positive, regions)


def _valid_payload() -> dict[str, np.ndarray]:
    classes = 1
    heads = 2
    patches = 784
    layers = 12
    positive = np.asarray([2], dtype=np.int64)
    raw = np.full((3, classes, patches), 1.0 / patches, dtype=np.float32)
    counts = np.zeros((patches, 22), dtype=np.uint16)
    counts[:, 0] = 256
    pre_offdiag = np.full((layers, heads, 20), 0.1, dtype=np.float32)
    pre_diagonal = np.full((layers, heads, 20), 0.2, dtype=np.float32)
    pre_class = pre_offdiag + pre_diagonal
    confusions = np.zeros((41, 21, 21), dtype=np.int64)
    confusions[:, 0, 0] = 448 * 448
    nan_regions = np.full((3, heads, classes, 3), np.nan, dtype=np.float32)
    payload = {
        "image_id": np.asarray("2007_000033"),
        "variant_code": np.asarray("C0"),
        "positive_class_ids": positive,
        "image_labels": np.eye(20, dtype=np.uint8)[2],
        "pair_class_ids": np.empty((0, 2), dtype=np.int64),
        "late_layers_one_based": np.asarray([10, 11, 12], dtype=np.int16),
        "thresholds": cam_threshold_grid(),
        "patch_label_counts": counts,
        "region_masks_rho05": np.full((classes, patches), 3, dtype=np.int8),
        "region_masks_rho07": np.full((classes, patches), 3, dtype=np.int8),
        "class_logits_all": np.zeros(20, dtype=np.float32),
        "patch_class_logits_all": np.zeros(20, dtype=np.float32),
        "patch_head_logits_positive": np.zeros((classes, patches), dtype=np.float32),
        "feature_post_l10_l12": np.zeros((3, classes, patches), dtype=np.float32),
        "feature_both_axis_removed_l10_l12": np.zeros(
            (3, classes, patches), dtype=np.float32
        ),
        "positive_pair_raw_cosine_l10_l12": np.empty((3, 0), dtype=np.float32),
        "positive_pair_residual_cosine_l10_l12": np.empty((3, 0), dtype=np.float32),
        "attention_c2p_raw_l10_l12": raw,
        "attention_c2p_conditional_l10_l12": raw.copy(),
        "attention_head_region_raw_rho05": nan_regions.copy(),
        "attention_head_region_conditional_rho05": nan_regions.copy(),
        "attention_head_region_raw_rho07": nan_regions.copy(),
        "attention_head_region_conditional_rho07": nan_regions.copy(),
        "c2c_pre_offdiag_mass": pre_offdiag,
        "c2c_pre_diagonal_mass": pre_diagonal,
        "c2c_pre_class_mass": pre_class,
        "c2c_post_offdiag_mass": pre_offdiag.copy(),
        "c2c_post_diagonal_mass": pre_diagonal.copy(),
        "c2c_post_class_mass": pre_class.copy(),
        "final_cam": np.zeros((classes, patches), dtype=np.float32),
        "threshold_confusions": confusions,
        "source_signal_sha256": np.asarray("a" * 64),
    }
    assert set(payload) == set(SIGNAL_KEYS)
    return payload


def test_c2c_runner_payload_schema_and_invariants_fail_closed():
    payload = _valid_payload()
    _validate_payload(payload, num_heads=2)

    broken = dict(payload)
    broken["attention_c2p_conditional_l10_l12"] = np.zeros_like(
        payload["attention_c2p_conditional_l10_l12"]
    )
    with pytest.raises(ValueError, match="do not sum to one"):
        _validate_payload(broken, num_heads=2)

    broken = dict(payload)
    broken["c2c_post_class_mass"] = np.zeros_like(payload["c2c_post_class_mass"])
    with pytest.raises(ValueError, match="decomposition"):
        _validate_payload(broken, num_heads=2)


def test_c2c_threshold_confusions_preserve_background_ties_and_void():
    thresholds = cam_threshold_grid()
    cams = np.zeros((1, 448, 448), dtype=np.float32)
    mask = np.zeros((448, 448), dtype=np.uint8)
    index = int(np.flatnonzero(np.isclose(thresholds, 0.45))[0])
    cams[0, 0, 0] = np.float32(0.45)
    cams[0, 0, 1] = np.nextafter(np.float32(0.45), np.float32(1.0))
    mask[0, :2] = 3
    mask[1, 0] = 255

    result = _threshold_confusions(cams, np.asarray([2]), mask, thresholds)

    assert result.shape == (41, 21, 21)
    assert result[index, 3, 0] == 1
    assert result[index, 3, 3] == 1
    np.testing.assert_array_equal(
        result.sum(axis=(1, 2)), np.full(41, 448 * 448 - 1, dtype=np.int64)
    )
