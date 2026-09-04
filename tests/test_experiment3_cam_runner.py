from __future__ import annotations

import numpy as np

from analysis.lazy_assignment.experiment3 import cam_layer_intervention
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    CAM_VARIANT_SPECS,
)
from analysis.lazy_assignment.experiment3.run_cam_layer_intervention import (
    SOURCE_DIAGNOSTIC_KEYS,
    _threshold_confusions,
)


def _naive_confusions(
    monkeypatch,
    normalized: np.ndarray,
    active_classes: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    monkeypatch.setattr(
        cam_layer_intervention,
        "upsample_and_normalize_active_cams",
        lambda _cam: normalized,
    )
    dummy_raw_cam = np.zeros((len(active_classes), 28, 28), dtype=np.float32)
    return np.stack(
        [
            cam_layer_intervention.raw_cam_confusion_at_threshold(
                dummy_raw_cam,
                active_classes,
                mask,
                float(threshold),
            )
            for threshold in thresholds
        ]
    )


def test_optimized_threshold_confusions_equal_naive_at_multiple_thresholds_and_void(
    monkeypatch,
):
    rng = np.random.default_rng(191)
    normalized = rng.uniform(0.0, 1.0, size=(3, 448, 448)).astype(np.float32)
    active_classes = np.asarray([1, 7, 18], dtype=np.int64)
    mask = rng.choice(
        np.asarray([0, 2, 8, 19, 255], dtype=np.uint8),
        size=(448, 448),
        p=(0.45, 0.15, 0.15, 0.15, 0.10),
    )
    thresholds = np.asarray([0.20, 0.31, 0.45, 0.53, 0.60], dtype=np.float64)

    optimized = _threshold_confusions(normalized, active_classes, mask, thresholds)
    naive = _naive_confusions(monkeypatch, normalized, active_classes, mask, thresholds)

    np.testing.assert_array_equal(optimized, naive)
    expected_valid = int(np.count_nonzero(mask != 255))
    np.testing.assert_array_equal(
        optimized.sum(axis=(1, 2)),
        np.full(len(thresholds), expected_valid, dtype=np.int64),
    )


def test_optimized_threshold_confusions_preserve_exact_bg_and_foreground_ties(
    monkeypatch,
):
    thresholds = np.asarray([0.20, 0.45, 0.60], dtype=np.float64)
    normalized = np.zeros((2, 448, 448), dtype=np.float32)
    active_classes = np.asarray([2, 7], dtype=np.int64)
    mask = np.zeros((448, 448), dtype=np.uint8)

    # Exact ties must remain background because the background channel is
    # prepended. The adjacent pixels are the smallest representable float32
    # values strictly above each corresponding threshold.
    for column, threshold in enumerate(thresholds):
        normalized[0, 0, 2 * column] = np.float32(threshold)
        normalized[0, 0, 2 * column + 1] = np.nextafter(
            np.float32(threshold), np.float32(1.0)
        )
        mask[0, 2 * column : 2 * column + 2] = 3

    # An above-threshold foreground/foreground tie selects the first active
    # class in both the optimized and background-prepended naive paths.
    normalized[:, 0, 8] = np.float32(0.9)
    mask[0, 8] = 3
    # Void must never enter any confusion matrix, regardless of confidence.
    normalized[1, 1, 0] = np.float32(1.0)
    mask[1, 0] = 255

    optimized = _threshold_confusions(normalized, active_classes, mask, thresholds)
    naive = _naive_confusions(monkeypatch, normalized, active_classes, mask, thresholds)

    np.testing.assert_array_equal(optimized, naive)
    expected_valid = 448 * 448 - 1
    np.testing.assert_array_equal(
        optimized.sum(axis=(1, 2)),
        np.full(len(thresholds), expected_valid, dtype=np.int64),
    )


def test_cam_runner_variant_order_and_experiment2_source_mapping_are_locked():
    assert [spec.code for spec in CAM_VARIANT_SPECS] == [
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ]
    assert [spec.name for spec in CAM_VARIANT_SPECS] == [
        "native_last3",
        "l10_only",
        "l11_only",
        "l12_only",
        "l10_l11",
        "l4_l6_control",
    ]
    assert SOURCE_DIAGNOSTIC_KEYS == {
        "B0": "final_cam",
        "B1": "diagnostic_c2p_cam_l10",
        "B2": "diagnostic_c2p_cam_l11",
        "B3": "diagnostic_c2p_cam_l12",
        "B5": "diagnostic_c2p_cam_mid3",
    }
    assert "B4" not in SOURCE_DIAGNOSTIC_KEYS
