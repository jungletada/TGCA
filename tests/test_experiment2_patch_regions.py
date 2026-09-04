"""Synthetic tests for class- and pair-specific patch ownership."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from analysis.lazy_assignment.experiment2.patch_regions import (
    PAIR_REGION_BACKGROUND,
    PAIR_REGION_MIXED,
    PAIR_REGION_OTHER_FOREGROUND,
    PAIR_REGION_TARGET_A,
    PAIR_REGION_TARGET_B,
    PAIR_REGION_VOID,
    REGION_BACKGROUND,
    REGION_MIXED,
    REGION_OTHER_FOREGROUND,
    REGION_TARGET,
    REGION_VOID,
    assign_pair_patch_regions,
    assign_pair_patch_regions_from_counts,
    assign_patch_regions,
    assign_patch_regions_from_counts,
    patch_label_counts,
)


def _class_specific_fixture() -> np.ndarray:
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[0:16, 0:16] = 1  # pure target for zero-based class 0
    mask[0:16, 16:32] = 2  # pure other foreground
    mask[0:16, 32:48] = 0  # pure background
    mixed = mask[0:16, 48:64]
    mixed[:, :] = 0
    mixed[0:8, 0:8] = 1
    mixed[0:8, 8:16] = 2
    mixed[8:16, 0:8] = 0
    mixed[8:16, 8:16] = 255  # 1/3 each among valid pixels

    void_heavy = np.full(256, 255, dtype=np.uint8)
    void_heavy[0:127] = 1  # valid fraction < 0.5
    mask[16:32, 0:16] = void_heavy.reshape(16, 16)

    exactly_valid = np.full(256, 255, dtype=np.uint8)
    exactly_valid[0:128] = 1  # exactly 0.5 valid, all target
    mask[16:32, 16:32] = exactly_valid.reshape(16, 16)

    rho_sensitive = np.zeros(256, dtype=np.uint8)
    rho_sensitive[0:160] = 1  # 62.5% target
    mask[16:32, 32:48] = rho_sensitive.reshape(16, 16)

    tied = np.zeros(256, dtype=np.uint8)
    tied[0:128] = 1  # exact 50/50 has no unique owner
    mask[16:32, 48:64] = tied.reshape(16, 16)
    return mask


def test_class_specific_regions_apply_validity_and_rho_rules() -> None:
    mask = _class_specific_fixture()
    assignment = assign_patch_regions(mask, target_class_id=0, rho=0.5)
    codes = assignment["region_codes"]
    expected = np.asarray(
        [
            [REGION_TARGET, REGION_OTHER_FOREGROUND, REGION_BACKGROUND, REGION_MIXED],
            [REGION_VOID, REGION_TARGET, REGION_TARGET, REGION_MIXED],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(codes, expected)
    assert assignment["valid_fraction"][1, 0] == pytest.approx(127 / 256)
    assert assignment["valid_fraction"][1, 1] == pytest.approx(0.5)
    assert assignment["target_fraction"][1, 2] == pytest.approx(160 / 256)
    assert assignment["composition"] == {
        "void": 1,
        "mixed": 2,
        "background": 1,
        "other_fg": 1,
        "target": 3,
    }
    assert sum(assignment["composition"].values()) == codes.size
    assert all(mask_.dtype == np.bool_ for mask_ in assignment["region_masks"].values())
    json.dumps(assignment["metadata"], sort_keys=True)


def test_rho_07_reaggregates_without_changing_pixel_counts() -> None:
    mask = _class_specific_fixture()
    main = assign_patch_regions(mask, target_class_id=0, rho=0.5)
    sensitivity = assign_patch_regions(mask, target_class_id=0, rho=0.7)

    assert main["region_codes"][1, 2] == REGION_TARGET
    assert sensitivity["region_codes"][1, 2] == REGION_MIXED
    for name in (
        "target_count",
        "other_foreground_count",
        "background_count",
        "void_count",
        "valid_count",
    ):
        np.testing.assert_array_equal(main[name], sensitivity[name])


def test_compact_label_counts_reproduce_class_assignment_at_both_rhos() -> None:
    mask = _class_specific_fixture()
    counts = patch_label_counts(mask, patch_size=16)
    assert counts.shape == (8, 22)
    assert counts.dtype == np.uint16
    np.testing.assert_array_equal(counts.sum(axis=1), np.full(8, 256))
    # ID 1 is column 1; void ID 255 is represented by final column 21.
    assert counts[0, 1] == 256
    assert counts[4, 21] == 129

    for rho in (0.5, 0.7):
        direct = assign_patch_regions(mask, 0, rho=rho)
        restored = assign_patch_regions_from_counts(
            counts, 0, rho=rho, grid_size=(2, 4)
        )
        np.testing.assert_array_equal(restored["region_codes"], direct["region_codes"])
        for key in (
            "target_count",
            "other_foreground_count",
            "background_count",
            "void_count",
        ):
            np.testing.assert_array_equal(restored[key], direct[key])


def test_pair_regions_keep_second_target_separate_from_other_foreground() -> None:
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[0:16, 0:16] = 1  # class A (zero-based 0)
    mask[0:16, 16:32] = 3  # class B (zero-based 2)
    mask[0:16, 32:48] = 7  # a third foreground class
    mask[0:16, 48:64] = 0

    void_heavy = np.full(256, 255, dtype=np.uint8)
    void_heavy[0:100] = 1
    mask[16:32, 0:16] = void_heavy.reshape(16, 16)
    mixed = np.zeros(256, dtype=np.uint8)
    mixed[0:80] = 1
    mixed[80:160] = 3
    mixed[160:240] = 7
    mask[16:32, 16:32] = mixed.reshape(16, 16)
    # Remaining two patches stay background.

    assignment = assign_pair_patch_regions(mask, 0, 2, rho=0.5)
    codes = assignment["region_codes"]
    assert codes[0].tolist() == [
        PAIR_REGION_TARGET_A,
        PAIR_REGION_TARGET_B,
        PAIR_REGION_OTHER_FOREGROUND,
        PAIR_REGION_BACKGROUND,
    ]
    assert codes[1, 0] == PAIR_REGION_VOID
    assert codes[1, 1] == PAIR_REGION_MIXED
    assert assignment["target_b_count"][0, 1] == 256
    assert assignment["other_foreground_count"][0, 1] == 0
    assert assignment["metadata"]["target_mask_a_id"] == 1
    assert assignment["metadata"]["target_mask_b_id"] == 3
    assert assignment["composition"]["target_a"] == 1
    assert assignment["composition"]["target_b"] == 1
    json.dumps(assignment["metadata"], sort_keys=True)

    restored = assign_pair_patch_regions_from_counts(
        patch_label_counts(mask), 0, 2, grid_size=(2, 4)
    )
    np.testing.assert_array_equal(restored["region_codes"], codes)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: assign_patch_regions(np.zeros((31, 32), dtype=np.uint8), 0),
            "not divisible",
        ),
        (
            lambda: assign_patch_regions(np.zeros((32, 32), dtype=np.uint8), 20),
            r"\[0, 19\]",
        ),
        (
            lambda: assign_patch_regions(np.zeros((32, 32), dtype=np.uint8), 0, rho=0),
            "rho",
        ),
        (
            lambda: assign_patch_regions(np.full((32, 32), 42, dtype=np.uint8), 0),
            "invalid VOC IDs",
        ),
        (
            lambda: assign_patch_regions(np.zeros((32, 32), dtype=np.float32), 0),
            "integer IDs",
        ),
        (
            lambda: assign_pair_patch_regions(
                torch.zeros((32, 32), dtype=torch.long), 1, 1
            ),
            "distinct classes",
        ),
    ],
)
def test_invalid_region_inputs_fail_loudly(call, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
