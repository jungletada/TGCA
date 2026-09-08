"""Deterministic contracts for the frozen multi-class selector scores."""

from __future__ import annotations

import inspect

import torch

from analysis.relational_selector.scores import (
    build_selector_scores,
    centered_dot_scores,
    raw_dot_scores,
    relative_ownership_scores,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20270908)
    classes = torch.randn(2, 20, 384, generator=generator)
    patches = torch.randn(2, 784, 384, generator=generator)
    raw = torch.randn(2, 20, 784, generator=generator)
    return classes, patches, raw


def test_selector_shapes_float32_and_finite() -> None:
    classes, patches, raw = _inputs()
    scores = build_selector_scores(classes.half(), patches.half(), raw.half())
    assert tuple(scores) == ("S0", "S1", "S2", "S3", "S4", "S5")
    for value in scores.values():
        assert value.shape == (2, 20, 784)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()


def test_centered_dot_is_common_class_shift_invariant() -> None:
    classes, patches, _ = _inputs()
    shift = torch.randn(2, 1, 384, generator=torch.Generator().manual_seed(9))
    baseline = centered_dot_scores(classes, patches)
    shifted = centered_dot_scores(classes + shift, patches)
    assert float((baseline - shifted).abs().max()) < 1e-5


def test_relative_ownership_is_common_class_logit_shift_invariant() -> None:
    classes, patches, _ = _inputs()
    q = raw_dot_scores(classes, patches)
    shift = torch.randn(2, 1, 784, generator=torch.Generator().manual_seed(11))
    baseline = relative_ownership_scores(q)
    shifted = relative_ownership_scores(q + shift)
    assert float((baseline - shifted).abs().max()) < 1e-5


def test_s3_is_exact_centered_raw_dot_and_s5_is_fixed_equal_sum() -> None:
    classes, patches, raw = _inputs()
    scores = build_selector_scores(classes, patches, raw)
    assert torch.allclose(scores["S3"], centered_dot_scores(classes, patches))
    from analysis.semantic_relations import spatial_minmax

    assert torch.allclose(scores["S5"], spatial_minmax(scores["S0"]) + spatial_minmax(scores["S4"]))


def test_score_construction_accepts_no_gt_or_image_labels() -> None:
    signature = inspect.signature(build_selector_scores)
    assert tuple(signature.parameters) == ("class_tokens", "patch_tokens", "raw_classifier_map")
