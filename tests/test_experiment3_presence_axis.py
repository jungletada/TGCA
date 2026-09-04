from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from analysis.lazy_assignment.experiment3.presence_axis import (
    TwoFoldPresenceAccumulator,
    ZeroNormError,
    axis_removed_cosine_maps,
    build_two_fold_split,
    decompose_along_direction,
    heldout_centered_projections,
    normalized_all_ones_direction,
    presence_projection_auroc,
    remove_direction,
    sha256_two_fold,
    token_axis_energy,
    token_pair_axis_metrics,
    validate_two_fold_split,
)


def _ids_by_fold(per_fold: int) -> tuple[list[str], list[str]]:
    result: tuple[list[str], list[str]] = ([], [])
    candidate = 0
    while min(len(values) for values in result) < per_fold:
        image_id = f"synthetic_{candidate:05d}"
        fold = sha256_two_fold(image_id)
        if len(result[fold]) < per_fold:
            result[fold].append(image_id)
        candidate += 1
    return result


def test_all_ones_axis_exactly_reconstructs_native_mean_logit() -> None:
    generator = torch.Generator().manual_seed(91)
    tokens = torch.randn(5, 20, 384, generator=generator, dtype=torch.float64)
    direction = normalized_all_ones_direction(384, dtype=torch.float64)
    parts = decompose_along_direction(tokens, direction)

    reconstructed_logits = parts.coefficients / math.sqrt(384)
    torch.testing.assert_close(
        reconstructed_logits, tokens.mean(dim=-1), rtol=0, atol=2e-15
    )
    torch.testing.assert_close(
        parts.parallel + parts.residual, tokens, rtol=0, atol=2e-15
    )
    orthogonality = torch.sum(parts.residual * direction, dim=-1)
    torch.testing.assert_close(
        orthogonality, torch.zeros_like(orthogonality), rtol=0, atol=6e-15
    )
    assert direction.norm().item() == pytest.approx(1.0, abs=2e-15)


def test_projection_normalizes_supplied_direction_and_rejects_bad_inputs() -> None:
    tokens = torch.tensor([[1.0, 2.0, 3.0]])
    residual_unit = remove_direction(tokens, torch.tensor([0.0, 0.0, 1.0]))
    residual_scaled = remove_direction(tokens, torch.tensor([0.0, 0.0, 7.0]))
    torch.testing.assert_close(residual_scaled, residual_unit)
    torch.testing.assert_close(residual_unit, torch.tensor([[1.0, 2.0, 0.0]]))

    with pytest.raises(ZeroNormError):
        remove_direction(tokens, torch.zeros(3))
    with pytest.raises(ValueError, match="NaN or Inf"):
        remove_direction(torch.tensor([[1.0, float("nan"), 2.0]]), torch.ones(3))
    with pytest.raises(ValueError, match="direction shape"):
        remove_direction(tokens, torch.ones(2))
    with pytest.raises(ValueError, match="positive integer"):
        normalized_all_ones_direction(0)
    with pytest.raises(TypeError, match="floating"):
        normalized_all_ones_direction(3, dtype=torch.int64)


def test_v0_v3_share_exact_axis_removed_numerator() -> None:
    generator = torch.Generator().manual_seed(12)
    classes = torch.randn(2, 3, 7, generator=generator, dtype=torch.float64)
    patches = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    direction = torch.randn(7, generator=generator, dtype=torch.float64)
    result = axis_removed_cosine_maps(classes, patches, direction)

    torch.testing.assert_close(
        result.residual_dot, result.direct_residual_dot, rtol=1e-13, atol=1e-13
    )
    raw_denominator = result.class_norms[:, :, None] * result.patch_norms[:, None]
    class_denominator = (
        result.residual_class_norms[:, :, None] * result.patch_norms[:, None]
    )
    patch_denominator = (
        result.class_norms[:, :, None] * result.residual_patch_norms[:, None]
    )
    both_denominator = (
        result.residual_class_norms[:, :, None] * result.residual_patch_norms[:, None]
    )
    torch.testing.assert_close(result.raw * raw_denominator, result.raw_dot)
    torch.testing.assert_close(
        result.class_only_removed * class_denominator, result.residual_dot
    )
    torch.testing.assert_close(
        result.patch_only_removed * patch_denominator, result.residual_dot
    )
    torch.testing.assert_close(
        result.both_removed * both_denominator, result.residual_dot
    )
    assert result.raw.shape == (2, 3, 5)
    assert bool(torch.isfinite(result.both_removed).all())


def test_axis_cosines_reject_zero_raw_or_residual_vectors() -> None:
    direction = normalized_all_ones_direction(4)
    valid_patch = torch.tensor([[[1.0, -1.0, 0.0, 0.0]]])
    with pytest.raises(ZeroNormError, match="class_tokens"):
        axis_removed_cosine_maps(torch.zeros(1, 1, 4), valid_patch, direction)
    with pytest.raises(ZeroNormError, match="axis-removed class_tokens"):
        axis_removed_cosine_maps(torch.ones(1, 1, 4), valid_patch, direction)
    with pytest.raises(ValueError, match="batch sizes"):
        axis_removed_cosine_maps(torch.randn(2, 1, 4), torch.randn(1, 2, 4), direction)


def test_token_pair_dot_decomposition_and_axis_energy() -> None:
    tokens = torch.tensor(
        [[[3.0, 1.0, -1.0], [2.0, -2.0, 1.0], [-1.0, 3.0, 2.0]]],
        dtype=torch.float64,
    )
    direction = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    result = token_pair_axis_metrics(tokens, direction)
    torch.testing.assert_close(
        result.raw_dot, result.axis_dot + result.residual_dot, rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        result.axis_energy, token_axis_energy(tokens, direction), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        torch.diagonal(result.raw_cosine, dim1=-2, dim2=-1),
        torch.ones(1, 3, dtype=torch.float64),
    )
    assert bool(((result.axis_energy >= 0) & (result.axis_energy <= 1)).all())


def test_sha256_split_is_deterministic_exhaustive_and_tamper_evident() -> None:
    fold_ids = _ids_by_fold(5)
    image_ids = fold_ids[0] + fold_ids[1]
    first = build_two_fold_split(image_ids)
    second = build_two_fold_split(list(reversed(image_ids)))
    assert first == second
    assert {fold for fold in first.values()} == {0, 1}
    assert validate_two_fold_split(first, expected_image_ids=image_ids) == first

    tampered = dict(first)
    tampered[image_ids[0]] = 1 - tampered[image_ids[0]]
    with pytest.raises(ValueError, match="not its deterministic"):
        validate_two_fold_split(tampered)
    with pytest.raises(ValueError, match="duplicates"):
        build_two_fold_split([image_ids[0], image_ids[0]])
    with pytest.raises(ValueError, match="split ID mismatch"):
        validate_two_fold_split(first, expected_image_ids=image_ids[:-1])
    with pytest.raises(ValueError, match="edge whitespace"):
        sha256_two_fold(" bad_id")


def _synthetic_accumulator(
    *, perturb_fold_zero: bool = False
) -> tuple[TwoFoldPresenceAccumulator, dict[str, tuple[np.ndarray, np.ndarray]]]:
    layers, classes, width = 2, 3, 5
    true_direction = np.asarray([1.0, 1.0, -1.0, 0.5, 0.0], dtype=np.float64)
    true_direction /= np.linalg.norm(true_direction)
    identities = np.asarray(
        [
            [0.2, -0.1, 0.4, 0.0, 0.3],
            [-0.3, 0.1, 0.0, 0.2, -0.1],
            [0.1, 0.3, -0.2, -0.1, 0.4],
        ]
    )
    patterns = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int64)
    ids = _ids_by_fold(len(patterns))
    accumulator = TwoFoldPresenceAccumulator(layers, classes, width)
    examples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fold in (0, 1):
        for index, image_id in enumerate(ids[fold]):
            labels = patterns[index]
            tokens = np.empty((layers, classes, width), dtype=np.float64)
            for layer in range(layers):
                scale = float(layer + 1)
                tokens[layer] = identities + labels[:, None] * scale * true_direction
            if perturb_fold_zero and fold == 0:
                perturbation = np.asarray([3.0, -2.0, 1.0, 4.0, -5.0])
                tokens = tokens + (index + 1) * perturbation
            assert accumulator.update(image_id, tokens, labels) == fold
            examples[image_id] = (tokens, labels)
    return accumulator, examples


def test_streaming_crossfit_direction_and_heldout_presence_auc() -> None:
    accumulator, examples = _synthetic_accumulator()
    registry = accumulator.finalize()
    true_direction = np.asarray([1.0, 1.0, -1.0, 0.5, 0.0], dtype=np.float64)
    true_direction /= np.linalg.norm(true_direction)

    expected = np.broadcast_to(true_direction, registry.shared_directions.shape)
    np.testing.assert_allclose(registry.shared_directions, expected, atol=1e-12)
    np.testing.assert_allclose(registry.class_alignment, 1.0, atol=1e-12)
    np.testing.assert_allclose(registry.loo_class_alignment, 1.0, atol=1e-12)
    assert registry.fit_fold_for_eval(0) == 1
    assert registry.fit_fold_for_eval(1) == 0

    scores_by_layer = [[], []]
    labels = []
    for image_id in sorted(examples):
        tokens, target = examples[image_id]
        projection = heldout_centered_projections(
            tokens, eval_fold=sha256_two_fold(image_id), registry=registry
        )
        for layer in range(2):
            scores_by_layer[layer].append(projection[layer])
        labels.append(target)
    labels_array = np.asarray(labels)
    for layer_scores in scores_by_layer:
        estimate = presence_projection_auroc(np.asarray(layer_scores), labels_array)
        assert estimate.micro == pytest.approx(1.0)
        assert estimate.macro_class == pytest.approx(1.0)
        np.testing.assert_allclose(estimate.classwise, 1.0)
        np.testing.assert_array_equal(estimate.positive_counts, 4)
        np.testing.assert_array_equal(estimate.negative_counts, 4)


def test_eval_fold_perturbation_cannot_change_its_fit_direction() -> None:
    baseline, _ = _synthetic_accumulator(perturb_fold_zero=False)
    perturbed, _ = _synthetic_accumulator(perturb_fold_zero=True)
    baseline_registry = baseline.finalize()
    perturbed_registry = perturbed.finalize()

    # Eval fold 0 always uses fit fold 1, which was not perturbed.
    baseline_means, baseline_direction, fit_fold = baseline_registry.heldout_parameters(
        0
    )
    changed_means, changed_direction, changed_fit_fold = (
        perturbed_registry.heldout_parameters(0)
    )
    assert fit_fold == changed_fit_fold == 1
    np.testing.assert_array_equal(changed_means, baseline_means)
    np.testing.assert_array_equal(changed_direction, baseline_direction)
    # The direction fitted on fold 0 is allowed to change.
    assert not np.array_equal(
        baseline_registry.fit_means[0], perturbed_registry.fit_means[0]
    )


def test_accumulator_rejects_duplicates_and_missing_class_status() -> None:
    ids = _ids_by_fold(1)
    accumulator = TwoFoldPresenceAccumulator(1, 2, 3)
    tokens = np.ones((1, 2, 3))
    accumulator.update(ids[0][0], tokens, [1, 0])
    with pytest.raises(ValueError, match="duplicate"):
        accumulator.update(ids[0][0], tokens, [1, 0])
    accumulator.update(ids[1][0], tokens, [1, 0])
    with pytest.raises(ValueError, match="positive sufficient statistic"):
        accumulator.finalize()


def test_synthetic_common_axis_recoupling_disappears_after_both_removed() -> None:
    direction = normalized_all_ones_direction(4, dtype=torch.float64)
    residual_a = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    residual_a /= residual_a.norm()
    residual_b = torch.tensor([1.0, 1.0, -2.0, 0.0], dtype=torch.float64)
    residual_b /= residual_b.norm()
    zero = torch.tensor(0.0, dtype=torch.float64)
    torch.testing.assert_close(torch.dot(residual_a, direction), zero)
    torch.testing.assert_close(torch.dot(residual_b, direction), zero)
    torch.testing.assert_close(torch.dot(residual_a, residual_b), zero)

    classes = torch.stack(
        (10.0 * direction + residual_a, 10.0 * direction + residual_b)
    )
    patches = torch.stack(
        (
            10.0 * direction + 0.1 * (residual_a + residual_b),
            3.0 * residual_a,
            3.0 * residual_b,
        )
    )
    maps = axis_removed_cosine_maps(
        classes.unsqueeze(0), patches.unsqueeze(0), direction
    )
    pair = token_pair_axis_metrics(classes.unsqueeze(0), direction)

    assert pair.raw_cosine[0, 0, 1].item() > 0.98
    assert abs(pair.residual_cosine[0, 0, 1].item()) < 1e-12
    assert maps.raw[0, 0].argmax().item() == 0
    assert maps.raw[0, 1].argmax().item() == 0
    assert maps.both_removed[0, 0].argmax().item() == 1
    assert maps.both_removed[0, 1].argmax().item() == 2


def test_presence_auroc_retains_undefined_class_without_imputation() -> None:
    scores = np.asarray([[0.1, 0.0], [0.9, 0.2], [0.2, 0.4], [0.8, 0.6]])
    labels = np.asarray([[0, 1], [1, 1], [0, 1], [1, 1]])
    result = presence_projection_auroc(scores, labels)
    assert result.classwise[0] == pytest.approx(1.0)
    assert np.isnan(result.classwise[1])
    assert result.macro_class == pytest.approx(1.0)
    assert np.isfinite(result.micro)
    with pytest.raises(ValueError, match="labels shape"):
        presence_projection_auroc(scores, labels[:, :1])
    with pytest.raises(ValueError, match="binary"):
        presence_projection_auroc(scores, np.asarray([[0, 2]] * 4))
