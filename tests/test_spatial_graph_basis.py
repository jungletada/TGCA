from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from analysis.spatial_graph_stability.basis import (
    cosine_affinity,
    generate_basis_transforms,
    last_channel_selector,
    orthogonality_error,
    transform_conv_input_basis,
    transform_linear_weight,
)
from analysis.spatial_graph_stability.basis_metrics import (
    channel_selection_agreement,
    compare_vote_maps,
    vote_counts,
)


def _official_reference(tokens: torch.Tensor):
    width = tokens.shape[-1]
    positions = torch.arange(
        -width // 2 + 1,
        width // 2 + 1,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    kernel = torch.exp(-0.5 * (positions / math.sqrt(width)).square())
    kernel = (kernel / kernel.max()).view(1, 1, width)
    spectrum = torch.fft.fft(tokens, dim=-1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1) * kernel
    low_pass = torch.fft.ifft(
        torch.fft.ifftshift(spectrum, dim=-1), dim=-1
    ).real
    stability = tokens / torch.abs(low_pass - tokens)
    indices = torch.topk(stability, k=1, dim=1, largest=True).indices
    pooled = torch.gather(tokens, 1, indices).mean(dim=1)
    return pooled, indices, stability, low_pass


def test_fixed_basis_transforms_are_orthogonal_and_reproducible():
    first = generate_basis_transforms(8, base_seed=17, haar_count=10)
    second = generate_basis_transforms(8, base_seed=17, haar_count=10)
    assert len(first) == 13
    assert [item.kind for item in first] == [
        "identity",
        "permutation",
        "signed_permutation",
        *(["haar"] * 10),
    ]
    for observed, reproduced in zip(first, second):
        np.testing.assert_array_equal(observed.matrix, reproduced.matrix)
        assert orthogonality_error(observed.matrix) < 1e-12


def test_linear_classifier_reparameterization_preserves_logits():
    generator = torch.Generator().manual_seed(101)
    query = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    weight = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    bias = torch.randn(4, generator=generator, dtype=torch.float64)
    rotation = torch.from_numpy(generate_basis_transforms(8)[3].matrix)
    expected = F.linear(query, weight, bias)
    observed = F.linear(
        query @ rotation,
        transform_linear_weight(weight, rotation),
        bias,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_conv_classifier_reparameterization_preserves_spatial_map():
    generator = torch.Generator().manual_seed(102)
    tokens = torch.randn(2, 4, 8, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 8, 3, 3, generator=generator, dtype=torch.float64)
    bias = torch.randn(3, generator=generator, dtype=torch.float64)
    rotation = torch.from_numpy(generate_basis_transforms(8)[4].matrix)
    grid = tokens.reshape(2, 2, 2, 8).permute(0, 3, 1, 2)
    transformed = (tokens @ rotation).reshape(2, 2, 2, 8).permute(0, 3, 1, 2)
    expected = F.conv2d(grid, weight, bias, padding=1)
    observed = F.conv2d(
        transformed,
        transform_conv_input_basis(weight, rotation),
        bias,
        padding=1,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-11, atol=1e-11)


def test_last_selector_exactly_matches_official_unclamped_reference():
    generator = torch.Generator().manual_seed(103)
    tokens = torch.randn(2, 7, 8, generator=generator, dtype=torch.float64)
    observed = last_channel_selector(tokens, topk=1, eps=None)
    expected = _official_reference(tokens)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


def test_cosine_affinity_is_orthogonal_basis_invariant():
    generator = torch.Generator().manual_seed(104)
    tokens = torch.randn(2, 9, 8, generator=generator, dtype=torch.float64)
    rotation = torch.from_numpy(generate_basis_transforms(8)[5].matrix)
    torch.testing.assert_close(
        cosine_affinity(tokens @ rotation),
        cosine_affinity(tokens),
        rtol=1e-11,
        atol=1e-11,
    )


def test_vote_metrics_and_explicit_channel_correspondence():
    indices = torch.tensor([[[0, 1, 1, 3]]], dtype=torch.long)
    votes = vote_counts(indices, 4)
    torch.testing.assert_close(votes, torch.tensor([[1, 2, 0, 1]]))
    comparison = compare_vote_maps(votes.numpy(), votes.numpy())
    np.testing.assert_allclose(comparison["vote_spearman"], 1.0)
    np.testing.assert_allclose(comparison["top10_jaccard"], 1.0)
    mapping = np.array([2, 0, 3, 1])
    transformed = indices.numpy()[:, :, mapping]
    agreement = channel_selection_agreement(indices.numpy(), transformed, mapping)
    np.testing.assert_allclose(agreement, 1.0)
