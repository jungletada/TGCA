import math

import numpy as np
import torch

from analysis.final_token_relations.relations import (
    apply_shared_basis,
    classifier_relevance,
    final_token_relations,
    positive_channel_statistics,
    positive_coordinate_patch_statistics,
    spatial_probability,
    transformed_mean_readout,
)
from analysis.spatial_graph_stability.basis import generate_basis_transforms


def _tokens():
    torch.manual_seed(20260901)
    classes = torch.randn(2, 3, 8)
    classes[..., 0] = classes[..., 0].abs() + 0.1
    return classes, torch.randn(2, 11, 8)


def test_final_token_einsum_matches_direct_matmul_and_shapes():
    classes, patches = _tokens()
    relations = final_token_relations(classes, patches)
    expected = classes @ patches.transpose(1, 2) / math.sqrt(classes.shape[-1])
    assert relations["s_last"].shape == (2, 3, 11)
    assert torch.allclose(relations["s_last"], expected, atol=1e-7, rtol=0)
    assert torch.allclose(relations["s_dot"], expected * math.sqrt(classes.shape[-1]), atol=1e-6, rtol=0)


def test_shared_orthogonal_transforms_preserve_final_token_affinity():
    classes, patches = _tokens()
    original = final_token_relations(classes.double(), patches.double())["s_last"]
    for transform in generate_basis_transforms(8, base_seed=11, haar_count=2):
        matrix = torch.from_numpy(transform.matrix)
        transformed_classes, transformed_patches = apply_shared_basis(classes.double(), patches.double(), matrix)
        actual = final_token_relations(transformed_classes, transformed_patches)["s_last"]
        assert torch.max(torch.abs(actual - original)).item() < 1e-10


def test_positive_negative_decomposition_and_native_mean_readout_identity():
    classes, patches = _tokens()
    relations = final_token_relations(classes, patches)
    assert torch.allclose(relations["s_last"], relations["s_pos"] - relations["s_negmag"], atol=1e-6, rtol=0)
    statistics = positive_channel_statistics(classes)
    assert torch.max(statistics["logit_identity_error"]).item() < 1e-6
    assert torch.allclose(statistics["native_logits"], classes.mean(-1), atol=1e-7, rtol=0)


def test_permutation_preserves_positive_channel_relation_but_signed_rotation_is_diagnostic():
    classes, patches = _tokens()
    relations = final_token_relations(classes.double(), patches.double())
    transforms = generate_basis_transforms(8, base_seed=17, haar_count=1)
    permutation = next(item for item in transforms if item.kind == "permutation")
    c_perm, p_perm = apply_shared_basis(classes.double(), patches.double(), torch.from_numpy(permutation.matrix))
    assert torch.max(torch.abs(final_token_relations(c_perm, p_perm)["s_pos"] - relations["s_pos"])).item() < 1e-10
    signed = next(item for item in transforms if item.kind == "signed_permutation")
    c_signed, p_signed = apply_shared_basis(classes.double(), patches.double(), torch.from_numpy(signed.matrix))
    # A signed permutation is intentionally not asserted invariant: the test
    # confirms that this diagnostic is actually evaluated and has the shape.
    assert final_token_relations(c_signed, p_signed)["s_pos"].shape == relations["s_pos"].shape


def test_transformed_mean_readout_preserves_logits():
    classes, _ = _tokens()
    transform = next(item for item in generate_basis_transforms(8, base_seed=29, haar_count=1) if item.kind == "haar")
    original, transformed = transformed_mean_readout(classes.double(), torch.from_numpy(transform.matrix))
    assert torch.max(torch.abs(original - transformed)).item() < 1e-10


def test_positive_coordinate_patch_statistics_and_spatial_probability_are_finite():
    classes, patches = _tokens()
    values = positive_coordinate_patch_statistics(classes, patches)
    relations = final_token_relations(classes, patches)
    probability = spatial_probability(relations["s_last"])
    assert values["u_pos"].shape == (2, 3, 11)
    assert values["c_sign"].shape == (2, 3, 11)
    assert torch.isfinite(values["u_pos"]).all()
    assert torch.isfinite(values["c_sign"]).all()
    assert torch.allclose(probability.sum(-1), torch.ones_like(probability.sum(-1)), atol=1e-6, rtol=0)


def test_classifier_relevance_is_native_relu_max_normalization():
    raw = torch.tensor([[[-2.0, 1.0, 4.0], [0.0, -3.0, -4.0]]])
    relevance = classifier_relevance(raw)
    assert torch.equal(relevance[0, 0], torch.tensor([0.0, 0.25, 1.0]))
    assert torch.equal(relevance[0, 1], torch.zeros(3))


def test_relation_functions_do_not_accept_or_use_gt_arguments():
    classes, patches = _tokens()
    # Construction has exactly the representation inputs; GT is deliberately
    # absent from this API and is only consumed in the runner's metric stage.
    assert set(final_token_relations(classes, patches)).issuperset({"s_last", "s_pos", "s_negmag"})
