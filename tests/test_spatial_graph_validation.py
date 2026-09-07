from __future__ import annotations

import inspect

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.lazy_assignment.experiment2.patch_regions import (
    REGION_BACKGROUND,
    REGION_MIXED,
    REGION_TARGET,
    REGION_VOID,
    assign_patch_regions,
)
from analysis.spatial_graph_stability.bootstrap import (
    exact_rank_metrics,
    histogram_rank_bootstrap,
    make_cluster_draws,
    rank_histograms_by_image,
    ratio_bootstrap,
)
from analysis.spatial_graph_stability.graph import (
    GRAPH_NAMES,
    PATCH_LABEL_MIXED,
    PATCH_LABEL_VOID,
    construct_local_graphs,
    dense_symmetric_adjacency,
    local_edge_index,
    semantic_patch_labels,
)
from analysis.spatial_graph_stability.basis import generate_basis_transforms
from analysis.spatial_graph_stability.graph_metrics import evaluate_graph_metrics


def test_local_candidate_edges_are_shared_finite_nonnegative_and_symmetric():
    edges = local_edge_index((3, 4))
    expected = 3 * 3 + 2 * 4 + 2 * 2 * 3
    assert edges.count == expected
    assert len(np.unique(np.stack((edges.source, edges.target), axis=1), axis=0)) == expected
    tokens = torch.randn(2, 12, 6, generator=torch.Generator().manual_seed(201))
    graphs = construct_local_graphs(tokens, edges)
    assert tuple(graphs) == GRAPH_NAMES
    for weights in graphs.values():
        assert weights.shape == (2, expected)
        assert torch.isfinite(weights).all()
        assert torch.all(weights >= 0)
        adjacency = dense_symmetric_adjacency(weights, edges)
        torch.testing.assert_close(adjacency, adjacency.transpose(1, 2))


def test_graph_constructor_has_no_gt_or_label_input():
    parameters = inspect.signature(construct_local_graphs).parameters
    assert tuple(parameters) == ("patch_tokens", "edges", "sigma_s")


def test_patch_gt_geometry_matches_experiment2_region_convention():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, 16:] = 1
    mask[16:, :8] = 1
    mask[16:, 8:16] = 2
    mask[16:, 16:] = 255
    mask[16:24, 16:] = 3  # exactly 50% valid, all valid pixels class 3
    labels = semantic_patch_labels(mask, patch_size=16)
    np.testing.assert_array_equal(
        labels,
        np.array([[0, 1], [PATCH_LABEL_MIXED, 3]], dtype=np.int8),
    )
    target_regions = assign_patch_regions(mask, target_class_id=0, patch_size=16)[
        "region_codes"
    ]
    assert target_regions[0, 0] == REGION_BACKGROUND
    assert target_regions[0, 1] == REGION_TARGET
    assert target_regions[1, 0] == REGION_MIXED
    # The final patch has exactly 50% valid pixels and a unique 100%-of-valid
    # semantic owner, so it is not void under Experiment 2's inclusive rule.
    assert target_regions[1, 1] != REGION_VOID
    assert labels[1, 1] != PATCH_LABEL_VOID


def test_b2_b3_graph_weights_are_orthogonal_basis_invariant():
    generator = torch.Generator().manual_seed(202)
    tokens = torch.randn(2, 12, 8, generator=generator, dtype=torch.float64)
    edges = local_edge_index((3, 4))
    rotation = torch.from_numpy(generate_basis_transforms(8)[3].matrix)
    first = construct_local_graphs(tokens, edges)
    second = construct_local_graphs(tokens @ rotation, edges)
    for name in ("B2_feature", "B3_spatial_feature"):
        torch.testing.assert_close(first[name], second[name], rtol=1e-11, atol=1e-11)


def test_bootstrap_resamples_whole_images_and_reuses_paired_draws():
    draws = make_cluster_draws(["a", "b", "c"], repeats=20, seed=203)
    numerator = np.array([1.0, 10.0, 100.0])
    denominator = np.array([2.0, 20.0, 200.0])
    observed = ratio_bootstrap(draws, numerator, denominator)
    expected_first = (
        draws.multiplicities[0].astype(float) @ numerator
    ) / (draws.multiplicities[0].astype(float) @ denominator)
    assert observed[0] == expected_first
    np.testing.assert_allclose(observed, 0.5)
    paired = ratio_bootstrap(draws, numerator * 2, denominator)
    np.testing.assert_allclose(paired - observed, 0.5)
    assert np.all(draws.multiplicities.sum(axis=1) == 3)


def test_histogram_cluster_bootstrap_matches_expanded_first_draw():
    image = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    target = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    score = np.array([0.1, 0.8, 0.9, 0.3, 0.4, 0.7, 0.2, 0.6])
    draws = make_cluster_draws(["a", "b", "c"], repeats=8, seed=204)
    positive, negative = rank_histograms_by_image(
        image, target, score, num_images=3, bins=101
    )
    auc, ap = histogram_rank_bootstrap(
        draws, positive, negative, device="cpu"
    )
    multiplicity = draws.multiplicities[0]
    expanded_target = np.concatenate(
        [target[image == index] for index in range(3) for _ in range(multiplicity[index])]
    )
    expanded_score = np.concatenate(
        [score[image == index] for index in range(3) for _ in range(multiplicity[index])]
    )
    assert auc[0] == roc_auc_score(expanded_target, expanded_score)
    assert ap[0] == average_precision_score(expanded_target, expanded_score)
    exact_auc, exact_ap = exact_rank_metrics(target, score)
    assert exact_auc == roc_auc_score(target, score)
    assert exact_ap == average_precision_score(target, score)


def test_full_voc_grid_has_expected_undirected_edge_count():
    assert local_edge_index((28, 28)).count == 2970


def test_metric_runner_skips_empty_label_count_strata_for_smoke_subsets():
    edges = local_edge_index((2, 2))
    image_ids = ["one", "two"]
    labels = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.int8)
    weights = {
        name: np.full((2, edges.count), 0.5 + 0.1 * index, dtype=np.float32)
        for index, name in enumerate(GRAPH_NAMES)
    }
    rows, _, _, audit = evaluate_graph_metrics(
        image_ids=image_ids,
        image_label_counts=np.array([1, 1], dtype=np.uint8),
        semantic_labels=labels,
        weights=weights,
        edges=edges,
        repeats=4,
        seed=205,
        bins=16,
        bootstrap_device="cpu",
    )
    assert rows
    assert {row["stratum"] for row in rows}.issubset({"all", "single_label"})
    assert audit["image_counts_by_stratum"]["two_label"] == 0
