"""Deterministic contracts for Phase C-D graph low-pass diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from analysis.spatial_graph_stability.graph import LocalEdges, local_edge_index
from analysis.spatial_graph_stability.lowpass import (
    bounded_stability,
    graph_energy,
    graph_lowpass,
    normalized_edge_weights,
)
from analysis.spatial_graph_stability.phase_metrics import (
    class_region_labels,
    within_map_quintiles,
)


def _dense_solution(values: torch.Tensor, weights: torch.Tensor, edges: LocalEdges, lam: float) -> torch.Tensor:
    batch, nodes, _ = values.shape
    adjacency = torch.zeros((batch, nodes, nodes), dtype=values.dtype)
    adjacency[:, edges.source, edges.target] = weights
    adjacency[:, edges.target, edges.source] = weights
    degree = adjacency.sum(-1)
    normalized = adjacency * torch.rsqrt(degree).unsqueeze(2) * torch.rsqrt(degree).unsqueeze(1)
    system = torch.eye(nodes, dtype=values.dtype).unsqueeze(0) + lam * (torch.eye(nodes, dtype=values.dtype).unsqueeze(0) - normalized)
    return torch.linalg.solve(system, values)


def test_sparse_cg_matches_direct_symmetric_normalized_solution():
    torch.manual_seed(11)
    edges = local_edge_index((4, 4))
    values = torch.randn(2, 16, 3, dtype=torch.float64)
    weights = torch.rand(2, edges.count, dtype=torch.float64) + 0.2
    sparse = graph_lowpass(values, weights, edges, lambda_value=1.0, tolerance=1e-6)
    dense = _dense_solution(values, weights, edges, 1.0)
    assert sparse.relative_residual.max().item() < 1e-5
    assert torch.allclose(sparse.values, dense, atol=3e-5, rtol=3e-5)


def test_lowpass_reduces_normalized_dirichlet_energy_and_stability_is_bounded():
    torch.manual_seed(12)
    edges = local_edge_index((5, 5))
    values = torch.randn(1, 25, 4)
    weights = torch.rand(1, edges.count) + 0.1
    result = graph_lowpass(values, weights, edges, lambda_value=2.0, tolerance=9e-6)
    normalized, _ = normalized_edge_weights(weights, edges)
    assert torch.all(graph_energy(result.values, normalized, edges) <= graph_energy(values, normalized, edges) + 1e-6)
    residual, scale, stability = bounded_stability(values, result.values)
    assert torch.isfinite(residual).all() and torch.isfinite(scale).all() and torch.isfinite(stability).all()
    assert torch.all(stability > 0) and torch.all(stability <= 1)


def test_lowpass_is_scale_equivariant_and_stability_is_scale_invariant():
    torch.manual_seed(13)
    edges = local_edge_index((4, 4))
    values = torch.randn(1, 16, 2, dtype=torch.float64)
    weights = torch.rand(1, edges.count, dtype=torch.float64) + 0.25
    original = graph_lowpass(values, weights, edges, lambda_value=1.0, tolerance=1e-6)
    scaled = graph_lowpass(7.0 * values, weights, edges, lambda_value=1.0, tolerance=1e-6)
    _, _, first_stability = bounded_stability(values, original.values)
    _, _, second_stability = bounded_stability(7.0 * values, scaled.values)
    assert torch.allclose(scaled.values, 7.0 * original.values, atol=5e-5, rtol=5e-5)
    assert torch.allclose(first_stability, second_stability, atol=5e-5, rtol=5e-5)


def test_phase_cd_region_labels_and_quintiles_obey_contract():
    semantic = np.asarray([0, 1, 2, -1, -2, 1], dtype=np.int8)
    assert class_region_labels(semantic, 0).tolist() == [2, 0, 1, 3, 4, 0]
    values = np.arange(10, dtype=float)
    quintiles = within_map_quintiles(values, np.ones(10, dtype=bool))
    assert set(quintiles.tolist()) == {0, 1, 2, 3, 4}
    # Highest values receive high relevance/stability quintiles.
    assert quintiles[-1] == 4 and quintiles[0] == 0
