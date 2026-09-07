"""Sparse local-graph low-pass filtering for the Phase C--D diagnostics.

The implementation deliberately operates on the undirected local edge list
rather than materialising a dense 784 x 784 adjacency.  It uses batched
conjugate gradients to solve every class map of an image simultaneously:

    (I + lambda * L_sym) X = M.

No matrix inverse, GT signal, or learned parameter is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import LocalEdges


@dataclass(frozen=True)
class LowPassResult:
    """Result and numerical audit for one batched graph solve."""

    values: torch.Tensor
    relative_residual: torch.Tensor  # [B, C]
    iterations: int
    converged: torch.Tensor  # [B, C]


def _edge_tensors(edges: LocalEdges, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(edges.source, dtype=torch.long, device=device),
        torch.as_tensor(edges.target, dtype=torch.long, device=device),
    )


def _validate(values: torch.Tensor, weights: torch.Tensor, edges: LocalEdges) -> None:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, C]")
    if weights.ndim != 2:
        raise ValueError("weights must have shape [B, E]")
    batch, nodes, _ = values.shape
    if nodes != edges.grid_size[0] * edges.grid_size[1]:
        raise ValueError("value node count does not match edge grid")
    if weights.shape != (batch, edges.count):
        raise ValueError("weights must have one local-edge weight per batch item")
    if not torch.is_floating_point(values) or not torch.is_floating_point(weights):
        raise TypeError("graph low-pass inputs must be floating tensors")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise ValueError("graph low-pass inputs contain NaN or Inf")
    if torch.any(weights < 0):
        raise ValueError("symmetric graph weights must be non-negative")


def normalized_edge_weights(
    weights: torch.Tensor, edges: LocalEdges, *, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return D^{-1/2} W D^{-1/2} edge weights and graph degrees.

    ``weights`` contains each undirected edge once.  The returned degree has
    shape ``[B, N]`` and is computed by scattering the same edge weight to both
    incident nodes.
    """

    if weights.ndim != 2 or weights.shape[1] != edges.count:
        raise ValueError("weights must have shape [B, E]")
    work_dtype = dtype or weights.dtype
    work = weights.to(dtype=work_dtype)
    source, target = _edge_tensors(edges, work.device)
    batch, _ = work.shape
    nodes = edges.grid_size[0] * edges.grid_size[1]
    degree = work.new_zeros((batch, nodes))
    index_source = source.view(1, -1).expand(batch, -1)
    index_target = target.view(1, -1).expand(batch, -1)
    degree.scatter_add_(1, index_source, work)
    degree.scatter_add_(1, index_target, work)
    if torch.any(degree <= 0) or not torch.isfinite(degree).all():
        raise RuntimeError("local graph has a non-positive or non-finite degree")
    inv_sqrt = torch.rsqrt(degree)
    normalized = work * inv_sqrt[:, source] * inv_sqrt[:, target]
    return normalized, degree


def normalized_laplacian_apply(
    values: torch.Tensor,
    normalized_weights: torch.Tensor,
    edges: LocalEdges,
) -> torch.Tensor:
    """Apply ``L_sym = I - D^{-1/2} W D^{-1/2}`` without a dense matrix."""

    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, C]")
    if normalized_weights.ndim != 2 or normalized_weights.shape[0] != values.shape[0]:
        raise ValueError("normalized weights must have shape [B, E]")
    if normalized_weights.shape[1] != edges.count:
        raise ValueError("normalized weights do not match edge list")
    source, target = _edge_tensors(edges, values.device)
    batch, nodes, channels = values.shape
    if nodes != edges.grid_size[0] * edges.grid_size[1]:
        raise ValueError("values do not match edge grid")
    weight = normalized_weights.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    adjacency_values = values.new_zeros((batch, nodes, channels))
    source_index = source.view(1, -1, 1).expand(batch, -1, channels)
    target_index = target.view(1, -1, 1).expand(batch, -1, channels)
    adjacency_values.scatter_add_(1, source_index, weight * values[:, target])
    adjacency_values.scatter_add_(1, target_index, weight * values[:, source])
    return values - adjacency_values


def graph_energy(
    values: torch.Tensor,
    normalized_weights: torch.Tensor,
    edges: LocalEdges,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return normalized Dirichlet energy per image and class, shape [B, C]."""

    laplacian_values = normalized_laplacian_apply(values, normalized_weights, edges)
    numerator = (values * laplacian_values).sum(dim=1)
    denominator = values.square().sum(dim=1).clamp_min(float(epsilon))
    result = numerator / denominator
    # Numerical roundoff can make a positive-semidefinite energy infinitesimally
    # negative.  Preserve meaningful values while removing that artifact.
    return result.clamp_min(0.0)


def graph_lowpass(
    values: torch.Tensor,
    weights: torch.Tensor,
    edges: LocalEdges,
    *,
    lambda_value: float,
    tolerance: float = 1e-7,
    max_iterations: int = 256,
    epsilon: float = 1e-12,
) -> LowPassResult:
    """Solve the symmetric normalized graph low-pass system with CG.

    The convergence condition is evaluated independently for every
    image/class right-hand side.  The returned residual is recomputed from the
    final solution, rather than relying on the iterative residual alone.
    """

    _validate(values, weights, edges)
    lambda_value = float(lambda_value)
    tolerance = float(tolerance)
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")

    original_dtype = values.dtype
    work_dtype = torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype
    rhs = values.to(dtype=work_dtype)
    normalized, _ = normalized_edge_weights(weights, edges, dtype=work_dtype)

    if lambda_value == 0.0:
        zero = torch.zeros((rhs.shape[0], rhs.shape[2]), dtype=rhs.dtype, device=rhs.device)
        return LowPassResult(rhs.to(dtype=original_dtype), zero, 0, torch.ones_like(zero, dtype=torch.bool))

    def system_apply(argument: torch.Tensor) -> torch.Tensor:
        return argument + lambda_value * normalized_laplacian_apply(
            argument, normalized, edges
        )

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rhs_norm = torch.sqrt(rhs.square().sum(dim=1)).clamp_min(float(epsilon))
    residual_sq = residual.square().sum(dim=1)
    converged = torch.sqrt(residual_sq) / rhs_norm <= tolerance
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        system_direction = system_apply(direction)
        denominator = (direction * system_direction).sum(dim=1)
        valid_denominator = denominator.abs() > float(epsilon)
        safe_denominator = torch.where(
            valid_denominator, denominator, torch.ones_like(denominator)
        )
        alpha = torch.where(
            valid_denominator, residual_sq / safe_denominator,
            torch.zeros_like(residual_sq),
        )
        active = ~converged
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        solution = solution + direction * alpha.unsqueeze(1)
        residual = residual - system_direction * alpha.unsqueeze(1)
        next_residual_sq = residual.square().sum(dim=1)
        converged = converged | (torch.sqrt(next_residual_sq) / rhs_norm <= tolerance)
        iterations = iteration
        if bool(converged.all()):
            residual_sq = next_residual_sq
            break
        valid_residual = residual_sq > float(epsilon)
        safe_residual = torch.where(
            valid_residual, residual_sq, torch.ones_like(residual_sq)
        )
        beta = torch.where(
            valid_residual, next_residual_sq / safe_residual,
            torch.zeros_like(next_residual_sq),
        )
        beta = torch.where(converged, torch.zeros_like(beta), beta)
        direction = residual + direction * beta.unsqueeze(1)
        residual_sq = next_residual_sq

    final_residual = system_apply(solution) - rhs
    relative = torch.sqrt(final_residual.square().sum(dim=1)) / rhs_norm
    final_converged = relative <= tolerance
    if not torch.isfinite(solution).all() or not torch.isfinite(relative).all():
        raise RuntimeError("graph low-pass CG produced NaN or Inf")
    return LowPassResult(
        solution.to(dtype=original_dtype), relative, iterations, final_converged
    )


def bounded_stability(
    values: torch.Tensor,
    smoothed: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute residual, within-map median scale, and bounded stability.

    Inputs use ``[B, N, C]`` layout.  The scale is one median per image/class,
    so a response's amplitude cannot by itself create a denominator singularity.
    """

    if values.shape != smoothed.shape or values.ndim != 3:
        raise ValueError("values and smoothed must share [B, N, C] shape")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    residual = (values - smoothed).abs()
    scale = residual.median(dim=1).values + epsilon
    stability = 1.0 / (1.0 + residual / scale.unsqueeze(1))
    if not torch.isfinite(stability).all() or torch.any(stability <= 0) or torch.any(stability > 1):
        raise RuntimeError("bounded graph stability is outside (0, 1]")
    return residual, scale, stability
