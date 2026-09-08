"""Pure final-token and positive-channel relation definitions.

This module deliberately knows nothing about VOC masks, datasets, attention, or
CAM refinement.  It operates only on raw post-Block-12 class/patch tokens.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


EPSILON = 1e-12


def _validate_tokens(class_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> None:
    if class_tokens.ndim != 3 or patch_tokens.ndim != 3:
        raise ValueError("class and patch tokens must have shape [B,C,D] / [B,N,D]")
    if class_tokens.shape[0] != patch_tokens.shape[0]:
        raise ValueError("class and patch token batch dimensions differ")
    if class_tokens.shape[-1] != patch_tokens.shape[-1]:
        raise ValueError("class and patch token embedding dimensions differ")
    if class_tokens.shape[1] < 1 or patch_tokens.shape[1] < 1:
        raise ValueError("class and patch token axes must be non-empty")
    if not torch.isfinite(class_tokens).all() or not torch.isfinite(patch_tokens).all():
        raise ValueError("raw token inputs must be finite")


def final_token_relations(
    class_tokens: torch.Tensor, patch_tokens: torch.Tensor, *, eps: float = EPSILON
) -> Mapping[str, torch.Tensor]:
    """Return raw final-token relations with consistent ``[B,C,N]`` shapes.

    ``s_last`` and its positive/negative decomposition include the required
    ``sqrt(D)`` scale.  ``s_dot`` is retained only as the explicitly requested
    unscaled comparison; ranking is identical to ``s_last``.
    """

    _validate_tokens(class_tokens, patch_tokens)
    if eps <= 0:
        raise ValueError("eps must be positive")
    # Production raw tokens are float32.  The explicit float64 branch is used
    # only by the orthogonal-basis regression, where rounding must not mask the
    # algebraic invariance being tested.
    work_dtype = (
        torch.float64
        if class_tokens.dtype == torch.float64 or patch_tokens.dtype == torch.float64
        else torch.float32
    )
    classes = class_tokens.to(dtype=work_dtype)
    patches = patch_tokens.to(dtype=work_dtype)
    scale = math.sqrt(classes.shape[-1])
    positive = torch.relu(classes)
    negative_magnitude = torch.relu(-classes)
    dot = torch.einsum("bcd,bnd->bcn", classes, patches)
    s_pos = torch.einsum("bcd,bnd->bcn", positive, patches) / scale
    s_negmag = torch.einsum("bcd,bnd->bcn", negative_magnitude, patches) / scale
    class_unit = F.normalize(classes, p=2, dim=-1, eps=eps)
    patch_unit = F.normalize(patches, p=2, dim=-1, eps=eps)
    cosine = torch.einsum("bcd,bnd->bcn", class_unit, patch_unit)
    patch_norm = torch.linalg.vector_norm(patches, dim=-1).unsqueeze(1).expand_as(dot)
    return {
        "s_dot": dot,
        "s_last": dot / scale,
        "s_pos": s_pos,
        "s_negmag": s_negmag,
        "s_cosine": cosine,
        "patch_norm": patch_norm,
        "positive_channels": positive,
        "negative_channels": negative_magnitude,
    }


def positive_channel_statistics(class_tokens: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Return native mean-readout contributions for every class token."""

    if class_tokens.ndim != 3 or class_tokens.shape[-1] < 1:
        raise ValueError("class_tokens must have shape [B,C,D] with D>0")
    values = class_tokens.float()
    positive = torch.relu(values)
    negative = torch.relu(-values)
    width = values.shape[-1]
    pos_mass = positive.sum(dim=-1)
    neg_mass = negative.sum(dim=-1)
    logits = values.mean(dim=-1)
    return {
        "positive_fraction": (values > 0).float().mean(dim=-1),
        "positive_mass": pos_mass,
        "negative_mass": neg_mass,
        "positive_negative_ratio": pos_mass / neg_mass.clamp_min(EPSILON),
        "native_logits": logits,
        "logit_identity_error": (width * logits - (pos_mass - neg_mass)).abs(),
    }


def spatial_probability(scores: torch.Tensor) -> torch.Tensor:
    """Spatial softmax used only for probability-mass diagnostics."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [B,C,N]")
    return torch.softmax(scores.float(), dim=-1)


def classifier_relevance(raw_classifier_map: torch.Tensor, *, eps: float = EPSILON) -> torch.Tensor:
    """Native 3x3 classifier reference ``ReLU(M) / max_j ReLU(M)``."""

    if raw_classifier_map.ndim != 3:
        raise ValueError("raw_classifier_map must have shape [B,C,N]")
    positive = torch.relu(raw_classifier_map.float())
    return positive / positive.amax(dim=-1, keepdim=True).clamp_min(eps)


def apply_shared_basis(
    class_tokens: torch.Tensor, patch_tokens: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a common right-orthogonal embedding transform ``Q``."""

    _validate_tokens(class_tokens, patch_tokens)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("basis matrix must be square")
    if matrix.shape[0] != class_tokens.shape[-1]:
        raise ValueError("basis dimension differs from token width")
    q = matrix.to(device=class_tokens.device, dtype=class_tokens.dtype)
    return class_tokens @ q, patch_tokens @ q


def transformed_mean_readout(
    class_tokens: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits under the mathematically equivalent transformed readout."""

    if class_tokens.ndim != 3:
        raise ValueError("class_tokens must have shape [B,C,D]")
    if matrix.ndim != 2 or matrix.shape != (class_tokens.shape[-1],) * 2:
        raise ValueError("matrix width differs from class token width")
    values = class_tokens.to(dtype=matrix.dtype, device=matrix.device)
    transformed = values @ matrix
    readout = torch.full(
        (values.shape[-1],), 1.0 / values.shape[-1],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    transformed_readout = matrix.transpose(0, 1) @ readout
    return values @ readout, transformed @ transformed_readout


def positive_coordinate_patch_statistics(
    class_tokens: torch.Tensor, patch_tokens: torch.Tensor
) -> Mapping[str, torch.Tensor]:
    """Compute U_pos and C_sign over each class token's positive coordinates."""

    _validate_tokens(class_tokens, patch_tokens)
    mask = class_tokens > 0
    count = mask.sum(dim=-1)
    expanded = mask.unsqueeze(2).expand(-1, -1, patch_tokens.shape[1], -1)
    patches = patch_tokens.float()
    numerator = (patches.unsqueeze(1) * mask.unsqueeze(2).float()).sum(dim=-1)
    denominator = count.unsqueeze(-1).float()
    mean = numerator / denominator.clamp_min(1.0)
    sign = (
        ((patches.unsqueeze(1) > 0) & expanded).sum(dim=-1).float()
        / denominator.clamp_min(1.0)
    )
    no_positive = count == 0
    if bool(no_positive.any()):
        mean = mean.masked_fill(no_positive.unsqueeze(-1), float("nan"))
        sign = sign.masked_fill(no_positive.unsqueeze(-1), float("nan"))
    return {"u_pos": mean, "c_sign": sign, "positive_channel_count": count}
