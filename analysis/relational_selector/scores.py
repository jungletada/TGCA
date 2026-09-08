"""GT-free frozen selector score definitions.

Every function here consumes only model representations or classifier outputs.
In particular, segmentation masks and image-level labels are deliberately not
accepted by this API.  All relation arithmetic is accumulated in float32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from analysis.lazy_assignment.score_utils import class_specific_patch_score
from analysis.semantic_relations import spatial_minmax


EPSILON = 1e-12


def _validate_tokens(class_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> tuple[int, int, int]:
    if class_tokens.ndim != 3 or patch_tokens.ndim != 3:
        raise ValueError("class_tokens and patch_tokens must be [B,C,D] and [B,P,D]")
    batch, classes, width = class_tokens.shape
    if patch_tokens.shape[0] != batch or patch_tokens.shape[2] != width:
        raise ValueError("class and patch token batch/embedding dimensions must agree")
    if classes < 2 or patch_tokens.shape[1] < 1 or width < 1:
        raise ValueError("requires at least two class tokens and one patch token")
    return int(batch), int(classes), int(width)


def raw_dot_scores(class_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
    """Return raw class--patch dot products divided by ``sqrt(D)`` in float32."""

    _, _, width = _validate_tokens(class_tokens, patch_tokens)
    score = torch.matmul(class_tokens.float(), patch_tokens.float().transpose(1, 2))
    score = score / math.sqrt(width)
    if not torch.isfinite(score).all():
        raise RuntimeError("raw class--patch relation contains non-finite values")
    return score


def centered_dot_scores(class_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
    """Subtract the all-class mean token before the raw dot-product relation."""

    _validate_tokens(class_tokens, patch_tokens)
    centered = class_tokens.float() - class_tokens.float().mean(dim=1, keepdim=True)
    return raw_dot_scores(centered, patch_tokens)


def relative_ownership_scores(q: torch.Tensor) -> torch.Tensor:
    """Class-vs-all-other class log-relative patch ownership score.

    ``q`` is indexed as ``[batch, class, patch]``.  The denominator is the
    stable log-mean-exp over all *other* classes, evaluated independently at
    every patch.  It is invariant to a common additive shift across classes.
    """

    if q.ndim != 3 or q.shape[1] < 2:
        raise ValueError("q must have shape [B,C,P] with C >= 2")
    values = q.float()
    batch, classes, patches = values.shape
    # [B, query-class, key-class, patch]; the key diagonal is excluded.
    expanded = values.unsqueeze(1).expand(batch, classes, classes, patches)
    diagonal = torch.eye(classes, device=values.device, dtype=torch.bool).view(1, classes, classes, 1)
    others = expanded.masked_fill(diagonal, float("-inf"))
    log_mean_other = torch.logsumexp(others, dim=2) - math.log(classes - 1)
    result = values - log_mean_other
    if not torch.isfinite(result).all():
        raise RuntimeError("relative ownership score contains non-finite values")
    return result


def build_selector_scores(
    class_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    raw_classifier_map: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Build the six preregistered frozen selector maps S0--S5.

    S0 must be the native raw 3x3 classifier output, before ReLU, min--max,
    GWRP, or any GT operation.  S1 is the existing cosine feature score. S2
    through S5 follow the exact frozen plan.  No function argument can carry
    semantic GT or image labels, which makes GT leakage structurally impossible
    at score construction.
    """

    batch, classes, _ = _validate_tokens(class_tokens, patch_tokens)
    raw = raw_classifier_map.float()
    if raw.shape != (batch, classes, patch_tokens.shape[1]):
        raise ValueError(
            "raw_classifier_map must have shape [B,C,P] matching class/patch tokens; "
            f"got {tuple(raw.shape)}"
        )
    if not torch.isfinite(raw).all():
        raise ValueError("raw_classifier_map contains non-finite values")
    s1 = class_specific_patch_score(class_tokens, patch_tokens)
    s2 = raw_dot_scores(class_tokens, patch_tokens)
    s3 = centered_dot_scores(class_tokens, patch_tokens)
    s4 = relative_ownership_scores(s2)
    s5 = spatial_minmax(raw) + spatial_minmax(s4)
    scores: dict[str, torch.Tensor] = {
        "S0": raw,
        "S1": s1.float(),
        "S2": s2,
        "S3": s3,
        "S4": s4,
        "S5": s5.float(),
    }
    for name, value in scores.items():
        if value.dtype != torch.float32 or value.shape != raw.shape or not torch.isfinite(value).all():
            raise RuntimeError(f"{name} violates the float32 finite [B,C,P] contract")
    return scores
