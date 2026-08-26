"""Token-group attention normalization shared by TGCA host models."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


SUPPORTED_MODES = frozenset(
    {"vanilla", "split_11", "split_05", "tgca", "tgca_bias", "tgca_gamma05"}
)


def build_mctformer_groups(
    num_classes: int,
    num_patches: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Return query/key group IDs for ``[class tokens, patch tokens]``."""
    if num_classes <= 0 or num_patches <= 0:
        raise ValueError("MCTformer+ requires positive class and patch token counts")
    group_ids = torch.cat(
        (
            torch.zeros(num_classes, dtype=torch.long, device=device),
            torch.ones(num_patches, dtype=torch.long, device=device),
        )
    )
    return group_ids, group_ids


def _expand_group_ids(group_ids: Tensor, batch_size: int, length: int, name: str) -> Tensor:
    if group_ids.ndim == 1:
        if group_ids.shape[0] != length:
            raise ValueError(f"{name} has length {group_ids.shape[0]}, expected {length}")
        group_ids = group_ids.unsqueeze(0).expand(batch_size, -1)
    elif group_ids.ndim == 2:
        if group_ids.shape != (batch_size, length):
            raise ValueError(
                f"{name} has shape {tuple(group_ids.shape)}, expected {(batch_size, length)}"
            )
    else:
        raise ValueError(f"{name} must have shape [N] or [B, N]")
    if group_ids.dtype == torch.bool or torch.is_floating_point(group_ids):
        raise TypeError(f"{name} must use an integer dtype")
    group_ids = group_ids.to(dtype=torch.long)
    if torch.any(group_ids < 0):
        raise ValueError(f"{name} cannot contain negative group IDs")
    return group_ids


def _expand_valid_mask(mask: Optional[Tensor], logits: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[None, None, None, :]
    elif mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    try:
        return torch.broadcast_to(mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"key_valid_mask shape {tuple(mask.shape)} is not broadcastable to "
            f"logits shape {tuple(logits.shape)}"
        ) from exc


def _masked_softmax(logits: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
    if valid_mask is None:
        return torch.softmax(logits, dim=-1)
    if torch.any(~valid_mask.any(dim=-1)):
        raise ValueError("Every attention row must contain at least one valid key")
    masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
    probabilities = torch.softmax(masked_logits, dim=-1)
    return probabilities.masked_fill(~valid_mask, 0.0)


def token_group_normalize(
    logits: Tensor,
    key_group_ids: Tensor,
    query_group_ids: Optional[Tensor] = None,
    key_valid_mask: Optional[Tensor] = None,
    mode: str = "tgca",
    gamma: float = 1.0,
    split_weights: Optional[Sequence[float]] = None,
    relation_bias: Optional[Tensor] = None,
) -> Tensor:
    """Normalize attention logits over heterogeneous key-token groups.

    All corrections and softmax accumulation are performed in float32. The
    returned probabilities use the input logit's dtype and are pre-dropout.
    """
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, H, Nq, Nk]")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown attention normalization mode: {mode!r}")

    batch_size, num_heads, num_queries, num_keys = logits.shape
    key_group_ids = _expand_group_ids(
        key_group_ids.to(device=logits.device), batch_size, num_keys, "key_group_ids"
    )
    if query_group_ids is None:
        query_group_ids = torch.zeros(
            (batch_size, num_queries), dtype=torch.long, device=logits.device
        )
    else:
        query_group_ids = _expand_group_ids(
            query_group_ids.to(device=logits.device),
            batch_size,
            num_queries,
            "query_group_ids",
        )

    valid_mask = (
        None if key_valid_mask is None else _expand_valid_mask(key_valid_mask, logits)
    )
    logits_fp32 = logits.float()
    num_key_groups = int(key_group_ids.max().item()) + 1
    num_query_groups = int(query_group_ids.max().item()) + 1

    if mode == "vanilla":
        probabilities = _masked_softmax(logits_fp32, valid_mask)
        return probabilities.to(dtype=logits.dtype)

    key_membership = torch.nn.functional.one_hot(
        key_group_ids, num_classes=num_key_groups
    ).to(dtype=torch.bool)
    if valid_mask is None:
        group_counts = key_membership.sum(dim=1)[:, None, None, :].expand(
            batch_size, num_heads, num_queries, num_key_groups
        )
        row_group_mask = None
    else:
        expanded_membership = key_membership[:, None, None, :, :]
        row_group_mask = valid_mask.unsqueeze(-1) & expanded_membership
        group_counts = row_group_mask.sum(dim=-2)

    if mode.startswith("split_"):
        if num_key_groups != 2:
            raise ValueError("Split softmax baselines require exactly two key groups")
        if torch.any(group_counts == 0):
            raise ValueError("Split softmax requires every key group to be present in every row")
        if split_weights is None:
            split_weights = (1.0, 1.0) if mode == "split_11" else (0.5, 0.5)
        if len(split_weights) != num_key_groups:
            raise ValueError("split_weights must contain one weight per key group")
        probabilities = torch.zeros_like(logits_fp32)
        for group_index, group_weight in enumerate(split_weights):
            if row_group_mask is None:
                group_mask = key_membership[..., group_index][
                    :, None, None, :
                ].expand(batch_size, num_heads, num_queries, num_keys)
            else:
                group_mask = row_group_mask[..., group_index]
            group_probabilities = _masked_softmax(logits_fp32, group_mask)
            probabilities = probabilities + float(group_weight) * group_probabilities
        return probabilities.to(dtype=logits.dtype)

    effective_gamma = 0.5 if mode == "tgca_gamma05" else float(gamma)
    if not math.isfinite(effective_gamma) or effective_gamma < 0:
        raise ValueError("gamma must be finite and non-negative")

    safe_counts = group_counts.clamp_min(1).to(dtype=torch.float32)
    count_log_by_key = torch.gather(
        safe_counts,
        dim=-1,
        index=key_group_ids[:, None, None, :].expand(
            batch_size, num_heads, num_queries, num_keys
        ),
    ).log()
    corrected_logits = logits_fp32 - effective_gamma * count_log_by_key

    if mode == "tgca_bias":
        if relation_bias is None:
            raise ValueError("tgca_bias requires relation_bias")
        if relation_bias.shape != (num_heads, num_query_groups, num_key_groups):
            raise ValueError(
                "relation_bias has shape "
                f"{tuple(relation_bias.shape)}, expected "
                f"{(num_heads, num_query_groups, num_key_groups)}"
            )
        bias = relation_bias.float()[None].expand(batch_size, -1, -1, -1)
        query_index = query_group_ids[:, None, :, None].expand(
            batch_size, num_heads, num_queries, num_key_groups
        )
        bias_by_query = torch.gather(bias, dim=2, index=query_index)
        bias_by_key = torch.gather(
            bias_by_query,
            dim=3,
            index=key_group_ids[:, None, :].expand(
                batch_size, num_heads, num_keys
            )[:, :, None, :].expand(batch_size, num_heads, num_queries, num_keys),
        )
        corrected_logits = corrected_logits + bias_by_key
    elif relation_bias is not None:
        raise ValueError("relation_bias is only valid in tgca_bias mode")

    probabilities = _masked_softmax(corrected_logits, valid_mask)
    return probabilities.to(dtype=logits.dtype)


class TokenGroupNormalizer(nn.Module):
    """Module wrapper around :func:`token_group_normalize`."""

    def __init__(
        self,
        num_heads: int,
        num_query_groups: int,
        num_key_groups: int,
        mode: str = "vanilla",
        gamma: float = 1.0,
        split_weights: Sequence[float] = (1.0, 1.0),
        learn_relation_bias: bool = False,
    ) -> None:
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown attention normalization mode: {mode!r}")
        if mode == "tgca_bias" and not learn_relation_bias:
            learn_relation_bias = True
        if learn_relation_bias and mode != "tgca_bias":
            raise ValueError("Relation bias can only be learned in tgca_bias mode")
        self.num_heads = int(num_heads)
        self.num_query_groups = int(num_query_groups)
        self.num_key_groups = int(num_key_groups)
        self.mode = mode
        self.gamma = float(gamma)
        if mode == "split_11":
            self.split_weights = (1.0, 1.0)
        elif mode == "split_05":
            self.split_weights = (0.5, 0.5)
        else:
            self.split_weights = tuple(float(value) for value in split_weights)
        if learn_relation_bias:
            self.relation_bias = nn.Parameter(
                torch.zeros(
                    self.num_heads, self.num_query_groups, self.num_key_groups
                )
            )
        else:
            self.register_parameter("relation_bias", None)

    def forward(
        self,
        logits: Tensor,
        key_group_ids: Tensor,
        query_group_ids: Optional[Tensor] = None,
        key_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return token_group_normalize(
            logits=logits,
            key_group_ids=key_group_ids,
            query_group_ids=query_group_ids,
            key_valid_mask=key_valid_mask,
            mode=self.mode,
            gamma=self.gamma,
            split_weights=self.split_weights,
            relation_bias=self.relation_bias,
        )

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, num_heads={self.num_heads}, "
            f"num_query_groups={self.num_query_groups}, "
            f"num_key_groups={self.num_key_groups}, gamma={self.gamma}"
        )
