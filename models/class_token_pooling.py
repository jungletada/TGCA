"""Class-wise weighted pooling for image-conditioned class-token initialization."""

import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ClassWiseWeightedPooling(nn.Module):
    """Pool patch tokens independently for every semantic class.

    The learnable queries are class-specific, while the softmax competition is
    restricted to patch locations.  Logits, normalization, and accumulation
    use float32 for mixed-precision stability; outputs are returned in the
    input patch-token dtype.
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        if self.num_classes < 1 or self.embed_dim < 1:
            raise ValueError('num_classes and embed_dim must be positive')
        self.class_queries = nn.Parameter(
            torch.empty(self.num_classes, self.embed_dim)
        )
        trunc_normal_(self.class_queries, std=0.02)

    def forward(self, patch_tokens: torch.Tensor):
        if patch_tokens.ndim != 3:
            raise ValueError(
                'patch_tokens must have shape [B, N, D], got '
                f'{tuple(patch_tokens.shape)}'
            )
        if patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f'expected embedding width {self.embed_dim}, got '
                f'{patch_tokens.shape[-1]}'
            )
        if patch_tokens.shape[1] < 1:
            raise ValueError('class-wise pooling requires at least one patch')

        with torch.cuda.amp.autocast(enabled=False):
            values = patch_tokens.float()
            queries = self.class_queries.float().unsqueeze(0).expand(
                values.shape[0], -1, -1
            )
            logits = torch.matmul(queries, values.transpose(1, 2)) / math.sqrt(
                self.embed_dim
            )
            attention = torch.softmax(logits, dim=-1)
            pooled = torch.matmul(attention, values)
        return pooled.to(patch_tokens.dtype), attention


class ResidualClassWiseWeightedPooling(ClassWiseWeightedPooling):
    """Class-wise pooling with the fixed first-round learnable residual scale."""

    def __init__(
            self, num_classes: int, embed_dim: int, initial_alpha: float = 0.1):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))


def class_token_pooling_diagnostics(
        attention: torch.Tensor, initial_class_tokens: torch.Tensor):
    """Return detached, loss-free CWP diagnostics for one mini-batch."""
    if attention.ndim != 3 or initial_class_tokens.ndim != 3:
        raise ValueError('CWP diagnostics expect [B,C,N] and [B,C,D] tensors')
    if attention.shape[:2] != initial_class_tokens.shape[:2]:
        raise ValueError('CWP attention and class-token shapes are inconsistent')

    with torch.no_grad():
        weights = attention.float()
        tokens = initial_class_tokens.float()
        normalized_entropy = -(
            weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        ).sum(dim=-1) / math.log(weights.shape[-1])

        def off_diagonal_cosine(values):
            values = torch.nn.functional.normalize(values, dim=-1)
            matrix = torch.matmul(values, values.transpose(1, 2))
            classes = matrix.shape[-1]
            mask = ~torch.eye(
                classes, dtype=torch.bool, device=matrix.device
            ).unsqueeze(0)
            return matrix.masked_select(mask).mean()

        return {
            'cwp_entropy_mean': normalized_entropy.mean().item(),
            'cwp_entropy_std': normalized_entropy.std(unbiased=False).item(),
            'cwp_entropy_min': normalized_entropy.min().item(),
            'cwp_entropy_max': normalized_entropy.max().item(),
            'cwp_attention_interclass_cosine': off_diagonal_cosine(weights).item(),
            'cwp_initial_token_interclass_cosine': off_diagonal_cosine(tokens).item(),
            'cwp_attention_row_sum_max_error': (
                weights.sum(dim=-1) - 1.0
            ).abs().max().item(),
        }
