"""Minimal persistent semantic latent read/write modules for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


PSL_VARIANTS = ("baseline", "read_only", "write_only", "read_write")


@dataclass(frozen=True)
class PersistentSemanticSpec:
    enabled: bool
    read: bool
    write: bool


_SPECS = {
    "baseline": PersistentSemanticSpec(False, False, False),
    "read_only": PersistentSemanticSpec(True, True, False),
    "write_only": PersistentSemanticSpec(True, False, True),
    "read_write": PersistentSemanticSpec(True, True, True),
}


def validate_psl_variant(variant: str) -> PersistentSemanticSpec:
    variant = variant.lower()
    if variant not in _SPECS:
        raise ValueError(
            f"Unknown persistent-semantic variant {variant!r}; "
            f"expected one of {PSL_VARIANTS}"
        )
    return _SPECS[variant]


def parse_interaction_layers(value):
    if isinstance(value, str):
        try:
            layers = tuple(int(item) for item in value.split(",") if item != "")
        except ValueError as error:
            raise ValueError("interaction layers must be comma-separated integers") from error
    else:
        layers = tuple(int(item) for item in value)
    if not layers or len(set(layers)) != len(layers) or tuple(sorted(layers)) != layers:
        raise ValueError("interaction layers must be unique and increasing")
    if any(layer < 0 for layer in layers):
        raise ValueError("interaction layers must be non-negative")
    return layers


class SemanticReadWrite(nn.Module):
    """One shared class-patch relation with optional read and write updates."""

    def __init__(self, dim: int, relation_dim: int, read: bool, write: bool):
        super().__init__()
        if dim <= 0 or relation_dim <= 0:
            raise ValueError("semantic and relation dimensions must be positive")
        self.dim = dim
        self.relation_dim = relation_dim
        self.read_enabled = bool(read)
        self.write_enabled = bool(write)
        self.scale = relation_dim ** -0.5

        # All variants instantiate the same projections for parameter matching.
        self.semantic_query = nn.Linear(dim, relation_dim)
        self.patch_key = nn.Linear(dim, relation_dim)
        self.patch_value = nn.Linear(dim, dim)
        self.semantic_value = nn.Linear(dim, dim)
        self.read_output = nn.Linear(dim, dim)
        self.write_output = nn.Linear(dim, dim)
        self.write_gate = nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def initialize_from_backbone_attention(self, attention):
        """Copy pretrained Q/K/V and output projections into the relation path."""
        if self.relation_dim != self.dim:
            raise ValueError("backbone initialization requires relation_dim == dim")
        query_weight, key_weight, value_weight = attention.qkv.weight.chunk(3, dim=0)
        self.semantic_query.weight.copy_(query_weight)
        self.patch_key.weight.copy_(key_weight)
        self.patch_value.weight.copy_(value_weight)
        self.semantic_value.weight.copy_(value_weight)
        if attention.qkv.bias is not None:
            query_bias, key_bias, value_bias = attention.qkv.bias.chunk(3, dim=0)
            self.semantic_query.bias.copy_(query_bias)
            self.patch_key.bias.copy_(key_bias)
            self.patch_value.bias.copy_(value_bias)
            self.semantic_value.bias.copy_(value_bias)
        self.read_output.load_state_dict(attention.proj.state_dict())
        self.write_output.load_state_dict(attention.proj.state_dict())
        self.write_gate.zero_()

    def forward(self, semantic_latents, patch_tokens):
        if semantic_latents.ndim != 3 or patch_tokens.ndim != 3:
            raise ValueError("semantic latents and patch tokens must be B x N x D")
        if semantic_latents.shape[0] != patch_tokens.shape[0]:
            raise ValueError("semantic latents and patch tokens must share a batch")
        if semantic_latents.shape[-1] != self.dim or patch_tokens.shape[-1] != self.dim:
            raise ValueError("semantic latents and patch tokens have the wrong width")

        # Copied DeiT Q/K/V weights expect the block's pre-norm input scale.
        semantic_input = F.layer_norm(semantic_latents, (self.dim,))
        patch_input = F.layer_norm(patch_tokens, (self.dim,))
        relation = (
            self.semantic_query(semantic_input)
            @ self.patch_key(patch_input).transpose(-2, -1)
        ) * self.scale
        read_attention = torch.softmax(relation.float(), dim=-1).to(relation.dtype)
        write_attention = torch.softmax(
            relation.transpose(-2, -1).float(), dim=-1
        ).to(relation.dtype)

        updated_semantic = semantic_latents
        if self.read_enabled:
            read_message = read_attention @ self.patch_value(patch_input)
            updated_semantic = semantic_latents + self.read_output(read_message)

        updated_patches = patch_tokens
        if self.write_enabled:
            # Read precedes write: values come from the image-conditioned latents.
            semantic_value_input = F.layer_norm(updated_semantic, (self.dim,))
            write_message = write_attention @ self.semantic_value(
                semantic_value_input)
            updated_patches = patch_tokens + self.write_gate * self.write_output(
                write_message
            )

        return updated_semantic, updated_patches, {
            "relation": relation,
            "read_attention": read_attention,
            "write_attention": write_attention,
            "semantic_latents": updated_semantic,
            "write_gate": self.write_gate,
        }
