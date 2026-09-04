from __future__ import annotations

import torch
import torch.nn as nn

from analysis.lazy_assignment.experiment2.signal_collector import (
    SignalCollector,
    class_to_patch_qk,
    split_qkv,
)
from models.vit import Block


class OneBlockHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=24,
                    num_heads=3,
                    qkv_bias=True,
                    qk_scale=0.37,
                    num_classes=20,
                    attention_normalization="vanilla",
                )
            ]
        )

    def forward(self, tokens):
        return self.blocks[0](tokens)


def test_qk_energy_and_softmax_reproduce_native_class_to_patch_attention():
    torch.manual_seed(227)
    host = OneBlockHost().eval()
    inputs = torch.randn(2, 24, 24)
    block = host.blocks[0]
    normalized = block.norm1(inputs)
    qkv_output = block.attn.qkv(normalized)
    query, key, value = split_qkv(qkv_output, num_heads=3)
    assert query.shape == key.shape == value.shape == (2, 3, 24, 8)

    manual_heads, manual_attention = class_to_patch_qk(
        qkv_output,
        num_heads=3,
        num_classes=20,
        num_patches=4,
        scale=block.attn.scale,
    )
    direct_heads = (
        query[:, :, :20] @ key[:, :, 20:].transpose(-2, -1)
    ) * block.attn.scale
    torch.testing.assert_close(manual_heads, direct_heads.float())

    _, native_attention = block(inputs)
    torch.testing.assert_close(
        manual_attention, native_attention[:, :, :20, 20:], rtol=0, atol=1e-7
    )

    with SignalCollector(host, num_classes=20) as collector:
        collector.clear(expected_num_patches=4)
        host(inputs)
        capture = collector.consume()

    torch.testing.assert_close(capture.qk_c2p_heads[0], manual_heads)
    torch.testing.assert_close(capture.qk_mean_scores[0], manual_heads.mean(dim=1))
    torch.testing.assert_close(
        capture.qk_head_std[0], manual_heads.std(dim=1, unbiased=False)
    )
    assert float(capture.qk_attention_max_abs_diff[0]) < 1e-7


def test_qk_softmax_uses_class_and_patch_keys_not_patch_only():
    torch.manual_seed(229)
    host = OneBlockHost().eval()
    inputs = torch.randn(1, 24, 24)
    block = host.blocks[0]
    qkv = block.attn.qkv(block.norm1(inputs))
    patch_energy, global_c2p = class_to_patch_qk(qkv, 3, 20, 4, block.attn.scale)
    patch_only = torch.softmax(patch_energy, dim=-1)

    # Global c2p mass is below one because 20 class keys remain in the denominator.
    assert torch.all(global_c2p.sum(dim=-1) < 1.0)
    assert not torch.allclose(global_c2p, patch_only)
    conditional = global_c2p / global_c2p.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(conditional, patch_only, rtol=1e-5, atol=1e-6)
