from __future__ import annotations

import torch
import torch.nn as nn

from analysis.lazy_assignment.experiment2.signal_collector import (
    SignalCollector,
    assert_no_change,
)
from analysis.lazy_assignment.score_utils import class_specific_patch_score
from models.vit import Block


class ToyNativeTransformer(nn.Module):
    def __init__(self, depth=2, num_classes=20, dim=24, heads=3):
        super().__init__()
        self.num_classes = num_classes
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=heads,
                    qkv_bias=True,
                    num_classes=num_classes,
                    attention_normalization="vanilla",
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens):
        attentions = []
        for block in self.blocks:
            tokens, attention = block(tokens)
            attentions.append(attention)
        return (
            tokens[:, : self.num_classes],
            tokens[:, self.num_classes :],
            attentions,
        )


def manual_trace(model, inputs):
    tokens = inputs
    norm_scores = []
    post_scores = []
    class_pairwise = []
    patch_norms = []
    attentions = []
    for block in model.blocks:
        normalized = block.norm1(tokens)
        norm_scores.append(
            class_specific_patch_score(
                normalized[:, : model.num_classes],
                normalized[:, model.num_classes :],
            )
        )
        tokens, attention = block(tokens)
        classes = tokens[:, : model.num_classes]
        patches = tokens[:, model.num_classes :]
        post_scores.append(class_specific_patch_score(classes, patches))
        class_unit = torch.nn.functional.normalize(classes.float(), dim=-1, eps=1e-12)
        class_pairwise.append(torch.einsum("bcd,bed->bce", class_unit, class_unit))
        patch_norms.append(patches.float().norm(dim=-1))
        attentions.append(attention)
    return {
        "norm": torch.stack(norm_scores),
        "post": torch.stack(post_scores),
        "class_pairwise": torch.stack(class_pairwise),
        "patch_norms": torch.stack(patch_norms),
        "attentions": attentions,
    }


def test_signal_collector_observes_native_path_without_changing_output():
    torch.manual_seed(221)
    model = ToyNativeTransformer().eval()
    inputs = torch.randn(2, 24, 24)
    reference = model(inputs)
    manual = manual_trace(model, inputs)

    with SignalCollector(model, num_classes=20) as collector:
        collector.clear(expected_num_patches=4)
        observed = model(inputs)
        capture = collector.consume()

    assert assert_no_change(reference, observed, tolerance=0.0) == 0.0
    assert capture.feature_post_scores.shape == (2, 2, 20, 4)
    assert capture.feature_norm_scores.shape == (2, 2, 20, 4)
    assert capture.qk_c2p_heads.shape == (2, 2, 3, 20, 4)
    assert capture.qk_mean_scores.shape == (2, 2, 20, 4)
    assert capture.qk_head_std.shape == (2, 2, 20, 4)
    assert capture.attn_c2p_raw.shape == (2, 2, 20, 4)
    assert capture.attn_c2p_conditional.shape == (2, 2, 20, 4)
    assert capture.attn_patch_mass.shape == (2, 2, 20)
    assert capture.class_token_pairwise_cosine.shape == (2, 2, 20, 20)
    assert capture.patch_norms.shape == (2, 2, 4)
    assert capture.last_class_tokens.shape == (2, 20, 24)
    assert capture.last_patch_tokens.shape == (2, 4, 24)
    assert capture.patch_to_patch_sum.shape == (2, 4, 4)

    torch.testing.assert_close(capture.feature_post_scores, manual["post"])
    torch.testing.assert_close(capture.feature_norm_scores, manual["norm"])
    torch.testing.assert_close(
        capture.class_token_pairwise_cosine, manual["class_pairwise"]
    )
    torch.testing.assert_close(capture.patch_norms, manual["patch_norms"])
    torch.testing.assert_close(
        capture.attn_c2p_conditional.sum(dim=-1),
        torch.ones_like(capture.attn_patch_mass),
    )
    torch.testing.assert_close(
        capture.attn_patch_mass, capture.attn_c2p_raw.sum(dim=-1)
    )

    p2p = torch.stack(
        [attention[:, :, 20:, 20:].mean(dim=1) for attention in manual["attentions"]]
    ).sum(dim=0)
    torch.testing.assert_close(capture.patch_to_patch_sum, p2p)
    assert float(capture.qk_attention_max_abs_diff.max()) < 1e-7
    assert float(capture.attention_row_sum_max_abs_error.max()) < 1e-6
    assert float(capture.pre_norm_input_max_abs_diff.max()) == 0.0
    assert float(capture.norm_qkv_input_max_abs_diff.max()) == 0.0

    assert all(not block._forward_hooks for block in model.blocks)
    assert all(not block._forward_pre_hooks for block in model.blocks)
    assert all(not block.norm1._forward_hooks for block in model.blocks)
    assert all(not block.attn.qkv._forward_hooks for block in model.blocks)


def test_signal_collector_rejects_extra_tokens_under_explicit_baseline_shape():
    model = ToyNativeTransformer(depth=1).eval()
    tokens_with_extra = torch.randn(1, 25, 24)
    with SignalCollector(model, num_classes=20) as collector:
        collector.clear(expected_num_patches=4)
        try:
            model(tokens_with_extra)
        except RuntimeError as error:
            assert "exactly 20 class + 4 patch tokens" in str(error)
        else:
            raise AssertionError("extra token was silently accepted")
