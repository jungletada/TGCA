from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from analysis.lazy_assignment.experiment3.c2c_intervention import (
    C2C_VARIANT_LAYERS_1BASED,
    C2CIntervention,
    self_reroute_c2c_attention,
    variant_layer_indices,
)
from models.vit import Block


NUM_CLASSES = 20


def _offdiag(block: torch.Tensor) -> torch.Tensor:
    classes = block.shape[-1]
    mask = ~torch.eye(classes, dtype=torch.bool, device=block.device)
    return block[..., mask]


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_equal(left_item, right_item)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
        return
    assert left == right


class TinyNativeTransformer(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        num_patches: int = 4,
        dim: int = 12,
        heads: int = 3,
        normalization: str = "vanilla",
    ) -> None:
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.num_class_tokens = NUM_CLASSES
        self.num_patches = num_patches
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=heads,
                    qkv_bias=True,
                    num_classes=NUM_CLASSES,
                    attention_normalization=normalization,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens: torch.Tensor):
        attentions = []
        for block in self.blocks:
            tokens, attention = block(tokens)
            attentions.append(attention)
        return tokens, attentions


class TinyTwoBlockCam(TinyNativeTransformer):
    def __init__(self, num_patches: int = 4, dim: int = 12) -> None:
        super().__init__(depth=2, num_patches=num_patches, dim=dim, heads=3)
        self.patch_head = nn.Linear(dim, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor):
        tokens, attentions = super().forward(tokens)
        classes = tokens[:, :NUM_CLASSES]
        patches = tokens[:, NUM_CLASSES:]
        patch_logits = self.patch_head(patches).transpose(1, 2)
        stacked = torch.stack(attentions)
        head_mean = stacked.mean(dim=2)
        c2p_layers = head_mean[:, :, :NUM_CLASSES, NUM_CLASSES:]
        p2p_sum = head_mean[:, :, NUM_CLASSES:, NUM_CLASSES:].sum(dim=0)
        c2p = c2p_layers.mean(dim=0)
        class_attention_cam = c2p * torch.relu(patch_logits)
        final_cam = torch.matmul(
            p2p_sum.unsqueeze(1), class_attention_cam.unsqueeze(-1)
        ).squeeze(-1)
        return {
            "class_tokens": classes,
            "class_logits": classes.mean(dim=-1),
            "patch_tokens": patches,
            "patch_logits": patch_logits,
            "c2p_layers": c2p_layers,
            "final_cam": final_cam,
            "attentions": attentions,
        }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pure_self_reroute_preserves_shape_dtype_device_and_all_invariants(dtype):
    torch.manual_seed(301)
    logits = torch.randn(2, 3, NUM_CLASSES + 5, NUM_CLASSES + 5, dtype=dtype)
    attention = torch.softmax(logits, dim=-1)
    rerouted = self_reroute_c2c_attention(attention)

    assert rerouted is not attention
    assert rerouted.shape == attention.shape
    assert rerouted.dtype == attention.dtype
    assert rerouted.device == attention.device
    torch.testing.assert_close(
        rerouted[..., NUM_CLASSES:, :],
        attention[..., NUM_CLASSES:, :],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rerouted[..., :NUM_CLASSES, NUM_CLASSES:],
        attention[..., :NUM_CLASSES, NUM_CLASSES:],
        rtol=0,
        atol=0,
    )

    native_c2c = attention[..., :NUM_CLASSES, :NUM_CLASSES]
    rerouted_c2c = rerouted[..., :NUM_CLASSES, :NUM_CLASSES]
    torch.testing.assert_close(
        rerouted_c2c.sum(dim=-1), native_c2c.sum(dim=-1), rtol=0, atol=0
    )
    row_sum_tolerance = 4 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        rerouted.sum(dim=-1),
        attention.sum(dim=-1),
        rtol=0,
        atol=row_sum_tolerance,
    )
    assert torch.count_nonzero(_offdiag(rerouted_c2c)) == 0
    torch.testing.assert_close(
        torch.diagonal(rerouted_c2c, dim1=-2, dim2=-1),
        native_c2c.sum(dim=-1),
        rtol=0,
        atol=0,
    )


def test_pure_self_reroute_rejects_invalid_attention_contract():
    with pytest.raises(ValueError, match=r"\[B,H,T,T\]"):
        self_reroute_c2c_attention(torch.ones(2, 3, 4))
    with pytest.raises(ValueError, match="square"):
        self_reroute_c2c_attention(torch.ones(1, 2, 21, 22))
    with pytest.raises(TypeError, match="floating-point"):
        self_reroute_c2c_attention(torch.ones(1, 2, 21, 21, dtype=torch.int64))
    bad = torch.ones(1, 2, 21, 21)
    bad[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        self_reroute_c2c_attention(bad)


def test_plan_variants_use_exact_one_based_layers_and_activate_only_those_layers():
    expected = {
        "C0": (),
        "C1": (12,),
        "C2": (11,),
        "C3": (10,),
        "C4": (10, 11),
        "C5": (10, 11, 12),
    }
    assert dict(C2C_VARIANT_LAYERS_1BASED) == expected
    torch.manual_seed(302)
    model = TinyNativeTransformer(depth=12, num_patches=2).eval()
    inputs = torch.randn(1, NUM_CLASSES + 2, 12)

    for variant, one_based in expected.items():
        context = C2CIntervention.from_variant(model, variant, expected_num_patches=2)
        with context:
            _, attentions = model(inputs)
        assert context.layers == variant_layer_indices(variant)
        assert context.layer_numbers_1based == one_based
        assert context.activation_counts_1based == {layer: 1 for layer in one_based}
        for layer_number, attention in enumerate(attentions, start=1):
            c2c = attention[..., :NUM_CLASSES, :NUM_CLASSES]
            if layer_number in one_based:
                assert torch.count_nonzero(_offdiag(c2c)) == 0
            else:
                assert bool(torch.all(_offdiag(c2c) > 0))


def test_context_changes_value_path_and_returned_attention_but_not_patch_rows():
    torch.manual_seed(303)
    model = TinyNativeTransformer(depth=1).eval()
    inputs = torch.randn(2, NUM_CLASSES + 4, 12)
    baseline_tokens, baseline_attentions = model(inputs)
    context = C2CIntervention(model, [0], expected_num_patches=4, expected_depth=1)

    with context:
        changed_tokens, changed_attentions = model(inputs)

    baseline_attention = baseline_attentions[0]
    changed_attention = changed_attentions[0]
    assert not torch.allclose(
        changed_tokens[:, :NUM_CLASSES], baseline_tokens[:, :NUM_CLASSES]
    )
    torch.testing.assert_close(
        changed_tokens[:, NUM_CLASSES:],
        baseline_tokens[:, NUM_CLASSES:],
        rtol=0,
        atol=0,
    )
    assert (
        torch.count_nonzero(
            _offdiag(changed_attention[..., :NUM_CLASSES, :NUM_CLASSES])
        )
        == 0
    )
    torch.testing.assert_close(
        changed_attention[..., :NUM_CLASSES, NUM_CLASSES:],
        baseline_attention[..., :NUM_CLASSES, NUM_CLASSES:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        changed_attention[..., NUM_CLASSES:, :],
        baseline_attention[..., NUM_CLASSES:, :],
        rtol=0,
        atol=0,
    )

    assert context.activation_counts == {0: 1}
    assert len(context.records) == 1
    record = context.records[0]
    assert record.layer_number == 1
    assert record.call_index == 1
    assert record.offdiag_pre_mass_mean > 0
    assert record.offdiag_pre_mass_max > 0
    assert record.offdiag_post_mass_max == 0
    assert record.patch_rows_max_abs_diff == 0
    assert record.class_to_patch_max_abs_diff == 0
    assert record.class_group_mass_max_abs_diff == 0
    assert record.diagonal_mass_max_abs_diff == 0
    assert record.row_sum_max_abs_diff <= 1e-6
    assert not model.blocks[0].attn.normalizer._forward_hooks


def test_last_block_intervention_is_a_structural_cam_negative_control():
    torch.manual_seed(304)
    model = TinyTwoBlockCam().eval()
    inputs = torch.randn(2, NUM_CLASSES + 4, 12)
    baseline = model(inputs)
    context = C2CIntervention(model, [1], expected_num_patches=4, expected_depth=2)

    with context:
        changed = model(inputs)

    assert not torch.allclose(changed["class_tokens"], baseline["class_tokens"])
    assert not torch.allclose(changed["class_logits"], baseline["class_logits"])
    for key in ("patch_tokens", "patch_logits", "c2p_layers", "final_cam"):
        torch.testing.assert_close(changed[key], baseline[key], rtol=0, atol=1e-6)
    assert context.activation_counts_1based == {2: 1}


def test_context_rejects_nonvanilla_training_and_wrong_runtime_layout():
    nonvanilla = TinyNativeTransformer(depth=1, normalization="tgca").eval()
    with pytest.raises(RuntimeError, match="must be vanilla"):
        with C2CIntervention(nonvanilla, [0], expected_num_patches=4, expected_depth=1):
            pass

    training = TinyNativeTransformer(depth=1).train()
    with pytest.raises(RuntimeError, match="inference-only"):
        with C2CIntervention(training, [0], expected_num_patches=4, expected_depth=1):
            pass

    model = TinyNativeTransformer(depth=1).eval()
    wrong_layout = torch.randn(1, NUM_CLASSES + 5, 12)
    with pytest.raises(RuntimeError, match="exact 20-class/patch layout"):
        with C2CIntervention(model, [0], expected_num_patches=4, expected_depth=1):
            model(wrong_layout)
    assert not model.blocks[0].attn.normalizer._forward_hooks


def test_context_cleanup_exception_nested_rejection_and_state_dict_unchanged():
    torch.manual_seed(305)
    model = TinyNativeTransformer(depth=2).eval()
    inputs = torch.randn(1, NUM_CLASSES + 4, 12)
    baseline = model(inputs)
    state_before = copy.deepcopy(model.state_dict())

    outer = C2CIntervention(model, [0, 1], expected_num_patches=4, expected_depth=2)
    with outer:
        with pytest.raises(RuntimeError, match="already active"):
            with C2CIntervention(model, [1], expected_num_patches=4, expected_depth=2):
                model(inputs)
        model(inputs)
    assert outer.activation_counts == {0: 1, 1: 1}

    with pytest.raises(ValueError, match="intentional"):
        with C2CIntervention(model, [0], expected_num_patches=4, expected_depth=2):
            raise ValueError("intentional body failure")

    for block in model.blocks:
        assert not block.attn.normalizer._forward_hooks
    restored = model(inputs)
    _assert_tree_equal(restored, baseline)
    state_after = model.state_dict()
    assert state_after.keys() == state_before.keys()
    for key in state_before:
        torch.testing.assert_close(state_after[key], state_before[key], rtol=0, atol=0)
