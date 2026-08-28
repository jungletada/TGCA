from functools import partial

import pytest
import torch
import torch.nn as nn

from models.mctformer_plus import MCTformerPlus
from models.vit import TOKEN_ROLE_SPECIALIZATIONS, validate_token_role_specialization


def small_model(mode):
    return MCTformerPlus(
        num_classes=2,
        input_size=32,
        img_size=32,
        patch_size=16,
        embed_dim=24,
        depth=6,
        num_heads=3,
        mlp_ratio=2,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        token_role_specialization=mode,
    )


def load_from_shared(specialized, shared):
    expanded = specialized.expand_token_role_state_dict(shared.state_dict())
    incompatibility = specialized.load_state_dict(expanded, strict=True)
    assert not incompatibility.missing_keys
    assert not incompatibility.unexpected_keys


def test_supported_token_role_specializations_are_explicit():
    assert TOKEN_ROLE_SPECIALIZATIONS == ("shared", "norm", "norm_qkv")
    assert validate_token_role_specialization("NORM_QKV") == "norm_qkv"
    with pytest.raises(ValueError, match="Unsupported token-role specialization"):
        validate_token_role_specialization("all_layers")


def test_shared_mode_preserves_historical_state_dict_surface():
    model = small_model("shared")
    names = tuple(model.state_dict())
    assert not any("class_norm" in name for name in names)
    assert not any("class_qkv" in name for name in names)


@pytest.mark.parametrize("mode", ("norm", "norm_qkv"))
def test_shared_weight_expansion_makes_specialized_model_initially_equivalent(mode):
    torch.manual_seed(17)
    shared = small_model("shared").eval()
    specialized = small_model(mode).eval()
    load_from_shared(specialized, shared)

    inputs = torch.randn(2, 3, 32, 32)
    shared_outputs = shared.forward_features(inputs)
    specialized_outputs = specialized.forward_features(inputs)
    for shared_tensor, specialized_tensor in zip(shared_outputs[:2], specialized_outputs[:2]):
        torch.testing.assert_close(
            specialized_tensor, shared_tensor, rtol=1e-5, atol=1e-6
        )
    for shared_attention, specialized_attention in zip(
        shared_outputs[2], specialized_outputs[2]
    ):
        torch.testing.assert_close(
            specialized_attention, shared_attention, rtol=1e-5, atol=1e-6
        )


def test_norm_mode_changes_only_class_role_at_the_norm_boundary():
    model = small_model("norm")
    block = model.blocks[0]
    with torch.no_grad():
        block.class_norm1.weight.fill_(2.0)
        block.class_norm1.bias.fill_(0.25)

    tokens = torch.randn(2, 7, 24)
    actual = block.apply_role_norm(tokens, block.norm1, block.class_norm1)
    shared = block.norm1(tokens)
    torch.testing.assert_close(actual[:, 2:], shared[:, 2:], rtol=0, atol=0)
    assert not torch.allclose(actual[:, :2], shared[:, :2])


def test_norm_qkv_specializes_only_the_first_third_of_blocks():
    model = small_model("norm_qkv")
    specialized = [block.attn.class_qkv is not None for block in model.blocks]
    assert specialized == [True, True, False, False, False, False]
    assert all(block.class_norm1 is not None for block in model.blocks)
    assert all(block.class_norm2 is not None for block in model.blocks)


@pytest.mark.parametrize("mode", ("norm", "norm_qkv"))
def test_specialized_parameters_receive_finite_gradients(mode):
    model = small_model(mode).train()
    outputs = model(torch.randn(2, 3, 32, 32))
    loss = sum(output.float().square().mean() for output in outputs[:3])
    loss.backward()

    specialized = [
        parameter
        for name, parameter in model.named_parameters()
        if ".class_norm" in name or ".class_qkv" in name
    ]
    assert specialized
    assert all(parameter.grad is not None for parameter in specialized)
    assert all(torch.isfinite(parameter.grad).all() for parameter in specialized)
