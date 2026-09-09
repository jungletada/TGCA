from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from models.mctformer_plus import (
    MCTformerPlus,
    build_mctformerplus,
    checkpoint_token_interaction,
    model_spec_from_instance,
    validate_mctformerplus_token_interaction_checkpoint,
)
from models.vit import Attention, Block


def _block(dim=24, heads=3):
    return Block(
        dim=dim,
        num_heads=heads,
        mlp_ratio=2,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        num_classes=3,
    )


def _lightweight_joint_kwargs(token_interaction=None):
    kwargs = {
        'num_classes': 3,
        'input_size': 32,
        'img_size': 32,
        'patch_size': 16,
        'embed_dim': 24,
        'depth': 2,
        'num_heads': 3,
        'mlp_ratio': 2,
        'qkv_bias': True,
        'drop_path_rate': 0.0,
    }
    if token_interaction is not None:
        kwargs['token_interaction'] = token_interaction
    return kwargs


def test_shared_cross_attention_matches_direct_qkv_slice_reference():
    torch.manual_seed(101)
    module = Attention(
        dim=24, num_heads=3, qkv_bias=True, num_classes=3
    ).eval()
    query_tokens = torch.randn(2, 3, 24)
    key_value_tokens = torch.randn(2, 5, 24)

    actual_output, actual_weights = module.cross_attention(
        query_tokens, key_value_tokens
    )

    dim = module.qkv.in_features
    bias = module.qkv.bias
    query = F.linear(query_tokens, module.qkv.weight[:dim], bias[:dim])
    key = F.linear(
        key_value_tokens,
        module.qkv.weight[dim:2 * dim],
        bias[dim:2 * dim],
    )
    value = F.linear(
        key_value_tokens,
        module.qkv.weight[2 * dim:],
        bias[2 * dim:],
    )
    query = query.reshape(2, 3, 3, 8).permute(0, 2, 1, 3)
    key = key.reshape(2, 5, 3, 8).permute(0, 2, 1, 3)
    value = value.reshape(2, 5, 3, 8).permute(0, 2, 1, 3)
    expected_weights = torch.softmax(
        (query @ key.transpose(-2, -1)) * module.scale, dim=-1
    )
    expected_output = (expected_weights @ value).transpose(1, 2).reshape(
        2, 3, 24
    )
    expected_output = module.proj_drop(module.proj(expected_output))

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_output, expected_output)


def test_four_attention_shapes_row_sums_order_and_shared_parameters(monkeypatch):
    torch.manual_seed(103)
    block = _block().eval()
    calls = []
    original_self = block.attn.self_attention
    original_cross = block.attn.cross_attention

    def traced_self(tokens):
        calls.append('p2p' if tokens.shape[1] == 5 else 'c2c')
        return original_self(tokens)

    def traced_cross(query_tokens, key_value_tokens):
        calls.append('c2p' if query_tokens.shape[1] == 3 else 'p2c')
        return original_cross(query_tokens, key_value_tokens)

    monkeypatch.setattr(block.attn, 'self_attention', traced_self)
    monkeypatch.setattr(block.attn, 'cross_attention', traced_cross)
    classes = torch.randn(2, 3, 24)
    patches = torch.randn(2, 5, 24)
    output_classes, output_patches, relations = (
        block.forward_decoupled_bidirectional(classes, patches)
    )

    assert calls == ['p2p', 'c2p', 'c2c', 'p2c']
    assert output_classes.shape == classes.shape
    assert output_patches.shape == patches.shape
    assert {name: tuple(value.shape) for name, value in relations.items()} == {
        'patch_to_patch': (2, 3, 5, 5),
        'class_to_patch': (2, 3, 3, 5),
        'class_to_class': (2, 3, 3, 3),
        'patch_to_class': (2, 3, 5, 3),
    }
    for weights in relations.values():
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0]),
            rtol=0, atol=1e-6,
        )

    reference = _block()
    assert list(block.state_dict()) == list(reference.state_dict())
    assert sum(p.numel() for p in block.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )


def test_decoupled_block_has_bidirectional_feature_and_gradient_exchange():
    torch.manual_seed(107)
    block = _block().eval()
    classes = torch.randn(1, 3, 24, requires_grad=True)
    patches = torch.randn(1, 5, 24, requires_grad=True)
    changed_classes = classes.detach() + torch.randn_like(classes)
    changed_patches = patches.detach() + torch.randn_like(patches)

    output_classes, output_patches, _ = (
        block.forward_decoupled_bidirectional(classes, patches)
    )
    _, patches_after_class_change, _ = (
        block.forward_decoupled_bidirectional(changed_classes, patches.detach())
    )
    classes_after_patch_change, _, _ = (
        block.forward_decoupled_bidirectional(classes.detach(), changed_patches)
    )
    assert not torch.equal(output_patches.detach(), patches_after_class_change)
    assert not torch.equal(output_classes.detach(), classes_after_patch_change)

    output_patches.square().mean().backward(retain_graph=True)
    assert classes.grad is not None
    assert torch.isfinite(classes.grad).all()
    assert classes.grad.abs().sum() > 0
    classes.grad = None
    patches.grad = None
    output_classes.square().mean().backward()
    assert patches.grad is not None
    assert torch.isfinite(patches.grad).all()
    assert patches.grad.abs().sum() > 0


def test_joint_default_is_numerically_identical_to_explicit_joint():
    torch.manual_seed(109)
    implicit = MCTformerPlus(**_lightweight_joint_kwargs()).eval()
    explicit = MCTformerPlus(
        **_lightweight_joint_kwargs('joint')
    ).eval()
    explicit.load_state_dict(deepcopy(implicit.state_dict()), strict=True)
    inputs = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        implicit_outputs = implicit(inputs)
        explicit_outputs = explicit(inputs)
    for left, right in zip(implicit_outputs, explicit_outputs):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_small_decoupled_forward_cct_relations_and_native_cam():
    torch.manual_seed(113)
    kwargs = {
        'num_classes': 2,
        'input_size': 32,
        'token_interaction': 'decoupled_bidirectional',
        'drop_path_rate': 0.0,
    }
    training_model = build_mctformerplus('small', **kwargs).eval()
    cam_model = build_mctformerplus('small', cam=True, **kwargs).eval()
    cam_model.load_state_dict(training_model.state_dict(), strict=True)
    inputs = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        outputs = training_model(inputs)
        classes, patches, records, all_classes, auxiliary = (
            cam_model.forward_features(inputs, return_aux=True)
        )
        cam_output = cam_model(inputs)

    assert outputs[0].shape == (1, 2)
    assert outputs[1].shape == (12, 1, 2, 384)
    assert outputs[2].shape == (1, 2)
    assert classes.shape == (1, 2, 384)
    assert patches.shape == (1, 4, 384)
    assert len(records) == len(all_classes) == 12
    assert auxiliary['token_interaction'] == 'decoupled_bidirectional'
    assert {name: tuple(value.shape) for name, value in records[-1].items()} == {
        'patch_to_patch': (1, 6, 4, 4),
        'class_to_patch': (1, 6, 2, 4),
        'class_to_class': (1, 6, 2, 2),
        'patch_to_class': (1, 6, 4, 2),
    }
    assert cam_output.shape == (1, 2, 2, 2)
    assert torch.isfinite(cam_output).all()

    with torch.inference_mode():
        patch_map = cam_model.head(
            patches.reshape(1, 2, 2, 384).permute(0, 3, 1, 2)
        )
        _, mean_attention = cam_model._stack_attention_records(records)
        class_to_patch = mean_attention['class_to_patch'][-3:].mean(0)
        expected = torch.sqrt(
            class_to_patch.reshape(1, 2, 2, 2) * torch.relu(patch_map)
        )
        patch_to_patch = mean_attention['patch_to_patch'].sum(dim=0)
        expected = torch.matmul(
            patch_to_patch.unsqueeze(1), expected.reshape(1, 2, 4, 1)
        ).reshape(1, 2, 2, 2)
    torch.testing.assert_close(cam_output, expected, rtol=0, atol=1e-6)


def test_decoupled_scope_parameter_count_and_checkpoint_contract():
    joint = build_mctformerplus(
        'small', num_classes=2, input_size=32
    )
    decoupled = build_mctformerplus(
        'small', num_classes=2, input_size=32,
        token_interaction='decoupled_bidirectional',
    )
    assert list(joint.state_dict()) == list(decoupled.state_dict())
    assert sum(p.numel() for p in joint.parameters()) == sum(
        p.numel() for p in decoupled.parameters()
    )
    decoupled.load_state_dict(joint.state_dict(), strict=True)
    assert model_spec_from_instance(decoupled)['token_interaction'] == (
        'decoupled_bidirectional'
    )

    checkpoint = {
        'model': decoupled.state_dict(),
        'model_spec': model_spec_from_instance(decoupled),
        'token_interaction': 'decoupled_bidirectional',
    }
    assert checkpoint_token_interaction(checkpoint) == 'decoupled_bidirectional'
    assert validate_mctformerplus_token_interaction_checkpoint(
        checkpoint, 'decoupled_bidirectional'
    ) == 'decoupled_bidirectional'
    assert checkpoint_token_interaction({'model': joint.state_dict()}) == 'joint'
    with pytest.raises(ValueError, match='does not match requested'):
        validate_mctformerplus_token_interaction_checkpoint(
            checkpoint, 'joint'
        )
    with pytest.raises(ValueError, match=r'restricted to MCTformer\+-Small'):
        build_mctformerplus(
            'tiny', num_classes=2, input_size=32,
            token_interaction='decoupled_bidirectional',
        )
    with pytest.raises(ValueError, match='first-round scope rejects'):
        build_mctformerplus(
            'small', num_classes=2, input_size=32,
            token_interaction='decoupled_bidirectional', final_norm=True,
        )


def test_full_decoupled_model_backward_reaches_original_training_branches():
    torch.manual_seed(123)
    model = build_mctformerplus(
        'small', num_classes=2, input_size=32,
        token_interaction='decoupled_bidirectional', drop_path_rate=0.0,
    ).train()
    class_logits, all_class_tokens, patch_logits = model(
        torch.randn(1, 3, 32, 32)
    )
    loss = (
        class_logits.square().mean()
        + all_class_tokens.square().mean()
        + patch_logits.square().mean()
    )
    loss.backward()

    for parameter in (
            model.cls_token, model.patch_embed.proj.weight,
            model.blocks[0].attn.qkv.weight,
            model.blocks[0].attn.proj.weight,
            model.blocks[0].mlp.fc1.weight, model.head.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_decoupled_amp_forward_backward_is_finite():
    torch.manual_seed(127)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    block = _block().to(device).train()
    classes = torch.randn(2, 3, 24, device=device, requires_grad=True)
    patches = torch.randn(2, 5, 24, device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, dtype=dtype):
        output_classes, output_patches, relations = (
            block.forward_decoupled_bidirectional(classes, patches)
        )
        loss = output_classes.square().mean() + output_patches.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in relations.values())
    for parameter in (
            block.norm1.weight, block.attn.qkv.weight,
            block.attn.proj.weight, block.norm2.weight,
            block.mlp.fc1.weight, block.mlp.fc2.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
