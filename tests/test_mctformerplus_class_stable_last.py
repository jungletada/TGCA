from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mctformer_plus import (
    ClassStableLastPooler,
    MCTformerPlus,
    MCTformerPlusCam,
    checkpoint_class_stable_last_enabled,
    validate_mctformerplus_final_norm_checkpoint,
)


def _small_kwargs(class_stable_last=True):
    return {
        'img_size': 32,
        'input_size': 32,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'num_classes': 20,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'attention_normalization': 'vanilla',
        'bcss_variant': 'e0',
        'psl_variant': 'baseline',
        'cti_bgt': False,
        'final_norm': False,
        'patch_final_norm': True,
        'last_mct': False,
        'class_stable_last': class_stable_last,
    }


def test_disabled_variant_is_numerically_identical_to_patch_final_ln_baseline():
    torch.manual_seed(201)
    baseline_kwargs = _small_kwargs(class_stable_last=False)
    baseline_kwargs.pop('class_stable_last')
    baseline = MCTformerPlus(**baseline_kwargs).eval()
    explicit_off = MCTformerPlus(
        **_small_kwargs(class_stable_last=False)
    ).eval()
    explicit_off.load_state_dict(baseline.state_dict(), strict=True)
    inputs = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        baseline_outputs = baseline(inputs)
        explicit_outputs = explicit_off(inputs)

    assert tuple(baseline.state_dict()) == tuple(explicit_off.state_dict())
    assert baseline.class_stable_last_pooler is None
    assert explicit_off.class_stable_last_pooler is None
    assert baseline.head.kernel_size == explicit_off.head.kernel_size == (3, 3)
    for observed, expected in zip(explicit_outputs, baseline_outputs):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_low_pass_preserves_token_shape_dtype_and_finite_values():
    torch.manual_seed(202)
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)
    pooler = ClassStableLastPooler(
        embed_dim=8, topk=1, sigma=math.sqrt(8), eps=1e-6
    )

    low_pass = pooler.low_pass(tokens)

    assert low_pass.shape == tokens.shape
    assert low_pass.dtype == tokens.dtype
    assert torch.isfinite(low_pass).all()


def test_original_and_low_pass_maps_use_the_same_three_by_three_classifier():
    torch.manual_seed(203)
    model = MCTformerPlus(**_small_kwargs()).eval()
    tokens = torch.randn(2, 4, 384)
    grid = tokens.reshape(2, 2, 2, 384).permute(0, 3, 1, 2).contiguous()
    calls = []

    def record(_module, inputs, output):
        calls.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = model.head.register_forward_hook(record)
    try:
        original_map = model.head(grid)
        pooled, indices, stability, selected = model.class_stable_last_pool(
            tokens, original_map
        )
    finally:
        handle.remove()

    assert model.head.kernel_size == (3, 3)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][1], original_map)
    expected_low_pass = model.class_stable_last_low_pass(tokens).reshape(
        2, 2, 2, 384
    ).permute(0, 3, 1, 2).contiguous()
    torch.testing.assert_close(calls[1][0], expected_low_pass)
    reference = model.class_stable_last_pooler(original_map, calls[1][1])
    for observed, expected in zip(
            (pooled, indices, stability, selected), reference):
        torch.testing.assert_close(observed, expected)


def test_class_response_stability_shape_and_finite_values():
    original = torch.tensor([[[[1.0, -2.0], [3.0, 4.0]]]])
    low_pass = original.clone()
    pooler = ClassStableLastPooler(embed_dim=4, eps=1e-6)

    stability = pooler.stability_score(original, low_pass)

    assert stability.shape == original.shape
    assert torch.isfinite(stability).all()
    torch.testing.assert_close(stability, original / 1e-6)


def test_topk_is_class_wise_over_space_and_gathers_original_map_values():
    original = torch.tensor([[
        [[10.0, 11.0], [12.0, 13.0]],
        [[20.0, 21.0], [22.0, 23.0]],
    ]])
    stability = torch.tensor([[
        [[1.0, 9.0], [2.0, 3.0]],
        [[8.0, 1.0], [10.0, 2.0]],
    ]])
    pooler = ClassStableLastPooler(embed_dim=4, topk=1)

    pooled, indices, selected = pooler.pool(original, stability)

    torch.testing.assert_close(indices, torch.tensor([[[1], [2]]]))
    torch.testing.assert_close(selected, torch.tensor([[[11.0], [22.0]]]))
    torch.testing.assert_close(pooled, torch.tensor([[11.0, 22.0]]))
    # The selected values intentionally differ from their selector scores.
    assert not torch.equal(selected, torch.tensor([[[9.0], [10.0]]]))


def test_forward_adds_no_low_pass_auxiliary_loss_output():
    torch.manual_seed(204)
    model = MCTformerPlus(**_small_kwargs()).eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 32, 32))

    assert len(outputs) == 3
    assert outputs[0].shape == outputs[2].shape == (1, 20)
    assert outputs[1].shape[:3] == (12, 1, 20)
    assert model.class_stable_last_configuration()['lowpass_role'] == 'selector only'


def test_patch_loss_gradient_reaches_norm_classifier_and_selected_original_map():
    torch.manual_seed(205)
    batch, height, width, embed_dim, classes = 2, 2, 3, 8, 4
    norm = nn.LayerNorm(embed_dim)
    classifier = nn.Conv2d(embed_dim, classes, kernel_size=3, padding=1)
    pooler = ClassStableLastPooler(embed_dim=embed_dim, topk=1)
    raw = torch.randn(
        batch, height * width, embed_dim, requires_grad=True
    )
    normalized = norm(raw)
    grid = normalized.reshape(
        batch, height, width, embed_dim
    ).permute(0, 3, 1, 2).contiguous()
    original_map = classifier(grid)
    original_map.retain_grad()
    low_pass = pooler.low_pass(normalized).reshape(
        batch, height, width, embed_dim
    ).permute(0, 3, 1, 2).contiguous()
    low_pass_map = classifier(low_pass)
    logits, indices, _, _ = pooler(original_map, low_pass_map)
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])

    F.multilabel_soft_margin_loss(logits, targets).backward()

    for gradient in (
        norm.weight.grad,
        norm.bias.grad,
        classifier.weight.grad,
        classifier.bias.grad,
        raw.grad,
        original_map.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    selected_mask = torch.zeros_like(original_map.flatten(2), dtype=torch.bool)
    selected_mask.scatter_(2, indices, True)
    assert original_map.grad.flatten(2)[selected_mask].abs().sum() > 0
    assert original_map.grad.flatten(2)[~selected_mask].abs().sum() == 0


def test_cam_gating_uses_class_stable_logits_but_cam_uses_original_map(monkeypatch):
    torch.manual_seed(206)
    model = MCTformerPlusCam(**_small_kwargs()).eval()
    inputs = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        cls, tokens, attention_heads, _ = model.forward_features(inputs)
        grid = tokens.reshape(1, 2, 2, 384).permute(0, 3, 1, 2).contiguous()
        original_map = model.head(grid)
        logits, _, _, _ = model.class_stable_last_pool(tokens, original_map)
        attention = torch.stack(attention_heads).mean(dim=2)
        expected_cam = model.get_cam(original_map, attention)
    monkeypatch.setattr(
        model, 'gwrp', lambda _map: (_ for _ in ()).throw(
            AssertionError('Class-Stable LaST gating must not call GWRP')
        )
    )

    with torch.inference_mode():
        cls_label, patch_label, cam = model.forward_with_label(inputs)

    torch.testing.assert_close(cls_label, (cls.mean(-1) > 0).to(inputs.dtype))
    torch.testing.assert_close(patch_label, (logits > 0).to(inputs.dtype))
    torch.testing.assert_close(cam, expected_cam, rtol=0, atol=0)


def test_scope_and_checkpoint_metadata_are_strict():
    assert checkpoint_class_stable_last_enabled({}) is False
    assert checkpoint_class_stable_last_enabled(
        {'class_stable_last': True}
    ) is True
    assert validate_mctformerplus_final_norm_checkpoint(
        {'patch_final_norm': True, 'class_stable_last': True},
        False, True, False, True,
    ) is False
    with pytest.raises(ValueError, match='requires patch_final_norm'):
        validate_mctformerplus_final_norm_checkpoint(
            {'class_stable_last': True}, False, False, False, True
        )
    with pytest.raises(ValueError, match='mutually exclusive'):
        mutually_exclusive = _small_kwargs()
        mutually_exclusive['last_mct'] = True
        MCTformerPlus(**mutually_exclusive)
    incompatible = dict(_small_kwargs())
    incompatible['attention_normalization'] = 'tgca'
    with pytest.raises(ValueError, match='vanilla attention'):
        MCTformerPlus(**incompatible)
