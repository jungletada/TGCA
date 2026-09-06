from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mctformer_plus import (
    LastPatchAggregator,
    MCTformerPlus,
    MCTformerPlusCam,
    checkpoint_last_mct_enabled,
    validate_mctformerplus_final_norm_checkpoint,
)


def _direct_last_reference(tokens, topk, sigma, eps):
    width = tokens.shape[-1]
    positions = torch.arange(
        -width // 2 + 1,
        width // 2 + 1,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = (kernel / kernel.max()).view(1, 1, width)
    spectrum = torch.fft.fft(tokens, dim=-1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1) * kernel
    low_pass = torch.fft.ifft(
        torch.fft.ifftshift(spectrum, dim=-1), dim=-1
    ).real
    stability = tokens / (low_pass - tokens).abs().clamp_min(eps)
    indices = torch.topk(
        stability, k=min(topk, tokens.shape[1]), dim=1, largest=True
    ).indices
    selected = torch.gather(tokens, dim=1, index=indices)
    return selected.mean(dim=1), indices, stability


def _classifier_stub(embed_dim=6, num_classes=3):
    model = MCTformerPlus.__new__(MCTformerPlus)
    nn.Module.__init__(model)
    model.last_mct = True
    model.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
    return model


def _small_kwargs(cam=False):
    kwargs = {
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
        'last_mct': True,
    }
    return MCTformerPlusCam(**kwargs) if cam else MCTformerPlus(**kwargs)


def test_last_aggregator_shapes_and_direct_repository_equivalence():
    torch.manual_seed(101)
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)
    aggregator = LastPatchAggregator(
        embed_dim=8, topk=2, sigma=math.sqrt(8), eps=1e-6
    )

    pooled, indices, stability = aggregator(tokens)
    reference = _direct_last_reference(
        tokens, topk=2, sigma=math.sqrt(8), eps=1e-6
    )

    assert pooled.shape == (2, 8)
    assert indices.shape == (2, 2, 8)
    assert stability.shape == (2, 7, 8)
    for observed, expected in zip((pooled, indices, stability), reference):
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(stability).all()


def test_last_topk_selects_independent_patch_per_channel(monkeypatch):
    tokens = torch.tensor([[
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ]])
    scores = torch.tensor([[
        [9.0, 1.0, 1.0, 1.0],
        [1.0, 9.0, 1.0, 9.0],
        [1.0, 1.0, 9.0, 1.0],
    ]])
    aggregator = LastPatchAggregator(embed_dim=4, topk=1)
    monkeypatch.setattr(aggregator, 'low_pass', torch.zeros_like)
    monkeypatch.setattr(
        aggregator, 'stability_score', lambda _tokens, _low_pass: scores
    )

    pooled, indices, _ = aggregator(tokens)

    expected_indices = torch.tensor([[[0, 1, 2, 1]]])
    expected_pooled = torch.tensor([[10.0, 21.0, 32.0, 23.0]])
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(pooled, expected_pooled)


def test_shared_one_by_one_classifier_matches_linear_and_spatial_scores():
    torch.manual_seed(102)
    model = _classifier_stub()
    pooled = torch.randn(4, 6)
    expected = F.linear(
        pooled, model.head.weight[:, :, 0, 0], model.head.bias
    )
    torch.testing.assert_close(model.last_classify(pooled), expected)

    single_patch = pooled.reshape(4, 6, 1, 1)
    spatial = model.head(single_patch)[:, :, 0, 0]
    torch.testing.assert_close(spatial, expected)


def test_last_patch_loss_gradients_reach_norm_selected_values_and_classifier():
    torch.manual_seed(103)
    norm = nn.LayerNorm(8)
    aggregator = LastPatchAggregator(embed_dim=8, topk=1)
    model = _classifier_stub(embed_dim=8, num_classes=3)
    raw_tokens = torch.randn(2, 5, 8, requires_grad=True)
    normalized = norm(raw_tokens)
    normalized.retain_grad()
    pooled, indices, _ = aggregator(normalized)
    logits = model.last_classify(pooled)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    F.multilabel_soft_margin_loss(logits, targets).backward()

    for gradient in (
        norm.weight.grad,
        norm.bias.grad,
        model.head.weight.grad,
        model.head.bias.grad,
        raw_tokens.grad,
        normalized.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    selected_mask = torch.zeros_like(normalized, dtype=torch.bool)
    selected_mask.scatter_(1, indices, True)
    assert normalized.grad[selected_mask].abs().sum() > 0
    assert normalized.grad[~selected_mask].abs().sum() == 0


def test_last_aggregator_amp_forward_backward_is_finite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    norm = nn.LayerNorm(8).to(device)
    aggregator = LastPatchAggregator(embed_dim=8, topk=1).to(device)
    model = _classifier_stub(embed_dim=8, num_classes=3).to(device)
    tokens = torch.randn(2, 5, 8, device=device, requires_grad=True)

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        normalized = norm(tokens)
        pooled, _, stability = aggregator(normalized)
        logits = model.last_classify(pooled)
        loss = logits.square().mean()
    loss.backward()

    assert torch.isfinite(stability).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
    assert torch.isfinite(model.head.weight.grad).all()


def test_last_mct_small_preserves_raw_class_and_cct_and_uses_patch_final_ln():
    torch.manual_seed(104)
    model = _small_kwargs().eval()
    baseline_kwargs = {
        **{
            'img_size': 32, 'input_size': 32, 'patch_size': 16,
            'embed_dim': 384, 'depth': 12, 'num_heads': 6,
            'mlp_ratio': 4, 'qkv_bias': True, 'num_classes': 20,
            'drop_rate': 0.0, 'drop_path_rate': 0.0,
            'attention_normalization': 'vanilla', 'bcss_variant': 'e0',
            'psl_variant': 'baseline', 'cti_bgt': False,
        },
        'patch_final_norm': False,
    }
    raw_model = MCTformerPlus(**baseline_kwargs).eval()
    transferable = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith('head.')
    }
    result = raw_model.load_state_dict(transferable, strict=False)
    assert result.missing_keys == ['head.weight', 'head.bias']
    assert not result.unexpected_keys
    inputs = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        raw_cls, raw_patch, raw_attn, raw_cct = raw_model.forward_features(inputs)
        cls, patch, attn, cct = model.forward_features(inputs)
        outputs = model(inputs)
        pooled, _, _ = model.last_aggregate(patch)
        expected_patch = model.norm(raw_patch)
        expected_patch_logits = model.last_classify(pooled)

    torch.testing.assert_close(cls, raw_cls, rtol=0, atol=0)
    torch.testing.assert_close(patch, expected_patch, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(cct), torch.stack(raw_cct), rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(attn), torch.stack(raw_attn), rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], cls.mean(dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], torch.stack(cct), rtol=0, atol=0)
    torch.testing.assert_close(
        outputs[2], expected_patch_logits, rtol=0, atol=0
    )
    assert model.head.kernel_size == (1, 1)


def test_last_mct_cam_uses_shared_spatial_head_and_pooled_gating(monkeypatch):
    torch.manual_seed(105)
    model = _small_kwargs(cam=True).eval()
    inputs = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        cls, patch, attention_heads, _ = model.forward_features(inputs)
        pooled, _, _ = model.last_aggregate(patch)
        expected_patch_label = (model.last_classify(pooled) > 0).to(inputs.dtype)
        patch_grid = patch.reshape(1, 2, 2, 384).permute(0, 3, 1, 2)
        spatial_scores = model.head(patch_grid.contiguous())
        attention = torch.stack(attention_heads).mean(dim=2)
        expected_cam = model.get_cam(spatial_scores, attention)
    monkeypatch.setattr(
        model, 'gwrp', lambda _scores: (_ for _ in ()).throw(
            AssertionError('Last-MCT CAM gating must not call GWRP')
        )
    )
    with torch.inference_mode():
        cls_label, patch_label, cam = model.forward_with_label(inputs)

    torch.testing.assert_close(cls_label, (cls.mean(-1) > 0).to(inputs.dtype))
    torch.testing.assert_close(patch_label, expected_patch_label)
    torch.testing.assert_close(cam, expected_cam, rtol=0, atol=0)


def test_last_mct_scope_and_checkpoint_metadata_are_strict():
    assert checkpoint_last_mct_enabled({}) is False
    assert checkpoint_last_mct_enabled({'last_mct': True}) is True
    assert validate_mctformerplus_final_norm_checkpoint(
        {'patch_final_norm': True, 'last_mct': True},
        False, True, True,
    ) is False
    with pytest.raises(ValueError, match='requires patch_final_norm'):
        validate_mctformerplus_final_norm_checkpoint(
            {'last_mct': True}, False, False, True
        )
    with pytest.raises(TypeError, match='must be a boolean'):
        checkpoint_last_mct_enabled({'last_mct': 1})
    with pytest.raises(ValueError, match='requires patch_final_norm'):
        MCTformerPlus(**{
            **{
                'img_size': 32, 'input_size': 32, 'patch_size': 16,
                'embed_dim': 384, 'depth': 12, 'num_heads': 6,
                'mlp_ratio': 4, 'qkv_bias': True, 'num_classes': 20,
                'attention_normalization': 'vanilla', 'bcss_variant': 'e0',
                'psl_variant': 'baseline', 'cti_bgt': False,
            },
            'last_mct': True,
        })
    with pytest.raises(ValueError, match=r'restricted to MCTformer\+-Small'):
        MCTformerPlus(
            img_size=32, input_size=32, patch_size=16, embed_dim=32,
            depth=2, num_heads=4, mlp_ratio=2, qkv_bias=True,
            num_classes=2, attention_normalization='vanilla',
            bcss_variant='e0', psl_variant='baseline', cti_bgt=False,
            patch_final_norm=True, last_mct=True,
        )
