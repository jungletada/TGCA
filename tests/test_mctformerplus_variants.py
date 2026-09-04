from __future__ import annotations

from pathlib import Path

import pytest
import torch
from timm.models import create_model

from models.mctformer_plus import (
    MCTFORMERPLUS_VARIANTS,
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
    resolve_mctformerplus_variant,
)


EXPECTED = {
    'tiny': ('mctformerplus_tiny', 192, 3),
    'small': ('mctformerplus', 384, 6),
    'base': ('mctformerplus_base', 768, 12),
}


@pytest.mark.parametrize('variant', ('tiny', 'small', 'base'))
def test_registered_architecture_contract(variant):
    name, width, heads = EXPECTED[variant]
    model = create_model(
        name, pretrained=False, num_classes=20, input_size=448
    )
    spec = model_spec_from_instance(model)
    assert resolve_mctformerplus_variant(name) == variant
    assert resolve_mctformerplus_variant(variant) == variant
    assert spec == {
        'family': 'MCTformer+',
        'variant': variant,
        'model_name': name,
        'patch_size': [16, 16],
        'embed_dim': width,
        'depth': 12,
        'num_heads': heads,
        'head_dim': 64,
        'mlp_ratio': 4,
        'cam_class_to_patch_layers': 3,
        'cam_patch_to_patch_layers': 12,
    }
    assert len(model.blocks) == 12
    assert all(block.attn.num_heads == heads for block in model.blocks)
    assert model.num_classes == 20


def test_parameter_count_is_strictly_increasing():
    counts = []
    for variant in ('tiny', 'small', 'base'):
        model = build_mctformerplus(variant, num_classes=20, input_size=448)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts == [5717012, 22050836, 86568980]
    assert counts[0] < counts[1] < counts[2]


@pytest.mark.parametrize('variant', ('tiny', 'small', 'base'))
def test_training_forward_shapes(variant):
    width = EXPECTED[variant][1]
    model = build_mctformerplus(
        variant, num_classes=20, input_size=224, drop_path_rate=0.0
    ).eval()
    with torch.inference_mode():
        outputs = model(torch.randn(2, 3, 224, 224))
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 20)
    assert outputs[1].shape == (12, 2, 20, width)
    assert outputs[2].shape == (2, 20)
    assert all(torch.isfinite(value).all() for value in outputs)


@pytest.mark.parametrize('variant', ('tiny', 'small', 'base'))
def test_cam_policy_and_224_shape(variant):
    model = build_mctformerplus(
        variant, cam=True, num_classes=20, input_size=448,
        drop_path_rate=0.0,
    ).eval()
    assert model.n_layers == 3
    assert len(model.blocks) == 12
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 20, 14, 14)
    assert torch.isfinite(output).all()


def test_cam_448_shape_for_all_variants_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip('Required 448 width smoke uses CUDA on the execution host')
    for variant in ('tiny', 'small', 'base'):
        model = build_mctformerplus(
            variant, cam=True, num_classes=20, input_size=448,
            drop_path_rate=0.0,
        ).cuda().eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 448, 448, device='cuda'))
        assert output.shape == (1, 20, 28, 28)
        assert torch.isfinite(output).all()
        del model, output
        torch.cuda.empty_cache()


def _checkpoint(variant):
    model = build_mctformerplus(variant, num_classes=20, input_size=224)
    return {
        'model': model.state_dict(),
        'model_spec': model_spec_from_instance(model),
    }


@pytest.mark.parametrize('variant', ('tiny', 'small', 'base'))
def test_checkpoint_round_trip_and_variant_resolution(tmp_path, variant):
    payload = _checkpoint(variant)
    path = tmp_path / f'{variant}.pth'
    torch.save(payload, path)
    loaded = torch.load(path, map_location='cpu')
    name = EXPECTED[variant][0]
    resolution = resolve_mctformerplus_checkpoint_variant(loaded, name)
    assert resolution['variant'] == variant
    assert not resolution['legacy_small_import']
    left = build_mctformerplus(variant, num_classes=20, input_size=224).eval()
    right = build_mctformerplus(variant, num_classes=20, input_size=224).eval()
    left.load_state_dict(loaded['model'], strict=True)
    right.load_state_dict(loaded['model'], strict=True)
    inputs = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        left_output = left(inputs)
        right_output = right(inputs)
    for first, second in zip(left_output, right_output):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('checkpoint_variant', 'requested'),
    (
        ('tiny', 'mctformerplus'),
        ('base', 'mctformerplus_tiny'),
        ('small', 'mctformerplus_base'),
    ),
)
def test_checkpoint_cli_variant_mismatch_fails(checkpoint_variant, requested):
    with pytest.raises(ValueError, match='does not match requested'):
        resolve_mctformerplus_checkpoint_variant(
            _checkpoint(checkpoint_variant), requested
        )


def test_legacy_checkpoint_allowed_only_for_canonical_small_name():
    payload = {'model': _checkpoint('small')['model']}
    resolution = resolve_mctformerplus_checkpoint_variant(
        payload, 'mctformerplus'
    )
    assert resolution['legacy_small_import']
    for name in ('mctformerplus_tiny', 'mctformerplus_base'):
        with pytest.raises(ValueError, match='lacks model_spec'):
            resolve_mctformerplus_checkpoint_variant(payload, name)


def test_checkpoint_spec_state_width_mismatch_fails():
    payload = _checkpoint('tiny')
    payload['model_spec'] = dict(payload['model_spec'])
    payload['model_spec']['embed_dim'] = 384
    with pytest.raises(ValueError, match='model_spec mismatch'):
        resolve_mctformerplus_checkpoint_variant(
            payload, 'mctformerplus_tiny'
        )


def test_fixed_architecture_kwargs_cannot_be_overridden():
    with pytest.raises(ValueError, match='fixes embed_dim'):
        build_mctformerplus('tiny', embed_dim=384)
    with pytest.raises(ValueError, match='Unknown MCTformer'):
        resolve_mctformerplus_variant('mctformerplus_tiny_extra')
