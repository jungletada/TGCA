from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from models.mctformer_plus import (
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
)


CACHE = Path('/home/peng/.cache/torch/hub/checkpoints')
OFFICIAL = {
    'tiny': (
        'deit_tiny_patch16_224-a1311bcf.pth',
        'a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97',
        192,
    ),
    'small': (
        'deit_small_patch16_224-cd65a155.pth',
        'cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f',
        384,
    ),
    'base': (
        'deit_base_patch16_224-b5f2ef4d.pth',
        'b5f2ef4d686982dcdab24fe285fd08fff40db01550d8d4833167a73dd85ca7a8',
        768,
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('variant', ('tiny', 'small', 'base'))
def test_exact_official_pretrained_adaptation(variant):
    filename, expected_hash, width = OFFICIAL[variant]
    path = CACHE / filename
    assert path.is_file(), f'Official checkpoint absent: {path}'
    assert sha256(path) == expected_hash
    source = torch.load(path, map_location='cpu')
    model = build_mctformerplus(variant, num_classes=20, input_size=448)
    original_head = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key in {'head.weight', 'head.bias'}
    }
    adapted, report = adapt_deit_checkpoint_for_mctformerplus(
        source, model, num_classes=20
    )
    incompatible = model.load_state_dict(adapted, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert report['passed']
    assert report['source_embed_dim'] == width
    assert report['target_embed_dim'] == width
    assert report['source_depth'] == report['target_depth'] == 12
    assert report['randomly_initialized_keys'] == ['head.bias', 'head.weight']
    assert report['unexpected_keys'] == []
    assert report['shape_mismatches'] == []
    assert adapted['cls_token'].shape == (1, 20, width)
    assert adapted['pos_embed_cls'].shape == (1, 20, width)
    assert adapted['pos_embed_pat'].shape == (1, 784, width)
    assert torch.equal(adapted['head.weight'], original_head['head.weight'])
    assert torch.equal(adapted['head.bias'], original_head['head.bias'])
    assert all(
        torch.isfinite(value).all()
        for value in adapted.values() if value.is_floating_point()
    )
    for layer in range(12):
        assert f'blocks.{layer}.attn.qkv.weight' in adapted


def test_pretrained_width_mismatch_fails():
    source = torch.load(CACHE / OFFICIAL['tiny'][0], map_location='cpu')
    model = build_mctformerplus('small', num_classes=20, input_size=448)
    with pytest.raises(ValueError, match='source embed_dim'):
        adapt_deit_checkpoint_for_mctformerplus(source, model, num_classes=20)


def test_pretrained_unknown_source_key_fails():
    source = torch.load(CACHE / OFFICIAL['tiny'][0], map_location='cpu')
    source = {'model': dict(source['model'])}
    source['model']['not_an_official_key'] = torch.zeros(1)
    model = build_mctformerplus('tiny', num_classes=20, input_size=448)
    with pytest.raises(ValueError, match='unexpected'):
        adapt_deit_checkpoint_for_mctformerplus(source, model, num_classes=20)
