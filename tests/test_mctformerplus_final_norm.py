from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mctformer_plus import (
    MCTformerPlus,
    MCTformerPlusCam,
    validate_mctformerplus_final_norm_checkpoint,
)
from utils import create_cam_model


def _kwargs(final_norm=False):
    return {
        'img_size': 32,
        'input_size': 32,
        'patch_size': 16,
        'embed_dim': 32,
        'depth': 2,
        'num_heads': 4,
        'mlp_ratio': 2,
        'qkv_bias': True,
        'num_classes': 2,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'attention_normalization': 'vanilla',
        'bcss_variant': 'e0',
        'psl_variant': 'baseline',
        'cti_bgt': False,
        'final_norm': final_norm,
    }


def _assert_tensor_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tensor_tree_equal(a, b)
    else:
        assert left == right


class _FinalNormMustNotRun(nn.Module):
    def forward(self, _tokens):
        raise AssertionError('final LayerNorm ran while final_norm=False')


def test_final_norm_false_is_default_and_does_not_call_norm():
    torch.manual_seed(11)
    explicit = MCTformerPlus(**_kwargs(final_norm=False)).eval()
    default_kwargs = _kwargs()
    default_kwargs.pop('final_norm')
    default = MCTformerPlus(**default_kwargs).eval()
    default.load_state_dict(explicit.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        explicit_output = explicit(inputs)
        default_output = default(inputs)
    _assert_tensor_tree_equal(explicit_output, default_output)

    explicit.norm = _FinalNormMustNotRun()
    with torch.inference_mode():
        without_norm_call = explicit(inputs)
    _assert_tensor_tree_equal(default_output, without_norm_call)


def test_final_norm_changes_only_final_split_and_keeps_raw_cct_tokens():
    torch.manual_seed(12)
    baseline = MCTformerPlus(**_kwargs(final_norm=False)).eval()
    final_ln = MCTformerPlus(**_kwargs(final_norm=True)).eval()
    final_ln.load_state_dict(baseline.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        raw_cls, raw_patch, raw_attn, raw_all_cls = baseline.forward_features(inputs)
        norm_cls, norm_patch, norm_attn, norm_all_cls = final_ln.forward_features(inputs)

    raw_sequence = torch.cat((raw_cls, raw_patch), dim=1)
    expected = final_ln.norm(raw_sequence)
    torch.testing.assert_close(norm_cls, expected[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(norm_patch, expected[:, 2:], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.stack(norm_all_cls), torch.stack(raw_all_cls), rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.stack(norm_attn), torch.stack(raw_attn), rtol=0, atol=0
    )
    assert norm_cls.shape == raw_cls.shape == (2, 2, 32)
    assert norm_patch.shape == raw_patch.shape == (2, 4, 32)
    assert torch.stack(norm_attn).shape == (2, 2, 4, 6, 6)


def test_final_norm_receives_nonzero_training_gradients():
    torch.manual_seed(13)
    model = MCTformerPlus(**_kwargs(final_norm=True)).train()
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    class_logits, _all_x_cls, patch_logits = model(inputs)
    loss = F.multilabel_soft_margin_loss(class_logits, targets)
    loss = loss + F.multilabel_soft_margin_loss(patch_logits, targets)
    loss.backward()

    for parameter in (model.norm.weight, model.norm.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.norm()) > 0.0


def test_final_norm_cam_uses_the_same_normalized_patch_tokens():
    torch.manual_seed(14)
    training_model = MCTformerPlus(**_kwargs(final_norm=True)).eval()
    cam_model = MCTformerPlusCam(**_kwargs(final_norm=True)).eval()
    cam_model.load_state_dict(training_model.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        train_features = training_model.forward_features(inputs)
        cam_features = cam_model.forward_features(inputs)
        class_tokens, patch_tokens, attention_heads, _ = cam_features
        patch_grid = patch_tokens.reshape(2, 2, 2, 32).permute(0, 3, 1, 2)
        patch_logits = cam_model.head(patch_grid.contiguous())
        head_mean = torch.stack(attention_heads).mean(dim=2)
        expected_cam = cam_model.get_cam(patch_logits, head_mean)
        native_cam = cam_model(inputs)

    torch.testing.assert_close(train_features[0], class_tokens, rtol=0, atol=0)
    torch.testing.assert_close(train_features[1], patch_tokens, rtol=0, atol=0)
    torch.testing.assert_close(expected_cam, native_cam, rtol=0, atol=0)
    assert native_cam.shape == (2, 2, 2, 2)


def test_cam_factory_and_checkpoint_metadata_require_explicit_match():
    args = Namespace(
        model='mctformerplus', num_classes=20, input_size=32,
        final_norm=True, attention_normalization='vanilla',
        attention_gamma=1.0, bcss_variant='e0',
        bcss_num_background_slots=1, bcss_tau=0.5, bcss_beta=0.5,
        bcss_cls_threshold=0.5, psl_variant='baseline',
        psl_interaction_layers=(11,), psl_relation_dim=384,
        psl_num_background_latents=1, cti_bgt=False,
        cti_bgt_weight=0.1, cti_bgt_n_layers=6,
        cti_bgt_affinity_start=4,
    )
    model = create_cam_model(args)
    assert model.final_norm is True

    assert validate_mctformerplus_final_norm_checkpoint({}, False) is False
    assert validate_mctformerplus_final_norm_checkpoint(
        {'final_norm': True}, True
    ) is True
    with pytest.raises(ValueError, match='does not match requested'):
        validate_mctformerplus_final_norm_checkpoint({}, True)
    with pytest.raises(ValueError, match='does not match requested'):
        validate_mctformerplus_final_norm_checkpoint(
            {'final_norm': True}, False
        )
    with pytest.raises(TypeError, match='must be a boolean'):
        validate_mctformerplus_final_norm_checkpoint(
            {'final_norm': 1}, True
        )


def test_final_norm_rejects_unrelated_persistent_semantic_path():
    with pytest.raises(ValueError, match='defined only for the native'):
        MCTformerPlus(**{
            **_kwargs(final_norm=True),
            'psl_variant': 'read_only',
            'psl_interaction_layers': (1,),
            'psl_relation_dim': 32,
        })
