from collections import OrderedDict

import pytest
import torch

from models.class_token_pooling import ResidualClassWiseWeightedPooling
from models.mctformer_plus import (
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
    checkpoint_class_token_init,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_class_token_init_checkpoint,
)


def _small(*, cam=False, class_token_init='residual_cwp'):
    return build_mctformerplus(
        'small', cam=cam, num_classes=20, input_size=32,
        drop_rate=0.0, drop_path_rate=0.0,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
        final_norm=False, patch_final_norm=False,
        last_mct=False, class_stable_last=False,
        class_token_init=class_token_init,
    )


def _synthetic_deit_source():
    baseline = _small(class_token_init='baseline')
    source = OrderedDict()
    for key, value in baseline.state_dict().items():
        if key in {'pos_embed_cls', 'pos_embed_pat', 'head.weight', 'head.bias'}:
            continue
        source[key] = value.detach().clone()
    source['cls_token'] = source['cls_token'][:, :1].clone()
    source['head.weight'] = torch.randn(1000, 384)
    source['head.bias'] = torch.randn(1000)
    return {'model': source}


def test_residual_pooler_parameters_shapes_and_normalization():
    torch.manual_seed(11)
    pooler = ResidualClassWiseWeightedPooling(4, 8)
    patches = torch.randn(2, 7, 8)
    pooled, attention = pooler(patches)
    assert pooled.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 7)
    assert pooler.alpha.shape == ()
    torch.testing.assert_close(pooler.alpha, torch.tensor(0.1))
    torch.testing.assert_close(
        attention.sum(-1), torch.ones(2, 4), rtol=0, atol=1e-6
    )


def test_residual_cwp_initial_token_formula_is_exact():
    torch.manual_seed(12)
    model = _small()
    model.train()
    image = torch.randn(2, 3, 32, 32)
    patches = model.patch_embed(image) + model.pos_embed_pat
    image_tokens, expected_attention = model.class_token_pooler(patches)
    expected = (
        model.cls_token.expand(2, -1, -1)
        + model.class_token_pooler.alpha * image_tokens
        + model.pos_embed_cls
    )
    *_, auxiliary = model.forward_features(image, return_aux=True)
    torch.testing.assert_close(
        auxiliary['initial_class_tokens'], expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        auxiliary['class_token_pooling_attention'],
        expected_attention, rtol=0, atol=0,
    )
    assert auxiliary['class_token_pooling_diagnostics'][
        'residual_cwp_alpha'
    ] == pytest.approx(0.1)


def test_residual_cwp_gradients_reach_prior_pooling_and_patch_embedding():
    torch.manual_seed(13)
    model = _small()
    model.train()
    image = torch.randn(2, 3, 32, 32)
    *_, auxiliary = model.forward_features(image, return_aux=True)
    loss = auxiliary['initial_class_tokens'].square().mean()
    loss.backward()
    parameters = (
        model.cls_token,
        model.class_token_pooler.alpha,
        model.class_token_pooler.class_queries,
        model.patch_embed.proj.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_residual_cwp_amp_forward_backward_is_finite():
    pooler = ResidualClassWiseWeightedPooling(20, 384).cuda()
    patches = torch.randn(
        2, 16, 384, device='cuda', dtype=torch.float16, requires_grad=True
    )
    pretrained_cls = torch.randn(
        1, 20, 384, device='cuda', dtype=torch.float16, requires_grad=True
    )
    with torch.cuda.amp.autocast():
        pooled, attention = pooler(patches)
        tokens = pretrained_cls + pooler.alpha * pooled
        loss = tokens.square().mean()
    loss.backward()
    for value in (
            tokens, attention, patches.grad, pretrained_cls.grad,
            pooler.class_queries.grad, pooler.alpha.grad):
        assert value is not None and torch.isfinite(value).all()


def test_residual_cwp_pretrained_policy_and_checkpoint_contract():
    torch.manual_seed(14)
    model = _small()
    original_queries = model.class_token_pooler.class_queries.detach().clone()
    original_alpha = model.class_token_pooler.alpha.detach().clone()
    source = _synthetic_deit_source()
    source_cls = source['model']['cls_token'].clone()
    adapted, report = adapt_deit_checkpoint_for_mctformerplus(
        source, model, num_classes=20
    )
    assert report['class_token_init'] == 'residual_cwp'
    assert report['deit_cls_token_used'] is True
    assert report['class_token_source_policy'] == 'repeated_for_all_classes'
    assert report['alpha_initialization'] == 0.1
    torch.testing.assert_close(
        adapted['cls_token'], source_cls.repeat(1, 20, 1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        adapted['class_token_pooler.class_queries'], original_queries,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        adapted['class_token_pooler.alpha'], original_alpha, rtol=0, atol=0
    )
    assert report['randomly_initialized_keys'] == [
        'class_token_pooler.alpha',
        'class_token_pooler.class_queries',
        'head.bias',
        'head.weight',
    ]
    model.load_state_dict(adapted, strict=True)

    checkpoint = {
        'model': model.state_dict(),
        'model_spec': model_spec_from_instance(model),
        'class_token_init': 'residual_cwp',
    }
    assert checkpoint_class_token_init(checkpoint) == 'residual_cwp'
    assert validate_mctformerplus_class_token_init_checkpoint(
        checkpoint, 'residual_cwp'
    ) == 'residual_cwp'
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, 'mctformerplus'
    )
    assert resolution['class_token_init'] == 'residual_cwp'


def test_residual_cwp_parameter_count_and_compatibility_scope():
    baseline = _small(class_token_init='baseline')
    residual = _small()
    assert sum(p.numel() for p in residual.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    ) == 20 * 384 + 1
    with pytest.raises(ValueError, match='Small'):
        build_mctformerplus(
            'tiny', num_classes=20, class_token_init='residual_cwp'
        )
    with pytest.raises(ValueError, match='final_norm=False'):
        build_mctformerplus(
            'small', num_classes=20, class_token_init='residual_cwp',
            final_norm=True,
        )


def test_residual_cwp_cam_uses_standard_outputs():
    torch.manual_seed(15)
    model = _small(cam=True).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        class_label, patch_label, cam = model.forward_with_label(image)
    assert class_label.shape == patch_label.shape == (1, 20)
    assert cam.shape == (1, 20, 2, 2)
    assert torch.isfinite(cam).all()
