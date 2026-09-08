from collections import OrderedDict

import pytest
import torch

from models.class_token_pooling import ClassWiseWeightedPooling
from models.mctformer_plus import (
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
    checkpoint_class_token_init,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_class_token_init_checkpoint,
)


def _small(*, cam=False, class_token_init='cwp'):
    return build_mctformerplus(
        'small',
        cam=cam,
        num_classes=20,
        input_size=32,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attention_normalization='vanilla',
        bcss_variant='e0',
        psl_variant='baseline',
        cti_bgt=False,
        final_norm=False,
        patch_final_norm=False,
        last_mct=False,
        class_stable_last=False,
        class_token_init=class_token_init,
    )


def test_cwp_shapes_patch_softmax_and_direct_reference():
    torch.manual_seed(3)
    pooler = ClassWiseWeightedPooling(num_classes=4, embed_dim=8)
    patches = torch.randn(2, 5, 8)
    pooled, attention = pooler(patches)
    assert pooled.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 5)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(2, 4), rtol=0, atol=1e-6
    )
    logits = torch.matmul(
        pooler.class_queries.float().unsqueeze(0),
        patches.float().transpose(1, 2),
    ) / (8 ** 0.5)
    reference_attention = torch.softmax(logits, dim=-1)
    reference_pooled = torch.matmul(reference_attention, patches.float())
    torch.testing.assert_close(attention, reference_attention, rtol=0, atol=0)
    torch.testing.assert_close(pooled, reference_pooled, rtol=0, atol=0)
    # Each class row normalizes over patches; there is no class-axis competition.
    assert not torch.allclose(attention.sum(dim=1), torch.ones(2, 5))


def test_cwp_is_image_conditioned_and_equal_queries_give_equal_outputs():
    torch.manual_seed(4)
    pooler = ClassWiseWeightedPooling(num_classes=3, embed_dim=6)
    patches = torch.randn(2, 7, 6)
    pooled, attention = pooler(patches)
    assert not torch.equal(attention[0], attention[1])
    assert not torch.equal(pooled[0], pooled[1])
    with torch.no_grad():
        pooler.class_queries.copy_(pooler.class_queries[:1].expand_as(
            pooler.class_queries
        ))
    pooled, attention = pooler(patches)
    torch.testing.assert_close(
        attention[:, :1].expand_as(attention), attention, rtol=0, atol=0
    )
    torch.testing.assert_close(
        pooled[:, :1].expand_as(pooled), pooled, rtol=0, atol=0
    )


def test_cwp_query_gradient_is_finite_and_nonzero():
    torch.manual_seed(5)
    pooler = ClassWiseWeightedPooling(num_classes=4, embed_dim=8)
    patches = torch.randn(2, 6, 8, requires_grad=True)
    pooled, _ = pooler(patches)
    pooled.square().sum().backward()
    assert pooler.class_queries.grad is not None
    assert torch.isfinite(pooler.class_queries.grad).all()
    assert pooler.class_queries.grad.abs().sum() > 0
    assert patches.grad is not None and patches.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cwp_amp_forward_backward_is_finite():
    pooler = ClassWiseWeightedPooling(20, 384).cuda()
    patches = torch.randn(
        2, 16, 384, device='cuda', dtype=torch.float16, requires_grad=True
    )
    with torch.cuda.amp.autocast():
        pooled, attention = pooler(patches)
        loss = pooled.square().mean()
    loss.backward()
    assert pooled.dtype == patches.dtype
    assert attention.dtype == torch.float32
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(attention).all()
    assert torch.isfinite(pooler.class_queries.grad).all()


def test_cwp_model_has_no_cls_token_and_exposes_layer_zero():
    torch.manual_seed(6)
    model = _small()
    named = dict(model.named_parameters())
    state = model.state_dict()
    assert model.cls_token is None
    assert 'cls_token' not in named and 'cls_token' not in state
    assert state['class_token_pooler.class_queries'].shape == (20, 384)
    model.train()
    features = model.forward_features(torch.randn(1, 3, 32, 32), return_aux=True)
    cls, patches, attention_layers, all_cls, auxiliary = features
    assert cls.shape == (1, 20, 384)
    assert patches.shape == (1, 4, 384)
    assert len(attention_layers) == len(all_cls) == 12
    assert auxiliary['initial_class_tokens'].shape == (1, 20, 384)
    assert auxiliary['initial_patch_tokens'].shape == (1, 4, 384)
    pooling = auxiliary['class_token_pooling_attention']
    assert pooling.shape == (1, 20, 4)
    assert (pooling.sum(-1) - 1).abs().max() < 1e-6
    assert all(torch.isfinite(torch.tensor(value)) for value in
               auxiliary['class_token_pooling_diagnostics'].values())


def test_baseline_default_and_explicit_paths_are_bit_exact():
    torch.manual_seed(7)
    implicit = _small(cam=True, class_token_init='baseline')
    torch.manual_seed(7)
    explicit = build_mctformerplus(
        'small', cam=True, num_classes=20, input_size=32,
        drop_rate=0.0, drop_path_rate=0.0,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
    )
    assert tuple(implicit.state_dict()) == tuple(explicit.state_dict())
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    image = torch.randn(1, 3, 32, 32)
    implicit.eval()
    explicit.eval()
    with torch.no_grad():
        torch.testing.assert_close(implicit(image), explicit(image), rtol=0, atol=0)
        implicit_features = implicit.forward_features(image)
        explicit_features = explicit.forward_features(image)
    for left, right in zip(implicit_features[:2], explicit_features[:2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


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


def test_cwp_pretrained_policy_and_checkpoint_contract():
    torch.manual_seed(8)
    model = _small()
    original_queries = model.class_token_pooler.class_queries.detach().clone()
    adapted, report = adapt_deit_checkpoint_for_mctformerplus(
        _synthetic_deit_source(), model, num_classes=20
    )
    assert report['class_token_init'] == 'cwp'
    assert report['deit_cls_token_used'] is False
    assert report['deit_cls_token_policy'] == 'discarded'
    assert report['class_pooling'] == 'class-wise weighted pooling'
    assert report['pooling_softmax_axis'] == 'patch'
    assert report['class_query_shape'] == [20, 384]
    assert report['randomly_initialized_keys'] == [
        'class_token_pooler.class_queries', 'head.bias', 'head.weight'
    ]
    assert 'cls_token' not in adapted
    torch.testing.assert_close(
        adapted['class_token_pooler.class_queries'], original_queries,
        rtol=0, atol=0,
    )
    model.load_state_dict(adapted, strict=True)

    checkpoint = {
        'model': model.state_dict(),
        'model_spec': model_spec_from_instance(model),
        'class_token_init': 'cwp',
    }
    assert checkpoint_class_token_init(checkpoint) == 'cwp'
    assert validate_mctformerplus_class_token_init_checkpoint(
        checkpoint, 'cwp'
    ) == 'cwp'
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, 'mctformerplus'
    )
    assert resolution['class_token_init'] == 'cwp'
    with pytest.raises(ValueError, match='explicit'):
        checkpoint_class_token_init({'model': model.state_dict()})
    with pytest.raises(ValueError, match='does not match'):
        validate_mctformerplus_class_token_init_checkpoint(checkpoint, 'baseline')


def test_cwp_rejects_incompatible_variants():
    with pytest.raises(ValueError, match='Small'):
        build_mctformerplus('tiny', num_classes=20, class_token_init='cwp')
    with pytest.raises(ValueError, match='final_norm=False'):
        build_mctformerplus(
            'small', num_classes=20, class_token_init='cwp', final_norm=True
        )
