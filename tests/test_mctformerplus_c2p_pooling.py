import pytest
import torch
from models.mctformer_plus import (
    build_mctformerplus, model_spec_from_instance,
    validate_mctformerplus_patch_pooling_checkpoint,
)


def model(pooling='c2p', **kwargs):
    return build_mctformerplus('small', input_size=32, num_classes=20,
                               patch_pooling=pooling, **kwargs)


@pytest.mark.parametrize('layers', ['last3', 'all'])
def test_weights_formula_last3_uniform_gap_and_gradients(layers):
    m = model(c2p_pooling_layers=layers)
    records = [torch.rand(2, 6, 24, 24, requires_grad=True) for _ in range(12)]
    logits = torch.randn(2, 20, 2, 2, requires_grad=True)
    w = m.c2p_spatial_weights(records, 4)
    used = records if layers == 'all' else records[-3:]
    direct = torch.stack(used).mean((0, 2))[:, :20, 20:]
    direct = direct / direct.sum(-1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(w, direct)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 20))
    z = m.c2p_pool(logits, records)
    assert z.shape == (2, 20)
    torch.testing.assert_close(z, (direct * logits.flatten(2)).sum(-1))
    z.square().mean().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    if layers == 'last3':
        assert all(t.grad is None for t in records[:-3])
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in used)
    uniform = [torch.ones_like(t) for t in records]
    torch.testing.assert_close(m.c2p_pool(logits, uniform), logits.mean((2, 3)))


@pytest.mark.parametrize('initializer', ['baseline', 'cwp', 'residual_cwp'])
def test_initialization_untouched_and_gwrp_exact(initializer):
    torch.manual_seed(9)
    a = model('gwrp', class_token_init=initializer).eval()
    torch.manual_seed(9)
    b = model('c2p', class_token_init=initializer).eval()
    assert a.class_token_init == b.class_token_init == initializer
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        cls, patches, _, raw = a.forward_features(x)
        maps = a.head(patches.reshape(2, 2, 2, 384).permute(0, 3, 1, 2).contiguous())
        sorted_values = maps.flatten(2).transpose(1, 2).sort(dim=1, descending=True).values
        decay = torch.logspace(0, 3, 4, base=a.decay_parameter)
        reference = (sorted_values * decay[None, :, None]).sum(1) / decay.sum()
        out = a(x)
        torch.testing.assert_close(out[2], reference, rtol=0, atol=0)
        torch.testing.assert_close(out[0], b(x)[0], rtol=0, atol=0)
        torch.testing.assert_close(out[1], torch.stack(raw), rtol=0, atol=0)


@pytest.mark.parametrize('layers', ['last3', 'all'])
@pytest.mark.parametrize('reduction', ['mean', 'product'])
def test_actual_training_attention_and_no_sort(monkeypatch, layers, reduction):
    m = model(c2p_pooling_layers=layers, c2p_pooling_reduction=reduction).train()
    records = []
    def save(module, args, output):
        output[1].retain_grad()
        records.append(output[1])
    handles = [b.attn.register_forward_hook(save) for b in m.blocks]
    def forbidden(*args, **kwargs):
        raise AssertionError('c2p must not sort or use GWRP')
    monkeypatch.setattr(torch, 'sort', forbidden)
    monkeypatch.setattr(m, 'gwrp', forbidden)
    out = m(torch.randn(2, 3, 32, 32))
    out[2].square().mean().backward()
    assert m.head.weight.grad.abs().sum() > 0
    for attention in (records if layers == 'all' else records[-3:]):
        assert attention.grad is not None
        assert attention.grad[:, :, :20, 20:].abs().sum() > 0
    for h in handles:
        h.remove()


@pytest.mark.parametrize('layers', ['last3', 'all'])
@pytest.mark.parametrize('reduction', ['mean', 'product'])
def test_checkpoint_and_native_cam_unchanged(tmp_path, layers, reduction):
    a = model('gwrp', cam=True).eval()
    b = model('c2p', cam=True, c2p_pooling_layers=layers, c2p_pooling_reduction=reduction).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    payload = {'model': b.state_dict(), 'model_spec': model_spec_from_instance(b)}
    path = tmp_path / 'checkpoint.pth'
    torch.save(payload, path)
    loaded = torch.load(path)
    validate_mctformerplus_patch_pooling_checkpoint(loaded, 'c2p', layers, reduction)
    with pytest.raises(ValueError, match='c2p_pooling_reduction'):
        validate_mctformerplus_patch_pooling_checkpoint(
            loaded, 'c2p', layers, 'product' if reduction == 'mean' else 'mean')
    with pytest.raises(ValueError, match='c2p_pooling_layers'):
        validate_mctformerplus_patch_pooling_checkpoint(loaded, 'c2p', 'all' if layers == 'last3' else 'last3')
    validate_mctformerplus_patch_pooling_checkpoint({}, 'gwrp')
    with pytest.raises(ValueError, match='patch_pooling'):
        validate_mctformerplus_patch_pooling_checkpoint(loaded, 'gwrp')
    classifier = model(c2p_pooling_layers=layers, c2p_pooling_reduction=reduction).eval()
    classifier.load_state_dict(loaded['model'], strict=True)
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    _, patch_labels, cams = b.forward_with_label(x)
    with torch.no_grad():
        expected = (classifier(x)[2] > 0).float()
    torch.testing.assert_close(patch_labels, expected, rtol=0, atol=0)
    torch.testing.assert_close(cams, b(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('layers', ['last3', 'all'])
@pytest.mark.parametrize('reduction', ['mean', 'product'])
def test_amp_finite(layers, reduction):
    m = model(c2p_pooling_layers=layers, c2p_pooling_reduction=reduction).cuda().train()
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(torch.randn(2, 3, 32, 32, device='cuda'))
        loss = out[2].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_old_checkpoint_last3_default_and_unchanged_initialization():
    validate_mctformerplus_patch_pooling_checkpoint({'model_spec': {'patch_pooling': 'c2p'}}, 'c2p')
    torch.manual_seed(7)
    a = model().eval()
    torch.manual_seed(7)
    b = model(c2p_pooling_layers='all').eval()
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        old, new = a(x), b(x)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    assert not torch.allclose(old[2], new[2])


def test_average_raw_attention_before_conditionalizing():
    m = model(c2p_pooling_layers='all')
    records = [torch.ones(1, 6, 24, 24) for _ in range(12)]
    # Unequal layer patch mass: averaging conditional maps would be incorrect.
    records[0][:, :, :20, 20:] = torch.tensor([90., 10., 0., 0.])
    w = m.c2p_spatial_weights(records, 4)
    reference = torch.tensor([101., 21., 11., 11.]) / 144
    torch.testing.assert_close(w, reference.expand(1, 20, 4))


@pytest.mark.parametrize('layers', ['last3', 'all'])
def test_product_reference_gradients_uniform_and_not_geometric_mean(layers):
    m = model(c2p_pooling_layers=layers, c2p_pooling_reduction='product')
    torch.manual_seed(21)
    records = [(torch.rand(2, 6, 24, 24, dtype=torch.float64) + .1).requires_grad_()
               for _ in range(12)]
    used = records if layers == 'all' else records[-3:]
    by_layer = torch.stack(used)[:, :, :, :20, 20:].mean(2)
    direct = by_layer.prod(0)
    direct = direct / direct.sum(-1, keepdim=True)
    weights = m.c2p_spatial_weights(records, 4)
    torch.testing.assert_close(weights, direct, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 20, dtype=torch.float64))
    logits = torch.randn(2, 20, 2, 2, dtype=torch.float64, requires_grad=True)
    pooled = m.c2p_pool(logits, records)
    assert pooled.shape == (2, 20)
    loss = pooled.square().mean()
    grads = torch.autograd.grad(loss, used + [logits], retain_graph=True)
    reference_loss = (direct * logits.flatten(2)).sum(-1).square().mean()
    refs = torch.autograd.grad(reference_loss, used + [logits])
    for grad, ref in zip(grads, refs):
        torch.testing.assert_close(grad, ref, rtol=1e-10, atol=1e-12)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    geometric = by_layer.log().mean(0).softmax(-1)
    assert not torch.allclose(weights, geometric)
    torch.testing.assert_close(m.c2p_pool(logits, [torch.ones_like(r) for r in records]), logits.mean((2, 3)))


def test_product_tiny_attention_stays_normalized():
    m = model(c2p_pooling_layers='all', c2p_pooling_reduction='product')
    records = [torch.full((1, 6, 24, 24), 1e-6) for _ in range(12)]
    records[0][:, :, :20, 20:] *= torch.tensor([1., 2., 3., 4.])
    weights = m.c2p_spatial_weights(records, 4)
    torch.testing.assert_close(weights, (torch.arange(1., 5.) / 10).expand(1, 20, 4), rtol=2e-5, atol=1e-6)
    assert torch.stack(records).mean(2).prod(0).eq(0).all()  # direct FP32 underflows
    # Exact AMP zeros are floored at the accumulation dtype's machine tiny,
    # not a tunable epsilon. Other spatial positions remain finite.
    records[0][:, :, :20, 20] = 0
    weights = m.c2p_spatial_weights(records, 4)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 20))


def test_product_flag_preserves_initialization_and_legacy_mean_exact():
    torch.manual_seed(8)
    mean = model(c2p_pooling_layers='all').eval()
    torch.manual_seed(8)
    product = model(c2p_pooling_layers='all', c2p_pooling_reduction='product').eval()
    for key in mean.state_dict():
        torch.testing.assert_close(mean.state_dict()[key], product.state_dict()[key], rtol=0, atol=0)
    records = [torch.rand(2, 6, 24, 24) for _ in range(12)]
    a = torch.stack([r[:, :, :20, 20:] for r in records]).mean((0, 2))
    expected = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(mean.c2p_spatial_weights(records, 4), expected, rtol=0, atol=0)
