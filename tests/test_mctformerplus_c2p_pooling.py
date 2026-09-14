import pytest
import torch
from models.mctformer_plus import (
    build_mctformerplus, model_spec_from_instance,
    validate_mctformerplus_patch_pooling_checkpoint,
)


def model(pooling='c2p', **kwargs):
    return build_mctformerplus('small', input_size=32, num_classes=20,
                               patch_pooling=pooling, **kwargs)


def test_weights_formula_last3_uniform_gap_and_gradients():
    m = model()
    records = [torch.rand(2, 6, 24, 24, requires_grad=True) for _ in range(12)]
    logits = torch.randn(2, 20, 2, 2, requires_grad=True)
    w = m.c2p_spatial_weights(records, 4)
    direct = torch.stack(records[-3:]).mean((0, 2))[:, :20, 20:]
    direct = direct / direct.sum(-1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(w, direct)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 20))
    z = m.c2p_pool(logits, records)
    assert z.shape == (2, 20)
    torch.testing.assert_close(z, (direct * logits.flatten(2)).sum(-1))
    z.square().mean().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert all(t.grad is None for t in records[:-3])
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in records[-3:])
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


def test_actual_training_attention_and_no_sort(monkeypatch):
    m = model().train()
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
    for attention in records[-3:]:
        assert attention.grad is not None
        assert attention.grad[:, :, :20, 20:].abs().sum() > 0
    for h in handles:
        h.remove()


def test_checkpoint_and_native_cam_unchanged(tmp_path):
    a = model('gwrp', cam=True).eval()
    b = model('c2p', cam=True).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    payload = {'model': b.state_dict(), 'model_spec': model_spec_from_instance(b)}
    path = tmp_path / 'checkpoint.pth'
    torch.save(payload, path)
    loaded = torch.load(path)
    validate_mctformerplus_patch_pooling_checkpoint(loaded, 'c2p')
    validate_mctformerplus_patch_pooling_checkpoint({}, 'gwrp')
    with pytest.raises(ValueError, match='patch_pooling'):
        validate_mctformerplus_patch_pooling_checkpoint(loaded, 'gwrp')
    classifier = model().eval()
    classifier.load_state_dict(loaded['model'], strict=True)
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    _, patch_labels, cams = b.forward_with_label(x)
    with torch.no_grad():
        expected = (classifier(x)[2] > 0).float()
    torch.testing.assert_close(patch_labels, expected, rtol=0, atol=0)
    torch.testing.assert_close(cams, b(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_amp_finite():
    m = model().cuda().train()
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(torch.randn(2, 3, 32, 32, device='cuda'))
        loss = out[2].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
