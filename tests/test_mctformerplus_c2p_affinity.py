import pytest
import torch

from models.mctformer_plus import (build_mctformerplus, model_spec_from_instance,
                                  validate_mctformerplus_patch_pooling_checkpoint)


def model(layers='last3', **kwargs):
    return build_mctformerplus('small', input_size=32, num_classes=20,
                              patch_pooling='c2p', c2p_pooling_layers=layers,
                              c2p_pooling_reduction='product', **kwargs)


@pytest.mark.parametrize('layers', ['last3', 'all'])
def test_affinity_reference_orientation_and_gradients(layers):
    torch.manual_seed(42)
    m = model(layers, c2p_pooling_affinity=True)
    records = [torch.rand(2, 6, 24, 24, dtype=torch.float64, requires_grad=True)
               for _ in range(12)]
    logits = torch.randn(2, 20, 2, 2, dtype=torch.float64, requires_grad=True)
    w = m.c2p_spatial_weights(records, 4)
    p = torch.stack([r[:, :, 20:, 20:].mean(1) for r in records]).sum(0)
    r = torch.einsum('bij,bcj->bci', p, w)
    reference = r / r.sum(-1, keepdim=True)
    actual = m.c2p_affinity_weights(w, records, 4)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, 20, dtype=torch.float64))
    wrong = w @ p
    wrong /= wrong.sum(-1, keepdim=True)
    assert not torch.allclose(actual, wrong)
    z = m.c2p_pool(logits, records)
    assert z.shape == (2, 20)
    torch.testing.assert_close(z, (reference * logits.flatten(2)).sum(-1))
    z.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    for i, record in enumerate(records):
        assert torch.isfinite(record.grad).all()
        assert record.grad[:, :, 20:, 20:].abs().sum() > 0  # ALL layers
        if layers == 'all' or i >= 9:
            assert record.grad[:, :, :20, 20:].abs().sum() > 0


def test_identity_constant_and_no_row_normalization():
    m = model(c2p_pooling_affinity=True)
    records = [torch.rand(2, 6, 24, 24) for _ in range(12)]
    logits = torch.randn(2, 20, 2, 2)
    w = m.c2p_spatial_weights(records, 4)
    for r in records:
        r[:, :, 20:, 20:] = torch.eye(4)
    torch.testing.assert_close(m.c2p_affinity_weights(w, records, 4), w)
    for r in records:
        r[:, :, 20:, 20:] = 3
    torch.testing.assert_close(m.c2p_pool(logits, records), logits.mean((2, 3)))
    # Row masses must survive; row-conditioning P would incorrectly give GAP.
    for r in records:
        r[:, :, 20:, 20:] = torch.arange(1., 5.)[:, None]
    expected = (torch.arange(1., 5.) / 10).expand(2, 20, 4)
    torch.testing.assert_close(m.c2p_affinity_weights(w, records, 4), expected)


@pytest.mark.parametrize('layers', ['last3', 'all'])
def test_disabled_exact_checkpoint_gating_native_cam(tmp_path, layers):
    torch.manual_seed(4)
    old = model(layers).eval()
    torch.manual_seed(4)
    disabled = model(layers, c2p_pooling_affinity=False).eval()
    enabled = model(layers, c2p_pooling_affinity=True).eval()
    enabled.load_state_dict(old.state_dict(), strict=True)
    for k in old.state_dict():
        torch.testing.assert_close(old.state_dict()[k], disabled.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for a, b in zip(old(x), disabled(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(old(x)[:2], enabled(x)[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    path = tmp_path / 'checkpoint.pth'
    torch.save({'model': enabled.state_dict(), 'model_spec': model_spec_from_instance(enabled)}, path)
    checkpoint = torch.load(path)
    validate_mctformerplus_patch_pooling_checkpoint(checkpoint, 'c2p', layers, 'product', True)
    with pytest.raises(ValueError, match='affinity'):
        validate_mctformerplus_patch_pooling_checkpoint(checkpoint, 'c2p', layers, 'product')
    validate_mctformerplus_patch_pooling_checkpoint({}, 'gwrp')
    cam_old = model(layers, cam=True).eval()
    cam_new = model(layers, cam=True, c2p_pooling_affinity=True).eval()
    for m in [cam_old, cam_new]:
        m.load_state_dict(checkpoint['model'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(cam_old(x), cam_new(x), rtol=0, atol=0)
        _, patch_labels, cams = cam_new.forward_with_label(x)
        torch.testing.assert_close(patch_labels, (enabled(x)[2] > 0).float(), rtol=0, atol=0)
        torch.testing.assert_close(cams, cam_old(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('layers', ['last3', 'all'])
def test_amp_real_forward_backward_and_fp32_propagation(layers):
    m = model(layers, c2p_pooling_affinity=True).cuda().train()
    records = [torch.rand(2, 6, 24, 24, device='cuda', dtype=torch.float16) for _ in range(12)]
    w = m.c2p_spatial_weights(records, 4)
    ref = m.c2p_affinity_weights(w, records, 4)
    with torch.autocast('cuda', dtype=torch.float16):
        actual = m.c2p_affinity_weights(w, records, 4)
        out = m(torch.randn(2, 3, 32, 32, device='cuda'))
        loss = out[2].square().mean()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(loss) and m.head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
