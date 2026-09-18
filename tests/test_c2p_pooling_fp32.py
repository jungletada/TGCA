"""Pre-cast pooling readout; native Transformer/CAM arithmetic stays unchanged."""
import pytest
import torch

from models.mctformer_plus import (
    build_mctformerplus, model_spec_from_instance, validate_c2p_precision_checkpoint,
)
from models.tgca import token_group_normalize


def model(**kwargs):
    return build_mctformerplus(
        'small', input_size=32, num_classes=20, patch_pooling='c2p',
        c2p_pooling_layers='all', c2p_pooling_reduction='product', **kwargs)


def probabilities(logits, keep_fp32):
    return token_group_normalize(
        logits, torch.tensor([0, 1, 1], device=logits.device),
        torch.tensor([0], device=logits.device), mode='vanilla', return_fp32=keep_fp32)


def test_precast_probabilities_keep_underflowed_patch_evidence():
    logits = torch.tensor([[[[0., -20., -18.]]]], dtype=torch.float16)
    legacy, repaired = probabilities(logits, False), probabilities(logits, True)
    assert legacy.dtype == torch.float16 and repaired.dtype == torch.float32
    assert torch.equal(legacy, repaired.half())
    assert (legacy[..., 1:] == 0).all()
    assert (repaired[..., 1:] > 0).all()
    w = repaired[..., 1:].log().softmax(-1)
    torch.testing.assert_close(w, torch.tensor([[[[0.11920292, 0.88079708]]]]))


def test_precast_backward_avoids_half_probability_gradient_overflow():
    results = []
    for keep_fp32 in (False, True):
        logits = torch.tensor([[[[0., -10., -11.]]]], dtype=torch.float16,
                              requires_grad=True)
        p = probabilities(logits, keep_fp32)
        p.retain_grad()
        # Same all-product formula, 12 layers, non-underflowing probabilities.
        w = (12 * p[..., 1:].float().log()).softmax(-1)
        (w[..., 1].sum() * 32768).backward()
        results.append((p.grad, logits.grad))
    assert not torch.isfinite(results[0][0]).all()
    assert not torch.isfinite(results[0][1]).all()
    assert torch.isfinite(results[1][0]).all()
    assert torch.isfinite(results[1][1]).all()
    assert results[1][1].abs().sum() > 0


def test_fp32_pool_formula_gradients_and_no_new_parameters():
    m = model(c2p_pooling_fp32=True)
    records = [torch.rand(2, 20, 4, requires_grad=True) for _ in range(12)]
    maps = torch.randn(2, 20, 2, 2, requires_grad=True)
    output = m.c2p_pool(maps, [], records)
    direct = torch.stack(records).double().prod(0)
    direct = direct / direct.sum(-1, keepdim=True)
    torch.testing.assert_close(output.double(), (direct * maps.flatten(2)).sum(-1),
                               rtol=2e-5, atol=2e-6)
    output.square().sum().backward()
    assert maps.grad.abs().sum() > 0
    assert all(r.grad is not None and torch.isfinite(r.grad).all() and r.grad.abs().sum() > 0
               for r in records)
    assert m.state_dict().keys() == model().state_dict().keys()
    with pytest.raises(ValueError, match='pre-cast'):
        m.c2p_pool(maps, [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_amp_native_outputs_rng_and_cam_exact_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    m = model(c2p_pooling_fp32=True).cuda().train()
    images = torch.randn(2, 3, 32, 32, device='cuda')
    # Reseeding also checks that collecting FP32 records consumes no RNG.
    outputs = []
    for enabled in (False, True):
        m.c2p_pooling_fp32 = enabled
        torch.manual_seed(9)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            outputs.append(m.forward_features(images, return_aux=True))
        if not enabled:
            rng = torch.cuda.get_rng_state()
        else:
            assert torch.equal(rng, torch.cuda.get_rng_state())
    for old, new in zip(outputs[0][:2], outputs[1][:2]):
        assert torch.equal(old, new)
    for field in (2, 3):
        assert all(torch.equal(a, b) for a, b in zip(outputs[0][field], outputs[1][field]))
    records = outputs[1][4]['c2p_pooling_fp32_records']
    assert len(records) == 12 and all(r.dtype == torch.float32 and r.shape == (2, 20, 4) for r in records)
    assert all(r.dtype == torch.float16 for r in outputs[1][2])
    with torch.autocast('cuda', dtype=torch.float16):
        output = m(images)
        loss = torch.nn.functional.multilabel_soft_margin_loss(output[2], torch.ones(2, 20, device='cuda'))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.head.weight.grad.abs().sum() > 0
    assert all(b.attn.qkv.weight.grad.abs().sum() > 0 for b in m.blocks)

    checkpoint = {'model': m.state_dict(), 'model_spec': model_spec_from_instance(m)}
    path = tmp_path / 'checkpoint.pth'
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location='cpu')
    validate_c2p_precision_checkpoint(loaded, True)
    with pytest.raises(ValueError, match='c2p_pooling_fp32'):
        validate_c2p_precision_checkpoint(loaded, False)
    validate_c2p_precision_checkpoint({}, False)
    with pytest.raises(ValueError, match='c2p_pooling_fp32'):
        validate_c2p_precision_checkpoint({}, True)
    cam = model(cam=True, c2p_pooling_fp32=True).cuda().eval()
    cam.load_state_dict(loaded['model'], strict=True)
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        repaired_cam = cam(images)
        _, gate, gated_cam = cam.forward_with_label(images)
        assert torch.equal(gate, (m(images)[2] > 0).float())
        assert torch.equal(gated_cam, repaired_cam)
        cam.c2p_pooling_fp32 = False
        assert torch.equal(cam(images), repaired_cam)


def test_default_keeps_legacy_metadata_and_weights():
    torch.manual_seed(3)
    legacy = model().eval()
    torch.manual_seed(3)
    explicit = model(c2p_pooling_fp32=False).eval()
    assert 'c2p_pooling_fp32' not in model_spec_from_instance(legacy)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for a, b in zip(legacy(x), explicit(x)):
            assert torch.equal(a, b)
