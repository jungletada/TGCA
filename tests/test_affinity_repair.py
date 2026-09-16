import pytest
import torch
from models.mctformer_plus import aggregate_p2p, propagate_c2p_weights, build_mctformerplus


def records(dtype=torch.float32, device='cpu'):
    torch.manual_seed(15)
    return [torch.rand(2, 3, 11, 11, dtype=dtype, device=device) for _ in range(12)]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_legacy_bitwise_equivalence(dtype):
    a = records(dtype)
    w = torch.rand(2, 4, 7, dtype=dtype).softmax(-1)
    p = None
    for record in a:
        layer = record[:, :, 4:, 4:].mean(1)
        p = layer if p is None else p + layer
    old = torch.bmm(w, p.transpose(-1, -2))
    old = old / old.sum(-1, keepdim=True).clamp_min(1e-8)
    actual = propagate_c2p_weights(w, aggregate_p2p(a, 4))
    torch.testing.assert_close(actual, old, rtol=0, atol=0)


def test_beta_zero_uniform_and_explicit_fallback():
    w = torch.rand(2, 3, 7).softmax(-1)
    p = torch.ones(2, 7, 7)
    torch.testing.assert_close(propagate_c2p_weights(w, p, beta=0), w, rtol=0, atol=0)
    torch.testing.assert_close(propagate_c2p_weights(w, p), torch.full_like(w, 1/7))
    for floor in ['min', 'mean']:
        actual, fallback = propagate_c2p_weights(w, p, floor=floor, return_fallback=True)
        torch.testing.assert_close(actual, w, rtol=0, atol=0)
        assert fallback.all()


def test_symmetry_power_entropy_and_sum_mean():
    a = records()
    symmetric = aggregate_p2p(a, 4, p_sym=True)
    torch.testing.assert_close(symmetric, symmetric.transpose(-1, -2), rtol=0, atol=0)
    p1 = aggregate_p2p(a, 4)
    p2 = aggregate_p2p(a, 4, p_alpha=2)
    def entropy(p):
        p = p / p.sum(-1, keepdim=True)
        return -(p * p.log()).sum(-1)
    assert (entropy(p2) <= entropy(p1) + 1e-6).all()
    w = torch.rand(2, 4, 7).softmax(-1)
    torch.testing.assert_close(propagate_c2p_weights(w, p1),
                               propagate_c2p_weights(w, aggregate_p2p(a, 4, p_reduce='mean')))


def test_orientation_gradients_rowmass_and_layers():
    a = [x.requires_grad_() for x in records()]
    w = torch.rand(2, 4, 7, requires_grad=True)
    p = aggregate_p2p(a, 4, p_layers='last3', p_reduce='rownorm_mean', p_alpha=2, p_sym=True)
    actual = propagate_c2p_weights(w.softmax(-1), p, floor='min', beta=.3)
    actual.square().sum().backward()
    assert w.grad.abs().sum() > 0 and torch.isfinite(w.grad).all()
    assert all(x.grad is None for x in a[:9])
    assert all(x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all() for x in a[9:])
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_amp_fp32_accumulation():
    a = [x.half().requires_grad_() for x in records(device='cuda')]
    w = torch.rand(2, 4, 7, device='cuda', requires_grad=True)
    with torch.autocast('cuda'):
        out = propagate_c2p_weights(w.softmax(-1), aggregate_p2p(a, 4, p_alpha=2), floor='min', beta=.3)
    assert out.dtype == torch.float32
    out.square().sum().backward()
    assert torch.isfinite(w.grad).all()
    assert all(torch.isfinite(x.grad).all() for x in a)


def test_preregistered_72_coverage_and_anchors():
    from analysis.affinity_repair import preregistered_configs, DEFAULT, MINIMAL, CONSERVATIVE, config_id
    configs = preregistered_configs()
    assert len(configs) == len({config_id(c) for c in configs}) == 72
    assert configs == preregistered_configs()
    assert all(c in configs for c in [DEFAULT, MINIMAL, CONSERVATIVE])
    assert {c['p_reduce'] for c in configs} == {'sum', 'mean', 'rownorm_mean'}
    assert {c['beta'] for c in configs} == {1., .5, .3, .1}


def test_weight_stats_class_pairs_and_dc():
    from analysis.affinity_repair import weight_stats
    w = torch.ones(2, 3, 784) / 784
    labels = torch.tensor([[1, 0, 0], [1, 1, 1]])
    stats = weight_stats(w, labels)
    torch.testing.assert_close(stats[:, 0], torch.ones(2))
    torch.testing.assert_close(stats[:, 2], torch.ones(2))
    assert torch.isnan(stats[0, 3]) and stats[1, 3] == 1


def test_cam_native_equivalence_and_repair_boundaries():
    from analysis.affinity_repair import repaired_cam
    torch.manual_seed(17)
    model = build_mctformerplus('small', cam=True, input_size=32, num_classes=20).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        _, patches, records, _ = model.forward_features(x)
        _, means = model._stack_attention_records(records)
        logits = model.head(patches.transpose(1, 2).reshape(2, 384, 2, 2))
        seed = (logits.relu().flatten(2) * means[-3:].mean(0)[:, :20, 20:]).sqrt()
        P = means[:, :, 20:, 20:].sum(0)
        cams, mask = repaired_cam(seed, P)
        torch.testing.assert_close(cams.reshape_as(logits), model.get_cam(logits, means), rtol=0, atol=0)
        assert not mask.any()
        torch.testing.assert_close(repaired_cam(seed, P, beta=0)[0], seed, rtol=0, atol=0)
        out, fallback = repaired_cam(seed, torch.ones_like(P), floor='min')
        torch.testing.assert_close(out, seed, rtol=0, atol=0)
        assert fallback.all()


def test_cam_bootstrap_paired_identical():
    import numpy as np
    from analysis.affinity_repair import cam_bootstrap
    x = np.stack([np.eye(21, dtype=np.int64) * i for i in range(1, 9)])
    ci = cam_bootstrap(x, x, reps=100)
    np.testing.assert_allclose(ci[:, 2], 0)


def test_repaired_model_checkpoint_and_unchanged_cam(tmp_path):
    from analysis.affinity_repair import CONSERVATIVE
    from models.mctformer_plus import model_spec_from_instance, validate_mctformerplus_patch_pooling_checkpoint
    kwargs = dict(input_size=32, num_classes=20, patch_pooling='c2p',
                  c2p_pooling_layers='all', c2p_pooling_reduction='product', c2p_pooling_affinity=True)
    trained = build_mctformerplus('small', affinity_repair=CONSERVATIVE, **kwargs).eval()
    baseline = build_mctformerplus('small', **kwargs).eval()
    baseline.load_state_dict(trained.state_dict(), strict=True)
    x = torch.randn(2, 3, 32, 32)
    output = trained(x)
    output[2].square().mean().backward()
    assert trained.head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in trained.parameters() if p.grad is not None)
    torch.testing.assert_close(output[0], baseline(x)[0], rtol=0, atol=0)
    path = tmp_path/'model.pth'
    torch.save(dict(model=trained.state_dict(), model_spec=model_spec_from_instance(trained)), path)
    payload = torch.load(path)
    validate_mctformerplus_patch_pooling_checkpoint(payload, 'c2p', 'all', 'product', True, CONSERVATIVE)
    with pytest.raises(ValueError, match='affinity_repair'):
        validate_mctformerplus_patch_pooling_checkpoint(payload, 'c2p', 'all', 'product', True)
    old = build_mctformerplus('small', cam=True, **kwargs).eval()
    new = build_mctformerplus('small', cam=True, affinity_repair=CONSERVATIVE, **kwargs).eval()
    for model in [old, new]:
        model.load_state_dict(payload['model'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
        torch.testing.assert_close(new.forward_with_label(x)[1], (trained(x)[2] > 0).float(), rtol=0, atol=0)
