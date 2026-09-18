import random

import numpy as np
import pytest
import torch

from vitprobe import artifacts, relations, spectral, stats
from vitprobe.state import isolated_eval


def test_uniform_onehot_and_missing():
    n = 8
    p = torch.full((2, 3, n), 1 / n)
    assert torch.allclose(stats.effective_support(p), torch.ones(2, 3))
    assert torch.allclose(stats.gini(p), torch.zeros(2, 3))
    p.zero_()
    p[..., 0] = 1
    assert torch.allclose(stats.effective_support(p), torch.full((2, 3), 1/n))
    assert torch.allclose(stats.gini(p), torch.full((2, 3), (n-1)/n))
    assert torch.isnan(stats.conditional_attention(torch.zeros_like(p))).all()


def test_white_noise_constant_and_known_correlation():
    f = torch.randn(12, 28, 28, 48)
    power, _, _, ac = spectral.spectrum_and_autocorrelation(f)
    assert .42 < spectral.high_freq_ratio(power).mean() < .58
    power, _, _, ac = spectral.spectrum_and_autocorrelation(torch.ones_like(f))
    assert not power.any()
    assert torch.isnan(spectral.high_freq_ratio(power)).all()
    assert not spectral.correlation_length(ac)['xi_valid'].any()
    result = spectral.correlation_length((-torch.arange(16)/2.).exp()[None])
    assert result['xi_valid'].all()
    assert torch.allclose(result['xi'], torch.tensor([2.]))


def test_low_precision_promoted():
    p = torch.rand(2, 3, 16).softmax(-1).half()
    for function in [stats.effective_support, stats.gini, stats.normalized_entropy]:
        assert function(p).dtype == torch.float32
        assert torch.equal(function(p), function(p.float()))


def test_missing_pairs_and_tiny_gamma():
    labels = torch.tensor([[1, 0, 0], [1, 1, 0]])
    a = torch.ones(2, 2, 3, 12) * 1e-30
    gamma = relations.class_agnosticism(a, labels)
    assert torch.isnan(gamma[0]).all()
    assert torch.allclose(gamma[1], torch.ones(2))
    result = relations.class_token_affinity(torch.randn(2, 3, 8), labels)
    assert torch.isnan(result['rho_cc'][0])


def test_isolated_eval_restores_all_state():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout())
    model.train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError):
        with isolated_eval(model, 'cpu'):
            random.random(); np.random.rand(); torch.rand(3)
            raise RuntimeError('intentional')
    assert [m.training for m in model.modules()] == modes
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), cpu)
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_patch_norm_zero_and_empty_high_set():
    p = torch.zeros(2, 784, 8)
    ratio = artifacts.patch_norm_stats(p)['ratio']
    assert not ratio.any()
    _, high, empty = artifacts.lowinfo_scores(torch.rand(2, 3, 56, 56), ratio)
    assert torch.isnan(high).all() and empty.all()
