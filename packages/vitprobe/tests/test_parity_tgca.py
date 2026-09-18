"""Optional source parity. Never loads checkpoints, datasets or historic metrics.

TGCA_PATH points to the source checkout. FP32 inputs; atol=1e-7, rtol=0,
equal_nan=True. A failure is a stop gate, not grounds to relax tolerances.
"""
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from vitprobe import artifacts, relations, spectral, stats


@pytest.fixture(scope='module')
def original():
    root = os.environ.get('TGCA_PATH')
    if not root:
        pytest.skip('TGCA_PATH unset: optional original-source parity unavailable')
    root = Path(root).resolve()
    if not (root / 'analysis/diagnostics/metrics.py').is_file():
        pytest.fail('TGCA_PATH is not a valid source checkout')
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(path) for name, path in {
        'stats': 'analysis.diagnostics.metrics',
        'weights': 'analysis.weight_stats',
        'artifact': 'analysis.artifact_probe',
        'affinity': 'analysis.affinity_repair',
        'spectral': 'analysis.diagnostics.spectral',
        'head': 'analysis.head_heatmap',
        'state': 'analysis.diagnostics.state',
        'graph': 'models.mctformer_plus',
        'bootstrap': 'analysis.c2p_pooling_attention',
        'confusion': 'tools.evaluate_cam_threshold_grid',
    }.items()}
    for module in modules.values():
        assert root in Path(module.__file__).resolve().parents
    yield modules
    sys.path.remove(str(root))


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-7, rtol=0, equal_nan=True)
    else:
        assert a == b


@pytest.mark.parametrize('name', ['conditional_attention', 'effective_support', 'gini'])
def test_stats_parity(original, name):
    x = torch.rand(3, 4, 31).softmax(-1)
    x[0, 0] = 0
    same(getattr(stats, name)(x), getattr(original['stats'], name)(x))


def test_entropy_and_weight_stats(original):
    w = torch.rand(3, 4, 31).softmax(-1)
    labels = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    same(stats.normalized_entropy(w), original['weights'].normalized_entropy(w))
    for fallback in [None, torch.rand(3, 4) > .5]:
        same(stats.weight_stats(w, labels, fallback),
             original['affinity'].weight_stats(w, labels, fallback))


def test_class_token_affinity(original):
    t = torch.randn(3, 4, 16)
    labels = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    same(relations.class_token_affinity(t, labels), original['stats'].class_token_affinity(t, labels))


@pytest.mark.parametrize('scale', [1., 1e-30])
def test_head_statistics_and_gamma(original, scale):
    a = torch.rand(5, 3, 4, 31) * scale
    labels = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    expected = original['head'].head_statistics(a, labels)
    same(relations.head_statistics(a, labels), expected)
    same(relations.class_agnosticism(a, labels), expected[..., 2])


def test_artifacts_and_overlap(original):
    p = torch.randn(2, 784, 16)
    p[:, :8] *= 10
    a = torch.rand(2, 784, 784)
    for name, x in [('patch_norm_stats', p), ('attractor_stats', a)]:
        same(getattr(artifacts, name)(x), getattr(original['artifact'], name)(x))
    ratio = artifacts.patch_norm_stats(p)['ratio']
    recv = artifacts.attractor_stats(a)['recv']
    same(relations.symptom_overlap(ratio, recv), original['artifact'].symptom_overlap(ratio, recv))
    rgb = torch.rand(2, 3, 112, 112)
    same(artifacts.lowinfo_scores(rgb, ratio), original['artifact'].lowinfo_scores(rgb, ratio))


@pytest.mark.parametrize('constant', [False, True])
def test_spectral(original, constant):
    f = torch.ones(2, 28, 28, 16) if constant else torch.randn(2, 28, 28, 16)
    result = spectral.spectrum_and_autocorrelation(f)
    same(result, original['spectral'].spectrum_and_autocorrelation(f))
    same(spectral.radial_power_spectrum(f), original['spectral'].radial_power_spectrum(f))
    same(spectral.high_freq_ratio(result[0]), original['spectral'].high_freq_ratio(result[0]))
    same(spectral.correlation_length(result[3]), original['spectral'].correlation_length(result[3]))


def test_state_parity(original):
    from vitprobe.state import isolated_eval
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout())
    model.train()
    model[0].eval()
    expected = [m.training for m in model.modules()]
    for context in [isolated_eval, original['state'].isolated_eval]:
        rng = torch.get_rng_state()
        with pytest.raises(RuntimeError, match='synthetic'):
            with context(model, 'cpu'):
                assert not any(m.training for m in model.modules())
                torch.rand(3)
                raise RuntimeError('synthetic')
        assert expected == [m.training for m in model.modules()]
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('class_first', [True, False])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('layers', ['all', 'last3', 'last6'])
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'rownorm_mean'])
@pytest.mark.parametrize('alpha,sym', [(1., False), (2., True)])
def test_graph_aggregation_parity(original, class_first, dtype, layers, reduce, alpha, sym):
    from vitprobe import TokenLayout
    from vitprobe.graph import aggregate_p2p
    layout = TokenLayout(n_cls=1, n_class=2, n_register=2, grid_hw=(2, 3),
                         class_first=class_first)
    records = [torch.rand(2, 3, layout.n_tokens, layout.n_tokens, dtype=dtype)
               for _ in range(8)]
    kwargs = dict(p_layers=layers, p_reduce=reduce, p_alpha=alpha, p_sym=sym)
    same(aggregate_p2p(records, layout, **kwargs),
         original['graph'].aggregate_p2p(records, layout.n_class,
                                         patch_slice=layout.patch_slice, **kwargs))


@pytest.mark.parametrize('floor', ['none', 'min', 'mean'])
@pytest.mark.parametrize('beta', [0., .3, 1.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16])
def test_graph_propagation_parity(original, floor, beta, dtype):
    from vitprobe.graph import propagate_weights
    w = torch.rand(2, 3, 12).softmax(-1).to(dtype)
    p = torch.rand(2, 12, 12).to(dtype)
    p[0] = 0
    p[1, 0] = p[1, 1]
    kwargs = dict(floor=floor, beta=beta, return_fallback=True)
    same(propagate_weights(w, p, **kwargs),
         original['graph'].propagate_c2p_weights(w, p, **kwargs))


@pytest.mark.parametrize('seed', [20260916, 20260914])
def test_image_bootstrap_parity(original, seed):
    from vitprobe.bootstrap import paired_image_bootstrap
    rng = np.random.default_rng(5)
    values = rng.normal(size=(2, 17, 2, 3))
    values[0, 0, 0, 0] = np.nan
    values[1, 2, 1, 2] = np.nan
    result = paired_image_bootstrap(values, seed=seed)
    expected = original['bootstrap'].paired_bootstrap(values, seed=seed)
    for actual, ref in zip(result, expected):
        np.testing.assert_allclose(actual, ref, atol=1e-7, rtol=0, equal_nan=True)


def test_confusion_bootstrap_parity(original):
    from vitprobe.bootstrap import paired_confusion_bootstrap
    rng = np.random.default_rng(20260916)
    reference = rng.integers(0, 50, size=(13, 21, 21))
    variant = rng.integers(0, 50, size=reference.shape)
    np.testing.assert_allclose(paired_confusion_bootstrap(variant, reference),
                               original['affinity'].cam_bootstrap(variant, reference),
                               atol=1e-7, rtol=0, equal_nan=True)


def test_threshold_confusions_and_metrics_parity(original):
    from vitprobe.sweep import image_threshold_confusions, confusion_metrics
    rng = np.random.default_rng(20260916)
    thresholds = np.asarray([0., .1, .2, .45, .59])
    scores = rng.choice(thresholds, (17, 23))  # explicitly exercise ties
    classes = rng.integers(1, 21, scores.shape)
    target = rng.integers(0, 21, scores.shape)
    target[0] = 255
    target[1] = 0
    scores[2] = -np.inf
    actual = image_threshold_confusions(scores, classes, target, thresholds, 21)
    expected = original['confusion'].image_threshold_confusions(scores, classes, target, thresholds)
    np.testing.assert_array_equal(actual, expected)
    actual_metrics = confusion_metrics(actual)
    for key, value in original['confusion'].confusion_metrics(expected).items():
        np.testing.assert_allclose(actual_metrics[key], value, atol=1e-7, rtol=0, equal_nan=True)


def test_layout_legacy_and_lowinfo_parity(original):
    from vitprobe import TokenLayout
    layout = TokenLayout.mctformer_plus()
    tokens = torch.randn(2, 804, 8)
    same(layout.patch_tokens(tokens), tokens[:, 20:])
    ratio = artifacts.patch_norm_stats(layout.patch_tokens(tokens))['ratio']
    rgb = torch.rand(2, 3, 112, 112)
    same(artifacts.lowinfo_scores(rgb, ratio, layout=layout),
         original['artifact'].lowinfo_scores(rgb, ratio))
