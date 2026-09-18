"""All currently extracted diagnostics on random non-semantic token layouts."""
import numpy as np
import torch

from vitprobe import TokenLayout, artifacts, relations, spectral, stats
from vitprobe.bootstrap import paired_image_bootstrap, paired_confusion_bootstrap
from vitprobe.graph import aggregate_p2p, propagate_weights
from vitprobe.state import isolated_eval
from vitprobe.sweep import ConfigSweep


def test_nonsemantic_layout_end_to_end():
    layout = TokenLayout.dinov3()
    tokens = torch.randn(2, layout.n_tokens, 12)
    attention = torch.rand(2, 2, layout.n_tokens, layout.n_tokens).softmax(-1)
    # Register queries are arbitrary query vectors, not semantic class labels.
    query_tokens = tokens[:, layout.register_slice]
    maps = layout.query_patch_attention(attention, 'register').mean(1)
    labels = torch.ones(2, 4, dtype=torch.bool)
    p = stats.conditional_attention(maps)
    for function in [stats.effective_support, stats.gini, stats.normalized_entropy]:
        assert torch.isfinite(function(p)).all()
    assert torch.isfinite(stats.weight_stats(p, labels)).all()
    assert torch.isfinite(relations.class_token_affinity(query_tokens, labels)['rho_cc']).all()
    assert torch.isfinite(relations.class_agnosticism(maps[None], labels)).all()
    assert relations.head_statistics(maps[None], labels).shape == (2, 1, 3)
    norm = artifacts.patch_norm_stats(layout.patch_tokens(tokens))
    attractor = artifacts.attractor_stats(layout.patch_attention(attention).mean(1))
    assert torch.isfinite(relations.symptom_overlap(norm['ratio'], attractor['recv'])).all()
    low = artifacts.lowinfo_scores(torch.rand(2, 3, 112, 112), norm['ratio'], layout=layout)
    assert torch.isfinite(low[0]).all()
    grid = layout.patch_grid(tokens)
    power, _, _, ac = spectral.spectrum_and_autocorrelation(grid)
    assert torch.isfinite(spectral.high_freq_ratio(power)).all()
    spectral.radial_power_spectrum(grid)
    assert spectral.correlation_length(ac)['xi_valid'].dtype == torch.bool
    propagated = propagate_weights(p, aggregate_p2p([attention], layout))
    assert torch.allclose(propagated.sum(-1), torch.ones(2, 4))
    model = torch.nn.Linear(12, 3).train()
    with isolated_eval(model, 'cpu'):
        assert not model.training
    assert model.training
    paired = np.stack([stats.gini(p).numpy(), stats.gini(propagated).numpy()])
    assert paired_image_bootstrap(paired, repetitions=101)[0].shape == (3, 4)
    sweep = ConfigSweep([None], 1, 3)
    target = np.asarray([0, 1, 2])
    counts = sweep.observe(0, target, target)
    ci = paired_confusion_bootstrap(counts, counts, repetitions=101)
    assert np.isfinite(ci).all()
    assert sweep.results()[0]['mean_iou'] == 1


def test_rectangular_rgb_adapter():
    layout = TokenLayout.dinov3(grid=(28, 32))
    ratio = torch.rand(2, layout.n_patch) + 1
    values = artifacts.lowinfo_scores(torch.rand(2, 3, 56, 64), ratio, layout=layout)
    assert torch.isfinite(values[0]).all()
