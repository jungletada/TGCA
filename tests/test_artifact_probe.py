import numpy as np
import pytest
import torch
from analysis.artifact_probe import (patch_norm_stats, attractor_stats, symptom_overlap,
                                     lowinfo_scores, frozen_eval, capture_patch_stats)


def test_norm_scale_and_outlier():
    x = torch.ones(2, 100, 4)
    x[:, 5] = 4
    a, b = patch_norm_stats(x), patch_norm_stats(x * 7)
    torch.testing.assert_close(a['ratio'], b['ratio'])
    torch.testing.assert_close(a['frac_hi'], torch.full((2,), .01))
    assert a['ratio'][0, 5] == 4


def test_receiver_column_direction_uniform_and_overlap():
    uniform = attractor_stats(torch.ones(2, 100, 100))
    torch.testing.assert_close(uniform['gini'], torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform['top_share'], torch.full((2,), .01))
    attention = torch.ones(1, 100, 100)
    attention[:, :, 7] = 100
    s = attractor_stats(attention)
    assert s['recv'].argmax() == 7 and s['top_share'].item() > .5
    torch.testing.assert_close(symptom_overlap(s['recv'], s['recv']), torch.ones(1))
    torch.testing.assert_close(symptom_overlap(s['recv'], s['recv'].roll(1, -1)), torch.zeros(1))


def test_lowinfo_flat_missing_and_gradient():
    ratio = torch.ones(1, 16)
    top, high, empty = lowinfo_scores(torch.zeros(1, 3, 16, 16), ratio, grid=4)
    assert top.item() == 0 and torch.isnan(high).all() and empty.item() == 1
    ratio[:, 0] = 4
    image = torch.arange(16.).reshape(1, 1, 1, 16).expand(1, 3, 16, 16)
    top, high, empty = lowinfo_scores(image, ratio, grid=4)
    torch.testing.assert_close(top, torch.ones(1))
    torch.testing.assert_close(high, torch.ones(1))
    assert empty.item() == 0


def test_eval_hooks_restore_even_on_error():
    from models.mctformer_plus import build_mctformerplus
    model = build_mctformerplus('small', input_size=32, num_classes=20).train()
    model.blocks[0].eval()
    modes = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError, match='deliberate'):
        with frozen_eval(model), capture_patch_stats(model) as stats:
            assert not model.training and not torch.is_grad_enabled()
            _, patches, records, _ = model.forward_features(torch.randn(1, 3, 32, 32))
            assert len(stats) == 12 and len(records) == 12
            torch.testing.assert_close(stats[12]['ratio'], patch_norm_stats(patches)['ratio'])
            raise RuntimeError('deliberate')
    assert modes == [m.training for m in model.modules()]
    assert all(not b._forward_hooks for b in model.blocks)


def test_bootstrap_cluster_not_patches():
    from analysis.c2p_pooling_attention import paired_bootstrap
    x = np.arange(12.).reshape(6, 2)
    point, ci, n = paired_bootstrap(np.stack([x, x + .2]), 100, 20260916)
    np.testing.assert_allclose(point[2], .2)
    np.testing.assert_allclose(ci[:, 2], .2)
    assert (n == 6).all()
