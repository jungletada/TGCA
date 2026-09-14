import numpy as np

from analysis.c2p_pooling_attention import METRICS, map_metrics, paired_bootstrap


def metrics(a):
    return dict(zip(METRICS, map_metrics(a)))


def test_identical_and_disjoint_peaks():
    a = np.zeros((2, 784))
    a[:, 5] = 1
    m = metrics(a)
    assert m['pair_top1_same'] == m['all_top1_same'] == 1
    assert m['pair_top10pct_jaccard'] == m['pair_cosine'] == 1
    a[1] = 0
    a[1, 100] = 1
    m = metrics(a)
    assert m['pair_top1_same'] == m['pair_cosine'] == 0


def test_uniform_and_single_label():
    m = metrics(np.ones((2, 784)))
    np.testing.assert_allclose(m['entropy_normalized'], 1)
    np.testing.assert_allclose(m['top1_mass'], 1 / 784)
    assert m['top1_tie_rate'] == 1
    assert np.isnan(m['pair_pearson'])
    assert np.isnan(metrics(np.ones((1, 784)))['pair_top1_same'])


def test_top10_percent_is_ceil_79_and_original_mass_recorded():
    a = np.stack([np.arange(1, 785), np.arange(784, 0, -1)]).astype(float)
    m = metrics(a)
    assert m['pair_top10pct_jaccard'] == 0
    a[1] = np.roll(a[0], 1)
    m = metrics(a)
    np.testing.assert_allclose(m['pair_top10pct_jaccard'], 78 / 80)
    np.testing.assert_allclose(m['pair_top10_jaccard'], 9 / 11)
    np.testing.assert_allclose(m['patch_group_mass'], a.sum(-1).mean())


def test_image_mean_not_pair_pooled():
    # Image one has one agreeing pair; image two has six disjoint pairs.
    a = metrics(np.ones((2, 20)))['pair_top1_same']
    b = metrics(np.eye(4, 20))['pair_top1_same']
    assert (a + b) / 2 == .5  # not 1/7


def test_bootstrap_uses_same_image_draws_and_finite_intersection():
    x = np.arange(60, dtype=float).reshape(10, 3, 2)
    point, ci, count = paired_bootstrap(np.stack([x, x + 2]), repetitions=200)
    np.testing.assert_allclose(point[2], 2)
    np.testing.assert_allclose(ci[:, 2], 2, atol=1e-12)
    assert (count == 10).all()
    y = x.copy()
    y[0, 0, 0] = np.nan
    point, ci, count = paired_bootstrap(np.stack([x, y]), repetitions=200)
    assert count[0, 0] == 9
    np.testing.assert_allclose(point[2], 0)
    np.testing.assert_allclose(ci[:, 2], 0)
