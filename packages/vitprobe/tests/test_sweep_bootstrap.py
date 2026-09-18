import numpy as np
import pytest

from vitprobe.bootstrap import paired_image_bootstrap, paired_confusion_bootstrap
from vitprobe.sweep import ConfigSweep, ConfusionAccumulator, image_threshold_confusions, confusion_metrics


def test_threshold_histogram_against_direct_labels():
    rng = np.random.default_rng(3)
    target = rng.integers(0, 4, (13, 17))
    target[0, :4] = 255
    scores = rng.choice([0., .2, .4, .6], target.shape)
    labels = rng.integers(1, 4, target.shape)
    thresholds = np.asarray([0., .2, .4, .6])
    hist = image_threshold_confusions(scores, labels, target, thresholds, 4)
    direct = ConfusionAccumulator(4, len(thresholds))
    delta = direct.observe(np.where(scores[None] > thresholds[:, None, None], labels, 0), target)
    np.testing.assert_array_equal(hist, delta)
    np.testing.assert_array_equal(hist, direct.confusion)
    assert direct.confusion.dtype == np.int64
    assert (hist.sum((1, 2)) == (target != 255).sum()).all()


def test_shared_forward_and_independent_config_totals():
    calls = []
    sweep = ConfigSweep([0, 1], 1, 3)
    samples = [(i, np.asarray([[0, 1, 2]])) for i in range(5)]
    def forward(x):
        calls.append(x)
        return np.asarray([[0, 1, 2]])
    def readout(features, config):
        return (features + config) % 3
    rows = sweep.run(samples, forward, readout)
    assert calls == list(range(5))
    assert isinstance(rows, list) and len(rows) == 2
    assert rows[0]['mean_iou'] == 1 and rows[1]['mean_iou'] == 0
    assert rows[0]['images'] == rows[1]['images'] == 5
    assert sweep.accumulators[0].confusion.sum() == 15
    assert sweep.accumulators[1].confusion.sum() == 15


def test_optional_dataframe_matches_records():
    pd = pytest.importorskip('pandas', reason='Optional dataframe extra not installed')
    sweep = ConfigSweep([{'setting': 1}], 1, 3)
    target = np.asarray([0, 1, 2])
    sweep.observe(0, target, target)
    frame = sweep.results(dataframe=True)
    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict('records') == sweep.results()


def test_invalid_input_rejected_without_mutating_totals():
    accumulator = ConfusionAccumulator(3)
    with pytest.raises(ValueError):
        accumulator.observe(np.asarray([0, 3]), np.asarray([0, 1]))
    with pytest.raises(ValueError):
        accumulator.observe_confusions(np.zeros((1, 3, 3), dtype=float))
    assert accumulator.observations == 0 and not accumulator.confusion.any()
    sweep = ConfigSweep([{}], 1, 3)
    with pytest.raises(ValueError):
        sweep.observe(-1, np.asarray([0]), np.asarray([0]))


def test_image_bootstrap_uses_paired_missing_intersection():
    values = np.array([[[1., 4.], [9., 2.], [3., np.nan]],
                       [[2., 5.], [10., 3.], [4., 1000.]]])
    point, ci, counts = paired_image_bootstrap(values, repetitions=101)
    np.testing.assert_array_equal(counts, [3, 2])
    np.testing.assert_allclose(point[2], [1, 1])
    np.testing.assert_allclose(ci[:, 2], np.ones((2, 2)))
    # The last model1-only observation is excluded, not treated as independent.
    assert point[1, 1] == 4


def test_missing_metric_is_nan_not_zero():
    values = np.ones((2, 5, 2))
    values[0, :, 0] = np.nan
    with pytest.warns(RuntimeWarning):
        point, ci, counts = paired_image_bootstrap(values, repetitions=101)
    assert counts[0] == 0 and np.isnan(point[:, 0]).all()
    assert np.isnan(ci[..., 0]).all()
    np.testing.assert_array_equal(ci[:, 2, 1], 0)


def test_global_confusion_estimator_not_mean_image_iou():
    reference = np.asarray([[[900, 0], [0, 100]], [[0, 9], [1, 0]]])
    variant = np.asarray([[[800, 100], [0, 100]], [[9, 0], [0, 1]]])
    reps, seed = 101, 4
    # Same image draws; independently compute the reference percentile.
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, reps, 100):
        weights = rng.multinomial(2, [.5, .5], size=min(100, reps-start))
        for weight in weights:
            a = confusion_metrics((reference * weight[:, None, None]).sum(0))['mean_iou']
            b = confusion_metrics((variant * weight[:, None, None]).sum(0))['mean_iou']
            samples.append([a, b, b-a])
    actual = paired_confusion_bootstrap(variant, reference, repetitions=reps, seed=seed)
    np.testing.assert_allclose(actual, np.percentile(samples, [2.5, 97.5], axis=0), atol=1e-7, rtol=0)
    assert not np.isclose(confusion_metrics(reference.sum(0))['mean_iou'],
                          confusion_metrics(reference)['mean_iou'].mean())


def test_bootstrap_identity_and_invalid_shapes():
    x = np.random.default_rng(2).integers(1, 10, (7, 4, 4))
    assert (paired_confusion_bootstrap(x, x, repetitions=101)[:, 2] == 0).all()
    with pytest.raises(ValueError):
        paired_confusion_bootstrap(x, x[:1])
    with pytest.raises(ValueError):
        paired_image_bootstrap(np.ones((2, 0, 3)))
