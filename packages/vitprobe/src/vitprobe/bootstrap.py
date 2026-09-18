"""Image-clustered paired estimators. Never pass patch or pair rows as images.

The two estimands are separate: mean image descriptors versus pooled-confusion
mIoU. Both resample whole images jointly, with the historical multinomial RNG
and batch size preserved. Default 5000 draws, seed 20260916. No seed uncertainty.
"""
import numpy as np

from .sweep import confusion_metrics


def paired_image_bootstrap(x, repetitions=5000, seed=20260916):
    """x [2, images, ...]; resample IMAGE IDs jointly, equal image weights.

    Finite intersection is used per metric for both models. The returned
    samples contain model 0, model 1, and paired (1 - 0), respectively.
    """
    x = np.asarray(x)
    if x.ndim < 2 or x.shape[0] != 2 or x.shape[1] == 0:
        raise ValueError('Expected [2,images,...] with nonempty paired image axis')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    shape = x.shape[2:]
    x = x.reshape(2, x.shape[1], -1)
    valid = np.isfinite(x).all(0)
    counts = valid.sum(0)
    clean = np.where(valid[None], x, 0)
    mean = clean.sum(1) / np.maximum(counts, 1)
    mean[:, counts == 0] = np.nan
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 100):
        weights = rng.multinomial(x.shape[1], np.full(x.shape[1], 1 / x.shape[1]),
                                  size=min(100, repetitions - start)).astype(float)
        denom = weights @ valid.astype(float)
        values = np.stack([weights @ clean[m] / np.where(denom > 0, denom, np.nan)
                           for m in range(2)])
        samples.append(np.concatenate([values, (values[1] - values[0])[None]], axis=0))
    samples = np.concatenate(samples, axis=1)
    ci = np.nanpercentile(samples, [2.5, 97.5], axis=1)
    point = np.concatenate([mean, (mean[1] - mean[0])[None]])
    return point.reshape((3,) + shape), ci.reshape((2, 3) + shape), counts.reshape(shape)




def paired_confusion_bootstrap(confusions, reference, repetitions=5000, seed=20260916):
    """Paired IMAGE resampling of fixed-threshold confusion, not image IoU means."""
    confusions, reference = np.asarray(confusions), np.asarray(reference)
    if (reference.ndim != 3 or confusions.shape != reference.shape
            or reference.shape[0] == 0 or reference.shape[1] < 2
            or reference.shape[1] != reference.shape[2]):
        raise ValueError('Expected matching nonempty [images,C,C] arrays, C>=2')
    if not all(np.isfinite(x).all() and (x >= 0).all() for x in (confusions, reference)):
        raise ValueError('Confusion counts must be finite and nonnegative')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    classes = reference.shape[-1]
    n = len(reference)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 100):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(100, repetitions-start)).astype(float)
        a = (weights @ reference.reshape(n, -1)).reshape(-1, classes, classes)
        b = (weights @ confusions.reshape(n, -1)).reshape(-1, classes, classes)
        ma, mb = confusion_metrics(a)['mean_iou'], confusion_metrics(b)['mean_iou']
        samples.append(np.stack([ma, mb, mb-ma], 1))
    return np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0)
