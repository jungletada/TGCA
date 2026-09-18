"""Streaming configuration scans and small, additive confusion statistics."""
import numpy as np


def image_threshold_confusions(scores, classes, target, thresholds,
                               num_classes, ignore_index=255):
    """Histogram all thresholds with strict score>threshold, class0 fallback.

    Returns [thresholds,C,C] int32 per-image counts as in the source. Targets
    indexed ignore_index are omitted; winner labels must be in 1..C-1.
    """
    scores, classes, target = map(np.asarray, (scores, classes, target))
    thresholds = np.asarray(thresholds)
    if (thresholds.ndim != 1 or not thresholds.size
            or not np.isfinite(thresholds).all() or (np.diff(thresholds) < 0).any()):
        raise ValueError("thresholds must be nonempty finite and sorted")
    if num_classes < 2:
        raise ValueError("At least two classes required")
    if not all(np.issubdtype(x.dtype, np.integer) for x in (classes, target)):
        raise ValueError("Labels must be integers")
    if np.isnan(scores).any():
        raise ValueError("Scores cannot contain NaN")
    if scores.shape != classes.shape or scores.shape != target.shape:
        raise ValueError(
            f'Score/target shape mismatch: {scores.shape}, {target.shape}'
        )
    valid = target != ignore_index
    target = target[valid].astype(np.int64, copy=False)
    scores = scores[valid]
    classes = classes[valid]
    if np.any((target < 0) | (target >= num_classes)):
        raise ValueError('Target contains an invalid non-void label')
    if np.any((classes < 1) | (classes >= num_classes)):
        raise ValueError('Invalid foreground prediction label')
    # k is the count of thresholds strictly below the score.  Therefore a
    # pixel is foreground at grid index i exactly when k > i, matching score>t.
    passed_count = np.searchsorted(thresholds, scores, side='left')
    pair = target * (num_classes - 1) + (classes - 1)
    width = len(thresholds) + 1
    histogram = np.bincount(
        pair * width + passed_count,
        minlength=num_classes * (num_classes - 1) * width,
    ).reshape(num_classes, num_classes - 1, width)
    foreground = np.cumsum(histogram[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    confusion = np.zeros(
        (len(thresholds), num_classes, num_classes), dtype=np.int32
    )
    confusion[:, :, 1:] = foreground.transpose(2, 0, 1).astype(np.int32)
    target_counts = np.bincount(target, minlength=num_classes)
    confusion[:, :, 0] = (
        target_counts[None] - confusion[:, :, 1:].sum(axis=2)
    ).astype(np.int32)
    if np.any(confusion < 0):
        raise RuntimeError('Negative confusion count produced')
    if not np.all(confusion.sum(axis=(1, 2)) == valid.sum()):
        raise RuntimeError('Threshold confusion does not conserve valid pixels')
    return confusion


def confusion_metrics(confusion):
    """Dataset-global metrics; class0 is background, absent-class IoU stays NaN.

    Preserves the source arithmetic; semantic and binary foreground metrics
    are distinct. Never average per-image IoU to emulate this estimator.
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    target = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    union = target + predicted - true_positive
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / union
        class_precision = true_positive / predicted
        class_recall = true_positive / target
    semantic_tp = true_positive[..., 1:].sum(axis=-1)
    pred_fg = predicted[..., 1:].sum(axis=-1)
    target_fg = target[..., 1:].sum(axis=-1)
    foreground_overlap = confusion[..., 1:, 1:].sum(axis=(-2, -1))
    false_positive_background = confusion[..., 0, 1:].sum(axis=-1)
    target_background = target[..., 0]
    valid_pixels = confusion.sum(axis=(-2, -1))
    correct_pixels = true_positive.sum(axis=-1)

    def divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full(np.broadcast_shapes(
                np.shape(numerator), np.shape(denominator)), np.nan),
            where=np.asarray(denominator) != 0,
        )

    return {
        'mean_iou': np.nanmean(iou, axis=-1),
        'foreground_mean_iou': np.nanmean(iou[..., 1:], axis=-1),
        'semantic_foreground_precision': divide(semantic_tp, pred_fg),
        'semantic_foreground_recall': divide(semantic_tp, target_fg),
        'binary_foreground_precision': divide(foreground_overlap, pred_fg),
        'binary_foreground_recall': divide(foreground_overlap, target_fg),
        'background_false_positive_rate': divide(
            false_positive_background, target_background
        ),
        'pixel_accuracy': divide(correct_pixels, valid_pixels),
        'class_iou': iou,
        'class_precision': class_precision,
        'class_recall': class_recall,
    }


class ConfusionAccumulator:
    """Int64 [threshold,C,C] totals; target rows, prediction columns.

    Retains no pixels. For paired bootstrap, retain each returned per-image
    increment externally with its image ID; totals alone cannot yield a paired
    image CI. The framework is new: NOT VALIDATED AGAINST TGCA RUNS.
    """
    def __init__(self, n_classes, n_thresholds=1, ignore_index=255):
        if any(type(v) is not int or v < 1 for v in (n_classes, n_thresholds)):
            raise ValueError('Class and threshold counts must be positive integers')
        self.n_classes = n_classes
        self.n_thresholds = n_thresholds
        self.ignore_index = ignore_index
        self.confusion = np.zeros((n_thresholds, n_classes, n_classes), dtype=np.int64)
        self.observations = 0

    def observe(self, prediction, target):
        """Add one image with [threshold,*spatial] integer prediction labels."""
        prediction, target = np.asarray(prediction), np.asarray(target)
        if self.n_thresholds == 1 and prediction.shape == target.shape:
            prediction = prediction[None]
        if prediction.shape != (self.n_thresholds,) + target.shape:
            raise ValueError('Prediction shape must be [threshold,*target.shape]')
        if not all(np.issubdtype(x.dtype, np.integer) for x in (prediction, target)):
            raise ValueError('Labels must be integers')
        valid = target != self.ignore_index
        y = target[valid].astype(np.int64, copy=False)
        p = prediction[:, valid].astype(np.int64, copy=False)
        if any(((x < 0) | (x >= self.n_classes)).any() for x in (y, p)):
            raise ValueError('Non-ignored label outside class range')
        counts = np.stack([
            np.bincount(y * self.n_classes + row,
                        minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
            for row in p
        ])
        self.observe_confusions(counts)
        return counts

    def observe_confusions(self, confusion):
        """Add an already computed per-image [threshold,C,C] count tensor."""
        confusion = np.asarray(confusion)
        if (confusion.shape != self.confusion.shape
                or not np.issubdtype(confusion.dtype, np.integer)
                or (confusion < 0).any()):
            raise ValueError('Invalid per-image confusion counts')
        self.confusion += confusion
        self.observations += 1


class ConfigSweep:
    """One shared forward per sample; configurations operate on the same features.

    73 configs x 50 thresholds x 21**2 x 8 bytes is about 12.9 MB of totals,
    without caching dense intermediate tensors. Configuration readouts still
    incur computation: they are not promised to be free. Callbacks must not
    mutate shared features. Model eval/RNG isolation is the caller's contract.
    NOT VALIDATED AGAINST TGCA RUNS: new generic callback interface.
    """
    def __init__(self, configs, n_thresholds, n_classes, *, ignore_index=255):
        self.configs = tuple(configs)
        if not self.configs:
            raise ValueError('At least one configuration required')
        self.accumulators = [
            ConfusionAccumulator(n_classes, n_thresholds, ignore_index)
            for _ in self.configs
        ]

    def _accumulator(self, config_id):
        if type(config_id) is not int or not 0 <= config_id < len(self.configs):
            raise ValueError('config_id must be a valid nonnegative positional ID')
        return self.accumulators[config_id]

    def observe(self, config_id, prediction, target):
        """Return the per-image increment, optionally retained for image CIs."""
        return self._accumulator(config_id).observe(prediction, target)

    def observe_confusions(self, config_id, confusion):
        self._accumulator(config_id).observe_confusions(confusion)

    def run(self, samples, forward, readout):
        """samples yields (inputs,target); readout(features,config) yields labels."""
        for inputs, target in samples:
            features = forward(inputs)
            for config_id, config in enumerate(self.configs):
                prediction = readout(features, config)
                self.observe(config_id, prediction, target)
        return self.results()

    def results(self, *, dataframe=False):
        """Plain records by default; optional pandas export requires [dataframe]."""
        rows = []
        for config_id, accumulator in enumerate(self.accumulators):
            metrics = confusion_metrics(accumulator.confusion)
            for threshold_id in range(accumulator.n_thresholds):
                row = dict(config_id=config_id, config=self.configs[config_id],
                           threshold_index=threshold_id,
                           images=accumulator.observations)
                for name, values in metrics.items():
                    value = np.asarray(values[threshold_id])
                    row[name] = value.item() if value.ndim == 0 else value.tolist()
                rows.append(row)
        if dataframe:
            try:
                import pandas as pd
            except ImportError as error:
                raise ImportError('Install vitprobe[dataframe] for pandas export') from error
            return pd.DataFrame(rows)
        return rows
