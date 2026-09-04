"""Frozen-model classification and raw-CAM evaluation for Experiment 2.

The helpers in this module operate only on already collected arrays.  They do
not load or mutate models, checkpoints, Experiment 1 artifacts, or VOC files.

Two details are deliberate:

* raw CAM evaluation reproduces the repository's single-scale 448 pipeline:
  bilinear 28x28 -> 448x448 interpolation with ``align_corners=False``,
  per-active-class min-max normalization, and a fixed 0.45 background channel;
* uncertainty resamples whole images.  MCTformer and MCTformer+ are evaluated
  with the same bootstrap multiplicities, so every reported delta is paired.

Classification AP bootstraps use integer image multiplicities over score
orders computed once per class/model.  Tied scores are accumulated as one
threshold group, matching the non-interpolated average-precision definition
without repeatedly calling scikit-learn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


__all__ = (
    "DEFAULT_BOOTSTRAP_REPEATS",
    "DEFAULT_BOOTSTRAP_SEED",
    "LABEL_STRATA",
    "RAW_CAM_BACKGROUND_THRESHOLD",
    "average_precision_from_scores",
    "classification_average_precision",
    "iou_from_confusion",
    "paired_cam_iou_bootstrap",
    "paired_classification_bootstrap",
    "raw_final_cam_confusion",
    "raw_final_cam_prediction",
    "upsample_and_normalize_active_cams",
    "voc_confusion_matrix",
)


VOC_FOREGROUND_CLASSES = 20
VOC_SEGMENTATION_CLASSES = 21
VOC_VOID_ID = 255
PATCH_GRID_SIZE = (28, 28)
TRANSFORMED_CROP_SIZE = (448, 448)
RAW_CAM_BACKGROUND_THRESHOLD = 0.45
NORMALIZE_CAM_EPSILON = 1e-8
DEFAULT_BOOTSTRAP_REPEATS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260901

LABEL_STRATA = (
    "all",
    "single_label",
    "exactly_2_labels",
    "3plus_labels",
)
_STRATUM_SEED_OFFSET = {
    "all": 0,
    "single_label": 1,
    "exactly_2_labels": 2,
    "3plus_labels": 3,
}
_MCTFORMER = "mctformer"
_MCTFORMER_PLUS = "mctformer_plus"
_PAIRED_DELTA = "mctformer_plus_minus_mctformer"


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise TypeError(f"{name} must be a numeric array")
    return array


def _cam_grid_tensor(final_cam: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, bool]:
    input_is_tensor = isinstance(final_cam, torch.Tensor)
    if input_is_tensor:
        cam = final_cam
    else:
        array = np.asarray(final_cam)
        if not array.flags.c_contiguous or not array.flags.writeable:
            array = np.array(array, copy=True, order="C")
        cam = torch.from_numpy(array)

    if cam.ndim == 2 and cam.shape[1] == PATCH_GRID_SIZE[0] * PATCH_GRID_SIZE[1]:
        cam = cam.reshape(cam.shape[0], *PATCH_GRID_SIZE)
    if cam.ndim != 3 or tuple(cam.shape[-2:]) != PATCH_GRID_SIZE:
        raise ValueError(
            f"final_cam must have shape [K,28,28] or [K,784], got {tuple(cam.shape)}"
        )
    if cam.shape[0] < 1:
        raise ValueError("final_cam must contain at least one active class")
    if not cam.is_floating_point():
        raise TypeError("final_cam must use a floating-point dtype")
    # The collected Experiment 2 artifacts are float32.  CPU interpolation does
    # not support every lower-precision dtype, so normalize those inputs in the
    # same float32 dtype used by the production evaluation path.
    if cam.dtype not in (torch.float32, torch.float64):
        cam = cam.float()
    if not bool(torch.isfinite(cam).all()):
        raise ValueError("final_cam must contain only finite values")
    return cam, input_is_tensor


def upsample_and_normalize_active_cams(
    final_cam: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reproduce ``make_cam.normalize_cam`` after exact 28 -> 448 upsampling.

    The returned container type follows the input: NumPy in, NumPy out; torch
    in, torch out.  Torch tensors stay on their input device.  The active-class
    order is not changed.
    """

    cam, input_is_tensor = _cam_grid_tensor(final_cam)
    upsampled = F.interpolate(
        cam.unsqueeze(0),
        size=TRANSFORMED_CROP_SIZE,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    channels = upsampled.shape[0]
    flat = upsampled.reshape(channels, -1)
    minimum = flat.min(dim=-1, keepdim=True)[0].reshape(channels, 1, 1)
    maximum = flat.max(dim=-1, keepdim=True)[0].reshape(channels, 1, 1)
    normalized = (upsampled - minimum) / (maximum - minimum + NORMALIZE_CAM_EPSILON)
    if input_is_tensor:
        return normalized
    return normalized.detach().cpu().numpy()


def _active_class_ids(
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
    expected: int,
) -> np.ndarray:
    classes = _as_numpy(active_class_ids, "active_class_ids")
    if classes.ndim != 1 or len(classes) != expected:
        raise ValueError(
            "active_class_ids must be a rank-one array matching CAM channels; "
            f"expected {expected}, got {classes.shape}"
        )
    if not np.issubdtype(classes.dtype, np.integer):
        raise TypeError("active_class_ids must use an integer dtype")
    classes = classes.astype(np.int64, copy=False)
    if np.any((classes < 0) | (classes >= VOC_FOREGROUND_CLASSES)):
        raise ValueError("active_class_ids must lie in the zero-based VOC range 0..19")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("active_class_ids must be unique")
    return classes


def raw_final_cam_prediction(
    final_cam: np.ndarray | torch.Tensor,
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Return the exact fixed-threshold raw-CAM prediction on the 448 crop.

    ``active_class_ids`` are zero-based image-level VOC class IDs.  The output
    uses semantic-mask IDs: 0 for background and 1..20 for foreground.  Ties
    with the fixed background score resolve to background, matching
    ``numpy.argmax`` over the repository's prepended background channel.
    """

    normalized = upsample_and_normalize_active_cams(final_cam)
    normalized_np = _as_numpy(normalized, "normalized CAM")
    classes = _active_class_ids(active_class_ids, normalized_np.shape[0])
    background = np.full(
        (1, *TRANSFORMED_CROP_SIZE),
        RAW_CAM_BACKGROUND_THRESHOLD,
        dtype=normalized_np.dtype,
    )
    scores = np.concatenate((background, normalized_np), axis=0)
    keys = np.concatenate((np.zeros(1, dtype=np.int64), classes + 1))
    return keys[np.argmax(scores, axis=0)]


def voc_confusion_matrix(
    prediction: np.ndarray | torch.Tensor,
    semantic_mask: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Build a fixed 21x21 VOC confusion matrix while ignoring mask ID 255."""

    pred = _as_numpy(prediction, "prediction")
    target = _as_numpy(semantic_mask, "semantic_mask")
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        raise ValueError(
            "prediction and semantic_mask must be same-shaped rank-two arrays; "
            f"got {pred.shape} and {target.shape}"
        )
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError("prediction must use an integer dtype")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("semantic_mask must use an integer dtype")
    pred = pred.astype(np.int64, copy=False)
    target = target.astype(np.int64, copy=False)
    if np.any((pred < 0) | (pred >= VOC_SEGMENTATION_CLASSES)):
        raise ValueError("prediction IDs must lie in 0..20")
    valid = target != VOC_VOID_ID
    if np.any((target[valid] < 0) | (target[valid] >= VOC_SEGMENTATION_CLASSES)):
        raise ValueError("non-void semantic_mask IDs must lie in 0..20")
    encoded = VOC_SEGMENTATION_CLASSES * target[valid] + pred[valid]
    return np.bincount(
        encoded,
        minlength=VOC_SEGMENTATION_CLASSES**2,
    ).reshape(VOC_SEGMENTATION_CLASSES, VOC_SEGMENTATION_CLASSES)


def raw_final_cam_confusion(
    final_cam: np.ndarray | torch.Tensor,
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
    transformed_semantic_mask: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Evaluate one raw final CAM against its exactly transformed VOC mask."""

    target = _as_numpy(transformed_semantic_mask, "transformed_semantic_mask")
    if target.shape != TRANSFORMED_CROP_SIZE:
        raise ValueError(
            "transformed_semantic_mask must be the exact 448x448 center crop, got "
            f"{target.shape}"
        )
    prediction = raw_final_cam_prediction(final_cam, active_class_ids)
    return voc_confusion_matrix(prediction, target)


def iou_from_confusion(
    confusion: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, float]:
    """Return class-wise IoU and the VOC macro mIoU from one 21x21 matrix."""

    matrix = _validated_confusions(confusion, "confusion", allow_batch=False)[0]
    iou = _iou_rows(matrix[None, ...])[0]
    return iou, _finite_mean(iou)


def average_precision_from_scores(
    labels: np.ndarray | torch.Tensor | Sequence[int],
    scores: np.ndarray | torch.Tensor | Sequence[float],
    sample_weight: np.ndarray | torch.Tensor | Sequence[float] | None = None,
) -> float:
    """Compute non-interpolated binary AP with exact tied-score grouping.

    A class with no positive-weight positive image returns NaN, so a bootstrap
    replicate that cannot identify AP is excluded from its percentile interval
    instead of being silently assigned zero.
    """

    truth = _validated_binary_vector(labels, "labels")
    values = _as_numpy(scores, "scores").astype(np.float64, copy=False)
    if values.ndim != 1 or values.shape != truth.shape:
        raise ValueError("scores must be rank one and have the same shape as labels")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    if sample_weight is None:
        weights = np.ones(len(truth), dtype=np.float64)
    else:
        weights = _as_numpy(sample_weight, "sample_weight").astype(
            np.float64, copy=False
        )
        if weights.ndim != 1 or weights.shape != truth.shape:
            raise ValueError(
                "sample_weight must be rank one and have the same shape as labels"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight must contain finite non-negative values")
    if not np.any(weights > 0):
        return float("nan")
    ordering = _average_precision_ordering(truth, values)
    result = _average_precision_for_draws(ordering, weights[None, :])
    return float(result[0])


def classification_average_precision(
    labels: np.ndarray | torch.Tensor,
    logits: np.ndarray | torch.Tensor,
    sample_weight: np.ndarray | torch.Tensor | Sequence[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Return all 20 class APs and their finite-class macro mean."""

    truth = _validated_image_labels(labels)
    values = _validated_logits(logits, len(truth), "logits")
    if sample_weight is None:
        weights = None
    else:
        weights = _as_numpy(sample_weight, "sample_weight")
        if weights.ndim != 1 or len(weights) != len(truth):
            raise ValueError("sample_weight must have one value per image")
    aps = np.asarray(
        [
            average_precision_from_scores(
                truth[:, class_id], values[:, class_id], weights
            )
            for class_id in range(VOC_FOREGROUND_CLASSES)
        ],
        dtype=np.float64,
    )
    return aps, _finite_mean(aps)


def _validated_binary_vector(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank one")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be numeric")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return array.astype(np.uint8, copy=False)


def _validated_image_labels(value: Any) -> np.ndarray:
    labels = _as_numpy(value, "image_labels")
    expected = (labels.shape[0], VOC_FOREGROUND_CLASSES) if labels.ndim == 2 else None
    if labels.ndim != 2 or labels.shape != expected:
        raise ValueError("image_labels must have shape [N,20]")
    if labels.shape[0] < 1:
        raise ValueError("image_labels must contain at least one image")
    numeric_or_bool = np.issubdtype(labels.dtype, np.number) or np.issubdtype(
        labels.dtype, np.bool_
    )
    if not numeric_or_bool or not np.isin(labels, (0, 1)).all():
        raise ValueError("image_labels must contain only binary 0/1 values")
    counts = labels.sum(axis=1)
    if np.any((counts < 1) | (counts > VOC_FOREGROUND_CLASSES)):
        raise ValueError("each image must have at least one positive foreground label")
    return labels.astype(np.uint8, copy=False)


def _validated_logits(value: Any, images: int, name: str) -> np.ndarray:
    logits = _as_numpy(value, name).astype(np.float64, copy=False)
    if logits.shape != (images, VOC_FOREGROUND_CLASSES):
        raise ValueError(f"{name} must have shape [{images},20], got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite values")
    return logits


def _validated_image_ids(image_ids: Sequence[str], images: int) -> np.ndarray:
    ids = np.asarray([str(value) for value in image_ids], dtype=str)
    if ids.ndim != 1 or len(ids) != images:
        raise ValueError(f"image_ids must contain exactly {images} entries")
    if np.any(ids == ""):
        raise ValueError("image_ids cannot contain empty strings")
    if len(np.unique(ids)) != images:
        raise ValueError("image_ids must be unique: inputs are one row per image")
    return ids


def _validated_confusions(
    value: Any,
    name: str,
    *,
    allow_batch: bool = True,
) -> np.ndarray:
    matrices = _as_numpy(value, name)
    expected_tail = (VOC_SEGMENTATION_CLASSES, VOC_SEGMENTATION_CLASSES)
    if not allow_batch:
        if matrices.ndim != 2 or tuple(matrices.shape) != expected_tail:
            raise ValueError(f"{name} must have shape [21,21], got {matrices.shape}")
        matrices = matrices[None, ...]
    elif matrices.ndim != 3 or tuple(matrices.shape[-2:]) != expected_tail:
        raise ValueError(f"{name} must have shape [N,21,21], got {matrices.shape}")
    if not np.issubdtype(matrices.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    matrices = matrices.astype(np.int64, copy=False)
    if np.any(matrices < 0):
        raise ValueError(f"{name} cannot contain negative counts")
    return matrices


def _sorted_paired_inputs(
    image_ids: Sequence[str],
    image_labels: Any,
    mctformer_values: np.ndarray,
    mctformer_plus_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = _validated_image_labels(image_labels)
    ids = _validated_image_ids(image_ids, len(labels))
    if len(mctformer_values) != len(labels) or len(mctformer_plus_values) != len(
        labels
    ):
        raise ValueError("both model arrays must contain exactly one row per image_id")
    order = np.argsort(ids, kind="stable")
    return (
        ids[order],
        labels[order],
        mctformer_values[order],
        mctformer_plus_values[order],
    )


def _stratum_masks(labels: np.ndarray) -> Mapping[str, np.ndarray]:
    count = labels.sum(axis=1)
    return {
        "all": np.ones(len(labels), dtype=bool),
        "single_label": count == 1,
        "exactly_2_labels": count == 2,
        "3plus_labels": count >= 3,
    }


def _bootstrap_multiplicities(images: int, repeats: int, seed: int) -> np.ndarray:
    if images < 1:
        raise ValueError("cannot bootstrap an empty image stratum")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    probabilities = np.full(images, 1.0 / images, dtype=np.float64)
    draws = rng.multinomial(images, probabilities, size=repeats)
    dtype = np.uint16 if images <= np.iinfo(np.uint16).max else np.uint32
    return draws.astype(dtype, copy=False)


def _average_precision_ordering(
    truth: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_truth = truth[order].astype(np.float64, copy=False)
    group_ends = np.concatenate(
        (np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), [len(order) - 1])
    ).astype(np.int64, copy=False)
    return order, ordered_truth, group_ends


def _average_precision_for_draws(
    ordering: tuple[np.ndarray, np.ndarray, np.ndarray],
    draw_weights: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    order, ordered_truth, group_ends = ordering
    if draw_weights.ndim != 2 or draw_weights.shape[1] != len(order):
        raise ValueError("draw_weights must have shape [R,N]")
    output = np.full(len(draw_weights), np.nan, dtype=np.float64)
    for offset in range(0, len(draw_weights), chunk_size):
        selected = draw_weights[offset : offset + chunk_size, order].astype(
            np.float64, copy=False
        )
        cumulative_total = np.cumsum(selected, axis=1)
        cumulative_positive = np.cumsum(selected * ordered_truth, axis=1)
        total_at_threshold = cumulative_total[:, group_ends]
        positive_at_threshold = cumulative_positive[:, group_ends]
        positive_increment = np.diff(
            np.pad(positive_at_threshold, ((0, 0), (1, 0))), axis=1
        )
        precision = np.divide(
            positive_at_threshold,
            total_at_threshold,
            out=np.zeros_like(positive_at_threshold),
            where=total_at_threshold > 0,
        )
        total_positive = positive_at_threshold[:, -1]
        numerator = np.sum(positive_increment * precision, axis=1)
        result = np.divide(
            numerator,
            total_positive,
            out=np.full(len(selected), np.nan, dtype=np.float64),
            where=total_positive > 0,
        )
        output[offset : offset + len(selected)] = result
    return output


def _classification_bootstrap_values(
    labels: np.ndarray,
    logits: np.ndarray,
    draws: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.full(VOC_FOREGROUND_CLASSES + 1, np.nan, dtype=np.float64)
    bootstrap = np.full(
        (len(draws), VOC_FOREGROUND_CLASSES + 1), np.nan, dtype=np.float64
    )
    unit_weights = np.ones(len(labels), dtype=np.float64)
    for class_id in range(VOC_FOREGROUND_CLASSES):
        ordering = _average_precision_ordering(labels[:, class_id], logits[:, class_id])
        point[class_id] = _average_precision_for_draws(
            ordering, unit_weights[None, :], chunk_size=1
        )[0]
        bootstrap[:, class_id] = _average_precision_for_draws(
            ordering, draws, chunk_size=chunk_size
        )
    point[-1] = _finite_mean(point[:-1])
    bootstrap[:, -1] = _finite_mean_rows(bootstrap[:, :-1])
    return point, bootstrap


def _iou_rows(confusions: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(confusions, axis1=1, axis2=2).astype(np.float64, copy=False)
    target = confusions.sum(axis=2, dtype=np.float64)
    predicted = confusions.sum(axis=1, dtype=np.float64)
    union = target + predicted - diagonal
    return np.divide(
        diagonal,
        union,
        out=np.full_like(diagonal, np.nan, dtype=np.float64),
        where=union > 0,
    )


def _cam_bootstrap_values(
    confusions: np.ndarray,
    draws: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.full(VOC_SEGMENTATION_CLASSES + 1, np.nan, dtype=np.float64)
    point[:-1] = _iou_rows(confusions.sum(axis=0, keepdims=True))[0]
    point[-1] = _finite_mean(point[:-1])
    bootstrap = np.full(
        (len(draws), VOC_SEGMENTATION_CLASSES + 1), np.nan, dtype=np.float64
    )
    flat = confusions.reshape(len(confusions), -1).astype(np.float64, copy=False)
    for offset in range(0, len(draws), chunk_size):
        selected = draws[offset : offset + chunk_size].astype(np.float64, copy=False)
        aggregated = (selected @ flat).reshape(
            len(selected), VOC_SEGMENTATION_CLASSES, VOC_SEGMENTATION_CLASSES
        )
        iou = _iou_rows(aggregated)
        bootstrap[offset : offset + len(selected), :-1] = iou
        bootstrap[offset : offset + len(selected), -1] = _finite_mean_rows(iou)
    return point, bootstrap


def _finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].mean())


def _finite_mean_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=1),
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    )


def _percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _append_metric_records(
    records: list[dict[str, object]],
    *,
    evaluation: str,
    label_stratum: str,
    metric: str,
    aggregation: str,
    class_id: int | None,
    metric_index: int,
    points: Mapping[str, np.ndarray],
    bootstraps: Mapping[str, np.ndarray],
    num_images: int,
    repeats: int,
    base_seed: int,
    draw_seed: int,
    point_support: int,
    point_support_kind: str,
    extra_identity: Mapping[str, object] | None = None,
) -> None:
    for series in (_MCTFORMER, _MCTFORMER_PLUS, _PAIRED_DELTA):
        samples = bootstraps[series][:, metric_index]
        low, high, valid_repeats = _percentile_interval(samples)
        record = {
            "evaluation": evaluation,
            "label_stratum": label_stratum,
            "model_or_delta": series,
            "metric": metric,
            "aggregation": aggregation,
            "class_id": class_id,
            "estimate": float(points[series][metric_index]),
            "ci_low": low,
            "ci_high": high,
            "num_images": int(num_images),
            "point_support": int(point_support),
            "point_support_kind": point_support_kind,
            "bootstrap_repeats": int(repeats),
            "bootstrap_valid_repeats": valid_repeats,
            "bootstrap_base_seed": int(base_seed),
            "bootstrap_draw_seed": int(draw_seed),
            "bootstrap_unit": "image",
            "paired_delta": series == _PAIRED_DELTA,
            "delta_definition": (
                "MCTformer+ - MCTformer" if series == _PAIRED_DELTA else ""
            ),
            "ci_method": "95% percentile",
        }
        if extra_identity:
            overlap = set(record).intersection(extra_identity)
            if overlap:
                raise ValueError(
                    f"extra_identity cannot replace record fields: {overlap}"
                )
            record.update(extra_identity)
        records.append(record)


def _validate_bootstrap_options(repeats: int, seed: int, chunk_size: int) -> None:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")


def paired_classification_bootstrap(
    image_ids: Sequence[str],
    image_labels: np.ndarray | torch.Tensor,
    mctformer_logits: np.ndarray | torch.Tensor,
    mctformer_plus_logits: np.ndarray | torch.Tensor,
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = 256,
    logit_source: str = "class_token",
) -> list[dict[str, object]]:
    """Summarize class-wise AP, macro mAP, and paired deltas by label stratum.

    Inputs contain one row per image and all 20 foreground classes.  The output
    contains MCTformer, MCTformer+, and ``MCTformer+ - MCTformer`` rows.  A
    single multiplicity matrix is reused for both models within each stratum.
    """

    _validate_bootstrap_options(repeats, seed, chunk_size)
    if not isinstance(logit_source, str) or not logit_source:
        raise ValueError("logit_source must be a non-empty string")
    labels = _validated_image_labels(image_labels)
    baseline = _validated_logits(mctformer_logits, len(labels), "mctformer_logits")
    comparison = _validated_logits(
        mctformer_plus_logits, len(labels), "mctformer_plus_logits"
    )
    _, labels, baseline, comparison = _sorted_paired_inputs(
        image_ids, labels, baseline, comparison
    )

    records: list[dict[str, object]] = []
    for stratum in LABEL_STRATA:
        mask = _stratum_masks(labels)[stratum]
        if not mask.any():
            continue
        stratum_labels = labels[mask]
        draw_seed = seed + _STRATUM_SEED_OFFSET[stratum]
        draws = _bootstrap_multiplicities(len(stratum_labels), repeats, draw_seed)
        baseline_point, baseline_boot = _classification_bootstrap_values(
            stratum_labels, baseline[mask], draws, chunk_size=chunk_size
        )
        comparison_point, comparison_boot = _classification_bootstrap_values(
            stratum_labels, comparison[mask], draws, chunk_size=chunk_size
        )
        points = {
            _MCTFORMER: baseline_point,
            _MCTFORMER_PLUS: comparison_point,
            _PAIRED_DELTA: comparison_point - baseline_point,
        }
        bootstraps = {
            _MCTFORMER: baseline_boot,
            _MCTFORMER_PLUS: comparison_boot,
            _PAIRED_DELTA: comparison_boot - baseline_boot,
        }
        positives = stratum_labels.sum(axis=0)
        for class_id in range(VOC_FOREGROUND_CLASSES):
            _append_metric_records(
                records,
                evaluation="classification",
                label_stratum=stratum,
                metric="average_precision",
                aggregation="classwise",
                class_id=class_id,
                metric_index=class_id,
                points=points,
                bootstraps=bootstraps,
                num_images=len(stratum_labels),
                repeats=repeats,
                base_seed=seed,
                draw_seed=draw_seed,
                point_support=int(positives[class_id]),
                point_support_kind="positive_images",
                extra_identity={"logit_source": logit_source},
            )
        _append_metric_records(
            records,
            evaluation="classification",
            label_stratum=stratum,
            metric="mean_average_precision",
            aggregation="macro_class",
            class_id=None,
            metric_index=VOC_FOREGROUND_CLASSES,
            points=points,
            bootstraps=bootstraps,
            num_images=len(stratum_labels),
            repeats=repeats,
            base_seed=seed,
            draw_seed=draw_seed,
            point_support=int(np.isfinite(baseline_point[:-1]).sum()),
            point_support_kind="finite_classes",
            extra_identity={"logit_source": logit_source},
        )
    return records


def paired_cam_iou_bootstrap(
    image_ids: Sequence[str],
    image_labels: np.ndarray | torch.Tensor,
    mctformer_confusions: np.ndarray | torch.Tensor,
    mctformer_plus_confusions: np.ndarray | torch.Tensor,
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = 128,
) -> list[dict[str, object]]:
    """Summarize raw-CAM class IoU, macro mIoU, and paired deltas.

    Each confusion matrix must have shape 21x21 and represent one transformed
    image after void pixels were excluded.  Matching GT row marginals are
    required for the two models, which catches accidental image misalignment.
    """

    _validate_bootstrap_options(repeats, seed, chunk_size)
    labels = _validated_image_labels(image_labels)
    baseline = _validated_confusions(mctformer_confusions, "mctformer_confusions")
    comparison = _validated_confusions(
        mctformer_plus_confusions, "mctformer_plus_confusions"
    )
    _, labels, baseline, comparison = _sorted_paired_inputs(
        image_ids, labels, baseline, comparison
    )
    if not np.array_equal(baseline.sum(axis=2), comparison.sum(axis=2)):
        raise ValueError(
            "paired CAM confusions must have identical per-image GT row marginals"
        )

    records: list[dict[str, object]] = []
    for stratum in LABEL_STRATA:
        mask = _stratum_masks(labels)[stratum]
        if not mask.any():
            continue
        baseline_subset = baseline[mask]
        comparison_subset = comparison[mask]
        draw_seed = seed + _STRATUM_SEED_OFFSET[stratum]
        draws = _bootstrap_multiplicities(len(baseline_subset), repeats, draw_seed)
        baseline_point, baseline_boot = _cam_bootstrap_values(
            baseline_subset, draws, chunk_size=chunk_size
        )
        comparison_point, comparison_boot = _cam_bootstrap_values(
            comparison_subset, draws, chunk_size=chunk_size
        )
        points = {
            _MCTFORMER: baseline_point,
            _MCTFORMER_PLUS: comparison_point,
            _PAIRED_DELTA: comparison_point - baseline_point,
        }
        bootstraps = {
            _MCTFORMER: baseline_boot,
            _MCTFORMER_PLUS: comparison_boot,
            _PAIRED_DELTA: comparison_boot - baseline_boot,
        }
        gt_pixels = baseline_subset.sum(axis=(0, 2))
        for class_id in range(VOC_SEGMENTATION_CLASSES):
            _append_metric_records(
                records,
                evaluation="raw_final_cam",
                label_stratum=stratum,
                metric="intersection_over_union",
                aggregation="classwise",
                class_id=class_id,
                metric_index=class_id,
                points=points,
                bootstraps=bootstraps,
                num_images=len(baseline_subset),
                repeats=repeats,
                base_seed=seed,
                draw_seed=draw_seed,
                point_support=int(gt_pixels[class_id]),
                point_support_kind="ground_truth_pixels",
                extra_identity={
                    "cam_stage": "final_cam",
                    "background_threshold": RAW_CAM_BACKGROUND_THRESHOLD,
                    "input_resolution": TRANSFORMED_CROP_SIZE[0],
                },
            )
        _append_metric_records(
            records,
            evaluation="raw_final_cam",
            label_stratum=stratum,
            metric="mean_intersection_over_union",
            aggregation="macro_class",
            class_id=None,
            metric_index=VOC_SEGMENTATION_CLASSES,
            points=points,
            bootstraps=bootstraps,
            num_images=len(baseline_subset),
            repeats=repeats,
            base_seed=seed,
            draw_seed=draw_seed,
            point_support=int(np.isfinite(baseline_point[:-1]).sum()),
            point_support_kind="finite_classes",
            extra_identity={
                "cam_stage": "final_cam",
                "background_threshold": RAW_CAM_BACKGROUND_THRESHOLD,
                "input_resolution": TRANSFORMED_CROP_SIZE[0],
            },
        )
    return records
