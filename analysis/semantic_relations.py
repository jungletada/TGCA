"""Deterministic metrics for class-to-patch and patch-to-class relations."""

from __future__ import annotations

import numpy as np
import torch


EPS = 1e-8


def conditional_relations(raw_class_to_patch, raw_patch_to_class):
    """Normalize the same raw relation along its two semantic axes."""
    if raw_class_to_patch.ndim != 3 or raw_patch_to_class.ndim != 3:
        raise ValueError("Expected head x class x patch and head x patch x class tensors")
    if raw_class_to_patch.shape[0] != raw_patch_to_class.shape[0]:
        raise ValueError("Class-to-patch and patch-to-class head counts differ")
    if raw_class_to_patch.shape[1:] != raw_patch_to_class.shape[1:][::-1]:
        raise ValueError("Class-to-patch and patch-to-class token counts differ")
    class_to_patch = torch.softmax(raw_class_to_patch.float(), dim=-1)
    patch_to_class = torch.softmax(raw_patch_to_class.float(), dim=-1)
    return class_to_patch, patch_to_class


def present_class_relation(raw_patch_to_class, active_classes):
    """Compute P(c|p) after applying only image-level class supervision."""
    if raw_patch_to_class.ndim != 3:
        raise ValueError("Expected a head x patch x class tensor")
    active_classes = torch.as_tensor(
        active_classes, dtype=torch.bool, device=raw_patch_to_class.device
    )
    if active_classes.ndim != 1 or active_classes.shape[0] != raw_patch_to_class.shape[-1]:
        raise ValueError("active_classes must match the class dimension")
    if not bool(active_classes.any()):
        raise ValueError("At least one image-level class must be active")
    logits = raw_patch_to_class.float().masked_fill(
        ~active_classes.view(1, 1, -1), -torch.inf
    )
    return torch.softmax(logits, dim=-1)


def spatial_minmax(scores):
    """Apply the baseline CAM's per-class spatial min-max normalization."""
    if scores.ndim < 2:
        raise ValueError("scores must end in class x patch dimensions")
    minimum = scores.amin(dim=-1, keepdim=True)
    maximum = scores.amax(dim=-1, keepdim=True)
    return (scores - minimum) / (maximum - minimum).clamp_min(EPS)


def mutual_relation(class_to_patch, patch_to_class):
    """Return the diagnostic geometric mean of P(p|c) and P(c|p)."""
    if class_to_patch.ndim != 2 or patch_to_class.ndim != 2:
        raise ValueError("Expected class x patch and patch x class matrices")
    if class_to_patch.shape != patch_to_class.transpose(0, 1).shape:
        raise ValueError("Relation shapes are incompatible")
    return torch.sqrt(
        class_to_patch.float().clamp_min(0)
        * patch_to_class.transpose(0, 1).float().clamp_min(0)
    )


def cam_prediction(scores, active_classes, threshold):
    """Predict VOC labels at patch resolution using a fixed background threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    active_classes = np.asarray(active_classes, dtype=bool)
    if scores.ndim != 2 or active_classes.shape != (scores.shape[0],):
        raise ValueError("scores and active_classes have incompatible shapes")
    if not active_classes.any():
        raise ValueError("At least one class must be active")
    masked = np.where(active_classes[:, None], scores, -np.inf)
    best_class = masked.argmax(axis=0)
    best_score = masked[best_class, np.arange(masked.shape[1])]
    prediction = np.zeros(masked.shape[1], dtype=np.int64)
    foreground = best_score > threshold
    prediction[foreground] = best_class[foreground] + 1
    return prediction


def semantic_prediction(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("Expected patch x class probabilities")
    return probabilities.argmax(axis=-1).astype(np.int64)


def confusion_matrix(target, prediction, num_classes, valid=None):
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    if valid is None:
        valid = np.ones(target.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool).reshape(-1)
    valid &= (target >= 0) & (target < num_classes)
    valid &= (prediction >= 0) & (prediction < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def confusion_summary(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    prediction_count = confusion.sum(axis=0).astype(np.float64)
    union = target_count + prediction_count - true_positive
    accuracy = np.divide(
        true_positive, target_count, out=np.full_like(true_positive, np.nan),
        where=target_count > 0,
    )
    iou = np.divide(
        true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0
    )
    total = confusion.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else None,
        "mean_iou": float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        "per_class_accuracy": accuracy,
        "per_class_iou": iou,
        "target_count": target_count,
    }


def foreground_counts(prediction, target):
    """Return semantic TP, binary overlap, predicted FG, target FG, and valid count."""
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    valid = target != 255
    prediction = prediction[valid]
    target = target[valid]
    predicted_foreground = prediction > 0
    target_foreground = target > 0
    return np.asarray(
        [
            np.logical_and(predicted_foreground, prediction == target).sum(),
            np.logical_and(predicted_foreground, target_foreground).sum(),
            predicted_foreground.sum(),
            target_foreground.sum(),
            np.logical_and(predicted_foreground, ~target_foreground).sum(),
            (~target_foreground).sum(),
            valid.sum(),
        ],
        dtype=np.int64,
    )


def four_region_masks(class_score, semantic_probability, threshold):
    class_high = np.asarray(class_score) > threshold
    semantic_high = np.asarray(semantic_probability) > threshold
    if class_high.shape != semantic_high.shape:
        raise ValueError("class and semantic relation shapes differ")
    return {
        "A": class_high & semantic_high,
        "B": class_high & ~semantic_high,
        "C": ~class_high & semantic_high,
        "D": ~class_high & ~semantic_high,
    }


def region_composition(mask, target, class_id):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if mask.shape != target.shape:
        raise ValueError("region mask and target shapes differ")
    return np.asarray(
        [
            np.logical_and(mask, target == class_id).sum(),
            np.logical_and(mask, (target > 0) & (target != class_id) & (target != 255)).sum(),
            np.logical_and(mask, target == 0).sum(),
            np.logical_and(mask, target == 255).sum(),
        ],
        dtype=np.int64,
    )
