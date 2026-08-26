#!/usr/bin/env python3
"""Collect fixed-threshold raw-CAM localization and error diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


NUM_CLASSES = 21
CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-dir", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def predict(cam_path, threshold):
    cam_dict = np.load(cam_path, allow_pickle=True).item()
    if not isinstance(cam_dict, dict) or not cam_dict:
        raise ValueError(f"Expected a non-empty CAM dictionary in {cam_path}")

    items = list(cam_dict.items())
    class_ids = np.asarray([int(key) + 1 for key, _ in items], dtype=np.int64)
    cams = np.stack([np.asarray(value) for _, value in items], axis=0)
    if cams.ndim != 3 or not np.isfinite(cams).all():
        raise ValueError(f"Invalid CAM tensor in {cam_path}: shape={cams.shape}")

    best_index = np.argmax(cams, axis=0)
    best_score = np.take_along_axis(cams, best_index[None], axis=0)[0]
    prediction = np.zeros(best_index.shape, dtype=np.int64)
    foreground = best_score > threshold
    prediction[foreground] = class_ids[best_index[foreground]]
    return prediction


def image_counts(prediction, target):
    valid = target >= 0
    prediction = prediction[valid]
    target = target[valid]
    pred_fg = prediction > 0
    target_fg = target > 0
    correct_semantic_fg = pred_fg & (prediction == target)
    foreground_overlap = pred_fg & target_fg
    false_positive_bg = pred_fg & ~target_fg
    return np.asarray(
        [
            correct_semantic_fg.sum(),
            foreground_overlap.sum(),
            pred_fg.sum(),
            target_fg.sum(),
            false_positive_bg.sum(),
            (~target_fg).sum(),
            valid.sum(),
            (prediction == target).sum(),
        ],
        dtype=np.int64,
    )


def bootstrap_intervals(counts, resamples, seed):
    if resamples <= 0:
        return {}
    generator = np.random.default_rng(seed)
    samples = {name: np.empty(resamples, dtype=np.float64) for name in (
        "semantic_foreground_precision",
        "semantic_foreground_recall",
        "binary_foreground_precision",
        "binary_foreground_recall",
        "background_false_positive_rate",
    )}
    batch_size = 250
    for begin in range(0, resamples, batch_size):
        end = min(begin + batch_size, resamples)
        indices = generator.integers(0, len(counts), size=(end - begin, len(counts)))
        totals = counts[indices].sum(axis=1).astype(np.float64)
        values = (
            totals[:, 0] / totals[:, 2],
            totals[:, 0] / totals[:, 3],
            totals[:, 1] / totals[:, 2],
            totals[:, 1] / totals[:, 3],
            totals[:, 4] / totals[:, 5],
        )
        for name, value in zip(samples, values):
            samples[name][begin:end] = value
    return {
        name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for name, values in samples.items()
    }


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.bootstrap_resamples < 0:
        raise ValueError("bootstrap-resamples must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines() if line.strip()]
    if not image_ids:
        raise ValueError("id-list contains no images")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image = []
    counts = []
    for index, image_id in enumerate(image_ids):
        prediction = predict(args.cam_dir / f"{image_id}.npy", args.threshold)
        target = np.asarray(
            Image.open(args.voc_root / "SegmentationClass" / f"{image_id}.png"),
            dtype=np.int64,
        )
        target[target == 255] = -1
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {image_id}: CAM={prediction.shape}, target={target.shape}"
            )
        valid = target >= 0
        confusion += np.bincount(
            NUM_CLASSES * target[valid] + prediction[valid],
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        item_counts = image_counts(prediction, target)
        counts.append(item_counts)
        semantic_tp, binary_tp, pred_fg, target_fg, fp_bg, target_bg, valid_pixels, correct = item_counts
        per_image.append(
            {
                "image_id": image_id,
                "semantic_foreground_precision": safe_ratio(semantic_tp, pred_fg),
                "semantic_foreground_recall": safe_ratio(semantic_tp, target_fg),
                "binary_foreground_precision": safe_ratio(binary_tp, pred_fg),
                "binary_foreground_recall": safe_ratio(binary_tp, target_fg),
                "background_false_positive_rate": safe_ratio(fp_bg, target_bg),
                "pixel_accuracy": safe_ratio(correct, valid_pixels),
                "semantic_correct_foreground_pixels": int(semantic_tp),
                "foreground_overlap_pixels": int(binary_tp),
                "predicted_foreground_pixels": int(pred_fg),
                "target_foreground_pixels": int(target_fg),
                "false_positive_background_pixels": int(fp_bg),
                "target_background_pixels": int(target_bg),
                "valid_pixels": int(valid_pixels),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(image_ids):
            print(f"images={index + 1}/{len(image_ids)}", flush=True)

    counts = np.stack(counts)
    totals = counts.sum(axis=0)
    gt_pixels = confusion.sum(axis=1)
    pred_pixels = confusion.sum(axis=0)
    correct_pixels = np.diag(confusion)
    union = gt_pixels + pred_pixels - correct_pixels
    with np.errstate(divide="ignore", invalid="ignore"):
        class_iou = correct_pixels / union
        class_precision = correct_pixels / pred_pixels
        class_recall = correct_pixels / gt_pixels

    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": None if np.isnan(class_iou[class_id]) else float(class_iou[class_id]),
                "precision": None if np.isnan(class_precision[class_id]) else float(class_precision[class_id]),
                "recall": None if np.isnan(class_recall[class_id]) else float(class_recall[class_id]),
                "true_positive_pixels": int(correct_pixels[class_id]),
                "predicted_pixels": int(pred_pixels[class_id]),
                "target_pixels": int(gt_pixels[class_id]),
                "union_pixels": int(union[class_id]),
            }
        )

    semantic_tp, binary_tp, pred_fg, target_fg, fp_bg, target_bg, valid_pixels, correct = totals
    metrics = {
        "dataset": "PASCAL VOC 2012",
        "split": args.id_list.stem,
        "num_images": len(image_ids),
        "num_classes_including_background": NUM_CLASSES,
        "background_threshold": args.threshold,
        "cam_postprocessing": "per-class min-max normalization from make_cam.py; no CRF",
        "threshold_rule": "background wins ties; foreground requires max CAM > threshold",
        "mean_iou_percent": 100.0 * float(np.nanmean(class_iou)),
        "foreground_mean_iou_percent": 100.0 * float(np.nanmean(class_iou[1:])),
        "pixel_accuracy_percent": 100.0 * safe_ratio(correct, valid_pixels),
        "semantic_foreground_precision_percent": 100.0 * safe_ratio(semantic_tp, pred_fg),
        "semantic_foreground_recall_percent": 100.0 * safe_ratio(semantic_tp, target_fg),
        "binary_foreground_precision_percent": 100.0 * safe_ratio(binary_tp, pred_fg),
        "binary_foreground_recall_percent": 100.0 * safe_ratio(binary_tp, target_fg),
        "background_false_positive_rate_percent": 100.0 * safe_ratio(fp_bg, target_bg),
        "predicted_foreground_semantic_error_rate_percent": 100.0 * (
            1.0 - safe_ratio(semantic_tp, pred_fg)
        ),
        "definitions": {
            "semantic_foreground_precision": "correct non-background class pixels / predicted foreground pixels",
            "semantic_foreground_recall": "correct non-background class pixels / target foreground pixels",
            "binary_foreground_precision": "predicted foreground on any target foreground / predicted foreground pixels",
            "binary_foreground_recall": "predicted foreground on any target foreground / target foreground pixels",
            "background_false_positive_rate": "predicted foreground on target background / target background pixels",
        },
        "bootstrap": {
            "unit": "image",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "ci95": bootstrap_intervals(
                counts, args.bootstrap_resamples, args.bootstrap_seed
            ),
        },
        "provenance": {
            "cam_dir": str(args.cam_dir.resolve()),
            "voc_root": str(args.voc_root.resolve()),
            "id_list": str(args.id_list.resolve()),
            "id_list_sha256": sha256(args.id_list),
        },
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics_by_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    with (args.output_dir / "metrics_by_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    np.savetxt(
        args.output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=","
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
