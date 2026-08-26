#!/usr/bin/env python3
"""Evaluate soft and thresholded CAM consistency across input resolutions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC_NAMES = (
    "mean_class_cam_cosine",
    "mean_class_binary_iou",
    "foreground_mask_iou",
    "semantic_mask_mean_iou",
    "semantic_pixel_agreement",
)


def parse_resolutions(value):
    resolutions = tuple(int(item) for item in value.split(","))
    if not resolutions or len(set(resolutions)) != len(resolutions):
        raise argparse.ArgumentTypeError("resolutions must be a non-empty unique list")
    return resolutions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=parse_resolutions, default=(224, 320, 448, 512))
    parser.add_argument("--reference-resolution", type=int, default=448)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    return parser.parse_args()


def load_cam(path):
    cam_dict = np.load(path, allow_pickle=True).item()
    if not isinstance(cam_dict, dict) or not cam_dict:
        raise ValueError(f"Expected a non-empty CAM dictionary in {path}")
    result = {int(key): np.asarray(value, dtype=np.float64) for key, value in cam_dict.items()}
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"Invalid CAM values in {path}")
    return result


def semantic_prediction(cams, threshold):
    class_ids = np.asarray(sorted(cams), dtype=np.int64)
    values = np.stack([cams[class_id] for class_id in class_ids])
    best_index = values.argmax(axis=0)
    best_score = np.take_along_axis(values, best_index[None], axis=0)[0]
    prediction = np.zeros(best_index.shape, dtype=np.int64)
    foreground = best_score > threshold
    prediction[foreground] = class_ids[best_index[foreground]] + 1
    return prediction


def binary_iou(left, right):
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum()) / float(union)


def semantic_mean_iou(left, right):
    values = []
    for class_id in np.union1d(np.unique(left), np.unique(right)):
        values.append(binary_iou(left == class_id, right == class_id))
    return float(np.mean(values))


def compare(cams, reference, threshold):
    if set(cams) != set(reference):
        raise ValueError("CAM class keys differ across resolutions")
    cosines = []
    binary_ious = []
    for class_id in sorted(reference):
        left = cams[class_id]
        right = reference[class_id]
        if left.shape != right.shape:
            raise ValueError(f"CAM shape differs for class {class_id}: {left.shape} != {right.shape}")
        denominator = np.linalg.norm(left) * np.linalg.norm(right) + 1e-8
        cosines.append(float(np.vdot(left, right) / denominator))
        binary_ious.append(binary_iou(left > threshold, right > threshold))
    prediction = semantic_prediction(cams, threshold)
    reference_prediction = semantic_prediction(reference, threshold)
    return {
        "mean_class_cam_cosine": float(np.mean(cosines)),
        "mean_class_binary_iou": float(np.mean(binary_ious)),
        "foreground_mask_iou": binary_iou(
            prediction > 0, reference_prediction > 0
        ),
        "semantic_mask_mean_iou": semantic_mean_iou(
            prediction, reference_prediction
        ),
        "semantic_pixel_agreement": float(np.mean(prediction == reference_prediction)),
    }


def bootstrap_mean(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 500
    for begin in range(0, resamples, batch_size):
        end = min(begin + batch_size, resamples)
        indices = generator.integers(0, len(values), size=(end - begin, len(values)))
        means[begin:end] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def main():
    args = parse_args()
    if args.reference_resolution not in args.resolutions:
        raise ValueError("reference-resolution must be included in resolutions")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    manifest_path = args.cam_root / "manifest.json"
    if not manifest_path.is_file() or not (args.cam_root / "COMPLETE").is_file():
        raise FileNotFoundError(f"Scale CAM generation is incomplete in {args.cam_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["resolutions"] != list(args.resolutions):
        raise ValueError("Scale CAM manifest resolution mismatch")

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines() if line.strip()]
    rows = []
    summary_values = defaultdict(list)
    for image_id in image_ids:
        reference = load_cam(
            args.cam_root / str(args.reference_resolution) / f"{image_id}.npy"
        )
        for resolution in args.resolutions:
            if resolution == args.reference_resolution:
                continue
            cams = load_cam(args.cam_root / str(resolution) / f"{image_id}.npy")
            metrics = compare(cams, reference, args.threshold)
            row = {
                "image_id": image_id,
                "resolution": resolution,
                "reference_resolution": args.reference_resolution,
                **metrics,
            }
            rows.append(row)
            for name, value in metrics.items():
                summary_values[(resolution, name)].append(value)

    summary = {
        str(resolution): {
            metric_name: bootstrap_mean(
                summary_values[(resolution, metric_name)],
                args.bootstrap_resamples,
                args.bootstrap_seed + resolution + metric_index,
            )
            for metric_index, metric_name in enumerate(METRIC_NAMES)
        }
        for resolution in args.resolutions
        if resolution != args.reference_resolution
    }
    metrics = {
        "host": manifest["host"],
        "normalization": manifest["normalization"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "num_images": len(image_ids),
        "resolutions": list(args.resolutions),
        "reference_resolution": args.reference_resolution,
        "background_threshold": args.threshold,
        "cam_normalization": manifest["cam_normalization"],
        "resolution_definition": manifest["resolution_definition"],
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics_by_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
