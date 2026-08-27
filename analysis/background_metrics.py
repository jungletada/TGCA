#!/usr/bin/env python3
"""Measure whether a slot/register map aligns with semantic background."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from scipy.stats import spearmanr

from analysis._common import finite_mean, load_segmentation


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--map-key",
        choices=("background_ownership", "background_attention", "background_raw_score",
                 "register_to_patch", "patch_to_register", "cam_complement"),
        required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-transform", choices=("none", "sigmoid", "max"), default="none")
    return parser.parse_args()


def background_score(sample, key):
    if key == "cam_complement":
        maps = np.maximum(sample["final_cam"].astype(np.float64), 0.0)
        label = sample["label"].astype(bool)
        active = maps[label]
        scale = active.max(axis=(1, 2), keepdims=True) + EPS
        return 1.0 - (active / scale).max(axis=0)
    score = sample[key].astype(np.float64)
    while score.ndim > 2:
        score = score.mean(axis=0)
    return score


def transform_score(score, transform):
    if transform == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    if transform == "max":
        return score / (score.max() + EPS)
    return score


def entropy(score):
    probability = np.maximum(score.reshape(-1), 0.0)
    probability /= probability.sum() + EPS
    return float(-(probability * np.log(probability + EPS)).sum())


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = []
    for path in sorted((args.dump_dir / "samples").glob("*.npz")):
        with np.load(path) as sample:
            if args.map_key != "cam_complement" and args.map_key not in sample:
                continue
            image_id = str(sample["image_id"])
            score = transform_score(background_score(sample, args.map_key), args.score_transform)
            feature_norm = sample["patch_feature_norm"].astype(np.float64)
        target = load_segmentation(args.voc_root, image_id, score.shape)
        valid = target != 255
        truth = target[valid] == 0
        values = score[valid]
        prediction = values >= args.threshold
        intersection = np.logical_and(prediction, truth).sum()
        union = np.logical_or(prediction, truth).sum()
        corr = spearmanr(values, feature_norm[valid]).statistic
        rows.append({
            "image_id": image_id,
            "background_iou": intersection / (union + EPS),
            "background_auprc": average_precision_score(truth, values),
            "background_balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "background_spearman": spearmanr(values, truth.astype(np.float64)).statistic,
            "feature_norm_spearman": corr,
            "map_entropy": entropy(score),
            "background_fraction": truth.mean(),
            "predicted_background_fraction": prediction.mean(),
        })
    if not rows:
        raise RuntimeError(f"No {args.map_key} maps found in {args.dump_dir}")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = tuple(key for key in rows[0] if key != "image_id")
    metrics = {
        "map_key": args.map_key,
        "score_transform": args.score_transform,
        "threshold": args.threshold,
        "num_images": len(rows),
        **{name: finite_mean([row[name] for row in rows]) for name in metric_names},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
