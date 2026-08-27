#!/usr/bin/env python3
"""Compute foreground leakage and cross-class collision from unified dumps."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from analysis._common import finite_mean, load_segmentation


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--map-key", choices=("patch_cam", "class_to_patch", "final_cam", "class_ownership"),
        default="final_cam")
    return parser.parse_args()


def overlap_score(first, second, region):
    numerator = np.minimum(first[region], second[region]).sum()
    denominator = np.maximum(first[region], second[region]).sum()
    return float(numerator / (denominator + EPS))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = []
    for path in sorted((args.dump_dir / "samples").glob("*.npz")):
        with np.load(path) as sample:
            if args.map_key not in sample:
                continue
            maps = sample[args.map_key].astype(np.float64)
            label = sample["label"].astype(bool)
            image_id = str(sample["image_id"])
        grid_size = maps.shape[-2:]
        target = load_segmentation(args.voc_root, image_id, grid_size)
        valid = target != 255
        background = (target == 0) & valid
        foreground = (target > 0) & valid
        active_ids = np.flatnonzero(label)
        active_maps = np.maximum(maps[active_ids], 0.0)
        total_mass = active_maps.sum()
        blr = float(active_maps[:, background].sum() / (total_mass + EPS))
        normalized = active_maps / (active_maps.sum(axis=(1, 2), keepdims=True) + EPS)
        cbl = float(normalized[:, background].sum(axis=1).mean())
        bg_collisions = []
        fg_collisions = []
        for first, second in combinations(active_maps, 2):
            bg_collisions.append(overlap_score(first, second, background))
            fg_collisions.append(overlap_score(first, second, foreground))
        rows.append({
            "image_id": image_id,
            "active_classes": len(active_ids),
            "blr": blr,
            "cbl": cbl,
            "ccs_bg": np.mean(bg_collisions) if bg_collisions else np.nan,
            "ccs_fg": np.mean(fg_collisions) if fg_collisions else np.nan,
        })
    if not rows:
        raise RuntimeError(f"No {args.map_key} maps found in {args.dump_dir}")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "map_key": args.map_key,
        "num_images": len(rows),
        "num_multilabel_images": sum(row["active_classes"] > 1 for row in rows),
        "blr": finite_mean([row["blr"] for row in rows]),
        "cbl": finite_mean([row["cbl"] for row in rows]),
        "ccs_bg": finite_mean([row["ccs_bg"] for row in rows]),
        "ccs_fg": finite_mean([row["ccs_fg"] for row in rows]),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
