#!/usr/bin/env python3
"""Small inference-only tau/beta screen on a fixed VOC subset."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from analysis._common import image_tensor, load_cam_model, load_labels, load_segmentation, read_ids


EPS = 1e-8


def comma_floats(value):
    result = tuple(float(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive comma-separated floats")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("e4", "e5", "e6"), default="e6")
    parser.add_argument(
        "--model",
        choices=("mctformerplus_tiny", "mctformerplus", "mctformerplus_base"),
        default="mctformerplus",
    )
    parser.add_argument("--taus", type=comma_floats, default=(0.35, 0.5, 0.75))
    parser.add_argument("--betas", type=comma_floats, default=(0.25, 0.5, 0.75))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def overlap(first, second, region):
    return np.minimum(first[region], second[region]).sum() / (
        np.maximum(first[region], second[region]).sum() + EPS)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    model, _ = load_cam_model(
        args.checkpoint,
        args.input_size,
        device,
        args.variant,
        model_name=args.model,
    )
    labels = load_labels(args.voc_root)
    image_ids = read_ids(args.id_list)[:args.max_images]
    settings = [(tau, beta) for tau in args.taus for beta in args.betas]
    accumulators = {
        setting: {"confusion": np.zeros((21, 21), dtype=np.int64), "cbl": [],
                  "ccs_bg": [], "bg_mean": [], "bg_fraction": []}
        for setting in settings
    }
    with torch.no_grad():
        for image_index, image_id in enumerate(image_ids):
            _, inputs = image_tensor(
                args.voc_root / "JPEGImages" / f"{image_id}.jpg",
                args.input_size, device)
            label = labels[image_id].astype(bool)
            active_ids = np.flatnonzero(label)
            target_tensor = torch.from_numpy(labels[image_id]).to(device).unsqueeze(0)
            for tau, beta in settings:
                model.bcss_final_tau = tau
                model.bcss_final_beta = beta
                model.set_bcss_epoch(8)
                output = model(inputs, active_labels=target_tensor, return_diagnostics=True)
                maps = output["final_cam"][0, active_ids].float().cpu().numpy()
                minimum = maps.min(axis=(1, 2), keepdims=True)
                maximum = maps.max(axis=(1, 2), keepdims=True)
                normalized = (maps - minimum) / (maximum - minimum + EPS)
                target = load_segmentation(args.voc_root, image_id, maps.shape[-2:])
                valid = target != 255
                background = (target == 0) & valid
                foreground = (target > 0) & valid
                class_choice = normalized.argmax(axis=0)
                score = normalized.max(axis=0)
                prediction = np.zeros(score.shape, dtype=np.int64)
                foreground_prediction = score > args.threshold
                prediction[foreground_prediction] = active_ids[class_choice[foreground_prediction]] + 1
                accumulator = accumulators[(tau, beta)]
                accumulator["confusion"] += np.bincount(
                    21 * target[valid] + prediction[valid], minlength=21 ** 2
                ).reshape(21, 21)
                positive_maps = np.maximum(maps, 0.0)
                spatial = positive_maps / (
                    positive_maps.sum(axis=(1, 2), keepdims=True) + EPS)
                accumulator["cbl"].append(spatial[:, background].sum(axis=1).mean())
                accumulator["ccs_bg"].extend(
                    overlap(first, second, background)
                    for first, second in combinations(positive_maps, 2))
                background_ownership = output["background_ownership"][0].float().cpu().numpy()
                accumulator["bg_mean"].append(background_ownership[valid].mean())
                accumulator["bg_fraction"].append(
                    (background_ownership[valid] >= 0.5).mean())
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(image_ids):
                print(f"evaluated {image_index + 1}/{len(image_ids)}")
    rows = []
    for (tau, beta), accumulator in accumulators.items():
        confusion = accumulator["confusion"]
        intersection = np.diag(confusion)
        union = confusion.sum(0) + confusion.sum(1) - intersection
        pred_fg = confusion[:, 1:].sum()
        target_fg = confusion[1:, :].sum()
        semantic_tp = intersection[1:].sum()
        rows.append({
            "tau": tau,
            "beta": beta,
            "num_images": len(image_ids),
            "single_scale_grid_miou_percent": 100.0 * np.nanmean(intersection / union),
            "semantic_precision_percent": 100.0 * semantic_tp / (pred_fg + EPS),
            "semantic_recall_percent": 100.0 * semantic_tp / (target_fg + EPS),
            "cbl": float(np.mean(accumulator["cbl"])),
            "ccs_bg": float(np.mean(accumulator["ccs_bg"])) if accumulator["ccs_bg"] else None,
            "background_mean": float(np.mean(accumulator["bg_mean"])),
            "background_fraction_at_0_5": float(np.mean(accumulator["bg_fraction"])),
        })
    rows.sort(key=lambda row: row["single_scale_grid_miou_percent"], reverse=True)
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "model": args.model,
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "selection_scope": "inference-only single-scale diagnostic; not a final CAM result",
        "rows": rows,
        "best_by_diagnostic_miou": rows[0],
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
