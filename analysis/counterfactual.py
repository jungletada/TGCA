#!/usr/bin/env python3
"""VOC context-only and object-only counterfactual evaluation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageFilter
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from analysis._common import finite_mean, load_cam_model, load_labels, load_segmentation, read_ids
from models.adapter_modules import resize_input_minbound


RGB_MEAN = np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32) * 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("e0", "e1", "e2", "e4", "e5", "e6"))
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--replacement", choices=("mean", "blur"), default="mean")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--save-images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_tensor(array, min_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    tensor = transform(PIL.Image.fromarray(array)).unsqueeze(0)
    return resize_input_minbound(tensor, min_size=min_size).to(device)


def replacement_image(image, mode):
    if mode == "mean":
        return np.broadcast_to(RGB_MEAN, image.shape).astype(np.uint8)
    return np.asarray(
        PIL.Image.fromarray(image).filter(PIL.ImageFilter.GaussianBlur(radius=15)),
        dtype=np.uint8,
    )


def normalize_map(values):
    minimum = values.min()
    return (values - minimum) / (values.max() - minimum + 1e-8)


def infer(model, array, target, input_size, device):
    inputs = to_tensor(array, input_size, device)
    active = target.expand(inputs.shape[0], -1)
    diagnostics = model(inputs, active_labels=active, return_diagnostics=True)
    probabilities = torch.sigmoid(diagnostics["class_logits"])[0]
    cams = diagnostics["final_cam"][0].float().cpu().numpy()
    if "background_ownership" in diagnostics:
        background = diagnostics["background_ownership"][0].float().cpu().numpy()
    elif "background_attention" in diagnostics:
        background = diagnostics["background_attention"][0].mean(0).float().cpu().numpy()
    elif "register_to_patch" in diagnostics:
        register = diagnostics["register_to_patch"][-3:, 0].mean(dim=(0, 1))
        background = register.reshape(cams.shape[-2:]).float().cpu().numpy()
    else:
        active_ids = target[0].bool().cpu().numpy()
        normalized = np.stack([normalize_map(values) for values in cams[active_ids]])
        background = 1.0 - normalized.max(axis=0)
    return probabilities.detach().cpu().numpy(), cams, background


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    image_dir = args.output_dir / "images"
    map_dir = args.output_dir / "maps"
    map_dir.mkdir()
    if args.save_images:
        image_dir.mkdir()
    device = torch.device(args.device)
    model, config = load_cam_model(
        args.checkpoint, args.input_size, device, args.variant)
    labels = load_labels(args.voc_root)
    image_ids = read_ids(args.id_list)
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]
    rows = []
    saved = 0
    with torch.no_grad():
        for index, image_id in enumerate(image_ids):
            image = np.asarray(PIL.Image.open(
                args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB"))
            mask = load_segmentation(args.voc_root, image_id, image.shape[:2])
            target = torch.from_numpy(labels[image_id]).float().to(device).unsqueeze(0)
            original_score, original_cams, original_bg = infer(
                model, image, target, args.input_size, device)
            fill = replacement_image(image, args.replacement)
            for class_index in np.flatnonzero(labels[image_id]):
                object_mask = mask == class_index + 1
                if not object_mask.any():
                    continue
                context = image.copy()
                context[object_mask] = fill[object_mask]
                object_only = fill.copy()
                object_only[object_mask] = image[object_mask]
                context_score, context_cams, context_bg = infer(
                    model, context, target, args.input_size, device)
                object_score, object_cams, object_bg = infer(
                    model, object_only, target, args.input_size, device)
                denominator = original_score[class_index] + 1e-8
                grid_mask = load_segmentation(
                    args.voc_root, image_id, original_cams.shape[-2:]) == class_index + 1
                original_cam = normalize_map(original_cams[class_index])
                context_cam = normalize_map(context_cams[class_index])
                object_cam = normalize_map(object_cams[class_index])
                rows.append({
                    "image_id": image_id,
                    "class_index": int(class_index),
                    "original_score": float(original_score[class_index]),
                    "context_only_score": float(context_score[class_index]),
                    "object_only_score": float(object_score[class_index]),
                    "crs": float(context_score[class_index] / denominator),
                    "ors": float(object_score[class_index] / denominator),
                    "original_object_cam_mean": float(original_cam[grid_mask].mean()),
                    "context_object_cam_mean": float(context_cam[grid_mask].mean()),
                    "object_only_object_cam_mean": float(object_cam[grid_mask].mean()),
                    "context_outside_object_cam_mean": float(context_cam[~grid_mask].mean()),
                    "original_background_mean": float(original_bg.mean()),
                    "context_background_mean": float(context_bg.mean()),
                    "object_background_mean": float(object_bg.mean()),
                })
                np.savez_compressed(
                    map_dir / f"{image_id}_c{class_index:02d}.npz",
                    image_id=np.asarray(image_id),
                    class_index=np.asarray(class_index, dtype=np.int16),
                    object_mask=grid_mask.astype(np.uint8),
                    original_cam=original_cam.astype(np.float32),
                    context_cam=context_cam.astype(np.float32),
                    object_only_cam=object_cam.astype(np.float32),
                    original_background=original_bg.astype(np.float32),
                    context_background=context_bg.astype(np.float32),
                    object_only_background=object_bg.astype(np.float32),
                )
                if saved < args.save_images:
                    stem = f"{image_id}_c{class_index:02d}"
                    PIL.Image.fromarray(context).save(image_dir / f"{stem}_context.jpg")
                    PIL.Image.fromarray(object_only).save(image_dir / f"{stem}_object.jpg")
                    saved += 1
            if (index + 1) % 25 == 0:
                print(f"evaluated {index + 1}/{len(image_ids)}")
    if not rows:
        raise RuntimeError("No class-specific VOC masks were available")
    with (args.output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "variant": config.get("variant", "e0"),
        "replacement": args.replacement,
        "num_image_classes": len(rows),
        "crs": finite_mean([row["crs"] for row in rows]),
        "ors": finite_mean([row["ors"] for row in rows]),
        "context_object_cam_mean": finite_mean(
            [row["context_object_cam_mean"] for row in rows]),
        "object_only_object_cam_mean": finite_mean(
            [row["object_only_object_cam_mean"] for row in rows]),
        "context_outside_object_cam_mean": finite_mean(
            [row["context_outside_object_cam_mean"] for row in rows]),
        "context_background_mean": finite_mean(
            [row["context_background_mean"] for row in rows]),
        "object_background_mean": finite_mean(
            [row["object_background_mean"] for row in rows]),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
