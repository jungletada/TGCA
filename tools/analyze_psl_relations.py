#!/usr/bin/env python3
"""Measure Phase 2 semantic read/write behavior on a fixed VOC subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._common import load_labels, segmentation_path
from models.mctformer_plus import (
    build_mctformerplus,
    resolve_mctformerplus_checkpoint_variant,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("mctformerplus_tiny", "mctformerplus", "mctformerplus_base"),
        default="mctformerplus",
    )
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_transform(image, mask, resolution):
    resize_size = int((256 / 224) * resolution)
    image = transform_functional.resize(
        image, resize_size, interpolation=transforms.InterpolationMode.BICUBIC)
    mask = transform_functional.resize(
        mask, resize_size, interpolation=transforms.InterpolationMode.NEAREST)
    image = transform_functional.center_crop(image, (resolution, resolution))
    mask = transform_functional.center_crop(mask, (resolution, resolution))
    tensor = transform_functional.normalize(
        transform_functional.to_tensor(image),
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
    )
    grid = resolution // 16
    mask = transform_functional.resize(
        mask, (grid, grid), interpolation=transforms.InterpolationMode.NEAREST)
    return tensor.unsqueeze(0), np.asarray(mask, dtype=np.uint8).reshape(-1)


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def confusion_summary(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    target = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = target + predicted - true_positive
    iou = np.divide(
        true_positive, union, out=np.full_like(true_positive, np.nan),
        where=union > 0)
    return {
        "semantic_accuracy": float(true_positive.sum() / confusion.sum()),
        "foreground_accuracy": float(true_positive[1:].sum() / target[1:].sum()),
        "background_accuracy": float(true_positive[0] / target[0]),
        "mean_iou": float(np.nanmean(iou)),
        "foreground_mean_iou": float(np.nanmean(iou[1:])),
        "per_class_iou": [float(value) if np.isfinite(value) else None for value in iou],
    }


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("psl", {"variant": "baseline"})
    if config.get("variant") == "baseline":
        raise ValueError("Phase 2 relation diagnostics require a PSL checkpoint")
    resolution = resolve_mctformerplus_checkpoint_variant(checkpoint, args.model)
    model = build_mctformerplus(
        resolution["variant"],
        cam=True,
        num_classes=20,
        input_size=args.input_size,
        attention_normalization="vanilla",
        bcss_variant="e0",
        psl_variant=config["variant"],
        psl_interaction_layers=tuple(config["interaction_layers_zero_based"]),
        psl_relation_dim=config["relation_dim"],
        psl_num_background_latents=config["num_background_latents"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines()
                 if line.strip()]
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    if not image_ids:
        raise ValueError("No images selected")
    labels = load_labels(args.voc_root)
    layers = tuple(config["interaction_layers_zero_based"])
    confusions = {
        layer: np.zeros((21, 21), dtype=np.int64) for layer in layers}
    layer_values = defaultdict(lambda: defaultdict(list))
    rows = []
    classification_logits = []
    classification_targets = []
    token_gaps = []
    background_gaps = []
    start = time.time()

    with torch.inference_mode():
        for image_index, image_id in enumerate(image_ids):
            image = PIL.Image.open(
                args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
            mask = PIL.Image.open(segmentation_path(args.voc_root, image_id))
            tensor, target = paired_transform(image, mask, args.input_size)
            classes, patches, _, _, auxiliary = model.forward_features(
                tensor.to(device), return_aux=True)
            classification_logits.append(classes.mean(-1)[0].float().cpu().numpy())
            classification_targets.append(
                np.asarray(labels[image_id], dtype=np.uint8))
            valid = target != 255

            for relation_item in auxiliary["psl_relations"]:
                layer = int(relation_item["layer"])
                read = relation_item["read_attention"][0].float()
                write = relation_item["write_attention"][0].float()
                relation = relation_item["relation"][0].float()
                prediction_index = write.argmax(dim=-1).cpu().numpy()
                prediction = np.where(prediction_index == 20, 0, prediction_index + 1)
                encoded = target[valid] * 21 + prediction[valid]
                confusions[layer] += np.bincount(
                    encoded, minlength=21 ** 2).reshape(21, 21)
                read_entropy = -(
                    read * read.clamp_min(1e-30).log()).sum(-1) / math.log(read.shape[-1])
                write_entropy = -(
                    write * write.clamp_min(1e-30).log()).sum(-1) / math.log(write.shape[-1])
                values = {
                    "read_foreground_entropy": float(read_entropy[:20].mean()),
                    "read_background_entropy": float(read_entropy[20]),
                    "write_entropy": float(write_entropy.mean()),
                    "write_background_mass": float(write[:, 20].mean()),
                    "relation_mean": float(relation.mean()),
                    "relation_std": float(relation.std(unbiased=False)),
                }
                for key, value in values.items():
                    layer_values[layer][key].append(value)
                rows.append({
                    "image_id": image_id,
                    "layer": layer,
                    **values,
                })

            patch_tensor = patches[0].float()
            semantic_tensor = auxiliary["semantic_latents"][0].float()
            image_gaps = []
            for class_index in range(20):
                class_mask = torch.from_numpy(target == class_index + 1).to(device)
                if bool(class_mask.any()):
                    centroid = patch_tensor[class_mask].mean(dim=0)
                    gap = 1.0 - F.cosine_similarity(
                        semantic_tensor[class_index], centroid, dim=0)
                    image_gaps.append(float(gap))
            if image_gaps:
                token_gaps.append(float(np.mean(image_gaps)))
            background_mask = torch.from_numpy(target == 0).to(device)
            if bool(background_mask.any()):
                centroid = patch_tensor[background_mask].mean(dim=0)
                background_gaps.append(float(1.0 - F.cosine_similarity(
                    semantic_tensor[20], centroid, dim=0)))
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(image_ids):
                print(
                    f"images={image_index + 1}/{len(image_ids)} "
                    f"elapsed_seconds={time.time() - start:.1f}", flush=True)

    classification_logits = np.stack(classification_logits)
    classification_targets = np.stack(classification_targets)
    class_ap = []
    for class_index in range(20):
        target = classification_targets[:, class_index]
        class_ap.append(
            average_precision_score(target, classification_logits[:, class_index])
            if len(np.unique(target)) == 2 else math.nan
        )

    layer_metrics = []
    for layer in layers:
        layer_metrics.append({
            "layer": layer,
            **confusion_summary(confusions[layer]),
            **{
                key: finite_mean(values)
                for key, values in layer_values[layer].items()
            },
            "write_gate": float(
                model.semantic_interactions[str(layer)].write_gate.detach().cpu()),
        })
    metrics = {
        "phase": 2,
        "model": resolution["model_name"],
        "model_spec": checkpoint.get("model_spec"),
        "variant": config["variant"],
        "num_images": len(image_ids),
        "split": args.id_list.stem,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "id_list_sha256": sha256(args.id_list),
        "configuration": config,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        "semantic_interaction_parameters": sum(
            parameter.numel()
            for parameter in model.semantic_interactions.parameters()),
        "background_latent_parameters": model.background_semantic_latent.numel(),
        "classification_map": float(np.nanmean(class_ap)),
        "per_class_ap": [
            float(value) if math.isfinite(value) else None for value in class_ap],
        "foreground_token_gap": finite_mean(token_gaps),
        "background_token_gap": finite_mean(background_gaps),
        "layer_metrics": layer_metrics,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "per_image_layer.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for layer, confusion in confusions.items():
        np.savetxt(
            args.output_dir / f"confusion_layer_{layer}.csv",
            confusion, fmt="%d", delimiter=",")
    (args.output_dir / "completion.json").write_text(json.dumps({
        "complete": True,
        "num_images": len(image_ids),
        "elapsed_seconds": metrics["elapsed_seconds"],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "complete": True,
        "variant": config["variant"],
        "num_images": len(image_ids),
        "classification_map": metrics["classification_map"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
