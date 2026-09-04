#!/usr/bin/env python3
"""Generate MCTformer+ CAMs at explicit short-side input resolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import (
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
)
from models.tgca import SUPPORTED_MODES


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_resolutions(value):
    resolutions = tuple(int(item) for item in value.split(","))
    if not resolutions or len(set(resolutions)) != len(resolutions):
        raise argparse.ArgumentTypeError("resolutions must be a non-empty unique list")
    if any(value <= 0 or value % 16 for value in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive multiples of 16")
    return resolutions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model", choices=(
            "mctformerplus_tiny", "mctformerplus", "mctformerplus_base"),
        default="mctformerplus")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=parse_resolutions, default=(224, 320, 448, 512))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_short_side(image, short_side):
    width, height = image.size
    if height <= width:
        new_height = short_side
        new_width = max(1, round(width * short_side / height))
    else:
        new_width = short_side
        new_height = max(1, round(height * short_side / width))
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def normalize_cam(cam):
    minimum = cam.flatten(1).amin(dim=1).view(-1, 1, 1)
    maximum = cam.flatten(1).amax(dim=1).view(-1, 1, 1)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def infer_flip_pair(model, tensor, output_size, device):
    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        original = model(tensor)
        flipped = model(torch.flip(tensor, dims=(-1,)))
        original = F.interpolate(original, output_size, mode="bilinear", align_corners=False)[0]
        flipped = F.interpolate(flipped, output_size, mode="bilinear", align_corners=False)[0]
    return original + torch.flip(flipped, dims=(-1,))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("attention_normalization", {})
    if checkpoint_config and checkpoint_config.get("mode") != args.mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint_config.get('mode')!r} does not match {args.mode!r}"
        )
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, args.model
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = build_mctformerplus(
        resolution['variant'],
        cam=True,
        num_classes=20,
        input_size=448,
        attention_normalization=args.mode,
        attention_gamma=1.0,
    )
    incompatibility = model.load_state_dict(state_dict, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(str(incompatibility))
    model.to(device).eval()

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines() if line.strip()]
    labels = np.load(
        args.voc_root / "ImageLabel" / "cls_labels.npy", allow_pickle=True
    ).item()
    if not image_ids:
        raise ValueError("id-list contains no images")

    args.output_dir.mkdir(parents=True)
    for resolution in args.resolutions:
        (args.output_dir / str(resolution)).mkdir()
    (args.output_dir / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    start_time = time.time()
    patch_count_ranges = {str(value): [None, None] for value in args.resolutions}
    for image_index, image_id in enumerate(image_ids):
        image = Image.open(
            args.voc_root / "JPEGImages" / f"{image_id}.jpg"
        ).convert("RGB")
        output_size = (image.height, image.width)
        valid_classes = np.flatnonzero(labels[image_id]).tolist()
        if not valid_classes:
            raise ValueError(f"Image {image_id} has no positive classes")
        for resolution in args.resolutions:
            resized = resize_short_side(image, resolution)
            tensor = image_to_tensor(resized)
            patch_count = (tensor.shape[-2] // 16) * (tensor.shape[-1] // 16)
            value_range = patch_count_ranges[str(resolution)]
            value_range[0] = patch_count if value_range[0] is None else min(value_range[0], patch_count)
            value_range[1] = patch_count if value_range[1] is None else max(value_range[1], patch_count)
            cam = infer_flip_pair(model, tensor, output_size, device)
            cam = normalize_cam(cam[valid_classes]).cpu().numpy()
            cam_dict = {
                class_id: cam[index]
                for index, class_id in enumerate(valid_classes)
            }
            np.save(args.output_dir / str(resolution) / f"{image_id}.npy", cam_dict)
        if (image_index + 1) % 25 == 0 or image_index + 1 == len(image_ids):
            print(
                f"images={image_index + 1}/{len(image_ids)} "
                f"elapsed_seconds={time.time() - start_time:.1f}",
                flush=True,
            )

    manifest = {
        "host": "MCTformer+",
        "model_spec": model_spec_from_instance(model),
        "variant_resolution": resolution,
        "normalization": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "voc_root": str(args.voc_root.resolve()),
        "id_list": str(args.id_list.resolve()),
        "id_list_sha256": sha256(args.id_list),
        "num_images": len(image_ids),
        "resolutions": list(args.resolutions),
        "resolution_definition": "short image side in pixels; aspect ratio preserved",
        "flip_augmentation": True,
        "cam_aggregation": "sum original and horizontally restored flip",
        "cam_normalization": "per-image, per-class min-max after flip aggregation",
        "patch_count_min_max_by_resolution": patch_count_ranges,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
