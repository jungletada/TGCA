#!/usr/bin/env python3
"""Measure class-token/patch-token separation around each pre-attention norm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import MCTformerPlusCam
from models.vit import TOKEN_ROLE_SPECIALIZATIONS


FIELDS = (
    "image_id", "layer", "stage", "class_patch_cosine",
    "patch_patch_cosine", "class_token_norm", "patch_token_norm",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token-role-specialization",
        choices=TOKEN_ROLE_SPECIALIZATIONS,
        required=True,
    )
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_statistics(tokens, class_count):
    normalized = F.normalize(tokens.float(), dim=-1)
    class_tokens = normalized[:, :class_count]
    patch_tokens = normalized[:, class_count:]
    class_patch = (
        class_tokens.sum(dim=1) * patch_tokens.sum(dim=1)
    ).sum(dim=-1) / (class_tokens.shape[1] * patch_tokens.shape[1])

    patch_sum = patch_tokens.sum(dim=1)
    patch_count = patch_tokens.shape[1]
    if patch_count > 1:
        patch_patch = (
            patch_sum.square().sum(dim=-1) - patch_count
        ) / (patch_count * (patch_count - 1))
    else:
        patch_patch = torch.full_like(class_patch, float("nan"))

    raw_norm = tokens.float().norm(dim=-1)
    return {
        "class_patch_cosine": class_patch.mean().item(),
        "patch_patch_cosine": patch_patch.mean().item(),
        "class_token_norm": raw_norm[:, :class_count].mean().item(),
        "patch_token_norm": raw_norm[:, class_count:].mean().item(),
    }


class RoleCollector:
    def __init__(self, writer, class_count):
        self.writer = writer
        self.class_count = class_count
        self.image_id = None
        self.handles = []
        self.values = defaultdict(list)

    def attach(self, model):
        for layer, block in enumerate(model.blocks):
            self.handles.append(
                block.register_forward_pre_hook(
                    self._make_hook(layer, "pre_norm1")
                )
            )
            self.handles.append(
                block.attn.register_forward_pre_hook(
                    self._make_hook(layer, "post_norm1")
                )
            )

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _make_hook(self, layer, stage):
        def hook(_module, inputs):
            if self.image_id is None:
                raise RuntimeError("Token-role diagnostic image context is not set")
            stats = token_statistics(inputs[0], self.class_count)
            row = {"image_id": self.image_id, "layer": layer, "stage": stage, **stats}
            self.writer.writerow(row)
            for name, value in stats.items():
                self.values[(layer, stage, name)].append(value)

        return hook


def build_transform(resolution):
    resize_size = int((256 / 224) * resolution)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def main():
    args = parse_args()
    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    if args.max_images <= 0:
        raise ValueError("max-images must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    role_config = checkpoint.get("token_role_specialization", {})
    checkpoint_role = role_config.get("mode", "shared")
    if checkpoint_role != args.token_role_specialization:
        raise ValueError(
            f"Checkpoint role specialization {checkpoint_role!r} does not match "
            f"{args.token_role_specialization!r}"
        )
    state_dict = checkpoint.get("model", checkpoint)
    model = MCTformerPlusCam(
        num_classes=20,
        input_size=448,
        attention_normalization="vanilla",
        token_role_specialization=args.token_role_specialization,
    )
    incompatibility = model.load_state_dict(state_dict, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(str(incompatibility))

    device = torch.device(args.device)
    model.to(device).eval()
    image_ids = [
        line.strip() for line in args.id_list.read_text().splitlines() if line.strip()
    ][:args.max_images]
    labels = np.load(
        args.voc_root / "ImageLabel" / "cls_labels.npy", allow_pickle=True
    ).item()
    transform = build_transform(args.resolution)
    classification_ap = []
    csv_path = args.output_dir / "token_role_cosine.csv"
    started = time.time()

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        collector = RoleCollector(writer, class_count=20)
        collector.attach(model)
        try:
            with torch.inference_mode():
                for index, image_id in enumerate(image_ids):
                    image = PIL.Image.open(
                        args.voc_root / "JPEGImages" / f"{image_id}.jpg"
                    ).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)
                    collector.image_id = image_id
                    class_tokens, _, _, _ = model.forward_features(tensor)
                    classification_ap.append(
                        average_precision_score(
                            labels[image_id],
                            class_tokens.mean(dim=-1)[0].float().cpu().numpy(),
                        )
                    )
                    if (index + 1) % 25 == 0 or index + 1 == len(image_ids):
                        print(f"images={index + 1}/{len(image_ids)}", flush=True)
        finally:
            collector.close()

    aggregates = []
    for (layer, stage, name), values in sorted(collector.values.items()):
        values = np.asarray(values, dtype=np.float64)
        aggregates.append(
            {
                "layer": layer,
                "stage": stage,
                "metric": name,
                "mean": float(values.mean()),
                "std_across_images": float(values.std()),
                "num_images": int(len(values)),
            }
        )

    post_minus_pre = []
    for layer in range(len(model.blocks)):
        for name in (
            "class_patch_cosine", "patch_patch_cosine",
            "class_token_norm", "patch_token_norm",
        ):
            before = np.asarray(
                collector.values[(layer, "pre_norm1", name)], dtype=np.float64
            )
            after = np.asarray(
                collector.values[(layer, "post_norm1", name)], dtype=np.float64
            )
            delta = after - before
            post_minus_pre.append(
                {
                    "layer": layer,
                    "metric": name,
                    "mean_delta": float(delta.mean()),
                    "std_delta_across_images": float(delta.std()),
                }
            )

    metrics = {
        "host": "MCTformer+",
        "dataset": "PASCAL VOC 2012 train",
        "token_role_specialization": args.token_role_specialization,
        "role_adaptation": "first 20 class tokens versus all spatial patch tokens",
        "resolution": args.resolution,
        "num_images": len(image_ids),
        "classification_map_percent": 100.0 * float(np.mean(classification_ap)),
        "elapsed_seconds": time.time() - started,
        "aggregates": aggregates,
        "post_norm1_minus_pre_norm1": post_minus_pre,
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "id_list": str(args.id_list.resolve()),
            "id_list_sha256": sha256(args.id_list),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
