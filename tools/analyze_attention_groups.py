#!/usr/bin/env python3
"""Measure MCTformer+ token-group attention mass across input resolutions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from sklearn.metrics import average_precision_score
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import MCTformerPlusCam
from models.tgca import SUPPORTED_MODES, TokenGroupNormalizer


GROUP_NAMES = {0: "class", 1: "patch"}
MASS_FIELDS = [
    "run_id", "dataset", "split", "image_id", "resolution", "patch_count",
    "class_count", "layer", "head", "query_group", "key_group", "mean_mass",
    "std_mass", "mean_entropy", "max_row_sum_error", "mean_group_log_evidence",
]
STABILITY_FIELDS = [
    "run_id", "image_id", "layer", "head", "query_group", "key_group",
    "mass_variance", "mass_slope_vs_log_patch_count",
]


def parse_resolutions(value):
    resolutions = tuple(int(item) for item in value.split(","))
    if not resolutions or any(item <= 0 or item % 16 for item in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive multiples of 16")
    return resolutions


def build_transform(resolution):
    resize_size = int((256 / 224) * resolution)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def bootstrap_mean_interval(values, seed, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 500
    for begin in range(0, resamples, batch_size):
        end = min(begin + batch_size, resamples)
        indices = generator.integers(0, len(values), size=(end - begin, len(values)))
        means[begin:end] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_mean_ci95": [float(lower), float(upper)],
        "num_images": int(len(values)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


class AttentionCollector:
    def __init__(self, writer, run_id, mode):
        self.writer = writer
        self.run_id = run_id
        self.mode = mode
        self.context = None
        self.handles = []
        self.max_row_sum_error = 0.0

    def attach(self, model):
        layer_index = 0
        for module in model.modules():
            if isinstance(module, TokenGroupNormalizer):
                self.handles.append(
                    module.register_forward_hook(self._make_hook(layer_index))
                )
                layer_index += 1
        if layer_index != 12:
            raise RuntimeError(f"Expected 12 attention normalizers, found {layer_index}")

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _make_hook(self, layer):
        def hook(module, inputs, attention):
            if self.context is None:
                raise RuntimeError("Attention diagnostics context is not set")
            logits, key_group_ids, query_group_ids = inputs[:3]
            if key_group_ids.ndim != 1 or query_group_ids.ndim != 1:
                raise RuntimeError("MCTformer+ diagnostics expect shared 1-D group IDs")
            expected_sum = 2.0 if self.mode == "split_11" else 1.0
            for query_group in (0, 1):
                query_mask = query_group_ids == query_group
                query_attention = attention[..., query_mask, :].float()
                query_logits = logits[..., query_mask, :].float()
                entropy = -(
                    query_attention
                    * query_attention.clamp_min(torch.finfo(torch.float32).tiny).log()
                ).sum(dim=-1).mean(dim=(0, 2))
                row_error = (
                    query_attention.sum(dim=-1) - expected_sum
                ).abs().amax(dim=(0, 2))
                for key_group in (0, 1):
                    key_mask = key_group_ids == key_group
                    mass = query_attention[..., key_mask].sum(dim=-1)
                    mean_mass = mass.mean(dim=(0, 2))
                    std_mass = mass.reshape(mass.shape[1], -1).std(dim=-1, unbiased=False)
                    group_log_evidence = (
                        torch.logsumexp(query_logits[..., key_mask], dim=-1)
                        - math.log(int(key_mask.sum()))
                    ).mean(dim=(0, 2))
                    arrays = torch.stack(
                        (mean_mass, std_mass, entropy, row_error, group_log_evidence),
                        dim=-1,
                    ).detach().cpu().numpy()
                    self.max_row_sum_error = max(
                        self.max_row_sum_error, float(arrays[:, 3].max())
                    )
                    for head, values in enumerate(arrays):
                        self.writer.writerow(
                            {
                                **self.context,
                                "layer": layer,
                                "head": head,
                                "query_group": GROUP_NAMES[query_group],
                                "key_group": GROUP_NAMES[key_group],
                                "mean_mass": f"{values[0]:.10g}",
                                "std_mass": f"{values[1]:.10g}",
                                "mean_entropy": f"{values[2]:.10g}",
                                "max_row_sum_error": f"{values[3]:.10g}",
                                "mean_group_log_evidence": f"{values[4]:.10g}",
                            }
                        )
        return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default="vanilla")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolutions", type=parse_resolutions, default=(224, 320, 448, 512))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / "attention_group_mass.csv",
        args.output_dir / "scale_stability.csv",
        args.output_dir / "metrics.json",
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite existing diagnostic outputs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("attention_normalization") if isinstance(checkpoint, dict) else None
    if checkpoint_config is not None and checkpoint_config.get("mode") != args.mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint_config.get('mode')!r} does not match requested {args.mode!r}"
        )
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = MCTformerPlusCam(
        num_classes=20,
        input_size=448,
        attention_normalization=args.mode,
        attention_gamma=args.gamma,
    )
    incompatibility = model.load_state_dict(state_dict, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(str(incompatibility))
    device = torch.device(args.device)
    model.to(device).eval()

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines() if line.strip()]
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]
    labels = np.load(args.voc_root / "ImageLabel" / "cls_labels.npy", allow_pickle=True).item()
    image_transforms = {resolution: build_transform(resolution) for resolution in args.resolutions}

    stability_values = defaultdict(lambda: {"variance": [], "slope": []})
    directional_image_slopes = defaultdict(list)
    classification_ap = defaultdict(list)
    start_time = time.time()
    mass_path, stability_path, metrics_path = output_paths
    with mass_path.open("w", newline="", encoding="utf-8") as mass_stream, stability_path.open(
        "w", newline="", encoding="utf-8"
    ) as stability_stream:
        mass_writer = csv.DictWriter(mass_stream, fieldnames=MASS_FIELDS)
        stability_writer = csv.DictWriter(stability_stream, fieldnames=STABILITY_FIELDS)
        mass_writer.writeheader()
        stability_writer.writeheader()
        collector = AttentionCollector(mass_writer, args.run_id, args.mode)
        collector.attach(model)
        try:
            with torch.inference_mode():
                for image_index, image_id in enumerate(image_ids):
                    image = PIL.Image.open(
                        args.voc_root / "JPEGImages" / f"{image_id}.jpg"
                    ).convert("RGB")
                    per_image_mass = defaultdict(list)
                    for resolution in args.resolutions:
                        tensor = image_transforms[resolution](image).unsqueeze(0).to(device)
                        patch_count = (resolution // 16) ** 2
                        collector.context = {
                            "run_id": args.run_id,
                            "dataset": "PASCAL VOC 2012",
                            "split": "train",
                            "image_id": image_id,
                            "resolution": resolution,
                            "patch_count": patch_count,
                            "class_count": 20,
                        }
                        class_tokens, _, weights, _ = model.forward_features(tensor)
                        class_logits = class_tokens.mean(dim=-1)[0].float().cpu().numpy()
                        classification_ap[resolution].append(
                            average_precision_score(labels[image_id], class_logits)
                        )
                        mass_stream.flush()
                        for layer, attention in enumerate(weights):
                            attention = attention.float()
                            query_groups = (
                                (0, slice(0, 20)),
                                (1, slice(20, None)),
                            )
                            key_groups = (
                                (0, slice(0, 20)),
                                (1, slice(20, None)),
                            )
                            for query_group, query_slice in query_groups:
                                for key_group, key_slice in key_groups:
                                    mass = attention[..., query_slice, key_slice].sum(-1).mean((0, 2))
                                    for head, value in enumerate(mass.cpu().tolist()):
                                        per_image_mass[(layer, head, query_group, key_group)].append(
                                            (math.log(patch_count), value)
                                        )
                    for (layer, head, query_group, key_group), values in per_image_mass.items():
                        x = np.asarray([value[0] for value in values], dtype=np.float64)
                        y = np.asarray([value[1] for value in values], dtype=np.float64)
                        variance = float(np.var(y))
                        slope = float(np.polyfit(x, y, deg=1)[0])
                        summary_key = (layer, head, query_group, key_group)
                        stability_values[summary_key]["variance"].append(variance)
                        stability_values[summary_key]["slope"].append(slope)
                        stability_writer.writerow(
                            {
                                "run_id": args.run_id,
                                "image_id": image_id,
                                "layer": layer,
                                "head": head,
                                "query_group": GROUP_NAMES[query_group],
                                "key_group": GROUP_NAMES[key_group],
                                "mass_variance": f"{variance:.10g}",
                                "mass_slope_vs_log_patch_count": f"{slope:.10g}",
                            }
                        )
                    for query_group in (0, 1):
                        for key_group in (0, 1):
                            all_direction_slopes = [
                                float(np.polyfit(
                                    np.asarray([item[0] for item in values]),
                                    np.asarray([item[1] for item in values]),
                                    deg=1,
                                )[0])
                                for (layer, head, q_group, k_group), values in per_image_mass.items()
                                if q_group == query_group and k_group == key_group
                            ]
                            last_direction_slopes = [
                                float(np.polyfit(
                                    np.asarray([item[0] for item in values]),
                                    np.asarray([item[1] for item in values]),
                                    deg=1,
                                )[0])
                                for (layer, head, q_group, k_group), values in per_image_mass.items()
                                if layer >= 9 and q_group == query_group and k_group == key_group
                            ]
                            directional_image_slopes[(query_group, key_group, "all_layers")].append(
                                float(np.mean(all_direction_slopes))
                            )
                            directional_image_slopes[(query_group, key_group, "last_3_layers")].append(
                                float(np.mean(last_direction_slopes))
                            )
                    if (image_index + 1) % 10 == 0 or image_index + 1 == len(image_ids):
                        print(
                            f"images={image_index + 1}/{len(image_ids)} "
                            f"elapsed_seconds={time.time() - start_time:.1f}",
                            flush=True,
                        )
        finally:
            collector.context = None
            collector.close()

    summary_rows = []
    all_variances = []
    all_slopes = []
    for (layer, head, query_group, key_group), values in sorted(stability_values.items()):
        variance_array = np.asarray(values["variance"])
        slope_array = np.asarray(values["slope"])
        all_variances.extend(variance_array.tolist())
        all_slopes.extend(slope_array.tolist())
        summary_rows.append(
            {
                "layer": layer,
                "head": head,
                "query_group": GROUP_NAMES[query_group],
                "key_group": GROUP_NAMES[key_group],
                "mean_mass_variance": float(variance_array.mean()),
                "median_mass_variance": float(np.median(variance_array)),
                "mean_mass_slope": float(slope_array.mean()),
                "median_mass_slope": float(np.median(slope_array)),
            }
        )
    with (args.output_dir / "scale_stability_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metrics = {
        "run_id": args.run_id,
        "host": "MCTformer+",
        "dataset": "PASCAL VOC 2012 train",
        "normalization": args.mode,
        "gamma": args.gamma,
        "checkpoint": str(args.checkpoint),
        "num_images": len(image_ids),
        "resolutions": list(args.resolutions),
        "patch_counts": [(resolution // 16) ** 2 for resolution in args.resolutions],
        "classification_mean_image_ap_percent": {
            str(resolution): 100.0 * float(np.mean(values))
            for resolution, values in classification_ap.items()
        },
        "mean_group_mass_variance": float(np.mean(all_variances)),
        "median_group_mass_variance": float(np.median(all_variances)),
        "mean_group_mass_slope_vs_log_patch_count": float(np.mean(all_slopes)),
        "maximum_row_sum_error": collector.max_row_sum_error,
        "directional_mass_slope": {
            f"{GROUP_NAMES[query_group]}_query_to_{GROUP_NAMES[key_group]}_key_{scope}":
                bootstrap_mean_interval(
                    values,
                    seed=2027 + query_group * 100 + key_group * 10 + (scope == "last_3_layers"),
                )
            for (query_group, key_group, scope), values in sorted(directional_image_slopes.items())
        },
        "elapsed_seconds": time.time() - start_time,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
