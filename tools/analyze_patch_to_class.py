#!/usr/bin/env python3
"""Run Persistent Semantic Latent Phase 0/1 on a frozen MCTformer+ baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
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
from torchvision.transforms import functional as transform_functional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._common import load_labels, segmentation_path
from analysis.semantic_relations import (
    cam_prediction,
    class_permutation_control,
    conditional_relations,
    conservative_diagnostic_gates,
    confusion_matrix,
    confusion_summary,
    foreground_counts,
    four_region_masks,
    mutual_relation,
    present_class_relation,
    region_composition,
    semantic_prediction,
    spatial_minmax,
)
from models.mctformer_plus import MCTformerPlusCam
from models.tgca import TokenGroupNormalizer


CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
GROUPS = (("class", slice(0, 20)), ("patch", slice(20, None)))
PHASE0_FIELDS = (
    "run_id", "split", "image_id", "resolution", "patch_count", "layer",
    "query_group", "key_group", "raw_logit_mean", "raw_logit_std",
    "group_log_evidence", "post_attention_mass", "post_attention_mass_std",
    "conditional_entropy", "row_sum_error",
)
PHASE1_IMAGE_FIELDS = (
    "run_id", "split", "image_id", "layer", "num_active_classes",
    "cp_cam_miou", "cp_fg_precision", "cp_fg_recall",
    "pc_all_fg_accuracy", "pc_all_fg_miou", "pc_all_fg_precision",
    "pc_all_fg_recall", "pc_present_fg_accuracy", "pc_present_fg_miou",
    "pc_present_fg_precision", "pc_present_fg_recall", "mutual_cam_miou",
    "mutual_fg_precision", "mutual_fg_recall", "cp_pc_overlap_iou",
    "region_c_target_purity", "cp_low_target_purity_reference",
    "region_c_purity_difference", "region_c_recovery_recall",
)


def parse_int_tuple(value):
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 or item % 16 for item in values):
        raise argparse.ArgumentTypeError("resolutions must be positive multiples of 16")
    return values


def parse_float_tuple(value):
    values = tuple(float(item) for item in value.split(","))
    if not values or any(not 0 < item < 1 for item in values):
        raise argparse.ArgumentTypeError("thresholds must be in (0, 1)")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--split", default="train_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resolutions", type=parse_int_tuple, default=(224, 320, 448, 512))
    parser.add_argument("--semantic-resolution", type=int, default=448)
    parser.add_argument(
        "--confirmatory-layer", type=int, default=-1,
        help="zero-based fixed primary layer; -1 retains exploratory best-layer review",
    )
    parser.add_argument(
        "--diagnostic-head", type=int, default=-1,
        help="zero-based head reported only as a secondary diagnostic",
    )
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument(
        "--semantic-thresholds", type=parse_float_tuple,
        default=(0.05, 0.10, 0.25, 0.50),
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-dumps", type=int, default=12)
    parser.add_argument("--raw-dump-images", type=int, default=2)
    parser.add_argument("--raw-dump-resolution", type=int, default=224)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--permutation-seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.semantic_resolution not in args.resolutions:
        parser.error("semantic resolution must be included in --resolutions")
    if args.raw_dump_resolution not in args.resolutions:
        parser.error("raw dump resolution must be included in --resolutions")
    if not 0 < args.cam_threshold < 1 or not 0 < args.semantic_threshold < 1:
        parser.error("diagnostic thresholds must be in (0, 1)")
    if args.semantic_threshold not in args.semantic_thresholds:
        parser.error("the primary semantic threshold must be in --semantic-thresholds")
    if args.confirmatory_layer < -1 or args.diagnostic_head < -1:
        parser.error("confirmatory layer and diagnostic head must be -1 or non-negative")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def summary_value(confusion, key):
    value = confusion_summary(confusion)[key]
    return float(value) if value is not None else math.nan


def mean_iou_for_image(target, prediction, num_classes, valid=None):
    return summary_value(confusion_matrix(target, prediction, num_classes, valid), "mean_iou")


def precision_recall(counts):
    _, overlap, predicted, target, _, _, _ = counts
    return safe_ratio(overlap, predicted), safe_ratio(overlap, target)


def bootstrap_mean(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": [None, None], "num_images": 0}
    if resamples <= 0:
        return {"mean": float(values.mean()), "ci95": [None, None], "num_images": len(values)}
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch = 250
    for begin in range(0, resamples, batch):
        end = min(begin + batch, resamples)
        indices = generator.integers(0, len(values), size=(end - begin, len(values)))
        means[begin:end] = values[indices].mean(axis=1)
    interval = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95": [float(interval[0]), float(interval[1])],
        "num_images": int(len(values)),
    }


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        value = float(value)
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_square += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self):
        if not self.count:
            return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
        }


class AttentionCapture:
    """Capture raw normalizer inputs without changing the model forward path."""

    METRICS = (
        "raw_logit_mean", "raw_logit_std", "group_log_evidence",
        "post_attention_mass", "post_attention_mass_std", "conditional_entropy",
        "row_sum_error",
    )

    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.handles = []
        self.context = None
        self.records = []
        self.semantic_relations = []
        self.full_raw = []

    def attach(self, model):
        for layer, module in enumerate(
            item for item in model.modules() if isinstance(item, TokenGroupNormalizer)
        ):
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        if len(self.handles) != len(model.blocks):
            raise RuntimeError(
                f"Expected {len(model.blocks)} attention normalizers, found {len(self.handles)}"
            )

    def close(self):
        for handle in self.handles:
            handle.remove()

    def begin(self, capture_semantic=False, capture_full=False):
        if self.context is not None:
            raise RuntimeError("Previous attention capture was not consumed")
        self.context = {
            "capture_semantic": capture_semantic,
            "capture_full": capture_full,
        }
        self.records = []
        self.semantic_relations = []
        self.full_raw = []

    def consume(self):
        if self.context is None:
            raise RuntimeError("Attention capture was not started")
        payload = (self.records, self.semantic_relations, self.full_raw)
        self.context = None
        self.records = []
        self.semantic_relations = []
        self.full_raw = []
        return payload

    def _hook(self, layer):
        def hook(_module, inputs, attention):
            if self.context is None:
                raise RuntimeError("Attention hook fired outside a capture context")
            logits = inputs[0].detach().float()
            attention = attention.detach().float()
            if logits.shape != attention.shape or logits.shape[0] != 1:
                raise RuntimeError(f"Unexpected attention shapes: {logits.shape}, {attention.shape}")
            row_error = (attention.sum(dim=-1) - 1.0).abs().amax(dim=(0, 2))
            layer_records = []
            for query_name, query_slice in GROUPS:
                for key_name, key_slice in GROUPS:
                    raw_block = logits[..., query_slice, key_slice]
                    attention_block = attention[..., query_slice, key_slice]
                    mass = attention_block.sum(dim=-1)
                    conditional = attention_block / mass.unsqueeze(-1).clamp_min(1e-30)
                    entropy = -(
                        conditional * conditional.clamp_min(1e-30).log()
                    ).sum(dim=-1)
                    key_count = raw_block.shape[-1]
                    values = {
                        "layer": layer,
                        "query_group": query_name,
                        "key_group": key_name,
                        "raw_logit_mean": raw_block.mean(dim=(0, 2, 3)),
                        "raw_logit_std": raw_block.flatten(2).std(dim=(0, 2), unbiased=False),
                        "group_log_evidence": (
                            torch.logsumexp(raw_block, dim=-1) - math.log(key_count)
                        ).mean(dim=(0, 2)),
                        "post_attention_mass": mass.mean(dim=(0, 2)),
                        "post_attention_mass_std": mass.flatten(2).std(
                            dim=(0, 2), unbiased=False
                        ),
                        "conditional_entropy": entropy.mean(dim=(0, 2)),
                        "row_sum_error": row_error,
                    }
                    layer_records.append({
                        key: (value.cpu().numpy() if torch.is_tensor(value) else value)
                        for key, value in values.items()
                    })
            self.records.extend(layer_records)
            class_slice = slice(0, self.num_classes)
            patch_slice = slice(self.num_classes, None)
            if self.context["capture_semantic"]:
                self.semantic_relations.append({
                    "layer": layer,
                    "raw_cp": logits[0, :, class_slice, patch_slice].cpu(),
                    "raw_pc": logits[0, :, patch_slice, class_slice].cpu(),
                    "post_cp": attention[0, :, class_slice, patch_slice].cpu(),
                    "post_pc": attention[0, :, patch_slice, class_slice].cpu(),
                })
            if self.context["capture_full"]:
                self.full_raw.append({
                    "raw": logits[0].half().cpu(),
                    "post": attention[0].half().cpu(),
                })
        return hook


def paired_transform(image, mask, resolution):
    resize_size = int((256 / 224) * resolution)
    image = transform_functional.resize(
        image, resize_size, interpolation=transforms.InterpolationMode.BICUBIC
    )
    mask = transform_functional.resize(
        mask, resize_size, interpolation=transforms.InterpolationMode.NEAREST
    )
    image = transform_functional.center_crop(image, (resolution, resolution))
    mask = transform_functional.center_crop(mask, (resolution, resolution))
    tensor = transform_functional.to_tensor(image)
    tensor = transform_functional.normalize(
        tensor, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    )
    patch_size = resolution // 16
    patch_mask = transform_functional.resize(
        mask, (patch_size, patch_size), interpolation=transforms.InterpolationMode.NEAREST
    )
    return image, tensor.unsqueeze(0), np.asarray(patch_mask, dtype=np.uint8)


def image_transform(image, resolution):
    resize_size = int((256 / 224) * resolution)
    pipeline = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return pipeline(image).unsqueeze(0)


def pc_foreground_prediction(probability, active_classes, threshold):
    predicted_class = semantic_prediction(probability)
    confidence = probability[np.arange(len(predicted_class)), predicted_class]
    foreground = active_classes[predicted_class] & (confidence > threshold)
    prediction = np.zeros(len(predicted_class), dtype=np.int64)
    prediction[foreground] = predicted_class[foreground] + 1
    return prediction


def update_relation_confusions(
    accumulators, source, layer, target, prediction, num_classes, valid=None
):
    accumulators[(source, layer)] += confusion_matrix(
        target, prediction, num_classes, valid
    )


def render_preview(path, image, target, active_classes, sample_maps, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_index = int(np.flatnonzero(active_classes)[0])
    chosen_layers = tuple(index for index in (2, 5, 8, 11) if index < len(sample_maps))
    figure, axes = plt.subplots(len(chosen_layers), 5, figsize=(12, 2.6 * len(chosen_layers)))
    if len(chosen_layers) == 1:
        axes = axes[None, :]
    target_class = target == class_index + 1
    for row, layer in enumerate(chosen_layers):
        cp = sample_maps[layer]["cp"][class_index]
        pc = sample_maps[layer]["pc"][:, class_index].reshape(target.shape)
        regions = four_region_masks(cp.reshape(-1), pc.reshape(-1), threshold)
        region_map = np.zeros(target.size, dtype=np.uint8)
        for value, name in enumerate(("A", "B", "C", "D"), start=1):
            region_map[regions[name]] = value
        panels = (np.asarray(image), target_class, cp, pc, region_map.reshape(target.shape))
        titles = (
            f"layer {layer + 1}: {CLASS_NAMES[class_index]}", "GT class", "P(p|c)",
            "P(c|p)", "A/B/C/D",
        )
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, column].imshow(panel, cmap=None if column == 0 else "viridis")
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "samples").mkdir()
    (args.output_dir / "raw_dumps").mkdir()
    (args.output_dir / "previews").mkdir()

    checkpoint_hash = sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 {checkpoint_hash} does not match "
            f"{args.expected_checkpoint_sha256}"
        )
    baseline_metrics = json.loads(args.baseline_metrics.read_text())
    if baseline_metrics.get("background_threshold") != 0.45:
        raise ValueError("The trusted baseline metrics must use threshold 0.45")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    normalization = checkpoint.get("attention_normalization", {})
    bcss = checkpoint.get("bcss", {})
    if normalization.get("mode", "vanilla") != "vanilla":
        raise ValueError("Phase 0/1 requires a vanilla-softmax baseline checkpoint")
    if bcss.get("variant", "e0") != "e0":
        raise ValueError("Phase 0/1 requires the E0 baseline without background/register slots")
    model = MCTformerPlusCam(
        num_classes=20,
        input_size=args.semantic_resolution,
        attention_normalization="vanilla",
        attention_gamma=1.0,
        bcss_variant="e0",
    )
    incompatibility = model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(str(incompatibility))
    device = torch.device(args.device)
    model.to(device).eval()

    image_ids = [line.strip() for line in args.id_list.read_text().splitlines() if line.strip()]
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    if not image_ids:
        raise RuntimeError("The selected dataset split is empty")
    baseline_split = baseline_metrics.get("split")
    if (
        args.max_images == 0
        and baseline_split == args.split
        and baseline_metrics.get("num_images") != len(image_ids)
    ):
        raise ValueError("Baseline metric image count does not match the requested split")
    image_labels = load_labels(args.voc_root)

    manifest = {
        "format_version": 1,
        "run_id": args.run_id,
        "phase": [0, 1],
        "method_status": "frozen-baseline diagnostic; no new architecture or training",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_attention_normalization": normalization,
        "checkpoint_bcss": bcss,
        "trusted_baseline_metrics": str(args.baseline_metrics.resolve()),
        "trusted_baseline_metrics_sha256": sha256(args.baseline_metrics),
        "trusted_baseline_raw_cam_miou_percent": baseline_metrics.get("mean_iou_percent"),
        "trusted_baseline_raw_cam_threshold": baseline_metrics.get("background_threshold"),
        "trusted_baseline_split": baseline_split,
        "baseline_count_match_required": baseline_split == args.split,
        "voc_root": str(args.voc_root.resolve()),
        "id_list": str(args.id_list.resolve()),
        "id_list_sha256": sha256(args.id_list),
        "split": args.split,
        "num_images": len(image_ids),
        "resolutions": list(args.resolutions),
        "semantic_resolution": args.semantic_resolution,
        "confirmatory_layer_zero_based": (
            args.confirmatory_layer if args.confirmatory_layer >= 0 else None
        ),
        "diagnostic_head_zero_based": (
            args.diagnostic_head if args.diagnostic_head >= 0 else None
        ),
        "cam_threshold": args.cam_threshold,
        "semantic_threshold": args.semantic_threshold,
        "semantic_threshold_sensitivity": list(args.semantic_thresholds),
        "geometry": "aspect-preserving Resize(256/224*r), CenterCrop(r), nearest GT to patch grid",
        "token_order": "20 class tokens followed by image patches; E0 has no register/background token",
        "head_aggregation": "softmax per head, then arithmetic mean across six heads",
        "gt_usage": "analysis only; P(c|p) semantic metrics are restricted to GT foreground",
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    phase0_moments = defaultdict(lambda: defaultdict(RunningMoments))
    classification_logits = defaultdict(list)
    classification_targets = []
    localization_confusions = defaultdict(lambda: np.zeros((21, 21), dtype=np.int64))
    semantic_confusions = defaultdict(lambda: np.zeros((20, 20), dtype=np.int64))
    head_cp_confusions = defaultdict(lambda: np.zeros((21, 21), dtype=np.int64))
    head_pc_confusions = defaultdict(lambda: np.zeros((20, 20), dtype=np.int64))
    foreground_totals = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    region_totals = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    region_class_totals = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    sensitivity_foreground = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    sensitivity_overlap = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    sensitivity_region_c = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    per_image_foreground_class_counts = []
    per_image_rows = []
    start_time = time.time()

    capture = AttentionCapture()
    capture.attach(model)
    phase0_path = args.output_dir / "phase0_per_image_layer.csv"
    try:
        with phase0_path.open("w", newline="", encoding="utf-8") as phase0_stream:
            phase0_writer = csv.DictWriter(phase0_stream, fieldnames=PHASE0_FIELDS)
            phase0_writer.writeheader()
            with torch.inference_mode():
                for image_index, image_id in enumerate(image_ids):
                    image = PIL.Image.open(
                        args.voc_root / "JPEGImages" / f"{image_id}.jpg"
                    ).convert("RGB")
                    mask_path = segmentation_path(args.voc_root, image_id)
                    if not mask_path.is_file():
                        raise FileNotFoundError(mask_path)
                    mask = PIL.Image.open(mask_path)
                    active_classes = np.asarray(image_labels[image_id], dtype=bool)
                    classification_targets.append(active_classes.astype(np.uint8))
                    semantic_payload = None
                    semantic_target = None
                    semantic_image = None

                    for resolution in args.resolutions:
                        if resolution == args.semantic_resolution:
                            semantic_image, tensor, semantic_target = paired_transform(
                                image, mask, resolution
                            )
                        else:
                            tensor = image_transform(image, resolution)
                        capture_semantic = resolution == args.semantic_resolution
                        capture_full = (
                            image_index < args.raw_dump_images
                            and resolution == args.raw_dump_resolution
                        )
                        capture.begin(capture_semantic, capture_full)
                        class_tokens, _, attention_weights, _ = model.forward_features(
                            tensor.to(device)
                        )
                        records, relations, full_raw = capture.consume()
                        classification_logits[resolution].append(
                            class_tokens.mean(dim=-1)[0].float().cpu().numpy()
                        )
                        patch_count = (resolution // 16) ** 2
                        for record in records:
                            head_values = []
                            for head in range(len(record["raw_logit_mean"])):
                                key = (
                                    resolution, record["layer"], head,
                                    record["query_group"], record["key_group"],
                                )
                                for metric in AttentionCapture.METRICS:
                                    phase0_moments[key][metric].add(record[metric][head])
                                head_values.append({
                                    metric: float(record[metric][head])
                                    for metric in AttentionCapture.METRICS
                                })
                            phase0_writer.writerow({
                                "run_id": args.run_id,
                                "split": args.split,
                                "image_id": image_id,
                                "resolution": resolution,
                                "patch_count": patch_count,
                                "layer": record["layer"],
                                "query_group": record["query_group"],
                                "key_group": record["key_group"],
                                **{
                                    metric: f"{np.mean([item[metric] for item in head_values]):.10g}"
                                    for metric in AttentionCapture.METRICS
                                },
                            })
                        if capture_full:
                            raw = torch.stack([item["raw"] for item in full_raw]).numpy()
                            post = torch.stack([item["post"] for item in full_raw]).numpy()
                            np.savez_compressed(
                                args.output_dir / "raw_dumps" / f"{image_id}_r{resolution}.npz",
                                image_id=image_id,
                                resolution=resolution,
                                class_count=20,
                                patch_count=patch_count,
                                raw_logits=raw,
                                post_attention=post,
                            )
                        if capture_semantic:
                            semantic_payload = relations
                        del attention_weights, class_tokens, tensor

                    if semantic_payload is None or semantic_target is None:
                        raise RuntimeError("Semantic resolution was not captured")
                    target_flat = semantic_target.reshape(-1).astype(np.int64)
                    valid = target_flat != 255
                    foreground_valid = valid & (target_flat > 0)
                    per_image_foreground_class_counts.append(
                        np.bincount(
                            target_flat[foreground_valid] - 1, minlength=20
                        ).astype(np.int64)
                    )
                    sample_maps = []

                    for relation in semantic_payload:
                        layer = relation["layer"]
                        cp_heads, pc_heads = conditional_relations(
                            relation["raw_cp"], relation["raw_pc"]
                        )
                        present_heads = present_class_relation(
                            relation["raw_pc"], active_classes
                        )
                        cp = cp_heads.mean(dim=0)
                        pc = pc_heads.mean(dim=0)
                        pc_present = present_heads.mean(dim=0)
                        mutual = mutual_relation(cp, pc)
                        cp_score = spatial_minmax(cp).numpy()
                        mutual_score = spatial_minmax(mutual).numpy()
                        post_cp_score = spatial_minmax(
                            relation["post_cp"].mean(dim=0)
                        ).numpy()
                        pc_array = pc.numpy()
                        pc_present_array = pc_present.numpy()
                        cp_prediction = cam_prediction(
                            cp_score, active_classes, args.cam_threshold
                        )
                        post_cp_prediction = cam_prediction(
                            post_cp_score, active_classes, args.cam_threshold
                        )
                        mutual_prediction = cam_prediction(
                            mutual_score, active_classes, args.cam_threshold
                        )
                        pc_prediction = semantic_prediction(pc_array)
                        pc_present_prediction = semantic_prediction(pc_present_array)
                        pc_fg_prediction = pc_foreground_prediction(
                            pc_array, active_classes, args.semantic_threshold
                        )
                        pc_present_fg_prediction = pc_foreground_prediction(
                            pc_present_array, active_classes, args.semantic_threshold
                        )

                        update_relation_confusions(
                            localization_confusions, "cp", layer, target_flat,
                            cp_prediction, 21, valid
                        )
                        update_relation_confusions(
                            localization_confusions, "post_cp", layer, target_flat,
                            post_cp_prediction, 21, valid
                        )
                        update_relation_confusions(
                            localization_confusions, "mutual", layer, target_flat,
                            mutual_prediction, 21, valid
                        )
                        update_relation_confusions(
                            semantic_confusions, "pc_all", layer, target_flat - 1,
                            pc_prediction, 20, foreground_valid
                        )
                        update_relation_confusions(
                            semantic_confusions, "pc_present", layer, target_flat - 1,
                            pc_present_prediction, 20, foreground_valid
                        )
                        for source, prediction in (
                            ("cp", cp_prediction),
                            ("post_cp", post_cp_prediction),
                            ("mutual", mutual_prediction),
                            ("pc_all", pc_fg_prediction),
                            ("pc_present", pc_present_fg_prediction),
                        ):
                            foreground_totals[(source, layer)] += foreground_counts(
                                prediction, target_flat
                            )

                        for head in range(cp_heads.shape[0]):
                            head_cp_score = spatial_minmax(cp_heads[head]).numpy()
                            head_cp_prediction = cam_prediction(
                                head_cp_score, active_classes, args.cam_threshold
                            )
                            head_pc_prediction = semantic_prediction(pc_heads[head].numpy())
                            head_cp_confusions[(layer, head)] += confusion_matrix(
                                target_flat, head_cp_prediction, 21, valid
                            )
                            head_pc_confusions[(layer, head)] += confusion_matrix(
                                target_flat - 1, head_pc_prediction, 20, foreground_valid
                            )

                        cp_high = cp_score[active_classes] > args.cam_threshold
                        pc_high = pc_array[:, active_classes].T > args.semantic_threshold
                        overlap_intersection = np.logical_and(cp_high, pc_high).sum()
                        overlap_union = np.logical_or(cp_high, pc_high).sum()
                        c_target = c_valid = low_target = low_valid = missed_target = 0
                        for class_index in np.flatnonzero(active_classes):
                            masks = four_region_masks(
                                cp_score[class_index], pc_array[:, class_index],
                                args.semantic_threshold,
                            )
                            for region_name, region_mask in masks.items():
                                composition = region_composition(
                                    region_mask, target_flat, class_index + 1
                                )
                                region_totals[(layer, region_name)] += composition
                                region_class_totals[(layer, region_name, class_index)] += composition
                            c_target += region_composition(
                                masks["C"], target_flat, class_index + 1
                            )[0]
                            c_valid += np.logical_and(masks["C"], valid).sum()
                            low = cp_score[class_index] <= args.cam_threshold
                            low_target += np.logical_and(low, target_flat == class_index + 1).sum()
                            low_valid += np.logical_and(low, valid).sum()
                            missed_target += np.logical_and(
                                low, target_flat == class_index + 1
                            ).sum()

                        for threshold in args.semantic_thresholds:
                            threshold_pc_prediction = pc_foreground_prediction(
                                pc_array, active_classes, threshold
                            )
                            threshold_present_prediction = pc_foreground_prediction(
                                pc_present_array, active_classes, threshold
                            )
                            sensitivity_foreground[(threshold, "pc_all", layer)] += (
                                foreground_counts(threshold_pc_prediction, target_flat)
                            )
                            sensitivity_foreground[(threshold, "pc_present", layer)] += (
                                foreground_counts(threshold_present_prediction, target_flat)
                            )
                            threshold_pc_high = (
                                pc_array[:, active_classes].T > threshold
                            )
                            sensitivity_overlap[(threshold, layer)] += np.asarray([
                                np.logical_and(cp_high, threshold_pc_high).sum(),
                                np.logical_or(cp_high, threshold_pc_high).sum(),
                            ], dtype=np.int64)
                            threshold_c_target = 0
                            threshold_c_valid = 0
                            threshold_missed_target = 0
                            for class_index in np.flatnonzero(active_classes):
                                cp_low = cp_score[class_index] <= args.cam_threshold
                                pc_high_for_class = pc_array[:, class_index] > threshold
                                region_c = cp_low & pc_high_for_class
                                threshold_c_target += np.logical_and(
                                    region_c, target_flat == class_index + 1
                                ).sum()
                                threshold_c_valid += np.logical_and(region_c, valid).sum()
                                threshold_missed_target += np.logical_and(
                                    cp_low, target_flat == class_index + 1
                                ).sum()
                            sensitivity_region_c[(threshold, layer)] += np.asarray([
                                threshold_c_target,
                                threshold_c_valid,
                                threshold_missed_target,
                            ], dtype=np.int64)

                        cp_counts = foreground_counts(cp_prediction, target_flat)
                        pc_counts = foreground_counts(pc_fg_prediction, target_flat)
                        pc_present_counts = foreground_counts(
                            pc_present_fg_prediction, target_flat
                        )
                        mutual_counts = foreground_counts(mutual_prediction, target_flat)
                        cp_precision, cp_recall = precision_recall(cp_counts)
                        pc_precision, pc_recall = precision_recall(pc_counts)
                        pc_present_precision, pc_present_recall = precision_recall(
                            pc_present_counts
                        )
                        mutual_precision, mutual_recall = precision_recall(mutual_counts)
                        c_purity = safe_ratio(c_target, c_valid)
                        low_purity = safe_ratio(low_target, low_valid)
                        row = {
                            "run_id": args.run_id,
                            "split": args.split,
                            "image_id": image_id,
                            "layer": layer,
                            "num_active_classes": int(active_classes.sum()),
                            "cp_cam_miou": mean_iou_for_image(
                                target_flat, cp_prediction, 21, valid
                            ),
                            "cp_fg_precision": cp_precision,
                            "cp_fg_recall": cp_recall,
                            "pc_all_fg_accuracy": summary_value(
                                confusion_matrix(
                                    target_flat - 1, pc_prediction, 20, foreground_valid
                                ), "accuracy"
                            ),
                            "pc_all_fg_miou": mean_iou_for_image(
                                target_flat - 1, pc_prediction, 20, foreground_valid
                            ),
                            "pc_all_fg_precision": pc_precision,
                            "pc_all_fg_recall": pc_recall,
                            "pc_present_fg_accuracy": summary_value(
                                confusion_matrix(
                                    target_flat - 1, pc_present_prediction, 20,
                                    foreground_valid,
                                ), "accuracy"
                            ),
                            "pc_present_fg_miou": mean_iou_for_image(
                                target_flat - 1, pc_present_prediction, 20,
                                foreground_valid,
                            ),
                            "pc_present_fg_precision": pc_present_precision,
                            "pc_present_fg_recall": pc_present_recall,
                            "mutual_cam_miou": mean_iou_for_image(
                                target_flat, mutual_prediction, 21, valid
                            ),
                            "mutual_fg_precision": mutual_precision,
                            "mutual_fg_recall": mutual_recall,
                            "cp_pc_overlap_iou": safe_ratio(
                                overlap_intersection, overlap_union
                            ),
                            "region_c_target_purity": c_purity,
                            "cp_low_target_purity_reference": low_purity,
                            "region_c_purity_difference": (
                                c_purity - low_purity
                                if c_purity is not None and low_purity is not None else None
                            ),
                            "region_c_recovery_recall": safe_ratio(c_target, missed_target),
                        }
                        per_image_rows.append(row)
                        sample_maps.append({
                            "cp": cp_score.reshape(20, *semantic_target.shape),
                            "pc": pc_array,
                            "mutual": mutual_score.reshape(20, *semantic_target.shape),
                        })

                    if image_index < args.sample_dumps:
                        np.savez_compressed(
                            args.output_dir / "samples" / f"{image_id}.npz",
                            image_id=image_id,
                            image_level_labels=active_classes,
                            patch_target=semantic_target,
                            conditional_cp=np.stack([item["cp"] for item in sample_maps]).astype(np.float16),
                            conditional_pc=np.stack([
                                item["pc"].T.reshape(20, *semantic_target.shape)
                                for item in sample_maps
                            ]).astype(np.float16),
                            mutual=np.stack([item["mutual"] for item in sample_maps]).astype(np.float16),
                        )
                        render_preview(
                            args.output_dir / "previews" / f"{image_id}.png",
                            semantic_image, semantic_target, active_classes, sample_maps,
                            args.semantic_threshold,
                        )
                    if (image_index + 1) % 10 == 0 or image_index + 1 == len(image_ids):
                        print(
                            f"images={image_index + 1}/{len(image_ids)} "
                            f"elapsed_seconds={time.time() - start_time:.1f}",
                            flush=True,
                        )
    finally:
        capture.close()

    with (args.output_dir / "phase0_layer_head_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "resolution", "layer", "head", "query_group", "key_group", "metric",
            "mean", "std_across_images", "min", "max", "num_images",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, metrics in sorted(phase0_moments.items()):
            resolution, layer, head, query_group, key_group = key
            for metric, moments in sorted(metrics.items()):
                summary = moments.summary()
                writer.writerow({
                    "resolution": resolution,
                    "layer": layer,
                    "head": head,
                    "query_group": query_group,
                    "key_group": key_group,
                    "metric": metric,
                    "mean": summary["mean"],
                    "std_across_images": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "num_images": summary["count"],
                })

    with (args.output_dir / "phase1_per_image_layer.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=PHASE1_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(per_image_rows)

    layer_rows = []
    bootstrap = {}
    bootstrap_metrics = (
        "cp_cam_miou", "pc_all_fg_accuracy", "pc_all_fg_miou",
        "pc_present_fg_accuracy", "pc_present_fg_miou", "mutual_cam_miou",
        "cp_pc_overlap_iou", "region_c_target_purity",
        "region_c_purity_difference", "region_c_recovery_recall",
    )
    for layer in range(len(model.blocks)):
        cp_summary = confusion_summary(localization_confusions[("cp", layer)])
        post_cp_summary = confusion_summary(localization_confusions[("post_cp", layer)])
        mutual_summary = confusion_summary(localization_confusions[("mutual", layer)])
        pc_summary = confusion_summary(semantic_confusions[("pc_all", layer)])
        pc_present_summary = confusion_summary(semantic_confusions[("pc_present", layer)])
        row = {
            "layer": layer,
            "cp_cam_miou": cp_summary["mean_iou"],
            "post_cp_cam_miou": post_cp_summary["mean_iou"],
            "pc_all_fg_accuracy": pc_summary["accuracy"],
            "pc_all_fg_miou": pc_summary["mean_iou"],
            "pc_present_fg_accuracy": pc_present_summary["accuracy"],
            "pc_present_fg_miou": pc_present_summary["mean_iou"],
            "mutual_cam_miou": mutual_summary["mean_iou"],
        }
        for source in ("cp", "pc_all", "pc_present", "mutual"):
            precision, recall = precision_recall(foreground_totals[(source, layer)])
            row[f"{source}_fg_precision"] = precision
            row[f"{source}_fg_recall"] = recall
        for metric in bootstrap_metrics:
            values = [
                item[metric] for item in per_image_rows
                if item["layer"] == layer and item[metric] is not None
            ]
            bootstrap[f"layer_{layer}_{metric}"] = bootstrap_mean(
                values, args.bootstrap_resamples, args.bootstrap_seed + layer
            )
            row[f"macro_image_{metric}"] = finite_mean(values)
        layer_rows.append(row)

    with (args.output_dir / "phase1_layer_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    with (args.output_dir / "phase1_per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = ("source", "layer", "class_id", "class_name", "accuracy", "iou", "patches")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in ("cp", "post_cp", "mutual"):
            for layer in range(len(model.blocks)):
                summary = confusion_summary(localization_confusions[(source, layer)])
                for class_index, class_name in enumerate(("background", *CLASS_NAMES)):
                    writer.writerow({
                        "source": source,
                        "layer": layer,
                        "class_id": class_index,
                        "class_name": class_name,
                        "accuracy": summary["per_class_accuracy"][class_index],
                        "iou": summary["per_class_iou"][class_index],
                        "patches": int(summary["target_count"][class_index]),
                    })
        for source in ("pc_all", "pc_present"):
            for layer in range(len(model.blocks)):
                summary = confusion_summary(semantic_confusions[(source, layer)])
                for class_index, class_name in enumerate(CLASS_NAMES):
                    writer.writerow({
                        "source": source,
                        "layer": layer,
                        "class_id": class_index + 1,
                        "class_name": class_name,
                        "accuracy": summary["per_class_accuracy"][class_index],
                        "iou": summary["per_class_iou"][class_index],
                        "patches": int(summary["target_count"][class_index]),
                    })

    with (args.output_dir / "phase1_head_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = ("layer", "head", "cp_cam_miou", "pc_all_fg_accuracy", "pc_all_fg_miou")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer, head in sorted(head_cp_confusions):
            cp_summary = confusion_summary(head_cp_confusions[(layer, head)])
            pc_summary = confusion_summary(head_pc_confusions[(layer, head)])
            writer.writerow({
                "layer": layer,
                "head": head,
                "cp_cam_miou": cp_summary["mean_iou"],
                "pc_all_fg_accuracy": pc_summary["accuracy"],
                "pc_all_fg_miou": pc_summary["mean_iou"],
            })

    with (args.output_dir / "phase1_region_composition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "layer", "region", "class_id", "class_name", "target_class",
            "other_foreground", "background", "ignore", "target_purity",
            "foreground_purity",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (layer, region, class_index), counts in sorted(region_class_totals.items()):
            target_count, other, background, ignore = counts
            valid_count = target_count + other + background
            writer.writerow({
                "layer": layer,
                "region": region,
                "class_id": class_index + 1,
                "class_name": CLASS_NAMES[class_index],
                "target_class": int(target_count),
                "other_foreground": int(other),
                "background": int(background),
                "ignore": int(ignore),
                "target_purity": safe_ratio(target_count, valid_count),
                "foreground_purity": safe_ratio(target_count + other, valid_count),
            })

    with (args.output_dir / "phase1_threshold_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "semantic_threshold", "layer", "pc_all_fg_precision",
            "pc_all_fg_recall", "pc_present_fg_precision",
            "pc_present_fg_recall", "cp_pc_overlap_iou",
            "region_c_target_purity", "region_c_recovery_recall",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for threshold in args.semantic_thresholds:
            for layer in range(len(model.blocks)):
                pc_precision, pc_recall = precision_recall(
                    sensitivity_foreground[(threshold, "pc_all", layer)]
                )
                present_precision, present_recall = precision_recall(
                    sensitivity_foreground[(threshold, "pc_present", layer)]
                )
                intersection, union = sensitivity_overlap[(threshold, layer)]
                c_target, c_valid, missed_target = sensitivity_region_c[
                    (threshold, layer)
                ]
                writer.writerow({
                    "semantic_threshold": threshold,
                    "layer": layer,
                    "pc_all_fg_precision": pc_precision,
                    "pc_all_fg_recall": pc_recall,
                    "pc_present_fg_precision": present_precision,
                    "pc_present_fg_recall": present_recall,
                    "cp_pc_overlap_iou": safe_ratio(intersection, union),
                    "region_c_target_purity": safe_ratio(c_target, c_valid),
                    "region_c_recovery_recall": safe_ratio(c_target, missed_target),
                })

    classification_targets = np.stack(classification_targets)
    classification = {}
    for resolution, values in classification_logits.items():
        logits = np.stack(values)
        per_class_ap = []
        for index in range(20):
            target = classification_targets[:, index]
            per_class_ap.append(
                average_precision_score(target, logits[:, index])
                if len(np.unique(target)) == 2 else math.nan
            )
        per_image_ap = [
            average_precision_score(classification_targets[index], logits[index])
            for index in range(len(logits))
        ]
        classification[str(resolution)] = {
            "class_map": float(np.nanmean(per_class_ap)),
            "mean_image_ap": float(np.mean(per_image_ap)),
            "per_class_ap": {
                name: (float(value) if math.isfinite(value) else None)
                for name, value in zip(CLASS_NAMES, per_class_ap)
            },
        }

    best_pc_present_layer = max(
        layer_rows,
        key=lambda row: row["pc_present_fg_accuracy"]
        if row["pc_present_fg_accuracy"] is not None else -math.inf,
    )["layer"]
    best_pc_all_layer = max(
        layer_rows,
        key=lambda row: row["pc_all_fg_accuracy"]
        if row["pc_all_fg_accuracy"] is not None else -math.inf,
    )["layer"]
    if args.confirmatory_layer >= len(model.blocks):
        raise ValueError(
            f"Confirmatory layer {args.confirmatory_layer} is outside "
            f"the {len(model.blocks)}-layer model"
        )
    if args.diagnostic_head >= model.blocks[0].attn.num_heads:
        raise ValueError(
            f"Diagnostic head {args.diagnostic_head} is outside "
            f"the {model.blocks[0].attn.num_heads}-head model"
        )
    selected_pc_layer = (
        args.confirmatory_layer if args.confirmatory_layer >= 0 else best_pc_all_layer
    )
    random_accuracy = 1.0 / 20.0
    best_pc_accuracy = layer_rows[best_pc_all_layer]["pc_all_fg_accuracy"]
    selected_pc_bootstrap = bootstrap[
        f"layer_{selected_pc_layer}_pc_all_fg_accuracy"
    ]
    minimum_region_images = max(30, math.ceil(0.05 * len(image_ids)))
    region_layers = (
        [selected_pc_layer]
        if args.confirmatory_layer >= 0 else list(range(len(model.blocks)))
    )
    eligible_region_layers = [
        layer for layer in region_layers
        if bootstrap[f"layer_{layer}_region_c_purity_difference"]["num_images"]
        >= minimum_region_images
    ]
    best_region_c_layer = (
        max(
            eligible_region_layers,
            key=lambda layer: bootstrap[
                f"layer_{layer}_region_c_purity_difference"
            ]["mean"],
        )
        if eligible_region_layers else None
    )
    maximum_recovery = max(
        row["macro_image_region_c_recovery_recall"] or 0.0
        for row in layer_rows
        if row["layer"] in region_layers
    )
    selected_region_bootstrap = (
        bootstrap[f"layer_{best_region_c_layer}_region_c_purity_difference"]
        if best_region_c_layer is not None
        else {"ci95": [None, None], "num_images": 0}
    )
    gates = conservative_diagnostic_gates(
        pc_accuracy_ci_lower=selected_pc_bootstrap["ci95"][0],
        random_accuracy=random_accuracy,
        maximum_recovery_recall=maximum_recovery,
        region_c_ci_lower=selected_region_bootstrap["ci95"][0],
        region_c_images=selected_region_bootstrap["num_images"],
        total_images=len(image_ids),
    )
    gates["decision"] = "diagnostic_only_pending_scientific_review"
    selected_confusion = semantic_confusions[("pc_all", selected_pc_layer)]
    target_counts = selected_confusion.sum(axis=1)
    majority_class_index = int(target_counts.argmax())
    majority_patch_accuracy = float(target_counts.max() / target_counts.sum())
    selected_image_rows = [
        item for item in per_image_rows if item["layer"] == selected_pc_layer
    ]
    if len(selected_image_rows) != len(per_image_foreground_class_counts):
        raise RuntimeError("Per-image semantic rows and target counts are misaligned")
    majority_image_accuracies = []
    accuracy_advantages = []
    for row, counts in zip(selected_image_rows, per_image_foreground_class_counts):
        count = int(counts.sum())
        accuracy = row["pc_all_fg_accuracy"]
        if count and accuracy is not None and math.isfinite(accuracy):
            majority_accuracy = float(counts[majority_class_index] / count)
            majority_image_accuracies.append(majority_accuracy)
            accuracy_advantages.append(float(accuracy - majority_accuracy))
    majority_bootstrap = bootstrap_mean(
        majority_image_accuracies,
        args.bootstrap_resamples,
        args.bootstrap_seed + 1000,
    )
    majority_advantage_bootstrap = bootstrap_mean(
        accuracy_advantages,
        args.bootstrap_resamples,
        args.bootstrap_seed + 2000,
    )
    majority_class_accuracy = majority_bootstrap["mean"]
    majority_control = {
        "class_index_zero_based": majority_class_index,
        "class_id_voc": majority_class_index + 1,
        "class_name": CLASS_NAMES[majority_class_index],
        "patch_weighted_accuracy": majority_patch_accuracy,
        "macro_image_accuracy": majority_class_accuracy,
        "macro_image_accuracy_ci95": majority_bootstrap["ci95"],
        "pc_all_minus_majority_macro_image_accuracy": (
            majority_advantage_bootstrap["mean"]
        ),
        "pc_all_minus_majority_macro_image_accuracy_ci95": (
            majority_advantage_bootstrap["ci95"]
        ),
        "num_images": majority_bootstrap["num_images"],
    }
    permutation_control = class_permutation_control(
        selected_confusion,
        resamples=args.permutation_resamples,
        seed=args.permutation_seed,
    )
    gates["pc_all_above_foreground_majority_class"] = bool(
        majority_advantage_bootstrap["ci95"][0] is not None
        and majority_advantage_bootstrap["ci95"][0] > 0
    )
    gates["class_identity_permutation_p_below_0_01"] = bool(
        permutation_control["empirical_p_greater_equal"] < 0.01
    )
    diagnostic_head_metrics = None
    if args.diagnostic_head >= 0:
        head_cp = confusion_summary(
            head_cp_confusions[(selected_pc_layer, args.diagnostic_head)]
        )
        head_pc = confusion_summary(
            head_pc_confusions[(selected_pc_layer, args.diagnostic_head)]
        )
        diagnostic_head_metrics = {
            "layer_zero_based": selected_pc_layer,
            "head_zero_based": args.diagnostic_head,
            "pc_all_fg_accuracy": head_pc["accuracy"],
            "pc_all_fg_miou": head_pc["mean_iou"],
            "cp_cam_miou": head_cp["mean_iou"],
            "role": "secondary_diagnostic_not_used_for_selection",
        }
    metrics = {
        "run_id": args.run_id,
        "phase": [0, 1],
        "num_images": len(image_ids),
        "elapsed_seconds": time.time() - start_time,
        "trusted_baseline": {
            "raw_cam_miou_percent": baseline_metrics.get("mean_iou_percent"),
            "threshold": baseline_metrics.get("background_threshold"),
            "checkpoint_sha256": checkpoint_hash,
        },
        "classification": classification,
        "phase1": {
            "selection_mode": (
                "fixed_confirmatory_layer"
                if args.confirmatory_layer >= 0 else "exploratory_best_layer"
            ),
            "confirmatory_layer_zero_based": (
                selected_pc_layer if args.confirmatory_layer >= 0 else None
            ),
            "selected_primary_layer_zero_based": selected_pc_layer,
            "best_pc_present_layer_zero_based": best_pc_present_layer,
            "best_pc_all_layer_zero_based": best_pc_all_layer,
            "best_pc_all_fg_accuracy": best_pc_accuracy,
            "uniform_20_class_random_accuracy": random_accuracy,
            "foreground_majority_class_accuracy": majority_class_accuracy,
            "foreground_majority_class_control": majority_control,
            "class_identity_permutation_control": permutation_control,
            "diagnostic_head": diagnostic_head_metrics,
            "best_region_c_enrichment_layer_zero_based": best_region_c_layer,
            "layer_metrics": layer_rows,
            "bootstrap": bootstrap,
            "gates": gates,
        },
        "definitions": {
            "cp": "softmax over patch keys per head, head mean, spatial min-max, fixed threshold",
            "pc_all": "softmax over all 20 class keys per head; semantic metrics use GT foreground only",
            "pc_present": "image-label-masked class-only softmax per head; GT foreground only",
            "mutual": "sqrt(P(p|c)*P(c|p)), then spatial min-max for fixed-threshold CAM",
            "region_high": f"strictly greater than {args.semantic_threshold}",
            "region_c_reference": "target-class purity among all P(p|c)-low patches",
            "layer_indexing": "zero based in files; add one for paper tables",
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "completion.json").write_text(
        json.dumps({
            "complete": True,
            "num_images": len(image_ids),
            "elapsed_seconds": metrics["elapsed_seconds"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "complete": True,
        "num_images": len(image_ids),
        "best_pc_present_layer": best_pc_present_layer,
        "best_pc_all_layer": best_pc_all_layer,
        "selected_primary_layer": selected_pc_layer,
        "best_pc_all_fg_accuracy": best_pc_accuracy,
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
