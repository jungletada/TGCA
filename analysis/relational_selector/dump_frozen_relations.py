#!/usr/bin/env python
"""One-pass frozen Task A/B score dump for native MCTformer+-Small.

This program never trains, backpropagates, alters a model parameter, or writes
below a source checkpoint/result tree.  It runs exactly one model forward per
VOC batch, captures all twelve raw post-block layers for Task A, and writes
only final-layer S0--S5 maps for the later GT evaluation of Task B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.final_token_relations.run_final_token_relations import (
    EMBED_DIM,
    NATIVE_CHECKPOINT_SHA256,
    NUM_CLASSES,
    PATCH_COUNT,
    _raw_classifier_map,
    load_native_mctformer,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import VOCSemanticDataset
from analysis.relational_selector.layer_capture import LayerRelationCollector
from analysis.relational_selector.scores import build_selector_scores
from analysis.relational_selector.task_a import evaluate_layer_relations, global_patch_regions
from analysis.spatial_graph_stability.provenance import (
    EXPECTED_VOC_IMAGES,
    RunLog,
    command_line,
    create_output,
    csv_dump,
    git_metadata,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    timestamp,
    write_environment_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8/checkpoints/baseline/mctformerplus_final.pth"),
    )
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument("--list-path", type=Path, default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="deterministic prefix for smoke only")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--allow-tracked-dirty", action="store_true", help="test-only escape hatch; full runs remain clean-source only")
    return parser.parse_args()


def _resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        payload = value.detach().cpu().contiguous().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(payload)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_sample(destination: Path, *, image_id: str, labels: np.ndarray, scores: dict[str, torch.Tensor]) -> None:
    values = {name: tensor.detach().cpu().numpy().astype(np.float32, copy=False) for name, tensor in scores.items()}
    np.savez_compressed(
        destination,
        image_id=np.asarray(image_id),
        labels=np.asarray(labels, dtype=np.uint8),
        grid=np.asarray((28, 28), dtype=np.int16),
        **values,
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    os.chdir(repo_root)
    require_environment()
    if args.limit < 0 or args.batch_size < 1:
        raise ValueError("limit must be non-negative and batch-size positive")
    if args.limit == 0 and args.allow_tracked_dirty:
        raise RuntimeError("--allow-tracked-dirty is permitted only for a bounded smoke run")
    if not args.allow_tracked_dirty:
        source_git = require_clean_tracked(repo_root)
    else:
        source_git = git_metadata(repo_root)
    output = create_output(_resolve(repo_root, args.output_dir))
    log = RunLog(output / "run.log")
    _seed_everything(args.seed)
    checkpoint = _resolve(repo_root, args.checkpoint)
    voc_root = _resolve(repo_root, args.voc_root)
    list_path = _resolve(repo_root, args.list_path)
    if sha256_file(checkpoint) != NATIVE_CHECKPOINT_SHA256:
        raise RuntimeError("requested checkpoint is not the frozen native MCTformer+ baseline")
    data = VOCSemanticDataset(voc_root, list_path, input_size=448, limit=args.limit)
    expected = args.limit or EXPECTED_VOC_IMAGES
    if len(data) != expected:
        raise RuntimeError(f"expected {expected} images, found {len(data)}")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model, checkpoint_metadata = load_native_mctformer(checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = model.to(device).eval()
    before = _parameter_sha256(model)
    collector = LayerRelationCollector(model, num_classes=NUM_CLASSES, patch_count=PATCH_COUNT, width=EMBED_DIM)
    sample_dir = output / "samples"
    sample_dir.mkdir()
    task_a_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    mu_batches: list[np.ndarray] = []
    global_index = 0
    log(f"start: images={len(data)} device={device} batch_size={args.batch_size} checkpoint={checkpoint}")
    try:
        with torch.no_grad():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                names = [str(value) for value in batch["name"]]
                masks = batch["mask"]
                if images.shape[-2:] != (448, 448) or labels.shape[1] != NUM_CLASSES:
                    raise RuntimeError("dataset geometry/label contract failed")
                collector.set_positive_labels(labels)
                final_classes, final_patches, _attention, _all_classes = model.forward_features(images)
                records = collector.consume()
                raw_map = _raw_classifier_map(model, final_patches)
                scores = dict(build_selector_scores(final_classes, final_patches, raw_map))
                if final_classes.shape[1:] != (NUM_CLASSES, EMBED_DIM) or final_patches.shape[1:] != (PATCH_COUNT, EMBED_DIM):
                    raise RuntimeError("final native token shape contract failed")
                mu_batch = torch.stack([record.mean_class for record in records], dim=1).cpu().numpy().astype(np.float32, copy=False)
                mu_batches.append(mu_batch)
                labels_cpu = labels.cpu().numpy()
                for local, image_id in enumerate(names):
                    label_values = labels_cpu[local]
                    region_codes = global_patch_regions(masks[local].numpy())
                    if region_codes.shape != (PATCH_COUNT,):
                        raise RuntimeError("semantic mask did not map to the native 28x28 patch grid")
                    index_rows.append({
                        "image_id": image_id, "image_index": global_index + local,
                        "label_count": int((label_values > 0).sum()),
                    })
                    for layer_index, record in enumerate(records, start=1):
                        task_a_rows.append(evaluate_layer_relations(
                            image_id=image_id, image_index=global_index + local,
                            labels=label_values, regions=region_codes, layer=layer_index,
                            raw=record.raw[local].cpu().numpy(),
                            residual=record.residual[local].cpu().numpy(),
                            common=record.common[local].cpu().numpy(),
                            positive_common=record.positive_common[local].cpu().numpy(),
                        ))
                    _save_sample(
                        sample_dir / f"{image_id}.npz", image_id=image_id,
                        labels=label_values, scores={name: value[local] for name, value in scores.items()},
                    )
                global_index += len(names)
                if batch_number == 1 or batch_number % 20 == 0 or global_index == len(data):
                    log(f"processed {global_index}/{len(data)} images")
    finally:
        collector.close()
    after = _parameter_sha256(model)
    if before != after:
        raise RuntimeError("frozen analysis changed model parameters")
    if global_index != len(data) or len(task_a_rows) != len(data) * 12:
        raise RuntimeError("incomplete Task A/B frozen dump")
    mu = np.concatenate(mu_batches, axis=0)
    if mu.shape != (len(data), 12, EMBED_DIM) or not np.isfinite(mu).all():
        raise RuntimeError(f"unexpected shared-mean artifact shape {mu.shape}")
    np.save(output / "mu_common.npy", mu)
    task_fields = list(task_a_rows[0])
    csv_dump(output / "task_a_per_image_layer.csv", task_a_rows, task_fields)
    csv_dump(output / "index.csv", index_rows, list(index_rows[0]))
    environment = write_environment_manifests(output)
    metadata = {
        "schema": "frozen_multi_class_token_relation_dump_v1",
        "created_at": timestamp(),
        "command": command_line(),
        "git": source_git,
        "checkpoint": checkpoint_metadata,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input": {"voc_root": str(voc_root), "list_path": str(list_path), "input_size": 448, "num_images": len(data), "limit": int(args.limit)},
        "contract": {
            "model": "native MCTformer+-Small", "frozen": True, "training_or_backward": False,
            "relations_dtype": "float32", "layers": 12, "class_tokens": 20, "patch_tokens": 784,
            "score_names": ["S0", "S1", "S2", "S3", "S4", "S5"],
            "score_construction_gt_free": True,
            "note": "docs/CHAT_HANDOFF.md and docs/RESEARCH_PLAN_FULL.md were absent at task start; live Git/tmux/checkpoint state used as operational truth.",
        },
        "model_parameter_sha256_before": before,
        "model_parameter_sha256_after": after,
        "environment_manifests": environment,
    }
    json_dump(output / "metadata.json", metadata)
    json_dump(output / "completion.json", {"status": "complete", "run_kind": "smoke" if args.limit else "full", "num_images": len(data), "created_at": timestamp()})
    log(f"complete: parameter_sha256={after}; task_a_rows={len(task_a_rows)}; samples={len(index_rows)}")


if __name__ == "__main__":
    main()
