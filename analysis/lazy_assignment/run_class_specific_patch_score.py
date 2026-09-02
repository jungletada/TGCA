#!/usr/bin/env python3
"""Extract layer-wise class-specific patch scores from frozen WSSS models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import PIL
import timm
import torch
import torchvision
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.score_utils import (  # noqa: E402
    LayerScoreSummary,
    infer_patch_grid,
)
from analysis.lazy_assignment.token_collector import BlockTokenCollector  # noqa: E402
from analysis.lazy_assignment.visualize_patch_score import (  # noqa: E402
    save_score_visualizations,
)
from analysis.lazy_assignment.voc_score_dataset import VOCScoreDataset  # noqa: E402
from utils import create_cam_model  # noqa: E402


LAST_VIT_REPOSITORY = "https://github.com/ChengShiest/LAST-ViT"
LAST_VIT_COMMIT = "cdeb884af65e7774f2da80f666d95cf09a76b717"
RUNTIME_SOURCE_PATHS = (
    "analysis/lazy_assignment/__init__.py",
    "analysis/lazy_assignment/run_class_specific_patch_score.py",
    "analysis/lazy_assignment/score_utils.py",
    "analysis/lazy_assignment/token_collector.py",
    "analysis/lazy_assignment/visualize_patch_score.py",
    "analysis/lazy_assignment/voc_score_dataset.py",
)


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: layer-wise class-specific patch score"
    )
    parser.add_argument(
        "--model", choices=("mctformerplus", "mctformerv2"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--list-path", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="process only the first N images; any positive value marks a smoke run",
    )
    parser.add_argument("--save-all-classes", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visualization-images", type=int, default=10)
    parser.add_argument("--visualization-max-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--allow-uncommitted-source",
        action="store_true",
        help="mechanical smoke only; full runs always require tracked, clean runtime source",
    )
    args = parser.parse_args()
    if args.input_size < 1 or args.input_size % 16:
        parser.error("--input-size must be a positive multiple of 16")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0 or args.limit < 0:
        parser.error("--num-workers and --limit must be non-negative")
    if args.visualization_images < 0 or args.visualization_max_classes < 0:
        parser.error("visualization counts must be non-negative")
    if args.allow_uncommitted_source and args.limit == 0:
        parser.error("--allow-uncommitted-source is forbidden for a full run")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: Sequence[str], check: bool = True) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_state() -> dict:
    commit = run_command(("git", "rev-parse", "HEAD"))
    branch = run_command(("git", "branch", "--show-current"))
    remote = run_command(("git", "config", "--get", "remote.origin.url"), check=False)
    status = run_command(("git", "status", "--short"), check=True).splitlines()
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"], cwd=str(REPO_ROOT), check=False
    ).returncode != 0
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=str(REPO_ROOT), check=False
    ).returncode != 0
    source_tracked = {}
    source_hashes = {}
    for relative in RUNTIME_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        source_tracked[relative] = tracked
        source_hashes[relative] = sha256_file(path)
    return {
        "repository_url": remote,
        "commit": commit,
        "branch": branch,
        "status_short": status,
        "tracked_dirty": bool(unstaged or staged),
        "runtime_source_tracked": source_tracked,
        "runtime_source_sha256": source_hashes,
    }


def enforce_source_safety(args: argparse.Namespace, state: Mapping[str, object]) -> None:
    tracked_dirty = bool(state["tracked_dirty"])
    tracked_sources = state["runtime_source_tracked"]
    assert isinstance(tracked_sources, Mapping)
    untracked_sources = [path for path, tracked in tracked_sources.items() if not tracked]
    if not args.allow_uncommitted_source and (tracked_dirty or untracked_sources):
        raise RuntimeError(
            "Analysis source is not in a clean, tracked Git state. "
            f"tracked_dirty={tracked_dirty}, untracked_runtime_sources={untracked_sources}. "
            "A limited mechanical smoke may explicitly use --allow-uncommitted-source; "
            "a full scientific run may not."
        )


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def checkpoint_payload(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> tuple[Mapping[str, torch.Tensor], str, bool]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping, got {type(payload)}")
    if "model" in payload:
        state = payload["model"]
        checkpoint_format = "wrapped_model"
    else:
        state = payload
        checkpoint_format = "raw_state_dict"
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("checkpoint state_dict must map string keys to tensors")

    prefixed = [key.startswith("module.") for key in state]
    stripped = False
    if any(prefixed):
        if not all(prefixed):
            raise ValueError("checkpoint mixes module.-prefixed and unprefixed keys")
        state = OrderedDict((key[len("module.") :], value) for key, value in state.items())
        stripped = True
    return state, checkpoint_format, stripped


def load_state_dict_strict(model: torch.nn.Module, payload) -> dict:
    state, checkpoint_format, stripped = extract_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict load unexpectedly returned missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "format": checkpoint_format,
        "module_prefix_stripped": stripped,
        "state_key_count": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def checkpoint_configuration(payload) -> dict:
    if not isinstance(payload, Mapping):
        return {
            "attention": {"mode": "vanilla", "gamma": 1.0, "relation_bias": False},
            "bcss": {"variant": "e0"},
            "psl": {"variant": "baseline"},
            "cti_bgt": {"enabled": False},
        }
    attention = payload.get("attention_normalization", {})
    if not isinstance(attention, Mapping):
        raise TypeError("checkpoint attention_normalization must be a mapping")
    bcss = payload.get("bcss", {"variant": "e0"})
    psl = payload.get("psl", {"variant": "baseline"})
    cti_bgt = payload.get("cti_bgt", {"enabled": False})
    if not isinstance(bcss, Mapping) or not isinstance(psl, Mapping) or not isinstance(cti_bgt, Mapping):
        raise TypeError("checkpoint BCSS/PSL/CTI configuration must be mappings")
    return {
        "attention": {
            "mode": attention.get("mode", "vanilla"),
            "gamma": float(attention.get("gamma", 1.0)),
            "relation_bias": bool(attention.get("relation_bias", False)),
        },
        "bcss": dict(bcss),
        "psl": dict(psl),
        "cti_bgt": dict(cti_bgt),
    }


def create_frozen_model(args: argparse.Namespace, payload):
    configuration = checkpoint_configuration(payload)
    if args.model == "mctformerplus":
        if configuration["bcss"].get("variant", "e0") != "e0":
            raise ValueError("Experiment 1 currently requires native MCTformer+ BCSS E0")
        if configuration["psl"].get("variant", "baseline") != "baseline":
            raise ValueError("Experiment 1 currently excludes persistent-semantic variants")
        if configuration["cti_bgt"].get("enabled", False):
            raise ValueError("Experiment 1 currently excludes CTI-BGT tokens")

    factory_args = SimpleNamespace(
        model=args.model,
        num_classes=20,
        input_size=args.input_size,
        attention_normalization=configuration["attention"]["mode"],
        attention_gamma=configuration["attention"]["gamma"],
        bcss_variant=configuration["bcss"].get("variant", "e0"),
        bcss_num_background_slots=configuration["bcss"].get("num_background_slots", 1),
        bcss_tau=configuration["bcss"].get("tau", 0.5),
        bcss_beta=configuration["bcss"].get("beta", 0.5),
        bcss_cls_threshold=configuration["bcss"].get("class_threshold", 0.5),
        psl_variant=configuration["psl"].get("variant", "baseline"),
        psl_interaction_layers=tuple(
            configuration["psl"].get("interaction_layers_zero_based", (11,))
        ),
        psl_relation_dim=configuration["psl"].get("relation_dim", 384),
        psl_num_background_latents=configuration["psl"].get(
            "num_background_latents", 1
        ),
        cti_bgt=configuration["cti_bgt"].get("enabled", False),
        cti_bgt_weight=configuration["cti_bgt"].get("weight", 0.1),
        cti_bgt_n_layers=configuration["cti_bgt"].get("n_layers", 6),
        cti_bgt_affinity_start=configuration["cti_bgt"].get("affinity_start", 4),
    )
    model = create_cam_model(factory_args)
    load_info = load_state_dict_strict(model, payload)
    if hasattr(model, "set_bcss_epoch"):
        model.set_bcss_epoch(8)
    model.eval()
    return model, configuration, load_info


def tensor_tree_max_abs_diff(left, right, path: str = "root") -> tuple[float, int]:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor mismatch at {path}")
        if left.numel() == 0:
            return 0.0, 1
        if left.is_floating_point() or left.is_complex():
            difference = float((left - right).abs().max().item())
        else:
            difference = 0.0 if torch.equal(left, right) else math.inf
        return difference, 1
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"sequence mismatch at {path}")
        maximum, count = 0.0, 0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference, item_count = tensor_tree_max_abs_diff(
                left_item, right_item, f"{path}[{index}]"
            )
            maximum = max(maximum, difference)
            count += item_count
        return maximum, count
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            raise AssertionError(f"mapping mismatch at {path}")
        maximum, count = 0.0, 0
        for key in left:
            difference, item_count = tensor_tree_max_abs_diff(
                left[key], right[key], f"{path}.{key}"
            )
            maximum = max(maximum, difference)
            count += item_count
        return maximum, count
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")
    return 0.0, 0


def numerical_no_change_guard(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_patches: int,
    tolerance: float = 1e-6,
) -> dict:
    with torch.inference_mode():
        baseline = model.forward_features(image)
        with BlockTokenCollector(model, num_classes=20) as collector:
            collector.clear(expected_num_patches=num_patches)
            instrumented = model.forward_features(image)
            capture = collector.consume()
        forward_difference, tensor_count = tensor_tree_max_abs_diff(
            baseline, instrumented
        )
        class_difference = float(
            (capture.last_class_tokens - instrumented[0]).abs().max().item()
        )
        patch_difference = float(
            (capture.last_patch_tokens - instrumented[1]).abs().max().item()
        )
        if forward_difference > tolerance:
            raise RuntimeError(
                f"hook changed forward output by {forward_difference}, tolerance {tolerance}"
            )
        if max(class_difference, patch_difference) > tolerance:
            raise RuntimeError("layer-12 hook tokens do not match forward_features outputs")
        if capture.scores.shape != (len(model.blocks), 1, 20, num_patches):
            raise RuntimeError(f"unexpected guard score shape {tuple(capture.scores.shape)}")
        if not torch.isfinite(capture.scores).all():
            raise RuntimeError("non-finite score in numerical guard")
        score_min = float(capture.scores.min().item())
        score_max = float(capture.scores.max().item())
        if score_min < -1.00001 or score_max > 1.00001:
            raise RuntimeError(f"cosine range violation [{score_min}, {score_max}]")
    del baseline, instrumented, capture
    return {
        "tolerance": tolerance,
        "forward_max_abs_diff": forward_difference,
        "compared_tensor_count": tensor_count,
        "layer12_class_max_abs_diff": class_difference,
        "layer12_patch_max_abs_diff": patch_difference,
        "score_min": score_min,
        "score_max": score_max,
        "passed": True,
    }


def environment_metadata(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
    }


def write_environment_manifests(output_dir: Path) -> None:
    pip_freeze = run_command((sys.executable, "-m", "pip", "freeze"))
    (output_dir / "pip_freeze.txt").write_text(pip_freeze + "\n", encoding="utf-8")
    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_executable:
        raise RuntimeError("cannot capture conda explicit manifest: conda executable not found")
    conda_explicit = run_command(
        (conda_executable, "list", "--explicit", "--prefix", sys.prefix)
    )
    (output_dir / "conda_explicit.txt").write_text(
        conda_explicit + "\n", encoding="utf-8"
    )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("summary rows are empty")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_score_file(
    path: Path,
    image_id: str,
    positive_class_ids: np.ndarray,
    saved_class_ids: np.ndarray,
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(
        path,
        image_id=np.asarray(image_id),
        positive_class_ids=np.asarray(positive_class_ids, dtype=np.int64),
        saved_class_ids=np.asarray(saved_class_ids, dtype=np.int64),
        scores_raw=np.asarray(scores, dtype=np.float32),
        grid_h=np.asarray(grid_h, dtype=np.int32),
        grid_w=np.asarray(grid_w, dtype=np.int32),
    )


def validate_saved_artifacts(output_dir: Path, expected_images: int, depth: int) -> dict:
    manifest_path = output_dir / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected_images:
        raise RuntimeError(f"manifest has {len(records)} rows, expected {expected_images}")
    image_ids = [record["image_id"] for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("manifest contains duplicate image IDs")
    score_files = sorted((output_dir / "scores").glob("*.npz"))
    if len(score_files) != expected_images:
        raise RuntimeError(f"found {len(score_files)} score files, expected {expected_images}")

    total_maps = 0
    score_min, score_max = math.inf, -math.inf
    total_bytes = 0
    for record in records:
        path = output_dir / record["score_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as artifact:
            image_id = str(artifact["image_id"].item())
            positive_ids = artifact["positive_class_ids"]
            saved_ids = artifact["saved_class_ids"]
            scores = artifact["scores_raw"]
            grid_h = int(artifact["grid_h"].item())
            grid_w = int(artifact["grid_w"].item())
        if image_id != record["image_id"]:
            raise RuntimeError(f"image ID mismatch in {path}")
        if positive_ids.tolist() != record["positive_class_ids"]:
            raise RuntimeError(f"positive class mismatch in {path}")
        if saved_ids.tolist() != record["saved_class_ids"]:
            raise RuntimeError(f"saved class mismatch in {path}")
        expected_shape = (depth, len(saved_ids), grid_h * grid_w)
        if scores.shape != expected_shape or scores.dtype != np.float32:
            raise RuntimeError(
                f"score payload {path} has {scores.shape}/{scores.dtype}, expected {expected_shape}/float32"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"non-finite saved scores in {path}")
        local_min, local_max = float(scores.min()), float(scores.max())
        if local_min < -1.00001 or local_max > 1.00001:
            raise RuntimeError(f"saved cosine range violation in {path}")
        score_min = min(score_min, local_min)
        score_max = max(score_max, local_max)
        total_maps += len(saved_ids)
    return {
        "independent_reload_passed": True,
        "manifest_rows": len(records),
        "unique_image_ids": len(set(image_ids)),
        "score_files": len(score_files),
        "saved_class_maps": total_maps,
        "score_min": score_min,
        "score_max": score_max,
        "compressed_score_bytes": total_bytes,
    }


def execute(args: argparse.Namespace) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "tgca-repro":
        raise RuntimeError(
            "Experiment 1 must run in the tgca-repro Conda environment; "
            f"active={os.environ.get('CONDA_DEFAULT_ENV')!r}"
        )
    source_state = git_state()
    enforce_source_safety(args, source_state)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_dir = output_dir / "scores"
    visualizations_dir = output_dir / "visualizations"
    scores_dir.mkdir()
    if args.save_visualizations:
        visualizations_dir.mkdir()
    log = RunLog(output_dir / "analysis.log")
    command = shlex.join([sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    write_environment_manifests(output_dir)

    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        checkpoint = args.checkpoint.resolve()
        checkpoint_hash = sha256_file(checkpoint)
        if (
            args.expected_checkpoint_sha256
            and checkpoint_hash != args.expected_checkpoint_sha256
        ):
            raise RuntimeError(
                f"checkpoint SHA-256 {checkpoint_hash} does not match expected "
                f"{args.expected_checkpoint_sha256}"
            )
        payload = checkpoint_payload(checkpoint)
        model, model_configuration, load_info = create_frozen_model(args, payload)
        model.to(device).eval()
        depth = len(model.blocks)
        patch_size = tuple(int(value) for value in model.patch_embed.patch_size)
        if depth != 12:
            raise RuntimeError(f"Experiment 1 expects 12 blocks, found {depth}")

        dataset = VOCScoreDataset(
            voc_root=args.voc_root,
            list_path=args.list_path,
            input_size=args.input_size,
            limit=args.limit,
        )
        if len(dataset) < 1:
            raise RuntimeError("analysis dataset is empty")

        list_path = args.list_path.resolve()
        labels_path = args.voc_root.resolve() / "ImageLabel" / "cls_labels.npy"
        run_kind = "smoke" if args.limit else "full"
        environment = environment_metadata(device)
        metadata = {
            "status": "running",
            "run_id": output_dir.name,
            "analysis": "experiment1_class_specific_patch_score",
            "run_kind": run_kind,
            "repository_url": source_state["repository_url"],
            "git": source_state,
            "uncommitted_source_explicitly_allowed": bool(args.allow_uncommitted_source),
            "last_vit_source": {
                "repository": LAST_VIT_REPOSITORY,
                "commit": LAST_VIT_COMMIT,
                "score_definition": "cosine similarity between pooled/class token and patch tokens",
            },
            "model": {
                "name": args.model,
                "class": model.__class__.__name__,
                "num_classes": 20,
                "depth": depth,
                "embed_dim": int(model.embed_dim),
                "patch_size": list(patch_size),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "configuration": model_configuration,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "epoch": payload.get("epoch") if isinstance(payload, Mapping) else None,
                "strict_load": load_info,
            },
            "dataset": {
                "name": "PASCAL VOC 2012",
                "split": list_path.stem,
                "voc_root": str(args.voc_root.resolve()),
                "list_path": str(list_path),
                "list_sha256": sha256_file(list_path),
                "labels_path": str(labels_path),
                "labels_sha256": sha256_file(labels_path),
                "num_samples": len(dataset),
                "limit": args.limit,
            },
            "input": {
                "size": args.input_size,
                "scale": 1.0,
                "horizontal_flip": False,
                "random_augmentation": False,
                "transform": "bicubic Resize(int(256/224*input_size)) -> CenterCrop(input_size) -> ToTensor -> ImageNet Normalize",
            },
            "representation": "post_block_pre_final_norm",
            "layer_indexing": "1-based in outputs, block index 0-based in code",
            "score": "float32 cosine(class_token, patch_token)",
            "score_shape_before_filter": [depth, "batch", 20, "num_patches"],
            "positive_class_filter": not args.save_all_classes,
            "segmentation_ground_truth_used": False,
            "cam_used": False,
            "environment": environment,
            "command": command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(output_dir / "metadata.json", metadata)
        log(
            f"strict checkpoint load passed: {checkpoint} sha256={checkpoint_hash}; "
            f"model={model.__class__.__name__}, depth={depth}, params={metadata['model']['parameters']}"
        )
        log(
            f"dataset={list_path} samples={len(dataset)} input={args.input_size} "
            f"batch={args.batch_size} device={device} run_kind={run_kind}"
        )

        first = dataset[0]
        guard_image = first["image"].unsqueeze(0).to(device)
        grid_h, grid_w = infer_patch_grid(
            guard_image.shape, patch_size, (args.input_size // patch_size[0]) * (args.input_size // patch_size[1])
        )
        numerical_guard = numerical_no_change_guard(
            model, guard_image, num_patches=grid_h * grid_w
        )
        metadata["numerical_no_change_guard"] = numerical_guard
        atomic_json(output_dir / "metadata.json", metadata)
        log(
            "numerical guard passed: forward max diff="
            f"{numerical_guard['forward_max_abs_diff']:.3g}, layer12 cls/patch="
            f"{numerical_guard['layer12_class_max_abs_diff']:.3g}/"
            f"{numerical_guard['layer12_patch_max_abs_diff']:.3g}"
        )
        del guard_image
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda" and not args.no_pin_memory),
            drop_last=False,
        )
        summary = LayerScoreSummary(depth)
        manifest_path = output_dir / "manifest.jsonl"
        visualizations_written: list[str] = []
        processed_images = 0
        max_layer12_class_diff = 0.0
        max_layer12_patch_diff = 0.0
        start = time.perf_counter()
        with manifest_path.open("w", encoding="utf-8") as manifest_stream:
            with BlockTokenCollector(model, num_classes=20) as collector:
                with torch.inference_mode():
                    for batch_index, batch in enumerate(loader):
                        images = batch["image"].to(
                            device,
                            non_blocking=(device.type == "cuda" and not args.no_pin_memory),
                        )
                        labels = batch["label"].cpu().numpy()
                        image_ids = list(batch["name"])
                        num_patches = (
                            images.shape[-2] // patch_size[0]
                        ) * (images.shape[-1] // patch_size[1])
                        grid_h, grid_w = infer_patch_grid(
                            images.shape, patch_size, num_patches
                        )
                        collector.clear(expected_num_patches=num_patches)
                        forward_output = model.forward_features(images)
                        capture = collector.consume()
                        expected_shape = (depth, images.shape[0], 20, num_patches)
                        if tuple(capture.scores.shape) != expected_shape:
                            raise RuntimeError(
                                f"score shape {tuple(capture.scores.shape)} != {expected_shape}"
                            )
                        class_difference = float(
                            (capture.last_class_tokens - forward_output[0]).abs().max().item()
                        )
                        patch_difference = float(
                            (capture.last_patch_tokens - forward_output[1]).abs().max().item()
                        )
                        max_layer12_class_diff = max(max_layer12_class_diff, class_difference)
                        max_layer12_patch_diff = max(max_layer12_patch_diff, patch_difference)
                        if max(class_difference, patch_difference) > 1e-6:
                            raise RuntimeError("layer-12 hook/final-token equivalence failed")
                        batch_scores = capture.scores.cpu().numpy()
                        if not np.isfinite(batch_scores).all():
                            raise RuntimeError("non-finite raw cosine score")
                        if float(batch_scores.min()) < -1.00001 or float(batch_scores.max()) > 1.00001:
                            raise RuntimeError("raw cosine score escaped [-1,1] tolerance")

                        for local_index, image_id in enumerate(image_ids):
                            positive_ids = np.flatnonzero(labels[local_index] > 0).astype(np.int64)
                            if not positive_ids.size:
                                raise RuntimeError(f"VOC image {image_id} has no positive class")
                            positive_scores = batch_scores[:, local_index, positive_ids, :]
                            summary.add_image(positive_scores)
                            if args.save_all_classes:
                                saved_ids = np.arange(20, dtype=np.int64)
                                saved_scores = batch_scores[:, local_index, :, :]
                            else:
                                saved_ids = positive_ids
                                saved_scores = positive_scores
                            score_path = scores_dir / f"{image_id}.npz"
                            save_score_file(
                                score_path,
                                image_id,
                                positive_ids,
                                saved_ids,
                                saved_scores,
                                grid_h,
                                grid_w,
                            )
                            record = {
                                "image_id": image_id,
                                "score_path": str(score_path.relative_to(output_dir)),
                                "positive_class_ids": positive_ids.tolist(),
                                "saved_class_ids": saved_ids.tolist(),
                                "num_layers": depth,
                                "num_patches": num_patches,
                                "grid_h": grid_h,
                                "grid_w": grid_w,
                            }
                            manifest_stream.write(json.dumps(record, sort_keys=True) + "\n")
                            if (
                                args.save_visualizations
                                and processed_images < args.visualization_images
                            ):
                                written = save_score_visualizations(
                                    image=batch["image"][local_index],
                                    positive_scores=positive_scores,
                                    positive_class_ids=positive_ids.tolist(),
                                    image_id=image_id,
                                    grid_h=grid_h,
                                    grid_w=grid_w,
                                    output_dir=visualizations_dir,
                                    max_classes=args.visualization_max_classes,
                                )
                                visualizations_written.extend(
                                    str(Path(path).relative_to(output_dir)) for path in written
                                )
                            processed_images += 1
                        manifest_stream.flush()
                        del forward_output, capture, batch_scores, images
                        if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                            log(
                                f"progress batch={batch_index + 1}/{len(loader)} "
                                f"images={processed_images}/{len(dataset)}"
                            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if processed_images != len(dataset):
            raise RuntimeError(f"processed {processed_images}, expected {len(dataset)}")
        summary_rows = summary.finish(args.model)
        write_summary(output_dir / "summary_by_layer.csv", summary_rows)
        artifact_validation = validate_saved_artifacts(output_dir, len(dataset), depth)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        completion = {
            "status": "complete",
            "run_kind": run_kind,
            "num_images": processed_images,
            "num_layers": depth,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "num_patches": grid_h * grid_w,
            "manifest_rows": artifact_validation["manifest_rows"],
            "score_files": artifact_validation["score_files"],
            "visualization_files": len(visualizations_written),
            "elapsed_seconds": elapsed,
            "images_per_second": processed_images / elapsed,
            "peak_gpu_memory_bytes": peak_memory,
            "layer12_class_max_abs_diff_all_batches": max_layer12_class_diff,
            "layer12_patch_max_abs_diff_all_batches": max_layer12_patch_diff,
            "numerical_no_change_guard": numerical_guard,
            "artifact_validation": artifact_validation,
            "checkpoint_sha256": checkpoint_hash,
            "git_commit": source_state["commit"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(output_dir / "completion.json", completion)
        metadata["status"] = "complete"
        metadata["completed_at"] = completion["completed_at"]
        metadata["completion"] = completion
        metadata["visualizations"] = visualizations_written
        atomic_json(output_dir / "metadata.json", metadata)
        log(
            f"COMPLETE images={processed_images} elapsed={elapsed:.2f}s "
            f"score_range=[{artifact_validation['score_min']:.6f}, "
            f"{artifact_validation['score_max']:.6f}]"
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(output_dir / "failure.json", failure)
        log(f"FAILED {type(error).__name__}: {error}")
        raise


def main() -> None:
    args = parse_args()
    execute(args)


if __name__ == "__main__":
    main()
