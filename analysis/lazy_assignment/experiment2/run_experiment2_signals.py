#!/usr/bin/env python3
"""Generate immutable, per-image Experiment 2 feature/attention/CAM signals."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.common import (  # noqa: E402
    assert_output_outside_inputs,
    git_metadata,
    json_dump,
    read_json,
    sha256_file,
    timestamp,
)
from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    RAW_CAM_BACKGROUND_THRESHOLD,
    raw_final_cam_confusion,
)
from analysis.lazy_assignment.experiment2.native_cam_stages import (  # noqa: E402
    assert_native_cam_equivalent,
    decompose_native_cam_reduced,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    REGION_BACKGROUND,
    REGION_OTHER_FOREGROUND,
    REGION_TARGET,
    assign_patch_regions_from_counts,
    patch_label_counts,
)
from analysis.lazy_assignment.experiment2.signal_collector import (  # noqa: E402
    SignalCapture,
    SignalCollector,
    assert_no_change,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    VOCSemanticDataset,
)
from analysis.lazy_assignment.run_class_specific_patch_score import (  # noqa: E402
    checkpoint_payload,
    create_frozen_model,
    write_environment_manifests,
)
from analysis.lazy_assignment.score_utils import class_specific_patch_score  # noqa: E402


MODEL_TO_EXPERIMENT1_NAME = {
    "mctformer": "mctformerv2",
    "mctformer_plus": "mctformerplus",
}
STRICT_EQUIVALENCE_TOLERANCE = 1e-6
# Summing 804 float32 probabilities can accumulate several unit roundoffs even
# when the underlying softmax is exactly normalized.  This tolerance applies
# only to diagnostic row sums, never to QK/CAM/Experiment-1 equivalence.
ATTENTION_ROW_SUM_TOLERANCE = 5e-6
SIGNAL_KEYS = frozenset(
    {
        "image_id",
        "positive_class_ids",
        "grid_h",
        "grid_w",
        "patch_label_counts",
        "region_masks_rho05",
        "region_masks_rho07",
        "feature_post_scores",
        "feature_norm_scores",
        "feature_final_norm_scores",
        "qk_mean_scores",
        "qk_head_std",
        "attn_c2p_raw",
        "attn_c2p_conditional",
        "attn_patch_mass",
        "patch_logits",
        "patch_cam",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "c2p_cam",
        "final_cam",
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
        "class_logits",
        "patch_class_logits",
        "class_logits_all",
        "patch_class_logits_all",
        "raw_final_cam_confusion_t045",
        "class_token_pairwise_cosine",
        "patch_norms",
        "qk_head_region_mean_rho05",
        "qk_head_region_mean_rho07",
    }
)
RUNTIME_SOURCES = (
    "analysis/lazy_assignment/experiment2/run_experiment2_signals.py",
    "analysis/lazy_assignment/experiment2/evaluation_metrics.py",
    "analysis/lazy_assignment/experiment2/signal_collector.py",
    "analysis/lazy_assignment/experiment2/native_cam_stages.py",
    "analysis/lazy_assignment/experiment2/voc_semantic_dataset.py",
    "analysis/lazy_assignment/experiment2/patch_regions.py",
    "analysis/lazy_assignment/experiment2/common.py",
    "analysis/lazy_assignment/run_class_specific_patch_score.py",
    "analysis/lazy_assignment/score_utils.py",
    "datasets_cam.py",
    "utils.py",
    "models/mctformer.py",
    "models/mctformer_plus.py",
    "models/vit.py",
    "models/tgca.py",
)


@dataclass(frozen=True)
class ResolvedInputs:
    source_metadata_path: Path
    source_metadata_sha256: str
    result_root: Path
    experiment1_metadata: Mapping[str, object]
    checkpoint: Path
    checkpoint_sha256: str
    voc_root: Path
    list_path: Path
    expected_images: int
    model_factory_name: str


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 2 frozen-model signal generation"
    )
    parser.add_argument(
        "--model", choices=tuple(MODEL_TO_EXPERIMENT1_NAME), required=True
    )
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--allow-uncommitted-source", action="store_true")
    return parser


def validate_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None
) -> None:
    def reject(message: str) -> None:
        if parser is None:
            raise ValueError(message)
        parser.error(message)

    if args.input_size != 448:
        reject("Experiment 2 signal generation is fixed to --input-size 448")
    if args.batch_size < 1:
        reject("--batch-size must be positive")
    if args.batch_size != 8:
        reject(
            "Experiment 1 used batch size 8; Experiment 2 requires --batch-size 8 "
            "for the <1e-6 streaming feature-score reproduction gate"
        )
    if args.num_workers < 0 or args.limit < 0:
        reject("--num-workers and --limit must be non-negative")
    if args.allow_uncommitted_source and args.limit <= 0:
        reject("--allow-uncommitted-source is permitted only with --limit > 0")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _required_path(value: object, context: str, kind: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path string")
    path = Path(value).expanduser().resolve()
    exists = path.is_file() if kind == "file" else path.is_dir()
    if not exists:
        raise FileNotFoundError(f"{context} does not resolve to a {kind}: {path}")
    return path


def resolve_inputs(source_metadata_path: Path, model: str) -> ResolvedInputs:
    """Resolve the stable audit schema with explicit mismatch errors."""

    if model not in MODEL_TO_EXPERIMENT1_NAME:
        raise ValueError(f"unsupported model key {model!r}")
    path = source_metadata_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = read_json(path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("integrity_passed") is not True
    ):
        raise RuntimeError(
            "source metadata is not a completed passing Experiment 2 input audit: "
            f"status={metadata.get('status')!r}, "
            f"integrity_passed={metadata.get('integrity_passed')!r}"
        )

    sources = _require_mapping(metadata.get("sources"), "source_metadata.sources")
    source = _require_mapping(sources.get(model), f"source_metadata.sources.{model}")
    expected_factory = MODEL_TO_EXPERIMENT1_NAME[model]
    if source.get("model_cli_name") != expected_factory:
        raise ValueError(
            f"source_metadata.sources.{model}.model_cli_name must be "
            f"{expected_factory!r}, got {source.get('model_cli_name')!r}"
        )
    result_root = _required_path(
        source.get("result_root"), f"sources.{model}.result_root", "directory"
    )
    for required in ("metadata.json", "manifest.jsonl", "completion.json"):
        if not (result_root / required).is_file():
            raise FileNotFoundError(result_root / required)
    if not (result_root / "scores").is_dir():
        raise FileNotFoundError(result_root / "scores")

    checkpoint_record = _require_mapping(
        source.get("checkpoint"), f"source_metadata.sources.{model}.checkpoint"
    )
    checkpoint = _required_path(
        checkpoint_record.get("path"), f"sources.{model}.checkpoint.path", "file"
    )
    expected_hash = checkpoint_record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(
            f"sources.{model}.checkpoint.sha256 must be a SHA-256 hex string"
        )
    actual_hash = sha256_file(checkpoint)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    dataset = _require_mapping(metadata.get("dataset"), "source_metadata.dataset")
    if dataset.get("input_size") != 448 or dataset.get("patch_size") != 16:
        raise ValueError(
            "source audit must specify input_size=448 and patch_size=16; got "
            f"{dataset.get('input_size')}/{dataset.get('patch_size')}"
        )
    voc_root = _required_path(dataset.get("voc_root"), "dataset.voc_root", "directory")
    list_path = _required_path(dataset.get("list_path"), "dataset.list_path", "file")
    expected_images = dataset.get("num_images")
    if not isinstance(expected_images, int) or expected_images < 1:
        raise ValueError("dataset.num_images must be a positive integer")

    experiment1_records = _require_mapping(
        metadata.get("experiment1_metadata"), "source_metadata.experiment1_metadata"
    )
    experiment1_metadata = _require_mapping(
        experiment1_records.get(model), f"experiment1_metadata.{model}"
    )
    recorded_name = _require_mapping(
        experiment1_metadata.get("model"), f"experiment1_metadata.{model}.model"
    ).get("name")
    if recorded_name != expected_factory:
        raise ValueError(
            f"Experiment 1 metadata model mismatch: {recorded_name!r} != {expected_factory!r}"
        )
    recorded_input = _require_mapping(
        experiment1_metadata.get("input"), f"experiment1_metadata.{model}.input"
    )
    if recorded_input.get("size") != 448:
        raise ValueError("Experiment 1 source was not generated at input size 448")
    if experiment1_metadata.get("representation") != "post_block_pre_final_norm":
        raise ValueError("Experiment 1 representation is not post_block_pre_final_norm")

    return ResolvedInputs(
        source_metadata_path=path,
        source_metadata_sha256=sha256_file(path),
        result_root=result_root,
        experiment1_metadata=experiment1_metadata,
        checkpoint=checkpoint,
        checkpoint_sha256=actual_hash,
        voc_root=voc_root,
        list_path=list_path,
        expected_images=expected_images,
        model_factory_name=expected_factory,
    )


def runtime_git_state() -> dict[str, object]:
    state = git_metadata(REPO_ROOT)
    unstaged = (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        != 0
    )
    staged = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        != 0
    )
    tracked: dict[str, bool] = {}
    hashes: dict[str, str] = {}
    for relative in RUNTIME_SOURCES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        tracked[relative] = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        hashes[relative] = sha256_file(source)
    state.update(
        {
            "tracked_dirty": bool(unstaged or staged),
            "runtime_source_tracked": tracked,
            "runtime_source_sha256": hashes,
        }
    )
    return state


def enforce_source_safety(
    args: argparse.Namespace, state: Mapping[str, object]
) -> None:
    tracked = _require_mapping(
        state.get("runtime_source_tracked"), "git.runtime_source_tracked"
    )
    untracked = [name for name, value in tracked.items() if value is not True]
    unsafe = bool(state.get("tracked_dirty")) or bool(untracked)
    if unsafe and not args.allow_uncommitted_source:
        raise RuntimeError(
            "full Experiment 2 runs require a clean tracked runtime source; "
            f"tracked_dirty={state.get('tracked_dirty')}, untracked={untracked}"
        )


def _conditional(values: torch.Tensor) -> torch.Tensor:
    denominator = values.sum(dim=-1, keepdim=True)
    if torch.any(denominator <= 0):
        raise RuntimeError("cannot conditionalize a non-positive spatial mass")
    return values / denominator


def _aggregate_c2p(model: str, layers: torch.Tensor) -> torch.Tensor:
    if layers.ndim != 4 or layers.shape[0] < 1:
        raise ValueError("c2p layers must have shape [L,B,C,P]")
    if model == "mctformer":
        return layers.sum(dim=0)
    if model == "mctformer_plus":
        return layers.mean(dim=0)
    raise ValueError(model)


def diagnostic_propagated_cam(
    model: str,
    patch_cam: torch.Tensor,
    c2p_layers: torch.Tensor,
    patch_to_patch_sum: torch.Tensor,
) -> torch.Tensor:
    """Apply one diagnostic c2p aggregation with the host's native fusion rule."""

    batch, classes, grid_h, grid_w = patch_cam.shape
    patches = grid_h * grid_w
    c2p = _aggregate_c2p(model, c2p_layers)
    if tuple(c2p.shape) != (batch, classes, patches):
        raise ValueError("diagnostic c2p shape is incompatible with patch CAM")
    c1 = c2p * patch_cam.flatten(2)
    if model == "mctformer_plus":
        c1 = torch.sqrt(c1)
    return torch.matmul(patch_to_patch_sum.unsqueeze(1), c1.unsqueeze(-1)).reshape(
        batch, classes, grid_h, grid_w
    )


def region_code_maps(
    counts: np.ndarray,
    positive_class_ids: np.ndarray,
    rho: float,
    grid_size: tuple[int, int],
) -> np.ndarray:
    codes = [
        np.asarray(
            assign_patch_regions_from_counts(
                counts,
                int(class_id),
                rho=rho,
                valid_fraction=0.5,
                grid_size=grid_size,
            )["region_codes"],
            dtype=np.int8,
        ).reshape(-1)
        for class_id in positive_class_ids
    ]
    return np.stack(codes, axis=0)


def qk_head_region_means(
    qk_heads: torch.Tensor,
    positive_class_ids: np.ndarray,
    region_codes: np.ndarray,
) -> np.ndarray:
    """Summarize QK heads over target/other/background; empty regions stay NaN."""

    if qk_heads.ndim != 4:
        raise ValueError("qk_heads must have shape [L,H,C,P]")
    layers, heads, classes, patches = qk_heads.shape
    positive = np.asarray(positive_class_ids, dtype=np.int64)
    if region_codes.shape != (len(positive), patches):
        raise ValueError("region code shape is incompatible with positive classes/QK")
    if np.any(positive < 0) or np.any(positive >= classes):
        raise ValueError("positive class ID outside QK class axis")
    selected = qk_heads.index_select(
        2, torch.as_tensor(positive, device=qk_heads.device)
    ).float()
    result = torch.full(
        (layers, heads, len(positive), 3),
        float("nan"),
        dtype=torch.float32,
        device=qk_heads.device,
    )
    for local_class in range(len(positive)):
        codes = torch.as_tensor(region_codes[local_class], device=qk_heads.device)
        for output_index, code in enumerate(
            (REGION_TARGET, REGION_OTHER_FOREGROUND, REGION_BACKGROUND)
        ):
            mask = codes == code
            if bool(mask.any()):
                result[:, :, local_class, output_index] = selected[
                    :, :, local_class, mask
                ].mean(dim=-1)
    return result.cpu().numpy()


def _index_class(
    tensor: torch.Tensor, class_ids: np.ndarray, dimension: int
) -> torch.Tensor:
    index = torch.as_tensor(class_ids, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dimension, index)


def _numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


def validate_signal_payload(payload: Mapping[str, np.ndarray]) -> None:
    missing = SIGNAL_KEYS.difference(payload)
    extra = set(payload).difference(SIGNAL_KEYS)
    if missing or extra:
        raise ValueError(
            f"signal schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    image_id = np.asarray(payload["image_id"])
    positive = np.asarray(payload["positive_class_ids"])
    grid_h = int(np.asarray(payload["grid_h"]).item())
    grid_w = int(np.asarray(payload["grid_w"]).item())
    if image_id.ndim != 0 or not isinstance(image_id.item(), str):
        raise TypeError("image_id must be a scalar string")
    if positive.ndim != 1 or positive.dtype != np.int64 or not len(positive):
        raise TypeError("positive_class_ids must be a non-empty int64 vector")
    classes, patches = len(positive), grid_h * grid_w
    counts = np.asarray(payload["patch_label_counts"])
    if counts.dtype != np.uint16 or counts.shape != (patches, 22):
        raise TypeError("patch_label_counts must be uint16[P,22]")
    if not np.all(counts.sum(axis=-1) == 256):
        raise ValueError("every patch_label_counts row must contain 256 pixels")
    for name in ("region_masks_rho05", "region_masks_rho07"):
        value = np.asarray(payload[name])
        if value.dtype != np.int8 or value.shape != (classes, patches):
            raise TypeError(f"{name} must be int8[K,P]")
        if np.any((value < 0) | (value > 4)):
            raise ValueError(f"{name} contains an invalid region code")

    layer_class_patch = (
        "feature_post_scores",
        "feature_norm_scores",
        "qk_mean_scores",
        "qk_head_std",
        "attn_c2p_raw",
        "attn_c2p_conditional",
    )
    for name in layer_class_patch:
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != (12, classes, patches):
            raise TypeError(f"{name} must be float32[12,K,P]")
    if np.asarray(payload["feature_final_norm_scores"]).shape != (classes, patches):
        raise ValueError("feature_final_norm_scores must be [K,P]")
    if np.asarray(payload["attn_patch_mass"]).shape != (12, classes):
        raise ValueError("attn_patch_mass must be [12,K]")
    if np.asarray(payload["attn_patch_mass"]).dtype != np.float32:
        raise TypeError("attn_patch_mass must be float32")

    class_patch_maps = (
        "feature_final_norm_scores",
        "patch_logits",
        "patch_cam",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "c2p_cam",
        "final_cam",
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
    )
    for name in class_patch_maps:
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != (classes, patches):
            raise TypeError(f"{name} must be float32[K,P]")
    for name in ("class_logits", "patch_class_logits"):
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != (classes,):
            raise TypeError(f"{name} must be float32[K]")
    for name in ("class_logits_all", "patch_class_logits_all"):
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != (20,):
            raise TypeError(f"{name} must be float32[20]")
    np.testing.assert_allclose(
        payload["class_logits_all"][positive], payload["class_logits"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        payload["patch_class_logits_all"][positive],
        payload["patch_class_logits"],
        rtol=0,
        atol=0,
    )
    confusion = np.asarray(payload["raw_final_cam_confusion_t045"])
    if confusion.dtype != np.int64 or confusion.shape != (21, 21):
        raise TypeError("raw_final_cam_confusion_t045 must be int64[21,21]")
    # Every 16x16 patch contributes exactly 256 semantic-mask pixels.  Derive
    # the transformed area from the declared grid so synthetic schema tests
    # remain valid while production 28x28 inputs still resolve to 448x448.
    expected_valid_pixels = grid_h * grid_w * 256 - int(counts[:, 21].sum())
    if (
        np.any(confusion < 0)
        or int(confusion.sum()) <= 0
        or int(confusion.sum()) != expected_valid_pixels
    ):
        raise ValueError("raw final-CAM confusion must contain non-negative pixels")
    pairwise = np.asarray(payload["class_token_pairwise_cosine"])
    if pairwise.dtype != np.float32 or pairwise.shape != (12, classes, classes):
        raise TypeError("class_token_pairwise_cosine must be float32[12,K,K]")
    norms = np.asarray(payload["patch_norms"])
    if norms.dtype != np.float32 or norms.shape != (12, patches):
        raise TypeError("patch_norms must be float32[12,P]")
    for name in ("qk_head_region_mean_rho05", "qk_head_region_mean_rho07"):
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.ndim != 4:
            raise TypeError(f"{name} must be float32[L,H,K,3]")
        if value.shape[0] != 12 or value.shape[2:] != (classes, 3):
            raise ValueError(f"{name} has invalid shape {value.shape}")
        if np.isinf(value).any():
            raise ValueError(f"{name} contains infinity")

    for name, value in payload.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and name not in {
            "qk_head_region_mean_rho05",
            "qk_head_region_mean_rho07",
        }:
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains a non-finite value")


def save_signal(path: Path, payload: Mapping[str, np.ndarray]) -> str:
    validate_signal_payload(payload)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)
    return sha256_file(path)


def _source_feature_scores(
    result_root: Path,
    image_id: str,
    positive_class_ids: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    path = result_root / "scores" / f"{image_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as artifact:
        if str(artifact["image_id"].item()) != image_id:
            raise RuntimeError(f"Experiment 1 image ID mismatch in {path}")
        if not np.array_equal(artifact["positive_class_ids"], positive_class_ids):
            raise RuntimeError(f"Experiment 1 positive classes mismatch in {path}")
        if not np.array_equal(artifact["saved_class_ids"], positive_class_ids):
            raise RuntimeError(f"Experiment 1 saved classes mismatch in {path}")
        scores = np.asarray(artifact["scores_raw"], dtype=np.float32)
        grid = (int(artifact["grid_h"].item()), int(artifact["grid_w"].item()))
    if scores.shape != expected_shape or grid != (28, 28):
        raise RuntimeError(
            f"Experiment 1 score shape/grid mismatch for {image_id}: {scores.shape}/{grid}"
        )
    return scores


def _make_payload(
    model_key: str,
    model: torch.nn.Module,
    image_id: str,
    positive_class_ids: np.ndarray,
    mask: torch.Tensor,
    capture: SignalCapture,
    batch_index: int,
    patch_logits_batch: torch.Tensor,
    stages: Mapping[str, torch.Tensor],
    diagnostic_batches: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    positive = np.asarray(positive_class_ids, dtype=np.int64)
    counts = patch_label_counts(mask, patch_size=16)
    region05 = region_code_maps(counts, positive, 0.5, (28, 28))
    region07 = region_code_maps(counts, positive, 0.7, (28, 28))

    def lcp(tensor: torch.Tensor) -> torch.Tensor:
        return _index_class(tensor[:, batch_index], positive, 1)

    qk_heads = capture.qk_c2p_heads[:, batch_index]
    qk05 = qk_head_region_means(qk_heads, positive, region05)
    qk07 = qk_head_region_means(qk_heads, positive, region07)
    last_tokens = torch.cat(
        (
            capture.last_class_tokens[batch_index],
            capture.last_patch_tokens[batch_index],
        ),
        dim=0,
    ).unsqueeze(0)
    # This inherited final LayerNorm is not called by either native CAM host.
    final_normalized = model.norm(last_tokens)
    final_norm_scores = class_specific_patch_score(
        final_normalized[:, :20], final_normalized[:, 20:]
    )[0]

    patch_logits = patch_logits_batch[batch_index].flatten(1)
    patch_cam = stages["patch_cam"][batch_index].flatten(1)
    official_raw = stages["official_c2p_flat"][batch_index]
    c2p_cam = stages["class_attention_cam"][batch_index].flatten(1)
    final_cam = stages["final_cam"][batch_index].flatten(1)
    positive_final_cam = _index_class(final_cam, positive, 0)
    c2p_layers = capture.attn_c2p_raw
    mid3_raw_batch = _aggregate_c2p(model_key, c2p_layers[3:6])

    patch_class_logits = (
        model.avgpool(patch_logits_batch).squeeze(-1).squeeze(-1)
        if model_key == "mctformer"
        else model.gwrp(patch_logits_batch)
    )
    class_logits = capture.last_class_tokens.mean(dim=-1)
    pairwise = capture.class_token_pairwise_cosine[:, batch_index]
    pairwise = _index_class(_index_class(pairwise, positive, 1), positive, 2)

    diagnostic = {}
    for name, batch_value in diagnostic_batches.items():
        value = batch_value[batch_index].flatten(1)
        diagnostic[name] = _numpy_float32(_index_class(value, positive, 0))

    payload: dict[str, np.ndarray] = {
        "image_id": np.asarray(image_id),
        "positive_class_ids": positive,
        "grid_h": np.asarray(28, dtype=np.int32),
        "grid_w": np.asarray(28, dtype=np.int32),
        "patch_label_counts": counts.astype(np.uint16, copy=False),
        "region_masks_rho05": region05,
        "region_masks_rho07": region07,
        "feature_post_scores": _numpy_float32(lcp(capture.feature_post_scores)),
        "feature_norm_scores": _numpy_float32(lcp(capture.feature_norm_scores)),
        "feature_final_norm_scores": _numpy_float32(
            _index_class(final_norm_scores, positive, 0)
        ),
        "qk_mean_scores": _numpy_float32(lcp(capture.qk_mean_scores)),
        "qk_head_std": _numpy_float32(lcp(capture.qk_head_std)),
        "attn_c2p_raw": _numpy_float32(lcp(capture.attn_c2p_raw)),
        "attn_c2p_conditional": _numpy_float32(lcp(capture.attn_c2p_conditional)),
        "attn_patch_mass": _numpy_float32(
            _index_class(capture.attn_patch_mass[:, batch_index], positive, 1)
        ),
        "patch_logits": _numpy_float32(_index_class(patch_logits, positive, 0)),
        "patch_cam": _numpy_float32(_index_class(patch_cam, positive, 0)),
        "attn_official_raw": _numpy_float32(_index_class(official_raw, positive, 0)),
        "attn_official_conditional": _numpy_float32(
            _index_class(_conditional(official_raw), positive, 0)
        ),
        "attn_mid3_raw": _numpy_float32(
            _index_class(mid3_raw_batch[batch_index], positive, 0)
        ),
        "attn_mid3_conditional": _numpy_float32(
            _index_class(_conditional(mid3_raw_batch[batch_index]), positive, 0)
        ),
        "c2p_cam": _numpy_float32(_index_class(c2p_cam, positive, 0)),
        "final_cam": _numpy_float32(_index_class(final_cam, positive, 0)),
        "class_logits": _numpy_float32(
            _index_class(class_logits[batch_index], positive, 0)
        ),
        "patch_class_logits": _numpy_float32(
            _index_class(patch_class_logits[batch_index], positive, 0)
        ),
        "class_logits_all": _numpy_float32(class_logits[batch_index]),
        "patch_class_logits_all": _numpy_float32(patch_class_logits[batch_index]),
        "raw_final_cam_confusion_t045": raw_final_cam_confusion(
            positive_final_cam,
            positive,
            mask,
        ).astype(np.int64, copy=False),
        "class_token_pairwise_cosine": _numpy_float32(pairwise),
        "patch_norms": _numpy_float32(capture.patch_norms[:, batch_index]),
        "qk_head_region_mean_rho05": qk05.astype(np.float32, copy=False),
        "qk_head_region_mean_rho07": qk07.astype(np.float32, copy=False),
        **diagnostic,
    }
    validate_signal_payload(payload)
    return payload


def _environment(device: torch.device) -> dict[str, object]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
    }


def _model_from_inputs(args: argparse.Namespace, inputs: ResolvedInputs):
    payload = checkpoint_payload(inputs.checkpoint)
    factory_args = SimpleNamespace(model=inputs.model_factory_name, input_size=448)
    model, configuration, load_info = create_frozen_model(factory_args, payload)
    return model, payload, configuration, load_info


def execute(args: argparse.Namespace) -> None:
    validate_args(args)
    if os.environ.get("CONDA_DEFAULT_ENV") != "tgca-repro":
        raise RuntimeError(
            "Experiment 2 signals must run in tgca-repro; active="
            f"{os.environ.get('CONDA_DEFAULT_ENV')!r}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    inputs = resolve_inputs(args.source_metadata, args.model)
    assert_output_outside_inputs(
        output_dir,
        [
            inputs.source_metadata_path,
            inputs.result_root,
            inputs.checkpoint,
            inputs.voc_root,
        ],
    )
    git_state = runtime_git_state()
    enforce_source_safety(args, git_state)

    output_dir.mkdir(parents=True, exist_ok=False)
    signal_dir = output_dir / "signals"
    signal_dir.mkdir()
    log = RunLog(output_dir / "run.log")
    command = shlex.join([sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    write_environment_manifests(output_dir)

    started = time.perf_counter()
    metadata: dict[str, object] = {
        "status": "running",
        "analysis": "experiment2_semantic_ownership_signals",
        "model": args.model,
        "run_kind": "smoke" if args.limit else "full",
        "execution": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "limit": args.limit,
            "allow_uncommitted_source": bool(args.allow_uncommitted_source),
        },
        "source_metadata": str(inputs.source_metadata_path),
        "source_metadata_sha256": inputs.source_metadata_sha256,
        "experiment1_result_root": str(inputs.result_root),
        "checkpoint": {
            "path": str(inputs.checkpoint),
            "sha256": inputs.checkpoint_sha256,
        },
        "dataset": {
            "voc_root": str(inputs.voc_root),
            "list_path": str(inputs.list_path),
            "input_size": 448,
            "patch_size": 16,
            "expected_images": inputs.expected_images,
            "transform": "bicubic short-side Resize(512) -> CenterCrop(448) -> ToTensor -> ImageNet Normalize; matched nearest-neighbor semantic-mask geometry",
        },
        "signal_schema": sorted(SIGNAL_KEYS),
        "region_code_order": ["target", "other_fg", "background", "mixed", "void"],
        "qk_head_region_last_axis": ["target", "other_fg", "background"],
        "numerical_tolerances": {
            "experiment1_feature_max_abs_diff_strictly_below": (
                STRICT_EQUIVALENCE_TOLERANCE
            ),
            "qk_attention_max_abs_diff_strictly_below": (STRICT_EQUIVALENCE_TOLERANCE),
            "native_cam_max_abs_diff_strictly_below": (STRICT_EQUIVALENCE_TOLERANCE),
            "float32_attention_row_sum_max_abs_error_at_most": (
                ATTENTION_ROW_SUM_TOLERANCE
            ),
        },
        "feature_final_norm": {
            "status": "analysis_only_non_native",
            "definition": "inherited model.norm applied to L12 post-block tokens",
            "native_hosts_call_this_module": False,
        },
        "cam": {
            "mctformer": "last3 c2p sum, no sqrt; all-layer p2p sum",
            "mctformer_plus": "last3 c2p mean, sqrt; all-layer p2p sum",
            "conditional_attention_used_in_native_cam": False,
        },
        "classification_logits": {
            "class_logits": "mean over L12 class-token embedding dimension",
            "mctformer_patch_class_logits": "native AdaptiveAvgPool2d over raw patch-head logits",
            "mctformer_plus_patch_class_logits": "native GWRP over raw patch-head logits",
        },
        "raw_final_cam_evaluation": {
            "geometry": "bilinear 28x28 to transformed 448x448 crop, align_corners=False",
            "normalization": "per-active-class min-max with epsilon 1e-8",
            "background_threshold": RAW_CAM_BACKGROUND_THRESHOLD,
            "void_id": 255,
            "active_classes": "ground-truth image-level positive classes",
        },
        "git": git_state,
        "environment": None,
        "command": command,
        "started_at": timestamp(),
    }
    json_dump(output_dir / "metadata.json", metadata)

    head_outputs: list[torch.Tensor] = []
    head_handle = None
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        metadata["environment"] = _environment(device)

        model, checkpoint, configuration, load_info = _model_from_inputs(args, inputs)
        model.to(device).eval()
        if len(model.blocks) != 12 or int(model.num_classes) != 20:
            raise RuntimeError(
                "Experiment 2 requires the native 12-layer, 20-class host"
            )
        metadata["model_configuration"] = configuration
        metadata["strict_checkpoint_load"] = load_info
        metadata["checkpoint_epoch"] = (
            checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
        )

        requested_images = args.limit if args.limit else inputs.expected_images
        # Experiment 1 was evaluated in ordered batches of eight.  A limited
        # smoke still evaluates the complete surrounding batch and only saves
        # the requested prefix, preserving the exact CUDA numerical path.
        context_images = (
            min(
                inputs.expected_images,
                int(np.ceil(requested_images / args.batch_size)) * args.batch_size,
            )
            if args.limit
            else inputs.expected_images
        )
        dataset = VOCSemanticDataset(
            inputs.voc_root,
            inputs.list_path,
            input_size=448,
            limit=context_images if args.limit else 0,
        )
        expected_processed = min(requested_images, inputs.expected_images)
        if len(dataset) != context_images:
            raise RuntimeError(
                f"dataset has {len(dataset)} context images, expected {context_images}"
            )
        metadata["execution"]["requested_images"] = expected_processed
        metadata["execution"]["batch_context_images"] = context_images

        # First-image read-only hook guard.  It compares native raw-grid CAMs.
        guard_image = dataset[0]["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            guard_plain = model(guard_image)
            with SignalCollector(model, num_classes=20) as guard_collector:
                guard_collector.clear(expected_num_patches=784)
                guard_hooked = model(guard_image)
                guard_capture = guard_collector.consume()
        guard_difference = assert_no_change(guard_plain, guard_hooked, tolerance=0.0)
        if (
            float(guard_capture.qk_attention_max_abs_diff.max())
            >= STRICT_EQUIVALENCE_TOLERANCE
        ):
            raise RuntimeError("first-image QK reconstruction guard failed")
        if (
            float(guard_capture.attention_row_sum_max_abs_error.max())
            > ATTENTION_ROW_SUM_TOLERANCE
        ):
            raise RuntimeError("first-image attention row-sum guard failed")
        metadata["first_image_no_change_guard"] = {
            "native_cam_max_abs_diff": guard_difference,
            "qk_attention_max_abs_diff": float(
                guard_capture.qk_attention_max_abs_diff.max()
            ),
            "attention_row_sum_max_abs_error": float(
                guard_capture.attention_row_sum_max_abs_error.max()
            ),
            "attention_row_sum_tolerance": ATTENTION_ROW_SUM_TOLERANCE,
            "passed": True,
        }
        del guard_plain, guard_hooked, guard_capture, guard_image
        if device.type == "cuda":
            torch.cuda.empty_cache()

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda" and not args.no_pin_memory),
            drop_last=False,
        )

        def head_hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("native patch head output must be a tensor")
            head_outputs.append(output.detach())
            return None

        head_handle = model.head.register_forward_hook(head_hook)
        manifest_path = output_dir / "manifest.jsonl"
        processed = 0
        maximum_exp1_difference = 0.0
        maximum_native_cam_difference = 0.0
        maximum_qk_difference = 0.0
        maximum_row_error = 0.0
        maximum_conditional_row_error = 0.0
        with manifest_path.open("x", encoding="utf-8") as manifest_stream:
            with SignalCollector(model, num_classes=20) as collector:
                with torch.inference_mode():
                    for batch_number, batch in enumerate(loader):
                        images = batch["image"].to(
                            device,
                            non_blocking=(
                                device.type == "cuda" and not args.no_pin_memory
                            ),
                        )
                        labels = batch["label"].cpu().numpy()
                        masks = batch["mask"]
                        image_ids = list(batch["name"])
                        head_outputs.clear()
                        collector.clear(expected_num_patches=784)
                        native_cam = model(images)
                        capture = collector.consume()
                        if len(head_outputs) != 1:
                            raise RuntimeError(
                                f"native patch head fired {len(head_outputs)} times"
                            )
                        patch_logits_batch = head_outputs[0]
                        stages = decompose_native_cam_reduced(
                            args.model,
                            patch_logits_batch,
                            capture.attn_c2p_raw,
                            capture.patch_to_patch_sum,
                            num_classes=20,
                        )
                        diagnostic_batches = {
                            name: diagnostic_propagated_cam(
                                args.model,
                                stages["patch_cam"],
                                capture.attn_c2p_raw[layer_slice],
                                capture.patch_to_patch_sum,
                            )
                            for name, layer_slice in {
                                "diagnostic_c2p_cam_l10": slice(9, 10),
                                "diagnostic_c2p_cam_l11": slice(10, 11),
                                "diagnostic_c2p_cam_l12": slice(11, 12),
                                "diagnostic_c2p_cam_mid3": slice(3, 6),
                            }.items()
                        }
                        native_difference = assert_native_cam_equivalent(
                            stages, native_cam, tolerance=1e-6
                        )
                        if native_difference >= 1e-6:
                            raise RuntimeError(
                                "native CAM reproduction must have max_abs_diff < 1e-6; "
                                f"got {native_difference}"
                            )
                        maximum_native_cam_difference = max(
                            maximum_native_cam_difference, native_difference
                        )
                        batch_qk = float(capture.qk_attention_max_abs_diff.max())
                        batch_rows = float(
                            capture.attention_row_sum_max_abs_error.max()
                        )
                        conditional_rows = float(
                            (capture.attn_c2p_conditional.float().sum(dim=-1) - 1.0)
                            .abs()
                            .max()
                            .item()
                        )
                        if (
                            batch_qk >= STRICT_EQUIVALENCE_TOLERANCE
                            or batch_rows > ATTENTION_ROW_SUM_TOLERANCE
                            or conditional_rows > ATTENTION_ROW_SUM_TOLERANCE
                        ):
                            raise RuntimeError(
                                "attention guard failed: "
                                f"qk={batch_qk}, rows={batch_rows}, "
                                f"conditional_rows={conditional_rows}"
                            )
                        maximum_qk_difference = max(maximum_qk_difference, batch_qk)
                        maximum_row_error = max(maximum_row_error, batch_rows)
                        maximum_conditional_row_error = max(
                            maximum_conditional_row_error, conditional_rows
                        )

                        for local_index, image_id in enumerate(image_ids):
                            if processed >= expected_processed:
                                break
                            positive = np.flatnonzero(labels[local_index] > 0).astype(
                                np.int64
                            )
                            if not len(positive):
                                raise RuntimeError(
                                    f"image {image_id} has no positive class"
                                )
                            payload = _make_payload(
                                args.model,
                                model,
                                image_id,
                                positive,
                                masks[local_index],
                                capture,
                                local_index,
                                patch_logits_batch,
                                stages,
                                diagnostic_batches,
                            )
                            source_scores = _source_feature_scores(
                                inputs.result_root,
                                image_id,
                                positive,
                                (12, len(positive), 784),
                            )
                            difference = float(
                                np.max(
                                    np.abs(
                                        payload["feature_post_scores"] - source_scores
                                    )
                                )
                            )
                            if difference >= 1e-6:
                                raise RuntimeError(
                                    f"Experiment 1 reproduction failed for {image_id}: "
                                    f"max_abs_diff={difference}"
                                )
                            maximum_exp1_difference = max(
                                maximum_exp1_difference, difference
                            )
                            signal_path = signal_dir / f"{image_id}.npz"
                            digest = save_signal(signal_path, payload)
                            record = {
                                "image_id": image_id,
                                "positive_class_ids": positive.tolist(),
                                "grid_h": 28,
                                "grid_w": 28,
                                "num_layers": 12,
                                "num_patches": 784,
                                "signal_path": str(signal_path.relative_to(output_dir)),
                                "artifact_sha256": digest,
                            }
                            manifest_stream.write(
                                json.dumps(record, sort_keys=True, allow_nan=False)
                                + "\n"
                            )
                            manifest_stream.flush()
                            processed += 1
                        log(
                            f"batch={batch_number + 1} processed={processed}/"
                            f"{expected_processed} native_diff={native_difference:.3g} "
                            f"qk_diff={batch_qk:.3g}"
                        )
                        del (
                            native_cam,
                            capture,
                            stages,
                            diagnostic_batches,
                            patch_logits_batch,
                            images,
                        )

        if processed != expected_processed:
            raise RuntimeError(
                f"processed {processed} images, expected {expected_processed}"
            )
        if len(list(signal_dir.glob("*.npz"))) != processed:
            raise RuntimeError("signal artifact count does not match processed images")
        if sha256_file(inputs.checkpoint) != inputs.checkpoint_sha256:
            raise RuntimeError("checkpoint changed during signal generation")
        if sha256_file(inputs.source_metadata_path) != inputs.source_metadata_sha256:
            raise RuntimeError("source metadata changed during signal generation")

        metadata.update(
            {
                "status": "complete",
                "processed_images": processed,
                "experiment1_feature_post_max_abs_diff": maximum_exp1_difference,
                "native_cam_max_abs_diff": maximum_native_cam_difference,
                "qk_attention_max_abs_diff": maximum_qk_difference,
                "attention_row_sum_max_abs_error": maximum_row_error,
                "conditional_attention_row_sum_max_abs_error": (
                    maximum_conditional_row_error
                ),
                "attention_row_sum_tolerance": ATTENTION_ROW_SUM_TOLERANCE,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at": timestamp(),
            }
        )
        json_dump(output_dir / "metadata.json", metadata)
        json_dump(
            output_dir / "completion.json",
            {
                "status": "complete",
                "run_kind": metadata["run_kind"],
                "model": args.model,
                "num_images": processed,
                "manifest": "manifest.jsonl",
                "signals": "signals",
                "experiment1_feature_post_max_abs_diff": maximum_exp1_difference,
                "native_cam_max_abs_diff": maximum_native_cam_difference,
                "qk_attention_max_abs_diff": maximum_qk_difference,
                "attention_row_sum_max_abs_error": maximum_row_error,
                "conditional_attention_row_sum_max_abs_error": (
                    maximum_conditional_row_error
                ),
                "completed_at": timestamp(),
            },
        )
        log(
            f"complete images={processed} exp1_diff={maximum_exp1_difference:.3g} "
            f"native_cam_diff={maximum_native_cam_difference:.3g}"
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "failed_at": timestamp(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": repr(error),
            }
        )
        json_dump(output_dir / "metadata.json", metadata)
        json_dump(
            output_dir / "failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_at": timestamp(),
            },
        )
        log(f"FAILED: {error!r}")
        raise
    finally:
        if head_handle is not None:
            head_handle.remove()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
