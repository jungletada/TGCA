#!/usr/bin/env python3
"""Run Validation A with two frozen deterministic passes and reduced outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.signal_collector import (  # noqa: E402
    SignalCollector,
    assert_no_change,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    EXPECTED_PATCHES,
    STRICT_TOLERANCE,
    enforce_production_source,
    json_dump,
    runtime_source_state,
    sha256_file,
)
from analysis.lazy_assignment.experiment3.presence_axis import (  # noqa: E402
    CrossFittedDirectionRegistry,
    TwoFoldPresenceAccumulator,
    axis_removed_cosine_maps,
    build_two_fold_split,
    normalized_all_ones_direction,
    sha256_two_fold,
    token_pair_axis_metrics,
)
from analysis.lazy_assignment.experiment3.runtime import (  # noqa: E402
    assert_inputs_unchanged,
    create_runtime_model,
    finish_metadata,
    initialize_run_directory,
    load_source_signal,
    make_dataset_and_loader,
    resolve_runtime_inputs,
    reload_npz_checked,
    runtime_environment,
    save_npz_atomic,
)


RUNTIME_SOURCES = (
    "analysis/lazy_assignment/experiment3/common.py",
    "analysis/lazy_assignment/experiment3/presence_axis.py",
    "analysis/lazy_assignment/experiment3/runtime.py",
    "analysis/lazy_assignment/experiment3/run_presence_axis_analysis.py",
    "analysis/lazy_assignment/experiment2/signal_collector.py",
    "analysis/lazy_assignment/experiment2/voc_semantic_dataset.py",
    "analysis/lazy_assignment/experiment2/run_experiment2_signals.py",
    "analysis/lazy_assignment/run_class_specific_patch_score.py",
    "analysis/lazy_assignment/score_utils.py",
    "models/mctformer.py",
    "models/mctformer_plus.py",
    "models/vit.py",
    "models/tgca.py",
)
CONTROL_LAYERS_ONE_BASED = (4, 5, 9, 10, 11, 12)
CONTROL_LAYER_INDICES = tuple(value - 1 for value in CONTROL_LAYERS_ONE_BASED)
DERIVED_KEYS = {
    "image_id",
    "eval_fold",
    "fit_fold",
    "positive_class_ids",
    "control_layer_ids",
    "raw_control_all",
    "class_removed_control_all",
    "patch_removed_control_all",
    "both_removed_control_all",
    "shared_both_removed_control_all",
    "feature_norm_control_all",
    "qk_control_all",
    "attention_conditional_control_all",
    "class_removed_scores",
    "patch_removed_scores",
    "both_removed_scores",
    "shared_both_removed_scores",
    "class_coefficients",
    "class_axis_energy",
    "class_norms",
    "class_residual_norms",
    "patch_coefficient_mean",
    "patch_coefficient_std",
    "patch_axis_energy_mean",
    "patch_axis_energy_std",
    "raw_pair_cosine_all",
    "residual_pair_cosine_all",
    "pair_axis_dot_all",
    "pair_residual_dot_all",
    "heldout_projection_all",
    "shared_axis_energy_all",
    "class_logits_all",
    "source_signal_sha256",
}


@dataclass(frozen=True)
class PresenceBatchCapture:
    raw_scores: torch.Tensor
    class_removed_scores: torch.Tensor
    patch_removed_scores: torch.Tensor
    both_removed_scores: torch.Tensor
    shared_both_removed_scores: torch.Tensor
    class_coefficients: torch.Tensor
    class_axis_energy: torch.Tensor
    class_norms: torch.Tensor
    class_residual_norms: torch.Tensor
    patch_coefficient_mean: torch.Tensor
    patch_coefficient_std: torch.Tensor
    patch_axis_energy_mean: torch.Tensor
    patch_axis_energy_std: torch.Tensor
    raw_pair_cosine: torch.Tensor
    residual_pair_cosine: torch.Tensor
    pair_axis_dot: torch.Tensor
    pair_residual_dot: torch.Tensor
    heldout_projection: torch.Tensor
    shared_axis_energy: torch.Tensor
    final_logit_identity_error: torch.Tensor
    decomposition_reconstruction_error: torch.Tensor
    residual_orthogonality_error: torch.Tensor
    residual_cosine_identity_error: torch.Tensor


class PostBlockClassTokenCollector:
    """Observe all post-block class tokens without retaining patch tokens."""

    def __init__(self, model: torch.nn.Module, num_classes: int = EXPECTED_CLASSES):
        self.model = model
        self.num_classes = num_classes
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.values: list[Optional[torch.Tensor]] = []
        self.active = False

    def register(self):
        if self.handles:
            raise RuntimeError("hooks already registered")
        for layer, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self.active:
                return None
            if not isinstance(output, (tuple, list)) or not isinstance(
                output[0], torch.Tensor
            ):
                raise TypeError("native block must return token tensor first")
            tokens = output[0]
            if (
                tokens.ndim != 3
                or tokens.shape[1] != EXPECTED_CLASSES + EXPECTED_PATCHES
            ):
                raise RuntimeError(
                    f"unexpected native token shape {tuple(tokens.shape)}"
                )
            if self.values[layer] is not None:
                raise RuntimeError(f"block {layer} fired twice")
            self.values[layer] = tokens[:, : self.num_classes].detach()
            return None

        return hook

    def clear(self) -> None:
        if not self.handles or self.active:
            raise RuntimeError("collector is unregistered or previous capture pending")
        self.values = [None] * len(self.model.blocks)
        self.active = True

    def consume(self) -> torch.Tensor:
        if not self.active or any(value is None for value in self.values):
            raise RuntimeError("incomplete class-token capture")
        result = torch.stack(self.values)  # type: ignore[arg-type]
        self.active = False
        self.values = []
        return result

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values = []
        self.active = False

    def __enter__(self):
        return self.register()

    def __exit__(self, exc_type, exc_value, tb):
        self.remove()


def _batched_direction_parts(
    tokens: torch.Tensor, directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [B,N,D] tokens using a different unit direction per image."""

    if tokens.ndim != 3 or directions.shape != (tokens.shape[0], tokens.shape[2]):
        raise ValueError("batched directions do not match tokens")
    directions = F.normalize(directions.float(), p=2, dim=-1, eps=1e-12)
    values = tokens.float()
    coefficients = torch.einsum("bnd,bd->bn", values, directions)
    residual = values - coefficients.unsqueeze(-1) * directions.unsqueeze(1)
    return coefficients, residual


class PresenceAxisCollector:
    """Reduce fixed/shared direction post-block statistics inside hooks."""

    def __init__(self, model: torch.nn.Module, registry: CrossFittedDirectionRegistry):
        self.model = model
        self.registry = registry
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.active = False
        self.eval_folds: Optional[np.ndarray] = None
        self.storage: dict[str, list[Optional[torch.Tensor]]] = {}

    def register(self):
        if self.handles:
            raise RuntimeError("presence hooks already registered")
        for layer, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_hook(self._hook(layer)))
        return self

    def clear(self, eval_folds: Sequence[int]) -> None:
        if not self.handles or self.active:
            raise RuntimeError("collector is unregistered or previous capture pending")
        folds = np.asarray(eval_folds, dtype=np.int64)
        if folds.ndim != 1 or not np.isin(folds, (0, 1)).all():
            raise ValueError("eval_folds must be a rank-one 0/1 vector")
        self.eval_folds = folds
        fields = PresenceBatchCapture.__dataclass_fields__
        self.storage = {name: [None] * len(self.model.blocks) for name in fields}
        self.active = True

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self.active:
                return None
            if self.eval_folds is None:
                raise RuntimeError("missing eval fold assignments")
            if not isinstance(output, (tuple, list)) or not isinstance(
                output[0], torch.Tensor
            ):
                raise TypeError("native block must return token tensor first")
            tokens = output[0]
            if (
                tokens.ndim != 3
                or tokens.shape[1] != EXPECTED_CLASSES + EXPECTED_PATCHES
            ):
                raise RuntimeError(f"unexpected token shape {tuple(tokens.shape)}")
            classes = tokens[:, :EXPECTED_CLASSES]
            patches = tokens[:, EXPECTED_CLASSES:]
            width = int(tokens.shape[-1])
            fixed = normalized_all_ones_direction(
                width, device=tokens.device, dtype=torch.float32
            )
            maps = axis_removed_cosine_maps(classes, patches, fixed)
            pairs = token_pair_axis_metrics(classes, fixed)

            fit_folds = 1 - self.eval_folds
            direction_np = self.registry.shared_directions[fit_folds, layer]
            mean_np = self.registry.fit_means[fit_folds, layer]
            directions = torch.as_tensor(
                direction_np, device=tokens.device, dtype=torch.float32
            )
            fit_means = torch.as_tensor(
                mean_np, device=tokens.device, dtype=torch.float32
            )
            shared_class_coeff, shared_classes = _batched_direction_parts(
                classes, directions
            )
            shared_patch_coeff, shared_patches = _batched_direction_parts(
                patches, directions
            )
            shared_class_norm = shared_classes.norm(dim=-1).clamp_min(1e-12)
            shared_patch_norm = shared_patches.norm(dim=-1).clamp_min(1e-12)
            shared_dot = torch.einsum("bcd,bpd->bcp", shared_classes, shared_patches)
            shared_scores = shared_dot / (
                shared_class_norm.unsqueeze(-1) * shared_patch_norm.unsqueeze(1)
            )
            centered = classes.float() - fit_means
            heldout = torch.einsum("bcd,bd->bc", centered, directions)
            shared_energy = shared_class_coeff.square() / classes.float().norm(
                dim=-1
            ).square().clamp_min(1e-12)

            class_energy = maps.class_coefficients.square() / maps.class_norms.square()
            patch_energy = maps.patch_coefficients.square() / maps.patch_norms.square()
            # Check the exact readout identity and token decomposition in
            # float64. Native float32 logits are independently compared with
            # immutable Experiment 2 below, so CUDA reduction-order error is
            # not conflated with the mathematical direction identity.
            classes64 = classes.double()
            fixed64 = normalized_all_ones_direction(
                width, device=tokens.device, dtype=torch.float64
            )
            coefficient64 = torch.einsum("bcd,d->bc", classes64, fixed64)
            residual64 = classes64 - coefficient64.unsqueeze(-1) * fixed64
            reconstructed64 = residual64 + coefficient64.unsqueeze(-1) * fixed64
            identity = (
                (classes64.mean(dim=-1) - coefficient64 / math.sqrt(width))
                .abs()
                .amax(dim=-1)
            )
            reconstruction_error = (
                (reconstructed64 - classes64).abs().flatten(1).amax(dim=-1)
            )
            orthogonality_error = (
                torch.einsum("bcd,d->bc", residual64, fixed64).abs().amax(dim=-1)
            )
            direct_residual_cosine = maps.direct_residual_dot / (
                maps.residual_class_norms.unsqueeze(-1)
                * maps.residual_patch_norms.unsqueeze(1)
            )
            values = {
                "raw_scores": maps.raw,
                "class_removed_scores": maps.class_only_removed,
                "patch_removed_scores": maps.patch_only_removed,
                "both_removed_scores": maps.both_removed,
                "shared_both_removed_scores": shared_scores,
                "class_coefficients": maps.class_coefficients,
                "class_axis_energy": class_energy,
                "class_norms": maps.class_norms,
                "class_residual_norms": maps.residual_class_norms,
                "patch_coefficient_mean": maps.patch_coefficients.mean(dim=-1),
                "patch_coefficient_std": maps.patch_coefficients.std(
                    dim=-1, unbiased=False
                ),
                "patch_axis_energy_mean": patch_energy.mean(dim=-1),
                "patch_axis_energy_std": patch_energy.std(dim=-1, unbiased=False),
                "raw_pair_cosine": pairs.raw_cosine,
                "residual_pair_cosine": pairs.residual_cosine,
                "pair_axis_dot": pairs.axis_dot,
                "pair_residual_dot": pairs.residual_dot,
                "heldout_projection": heldout,
                "shared_axis_energy": shared_energy,
                "final_logit_identity_error": identity,
                "decomposition_reconstruction_error": reconstruction_error,
                "residual_orthogonality_error": orthogonality_error,
                "residual_cosine_identity_error": (
                    maps.both_removed - direct_residual_cosine
                )
                .abs()
                .flatten(1)
                .amax(dim=-1),
            }
            for name, value in values.items():
                if self.storage[name][layer] is not None:
                    raise RuntimeError(f"presence layer {layer} fired twice")
                self.storage[name][layer] = value.detach()
            return None

        return hook

    def consume(self) -> PresenceBatchCapture:
        if not self.active:
            raise RuntimeError("no active presence capture")
        values: dict[str, torch.Tensor] = {}
        for name, layers in self.storage.items():
            if any(value is None for value in layers):
                raise RuntimeError(f"incomplete presence capture: {name}")
            values[name] = torch.stack(layers)  # type: ignore[arg-type]
        self.active = False
        self.eval_folds = None
        self.storage = {}
        return PresenceBatchCapture(**values)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False
        self.eval_folds = None
        self.storage = {}

    def __enter__(self):
        return self.register()

    def __exit__(self, exc_type, exc_value, tb):
        self.remove()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("mctformer", "mctformer_plus"), required=True
    )
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--allow-uncommitted-source", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size != 8 or args.num_workers < 0 or args.limit < 0:
        raise ValueError("batch-size must be 8; workers/limit must be non-negative")
    return args


def _registry_payload(registry: CrossFittedDirectionRegistry) -> dict[str, np.ndarray]:
    return {
        "fit_means": registry.fit_means,
        "class_deltas": registry.class_deltas,
        "shared_directions": registry.shared_directions,
        "loo_shared_directions": registry.loo_shared_directions,
        "class_alignment": registry.class_alignment,
        "loo_class_alignment": registry.loo_class_alignment,
        "total_counts": registry.total_counts,
        "positive_counts": registry.positive_counts,
        "negative_counts": registry.negative_counts,
    }


def _selected(
    tensor: torch.Tensor, batch: int, positive: np.ndarray, class_dim: int
) -> np.ndarray:
    index = torch.as_tensor(positive, dtype=torch.long, device=tensor.device)
    value = tensor[:, batch].index_select(class_dim, index)
    return value.float().cpu().numpy().astype(np.float32, copy=False)


def _validate_direction_artifact(
    path: Path, *, expected_sha256: str
) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("shared-presence direction artifact hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    expected_keys = {
        "fit_means",
        "class_deltas",
        "shared_directions",
        "loo_shared_directions",
        "class_alignment",
        "loo_class_alignment",
        "total_counts",
        "positive_counts",
        "negative_counts",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("shared-presence direction schema mismatch")
    shapes = {
        "fit_means": (2, EXPECTED_LAYERS, EXPECTED_CLASSES, 384),
        "class_deltas": (2, EXPECTED_LAYERS, EXPECTED_CLASSES, 384),
        "shared_directions": (2, EXPECTED_LAYERS, 384),
        "loo_shared_directions": (2, EXPECTED_LAYERS, EXPECTED_CLASSES, 384),
        "class_alignment": (2, EXPECTED_LAYERS, EXPECTED_CLASSES),
        "loo_class_alignment": (2, EXPECTED_LAYERS, EXPECTED_CLASSES),
        "total_counts": (2, EXPECTED_LAYERS, EXPECTED_CLASSES),
        "positive_counts": (2, EXPECTED_LAYERS, EXPECTED_CLASSES),
        "negative_counts": (2, EXPECTED_LAYERS, EXPECTED_CLASSES),
    }
    for key, shape in shapes.items():
        if payload[key].shape != shape:
            raise RuntimeError(
                f"shared-presence direction shape mismatch {key}: "
                f"{payload[key].shape} != {shape}"
            )
        if payload[key].dtype.kind in "fc" and not np.isfinite(payload[key]).all():
            raise RuntimeError(f"non-finite shared-presence direction field: {key}")
    for key in ("shared_directions", "loo_shared_directions"):
        norms = np.linalg.norm(payload[key].astype(np.float64), axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-6, rtol=0):
            raise RuntimeError(f"non-unit shared-presence directions: {key}")
    return {
        "artifact_sha256": expected_sha256,
        "schema": sorted(expected_keys),
        "passed": True,
    }


def _validate_artifact_tree(
    output: Path,
    *,
    expected_image_ids: Sequence[str],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    manifest_path = output / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(record.get("image_id", "")) for record in records]
    if ids != list(expected_image_ids):
        raise RuntimeError("derived presence manifest order/membership mismatch")
    for record in records:
        image_id = str(record["image_id"])
        path = (output / str(record["signal_path"])).resolve()
        try:
            path.relative_to(output.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"derived presence path escapes output: {path}"
            ) from error
        payload = reload_npz_checked(
            path,
            expected_sha256=str(record["artifact_sha256"]),
            expected_image_id=image_id,
        )
        if set(payload) != DERIVED_KEYS:
            raise RuntimeError(f"derived presence schema mismatch: {path}")
        positive = np.asarray(payload["positive_class_ids"], dtype=np.int64)
        p = len(positive)
        shapes = {
            "eval_fold": (),
            "fit_fold": (),
            "control_layer_ids": (len(CONTROL_LAYERS_ONE_BASED),),
            "raw_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "class_removed_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "patch_removed_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "both_removed_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "shared_both_removed_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "feature_norm_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "qk_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "attention_conditional_control_all": (
                len(CONTROL_LAYERS_ONE_BASED),
                EXPECTED_CLASSES,
                EXPECTED_PATCHES,
            ),
            "class_removed_scores": (EXPECTED_LAYERS, p, EXPECTED_PATCHES),
            "patch_removed_scores": (EXPECTED_LAYERS, p, EXPECTED_PATCHES),
            "both_removed_scores": (EXPECTED_LAYERS, p, EXPECTED_PATCHES),
            "shared_both_removed_scores": (EXPECTED_LAYERS, p, EXPECTED_PATCHES),
            "class_coefficients": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "class_axis_energy": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "class_norms": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "class_residual_norms": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "patch_coefficient_mean": (EXPECTED_LAYERS,),
            "patch_coefficient_std": (EXPECTED_LAYERS,),
            "patch_axis_energy_mean": (EXPECTED_LAYERS,),
            "patch_axis_energy_std": (EXPECTED_LAYERS,),
            "raw_pair_cosine_all": (
                EXPECTED_LAYERS,
                EXPECTED_CLASSES,
                EXPECTED_CLASSES,
            ),
            "residual_pair_cosine_all": (
                EXPECTED_LAYERS,
                EXPECTED_CLASSES,
                EXPECTED_CLASSES,
            ),
            "pair_axis_dot_all": (
                EXPECTED_LAYERS,
                EXPECTED_CLASSES,
                EXPECTED_CLASSES,
            ),
            "pair_residual_dot_all": (
                EXPECTED_LAYERS,
                EXPECTED_CLASSES,
                EXPECTED_CLASSES,
            ),
            "heldout_projection_all": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "shared_axis_energy_all": (EXPECTED_LAYERS, EXPECTED_CLASSES),
            "class_logits_all": (EXPECTED_CLASSES,),
        }
        for key, shape in shapes.items():
            if payload[key].shape != shape:
                raise RuntimeError(
                    f"derived presence shape mismatch {path}:{key}: "
                    f"{payload[key].shape} != {shape}"
                )
        if record.get("positive_class_ids") != positive.tolist():
            raise RuntimeError(f"derived presence manifest class mismatch: {path}")
        if not np.array_equal(
            payload["control_layer_ids"], np.asarray(CONTROL_LAYERS_ONE_BASED)
        ):
            raise RuntimeError(f"derived presence control layers mismatch: {path}")
        if int(payload["fit_fold"].item()) != 1 - int(payload["eval_fold"].item()):
            raise RuntimeError(f"derived presence cross-fit fold mismatch: {path}")
        if str(payload["source_signal_sha256"].item()) != source_hashes[image_id]:
            raise RuntimeError(f"derived presence source linkage mismatch: {path}")
        for key in (
            "raw_control_all",
            "class_removed_control_all",
            "patch_removed_control_all",
            "both_removed_control_all",
            "shared_both_removed_control_all",
            "feature_norm_control_all",
            "class_removed_scores",
            "patch_removed_scores",
            "both_removed_scores",
            "shared_both_removed_scores",
            "raw_pair_cosine_all",
            "residual_pair_cosine_all",
        ):
            if np.max(np.abs(payload[key])) > 1.00001:
                raise RuntimeError(f"derived cosine leaves [-1,1]: {path}:{key}")
        conditional = payload["attention_conditional_control_all"]
        if np.any(conditional < 0) or not np.allclose(
            conditional.sum(axis=-1), 1.0, atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"derived conditional attention invalid: {path}")
        for key in ("class_axis_energy", "shared_axis_energy_all"):
            if np.min(payload[key]) < -1e-7 or np.max(payload[key]) > 1.00001:
                raise RuntimeError(f"derived energy outside [0,1]: {path}:{key}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts_reloaded": len(records),
        "schema": sorted(DERIVED_KEYS),
        "passed": True,
    }


def execute(args: argparse.Namespace) -> None:
    inputs = resolve_runtime_inputs(args.source_metadata, args.model)
    state = runtime_source_state(RUNTIME_SOURCES)
    enforce_production_source(
        state,
        allow_uncommitted=bool(args.allow_uncommitted_source),
        limit=int(args.limit),
    )
    output, metadata, log = initialize_run_directory(
        args.output_dir,
        analysis="experiment3_validation_a_presence_axis",
        model=args.model,
        inputs=inputs,
        execution={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "limit": args.limit,
            "seed": args.seed,
            "allow_uncommitted_source": bool(args.allow_uncommitted_source),
            "direction_fit_images": EXPECTED_IMAGES,
        },
        git=state,
    )
    started = time.perf_counter()
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        metadata["environment"] = runtime_environment(device)
        model, checkpoint, configuration, load_info = create_runtime_model(
            args.model, inputs, device
        )
        metadata["model_configuration"] = configuration
        metadata["strict_checkpoint_load"] = load_info
        metadata["checkpoint_epoch"] = (
            checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
        )

        # The cross-fit registry always uses the full prespecified split.  A
        # smoke limits only held-out map evaluation, never direction fitting.
        fit_dataset, fit_loader, _, _ = make_dataset_and_loader(
            inputs,
            limit=0,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            pin_memory=not args.no_pin_memory,
        )
        all_ids = [str(value) for value in fit_dataset.image_ids]
        assignments = build_two_fold_split(all_ids)
        with (output / "split_manifest.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_id", "eval_fold"))
            writer.writeheader()
            writer.writerows(
                {"image_id": image_id, "eval_fold": assignments[image_id]}
                for image_id in all_ids
            )

        accumulator = TwoFoldPresenceAccumulator(EXPECTED_LAYERS, EXPECTED_CLASSES, 384)
        fit_processed = 0
        with PostBlockClassTokenCollector(model) as fit_collector:
            with torch.inference_mode():
                for batch_number, batch in enumerate(fit_loader):
                    images = batch["image"].to(
                        device,
                        non_blocking=(device.type == "cuda" and not args.no_pin_memory),
                    )
                    image_ids = list(batch["name"])
                    labels = batch["label"]
                    fit_collector.clear()
                    model(images)
                    tokens = fit_collector.consume().cpu()
                    for local, image_id in enumerate(image_ids):
                        accumulator.update(image_id, tokens[:, local], labels[local])
                        fit_processed += 1
                    log(
                        f"fit batch={batch_number + 1} images={fit_processed}/{EXPECTED_IMAGES}"
                    )
                    del images, tokens
        if fit_processed != EXPECTED_IMAGES:
            raise RuntimeError(f"direction fit processed {fit_processed} images")
        registry = accumulator.finalize()
        direction_path = output / "shared_presence_directions.npz"
        direction_sha = save_npz_atomic(direction_path, _registry_payload(registry))
        fold_counts = [len(values) for values in registry.image_ids_by_fold]
        metadata["cross_fit"] = {
            "split": "sha256(image_id UTF-8) parity; eval uses opposite fit fold",
            "fold_counts": fold_counts,
            "split_manifest": str(output / "split_manifest.csv"),
            "split_manifest_sha256": sha256_file(output / "split_manifest.csv"),
            "direction_artifact": str(direction_path),
            "direction_artifact_sha256": direction_sha,
            "minimum_positive_count": int(registry.positive_counts.min()),
            "minimum_negative_count": int(registry.negative_counts.min()),
            "fit_uncertainty_note": "reported fixed-OOF bootstrap is conditional on these two fitted directions",
        }

        eval_dataset, eval_loader, requested, context = make_dataset_and_loader(
            inputs,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            pin_memory=not args.no_pin_memory,
        )
        metadata["execution"]["requested_images"] = requested
        metadata["execution"]["batch_context_images"] = context

        # First-image instrumentation must be exactly observational.
        guard = eval_dataset[0]["image"].unsqueeze(0).to(device)
        guard_fold = [sha256_two_fold(str(eval_dataset[0]["name"]))]
        with torch.inference_mode():
            plain = model(guard)
            with SignalCollector(model, num_classes=EXPECTED_CLASSES) as signal_guard:
                with PresenceAxisCollector(model, registry) as axis_guard:
                    signal_guard.clear(expected_num_patches=EXPECTED_PATCHES)
                    axis_guard.clear(guard_fold)
                    observed = model(guard)
                    signal_guard.consume()
                    axis_capture = axis_guard.consume()
        guard_diff = assert_no_change(plain, observed, tolerance=0.0)
        metadata["first_image_no_change_guard"] = {
            "native_output_max_abs_diff": guard_diff,
            "final_logit_identity_max_abs_error": float(
                axis_capture.final_logit_identity_error.max().item()
            ),
            "passed": guard_diff == 0.0,
        }
        del guard, plain, observed, axis_capture

        signal_dir = output / "signals"
        signal_dir.mkdir()
        processed = 0
        maxima = {
            "feature_post_source": 0.0,
            "feature_norm_source": 0.0,
            "qk_source": 0.0,
            "attention_source": 0.0,
            "conditional_attention_source": 0.0,
            "class_pair_cosine_source": 0.0,
            "patch_norm_source": 0.0,
            "class_logits_source": 0.0,
            "raw_axis_vs_collector": 0.0,
            "final_logit_identity": 0.0,
            "decomposition_reconstruction": 0.0,
            "residual_orthogonality": 0.0,
            "residual_cosine_identity": 0.0,
        }
        with (output / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
            with SignalCollector(model, num_classes=EXPECTED_CLASSES) as collector:
                with PresenceAxisCollector(model, registry) as axis_collector:
                    with torch.inference_mode():
                        for batch_number, batch in enumerate(eval_loader):
                            images = batch["image"].to(
                                device,
                                non_blocking=(
                                    device.type == "cuda" and not args.no_pin_memory
                                ),
                            )
                            image_ids = list(batch["name"])
                            labels = batch["label"].cpu().numpy()
                            folds = [sha256_two_fold(value) for value in image_ids]
                            collector.clear(expected_num_patches=EXPECTED_PATCHES)
                            axis_collector.clear(folds)
                            model(images)
                            capture = collector.consume()
                            axis = axis_collector.consume()
                            maxima["raw_axis_vs_collector"] = max(
                                maxima["raw_axis_vs_collector"],
                                float(
                                    (axis.raw_scores - capture.feature_post_scores)
                                    .abs()
                                    .max()
                                    .item()
                                ),
                            )
                            maxima["final_logit_identity"] = max(
                                maxima["final_logit_identity"],
                                float(axis.final_logit_identity_error.max().item()),
                            )
                            maxima["decomposition_reconstruction"] = max(
                                maxima["decomposition_reconstruction"],
                                float(
                                    axis.decomposition_reconstruction_error.max().item()
                                ),
                            )
                            maxima["residual_orthogonality"] = max(
                                maxima["residual_orthogonality"],
                                float(axis.residual_orthogonality_error.max().item()),
                            )
                            maxima["residual_cosine_identity"] = max(
                                maxima["residual_cosine_identity"],
                                float(axis.residual_cosine_identity_error.max().item()),
                            )
                            for local, image_id in enumerate(image_ids):
                                if processed >= requested:
                                    break
                                positive = np.flatnonzero(labels[local] > 0).astype(
                                    np.int64
                                )
                                source = load_source_signal(inputs, image_id)
                                index = torch.as_tensor(
                                    positive, dtype=torch.long, device=device
                                )
                                comparisons = {
                                    "feature_post_source": capture.feature_post_scores[
                                        :, local
                                    ].index_select(1, index),
                                    "feature_norm_source": capture.feature_norm_scores[
                                        :, local
                                    ].index_select(1, index),
                                    "qk_source": capture.qk_mean_scores[
                                        :, local
                                    ].index_select(1, index),
                                    "attention_source": capture.attn_c2p_raw[
                                        :, local
                                    ].index_select(1, index),
                                    "conditional_attention_source": capture.attn_c2p_conditional[
                                        :, local
                                    ].index_select(1, index),
                                    "patch_norm_source": capture.patch_norms[:, local],
                                }
                                source_keys = {
                                    "feature_post_source": "feature_post_scores",
                                    "feature_norm_source": "feature_norm_scores",
                                    "qk_source": "qk_mean_scores",
                                    "attention_source": "attn_c2p_raw",
                                    "conditional_attention_source": "attn_c2p_conditional",
                                    "patch_norm_source": "patch_norms",
                                }
                                for name, current in comparisons.items():
                                    reference = torch.as_tensor(
                                        source[source_keys[name]], device=device
                                    )
                                    difference = float(
                                        (current.float() - reference.float())
                                        .abs()
                                        .max()
                                        .item()
                                    )
                                    maxima[name] = max(maxima[name], difference)
                                    if difference >= STRICT_TOLERANCE:
                                        raise RuntimeError(
                                            f"{name} reproduction failed for {image_id}: {difference}"
                                        )
                                pair_current = (
                                    capture.class_token_pairwise_cosine[:, local]
                                    .index_select(1, index)
                                    .index_select(2, index)
                                )
                                pair_reference = torch.as_tensor(
                                    source["class_token_pairwise_cosine"],
                                    device=device,
                                )
                                pair_difference = float(
                                    (pair_current.float() - pair_reference.float())
                                    .abs()
                                    .max()
                                    .item()
                                )
                                maxima["class_pair_cosine_source"] = max(
                                    maxima["class_pair_cosine_source"], pair_difference
                                )
                                if pair_difference >= STRICT_TOLERANCE:
                                    raise RuntimeError(
                                        "class-pair cosine reproduction failed for "
                                        f"{image_id}: {pair_difference}"
                                    )
                                class_logits = capture.last_class_tokens[local].mean(
                                    dim=-1
                                )
                                logit_reference = torch.as_tensor(
                                    source["class_logits_all"], device=device
                                )
                                logit_diff = float(
                                    (class_logits.float() - logit_reference.float())
                                    .abs()
                                    .max()
                                    .item()
                                )
                                maxima["class_logits_source"] = max(
                                    maxima["class_logits_source"], logit_diff
                                )
                                if logit_diff >= STRICT_TOLERANCE:
                                    raise RuntimeError(
                                        f"class-logit reproduction failed for {image_id}: {logit_diff}"
                                    )

                                def all_class(value: torch.Tensor) -> np.ndarray:
                                    return (
                                        value[:, local]
                                        .float()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float32, copy=False)
                                    )

                                def pos_class(value: torch.Tensor) -> np.ndarray:
                                    return (
                                        value[:, local]
                                        .index_select(
                                            1,
                                            torch.as_tensor(
                                                positive, device=value.device
                                            ),
                                        )
                                        .float()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float32, copy=False)
                                    )

                                def control_all(value: torch.Tensor) -> np.ndarray:
                                    layer_index = torch.as_tensor(
                                        CONTROL_LAYER_INDICES,
                                        dtype=torch.long,
                                        device=value.device,
                                    )
                                    return (
                                        value[:, local]
                                        .index_select(0, layer_index)
                                        .float()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float32, copy=False)
                                    )

                                payload = {
                                    "image_id": np.asarray(image_id),
                                    "eval_fold": np.asarray(
                                        folds[local], dtype=np.int8
                                    ),
                                    "fit_fold": np.asarray(
                                        1 - folds[local], dtype=np.int8
                                    ),
                                    "positive_class_ids": positive,
                                    "control_layer_ids": np.asarray(
                                        CONTROL_LAYERS_ONE_BASED, dtype=np.int16
                                    ),
                                    "raw_control_all": control_all(axis.raw_scores),
                                    "class_removed_control_all": control_all(
                                        axis.class_removed_scores
                                    ),
                                    "patch_removed_control_all": control_all(
                                        axis.patch_removed_scores
                                    ),
                                    "both_removed_control_all": control_all(
                                        axis.both_removed_scores
                                    ),
                                    "shared_both_removed_control_all": control_all(
                                        axis.shared_both_removed_scores
                                    ),
                                    "feature_norm_control_all": control_all(
                                        capture.feature_norm_scores
                                    ),
                                    "qk_control_all": control_all(
                                        capture.qk_mean_scores
                                    ),
                                    "attention_conditional_control_all": control_all(
                                        capture.attn_c2p_conditional
                                    ),
                                    "class_removed_scores": pos_class(
                                        axis.class_removed_scores
                                    ),
                                    "patch_removed_scores": pos_class(
                                        axis.patch_removed_scores
                                    ),
                                    "both_removed_scores": pos_class(
                                        axis.both_removed_scores
                                    ),
                                    "shared_both_removed_scores": pos_class(
                                        axis.shared_both_removed_scores
                                    ),
                                    "class_coefficients": all_class(
                                        axis.class_coefficients
                                    ),
                                    "class_axis_energy": all_class(
                                        axis.class_axis_energy
                                    ),
                                    "class_norms": all_class(axis.class_norms),
                                    "class_residual_norms": all_class(
                                        axis.class_residual_norms
                                    ),
                                    "patch_coefficient_mean": axis.patch_coefficient_mean[
                                        :, local
                                    ]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "patch_coefficient_std": axis.patch_coefficient_std[
                                        :, local
                                    ]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "patch_axis_energy_mean": axis.patch_axis_energy_mean[
                                        :, local
                                    ]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "patch_axis_energy_std": axis.patch_axis_energy_std[
                                        :, local
                                    ]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "raw_pair_cosine_all": all_class(
                                        axis.raw_pair_cosine
                                    ),
                                    "residual_pair_cosine_all": all_class(
                                        axis.residual_pair_cosine
                                    ),
                                    "pair_axis_dot_all": all_class(axis.pair_axis_dot),
                                    "pair_residual_dot_all": all_class(
                                        axis.pair_residual_dot
                                    ),
                                    "heldout_projection_all": all_class(
                                        axis.heldout_projection
                                    ),
                                    "shared_axis_energy_all": all_class(
                                        axis.shared_axis_energy
                                    ),
                                    "class_logits_all": class_logits.float()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "source_signal_sha256": np.asarray(
                                        inputs.source_artifact_sha256[image_id]
                                    ),
                                }
                                path = signal_dir / f"{image_id}.npz"
                                digest = save_npz_atomic(path, payload)
                                manifest.write(
                                    json.dumps(
                                        {
                                            "image_id": image_id,
                                            "eval_fold": folds[local],
                                            "positive_class_ids": positive.tolist(),
                                            "signal_path": str(
                                                path.relative_to(output)
                                            ),
                                            "artifact_sha256": digest,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                manifest.flush()
                                processed += 1
                            log(
                                f"eval batch={batch_number + 1} images={processed}/{requested}"
                            )
                            del images, capture, axis

        if processed != requested or len(list(signal_dir.glob("*.npz"))) != requested:
            raise RuntimeError(
                f"saved {processed} presence artifacts, expected {requested}"
            )
        source_strict_names = (
            "feature_post_source",
            "feature_norm_source",
            "qk_source",
            "attention_source",
            "conditional_attention_source",
            "class_pair_cosine_source",
            "patch_norm_source",
            "class_logits_source",
        )
        algebra_strict_names = (
            "final_logit_identity",
            "decomposition_reconstruction",
            "residual_orthogonality",
        )
        if any(maxima[name] >= STRICT_TOLERANCE for name in source_strict_names):
            raise RuntimeError(f"presence numerical gate failed: {maxima}")
        if any(maxima[name] >= STRICT_TOLERANCE for name in algebra_strict_names):
            raise RuntimeError(f"presence algebraic gate failed: {maxima}")
        if maxima["raw_axis_vs_collector"] > 5e-6:
            raise RuntimeError(f"presence raw-cosine cross-check failed: {maxima}")
        if maxima["residual_cosine_identity"] > 5e-6:
            raise RuntimeError(f"presence residual-cosine gate failed: {maxima}")
        direction_validation = _validate_direction_artifact(
            direction_path, expected_sha256=direction_sha
        )
        artifact_validation = _validate_artifact_tree(
            output,
            expected_image_ids=[
                str(value) for value in eval_dataset.image_ids[:requested]
            ],
            source_hashes=inputs.source_artifact_sha256,
        )
        assert_inputs_unchanged(inputs)
        finish_metadata(
            output,
            metadata,
            started=started,
            updates={
                "processed_images": processed,
                "direction_fit_images": fit_processed,
                "numerical_max_abs_differences": maxima,
                "signal_manifest": "manifest.jsonl",
                "signals": "signals",
                "direction_artifact_validation": direction_validation,
                "derived_artifact_validation": artifact_validation,
                "timing_note": (
                    "V0-V3 are post-block Lk; Experiment 2 V4 norm1 Lk is the input "
                    "to block Lk. Aligned analysis compares post Lk with norm1 L(k+1)."
                ),
            },
        )
        log(f"complete images={processed} fit_images={fit_processed} maxima={maxima}")
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        json_dump(output / "metadata.json", metadata)
        raise


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
