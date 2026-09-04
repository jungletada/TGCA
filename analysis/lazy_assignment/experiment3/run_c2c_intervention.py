#!/usr/bin/env python3
"""Run Experiment 3 Validation C frozen-model C2C interventions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    upsample_and_normalize_active_cams,
)
from analysis.lazy_assignment.experiment2.native_cam_stages import (  # noqa: E402
    decompose_native_cam_reduced,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    REGION_BACKGROUND,
    REGION_OTHER_FOREGROUND,
    REGION_TARGET,
)
from analysis.lazy_assignment.experiment2.signal_collector import (  # noqa: E402
    SignalCapture,
    SignalCollector,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (  # noqa: E402
    cam_threshold_grid,
)
from analysis.lazy_assignment.experiment3.c2c_intervention import (  # noqa: E402
    C2C_VARIANT_LAYERS_1BASED,
    C2CIntervention,
    variant_layer_indices,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    EXPECTED_PATCHES,
    STRICT_TOLERANCE,
    assert_new_output,
    enforce_production_source,
    json_dump,
    runtime_source_state,
    sha256_file,
)
from analysis.lazy_assignment.experiment3.presence_axis import (  # noqa: E402
    axis_removed_cosine_maps,
    normalized_all_ones_direction,
    token_pair_axis_metrics,
)
from analysis.lazy_assignment.experiment3.runtime import (  # noqa: E402
    RuntimeInputs,
    assert_inputs_unchanged,
    create_runtime_model,
    finish_metadata,
    initialize_run_directory,
    load_source_signal,
    make_dataset_and_loader,
    mask_region_codes,
    reload_npz_checked,
    resolve_runtime_inputs,
    runtime_environment,
    save_npz_atomic,
    source_signal_path,
)


VARIANT_CODES = tuple(C2C_VARIANT_LAYERS_1BASED)
LATE_LAYER_INDICES = (9, 10, 11)
LATE_LAYER_NUMBERS = tuple(layer + 1 for layer in LATE_LAYER_INDICES)
REGION_CODES = (REGION_TARGET, REGION_OTHER_FOREGROUND, REGION_BACKGROUND)
HEAD_REGION_NAMES = ("target", "other_foreground", "background")
SEGMENTATION_CLASSES = 21

RUNTIME_SOURCES = (
    "analysis/lazy_assignment/experiment3/common.py",
    "analysis/lazy_assignment/experiment3/runtime.py",
    "analysis/lazy_assignment/experiment3/presence_axis.py",
    "analysis/lazy_assignment/experiment3/cam_layer_intervention.py",
    "analysis/lazy_assignment/experiment3/c2c_intervention.py",
    "analysis/lazy_assignment/experiment3/run_c2c_intervention.py",
    "analysis/lazy_assignment/experiment2/evaluation_metrics.py",
    "analysis/lazy_assignment/experiment2/native_cam_stages.py",
    "analysis/lazy_assignment/experiment2/patch_regions.py",
    "analysis/lazy_assignment/experiment2/signal_collector.py",
    "analysis/lazy_assignment/experiment2/voc_semantic_dataset.py",
    "analysis/lazy_assignment/experiment2/run_experiment2_signals.py",
    "analysis/lazy_assignment/run_class_specific_patch_score.py",
    "analysis/lazy_assignment/score_utils.py",
    "datasets_cam.py",
    "utils.py",
    "models/mctformer.py",
    "models/mctformer_plus.py",
    "models/vit.py",
    "models/tgca.py",
)

SOURCE_EQUIVALENCE_FIELDS = (
    "feature_post_scores",
    "attention_raw",
    "attention_conditional",
    "class_logits_all",
    "patch_class_logits_all",
    "patch_head_logits_positive",
    "final_cam",
)

SIGNAL_KEYS = frozenset(
    {
        "image_id",
        "variant_code",
        "positive_class_ids",
        "image_labels",
        "pair_class_ids",
        "late_layers_one_based",
        "thresholds",
        "patch_label_counts",
        "region_masks_rho05",
        "region_masks_rho07",
        "class_logits_all",
        "patch_class_logits_all",
        "patch_head_logits_positive",
        "feature_post_l10_l12",
        "feature_both_axis_removed_l10_l12",
        "positive_pair_raw_cosine_l10_l12",
        "positive_pair_residual_cosine_l10_l12",
        "attention_c2p_raw_l10_l12",
        "attention_c2p_conditional_l10_l12",
        "attention_head_region_raw_rho05",
        "attention_head_region_conditional_rho05",
        "attention_head_region_raw_rho07",
        "attention_head_region_conditional_rho07",
        "c2c_pre_offdiag_mass",
        "c2c_pre_diagonal_mass",
        "c2c_pre_class_mass",
        "c2c_post_offdiag_mass",
        "c2c_post_diagonal_mass",
        "c2c_post_class_mass",
        "final_cam",
        "threshold_confusions",
        "source_signal_sha256",
    }
)
ALLOW_NAN_SIGNAL_KEYS = frozenset(
    {
        "attention_head_region_raw_rho05",
        "attention_head_region_conditional_rho05",
        "attention_head_region_raw_rho07",
        "attention_head_region_conditional_rho07",
    }
)


@dataclass(frozen=True)
class C2CRunnerCapture:
    late_class_tokens: torch.Tensor  # [3,B,C,D]
    late_patch_tokens: torch.Tensor  # [3,B,P,D]
    c2p_heads: torch.Tensor  # [12,B,H,C,P]
    pre_offdiag_mass: torch.Tensor  # [12,B,H,C]
    pre_diagonal_mass: torch.Tensor  # [12,B,H,C]
    pre_class_mass: torch.Tensor  # [12,B,H,C]
    post_offdiag_mass: torch.Tensor  # [12,B,H,C]
    post_diagonal_mass: torch.Tensor  # [12,B,H,C]
    post_class_mass: torch.Tensor  # [12,B,H,C]


def _c2c_components(attention: torch.Tensor) -> tuple[torch.Tensor, ...]:
    c2c = attention[..., :EXPECTED_CLASSES, :EXPECTED_CLASSES]
    diagonal = torch.diagonal(c2c, dim1=-2, dim2=-1).detach().clone()
    class_mass = c2c.sum(dim=-1).detach()
    mask = torch.eye(EXPECTED_CLASSES, dtype=torch.bool, device=attention.device).view(
        1, 1, EXPECTED_CLASSES, EXPECTED_CLASSES
    )
    offdiag_mass = c2c.masked_fill(mask, 0).sum(dim=-1).detach()
    return offdiag_mass, diagonal, class_mass


class C2CRunnerCollector:
    """Observe compact pre/post C2C signals without changing module outputs."""

    def __init__(self, model: nn.Module):
        if not hasattr(model, "blocks") or len(model.blocks) != EXPECTED_LAYERS:
            raise TypeError("collector requires the native 12-block host")
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._reset()

    def _reset(self) -> None:
        self._pre: list[tuple[torch.Tensor, ...] | None] = [None] * EXPECTED_LAYERS
        self._post: list[tuple[torch.Tensor, ...] | None] = [None] * EXPECTED_LAYERS
        self._c2p: list[torch.Tensor | None] = [None] * EXPECTED_LAYERS
        self._late_classes: dict[int, torch.Tensor] = {}
        self._late_patches: dict[int, torch.Tensor] = {}

    def register(self) -> "C2CRunnerCollector":
        if self.handles:
            raise RuntimeError("C2C runner collector is already registered")
        for layer, block in enumerate(self.model.blocks):
            self.handles.append(
                block.attn.normalizer.register_forward_hook(
                    self._make_normalizer_hook(layer)
                )
            )
            self.handles.append(
                block.register_forward_hook(self._make_block_hook(layer))
            )
        return self

    def clear(self) -> None:
        if not self.handles:
            raise RuntimeError("register C2C runner collector before clear")
        if self._active:
            raise RuntimeError("previous C2C runner capture was not consumed")
        self._reset()
        self._active = True

    def _validate_attention(self, layer: int, value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"block {layer} attention must be a tensor")
        if value.ndim != 4:
            raise RuntimeError(
                f"block {layer} attention must have shape [B,H,T,T], "
                f"got {tuple(value.shape)}"
            )
        expected = (
            value.shape[0],
            int(self.model.blocks[layer].attn.num_heads),
            EXPECTED_CLASSES + EXPECTED_PATCHES,
            EXPECTED_CLASSES + EXPECTED_PATCHES,
        )
        if tuple(value.shape) != expected or not value.is_floating_point():
            raise RuntimeError(
                f"block {layer} attention {tuple(value.shape)}/{value.dtype} != "
                f"{expected}/floating"
            )
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"block {layer} attention is non-finite")
        return value

    def _make_normalizer_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._pre[layer] is not None:
                raise RuntimeError(f"normalizer layer {layer} fired more than once")
            attention = self._validate_attention(layer, output)
            # This observer is registered before the temporary intervention
            # hook, so it records the native matrix for the current inputs.
            self._pre[layer] = _c2c_components(attention)
            return None

        return hook

    def _make_block_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._post[layer] is not None or self._c2p[layer] is not None:
                raise RuntimeError(f"block layer {layer} fired more than once")
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise TypeError("native block must return (tokens, attention)")
            tokens, value = output[0], output[1]
            if not isinstance(tokens, torch.Tensor):
                raise TypeError("native block tokens must be a tensor")
            attention = self._validate_attention(layer, value)
            expected_tokens = EXPECTED_CLASSES + EXPECTED_PATCHES
            if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"block {layer} tokens must have shape [B,{expected_tokens},D]"
                )
            self._post[layer] = _c2c_components(attention)
            # Clone the reduced view; detach alone would retain the full T x T
            # attention allocation through its underlying storage.
            self._c2p[layer] = (
                attention[..., :EXPECTED_CLASSES, EXPECTED_CLASSES:].detach().clone()
            )
            if layer in LATE_LAYER_INDICES:
                self._late_classes[layer] = (
                    tokens[:, :EXPECTED_CLASSES].detach().clone()
                )
                self._late_patches[layer] = (
                    tokens[:, EXPECTED_CLASSES:].detach().clone()
                )
            return None

        return hook

    @staticmethod
    def _stack(values: Sequence[torch.Tensor | None], name: str) -> torch.Tensor:
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            raise RuntimeError(f"missing {name} captures for layers {missing}")
        return torch.stack(list(values))  # type: ignore[arg-type]

    @staticmethod
    def _component(
        values: Sequence[tuple[torch.Tensor, ...] | None],
        index: int,
        name: str,
    ) -> torch.Tensor:
        missing = [layer for layer, value in enumerate(values) if value is None]
        if missing:
            raise RuntimeError(f"missing {name} captures for layers {missing}")
        return torch.stack([value[index] for value in values])  # type: ignore[index]

    def consume(self) -> C2CRunnerCapture:
        if not self._active:
            raise RuntimeError("no active C2C runner capture")
        missing_late = [
            layer
            for layer in LATE_LAYER_INDICES
            if layer not in self._late_classes or layer not in self._late_patches
        ]
        if missing_late:
            raise RuntimeError(f"missing late token captures: {missing_late}")
        capture = C2CRunnerCapture(
            late_class_tokens=torch.stack(
                [self._late_classes[layer] for layer in LATE_LAYER_INDICES]
            ),
            late_patch_tokens=torch.stack(
                [self._late_patches[layer] for layer in LATE_LAYER_INDICES]
            ),
            c2p_heads=self._stack(self._c2p, "C2P head"),
            pre_offdiag_mass=self._component(self._pre, 0, "pre offdiag"),
            pre_diagonal_mass=self._component(self._pre, 1, "pre diagonal"),
            pre_class_mass=self._component(self._pre, 2, "pre class mass"),
            post_offdiag_mass=self._component(self._post, 0, "post offdiag"),
            post_diagonal_mass=self._component(self._post, 1, "post diagonal"),
            post_class_mass=self._component(self._post, 2, "post class mass"),
        )
        self._active = False
        self._reset()
        return capture

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self._active = False
        self._reset()

    def __enter__(self) -> "C2CRunnerCollector":
        return self.register()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()


@dataclass(frozen=True)
class ImageContext:
    local_index: int
    image_id: str
    labels: np.ndarray
    positive: np.ndarray
    mask: np.ndarray
    patch_counts: np.ndarray
    region_rho05: np.ndarray
    region_rho07: np.ndarray
    source: Mapping[str, np.ndarray]
    source_sha256: str


@dataclass(frozen=True)
class ControlSnapshot:
    patch_tokens: torch.Tensor
    patch_logits: torch.Tensor
    c2p_heads: torch.Tensor
    final_cam: torch.Tensor


def _as_float32(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _tensor_difference(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    if left.dtype != right.dtype:
        raise RuntimeError(f"{name} dtype mismatch: {left.dtype} != {right.dtype}")
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    if left_cpu.shape != right_cpu.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {tuple(left_cpu.shape)} != "
            f"{tuple(right_cpu.shape)}"
        )
    if not left_cpu.numel():
        return 0.0
    return float((left_cpu - right_cpu).abs().max().item())


def _array_difference(left: np.ndarray, right: np.ndarray, name: str) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {left_array.shape} != {right_array.shape}"
        )
    if not left_array.size:
        return 0.0
    return float(
        np.max(
            np.abs(
                left_array.astype(np.float64, copy=False)
                - right_array.astype(np.float64, copy=False)
            )
        )
    )


def _threshold_confusions(
    normalized_cam: np.ndarray,
    positive_class_ids: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(normalized_cam)
    positive = np.asarray(positive_class_ids, dtype=np.int64)
    target = np.asarray(mask, dtype=np.int64)
    if values.shape != (len(positive), 448, 448) or target.shape != (448, 448):
        raise ValueError("normalized CAM, positive classes, and mask shapes disagree")
    if not np.isfinite(values).all() or not np.all(np.diff(thresholds) > 0):
        raise ValueError("CAM/threshold values must be finite and ordered")
    comparison_thresholds = np.asarray(thresholds, dtype=values.dtype)
    if not np.isfinite(comparison_thresholds).all() or not np.all(
        np.diff(comparison_thresholds) > 0
    ):
        raise ValueError("threshold grid is invalid in the CAM comparison dtype")
    foreground_offset = values.argmax(axis=0)
    confidence = values.max(axis=0)
    foreground_prediction = positive[foreground_offset] + 1
    valid = target != 255
    gt = target[valid]
    predicted = foreground_prediction[valid]
    pass_count = np.searchsorted(comparison_thresholds, confidence[valid], side="left")
    bins = len(thresholds) + 1
    encoded = (gt * SEGMENTATION_CLASSES + predicted) * bins + pass_count
    histogram = np.bincount(
        encoded, minlength=SEGMENTATION_CLASSES * SEGMENTATION_CLASSES * bins
    ).reshape(SEGMENTATION_CLASSES, SEGMENTATION_CLASSES, bins)
    cumulative = histogram[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]
    total_target = np.bincount(gt, minlength=SEGMENTATION_CLASSES).astype(
        np.int64, copy=False
    )
    output = np.zeros(
        (len(thresholds), SEGMENTATION_CLASSES, SEGMENTATION_CLASSES),
        dtype=np.int64,
    )
    for index in range(len(thresholds)):
        foreground = cumulative[:, :, index + 1]
        output[index] = foreground
        output[index, :, 0] = total_target - foreground.sum(axis=1)
    return output


def _positive_pairs(positive: np.ndarray) -> np.ndarray:
    pairs = [
        (int(positive[left]), int(positive[right]))
        for left in range(len(positive))
        for right in range(left + 1, len(positive))
    ]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _head_region_means(
    values: torch.Tensor,
    positive: np.ndarray,
    region_codes: np.ndarray,
) -> np.ndarray:
    if (
        values.ndim != 4
        or values.shape[0] != 3
        or values.shape[2:]
        != (
            EXPECTED_CLASSES,
            EXPECTED_PATCHES,
        )
    ):
        raise ValueError("head attention must have shape [3,H,20,784]")
    if region_codes.shape != (len(positive), EXPECTED_PATCHES):
        raise ValueError("region code map does not match positive classes")
    result = torch.full(
        (3, values.shape[1], len(positive), len(REGION_CODES)),
        float("nan"),
        dtype=torch.float32,
        device=values.device,
    )
    for local_class, class_id in enumerate(positive):
        class_values = values[:, :, int(class_id)]
        codes = torch.as_tensor(region_codes[local_class], device=values.device)
        for region_index, code in enumerate(REGION_CODES):
            selected = codes == code
            if bool(selected.any()):
                result[:, :, local_class, region_index] = (
                    class_values[..., selected].float().mean(dim=-1)
                )
    return result.cpu().numpy().astype(np.float32, copy=False)


def _late_axis_products(
    capture: C2CRunnerCapture,
    feature_capture: SignalCapture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    direction = normalized_all_ones_direction(
        capture.late_class_tokens.shape[-1],
        device=capture.late_class_tokens.device,
        dtype=torch.float32,
    )
    both_removed = []
    pair_raw = []
    pair_residual = []
    maximum_raw_difference = 0.0
    for late_offset, layer in enumerate(LATE_LAYER_INDICES):
        maps = axis_removed_cosine_maps(
            capture.late_class_tokens[late_offset],
            capture.late_patch_tokens[late_offset],
            direction,
        )
        difference = float(
            (maps.raw.float() - feature_capture.feature_post_scores[layer].float())
            .abs()
            .max()
            .item()
        )
        maximum_raw_difference = max(maximum_raw_difference, difference)
        # This is an independent float32 cosine re-computation, not the exact
        # Experiment 2 collector path. CUDA reduction order can differ by a
        # few ulps; the exact immutable-source reproduction is gated
        # separately at STRICT_TOLERANCE below.
        if difference > 5e-6:
            raise RuntimeError(
                f"fixed-axis raw feature reproduction failed at L{layer + 1}: "
                f"{difference}"
            )
        both_removed.append(maps.both_removed.detach())
        pair = token_pair_axis_metrics(
            capture.late_class_tokens[late_offset], direction
        )
        pair_raw.append(pair.raw_cosine.detach())
        pair_residual.append(pair.residual_cosine.detach())
    return (
        torch.stack(both_removed),
        torch.stack(pair_raw),
        torch.stack(pair_residual),
        maximum_raw_difference,
    )


def _patch_class_logits(
    model_key: str, model: nn.Module, patch_logits: torch.Tensor
) -> torch.Tensor:
    if model_key == "mctformer":
        return model.avgpool(patch_logits).squeeze(-1).squeeze(-1)
    if model_key == "mctformer_plus":
        return model.gwrp(patch_logits)
    raise ValueError(model_key)


def _validate_intervention_capture(
    intervention: C2CIntervention,
    capture: C2CRunnerCapture,
    *,
    variant: str,
    hook_counts_before: tuple[int, ...],
    hook_counts_after: tuple[int, ...],
) -> dict[str, float]:
    expected_layers = variant_layer_indices(variant)
    expected_counts = {layer: 1 for layer in expected_layers}
    if intervention.layers != expected_layers:
        raise RuntimeError(f"{variant} layer mapping drifted: {intervention.layers}")
    if intervention.activation_counts != expected_counts:
        raise RuntimeError(
            f"{variant} activation counts {intervention.activation_counts} != "
            f"{expected_counts}"
        )
    if len(intervention.records) != len(expected_layers):
        raise RuntimeError(f"{variant} structural record count is invalid")
    if hook_counts_before != hook_counts_after:
        raise RuntimeError(f"{variant} intervention hooks were not fully removed")

    maxima = {
        "selected_offdiag_post": 0.0,
        "selected_diagonal_assignment": 0.0,
        "selected_class_mass": 0.0,
        "unselected_offdiag": 0.0,
        "unselected_diagonal": 0.0,
        "unselected_class_mass": 0.0,
        "record_pre_offdiag_mean": 0.0,
        "record_pre_offdiag_max": 0.0,
    }
    records_by_layer = {record.layer_index: record for record in intervention.records}
    for layer in range(EXPECTED_LAYERS):
        if layer in expected_layers:
            maxima["selected_offdiag_post"] = max(
                maxima["selected_offdiag_post"],
                float(capture.post_offdiag_mass[layer].abs().max().item()),
            )
            maxima["selected_diagonal_assignment"] = max(
                maxima["selected_diagonal_assignment"],
                float(
                    (capture.post_diagonal_mass[layer] - capture.pre_class_mass[layer])
                    .abs()
                    .max()
                    .item()
                ),
            )
            maxima["selected_class_mass"] = max(
                maxima["selected_class_mass"],
                float(
                    (capture.post_class_mass[layer] - capture.pre_class_mass[layer])
                    .abs()
                    .max()
                    .item()
                ),
            )
            record = records_by_layer[layer]
            pre_mean = float(capture.pre_offdiag_mass[layer].float().mean().item())
            pre_max = float(capture.pre_offdiag_mass[layer].float().max().item())
            if pre_max <= 0:
                raise RuntimeError(f"{variant} L{layer + 1} treatment has zero mass")
            maxima["record_pre_offdiag_mean"] = max(
                maxima["record_pre_offdiag_mean"],
                abs(pre_mean - record.offdiag_pre_mass_mean),
            )
            maxima["record_pre_offdiag_max"] = max(
                maxima["record_pre_offdiag_max"],
                abs(pre_max - record.offdiag_pre_mass_max),
            )
        else:
            maxima["unselected_offdiag"] = max(
                maxima["unselected_offdiag"],
                float(
                    (capture.post_offdiag_mass[layer] - capture.pre_offdiag_mass[layer])
                    .abs()
                    .max()
                    .item()
                ),
            )
            maxima["unselected_diagonal"] = max(
                maxima["unselected_diagonal"],
                float(
                    (
                        capture.post_diagonal_mass[layer]
                        - capture.pre_diagonal_mass[layer]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
            maxima["unselected_class_mass"] = max(
                maxima["unselected_class_mass"],
                float(
                    (capture.post_class_mass[layer] - capture.pre_class_mass[layer])
                    .abs()
                    .max()
                    .item()
                ),
            )
    failures = {
        name: value for name, value in maxima.items() if value >= STRICT_TOLERANCE
    }
    if failures:
        raise RuntimeError(
            f"{variant} runner/intervention invariant failure: {failures}"
        )
    return maxima


def _control_snapshot(
    feature_capture: SignalCapture,
    runner_capture: C2CRunnerCapture,
    patch_logits: torch.Tensor,
    final_cam: torch.Tensor,
) -> ControlSnapshot:
    return ControlSnapshot(
        patch_tokens=feature_capture.last_patch_tokens.detach().cpu().clone(),
        patch_logits=patch_logits.detach().cpu().clone(),
        c2p_heads=runner_capture.c2p_heads.detach().cpu().clone(),
        final_cam=final_cam.detach().cpu().clone(),
    )


def _compare_control(
    reference: ControlSnapshot,
    feature_capture: SignalCapture,
    runner_capture: C2CRunnerCapture,
    patch_logits: torch.Tensor,
    final_cam: torch.Tensor,
    *,
    name: str,
) -> dict[str, float]:
    differences = {
        "final_patch_tokens": _tensor_difference(
            feature_capture.last_patch_tokens, reference.patch_tokens, name
        ),
        "patch_head_logits": _tensor_difference(
            patch_logits, reference.patch_logits, name
        ),
        "all_layer_all_head_c2p": _tensor_difference(
            runner_capture.c2p_heads, reference.c2p_heads, name
        ),
        "native_final_cam": _tensor_difference(final_cam, reference.final_cam, name),
    }
    failures = {
        field: value
        for field, value in differences.items()
        if value >= STRICT_TOLERANCE
    }
    if failures:
        raise RuntimeError(f"{name} structural negative control failed: {failures}")
    return differences


def _verify_c0_source(
    context: ImageContext,
    feature_capture: SignalCapture,
    patch_logits: torch.Tensor,
    patch_class_logits: torch.Tensor,
    final_cam: torch.Tensor,
) -> dict[str, float]:
    source = context.source
    positive_index = torch.as_tensor(
        context.positive, dtype=torch.long, device=patch_logits.device
    )
    local = context.local_index
    current = {
        "feature_post_scores": _as_float32(
            feature_capture.feature_post_scores[:, local].index_select(
                1, positive_index
            )
        ),
        "attention_raw": _as_float32(
            feature_capture.attn_c2p_raw[:, local].index_select(1, positive_index)
        ),
        "attention_conditional": _as_float32(
            feature_capture.attn_c2p_conditional[:, local].index_select(
                1, positive_index
            )
        ),
        "class_logits_all": _as_float32(
            feature_capture.last_class_tokens[local].mean(dim=-1)
        ),
        "patch_class_logits_all": _as_float32(patch_class_logits[local]),
        "patch_head_logits_positive": _as_float32(
            patch_logits[local].index_select(0, positive_index).flatten(1)
        ),
        "final_cam": _as_float32(
            final_cam[local].index_select(0, positive_index).flatten(1)
        ),
    }
    source_fields = {
        "feature_post_scores": source["feature_post_scores"],
        "attention_raw": source["attn_c2p_raw"],
        "attention_conditional": source["attn_c2p_conditional"],
        "class_logits_all": source["class_logits_all"],
        "patch_class_logits_all": source["patch_class_logits_all"],
        "patch_head_logits_positive": source["patch_logits"],
        "final_cam": source["final_cam"],
    }
    differences = {
        name: _array_difference(current[name], source_fields[name], name)
        for name in SOURCE_EQUIVALENCE_FIELDS
    }
    failures = {
        name: difference
        for name, difference in differences.items()
        if difference >= STRICT_TOLERANCE
    }
    if failures:
        raise RuntimeError(
            f"C0 Experiment 2 equivalence failed for {context.image_id}: {failures}"
        )
    return differences


def _pair_values(values: torch.Tensor, local: int, pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return np.empty((3, 0), dtype=np.float32)
    left = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=values.device)
    right = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=values.device)
    return _as_float32(values[:, local][:, left, right])


def _build_payload(
    *,
    variant: str,
    context: ImageContext,
    feature_capture: SignalCapture,
    runner_capture: C2CRunnerCapture,
    both_removed: torch.Tensor,
    pair_raw: torch.Tensor,
    pair_residual: torch.Tensor,
    patch_logits: torch.Tensor,
    patch_class_logits: torch.Tensor,
    final_cam: torch.Tensor,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    local = context.local_index
    positive = context.positive
    positive_index = torch.as_tensor(
        positive, dtype=torch.long, device=patch_logits.device
    )
    late_index = torch.as_tensor(
        LATE_LAYER_INDICES,
        dtype=torch.long,
        device=patch_logits.device,
    )
    pairs = _positive_pairs(positive)

    # Keep all 20 class rows here because region lookup is keyed by the
    # original VOC class ID; the stored region summaries are reduced to the
    # positive classes inside _head_region_means.
    late_heads = runner_capture.c2p_heads.index_select(0, late_index)[:, local]
    head_mass = late_heads.sum(dim=-1, keepdim=True)
    if bool(torch.any(head_mass <= 0)):
        raise RuntimeError("per-head class-to-patch mass is non-positive")
    late_heads_conditional = late_heads / head_mass
    late_raw = feature_capture.attn_c2p_raw.index_select(0, late_index)[
        :, local
    ].index_select(1, positive_index)
    late_conditional = feature_capture.attn_c2p_conditional.index_select(0, late_index)[
        :, local
    ].index_select(1, positive_index)
    positive_final = final_cam[local].index_select(0, positive_index).flatten(1)
    normalized = upsample_and_normalize_active_cams(
        positive_final.reshape(len(positive), 28, 28)
    )
    if isinstance(normalized, torch.Tensor):
        normalized_np = normalized.detach().cpu().numpy()
    else:
        normalized_np = np.asarray(normalized)
    confusions = _threshold_confusions(
        normalized_np, positive, context.mask, thresholds
    )

    payload = {
        "image_id": np.asarray(context.image_id),
        "variant_code": np.asarray(variant),
        "positive_class_ids": positive.astype(np.int64, copy=False),
        "image_labels": context.labels.astype(np.uint8, copy=False),
        "pair_class_ids": pairs,
        "late_layers_one_based": np.asarray(LATE_LAYER_NUMBERS, dtype=np.int16),
        "thresholds": thresholds.astype(np.float64, copy=False),
        "patch_label_counts": context.patch_counts,
        "region_masks_rho05": context.region_rho05,
        "region_masks_rho07": context.region_rho07,
        "class_logits_all": _as_float32(
            feature_capture.last_class_tokens[local].mean(dim=-1)
        ),
        "patch_class_logits_all": _as_float32(patch_class_logits[local]),
        "patch_head_logits_positive": _as_float32(
            patch_logits[local].index_select(0, positive_index).flatten(1)
        ),
        "feature_post_l10_l12": _as_float32(
            feature_capture.feature_post_scores.index_select(0, late_index)[
                :, local
            ].index_select(1, positive_index)
        ),
        "feature_both_axis_removed_l10_l12": _as_float32(
            both_removed[:, local].index_select(1, positive_index)
        ),
        "positive_pair_raw_cosine_l10_l12": _pair_values(pair_raw, local, pairs),
        "positive_pair_residual_cosine_l10_l12": _pair_values(
            pair_residual, local, pairs
        ),
        "attention_c2p_raw_l10_l12": _as_float32(late_raw),
        "attention_c2p_conditional_l10_l12": _as_float32(late_conditional),
        "attention_head_region_raw_rho05": _head_region_means(
            late_heads, positive, context.region_rho05
        ),
        "attention_head_region_conditional_rho05": _head_region_means(
            late_heads_conditional, positive, context.region_rho05
        ),
        "attention_head_region_raw_rho07": _head_region_means(
            late_heads, positive, context.region_rho07
        ),
        "attention_head_region_conditional_rho07": _head_region_means(
            late_heads_conditional, positive, context.region_rho07
        ),
        "c2c_pre_offdiag_mass": _as_float32(runner_capture.pre_offdiag_mass[:, local]),
        "c2c_pre_diagonal_mass": _as_float32(
            runner_capture.pre_diagonal_mass[:, local]
        ),
        "c2c_pre_class_mass": _as_float32(runner_capture.pre_class_mass[:, local]),
        "c2c_post_offdiag_mass": _as_float32(
            runner_capture.post_offdiag_mass[:, local]
        ),
        "c2c_post_diagonal_mass": _as_float32(
            runner_capture.post_diagonal_mass[:, local]
        ),
        "c2c_post_class_mass": _as_float32(runner_capture.post_class_mass[:, local]),
        "final_cam": _as_float32(positive_final),
        "threshold_confusions": confusions.astype(np.int64, copy=False),
        "source_signal_sha256": np.asarray(context.source_sha256),
    }
    _validate_payload(payload, num_heads=int(late_heads.shape[1]))
    return payload


def _validate_payload(payload: Mapping[str, np.ndarray], *, num_heads: int) -> None:
    missing = SIGNAL_KEYS.difference(payload)
    extra = set(payload).difference(SIGNAL_KEYS)
    if missing or extra:
        raise ValueError(
            f"signal schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    image_id = np.asarray(payload["image_id"])
    variant = str(np.asarray(payload["variant_code"]).item())
    positive = np.asarray(payload["positive_class_ids"])
    pairs = np.asarray(payload["pair_class_ids"])
    classes = len(positive)
    pair_count = classes * (classes - 1) // 2
    if image_id.ndim != 0 or not isinstance(image_id.item(), str):
        raise TypeError("image_id must be a scalar string")
    if variant not in VARIANT_CODES:
        raise ValueError(f"invalid C2C variant {variant}")
    if (
        positive.dtype != np.int64
        or positive.ndim != 1
        or not classes
        or np.any(np.diff(positive) <= 0)
    ):
        raise TypeError("positive_class_ids must be sorted unique int64")
    if pairs.dtype != np.int64 or pairs.shape != (pair_count, 2):
        raise TypeError("pair_class_ids has an invalid shape or dtype")
    if not np.array_equal(pairs, _positive_pairs(positive)):
        raise ValueError("pair_class_ids are not the exact unordered positive pairs")
    labels = np.asarray(payload["image_labels"])
    if labels.dtype != np.uint8 or labels.shape != (EXPECTED_CLASSES,):
        raise TypeError("image_labels must be uint8[20]")
    if not np.array_equal(np.flatnonzero(labels), positive):
        raise ValueError("positive classes do not match image labels")
    late_layers = np.asarray(payload["late_layers_one_based"])
    if late_layers.dtype != np.int16 or not np.array_equal(
        late_layers, np.asarray(LATE_LAYER_NUMBERS, dtype=np.int16)
    ):
        raise ValueError("late layer numbering drifted")
    thresholds = np.asarray(payload["thresholds"])
    if thresholds.dtype != np.float64 or thresholds.shape != (41,):
        raise TypeError("thresholds must be float64[41]")
    if not np.array_equal(thresholds, cam_threshold_grid()):
        raise ValueError("threshold grid drifted from the pre-registered grid")

    counts = np.asarray(payload["patch_label_counts"])
    if counts.dtype != np.uint16 or counts.shape != (EXPECTED_PATCHES, 22):
        raise TypeError("patch_label_counts must be uint16[784,22]")
    if not np.all(counts.sum(axis=-1) == 256):
        raise ValueError("each patch must contain exactly 256 mask pixels")
    for name in ("region_masks_rho05", "region_masks_rho07"):
        regions = np.asarray(payload[name])
        if regions.dtype != np.int8 or regions.shape != (classes, EXPECTED_PATCHES):
            raise TypeError(f"{name} must be int8[K,784]")
        if np.any((regions < 0) | (regions > 4)):
            raise ValueError(f"{name} contains an invalid region code")

    expected_float_shapes = {
        "class_logits_all": (EXPECTED_CLASSES,),
        "patch_class_logits_all": (EXPECTED_CLASSES,),
        "patch_head_logits_positive": (classes, EXPECTED_PATCHES),
        "feature_post_l10_l12": (3, classes, EXPECTED_PATCHES),
        "feature_both_axis_removed_l10_l12": (
            3,
            classes,
            EXPECTED_PATCHES,
        ),
        "positive_pair_raw_cosine_l10_l12": (3, pair_count),
        "positive_pair_residual_cosine_l10_l12": (3, pair_count),
        "attention_c2p_raw_l10_l12": (3, classes, EXPECTED_PATCHES),
        "attention_c2p_conditional_l10_l12": (
            3,
            classes,
            EXPECTED_PATCHES,
        ),
        "c2c_pre_offdiag_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "c2c_pre_diagonal_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "c2c_pre_class_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "c2c_post_offdiag_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "c2c_post_diagonal_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "c2c_post_class_mass": (EXPECTED_LAYERS, num_heads, EXPECTED_CLASSES),
        "final_cam": (classes, EXPECTED_PATCHES),
    }
    for name, expected in expected_float_shapes.items():
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != expected:
            raise TypeError(
                f"{name} must be float32{expected}, got {value.shape}/{value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    head_shape = (3, num_heads, classes, len(REGION_CODES))
    for name in (
        "attention_head_region_raw_rho05",
        "attention_head_region_conditional_rho05",
        "attention_head_region_raw_rho07",
        "attention_head_region_conditional_rho07",
    ):
        value = np.asarray(payload[name])
        if value.dtype != np.float32 or value.shape != head_shape:
            raise TypeError(f"{name} must be float32{head_shape}")
        if np.isinf(value).any():
            raise ValueError(f"{name} contains infinity")
    conditional = np.asarray(payload["attention_c2p_conditional_l10_l12"])
    if not np.allclose(conditional.sum(axis=-1), 1.0, rtol=0, atol=5e-6):
        raise ValueError("conditional C2P rows do not sum to one")
    raw = np.asarray(payload["attention_c2p_raw_l10_l12"])
    reproduced_conditional = raw / raw.sum(axis=-1, keepdims=True)
    if not np.allclose(conditional, reproduced_conditional, rtol=0, atol=5e-7):
        raise ValueError("conditional C2P does not reproduce from raw C2P")

    pre_offdiag = np.asarray(payload["c2c_pre_offdiag_mass"])
    pre_diagonal = np.asarray(payload["c2c_pre_diagonal_mass"])
    pre_class = np.asarray(payload["c2c_pre_class_mass"])
    post_offdiag = np.asarray(payload["c2c_post_offdiag_mass"])
    post_diagonal = np.asarray(payload["c2c_post_diagonal_mass"])
    post_class = np.asarray(payload["c2c_post_class_mass"])
    if np.any(pre_offdiag < 0) or np.any(post_offdiag < 0):
        raise ValueError("C2C off-diagonal mass must be non-negative")
    for name, observed, expected in (
        ("pre C2C decomposition", pre_diagonal + pre_offdiag, pre_class),
        ("post C2C decomposition", post_diagonal + post_offdiag, post_class),
    ):
        if not np.allclose(observed, expected, rtol=0, atol=STRICT_TOLERANCE):
            raise ValueError(f"{name} failed")
    selected_layers = set(variant_layer_indices(variant))
    for layer in range(EXPECTED_LAYERS):
        if layer in selected_layers:
            invariants = (
                post_offdiag[layer],
                post_diagonal[layer] - pre_class[layer],
                post_class[layer] - pre_class[layer],
            )
        else:
            invariants = (
                post_offdiag[layer] - pre_offdiag[layer],
                post_diagonal[layer] - pre_diagonal[layer],
                post_class[layer] - pre_class[layer],
            )
        if max(float(np.abs(value).max()) for value in invariants) >= STRICT_TOLERANCE:
            raise ValueError(f"C2C intervention invariant failed at L{layer + 1}")
    confusions = np.asarray(payload["threshold_confusions"])
    if confusions.dtype != np.int64 or confusions.shape != (41, 21, 21):
        raise TypeError("threshold_confusions must be int64[41,21,21]")
    valid_pixels = 448 * 448 - int(counts[:, 21].sum())
    if np.any(confusions < 0) or not np.all(
        confusions.sum(axis=(1, 2)) == valid_pixels
    ):
        raise ValueError("threshold confusion matrices have invalid pixel totals")
    source_hash = str(np.asarray(payload["source_signal_sha256"]).item())
    if len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError("source_signal_sha256 is invalid")


def _image_contexts(
    *,
    inputs: RuntimeInputs,
    image_ids: Sequence[str],
    labels: np.ndarray,
    masks: torch.Tensor,
    remaining: int,
    source_hashes: dict[str, str],
) -> list[ImageContext]:
    if remaining < 1 or remaining > len(image_ids):
        raise ValueError(
            f"remaining image count {remaining} is outside batch size {len(image_ids)}"
        )
    if labels.shape != (len(image_ids), EXPECTED_CLASSES):
        raise RuntimeError(
            f"batch labels have shape {labels.shape}, expected "
            f"({len(image_ids)},{EXPECTED_CLASSES})"
        )
    if masks.ndim != 3 or tuple(masks.shape) != (len(image_ids), 448, 448):
        raise RuntimeError(
            f"batch masks have shape {tuple(masks.shape)}, expected "
            f"({len(image_ids)},448,448)"
        )
    contexts = []
    for local, image_id in enumerate(image_ids[:remaining]):
        row = np.asarray(labels[local], dtype=np.uint8)
        if not np.isin(row, (0, 1)).all():
            raise RuntimeError(f"image {image_id} has non-binary labels")
        positive = np.flatnonzero(row > 0).astype(np.int64)
        if not len(positive):
            raise RuntimeError(f"image {image_id} has no positive class")
        if image_id in source_hashes:
            raise RuntimeError(f"duplicate image encountered: {image_id}")
        path = source_signal_path(inputs, image_id)
        digest = sha256_file(path)
        if digest != inputs.source_artifact_sha256[image_id]:
            raise RuntimeError(f"audited source hash mismatch for {image_id}")
        source_hashes[image_id] = digest
        source = load_source_signal(inputs, image_id)
        if not np.array_equal(source["positive_class_ids"], positive):
            raise RuntimeError(f"source positive-class mismatch for {image_id}")
        counts, rho05, rho07 = mask_region_codes(masks[local], positive)
        for name, observed, expected in (
            ("patch counts", counts, source["patch_label_counts"]),
            ("rho05 regions", rho05, source["region_masks_rho05"]),
            ("rho07 regions", rho07, source["region_masks_rho07"]),
        ):
            if not np.array_equal(observed, expected):
                raise RuntimeError(f"{name} mismatch for {image_id}")
        contexts.append(
            ImageContext(
                local_index=local,
                image_id=image_id,
                labels=row,
                positive=positive,
                mask=masks[local].cpu().numpy().copy(),
                patch_counts=counts,
                region_rho05=rho05,
                region_rho07=rho07,
                source=source,
                source_sha256=digest,
            )
        )
    return contexts


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank manifest row {number}: {path}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"manifest row {number} is not an object: {path}")
        rows.append(value)
    return rows


def _verify_derived_artifacts(
    output: Path,
    expected_records: Sequence[Mapping[str, object]],
    *,
    requested: int,
    num_heads: int,
) -> dict[str, object]:
    """Reload every output artifact and fail before the completion marker."""

    manifest_path = output / "manifest.jsonl"
    observed = _read_jsonl(manifest_path)
    expected = [dict(row) for row in expected_records]
    expected_count = requested * len(VARIANT_CODES)
    if len(observed) != expected_count or observed != expected:
        raise RuntimeError(
            f"global manifest mismatch: observed={len(observed)}, "
            f"expected={expected_count}"
        )

    verified: set[tuple[str, str]] = set()
    for number, row in enumerate(observed, 1):
        required = {
            "image_id",
            "variant",
            "layers_one_based",
            "positive_class_ids",
            "positive_pair_count",
            "artifact_path",
            "artifact_sha256",
            "source_signal_sha256",
        }
        if set(row) != required:
            raise RuntimeError(
                f"manifest row {number} keys differ: {sorted(set(row) ^ required)}"
            )
        image_id = str(row["image_id"])
        variant = str(row["variant"])
        if variant not in VARIANT_CODES:
            raise RuntimeError(f"manifest row {number} has invalid variant {variant}")
        key = (variant, image_id)
        if key in verified:
            raise RuntimeError(f"duplicate derived artifact manifest key: {key}")
        verified.add(key)
        expected_relative = Path("signals") / variant / f"{image_id}.npz"
        if str(row["artifact_path"]) != str(expected_relative):
            raise RuntimeError(
                f"manifest row {number} has invalid artifact path "
                f"{row['artifact_path']!r}"
            )
        if row["layers_one_based"] != list(C2C_VARIANT_LAYERS_1BASED[variant]):
            raise RuntimeError(f"manifest row {number} layer assignment drifted")
        path = output / expected_relative
        payload = reload_npz_checked(
            path,
            expected_sha256=str(row["artifact_sha256"]),
            expected_image_id=image_id,
            allow_nan_keys=ALLOW_NAN_SIGNAL_KEYS,
        )
        _validate_payload(payload, num_heads=num_heads)
        if str(payload["variant_code"].item()) != variant:
            raise RuntimeError(f"derived artifact variant mismatch: {path}")
        positive = np.asarray(payload["positive_class_ids"])
        if positive.tolist() != row["positive_class_ids"]:
            raise RuntimeError(f"derived artifact class IDs mismatch: {path}")
        if int(row["positive_pair_count"]) != len(payload["pair_class_ids"]):
            raise RuntimeError(f"derived artifact pair count mismatch: {path}")
        if str(payload["source_signal_sha256"].item()) != str(
            row["source_signal_sha256"]
        ):
            raise RuntimeError(f"derived artifact source hash mismatch: {path}")

    variant_manifest_hashes = {}
    for variant in VARIANT_CODES:
        path = output / "signals" / variant / "manifest.jsonl"
        variant_rows = _read_jsonl(path)
        expected_variant = [row for row in expected if row["variant"] == variant]
        if variant_rows != expected_variant or len(variant_rows) != requested:
            raise RuntimeError(f"{variant} manifest content/count mismatch")
        variant_manifest_hashes[variant] = sha256_file(path)
    return {
        "verified_artifact_count": len(verified),
        "global_manifest_sha256": sha256_file(manifest_path),
        "variant_manifest_sha256": variant_manifest_hashes,
    }


def _verify_runtime_source_hashes(state: Mapping[str, object]) -> int:
    hashes = state.get("runtime_source_sha256")
    if not isinstance(hashes, Mapping):
        raise TypeError("runtime source metadata lacks SHA-256 records")
    for relative, expected in hashes.items():
        path = REPO_ROOT / str(relative)
        if sha256_file(path) != str(expected):
            raise RuntimeError(f"runtime source changed during inference: {relative}")
    return len(hashes)


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
    if args.batch_size != 8:
        raise ValueError("Validation C requires --batch-size 8")
    if args.num_workers < 0 or args.limit < 0:
        raise ValueError("--num-workers and --limit must be non-negative")
    if args.limit >= EXPECTED_IMAGES:
        raise ValueError(
            f"use --limit 0 for the full {EXPECTED_IMAGES}-image run; "
            "a positive --limit is smoke-only"
        )
    return args


def execute(args: argparse.Namespace) -> None:
    if args.batch_size != 8:
        raise ValueError("Validation C requires exact --batch-size 8")
    if args.limit < 0 or args.limit >= EXPECTED_IMAGES:
        raise ValueError(
            f"--limit must be 0 (full) or in [1,{EXPECTED_IMAGES - 1}] (smoke)"
        )
    if args.allow_uncommitted_source and args.limit == 0:
        raise ValueError("uncommitted runtime sources are allowed only for smoke runs")
    inputs = resolve_runtime_inputs(args.source_metadata, args.model)
    output_target = assert_new_output(
        args.output_dir,
        (
            inputs.source_metadata_path,
            inputs.experiment2_root,
            inputs.checkpoint,
            inputs.voc_root,
        ),
    )
    state = runtime_source_state(RUNTIME_SOURCES)
    enforce_production_source(
        state,
        allow_uncommitted=bool(args.allow_uncommitted_source),
        limit=int(args.limit),
    )
    output, metadata, log = initialize_run_directory(
        output_target,
        analysis="experiment3_validation_c_late_c2c_causal_intervention",
        model=args.model,
        inputs=inputs,
        execution={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "limit": args.limit,
            "seed": args.seed,
            "allow_uncommitted_source": bool(args.allow_uncommitted_source),
            "variant_order": list(VARIANT_CODES),
        },
        git=state,
    )
    started = time.perf_counter()
    head_outputs: list[torch.Tensor] = []
    head_handle = None
    source_hashes: dict[str, str] = {}
    structural_records: list[dict[str, object]] = []
    activation_audits: list[dict[str, object]] = []
    derived_records: list[dict[str, object]] = []
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
        dataset, loader, requested, context_images = make_dataset_and_loader(
            inputs,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            pin_memory=not args.no_pin_memory,
        )
        del dataset
        metadata["execution"]["requested_images"] = requested
        metadata["execution"]["batch_context_images"] = context_images
        thresholds = cam_threshold_grid()
        if thresholds.shape != (41,):
            raise RuntimeError("Validation C requires the pre-registered 41 thresholds")
        metadata["intervention_contract"] = {
            "mode": "mass_preserving_self_reroute",
            "variant_layers_one_based": {
                name: list(layers) for name, layers in C2C_VARIANT_LAYERS_1BASED.items()
            },
            "operator": "all off-diagonal class-key mass moves to matching self key independently for every class query and head",
            "unchanged_at_intervened_layer": [
                "class_to_patch_weights",
                "class_group_mass",
                "row_sum",
                "all_patch_query_rows",
            ],
            "token_layout": "20 leading class tokens + 784 trailing patch tokens",
            "normalization": "vanilla global softmax",
            "late_layers_one_based": list(LATE_LAYER_NUMBERS),
            "head_region_order": list(HEAD_REGION_NAMES),
            "region_rhos": [0.5, 0.7],
            "fixed_axis": "normalized all-ones embedding direction",
            "thresholds": thresholds.tolist(),
            "primary_cam_threshold": 0.45,
            "prediction": "GT-positive gating; bilinear 28->448 align_corners=False; per-class minmax; background-first ties",
        }

        def head_hook(_module, _inputs, value):
            if not isinstance(value, torch.Tensor):
                raise TypeError("patch head output must be a tensor")
            head_outputs.append(value.detach())
            return None

        head_handle = model.head.register_forward_hook(head_hook)
        signal_root = output / "signals"
        signal_root.mkdir()
        signal_dirs = {}
        for variant in VARIANT_CODES:
            signal_dirs[variant] = signal_root / variant
            signal_dirs[variant].mkdir()

        saved_counts = {variant: 0 for variant in VARIANT_CODES}
        maximum_native_difference = {variant: 0.0 for variant in VARIANT_CODES}
        maximum_source_difference = {name: 0.0 for name in SOURCE_EQUIVALENCE_FIELDS}
        maximum_control_difference = {
            "C1_minus_C0": {
                "final_patch_tokens": 0.0,
                "patch_head_logits": 0.0,
                "all_layer_all_head_c2p": 0.0,
                "native_final_cam": 0.0,
            },
            "C5_minus_C4": {
                "final_patch_tokens": 0.0,
                "patch_head_logits": 0.0,
                "all_layer_all_head_c2p": 0.0,
                "native_final_cam": 0.0,
            },
        }
        maximum_axis_raw_difference = 0.0
        maximum_head_mean_difference = 0.0
        maximum_intervention_checks: dict[str, float] = {}
        processed_images = 0
        control_snapshots: dict[str, ControlSnapshot] = {}

        with ExitStack() as stack:
            global_manifest = stack.enter_context(
                (output / "manifest.jsonl").open("x", encoding="utf-8")
            )
            variant_manifests = {
                variant: stack.enter_context(
                    (signal_dirs[variant] / "manifest.jsonl").open(
                        "x", encoding="utf-8"
                    )
                )
                for variant in VARIANT_CODES
            }
            with (
                SignalCollector(
                    model, num_classes=EXPECTED_CLASSES
                ) as feature_collector,
                C2CRunnerCollector(model) as c2c_collector,
            ):
                with torch.inference_mode():
                    for batch_number, batch in enumerate(loader, start=1):
                        images = batch["image"].to(
                            device,
                            non_blocking=(
                                device.type == "cuda" and not args.no_pin_memory
                            ),
                        )
                        image_ids = list(batch["name"])
                        labels = batch["label"].cpu().numpy()
                        masks = batch["mask"]
                        remaining = min(len(image_ids), requested - processed_images)
                        image_contexts = _image_contexts(
                            inputs=inputs,
                            image_ids=image_ids,
                            labels=labels,
                            masks=masks,
                            remaining=remaining,
                            source_hashes=source_hashes,
                        )
                        for variant in VARIANT_CODES:
                            head_outputs.clear()
                            feature_collector.clear(
                                expected_num_patches=EXPECTED_PATCHES
                            )
                            c2c_collector.clear()
                            hook_counts_before = tuple(
                                len(block.attn.normalizer._forward_hooks)
                                for block in model.blocks
                            )
                            intervention = C2CIntervention.from_variant(model, variant)
                            with intervention:
                                native_cam = model(images)
                            if not isinstance(native_cam, torch.Tensor):
                                raise TypeError(
                                    f"{variant} native CAM output must be a tensor"
                                )
                            expected_cam_shape = (
                                len(image_ids),
                                EXPECTED_CLASSES,
                                28,
                                28,
                            )
                            if tuple(native_cam.shape) != expected_cam_shape:
                                raise RuntimeError(
                                    f"{variant} native CAM shape "
                                    f"{tuple(native_cam.shape)} != "
                                    f"{expected_cam_shape}"
                                )
                            if not bool(torch.isfinite(native_cam).all()):
                                raise RuntimeError(
                                    f"{variant} native CAM is non-finite"
                                )
                            hook_counts_after = tuple(
                                len(block.attn.normalizer._forward_hooks)
                                for block in model.blocks
                            )
                            feature_capture = feature_collector.consume()
                            runner_capture = c2c_collector.consume()
                            check_values = _validate_intervention_capture(
                                intervention,
                                runner_capture,
                                variant=variant,
                                hook_counts_before=hook_counts_before,
                                hook_counts_after=hook_counts_after,
                            )
                            for name, value in check_values.items():
                                maximum_intervention_checks[name] = max(
                                    maximum_intervention_checks.get(name, 0.0), value
                                )
                            activation_audits.append(
                                {
                                    "batch_number": batch_number,
                                    "variant": variant,
                                    "context_image_ids": json.dumps(image_ids),
                                    "saved_image_ids": json.dumps(
                                        [item.image_id for item in image_contexts]
                                    ),
                                    "layers_one_based": json.dumps(
                                        list(intervention.layer_numbers_1based)
                                    ),
                                    "activation_counts_one_based": json.dumps(
                                        intervention.activation_counts_1based,
                                        sort_keys=True,
                                    ),
                                    "hook_counts_restored": (
                                        hook_counts_before == hook_counts_after
                                    ),
                                }
                            )
                            for record in intervention.records:
                                row = {
                                    "batch_number": batch_number,
                                    "variant": variant,
                                    "context_image_ids": json.dumps(image_ids),
                                    "saved_image_ids": json.dumps(
                                        [item.image_id for item in image_contexts]
                                    ),
                                    "variant_layers_one_based": json.dumps(
                                        list(intervention.layer_numbers_1based)
                                    ),
                                    "activation_count": intervention.activation_counts[
                                        record.layer_index
                                    ],
                                    **record.to_dict(),
                                }
                                structural_records.append(row)

                            if len(head_outputs) != 1:
                                raise RuntimeError(
                                    f"{variant} patch head fired "
                                    f"{len(head_outputs)} times"
                                )
                            patch_logits = head_outputs[0]
                            patch_class_logits = _patch_class_logits(
                                args.model, model, patch_logits
                            )
                            native_stages = decompose_native_cam_reduced(
                                args.model,
                                patch_logits,
                                feature_capture.attn_c2p_raw,
                                feature_capture.patch_to_patch_sum,
                                num_classes=EXPECTED_CLASSES,
                            )
                            reconstructed_cam = native_stages["final_cam"]
                            if (
                                reconstructed_cam.shape != native_cam.shape
                                or reconstructed_cam.dtype != native_cam.dtype
                                or reconstructed_cam.device != native_cam.device
                            ):
                                raise RuntimeError(
                                    f"{variant} reconstructed/native CAM metadata "
                                    "mismatch"
                                )
                            native_difference = float(
                                (reconstructed_cam.float() - native_cam.float())
                                .abs()
                                .max()
                                .item()
                            )
                            if native_difference >= STRICT_TOLERANCE:
                                raise RuntimeError(
                                    f"{variant} native CAM reconstruction failed: "
                                    f"{native_difference}"
                                )
                            maximum_native_difference[variant] = max(
                                maximum_native_difference[variant], native_difference
                            )
                            head_mean_difference = float(
                                (
                                    runner_capture.c2p_heads.float().mean(dim=2)
                                    - feature_capture.attn_c2p_raw.float()
                                )
                                .abs()
                                .max()
                                .item()
                            )
                            if head_mean_difference >= STRICT_TOLERANCE:
                                raise RuntimeError(
                                    f"{variant} per-head C2P capture mismatch: "
                                    f"{head_mean_difference}"
                                )
                            maximum_head_mean_difference = max(
                                maximum_head_mean_difference, head_mean_difference
                            )
                            (
                                both_removed,
                                pair_raw,
                                pair_residual,
                                axis_raw_difference,
                            ) = _late_axis_products(runner_capture, feature_capture)
                            maximum_axis_raw_difference = max(
                                maximum_axis_raw_difference, axis_raw_difference
                            )

                            if variant == "C0":
                                control_snapshots["C0"] = _control_snapshot(
                                    feature_capture,
                                    runner_capture,
                                    patch_logits,
                                    native_cam,
                                )
                                for item in image_contexts:
                                    differences = _verify_c0_source(
                                        item,
                                        feature_capture,
                                        patch_logits,
                                        patch_class_logits,
                                        native_cam,
                                    )
                                    for name, difference in differences.items():
                                        maximum_source_difference[name] = max(
                                            maximum_source_difference[name], difference
                                        )
                            elif variant == "C1":
                                differences = _compare_control(
                                    control_snapshots.pop("C0"),
                                    feature_capture,
                                    runner_capture,
                                    patch_logits,
                                    native_cam,
                                    name="C1_minus_C0",
                                )
                                for name, difference in differences.items():
                                    maximum_control_difference["C1_minus_C0"][name] = (
                                        max(
                                            maximum_control_difference["C1_minus_C0"][
                                                name
                                            ],
                                            difference,
                                        )
                                    )
                            elif variant == "C4":
                                control_snapshots["C4"] = _control_snapshot(
                                    feature_capture,
                                    runner_capture,
                                    patch_logits,
                                    native_cam,
                                )
                            elif variant == "C5":
                                differences = _compare_control(
                                    control_snapshots.pop("C4"),
                                    feature_capture,
                                    runner_capture,
                                    patch_logits,
                                    native_cam,
                                    name="C5_minus_C4",
                                )
                                for name, difference in differences.items():
                                    maximum_control_difference["C5_minus_C4"][name] = (
                                        max(
                                            maximum_control_difference["C5_minus_C4"][
                                                name
                                            ],
                                            difference,
                                        )
                                    )

                            for item in image_contexts:
                                payload = _build_payload(
                                    variant=variant,
                                    context=item,
                                    feature_capture=feature_capture,
                                    runner_capture=runner_capture,
                                    both_removed=both_removed,
                                    pair_raw=pair_raw,
                                    pair_residual=pair_residual,
                                    patch_logits=patch_logits,
                                    patch_class_logits=patch_class_logits,
                                    final_cam=native_cam,
                                    thresholds=thresholds,
                                )
                                path = signal_dirs[variant] / f"{item.image_id}.npz"
                                digest = save_npz_atomic(path, payload)
                                record = {
                                    "image_id": item.image_id,
                                    "variant": variant,
                                    "layers_one_based": list(
                                        C2C_VARIANT_LAYERS_1BASED[variant]
                                    ),
                                    "positive_class_ids": item.positive.tolist(),
                                    "positive_pair_count": int(
                                        len(item.positive)
                                        * (len(item.positive) - 1)
                                        // 2
                                    ),
                                    "artifact_path": str(path.relative_to(output)),
                                    "artifact_sha256": digest,
                                    "source_signal_sha256": item.source_sha256,
                                }
                                serialized = json.dumps(
                                    record, sort_keys=True, allow_nan=False
                                )
                                global_manifest.write(serialized + "\n")
                                global_manifest.flush()
                                variant_manifests[variant].write(serialized + "\n")
                                variant_manifests[variant].flush()
                                derived_records.append(record)
                                saved_counts[variant] += 1

                            del (
                                native_cam,
                                feature_capture,
                                runner_capture,
                                patch_logits,
                                patch_class_logits,
                                native_stages,
                                both_removed,
                                pair_raw,
                                pair_residual,
                            )
                        if control_snapshots:
                            raise RuntimeError(
                                f"unconsumed control snapshots: {tuple(control_snapshots)}"
                            )
                        processed_images += len(image_contexts)
                        log(
                            f"batch={batch_number} images={processed_images}/{requested} "
                            f"variants={','.join(VARIANT_CODES)}"
                        )
                        del images, image_contexts

        if processed_images != requested:
            raise RuntimeError(
                f"processed {processed_images} images, expected {requested}"
            )
        for variant in VARIANT_CODES:
            count = len(list(signal_dirs[variant].glob("*.npz")))
            if saved_counts[variant] != requested or count != requested:
                raise RuntimeError(
                    f"{variant} saved {saved_counts[variant]}/{count}, expected "
                    f"{requested}"
                )
        if len(source_hashes) != requested:
            raise RuntimeError(
                f"verified {len(source_hashes)} source signals, expected {requested}"
            )
        for image_id, expected_hash in source_hashes.items():
            actual_hash = sha256_file(source_signal_path(inputs, image_id))
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Experiment 2 signal changed during inference: {image_id}"
                )
        assert_inputs_unchanged(inputs)
        runtime_sources_verified = _verify_runtime_source_hashes(state)
        for block in model.blocks:
            if block.attn.normalizer._forward_hooks:
                raise RuntimeError("C2C/runner normalizer hook leaked after collection")

        num_heads = int(model.blocks[0].attn.num_heads)
        if any(int(block.attn.num_heads) != num_heads for block in model.blocks):
            raise RuntimeError("attention head count differs across model blocks")
        derived_verification = _verify_derived_artifacts(
            output,
            derived_records,
            requested=requested,
            num_heads=num_heads,
        )

        _write_csv(output / "structural_records.csv", structural_records)
        json_dump(
            output / "structural_records.json",
            {
                "schema_version": 1,
                "records": structural_records,
                "activation_audits": activation_audits,
            },
        )
        finish_metadata(
            output,
            metadata,
            started=started,
            updates={
                "processed_images": processed_images,
                "saved_artifacts": int(sum(saved_counts.values())),
                "saved_by_variant": saved_counts,
                "manifest": "manifest.jsonl",
                "signals": "signals",
                "structural_records_csv": "structural_records.csv",
                "structural_records_json": "structural_records.json",
                "structural_record_count": len(structural_records),
                "activation_audit_count": len(activation_audits),
                "native_cam_max_abs_diff_by_variant": maximum_native_difference,
                "c0_experiment2_max_abs_diff": maximum_source_difference,
                "negative_control_max_abs_diff": maximum_control_difference,
                "axis_raw_feature_max_abs_diff": maximum_axis_raw_difference,
                "head_mean_c2p_max_abs_diff": maximum_head_mean_difference,
                "intervention_capture_max_abs_diff": maximum_intervention_checks,
                "source_signal_hashes_verified": len(source_hashes),
                "runtime_source_hashes_verified": runtime_sources_verified,
                "derived_artifact_verification": derived_verification,
                "derived_artifact_integrity_passed": True,
                "source_integrity_passed": True,
                "signal_schema": sorted(SIGNAL_KEYS),
                "structural_records_csv_sha256": sha256_file(
                    output / "structural_records.csv"
                ),
                "structural_records_json_sha256": sha256_file(
                    output / "structural_records.json"
                ),
            },
        )
        log(
            f"complete images={processed_images} artifacts={sum(saved_counts.values())} "
            f"native={maximum_native_difference} controls={maximum_control_difference}"
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "structural_records_collected": len(structural_records),
                "activation_audits_collected": len(activation_audits),
            }
        )
        json_dump(output / "metadata.json", metadata)
        log(f"FAILED: {error!r}")
        raise
    finally:
        if head_handle is not None:
            head_handle.remove()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
