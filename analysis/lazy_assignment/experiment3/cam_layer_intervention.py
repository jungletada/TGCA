"""Pure CAM layer-readout helpers for Experiment 3 Validation B.

The functions in this module deliberately separate three operations that are
easy to conflate:

* host-native aggregation of *raw* class-to-patch attention;
* host-native fusion with the non-negative patch CAM; and
* propagation with one supplied, fixed all-layer patch-to-patch matrix.

No function loads a model, checkpoint, dataset, or Experiment 2 artifact.  In
particular, patch-conditionalized attention must never be passed off as native
CAM attention: callers are responsible for supplying the raw head-mean
softmax weights recorded by Experiment 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from analysis.lazy_assignment.experiment2.evaluation_metrics import (
    iou_from_confusion,
    upsample_and_normalize_active_cams,
    voc_confusion_matrix,
)


NUM_LAYERS = 12
VOC_FOREGROUND_CLASSES = 20
VOC_SEGMENTATION_CLASSES = 21
TRANSFORMED_CROP_SIZE = (448, 448)


@dataclass(frozen=True)
class CamVariantSpec:
    """One pre-registered Validation B attention readout."""

    code: str
    name: str
    layers_one_based: tuple[int, ...]


CAM_VARIANT_SPECS = (
    CamVariantSpec("B0", "native_last3", (10, 11, 12)),
    CamVariantSpec("B1", "l10_only", (10,)),
    CamVariantSpec("B2", "l11_only", (11,)),
    CamVariantSpec("B3", "l12_only", (12,)),
    CamVariantSpec("B4", "l10_l11", (10, 11)),
    CamVariantSpec("B5", "l4_l6_control", (4, 5, 6)),
)


@dataclass(frozen=True)
class CamReadoutStages:
    """Host-native outputs for one CAM attention-source variant."""

    variant_code: str
    variant_name: str
    layers_one_based: tuple[int, ...]
    raw_c2p: torch.Tensor
    preprop_cam: torch.Tensor
    final_cam: torch.Tensor


@dataclass(frozen=True)
class NativeBestThreshold:
    """A deterministic threshold selected solely from native B0."""

    index: int
    threshold: float
    native_metric: float


_SPEC_BY_ALIAS = {
    alias.lower(): spec
    for spec in CAM_VARIANT_SPECS
    for alias in (spec.code, spec.name)
}
_SPEC_BY_ALIAS.update(
    {
        "native-last3": CAM_VARIANT_SPECS[0],
        "last3": CAM_VARIANT_SPECS[0],
        "l10-l11": CAM_VARIANT_SPECS[4],
        "mid3": CAM_VARIANT_SPECS[5],
        "mid3_control": CAM_VARIANT_SPECS[5],
        "l4-l6": CAM_VARIANT_SPECS[5],
    }
)


def _host_kind(host: str) -> str:
    if not isinstance(host, str) or not host.strip():
        raise TypeError("host must be a non-empty string")
    normalized = host.lower().replace("+", "plus").replace("_", "").replace("-", "")
    if normalized in {"mctformer", "mctformerv2", "mctformerv2cam"}:
        return "mctformer"
    if normalized in {"mctformerplus", "mctformerpluscam"}:
        return "mctformer_plus"
    raise ValueError(f"unsupported CAM host: {host!r}")


def resolve_cam_variant(variant: str | CamVariantSpec) -> CamVariantSpec:
    """Resolve a B0--B5 code or stable descriptive name."""

    if isinstance(variant, CamVariantSpec):
        if variant not in CAM_VARIANT_SPECS:
            raise ValueError("custom CAM variants are outside the pre-registered set")
        return variant
    if not isinstance(variant, str) or not variant.strip():
        raise TypeError("variant must be a non-empty string or CamVariantSpec")
    try:
        return _SPEC_BY_ALIAS[variant.strip().lower()]
    except KeyError as error:
        allowed = ", ".join(spec.code for spec in CAM_VARIANT_SPECS)
        raise ValueError(
            f"unknown CAM variant {variant!r}; expected {allowed}"
        ) from error


def _floating_tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, got shape {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.numel() == 0 or any(int(size) < 1 for size in value.shape):
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def _nonnegative(value: torch.Tensor, name: str) -> None:
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative")


def aggregate_raw_c2p(
    host: str,
    c2p_layers: torch.Tensor,
    variant: str | CamVariantSpec,
) -> torch.Tensor:
    """Aggregate raw head-mean ``A_c2p`` with the exact host convention.

    Args:
        host: MCTformer or MCTformer+ (common aliases are accepted).
        c2p_layers: Raw attention with shape ``[12,B,C,P]``.
        variant: One of B0--B5 or its descriptive name.

    MCTformer sums selected layers.  MCTformer+ averages them.  No spatial
    conditionalization or other rescaling is performed.
    """

    kind = _host_kind(host)
    layers = _floating_tensor(c2p_layers, "c2p_layers", 4)
    if layers.shape[0] != NUM_LAYERS:
        raise ValueError(
            f"c2p_layers must contain exactly {NUM_LAYERS} layers, "
            f"got {layers.shape[0]}"
        )
    _nonnegative(layers, "c2p_layers")
    spec = resolve_cam_variant(variant)
    indices = torch.as_tensor(
        [layer - 1 for layer in spec.layers_one_based],
        dtype=torch.long,
        device=layers.device,
    )
    selected = layers.index_select(0, indices)
    if kind == "mctformer":
        return selected.sum(dim=0)
    return selected.mean(dim=0)


def construct_cam_readout(
    host: str,
    patch_cam: torch.Tensor,
    c2p_layers: torch.Tensor,
    patch_to_patch_sum: torch.Tensor,
    variant: str | CamVariantSpec,
) -> CamReadoutStages:
    """Construct one exact native-formula CAM with a fixed supplied A_p2p.

    ``patch_cam`` may be ``[B,C,P]`` or ``[B,C,H,W]``.  Returned pre- and
    post-propagation CAMs preserve that shape.  ``patch_to_patch_sum`` must be
    ``[B,P,P]`` and is interpreted in native query-by-key orientation, i.e.
    ``output[query] = A_p2p[query,key] @ CAM[key]``.
    """

    kind = _host_kind(host)
    spec = resolve_cam_variant(variant)
    if not isinstance(patch_cam, torch.Tensor):
        raise TypeError("patch_cam must be a torch.Tensor")
    if patch_cam.ndim not in (3, 4):
        raise ValueError(
            "patch_cam must have shape [B,C,P] or [B,C,H,W], "
            f"got {tuple(patch_cam.shape)}"
        )
    patch = _floating_tensor(patch_cam, "patch_cam", patch_cam.ndim)
    layers = _floating_tensor(c2p_layers, "c2p_layers", 4)
    p2p = _floating_tensor(patch_to_patch_sum, "patch_to_patch_sum", 3)
    _nonnegative(patch, "patch_cam")
    _nonnegative(p2p, "patch_to_patch_sum")
    if patch.dtype != layers.dtype or patch.device != layers.device:
        raise ValueError("patch_cam and c2p_layers must share dtype and device")
    if p2p.dtype != patch.dtype or p2p.device != patch.device:
        raise ValueError(
            "patch_to_patch_sum, patch_cam, and c2p_layers must share dtype/device"
        )

    raw_c2p = aggregate_raw_c2p(kind, layers, spec)
    batch, classes = (int(value) for value in patch.shape[:2])
    flat_patch = patch.flatten(start_dim=2)
    patches = int(flat_patch.shape[-1])
    if tuple(raw_c2p.shape) != (batch, classes, patches):
        raise ValueError(
            "aggregated c2p and patch CAM shapes disagree: "
            f"{tuple(raw_c2p.shape)} versus {(batch, classes, patches)}"
        )
    if tuple(p2p.shape) != (batch, patches, patches):
        raise ValueError(
            "patch_to_patch_sum must have shape "
            f"{(batch, patches, patches)}, got {tuple(p2p.shape)}"
        )

    fused = raw_c2p * flat_patch
    if kind == "mctformer_plus":
        fused = torch.sqrt(fused)
    propagated = torch.matmul(p2p.unsqueeze(1), fused.unsqueeze(-1)).squeeze(-1)
    preprop_cam = fused.reshape_as(patch)
    final_cam = propagated.reshape_as(patch)
    return CamReadoutStages(
        variant_code=spec.code,
        variant_name=spec.name,
        layers_one_based=spec.layers_one_based,
        raw_c2p=raw_c2p,
        preprop_cam=preprop_cam,
        final_cam=final_cam,
    )


def construct_all_cam_readouts(
    host: str,
    patch_cam: torch.Tensor,
    c2p_layers: torch.Tensor,
    patch_to_patch_sum: torch.Tensor,
) -> dict[str, CamReadoutStages]:
    """Construct B0--B5 with one unchanged patch CAM and A_p2p tensor."""

    return {
        spec.code: construct_cam_readout(
            host, patch_cam, c2p_layers, patch_to_patch_sum, spec
        )
        for spec in CAM_VARIANT_SPECS
    }


def cam_threshold_grid() -> np.ndarray:
    """Return the pre-registered 0.20--0.60 inclusive grid (41 points)."""

    return np.arange(20, 61, dtype=np.float64) / 100.0


def _active_class_ids(
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
    expected: int,
) -> np.ndarray:
    if isinstance(active_class_ids, torch.Tensor):
        classes = active_class_ids.detach().cpu().numpy()
    else:
        classes = np.asarray(active_class_ids)
    if classes.ndim != 1 or len(classes) != expected:
        raise ValueError(
            "active_class_ids must be rank one and match the CAM channels; "
            f"expected {expected}, got {classes.shape}"
        )
    if not np.issubdtype(classes.dtype, np.integer):
        raise TypeError("active_class_ids must have an integer dtype")
    classes = classes.astype(np.int64, copy=False)
    if np.any((classes < 0) | (classes >= VOC_FOREGROUND_CLASSES)):
        raise ValueError("active_class_ids must lie in the zero-based VOC range 0..19")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("active_class_ids must be unique")
    return classes


def raw_cam_prediction_at_threshold(
    final_cam: np.ndarray | torch.Tensor,
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Apply the exact Experiment 2 CAM evaluation at one BG threshold.

    The background channel is prepended, so an exact foreground/background
    tie resolves to background, just as in the official NumPy ``argmax`` path.
    """

    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and lie in [0,1]")
    normalized = upsample_and_normalize_active_cams(final_cam)
    if isinstance(normalized, torch.Tensor):
        values = normalized.detach().cpu().numpy()
    else:
        values = np.asarray(normalized)
    classes = _active_class_ids(active_class_ids, int(values.shape[0]))
    background = np.full(
        (1, *TRANSFORMED_CROP_SIZE), threshold_value, dtype=values.dtype
    )
    scores = np.concatenate((background, values), axis=0)
    semantic_ids = np.concatenate((np.zeros(1, dtype=np.int64), classes + 1), axis=0)
    return semantic_ids[np.argmax(scores, axis=0)]


def raw_cam_confusion_at_threshold(
    final_cam: np.ndarray | torch.Tensor,
    active_class_ids: Sequence[int] | np.ndarray | torch.Tensor,
    transformed_semantic_mask: np.ndarray | torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Return one void-excluding 21x21 confusion at an explicit threshold."""

    if isinstance(transformed_semantic_mask, torch.Tensor):
        target_shape = tuple(transformed_semantic_mask.shape)
    else:
        target_shape = tuple(np.asarray(transformed_semantic_mask).shape)
    if target_shape != TRANSFORMED_CROP_SIZE:
        raise ValueError(
            "transformed_semantic_mask must have shape "
            f"{TRANSFORMED_CROP_SIZE}, got {target_shape}"
        )
    prediction = raw_cam_prediction_at_threshold(final_cam, active_class_ids, threshold)
    return voc_confusion_matrix(prediction, transformed_semantic_mask)


def _validated_confusion(confusion: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(confusion, torch.Tensor):
        value = confusion.detach().cpu().numpy()
    else:
        value = np.asarray(confusion)
    if value.shape != (VOC_SEGMENTATION_CLASSES, VOC_SEGMENTATION_CLASSES):
        raise ValueError("confusion must have shape [21,21]")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError("confusion must have an integer dtype")
    if np.any(value < 0):
        raise ValueError("confusion counts must be non-negative")
    if int(value.sum()) <= 0:
        raise ValueError("confusion must contain at least one evaluated pixel")
    return value.astype(np.int64, copy=False)


def _ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return float(numerator / denominator) if int(denominator) > 0 else float("nan")


def cam_metrics_from_confusion(
    confusion: np.ndarray | torch.Tensor,
) -> dict[str, float | np.ndarray]:
    """Compute mIoU and two explicitly different foreground P/R definitions.

    ``binary_foreground_*`` treats any foreground ID as foreground, regardless
    of which class was predicted.  ``semantic_correct_foreground_*`` requires
    the foreground class ID to be correct; an inter-foreground confusion is
    therefore an error.  Rows are GT and columns are predictions.
    """

    matrix = _validated_confusion(confusion)
    per_class_iou, mean_iou = iou_from_confusion(matrix)
    diagonal = np.diag(matrix).astype(np.int64, copy=False)
    predicted = matrix.sum(axis=0, dtype=np.int64)
    target = matrix.sum(axis=1, dtype=np.int64)
    per_class_precision = np.divide(
        diagonal,
        predicted,
        out=np.full(VOC_SEGMENTATION_CLASSES, np.nan, dtype=np.float64),
        where=predicted > 0,
    )
    per_class_recall = np.divide(
        diagonal,
        target,
        out=np.full(VOC_SEGMENTATION_CLASSES, np.nan, dtype=np.float64),
        where=target > 0,
    )

    binary_true_positive = int(matrix[1:, 1:].sum(dtype=np.int64))
    binary_false_positive = int(matrix[0, 1:].sum(dtype=np.int64))
    binary_false_negative = int(matrix[1:, 0].sum(dtype=np.int64))
    semantic_true_positive = int(diagonal[1:].sum(dtype=np.int64))
    predicted_foreground = int(matrix[:, 1:].sum(dtype=np.int64))
    target_foreground = int(matrix[1:, :].sum(dtype=np.int64))

    return {
        "per_class_iou": per_class_iou,
        "mean_iou": float(mean_iou),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "binary_foreground_precision": _ratio(
            binary_true_positive, binary_true_positive + binary_false_positive
        ),
        "binary_foreground_recall": _ratio(
            binary_true_positive, binary_true_positive + binary_false_negative
        ),
        "semantic_correct_foreground_precision": _ratio(
            semantic_true_positive, predicted_foreground
        ),
        "semantic_correct_foreground_recall": _ratio(
            semantic_true_positive, target_foreground
        ),
    }


def native_best_threshold_anchor(
    thresholds: Sequence[float] | np.ndarray,
    native_metric_values: Sequence[float] | np.ndarray,
) -> NativeBestThreshold:
    """Select native B0's best threshold with a deterministic low-tie rule.

    Thresholds must be strictly increasing.  When B0 has an exact plateau at
    its maximum, the smallest threshold is selected.  Variants must then be
    sampled at this returned index rather than selecting their own threshold.
    """

    grid = np.asarray(thresholds, dtype=np.float64)
    values = np.asarray(native_metric_values, dtype=np.float64)
    if grid.ndim != 1 or values.ndim != 1 or len(grid) != len(values) or not len(grid):
        raise ValueError(
            "thresholds and native_metric_values must be equal non-empty vectors"
        )
    if not np.isfinite(grid).all() or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("thresholds must be finite and lie in [0,1]")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("thresholds must be strictly increasing")
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError("native_metric_values must contain a finite value and no Inf")
    maximum = float(np.nanmax(values))
    index = int(np.flatnonzero(values == maximum)[0])
    return NativeBestThreshold(
        index=index,
        threshold=float(grid[index]),
        native_metric=float(values[index]),
    )


def values_at_native_anchor(
    curves: Mapping[str, Sequence[float] | np.ndarray],
    anchor: NativeBestThreshold,
) -> dict[str, float]:
    """Sample every variant curve at a precomputed native-B0 anchor."""

    if not isinstance(anchor, NativeBestThreshold):
        raise TypeError("anchor must be a NativeBestThreshold")
    if not curves:
        raise ValueError("curves must be non-empty")
    sampled: dict[str, float] = {}
    expected_length: int | None = None
    for name, curve in curves.items():
        if not isinstance(name, str) or not name:
            raise TypeError("curve names must be non-empty strings")
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"curve {name!r} must be rank one")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("all curves must have equal length")
        if anchor.index < 0 or anchor.index >= len(values):
            raise ValueError("anchor index lies outside a supplied curve")
        sampled[name] = float(values[anchor.index])
    return sampled
