"""Pure utilities for Experiment 3 Validation A presence-axis analysis.

The fixed direction used by the native class-token readout is the normalized
all-ones vector.  This module keeps that exact algebra separate from the
cross-fitted, data-derived presence direction used as an intermediate-layer
control.  It contains no model or filesystem mutation code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


DEFAULT_EPSILON = 1e-12
NUM_CROSS_FIT_FOLDS = 2


class ZeroNormError(ValueError):
    """Raised when a direction or token required by a cosine has zero norm."""


def _torch_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have a non-empty final embedding axis")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def normalized_all_ones_direction(
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``1 / sqrt(width)`` on the requested device and floating dtype."""

    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError("direction dtype must be floating")
    return torch.full(
        (width,),
        1.0 / math.sqrt(width),
        dtype=dtype,
        device=device,
    )


def normalize_torch_direction(
    direction: torch.Tensor, *, epsilon: float = DEFAULT_EPSILON
) -> torch.Tensor:
    """Normalize a finite one-dimensional torch direction."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    value = _torch_float_tensor(direction, "direction")
    if value.ndim != 1:
        raise ValueError("direction must have shape [D]")
    norm = value.norm(p=2)
    if float(norm) <= epsilon:
        raise ZeroNormError("direction has zero or near-zero norm")
    return value / norm


def _unit_torch_direction(
    direction: torch.Tensor,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    epsilon: float,
) -> torch.Tensor:
    if not isinstance(direction, torch.Tensor):
        raise TypeError("direction must be a torch.Tensor")
    value = normalize_torch_direction(
        direction.to(device=device, dtype=dtype), epsilon=epsilon
    )
    if value.shape != (width,):
        raise ValueError(f"direction shape {tuple(value.shape)} != ({width},)")
    return value


@dataclass(frozen=True)
class DirectionDecomposition:
    """Orthogonal decomposition along one normalized direction."""

    coefficients: torch.Tensor
    parallel: torch.Tensor
    residual: torch.Tensor


def decompose_along_direction(
    tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> DirectionDecomposition:
    """Decompose ``tokens`` as ``coefficient * direction + residual``.

    ``tokens`` may have any leading dimensions and must end in embedding width
    ``D``.  A supplied non-unit direction is normalized explicitly, avoiding a
    silent projection-scale error.
    """

    value = _torch_float_tensor(tokens, "tokens")
    unit = _unit_torch_direction(
        direction,
        value.shape[-1],
        device=value.device,
        dtype=value.dtype,
        epsilon=epsilon,
    )
    coefficients = torch.sum(value * unit, dim=-1)
    parallel = coefficients.unsqueeze(-1) * unit
    return DirectionDecomposition(
        coefficients=coefficients,
        parallel=parallel,
        residual=value - parallel,
    )


def project_onto_direction(
    tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Return the vector projection of tokens onto ``direction``."""

    return decompose_along_direction(tokens, direction, epsilon=epsilon).parallel


def remove_direction(
    tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Return the component orthogonal to ``direction``."""

    return decompose_along_direction(tokens, direction, epsilon=epsilon).residual


def _strict_norm(values: torch.Tensor, name: str, epsilon: float) -> torch.Tensor:
    norms = values.norm(p=2, dim=-1)
    if bool((norms <= epsilon).any()):
        count = int((norms <= epsilon).sum().item())
        raise ZeroNormError(f"{name} contains {count} zero or near-zero vectors")
    return norms


@dataclass(frozen=True)
class AxisCosineMaps:
    """Raw and axis-removed class-to-patch cosine maps.

    The three removed variants use the direct residual-vector dot product as
    their numerator; only their norm denominators differ. ``residual_dot``
    retains the algebraically equivalent ``raw_dot - axis_dot`` value as an
    independent numerical identity guard. The direct form avoids cancellation
    when the shared-axis contribution is large.
    """

    raw: torch.Tensor
    class_only_removed: torch.Tensor
    patch_only_removed: torch.Tensor
    both_removed: torch.Tensor
    raw_dot: torch.Tensor
    residual_dot: torch.Tensor
    direct_residual_dot: torch.Tensor
    class_coefficients: torch.Tensor
    patch_coefficients: torch.Tensor
    class_norms: torch.Tensor
    patch_norms: torch.Tensor
    residual_class_norms: torch.Tensor
    residual_patch_norms: torch.Tensor


def axis_removed_cosine_maps(
    class_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> AxisCosineMaps:
    """Compute V0--V3 cosine maps for batched class and patch tokens.

    Inputs have shapes ``[B,C,D]`` and ``[B,P,D]``.  Computation is promoted
    to float32 when inputs are lower precision, matching the Experiment 1
    score contract.  Zero raw or residual vectors are rejected rather than
    silently converted into a cosine of zero.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    classes = _torch_float_tensor(class_tokens, "class_tokens")
    patches = _torch_float_tensor(patch_tokens, "patch_tokens")
    if classes.ndim != 3 or patches.ndim != 3:
        raise ValueError("class_tokens and patch_tokens must have rank 3")
    if classes.shape[0] != patches.shape[0]:
        raise ValueError("class and patch batch sizes differ")
    if classes.shape[-1] != patches.shape[-1]:
        raise ValueError("class and patch embedding widths differ")
    if classes.shape[1] < 1 or patches.shape[1] < 1:
        raise ValueError("class and patch token axes must be non-empty")
    if classes.device != patches.device:
        raise ValueError("class and patch tensors must be on the same device")

    work_dtype = torch.float64 if classes.dtype == torch.float64 else torch.float32
    classes = classes.to(dtype=work_dtype)
    patches = patches.to(dtype=work_dtype)
    unit = _unit_torch_direction(
        direction,
        classes.shape[-1],
        device=classes.device,
        dtype=work_dtype,
        epsilon=epsilon,
    )
    class_parts = decompose_along_direction(classes, unit, epsilon=epsilon)
    patch_parts = decompose_along_direction(patches, unit, epsilon=epsilon)

    class_norms = _strict_norm(classes, "class_tokens", epsilon)
    patch_norms = _strict_norm(patches, "patch_tokens", epsilon)
    residual_class_norms = _strict_norm(
        class_parts.residual, "axis-removed class_tokens", epsilon
    )
    residual_patch_norms = _strict_norm(
        patch_parts.residual, "axis-removed patch_tokens", epsilon
    )

    raw_dot = torch.einsum("bcd,bpd->bcp", classes, patches)
    axis_dot = torch.einsum(
        "bc,bp->bcp", class_parts.coefficients, patch_parts.coefficients
    )
    residual_dot = raw_dot - axis_dot
    direct_residual_dot = torch.einsum(
        "bcd,bpd->bcp", class_parts.residual, patch_parts.residual
    )

    raw_denominator = torch.einsum("bc,bp->bcp", class_norms, patch_norms)
    class_removed_denominator = torch.einsum(
        "bc,bp->bcp", residual_class_norms, patch_norms
    )
    patch_removed_denominator = torch.einsum(
        "bc,bp->bcp", class_norms, residual_patch_norms
    )
    both_removed_denominator = torch.einsum(
        "bc,bp->bcp", residual_class_norms, residual_patch_norms
    )
    result = AxisCosineMaps(
        raw=raw_dot / raw_denominator,
        class_only_removed=direct_residual_dot / class_removed_denominator,
        patch_only_removed=direct_residual_dot / patch_removed_denominator,
        both_removed=direct_residual_dot / both_removed_denominator,
        raw_dot=raw_dot,
        residual_dot=residual_dot,
        direct_residual_dot=direct_residual_dot,
        class_coefficients=class_parts.coefficients,
        patch_coefficients=patch_parts.coefficients,
        class_norms=class_norms,
        patch_norms=patch_norms,
        residual_class_norms=residual_class_norms,
        residual_patch_norms=residual_patch_norms,
    )
    for name in (
        "raw",
        "class_only_removed",
        "patch_only_removed",
        "both_removed",
    ):
        value = getattr(result, name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} cosine map contains NaN or Inf")
    return result


def token_axis_energy(
    tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Return squared directional energy divided by squared token norm."""

    value = _torch_float_tensor(tokens, "tokens")
    parts = decompose_along_direction(value, direction, epsilon=epsilon)
    norms = _strict_norm(value, "tokens", epsilon)
    energy = parts.coefficients.square() / norms.square()
    if bool(((energy < -1e-6) | (energy > 1.0 + 1e-5)).any()):
        raise RuntimeError("axis energy escaped its [0,1] numerical range")
    return energy.clamp(0.0, 1.0)


@dataclass(frozen=True)
class TokenPairAxisMetrics:
    """Pairwise class-token decomposition for tensors shaped ``[...,C,D]``."""

    raw_cosine: torch.Tensor
    residual_cosine: torch.Tensor
    raw_dot: torch.Tensor
    axis_dot: torch.Tensor
    residual_dot: torch.Tensor
    coefficients: torch.Tensor
    axis_energy: torch.Tensor
    token_norms: torch.Tensor
    residual_norms: torch.Tensor


def token_pair_axis_metrics(
    class_tokens: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> TokenPairAxisMetrics:
    """Compute raw/residual pair cosine and additive dot-product components."""

    values = _torch_float_tensor(class_tokens, "class_tokens")
    if values.ndim < 2 or values.shape[-2] < 1:
        raise ValueError("class_tokens must have shape [...,C,D]")
    work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    values = values.to(dtype=work_dtype)
    parts = decompose_along_direction(values, direction, epsilon=epsilon)
    token_norms = _strict_norm(values, "class_tokens", epsilon)
    residual_norms = _strict_norm(parts.residual, "axis-removed class_tokens", epsilon)
    raw_dot = torch.matmul(values, values.transpose(-2, -1))
    axis_dot = parts.coefficients.unsqueeze(-1) * parts.coefficients.unsqueeze(-2)
    residual_dot = torch.matmul(parts.residual, parts.residual.transpose(-2, -1))
    raw_denominator = token_norms.unsqueeze(-1) * token_norms.unsqueeze(-2)
    residual_denominator = residual_norms.unsqueeze(-1) * residual_norms.unsqueeze(-2)
    return TokenPairAxisMetrics(
        raw_cosine=raw_dot / raw_denominator,
        residual_cosine=residual_dot / residual_denominator,
        raw_dot=raw_dot,
        axis_dot=axis_dot,
        residual_dot=residual_dot,
        coefficients=parts.coefficients,
        axis_energy=parts.coefficients.square() / token_norms.square(),
        token_norms=token_norms,
        residual_norms=residual_norms,
    )


def sha256_two_fold(image_id: str) -> int:
    """Assign an exact image ID to a deterministic SHA-256 parity fold."""

    if not isinstance(image_id, str):
        raise TypeError("image_id must be a string")
    if not image_id or image_id != image_id.strip():
        raise ValueError("image_id must be non-empty and contain no edge whitespace")
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return int(digest[-1] & 1)


def validate_two_fold_split(
    assignments: Mapping[str, int],
    *,
    expected_image_ids: Iterable[str] | None = None,
    require_both_folds: bool = True,
) -> dict[str, int]:
    """Validate deterministic, exhaustive, duplicate-free two-fold assignments."""

    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("assignments must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for image_id, fold in assignments.items():
        if not isinstance(image_id, str):
            raise TypeError("every image ID must be a string")
        expected_fold = sha256_two_fold(image_id)
        if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)):
            raise TypeError(f"fold for {image_id} must be integer 0 or 1")
        fold = int(fold)
        if fold not in (0, 1):
            raise ValueError(f"fold for {image_id} is outside {{0,1}}")
        if fold != expected_fold:
            raise ValueError(
                f"fold for {image_id} is not its deterministic SHA-256 assignment"
            )
        normalized[image_id] = fold
    if expected_image_ids is not None:
        expected = list(expected_image_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected_image_ids contains duplicates")
        if set(expected) != set(normalized):
            missing = sorted(set(expected).difference(normalized))
            extra = sorted(set(normalized).difference(expected))
            raise ValueError(f"split ID mismatch: missing={missing}, extra={extra}")
    counts = {
        fold: sum(value == fold for value in normalized.values()) for fold in (0, 1)
    }
    if require_both_folds and min(counts.values()) == 0:
        raise ValueError("both deterministic folds must be non-empty")
    return normalized


def build_two_fold_split(image_ids: Sequence[str]) -> dict[str, int]:
    """Build and validate the deterministic two-fold image split."""

    ids = list(image_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("image_ids contains duplicates")
    assignments = {image_id: sha256_two_fold(image_id) for image_id in ids}
    return validate_two_fold_split(assignments, expected_image_ids=ids)


def _numpy_float_array(
    value: np.ndarray | Sequence[float], name: str, *, ndim: int | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def normalize_numpy_direction(
    value: np.ndarray | Sequence[float],
    *,
    axis: int = -1,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Normalize finite numpy vectors and reject any zero-norm slice."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    array = _numpy_float_array(value, "direction")
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= epsilon):
        raise ZeroNormError("direction contains a zero or near-zero vector")
    return array / norms


@dataclass(frozen=True)
class CrossFittedDirectionRegistry:
    """Two-fold presence directions indexed explicitly by *fit* fold."""

    fit_means: np.ndarray  # [fit_fold,L,C,D]
    class_deltas: np.ndarray  # [fit_fold,L,C,D]
    shared_directions: np.ndarray  # [fit_fold,L,D]
    loo_shared_directions: np.ndarray  # [fit_fold,L,C,D]
    class_alignment: np.ndarray  # [fit_fold,L,C]
    loo_class_alignment: np.ndarray  # [fit_fold,L,C]
    total_counts: np.ndarray  # [fit_fold,L,C]
    positive_counts: np.ndarray  # [fit_fold,L,C]
    negative_counts: np.ndarray  # [fit_fold,L,C]
    image_ids_by_fold: tuple[tuple[str, ...], tuple[str, ...]]

    def __post_init__(self) -> None:
        means = np.asarray(self.fit_means)
        if means.ndim != 4 or means.shape[0] != NUM_CROSS_FIT_FOLDS:
            raise ValueError("fit_means must have shape [2,L,C,D]")
        folds, layers, classes, width = means.shape
        expected_shapes = {
            "class_deltas": (folds, layers, classes, width),
            "shared_directions": (folds, layers, width),
            "loo_shared_directions": (folds, layers, classes, width),
            "class_alignment": (folds, layers, classes),
            "loo_class_alignment": (folds, layers, classes),
            "total_counts": (folds, layers, classes),
            "positive_counts": (folds, layers, classes),
            "negative_counts": (folds, layers, classes),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if len(self.image_ids_by_fold) != NUM_CROSS_FIT_FOLDS:
            raise ValueError("image_ids_by_fold must contain exactly two folds")
        left, right = (set(values) for values in self.image_ids_by_fold)
        if not left or not right or left.intersection(right):
            raise ValueError("fit image folds must be non-empty and disjoint")
        for fold, image_ids in enumerate(self.image_ids_by_fold):
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(f"image fold {fold} contains duplicate IDs")
            if any(sha256_two_fold(image_id) != fold for image_id in image_ids):
                raise ValueError(f"image fold {fold} violates SHA-256 assignment")
            if not np.all(np.asarray(self.total_counts[fold]) == len(image_ids)):
                raise ValueError(f"total_counts for fold {fold} do not match image IDs")
        if not np.array_equal(
            np.asarray(self.positive_counts) + np.asarray(self.negative_counts),
            np.asarray(self.total_counts),
        ):
            raise ValueError("positive and negative counts do not sum to total counts")
        shared_norms = np.linalg.norm(self.shared_directions, axis=-1)
        loo_norms = np.linalg.norm(self.loo_shared_directions, axis=-1)
        if not np.allclose(shared_norms, 1.0, rtol=0, atol=1e-10):
            raise ValueError("shared directions are not unit normalized")
        if not np.allclose(loo_norms, 1.0, rtol=0, atol=1e-10):
            raise ValueError("LOO shared directions are not unit normalized")

    @property
    def num_layers(self) -> int:
        return int(self.fit_means.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.fit_means.shape[2])

    @property
    def width(self) -> int:
        return int(self.fit_means.shape[3])

    @staticmethod
    def fit_fold_for_eval(eval_fold: int) -> int:
        if isinstance(eval_fold, bool) or int(eval_fold) not in (0, 1):
            raise ValueError("eval_fold must be 0 or 1")
        return 1 - int(eval_fold)

    def heldout_parameters(self, eval_fold: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Return fit means/directions learned only on the opposite fold."""

        fit_fold = self.fit_fold_for_eval(eval_fold)
        return (
            self.fit_means[fit_fold],
            self.shared_directions[fit_fold],
            fit_fold,
        )


class TwoFoldPresenceAccumulator:
    """Streaming sufficient statistics for two-fold presence directions.

    Each update consumes one image's all-class post-block tokens ``[L,C,D]``
    and binary image-level labels ``[C]``.  No patch vectors are retained.
    """

    def __init__(self, num_layers: int, num_classes: int, width: int):
        for name, value in (
            ("num_layers", num_layers),
            ("num_classes", num_classes),
            ("width", width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes < 2:
            raise ValueError("at least two classes are required for LOO alignment")
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.width = width
        vector_shape = (NUM_CROSS_FIT_FOLDS, num_layers, num_classes, width)
        count_shape = (NUM_CROSS_FIT_FOLDS, num_layers, num_classes)
        self._total_sum = np.zeros(vector_shape, dtype=np.float64)
        self._positive_sum = np.zeros(vector_shape, dtype=np.float64)
        self._negative_sum = np.zeros(vector_shape, dtype=np.float64)
        self._total_count = np.zeros(count_shape, dtype=np.int64)
        self._positive_count = np.zeros(count_shape, dtype=np.int64)
        self._negative_count = np.zeros(count_shape, dtype=np.int64)
        self._image_ids: list[list[str]] = [[], []]
        self._seen: set[str] = set()

    def update(
        self,
        image_id: str,
        class_tokens: np.ndarray | torch.Tensor,
        presence_labels: np.ndarray | torch.Tensor | Sequence[int],
    ) -> int:
        """Add one image and return its deterministic fold."""

        if image_id in self._seen:
            raise ValueError(f"duplicate image_id update: {image_id}")
        if isinstance(class_tokens, torch.Tensor):
            tokens = class_tokens.detach().to(dtype=torch.float64).cpu().numpy()
        else:
            tokens = np.asarray(class_tokens)
        tokens = _numpy_float_array(tokens, "class_tokens", ndim=3)
        expected = (self.num_layers, self.num_classes, self.width)
        if tokens.shape != expected:
            raise ValueError(f"class_tokens shape {tokens.shape} != {expected}")
        if isinstance(presence_labels, torch.Tensor):
            labels = presence_labels.detach().cpu().numpy()
        else:
            labels = np.asarray(presence_labels)
        if labels.shape != (self.num_classes,):
            raise ValueError(
                f"presence_labels shape {labels.shape} != ({self.num_classes},)"
            )
        if not np.all(np.isin(labels, (0, 1, False, True))):
            raise ValueError("presence_labels must be binary")
        positive = labels.astype(bool, copy=False)
        negative = ~positive
        fold = sha256_two_fold(image_id)

        self._total_sum[fold] += tokens
        self._total_count[fold] += 1
        # Index the fold first so NumPy boolean indexing retains the declared
        # [layer, class, width] axis order instead of moving the advanced class
        # index to the front.
        self._positive_sum[fold][:, positive, :] += tokens[:, positive, :]
        self._positive_count[fold][:, positive] += 1
        self._negative_sum[fold][:, negative, :] += tokens[:, negative, :]
        self._negative_count[fold][:, negative] += 1
        self._image_ids[fold].append(image_id)
        self._seen.add(image_id)
        return fold

    def finalize(
        self, *, epsilon: float = DEFAULT_EPSILON
    ) -> CrossFittedDirectionRegistry:
        """Build exact fit-fold means, deltas, shared and LOO directions."""

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not self._seen:
            raise ValueError("cannot finalize an empty accumulator")
        if any(not values for values in self._image_ids):
            raise ValueError("both deterministic folds must contain images")
        for name, counts in (
            ("total", self._total_count),
            ("positive", self._positive_count),
            ("negative", self._negative_count),
        ):
            if np.any(counts <= 0):
                indices = np.argwhere(counts <= 0)
                raise ValueError(
                    f"{name} sufficient statistic is empty at {indices.tolist()}"
                )

        fit_means = self._total_sum / self._total_count[..., None]
        positive_means = self._positive_sum / self._positive_count[..., None]
        negative_means = self._negative_sum / self._negative_count[..., None]
        # Subtracting the same fit-only class identity mean from both groups
        # cancels algebraically.  The explicit means remain in the registry for
        # centered held-out projections.
        centered_positive = positive_means - fit_means
        centered_negative = negative_means - fit_means
        class_deltas = centered_positive - centered_negative

        shared_raw = class_deltas.mean(axis=2)
        shared = normalize_numpy_direction(shared_raw, axis=-1, epsilon=epsilon)
        delta_unit = normalize_numpy_direction(class_deltas, axis=-1, epsilon=epsilon)
        class_alignment = np.einsum("flcd,fld->flc", delta_unit, shared)

        loo_raw = (class_deltas.sum(axis=2, keepdims=True) - class_deltas) / float(
            self.num_classes - 1
        )
        loo = normalize_numpy_direction(loo_raw, axis=-1, epsilon=epsilon)
        loo_alignment = np.einsum("flcd,flcd->flc", delta_unit, loo)

        return CrossFittedDirectionRegistry(
            fit_means=fit_means.copy(),
            class_deltas=class_deltas.copy(),
            shared_directions=shared.copy(),
            loo_shared_directions=loo.copy(),
            class_alignment=class_alignment.copy(),
            loo_class_alignment=loo_alignment.copy(),
            total_counts=self._total_count.copy(),
            positive_counts=self._positive_count.copy(),
            negative_counts=self._negative_count.copy(),
            image_ids_by_fold=(
                tuple(sorted(self._image_ids[0])),
                tuple(sorted(self._image_ids[1])),
            ),
        )


def heldout_centered_projections(
    class_tokens: np.ndarray | torch.Tensor,
    *,
    eval_fold: int,
    registry: CrossFittedDirectionRegistry,
) -> np.ndarray:
    """Project one held-out image after fit-only per-class identity centering."""

    if isinstance(class_tokens, torch.Tensor):
        tokens = class_tokens.detach().to(dtype=torch.float64).cpu().numpy()
    else:
        tokens = np.asarray(class_tokens)
    tokens = _numpy_float_array(tokens, "class_tokens", ndim=3)
    expected = (registry.num_layers, registry.num_classes, registry.width)
    if tokens.shape != expected:
        raise ValueError(f"class_tokens shape {tokens.shape} != {expected}")
    fit_means, directions, _ = registry.heldout_parameters(eval_fold)
    centered = tokens - fit_means
    result = np.einsum("lcd,ld->lc", centered, directions)
    if not np.isfinite(result).all():
        raise RuntimeError("held-out projection contains NaN or Inf")
    return result


@dataclass(frozen=True)
class PresenceAUROC:
    """Pooled and equal-class presence discrimination."""

    micro: float
    macro_class: float
    classwise: np.ndarray
    positive_counts: np.ndarray
    negative_counts: np.ndarray


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels.astype(np.int8), scores))


def presence_projection_auroc(
    scores: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[Sequence[int]],
) -> PresenceAUROC:
    """Compute pooled and per-class AUROC from OOF image-by-class scores.

    This helper computes point estimates only.  A caller performing bootstrap
    inference must resample whole image rows and recompute this function for
    every repeat; flattening image-class pairs before resampling is invalid.
    """

    values = _numpy_float_array(scores, "scores", ndim=2)
    target = np.asarray(labels)
    if target.shape != values.shape:
        raise ValueError(f"labels shape {target.shape} != scores shape {values.shape}")
    if not np.all(np.isin(target, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    target = target.astype(bool, copy=False)
    classwise = np.asarray(
        [_binary_auroc(target[:, c], values[:, c]) for c in range(values.shape[1])],
        dtype=np.float64,
    )
    finite = np.isfinite(classwise)
    macro = float(classwise[finite].mean()) if finite.any() else float("nan")
    return PresenceAUROC(
        micro=_binary_auroc(target.reshape(-1), values.reshape(-1)),
        macro_class=macro,
        classwise=classwise,
        positive_counts=target.sum(axis=0).astype(np.int64),
        negative_counts=(~target).sum(axis=0).astype(np.int64),
    )


__all__ = [
    "AxisCosineMaps",
    "CrossFittedDirectionRegistry",
    "DEFAULT_EPSILON",
    "DirectionDecomposition",
    "NUM_CROSS_FIT_FOLDS",
    "PresenceAUROC",
    "TokenPairAxisMetrics",
    "TwoFoldPresenceAccumulator",
    "ZeroNormError",
    "axis_removed_cosine_maps",
    "build_two_fold_split",
    "decompose_along_direction",
    "heldout_centered_projections",
    "normalize_numpy_direction",
    "normalize_torch_direction",
    "normalized_all_ones_direction",
    "presence_projection_auroc",
    "project_onto_direction",
    "remove_direction",
    "sha256_two_fold",
    "token_axis_energy",
    "token_pair_axis_metrics",
    "validate_two_fold_split",
]
