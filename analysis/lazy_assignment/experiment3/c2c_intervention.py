"""Inference-only late class-to-class attention intervention.

The intervention is deliberately implemented as a temporary forward hook on
each selected block's ``attn.normalizer``.  At that point the tensor is the
native, per-head, pre-dropout attention matrix used by both the value path and
the attention returned by :class:`models.vit.Attention`.

No module parameter or buffer is added, and the context manager removes every
hook on normal exit and on exceptions.
"""

from __future__ import annotations

import math
import threading
import weakref
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn as nn


__all__ = [
    "C2C_VARIANT_LAYERS_1BASED",
    "C2CIntervention",
    "C2CInterventionRecord",
    "self_reroute_c2c_attention",
    "variant_layer_indices",
]


C2C_VARIANT_LAYERS_1BASED: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "C0": (),
        "C1": (12,),
        "C2": (11,),
        "C3": (10,),
        "C4": (10, 11),
        "C5": (10, 11, 12),
    }
)

_ACTIVE_MODELS: weakref.WeakSet[nn.Module] = weakref.WeakSet()
_ACTIVE_MODELS_LOCK = threading.RLock()


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    return num_classes


def _validate_attention(attention: torch.Tensor, num_classes: int) -> None:
    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    if attention.ndim != 4:
        raise ValueError("attention must have shape [B,H,T,T]")
    if attention.shape[0] < 1 or attention.shape[1] < 1:
        raise ValueError("attention batch and head dimensions must be positive")
    if attention.shape[-2] != attention.shape[-1]:
        raise ValueError("C2C self-reroute requires square self-attention")
    if attention.shape[-1] <= num_classes:
        raise ValueError("attention must contain at least one patch token")
    if not torch.is_floating_point(attention):
        raise TypeError("attention must use a floating-point dtype")
    if not bool(torch.isfinite(attention).all()):
        raise ValueError("attention contains non-finite values")
    if bool(torch.any(attention < 0)):
        raise ValueError("attention probabilities must be non-negative")


def self_reroute_c2c_attention(
    attention: torch.Tensor,
    num_classes: int = 20,
) -> torch.Tensor:
    """Mass-preservingly reroute every off-diagonal C2C read to self.

    Args:
        attention: Native per-head attention with shape ``[B,H,T,T]`` and
            leading class tokens.
        num_classes: Number of leading class-query/class-key tokens.

    Returns:
        A new tensor with the same shape, dtype, and device.  For every class
        query and head, its complete class-key mass is placed on the matching
        diagonal.  Class-to-patch entries and all patch-query rows are copied
        without modification.
    """

    num_classes = _validate_num_classes(num_classes)
    _validate_attention(attention, num_classes)

    class_block = attention[..., :num_classes, :num_classes]
    class_mass = class_block.sum(dim=-1)
    rerouted = attention.clone()
    rerouted[..., :num_classes, :num_classes] = 0
    diagonal = torch.arange(num_classes, device=attention.device)
    rerouted[..., diagonal, diagonal] = class_mass
    return rerouted


def variant_layer_indices(variant: str, expected_depth: int = 12) -> tuple[int, ...]:
    """Return the plan's variant layers as zero-based block indices."""

    if not isinstance(variant, str):
        raise TypeError("variant must be a string")
    name = variant.strip().upper()
    if name not in C2C_VARIANT_LAYERS_1BASED:
        raise ValueError(
            f"unknown C2C variant {variant!r}; expected one of "
            f"{tuple(C2C_VARIANT_LAYERS_1BASED)}"
        )
    if isinstance(expected_depth, bool) or not isinstance(expected_depth, int):
        raise TypeError("expected_depth must be an integer")
    if expected_depth < 1:
        raise ValueError("expected_depth must be positive")
    one_based = C2C_VARIANT_LAYERS_1BASED[name]
    if one_based and max(one_based) > expected_depth:
        raise ValueError(
            f"variant {name} requires layer {max(one_based)}, "
            f"but expected_depth={expected_depth}"
        )
    return tuple(layer - 1 for layer in one_based)


@dataclass(frozen=True)
class C2CInterventionRecord:
    """Structural diagnostics for one activation of one selected layer."""

    layer_index: int
    layer_number: int
    call_index: int
    batch_size: int
    num_heads: int
    num_tokens: int
    dtype: str
    device: str
    patch_rows_max_abs_diff: float
    class_to_patch_max_abs_diff: float
    class_group_mass_max_abs_diff: float
    row_sum_max_abs_diff: float
    native_row_sum_max_abs_error: float
    rerouted_row_sum_max_abs_error: float
    diagonal_mass_max_abs_diff: float
    offdiag_pre_mass_mean: float
    offdiag_pre_mass_max: float
    offdiag_post_mass_mean: float
    offdiag_post_mass_max: float
    offdiag_post_weight_max_abs: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly representation."""

        return asdict(self)


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().item())


def _mean(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean().item())


def _maximum(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().max().item())


def _offdiag_mass(class_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = class_block.shape[-1]
    diagonal_mask = torch.eye(
        classes, dtype=torch.bool, device=class_block.device
    ).view(1, 1, classes, classes)
    offdiag_weights = class_block.masked_fill(diagonal_mask, 0)
    return offdiag_weights.sum(dim=-1), offdiag_weights


def _group_ids_match(
    group_ids: object,
    *,
    batch_size: int,
    num_classes: int,
    num_patches: int,
    device: torch.device,
    name: str,
) -> None:
    if not isinstance(group_ids, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if torch.is_floating_point(group_ids) or group_ids.dtype == torch.bool:
        raise TypeError(f"{name} must use an integer dtype")
    total = num_classes + num_patches
    expected = torch.cat(
        (
            torch.zeros(num_classes, dtype=torch.long, device=device),
            torch.ones(num_patches, dtype=torch.long, device=device),
        )
    )
    actual = group_ids.to(device=device, dtype=torch.long)
    if actual.ndim == 1:
        valid_shape = actual.shape == (total,)
        matches = valid_shape and torch.equal(actual, expected)
    elif actual.ndim == 2:
        valid_shape = actual.shape == (batch_size, total)
        matches = valid_shape and torch.equal(
            actual, expected.unsqueeze(0).expand(batch_size, -1)
        )
    else:
        valid_shape = False
        matches = False
    if not matches:
        raise RuntimeError(
            f"{name} must encode exact leading class / trailing patch groups; "
            f"shape={tuple(actual.shape)}, expected total={total}, "
            f"valid_shape={valid_shape}"
        )


class C2CIntervention:
    """Exception-safe context manager for analysis-only C2C self-reroute.

    ``layers`` uses zero-based block indices, matching the implementation
    interface in the Experiment 3 plan.  Use :meth:`from_variant` for the
    plan's C0--C5 names.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[int],
        *,
        mode: str = "self_reroute",
        num_classes: int = 20,
        expected_num_patches: int = 784,
        expected_depth: int | None = 12,
        invariant_tolerance: float = 1e-6,
        native_row_tolerance: float = 5e-6,
        variant_name: str = "custom",
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if mode != "self_reroute":
            raise ValueError("only mode='self_reroute' is supported")
        self.model = model
        self.mode = mode
        self.num_classes = _validate_num_classes(num_classes)
        if isinstance(expected_num_patches, bool) or not isinstance(
            expected_num_patches, int
        ):
            raise TypeError("expected_num_patches must be an integer")
        if expected_num_patches < 1:
            raise ValueError("expected_num_patches must be positive")
        self.expected_num_patches = expected_num_patches
        if expected_depth is not None and (
            isinstance(expected_depth, bool) or not isinstance(expected_depth, int)
        ):
            raise TypeError("expected_depth must be an integer or None")
        if expected_depth is not None and expected_depth < 1:
            raise ValueError("expected_depth must be positive")
        self.expected_depth = expected_depth
        if not math.isfinite(invariant_tolerance) or invariant_tolerance < 0:
            raise ValueError("invariant_tolerance must be finite and non-negative")
        if not math.isfinite(native_row_tolerance) or native_row_tolerance < 0:
            raise ValueError("native_row_tolerance must be finite and non-negative")
        self.invariant_tolerance = float(invariant_tolerance)
        self.native_row_tolerance = float(native_row_tolerance)
        self.variant_name = str(variant_name)

        normalized_layers: list[int] = []
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise TypeError("layer indices must be integers")
            normalized_layers.append(layer)
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("layer indices must be unique")
        self.layers = tuple(sorted(normalized_layers))

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._records: list[C2CInterventionRecord] = []
        self._activation_counts = {layer: 0 for layer in self.layers}
        self._entered = False
        self._used = False

    @classmethod
    def from_variant(
        cls,
        model: nn.Module,
        variant: str,
        **kwargs: object,
    ) -> "C2CIntervention":
        """Construct one of the plan-defined C0--C5 interventions."""

        expected_depth_value = kwargs.get("expected_depth", 12)
        if expected_depth_value is None:
            if not hasattr(model, "blocks"):
                raise TypeError("model must expose .blocks")
            variant_depth = len(model.blocks)  # type: ignore[arg-type]
        elif isinstance(expected_depth_value, bool) or not isinstance(
            expected_depth_value, int
        ):
            raise TypeError("expected_depth must be an integer or None")
        else:
            variant_depth = expected_depth_value
        name = variant.strip().upper()
        layers = variant_layer_indices(name, expected_depth=variant_depth)
        return cls(model, layers, variant_name=name, **kwargs)

    @property
    def records(self) -> tuple[C2CInterventionRecord, ...]:
        return tuple(self._records)

    @property
    def activation_counts(self) -> dict[int, int]:
        """Return zero-based layer activation counts."""

        return dict(self._activation_counts)

    @property
    def activation_counts_1based(self) -> dict[int, int]:
        return {layer + 1: count for layer, count in self._activation_counts.items()}

    @property
    def layer_numbers_1based(self) -> tuple[int, ...]:
        return tuple(layer + 1 for layer in self.layers)

    def _validate_model(self) -> None:
        if not hasattr(self.model, "blocks"):
            raise TypeError("model must expose a .blocks sequence")
        blocks = self.model.blocks
        try:
            depth = len(blocks)
        except TypeError as error:
            raise TypeError("model.blocks must be a sized sequence") from error
        if self.expected_depth is not None and depth != self.expected_depth:
            raise RuntimeError(
                f"model depth {depth} does not match expected_depth={self.expected_depth}"
            )
        if self.model.training:
            raise RuntimeError("C2C intervention is inference-only; call model.eval()")
        model_classes = getattr(self.model, "num_classes", self.num_classes)
        if int(model_classes) != self.num_classes:
            raise RuntimeError(
                f"model.num_classes={model_classes} != {self.num_classes}"
            )
        class_tokens = getattr(self.model, "num_class_tokens", self.num_classes)
        if int(class_tokens) != self.num_classes:
            raise RuntimeError("extra class/background tokens are not supported")

        for layer in self.layers:
            if layer < 0 or layer >= depth:
                raise IndexError(f"layer index {layer} is outside model depth {depth}")
            block = blocks[layer]
            attention = getattr(block, "attn", None)
            normalizer = getattr(attention, "normalizer", None)
            if not isinstance(normalizer, nn.Module):
                raise TypeError(f"block {layer} lacks attn.normalizer")
            if getattr(normalizer, "mode", None) != "vanilla":
                raise RuntimeError(
                    f"block {layer} attention normalization must be vanilla"
                )
            if int(getattr(attention, "num_classes", -1)) != self.num_classes:
                raise RuntimeError(
                    f"block {layer} attention class-token count is not "
                    f"{self.num_classes}"
                )

    def _make_hook(self, layer: int):
        block = self.model.blocks[layer]

        def hook(
            module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> torch.Tensor:
            if not self._entered:
                raise RuntimeError("C2C hook fired outside its active context")
            if self.model.training or block.training:
                raise RuntimeError("C2C intervention cannot run in training mode")
            if getattr(module, "mode", None) != "vanilla":
                raise RuntimeError("attention normalizer changed away from vanilla")
            if not isinstance(output, torch.Tensor):
                raise TypeError("attention normalizer output must be a tensor")
            _validate_attention(output, self.num_classes)

            expected_tokens = self.num_classes + self.expected_num_patches
            expected_shape = (
                output.shape[0],
                int(block.attn.num_heads),
                expected_tokens,
                expected_tokens,
            )
            if tuple(output.shape) != expected_shape:
                raise RuntimeError(
                    f"block {layer} attention shape {tuple(output.shape)} != "
                    f"{expected_shape}; exact 20-class/patch layout is required"
                )
            if len(inputs) != 3:
                raise RuntimeError(
                    "native normalizer must receive logits, key groups, and query groups"
                )
            _group_ids_match(
                inputs[1],
                batch_size=output.shape[0],
                num_classes=self.num_classes,
                num_patches=self.expected_num_patches,
                device=output.device,
                name="key_group_ids",
            )
            _group_ids_match(
                inputs[2],
                batch_size=output.shape[0],
                num_classes=self.num_classes,
                num_patches=self.expected_num_patches,
                device=output.device,
                name="query_group_ids",
            )

            native_row_error = _max_abs(output.float().sum(dim=-1) - 1.0)
            if native_row_error > self.native_row_tolerance:
                raise RuntimeError(
                    f"block {layer} native attention row-sum error "
                    f"{native_row_error} exceeds {self.native_row_tolerance}"
                )

            rerouted = self_reroute_c2c_attention(output, self.num_classes)
            classes = self.num_classes
            native_c2c = output[..., :classes, :classes]
            rerouted_c2c = rerouted[..., :classes, :classes]
            native_class_mass = native_c2c.sum(dim=-1)
            rerouted_class_mass = rerouted_c2c.sum(dim=-1)
            pre_offdiag_mass, _ = _offdiag_mass(native_c2c)
            post_offdiag_mass, post_offdiag_weights = _offdiag_mass(rerouted_c2c)
            new_diagonal = torch.diagonal(rerouted_c2c, dim1=-2, dim2=-1)

            call_index = self._activation_counts[layer] + 1
            record = C2CInterventionRecord(
                layer_index=layer,
                layer_number=layer + 1,
                call_index=call_index,
                batch_size=int(output.shape[0]),
                num_heads=int(output.shape[1]),
                num_tokens=int(output.shape[-1]),
                dtype=str(output.dtype),
                device=str(output.device),
                patch_rows_max_abs_diff=_max_abs(
                    rerouted[..., classes:, :] - output[..., classes:, :]
                ),
                class_to_patch_max_abs_diff=_max_abs(
                    rerouted[..., :classes, classes:] - output[..., :classes, classes:]
                ),
                class_group_mass_max_abs_diff=_max_abs(
                    rerouted_class_mass - native_class_mass
                ),
                row_sum_max_abs_diff=_max_abs(
                    rerouted.float().sum(dim=-1) - output.float().sum(dim=-1)
                ),
                native_row_sum_max_abs_error=native_row_error,
                rerouted_row_sum_max_abs_error=_max_abs(
                    rerouted.float().sum(dim=-1) - 1.0
                ),
                diagonal_mass_max_abs_diff=_max_abs(new_diagonal - native_class_mass),
                offdiag_pre_mass_mean=_mean(pre_offdiag_mass),
                offdiag_pre_mass_max=_maximum(pre_offdiag_mass),
                offdiag_post_mass_mean=_mean(post_offdiag_mass),
                offdiag_post_mass_max=_maximum(post_offdiag_mass),
                offdiag_post_weight_max_abs=_max_abs(post_offdiag_weights),
            )
            self._activation_counts[layer] = call_index
            self._records.append(record)
            self._assert_structural_record(record)
            return rerouted

        return hook

    def _assert_structural_record(self, record: C2CInterventionRecord) -> None:
        exact_zero_fields = {
            "patch_rows_max_abs_diff": record.patch_rows_max_abs_diff,
            "class_to_patch_max_abs_diff": record.class_to_patch_max_abs_diff,
            "diagonal_mass_max_abs_diff": record.diagonal_mass_max_abs_diff,
            "offdiag_post_mass_mean": record.offdiag_post_mass_mean,
            "offdiag_post_mass_max": record.offdiag_post_mass_max,
            "offdiag_post_weight_max_abs": record.offdiag_post_weight_max_abs,
        }
        nonzero = {
            name: value for name, value in exact_zero_fields.items() if value != 0
        }
        if nonzero:
            raise RuntimeError(
                f"layer {record.layer_number} exact C2C invariant failure: {nonzero}"
            )
        tolerance_fields = {
            "class_group_mass_max_abs_diff": record.class_group_mass_max_abs_diff,
            "row_sum_max_abs_diff": record.row_sum_max_abs_diff,
        }
        exceeded = {
            name: value
            for name, value in tolerance_fields.items()
            if value > self.invariant_tolerance
        }
        if exceeded:
            raise RuntimeError(
                f"layer {record.layer_number} numerical C2C invariant failure: "
                f"{exceeded}, tolerance={self.invariant_tolerance}"
            )
        # The causal invariant is equality with the native row mass. The
        # immutable vanilla CUDA softmax itself can be 1.192e-6 away from one
        # (as already audited in Experiment 2), so requiring the rerouted row
        # to be closer to one than its source would be a false gate.
        if record.rerouted_row_sum_max_abs_error > self.native_row_tolerance:
            raise RuntimeError(
                f"layer {record.layer_number} rerouted attention row-sum error "
                f"{record.rerouted_row_sum_max_abs_error} exceeds native-row "
                f"tolerance={self.native_row_tolerance}"
            )

    def __enter__(self) -> "C2CIntervention":
        if self._entered:
            raise RuntimeError("C2CIntervention contexts cannot be nested")
        if self._used:
            raise RuntimeError("a C2CIntervention instance is single-use")
        self._used = True
        self._validate_model()

        with _ACTIVE_MODELS_LOCK:
            if self.model in _ACTIVE_MODELS:
                raise RuntimeError(
                    "another C2CIntervention is already active for this model"
                )
            _ACTIVE_MODELS.add(self.model)
        self._entered = True
        try:
            for layer in self.layers:
                normalizer = self.model.blocks[layer].attn.normalizer
                self._handles.append(
                    normalizer.register_forward_hook(self._make_hook(layer))
                )
        except BaseException:
            self._remove()
            raise
        return self

    def _remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._entered = False
        with _ACTIVE_MODELS_LOCK:
            _ACTIVE_MODELS.discard(self.model)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._remove()
        if exc_type is None and self.layers:
            counts = tuple(self._activation_counts.values())
            if any(count < 1 for count in counts):
                raise RuntimeError(
                    "not every selected C2C layer activated: "
                    f"{self.activation_counts_1based}"
                )
            if len(set(counts)) != 1:
                raise RuntimeError(
                    "selected C2C layers activated unequal numbers of times: "
                    f"{self.activation_counts_1based}"
                )
