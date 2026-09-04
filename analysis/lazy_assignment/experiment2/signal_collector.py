"""Read-only signal hooks for Experiment 2 semantic-ownership analysis.

The collector observes the native pre-norm Transformer path without replacing
any module output.  Full ``[B, H, T, T]`` attention matrices are reduced inside
the block hook; only the all-layer, head-mean patch-to-patch sum required by the
native CAM is retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from analysis.lazy_assignment.score_utils import class_specific_patch_score


@dataclass(frozen=True)
class SignalCapture:
    """Reduced signals from one complete native forward.

    Layer is the leading dimension for every layer-wise field.  The transient
    ``qk_c2p_heads`` field is intentionally not a full attention square; callers
    may use it for per-head region summaries and then release the capture.
    """

    feature_post_scores: torch.Tensor  # [L, B, C, P]
    feature_norm_scores: torch.Tensor  # [L, B, C, P]
    qk_mean_scores: torch.Tensor  # [L, B, C, P]
    qk_head_std: torch.Tensor  # [L, B, C, P]
    qk_c2p_heads: torch.Tensor  # [L, B, H, C, P], transient
    attn_c2p_raw: torch.Tensor  # [L, B, C, P]
    attn_c2p_conditional: torch.Tensor  # [L, B, C, P]
    attn_patch_mass: torch.Tensor  # [L, B, C]
    class_token_pairwise_cosine: torch.Tensor  # [L, B, C, C]
    patch_norms: torch.Tensor  # [L, B, P]
    last_class_tokens: torch.Tensor  # [B, C, D]
    last_patch_tokens: torch.Tensor  # [B, P, D]
    patch_to_patch_sum: torch.Tensor  # [B, P, P]
    qk_attention_max_abs_diff: torch.Tensor  # [L]
    attention_row_sum_max_abs_error: torch.Tensor  # [L]
    pre_norm_input_max_abs_diff: torch.Tensor  # [L]
    norm_qkv_input_max_abs_diff: torch.Tensor  # [L]


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(
            "tensor metadata mismatch: "
            f"{tuple(left.shape)}/{left.dtype} != {tuple(right.shape)}/{right.dtype}"
        )
    if not left.numel():
        return 0.0
    if left.is_floating_point() or left.is_complex():
        return float((left - right).abs().max().item())
    return 0.0 if torch.equal(left, right) else math.inf


def tensor_tree_max_abs_diff(left: object, right: object, path: str = "root") -> float:
    """Return the maximum tensor difference in identically structured outputs."""

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"type mismatch at {path}")
        return _max_abs_difference(left, right)
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"sequence mismatch at {path}")
        return max(
            (
                tensor_tree_max_abs_diff(a, b, f"{path}[{index}]")
                for index, (a, b) in enumerate(zip(left, right))
            ),
            default=0.0,
        )
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            raise AssertionError(f"mapping mismatch at {path}")
        return max(
            (
                tensor_tree_max_abs_diff(left[key], right[key], f"{path}.{key}")
                for key in left
            ),
            default=0.0,
        )
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")
    return 0.0


def assert_no_change(
    reference: object, observed: object, tolerance: float = 0.0
) -> float:
    """Assert that instrumentation did not change a nested native output."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    difference = tensor_tree_max_abs_diff(reference, observed)
    if difference > tolerance:
        raise AssertionError(
            f"instrumented output changed by {difference}, tolerance={tolerance}"
        )
    return difference


def split_qkv(qkv_output: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, ...]:
    """Apply the exact reshape/permute convention used by ``Attention.forward``."""

    if qkv_output.ndim != 3:
        raise ValueError("qkv output must have shape [B, T, 3D]")
    batch, tokens, triple_dim = qkv_output.shape
    if num_heads < 1 or triple_dim % (3 * num_heads):
        raise ValueError("qkv width must be divisible by 3 * num_heads")
    head_dim = triple_dim // (3 * num_heads)
    qkv = qkv_output.reshape(batch, tokens, 3, num_heads, head_dim).permute(
        2, 0, 3, 1, 4
    )
    return qkv[0], qkv[1], qkv[2]


def class_to_patch_qk(
    qkv_output: torch.Tensor,
    num_heads: int,
    num_classes: int,
    num_patches: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return c2p QK energies and vanilla-softmax c2p probabilities.

    Only class-query logits ``[B,H,C,T]`` are materialized.  Softmax still uses
    all class and patch keys, exactly matching vanilla global attention.
    """

    query, key, _ = split_qkv(qkv_output, num_heads)
    total = num_classes + num_patches
    if query.shape[-2] != total:
        raise ValueError(
            f"expected exactly {num_classes}+{num_patches} baseline tokens, "
            f"got {query.shape[-2]}"
        )
    class_query = query[:, :, :num_classes]
    all_energy = (class_query @ key.transpose(-2, -1)) * float(scale)
    patch_energy = all_energy[..., num_classes:total]
    expected_attention = torch.softmax(all_energy.float(), dim=-1).to(
        dtype=all_energy.dtype
    )[..., num_classes:total]
    return patch_energy.float(), expected_attention


def _class_pairwise_cosine(class_tokens: torch.Tensor) -> torch.Tensor:
    unit = torch.nn.functional.normalize(class_tokens.float(), p=2, dim=-1, eps=1e-12)
    return torch.einsum("bcd,bed->bce", unit, unit)


class SignalCollector:
    """Collect Experiment 2 signals with read-only forward hooks.

    The supported production contract is the native baseline ordering
    ``[class tokens, patch tokens]``.  An explicit expected patch count makes
    any extra register/background token a hard error.
    """

    def __init__(self, model: nn.Module, num_classes: int = 20):
        if not hasattr(model, "blocks"):
            raise TypeError("model must expose a .blocks sequence")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.model = model
        self.num_classes = int(num_classes)
        self.depth = len(model.blocks)
        if self.depth < 1:
            raise ValueError("model.blocks must not be empty")
        for index, block in enumerate(model.blocks):
            if not all(hasattr(block, name) for name in ("norm1", "attn")):
                raise TypeError(f"block {index} lacks native norm1/attn modules")
            if not hasattr(block.attn, "qkv") or not hasattr(block.attn, "num_heads"):
                raise TypeError(f"block {index} lacks native attention qkv metadata")
            normalizer = getattr(block.attn, "normalizer", None)
            if (
                normalizer is not None
                and getattr(normalizer, "mode", "vanilla") != "vanilla"
            ):
                raise ValueError(
                    "Experiment 2 signal collection requires vanilla attention"
                )

        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._expected_num_patches: Optional[int] = None
        self._reset_storage()

    def _reset_storage(self) -> None:
        self._pre_tokens: list[Optional[torch.Tensor]] = []
        self._norm_tokens: list[Optional[torch.Tensor]] = []
        self._expected_c2p_heads: list[Optional[torch.Tensor]] = []
        self._feature_post: list[Optional[torch.Tensor]] = []
        self._feature_norm: list[Optional[torch.Tensor]] = []
        self._qk_heads: list[Optional[torch.Tensor]] = []
        self._qk_mean: list[Optional[torch.Tensor]] = []
        self._qk_std: list[Optional[torch.Tensor]] = []
        self._attn_raw: list[Optional[torch.Tensor]] = []
        self._attn_conditional: list[Optional[torch.Tensor]] = []
        self._attn_mass: list[Optional[torch.Tensor]] = []
        self._class_pairwise: list[Optional[torch.Tensor]] = []
        self._patch_norms: list[Optional[torch.Tensor]] = []
        self._qk_diff: list[Optional[float]] = []
        self._row_error: list[Optional[float]] = []
        self._pre_norm_diff: list[Optional[float]] = []
        self._norm_qkv_diff: list[Optional[float]] = []
        self._patch_to_patch_sum: Optional[torch.Tensor] = None
        self._last_class_tokens: Optional[torch.Tensor] = None
        self._last_patch_tokens: Optional[torch.Tensor] = None

    def register(self) -> "SignalCollector":
        if self.handles:
            raise RuntimeError("collector hooks are already registered")
        for layer, block in enumerate(self.model.blocks):
            self.handles.append(
                block.register_forward_pre_hook(self._make_block_pre_hook(layer))
            )
            self.handles.append(
                block.norm1.register_forward_hook(self._make_norm_hook(layer))
            )
            self.handles.append(
                block.attn.qkv.register_forward_hook(self._make_qkv_hook(layer))
            )
            self.handles.append(
                block.register_forward_hook(self._make_block_hook(layer))
            )
        return self

    def clear(self, expected_num_patches: Optional[int] = None) -> None:
        if not self.handles:
            raise RuntimeError("register hooks before starting a capture")
        if self._active:
            raise RuntimeError("previous capture has not been consumed")
        if expected_num_patches is not None and expected_num_patches < 1:
            raise ValueError("expected_num_patches must be positive")
        self._expected_num_patches = (
            None if expected_num_patches is None else int(expected_num_patches)
        )
        self._reset_storage()
        for name in (
            "_pre_tokens",
            "_norm_tokens",
            "_expected_c2p_heads",
            "_feature_post",
            "_feature_norm",
            "_qk_heads",
            "_qk_mean",
            "_qk_std",
            "_attn_raw",
            "_attn_conditional",
            "_attn_mass",
            "_class_pairwise",
            "_patch_norms",
            "_qk_diff",
            "_row_error",
            "_pre_norm_diff",
            "_norm_qkv_diff",
        ):
            setattr(self, name, [None] * self.depth)
        self._active = True

    def _token_parts(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise TypeError("native block tokens must have shape [B,T,D]")
        if self._expected_num_patches is None:
            inferred = int(tokens.shape[1]) - self.num_classes
            if inferred < 1:
                raise RuntimeError("token sequence contains no patch tokens")
            self._expected_num_patches = inferred
        expected = self.num_classes + self._expected_num_patches
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"expected exactly {self.num_classes} class + "
                f"{self._expected_num_patches} patch tokens, got {tokens.shape[1]}"
            )
        return tokens[:, : self.num_classes], tokens[:, self.num_classes : expected]

    def _make_block_pre_hook(self, layer: int):
        def hook(_module: nn.Module, inputs: Tuple[object, ...]):
            if not self._active:
                return None
            if self._pre_tokens[layer] is not None:
                raise RuntimeError(f"block {layer} fired more than once")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"block {layer} input[0] must be a tensor")
            self._token_parts(inputs[0])
            self._pre_tokens[layer] = inputs[0].detach()
            return None

        return hook

    def _make_norm_hook(self, layer: int):
        def hook(_module: nn.Module, inputs: Tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._norm_tokens[layer] is not None:
                raise RuntimeError(f"norm1 {layer} fired more than once")
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise TypeError("norm1 hook requires tensor input and output")
            pre = self._pre_tokens[layer]
            if pre is None:
                raise RuntimeError("norm1 fired before its block pre-hook")
            self._pre_norm_diff[layer] = _max_abs_difference(pre, inputs[0].detach())
            classes, patches = self._token_parts(output)
            self._feature_norm[layer] = class_specific_patch_score(classes, patches)
            self._norm_tokens[layer] = output.detach()
            return None

        return hook

    def _make_qkv_hook(self, layer: int):
        block = self.model.blocks[layer]

        def hook(_module: nn.Module, inputs: Tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._qk_heads[layer] is not None:
                raise RuntimeError(f"qkv {layer} fired more than once")
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise TypeError("qkv hook requires tensor input and output")
            norm = self._norm_tokens[layer]
            if norm is None:
                raise RuntimeError("qkv fired before norm1")
            self._norm_qkv_diff[layer] = _max_abs_difference(norm, inputs[0].detach())
            assert self._expected_num_patches is not None
            energy, expected = class_to_patch_qk(
                output,
                int(block.attn.num_heads),
                self.num_classes,
                self._expected_num_patches,
                float(block.attn.scale),
            )
            energy = energy.detach()
            self._qk_heads[layer] = energy
            self._qk_mean[layer] = energy.mean(dim=1)
            self._qk_std[layer] = energy.std(dim=1, unbiased=False)
            self._expected_c2p_heads[layer] = expected.detach()
            return None

        return hook

    def _make_block_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: Tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._feature_post[layer] is not None:
                raise RuntimeError(f"block {layer} output fired more than once")
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise TypeError("native block must return (tokens, attention)")
            tokens, attention = output[0], output[1]
            if not isinstance(tokens, torch.Tensor) or not isinstance(
                attention, torch.Tensor
            ):
                raise TypeError("native block tokens and attention must be tensors")
            classes, patches = self._token_parts(tokens)
            assert self._expected_num_patches is not None
            expected_tokens = self.num_classes + self._expected_num_patches
            expected_shape = (
                tokens.shape[0],
                int(self.model.blocks[layer].attn.num_heads),
                expected_tokens,
                expected_tokens,
            )
            if tuple(attention.shape) != expected_shape:
                raise RuntimeError(
                    f"attention {layer} shape {tuple(attention.shape)} != {expected_shape}"
                )

            self._feature_post[layer] = class_specific_patch_score(classes, patches)
            self._class_pairwise[layer] = _class_pairwise_cosine(classes)
            self._patch_norms[layer] = patches.float().norm(dim=-1)

            c2p_heads = attention[
                :, :, : self.num_classes, self.num_classes : expected_tokens
            ]
            expected_c2p = self._expected_c2p_heads[layer]
            if expected_c2p is None:
                raise RuntimeError("block output fired before qkv")
            self._qk_diff[layer] = float(
                (c2p_heads.float() - expected_c2p.float()).abs().max().item()
            )

            raw = c2p_heads.mean(dim=1).float()
            mass = raw.sum(dim=-1)
            if torch.any(mass <= 0) or not torch.isfinite(mass).all():
                raise RuntimeError("class-to-patch attention has invalid patch mass")
            self._attn_raw[layer] = raw.detach()
            self._attn_mass[layer] = mass.detach()
            self._attn_conditional[layer] = (raw / mass.unsqueeze(-1)).detach()

            p2p = (
                attention[
                    :,
                    :,
                    self.num_classes : expected_tokens,
                    self.num_classes : expected_tokens,
                ]
                .mean(dim=1)
                .float()
            )
            self._patch_to_patch_sum = (
                p2p.detach()
                if self._patch_to_patch_sum is None
                else self._patch_to_patch_sum + p2p.detach()
            )
            self._row_error[layer] = float(
                (attention.float().sum(dim=-1) - 1.0).abs().max().item()
            )
            if layer == self.depth - 1:
                self._last_class_tokens = classes.detach()
                self._last_patch_tokens = patches.detach()

            # Release temporary full-width token references and expected heads.
            self._pre_tokens[layer] = None
            self._norm_tokens[layer] = None
            self._expected_c2p_heads[layer] = None
            return None

        return hook

    @staticmethod
    def _stack_complete(
        values: Sequence[Optional[torch.Tensor]], name: str
    ) -> torch.Tensor:
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            raise RuntimeError(f"missing {name} captures for layers {missing}")
        return torch.stack(list(values))  # type: ignore[arg-type]

    @staticmethod
    def _float_tensor_complete(
        values: Sequence[Optional[float]], name: str
    ) -> torch.Tensor:
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            raise RuntimeError(f"missing {name} captures for layers {missing}")
        return torch.tensor(list(values), dtype=torch.float64)  # type: ignore[arg-type]

    def consume(self) -> SignalCapture:
        if not self._active:
            raise RuntimeError("no active capture to consume")
        if (
            self._last_class_tokens is None
            or self._last_patch_tokens is None
            or self._patch_to_patch_sum is None
        ):
            raise RuntimeError("forward did not reach the final native block")
        capture = SignalCapture(
            feature_post_scores=self._stack_complete(
                self._feature_post, "post-feature"
            ),
            feature_norm_scores=self._stack_complete(
                self._feature_norm, "normalized-feature"
            ),
            qk_mean_scores=self._stack_complete(self._qk_mean, "QK mean"),
            qk_head_std=self._stack_complete(self._qk_std, "QK head std"),
            qk_c2p_heads=self._stack_complete(self._qk_heads, "QK heads"),
            attn_c2p_raw=self._stack_complete(self._attn_raw, "raw c2p"),
            attn_c2p_conditional=self._stack_complete(
                self._attn_conditional, "conditional c2p"
            ),
            attn_patch_mass=self._stack_complete(self._attn_mass, "patch mass"),
            class_token_pairwise_cosine=self._stack_complete(
                self._class_pairwise, "class pairwise cosine"
            ),
            patch_norms=self._stack_complete(self._patch_norms, "patch norms"),
            last_class_tokens=self._last_class_tokens,
            last_patch_tokens=self._last_patch_tokens,
            patch_to_patch_sum=self._patch_to_patch_sum,
            qk_attention_max_abs_diff=self._float_tensor_complete(
                self._qk_diff, "QK equivalence"
            ),
            attention_row_sum_max_abs_error=self._float_tensor_complete(
                self._row_error, "attention row sum"
            ),
            pre_norm_input_max_abs_diff=self._float_tensor_complete(
                self._pre_norm_diff, "pre/norm input equivalence"
            ),
            norm_qkv_input_max_abs_diff=self._float_tensor_complete(
                self._norm_qkv_diff, "norm/QKV input equivalence"
            ),
        )
        self._active = False
        self._reset_storage()
        return capture

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._active = False
        self._expected_num_patches = None
        self._reset_storage()

    def __enter__(self) -> "SignalCollector":
        return self.register()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()
