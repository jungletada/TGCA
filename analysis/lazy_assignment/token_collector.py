"""Read-only block hooks for class/patch representation alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from analysis.lazy_assignment.score_utils import class_specific_patch_score


@dataclass(frozen=True)
class TokenCapture:
    """One complete model-forward capture."""

    scores: torch.Tensor
    last_class_tokens: torch.Tensor
    last_patch_tokens: torch.Tensor


class BlockTokenCollector:
    """Collect post-block cosine scores without replacing block outputs.

    Hooks stay inert until :meth:`clear` starts a capture.  This makes the
    lifecycle explicit and prevents stale scores from being reused across
    batches.
    """

    def __init__(self, model: nn.Module, num_classes: int):
        if not hasattr(model, "blocks"):
            raise TypeError("model must expose a .blocks sequence")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.model = model
        self.num_classes = int(num_classes)
        self.depth = len(model.blocks)
        if self.depth < 1:
            raise ValueError("model.blocks must not be empty")
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._expected_num_patches: Optional[int] = None
        self._scores: list[Optional[torch.Tensor]] = []
        self._last_class_tokens: Optional[torch.Tensor] = None
        self._last_patch_tokens: Optional[torch.Tensor] = None

    def register(self) -> "BlockTokenCollector":
        if self.handles:
            raise RuntimeError("collector hooks are already registered")
        for layer_index, block in enumerate(self.model.blocks):
            self.handles.append(
                block.register_forward_hook(self._make_hook(layer_index))
            )
        return self

    def clear(self, expected_num_patches: Optional[int] = None) -> None:
        if not self.handles:
            raise RuntimeError("register collector hooks before starting a capture")
        if self._active:
            raise RuntimeError("previous capture has not been consumed")
        if expected_num_patches is not None and expected_num_patches < 1:
            raise ValueError("expected_num_patches must be positive")
        self._expected_num_patches = (
            None if expected_num_patches is None else int(expected_num_patches)
        )
        self._scores = [None] * self.depth
        self._last_class_tokens = None
        self._last_patch_tokens = None
        self._active = True

    def consume(self) -> TokenCapture:
        if not self._active:
            raise RuntimeError("no active capture to consume")
        missing = [index for index, value in enumerate(self._scores) if value is None]
        if missing:
            raise RuntimeError(f"missing block captures for zero-based layers {missing}")
        if self._last_class_tokens is None or self._last_patch_tokens is None:
            raise RuntimeError("final-layer tokens were not captured")
        capture = TokenCapture(
            scores=torch.stack(self._scores),  # type: ignore[arg-type]
            last_class_tokens=self._last_class_tokens,
            last_patch_tokens=self._last_patch_tokens,
        )
        self._active = False
        self._scores = []
        self._last_class_tokens = None
        self._last_patch_tokens = None
        return capture

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._active = False
        self._scores = []
        self._last_class_tokens = None
        self._last_patch_tokens = None

    def _make_hook(self, layer_index: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            if not self._active:
                return None
            if self._scores[layer_index] is not None:
                raise RuntimeError(f"block {layer_index} fired more than once in one capture")
            if not isinstance(output, (tuple, list)) or not output:
                raise TypeError(
                    f"block {layer_index} must return (tokens, attention), got {type(output)}"
                )
            tokens = output[0]
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise TypeError(
                    f"block {layer_index} output[0] must be [B, N, D] tensor"
                )
            if tokens.shape[1] <= self.num_classes:
                raise RuntimeError(
                    f"block {layer_index} has {tokens.shape[1]} tokens for "
                    f"{self.num_classes} classes"
                )
            num_patches = self._expected_num_patches
            if num_patches is None:
                num_patches = tokens.shape[1] - self.num_classes
            expected_tokens = self.num_classes + num_patches
            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"block {layer_index} produced {tokens.shape[1]} tokens; expected "
                    f"exactly {self.num_classes} class + {num_patches} patch tokens. "
                    "Experiment 1 does not silently absorb register/background tokens."
                )

            class_tokens = tokens[:, : self.num_classes]
            patch_tokens = tokens[:, self.num_classes : expected_tokens]
            self._scores[layer_index] = class_specific_patch_score(
                class_tokens.detach(), patch_tokens.detach()
            ).detach()
            if layer_index == self.depth - 1:
                self._last_class_tokens = class_tokens.detach()
                self._last_patch_tokens = patch_tokens.detach()
            return None

        return hook

    def __enter__(self) -> "BlockTokenCollector":
        return self.register()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()
