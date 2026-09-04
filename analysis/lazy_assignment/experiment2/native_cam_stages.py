"""Exact, external decomposition of native MCTformer CAM generation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _host_kind(model_name: str) -> str:
    normalized = (
        model_name.lower().replace("+", "plus").replace("_", "").replace("-", "")
    )
    if "plus" in normalized:
        return "mctformerplus"
    if normalized in {"mctformer", "mctformerv2", "mctformerv2cam"}:
        return "mctformer"
    raise ValueError(f"unsupported native CAM host: {model_name!r}")


def patch_logits_from_tokens(
    model: torch.nn.Module,
    patch_tokens: torch.Tensor,
    grid_size: Tuple[int, int],
) -> torch.Tensor:
    """Apply the native 2-D patch head without changing token ordering."""

    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must have shape [B,P,D]")
    grid_h, grid_w = (int(value) for value in grid_size)
    if grid_h < 1 or grid_w < 1 or grid_h * grid_w != patch_tokens.shape[1]:
        raise ValueError(
            f"grid {grid_size} is incompatible with {patch_tokens.shape[1]} patches"
        )
    if not hasattr(model, "head"):
        raise TypeError("model must expose its native patch head")
    batch, _, width = patch_tokens.shape
    patch_grid = patch_tokens.reshape(batch, grid_h, grid_w, width)
    patch_grid = patch_grid.permute(0, 3, 1, 2).contiguous()
    return model.head(patch_grid)


def _validate_reduced_inputs(
    patch_logits: torch.Tensor,
    c2p_layers: torch.Tensor,
    patch_to_patch_sum: torch.Tensor,
    num_classes: int,
) -> Tuple[int, int, int, int]:
    if patch_logits.ndim != 4:
        raise ValueError("patch_logits must have shape [B,C,Hp,Wp]")
    batch, channels, grid_h, grid_w = patch_logits.shape
    if num_classes < 1 or channels != num_classes:
        raise ValueError(
            f"patch logits contain {channels} channels, expected {num_classes}"
        )
    patches = grid_h * grid_w
    if c2p_layers.ndim != 4:
        raise ValueError("c2p_layers must have shape [L,B,C,P]")
    if c2p_layers.shape[0] < 3:
        raise ValueError("native CAM requires at least the final three layers")
    if tuple(c2p_layers.shape[1:]) != (batch, num_classes, patches):
        raise ValueError(
            f"c2p shape {tuple(c2p_layers.shape)} is incompatible with patch logits"
        )
    if tuple(patch_to_patch_sum.shape) != (batch, patches, patches):
        raise ValueError(
            "patch_to_patch_sum must have shape "
            f"{(batch, patches, patches)}, got {tuple(patch_to_patch_sum.shape)}"
        )
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (patch_logits, c2p_layers, patch_to_patch_sum)
    ):
        raise ValueError("native CAM inputs must be finite")
    return batch, patches, grid_h, grid_w


def decompose_native_cam_reduced(
    model_name: str,
    patch_logits: torch.Tensor,
    c2p_layers: torch.Tensor,
    patch_to_patch_sum: torch.Tensor,
    num_classes: int = 20,
) -> Dict[str, torch.Tensor]:
    """Decompose native CAM from reduced attention tensors.

    ``c2p_layers`` must already be head-mean raw global attention.  It must not
    be patch-conditionalized.  ``patch_to_patch_sum`` is the unnormalized sum
    of head-mean p2p submatrices over all available layers.
    """

    batch, _, grid_h, grid_w = _validate_reduced_inputs(
        patch_logits, c2p_layers, patch_to_patch_sum, num_classes
    )
    kind = _host_kind(model_name)

    # Both native implementations detach the patch-head map before ReLU.
    patch_raw = patch_logits.detach().clone()
    patch_cam = F.relu(patch_raw)
    if kind == "mctformer":
        official_c2p_flat = c2p_layers[-3:].sum(dim=0)
        class_attention_flat = official_c2p_flat * patch_cam.flatten(2)
    else:
        official_c2p_flat = c2p_layers[-3:].mean(dim=0)
        class_attention_flat = torch.sqrt(official_c2p_flat * patch_cam.flatten(2))

    class_attention_cam = class_attention_flat.reshape(
        batch, num_classes, grid_h, grid_w
    )
    final_cam = torch.matmul(
        patch_to_patch_sum.unsqueeze(1),
        class_attention_flat.unsqueeze(-1),
    ).reshape(batch, num_classes, grid_h, grid_w)
    official_c2p = official_c2p_flat.reshape(batch, num_classes, grid_h, grid_w)
    return {
        "patch_logits": patch_raw,
        "patch_cam": patch_cam,
        "c2p_layers": c2p_layers,
        "official_c2p_flat": official_c2p_flat,
        "official_c2p": official_c2p,
        "class_attention_cam": class_attention_cam,
        "official_p2p": patch_to_patch_sum,
        "final_cam": final_cam,
    }


def decompose_native_cam(
    model_name: str,
    patch_logits: torch.Tensor,
    head_mean_attentions: torch.Tensor,
    num_classes: int = 20,
) -> Dict[str, torch.Tensor]:
    """Reproduce the exact native patch, c2p-refined, and propagated CAMs.

    Args:
        model_name: MCTformer/MCTformerV2 or MCTformer+ host identifier.
        patch_logits: Native patch-head output ``[B,C,Hp,Wp]`` before ReLU.
        head_mean_attentions: Softmax attention after head averaging, with
            shape ``[L,B,C+P,C+P]``.
        num_classes: Number of leading class tokens/channels.
    """

    if head_mean_attentions.ndim != 4:
        raise ValueError("head_mean_attentions must have shape [L,B,C+P,C+P]")
    batch, channels, grid_h, grid_w = patch_logits.shape
    patches = grid_h * grid_w
    total = num_classes + patches
    expected = (batch, total, total)
    if tuple(head_mean_attentions.shape[1:]) != expected:
        raise ValueError(
            f"attention shape {tuple(head_mean_attentions.shape)} must be "
            f"[L,{batch},{total},{total}]; extra tokens are not accepted"
        )
    c2p_layers = head_mean_attentions[:, :, :num_classes, num_classes:total]
    patch_to_patch_sum = head_mean_attentions[
        :, :, num_classes:total, num_classes:total
    ].sum(dim=0)
    return decompose_native_cam_reduced(
        model_name,
        patch_logits,
        c2p_layers,
        patch_to_patch_sum,
        num_classes=num_classes,
    )


def native_cam_max_abs_diff(
    stages: Dict[str, torch.Tensor], native_cam: torch.Tensor
) -> float:
    """Return max absolute difference between decomposition and native output."""

    if "final_cam" not in stages:
        raise KeyError("stages does not contain final_cam")
    decomposed = stages["final_cam"]
    if decomposed.shape != native_cam.shape or decomposed.dtype != native_cam.dtype:
        raise AssertionError(
            "native CAM metadata mismatch: "
            f"{tuple(decomposed.shape)}/{decomposed.dtype} != "
            f"{tuple(native_cam.shape)}/{native_cam.dtype}"
        )
    if not decomposed.numel():
        return 0.0
    return float((decomposed - native_cam).abs().max().item())


def assert_native_cam_equivalent(
    stages: Dict[str, torch.Tensor],
    native_cam: torch.Tensor,
    tolerance: float = 1e-6,
) -> float:
    """Assert raw-grid native CAM equivalence before any post-processing."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    difference = native_cam_max_abs_diff(stages, native_cam)
    if difference > tolerance:
        raise AssertionError(
            f"native CAM differs by {difference}, tolerance={tolerance}"
        )
    return difference
