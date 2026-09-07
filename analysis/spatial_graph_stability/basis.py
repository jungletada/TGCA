"""Equivalent embedding-basis transforms and the official LaST selector."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BasisTransform:
    name: str
    kind: str
    dimension: int
    seed: int | None
    matrix: np.ndarray
    channel_correspondence: np.ndarray | None = None
    channel_signs: np.ndarray | None = None

    def metadata(self) -> dict[str, object]:
        matrix = np.ascontiguousarray(self.matrix.astype(np.float64, copy=False))
        return {
            "name": self.name,
            "kind": self.kind,
            "dimension": self.dimension,
            "seed": self.seed,
            "matrix_dtype": str(matrix.dtype),
            "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "orthogonality_max_abs_error": orthogonality_error(matrix),
            "channel_correspondence": (
                self.channel_correspondence.tolist()
                if self.channel_correspondence is not None
                else None
            ),
            "channel_signs": (
                self.channel_signs.tolist()
                if self.channel_signs is not None
                else None
            ),
        }


def orthogonality_error(matrix: np.ndarray | torch.Tensor) -> float:
    if isinstance(matrix, torch.Tensor):
        value = matrix.detach().cpu().double().numpy()
    else:
        value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"orthogonal matrix must be square, got {value.shape}")
    identity = np.eye(value.shape[0], dtype=np.float64)
    return float(np.max(np.abs(value.T @ value - identity)))


def _haar_matrix(dimension: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    gaussian = generator.standard_normal((dimension, dimension), dtype=np.float64)
    q, upper = np.linalg.qr(gaussian)
    diagonal = np.diag(upper)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return np.ascontiguousarray(q * signs[None, :], dtype=np.float64)


def generate_basis_transforms(
    dimension: int,
    *,
    base_seed: int = 20260901,
    haar_count: int = 10,
) -> tuple[BasisTransform, ...]:
    dimension = int(dimension)
    if dimension < 1 or haar_count < 0:
        raise ValueError("dimension must be positive and haar_count non-negative")
    identity = np.eye(dimension, dtype=np.float64)
    transforms: list[BasisTransform] = [
        BasisTransform("identity", "identity", dimension, None, identity)
    ]

    permutation_seed = int(base_seed)
    permutation = np.random.default_rng(permutation_seed).permutation(dimension)
    permutation_matrix = identity[:, permutation]
    transforms.append(
        BasisTransform(
            "permutation",
            "permutation",
            dimension,
            permutation_seed,
            np.ascontiguousarray(permutation_matrix),
            channel_correspondence=permutation.astype(np.int64),
            channel_signs=np.ones(dimension, dtype=np.int8),
        )
    )

    signed_seed = int(base_seed) + 1
    signed_generator = np.random.default_rng(signed_seed)
    signed_permutation = signed_generator.permutation(dimension)
    signs = signed_generator.choice(np.array([-1, 1], dtype=np.int8), dimension)
    signed_matrix = identity[:, signed_permutation] * signs[None, :]
    transforms.append(
        BasisTransform(
            "signed_permutation",
            "signed_permutation",
            dimension,
            signed_seed,
            np.ascontiguousarray(signed_matrix),
            channel_correspondence=signed_permutation.astype(np.int64),
            channel_signs=signs,
        )
    )

    for index in range(int(haar_count)):
        seed = int(base_seed) + 10 + index
        transforms.append(
            BasisTransform(
                f"haar_{index:02d}",
                "haar",
                dimension,
                seed,
                _haar_matrix(dimension, seed),
            )
        )
    return tuple(transforms)


def gaussian_channel_kernel(
    width: int,
    *,
    sigma: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    width = int(width)
    sigma = math.sqrt(width) if sigma is None else float(sigma)
    if width < 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("width and sigma must be finite and positive")
    positions = torch.arange(
        -width // 2 + 1,
        width // 2 + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-0.5 * (positions / sigma).square())
    return kernel / kernel.max()


def last_channel_selector(
    patch_tokens: torch.Tensor,
    *,
    topk: int = 1,
    sigma: float | None = None,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the official LaST channel selector on ``[B,N,D]`` tokens.

    The official supervised repository uses no denominator clamp. Passing
    ``eps=None`` therefore reproduces it exactly. A positive ``eps`` exists
    only for explicit robustness controls and is never used in Phase A.
    """

    if patch_tokens.ndim != 3:
        raise ValueError(
            f"patch tokens must have shape [B,N,D], got {patch_tokens.shape}"
        )
    if patch_tokens.shape[1] < 1 or patch_tokens.shape[2] < 1:
        raise ValueError("LaST selector requires non-empty patch/channel axes")
    if topk < 1 or topk > patch_tokens.shape[1]:
        raise ValueError("topk must be in [1, number of patches]")
    if eps is not None and (not math.isfinite(float(eps)) or float(eps) <= 0):
        raise ValueError("eps must be positive when supplied")

    width = patch_tokens.shape[-1]
    kernel = gaussian_channel_kernel(
        width,
        sigma=sigma,
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    ).view(1, 1, width)
    spectrum = torch.fft.fft(patch_tokens, dim=-1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1) * kernel
    low_pass = torch.fft.ifft(
        torch.fft.ifftshift(spectrum, dim=-1), dim=-1
    ).real
    denominator = (low_pass - patch_tokens).abs()
    if eps is not None:
        denominator = denominator.clamp_min(float(eps))
    stability = patch_tokens / denominator
    indices = torch.topk(stability, k=topk, dim=1, largest=True).indices
    selected = torch.gather(patch_tokens, dim=1, index=indices)
    pooled = selected.mean(dim=1)
    return pooled, indices, stability, low_pass


def transform_linear_weight(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or rotation.ndim != 2:
        raise ValueError("linear weight and rotation must both be matrices")
    if rotation.shape[0] != rotation.shape[1] or weight.shape[1] != rotation.shape[0]:
        raise ValueError("linear weight and rotation dimensions do not match")
    return weight @ rotation


def transform_conv_input_basis(
    weight: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    if weight.ndim != 4 or rotation.ndim != 2:
        raise ValueError("expected Conv2d weight [O,I,H,W] and matrix [I,I]")
    if rotation.shape[0] != rotation.shape[1] or weight.shape[1] != rotation.shape[0]:
        raise ValueError("convolution weight and rotation dimensions do not match")
    return torch.einsum("oihw,ij->ojhw", weight, rotation)


def cosine_affinity(patch_tokens: torch.Tensor) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError("patch tokens must have shape [B,N,D]")
    unit = F.normalize(patch_tokens, p=2, dim=-1, eps=1e-12)
    return ((unit @ unit.transpose(-1, -2)) + 1.0) * 0.5


def transform_metadata(transforms: Iterable[BasisTransform]) -> list[dict[str, object]]:
    return [transform.metadata() for transform in transforms]
