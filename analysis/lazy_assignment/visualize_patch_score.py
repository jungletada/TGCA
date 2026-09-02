"""Sanity visualizations for raw and per-map-normalized cosine scores."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


VOC_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def input_tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected normalized [3,H,W] input, got {tuple(image.shape)}")
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(3, 1, 1)
    rgb = image.detach().cpu().float() * std + mean
    return rgb.clamp(0, 1).permute(1, 2, 0).numpy()


def minmax_map(score_map: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    return (values - minimum) / (maximum - minimum + eps)


def _save_one_figure(
    image: np.ndarray,
    layer_maps: list[np.ndarray],
    layer_numbers: Sequence[int],
    title: str,
    path: Path,
    normalized: bool,
) -> None:
    figure, axes = plt.subplots(1, 1 + len(layer_maps), figsize=(3.2 * (1 + len(layer_maps)), 3.4))
    axes[0].imshow(image)
    axes[0].set_title("Model input")
    axes[0].axis("off")
    image_handle = None
    for axis, layer_number, score_map in zip(axes[1:], layer_numbers, layer_maps):
        shown = minmax_map(score_map) if normalized else score_map
        image_handle = axis.imshow(
            shown,
            cmap="viridis" if normalized else "coolwarm",
            interpolation="bilinear",
            vmin=0.0 if normalized else -1.0,
            vmax=1.0,
        )
        axis.set_title(f"Layer {layer_number}")
        axis.axis("off")
    figure.suptitle(title)
    if image_handle is not None:
        figure.colorbar(image_handle, ax=list(axes[1:]), fraction=0.025, pad=0.02)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_score_visualizations(
    image: torch.Tensor,
    positive_scores: np.ndarray,
    positive_class_ids: Sequence[int],
    image_id: str,
    grid_h: int,
    grid_w: int,
    output_dir: Union[str, Path],
    layer_indices: Sequence[int] = (0, 3, 7, 11),
    max_classes: int = 3,
) -> list[str]:
    """Save raw-cosine and min-max figures for up to ``max_classes``."""

    scores = np.asarray(positive_scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"positive_scores must be [L,K,P], got {scores.shape}")
    if scores.shape[1] != len(positive_class_ids):
        raise ValueError("positive score/class-ID counts differ")
    if scores.shape[-1] != grid_h * grid_w:
        raise ValueError("score token count does not match the declared grid")
    if max_classes < 1:
        return []
    for layer_index in layer_indices:
        if not 0 <= layer_index < scores.shape[0]:
            raise ValueError(f"layer index {layer_index} is outside depth {scores.shape[0]}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rgb = input_tensor_to_rgb(image)
    layer_numbers = [index + 1 for index in layer_indices]
    written: list[str] = []
    for local_index, class_id in enumerate(positive_class_ids[:max_classes]):
        class_name = (
            VOC_CLASS_NAMES[int(class_id)]
            if 0 <= int(class_id) < len(VOC_CLASS_NAMES)
            else f"class{int(class_id)}"
        )
        layer_maps = [
            scores[layer_index, local_index].reshape(grid_h, grid_w)
            for layer_index in layer_indices
        ]
        stem = f"{image_id}_class{int(class_id):02d}_{class_name}"
        raw_path = destination / f"{stem}_raw_cosine.png"
        minmax_path = destination / f"{stem}_minmax.png"
        _save_one_figure(
            rgb,
            layer_maps,
            layer_numbers,
            f"{image_id} — {class_name} — raw cosine",
            raw_path,
            normalized=False,
        )
        _save_one_figure(
            rgb,
            layer_maps,
            layer_numbers,
            f"{image_id} — {class_name} — per-map min-max (visual only)",
            minmax_path,
            normalized=True,
        )
        written.extend((str(raw_path), str(minmax_path)))
    return written
