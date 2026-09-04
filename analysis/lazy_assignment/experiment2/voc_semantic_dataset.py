"""Deterministic, jointly transformed VOC RGB images and semantic masks.

The RGB branch deliberately reproduces the Experiment 1 validation transform:
resize the short side to ``int(256 / 224 * input_size)`` with bicubic
interpolation, center-crop to ``input_size``, convert to a tensor, and apply the
ImageNet normalization.  The semantic mask follows exactly the same geometric
operations with nearest-neighbor interpolation and remains an integer tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import PIL.Image
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from datasets_cam import load_image_label_list_from_npy_voc, load_img_name_list


VOC_BACKGROUND_ID = 0
VOC_FOREGROUND_IDS = frozenset(range(1, 21))
VOC_VOID_ID = 255
VOC_ALLOWED_MASK_IDS = frozenset({VOC_BACKGROUND_ID, VOC_VOID_ID, *VOC_FOREGROUND_IDS})


def _validate_mask_ids(mask: np.ndarray, *, source: str) -> None:
    observed = {int(value) for value in np.unique(mask)}
    invalid = sorted(observed.difference(VOC_ALLOWED_MASK_IDS))
    if invalid:
        raise ValueError(
            f"{source} contains invalid VOC semantic IDs {invalid}; expected only "
            "0, 1..20, or 255"
        )


class Experiment2JointTransform:
    """Apply the Experiment 1 RGB transform and matching mask geometry.

    ``geometry`` in the return value is composed only of JSON-serializable
    scalars and lists.  Sizes are represented in ``[height, width]`` order;
    ``crop_box_ltrb`` follows the PIL convention ``[left, top, right, bottom]``.
    """

    def __init__(self, input_size: int = 448) -> None:
        if int(input_size) < 1:
            raise ValueError(f"input_size must be positive, got {input_size}")
        self.input_size = int(input_size)
        self.resize_short_side = int((256 / 224) * self.input_size)

    def __call__(
        self, image: PIL.Image.Image, mask: PIL.Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if not isinstance(image, PIL.Image.Image):
            raise TypeError(f"image must be a PIL image, got {type(image)!r}")
        if not isinstance(mask, PIL.Image.Image):
            raise TypeError(f"mask must be a PIL image, got {type(mask)!r}")

        image = image.convert("RGB")
        raw_width, raw_height = image.size
        mask_width, mask_height = mask.size
        if (mask_width, mask_height) != (raw_width, raw_height):
            raise ValueError(
                "RGB image and semantic mask must have identical raw geometry; "
                f"got image={(raw_height, raw_width)} and "
                f"mask={(mask_height, mask_width)}"
            )

        raw_mask = np.asarray(mask, dtype=np.uint8)
        _validate_mask_ids(raw_mask, source="raw semantic mask")

        resized_image = transform_functional.resize(
            image,
            self.resize_short_side,
            interpolation=InterpolationMode.BICUBIC,
        )
        resized_mask = transform_functional.resize(
            mask,
            self.resize_short_side,
            interpolation=InterpolationMode.NEAREST,
        )
        resized_width, resized_height = resized_image.size
        if resized_mask.size != resized_image.size:
            raise RuntimeError(
                "joint resize produced different RGB and mask shapes: "
                f"{resized_image.size} versus {resized_mask.size}"
            )
        if resized_height < self.input_size or resized_width < self.input_size:
            raise RuntimeError(
                "resized input is unexpectedly smaller than the requested crop: "
                f"resized={(resized_height, resized_width)}, crop={self.input_size}"
            )

        # torchvision.transforms.functional.center_crop uses round rather than
        # floor for odd differences.  Recording the same offsets makes the
        # geometry metadata an exact account of the operation below.
        crop_top = int(round((resized_height - self.input_size) / 2.0))
        crop_left = int(round((resized_width - self.input_size) / 2.0))
        cropped_image = transform_functional.center_crop(
            img=resized_image, output_size=self.input_size
        )
        cropped_mask = transform_functional.center_crop(
            img=resized_mask, output_size=self.input_size
        )

        image_tensor = transform_functional.to_tensor(cropped_image)
        image_tensor = transform_functional.normalize(
            image_tensor,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        mask_array = np.array(cropped_mask, dtype=np.uint8, copy=True)
        _validate_mask_ids(mask_array, source="transformed semantic mask")
        mask_tensor = torch.from_numpy(mask_array).to(dtype=torch.int64)

        geometry: dict[str, object] = {
            "original_size": [int(raw_height), int(raw_width)],
            "raw_size": [int(raw_height), int(raw_width)],
            "resize_short_side": int(self.resize_short_side),
            "resized_size": [int(resized_height), int(resized_width)],
            "crop_size": [int(self.input_size), int(self.input_size)],
            "crop_offset": [int(crop_top), int(crop_left)],
            "crop_box_ltrb": [
                int(crop_left),
                int(crop_top),
                int(crop_left + self.input_size),
                int(crop_top + self.input_size),
            ],
            "output_size": [int(self.input_size), int(self.input_size)],
            "image_interpolation": "bicubic",
            "mask_interpolation": "nearest",
            "size_order": "height_width",
            "horizontal_flip": False,
        }
        return image_tensor, mask_tensor, geometry


def build_joint_transform(input_size: int = 448) -> Experiment2JointTransform:
    """Build the deterministic RGB/mask transform used by Experiment 2."""

    return Experiment2JointTransform(input_size=input_size)


def resolve_semantic_mask_path(
    voc_root: Union[str, Path],
    image_id: str,
    mask_directories: Sequence[str] = ("SegmentationClass", "SegmentationClassAug"),
) -> Path:
    """Resolve a VOC semantic mask without mutating or synthesizing GT files."""

    root = Path(voc_root).resolve()
    tried: list[Path] = []
    for directory in mask_directories:
        candidate = root / str(directory) / f"{image_id}.png"
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    paths = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(f"semantic mask for {image_id!r} not found; tried {paths}")


class VOCSemanticDataset(Dataset):
    """VOC evaluation dataset with jointly aligned RGB and semantic GT.

    Each sample exposes exactly the integration keys used by the Experiment 2
    collector: ``name``, ``image``, ``label``, ``mask``, and ``mask_geometry``.
    Labels are zero-based 20-way image-level targets while semantic mask IDs use
    the native VOC convention (0 background, 1..20 foreground, 255 void).
    """

    def __init__(
        self,
        voc_root: Union[str, Path],
        list_path: Union[str, Path],
        input_size: int = 448,
        limit: int = 0,
        transform: Experiment2JointTransform | None = None,
        mask_directories: Sequence[str] = ("SegmentationClass", "SegmentationClassAug"),
    ) -> None:
        self.voc_root = Path(voc_root).resolve()
        self.list_path = Path(list_path).resolve()
        if not self.voc_root.is_dir():
            raise FileNotFoundError(self.voc_root)
        if not self.list_path.is_file():
            raise FileNotFoundError(self.list_path)
        if int(input_size) < 1:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if int(limit) < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if not mask_directories:
            raise ValueError("mask_directories must contain at least one directory")

        image_ids = load_img_name_list(str(self.list_path))
        if limit:
            image_ids = image_ids[: int(limit)]
        if not image_ids:
            raise ValueError(f"no image IDs found in {self.list_path}")

        self.image_ids = image_ids
        self.labels = load_image_label_list_from_npy_voc(
            str(self.voc_root), self.image_ids
        )
        self.transform = transform or build_joint_transform(input_size=int(input_size))
        self.mask_directories = tuple(str(value) for value in mask_directories)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id = self.image_ids[index]
        image_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        mask_path = resolve_semantic_mask_path(
            self.voc_root,
            image_id,
            mask_directories=self.mask_directories,
        )

        with PIL.Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with PIL.Image.open(mask_path) as source_mask:
            mask = source_mask.copy()
        image_tensor, mask_tensor, geometry = self.transform(image, mask)

        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        if label.shape != (20,):
            raise ValueError(
                f"image-level label for {image_id!r} has shape {tuple(label.shape)}, "
                "expected (20,)"
            )
        return {
            "name": image_id,
            "image": image_tensor,
            "label": label,
            "mask": mask_tensor,
            "mask_geometry": geometry,
        }

    def __len__(self) -> int:
        return len(self.image_ids)
