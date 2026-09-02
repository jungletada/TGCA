"""VOC validation dataset wrapper that preserves image identifiers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import PIL.Image
import torch
from torch.utils.data import Dataset

from datasets_cam import (
    build_transform,
    load_image_label_list_from_npy_voc,
    load_img_name_list,
)


class VOCScoreDataset(Dataset):
    """Return deterministic model input, image-level labels, and image ID."""

    def __init__(
        self,
        voc_root: Union[str, Path],
        list_path: Union[str, Path],
        input_size: int,
        limit: int = 0,
        transform=None,
    ):
        self.voc_root = Path(voc_root).resolve()
        self.list_path = Path(list_path).resolve()
        if not self.voc_root.is_dir():
            raise FileNotFoundError(self.voc_root)
        if not self.list_path.is_file():
            raise FileNotFoundError(self.list_path)
        if input_size < 1:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")

        image_ids = load_img_name_list(str(self.list_path))
        if limit:
            image_ids = image_ids[:limit]
        if not image_ids:
            raise ValueError(f"no image IDs found in {self.list_path}")
        self.image_ids = image_ids
        self.labels = load_image_label_list_from_npy_voc(
            str(self.voc_root), self.image_ids
        )
        self.transform = transform or build_transform(
            is_train=False,
            make_cam=False,
            args=SimpleNamespace(input_size=int(input_size)),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id = self.image_ids[index]
        image_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = PIL.Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return {"name": image_id, "image": image_tensor, "label": label}

    def __len__(self) -> int:
        return len(self.image_ids)
