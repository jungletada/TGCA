"""Tests for Experiment 2's joint deterministic RGB/mask transform."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch
from torchvision import transforms

from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (
    Experiment2JointTransform,
    VOCSemanticDataset,
    resolve_semantic_mask_path,
)
from datasets_cam import build_transform


def _synthetic_pair(height: int = 601, width: int = 799):
    generator = np.random.default_rng(20260903)
    image_array = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    mask_array[: height // 2, : width // 2] = 1
    mask_array[: height // 2, width // 2 :] = 7
    mask_array[height // 2 :, : width // 2] = 20
    mask_array[height // 2 :, width // 2 :] = 255
    return PIL.Image.fromarray(image_array, mode="RGB"), PIL.Image.fromarray(mask_array)


def test_joint_rgb_output_exactly_matches_experiment1_transform() -> None:
    image, mask = _synthetic_pair()
    joint = Experiment2JointTransform(input_size=448)
    actual_image, actual_mask, geometry = joint(image, mask)
    experiment1 = build_transform(
        is_train=False,
        make_cam=False,
        args=SimpleNamespace(input_size=448),
    )
    expected_image = experiment1(image)

    assert actual_image.shape == (3, 448, 448)
    assert actual_mask.shape == (448, 448)
    torch.testing.assert_close(actual_image, expected_image, rtol=0.0, atol=0.0)
    assert float((actual_image - expected_image).abs().max()) < 1e-6
    assert geometry["raw_size"] == [601, 799]
    assert geometry["resize_short_side"] == 512
    assert geometry["resized_size"] == [512, int(512 * 799 / 601)]
    assert geometry["crop_offset"] == [32, 116]
    assert geometry["crop_box_ltrb"] == [116, 32, 564, 480]
    # The geometry metadata must be directly recordable in run manifests.
    json.dumps(geometry, sort_keys=True)


def test_mask_uses_matching_geometry_and_nearest_neighbor_only() -> None:
    image, mask = _synthetic_pair(height=513, width=701)
    _, actual_mask, geometry = Experiment2JointTransform(input_size=448)(image, mask)
    expected_mask = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(448),
        ]
    )(mask)
    expected_tensor = torch.from_numpy(
        np.array(expected_mask, dtype=np.uint8, copy=True)
    ).long()

    torch.testing.assert_close(actual_mask, expected_tensor, rtol=0.0, atol=0.0)
    assert set(actual_mask.unique().tolist()).issubset({0, 1, 7, 20, 255})
    assert geometry["image_interpolation"] == "bicubic"
    assert geometry["mask_interpolation"] == "nearest"
    assert geometry["horizontal_flip"] is False


def test_joint_transform_rejects_misaligned_or_invalid_masks() -> None:
    image, mask = _synthetic_pair(height=64, width=80)
    transform = Experiment2JointTransform(input_size=48)
    with pytest.raises(ValueError, match="identical raw geometry"):
        transform(image, mask.crop((0, 0, 79, 64)))

    invalid = np.asarray(mask).copy()
    invalid[0, 0] = 42
    with pytest.raises(ValueError, match="invalid VOC semantic IDs"):
        transform(image, PIL.Image.fromarray(invalid))


def test_voc_semantic_dataset_exposes_joint_inputs_and_aug_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "VOC2012"
    for directory in ("JPEGImages", "SegmentationClassAug", "ImageLabel"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    image_id = "2007_000001"
    image, mask = _synthetic_pair(height=80, width=100)
    image.save(root / "JPEGImages" / f"{image_id}.jpg")
    mask.save(root / "SegmentationClassAug" / f"{image_id}.png")
    label = np.zeros(20, dtype=np.float32)
    label[[0, 6, 19]] = 1.0
    np.save(root / "ImageLabel" / "cls_labels.npy", {image_id: label})
    list_path = root / "val.txt"
    list_path.write_text(image_id + "\n", encoding="utf-8")

    assert (
        resolve_semantic_mask_path(root, image_id).parent.name == "SegmentationClassAug"
    )
    dataset = VOCSemanticDataset(root, list_path, input_size=64)
    sample = dataset[0]

    assert set(sample) == {"name", "image", "label", "mask", "mask_geometry"}
    assert sample["name"] == image_id
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == torch.int64
    torch.testing.assert_close(sample["label"], torch.from_numpy(label))
    assert sample["mask_geometry"]["raw_size"] == [80, 100]
    assert len(dataset) == 1
