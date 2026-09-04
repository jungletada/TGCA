"""Patch-level semantic-region assignment for Experiment 2.

All class indices accepted by this module are zero-based image-level indices
(``0..19``).  They are converted to native VOC semantic IDs (``1..20``) only
inside the counting helpers.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from .voc_semantic_dataset import VOC_ALLOWED_MASK_IDS, VOC_VOID_ID


REGION_TARGET = 0
REGION_OTHER_FOREGROUND = 1
REGION_BACKGROUND = 2
REGION_MIXED = 3
REGION_VOID = 4

REGION_NAME_TO_CODE: Mapping[str, int] = {
    "target": REGION_TARGET,
    "other_fg": REGION_OTHER_FOREGROUND,
    "background": REGION_BACKGROUND,
    "mixed": REGION_MIXED,
    "void": REGION_VOID,
}
REGION_CODE_TO_NAME: Mapping[int, str] = {
    code: name for name, code in REGION_NAME_TO_CODE.items()
}

PAIR_REGION_TARGET_A = 0
PAIR_REGION_TARGET_B = 1
PAIR_REGION_OTHER_FOREGROUND = 2
PAIR_REGION_BACKGROUND = 3
PAIR_REGION_MIXED = 4
PAIR_REGION_VOID = 5

PAIR_REGION_NAME_TO_CODE: Mapping[str, int] = {
    "target_a": PAIR_REGION_TARGET_A,
    "target_b": PAIR_REGION_TARGET_B,
    "other_fg": PAIR_REGION_OTHER_FOREGROUND,
    "background": PAIR_REGION_BACKGROUND,
    "mixed": PAIR_REGION_MIXED,
    "void": PAIR_REGION_VOID,
}
PAIR_REGION_CODE_TO_NAME: Mapping[int, str] = {
    code: name for name, code in PAIR_REGION_NAME_TO_CODE.items()
}

# The compact count artifact has one column for every legal non-void class ID,
# followed by the VOC void ID.  Keeping this order explicit prevents an
# off-by-one error when zero-based image-level class IDs are converted to masks.
PATCH_LABEL_IDS = tuple(range(21)) + (VOC_VOID_ID,)
PATCH_LABEL_ID_TO_COLUMN: Mapping[int, int] = {
    label_id: index for index, label_id in enumerate(PATCH_LABEL_IDS)
}


def _patch_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError(f"patch_size must contain two values, got {pair}")
    if pair[0] < 1 or pair[1] < 1:
        raise ValueError(f"patch_size values must be positive, got {pair}")
    return pair


def _validated_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.device.type != "cpu":
            mask = mask.detach().cpu()
        array = mask.numpy()
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"semantic mask must be rank 2, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"semantic mask must contain integer IDs, got {array.dtype}")
    array = array.astype(np.int16, copy=False)
    observed = {int(value) for value in np.unique(array)}
    invalid = sorted(observed.difference(VOC_ALLOWED_MASK_IDS))
    if invalid:
        raise ValueError(
            f"semantic mask contains invalid VOC IDs {invalid}; expected only "
            "0, 1..20, or 255"
        )
    return array


def _patch_pixels(
    mask: np.ndarray | torch.Tensor, patch_size: int | Sequence[int]
) -> tuple[np.ndarray, tuple[int, int]]:
    array = _validated_mask(mask)
    patch_h, patch_w = _patch_pair(patch_size)
    height, width = array.shape
    if height % patch_h or width % patch_w:
        raise ValueError(
            f"mask shape {(height, width)} is not divisible by patch size "
            f"{(patch_h, patch_w)}"
        )
    grid_h, grid_w = height // patch_h, width // patch_w
    pixels = array.reshape(grid_h, patch_h, grid_w, patch_w)
    pixels = pixels.transpose(0, 2, 1, 3).reshape(grid_h, grid_w, -1)
    return pixels, (patch_h, patch_w)


def _validate_threshold(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _unique_dominant_assignment(
    valid_count: np.ndarray,
    category_counts: np.ndarray,
    category_codes: Sequence[int],
    *,
    mixed_code: int,
    void_code: int,
    rho: float,
    min_valid_fraction: float,
    patch_area: int,
) -> np.ndarray:
    """Assign categories with a unique maximum, otherwise leave them mixed.

    An exact tie at the threshold has no unique semantic owner and is therefore
    treated as mixed rather than being resolved by arbitrary category order.
    """

    grid_shape = valid_count.shape
    codes = np.full(grid_shape, np.uint8(mixed_code), dtype=np.uint8)
    sufficiently_valid = (
        valid_count.astype(np.float64) / patch_area >= min_valid_fraction
    )
    codes[~sufficiently_valid] = np.uint8(void_code)

    safe_denominator = np.maximum(valid_count, 1)[..., None]
    fractions = category_counts.astype(np.float64) / safe_denominator
    maximum = fractions.max(axis=-1)
    maximizer_count = np.isclose(fractions, maximum[..., None], rtol=0.0, atol=0.0).sum(
        axis=-1
    )
    dominant = sufficiently_valid & (maximum >= rho) & (maximizer_count == 1)
    winning_category = fractions.argmax(axis=-1)
    for index, code in enumerate(category_codes):
        codes[dominant & (winning_category == index)] = np.uint8(code)
    return codes


def _region_masks(
    codes: np.ndarray, codebook: Mapping[str, int]
) -> dict[str, np.ndarray]:
    return {name: codes == code for name, code in codebook.items()}


def _summary(codes: np.ndarray, codebook: Mapping[str, int]) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(codes == code)) for name, code in codebook.items()
    }


def patch_label_counts(
    mask: np.ndarray | torch.Tensor,
    patch_size: int | Sequence[int] = 16,
) -> np.ndarray:
    """Return compact per-patch counts for VOC IDs ``0..20, 255``.

    The shape is ``[P, 22]`` in row-major patch order.  Columns ``0..20`` map
    directly to semantic IDs ``0..20`` and column 21 stores ID 255 (void).
    This is sufficient to re-run either class- or pair-specific region
    assignment for any ``rho`` without retaining the 448x448 semantic mask.
    """

    pixels, _ = _patch_pixels(mask, patch_size)
    flat_pixels = pixels.reshape(-1, pixels.shape[-1])
    counts = np.stack(
        [
            np.count_nonzero(flat_pixels == label_id, axis=-1)
            for label_id in PATCH_LABEL_IDS
        ],
        axis=-1,
    )
    return counts.astype(np.uint16, copy=False)


def _validated_label_counts(
    label_counts: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, int]:
    if isinstance(label_counts, torch.Tensor):
        if label_counts.device.type != "cpu":
            label_counts = label_counts.detach().cpu()
        counts = label_counts.numpy()
    else:
        counts = np.asarray(label_counts)
    if counts.ndim != 2 or counts.shape[1] != len(PATCH_LABEL_IDS):
        raise ValueError(
            "label_counts must have shape [num_patches, 22] for IDs 0..20,255; "
            f"got {counts.shape}"
        )
    if not np.issubdtype(counts.dtype, np.integer):
        raise TypeError(f"label_counts must be integers, got {counts.dtype}")
    if np.any(counts < 0):
        raise ValueError("label_counts cannot contain negative values")
    counts = counts.astype(np.int64, copy=False)
    patch_areas = counts.sum(axis=-1)
    if np.any(patch_areas <= 0) or np.any(patch_areas != patch_areas[0]):
        raise ValueError("every patch must have the same positive total pixel count")
    return counts, int(patch_areas[0])


def _output_shape(num_patches: int, grid_size: Sequence[int] | None) -> tuple[int, ...]:
    if grid_size is None:
        return (int(num_patches),)
    values = tuple(int(value) for value in grid_size)
    if len(values) != 2 or min(values) < 1:
        raise ValueError(f"grid_size must contain two positive values, got {values}")
    if values[0] * values[1] != int(num_patches):
        raise ValueError(
            f"grid_size {values} implies {values[0] * values[1]} patches, "
            f"but counts contain {num_patches}"
        )
    return values


def assign_patch_regions_from_counts(
    label_counts: np.ndarray | torch.Tensor,
    target_class_id: int,
    *,
    rho: float = 0.5,
    valid_fraction: float = 0.5,
    grid_size: Sequence[int] | None = None,
) -> dict[str, object]:
    """Assign class-specific semantic regions from compact patch counts."""

    target_class_id = int(target_class_id)
    if not 0 <= target_class_id < 20:
        raise ValueError(f"target_class_id must be in [0, 19], got {target_class_id}")
    rho = _validate_threshold("rho", rho)
    valid_fraction_threshold = _validate_threshold("valid_fraction", valid_fraction)
    counts, patch_area = _validated_label_counts(label_counts)
    shape = _output_shape(len(counts), grid_size)
    target_mask_id = target_class_id + 1

    target_count = counts[:, target_mask_id]
    foreground_count = counts[:, 1:21].sum(axis=-1)
    other_foreground_count = foreground_count - target_count
    background_count = counts[:, 0]
    void_count = counts[:, PATCH_LABEL_ID_TO_COLUMN[VOC_VOID_ID]]
    valid_count = patch_area - void_count
    category_counts = np.stack(
        [target_count, other_foreground_count, background_count], axis=-1
    )
    codes = _unique_dominant_assignment(
        valid_count,
        category_counts,
        [REGION_TARGET, REGION_OTHER_FOREGROUND, REGION_BACKGROUND],
        mixed_code=REGION_MIXED,
        void_code=REGION_VOID,
        rho=rho,
        min_valid_fraction=valid_fraction_threshold,
        patch_area=patch_area,
    ).reshape(shape)

    def shaped(values: np.ndarray, dtype=None) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(shape)

    target_count = shaped(target_count, np.uint16)
    other_foreground_count = shaped(other_foreground_count, np.uint16)
    background_count = shaped(background_count, np.uint16)
    void_count = shaped(void_count, np.uint16)
    valid_count = shaped(valid_count, np.uint16)
    safe_valid = np.maximum(valid_count, 1).astype(np.float32)
    composition = _summary(codes, REGION_NAME_TO_CODE)
    return {
        "region_codes": codes,
        "region_masks": _region_masks(codes, REGION_NAME_TO_CODE),
        "target_count": target_count,
        "other_foreground_count": other_foreground_count,
        "background_count": background_count,
        "void_count": void_count,
        "valid_count": valid_count,
        "target_fraction": target_count.astype(np.float32) / safe_valid,
        "other_foreground_fraction": other_foreground_count.astype(np.float32)
        / safe_valid,
        "background_fraction": background_count.astype(np.float32) / safe_valid,
        "void_fraction": void_count.astype(np.float32) / float(patch_area),
        "valid_fraction": valid_count.astype(np.float32) / float(patch_area),
        "composition": composition,
        "metadata": {
            "target_class_id": target_class_id,
            "target_mask_id": target_mask_id,
            "patch_area": patch_area,
            "grid_size": list(shape) if len(shape) == 2 else None,
            "rho": rho,
            "minimum_valid_fraction": valid_fraction_threshold,
            "tie_policy": "mixed",
            "region_codebook": dict(REGION_NAME_TO_CODE),
            "num_patches": int(codes.size),
            "num_patches_by_region": composition,
            "label_count_columns": list(PATCH_LABEL_IDS),
        },
    }


def assign_patch_regions(
    mask: np.ndarray | torch.Tensor,
    target_class_id: int,
    patch_size: int | Sequence[int] = 16,
    rho: float = 0.5,
    valid_fraction: float = 0.5,
) -> dict[str, object]:
    """Assign target/other-foreground/background/mixed/void patch regions.

    ``target_class_id`` is zero-based.  Returned arrays retain the two-dimensional
    patch grid.  ``metadata`` and ``composition`` contain only JSON-serializable
    values, while all dense arrays can be written directly to NPZ artifacts.
    """

    pixels, (patch_h, patch_w) = _patch_pixels(mask, patch_size)
    grid_size = pixels.shape[:2]
    result = assign_patch_regions_from_counts(
        patch_label_counts(mask, patch_size),
        target_class_id,
        rho=rho,
        valid_fraction=valid_fraction,
        grid_size=grid_size,
    )
    result["metadata"]["patch_size"] = [patch_h, patch_w]
    return result


def assign_pair_patch_regions_from_counts(
    label_counts: np.ndarray | torch.Tensor,
    target_class_a_id: int,
    target_class_b_id: int,
    *,
    rho: float = 0.5,
    valid_fraction: float = 0.5,
    grid_size: Sequence[int] | None = None,
) -> dict[str, object]:
    """Assign pair-specific semantic regions from compact patch counts."""

    class_a = int(target_class_a_id)
    class_b = int(target_class_b_id)
    if not 0 <= class_a < 20 or not 0 <= class_b < 20:
        raise ValueError(
            "target class IDs must both be in [0, 19], got "
            f"{(target_class_a_id, target_class_b_id)}"
        )
    if class_a == class_b:
        raise ValueError("pair-specific assignment requires two distinct classes")
    rho = _validate_threshold("rho", rho)
    valid_fraction_threshold = _validate_threshold("valid_fraction", valid_fraction)
    counts, patch_area = _validated_label_counts(label_counts)
    shape = _output_shape(len(counts), grid_size)
    mask_a = class_a + 1
    mask_b = class_b + 1

    target_a_count = counts[:, mask_a]
    target_b_count = counts[:, mask_b]
    foreground_count = counts[:, 1:21].sum(axis=-1)
    other_foreground_count = foreground_count - target_a_count - target_b_count
    background_count = counts[:, 0]
    void_count = counts[:, PATCH_LABEL_ID_TO_COLUMN[VOC_VOID_ID]]
    valid_count = patch_area - void_count
    category_counts = np.stack(
        [target_a_count, target_b_count, other_foreground_count, background_count],
        axis=-1,
    )
    codes = _unique_dominant_assignment(
        valid_count,
        category_counts,
        [
            PAIR_REGION_TARGET_A,
            PAIR_REGION_TARGET_B,
            PAIR_REGION_OTHER_FOREGROUND,
            PAIR_REGION_BACKGROUND,
        ],
        mixed_code=PAIR_REGION_MIXED,
        void_code=PAIR_REGION_VOID,
        rho=rho,
        min_valid_fraction=valid_fraction_threshold,
        patch_area=patch_area,
    ).reshape(shape)

    def shaped(values: np.ndarray, dtype=None) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(shape)

    target_a_count = shaped(target_a_count, np.uint16)
    target_b_count = shaped(target_b_count, np.uint16)
    other_foreground_count = shaped(other_foreground_count, np.uint16)
    background_count = shaped(background_count, np.uint16)
    void_count = shaped(void_count, np.uint16)
    valid_count = shaped(valid_count, np.uint16)
    safe_valid = np.maximum(valid_count, 1).astype(np.float32)
    composition = _summary(codes, PAIR_REGION_NAME_TO_CODE)
    return {
        "region_codes": codes,
        "region_masks": _region_masks(codes, PAIR_REGION_NAME_TO_CODE),
        "target_a_count": target_a_count,
        "target_b_count": target_b_count,
        "other_foreground_count": other_foreground_count,
        "background_count": background_count,
        "void_count": void_count,
        "valid_count": valid_count,
        "target_a_fraction": target_a_count.astype(np.float32) / safe_valid,
        "target_b_fraction": target_b_count.astype(np.float32) / safe_valid,
        "other_foreground_fraction": other_foreground_count.astype(np.float32)
        / safe_valid,
        "background_fraction": background_count.astype(np.float32) / safe_valid,
        "void_fraction": void_count.astype(np.float32) / float(patch_area),
        "valid_fraction": valid_count.astype(np.float32) / float(patch_area),
        "composition": composition,
        "metadata": {
            "target_class_a_id": class_a,
            "target_class_b_id": class_b,
            "target_mask_a_id": mask_a,
            "target_mask_b_id": mask_b,
            "patch_area": patch_area,
            "grid_size": list(shape) if len(shape) == 2 else None,
            "rho": rho,
            "minimum_valid_fraction": valid_fraction_threshold,
            "tie_policy": "mixed",
            "region_codebook": dict(PAIR_REGION_NAME_TO_CODE),
            "num_patches": int(codes.size),
            "num_patches_by_region": composition,
            "label_count_columns": list(PATCH_LABEL_IDS),
        },
    }


def assign_pair_patch_regions(
    mask: np.ndarray | torch.Tensor,
    target_class_a_id: int,
    target_class_b_id: int,
    patch_size: int | Sequence[int] = 16,
    rho: float = 0.5,
    valid_fraction: float = 0.5,
) -> dict[str, object]:
    """Assign pair-specific target-a/target-b/other/background regions."""

    pixels, (patch_h, patch_w) = _patch_pixels(mask, patch_size)
    grid_size = pixels.shape[:2]
    result = assign_pair_patch_regions_from_counts(
        patch_label_counts(mask, patch_size),
        target_class_a_id,
        target_class_b_id,
        rho=rho,
        valid_fraction=valid_fraction,
        grid_size=grid_size,
    )
    result["metadata"]["patch_size"] = [patch_h, patch_w]
    return result
