#!/usr/bin/env python3
"""Read-only provenance and input audit for Experiment 2.

This command only writes to a new output directory.  Experiment 1 results,
checkpoints, RGB images, and VOC semantic masks are treated as immutable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

# Support direct execution as documented: ``python analysis/.../audit_*.py``.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from analysis.lazy_assignment.experiment2.common import (  # noqa: E402
    LOW_LEVEL_SOURCE_PATHS,
    VOC_CLASS_NAMES,
    assert_output_outside_inputs,
    csv_dump,
    git_blob,
    git_metadata,
    json_dump,
    read_json,
    resolve_completed_experiment1_root,
    resolve_completed_paired_analysis_root,
    sha256_bytes,
    sha256_file,
    timestamp,
)


NPZ_REQUIRED_KEYS = {
    "image_id",
    "positive_class_ids",
    "saved_class_ids",
    "scores_raw",
    "grid_h",
    "grid_w",
}
EXPECTED_TRANSFORM = (
    "bicubic Resize(int(256/224*input_size)) -> CenterCrop(input_size) -> "
    "ToTensor -> ImageNet Normalize"
)
FILE_MANIFEST_FIELDS = (
    "source_group",
    "model",
    "root",
    "relative_path",
    "absolute_path",
    "size_bytes",
    "mtime_ns",
    "sha256",
)
GT_MANIFEST_FIELDS = (
    "image_id",
    "jpeg_path",
    "jpeg_size_bytes",
    "jpeg_sha256",
    "mask_path",
    "mask_size_bytes",
    "mask_sha256",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "mask_mode",
    "raw_mask_values",
    "image_positive_class_ids",
    "raw_mask_foreground_class_ids",
    "raw_mask_only_class_ids",
    "raw_label_only_class_ids",
    "raw_class_sets_match",
    "crop_width",
    "crop_height",
    "cropped_mask_values",
    "cropped_mask_foreground_class_ids",
    "cropped_mask_only_class_ids",
    "cropped_label_only_class_ids",
    "cropped_class_sets_match",
    "num_positive_classes",
    "num_target_absent_after_crop",
    "num_without_target_patch_rho05",
    "num_without_target_patch_rho07",
)


@dataclass(frozen=True)
class AuditConfig:
    repo_root: Path
    voc_root: Path
    val_list: Path
    output_dir: Path
    mctformer_search_root: Path
    mctformer_plus_search_root: Path
    paired_analysis_search_root: Path
    mctformer_result_root: Optional[Path] = None
    mctformer_plus_result_root: Optional[Path] = None
    paired_analysis_root: Optional[Path] = None
    resize_size: int = 512
    crop_size: int = 448
    expected_layers: int = 12
    expected_grid_size: int = 28


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    repo = default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--voc-root", type=Path)
    parser.add_argument("--val-list", type=Path)
    parser.add_argument("--mctformer-search-root", type=Path)
    parser.add_argument("--mctformer-result-root", type=Path)
    parser.add_argument("--mctformer-plus-search-root", type=Path)
    parser.add_argument("--mctformer-plus-result-root", type=Path)
    parser.add_argument("--paired-analysis-search-root", type=Path)
    parser.add_argument("--paired-analysis-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=448)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    repo_root = args.repo_root.expanduser().resolve()
    voc_root = (
        args.voc_root.expanduser().resolve()
        if args.voc_root is not None
        else (repo_root / "data/VOCdevkit/VOC2012").resolve()
    )
    val_list = (
        args.val_list.expanduser().resolve()
        if args.val_list is not None
        else voc_root / "ImageLists/val_id.txt"
    )
    return AuditConfig(
        repo_root=repo_root,
        voc_root=voc_root,
        val_list=val_list,
        output_dir=args.output_dir.expanduser().resolve(),
        mctformer_search_root=(
            args.mctformer_search_root
            if args.mctformer_search_root is not None
            else repo_root
            / "results/lazy_assignment/experiment1_class_patch_score/mctformer"
        ),
        mctformer_plus_search_root=(
            args.mctformer_plus_search_root
            if args.mctformer_plus_search_root is not None
            else repo_root
            / "results/lazy_assignment/experiment1_class_patch_score/mctformer_plus"
        ),
        paired_analysis_search_root=(
            args.paired_analysis_search_root
            if args.paired_analysis_search_root is not None
            else repo_root / "results/lazy_assignment/experiment1_analysis"
        ),
        mctformer_result_root=args.mctformer_result_root,
        mctformer_plus_result_root=args.mctformer_plus_result_root,
        paired_analysis_root=args.paired_analysis_root,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
    )


def _issue(
    issues: list[dict[str, object]],
    severity: str,
    code: str,
    details: object,
    path: Optional[Path] = None,
    image_id: str = "",
    model: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "details": str(details),
            "path": str(path) if path is not None else "",
            "image_id": image_id,
            "model": model,
        }
    )


def _load_val_ids(path: Path, issues: list[dict[str, object]]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicates = sorted(key for key, count in Counter(image_ids).items() if count > 1)
    if duplicates:
        _issue(issues, "error", "duplicate_val_ids", duplicates, path)
    if not image_ids:
        _issue(issues, "error", "empty_val_list", "no image IDs", path)
    return image_ids


def _load_labels(
    path: Path, image_ids: list[str], issues: list[dict[str, object]]
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"expected label dictionary in {path}")
    labels: dict[str, np.ndarray] = {}
    for image_id in image_ids:
        if image_id not in payload:
            _issue(
                issues,
                "error",
                "missing_image_label",
                image_id,
                path,
                image_id=image_id,
            )
            continue
        label = np.asarray(payload[image_id])
        if label.shape != (len(VOC_CLASS_NAMES),):
            _issue(
                issues,
                "error",
                "invalid_label_shape",
                label.shape,
                path,
                image_id=image_id,
            )
        if not np.all(np.isin(label, (0, 1))):
            _issue(
                issues,
                "error",
                "invalid_image_label_values",
                np.unique(label).tolist(),
                path,
                image_id=image_id,
            )
        labels[image_id] = label
    return labels


def _metadata_checkpoint_path(metadata: dict[str, Any], repo_root: Path) -> Path:
    raw = metadata.get("checkpoint", {}).get("path")
    if not isinstance(raw, str) or not raw:
        raise KeyError("metadata.checkpoint.path is absent")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_manifest(
    path: Path, issues: list[dict[str, object]], model: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        _issue(issues, "error", "missing_experiment1_manifest", "", path, model=model)
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            _issue(
                issues,
                "error",
                "invalid_manifest_json",
                f"line {line_number}: {error}",
                path,
                model=model,
            )
            continue
        if not isinstance(value, dict):
            _issue(
                issues,
                "error",
                "manifest_row_not_object",
                line_number,
                path,
                model=model,
            )
            continue
        records.append(value)
    return records


def audit_experiment1_result(
    model: str,
    root: Path,
    expected_ids: list[str],
    labels: dict[str, np.ndarray],
    config: AuditConfig,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    metadata_path = root / "metadata.json"
    completion_path = root / "completion.json"
    metadata = read_json(metadata_path)
    completion = read_json(completion_path)
    records = _read_manifest(root / "manifest.jsonl", issues, model)
    manifest_ids = [str(record.get("image_id", "")) for record in records]
    if manifest_ids != expected_ids:
        _issue(
            issues,
            "error",
            "manifest_order_or_membership_mismatch",
            {
                "rows": len(manifest_ids),
                "expected_rows": len(expected_ids),
                "same_set": set(manifest_ids) == set(expected_ids),
            },
            root / "manifest.jsonl",
            model=model,
        )
    duplicate_ids = sorted(
        key for key, value in Counter(manifest_ids).items() if value > 1
    )
    if duplicate_ids:
        _issue(
            issues, "error", "duplicate_manifest_ids", duplicate_ids, root, model=model
        )

    actual_score_paths = {path.resolve() for path in (root / "scores").glob("*.npz")}
    recorded_score_paths: set[Path] = set()
    valid_npz = 0
    image_class_pairs = 0
    schema_distribution: Counter[tuple[str, ...]] = Counter()
    dtype_distribution: Counter[str] = Counter()
    for record in records:
        image_id = str(record.get("image_id", ""))
        relative_score = record.get("score_path")
        if not isinstance(relative_score, str):
            _issue(issues, "error", "missing_score_path", record, root, image_id, model)
            continue
        score_path = (root / relative_score).resolve()
        recorded_score_paths.add(score_path)
        if not score_path.is_file():
            _issue(
                issues, "error", "missing_score_npz", "", score_path, image_id, model
            )
            continue
        expected_positive = (
            np.flatnonzero(labels[image_id] > 0).astype(np.int64)
            if image_id in labels
            else np.empty(0, dtype=np.int64)
        )
        record_positive = np.asarray(
            record.get("positive_class_ids", []), dtype=np.int64
        )
        record_saved = np.asarray(record.get("saved_class_ids", []), dtype=np.int64)
        if not np.array_equal(record_positive, expected_positive) or not np.array_equal(
            record_saved, expected_positive
        ):
            _issue(
                issues,
                "error",
                "manifest_positive_ids_mismatch",
                {
                    "manifest": record_positive.tolist(),
                    "saved": record_saved.tolist(),
                    "expected": expected_positive.tolist(),
                },
                root / "manifest.jsonl",
                image_id,
                model,
            )
        try:
            with np.load(score_path, allow_pickle=False) as artifact:
                keys = set(artifact.files)
                schema_distribution[tuple(sorted(keys))] += 1
                if not NPZ_REQUIRED_KEYS.issubset(keys):
                    raise KeyError(f"keys={sorted(keys)}")
                stored_id = str(artifact["image_id"].item())
                positive = np.asarray(
                    artifact["positive_class_ids"], dtype=np.int64
                ).reshape(-1)
                saved = np.asarray(artifact["saved_class_ids"], dtype=np.int64).reshape(
                    -1
                )
                scores = np.asarray(artifact["scores_raw"])
                grid_h = int(artifact["grid_h"].item())
                grid_w = int(artifact["grid_w"].item())
        except Exception as error:
            _issue(
                issues,
                "error",
                "npz_read_or_schema_error",
                repr(error),
                score_path,
                image_id,
                model,
            )
            continue
        expected_shape = (
            config.expected_layers,
            len(expected_positive),
            config.expected_grid_size * config.expected_grid_size,
        )
        if stored_id != image_id:
            _issue(
                issues,
                "error",
                "npz_image_id_mismatch",
                stored_id,
                score_path,
                image_id,
                model,
            )
        if not np.array_equal(positive, expected_positive) or not np.array_equal(
            saved, expected_positive
        ):
            _issue(
                issues,
                "error",
                "npz_positive_ids_mismatch",
                {
                    "positive": positive.tolist(),
                    "saved": saved.tolist(),
                    "expected": expected_positive.tolist(),
                },
                score_path,
                image_id,
                model,
            )
        if scores.shape != expected_shape:
            _issue(
                issues,
                "error",
                "npz_score_shape_mismatch",
                scores.shape,
                score_path,
                image_id,
                model,
            )
        if (grid_h, grid_w) != (config.expected_grid_size, config.expected_grid_size):
            _issue(
                issues,
                "error",
                "npz_grid_mismatch",
                (grid_h, grid_w),
                score_path,
                image_id,
                model,
            )
        if not np.isfinite(scores).all():
            _issue(
                issues, "error", "npz_nonfinite_scores", "", score_path, image_id, model
            )
        if scores.size and (scores.min() < -1.00001 or scores.max() > 1.00001):
            _issue(
                issues,
                "error",
                "npz_cosine_range_violation",
                (float(scores.min()), float(scores.max())),
                score_path,
                image_id,
                model,
            )
        dtype_distribution[str(scores.dtype)] += 1
        image_class_pairs += len(positive)
        valid_npz += 1

    missing_recorded = sorted(
        str(path) for path in recorded_score_paths - actual_score_paths
    )
    orphan_scores = sorted(
        str(path) for path in actual_score_paths - recorded_score_paths
    )
    if missing_recorded:
        _issue(
            issues, "error", "manifest_npz_missing", missing_recorded, root, model=model
        )
    if orphan_scores:
        _issue(issues, "error", "orphan_score_npz", orphan_scores, root, model=model)

    actual_list_hash = sha256_file(config.val_list)
    actual_labels_hash = sha256_file(config.voc_root / "ImageLabel/cls_labels.npy")
    metadata_checks = {
        "completion_status": completion.get("status") == "complete",
        "run_kind_full": completion.get("run_kind") == "full",
        "val_list_sha256": metadata.get("dataset", {}).get("list_sha256")
        == actual_list_hash,
        "labels_sha256": metadata.get("dataset", {}).get("labels_sha256")
        == actual_labels_hash,
        "input_size_448": metadata.get("input", {}).get("size") == config.crop_size,
        "single_scale": metadata.get("input", {}).get("scale") == 1.0,
        "no_horizontal_flip": metadata.get("input", {}).get("horizontal_flip") is False,
        "deterministic_transform": metadata.get("input", {}).get("transform")
        == EXPECTED_TRANSFORM,
        "post_block_representation": metadata.get("representation")
        == "post_block_pre_final_norm",
        "positive_class_filter": metadata.get("positive_class_filter") is True,
    }
    for name, passed in metadata_checks.items():
        if not passed:
            _issue(
                issues,
                "error",
                "upstream_metadata_mismatch",
                name,
                metadata_path,
                model=model,
            )
    return {
        "root": str(root),
        "metadata_path": str(metadata_path),
        "completion_path": str(completion_path),
        "manifest_rows": len(records),
        "unique_manifest_ids": len(set(manifest_ids)),
        "manifest_order_matches_val": manifest_ids == expected_ids,
        "score_npz_files": len(actual_score_paths),
        "valid_score_npz_files": valid_npz,
        "image_class_pairs": image_class_pairs,
        "schema_distribution": {
            "|".join(key): value for key, value in schema_distribution.items()
        },
        "dtype_distribution": dict(dtype_distribution),
        "metadata_checks": metadata_checks,
        "metadata": metadata,
        "completion": completion,
    }


def _target_patch_visibility(mask: np.ndarray, class_id: int, rho: float) -> bool:
    patch_size = 16
    grid = mask.shape[0] // patch_size
    patches = (
        mask.reshape(grid, patch_size, grid, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(grid * grid, patch_size * patch_size)
    )
    valid = patches != 255
    valid_count = valid.sum(axis=1)
    target = ((patches == class_id + 1) & valid).sum(axis=1)
    other = (
        (patches >= 1)
        & (patches <= len(VOC_CLASS_NAMES))
        & (patches != class_id + 1)
        & valid
    ).sum(axis=1)
    background = ((patches == 0) & valid).sum(axis=1)
    denominator = np.maximum(valid_count, 1)
    target_fraction = target / denominator
    target_is_dominant = (target >= other) & (target >= background)
    eligible = valid_count >= patch_size * patch_size * 0.5
    return bool(np.any(eligible & target_is_dominant & (target_fraction >= rho)))


def audit_gt(
    config: AuditConfig,
    image_ids: list[str],
    labels: dict[str, np.ndarray],
    hash_by_path: dict[Path, str],
    issues: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    allowed = set(range(len(VOC_CLASS_NAMES) + 1)) | {255}
    raw_mismatches: list[dict[str, object]] = []
    cropped_mismatch_images = 0
    target_pairs = 0
    target_absent_pairs = 0
    without_rho05 = 0
    without_rho07 = 0
    masks_with_void = 0
    for image_id in image_ids:
        jpeg_path = (config.voc_root / "JPEGImages" / f"{image_id}.jpg").resolve()
        mask_path = (
            config.voc_root / "SegmentationClass" / f"{image_id}.png"
        ).resolve()
        if not jpeg_path.is_file():
            _issue(issues, "error", "missing_jpeg", "", jpeg_path, image_id=image_id)
            continue
        if not mask_path.is_file():
            _issue(
                issues,
                "error",
                "missing_semantic_mask",
                "",
                mask_path,
                image_id=image_id,
            )
            continue
        try:
            with Image.open(jpeg_path) as image:
                image.load()
                image_width, image_height = image.size
            with Image.open(mask_path) as source_mask:
                source_mask.load()
                mask_width, mask_height = source_mask.size
                mask_mode = source_mask.mode
                raw_mask = np.asarray(source_mask)
                cropped_pil = transform_functional.center_crop(
                    transform_functional.resize(
                        source_mask,
                        config.resize_size,
                        interpolation=InterpolationMode.NEAREST,
                    ),
                    [config.crop_size, config.crop_size],
                )
                cropped_mask = np.asarray(cropped_pil)
        except Exception as error:
            _issue(
                issues,
                "error",
                "image_or_mask_decode_error",
                repr(error),
                mask_path,
                image_id,
            )
            continue
        if (image_width, image_height) != (mask_width, mask_height):
            _issue(
                issues,
                "error",
                "image_mask_dimension_mismatch",
                ((image_width, image_height), (mask_width, mask_height)),
                mask_path,
                image_id,
            )
        raw_values = set(int(value) for value in np.unique(raw_mask))
        cropped_values = set(int(value) for value in np.unique(cropped_mask))
        illegal_raw = sorted(raw_values - allowed)
        illegal_cropped = sorted(cropped_values - allowed)
        if illegal_raw or illegal_cropped:
            _issue(
                issues,
                "error",
                "illegal_semantic_mask_labels",
                {"raw": illegal_raw, "cropped": illegal_cropped},
                mask_path,
                image_id,
            )
        if 255 in raw_values:
            masks_with_void += 1
        positive = set(np.flatnonzero(labels[image_id] > 0).astype(int).tolist())
        raw_foreground = {
            value - 1 for value in raw_values if 1 <= value <= len(VOC_CLASS_NAMES)
        }
        cropped_foreground = {
            value - 1 for value in cropped_values if 1 <= value <= len(VOC_CLASS_NAMES)
        }
        raw_mask_only = sorted(raw_foreground - positive)
        raw_label_only = sorted(positive - raw_foreground)
        cropped_mask_only = sorted(cropped_foreground - positive)
        cropped_label_only = sorted(positive - cropped_foreground)
        if raw_mask_only or raw_label_only:
            mismatch = {
                "image_id": image_id,
                "mask_only_class_ids": raw_mask_only,
                "label_only_class_ids": raw_label_only,
            }
            raw_mismatches.append(mismatch)
            _issue(
                issues,
                "warning",
                "mask_image_label_class_set_mismatch",
                mismatch,
                mask_path,
                image_id,
            )
        if cropped_mask_only or cropped_label_only:
            cropped_mismatch_images += 1
        absent_here = 0
        no_rho05_here = 0
        no_rho07_here = 0
        for class_id in sorted(positive):
            target_pairs += 1
            if not np.any(cropped_mask == class_id + 1):
                target_absent_pairs += 1
                absent_here += 1
            if not _target_patch_visibility(cropped_mask, class_id, 0.5):
                without_rho05 += 1
                no_rho05_here += 1
            if not _target_patch_visibility(cropped_mask, class_id, 0.7):
                without_rho07 += 1
                no_rho07_here += 1
        rows.append(
            {
                "image_id": image_id,
                "jpeg_path": str(jpeg_path),
                "jpeg_size_bytes": jpeg_path.stat().st_size,
                "jpeg_sha256": hash_by_path[jpeg_path],
                "mask_path": str(mask_path),
                "mask_size_bytes": mask_path.stat().st_size,
                "mask_sha256": hash_by_path[mask_path],
                "image_width": image_width,
                "image_height": image_height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_mode": mask_mode,
                "raw_mask_values": json.dumps(sorted(raw_values)),
                "image_positive_class_ids": json.dumps(sorted(positive)),
                "raw_mask_foreground_class_ids": json.dumps(sorted(raw_foreground)),
                "raw_mask_only_class_ids": json.dumps(raw_mask_only),
                "raw_label_only_class_ids": json.dumps(raw_label_only),
                "raw_class_sets_match": not raw_mask_only and not raw_label_only,
                "crop_width": cropped_mask.shape[1],
                "crop_height": cropped_mask.shape[0],
                "cropped_mask_values": json.dumps(sorted(cropped_values)),
                "cropped_mask_foreground_class_ids": json.dumps(
                    sorted(cropped_foreground)
                ),
                "cropped_mask_only_class_ids": json.dumps(cropped_mask_only),
                "cropped_label_only_class_ids": json.dumps(cropped_label_only),
                "cropped_class_sets_match": not cropped_mask_only
                and not cropped_label_only,
                "num_positive_classes": len(positive),
                "num_target_absent_after_crop": absent_here,
                "num_without_target_patch_rho05": no_rho05_here,
                "num_without_target_patch_rho07": no_rho07_here,
            }
        )
    summary = {
        "expected_images": len(image_ids),
        "audited_images": len(rows),
        "masks_with_void": masks_with_void,
        "raw_mask_image_label_mismatch_count": len(raw_mismatches),
        "raw_mask_image_label_mismatches": raw_mismatches,
        "cropped_mask_image_label_mismatch_count": cropped_mismatch_images,
        "positive_image_class_pairs": target_pairs,
        "positive_pairs_with_no_target_pixels_after_crop": target_absent_pairs,
        "positive_pairs_without_target_dominant_patch_rho05": without_rho05,
        "positive_pairs_without_target_dominant_patch_rho07": without_rho07,
        "target_absence_is_integrity_failure": False,
        "empty_target_auc_policy": "record AUROC/AUPRC as NA; retain full-set ownership metrics and report a target-visible subset",
        "class_mapping": "image-level class_id 0..19 maps to VOC mask value class_id+1",
        "mask_values": "0=background, 1..20=foreground, 255=void",
        "transform": {
            "image": "scalar Resize(512, bicubic) -> CenterCrop(448) -> ToTensor -> ImageNet Normalize",
            "mask": "same scalar resize/crop geometry with nearest-neighbor interpolation",
            "resize_size": config.resize_size,
            "crop_size": config.crop_size,
            "patch_grid": f"{config.expected_grid_size}x{config.expected_grid_size}",
        },
    }
    return rows, summary


def _manifest_row(
    source_group: str, model: str, root: Path, path: Path
) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError:
        relative = Path(resolved.name)
    return {
        "source_group": source_group,
        "model": model,
        "root": str(root.resolve()),
        "relative_path": str(relative),
        "absolute_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def build_before_manifest(
    config: AuditConfig,
    source_roots: dict[str, Path],
    paired_root: Path,
    checkpoints: dict[str, Path],
    image_ids: list[str],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, root in source_roots.items():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(_manifest_row("experiment1_result", model, root, path))
    for path in sorted(item for item in paired_root.rglob("*") if item.is_file()):
        rows.append(
            _manifest_row("experiment1_paired_analysis", "paired", paired_root, path)
        )
    for model, path in checkpoints.items():
        if path.is_file():
            rows.append(_manifest_row("checkpoint", model, path.parent, path))
        else:
            _issue(issues, "error", "missing_checkpoint", "", path, model=model)
    for image_id in image_ids:
        for group, relative in (
            ("voc_jpeg", Path("JPEGImages") / f"{image_id}.jpg"),
            ("voc_semantic_mask", Path("SegmentationClass") / f"{image_id}.png"),
        ):
            path = config.voc_root / relative
            if path.is_file():
                rows.append(_manifest_row(group, "", config.voc_root, path))
            else:
                _issue(issues, "error", f"missing_{group}", "", path, image_id=image_id)
    for group, path in (
        ("voc_val_list", config.val_list),
        ("voc_image_labels", config.voc_root / "ImageLabel/cls_labels.npy"),
    ):
        if path.is_file():
            rows.append(_manifest_row(group, "", config.voc_root, path))
    for relative in LOW_LEVEL_SOURCE_PATHS:
        path = config.repo_root / relative
        if path.is_file():
            rows.append(_manifest_row("runtime_source", "", config.repo_root, path))
        else:
            _issue(issues, "error", "missing_low_level_source", relative, path)
    rows.sort(
        key=lambda row: (
            str(row["source_group"]),
            str(row["model"]),
            str(row["absolute_path"]),
        )
    )
    return rows


def verify_checkpoints(
    metadata_by_model: dict[str, dict[str, Any]],
    paired_metadata: dict[str, Any],
    config: AuditConfig,
    issues: list[dict[str, object]],
) -> tuple[dict[str, Path], dict[str, object]]:
    paths: dict[str, Path] = {}
    results: dict[str, object] = {}
    paired_hashes = paired_metadata.get("source_checkpoints", {})
    for model, metadata in metadata_by_model.items():
        try:
            path = _metadata_checkpoint_path(metadata, config.repo_root)
        except KeyError as error:
            _issue(
                issues,
                "error",
                "checkpoint_path_missing_from_metadata",
                error,
                model=model,
            )
            continue
        paths[model] = path
        expected = metadata.get("checkpoint", {}).get("sha256")
        actual = sha256_file(path) if path.is_file() else None
        paired_expected = paired_hashes.get(model)
        matches = bool(actual and actual == expected and paired_expected == expected)
        if not matches:
            _issue(
                issues,
                "error",
                "checkpoint_hash_or_linkage_mismatch",
                {"actual": actual, "experiment1": expected, "paired": paired_expected},
                path,
                model=model,
            )
        stat = path.stat() if path.is_file() else None
        results[model] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": stat.st_size if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else None,
            "sha256_expected_experiment1": expected,
            "sha256_expected_paired_analysis": paired_expected,
            "sha256_actual": actual,
            "all_hashes_match": matches,
            "checkpoint_epoch_from_metadata": metadata.get("checkpoint", {}).get(
                "epoch"
            ),
            "strict_load_from_experiment1": metadata.get("checkpoint", {}).get(
                "strict_load"
            ),
        }
    return paths, {
        "generated_at": timestamp(),
        "models": results,
        "passed": len(results) == 2
        and all(value["all_hashes_match"] for value in results.values()),
    }


def audit_source_provenance(
    config: AuditConfig,
    metadata_by_model: dict[str, dict[str, Any]],
    issues: list[dict[str, object]],
) -> dict[str, object]:
    low_level: dict[str, object] = {}
    commits = {
        model: str(metadata.get("git", {}).get("commit", ""))
        for model, metadata in metadata_by_model.items()
    }
    for relative in LOW_LEVEL_SOURCE_PATHS:
        current_path = config.repo_root / relative
        current_hash = sha256_file(current_path) if current_path.is_file() else None
        by_commit: dict[str, object] = {}
        for model, commit in commits.items():
            blob = git_blob(config.repo_root, commit, relative)
            recorded_hash = sha256_bytes(blob) if blob is not None else None
            matches = (
                recorded_hash == current_hash if recorded_hash is not None else None
            )
            by_commit[model] = {
                "commit": commit,
                "git_blob_available": blob is not None,
                "sha256_at_commit": recorded_hash,
                "matches_current": matches,
            }
            if matches is False:
                _issue(
                    issues,
                    "error",
                    "low_level_source_drift",
                    {
                        "commit": commit,
                        "sha256_at_commit": recorded_hash,
                        "current": current_hash,
                    },
                    current_path,
                    model=model,
                )
        low_level[relative] = {
            "current_path": str(current_path.resolve()),
            "current_sha256": current_hash,
            "experiment1_commits": by_commit,
            "explicitly_hashed_in_experiment1_runtime_metadata": {
                model: relative
                in metadata.get("git", {}).get("runtime_source_sha256", {})
                for model, metadata in metadata_by_model.items()
            },
        }

    runtime_sources: dict[str, object] = {}
    for model, metadata in metadata_by_model.items():
        checks: dict[str, object] = {}
        for relative, expected_hash in (
            metadata.get("git", {}).get("runtime_source_sha256", {}).items()
        ):
            path = config.repo_root / relative
            actual_hash = sha256_file(path) if path.is_file() else None
            matches = actual_hash == expected_hash
            checks[relative] = {
                "path": str(path.resolve()),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "matches": matches,
            }
            if not matches:
                _issue(
                    issues,
                    "error",
                    "experiment1_runtime_source_drift",
                    checks[relative],
                    path,
                    model=model,
                )
        runtime_sources[model] = checks
    return {
        "low_level_sources": low_level,
        "experiment1_runtime_sources": runtime_sources,
        "note": "Experiment 1 metadata did not explicitly hash every low-level dependency; recorded Git commits are used to recover and compare those blobs.",
    }


def _linkage_checks(
    source_roots: dict[str, Path],
    result_summaries: dict[str, dict[str, object]],
    paired_root: Path,
    paired_metadata: dict[str, Any],
    config: AuditConfig,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    recorded_roots = {
        model: Path(value).expanduser().resolve() if isinstance(value, str) else None
        for model, value in paired_metadata.get("source_roots", {}).items()
    }
    source_root_match = {
        model: recorded_roots.get(model) == root.resolve()
        for model, root in source_roots.items()
    }
    common_ids = all(
        summary["manifest_order_matches_val"] for summary in result_summaries.values()
    )
    pair_counts = {
        model: summary["image_class_pairs"]
        for model, summary in result_summaries.items()
    }
    checks = {
        "paired_analysis_status_complete": paired_metadata.get("status") == "complete",
        "paired_analysis_integrity_passed": paired_metadata.get(
            "source_immutability_passed"
        )
        is True,
        "source_root_match": all(source_root_match.values()),
        "both_manifest_orders_match_val": common_ids,
        "image_class_pair_counts_match": len(set(pair_counts.values())) == 1,
        "val_list_sha256_match_between_models": len(
            {
                summary["metadata"].get("dataset", {}).get("list_sha256")
                for summary in result_summaries.values()
            }
        )
        == 1,
        "transform_match_between_models": len(
            {
                summary["metadata"].get("input", {}).get("transform")
                for summary in result_summaries.values()
            }
        )
        == 1,
    }
    for name, passed in checks.items():
        if not passed:
            _issue(issues, "error", "experiment1_linkage_failure", name, paired_root)
    return {
        "paired_analysis_root": str(paired_root),
        "paired_run_metadata_path": str(paired_root / "run_metadata.json"),
        "recorded_source_roots": {
            model: str(value) if value is not None else None
            for model, value in recorded_roots.items()
        },
        "resolved_source_roots": {
            model: str(root) for model, root in source_roots.items()
        },
        "source_root_match_by_model": source_root_match,
        "image_class_pair_counts": pair_counts,
        "checks": checks,
        "expected_transform": EXPECTED_TRANSFORM,
        "actual_val_list_sha256": sha256_file(config.val_list),
    }


def _markdown_report(
    source_roots: dict[str, Path],
    paired_root: Path,
    result_summaries: dict[str, dict[str, object]],
    checkpoint_report: dict[str, object],
    gt_summary: dict[str, object],
    manifest_rows: list[dict[str, object]],
    issues: list[dict[str, object]],
    passed: bool,
) -> str:
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    lines = [
        "# Experiment 2 Input Audit",
        "",
        f"Generated: `{timestamp()}`",
        "",
        f"Overall integrity: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Immutable upstream inputs",
        "",
        "| Input | Resolved path | Files / pairs |",
        "|---|---|---:|",
    ]
    for model, root in source_roots.items():
        summary = result_summaries[model]
        lines.append(
            f"| {model} Experiment 1 | `{root}` | {summary['score_npz_files']} NPZ / {summary['image_class_pairs']} pairs |"
        )
    lines.extend(
        [
            f"| paired Experiment 1 analysis | `{paired_root}` | complete |",
            "",
            "The before-analysis manifest contains "
            f"**{len(manifest_rows):,} files** and full SHA-256 values. No upstream file was written.",
            "",
            "## Checkpoints",
            "",
            "| Model | Path | Bytes | SHA-256 match |",
            "|---|---|---:|---:|",
        ]
    )
    for model, value in checkpoint_report["models"].items():
        lines.append(
            f"| {model} | `{value['path']}` | {value['size_bytes']} | {value['all_hashes_match']} |"
        )
    lines.extend(
        [
            "",
            "## VOC semantic GT",
            "",
            f"- Val images audited: **{gt_summary['audited_images']}/{gt_summary['expected_images']}**.",
            f"- Positive image-class pairs: **{gt_summary['positive_image_class_pairs']}**.",
            f"- Raw mask/ImageLabel class-set mismatches: **{gt_summary['raw_mask_image_label_mismatch_count']}**.",
            f"- Positive pairs with no target pixels after the matched center crop: **{gt_summary['positive_pairs_with_no_target_pixels_after_crop']}**.",
            f"- Positive pairs without a target-dominant patch at rho=0.5 / 0.7: **{gt_summary['positive_pairs_without_target_dominant_patch_rho05']} / {gt_summary['positive_pairs_without_target_dominant_patch_rho07']}**.",
            "",
            "Target absence after the deterministic crop is a documented analysis stratum, not an integrity failure. Target-vs-background/other AUC and AP must be NA when a comparison set is empty.",
            "",
            "## Issues",
            "",
            f"- Errors: **{len(errors)}**",
            f"- Warnings: **{len(warnings)}**",
        ]
    )
    if issues:
        lines.extend(
            ["", "| Severity | Code | Image/model | Details |", "|---|---|---|---|"]
        )
        for issue in issues[:100]:
            owner = issue["image_id"] or issue["model"] or "-"
            details = str(issue["details"]).replace("|", "\\|")
            lines.append(
                f"| {issue['severity']} | {issue['code']} | {owner} | {details} |"
            )
        if len(issues) > 100:
            lines.append(
                f"| info | truncated | - | {len(issues) - 100} additional issues in source_metadata.json |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This audit establishes input identity, geometry, and provenance only. It does not establish semantic leakage, attention behavior, CAM behavior, or causality.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(config: AuditConfig) -> dict[str, object]:
    if config.resize_size < config.crop_size:
        raise ValueError("resize_size must be at least crop_size")
    if config.crop_size % 16:
        raise ValueError("crop_size must be divisible by the 16-pixel patch size")
    if config.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing audit output: {config.output_dir}"
        )

    issues: list[dict[str, object]] = []
    mctformer_discovery = resolve_completed_experiment1_root(
        config.mctformer_search_root, "mctformer", config.mctformer_result_root
    )
    plus_discovery = resolve_completed_experiment1_root(
        config.mctformer_plus_search_root,
        "mctformer_plus",
        config.mctformer_plus_result_root,
    )
    source_roots = {
        "mctformer": mctformer_discovery.root,
        "mctformer_plus": plus_discovery.root,
    }
    paired_discovery = resolve_completed_paired_analysis_root(
        config.paired_analysis_search_root,
        source_roots,
        config.paired_analysis_root,
    )
    paired_root = paired_discovery.root
    assert_output_outside_inputs(
        config.output_dir,
        [*source_roots.values(), paired_root, config.voc_root],
    )

    image_ids = _load_val_ids(config.val_list, issues)
    labels_path = config.voc_root / "ImageLabel/cls_labels.npy"
    labels = _load_labels(labels_path, image_ids, issues)
    metadata_by_model = {
        model: read_json(root / "metadata.json") for model, root in source_roots.items()
    }
    paired_metadata = read_json(paired_root / "run_metadata.json")
    checkpoint_paths, checkpoint_report = verify_checkpoints(
        metadata_by_model, paired_metadata, config, issues
    )

    before_manifest = build_before_manifest(
        config, source_roots, paired_root, checkpoint_paths, image_ids, issues
    )
    hash_by_path = {
        Path(str(row["absolute_path"])): str(row["sha256"]) for row in before_manifest
    }
    result_summaries = {
        model: audit_experiment1_result(model, root, image_ids, labels, config, issues)
        for model, root in source_roots.items()
    }
    gt_rows, gt_summary = audit_gt(config, image_ids, labels, hash_by_path, issues)
    provenance = audit_source_provenance(config, metadata_by_model, issues)
    linkage = _linkage_checks(
        source_roots,
        result_summaries,
        paired_root,
        paired_metadata,
        config,
        issues,
    )
    linkage.update(
        {
            "generated_at": timestamp(),
            "discovery": {
                "mctformer": list(mctformer_discovery.inspected),
                "mctformer_plus": list(plus_discovery.inspected),
                "paired_analysis": list(paired_discovery.inspected),
            },
            "result_audits": {
                model: {
                    key: value
                    for key, value in summary.items()
                    if key not in {"metadata", "completion"}
                }
                for model, summary in result_summaries.items()
            },
            "source_provenance": provenance,
            "sources": {
                model: {
                    "result_root": str(source_roots[model]),
                    "model_cli_name": (
                        "mctformerv2" if model == "mctformer" else "mctformerplus"
                    ),
                    "checkpoint": {
                        "path": checkpoint_report["models"][model]["path"],
                        "sha256": checkpoint_report["models"][model]["sha256_actual"],
                    },
                }
                for model in ("mctformer", "mctformer_plus")
            },
            "dataset": {
                "voc_root": str(config.voc_root),
                "list_path": str(config.val_list),
                "labels_path": str(labels_path),
                "input_size": config.crop_size,
                "patch_size": 16,
                "num_images": len(image_ids),
            },
        }
    )
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    passed = not errors
    linkage["integrity_passed"] = passed
    linkage["error_count"] = len(errors)
    linkage["warning_count"] = len(warnings)

    source_metadata: dict[str, object] = {
        "generated_at": timestamp(),
        "status": "complete" if passed else "failed",
        "integrity_passed": passed,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": issues,
        "repo": git_metadata(config.repo_root),
        "sources": {
            model: {
                "result_root": str(source_roots[model]),
                "model_cli_name": (
                    "mctformerv2" if model == "mctformer" else "mctformerplus"
                ),
                "checkpoint": {
                    "path": checkpoint_report["models"][model]["path"],
                    "sha256": checkpoint_report["models"][model]["sha256_actual"],
                },
            }
            for model in ("mctformer", "mctformer_plus")
        },
        "paired_analysis_root": str(paired_root),
        "dataset": {
            "voc_root": str(config.voc_root),
            "list_path": str(config.val_list),
            "labels_path": str(labels_path),
            "input_size": config.crop_size,
            "patch_size": 16,
            "num_images": len(image_ids),
        },
        "resolved_inputs": {
            "voc_root": str(config.voc_root),
            "val_list": str(config.val_list),
            "labels_path": str(labels_path),
            "mctformer_result_root": str(source_roots["mctformer"]),
            "mctformer_plus_result_root": str(source_roots["mctformer_plus"]),
            "paired_analysis_root": str(paired_root),
        },
        "experiment1_metadata": metadata_by_model,
        "paired_analysis_metadata": paired_metadata,
        "source_provenance": provenance,
        "gt_summary": gt_summary,
        "before_manifest": {
            "row_count": len(before_manifest),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in before_manifest),
            "source_group_counts": dict(
                Counter(str(row["source_group"]) for row in before_manifest)
            ),
            "scope": "all Experiment 1 model-result and paired-analysis files; exact checkpoints; VOC val JPEGs, semantic masks, list and image labels; current low-level runtime sources",
        },
    }

    config.output_dir.mkdir(parents=True, exist_ok=False)
    csv_dump(
        config.output_dir / "file_manifest_before.csv",
        before_manifest,
        FILE_MANIFEST_FIELDS,
    )
    csv_dump(config.output_dir / "gt_manifest.csv", gt_rows, GT_MANIFEST_FIELDS)
    json_dump(config.output_dir / "checkpoint_verification.json", checkpoint_report)
    json_dump(config.output_dir / "experiment1_linkage.json", linkage)
    json_dump(config.output_dir / "source_metadata.json", source_metadata)
    (config.output_dir / "INPUT_AUDIT.md").write_text(
        _markdown_report(
            source_roots,
            paired_root,
            result_summaries,
            checkpoint_report,
            gt_summary,
            before_manifest,
            issues,
            passed,
        ),
        encoding="utf-8",
    )
    return source_metadata


def main(argv: Optional[list[str]] = None) -> int:
    config = config_from_args(parse_args(argv))
    report = run_audit(config)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "integrity_passed": report["integrity_passed"],
                "errors": report["error_count"],
                "warnings": report["warning_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["integrity_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
