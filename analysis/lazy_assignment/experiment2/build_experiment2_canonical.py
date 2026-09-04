#!/usr/bin/env python3
"""Build immutable-source canonical tables for Experiment 2.

This entry point consumes completed per-image signal artifacts.  It never
writes below either input root.  Every signal NPZ is schema-checked, linked to
its manifest SHA-256, and re-hashed after table construction before any
canonical output is finalized.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shlex
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.canonical_io import (  # noqa: E402
    SCHEMA_VERSION,
    StreamingCanonicalWriter,
)
from analysis.lazy_assignment.experiment2.common import (  # noqa: E402
    VOC_CLASS_NAMES,
    assert_output_outside_inputs,
    git_metadata,
    sha256_file,
)
from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    TOPK_RATIOS,
    map_overlap_metrics,
    region_map_metrics,
    stable_topk_mask,
)
from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    RAW_CAM_BACKGROUND_THRESHOLD,
)
from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (  # noqa: E402
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.metrics_stage_linkage import (  # noqa: E402
    stage_transition_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    REGION_BACKGROUND,
    REGION_VOID,
    assign_pair_patch_regions_from_counts,
    assign_patch_regions_from_counts,
)


MODEL_KEYS = ("mctformer", "mctformer_plus")
REQUIRED_NPZ_KEYS = frozenset(
    {
        "image_id",
        "positive_class_ids",
        "grid_h",
        "grid_w",
        "patch_label_counts",
        "region_masks_rho05",
        "region_masks_rho07",
        "feature_post_scores",
        "feature_norm_scores",
        "feature_final_norm_scores",
        "qk_mean_scores",
        "qk_head_std",
        "attn_c2p_raw",
        "attn_c2p_conditional",
        "attn_patch_mass",
        "patch_logits",
        "patch_cam",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "c2p_cam",
        "final_cam",
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
        "class_logits",
        "patch_class_logits",
        "class_logits_all",
        "patch_class_logits_all",
        "raw_final_cam_confusion_t045",
        "class_token_pairwise_cosine",
        "patch_norms",
        "qk_head_region_mean_rho05",
        "qk_head_region_mean_rho07",
    }
)

FLOAT_MAP_KEYS = (
    "feature_post_scores",
    "feature_norm_scores",
    "feature_final_norm_scores",
    "qk_mean_scores",
    "qk_head_std",
    "attn_c2p_raw",
    "attn_c2p_conditional",
    "attn_patch_mass",
    "patch_logits",
    "patch_cam",
    "attn_official_raw",
    "attn_official_conditional",
    "attn_mid3_raw",
    "attn_mid3_conditional",
    "c2p_cam",
    "final_cam",
    "diagnostic_c2p_cam_l10",
    "diagnostic_c2p_cam_l11",
    "diagnostic_c2p_cam_l12",
    "diagnostic_c2p_cam_mid3",
    "class_logits",
    "patch_class_logits",
    "class_logits_all",
    "patch_class_logits_all",
    "class_token_pairwise_cosine",
    "patch_norms",
    "qk_head_region_mean_rho05",
    "qk_head_region_mean_rho07",
)

PATCH_NORM_JOINT_COLUMNS = (
    "post_cosine_patch_l2norm_pearson_valid",
    "post_cosine_patch_l2norm_pearson_bg",
    "feature_top10_bg_patch_l2norm_mean",
    "feature_top10_bg_patch_l2norm_enrichment_vs_bg",
    "feature_top10_bg_below_valid_l2norm_median_fraction",
    "feature_top10_bg_above_valid_l2norm_q75_fraction",
    "feature_top10_bg_patch_count",
)


@dataclass(frozen=True)
class ManifestEntry:
    model: str
    line_number: int
    image_id: str
    positive_class_ids: tuple[int, ...]
    grid_h: int
    grid_w: int
    num_layers: int
    num_patches: int
    artifact_path: Path
    relative_path: str
    artifact_sha256: str


@dataclass(frozen=True)
class SignalRoot:
    model: str
    root: Path
    manifest_path: Path
    metadata_path: Path
    completion_path: Path
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _scalar_string(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes signal root: {relative!r}") from error
    return path


def _manifest_signal_path(
    root: Path, row: Mapping[str, object], image_id: str
) -> tuple[Path, str]:
    raw = row.get("signal_path", row.get("npz_path"))
    if raw is None:
        raw = f"signals/{image_id}.npz"
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"invalid signal_path for {image_id!r}: {raw!r}")
    relative = str(Path(raw))
    path = _inside_root(root, relative)
    try:
        path.relative_to(root / "signals")
    except ValueError as error:
        raise ValueError(
            f"signal artifact must be below {root / 'signals'}: {path}"
        ) from error
    return path, relative


def load_signal_root(
    model: str,
    root: Path,
    *,
    require_full: bool = True,
) -> SignalRoot:
    """Validate completion and parse one deterministic signal manifest."""

    if model not in MODEL_KEYS:
        raise ValueError(f"unsupported model key {model!r}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    completion_path = root / "completion.json"
    manifest_path = root / "manifest.jsonl"
    metadata_candidates = [root / "metadata.json", root / "run_metadata.json"]
    metadata_paths = [path for path in metadata_candidates if path.is_file()]
    if len(metadata_paths) != 1:
        raise RuntimeError(
            f"expected exactly one metadata.json/run_metadata.json in {root}, "
            f"found {metadata_paths}"
        )
    metadata_path = metadata_paths[0]
    completion = _read_json(completion_path)
    if completion.get("status") != "complete":
        raise RuntimeError(f"signal root is not complete: {completion_path}")
    if require_full and completion.get("run_kind") != "full":
        raise RuntimeError(f"full canonical build rejects non-full root: {root}")
    metadata = _read_json(metadata_path)
    recorded_model = metadata.get("model")
    if isinstance(recorded_model, Mapping):
        recorded_model = recorded_model.get("name")
    if recorded_model != model:
        raise RuntimeError(
            f"signal metadata model {recorded_model!r} does not match requested {model!r}"
        )
    if completion.get("model") not in (None, model):
        raise RuntimeError(
            f"signal completion model {completion.get('model')!r} does not match {model!r}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"manifest line {line_number} is not an object")
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in seen:
            raise ValueError(
                f"empty or duplicate image_id at manifest line {line_number}"
            )
        seen.add(image_id)
        positives = tuple(int(value) for value in row.get("positive_class_ids", []))
        if not positives or tuple(sorted(set(positives))) != positives:
            raise ValueError(
                f"positive_class_ids must be nonempty, sorted, and unique for {image_id}"
            )
        if min(positives) < 0 or max(positives) >= 20:
            raise ValueError(f"positive class outside [0,19] for {image_id}")
        grid_h = int(row.get("grid_h", 0))
        grid_w = int(row.get("grid_w", 0))
        layers = int(row.get("num_layers", 0))
        patches = int(row.get("num_patches", 0))
        if min(grid_h, grid_w, layers, patches) < 1 or grid_h * grid_w != patches:
            raise ValueError(f"invalid layer/patch geometry for {image_id}")
        path, relative = _manifest_signal_path(root, row, image_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_hash = row.get("artifact_sha256")
        if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
            raise ValueError(f"manifest lacks a valid artifact_sha256 for {image_id}")
        actual_hash = sha256_file(path)
        if actual_hash != artifact_hash:
            raise RuntimeError(
                f"artifact SHA-256 mismatch for {image_id}: "
                f"manifest={artifact_hash}, actual={actual_hash}"
            )
        entries.append(
            ManifestEntry(
                model=model,
                line_number=line_number,
                image_id=image_id,
                positive_class_ids=positives,
                grid_h=grid_h,
                grid_w=grid_w,
                num_layers=layers,
                num_patches=patches,
                artifact_path=path,
                relative_path=relative,
                artifact_sha256=actual_hash,
            )
        )
    if not entries:
        raise ValueError(f"empty signal manifest: {manifest_path}")
    actual_npz = {path.resolve() for path in (root / "signals").glob("*.npz")}
    recorded_npz = {entry.artifact_path for entry in entries}
    if actual_npz != recorded_npz:
        raise RuntimeError(
            f"manifest/files mismatch in {root}: unrecorded={sorted(actual_npz - recorded_npz)}, "
            f"missing={sorted(recorded_npz - actual_npz)}"
        )
    return SignalRoot(
        model=model,
        root=root,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        completion_path=completion_path,
        entries=tuple(entries),
    )


def assert_exact_manifest_match(left: SignalRoot, right: SignalRoot) -> None:
    """Require identical ordered image/class/grid identities across models."""

    def identity(entry: ManifestEntry) -> tuple[object, ...]:
        return (
            entry.image_id,
            entry.positive_class_ids,
            entry.grid_h,
            entry.grid_w,
            entry.num_layers,
            entry.num_patches,
        )

    left_rows = [identity(entry) for entry in left.entries]
    right_rows = [identity(entry) for entry in right.entries]
    if left_rows != right_rows:
        first_difference = next(
            (
                index
                for index, values in enumerate(
                    itertools.zip_longest(left_rows, right_rows)
                )
                if values[0] != values[1]
            ),
            None,
        )
        raise RuntimeError(
            "signal manifests do not match exactly in order, positives, or geometry; "
            f"first_difference={first_difference}, left_rows={len(left_rows)}, "
            f"right_rows={len(right_rows)}"
        )


def snapshot_root(root: Path) -> dict[str, FileSnapshot]:
    snapshots: dict[str, FileSnapshot] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        stat = path.stat()
        relative = str(path.relative_to(root))
        snapshots[relative] = FileSnapshot(
            path=path,
            relative_path=relative,
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            sha256=sha256_file(path),
        )
    return snapshots


def assert_snapshot_unchanged(
    root: Path, before: Mapping[str, FileSnapshot]
) -> dict[str, FileSnapshot]:
    after = snapshot_root(root)
    if set(after) != set(before):
        raise RuntimeError(f"immutable source membership changed while reading {root}")
    changed = [
        relative
        for relative in before
        if (
            before[relative].size_bytes,
            before[relative].mtime_ns,
            before[relative].sha256,
        )
        != (
            after[relative].size_bytes,
            after[relative].mtime_ns,
            after[relative].sha256,
        )
    ]
    if changed:
        raise RuntimeError(
            f"immutable source files changed while reading {root}: {changed[:10]}"
        )
    return after


def _require_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    if tuple(array.shape) != expected:
        raise ValueError(f"{name} shape {array.shape} != {expected}")


def _require_float32(name: str, array: np.ndarray, *, allow_nan: bool = False) -> None:
    if array.dtype != np.float32:
        raise TypeError(f"{name} must be float32, got {array.dtype}")
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        raise ValueError(f"{name} contains invalid non-finite values")


def load_and_validate_artifact(
    entry: ManifestEntry,
    *,
    expected_layers: int = 12,
    expected_grid: Optional[tuple[int, int]] = (28, 28),
) -> dict[str, np.ndarray]:
    """Load one NPZ and enforce the agreed signal and numerical contracts."""

    with np.load(entry.artifact_path, allow_pickle=False) as archive:
        missing = REQUIRED_NPZ_KEYS.difference(archive.files)
        if missing:
            raise KeyError(f"{entry.artifact_path} misses keys {sorted(missing)}")
        artifact = {key: np.array(archive[key], copy=True) for key in REQUIRED_NPZ_KEYS}
    image_id = _scalar_string(artifact["image_id"])
    if image_id != entry.image_id:
        raise ValueError(f"NPZ image_id {image_id!r} != manifest {entry.image_id!r}")
    positives = np.asarray(artifact["positive_class_ids"])
    if not np.issubdtype(positives.dtype, np.integer):
        raise TypeError("positive_class_ids must be integers")
    positives = positives.astype(np.int64, copy=False).reshape(-1)
    if tuple(positives.tolist()) != entry.positive_class_ids:
        raise ValueError(f"NPZ/manifest positive classes differ for {entry.image_id}")
    grid_h = int(np.asarray(artifact["grid_h"]).item())
    grid_w = int(np.asarray(artifact["grid_w"]).item())
    if (grid_h, grid_w) != (entry.grid_h, entry.grid_w):
        raise ValueError(f"NPZ/manifest grid differs for {entry.image_id}")
    if expected_grid is not None and (grid_h, grid_w) != tuple(expected_grid):
        raise ValueError(
            f"production grid {(grid_h, grid_w)} != expected {expected_grid}"
        )
    if entry.num_layers != expected_layers:
        raise ValueError(f"expected {expected_layers} layers, got {entry.num_layers}")
    layers, classes, patches = expected_layers, len(positives), grid_h * grid_w

    shape_contract = {
        "patch_label_counts": (patches, 22),
        "region_masks_rho05": (classes, patches),
        "region_masks_rho07": (classes, patches),
        "feature_post_scores": (layers, classes, patches),
        "feature_norm_scores": (layers, classes, patches),
        "feature_final_norm_scores": (classes, patches),
        "qk_mean_scores": (layers, classes, patches),
        "qk_head_std": (layers, classes, patches),
        "attn_c2p_raw": (layers, classes, patches),
        "attn_c2p_conditional": (layers, classes, patches),
        "attn_patch_mass": (layers, classes),
        "patch_logits": (classes, patches),
        "patch_cam": (classes, patches),
        "attn_official_raw": (classes, patches),
        "attn_official_conditional": (classes, patches),
        "attn_mid3_raw": (classes, patches),
        "attn_mid3_conditional": (classes, patches),
        "c2p_cam": (classes, patches),
        "final_cam": (classes, patches),
        "diagnostic_c2p_cam_l10": (classes, patches),
        "diagnostic_c2p_cam_l11": (classes, patches),
        "diagnostic_c2p_cam_l12": (classes, patches),
        "diagnostic_c2p_cam_mid3": (classes, patches),
        "class_logits": (classes,),
        "patch_class_logits": (classes,),
        "class_logits_all": (20,),
        "patch_class_logits_all": (20,),
        "raw_final_cam_confusion_t045": (21, 21),
        "class_token_pairwise_cosine": (layers, classes, classes),
        "patch_norms": (layers, patches),
    }
    for name, shape in shape_contract.items():
        _require_shape(name, artifact[name], shape)
    for name in FLOAT_MAP_KEYS:
        _require_float32(
            name,
            artifact[name],
            allow_nan=name.startswith("qk_head_region_mean_"),
        )
    for name in ("region_masks_rho05", "region_masks_rho07", "patch_label_counts"):
        if not np.issubdtype(artifact[name].dtype, np.integer):
            raise TypeError(f"{name} must be integer-valued")
    confusion = artifact["raw_final_cam_confusion_t045"]
    if confusion.dtype != np.int64 or np.any(confusion < 0):
        raise TypeError("raw_final_cam_confusion_t045 must be non-negative int64")
    counts = artifact["patch_label_counts"]
    if np.any(counts < 0) or not np.all(counts.sum(axis=-1) == 256):
        raise ValueError("patch_label_counts must contain 256 pixels per patch")
    expected_valid_pixels = grid_h * grid_w * 256 - int(counts[:, 21].sum())
    if int(confusion.sum()) != expected_valid_pixels:
        raise ValueError("raw final-CAM confusion pixel count disagrees with GT counts")
    for key, rho in (("region_masks_rho05", 0.5), ("region_masks_rho07", 0.7)):
        observed = artifact[key]
        if np.any((observed < 0) | (observed > 4)):
            raise ValueError(f"{key} contains an invalid region code")
        for class_offset, class_id in enumerate(positives):
            expected = assign_patch_regions_from_counts(
                counts, int(class_id), rho=rho, grid_size=None
            )["region_codes"]
            if not np.array_equal(observed[class_offset], expected):
                raise ValueError(
                    f"{key} does not reproduce from patch counts for "
                    f"{entry.image_id}/class {class_id}"
                )

    head_shape_05 = artifact["qk_head_region_mean_rho05"].shape
    head_shape_07 = artifact["qk_head_region_mean_rho07"].shape
    if (
        len(head_shape_05) != 4
        or head_shape_05[0] != layers
        or head_shape_05[2:] != (classes, 3)
    ):
        raise ValueError(f"invalid qk_head_region_mean_rho05 shape {head_shape_05}")
    if head_shape_07 != head_shape_05:
        raise ValueError("rho=.5/.7 QK head-region summaries have different shapes")
    if head_shape_05[1] < 1:
        raise ValueError("QK head-region summary has no heads")

    if np.any(artifact["qk_head_std"] < -1e-7):
        raise ValueError("qk_head_std contains a negative value")
    for key in (
        "attn_c2p_raw",
        "attn_c2p_conditional",
        "attn_patch_mass",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "patch_cam",
        "c2p_cam",
        "final_cam",
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
        "patch_norms",
    ):
        if np.any(artifact[key] < -1e-7):
            raise ValueError(f"{key} contains a negative value")
    np.testing.assert_allclose(
        artifact["attn_c2p_raw"].sum(axis=-1),
        artifact["attn_patch_mass"],
        rtol=0,
        atol=1e-6,
    )
    for key in (
        "attn_c2p_conditional",
        "attn_official_conditional",
        "attn_mid3_conditional",
    ):
        np.testing.assert_allclose(artifact[key].sum(axis=-1), 1.0, rtol=0, atol=1e-5)
    for raw, conditional in (
        ("attn_official_raw", "attn_official_conditional"),
        ("attn_mid3_raw", "attn_mid3_conditional"),
    ):
        expected = artifact[raw] / artifact[raw].sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(artifact[conditional], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        artifact["patch_cam"],
        np.maximum(artifact["patch_logits"], 0.0),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        artifact["class_logits_all"][positives],
        artifact["class_logits"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        artifact["patch_class_logits_all"][positives],
        artifact["patch_class_logits"],
        rtol=0,
        atol=0,
    )
    pairwise = artifact["class_token_pairwise_cosine"]
    np.testing.assert_allclose(pairwise, pairwise.transpose(0, 2, 1), rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.diagonal(pairwise, axis1=1, axis2=2), 1.0, rtol=0, atol=1e-5
    )
    artifact["positive_class_ids"] = positives
    return artifact


def _label_stratum(num_positive_classes: int) -> str:
    if num_positive_classes == 1:
        return "single_label"
    if num_positive_classes == 2:
        return "exactly_2_labels"
    if num_positive_classes >= 3:
        return "3plus_labels"
    raise ValueError("an analyzed image must have at least one positive class")


def _classification_status(class_logit: float, patch_logit: float) -> str:
    class_positive = class_logit > 0.0
    patch_positive = patch_logit > 0.0
    if class_positive and patch_positive:
        return "both_positive"
    if class_positive:
        return "class_only_positive"
    if patch_positive:
        return "patch_only_positive"
    return "neither_positive"


def _class_base(
    model: str,
    image_id: str,
    positives: np.ndarray,
    class_offset: int,
    artifact: Mapping[str, np.ndarray],
) -> dict[str, object]:
    class_id = int(positives[class_offset])
    class_logit = float(artifact["class_logits"][class_offset])
    patch_logit = float(artifact["patch_class_logits"][class_offset])
    return {
        "model": model,
        "image_id": image_id,
        "class_id": class_id,
        "class_name": VOC_CLASS_NAMES[class_id],
        "class_offset": int(class_offset),
        "num_positive_classes": int(len(positives)),
        "label_stratum": _label_stratum(len(positives)),
        "class_logit": class_logit,
        "patch_class_logit": patch_logit,
        "class_token_positive": bool(class_logit > 0.0),
        "patch_head_positive": bool(patch_logit > 0.0),
        "classification_status": _classification_status(class_logit, patch_logit),
    }


def _pair_base(
    model: str,
    image_id: str,
    positives: np.ndarray,
    offset_a: int,
    offset_b: int,
    artifact: Mapping[str, np.ndarray],
) -> dict[str, object]:
    class_a = int(positives[offset_a])
    class_b = int(positives[offset_b])
    class_logits = artifact["class_logits"]
    patch_logits = artifact["patch_class_logits"]
    return {
        "model": model,
        "image_id": image_id,
        "class_a": class_a,
        "class_b": class_b,
        "class_a_name": VOC_CLASS_NAMES[class_a],
        "class_b_name": VOC_CLASS_NAMES[class_b],
        "class_a_offset": int(offset_a),
        "class_b_offset": int(offset_b),
        "num_positive_classes": int(len(positives)),
        "label_stratum": _label_stratum(len(positives)),
        "classification_status_a": _classification_status(
            float(class_logits[offset_a]), float(patch_logits[offset_a])
        ),
        "classification_status_b": _classification_status(
            float(class_logits[offset_b]), float(patch_logits[offset_b])
        ),
    }


def _classification_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    positives = {int(value) for value in artifact["positive_class_ids"]}
    class_logits = artifact["class_logits_all"]
    patch_logits = artifact["patch_class_logits_all"]
    return [
        {
            "model": model,
            "image_id": entry.image_id,
            "class_id": class_id,
            "class_name": VOC_CLASS_NAMES[class_id],
            "target": class_id in positives,
            "class_logit": float(class_logits[class_id]),
            "patch_class_logit": float(patch_logits[class_id]),
            "num_positive_classes": len(positives),
            "label_stratum": _label_stratum(len(positives)),
        }
        for class_id in range(20)
    ]


def _cam_confusion_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    confusion = artifact["raw_final_cam_confusion_t045"]
    positive_count = len(artifact["positive_class_ids"])
    base = {
        "model": model,
        "image_id": entry.image_id,
        "num_positive_classes": positive_count,
        "label_stratum": _label_stratum(positive_count),
        "cam_stage": "final_cam",
        "input_resolution": 448,
        "background_threshold": RAW_CAM_BACKGROUND_THRESHOLD,
        "normalization": "per-active-class min-max after bilinear upsampling",
    }
    return [
        {
            **base,
            "gt_class_id": gt_class_id,
            "pred_class_id": pred_class_id,
            "pixel_count": int(confusion[gt_class_id, pred_class_id]),
        }
        for gt_class_id in range(21)
        for pred_class_id in range(21)
    ]


def _rho_regions(artifact: Mapping[str, np.ndarray], rho: float) -> np.ndarray:
    if rho == 0.5:
        return artifact["region_masks_rho05"]
    if rho == 0.7:
        return artifact["region_masks_rho07"]
    raise ValueError(f"unsupported preregistered rho {rho}")


def _finite_json(values: np.ndarray) -> str:
    return json.dumps(
        [
            float(value) if np.isfinite(value) else None
            for value in np.asarray(values).reshape(-1)
        ],
        separators=(",", ":"),
        allow_nan=False,
    )


def _derived_feature_probe_maps(
    artifact: Mapping[str, np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return active-class relative and class-softmax probe controls."""

    scores = np.asarray(artifact["feature_post_scores"], dtype=np.float32)
    if scores.shape[1] < 2:
        return None, None
    relative = np.empty_like(scores)
    for class_offset in range(scores.shape[1]):
        other = np.concatenate(
            (scores[:, :class_offset], scores[:, class_offset + 1 :]), axis=1
        )
        relative[:, class_offset] = scores[:, class_offset] - other.max(axis=1)
    logits = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(logits).astype(np.float32, copy=False)
    active_softmax = exponent / exponent.sum(axis=1, keepdims=True)
    return relative.astype(np.float32, copy=False), active_softmax.astype(
        np.float32, copy=False
    )


def _signal_role(signal: str) -> str:
    if signal in {
        "feature_post_relative",
        "feature_post_active_softmax",
        "patch_norm",
        "feature_final_norm",
        "qk_head_std",
    }:
        return "probe_control"
    if signal.startswith("diagnostic_") or signal.startswith("attn_mid3"):
        return "diagnostic"
    return "primary"


def _canonical_region_metrics(
    values: np.ndarray,
    region_labels: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    nonnegative_mass: bool,
) -> dict[str, object]:
    """Apply shared metrics and the plan's all-patch CBL denominator.

    Ranking excludes void patches, but conditional attention/CAM background
    mass is defined after normalizing the complete spatial map.  It therefore
    must not be re-normalized after dropping void patches.
    """

    result = region_map_metrics(
        values,
        region_labels,
        grid_h=grid_h,
        grid_w=grid_w,
        nonnegative_mass=nonnegative_mass,
    )
    result["void_hit"] = False
    result["c_pim_target_hit"] = bool(result["target_hit"])
    result["other_hit"] = bool(result["other_fg_hit"])
    for ratio in TOPK_RATIOS:
        suffix = f"{int(round(100 * ratio)):02d}"
        result[f"other_top{suffix}_fraction"] = result[f"other_fg_top{suffix}_fraction"]
        result[f"other_tail_enrich_{suffix}"] = result[f"other_fg_tail_enrich_{suffix}"]
    if nonnegative_mass:
        spatial = np.asarray(values, dtype=np.float64).reshape(-1)
        labels = np.asarray(region_labels).reshape(-1)
        total = float(spatial.sum())
        result["conditional_bg_mass"] = (
            float(spatial[labels == 2].sum() / total) if total > 1e-12 else float("nan")
        )
    return result


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        raise ValueError("Pearson inputs must have equal length")
    finite = np.isfinite(left) & np.isfinite(right)
    if (
        int(finite.sum()) < 2
        or np.ptp(left[finite]) <= 1e-12
        or np.ptp(right[finite]) <= 1e-12
    ):
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _feature_patch_norm_controls(
    feature_scores: np.ndarray,
    patch_l2norm: np.ndarray,
    region_labels: np.ndarray,
) -> dict[str, float | int]:
    """Joint post-block-cosine/patch-norm controls for one image/class/layer.

    Thresholds are within-image controls fixed before the full run. ``low`` is
    at or below the median valid-patch norm and ``high`` is at or above its
    75th percentile. These measurements diagnose norm concentration; they do
    not by themselves establish a register token or a semantic shortcut.
    """

    scores = np.asarray(feature_scores, dtype=np.float64).reshape(-1)
    norms = np.asarray(patch_l2norm, dtype=np.float64).reshape(-1)
    regions = np.asarray(region_labels).reshape(-1)
    if scores.size != norms.size or scores.size != regions.size:
        raise ValueError("feature, patch-norm, and region maps must have equal size")
    if not np.isfinite(scores).all() or not np.isfinite(norms).all():
        raise ValueError("feature and patch-norm controls require finite maps")
    valid = regions != REGION_VOID
    background = regions == REGION_BACKGROUND
    if not bool(valid.any()):
        raise ValueError("patch-norm control has no valid patches")
    top10 = stable_topk_mask(scores, 0.10, eligible=valid).reshape(-1)
    top10_background = top10 & background
    selected_norms = norms[top10_background]
    background_norms = norms[background]
    valid_norms = norms[valid]
    selected_count = int(selected_norms.size)
    selected_mean = float(selected_norms.mean()) if selected_count else float("nan")
    background_mean = (
        float(background_norms.mean()) if background_norms.size else float("nan")
    )
    enrichment = (
        selected_mean / background_mean
        if np.isfinite(selected_mean) and background_mean > 1e-12
        else float("nan")
    )
    median = float(np.median(valid_norms))
    q75 = float(np.quantile(valid_norms, 0.75))
    return {
        "post_cosine_patch_l2norm_pearson_valid": _finite_pearson(
            scores[valid], norms[valid]
        ),
        "post_cosine_patch_l2norm_pearson_bg": _finite_pearson(
            scores[background], norms[background]
        ),
        "feature_top10_bg_patch_l2norm_mean": selected_mean,
        "feature_top10_bg_patch_l2norm_enrichment_vs_bg": enrichment,
        "feature_top10_bg_below_valid_l2norm_median_fraction": (
            float(np.mean(selected_norms <= median)) if selected_count else float("nan")
        ),
        "feature_top10_bg_above_valid_l2norm_q75_fraction": (
            float(np.mean(selected_norms >= q75)) if selected_count else float("nan")
        ),
        "feature_top10_bg_patch_count": selected_count,
    }


def _layer_signal_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positives = artifact["positive_class_ids"]
    grid_h, grid_w = entry.grid_h, entry.grid_w
    relative, active_softmax = _derived_feature_probe_maps(artifact)
    layered = [
        ("feature_post", artifact["feature_post_scores"], False, True),
        ("feature_norm", artifact["feature_norm_scores"], False, True),
        ("qk_mean", artifact["qk_mean_scores"], False, True),
        ("qk_head_std", artifact["qk_head_std"], False, True),
        ("attn_c2p_raw", artifact["attn_c2p_raw"], False, True),
        ("attn_c2p_conditional", artifact["attn_c2p_conditional"], True, True),
        ("patch_norm", artifact["patch_norms"], False, False),
    ]
    if relative is not None and active_softmax is not None:
        layered.extend(
            [
                ("feature_post_relative", relative, False, True),
                ("feature_post_active_softmax", active_softmax, False, True),
            ]
        )
    aggregates = (
        (
            "feature_final_norm",
            12,
            "final_norm",
            artifact["feature_final_norm_scores"],
            False,
        ),
        (
            "attn_official_raw",
            0,
            "official_last3",
            artifact["attn_official_raw"],
            False,
        ),
        (
            "attn_official_conditional",
            0,
            "official_last3",
            artifact["attn_official_conditional"],
            True,
        ),
        ("attn_mid3_raw", 0, "mid3", artifact["attn_mid3_raw"], False),
        (
            "attn_mid3_conditional",
            0,
            "mid3",
            artifact["attn_mid3_conditional"],
            True,
        ),
    )
    for rho in (0.5, 0.7):
        regions = _rho_regions(artifact, rho)
        qk_head_regions = artifact[
            "qk_head_region_mean_rho05" if rho == 0.5 else "qk_head_region_mean_rho07"
        ]
        for class_offset in range(len(positives)):
            base = _class_base(model, entry.image_id, positives, class_offset, artifact)
            region_labels = regions[class_offset]
            for signal, maps, nonnegative_mass, class_specific in layered:
                for layer in range(entry.num_layers):
                    row = {
                        **base,
                        "layer": layer + 1,
                        "layer_label": f"L{layer + 1}",
                        "signal": signal,
                        "signal_role": _signal_role(signal),
                        "rho": rho,
                        "attn_patch_mass": (
                            float(artifact["attn_patch_mass"][layer, class_offset])
                            if signal.startswith("attn_c2p")
                            else float("nan")
                        ),
                        "num_heads": int(qk_head_regions.shape[1]),
                        "qk_head_target_means_json": None,
                        "qk_head_other_fg_means_json": None,
                        "qk_head_bg_means_json": None,
                        **{name: float("nan") for name in PATCH_NORM_JOINT_COLUMNS},
                    }
                    for head in range(6):
                        for region_name in ("target", "other_fg", "bg"):
                            row[f"qk_head{head}_{region_name}_mean"] = float("nan")
                    if signal == "qk_mean":
                        head_values = qk_head_regions[layer, :, class_offset]
                        row.update(
                            {
                                "qk_head_target_means_json": _finite_json(
                                    head_values[:, 0]
                                ),
                                "qk_head_other_fg_means_json": _finite_json(
                                    head_values[:, 1]
                                ),
                                "qk_head_bg_means_json": _finite_json(
                                    head_values[:, 2]
                                ),
                            }
                        )
                        for head in range(min(6, head_values.shape[0])):
                            row[f"qk_head{head}_target_mean"] = float(
                                head_values[head, 0]
                            )
                            row[f"qk_head{head}_other_fg_mean"] = float(
                                head_values[head, 1]
                            )
                            row[f"qk_head{head}_bg_mean"] = float(head_values[head, 2])
                    row.update(
                        _canonical_region_metrics(
                            maps[layer, class_offset]
                            if class_specific
                            else maps[layer],
                            region_labels,
                            grid_h=grid_h,
                            grid_w=grid_w,
                            nonnegative_mass=nonnegative_mass,
                        )
                    )
                    if signal == "feature_post":
                        row.update(
                            _feature_patch_norm_controls(
                                maps[layer, class_offset],
                                artifact["patch_norms"][layer],
                                region_labels,
                            )
                        )
                    rows.append(row)
            for signal, layer, layer_label, maps, nonnegative_mass in aggregates:
                row = {
                    **base,
                    "layer": layer,
                    "layer_label": layer_label,
                    "signal": signal,
                    "signal_role": _signal_role(signal),
                    "rho": rho,
                    "attn_patch_mass": float(maps[class_offset].sum())
                    if signal.endswith("_raw")
                    else float("nan"),
                    "num_heads": int(qk_head_regions.shape[1]),
                    "qk_head_target_means_json": None,
                    "qk_head_other_fg_means_json": None,
                    "qk_head_bg_means_json": None,
                    **{name: float("nan") for name in PATCH_NORM_JOINT_COLUMNS},
                }
                for head in range(6):
                    for region_name in ("target", "other_fg", "bg"):
                        row[f"qk_head{head}_{region_name}_mean"] = float("nan")
                row.update(
                    _canonical_region_metrics(
                        maps[class_offset],
                        region_labels,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        nonnegative_mass=nonnegative_mass,
                    )
                )
                rows.append(row)
    return rows


def _cam_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positives = artifact["positive_class_ids"]
    stages = (
        ("patch_cam", artifact["patch_cam"], "native", 12),
        ("c2p_cam", artifact["c2p_cam"], "native", 0),
        ("final_cam", artifact["final_cam"], "native", 0),
        (
            "diagnostic_c2p_cam_l10",
            artifact["diagnostic_c2p_cam_l10"],
            "diagnostic",
            10,
        ),
        (
            "diagnostic_c2p_cam_l11",
            artifact["diagnostic_c2p_cam_l11"],
            "diagnostic",
            11,
        ),
        (
            "diagnostic_c2p_cam_l12",
            artifact["diagnostic_c2p_cam_l12"],
            "diagnostic",
            12,
        ),
        (
            "diagnostic_c2p_cam_mid3",
            artifact["diagnostic_c2p_cam_mid3"],
            "diagnostic",
            0,
        ),
    )
    for rho in (0.5, 0.7):
        regions = _rho_regions(artifact, rho)
        for class_offset in range(len(positives)):
            base = _class_base(model, entry.image_id, positives, class_offset, artifact)
            for stage, maps, stage_kind, source_layer in stages:
                row = {
                    **base,
                    "stage": stage,
                    "stage_kind": stage_kind,
                    "source_layer": source_layer,
                    "rho": rho,
                }
                row.update(
                    _canonical_region_metrics(
                        maps[class_offset],
                        regions[class_offset],
                        grid_h=entry.grid_h,
                        grid_w=entry.grid_w,
                        nonnegative_mass=True,
                    )
                )
                rows.append(row)
    return rows


def _transition_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positives = artifact["positive_class_ids"]
    layered = (
        (
            "feature_post_to_attn",
            "feature_post",
            "attn_c2p_conditional",
            artifact["feature_post_scores"],
            artifact["attn_c2p_conditional"],
            "post-block feature versus same-index pre-block attention",
        ),
        (
            "feature_norm_to_qk",
            "feature_norm",
            "qk_mean",
            artifact["feature_norm_scores"],
            artifact["qk_mean_scores"],
            "same pre-attention normalized geometry",
        ),
        (
            "qk_to_attn",
            "qk_mean",
            "attn_c2p_conditional",
            artifact["qk_mean_scores"],
            artifact["attn_c2p_conditional"],
            "same pre-attention QK-to-softmax routing",
        ),
    )
    finals = (
        (
            "feature_l12_to_patch_cam",
            "feature_post",
            "patch_cam",
            12,
            artifact["feature_post_scores"][11],
            artifact["patch_cam"],
            "final post-block representation to native patch-head CAM",
        ),
        (
            "official_attn_to_c2p_cam",
            "attn_official_conditional",
            "c2p_cam",
            0,
            artifact["attn_official_conditional"],
            artifact["c2p_cam"],
            "official last-three attention to native class-attention CAM",
        ),
        (
            "c2p_cam_to_final_cam",
            "c2p_cam",
            "final_cam",
            0,
            artifact["c2p_cam"],
            artifact["final_cam"],
            "native patch-to-patch propagation",
        ),
    )
    for rho in (0.5, 0.7):
        regions = _rho_regions(artifact, rho)
        for class_offset in range(len(positives)):
            base = _class_base(model, entry.image_id, positives, class_offset, artifact)
            for (
                transition,
                source_signal,
                destination_signal,
                source,
                destination,
                note,
            ) in layered:
                for layer in range(entry.num_layers):
                    for ratio in TOPK_RATIOS:
                        row = {
                            **base,
                            "transition": transition,
                            "source_signal": source_signal,
                            "destination_signal": destination_signal,
                            "layer": layer + 1,
                            "layer_label": f"L{layer + 1}",
                            "rho": rho,
                            "timing_note": note,
                        }
                        row.update(
                            stage_transition_metrics(
                                source[layer, class_offset],
                                destination[layer, class_offset],
                                regions[class_offset],
                                ratio=ratio,
                            )
                        )
                        rows.append(row)
            for (
                transition,
                source_signal,
                destination_signal,
                layer,
                source,
                destination,
                note,
            ) in finals:
                for ratio in TOPK_RATIOS:
                    row = {
                        **base,
                        "transition": transition,
                        "source_signal": source_signal,
                        "destination_signal": destination_signal,
                        "layer": layer,
                        "layer_label": f"L{layer}" if layer else "aggregate",
                        "rho": rho,
                        "timing_note": note,
                    }
                    row.update(
                        stage_transition_metrics(
                            source[class_offset],
                            destination[class_offset],
                            regions[class_offset],
                            ratio=ratio,
                        )
                    )
                    rows.append(row)
    return rows


def _pair_layer_signal_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Cross-class map diversity without semantic ownership aggregation."""

    positives = artifact["positive_class_ids"]
    if len(positives) < 2:
        return []
    pair_regions = {
        offsets: assign_pair_patch_regions_from_counts(
            artifact["patch_label_counts"],
            int(positives[offsets[0]]),
            int(positives[offsets[1]]),
            rho=0.5,
        )["region_codes"]
        for offsets in itertools.combinations(range(len(positives)), 2)
    }
    relative, active_softmax = _derived_feature_probe_maps(artifact)
    layered = [
        ("feature_post", artifact["feature_post_scores"]),
        ("feature_norm", artifact["feature_norm_scores"]),
        ("qk_mean", artifact["qk_mean_scores"]),
        ("qk_head_std", artifact["qk_head_std"]),
        ("attn_c2p_conditional", artifact["attn_c2p_conditional"]),
    ]
    if relative is not None and active_softmax is not None:
        layered.extend(
            [
                ("feature_post_relative", relative),
                ("feature_post_active_softmax", active_softmax),
            ]
        )
    aggregates = (
        ("feature_final_norm", 12, "final_norm", artifact["feature_final_norm_scores"]),
        (
            "attn_official_conditional",
            0,
            "official_last3",
            artifact["attn_official_conditional"],
        ),
        ("attn_mid3_conditional", 0, "mid3", artifact["attn_mid3_conditional"]),
    )
    rows: list[dict[str, object]] = []
    for (offset_a, offset_b), labels in pair_regions.items():
        base = _pair_base(
            model, entry.image_id, positives, offset_a, offset_b, artifact
        )
        eligible = labels != 5
        for signal, maps in layered:
            for layer in range(entry.num_layers):
                cosine = float(
                    artifact["class_token_pairwise_cosine"][layer, offset_a, offset_b]
                )
                for ratio in TOPK_RATIOS:
                    row = {
                        **base,
                        "layer": layer + 1,
                        "layer_label": f"L{layer + 1}",
                        "signal": signal,
                        "signal_role": _signal_role(signal),
                        "topk_ratio": ratio,
                        "class_token_cosine": cosine,
                    }
                    row.update(
                        map_overlap_metrics(
                            maps[layer, offset_a],
                            maps[layer, offset_b],
                            ratio=ratio,
                            eligible=eligible,
                        )
                    )
                    rows.append(row)
        for signal, layer, layer_label, maps in aggregates:
            for ratio in TOPK_RATIOS:
                row = {
                    **base,
                    "layer": layer,
                    "layer_label": layer_label,
                    "signal": signal,
                    "signal_role": _signal_role(signal),
                    "topk_ratio": ratio,
                    "class_token_cosine": float(
                        artifact["class_token_pairwise_cosine"][11, offset_a, offset_b]
                    ),
                }
                row.update(
                    map_overlap_metrics(
                        maps[offset_a], maps[offset_b], ratio=ratio, eligible=eligible
                    )
                )
                rows.append(row)
    return rows


def _class_token_pair_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    positives = artifact["positive_class_ids"]
    rows: list[dict[str, object]] = []
    for offset_a, offset_b in itertools.combinations(range(len(positives)), 2):
        base = _pair_base(
            model, entry.image_id, positives, offset_a, offset_b, artifact
        )
        pair_labels = assign_pair_patch_regions_from_counts(
            artifact["patch_label_counts"],
            int(positives[offset_a]),
            int(positives[offset_b]),
            rho=0.5,
        )["region_codes"]
        eligible = pair_labels != 5
        for layer in range(entry.num_layers):
            left = artifact["feature_post_scores"][layer, offset_a]
            right = artifact["feature_post_scores"][layer, offset_b]
            attention_left = artifact["attn_c2p_conditional"][layer, offset_a]
            attention_right = artifact["attn_c2p_conditional"][layer, offset_b]
            qk_left = artifact["qk_mean_scores"][layer, offset_a]
            qk_right = artifact["qk_mean_scores"][layer, offset_b]
            row = {
                **base,
                "layer": layer + 1,
                "layer_label": f"L{layer + 1}",
                "class_token_cosine": float(
                    artifact["class_token_pairwise_cosine"][layer, offset_a, offset_b]
                ),
            }
            spearman = map_overlap_metrics(left, right, ratio=0.10, eligible=eligible)[
                "spearman"
            ]
            row["feature_post_spearman"] = spearman
            for ratio in TOPK_RATIOS:
                suffix = f"{int(round(100 * ratio)):02d}"
                metrics = map_overlap_metrics(
                    left, right, ratio=ratio, eligible=eligible
                )
                row[f"feature_post_top{suffix}_jaccard"] = metrics["topk_jaccard"]
                row[f"feature_post_top{suffix}_overlap_coefficient"] = metrics[
                    "topk_overlap_coefficient"
                ]
            attention_metrics = map_overlap_metrics(
                attention_left, attention_right, ratio=0.10, eligible=eligible
            )
            qk_metrics = map_overlap_metrics(
                qk_left, qk_right, ratio=0.10, eligible=eligible
            )
            row.update(
                {
                    "attn_c2p_spearman": attention_metrics["spearman"],
                    "attn_c2p_top10_jaccard": attention_metrics["topk_jaccard"],
                    "attn_c2p_top10_overlap_coefficient": attention_metrics[
                        "topk_overlap_coefficient"
                    ],
                    "qk_mean_spearman": qk_metrics["spearman"],
                    "qk_mean_top10_jaccard": qk_metrics["topk_jaccard"],
                    "qk_mean_top10_overlap_coefficient": qk_metrics[
                        "topk_overlap_coefficient"
                    ],
                }
            )
            rows.append(row)
    return rows


def _shared_signal_descriptors(
    artifact: Mapping[str, np.ndarray],
) -> Iterable[tuple[str, int, str, np.ndarray, Optional[np.ndarray], Optional[str]]]:
    relative, active_softmax = _derived_feature_probe_maps(artifact)
    layered = [
        ("feature_post", artifact["feature_post_scores"]),
        ("feature_norm", artifact["feature_norm_scores"]),
        ("qk_mean", artifact["qk_mean_scores"]),
        ("qk_head_std", artifact["qk_head_std"]),
        ("attn_c2p_conditional", artifact["attn_c2p_conditional"]),
    ]
    if relative is not None and active_softmax is not None:
        layered.extend(
            [
                ("feature_post_relative", relative),
                ("feature_post_active_softmax", active_softmax),
            ]
        )
    for signal, maps in layered:
        for layer in range(maps.shape[0]):
            previous = maps[layer - 1] if layer + 1 in (10, 11, 12) else None
            previous_label = f"L{layer}" if previous is not None else None
            yield (
                signal,
                layer + 1,
                f"L{layer + 1}",
                maps[layer],
                previous,
                previous_label,
            )

    for signal, layer, label, maps in (
        ("feature_final_norm", 12, "final_norm", artifact["feature_final_norm_scores"]),
        (
            "attn_official_conditional",
            0,
            "official_last3",
            artifact["attn_official_conditional"],
        ),
        ("attn_mid3_conditional", 0, "mid3", artifact["attn_mid3_conditional"]),
        ("patch_cam", 12, "patch_cam", artifact["patch_cam"]),
        ("c2p_cam", 0, "c2p_cam", artifact["c2p_cam"]),
        ("final_cam", 0, "final_cam", artifact["final_cam"]),
    ):
        yield signal, layer, label, maps, None, None

    diagnostics = (
        ("diagnostic_c2p_cam_l10", 10, artifact["diagnostic_c2p_cam_l10"]),
        ("diagnostic_c2p_cam_l11", 11, artifact["diagnostic_c2p_cam_l11"]),
        ("diagnostic_c2p_cam_l12", 12, artifact["diagnostic_c2p_cam_l12"]),
    )
    previous: Optional[np.ndarray] = None
    previous_label: Optional[str] = None
    for signal, layer, maps in diagnostics:
        yield signal, layer, f"diagnostic_L{layer}", maps, previous, previous_label
        previous = maps
        previous_label = f"diagnostic_L{layer}"
    yield (
        "diagnostic_c2p_cam_mid3",
        0,
        "diagnostic_mid3",
        artifact["diagnostic_c2p_cam_mid3"],
        None,
        None,
    )


def _shared_ownership_rows(
    model: str,
    entry: ManifestEntry,
    artifact: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    positives = artifact["positive_class_ids"]
    if len(positives) < 2:
        return []
    descriptors = list(_shared_signal_descriptors(artifact))
    rows: list[dict[str, object]] = []
    for offset_a, offset_b in itertools.combinations(range(len(positives)), 2):
        base = _pair_base(
            model, entry.image_id, positives, offset_a, offset_b, artifact
        )
        for rho in (0.5, 0.7):
            pair_regions = assign_pair_patch_regions_from_counts(
                artifact["patch_label_counts"],
                int(positives[offset_a]),
                int(positives[offset_b]),
                rho=rho,
            )["region_codes"]
            for signal, layer, label, maps, previous, previous_label in descriptors:
                for ratio in TOPK_RATIOS:
                    row = {
                        **base,
                        "layer": layer,
                        "layer_or_stage": label,
                        "signal": signal,
                        "signal_role": _signal_role(signal),
                        "rho": rho,
                        "previous_layer_or_stage": previous_label,
                        "new_shared_transition": (
                            f"{previous_label}_to_{label}"
                            if previous_label is not None
                            else None
                        ),
                    }
                    row.update(
                        shared_support_metrics(
                            maps[offset_a],
                            maps[offset_b],
                            pair_regions,
                            ratio=ratio,
                            previous_scores_a=(
                                previous[offset_a] if previous is not None else None
                            ),
                            previous_scores_b=(
                                previous[offset_b] if previous is not None else None
                            ),
                        )
                    )
                    # Preserve the plan's concise BG spelling alongside the
                    # metrics module's semantically explicit name.
                    row["shared_bg_fraction"] = row["shared_background_fraction"]
                    row["shared_bg_enrichment"] = row["shared_background_enrichment"]
                    row["new_shared_bg_fraction"] = row[
                        "new_shared_background_fraction"
                    ]
                    rows.append(row)
    return rows


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_index_row(
    source: SignalRoot,
    entry: ManifestEntry,
    snapshot: Mapping[str, FileSnapshot],
) -> dict[str, object]:
    artifact_snapshot = snapshot[entry.relative_path]
    return {
        "model": source.model,
        "image_id": entry.image_id,
        "signal_root": str(source.root),
        "manifest_path": str(source.manifest_path),
        "manifest_line": entry.line_number,
        "artifact_path": str(entry.artifact_path),
        "artifact_relative_path": entry.relative_path,
        "artifact_size_bytes": artifact_snapshot.size_bytes,
        "artifact_mtime_ns": artifact_snapshot.mtime_ns,
        "artifact_sha256": artifact_snapshot.sha256,
        "manifest_artifact_sha256": entry.artifact_sha256,
        "hash_verified": bool(artifact_snapshot.sha256 == entry.artifact_sha256),
        "source_unchanged": True,
        "positive_class_ids_json": json.dumps(entry.positive_class_ids),
        "num_positive_classes": len(entry.positive_class_ids),
        "grid_h": entry.grid_h,
        "grid_w": entry.grid_w,
        "num_layers": entry.num_layers,
        "num_patches": entry.num_patches,
        "metadata_path": str(source.metadata_path),
        "metadata_sha256": snapshot[
            str(source.metadata_path.relative_to(source.root))
        ].sha256,
        "completion_path": str(source.completion_path),
        "completion_sha256": snapshot[
            str(source.completion_path.relative_to(source.root))
        ].sha256,
    }


def _tree_digest(snapshot: Mapping[str, FileSnapshot]) -> str:
    digest = hashlib.sha256()
    for relative, item in sorted(snapshot.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_metadata(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def build_canonical_tables(
    signal_roots: Mapping[str, Path],
    output_dir: Path,
    *,
    require_full: bool = True,
    expected_layers: int = 12,
    expected_grid: Optional[tuple[int, int]] = (28, 28),
    flush_rows: int = 10_000,
    command: Optional[str] = None,
) -> dict[str, object]:
    """Build all canonical tables from two exact-matched signal roots."""

    if set(signal_roots) != set(MODEL_KEYS):
        raise ValueError(f"signal_roots must contain exactly {MODEL_KEYS}")
    sources = {
        model: load_signal_root(model, signal_roots[model], require_full=require_full)
        for model in MODEL_KEYS
    }
    assert_exact_manifest_match(sources["mctformer"], sources["mctformer_plus"])
    output_dir = Path(output_dir).expanduser().resolve()
    assert_output_outside_inputs(
        output_dir, [source.root for source in sources.values()]
    )
    before = {model: snapshot_root(source.root) for model, source in sources.items()}
    for model, source in sources.items():
        for entry in source.entries:
            captured = before[model].get(entry.relative_path)
            if captured is None or captured.sha256 != entry.artifact_sha256:
                raise RuntimeError(
                    f"artifact changed between manifest validation and source snapshot: "
                    f"{entry.artifact_path}"
                )
    writer = StreamingCanonicalWriter(output_dir, flush_rows=flush_rows)
    patch_count_digests: dict[str, str] = {}
    processed = 0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        for model in MODEL_KEYS:
            source = sources[model]
            for entry in source.entries:
                artifact = load_and_validate_artifact(
                    entry,
                    expected_layers=expected_layers,
                    expected_grid=expected_grid,
                )
                count_digest = _array_digest(artifact["patch_label_counts"])
                if model == "mctformer":
                    patch_count_digests[entry.image_id] = count_digest
                elif patch_count_digests.get(entry.image_id) != count_digest:
                    raise RuntimeError(
                        f"cross-model patch-label counts differ for {entry.image_id}"
                    )
                writer.append(
                    "per_image_class_layer_signal",
                    _layer_signal_rows(model, entry, artifact),
                )
                writer.append(
                    "per_image_class_cam_stage",
                    _cam_rows(model, entry, artifact),
                )
                writer.append(
                    "per_image_class_stage_transition",
                    _transition_rows(model, entry, artifact),
                )
                writer.append(
                    "per_multilabel_class_pair_layer_signal",
                    _pair_layer_signal_rows(model, entry, artifact),
                )
                writer.append(
                    "per_shared_patch_ownership",
                    _shared_ownership_rows(model, entry, artifact),
                )
                writer.append(
                    "per_class_token_pair_layer",
                    _class_token_pair_rows(model, entry, artifact),
                )
                writer.append(
                    "per_image_classification",
                    _classification_rows(model, entry, artifact),
                )
                writer.append(
                    "per_image_cam_confusion",
                    _cam_confusion_rows(model, entry, artifact),
                )
                processed += 1

        after = {
            model: assert_snapshot_unchanged(source.root, before[model])
            for model, source in sources.items()
        }
        for model in MODEL_KEYS:
            source = sources[model]
            writer.append(
                "source_index",
                [
                    _source_index_row(source, entry, after[model])
                    for entry in source.entries
                ],
            )
        tables = writer.close()
    except Exception:
        writer.abort()
        raise

    metadata: dict[str, object] = {
        "status": "complete",
        "analysis": "experiment2_semantic_ownership_canonical",
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": command,
        "git": git_metadata(REPO_ROOT),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
        },
        "runtime_source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("canonical_io.py").resolve(),
                Path(__file__).with_name("metrics_region.py").resolve(),
                Path(__file__).with_name("metrics_stage_linkage.py").resolve(),
                Path(__file__).with_name("metrics_shared_ownership.py").resolve(),
                Path(__file__).with_name("patch_regions.py").resolve(),
                Path(__file__).with_name("evaluation_metrics.py").resolve(),
            )
        },
        "source_roots": {model: str(source.root) for model, source in sources.items()},
        "source_manifests_exact_match": True,
        "source_immutability_verified": True,
        "source_tree_before_after": {
            model: {
                "num_files": len(before[model]),
                "tree_sha256_before": _tree_digest(before[model]),
                "tree_sha256_after": _tree_digest(after[model]),
            }
            for model in MODEL_KEYS
        },
        "num_manifest_images_per_model": len(sources["mctformer"].entries),
        "num_artifacts_processed": processed,
        "expected_layers": expected_layers,
        "expected_grid": list(expected_grid) if expected_grid is not None else None,
        "rho_values": [0.5, 0.7],
        "topk_ratios": list(TOPK_RATIOS),
        "undefined_auc_policy": "NaN retained; orientation is never flipped",
        "delta_definition": "MCTformer+ minus MCTformer on exact common keys",
        "probe_controls": {
            "feature_post_relative": (
                "multi-label only: S_cj minus max score of other active classes"
            ),
            "feature_post_active_softmax": (
                "multi-label only: softmax across active classes independently at each patch"
            ),
            "patch_norm": "per-layer post-block patch-token L2 norm",
            "feature_patch_norm_joint": (
                "post-block cosine/norm Pearson controls plus L12-style top-10% "
                "background norm mean, enrichment versus all background, and "
                "within-image median/q75 norm fractions; thresholds fixed before "
                "the full run and interpreted only as norm-concentration diagnostics"
            ),
            "feature_final_norm": (
                "analysis-only inherited final LayerNorm; neither native host calls it"
            ),
            "qk_head_region_summary": (
                "head0..head5 target/other_fg/background means plus JSON preserving all heads"
            ),
            "claim_status": "probe controls, not model stages or proposed methods",
        },
        "frozen_checkpoint_evaluation": {
            "classification": "all-20 class-token and native patch-head logits",
            "raw_final_cam": (
                "single-scale transformed 448 crop; bilinear upsampling; per-active-"
                "class min-max; fixed background threshold 0.45; void ignored"
            ),
        },
        "tables": tables,
    }
    _write_metadata(output_dir / "canonical_metadata.json", metadata)
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-signal-root", type=Path, required=True)
    parser.add_argument("--mctformer-plus-signal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=12)
    parser.add_argument("--expected-grid-h", type=int, default=28)
    parser.add_argument("--expected-grid-w", type=int, default=28)
    parser.add_argument("--flush-rows", type=int, default=10_000)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="allow completed smoke roots; production builds must omit this flag",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.expected_layers,
            args.expected_grid_h,
            args.expected_grid_w,
            args.flush_rows,
        )
        < 1
    ):
        parser.error("layer, grid, and flush sizes must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    command = " ".join(shlex.quote(value) for value in sys.argv)
    metadata = build_canonical_tables(
        {
            "mctformer": args.mctformer_signal_root,
            "mctformer_plus": args.mctformer_plus_signal_root,
        },
        args.output_dir,
        require_full=not args.allow_smoke,
        expected_layers=args.expected_layers,
        expected_grid=(args.expected_grid_h, args.expected_grid_w),
        flush_rows=args.flush_rows,
        command=command,
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "output_dir": str(args.output_dir.resolve()),
                "tables": {
                    name: value["rows"] for name, value in metadata["tables"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
