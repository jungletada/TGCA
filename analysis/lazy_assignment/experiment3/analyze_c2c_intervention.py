#!/usr/bin/env python3
"""Analyze Experiment 3 Validation C C2C intervention artifacts offline.

The C runner roots, their linked Experiment 2 signals, checkpoints, and VOC
metadata are immutable inputs.  This analysis verifies every referenced file
before deriving canonical tables and uses whole-image clustered paired
bootstrap intervals for every inferential comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    paired_classification_bootstrap,
)
from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    jaccard,
    region_map_metrics,
    spatial_spearman,
    stable_topk_mask,
)
from analysis.lazy_assignment.experiment2.metrics_stage_linkage import (  # noqa: E402
    stage_transition_metrics,
)
from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (  # noqa: E402
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    assign_pair_patch_regions_from_counts,
)
from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (  # noqa: E402
    ImageBootstrapDraws,
    image_multinomial_draws,
    paired_clustered_mean_summary,
    paired_confusion_metric_summary,
)
from analysis.lazy_assignment.experiment3.c2c_intervention import (  # noqa: E402
    C2C_VARIANT_LAYERS_1BASED,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (  # noqa: E402
    cam_metrics_from_confusion,
    cam_threshold_grid,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    EXPECTED_MULTILABEL_IMAGES,
    EXPECTED_PATCHES,
    EXPECTED_POSITIVE_PAIRS,
    STRICT_TOLERANCE,
    assert_new_output,
    git_state,
    json_dump,
    load_image_labels,
    ordered_val_ids,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)
from analysis.lazy_assignment.experiment3.run_c2c_intervention import (  # noqa: E402
    ALLOW_NAN_SIGNAL_KEYS,
    HEAD_REGION_NAMES,
    LATE_LAYER_NUMBERS,
    SIGNAL_KEYS,
    SOURCE_EQUIVALENCE_FIELDS,
    VARIANT_CODES,
    _validate_payload,
)


ANALYSIS_NAME = "experiment3_validation_c_late_c2c_causal_analysis"
RUN_ANALYSIS_NAME = "experiment3_validation_c_late_c2c_causal_intervention"
MODEL_ORDER = ("mctformer", "mctformer_plus")
PRIMARY_MODEL = "mctformer_plus"
PRIMARY_VARIANT = "C4"
BASELINE_VARIANT = "C0"
CONTRASTS = ("C4", "C1", "C2", "C3", "C5")
CONTRAST_ROLE = {
    "C4": "primary",
    "C1": "structural_negative_control",
    "C2": "secondary",
    "C3": "secondary",
    "C5": "secondary",
}
LABEL_STRATA = ("all", "single_label", "exactly_2_labels", "3plus_labels")
STRATUM_SEED_OFFSET = {
    "all": 0,
    "single_label": 1,
    "exactly_2_labels": 2,
    "3plus_labels": 3,
}
PRIMARY_THRESHOLD = 0.45
PRIMARY_THRESHOLD_INDEX = int(
    np.flatnonzero(cam_threshold_grid() == PRIMARY_THRESHOLD)[0]
)
CONFUSION_ENCODING = "21x21 int64 little-endian C-order"
CORRECTNESS_LEVELS = (
    "both_correct",
    "class_only_correct",
    "patch_only_correct",
    "both_incorrect",
)
REGION_METRICS = (
    "cpim",
    "target_mean",
    "other_fg_mean",
    "bg_mean",
    "target_bg_mean_margin",
    "target_other_mean_margin",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "conditional_bg_mass",
)
PAIR_METRICS = (
    "feature_top10_jaccard",
    "feature_axis_removed_top10_jaccard",
    "attention_top10_jaccard",
    "token_pair_raw_cosine",
    "token_pair_residual_cosine",
)
HEAD_REGION_METRICS = (
    "raw_target_mean",
    "raw_other_fg_mean",
    "raw_bg_mean",
    "raw_target_other_margin",
    "conditional_target_mean",
    "conditional_other_fg_mean",
    "conditional_bg_mean",
    "conditional_target_other_margin",
    "conditional_bg_mass",
)
C2C_METRICS = (
    "pre_offdiag_mass",
    "pre_diagonal_mass",
    "pre_class_mass",
    "post_offdiag_mass",
    "post_diagonal_mass",
    "post_class_mass",
)
TRANSITION_METRICS = (
    "spearman",
    "topk_jaccard",
    "topk_overlap_coefficient",
    "introduced_size",
    "removed_size",
    "survive_target",
    "survive_other_fg",
    "survive_background",
    "destination_retained_target",
    "destination_retained_other_fg",
    "destination_retained_background",
    "introduced_target_fraction",
    "introduced_other_fg_fraction",
    "introduced_background_fraction",
    "removed_target_fraction",
    "removed_other_fg_fraction",
    "removed_background_fraction",
)
SHARED_SUPPORT_METRICS = (
    "shared_set_size",
    "topk_jaccard",
    "topk_overlap_coefficient",
    "shared_target_a_fraction",
    "shared_target_a_enrichment",
    "shared_target_b_fraction",
    "shared_target_b_enrichment",
    "shared_other_fg_fraction",
    "shared_other_fg_enrichment",
    "shared_background_fraction",
    "shared_background_enrichment",
    "shared_mixed_fraction",
    "shared_mixed_enrichment",
)
POSITIVE_RECALL_METRICS = (
    "class_token_positive_recall",
    "patch_head_positive_recall",
)
RUN_MANIFEST_KEYS = {
    "image_id",
    "variant",
    "layers_one_based",
    "positive_class_ids",
    "positive_pair_count",
    "artifact_path",
    "artifact_sha256",
    "source_signal_sha256",
}
SOURCE_REQUIRED_KEYS = {
    "image_id",
    "positive_class_ids",
    "feature_post_scores",
    "attn_c2p_raw",
    "attn_c2p_conditional",
    "class_logits_all",
    "patch_class_logits_all",
    "patch_logits",
    "patch_cam",
    "c2p_cam",
    "final_cam",
}


@dataclass(frozen=True)
class ValidatedRun:
    model: str
    root: Path
    metadata: Mapping[str, object]
    completion: Mapping[str, object]
    manifest: tuple[Mapping[str, object], ...]
    records: Mapping[tuple[str, str], Mapping[str, object]]
    image_ids: tuple[str, ...]
    positives: tuple[tuple[int, ...], ...]
    run_kind: str
    num_heads: int
    source_root: Path
    source_records: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ValidatedInputs:
    runs: Mapping[str, ValidatedRun]
    models: tuple[str, ...]
    image_ids: tuple[str, ...]
    labels: np.ndarray
    all_image_ids: tuple[str, ...]
    source_metadata_path: Path
    source_metadata: Mapping[str, object]
    immutable_paths: tuple[Path, ...]
    control_hashes: Mapping[str, str]
    verified_artifacts: int
    verified_source_artifacts: int


@dataclass(frozen=True)
class CollectedRun:
    classification: pd.DataFrame
    region: pd.DataFrame
    pairs: pd.DataFrame
    head_regions: pd.DataFrame
    c2c: pd.DataFrame
    transitions: pd.DataFrame
    shared_support: pd.DataFrame
    fixed_cam: pd.DataFrame
    fixed_confusions: np.ndarray
    aggregate_confusions: np.ndarray
    class_logits: np.ndarray
    patch_logits: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-plus-run-root", type=Path, required=True)
    parser.add_argument("--mctformer-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow a smoke runner root and, only there, fewer than 5000 repeats.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive")
    if args.bootstrap_seed < 0:
        raise ValueError("--bootstrap-seed must be non-negative")
    return args


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL row {number}: {path}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {number} is not an object: {path}")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"empty JSONL file: {path}")
    return rows


def _resolved_child(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes immutable run root: {relative!r}") from error
    return path


def _label_stratum(count: int) -> str:
    if count == 1:
        return "single_label"
    if count == 2:
        return "exactly_2_labels"
    if count >= 3:
        return "3plus_labels"
    raise ValueError("image has no positive class")


def _correctness_status(class_correct: bool, patch_correct: bool) -> str:
    if class_correct and patch_correct:
        return "both_correct"
    if class_correct:
        return "class_only_correct"
    if patch_correct:
        return "patch_only_correct"
    return "both_incorrect"


def encode_confusion(confusion: np.ndarray) -> bytes:
    value = np.asarray(confusion)
    if value.shape != (21, 21) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("confusion must be integer[21,21]")
    if np.any(value < 0):
        raise ValueError("confusion must be non-negative")
    return value.astype("<i8", copy=False).tobytes(order="C")


def decode_confusion(blob: bytes) -> np.ndarray:
    if not isinstance(blob, bytes) or len(blob) != 21 * 21 * 8:
        raise ValueError("invalid encoded 21x21 int64 confusion")
    return np.frombuffer(blob, dtype="<i8").reshape(21, 21).copy()


def _nested_finite_maximum(value: object, name: str) -> float:
    leaves: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} contains a non-numeric leaf")
        number = float(item)
        if not math.isfinite(number):
            raise RuntimeError(f"{name} contains a non-finite leaf")
        leaves.append(abs(number))

    visit(value)
    if not leaves:
        raise RuntimeError(f"{name} contains no numeric leaves")
    return max(leaves)


def _validate_run_metadata(
    model: str,
    root: Path,
) -> tuple[dict[str, object], dict[str, object], str, int]:
    metadata = read_json(root / "metadata.json")
    completion = read_json(root / "completion.json")
    if (
        metadata.get("status") != "complete"
        or completion.get("status") != "complete"
        or metadata.get("analysis") != RUN_ANALYSIS_NAME
        or completion.get("analysis") != RUN_ANALYSIS_NAME
        or metadata.get("model") != model
        or completion.get("model") != model
    ):
        raise RuntimeError(f"incomplete or misidentified Validation C run: {root}")
    run_kind = str(metadata.get("run_kind", ""))
    if run_kind not in {"smoke", "full"} or completion.get("run_kind") != run_kind:
        raise RuntimeError(f"invalid run kind linkage: {root}")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise TypeError(f"Validation C metadata lacks execution record: {root}")
    count = int(execution.get("requested_images", -1))
    if (
        int(execution.get("batch_size", -1)) != 8
        or tuple(execution.get("variant_order", ())) != VARIANT_CODES
        or int(metadata.get("processed_images", -1)) != count
        or int(completion.get("num_images", -1)) != count
        or count < 1
    ):
        raise RuntimeError(f"Validation C execution/count contract failed: {root}")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping) or (
        int(dataset.get("input_size", -1)) != 448
        or int(dataset.get("patch_size", -1)) != 16
    ):
        raise RuntimeError(f"Validation C geometry contract failed: {root}")
    contract = metadata.get("intervention_contract")
    if not isinstance(contract, Mapping):
        raise TypeError(f"Validation C intervention contract missing: {root}")
    variants = contract.get("variant_layers_one_based")
    expected_variants = {
        key: list(value) for key, value in C2C_VARIANT_LAYERS_1BASED.items()
    }
    if (
        variants != expected_variants
        or tuple(contract.get("late_layers_one_based", ())) != LATE_LAYER_NUMBERS
        or tuple(contract.get("head_region_order", ())) != HEAD_REGION_NAMES
        or tuple(contract.get("region_rhos", ())) != (0.5, 0.7)
        or not np.array_equal(contract.get("thresholds"), cam_threshold_grid())
        or float(contract.get("primary_cam_threshold", math.nan)) != PRIMARY_THRESHOLD
        or contract.get("mode") != "mass_preserving_self_reroute"
        or contract.get("normalization") != "vanilla global softmax"
    ):
        raise RuntimeError(f"Validation C intervention contract drifted: {root}")
    if metadata.get("derived_artifact_integrity_passed") is not True:
        raise RuntimeError(f"runner derived-artifact verification did not pass: {root}")
    if metadata.get("source_integrity_passed") is not True:
        raise RuntimeError(f"runner source verification did not pass: {root}")
    if set(metadata.get("signal_schema", ())) != SIGNAL_KEYS:
        raise RuntimeError(f"runner signal schema metadata drifted: {root}")
    for key in (
        "native_cam_max_abs_diff_by_variant",
        "c0_experiment2_max_abs_diff",
        "negative_control_max_abs_diff",
        "head_mean_c2p_max_abs_diff",
        "intervention_capture_max_abs_diff",
    ):
        if _nested_finite_maximum(metadata.get(key), key) >= STRICT_TOLERANCE:
            raise RuntimeError(f"runner numerical gate failed for {key}: {root}")
    axis_difference = float(metadata.get("axis_raw_feature_max_abs_diff", math.inf))
    if not math.isfinite(axis_difference) or axis_difference > 5e-6:
        raise RuntimeError(f"runner alternate-cosine gate failed: {root}")
    return metadata, completion, run_kind, count


def _source_manifest(
    root: Path,
    control_hashes: Mapping[str, object],
    *,
    expected_images: int,
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    paths = {
        "metadata.json": root / "metadata.json",
        "completion.json": root / "completion.json",
        "manifest.jsonl": root / "manifest.jsonl",
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        expected = str(control_hashes.get(name, ""))
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"linked Experiment 2 control hash mismatch: {path}")
        hashes[str(path)] = actual
    rows = _read_jsonl(paths["manifest.jsonl"])
    records: dict[str, Mapping[str, object]] = {}
    for number, row in enumerate(rows, 1):
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in records:
            raise RuntimeError(
                f"invalid/duplicate Experiment 2 manifest row {number}: {root}"
            )
        records[image_id] = row
    if len(records) != expected_images:
        raise RuntimeError(
            f"Experiment 2 manifest has {len(records)} images, expected "
            f"{expected_images}: {root}"
        )
    return records, hashes


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _expected_global_order(
    image_ids: Sequence[str], records: Mapping[tuple[str, str], Mapping[str, object]]
) -> list[Mapping[str, object]]:
    output: list[Mapping[str, object]] = []
    for start in range(0, len(image_ids), 8):
        batch = image_ids[start : start + 8]
        for variant in VARIANT_CODES:
            output.extend(records[(variant, image_id)] for image_id in batch)
    return output


def _native_stage_maps(
    model: str, payload: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the exact native patch, pre-propagation, and final CAM maps."""

    patch_logits = np.asarray(payload["patch_head_logits_positive"])
    attention = np.asarray(payload["attention_c2p_raw_l10_l12"])
    final = np.asarray(payload["final_cam"])
    if (
        patch_logits.ndim != 2
        or patch_logits.shape[1] != EXPECTED_PATCHES
        or attention.shape != (3, *patch_logits.shape)
        or final.shape != patch_logits.shape
        or patch_logits.dtype != np.float32
        or attention.dtype != np.float32
        or final.dtype != np.float32
    ):
        raise RuntimeError("native CAM stage schema/shape/dtype differs")
    patch_cam = np.maximum(patch_logits, np.float32(0.0))
    if model == "mctformer_plus":
        attention_aggregate = attention.mean(axis=0, dtype=np.float32)
        preprop = np.sqrt(attention_aggregate * patch_cam)
    elif model == "mctformer":
        attention_aggregate = attention.sum(axis=0, dtype=np.float32)
        preprop = attention_aggregate * patch_cam
    else:
        raise ValueError(f"unsupported native CAM host: {model}")
    if not all(np.isfinite(value).all() for value in (patch_cam, preprop, final)):
        raise RuntimeError("native CAM stage reconstruction is non-finite")
    return patch_cam, preprop, final


def _validate_source_equivalence(
    model: str,
    c0: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    positive = np.asarray(c0["positive_class_ids"])
    patch_cam, preprop, _ = _native_stage_maps(model, c0)
    checks = {
        "feature_post_scores": (
            c0["feature_post_l10_l12"],
            source["feature_post_scores"][-3:],
        ),
        "attention_raw": (
            c0["attention_c2p_raw_l10_l12"],
            source["attn_c2p_raw"][-3:],
        ),
        "attention_conditional": (
            c0["attention_c2p_conditional_l10_l12"],
            source["attn_c2p_conditional"][-3:],
        ),
        "class_logits_all": (c0["class_logits_all"], source["class_logits_all"]),
        "patch_class_logits_all": (
            c0["patch_class_logits_all"],
            source["patch_class_logits_all"],
        ),
        "patch_head_logits_positive": (
            c0["patch_head_logits_positive"],
            source["patch_logits"],
        ),
        "patch_cam": (patch_cam, source["patch_cam"]),
        "c2p_cam": (preprop, source["c2p_cam"]),
        "final_cam": (c0["final_cam"], source["final_cam"]),
    }
    if not np.array_equal(source["positive_class_ids"], positive):
        raise RuntimeError(f"C0/Experiment 2 positive classes differ: {path}")
    for name in (*SOURCE_EQUIVALENCE_FIELDS, "patch_cam", "c2p_cam"):
        left, right = checks[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"C0/Experiment 2 {name} metadata differ: {path}")
        difference = float(np.max(np.abs(left.astype(np.float64) - right)))
        if difference >= STRICT_TOLERANCE:
            raise RuntimeError(
                f"C0/Experiment 2 {name} differs by {difference}: {path}"
            )


def _validate_direct_negative_controls(
    payloads: Mapping[str, Mapping[str, np.ndarray]], image_id: str
) -> None:
    fields = (
        "patch_class_logits_all",
        "patch_head_logits_positive",
        "attention_c2p_raw_l10_l12",
        "attention_c2p_conditional_l10_l12",
        "final_cam",
    )
    for baseline, comparison in (("C0", "C1"), ("C4", "C5")):
        for field in fields:
            left = payloads[baseline][field]
            right = payloads[comparison][field]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(
                    f"{comparison}-{baseline} {field} metadata differs: {image_id}"
                )
            difference = float(np.max(np.abs(left.astype(np.float64) - right)))
            if difference >= STRICT_TOLERANCE:
                raise RuntimeError(
                    f"{comparison}-{baseline} {field} differs by {difference}: "
                    f"{image_id}"
                )


def _validate_run(
    model: str,
    root_value: Path,
    *,
    all_image_ids: Sequence[str],
    labels: np.ndarray,
    source_metadata_path: Path,
) -> tuple[ValidatedRun, dict[str, str], int, int]:
    root = root_value.expanduser().resolve()
    metadata, completion, run_kind, count = _validate_run_metadata(model, root)
    if Path(str(metadata.get("source_metadata", ""))).resolve() != source_metadata_path:
        raise RuntimeError(f"runner source-metadata path differs: {root}")
    if sha256_file(source_metadata_path) != str(
        metadata.get("source_metadata_sha256", "")
    ):
        raise RuntimeError(f"runner source-metadata hash differs: {root}")
    expected_checkpoint = metadata.get("checkpoint")
    if not isinstance(expected_checkpoint, Mapping):
        raise TypeError(f"runner checkpoint metadata missing: {root}")
    checkpoint_path = Path(str(expected_checkpoint.get("path", ""))).resolve()
    if sha256_file(checkpoint_path) != str(expected_checkpoint.get("sha256", "")):
        raise RuntimeError(f"runner checkpoint hash differs: {root}")

    source_root = Path(str(metadata.get("experiment2_signal_root", ""))).resolve()
    source_controls = metadata.get("experiment2_signal_control_sha256")
    if not isinstance(source_controls, Mapping):
        raise TypeError(f"runner Experiment 2 control hashes missing: {root}")
    source_records, source_hashes = _source_manifest(
        source_root,
        source_controls,
        expected_images=len(all_image_ids),
    )

    manifest_path = root / "manifest.jsonl"
    global_rows = _read_jsonl(manifest_path)
    if len(global_rows) != count * len(VARIANT_CODES):
        raise RuntimeError(f"global Validation C manifest count differs: {root}")
    records: dict[tuple[str, str], Mapping[str, object]] = {}
    for number, row in enumerate(global_rows, 1):
        if set(row) != RUN_MANIFEST_KEYS:
            raise RuntimeError(f"global manifest row {number} schema differs: {root}")
        image_id = str(row["image_id"])
        variant = str(row["variant"])
        key = (variant, image_id)
        if variant not in VARIANT_CODES or not image_id or key in records:
            raise RuntimeError(
                f"invalid/duplicate global manifest row {number}: {root}"
            )
        records[key] = row

    c0_rows = [row for row in global_rows if row["variant"] == BASELINE_VARIANT]
    image_ids = tuple(str(row["image_id"]) for row in c0_rows)
    if len(image_ids) != count or len(set(image_ids)) != count:
        raise RuntimeError(f"C0 membership is incomplete/duplicated: {root}")
    expected_ids = tuple(all_image_ids[:count])
    if image_ids != expected_ids:
        raise RuntimeError(f"Validation C image order is not the VOC prefix: {root}")
    if run_kind == "full":
        if count != EXPECTED_IMAGES:
            raise RuntimeError(f"full Validation C run has {count} images: {root}")
    elif count >= EXPECTED_IMAGES:
        raise RuntimeError(f"smoke Validation C run is not a strict prefix: {root}")

    label_lookup = {
        image_id: labels[index] for index, image_id in enumerate(all_image_ids)
    }
    positives = tuple(
        tuple(np.flatnonzero(label_lookup[image_id] > 0).astype(int).tolist())
        for image_id in image_ids
    )
    expected_global = _expected_global_order(image_ids, records)
    if global_rows != expected_global:
        raise RuntimeError(f"global Validation C manifest order differs: {root}")
    derived = metadata.get("derived_artifact_verification")
    if not isinstance(derived, Mapping):
        raise TypeError(f"runner derived verification metadata missing: {root}")
    if int(derived.get("verified_artifact_count", -1)) != len(
        global_rows
    ) or sha256_file(manifest_path) != str(derived.get("global_manifest_sha256", "")):
        raise RuntimeError(f"runner global manifest verification differs: {root}")

    variant_hashes = derived.get("variant_manifest_sha256")
    if not isinstance(variant_hashes, Mapping):
        raise TypeError(f"runner variant manifest hashes missing: {root}")
    for variant in VARIANT_CODES:
        path = root / "signals" / variant / "manifest.jsonl"
        rows = _read_jsonl(path)
        expected_rows = [records[(variant, image_id)] for image_id in image_ids]
        if rows != expected_rows or sha256_file(path) != str(
            variant_hashes.get(variant, "")
        ):
            raise RuntimeError(f"{variant} manifest content/hash differs: {root}")

    structural_json = _resolved_child(root, metadata.get("structural_records_json"))
    structural_csv = _resolved_child(root, metadata.get("structural_records_csv"))
    if sha256_file(structural_json) != str(
        metadata.get("structural_records_json_sha256", "")
    ) or sha256_file(structural_csv) != str(
        metadata.get("structural_records_csv_sha256", "")
    ):
        raise RuntimeError(f"runner structural record hash differs: {root}")
    structural = read_json(structural_json)
    expected_batches = int(math.ceil(count / 8))
    if (
        len(structural.get("records", ()))
        != int(metadata.get("structural_record_count", -1))
        or len(structural.get("activation_audits", ()))
        != int(metadata.get("activation_audit_count", -1))
        or int(metadata.get("structural_record_count", -1)) != expected_batches * 8
        or int(metadata.get("activation_audit_count", -1)) != expected_batches * 6
    ):
        raise RuntimeError(f"runner structural record counts differ: {root}")

    artifact_count = 0
    source_count = 0
    num_heads: int | None = None
    for image_id, expected_positive in zip(image_ids, positives):
        payloads: dict[str, Mapping[str, np.ndarray]] = {}
        source_record = source_records.get(image_id)
        if not isinstance(source_record, Mapping):
            raise RuntimeError(f"image absent from Experiment 2 manifest: {image_id}")
        source_relative = source_record.get("signal_path")
        source_path = _resolved_child(source_root, source_relative)
        source_digest = sha256_file(source_path)
        if source_digest != str(source_record.get("artifact_sha256", "")):
            raise RuntimeError(
                f"Experiment 2 source artifact hash differs: {source_path}"
            )
        source = _load_npz(source_path)
        if not SOURCE_REQUIRED_KEYS.issubset(source):
            raise RuntimeError(
                f"Experiment 2 source schema is incomplete: {source_path}"
            )
        if str(source["image_id"].item()) != image_id:
            raise RuntimeError(
                f"Experiment 2 source image identity differs: {source_path}"
            )
        source_count += 1
        for variant in VARIANT_CODES:
            row = records[(variant, image_id)]
            if (
                row["layers_one_based"] != list(C2C_VARIANT_LAYERS_1BASED[variant])
                or row["positive_class_ids"] != list(expected_positive)
                or int(row["positive_pair_count"])
                != len(expected_positive) * (len(expected_positive) - 1) // 2
                or str(row["source_signal_sha256"]) != source_digest
            ):
                raise RuntimeError(
                    f"manifest identity/linkage differs: {variant}/{image_id}"
                )
            expected_relative = Path("signals") / variant / f"{image_id}.npz"
            if str(row["artifact_path"]) != str(expected_relative):
                raise RuntimeError(
                    f"manifest artifact path differs: {variant}/{image_id}"
                )
            path = root / expected_relative
            if sha256_file(path) != str(row["artifact_sha256"]):
                raise RuntimeError(f"Validation C artifact hash differs: {path}")
            payload = _load_npz(path)
            if set(payload) != SIGNAL_KEYS:
                raise RuntimeError(f"Validation C artifact schema differs: {path}")
            heads = int(payload["c2c_pre_offdiag_mass"].shape[1])
            if num_heads is None:
                num_heads = heads
            elif num_heads != heads:
                raise RuntimeError(f"attention head count differs: {path}")
            _validate_payload(payload, num_heads=heads)
            if (
                str(payload["image_id"].item()) != image_id
                or str(payload["variant_code"].item()) != variant
                or not np.array_equal(
                    payload["positive_class_ids"],
                    np.asarray(expected_positive, dtype=np.int64),
                )
                or str(payload["source_signal_sha256"].item()) != source_digest
            ):
                raise RuntimeError(f"Validation C artifact identity differs: {path}")
            for key, value in payload.items():
                if value.dtype.kind in "fc":
                    valid = (
                        not np.isinf(value).any()
                        if key in ALLOW_NAN_SIGNAL_KEYS
                        else np.isfinite(value).all()
                    )
                    if not valid:
                        raise RuntimeError(
                            f"non-finite Validation C artifact: {path}:{key}"
                        )
            payloads[variant] = payload
            artifact_count += 1
        _validate_source_equivalence(model, payloads["C0"], source, source_path)
        _validate_direct_negative_controls(payloads, image_id)

    if num_heads is None:
        raise RuntimeError(f"Validation C run contains no artifacts: {root}")
    expected_saved = {variant: count for variant in VARIANT_CODES}
    if (
        metadata.get("saved_by_variant") != expected_saved
        or int(metadata.get("saved_artifacts", -1)) != count * len(VARIANT_CODES)
        or int(metadata.get("source_signal_hashes_verified", -1)) != count
    ):
        raise RuntimeError(f"runner saved/source counts differ: {root}")
    controls = {
        str(root / "metadata.json"): sha256_file(root / "metadata.json"),
        str(root / "completion.json"): sha256_file(root / "completion.json"),
        str(manifest_path): sha256_file(manifest_path),
        str(structural_json): sha256_file(structural_json),
        str(structural_csv): sha256_file(structural_csv),
        str(checkpoint_path): sha256_file(checkpoint_path),
        **source_hashes,
    }
    for variant in VARIANT_CODES:
        path = root / "signals" / variant / "manifest.jsonl"
        controls[str(path)] = sha256_file(path)
    return (
        ValidatedRun(
            model=model,
            root=root,
            metadata=metadata,
            completion=completion,
            manifest=tuple(global_rows),
            records=records,
            image_ids=image_ids,
            positives=positives,
            run_kind=run_kind,
            num_heads=num_heads,
            source_root=source_root,
            source_records=source_records,
        ),
        controls,
        artifact_count,
        source_count,
    )


def validate_inputs(args: argparse.Namespace) -> ValidatedInputs:
    plus_root = args.mctformer_plus_run_root.expanduser().resolve()
    plus_metadata = read_json(plus_root / "metadata.json")
    source_metadata_path = Path(str(plus_metadata.get("source_metadata", ""))).resolve()
    source_metadata = read_json(source_metadata_path)
    if (
        source_metadata.get("status") != "complete"
        or source_metadata.get("integrity_passed") is not True
    ):
        raise RuntimeError("Experiment 3 source audit is not complete/passed")
    if sha256_file(source_metadata_path) != str(
        plus_metadata.get("source_metadata_sha256", "")
    ):
        raise RuntimeError("MCTformer+ source metadata hash differs")
    dataset = source_metadata.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("Experiment 3 source audit lacks dataset metadata")
    all_image_ids = tuple(
        ordered_val_ids(Path(str(dataset.get("list_path", ""))).resolve())
    )
    labels = load_image_labels(
        Path(str(dataset.get("labels_path", ""))).resolve(), all_image_ids
    )
    if (
        len(all_image_ids) != int(dataset.get("num_images", -1))
        or int(labels.sum()) != int(dataset.get("positive_image_class_pairs", -1))
        or int(np.count_nonzero(labels.sum(axis=1) >= 2))
        != int(dataset.get("multilabel_images", -1))
        or int(dataset.get("input_size", -1)) != 448
        or int(dataset.get("patch_size", -1)) != 16
    ):
        raise RuntimeError("source-audit dataset cardinality/geometry differs")

    requested_roots: dict[str, Path] = {PRIMARY_MODEL: plus_root}
    if args.mctformer_run_root is not None:
        requested_roots["mctformer"] = args.mctformer_run_root.expanduser().resolve()
    models = tuple(model for model in MODEL_ORDER if model in requested_roots)
    runs: dict[str, ValidatedRun] = {}
    controls: dict[str, str] = {
        str(source_metadata_path): sha256_file(source_metadata_path),
        str(Path(str(dataset["list_path"])).resolve()): sha256_file(
            Path(str(dataset["list_path"])).resolve()
        ),
        str(Path(str(dataset["labels_path"])).resolve()): sha256_file(
            Path(str(dataset["labels_path"])).resolve()
        ),
    }
    immutable_paths = [source_metadata_path]
    artifact_count = 0
    source_count = 0
    for model in models:
        run, run_controls, verified, verified_sources = _validate_run(
            model,
            requested_roots[model],
            all_image_ids=all_image_ids,
            labels=labels,
            source_metadata_path=source_metadata_path,
        )
        runs[model] = run
        overlap = set(controls).intersection(run_controls)
        for key in overlap:
            if controls[key] != run_controls[key]:
                raise RuntimeError(f"input control hash conflicts: {key}")
        controls.update(run_controls)
        immutable_paths.extend((run.root, run.source_root))
        artifact_count += verified
        source_count += verified_sources

    plus = runs[PRIMARY_MODEL]
    selected_labels = np.stack(
        [labels[all_image_ids.index(image_id)] for image_id in plus.image_ids]
    )
    for model, run in runs.items():
        if (
            run.run_kind != plus.run_kind
            or run.image_ids != plus.image_ids
            or run.positives != plus.positives
        ):
            raise RuntimeError(
                f"optional {model} run is not exactly paired with MCTformer+"
            )
        linked_source = Path(str(run.metadata.get("source_metadata", ""))).resolve()
        if linked_source != source_metadata_path:
            raise RuntimeError(f"{model} uses a different source audit")
    if plus.run_kind == "full":
        if (
            len(plus.image_ids) != EXPECTED_IMAGES
            or sum(len(value) for value in plus.positives) != EXPECTED_POSITIVE_PAIRS
            or sum(len(value) >= 2 for value in plus.positives)
            != EXPECTED_MULTILABEL_IMAGES
        ):
            raise RuntimeError("full Validation C cardinality gate failed")
    return ValidatedInputs(
        runs=runs,
        models=models,
        image_ids=plus.image_ids,
        labels=selected_labels,
        all_image_ids=all_image_ids,
        source_metadata_path=source_metadata_path,
        source_metadata=source_metadata,
        immutable_paths=tuple(dict.fromkeys(immutable_paths)),
        control_hashes=controls,
        verified_artifacts=artifact_count,
        verified_source_artifacts=source_count,
    )


def _map_metric_record(
    values: np.ndarray, regions: np.ndarray, *, mass: bool
) -> dict[str, float]:
    result = region_map_metrics(
        values,
        regions,
        grid_h=28,
        grid_w=28,
        nonnegative_mass=mass,
    )
    conditional_bg_mass = float("nan")
    if mass:
        complete_map = np.asarray(values, dtype=np.float64).reshape(-1)
        region_codes = np.asarray(regions).reshape(-1)
        if (
            complete_map.size != EXPECTED_PATCHES
            or region_codes.shape != complete_map.shape
        ):
            raise ValueError("conditional mass map and region geometry differ")
        if float(complete_map.min()) < -1e-12:
            raise ValueError("conditional mass requires a nonnegative complete map")
        complete_mass = float(complete_map.sum())
        conditional_bg_mass = (
            float(complete_map[region_codes == 2].sum() / complete_mass)
            if complete_mass > 1e-12
            else float("nan")
        )
    return {
        "cpim": float(bool(result["target_hit"])),
        "target_mean": float(result["target_mean"]),
        "other_fg_mean": float(result["other_fg_mean"]),
        "bg_mean": float(result["bg_mean"]),
        "target_bg_mean_margin": float(result["target_bg_mean_margin"]),
        "target_other_mean_margin": float(result["target_other_mean_margin"]),
        "target_top10_fraction": float(result["target_top10_fraction"]),
        "other_fg_top10_fraction": float(result["other_fg_top10_fraction"]),
        "bg_top10_fraction": float(result["bg_top10_fraction"]),
        "auc_target_bg": float(result["auc_target_bg"]),
        "ap_target_bg": float(result["ap_target_bg"]),
        "auc_target_other": float(result["auc_target_other"]),
        "ap_target_other": float(result["ap_target_other"]),
        # Experiment 2 contract: denominator includes all 784 patches,
        # including mixed and void, rather than only non-void patches.
        "conditional_bg_mass": conditional_bg_mass,
    }


def _pair_jaccard(left: np.ndarray, right: np.ndarray, eligible: np.ndarray) -> float:
    return jaccard(
        stable_topk_mask(left, 0.10, eligible),
        stable_topk_mask(right, 0.10, eligible),
    )


def _stage_transition_record(
    source: np.ndarray,
    destination: np.ndarray,
    regions: np.ndarray,
) -> dict[str, float | int]:
    result = stage_transition_metrics(
        source,
        destination,
        regions,
        ratio=0.10,
    )
    region_codes = np.asarray(regions).reshape(-1)
    eligible = region_codes != 4
    source_flat = np.asarray(source).reshape(-1)
    destination_flat = np.asarray(destination).reshape(-1)
    # The shared Experiment 2 helper limits top-k selection but computes its
    # correlation globally. Validation C uses the same non-void support for
    # both types of transition statistic.
    result["spearman"] = spatial_spearman(
        source_flat[eligible], destination_flat[eligible]
    )
    return result


def _classification_statuses(
    c0: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = np.asarray(c0["image_labels"], dtype=np.uint8)
    class_correct = (np.asarray(c0["class_logits_all"]) >= 0) == labels.astype(bool)
    patch_correct = (np.asarray(c0["patch_class_logits_all"]) >= 0) == labels.astype(
        bool
    )
    statuses = [
        _correctness_status(bool(class_correct[index]), bool(patch_correct[index]))
        for index in range(len(labels))
    ]
    return class_correct, patch_correct, statuses


def _collect_run(run: ValidatedRun) -> CollectedRun:
    classification_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    c2c_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    shared_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    num_images = len(run.image_ids)
    num_variants = len(VARIANT_CODES)
    fixed_confusions = np.zeros((num_images, num_variants, 21, 21), dtype=np.int64)
    aggregate = np.zeros((len(LABEL_STRATA), num_variants, 41, 21, 21), dtype=np.int64)
    class_logits = np.empty((num_variants, num_images, 20), dtype=np.float32)
    patch_logits = np.empty_like(class_logits)

    for image_offset, (image_id, expected_positive) in enumerate(
        zip(run.image_ids, run.positives)
    ):
        payloads = {
            variant: _load_npz(
                run.root / str(run.records[(variant, image_id)]["artifact_path"])
            )
            for variant in VARIANT_CODES
        }
        c0 = payloads[BASELINE_VARIANT]
        positive = np.asarray(expected_positive, dtype=np.int64)
        label_count = len(positive)
        label_stratum = _label_stratum(label_count)
        class_correct, patch_correct, statuses = _classification_statuses(c0)
        pair_ids = np.asarray(c0["pair_class_ids"])

        reference_marginal = None
        for variant_offset, variant in enumerate(VARIANT_CODES):
            payload = payloads[variant]
            patch_cam, preprop_cam, final_cam = _native_stage_maps(run.model, payload)
            if not np.array_equal(payload["pair_class_ids"], pair_ids):
                raise RuntimeError(f"pair order differs: {variant}/{image_id}")
            class_logits[variant_offset, image_offset] = payload["class_logits_all"]
            patch_logits[variant_offset, image_offset] = payload[
                "patch_class_logits_all"
            ]
            for class_id in range(20):
                classification_rows.append(
                    {
                        "image_id": image_id,
                        "model": run.model,
                        "variant_code": variant,
                        "class_id": class_id,
                        "target": int(c0["image_labels"][class_id]),
                        "class_token_logit": float(
                            payload["class_logits_all"][class_id]
                        ),
                        "patch_head_logit": float(
                            payload["patch_class_logits_all"][class_id]
                        ),
                        "class_token_positive_recall": float(
                            payload["class_logits_all"][class_id] >= 0
                        ),
                        "patch_head_positive_recall": float(
                            payload["patch_class_logits_all"][class_id] >= 0
                        ),
                        "label_stratum": label_stratum,
                        "num_positive_classes": label_count,
                        "baseline_class_token_correct": bool(class_correct[class_id]),
                        "baseline_patch_head_correct": bool(patch_correct[class_id]),
                        "baseline_correctness": statuses[class_id],
                    }
                )

            confusions = np.asarray(payload["threshold_confusions"], dtype=np.int64)
            marginal = confusions.sum(axis=2)
            if reference_marginal is None:
                reference_marginal = marginal
            elif not np.array_equal(marginal, reference_marginal):
                raise RuntimeError(
                    f"variant/threshold GT marginals differ: {variant}/{image_id}"
                )
            fixed = confusions[PRIMARY_THRESHOLD_INDEX]
            fixed_confusions[image_offset, variant_offset] = fixed
            for stratum in ("all", label_stratum):
                aggregate[LABEL_STRATA.index(stratum), variant_offset] += confusions
            fixed_rows.append(
                {
                    "image_id": image_id,
                    "model": run.model,
                    "variant_code": variant,
                    "threshold": PRIMARY_THRESHOLD,
                    "label_stratum": label_stratum,
                    "num_positive_classes": label_count,
                    "confusion_encoding": CONFUSION_ENCODING,
                    "confusion": encode_confusion(fixed),
                    "evaluated_pixels": int(fixed.sum()),
                }
            )

            maps = {
                "feature_raw": payload["feature_post_l10_l12"],
                "feature_axis_removed": payload["feature_both_axis_removed_l10_l12"],
                "attention_raw": payload["attention_c2p_raw_l10_l12"],
                "attention_conditional": payload["attention_c2p_conditional_l10_l12"],
            }
            for rho_name, rho_value, regions in (
                ("rho05", 0.5, payload["region_masks_rho05"]),
                ("rho07", 0.7, payload["region_masks_rho07"]),
            ):
                for class_offset, class_id in enumerate(positive):
                    transition_identity = {
                        "image_id": image_id,
                        "model": run.model,
                        "variant_code": variant,
                        "class_id": int(class_id),
                        "rho_name": rho_name,
                        "rho": rho_value,
                        "label_stratum": label_stratum,
                        "num_positive_classes": label_count,
                        "baseline_correctness": statuses[int(class_id)],
                    }
                    for (
                        transition,
                        source_signal,
                        destination_signal,
                        source_map,
                        destination_map,
                    ) in (
                        (
                            "patch_to_preprop",
                            "patch_cam",
                            "class_attention_cam",
                            patch_cam[class_offset],
                            preprop_cam[class_offset],
                        ),
                        (
                            "preprop_to_final",
                            "class_attention_cam",
                            "final_cam",
                            preprop_cam[class_offset],
                            final_cam[class_offset],
                        ),
                    ):
                        transition_rows.append(
                            {
                                **transition_identity,
                                "transition": transition,
                                "source_signal": source_signal,
                                "destination_signal": destination_signal,
                                **_stage_transition_record(
                                    source_map,
                                    destination_map,
                                    regions[class_offset],
                                ),
                            }
                        )
                for layer_offset, layer in enumerate(LATE_LAYER_NUMBERS):
                    for class_offset, class_id in enumerate(positive):
                        identity = {
                            "image_id": image_id,
                            "model": run.model,
                            "variant_code": variant,
                            "layer": layer,
                            "class_id": int(class_id),
                            "rho_name": rho_name,
                            "rho": rho_value,
                            "label_stratum": label_stratum,
                            "num_positive_classes": label_count,
                            "baseline_correctness": statuses[int(class_id)],
                        }
                        for family, values in maps.items():
                            region_rows.append(
                                {
                                    **identity,
                                    "map_family": family,
                                    **_map_metric_record(
                                        values[layer_offset, class_offset],
                                        regions[class_offset],
                                        mass=family == "attention_conditional",
                                    ),
                                }
                            )

                    raw_heads = payload[f"attention_head_region_raw_{rho_name}"][
                        layer_offset
                    ]
                    conditional_heads = payload[
                        f"attention_head_region_conditional_{rho_name}"
                    ][layer_offset]
                    for head in range(run.num_heads):
                        for class_offset, class_id in enumerate(positive):
                            raw_values = raw_heads[head, class_offset]
                            conditional_values = conditional_heads[head, class_offset]
                            bg_count = int(np.count_nonzero(regions[class_offset] == 2))
                            head_rows.append(
                                {
                                    "image_id": image_id,
                                    "model": run.model,
                                    "variant_code": variant,
                                    "layer": layer,
                                    "head": head,
                                    "class_id": int(class_id),
                                    "rho_name": rho_name,
                                    "rho": rho_value,
                                    "label_stratum": label_stratum,
                                    "num_positive_classes": label_count,
                                    "baseline_correctness": statuses[int(class_id)],
                                    "raw_target_mean": float(raw_values[0]),
                                    "raw_other_fg_mean": float(raw_values[1]),
                                    "raw_bg_mean": float(raw_values[2]),
                                    "raw_target_other_margin": float(
                                        raw_values[0] - raw_values[1]
                                    ),
                                    "conditional_target_mean": float(
                                        conditional_values[0]
                                    ),
                                    "conditional_other_fg_mean": float(
                                        conditional_values[1]
                                    ),
                                    "conditional_bg_mean": float(conditional_values[2]),
                                    "conditional_target_other_margin": float(
                                        conditional_values[0] - conditional_values[1]
                                    ),
                                    "conditional_bg_mass": float(
                                        conditional_values[2] * bg_count
                                    ),
                                }
                            )

            for layer_index in range(EXPECTED_LAYERS):
                for head in range(run.num_heads):
                    for class_id in positive:
                        c2c_rows.append(
                            {
                                "image_id": image_id,
                                "model": run.model,
                                "variant_code": variant,
                                "layer": layer_index + 1,
                                "head": head,
                                "class_id": int(class_id),
                                "label_stratum": label_stratum,
                                "num_positive_classes": label_count,
                                "baseline_correctness": statuses[int(class_id)],
                                "pre_offdiag_mass": float(
                                    payload["c2c_pre_offdiag_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                                "pre_diagonal_mass": float(
                                    payload["c2c_pre_diagonal_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                                "pre_class_mass": float(
                                    payload["c2c_pre_class_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                                "post_offdiag_mass": float(
                                    payload["c2c_post_offdiag_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                                "post_diagonal_mass": float(
                                    payload["c2c_post_diagonal_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                                "post_class_mass": float(
                                    payload["c2c_post_class_mass"][
                                        layer_index, head, class_id
                                    ]
                                ),
                            }
                        )

            for pair_offset, (class_a, class_b) in enumerate(pair_ids):
                offset_a = int(np.flatnonzero(positive == class_a)[0])
                offset_b = int(np.flatnonzero(positive == class_b)[0])
                pair_regions_rho05 = np.asarray(
                    assign_pair_patch_regions_from_counts(
                        payload["patch_label_counts"],
                        int(class_a),
                        int(class_b),
                        rho=0.5,
                        grid_size=(28, 28),
                    )["region_codes"]
                )
                eligible = pair_regions_rho05.reshape(-1) != 5
                both_correct = bool(
                    class_correct[class_a]
                    and patch_correct[class_a]
                    and class_correct[class_b]
                    and patch_correct[class_b]
                )
                for layer_offset, layer in enumerate(LATE_LAYER_NUMBERS):
                    pair_rows.append(
                        {
                            "image_id": image_id,
                            "model": run.model,
                            "variant_code": variant,
                            "layer": layer,
                            "class_a": int(class_a),
                            "class_b": int(class_b),
                            "class_pair": f"{int(class_a):02d}-{int(class_b):02d}",
                            "label_stratum": label_stratum,
                            "num_positive_classes": label_count,
                            "class_a_baseline_correctness": statuses[int(class_a)],
                            "class_b_baseline_correctness": statuses[int(class_b)],
                            "pair_baseline_correctness": (
                                "both_classes_both_endpoints_correct"
                                if both_correct
                                else "either_class_or_endpoint_incorrect"
                            ),
                            "feature_top10_jaccard": _pair_jaccard(
                                payload["feature_post_l10_l12"][layer_offset, offset_a],
                                payload["feature_post_l10_l12"][layer_offset, offset_b],
                                eligible,
                            ),
                            "feature_axis_removed_top10_jaccard": _pair_jaccard(
                                payload["feature_both_axis_removed_l10_l12"][
                                    layer_offset, offset_a
                                ],
                                payload["feature_both_axis_removed_l10_l12"][
                                    layer_offset, offset_b
                                ],
                                eligible,
                            ),
                            "attention_top10_jaccard": _pair_jaccard(
                                payload["attention_c2p_conditional_l10_l12"][
                                    layer_offset, offset_a
                                ],
                                payload["attention_c2p_conditional_l10_l12"][
                                    layer_offset, offset_b
                                ],
                                eligible,
                            ),
                            "token_pair_raw_cosine": float(
                                payload["positive_pair_raw_cosine_l10_l12"][
                                    layer_offset, pair_offset
                                ]
                            ),
                            "token_pair_residual_cosine": float(
                                payload["positive_pair_residual_cosine_l10_l12"][
                                    layer_offset, pair_offset
                                ]
                            ),
                        }
                    )

                shared_map_families: list[tuple[str, int, np.ndarray]] = []
                for layer_offset, layer in enumerate(LATE_LAYER_NUMBERS):
                    shared_map_families.extend(
                        (
                            (
                                "feature_raw",
                                layer,
                                payload["feature_post_l10_l12"][layer_offset],
                            ),
                            (
                                "feature_axis_removed",
                                layer,
                                payload["feature_both_axis_removed_l10_l12"][
                                    layer_offset
                                ],
                            ),
                            (
                                "attention_raw",
                                layer,
                                payload["attention_c2p_raw_l10_l12"][layer_offset],
                            ),
                            (
                                "attention_conditional",
                                layer,
                                payload["attention_c2p_conditional_l10_l12"][
                                    layer_offset
                                ],
                            ),
                        )
                    )
                shared_map_families.extend(
                    (
                        ("patch_cam", 0, patch_cam),
                        ("preprop_cam", 0, preprop_cam),
                        ("final_cam", 0, final_cam),
                    )
                )
                for rho_name, rho_value in (("rho05", 0.5), ("rho07", 0.7)):
                    pair_regions = (
                        pair_regions_rho05
                        if rho_name == "rho05"
                        else np.asarray(
                            assign_pair_patch_regions_from_counts(
                                payload["patch_label_counts"],
                                int(class_a),
                                int(class_b),
                                rho=rho_value,
                                grid_size=(28, 28),
                            )["region_codes"]
                        )
                    )
                    for family, layer, family_maps in shared_map_families:
                        metrics = shared_support_metrics(
                            family_maps[offset_a],
                            family_maps[offset_b],
                            pair_regions,
                            ratio=0.10,
                        )
                        shared_rows.append(
                            {
                                "image_id": image_id,
                                "model": run.model,
                                "variant_code": variant,
                                "class_a": int(class_a),
                                "class_b": int(class_b),
                                "class_pair": (
                                    f"{int(class_a):02d}-{int(class_b):02d}"
                                ),
                                "map_family": family,
                                "layer": layer,
                                "layer_label": f"L{layer}" if layer else "aggregate",
                                "rho_name": rho_name,
                                "rho": rho_value,
                                "label_stratum": label_stratum,
                                "num_positive_classes": label_count,
                                "pair_baseline_correctness": (
                                    "both_classes_both_endpoints_correct"
                                    if both_correct
                                    else "either_class_or_endpoint_incorrect"
                                ),
                                **{key: metrics[key] for key in SHARED_SUPPORT_METRICS},
                            }
                        )

    return CollectedRun(
        classification=pd.DataFrame.from_records(classification_rows),
        region=pd.DataFrame.from_records(region_rows),
        pairs=pd.DataFrame.from_records(pair_rows),
        head_regions=pd.DataFrame.from_records(head_rows),
        c2c=pd.DataFrame.from_records(c2c_rows),
        transitions=pd.DataFrame.from_records(transition_rows),
        shared_support=pd.DataFrame.from_records(shared_rows),
        fixed_cam=pd.DataFrame.from_records(fixed_rows),
        fixed_confusions=fixed_confusions,
        aggregate_confusions=aggregate,
        class_logits=class_logits,
        patch_logits=patch_logits,
    )


def _threshold_tables(
    collected: Mapping[str, CollectedRun],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    curve_summary_rows: list[dict[str, object]] = []
    thresholds = cam_threshold_grid()
    scalar_metrics = (
        "mean_iou",
        "binary_foreground_precision",
        "binary_foreground_recall",
        "semantic_correct_foreground_precision",
        "semantic_correct_foreground_recall",
    )
    for model, run in collected.items():
        for stratum_offset, stratum in enumerate(LABEL_STRATA):
            for variant_offset, variant in enumerate(VARIANT_CODES):
                variant_rows: list[dict[str, object]] = []
                for threshold_offset, threshold in enumerate(thresholds):
                    confusion = run.aggregate_confusions[
                        stratum_offset, variant_offset, threshold_offset
                    ]
                    if int(confusion.sum()) == 0:
                        continue
                    metrics = cam_metrics_from_confusion(confusion)
                    row = {
                        "model": model,
                        "label_stratum": stratum,
                        "variant_code": variant,
                        "threshold": float(threshold),
                        **{key: float(metrics[key]) for key in scalar_metrics},
                        "evaluated_pixels": int(confusion.sum()),
                    }
                    curve_rows.append(row)
                    variant_rows.append(row)
                    canonical_rows.append(
                        {
                            "model": model,
                            "label_stratum": stratum,
                            "variant_code": variant,
                            "threshold": float(threshold),
                            "confusion_encoding": CONFUSION_ENCODING,
                            "confusion": encode_confusion(confusion),
                            "evaluated_pixels": int(confusion.sum()),
                        }
                    )
                    for class_id, iou in enumerate(metrics["per_class_iou"]):
                        class_rows.append(
                            {
                                "model": model,
                                "label_stratum": stratum,
                                "variant_code": variant,
                                "threshold": float(threshold),
                                "semantic_class_id": class_id,
                                "iou": float(iou),
                            }
                        )
                if variant_rows:
                    frame = pd.DataFrame.from_records(variant_rows).sort_values(
                        "threshold"
                    )
                    best = frame.iloc[int(np.nanargmax(frame["mean_iou"].to_numpy()))]
                    for metric in scalar_metrics:
                        values = frame[metric].to_numpy(dtype=np.float64)
                        curve_summary_rows.append(
                            {
                                "model": model,
                                "label_stratum": stratum,
                                "variant_code": variant,
                                "metric": metric,
                                "normalized_curve_auc": float(
                                    np.trapz(values, frame["threshold"])
                                    / (thresholds[-1] - thresholds[0])
                                ),
                                "variant_best_mean_iou_diagnostic": float(
                                    best["mean_iou"]
                                ),
                                "variant_best_threshold_diagnostic": float(
                                    best["threshold"]
                                ),
                            }
                        )
    return (
        pd.DataFrame.from_records(curve_rows),
        pd.DataFrame.from_records(class_rows),
        pd.DataFrame.from_records(canonical_rows),
        pd.DataFrame.from_records(curve_summary_rows),
    )


def _point_summary(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    class_col: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_cols, key_tuple))
        for metric in value_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    **identity,
                    "aggregation": "micro",
                    "metric": metric,
                    "estimate": float(finite.mean()) if len(finite) else np.nan,
                    "num_rows": int(len(finite)),
                    "num_images": int(group.loc[finite.index, "image_id"].nunique()),
                    "num_classes": np.nan,
                }
            )
            if class_col is not None:
                class_means = (
                    group.assign(_metric=values)
                    .groupby(class_col, sort=True)["_metric"]
                    .mean()
                )
                finite_classes = class_means[np.isfinite(class_means)]
                rows.append(
                    {
                        **identity,
                        "aggregation": "macro_class",
                        "metric": metric,
                        "estimate": (
                            float(finite_classes.mean())
                            if len(finite_classes)
                            else np.nan
                        ),
                        "num_rows": int(len(finite)),
                        "num_images": int(
                            group.loc[finite.index, "image_id"].nunique()
                        ),
                        "num_classes": int(len(finite_classes)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _stratum_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(labels).sum(axis=1)
    return {
        "all": np.ones(len(counts), dtype=bool),
        "single_label": counts == 1,
        "exactly_2_labels": counts == 2,
        "3plus_labels": counts >= 3,
    }


def _shared_draws(
    image_ids: Sequence[str],
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, ImageBootstrapDraws]:
    ids = np.asarray(image_ids)
    output = {}
    for stratum, mask in _stratum_masks(labels).items():
        if mask.any():
            output[stratum] = image_multinomial_draws(
                ids[mask].tolist(),
                repeats=repeats,
                seed=seed + STRATUM_SEED_OFFSET[stratum],
            )
    return output


def _filter_stratum(frame: pd.DataFrame, stratum: str) -> pd.DataFrame:
    return frame if stratum == "all" else frame[frame["label_stratum"] == stratum]


def _paired_mean(
    frame: pd.DataFrame,
    *,
    comparison: str,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws,
    identity: Mapping[str, object],
    class_col: str = "class_id",
    include_macro_class: bool = True,
) -> pd.DataFrame:
    return paired_clustered_mean_summary(
        frame,
        system_col="variant_code",
        baseline=BASELINE_VARIANT,
        comparison=comparison,
        key_cols=key_cols,
        value_cols=value_cols,
        draws=draws,
        class_col=class_col,
        include_macro_class=include_macro_class,
        identity={
            **identity,
            "comparison_role": CONTRAST_ROLE[comparison],
        },
    )


def _region_bootstrap(
    region: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model in sorted(region["model"].unique()):
        model_frame = region[region["model"] == model]
        for family in (
            "feature_raw",
            "feature_axis_removed",
            "attention_conditional",
        ):
            rho_names = (
                ("rho05", "rho07") if family == "attention_conditional" else ("rho05",)
            )
            strata = LABEL_STRATA if family == "attention_conditional" else ("all",)
            for rho_name in rho_names:
                for layer in LATE_LAYER_NUMBERS:
                    selected_group = model_frame[
                        (model_frame["map_family"] == family)
                        & (model_frame["rho_name"] == rho_name)
                        & (model_frame["layer"] == layer)
                    ]
                    for stratum in strata:
                        selected = _filter_stratum(selected_group, stratum)
                        if selected.empty:
                            continue
                        for comparison in CONTRASTS:
                            outputs.append(
                                _paired_mean(
                                    selected,
                                    comparison=comparison,
                                    key_cols=("image_id", "class_id"),
                                    value_cols=REGION_METRICS,
                                    draws=draws[stratum],
                                    identity={
                                        "model": model,
                                        "summary_stratum": stratum,
                                        "map_family": family,
                                        "layer": layer,
                                        "rho_name": rho_name,
                                    },
                                )
                            )
        correctness_group = model_frame[
            (model_frame["layer"] == 12)
            & (model_frame["rho_name"] == "rho05")
            & model_frame["map_family"].isin(
                ("feature_raw", "feature_axis_removed", "attention_conditional")
            )
        ]
        for family in sorted(correctness_group["map_family"].unique()):
            family_group = correctness_group[correctness_group["map_family"] == family]
            for status in CORRECTNESS_LEVELS:
                selected = family_group[family_group["baseline_correctness"] == status]
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_id"),
                            value_cols=REGION_METRICS,
                            draws=draws["all"],
                            identity={
                                "model": model,
                                "summary_stratum": f"baseline_correctness:{status}",
                                "map_family": family,
                                "layer": 12,
                                "rho_name": "rho05",
                            },
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _pair_bootstrap(
    pairs: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    if pairs.empty:
        return pd.DataFrame()
    for model in sorted(pairs["model"].unique()):
        model_frame = pairs[pairs["model"] == model]
        for layer in LATE_LAYER_NUMBERS:
            layer_frame = model_frame[model_frame["layer"] == layer]
            for stratum in ("all", "exactly_2_labels", "3plus_labels"):
                selected = _filter_stratum(layer_frame, stratum)
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_a", "class_b"),
                            value_cols=PAIR_METRICS,
                            draws=draws[stratum],
                            identity={
                                "model": model,
                                "summary_stratum": stratum,
                                "layer": layer,
                            },
                            class_col="class_pair",
                            include_macro_class=False,
                        )
                    )
        correctness_group = model_frame[model_frame["layer"] == 12]
        for status in (
            "both_classes_both_endpoints_correct",
            "either_class_or_endpoint_incorrect",
        ):
            selected = correctness_group[
                correctness_group["pair_baseline_correctness"] == status
            ]
            if selected.empty:
                continue
            for comparison in CONTRASTS:
                outputs.append(
                    _paired_mean(
                        selected,
                        comparison=comparison,
                        key_cols=("image_id", "class_a", "class_b"),
                        value_cols=PAIR_METRICS,
                        draws=draws["all"],
                        identity={
                            "model": model,
                            "summary_stratum": f"baseline_correctness:{status}",
                            "layer": 12,
                        },
                        class_col="class_pair",
                        include_macro_class=False,
                    )
                )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _head_region_bootstrap(
    head_regions: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model in sorted(head_regions["model"].unique()):
        model_frame = head_regions[
            (head_regions["model"] == model) & (head_regions["rho_name"] == "rho05")
        ]
        for layer in LATE_LAYER_NUMBERS:
            layer_frame = model_frame[model_frame["layer"] == layer]
            strata = LABEL_STRATA if layer == 12 else ("all",)
            for stratum in strata:
                selected = _filter_stratum(layer_frame, stratum)
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_id", "head"),
                            value_cols=HEAD_REGION_METRICS,
                            draws=draws[stratum],
                            identity={
                                "model": model,
                                "summary_stratum": stratum,
                                "layer": layer,
                                "rho_name": "rho05",
                            },
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _c2c_bootstrap(
    c2c: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model in sorted(c2c["model"].unique()):
        model_frame = c2c[c2c["model"] == model]
        for layer in LATE_LAYER_NUMBERS:
            layer_frame = model_frame[model_frame["layer"] == layer]
            strata = LABEL_STRATA if layer == 12 else ("all",)
            for stratum in strata:
                selected = _filter_stratum(layer_frame, stratum)
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_id", "head"),
                            value_cols=C2C_METRICS,
                            draws=draws[stratum],
                            identity={
                                "model": model,
                                "summary_stratum": stratum,
                                "layer": layer,
                            },
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _transition_bootstrap(
    transitions: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model in sorted(transitions["model"].unique()):
        model_frame = transitions[transitions["model"] == model]
        for transition in sorted(model_frame["transition"].unique()):
            transition_frame = model_frame[model_frame["transition"] == transition]
            for rho_name in ("rho05", "rho07"):
                rho_frame = transition_frame[transition_frame["rho_name"] == rho_name]
                for stratum in LABEL_STRATA:
                    selected = _filter_stratum(rho_frame, stratum)
                    if selected.empty:
                        continue
                    for comparison in CONTRASTS:
                        outputs.append(
                            _paired_mean(
                                selected,
                                comparison=comparison,
                                key_cols=("image_id", "class_id"),
                                value_cols=TRANSITION_METRICS,
                                draws=draws[stratum],
                                identity={
                                    "model": model,
                                    "summary_stratum": stratum,
                                    "transition": transition,
                                    "rho_name": rho_name,
                                },
                            )
                        )
        correctness_group = model_frame[model_frame["rho_name"] == "rho05"]
        for transition in sorted(correctness_group["transition"].unique()):
            transition_frame = correctness_group[
                correctness_group["transition"] == transition
            ]
            for status in CORRECTNESS_LEVELS:
                selected = transition_frame[
                    transition_frame["baseline_correctness"] == status
                ]
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_id"),
                            value_cols=TRANSITION_METRICS,
                            draws=draws["all"],
                            identity={
                                "model": model,
                                "summary_stratum": (f"baseline_correctness:{status}"),
                                "transition": transition,
                                "rho_name": "rho05",
                            },
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _shared_support_bootstrap(
    shared: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    if shared.empty:
        return pd.DataFrame()
    group_columns = ("map_family", "layer", "rho_name")
    for model in sorted(shared["model"].unique()):
        model_frame = shared[shared["model"] == model]
        for keys, group in model_frame.groupby(list(group_columns), sort=True):
            identity = dict(zip(group_columns, keys))
            for stratum in ("all", "exactly_2_labels", "3plus_labels"):
                selected = _filter_stratum(group, stratum)
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_a", "class_b"),
                            value_cols=SHARED_SUPPORT_METRICS,
                            draws=draws[stratum],
                            identity={
                                "model": model,
                                "summary_stratum": stratum,
                                **identity,
                            },
                            class_col="class_pair",
                            include_macro_class=False,
                        )
                    )
        correctness_group = model_frame[model_frame["rho_name"] == "rho05"]
        for keys, group in correctness_group.groupby(
            ["map_family", "layer"], sort=True
        ):
            family, layer = keys
            for status in (
                "both_classes_both_endpoints_correct",
                "either_class_or_endpoint_incorrect",
            ):
                selected = group[group["pair_baseline_correctness"] == status]
                if selected.empty:
                    continue
                for comparison in CONTRASTS:
                    outputs.append(
                        _paired_mean(
                            selected,
                            comparison=comparison,
                            key_cols=("image_id", "class_a", "class_b"),
                            value_cols=SHARED_SUPPORT_METRICS,
                            draws=draws["all"],
                            identity={
                                "model": model,
                                "summary_stratum": (f"baseline_correctness:{status}"),
                                "map_family": family,
                                "layer": layer,
                                "rho_name": "rho05",
                            },
                            class_col="class_pair",
                            include_macro_class=False,
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _positive_recall_bootstrap(
    classification: pd.DataFrame,
    draws: Mapping[str, ImageBootstrapDraws],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    positive = classification[classification["target"] == 1]
    if positive.empty:
        raise RuntimeError("positive-label recall has no positive labels")
    for model in sorted(positive["model"].unique()):
        model_frame = positive[positive["model"] == model]
        for stratum in LABEL_STRATA:
            selected = _filter_stratum(model_frame, stratum)
            if selected.empty:
                continue
            for comparison in CONTRASTS:
                outputs.append(
                    _paired_mean(
                        selected,
                        comparison=comparison,
                        key_cols=("image_id", "class_id"),
                        value_cols=POSITIVE_RECALL_METRICS,
                        draws=draws[stratum],
                        identity={
                            "model": model,
                            "summary_stratum": stratum,
                        },
                    )
                )
                for class_id in sorted(selected["class_id"].unique()):
                    class_frame = selected[selected["class_id"] == class_id]
                    class_result = _paired_mean(
                        class_frame,
                        comparison=comparison,
                        key_cols=("image_id", "class_id"),
                        value_cols=POSITIVE_RECALL_METRICS,
                        draws=draws[stratum],
                        identity={
                            "model": model,
                            "summary_stratum": stratum,
                        },
                        include_macro_class=False,
                    )
                    class_result["aggregation"] = "classwise"
                    class_result["semantic_class_id"] = int(class_id)
                    outputs.append(class_result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _iou_vector(confusion: np.ndarray) -> np.ndarray:
    value = np.asarray(confusion, dtype=np.float64)
    diagonal = np.diag(value)
    union = value.sum(axis=1) + value.sum(axis=0) - diagonal
    return np.divide(
        diagonal,
        union,
        out=np.full(21, np.nan, dtype=np.float64),
        where=union > 0,
    )


def _finite_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _classwise_iou_bootstrap(
    image_ids: Sequence[str],
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    comparison_name: str,
    draws: ImageBootstrapDraws,
    identity: Mapping[str, object],
) -> pd.DataFrame:
    if baseline.shape != comparison.shape or baseline.shape != (
        len(image_ids),
        21,
        21,
    ):
        raise ValueError("paired class-wise IoU stacks have invalid shapes")
    lookup = {image_id: index for index, image_id in enumerate(image_ids)}
    if set(lookup) != set(draws.image_ids):
        raise ValueError("paired class-wise IoU image IDs differ from shared draws")
    order = np.asarray([lookup[image_id] for image_id in draws.image_ids])
    left = baseline[order].astype(np.float64, copy=False)
    right = comparison[order].astype(np.float64, copy=False)
    if not np.array_equal(left.sum(axis=2), right.sum(axis=2)):
        raise ValueError("paired class-wise IoU GT marginals differ")
    left_point = _iou_vector(left.sum(axis=0))
    right_point = _iou_vector(right.sum(axis=0))
    left_samples = np.full((draws.repeats, 21), np.nan, dtype=np.float64)
    right_samples = np.full_like(left_samples, np.nan)
    left_flat = left.reshape(len(left), -1)
    right_flat = right.reshape(len(right), -1)
    for start in range(0, draws.repeats, 128):
        weights = draws.multiplicities[start : start + 128].astype(
            np.float64, copy=False
        )
        left_aggregate = (weights @ left_flat).reshape(-1, 21, 21)
        right_aggregate = (weights @ right_flat).reshape(-1, 21, 21)
        for offset in range(len(weights)):
            left_samples[start + offset] = _iou_vector(left_aggregate[offset])
            right_samples[start + offset] = _iou_vector(right_aggregate[offset])
    rows: list[dict[str, object]] = []
    series = {
        BASELINE_VARIANT: (left_point, left_samples),
        comparison_name: (right_point, right_samples),
        f"{comparison_name}_minus_{BASELINE_VARIANT}": (
            right_point - left_point,
            right_samples - left_samples,
        ),
    }
    for name, (points, samples) in series.items():
        for class_id in range(21):
            low, high, valid = _finite_interval(samples[:, class_id])
            rows.append(
                {
                    **identity,
                    "series": name,
                    "metric": "intersection_over_union",
                    "aggregation": "classwise",
                    "semantic_class_id": class_id,
                    "estimate": float(points[class_id]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_repeats": draws.repeats,
                    "bootstrap_valid_repeats": valid,
                    "bootstrap_seed": draws.seed,
                    "bootstrap_unit": "image",
                    "ci_method": "95% percentile",
                    "paired_delta": name
                    == f"{comparison_name}_minus_{BASELINE_VARIANT}",
                    "delta_definition": (
                        f"{comparison_name} - {BASELINE_VARIANT}"
                        if name == f"{comparison_name}_minus_{BASELINE_VARIANT}"
                        else ""
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _cam_bootstraps(
    validated: ValidatedInputs,
    collected: Mapping[str, CollectedRun],
    draws: Mapping[str, ImageBootstrapDraws],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scalar_outputs: list[pd.DataFrame] = []
    class_outputs: list[pd.DataFrame] = []
    ids = np.asarray(validated.image_ids)
    masks = _stratum_masks(validated.labels)
    for model, run in collected.items():
        for stratum in LABEL_STRATA:
            mask = masks[stratum]
            if not mask.any():
                continue
            selected_ids = ids[mask].tolist()
            baseline = run.fixed_confusions[mask, VARIANT_CODES.index("C0")]
            for comparison in CONTRASTS:
                compared = run.fixed_confusions[mask, VARIANT_CODES.index(comparison)]
                identity = {
                    "model": model,
                    "label_stratum": stratum,
                    "threshold": PRIMARY_THRESHOLD,
                    "comparison_role": CONTRAST_ROLE[comparison],
                }
                scalar_outputs.append(
                    paired_confusion_metric_summary(
                        selected_ids,
                        baseline,
                        compared,
                        baseline_name=BASELINE_VARIANT,
                        comparison_name=comparison,
                        draws=draws[stratum],
                        identity=identity,
                    )
                )
                class_outputs.append(
                    _classwise_iou_bootstrap(
                        selected_ids,
                        baseline,
                        compared,
                        comparison_name=comparison,
                        draws=draws[stratum],
                        identity=identity,
                    )
                )
    return (
        pd.concat(scalar_outputs, ignore_index=True),
        pd.concat(class_outputs, ignore_index=True),
    )


def _classification_bootstraps(
    validated: ValidatedInputs,
    collected: Mapping[str, CollectedRun],
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for model, run in collected.items():
        for source, values in (
            ("class_token", run.class_logits),
            ("patch_head", run.patch_logits),
        ):
            baseline = values[VARIANT_CODES.index(BASELINE_VARIANT)]
            for comparison in CONTRASTS:
                compared = values[VARIANT_CODES.index(comparison)]
                frame = pd.DataFrame.from_records(
                    paired_classification_bootstrap(
                        validated.image_ids,
                        validated.labels,
                        baseline,
                        compared,
                        repeats=repeats,
                        seed=seed,
                        logit_source=source,
                    )
                )
                mapping = {
                    "mctformer": BASELINE_VARIANT,
                    "mctformer_plus": comparison,
                    "mctformer_plus_minus_mctformer": (
                        f"{comparison}_minus_{BASELINE_VARIANT}"
                    ),
                }
                frame["series"] = frame["model_or_delta"].map(mapping)
                if frame["series"].isna().any():
                    raise RuntimeError("classification bootstrap series mapping failed")
                frame["model"] = model
                frame["comparison_role"] = CONTRAST_ROLE[comparison]
                frame["delta_definition"] = np.where(
                    frame["paired_delta"],
                    f"{comparison} - {BASELINE_VARIANT}",
                    "",
                )
                outputs.append(frame)
    output = pd.concat(outputs, ignore_index=True)
    rows: list[dict[str, object]] = []
    selected = output[
        (output["model"] == PRIMARY_MODEL)
        & (output["series"] == "C4_minus_C0")
        & (output["label_stratum"] == "all")
        & (output["metric"] == "mean_average_precision")
        & (output["aggregation"] == "macro_class")
    ]
    for _, row in selected.iterrows():
        lower = float(row["ci_low"])
        rows.append(
            {
                "model": PRIMARY_MODEL,
                "comparison": "C4_minus_C0",
                "logit_source": row["logit_source"],
                "margin": -0.003,
                "delta_map": float(row["estimate"]),
                "ci_low": lower,
                "ci_high": float(row["ci_high"]),
                "noninferiority_pass": bool(lower > -0.003),
                "decision_rule": "95% percentile lower CI > -0.003",
            }
        )
    if len(rows) != 2:
        raise RuntimeError("classification non-inferiority rows are incomplete")
    return output, pd.DataFrame.from_records(rows)


def _delta_text(
    frame: pd.DataFrame,
    *,
    model: str,
    series: str,
    metric: str,
    aggregation: str | None = None,
    **filters: object,
) -> str:
    required = {"model", "series", "metric"}
    if frame.empty or not required.issubset(frame.columns):
        return "N/A"
    selected = frame[
        (frame["model"] == model)
        & (frame["series"] == series)
        & (frame["metric"] == metric)
    ]
    if aggregation is not None and "aggregation" in selected:
        selected = selected[selected["aggregation"] == aggregation]
    for name, value in filters.items():
        if name not in selected:
            return "N/A"
        selected = selected[selected[name] == value]
    if len(selected) != 1:
        return "N/A"
    row = selected.iloc[0]
    return (
        f"{float(row['estimate']):+.4f} "
        f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
    )


def _write_report(
    output: Path,
    validated: ValidatedInputs,
    curves: pd.DataFrame,
    cam_bootstrap: pd.DataFrame,
    region_bootstrap: pd.DataFrame,
    pair_bootstrap: pd.DataFrame,
    transition_bootstrap: pd.DataFrame,
    shared_support_bootstrap: pd.DataFrame,
    positive_recall_bootstrap: pd.DataFrame,
    noninferiority: pd.DataFrame,
) -> None:
    plus_curves = curves[
        (curves["model"] == PRIMARY_MODEL)
        & (curves["label_stratum"] == "all")
        & np.isclose(curves["threshold"], PRIMARY_THRESHOLD)
    ].set_index("variant_code")
    miou_values = ", ".join(
        f"{variant}={100.0 * float(plus_curves.loc[variant, 'mean_iou']):.2f}%"
        for variant in VARIANT_CODES
    )
    cam_delta = _delta_text(
        cam_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="mean_iou",
        label_stratum="all",
    )
    c1_cam = _delta_text(
        cam_bootstrap,
        model=PRIMARY_MODEL,
        series="C1_minus_C0",
        metric="mean_iou",
        label_stratum="all",
    )
    auc_delta = _delta_text(
        region_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="auc_target_other",
        aggregation="micro",
        summary_stratum="all",
        map_family="attention_conditional",
        layer=12,
        rho_name="rho05",
    )
    ap_delta = _delta_text(
        region_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="ap_target_other",
        aggregation="micro",
        summary_stratum="all",
        map_family="attention_conditional",
        layer=12,
        rho_name="rho05",
    )
    bg_delta = _delta_text(
        region_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="conditional_bg_mass",
        aggregation="micro",
        summary_stratum="all",
        map_family="attention_conditional",
        layer=12,
        rho_name="rho05",
    )
    jaccard_delta = _delta_text(
        pair_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="attention_top10_jaccard",
        aggregation="micro",
        summary_stratum="all",
        layer=12,
    )
    patch_to_preprop_bg = _delta_text(
        transition_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="introduced_background_fraction",
        aggregation="micro",
        summary_stratum="all",
        transition="patch_to_preprop",
        rho_name="rho05",
    )
    preprop_to_final_bg = _delta_text(
        transition_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="introduced_background_fraction",
        aggregation="micro",
        summary_stratum="all",
        transition="preprop_to_final",
        rho_name="rho05",
    )
    final_shared_bg = _delta_text(
        shared_support_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="shared_background_fraction",
        aggregation="micro",
        summary_stratum="all",
        map_family="final_cam",
        layer=0,
        rho_name="rho05",
    )
    class_recall_delta = _delta_text(
        positive_recall_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="class_token_positive_recall",
        aggregation="macro_class",
        summary_stratum="all",
    )
    patch_recall_delta = _delta_text(
        positive_recall_bootstrap,
        model=PRIMARY_MODEL,
        series="C4_minus_C0",
        metric="patch_head_positive_recall",
        aggregation="macro_class",
        summary_stratum="all",
    )
    ni = ", ".join(
        f"{row.logit_source}: delta={float(row.delta_map):+.4f}, "
        f"CI=[{float(row.ci_low):+.4f},{float(row.ci_high):+.4f}], "
        f"pass={bool(row.noninferiority_pass)}"
        for row in noninferiority.itertuples()
    )
    scope = (
        "This is a smoke analysis: computation and provenance are validated, "
        "but population conclusions are not authorized."
        if validated.runs[PRIMARY_MODEL].run_kind == "smoke"
        else "All intervals resample complete VOC images; patches, classes, pairs, and heads from one image remain clustered."
    )
    lines = [
        "# Validation C: Late C2C Causal Intervention",
        "",
        "## Audited scope",
        "",
        f"- [Fact] Run kind: `{validated.runs[PRIMARY_MODEL].run_kind}`; models: {', '.join(validated.models)}; exactly matched images: {len(validated.image_ids):,}.",
        "- [Fact] C0--C5 are frozen-model analysis interventions. C4 (L10--L11 self-reroute) versus C0 is primary; C1 is the L12 structural negative control; C2/C3/C5 are secondary.",
        "- [Fact] Every runner artifact, manifest, linked Experiment 2 source, checkpoint, schema, and numerical equivalence gate was revalidated before analysis.",
        f"- [Fact] {scope}",
        "",
        "## Pre-registered primary endpoints",
        "",
        f"- [Fact] MCTformer+ raw-CAM mIoU at threshold 0.45: {miou_values}.",
        f"- [Statistical inference] MCTformer+ C4-C0 L12 target-vs-other AUROC: {auc_delta}; AUPRC: {ap_delta}.",
        f"- [Statistical inference] MCTformer+ C4-C0 L12 positive-class-pair attention top10 Jaccard: {jaccard_delta}.",
        f"- [Statistical inference] MCTformer+ C4-C0 final raw-CAM mIoU at 0.45: {cam_delta}.",
        f"- [Statistical inference] MCTformer+ C4-C0 L12 conditional background mass: {bg_delta}.",
        f"- [Statistical inference] MCTformer+ C4-C0 introduced-background fraction at patch-CAM→pre-propagation: {patch_to_preprop_bg}; at pre-propagation→final propagation: {preprop_to_final_bg}.",
        f"- [Statistical inference] MCTformer+ C4-C0 final-CAM positive-class-pair shared-support background fraction: {final_shared_bg}.",
        "",
        "## Controls and classification constraint",
        "",
        f"- [Fact] MCTformer+ C1-C0 final-CAM negative-control delta is {c1_cam}; direct source tensors were also required to remain below 1e-6.",
        f"- [Statistical inference] C4-C0 classification non-inferiority rule is lower 95% CI > -0.003. Results: {ni}.",
        f"- [Statistical inference] MCTformer+ C4-C0 macro-class positive-label recall at logit >= 0: class token {class_recall_delta}; patch head {patch_recall_delta}.",
        "- [Fact] Results are reported for all/single-label/exactly-two/three-plus label strata, per semantic CAM class, and C0-defined four-way class-token/patch-head correctness strata.",
        "- [Fact] Positive-label recall at the fixed logit >= 0 decision rule is reported as micro, equal-class macro, and class-wise estimates with the same whole-image bootstrap draws.",
        "- [Fact] The common 0.20--0.60 threshold sweep is diagnostic; variant-specific best thresholds do not replace the fixed 0.45 endpoint.",
        "",
        "## Interpretation boundaries",
        "",
        "- [Mechanistic interpretation] A C4-C0 change identifies the effect of this exact mass-preserving L10--L11 C2C value-reroute operator on downstream representations, routing, and CAM output; it does not prove that naturally occurring off-diagonal mixing is the unique cause of localization error.",
        "- [Mechanistic interpretation] Concordant reductions in class-pair overlap with improved target-vs-other discrimination and CAM mIoU would be consistent with late inter-class value mixing contributing to routing recoupling.",
        "- [Fact] For every intervention, patch-CAM is ReLU(patch-head logits), pre-propagation CAM uses the native last-three-layer raw A_c2p rule (MCTformer+: sqrt(mean(A) * patch-CAM); MCTformer: sum(A) * patch-CAM), and final CAM is the stored native propagated map. Their patch→pre-propagation and pre-propagation→final top-10% survival/introduction/removal metrics are therefore identifiable.",
        "- [Unsupported] This analysis does not validate a proposed method, training intervention, background-leakage claim, or causal shortcut beyond the explicitly evaluated frozen-model operator.",
        "",
        "## Artifact guide",
        "",
        "Canonical Parquets preserve classification rows, fixed-threshold image confusions, aggregate threshold confusions, late representation/attention region metrics, positive class-pair metrics, semantic ownership of shared supports, per-head region means, per-head/query C2C mass, and reconstructed native CAM-stage transitions. Compact CSVs contain point summaries, threshold curves, paired whole-image bootstrap intervals, per-class IoU, positive-label recall, classification AP/mAP, and the non-inferiority decision.",
        "",
    ]
    (output / "VALIDATION_C_LATE_C2C_CAUSAL.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _assert_inputs_unchanged(validated: ValidatedInputs) -> tuple[int, int]:
    for path_text, expected in validated.control_hashes.items():
        path = Path(path_text)
        if sha256_file(path) != expected:
            raise RuntimeError(
                f"immutable input control changed during analysis: {path}"
            )
    artifact_count = 0
    source_seen: set[Path] = set()
    for run in validated.runs.values():
        for row in run.manifest:
            path = _resolved_child(run.root, row["artifact_path"])
            if sha256_file(path) != str(row["artifact_sha256"]):
                raise RuntimeError(
                    f"Validation C artifact changed during analysis: {path}"
                )
            artifact_count += 1
        for image_id in run.image_ids:
            row = run.source_records[image_id]
            path = _resolved_child(run.source_root, row["signal_path"])
            if path in source_seen:
                continue
            if sha256_file(path) != str(row["artifact_sha256"]):
                raise RuntimeError(
                    f"Experiment 2 source changed during analysis: {path}"
                )
            source_seen.add(path)
    return artifact_count, len(source_seen)


def _validate_bootstrap_policy(
    run_kinds: set[str], repeats: int, *, allow_smoke: bool
) -> None:
    if run_kinds not in ({"smoke"}, {"full"}):
        raise RuntimeError(f"analysis inputs have inconsistent run kinds: {run_kinds}")
    if run_kinds == {"smoke"} and not allow_smoke:
        raise RuntimeError("smoke inputs require explicit --allow-smoke")
    if repeats != BOOTSTRAP_REPEATS and not (run_kinds == {"smoke"} and allow_smoke):
        raise RuntimeError(
            f"production analysis requires exactly {BOOTSTRAP_REPEATS} bootstrap "
            "repeats; a different count requires a smoke run and --allow-smoke"
        )


def execute(args: argparse.Namespace) -> Path:
    require_tgca_repro()
    if args.bootstrap_repeats < 1 or args.bootstrap_seed < 0:
        raise ValueError("bootstrap repeats must be positive and seed non-negative")
    validated = validate_inputs(args)
    run_kinds = {run.run_kind for run in validated.runs.values()}
    smoke_override = bool(getattr(args, "allow_smoke", False))
    _validate_bootstrap_policy(
        run_kinds, args.bootstrap_repeats, allow_smoke=smoke_override
    )
    output = assert_new_output(args.output_dir, validated.immutable_paths)
    output.mkdir(parents=True, exist_ok=False)

    collected = {
        model: _collect_run(validated.runs[model]) for model in validated.models
    }
    classification = pd.concat(
        [collected[model].classification for model in validated.models],
        ignore_index=True,
    )
    region = pd.concat(
        [collected[model].region for model in validated.models], ignore_index=True
    )
    pairs = pd.concat(
        [collected[model].pairs for model in validated.models], ignore_index=True
    )
    head_regions = pd.concat(
        [collected[model].head_regions for model in validated.models],
        ignore_index=True,
    )
    c2c = pd.concat(
        [collected[model].c2c for model in validated.models], ignore_index=True
    )
    transitions = pd.concat(
        [collected[model].transitions for model in validated.models],
        ignore_index=True,
    )
    shared_support = pd.concat(
        [collected[model].shared_support for model in validated.models],
        ignore_index=True,
    )
    fixed_cam = pd.concat(
        [collected[model].fixed_cam for model in validated.models], ignore_index=True
    )
    curves, per_class, aggregate_canonical, curve_summary = _threshold_tables(collected)
    fixed_metrics = curves[np.isclose(curves["threshold"], PRIMARY_THRESHOLD)].copy()

    region_summary = _point_summary(
        region,
        group_cols=("model", "variant_code", "layer", "rho_name", "map_family"),
        value_cols=REGION_METRICS,
        class_col="class_id",
    )
    pair_summary = _point_summary(
        pairs,
        group_cols=("model", "variant_code", "layer"),
        value_cols=PAIR_METRICS,
        class_col=None,
    )
    head_summary = _point_summary(
        head_regions,
        group_cols=("model", "variant_code", "layer", "rho_name"),
        value_cols=HEAD_REGION_METRICS,
        class_col="class_id",
    )
    c2c_summary = _point_summary(
        c2c,
        group_cols=("model", "variant_code", "layer"),
        value_cols=C2C_METRICS,
        class_col="class_id",
    )
    transition_summary = _point_summary(
        transitions,
        group_cols=("model", "variant_code", "transition", "rho_name"),
        value_cols=TRANSITION_METRICS,
        class_col="class_id",
    )
    shared_support_summary = _point_summary(
        shared_support,
        group_cols=(
            "model",
            "variant_code",
            "map_family",
            "layer",
            "rho_name",
        ),
        value_cols=SHARED_SUPPORT_METRICS,
        class_col=None,
    )

    draws = _shared_draws(
        validated.image_ids,
        validated.labels,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    region_bootstrap = _region_bootstrap(region, draws)
    pair_bootstrap = _pair_bootstrap(pairs, draws)
    head_bootstrap = _head_region_bootstrap(head_regions, draws)
    c2c_bootstrap = _c2c_bootstrap(c2c, draws)
    transition_bootstrap = _transition_bootstrap(transitions, draws)
    shared_support_bootstrap = _shared_support_bootstrap(shared_support, draws)
    positive_recall_bootstrap = _positive_recall_bootstrap(classification, draws)
    cam_bootstrap, cam_classwise_bootstrap = _cam_bootstraps(
        validated, collected, draws
    )
    classification_bootstrap, noninferiority = _classification_bootstraps(
        validated,
        collected,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    parquet_tables = {
        "canonical_classification.parquet": classification,
        "canonical_image_cam_t045.parquet": fixed_cam,
        "canonical_aggregate_threshold_confusions.parquet": aggregate_canonical,
        "canonical_image_class_region_metrics.parquet": region,
        "canonical_positive_class_pair_metrics.parquet": pairs,
        "canonical_head_region_metrics.parquet": head_regions,
        "canonical_c2c_head_query_metrics.parquet": c2c,
        "canonical_stage_transitions.parquet": transitions,
        "canonical_shared_support_ownership.parquet": shared_support,
    }
    csv_tables = {
        "threshold_curves.csv": curves,
        "per_class_iou_thresholds.csv": per_class,
        "fixed_t045_metrics.csv": fixed_metrics,
        "threshold_curve_summary.csv": curve_summary,
        "region_metric_summary.csv": region_summary,
        "class_pair_metric_summary.csv": pair_summary,
        "head_region_summary.csv": head_summary,
        "c2c_mass_summary.csv": c2c_summary,
        "stage_transition_summary.csv": transition_summary,
        "shared_support_summary.csv": shared_support_summary,
        "paired_region_bootstrap.csv": region_bootstrap,
        "paired_class_pair_bootstrap.csv": pair_bootstrap,
        "paired_head_region_bootstrap.csv": head_bootstrap,
        "paired_c2c_bootstrap.csv": c2c_bootstrap,
        "paired_stage_transition_bootstrap.csv": transition_bootstrap,
        "paired_shared_support_bootstrap.csv": shared_support_bootstrap,
        "paired_positive_recall_bootstrap.csv": positive_recall_bootstrap,
        "paired_cam_bootstrap.csv": cam_bootstrap,
        "paired_cam_classwise_iou_bootstrap.csv": cam_classwise_bootstrap,
        "paired_classification_bootstrap.csv": classification_bootstrap,
        "classification_noninferiority.csv": noninferiority,
    }
    for name, frame in parquet_tables.items():
        frame.to_parquet(output / name, index=False)
    for name, frame in csv_tables.items():
        frame.to_csv(output / name, index=False)

    _write_report(
        output,
        validated,
        curves,
        cam_bootstrap,
        region_bootstrap,
        pair_bootstrap,
        transition_bootstrap,
        shared_support_bootstrap,
        positive_recall_bootstrap,
        noninferiority,
    )
    command = shlex.join([sys.executable, *sys.argv])
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")

    canonical_files: dict[str, object] = {}
    for name, frame in parquet_tables.items():
        path = output / name
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != len(frame):
            raise RuntimeError(f"written canonical row count differs: {path}")
        if parquet.schema.names != list(frame.columns):
            raise RuntimeError(f"written canonical schema differs: {path}")
        canonical_files[name] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    canonical_metadata = {
        "status": "complete",
        "schema_version": 1,
        "analysis": ANALYSIS_NAME,
        "models": list(validated.models),
        "variant_order": list(VARIANT_CODES),
        "late_layers_one_based": list(LATE_LAYER_NUMBERS),
        "primary_contrast": "C4 - C0",
        "negative_control": "C1 - C0",
        "bootstrap_unit": "image",
        "classification_correctness": (
            "C0-defined at logit >= 0 independently for class-token and "
            "patch-head endpoints"
        ),
        "native_stage_reconstruction": {
            "patch_cam": "ReLU(patch_head_logits_positive)",
            "mctformer_plus_preprop": "sqrt(mean(raw A_c2p L10:L12) * patch_cam)",
            "mctformer_preprop": "sum(raw A_c2p L10:L12) * patch_cam",
            "final_cam": "stored native propagated CAM",
        },
        "conditional_bg_mass_denominator": (
            "complete 784-patch spatial-map sum, including mixed and void"
        ),
        "files": canonical_files,
    }
    json_dump(output / "canonical_metadata.json", canonical_metadata)

    verified_after, sources_after = _assert_inputs_unchanged(validated)
    if (
        verified_after != validated.verified_artifacts
        or sources_after != validated.verified_source_artifacts
    ):
        raise RuntimeError("before/after immutable artifact counts differ")
    generated: dict[str, object] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "analysis_metadata.json":
            continue
        generated[str(path.relative_to(output))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    label_counts = validated.labels.sum(axis=1)
    metadata = {
        "status": "complete",
        "analysis": ANALYSIS_NAME,
        "run_kind": validated.runs[PRIMARY_MODEL].run_kind,
        "models": list(validated.models),
        "num_images": len(validated.image_ids),
        "positive_image_class_pairs": int(validated.labels.sum()),
        "multilabel_images": int(np.count_nonzero(label_counts >= 2)),
        "variants": list(VARIANT_CODES),
        "contrasts": [
            {
                "baseline": BASELINE_VARIANT,
                "comparison": variant,
                "role": CONTRAST_ROLE[variant],
            }
            for variant in CONTRASTS
        ],
        "primary_endpoints": [
            "MCTformer+ L12 target-vs-other AUROC/AUPRC",
            "MCTformer+ L12 positive-class-pair attention top10 Jaccard",
            "MCTformer+ final raw-CAM mIoU at threshold 0.45",
        ],
        "classification_noninferiority_margin": -0.003,
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "whole image multinomial multiplicity",
            "ci": "95% percentile",
            "shared_draws": {
                stratum: {
                    "seed": item.seed,
                    "images": len(item.image_ids),
                    "repeats": item.repeats,
                }
                for stratum, item in draws.items()
            },
            "same_draws_reused_across_variants_metrics_and_models": True,
            "smoke_override": smoke_override,
        },
        "baseline_correctness_policy": (
            "C0, logit >= 0; four exact joint class-token/patch-head statuses"
        ),
        "stage_transition_identifiability": (
            "identified from exact native reconstruction of patch_cam and "
            "class_attention_cam plus stored final_cam"
        ),
        "source_metadata": str(validated.source_metadata_path),
        "run_roots": {
            model: str(validated.runs[model].root) for model in validated.models
        },
        "verified_input_artifacts_before_after": verified_after,
        "verified_experiment2_sources_before_after": sources_after,
        "input_hashes_before_and_after_equal": True,
        "input_control_hashes": dict(validated.control_hashes),
        "canonical_metadata": "canonical_metadata.json",
        "generated_files": generated,
        "git": git_state(REPO_ROOT),
        "command": command,
        "completed_at": timestamp(),
    }
    json_dump(output / "analysis_metadata.json", metadata)
    return output


def main() -> None:
    args = parse_args()
    output = execute(args)
    print(json.dumps({"status": "complete", "output_dir": str(output)}))


if __name__ == "__main__":
    main()
