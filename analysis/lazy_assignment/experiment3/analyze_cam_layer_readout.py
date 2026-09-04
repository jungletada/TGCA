#!/usr/bin/env python3
"""Offline analysis for Experiment 3 Validation B CAM layer readouts.

The two CAM run roots and every linked Experiment 2 signal are immutable
inputs.  This program verifies those inputs before creating a new output
directory, then derives image-clustered statistics without loading a model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    region_map_metrics,
)
from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (  # noqa: E402
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.metrics_stage_linkage import (  # noqa: E402
    stage_transition_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    REGION_BACKGROUND,
    assign_pair_patch_regions_from_counts,
)
from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (  # noqa: E402
    ImageBootstrapDraws,
    image_multinomial_draws,
    paired_confusion_metric_summary,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (  # noqa: E402
    CAM_VARIANT_SPECS,
    cam_metrics_from_confusion,
    cam_threshold_grid,
    native_best_threshold_anchor,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    EXPECTED_IMAGES,
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


ANALYSIS_NAME = "experiment3_validation_b_cam_layer_readout_analysis"
MODEL_ORDER = ("mctformer", "mctformer_plus")
VARIANT_CODES = tuple(spec.code for spec in CAM_VARIANT_SPECS)
VARIANT_NAMES = {spec.code: spec.name for spec in CAM_VARIANT_SPECS}
RHO_SPECS = (("rho05", 0.5), ("rho07", 0.7))
STRATA = ("all", "single", "exactly_2", "3_plus")
PRIMARY_THRESHOLD = 0.45
PRIMARY_COMPARISONS = ("B1", "B4")
CONFUSION_ENCODING = "21x21 int64 little-endian C-order"

B_ARTIFACT_KEYS = {
    "image_id",
    "positive_class_ids",
    "variant_codes",
    "thresholds",
    "attention_raw",
    "attention_conditional",
    "preprop_cam",
    "final_cam",
    "confusions",
    "source_signal_sha256",
}
SOURCE_KEYS = {
    "image_id",
    "positive_class_ids",
    "grid_h",
    "grid_w",
    "patch_label_counts",
    "region_masks_rho05",
    "region_masks_rho07",
    "patch_cam",
    "final_cam",
    "raw_final_cam_confusion_t045",
}
REGION_OUTPUT_METRICS = (
    "cpim",
    "target_top10_fraction",
    "other_top10_fraction",
    "background_top10_fraction",
    "target_top10_enrichment",
    "other_top10_enrichment",
    "background_top10_enrichment",
    "target_bg_auroc",
    "target_bg_average_precision",
    "target_other_auroc",
    "target_other_average_precision",
    "conditional_background_mass",
)
PAIR_OUTPUT_METRICS = (
    "top10_jaccard",
    "shared_set_size",
    "shared_target_a_fraction",
    "shared_target_b_fraction",
    "shared_other_foreground_fraction",
    "shared_background_fraction",
    "shared_mixed_fraction",
    "shared_target_a_enrichment",
    "shared_target_b_enrichment",
    "shared_other_foreground_enrichment",
    "shared_background_enrichment",
)
TRANSITION_OUTPUT_METRICS = (
    "top10_jaccard",
    "top10_overlap_coefficient",
    "introduced_size",
    "removed_size",
    "survive_target",
    "survive_other_foreground",
    "survive_background",
    "introduced_target_fraction",
    "introduced_other_foreground_fraction",
    "introduced_background_fraction",
    "removed_target_fraction",
    "removed_other_foreground_fraction",
    "removed_background_fraction",
)


@dataclass(frozen=True)
class ValidatedRun:
    model: str
    root: Path
    metadata: Mapping[str, object]
    completion: Mapping[str, object]
    manifest: tuple[Mapping[str, object], ...]
    source_root: Path
    source_manifest: Mapping[str, Mapping[str, object]]
    run_kind: str


@dataclass(frozen=True)
class ValidatedInputs:
    source_metadata_path: Path
    source_metadata: Mapping[str, object]
    linkage_path: Path
    linkage: Mapping[str, object]
    image_ids: tuple[str, ...]
    labels: np.ndarray
    runs: Mapping[str, ValidatedRun]
    input_hashes: Mapping[str, str]
    consumed_file_hashes: Mapping[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-run-root", type=Path, required=True)
    parser.add_argument("--mctformer-plus-run-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow smoke inputs and, only there, fewer than 5000 repeats.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap-seed must be non-negative")
    return args


def _validate_bootstrap_policy(
    run_kinds: set[str], repeats: int, *, allow_smoke: bool
) -> None:
    """Lock production inference to the pre-registered bootstrap contract."""

    if repeats != BOOTSTRAP_REPEATS and not (run_kinds == {"smoke"} and allow_smoke):
        raise RuntimeError(
            f"production analysis requires exactly {BOOTSTRAP_REPEATS} bootstrap "
            "repeats; a different count requires smoke inputs and --allow-smoke"
        )
    if run_kinds == {"smoke"} and not allow_smoke:
        raise RuntimeError("smoke inputs require explicit --allow-smoke")
    if run_kinds not in ({"smoke"}, {"full"}):
        raise RuntimeError(f"analysis inputs have inconsistent run kinds: {run_kinds}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _resolved_child(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes run root: {relative!r}") from error
    return path


def _record_consumed_hash(
    records: dict[str, str], path: Path, expected_sha256: object
) -> None:
    """Record one already-validated immutable input without hiding conflicts."""

    resolved = str(path.resolve())
    digest = str(expected_sha256)
    if not digest:
        raise RuntimeError(f"immutable input lacks a SHA-256 digest: {resolved}")
    previous = records.get(resolved)
    if previous is not None and previous != digest:
        raise RuntimeError(f"conflicting immutable input hashes: {resolved}")
    records[resolved] = digest


def _assert_consumed_files_unchanged(
    expected_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Re-hash every B artifact, Experiment 2 artifact, and checkpoint."""

    observed: dict[str, str] = {}
    for path_text, expected in sorted(expected_hashes.items()):
        path = Path(path_text)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"consumed immutable input changed during analysis: {path}"
            )
        observed[path_text] = actual
    if not observed:
        raise RuntimeError("no consumed immutable files were registered")
    return observed


def _stratum(num_positive: int) -> str:
    if num_positive == 1:
        return "single"
    if num_positive == 2:
        return "exactly_2"
    if num_positive >= 3:
        return "3_plus"
    raise ValueError("every evaluated image must have a positive class")


def encode_confusion(confusion: np.ndarray) -> bytes:
    """Encode one exact 21x21 confusion for compact canonical Parquet storage."""

    value = np.asarray(confusion)
    if value.shape != (21, 21) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("confusion must be an integer 21x21 matrix")
    if np.any(value < 0):
        raise ValueError("confusion must be non-negative")
    return value.astype("<i8", copy=False).tobytes(order="C")


def decode_confusion(blob: bytes) -> np.ndarray:
    """Decode :func:`encode_confusion` output and return an owned array."""

    if not isinstance(blob, bytes) or len(blob) != 21 * 21 * 8:
        raise ValueError("invalid encoded 21x21 int64 confusion")
    return np.frombuffer(blob, dtype="<i8").reshape(21, 21).copy()


def normalized_curve_auc(thresholds: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoidal curve AUC divided by the threshold interval width."""

    x = np.asarray(thresholds, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("thresholds and values must be equal vectors of length >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("curve coordinates must be finite")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("thresholds must be strictly increasing")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _expected_variant_contract(metadata: Mapping[str, object]) -> None:
    contract = metadata.get("cam_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("CAM metadata lacks cam_contract")
    if tuple(contract.get("variant_order", ())) != VARIANT_CODES:
        raise RuntimeError("CAM variant order is not the pre-registered B0--B5 order")
    variants = contract.get("variants")
    if not isinstance(variants, Mapping):
        raise TypeError("CAM contract lacks variant mapping")
    for spec in CAM_VARIANT_SPECS:
        record = variants.get(spec.code)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"CAM contract lacks {spec.code}")
        if (
            record.get("name") != spec.name
            or tuple(record.get("layers_one_based", ())) != spec.layers_one_based
        ):
            raise RuntimeError(f"CAM contract mismatch for {spec.code}")
    thresholds = np.asarray(contract.get("thresholds"), dtype=np.float64)
    if not np.array_equal(thresholds, cam_threshold_grid()):
        raise RuntimeError("CAM threshold grid differs from the pre-registration")
    if float(contract.get("primary_threshold", float("nan"))) != PRIMARY_THRESHOLD:
        raise RuntimeError("CAM primary threshold must be exactly 0.45")


def _validate_source_manifest(
    root: Path,
    linkage_record: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    manifest_path = root / "manifest.jsonl"
    if sha256_file(manifest_path) != str(linkage_record.get("manifest_sha256", "")):
        raise RuntimeError(f"Experiment 2 manifest hash mismatch: {manifest_path}")
    if sha256_file(root / "metadata.json") != str(
        linkage_record.get("metadata_sha256", "")
    ):
        raise RuntimeError(f"Experiment 2 metadata hash mismatch: {root}")
    if sha256_file(root / "completion.json") != str(
        linkage_record.get("completion_sha256", "")
    ):
        raise RuntimeError(f"Experiment 2 completion hash mismatch: {root}")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(linkage_record.get("manifest_rows", -1)):
        raise RuntimeError(f"Experiment 2 manifest row-count mismatch: {root}")
    ids = [str(row.get("image_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate Experiment 2 signal IDs: {root}")
    return {image_id: row for image_id, row in zip(ids, rows)}


def _validate_b_artifact(
    path: Path,
    record: Mapping[str, object],
    source_root: Path,
    source_record: Mapping[str, object],
    expected_positive: np.ndarray,
) -> None:
    if sha256_file(path) != str(record.get("artifact_sha256", "")):
        raise RuntimeError(f"CAM artifact hash mismatch: {path}")
    source_path = _resolved_child(source_root, source_record["signal_path"])
    source_digest = sha256_file(source_path)
    if source_digest != str(source_record.get("artifact_sha256", "")):
        raise RuntimeError(f"Experiment 2 signal hash mismatch: {source_path}")
    with (
        np.load(path, allow_pickle=False) as artifact,
        np.load(source_path, allow_pickle=False) as source,
    ):
        if set(artifact.files) != B_ARTIFACT_KEYS:
            raise RuntimeError(f"unexpected CAM artifact schema: {path}")
        if not SOURCE_KEYS.issubset(source.files):
            raise RuntimeError(f"incomplete Experiment 2 signal schema: {source_path}")
        image_id = str(record.get("image_id", ""))
        positive = np.asarray(artifact["positive_class_ids"])
        k = len(expected_positive)
        if (
            artifact["image_id"].shape != ()
            or str(artifact["image_id"].item()) != image_id
            or positive.dtype != np.int64
            or not np.array_equal(positive, expected_positive)
            or list(record.get("positive_class_ids", [])) != expected_positive.tolist()
        ):
            raise RuntimeError(f"CAM identity/positive-label mismatch: {path}")
        if tuple(np.asarray(artifact["variant_codes"]).tolist()) != VARIANT_CODES:
            raise RuntimeError(f"CAM variant-code mismatch: {path}")
        if artifact["thresholds"].dtype != np.float64 or not np.array_equal(
            artifact["thresholds"], cam_threshold_grid()
        ):
            raise RuntimeError(f"CAM threshold mismatch: {path}")
        float_shapes = {
            "attention_raw": (6, k, EXPECTED_PATCHES),
            "attention_conditional": (6, k, EXPECTED_PATCHES),
            "preprop_cam": (6, k, EXPECTED_PATCHES),
            "final_cam": (6, k, EXPECTED_PATCHES),
        }
        for key, shape in float_shapes.items():
            values = artifact[key]
            if values.dtype != np.float32 or values.shape != shape:
                raise RuntimeError(f"{key} schema mismatch in {path}")
            if not np.isfinite(values).all() or float(values.min()) < 0.0:
                raise RuntimeError(f"{key} contains invalid values in {path}")
        conditional_sums = artifact["attention_conditional"].sum(axis=-1)
        if not np.allclose(conditional_sums, 1.0, atol=2e-6, rtol=0.0):
            raise RuntimeError(f"conditional attention is not normalized: {path}")
        confusions = artifact["confusions"]
        if confusions.dtype != np.int64 or confusions.shape != (6, 41, 21, 21):
            raise RuntimeError(f"confusion schema mismatch: {path}")
        if np.any(confusions < 0) or np.any(confusions.sum(axis=(2, 3)) <= 0):
            raise RuntimeError(f"invalid confusion counts: {path}")
        gt_marginal = confusions.sum(axis=3)
        if not np.all(gt_marginal == gt_marginal[0, 0]):
            raise RuntimeError(f"variant/threshold GT marginals differ: {path}")
        if (
            artifact["source_signal_sha256"].shape != ()
            or str(artifact["source_signal_sha256"].item()) != source_digest
            or str(source["image_id"].item()) != image_id
            or not np.array_equal(source["positive_class_ids"], expected_positive)
        ):
            raise RuntimeError(f"Experiment 2 source linkage mismatch: {path}")
        if source["region_masks_rho05"].shape != (k, EXPECTED_PATCHES) or source[
            "region_masks_rho07"
        ].shape != (k, EXPECTED_PATCHES):
            raise RuntimeError(f"source region-mask shape mismatch: {source_path}")
        if source["patch_label_counts"].shape != (EXPECTED_PATCHES, 22):
            raise RuntimeError(f"source patch-count shape mismatch: {source_path}")
        if source["patch_cam"].shape != (k, EXPECTED_PATCHES):
            raise RuntimeError(f"source patch-CAM shape mismatch: {source_path}")
        if source["final_cam"].shape != (k, EXPECTED_PATCHES):
            raise RuntimeError(f"source final-CAM shape mismatch: {source_path}")
        native_diff = float(
            np.max(np.abs(artifact["final_cam"][0] - source["final_cam"]))
        )
        if native_diff > STRICT_TOLERANCE:
            raise RuntimeError(f"B0/source native CAM mismatch {native_diff}: {path}")
        primary_index = int(np.flatnonzero(cam_threshold_grid() == 0.45)[0])
        if not np.array_equal(
            confusions[0, primary_index], source["raw_final_cam_confusion_t045"]
        ):
            raise RuntimeError(f"B0/source threshold-0.45 confusion mismatch: {path}")


def _validate_run(
    model: str,
    root: Path,
    source_metadata_path: Path,
    source_metadata: Mapping[str, object],
    linkage: Mapping[str, object],
    all_image_ids: Sequence[str],
    labels: np.ndarray,
) -> ValidatedRun:
    root = root.expanduser().resolve()
    metadata = read_json(root / "metadata.json")
    completion = read_json(root / "completion.json")
    if (
        metadata.get("status") != "complete"
        or completion.get("status") != "complete"
        or metadata.get("analysis") != "experiment3_validation_b_cam_layer_readout"
        or completion.get("analysis") != "experiment3_validation_b_cam_layer_readout"
        or metadata.get("model") != model
        or completion.get("model") != model
    ):
        raise RuntimeError(f"incomplete or misidentified CAM run: {root}")
    run_kind = str(metadata.get("run_kind", ""))
    if run_kind not in {"full", "smoke"} or completion.get("run_kind") != run_kind:
        raise RuntimeError(f"invalid/inconsistent run kind: {root}")
    _expected_variant_contract(metadata)
    source_digest = sha256_file(source_metadata_path)
    if str(metadata.get("source_metadata_sha256", "")) != source_digest:
        raise RuntimeError(f"source-metadata hash mismatch: {root}")
    signal_roots = source_metadata.get("signal_roots")
    if not isinstance(signal_roots, Mapping):
        raise TypeError("source metadata lacks signal_roots")
    source_root = Path(str(signal_roots[model])).resolve()
    if Path(str(metadata.get("experiment2_signal_root", ""))).resolve() != source_root:
        raise RuntimeError(f"Experiment 2 signal-root mismatch: {root}")
    checkpoint = metadata.get("checkpoint")
    expected_checkpoint = source_metadata.get("checkpoints", {}).get(model)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        expected_checkpoint, Mapping
    ):
        raise TypeError(f"missing checkpoint linkage for {model}")
    if (
        Path(str(checkpoint.get("path", ""))).resolve()
        != Path(str(expected_checkpoint.get("path", ""))).resolve()
        or checkpoint.get("sha256") != expected_checkpoint.get("actual_sha256")
        or sha256_file(Path(str(checkpoint["path"])).resolve())
        != str(checkpoint["sha256"])
    ):
        raise RuntimeError(f"checkpoint linkage/hash mismatch: {root}")
    manifest = _read_jsonl(root / "manifest.jsonl")
    ids = [str(record.get("image_id", "")) for record in manifest]
    requested = int(metadata.get("execution", {}).get("requested_images", -1))
    processed = int(metadata.get("processed_images", -1))
    if len(ids) != len(set(ids)) or not len(ids):
        raise RuntimeError(f"duplicate/empty CAM manifest: {root}")
    if (
        len(ids) != requested
        or len(ids) != processed
        or len(ids) != int(completion.get("num_images", -1))
    ):
        raise RuntimeError(f"CAM manifest/count metadata mismatch: {root}")
    if run_kind == "full":
        if ids != list(all_image_ids) or len(ids) != int(
            source_metadata.get("dataset", {}).get("num_images", -1)
        ):
            raise RuntimeError(f"full CAM run membership/order mismatch: {root}")
    elif ids != list(all_image_ids[: len(ids)]) or len(ids) >= len(all_image_ids):
        raise RuntimeError(f"smoke CAM run must be a strict ordered VOC prefix: {root}")
    linkage_signals = linkage.get("signals")
    if not isinstance(linkage_signals, Mapping) or not isinstance(
        linkage_signals.get(model), Mapping
    ):
        raise TypeError(f"Experiment 2 linkage lacks {model}")
    source_manifest = _validate_source_manifest(source_root, linkage_signals[model])
    label_lookup = {
        image_id: labels[index] for index, image_id in enumerate(all_image_ids)
    }
    for record in manifest:
        image_id = str(record.get("image_id", ""))
        if image_id not in source_manifest or image_id not in label_lookup:
            raise RuntimeError(
                f"CAM image absent from linked source/VOC labels: {image_id}"
            )
        path = _resolved_child(root, record.get("artifact_path"))
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_positive = np.flatnonzero(label_lookup[image_id] > 0).astype(np.int64)
        _validate_b_artifact(
            path,
            record,
            source_root,
            source_manifest[image_id],
            expected_positive,
        )
    direct_diffs = metadata.get("source_variant_final_cam_max_abs_diff")
    if not isinstance(direct_diffs, Mapping):
        raise TypeError(f"CAM metadata lacks source equivalence values: {root}")
    for code in ("B0", "B1", "B2", "B3", "B5"):
        if float(direct_diffs.get(code, math.inf)) > STRICT_TOLERANCE:
            raise RuntimeError(f"source equivalence failed for {model}/{code}")
    for key in (
        "source_patch_cam_max_abs_diff",
        "native_cam_max_abs_diff",
        "mask_patch_count_max_abs_diff",
    ):
        if float(metadata.get(key, math.inf)) > STRICT_TOLERANCE:
            raise RuntimeError(f"run numerical equivalence failed for {model}/{key}")
    return ValidatedRun(
        model=model,
        root=root,
        metadata=metadata,
        completion=completion,
        manifest=tuple(manifest),
        source_root=source_root,
        source_manifest=source_manifest,
        run_kind=run_kind,
    )


def validate_inputs(args: argparse.Namespace) -> ValidatedInputs:
    source_path = args.source_metadata.expanduser().resolve()
    source = read_json(source_path)
    if source.get("status") != "complete" or source.get("integrity_passed") is not True:
        raise RuntimeError("Experiment 3 source audit is not complete/passed")
    dataset = source.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("source audit lacks dataset metadata")
    all_image_ids = ordered_val_ids(Path(str(dataset["list_path"])).resolve())
    labels = load_image_labels(
        Path(str(dataset["labels_path"])).resolve(), all_image_ids
    )
    if len(all_image_ids) != int(dataset.get("num_images", -1)):
        raise RuntimeError("source-audit VOC image count no longer matches")
    if int(labels.sum()) != int(dataset.get("positive_image_class_pairs", -1)):
        raise RuntimeError("source-audit positive-pair count no longer matches")
    if int(np.count_nonzero(labels.sum(axis=1) >= 2)) != int(
        dataset.get("multilabel_images", -1)
    ):
        raise RuntimeError("source-audit multi-label count no longer matches")
    linkage_path = Path(str(source.get("experiment2_linkage", ""))).resolve()
    linkage = read_json(linkage_path)
    if linkage.get("status") != "complete":
        raise RuntimeError("Experiment 2 linkage audit is incomplete")
    if (
        Path(str(linkage.get("experiment2_root", ""))).resolve()
        != Path(str(source.get("experiment2_root", ""))).resolve()
    ):
        raise RuntimeError("Experiment 2 root differs across audit/linkage metadata")
    roots = {
        "mctformer": args.mctformer_run_root,
        "mctformer_plus": args.mctformer_plus_run_root,
    }
    runs = {
        model: _validate_run(
            model,
            roots[model],
            source_path,
            source,
            linkage,
            all_image_ids,
            labels,
        )
        for model in MODEL_ORDER
    }
    left_ids = [str(row["image_id"]) for row in runs["mctformer"].manifest]
    right_ids = [str(row["image_id"]) for row in runs["mctformer_plus"].manifest]
    left_positive = [row["positive_class_ids"] for row in runs["mctformer"].manifest]
    right_positive = [
        row["positive_class_ids"] for row in runs["mctformer_plus"].manifest
    ]
    if (
        runs["mctformer"].run_kind != runs["mctformer_plus"].run_kind
        or left_ids != right_ids
        or left_positive != right_positive
    ):
        raise RuntimeError("MCTformer/MCTformer+ run kind, order, or labels differ")
    if runs["mctformer"].run_kind == "full":
        if (
            len(left_ids) != EXPECTED_IMAGES
            or sum(len(value) for value in left_positive) != EXPECTED_POSITIVE_PAIRS
            or sum(len(value) >= 2 for value in left_positive)
            != EXPECTED_MULTILABEL_IMAGES
            or int(dataset.get("input_size", -1)) != 448
            or int(dataset.get("patch_size", -1)) != 16
            or int(dataset.get("num_images", -1)) != EXPECTED_IMAGES
        ):
            raise RuntimeError("full-run VOC cardinality/geometry contract failed")
    input_hashes = {
        "source_metadata": sha256_file(source_path),
        "experiment2_linkage": sha256_file(linkage_path),
    }
    for model, run in runs.items():
        input_hashes[f"{model}_run_metadata"] = sha256_file(run.root / "metadata.json")
        input_hashes[f"{model}_run_completion"] = sha256_file(
            run.root / "completion.json"
        )
        input_hashes[f"{model}_run_manifest"] = sha256_file(run.root / "manifest.jsonl")
        input_hashes[f"{model}_source_manifest"] = sha256_file(
            run.source_root / "manifest.jsonl"
        )
    consumed_file_hashes: dict[str, str] = {}
    for run in runs.values():
        checkpoint = run.metadata["checkpoint"]
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"validated checkpoint record disappeared for {run.model}")
        _record_consumed_hash(
            consumed_file_hashes,
            Path(str(checkpoint["path"])),
            checkpoint["sha256"],
        )
        for record in run.manifest:
            image_id = str(record["image_id"])
            source_record = run.source_manifest[image_id]
            _record_consumed_hash(
                consumed_file_hashes,
                _resolved_child(run.root, record["artifact_path"]),
                record["artifact_sha256"],
            )
            _record_consumed_hash(
                consumed_file_hashes,
                _resolved_child(run.source_root, source_record["signal_path"]),
                source_record["artifact_sha256"],
            )
    selected_labels = np.stack(
        [labels[all_image_ids.index(image_id)] for image_id in left_ids]
    )
    return ValidatedInputs(
        source_metadata_path=source_path,
        source_metadata=source,
        linkage_path=linkage_path,
        linkage=linkage,
        image_ids=tuple(left_ids),
        labels=selected_labels,
        runs=runs,
        input_hashes=input_hashes,
        consumed_file_hashes=consumed_file_hashes,
    )


def _region_record(metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        "cpim": float(bool(metrics["target_hit"])),
        "target_top10_fraction": float(metrics["target_top10_fraction"]),
        "other_top10_fraction": float(metrics["other_fg_top10_fraction"]),
        "background_top10_fraction": float(metrics["bg_top10_fraction"]),
        "target_top10_enrichment": float(metrics["target_tail_enrich_10"]),
        "other_top10_enrichment": float(metrics["other_fg_tail_enrich_10"]),
        "background_top10_enrichment": float(metrics["bg_tail_enrich_10"]),
        "target_bg_auroc": float(metrics["auc_target_bg"]),
        "target_bg_average_precision": float(metrics["ap_target_bg"]),
        "target_other_auroc": float(metrics["auc_target_other"]),
        "target_other_average_precision": float(metrics["ap_target_other"]),
        "conditional_background_mass": float(metrics["conditional_bg_mass"]),
    }


def _canonical_region_metrics(
    values: np.ndarray, regions: np.ndarray
) -> Mapping[str, object]:
    """Match Experiment 2's complete-spatial-map BG mass denominator."""

    scores = np.asarray(values, dtype=np.float64)
    codes = np.asarray(regions)
    metrics = region_map_metrics(
        scores,
        codes,
        grid_h=28,
        grid_w=28,
        nonnegative_mass=True,
    )
    total = float(scores.sum())
    metrics["conditional_bg_mass"] = (
        float(scores[codes == REGION_BACKGROUND].sum() / total)
        if total > 1e-12
        else float("nan")
    )
    return metrics


def _pair_record(metrics: Mapping[str, object]) -> dict[str, float | int]:
    return {
        "top10_jaccard": float(metrics["topk_jaccard"]),
        "shared_set_size": int(metrics["shared_set_size"]),
        "shared_target_a_fraction": float(metrics["shared_target_a_fraction"]),
        "shared_target_b_fraction": float(metrics["shared_target_b_fraction"]),
        "shared_other_foreground_fraction": float(metrics["shared_other_fg_fraction"]),
        "shared_background_fraction": float(metrics["shared_background_fraction"]),
        # Experiment 2 names this union bucket mixed/void. Shared-support
        # construction excludes void patches, so its realized content here is
        # the mixed region while retaining the upstream key contract.
        "shared_mixed_fraction": float(metrics["shared_mixed_void_fraction"]),
        "shared_target_a_enrichment": float(metrics["shared_target_a_enrichment"]),
        "shared_target_b_enrichment": float(metrics["shared_target_b_enrichment"]),
        "shared_other_foreground_enrichment": float(
            metrics["shared_other_fg_enrichment"]
        ),
        "shared_background_enrichment": float(metrics["shared_background_enrichment"]),
    }


def _transition_record(metrics: Mapping[str, object]) -> dict[str, float | int]:
    return {
        "top10_jaccard": float(metrics["topk_jaccard"]),
        "top10_overlap_coefficient": float(metrics["topk_overlap_coefficient"]),
        "introduced_size": int(metrics["introduced_size"]),
        "removed_size": int(metrics["removed_size"]),
        "survive_target": float(metrics["survive_target"]),
        "survive_other_foreground": float(metrics["survive_other_fg"]),
        "survive_background": float(metrics["survive_background"]),
        "introduced_target_fraction": float(metrics["introduced_target_fraction"]),
        "introduced_other_foreground_fraction": float(
            metrics["introduced_other_fg_fraction"]
        ),
        "introduced_background_fraction": float(
            metrics["introduced_background_fraction"]
        ),
        "removed_target_fraction": float(metrics["removed_target_fraction"]),
        "removed_other_foreground_fraction": float(
            metrics["removed_other_fg_fraction"]
        ),
        "removed_background_fraction": float(metrics["removed_background_fraction"]),
    }


def _collect_run(
    run: ValidatedRun,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    thresholds = cam_threshold_grid()
    primary_index = int(np.flatnonzero(thresholds == PRIMARY_THRESHOLD)[0])
    stratum_index = {name: index for index, name in enumerate(STRATA)}
    aggregate = np.zeros((len(STRATA), 6, 41, 21, 21), dtype=np.int64)
    fixed_stacks: list[np.ndarray] = []
    fixed_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for record in run.manifest:
        image_id = str(record["image_id"])
        artifact_path = _resolved_child(run.root, record["artifact_path"])
        source_record = run.source_manifest[image_id]
        source_path = _resolved_child(run.source_root, source_record["signal_path"])
        with (
            np.load(artifact_path, allow_pickle=False) as artifact,
            np.load(source_path, allow_pickle=False) as source,
        ):
            positive = np.asarray(artifact["positive_class_ids"], dtype=np.int64)
            exact_stratum = _stratum(len(positive))
            confusions = np.asarray(artifact["confusions"], dtype=np.int64)
            aggregate[stratum_index["all"]] += confusions
            aggregate[stratum_index[exact_stratum]] += confusions
            fixed = confusions[:, primary_index].copy()
            fixed_stacks.append(fixed)
            for variant_index, code in enumerate(VARIANT_CODES):
                fixed_rows.append(
                    {
                        "image_id": image_id,
                        "model": run.model,
                        "label_stratum": exact_stratum,
                        "num_positive_classes": len(positive),
                        "variant_code": code,
                        "variant_name": VARIANT_NAMES[code],
                        "threshold": PRIMARY_THRESHOLD,
                        "confusion_encoding": CONFUSION_ENCODING,
                        "confusion": encode_confusion(fixed[variant_index]),
                        "evaluated_pixels": int(fixed[variant_index].sum()),
                    }
                )
            stage_arrays = {
                "attention": np.asarray(artifact["attention_conditional"]),
                "preprop": np.asarray(artifact["preprop_cam"]),
                "final": np.asarray(artifact["final_cam"]),
            }
            patch_cam = np.asarray(source["patch_cam"], dtype=np.float64)
            label_counts = np.asarray(source["patch_label_counts"])
            for rho_name, rho in RHO_SPECS:
                regions = np.asarray(source[f"region_masks_{rho_name}"])
                for class_offset, class_id in enumerate(positive):
                    base_identity = {
                        "image_id": image_id,
                        "model": run.model,
                        "class_id": int(class_id),
                        "label_stratum": exact_stratum,
                        "num_positive_classes": len(positive),
                        "rho": rho,
                    }
                    patch_metrics = _canonical_region_metrics(
                        patch_cam[class_offset], regions[class_offset]
                    )
                    region_rows.append(
                        {
                            **base_identity,
                            "variant_code": "SOURCE",
                            "variant_name": "fixed_patch_cam",
                            "stage": "patch_cam",
                            **_region_record(patch_metrics),
                        }
                    )
                    for variant_index, code in enumerate(VARIANT_CODES):
                        for stage, values in stage_arrays.items():
                            metrics = _canonical_region_metrics(
                                values[variant_index, class_offset],
                                regions[class_offset],
                            )
                            region_rows.append(
                                {
                                    **base_identity,
                                    "variant_code": code,
                                    "variant_name": VARIANT_NAMES[code],
                                    "stage": stage,
                                    **_region_record(metrics),
                                }
                            )
                        for transition_name, source_map, destination_map in (
                            (
                                "patch_cam_to_preprop",
                                patch_cam[class_offset],
                                stage_arrays["preprop"][variant_index, class_offset],
                            ),
                            (
                                "preprop_to_final",
                                stage_arrays["preprop"][variant_index, class_offset],
                                stage_arrays["final"][variant_index, class_offset],
                            ),
                        ):
                            transition = stage_transition_metrics(
                                source_map,
                                destination_map,
                                regions[class_offset],
                                ratio=0.10,
                            )
                            transition_rows.append(
                                {
                                    **base_identity,
                                    "variant_code": code,
                                    "variant_name": VARIANT_NAMES[code],
                                    "transition": transition_name,
                                    **_transition_record(transition),
                                }
                            )
                for offset_a, offset_b in itertools.combinations(
                    range(len(positive)), 2
                ):
                    class_a = int(positive[offset_a])
                    class_b = int(positive[offset_b])
                    pair_regions = np.asarray(
                        assign_pair_patch_regions_from_counts(
                            label_counts,
                            class_a,
                            class_b,
                            rho=rho,
                            grid_size=(28, 28),
                        )["region_codes"]
                    )
                    pair_identity = {
                        "image_id": image_id,
                        "model": run.model,
                        "class_a": class_a,
                        "class_b": class_b,
                        "class_pair": f"{class_a:02d}-{class_b:02d}",
                        "label_stratum": exact_stratum,
                        "num_positive_classes": len(positive),
                        "rho": rho,
                    }
                    for variant_index, code in enumerate(VARIANT_CODES):
                        for stage, values in stage_arrays.items():
                            shared = shared_support_metrics(
                                values[variant_index, offset_a],
                                values[variant_index, offset_b],
                                pair_regions,
                                ratio=0.10,
                            )
                            pair_rows.append(
                                {
                                    **pair_identity,
                                    "variant_code": code,
                                    "variant_name": VARIANT_NAMES[code],
                                    "stage": stage,
                                    **_pair_record(shared),
                                }
                            )
    return (
        pd.DataFrame.from_records(fixed_rows),
        pd.DataFrame.from_records(region_rows),
        pd.DataFrame.from_records(pair_rows),
        pd.DataFrame.from_records(transition_rows),
        np.stack(fixed_stacks),
        aggregate,
    )


def _threshold_tables(
    aggregates: Mapping[str, np.ndarray],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    thresholds = cam_threshold_grid()
    curve_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        values = aggregates[model]
        for stratum_index, stratum in enumerate(STRATA):
            for variant_index, code in enumerate(VARIANT_CODES):
                for threshold_index, threshold in enumerate(thresholds):
                    confusion = values[stratum_index, variant_index, threshold_index]
                    if not int(confusion.sum()):
                        continue
                    metrics = cam_metrics_from_confusion(confusion)
                    row = {
                        "model": model,
                        "label_stratum": stratum,
                        "variant_code": code,
                        "variant_name": VARIANT_NAMES[code],
                        "threshold": float(threshold),
                        "mean_iou": float(metrics["mean_iou"]),
                        "binary_foreground_precision": float(
                            metrics["binary_foreground_precision"]
                        ),
                        "binary_foreground_recall": float(
                            metrics["binary_foreground_recall"]
                        ),
                        "semantic_correct_foreground_precision": float(
                            metrics["semantic_correct_foreground_precision"]
                        ),
                        "semantic_correct_foreground_recall": float(
                            metrics["semantic_correct_foreground_recall"]
                        ),
                        "evaluated_pixels": int(confusion.sum()),
                    }
                    curve_rows.append(row)
                    canonical_rows.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "model",
                                    "label_stratum",
                                    "variant_code",
                                    "variant_name",
                                    "threshold",
                                )
                            },
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
                                "variant_code": code,
                                "variant_name": VARIANT_NAMES[code],
                                "threshold": float(threshold),
                                "semantic_class_id": class_id,
                                "iou": float(iou),
                            }
                        )
    curves = pd.DataFrame.from_records(curve_rows)
    per_class = pd.DataFrame.from_records(class_rows)
    canonical = pd.DataFrame.from_records(canonical_rows)
    fixed = curves[np.isclose(curves["threshold"], PRIMARY_THRESHOLD)].copy()
    auc_rows: list[dict[str, object]] = []
    curve_metrics = (
        "mean_iou",
        "binary_foreground_precision",
        "binary_foreground_recall",
        "semantic_correct_foreground_precision",
        "semantic_correct_foreground_recall",
    )
    for keys, group in curves.groupby(
        ["model", "label_stratum", "variant_code", "variant_name"], sort=True
    ):
        group = group.sort_values("threshold")
        best = group.iloc[int(np.nanargmax(group["mean_iou"].to_numpy()))]
        for metric in curve_metrics:
            metric_values = group[metric].to_numpy(dtype=np.float64)
            auc_rows.append(
                {
                    "model": keys[0],
                    "label_stratum": keys[1],
                    "variant_code": keys[2],
                    "variant_name": keys[3],
                    "metric": metric,
                    "normalized_curve_auc": (
                        normalized_curve_auc(group["threshold"], metric_values)
                        if np.isfinite(metric_values).all()
                        else np.nan
                    ),
                    "finite_threshold_points": int(np.isfinite(metric_values).sum()),
                    "variant_best_mean_iou": float(best["mean_iou"]),
                    "variant_best_threshold_diagnostic_only": float(best["threshold"]),
                }
            )
    anchor_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        native = curves[
            (curves["model"] == model)
            & (curves["label_stratum"] == "all")
            & (curves["variant_code"] == "B0")
        ].sort_values("threshold")
        anchor = native_best_threshold_anchor(native["threshold"], native["mean_iou"])
        selected = curves[
            (curves["model"] == model)
            & np.isclose(curves["threshold"], anchor.threshold)
        ].copy()
        for _, row in selected.iterrows():
            anchor_rows.append(
                {
                    **row.to_dict(),
                    "native_b0_anchor_threshold": anchor.threshold,
                    "native_b0_anchor_mean_iou_all": anchor.native_metric,
                    "anchor_selection": "host B0/all only; lowest-threshold tie",
                }
            )
    return (
        curves,
        per_class,
        fixed,
        pd.DataFrame.from_records(auc_rows),
        pd.DataFrame.from_records(anchor_rows),
        canonical,
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
    expanded = pd.concat(
        [
            frame.assign(summary_stratum="all"),
            frame.assign(summary_stratum=frame["label_stratum"]),
        ],
        ignore_index=True,
    )
    full_groups = [*group_cols, "summary_stratum"]
    for keys, group in expanded.groupby(full_groups, sort=True):
        identity = dict(zip(full_groups, keys if isinstance(keys, tuple) else (keys,)))
        for metric in value_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    **identity,
                    "aggregation": "micro",
                    "class_id": np.nan,
                    "metric": metric,
                    "estimate": float(finite.mean()) if len(finite) else np.nan,
                    "num_rows": int(len(finite)),
                    "num_images": int(group.loc[finite.index, "image_id"].nunique()),
                }
            )
            if class_col is None:
                continue
            class_means = (
                group.assign(_value=values)
                .groupby(class_col, sort=True)["_value"]
                .mean()
            )
            finite_classes = class_means[np.isfinite(class_means)]
            rows.append(
                {
                    **identity,
                    "aggregation": "macro_class",
                    "class_id": np.nan,
                    "metric": metric,
                    "estimate": (
                        float(finite_classes.mean()) if len(finite_classes) else np.nan
                    ),
                    "num_rows": int(len(finite)),
                    "num_images": int(group.loc[finite.index, "image_id"].nunique()),
                }
            )
            for class_id, estimate in finite_classes.items():
                subset = group[group[class_col] == class_id]
                rows.append(
                    {
                        **identity,
                        "aggregation": "class_wise",
                        "class_id": class_id,
                        "metric": metric,
                        "estimate": float(estimate),
                        "num_rows": int(np.isfinite(subset[metric]).sum()),
                        "num_images": int(subset["image_id"].nunique()),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _finite_ci(samples: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _paired_class_bootstrap(
    frame: pd.DataFrame,
    *,
    comparison: str,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    class_col: str,
    draws: ImageBootstrapDraws,
    identity: Mapping[str, object],
) -> pd.DataFrame:
    """Efficient paired micro, equal-class, and class-wise image bootstrap.

    Every metric first uses the exact finite intersection of B0 and the compared
    readout.  Whole-image multiplicities are then applied to sufficient
    statistics; rows, classes, and class pairs are never resampled separately.
    """

    required = {"variant_code", "image_id", class_col, *key_cols, *value_cols}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"paired bootstrap frame lacks columns: {sorted(missing)}")
    if comparison not in PRIMARY_COMPARISONS:
        raise ValueError(f"unsupported paired B comparison: {comparison}")
    if "image_id" not in key_cols or class_col not in key_cols:
        raise ValueError("paired keys must contain image_id and the class axis")
    selected = frame[frame["variant_code"].isin(("B0", comparison))]
    if selected.duplicated(["variant_code", *key_cols]).any():
        raise ValueError("paired B readouts contain duplicate system/key rows")
    baseline = selected[selected["variant_code"] == "B0"][[*key_cols, *value_cols]]
    compared = selected[selected["variant_code"] == comparison][
        [*key_cols, *value_cols]
    ]
    paired = baseline.merge(
        compared,
        on=list(key_cols),
        how="outer",
        suffixes=("_baseline", "_comparison"),
        indicator=True,
        validate="one_to_one",
    )
    if paired.empty or not bool((paired["_merge"] == "both").all()):
        raise ValueError("paired B readouts must have identical non-empty keys")
    paired = paired.drop(columns="_merge")
    draw_lookup = {image_id: index for index, image_id in enumerate(draws.image_ids)}
    frame_images = set(paired["image_id"].astype(str))
    if not frame_images.issubset(draw_lookup):
        raise ValueError("paired B rows reference images outside supplied draws")
    class_values = tuple(sorted(paired[class_col].unique().tolist()))
    if not class_values or bool(paired[class_col].isna().any()):
        raise ValueError(f"{class_col} must define a non-empty class axis")
    class_lookup = {value: index for index, value in enumerate(class_values)}
    image_indices_all = np.asarray(
        [draw_lookup[str(value)] for value in paired["image_id"]], dtype=np.int64
    )
    class_indices_all = np.asarray(
        [class_lookup[value] for value in paired[class_col]], dtype=np.int64
    )
    multiplicities = np.asarray(draws.multiplicities, dtype=np.float64)
    series_names = ("B0", comparison, f"{comparison}_minus_B0")
    reserved = {
        "metric",
        "aggregation",
        "series",
        "estimate",
        "ci_low",
        "ci_high",
        "paired_delta",
    }
    overlap = reserved.intersection(identity)
    if overlap:
        raise ValueError(
            f"bootstrap identity replaces statistic fields: {sorted(overlap)}"
        )

    def record(
        *,
        metric: str,
        aggregation: str,
        series_index: int,
        estimate: float,
        samples: np.ndarray,
        finite_images: int,
        finite_rows: int,
        finite_classes: int | None,
        class_value: object | None = None,
    ) -> dict[str, object]:
        low, high, valid = _finite_ci(samples)
        series = series_names[series_index]
        result: dict[str, object] = {
            **identity,
            "metric": metric,
            "aggregation": aggregation,
            "series": series,
            "estimate": float(estimate),
            "ci_low": low,
            "ci_high": high,
            "num_images": finite_images,
            "num_images_total": len(draws.image_ids),
            "num_rows": finite_rows,
            "num_rows_total": len(paired),
            "num_classes": finite_classes,
            "bootstrap_repeats": draws.repeats,
            "bootstrap_valid_repeats": valid,
            "bootstrap_valid_fraction": valid / draws.repeats,
            "bootstrap_seed": draws.seed,
            "bootstrap_unit": "image",
            "ci_method": "95% percentile",
            "paired_delta": series_index == 2,
            "delta_definition": (f"{comparison} - B0" if series_index == 2 else ""),
            "class_axis": class_col,
            "class_value": class_value,
            class_col: class_value,
        }
        return result

    rows: list[dict[str, object]] = []
    for metric in value_cols:
        left = pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        right = pd.to_numeric(paired[f"{metric}_comparison"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(left) & np.isfinite(right)
        if not finite.any():
            continue
        image_indices = image_indices_all[finite]
        class_indices = class_indices_all[finite]
        values = np.column_stack(
            (left[finite], right[finite], right[finite] - left[finite])
        )

        image_sums = np.zeros((len(draws.image_ids), 3), dtype=np.float64)
        image_counts = np.zeros(len(draws.image_ids), dtype=np.float64)
        np.add.at(image_sums, image_indices, values)
        np.add.at(image_counts, image_indices, 1.0)
        active_images = image_counts > 0
        micro_point = image_sums.sum(axis=0) / image_counts.sum()
        micro_numerator = multiplicities[:, active_images] @ image_sums[active_images]
        micro_denominator = (
            multiplicities[:, active_images] @ image_counts[active_images]
        )
        micro_samples = np.divide(
            micro_numerator,
            micro_denominator[:, None],
            out=np.full_like(micro_numerator, np.nan),
            where=micro_denominator[:, None] > 0,
        )

        class_points = np.full((len(class_values), 3), np.nan, dtype=np.float64)
        class_samples = np.full(
            (draws.repeats, len(class_values), 3), np.nan, dtype=np.float64
        )
        for class_index, class_value in enumerate(class_values):
            in_class = class_indices == class_index
            if not in_class.any():
                continue
            sums = np.zeros((len(draws.image_ids), 3), dtype=np.float64)
            counts = np.zeros(len(draws.image_ids), dtype=np.float64)
            np.add.at(sums, image_indices[in_class], values[in_class])
            np.add.at(counts, image_indices[in_class], 1.0)
            active = counts > 0
            class_points[class_index] = sums.sum(axis=0) / counts.sum()
            numerator = multiplicities[:, active] @ sums[active]
            denominator = multiplicities[:, active] @ counts[active]
            class_samples[:, class_index] = np.divide(
                numerator,
                denominator[:, None],
                out=np.full_like(numerator, np.nan),
                where=denominator[:, None] > 0,
            )
            for series_index in range(3):
                rows.append(
                    record(
                        metric=metric,
                        aggregation="class_wise",
                        series_index=series_index,
                        estimate=class_points[class_index, series_index],
                        samples=class_samples[:, class_index, series_index],
                        finite_images=int(active.sum()),
                        finite_rows=int(in_class.sum()),
                        finite_classes=1,
                        class_value=class_value,
                    )
                )

        macro_point = np.nanmean(class_points, axis=0)
        valid_class_counts = np.isfinite(class_samples).sum(axis=1)
        macro_samples = np.divide(
            np.nansum(class_samples, axis=1),
            valid_class_counts,
            out=np.full((draws.repeats, 3), np.nan),
            where=valid_class_counts > 0,
        )
        for series_index in range(3):
            rows.append(
                record(
                    metric=metric,
                    aggregation="micro",
                    series_index=series_index,
                    estimate=micro_point[series_index],
                    samples=micro_samples[:, series_index],
                    finite_images=int(active_images.sum()),
                    finite_rows=int(finite.sum()),
                    finite_classes=None,
                )
            )
            rows.append(
                record(
                    metric=metric,
                    aggregation="macro_class",
                    series_index=series_index,
                    estimate=macro_point[series_index],
                    samples=macro_samples[:, series_index],
                    finite_images=int(active_images.sum()),
                    finite_rows=int(finite.sum()),
                    finite_classes=int(
                        np.isfinite(class_points[:, series_index]).sum()
                    ),
                )
            )
    return pd.DataFrame.from_records(rows)


def _paired_region_bootstrap(
    region: pd.DataFrame,
    pair: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_outputs: list[pd.DataFrame] = []
    pair_outputs: list[pd.DataFrame] = []
    for stratum in STRATA:
        if stratum == "all":
            region_stratum = region
            pair_stratum = pair
        else:
            region_stratum = region[region["label_stratum"] == stratum]
            pair_stratum = pair[pair["label_stratum"] == stratum]
        if region_stratum.empty:
            continue
        # The exact same whole-image draws are reconstructed for both hosts.
        draws = image_multinomial_draws(
            region_stratum["image_id"].unique().tolist(), repeats=repeats, seed=seed
        )
        for model in MODEL_ORDER:
            model_region = region_stratum[
                (region_stratum["model"] == model)
                & np.isclose(region_stratum["rho"], 0.5)
                & region_stratum["variant_code"].isin(("B0", "B1", "B4"))
            ]
            for stage in ("attention", "preprop", "final"):
                stage_region = model_region[model_region["stage"] == stage]
                if stage_region.empty:
                    continue
                for comparison in PRIMARY_COMPARISONS:
                    region_outputs.append(
                        _paired_class_bootstrap(
                            stage_region,
                            comparison=comparison,
                            key_cols=("image_id", "class_id"),
                            value_cols=REGION_OUTPUT_METRICS,
                            class_col="class_id",
                            draws=draws,
                            identity={
                                "model": model,
                                "label_stratum": stratum,
                                "stage": stage,
                                "rho": 0.5,
                                "comparison_role": (
                                    "primary" if comparison == "B1" else "secondary"
                                ),
                            },
                        )
                    )
        if pair_stratum.empty or stratum == "single":
            continue
        pair_draws = image_multinomial_draws(
            pair_stratum["image_id"].unique().tolist(), repeats=repeats, seed=seed
        )
        for model in MODEL_ORDER:
            model_pair = pair_stratum[
                (pair_stratum["model"] == model)
                & np.isclose(pair_stratum["rho"], 0.5)
                & pair_stratum["variant_code"].isin(("B0", "B1", "B4"))
            ]
            for stage in ("attention", "preprop", "final"):
                stage_pair = model_pair[model_pair["stage"] == stage]
                if stage_pair.empty:
                    continue
                for comparison in PRIMARY_COMPARISONS:
                    pair_outputs.append(
                        _paired_class_bootstrap(
                            stage_pair,
                            comparison=comparison,
                            key_cols=(
                                "image_id",
                                "class_a",
                                "class_b",
                                "class_pair",
                            ),
                            value_cols=PAIR_OUTPUT_METRICS,
                            class_col="class_pair",
                            draws=pair_draws,
                            identity={
                                "model": model,
                                "label_stratum": stratum,
                                "stage": stage,
                                "rho": 0.5,
                                "comparison_role": (
                                    "primary" if comparison == "B1" else "secondary"
                                ),
                            },
                        )
                    )
    return (
        pd.concat(region_outputs, ignore_index=True)
        if region_outputs
        else pd.DataFrame(),
        pd.concat(pair_outputs, ignore_index=True) if pair_outputs else pd.DataFrame(),
    )


def _paired_transition_bootstrap(
    transition: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for stratum in STRATA:
        selected_stratum = (
            transition
            if stratum == "all"
            else transition[transition["label_stratum"] == stratum]
        )
        if selected_stratum.empty:
            continue
        draws = image_multinomial_draws(
            selected_stratum["image_id"].unique().tolist(),
            repeats=repeats,
            seed=seed,
        )
        for model in MODEL_ORDER:
            model_frame = selected_stratum[
                (selected_stratum["model"] == model)
                & np.isclose(selected_stratum["rho"], 0.5)
                & selected_stratum["variant_code"].isin(("B0", "B1", "B4"))
            ]
            for transition_name in ("patch_cam_to_preprop", "preprop_to_final"):
                stage_frame = model_frame[model_frame["transition"] == transition_name]
                if stage_frame.empty:
                    continue
                for comparison in PRIMARY_COMPARISONS:
                    outputs.append(
                        _paired_class_bootstrap(
                            stage_frame,
                            comparison=comparison,
                            key_cols=("image_id", "class_id"),
                            value_cols=TRANSITION_OUTPUT_METRICS,
                            class_col="class_id",
                            draws=draws,
                            identity={
                                "model": model,
                                "label_stratum": stratum,
                                "transition": transition_name,
                                "rho": 0.5,
                                "comparison_role": (
                                    "primary" if comparison == "B1" else "secondary"
                                ),
                            },
                        )
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _paired_cam_bootstrap(
    image_ids: Sequence[str],
    label_counts: Sequence[int],
    fixed_stacks: Mapping[str, np.ndarray],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    ids = np.asarray(image_ids)
    counts = np.asarray(label_counts)
    for stratum in STRATA:
        if stratum == "all":
            selected = np.ones(len(ids), dtype=bool)
        elif stratum == "single":
            selected = counts == 1
        elif stratum == "exactly_2":
            selected = counts == 2
        else:
            selected = counts >= 3
        if not selected.any():
            continue
        selected_ids = ids[selected].tolist()
        draws = image_multinomial_draws(selected_ids, repeats=repeats, seed=seed)
        for model in MODEL_ORDER:
            stacks = fixed_stacks[model][selected]
            for comparison in PRIMARY_COMPARISONS:
                comparison_index = VARIANT_CODES.index(comparison)
                summary = paired_confusion_metric_summary(
                    selected_ids,
                    stacks[:, 0],
                    stacks[:, comparison_index],
                    baseline_name="B0",
                    comparison_name=comparison,
                    draws=draws,
                    identity={
                        "model": model,
                        "label_stratum": stratum,
                        "threshold": PRIMARY_THRESHOLD,
                        "comparison_role": (
                            "primary" if comparison == "B1" else "secondary"
                        ),
                    },
                )
                outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def _write_plots(
    output: Path, curves: pd.DataFrame, region_summary: pd.DataFrame
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir()
    for model in MODEL_ORDER:
        selected = curves[
            (curves["model"] == model) & (curves["label_stratum"] == "all")
        ]
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        for code in VARIANT_CODES:
            group = selected[selected["variant_code"] == code].sort_values("threshold")
            axis.plot(group["threshold"], 100.0 * group["mean_iou"], label=code)
        axis.axvline(PRIMARY_THRESHOLD, color="black", linestyle="--", linewidth=0.8)
        axis.set(xlabel="Background threshold", ylabel="Raw CAM mIoU (%)", title=model)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{model}_miou_threshold_curve.png", dpi=180)
        plt.close(fig)
    selected_region = region_summary[
        (region_summary["summary_stratum"] == "all")
        & (region_summary["aggregation"] == "micro")
        & (region_summary["stage"] == "attention")
        & np.isclose(region_summary["rho"], 0.5)
        & (region_summary["metric"] == "target_other_auroc")
        & region_summary["variant_code"].isin(VARIANT_CODES)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for axis, model in zip(axes, MODEL_ORDER):
        group = selected_region[selected_region["model"] == model].set_index(
            "variant_code"
        )
        values = [group.loc[code, "estimate"] for code in VARIANT_CODES]
        axis.bar(VARIANT_CODES, values)
        axis.set(
            title=model, xlabel="Diagnostic readout", ylabel="Target-vs-other AUROC"
        )
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "attention_target_other_auroc.png", dpi=180)
    plt.close(fig)


def _delta_sentence(
    bootstrap: pd.DataFrame,
    model: str,
    comparison: str,
    metric: str,
    *,
    filters: Mapping[str, object] | None = None,
) -> str:
    rows = bootstrap[
        (bootstrap["model"] == model)
        & (bootstrap["label_stratum"] == "all")
        & (bootstrap["series"] == f"{comparison}_minus_B0")
        & (bootstrap["metric"] == metric)
    ]
    if "aggregation" in rows.columns:
        rows = rows[rows["aggregation"] == "micro"]
    for column, value in (filters or {}).items():
        if column not in rows.columns:
            raise RuntimeError(f"bootstrap table lacks report filter column: {column}")
        if isinstance(value, float):
            rows = rows[np.isclose(pd.to_numeric(rows[column]), value)]
        else:
            rows = rows[rows[column] == value]
    if rows.empty:
        return "N/A"
    if len(rows) != 1:
        raise RuntimeError(
            f"report statistic is ambiguous for {model}/{comparison}/{metric}: "
            f"{len(rows)} rows"
        )
    row = rows.iloc[0]
    return f"{float(row['estimate']):+.4f} [{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"


def _write_report(
    output: Path,
    validated: ValidatedInputs,
    fixed: pd.DataFrame,
    anchor: pd.DataFrame,
    cam_bootstrap: pd.DataFrame,
    region_bootstrap: pd.DataFrame,
    pair_bootstrap: pd.DataFrame,
    transition_bootstrap: pd.DataFrame,
) -> None:
    run_kind = validated.runs["mctformer"].run_kind
    lines = [
        "# Validation B: CAM Layer Readout",
        "",
        f"- [Fact] Run kind: `{run_kind}`; matched images: {len(validated.image_ids):,}.",
        "- [Fact] B0--B5 are frozen-model diagnostic readouts, not trained or proposed methods.",
        "- [Fact] Both hosts retain their native CAM fusion formula and one fixed all-layer A_p2p per image.",
        "- [Fact] Primary raw-CAM endpoint uses threshold 0.45; the common robustness grid is 0.20--0.60 by 0.01.",
        "- [Fact] Foreground precision/recall are reported in both binary-FG and semantic-correct-FG definitions.",
        "- [Fact] Object-size stratification is N/A: the existing matched GT tooling does not define a pre-registered object-size estimator.",
        "",
        "## Fixed-threshold findings",
        "",
    ]
    for model in MODEL_ORDER:
        subset = fixed[
            (fixed["model"] == model) & (fixed["label_stratum"] == "all")
        ].set_index("variant_code")
        values = ", ".join(
            f"{code}={100.0 * float(subset.loc[code, 'mean_iou']):.2f}%"
            for code in VARIANT_CODES
        )
        lines.extend(
            [
                f"- [Fact] `{model}` raw-CAM mIoU at 0.45: {values}.",
                f"- [Statistical inference] `{model}` B1-B0 delta mIoU (95% image-clustered percentile CI): {_delta_sentence(cam_bootstrap, model, 'B1', 'mean_iou')}.",
                f"- [Statistical inference] `{model}` B4-B0 delta mIoU (95% image-clustered percentile CI): {_delta_sentence(cam_bootstrap, model, 'B4', 'mean_iou')}.",
            ]
        )
    lines.extend(["", "## Native-B0 anchor and ownership", ""])
    for model in MODEL_ORDER:
        host_anchor = anchor[
            (anchor["model"] == model)
            & (anchor["label_stratum"] == "all")
            & (anchor["variant_code"] == "B0")
        ].iloc[0]
        lines.append(
            f"- [Fact] `{model}` native-B0 best-threshold anchor is {float(host_anchor['native_b0_anchor_threshold']):.2f}; every B variant is sampled at that same host anchor in `native_b0_anchor.csv`."
        )
        region_delta = _delta_sentence(
            region_bootstrap,
            model,
            "B1",
            "target_other_auroc",
            filters={"stage": "attention", "rho": 0.5},
        )
        preprop_delta = _delta_sentence(
            region_bootstrap,
            model,
            "B1",
            "target_other_auroc",
            filters={"stage": "preprop", "rho": 0.5},
        )
        final_delta = _delta_sentence(
            region_bootstrap,
            model,
            "B1",
            "target_other_auroc",
            filters={"stage": "final", "rho": 0.5},
        )
        pair_delta = _delta_sentence(
            pair_bootstrap,
            model,
            "B1",
            "top10_jaccard",
            filters={"stage": "attention", "rho": 0.5},
        )
        introduced_bg_delta = _delta_sentence(
            transition_bootstrap,
            model,
            "B1",
            "introduced_background_fraction",
            filters={"transition": "preprop_to_final", "rho": 0.5},
        )
        lines.extend(
            [
                f"- [Statistical inference] `{model}` attention-level B1-B0 target-vs-other AUROC delta: {region_delta}.",
                f"- [Statistical inference] `{model}` pre-propagation/final-CAM B1-B0 target-vs-other AUROC deltas: {preprop_delta} / {final_delta}.",
                f"- [Statistical inference] `{model}` attention-level positive-class-pair top10 Jaccard delta: {pair_delta}.",
                f"- [Statistical inference] `{model}` B1-B0 introduced-background fraction for pre-propagation→final: {introduced_bg_delta}.",
            ]
        )
    scope = (
        "Smoke output validates computation and provenance only; it cannot support a population conclusion."
        if run_kind == "smoke"
        else "Intervals resample whole VOC images; patches, image-class rows, and class-pair rows from one image remain clustered."
    )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            f"- [Fact] {scope}",
            "- [Mechanistic interpretation] A readout-level change that is already visible in A_c2p and survives pre-propagation/final CAM is consistent with late-layer routing quality carrying through the native CAM pipeline.",
            "- [Mechanistic interpretation] A change in attention ownership without a corresponding fixed-threshold CAM gain is consistent with compensation by patch-CAM fusion, A_p2p propagation, or normalization; it does not identify which component is causal.",
            "- [Unsupported] These diagnostics do not establish background leakage, attention/CAM causality, a deployable layer-selection rule, or a new method.",
            "- [Unsupported] Variant-specific best thresholds are diagnostic only and must not replace the fixed 0.45 endpoint or host-native-B0 anchor.",
            "",
            "## Artifacts",
            "",
            "Canonical Parquets preserve fixed-threshold per-image confusions, aggregate threshold confusions, image-class region metrics, class-pair shared support, and stage transitions. Compact CSV tables provide threshold curves, class IoU, anchors, normalized curve AUC, point summaries, and paired bootstrap intervals.",
            "",
        ]
    )
    (output / "VALIDATION_B_CAM_LAYER_READOUT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def execute(args: argparse.Namespace) -> Path:
    require_tgca_repro()
    validated = validate_inputs(args)
    _validate_bootstrap_policy(
        {run.run_kind for run in validated.runs.values()},
        args.bootstrap_repeats,
        allow_smoke=bool(args.allow_smoke),
    )
    immutable = [
        validated.source_metadata_path,
        validated.linkage_path,
        *(run.root for run in validated.runs.values()),
        *(run.source_root for run in validated.runs.values()),
    ]
    output = assert_new_output(args.output_dir, immutable)
    output.mkdir(parents=True, exist_ok=False)
    fixed_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []
    fixed_stacks: dict[str, np.ndarray] = {}
    aggregates: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        fixed, region, pair, transition, stacks, aggregate = _collect_run(
            validated.runs[model]
        )
        fixed_frames.append(fixed)
        region_frames.append(region)
        pair_frames.append(pair)
        transition_frames.append(transition)
        fixed_stacks[model] = stacks
        aggregates[model] = aggregate
    fixed_canonical = pd.concat(fixed_frames, ignore_index=True)
    region = pd.concat(region_frames, ignore_index=True)
    pair = pd.concat(pair_frames, ignore_index=True)
    transition = pd.concat(transition_frames, ignore_index=True)
    (
        curves,
        per_class,
        fixed_metrics,
        curve_auc,
        anchors,
        aggregate_canonical,
    ) = _threshold_tables(aggregates)
    region_summary = _point_summary(
        region,
        group_cols=("model", "variant_code", "variant_name", "stage", "rho"),
        value_cols=REGION_OUTPUT_METRICS,
        class_col="class_id",
    )
    pair_summary = _point_summary(
        pair,
        group_cols=("model", "variant_code", "variant_name", "stage", "rho"),
        value_cols=PAIR_OUTPUT_METRICS,
        class_col="class_pair",
    )
    transition_summary = _point_summary(
        transition,
        group_cols=("model", "variant_code", "variant_name", "transition", "rho"),
        value_cols=TRANSITION_OUTPUT_METRICS,
        class_col="class_id",
    )
    label_counts = [
        len(row["positive_class_ids"]) for row in validated.runs["mctformer"].manifest
    ]
    cam_bootstrap = _paired_cam_bootstrap(
        validated.image_ids,
        label_counts,
        fixed_stacks,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    region_bootstrap, pair_bootstrap = _paired_region_bootstrap(
        region,
        pair,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    transition_bootstrap = _paired_transition_bootstrap(
        transition,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    parquet_tables = {
        "canonical_image_cam_t045.parquet": fixed_canonical,
        "canonical_aggregate_threshold_confusions.parquet": aggregate_canonical,
        "canonical_image_class_region_metrics.parquet": region,
        "canonical_positive_class_pair_metrics.parquet": pair,
        "canonical_stage_transitions.parquet": transition,
    }
    csv_tables = {
        "threshold_curves.csv": curves,
        "per_class_iou_thresholds.csv": per_class,
        "fixed_t045_metrics.csv": fixed_metrics,
        "normalized_curve_auc.csv": curve_auc,
        "native_b0_anchor.csv": anchors,
        "region_metric_summary.csv": region_summary,
        "class_pair_metric_summary.csv": pair_summary,
        "stage_transition_summary.csv": transition_summary,
        "paired_cam_bootstrap.csv": cam_bootstrap,
        "paired_region_bootstrap.csv": region_bootstrap,
        "paired_class_pair_bootstrap.csv": pair_bootstrap,
        "paired_stage_transition_bootstrap.csv": transition_bootstrap,
    }
    for name, frame in parquet_tables.items():
        frame.to_parquet(output / name, index=False)
    for name, frame in csv_tables.items():
        frame.to_csv(output / name, index=False)
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
    json_dump(
        output / "canonical_metadata.json",
        {
            "status": "complete",
            "schema_version": 1,
            "analysis": ANALYSIS_NAME,
            "models": list(MODEL_ORDER),
            "variant_order": list(VARIANT_CODES),
            "thresholds": cam_threshold_grid().tolist(),
            "primary_threshold": PRIMARY_THRESHOLD,
            "files": canonical_files,
        },
    )
    _write_plots(output, curves, region_summary)
    _write_report(
        output,
        validated,
        fixed_metrics,
        anchors,
        cam_bootstrap,
        region_bootstrap,
        pair_bootstrap,
        transition_bootstrap,
    )
    command = " ".join(shlex.quote(value) for value in sys.argv)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    post_hashes = {
        "source_metadata": sha256_file(validated.source_metadata_path),
        "experiment2_linkage": sha256_file(validated.linkage_path),
    }
    for model, run in validated.runs.items():
        post_hashes[f"{model}_run_metadata"] = sha256_file(run.root / "metadata.json")
        post_hashes[f"{model}_run_completion"] = sha256_file(
            run.root / "completion.json"
        )
        post_hashes[f"{model}_run_manifest"] = sha256_file(run.root / "manifest.jsonl")
        post_hashes[f"{model}_source_manifest"] = sha256_file(
            run.source_root / "manifest.jsonl"
        )
    if post_hashes != dict(validated.input_hashes):
        raise RuntimeError(
            "immutable source metadata/manifests changed during analysis"
        )
    consumed_after = _assert_consumed_files_unchanged(validated.consumed_file_hashes)
    consumed_manifest = pd.DataFrame.from_records(
        [
            {
                "path": path,
                "sha256_before": digest,
                "sha256_after": consumed_after[path],
                "unchanged": consumed_after[path] == digest,
            }
            for path, digest in sorted(validated.consumed_file_hashes.items())
        ]
    )
    consumed_manifest_path = output / "consumed_input_manifest.csv"
    consumed_manifest.to_csv(consumed_manifest_path, index=False)
    generated: dict[str, object] = {}
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        if path.name == "analysis_metadata.json":
            continue
        generated[str(path.relative_to(output))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metadata = {
        "status": "complete",
        "analysis": ANALYSIS_NAME,
        "run_kind": validated.runs["mctformer"].run_kind,
        "models": list(MODEL_ORDER),
        "num_images": len(validated.image_ids),
        "positive_image_class_pairs": int(sum(label_counts)),
        "multilabel_images": int(sum(value >= 2 for value in label_counts)),
        "variants": list(VARIANT_CODES),
        "thresholds": cam_threshold_grid().tolist(),
        "primary_threshold": PRIMARY_THRESHOLD,
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "whole image multinomial multiplicity",
            "ci": "95% percentile",
            "same_draws_reused_for_paired_variants_and_hosts": True,
        },
        "object_size_strata": "N/A; no pre-registered existing GT tool",
        "source_metadata": str(validated.source_metadata_path),
        "source_metadata_sha256": validated.input_hashes["source_metadata"],
        "run_roots": {model: str(run.root) for model, run in validated.runs.items()},
        "source_signal_roots": {
            model: str(run.source_root) for model, run in validated.runs.items()
        },
        "input_hashes_before_and_after_equal": True,
        "input_hashes": dict(validated.input_hashes),
        "source_immutability_verified": True,
        "consumed_immutable_files": len(consumed_after),
        "consumed_input_manifest": {
            "path": str(consumed_manifest_path),
            "rows": len(consumed_manifest),
            "sha256": sha256_file(consumed_manifest_path),
        },
        "paired_bootstrap_coverage": {
            "region_stages": ["attention", "preprop", "final"],
            "transition_stages": ["patch_cam_to_preprop", "preprop_to_final"],
            "rho": 0.5,
            "aggregations": ["micro", "macro_class", "class_wise"],
            "class_pair_macro_axis": "class_pair",
        },
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
