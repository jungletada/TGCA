#!/usr/bin/env python3
"""Canonicalize and analyze Experiment 3 Validation A outputs.

This program is deliberately inference-free.  It treats the two Presence-Axis
run roots and their linked Experiment 2 signal roots as immutable inputs,
validates every consumed artifact, writes bounded canonical Parquet tables,
and performs only whole-image clustered inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.stats import rankdata  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.build_experiment2_canonical import (  # noqa: E402
    ManifestEntry as Experiment2ManifestEntry,
    assert_exact_manifest_match as assert_experiment2_manifest_match,
    load_and_validate_artifact as load_experiment2_artifact,
    load_signal_root as load_experiment2_signal_root,
)
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES  # noqa: E402
from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    TOPK_RATIOS,
    map_overlap_metrics,
    region_map_metrics,
    stable_topk_mask,
)
from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (  # noqa: E402
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    PAIR_REGION_VOID,
    REGION_BACKGROUND,
    REGION_VOID,
    assign_pair_patch_regions_from_counts,
)
from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPEATS,
    ImageBootstrapDraws,
    image_multinomial_draws,
    paired_clustered_mean_summary,
    summarize_clustered_means,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    STRICT_TOLERANCE,
    assert_new_output,
    json_dump,
    ordered_val_ids,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)
from analysis.lazy_assignment.experiment3.presence_axis import (  # noqa: E402
    sha256_two_fold,
)


MODEL_ORDER = ("mctformer", "mctformer_plus")
CONTROL_LAYERS = (4, 5, 9, 10, 11, 12)
PRIMARY_LATE_LAYERS = (10, 11, 12)
TOPK_RATIO_VALUES = tuple(float(value) for value in TOPK_RATIOS)
REGION_THRESHOLDS = (0.5, 0.7)
CLASS_LOGIT_THRESHOLD = 0.0
FINAL_DIRECTION_ALIGNMENT_THRESHOLD = 0.90
PRESENCE_AUROC_NULL = 0.50
CANONICAL_SCHEMA_VERSION = "experiment3-validation-a-v1"
PAIR_COUNT = EXPECTED_CLASSES * (EXPECTED_CLASSES - 1) // 2
SEMANTIC_OVERLAP_ESTIMAND = (
    "rho=0.5 semantic patch regions; Spearman and top-k supports both exclude "
    "pair-region void for positive class pairs and target-class region void for "
    "within-class probe linkage"
)

DECISION_RULES: Mapping[str, object] = {
    "qualitative_rule_pre_registered_in_plan": True,
    "numeric_operationalization_frozen_before_full_voc_analysis": True,
    "numeric_preregistration_claimed": False,
    "primary_model": "mctformer_plus",
    "primary_layers": list(PRIMARY_LATE_LAYERS),
    "primary_layer_pooling": "L10-L12 rows pooled; image remains bootstrap cluster",
    "decrease_support": "paired 95% percentile CI upper bound < 0",
    "increase_support": "paired 95% percentile CI lower bound > 0",
    "presence_projection_support": "OOF AUROC 95% CI lower bound > 0.5",
    "signed_final_direction_alignment_threshold": FINAL_DIRECTION_ALIGNMENT_THRESHOLD,
    "class_prediction_threshold": CLASS_LOGIT_THRESHOLD,
    "bootstrap_repeats": DEFAULT_BOOTSTRAP_REPEATS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_unit": "image",
    "fit_uncertainty": "conditional on the two fixed cross-fitted directions",
    "semantic_overlap_estimand": SEMANTIC_OVERLAP_ESTIMAND,
    "per_layer_policy": "descriptive; no layer selected from observed results",
}

PRESENCE_REQUIRED_KEYS = frozenset(
    {
        "image_id",
        "eval_fold",
        "fit_fold",
        "positive_class_ids",
        "class_removed_scores",
        "patch_removed_scores",
        "both_removed_scores",
        "shared_both_removed_scores",
        "class_coefficients",
        "class_axis_energy",
        "class_norms",
        "class_residual_norms",
        "patch_coefficient_mean",
        "patch_coefficient_std",
        "patch_axis_energy_mean",
        "patch_axis_energy_std",
        "raw_pair_cosine_all",
        "residual_pair_cosine_all",
        "pair_axis_dot_all",
        "pair_residual_dot_all",
        "heldout_projection_all",
        "shared_axis_energy_all",
        "class_logits_all",
        "source_signal_sha256",
        "control_layer_ids",
        "raw_control_all",
        "class_removed_control_all",
        "patch_removed_control_all",
        "both_removed_control_all",
        "shared_both_removed_control_all",
        "feature_norm_control_all",
        "qk_control_all",
        "attention_conditional_control_all",
    }
)
PRESENCE_OPTIONAL_KEYS = frozenset(
    {"schema_version", "grid_h", "grid_w", "num_layers", "num_patches"}
)
DIRECTION_KEYS = frozenset(
    {
        "fit_means",
        "class_deltas",
        "shared_directions",
        "loo_shared_directions",
        "class_alignment",
        "loo_class_alignment",
        "total_counts",
        "positive_counts",
        "negative_counts",
    }
)

DERIVED_VARIANTS = (
    "raw",
    "class_removed",
    "patch_removed",
    "both_removed",
    "shared_both_removed",
)
SHARED_OWNERSHIP_VARIANTS = (*DERIVED_VARIANTS, "norm_timing_aligned")
POSITIVE_PROBE_VARIANTS = (
    *DERIVED_VARIANTS,
    "norm1_pre_same_index",
    "norm_timing_aligned",
    "qk_mean",
    "attn_c2p_conditional",
)
PROBE_LINKS = (
    ("raw_to_both_removed", "raw", "both_removed"),
    ("raw_to_norm_timing_aligned", "raw", "norm_timing_aligned"),
    (
        "both_removed_to_norm_timing_aligned",
        "both_removed",
        "norm_timing_aligned",
    ),
    ("raw_to_qk", "raw", "qk_mean"),
    ("both_removed_to_qk", "both_removed", "qk_mean"),
    ("raw_to_attention", "raw", "attn_c2p_conditional"),
    ("both_removed_to_attention", "both_removed", "attn_c2p_conditional"),
)

REGION_METRIC_FIELDS = (
    "has_target_region",
    "target_hit",
    "other_fg_hit",
    "background_hit",
    "mixed_hit",
    "degenerate_map",
    "num_target",
    "num_other_fg",
    "num_bg",
    "num_mixed",
    "num_void",
    "num_valid",
    "target_top05_fraction",
    "other_fg_top05_fraction",
    "bg_top05_fraction",
    "target_tail_enrich_05",
    "other_fg_tail_enrich_05",
    "bg_tail_enrich_05",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "target_tail_enrich_10",
    "other_fg_tail_enrich_10",
    "bg_tail_enrich_10",
    "target_top20_fraction",
    "other_fg_top20_fraction",
    "bg_top20_fraction",
    "target_tail_enrich_20",
    "other_fg_tail_enrich_20",
    "bg_tail_enrich_20",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "conditional_bg_mass",
    "target_bg_mean_margin",
    "target_other_mean_margin",
    "score_std",
    "total_variation_over_std",
)


def _field(name: str, dtype: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, dtype, nullable=nullable)


BASE_IMAGE_FIELDS = (
    _field("model", pa.string()),
    _field("image_id", pa.string()),
    _field("eval_fold", pa.int8()),
    _field("fit_fold", pa.int8()),
    _field("num_positive_classes", pa.int8()),
    _field("label_stratum", pa.string()),
)

CANONICAL_SCHEMAS: Mapping[str, pa.Schema] = {
    "token_axis": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_id", pa.int8()),
            _field("class_name", pa.string()),
            _field("target", pa.bool_()),
            _field("class_logit", pa.float32()),
            _field("predicted_positive", pa.bool_()),
            _field("presence_status", pa.string()),
            _field("prediction_status", pa.string()),
            _field("class_coefficient", pa.float32()),
            _field("class_axis_energy", pa.float32()),
            _field("class_norm", pa.float32()),
            _field("class_residual_norm", pa.float32()),
            _field("shared_axis_energy", pa.float32()),
        ]
    ),
    "token_pairs": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_a", pa.int8()),
            _field("class_b", pa.int8()),
            _field("class_a_name", pa.string()),
            _field("class_b_name", pa.string()),
            _field("target_a", pa.bool_()),
            _field("target_b", pa.bool_()),
            _field("predicted_positive_a", pa.bool_()),
            _field("predicted_positive_b", pa.bool_()),
            _field("presence_pair_status", pa.string()),
            _field("prediction_pair_status", pa.string()),
            _field("raw_pair_cosine", pa.float32()),
            _field("residual_pair_cosine", pa.float32()),
            _field("cosine_delta", pa.float32()),
            _field("axis_dot", pa.float32()),
            _field("residual_dot", pa.float32()),
            _field("reconstructed_dot", pa.float32()),
        ]
    ),
    "patch_axis": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("patch_coefficient_mean", pa.float32()),
            _field("patch_coefficient_std", pa.float32()),
            _field("patch_axis_energy_mean", pa.float32()),
            _field("patch_axis_energy_std", pa.float32()),
        ]
    ),
    "oof_projection": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_id", pa.int8()),
            _field("class_name", pa.string()),
            _field("target", pa.bool_()),
            _field("predicted_positive", pa.bool_()),
            _field("heldout_projection", pa.float32()),
            _field("shared_axis_energy", pa.float32()),
        ]
    ),
    "probe_region": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_id", pa.int8()),
            _field("class_name", pa.string()),
            _field("class_logit", pa.float32()),
            _field("predicted_positive", pa.bool_()),
            _field("variant", pa.string()),
            _field("representation_timing", pa.string()),
            _field("rho", pa.float32()),
            *[
                _field(
                    name,
                    pa.bool_()
                    if name
                    in {
                        "has_target_region",
                        "target_hit",
                        "other_fg_hit",
                        "background_hit",
                        "mixed_hit",
                        "degenerate_map",
                    }
                    else (pa.int32() if name.startswith("num_") else pa.float32()),
                    nullable=name
                    not in {
                        "has_target_region",
                        "target_hit",
                        "other_fg_hit",
                        "background_hit",
                        "mixed_hit",
                        "degenerate_map",
                        "num_target",
                        "num_other_fg",
                        "num_bg",
                        "num_mixed",
                        "num_void",
                        "num_valid",
                    },
                )
                for name in REGION_METRIC_FIELDS
            ],
        ]
    ),
    "positive_map_overlap": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_a", pa.int8()),
            _field("class_b", pa.int8()),
            _field("predicted_positive_a", pa.bool_()),
            _field("predicted_positive_b", pa.bool_()),
            _field("prediction_pair_status", pa.string()),
            _field("variant", pa.string()),
            _field("representation_timing", pa.string()),
            _field("topk_ratio", pa.float32()),
            _field("spearman", pa.float32(), nullable=True),
            _field("topk_jaccard", pa.float32()),
            _field("topk_overlap_coefficient", pa.float32()),
        ]
    ),
    "shared_ownership": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_a", pa.int8()),
            _field("class_b", pa.int8()),
            _field("variant", pa.string()),
            _field("representation_timing", pa.string()),
            _field("rho", pa.float32()),
            _field("topk_ratio", pa.float32()),
            _field("shared_set_size", pa.int16()),
            _field("topk_jaccard", pa.float32()),
            _field("topk_overlap_coefficient", pa.float32()),
            _field("shared_target_a_fraction", pa.float32(), nullable=True),
            _field("shared_target_b_fraction", pa.float32(), nullable=True),
            _field("shared_other_fg_fraction", pa.float32(), nullable=True),
            _field("shared_background_fraction", pa.float32(), nullable=True),
            _field("shared_mixed_void_fraction", pa.float32(), nullable=True),
            _field("shared_target_a_enrichment", pa.float32(), nullable=True),
            _field("shared_target_b_enrichment", pa.float32(), nullable=True),
            _field("shared_other_fg_enrichment", pa.float32(), nullable=True),
            _field("shared_background_enrichment", pa.float32(), nullable=True),
        ]
    ),
    "probe_linkage": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("class_id", pa.int8()),
            _field("predicted_positive", pa.bool_()),
            _field("link", pa.string()),
            _field("topk_ratio", pa.float32()),
            _field("spearman", pa.float32(), nullable=True),
            _field("topk_jaccard", pa.float32()),
            _field("topk_overlap_coefficient", pa.float32()),
        ]
    ),
    "all_class_control_map_strata": pa.schema(
        [
            *BASE_IMAGE_FIELDS,
            _field("layer", pa.int8()),
            _field("variant", pa.string()),
            _field("representation_timing", pa.string()),
            _field("presence_pair_status", pa.string()),
            _field("prediction_pair_status", pa.string()),
            _field("num_pairs", pa.int16()),
            _field("spearman_mean", pa.float32(), nullable=True),
            _field("top05_jaccard_mean", pa.float32()),
            _field("top10_jaccard_mean", pa.float32()),
            _field("top20_jaccard_mean", pa.float32()),
        ]
    ),
    "source_index": pa.schema(
        [
            _field("model", pa.string()),
            _field("image_id", pa.string()),
            _field("presence_artifact_path", pa.string()),
            _field("presence_artifact_sha256", pa.string()),
            _field("experiment2_artifact_path", pa.string()),
            _field("experiment2_artifact_sha256", pa.string()),
            _field("source_hash_link_verified", pa.bool_()),
            _field("presence_schema_verified", pa.bool_()),
        ]
    ),
}


@dataclass(frozen=True)
class PresenceManifestEntry:
    model: str
    image_id: str
    eval_fold: int
    positive_class_ids: tuple[int, ...]
    artifact_path: Path
    artifact_sha256: str


@dataclass(frozen=True)
class PresenceRun:
    model: str
    root: Path
    run_kind: str
    processed_images: int
    metadata: Mapping[str, object]
    completion: Mapping[str, object]
    entries: tuple[PresenceManifestEntry, ...]
    split_ids: tuple[str, ...]
    split_assignments: Mapping[str, int]
    directions: Mapping[str, np.ndarray]
    watched_hashes: Mapping[Path, str]


class AtomicParquetWriter:
    """Bounded, typed, atomic Parquet writer with close-time verification."""

    def __init__(self, directory: Path, name: str, schema: pa.Schema):
        self.name = name
        self.schema = schema.with_metadata(
            {
                b"experiment": b"Experiment 3 Validation A Presence Axis",
                b"schema_version": CANONICAL_SCHEMA_VERSION.encode(),
                b"table_name": name.encode(),
            }
        )
        self.final_path = directory / f"{name}.parquet"
        self.temporary_path = directory / f"{name}.parquet.tmp"
        if self.final_path.exists() or self.temporary_path.exists():
            raise FileExistsError(self.final_path)
        self.writer = pq.ParquetWriter(
            self.temporary_path,
            self.schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.rows = 0
        self.closed = False

    def append(self, rows: Sequence[Mapping[str, object]]) -> None:
        if self.closed:
            raise RuntimeError(f"writer {self.name} is closed")
        if not rows:
            return
        expected = set(self.schema.names)
        for index, row in enumerate(rows):
            if set(row) != expected:
                raise ValueError(
                    f"{self.name} row {index} schema mismatch: "
                    f"missing={sorted(expected - set(row))}, "
                    f"extra={sorted(set(row) - expected)}"
                )
        table = pa.Table.from_pylist(list(rows), schema=self.schema)
        self.writer.write_table(table, row_group_size=len(rows))
        self.rows += len(rows)

    def close(self) -> dict[str, object]:
        if self.closed:
            raise RuntimeError(f"writer {self.name} already closed")
        self.writer.close()
        parquet = pq.ParquetFile(self.temporary_path)
        if parquet.metadata.num_rows != self.rows:
            raise RuntimeError(f"{self.name} row-count round trip failed")
        if parquet.schema_arrow.remove_metadata() != self.schema.remove_metadata():
            raise RuntimeError(f"{self.name} schema round trip failed")
        self.temporary_path.replace(self.final_path)
        self.closed = True
        return {
            "path": str(self.final_path),
            "rows": self.rows,
            "columns": self.schema.names,
            "sha256": sha256_file(self.final_path),
            "compression": "zstd",
            "roundtrip_verified": True,
        }

    def abort(self) -> None:
        if not self.closed:
            try:
                self.writer.close()
            finally:
                if self.temporary_path.exists():
                    self.temporary_path.unlink()
                self.closed = True


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise TypeError(f"{name} must be a non-empty scalar string")
    return item


def _sha256_shape(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} is not hexadecimal") from error
    return value


def _label_stratum(count: int) -> str:
    if count == 1:
        return "single_label"
    if count == 2:
        return "2_label"
    if count >= 3:
        return "3plus_label"
    raise ValueError("every VOC validation image must have a positive class")


def _pair_status(left: bool, right: bool, *, positive_name: str) -> str:
    count = int(left) + int(right)
    if count == 2:
        return f"both_{positive_name}"
    if count == 1:
        return f"one_{positive_name}"
    return f"neither_{positive_name}"


def _float_or_none(value: object) -> Optional[float]:
    result = float(value)
    if math.isinf(result):
        raise ValueError("infinite metric is forbidden")
    return result if math.isfinite(result) else None


def _json_safe(value: object) -> object:
    """Convert pandas/NumPy scalars and missing values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL manifest: {path}")
    return rows


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes root {root}: {relative!r}") from error
    return path


def _validate_direction_registry(
    path: Path, split_counts: tuple[int, int]
) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != DIRECTION_KEYS:
            raise ValueError(
                f"direction schema mismatch: missing={sorted(DIRECTION_KEYS - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - DIRECTION_KEYS)}"
            )
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    shapes = {
        "fit_means": (2, 12, 20, 384),
        "class_deltas": (2, 12, 20, 384),
        "shared_directions": (2, 12, 384),
        "loo_shared_directions": (2, 12, 20, 384),
        "class_alignment": (2, 12, 20),
        "loo_class_alignment": (2, 12, 20),
        "total_counts": (2, 12, 20),
        "positive_counts": (2, 12, 20),
        "negative_counts": (2, 12, 20),
    }
    for name, shape in shapes.items():
        value = payload[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid direction field {name}: {value.shape}")
    for fold, count in enumerate(split_counts):
        if not np.all(payload["total_counts"][fold] == count):
            raise ValueError(f"direction total counts mismatch fold {fold}")
    if not np.array_equal(
        payload["positive_counts"] + payload["negative_counts"],
        payload["total_counts"],
    ):
        raise ValueError("direction positive/negative counts do not sum to totals")
    if np.any(payload["positive_counts"] <= 0) or np.any(
        payload["negative_counts"] <= 0
    ):
        raise ValueError("direction registry contains an empty presence group")
    shared_norms = np.linalg.norm(payload["shared_directions"], axis=-1)
    loo_norms = np.linalg.norm(payload["loo_shared_directions"], axis=-1)
    if not np.allclose(shared_norms, 1.0, rtol=0, atol=1e-10):
        raise ValueError("shared directions are not unit norm")
    if not np.allclose(loo_norms, 1.0, rtol=0, atol=1e-10):
        raise ValueError("LOO directions are not unit norm")
    return payload


def load_presence_run(
    model: str,
    root: Path,
    *,
    source_metadata_path: Path,
    source_metadata_sha256: str,
    expected_split_ids: Sequence[str],
    allow_smoke: bool,
) -> PresenceRun:
    """Fail-closed validation of one completed Presence-Axis run root."""

    if model not in MODEL_ORDER:
        raise ValueError(f"unsupported model {model!r}")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        name: root / filename
        for name, filename in {
            "metadata": "metadata.json",
            "completion": "completion.json",
            "manifest": "manifest.jsonl",
            "split": "split_manifest.csv",
            "directions": "shared_presence_directions.npz",
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = read_json(paths["metadata"])
    completion = read_json(paths["completion"])
    if metadata.get("status") != "complete" or completion.get("status") != "complete":
        raise RuntimeError(f"Presence-Axis run is incomplete: {root}")
    if metadata.get("analysis") != "experiment3_validation_a_presence_axis":
        raise RuntimeError(f"wrong analysis kind in {root}")
    if metadata.get("model") != model or completion.get("model") != model:
        raise RuntimeError(f"model linkage mismatch in {root}")
    run_kind = str(metadata.get("run_kind"))
    if run_kind not in {"full", "smoke"} or completion.get("run_kind") != run_kind:
        raise RuntimeError(f"invalid/mismatched run_kind in {root}")
    if run_kind == "smoke" and not allow_smoke:
        raise RuntimeError(
            "smoke outputs require --allow-smoke and cannot support final claims"
        )
    recorded_source = Path(str(metadata.get("source_metadata", ""))).resolve()
    if recorded_source != source_metadata_path:
        raise RuntimeError(f"source metadata path mismatch in {root}")
    if metadata.get("source_metadata_sha256") != source_metadata_sha256:
        raise RuntimeError(f"source metadata hash mismatch in {root}")

    processed = int(metadata.get("processed_images", -1))
    if int(completion.get("num_images", -1)) != processed or processed < 1:
        raise RuntimeError(f"processed-image metadata mismatch in {root}")
    if run_kind == "full" and processed != EXPECTED_IMAGES:
        raise RuntimeError(f"full run has {processed}, expected {EXPECTED_IMAGES}")
    if int(metadata.get("direction_fit_images", -1)) != EXPECTED_IMAGES:
        raise RuntimeError("directions were not fitted on the full prespecified split")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping) or int(execution.get("batch_size", -1)) != 8:
        raise RuntimeError("Presence-Axis run did not use matched batch size 8")
    guard = metadata.get("first_image_no_change_guard")
    if not isinstance(guard, Mapping) or guard.get("passed") is not True:
        raise RuntimeError("instrumentation no-change guard did not pass")
    maxima = metadata.get("numerical_max_abs_differences")
    required_maxima = {
        "feature_post_source",
        "feature_norm_source",
        "qk_source",
        "attention_source",
        "class_logits_source",
        "conditional_attention_source",
        "class_pair_cosine_source",
        "patch_norm_source",
        "raw_axis_vs_collector",
        "final_logit_identity",
        "decomposition_reconstruction",
        "residual_orthogonality",
        "residual_cosine_identity",
    }
    if not isinstance(maxima, Mapping) or not required_maxima.issubset(maxima):
        raise RuntimeError("Presence-Axis run lacks required numerical gates")
    for name in required_maxima:
        value = float(maxima[name])
        tolerance = (
            5e-6
            if name in {"residual_cosine_identity", "raw_axis_vs_collector"}
            else STRICT_TOLERANCE
        )
        failed = value > tolerance if tolerance == 5e-6 else value >= tolerance
        if not math.isfinite(value) or failed:
            raise RuntimeError(f"failed numerical gate {name}={value} in {root}")
    for validation_name in (
        "direction_artifact_validation",
        "derived_artifact_validation",
    ):
        validation = metadata.get(validation_name)
        if not isinstance(validation, Mapping) or validation.get("passed") is not True:
            raise RuntimeError(f"missing/failed {validation_name} in {root}")
    derived_validation = metadata["derived_artifact_validation"]
    if int(derived_validation.get("artifacts_reloaded", -1)) != processed:
        raise RuntimeError(
            "derived artifact reload count does not match processed images"
        )
    if set(derived_validation.get("schema", [])) != PRESENCE_REQUIRED_KEYS:
        raise RuntimeError(
            "runner-recorded derived schema does not match analyzer contract"
        )

    split = pd.read_csv(paths["split"], dtype={"image_id": str, "eval_fold": int})
    if list(split.columns) != ["image_id", "eval_fold"]:
        raise ValueError(f"split manifest schema mismatch in {root}")
    split_ids = tuple(split["image_id"].tolist())
    if split_ids != tuple(expected_split_ids) or len(set(split_ids)) != len(split_ids):
        raise RuntimeError(f"split membership/order mismatch in {root}")
    assignments = dict(zip(split_ids, split["eval_fold"].astype(int)))
    if any(
        fold not in (0, 1) or fold != sha256_two_fold(image_id)
        for image_id, fold in assignments.items()
    ):
        raise RuntimeError(f"split assignment mismatch in {root}")
    crossfit = metadata.get("cross_fit")
    if not isinstance(crossfit, Mapping):
        raise RuntimeError("missing cross-fit metadata")
    if sha256_file(paths["split"]) != str(crossfit.get("split_manifest_sha256")):
        raise RuntimeError("split manifest hash mismatch")
    counts = (
        sum(value == 0 for value in assignments.values()),
        sum(value == 1 for value in assignments.values()),
    )
    if list(counts) != list(crossfit.get("fold_counts", [])):
        raise RuntimeError("cross-fit fold counts mismatch")
    direction_hash = sha256_file(paths["directions"])
    if direction_hash != str(crossfit.get("direction_artifact_sha256")):
        raise RuntimeError("direction artifact hash mismatch")
    directions = _validate_direction_registry(paths["directions"], counts)

    rows = _read_jsonl(paths["manifest"])
    if len(rows) != processed:
        raise RuntimeError(f"manifest rows {len(rows)} != processed {processed}")
    entries: list[PresenceManifestEntry] = []
    seen: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in seen:
            raise ValueError(f"empty/duplicate manifest image ID in {root}")
        seen.add(image_id)
        fold = int(row.get("eval_fold", -1))
        if assignments.get(image_id) != fold:
            raise RuntimeError(f"manifest fold mismatch for {image_id}")
        positives = tuple(int(value) for value in row.get("positive_class_ids", []))
        if not positives or tuple(sorted(set(positives))) != positives:
            raise ValueError(f"invalid positive classes for {image_id}")
        relative = str(row.get("signal_path", ""))
        artifact_path = _inside(root, relative)
        try:
            artifact_path.relative_to(root / "signals")
        except ValueError as error:
            raise ValueError(
                f"artifact outside signals directory: {artifact_path}"
            ) from error
        expected_hash = _sha256_shape(
            str(row.get("artifact_sha256", "")), "artifact hash"
        )
        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"artifact hash mismatch for {image_id}")
        entries.append(
            PresenceManifestEntry(
                model=model,
                image_id=image_id,
                eval_fold=fold,
                positive_class_ids=positives,
                artifact_path=artifact_path,
                artifact_sha256=actual_hash,
            )
        )
    expected_entries = tuple(expected_split_ids[:processed])
    if tuple(entry.image_id for entry in entries) != expected_entries:
        raise RuntimeError(
            f"Presence manifest is not the deterministic VOC prefix in {root}"
        )
    actual_artifacts = {path.resolve() for path in (root / "signals").glob("*.npz")}
    recorded_artifacts = {entry.artifact_path for entry in entries}
    if actual_artifacts != recorded_artifacts:
        raise RuntimeError(f"manifest/files membership mismatch in {root}")
    watched = {path: sha256_file(path) for path in paths.values()}
    watched.update({entry.artifact_path: entry.artifact_sha256 for entry in entries})
    return PresenceRun(
        model=model,
        root=root,
        run_kind=run_kind,
        processed_images=processed,
        metadata=metadata,
        completion=completion,
        entries=tuple(entries),
        split_ids=split_ids,
        split_assignments=assignments,
        directions=directions,
        watched_hashes=watched,
    )


def load_and_validate_presence_artifact(
    entry: PresenceManifestEntry,
    *,
    experiment2_entry: Experiment2ManifestEntry,
    experiment2_artifact: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    """Validate one derived NPZ and its exact Experiment 2 linkage."""

    if sha256_file(entry.artifact_path) != entry.artifact_sha256:
        raise RuntimeError(
            f"presence artifact changed before load: {entry.artifact_path}"
        )
    with np.load(entry.artifact_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = PRESENCE_REQUIRED_KEYS - keys
        extra = keys - PRESENCE_REQUIRED_KEYS - PRESENCE_OPTIONAL_KEYS
        if missing or extra:
            raise ValueError(
                f"Presence artifact schema mismatch for {entry.image_id}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    if _scalar_string(payload["image_id"], "image_id") != entry.image_id:
        raise RuntimeError("presence artifact image identity mismatch")
    positive = np.asarray(payload["positive_class_ids"])
    if (
        positive.dtype != np.int64
        or tuple(positive.tolist()) != entry.positive_class_ids
    ):
        raise TypeError("positive_class_ids must exactly match manifest int64 IDs")
    if (
        experiment2_entry.image_id != entry.image_id
        or experiment2_entry.positive_class_ids != entry.positive_class_ids
    ):
        raise RuntimeError("Presence/Experiment 2 identity or positives mismatch")
    eval_fold = int(np.asarray(payload["eval_fold"]).item())
    fit_fold = int(np.asarray(payload["fit_fold"]).item())
    if eval_fold != entry.eval_fold or fit_fold != 1 - eval_fold:
        raise RuntimeError("artifact cross-fit fold mismatch")
    source_hash = _sha256_shape(
        _scalar_string(payload["source_signal_sha256"], "source_signal_sha256"),
        "source_signal_sha256",
    )
    actual_source_hash = sha256_file(experiment2_entry.artifact_path)
    if (
        source_hash != experiment2_entry.artifact_sha256
        or source_hash != actual_source_hash
    ):
        raise RuntimeError("source Experiment 2 SHA linkage mismatch")

    classes = len(positive)
    shapes = {
        "class_removed_scores": (12, classes, 784),
        "patch_removed_scores": (12, classes, 784),
        "both_removed_scores": (12, classes, 784),
        "shared_both_removed_scores": (12, classes, 784),
        "class_coefficients": (12, 20),
        "class_axis_energy": (12, 20),
        "class_norms": (12, 20),
        "class_residual_norms": (12, 20),
        "patch_coefficient_mean": (12,),
        "patch_coefficient_std": (12,),
        "patch_axis_energy_mean": (12,),
        "patch_axis_energy_std": (12,),
        "raw_pair_cosine_all": (12, 20, 20),
        "residual_pair_cosine_all": (12, 20, 20),
        "pair_axis_dot_all": (12, 20, 20),
        "pair_residual_dot_all": (12, 20, 20),
        "heldout_projection_all": (12, 20),
        "shared_axis_energy_all": (12, 20),
        "class_logits_all": (20,),
        "control_layer_ids": (len(CONTROL_LAYERS),),
        "raw_control_all": (len(CONTROL_LAYERS), 20, 784),
        "class_removed_control_all": (len(CONTROL_LAYERS), 20, 784),
        "patch_removed_control_all": (len(CONTROL_LAYERS), 20, 784),
        "both_removed_control_all": (len(CONTROL_LAYERS), 20, 784),
        "shared_both_removed_control_all": (len(CONTROL_LAYERS), 20, 784),
        "feature_norm_control_all": (len(CONTROL_LAYERS), 20, 784),
        "qk_control_all": (len(CONTROL_LAYERS), 20, 784),
        "attention_conditional_control_all": (
            len(CONTROL_LAYERS),
            20,
            784,
        ),
    }
    integer_names = {"control_layer_ids"}
    for name, shape in shapes.items():
        value = np.asarray(payload[name])
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if name in integer_names:
            if not np.issubdtype(value.dtype, np.integer):
                raise TypeError(f"{name} must be integer")
        else:
            if value.dtype != np.float32:
                raise TypeError(f"{name} must be float32, got {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
    if tuple(int(value) for value in payload["control_layer_ids"]) != CONTROL_LAYERS:
        raise ValueError("control_layer_ids must be one-based [4,5,9,10,11,12]")
    if np.any(payload["class_norms"] <= 0) or np.any(
        payload["class_residual_norms"] <= 0
    ):
        raise ValueError("class token norms must be strictly positive")
    for name in (
        "class_axis_energy",
        "shared_axis_energy_all",
        "patch_axis_energy_mean",
    ):
        if np.any(payload[name] < -1e-6) or np.any(payload[name] > 1.0 + 1e-5):
            raise ValueError(f"{name} escaped [0,1]")
    for name in (
        "class_removed_scores",
        "patch_removed_scores",
        "both_removed_scores",
        "shared_both_removed_scores",
        "raw_control_all",
        "class_removed_control_all",
        "patch_removed_control_all",
        "both_removed_control_all",
        "shared_both_removed_control_all",
        "feature_norm_control_all",
        "raw_pair_cosine_all",
        "residual_pair_cosine_all",
    ):
        if np.max(np.abs(payload[name])) > 1.0 + 1e-4:
            raise ValueError(f"{name} escaped cosine range")

    conditional_control = payload["attention_conditional_control_all"]
    if np.any(conditional_control < 0.0) or np.any(conditional_control > 1.0):
        raise ValueError("attention_conditional_control_all escaped [0,1]")
    if not np.allclose(conditional_control.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(
            "attention_conditional_control_all does not sum to one over patches"
        )

    raw_pair = payload["raw_pair_cosine_all"]
    residual_pair = payload["residual_pair_cosine_all"]
    if not np.allclose(raw_pair, raw_pair.transpose(0, 2, 1), rtol=0, atol=1e-6):
        raise ValueError("raw class-pair cosine is not symmetric")
    if not np.allclose(
        residual_pair, residual_pair.transpose(0, 2, 1), rtol=0, atol=1e-6
    ):
        raise ValueError("residual class-pair cosine is not symmetric")
    if not np.allclose(np.diagonal(raw_pair, axis1=1, axis2=2), 1.0, rtol=0, atol=1e-5):
        raise ValueError("raw class-pair cosine diagonal is not one")
    reconstructed_dot = payload["pair_axis_dot_all"] + payload["pair_residual_dot_all"]
    expected_dot = raw_pair * (
        payload["class_norms"][:, :, None] * payload["class_norms"][:, None, :]
    )
    # All independently persisted operands are float32.  A conventional
    # relative-to-result tolerance fails under legitimate cancellation between
    # O(1e4) axis/residual terms.  Bound reload error by their operation scale;
    # the runner already enforces the underlying float64 identity at <1e-6.
    norm_products = (
        payload["class_norms"][:, :, None] * payload["class_norms"][:, None, :]
    )
    dot_roundoff_bound = (
        64.0
        * np.finfo(np.float32).eps
        * (
            np.abs(payload["pair_axis_dot_all"])
            + np.abs(payload["pair_residual_dot_all"])
            + np.abs(expected_dot)
            + np.abs(norm_products)
        )
        + 1e-6
    )
    if np.any(np.abs(reconstructed_dot - expected_dot) > dot_roundoff_bound):
        raise ValueError("axis/residual pair-dot reconstruction failed")
    # The runner verifies the identity in float64 at <1e-6.  These two saved
    # operands are float32, so their independent reload check allows three
    # float32 ulps at the observed logit scale.
    if (
        float(
            np.max(
                np.abs(
                    payload["class_coefficients"][11] / math.sqrt(384.0)
                    - payload["class_logits_all"]
                )
            )
        )
        > 3e-6
    ):
        raise ValueError("saved L12 coefficient does not reproduce class logits")

    source_positive = np.asarray(
        experiment2_artifact["positive_class_ids"], dtype=np.int64
    )
    if not np.array_equal(source_positive, positive):
        raise RuntimeError("Experiment 2 positive IDs changed")
    if (
        float(
            np.max(
                np.abs(
                    payload["class_logits_all"]
                    - experiment2_artifact["class_logits_all"]
                )
            )
        )
        >= STRICT_TOLERANCE
    ):
        raise RuntimeError("saved class logits do not reproduce Experiment 2")
    control_zero = np.asarray(CONTROL_LAYERS, dtype=np.int64) - 1
    positive_index = positive.astype(np.int64)
    equivalences = {
        "raw_control_all": experiment2_artifact["feature_post_scores"][control_zero],
        "class_removed_control_all": payload["class_removed_scores"][control_zero],
        "patch_removed_control_all": payload["patch_removed_scores"][control_zero],
        "both_removed_control_all": payload["both_removed_scores"][control_zero],
        "shared_both_removed_control_all": payload["shared_both_removed_scores"][
            control_zero
        ],
        "feature_norm_control_all": experiment2_artifact["feature_norm_scores"][
            control_zero
        ],
        "qk_control_all": experiment2_artifact["qk_mean_scores"][control_zero],
        "attention_conditional_control_all": experiment2_artifact[
            "attn_c2p_conditional"
        ][control_zero],
    }
    for name, reference in equivalences.items():
        observed = payload[name][:, positive_index]
        difference = float(np.max(np.abs(observed - reference)))
        tolerance = 5e-6 if name == "raw_control_all" else STRICT_TOLERANCE
        failed = (
            difference > tolerance
            if name == "raw_control_all"
            else difference >= tolerance
        )
        if failed:
            raise RuntimeError(f"positive slice equivalence failed for {name}")
    source_pair = experiment2_artifact["class_token_pairwise_cosine"]
    observed_pair = raw_pair[:, positive_index][:, :, positive_index]
    # The runner separately proves collector pair-cosine/source equivalence at
    # STRICT_TOLERANCE.  This all-class decomposition recomputes the same raw
    # cosine with a different reduction kernel and is covered by the runner's
    # independent raw-axis/collector gate of 5e-6.
    if float(np.max(np.abs(observed_pair - source_pair))) > 5e-6:
        raise RuntimeError("class-token pair cosine does not reproduce Experiment 2")
    return payload


def aligned_normalized_maps(
    source: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Align post-block Lk with norm1 L(k+1), using final LN only at L12."""

    norm = np.asarray(source["feature_norm_scores"])
    final_norm = np.asarray(source["feature_final_norm_scores"])
    if norm.ndim != 3 or norm.shape[0] != EXPECTED_LAYERS:
        raise ValueError("feature_norm_scores must have shape [12,K,P]")
    if final_norm.shape != norm.shape[1:]:
        raise ValueError("feature_final_norm_scores must have shape [K,P]")
    aligned = np.concatenate((norm[1:], final_norm[None]), axis=0)
    timings = tuple(
        [f"post_L{layer}_vs_norm1_L{layer + 1}" for layer in range(1, 12)]
        + ["post_L12_vs_final_norm_analysis_only"]
    )
    return aligned, timings


def pairwise_map_geometry(
    maps: np.ndarray,
    *,
    eligible: Optional[np.ndarray] = None,
    ratios: Sequence[float] = TOPK_RATIO_VALUES,
) -> Mapping[str, np.ndarray]:
    """Vectorized all-class Spearman and exact stable top-k Jaccard matrices."""

    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("maps must be finite [C,P] with C>=2")
    if eligible is None:
        valid = np.ones(values.shape[1], dtype=bool)
    else:
        valid = np.asarray(eligible, dtype=bool).reshape(-1)
        if valid.shape != (values.shape[1],) or not valid.any():
            raise ValueError("eligible must be a non-empty patch mask")
    ranks = rankdata(values[:, valid], axis=1, method="average")
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = norms[:, None] * norms[None, :]
    spearman = np.divide(
        centered @ centered.T,
        denominator,
        out=np.full((values.shape[0], values.shape[0]), np.nan),
        where=denominator > 1e-12,
    )
    output: dict[str, np.ndarray] = {"spearman": spearman}
    for ratio in ratios:
        masks = np.stack(
            [stable_topk_mask(row, ratio, valid).reshape(-1) for row in values]
        )
        intersections = masks.astype(np.int16) @ masks.astype(np.int16).T
        sizes = masks.sum(axis=1)
        unions = sizes[:, None] + sizes[None, :] - intersections
        jaccard = np.divide(
            intersections,
            unions,
            out=np.ones_like(intersections, dtype=np.float64),
            where=unions > 0,
        )
        output[f"top{int(round(100 * ratio)):02d}_jaccard"] = jaccard
    return output


def eligible_map_overlap_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    ratio: float,
    eligible: np.ndarray,
) -> Mapping[str, float]:
    """Compute Spearman and top-k geometry on exactly the eligible patches."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.asarray(eligible, dtype=bool).reshape(-1)
    if left_values.shape != right_values.shape or valid.shape != left_values.shape:
        raise ValueError("maps and eligible mask must have equal flattened shape")
    if not valid.any():
        raise ValueError("eligible mask contains no non-void patches")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("maps contain NaN/Inf")
    # Slicing, rather than merely forwarding ``eligible``, is necessary because
    # Experiment 2's shared helper limits top-k but computes Spearman globally.
    return map_overlap_metrics(left_values[valid], right_values[valid], ratio=ratio)


def _variant_maps(
    presence: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray]
) -> tuple[Mapping[str, np.ndarray], Mapping[str, str]]:
    aligned, aligned_timing = aligned_normalized_maps(source)
    maps = {
        "raw": np.asarray(source["feature_post_scores"]),
        "class_removed": np.asarray(presence["class_removed_scores"]),
        "patch_removed": np.asarray(presence["patch_removed_scores"]),
        "both_removed": np.asarray(presence["both_removed_scores"]),
        "shared_both_removed": np.asarray(presence["shared_both_removed_scores"]),
        "norm1_pre_same_index": np.asarray(source["feature_norm_scores"]),
        "norm_timing_aligned": aligned,
        "qk_mean": np.asarray(source["qk_mean_scores"]),
        "attn_c2p_conditional": np.asarray(source["attn_c2p_conditional"]),
    }
    timing = {
        "raw": "post_block_Lk",
        "class_removed": "post_block_Lk_fixed_axis_removed",
        "patch_removed": "post_block_Lk_fixed_axis_removed",
        "both_removed": "post_block_Lk_fixed_axis_removed",
        "shared_both_removed": "post_block_Lk_crossfit_axis_removed",
        "norm1_pre_same_index": "norm1_input_to_block_Lk_not_post_aligned",
        "qk_mean": "pre_attention_QK_Lk",
        "attn_c2p_conditional": "pre_attention_softmax_Lk",
    }
    # Timing varies by layer for the aligned variant and is filled per row.
    timing["norm_timing_aligned"] = "|".join(aligned_timing)
    return maps, timing


def _control_maps(presence: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return {
        "raw": np.asarray(presence["raw_control_all"]),
        "class_removed": np.asarray(presence["class_removed_control_all"]),
        "patch_removed": np.asarray(presence["patch_removed_control_all"]),
        "both_removed": np.asarray(presence["both_removed_control_all"]),
        "shared_both_removed": np.asarray(presence["shared_both_removed_control_all"]),
        "norm1_pre_same_index": np.asarray(presence["feature_norm_control_all"]),
        "qk_mean": np.asarray(presence["qk_control_all"]),
        "attn_c2p_conditional": np.asarray(
            presence["attention_conditional_control_all"]
        ),
    }


CONTROL_TIMINGS = {
    "raw": "post_block_Lk",
    "class_removed": "post_block_Lk_fixed_axis_removed",
    "patch_removed": "post_block_Lk_fixed_axis_removed",
    "both_removed": "post_block_Lk_fixed_axis_removed",
    "shared_both_removed": "post_block_Lk_crossfit_axis_removed",
    "norm1_pre_same_index": "norm1_input_to_block_Lk_not_post_aligned",
    "qk_mean": "pre_attention_QK_Lk",
    "attn_c2p_conditional": "pre_attention_softmax_Lk",
}


def _region_values(
    values: np.ndarray, regions: np.ndarray, *, nonnegative_mass: bool
) -> Mapping[str, object]:
    metrics = region_map_metrics(
        values,
        regions,
        grid_h=28,
        grid_w=28,
        nonnegative_mass=nonnegative_mass,
    )
    # Experiment 2 defines conditional attention mass over the complete spatial
    # map, not a void-dropped renormalization.
    if nonnegative_mass:
        total = float(np.asarray(values, dtype=np.float64).sum())
        metrics["conditional_bg_mass"] = (
            float(
                np.asarray(values)[np.asarray(regions) == REGION_BACKGROUND].sum()
                / total
            )
            if total > 1e-12
            else float("nan")
        )
    output: dict[str, object] = {}
    for name in REGION_METRIC_FIELDS:
        value = metrics[name]
        output[name] = (
            bool(value)
            if name
            in {
                "has_target_region",
                "target_hit",
                "other_fg_hit",
                "background_hit",
                "mixed_hit",
                "degenerate_map",
            }
            else (int(value) if name.startswith("num_") else _float_or_none(value))
        )
    return output


def _base(
    model: str,
    image_id: str,
    eval_fold: int,
    positive_count: int,
) -> dict[str, object]:
    return {
        "model": model,
        "image_id": image_id,
        "eval_fold": eval_fold,
        "fit_fold": 1 - eval_fold,
        "num_positive_classes": positive_count,
        "label_stratum": _label_stratum(positive_count),
    }


def _positive_pair_indices(positive: np.ndarray) -> list[tuple[int, int, int, int]]:
    return [
        (offset_a, offset_b, int(positive[offset_a]), int(positive[offset_b]))
        for offset_a in range(len(positive))
        for offset_b in range(offset_a + 1, len(positive))
    ]


def _canonical_rows_for_image(
    model: str,
    entry: PresenceManifestEntry,
    presence: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
) -> Mapping[str, list[dict[str, object]]]:
    positive = np.asarray(entry.positive_class_ids, dtype=np.int64)
    positive_set = set(int(value) for value in positive)
    logits = np.asarray(presence["class_logits_all"])
    predicted = logits > CLASS_LOGIT_THRESHOLD
    base = _base(model, entry.image_id, entry.eval_fold, len(positive))
    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in CANONICAL_SCHEMAS if name != "source_index"
    }
    for layer in range(EXPECTED_LAYERS):
        for class_id in range(EXPECTED_CLASSES):
            target = class_id in positive_set
            common = {
                **base,
                "layer": layer + 1,
                "class_id": class_id,
                "class_name": VOC_CLASS_NAMES[class_id],
                "target": target,
                "predicted_positive": bool(predicted[class_id]),
            }
            rows["token_axis"].append(
                {
                    **common,
                    "class_logit": float(logits[class_id]),
                    "presence_status": "GT_positive" if target else "GT_negative",
                    "prediction_status": (
                        "predicted_positive"
                        if predicted[class_id]
                        else "predicted_negative"
                    ),
                    "class_coefficient": float(
                        presence["class_coefficients"][layer, class_id]
                    ),
                    "class_axis_energy": float(
                        presence["class_axis_energy"][layer, class_id]
                    ),
                    "class_norm": float(presence["class_norms"][layer, class_id]),
                    "class_residual_norm": float(
                        presence["class_residual_norms"][layer, class_id]
                    ),
                    "shared_axis_energy": float(
                        presence["shared_axis_energy_all"][layer, class_id]
                    ),
                }
            )
            rows["oof_projection"].append(
                {
                    **base,
                    "layer": layer + 1,
                    "class_id": class_id,
                    "class_name": VOC_CLASS_NAMES[class_id],
                    "target": target,
                    "predicted_positive": bool(predicted[class_id]),
                    "heldout_projection": float(
                        presence["heldout_projection_all"][layer, class_id]
                    ),
                    "shared_axis_energy": float(
                        presence["shared_axis_energy_all"][layer, class_id]
                    ),
                }
            )
        for class_a in range(EXPECTED_CLASSES):
            for class_b in range(class_a + 1, EXPECTED_CLASSES):
                target_a, target_b = class_a in positive_set, class_b in positive_set
                raw = float(presence["raw_pair_cosine_all"][layer, class_a, class_b])
                residual = float(
                    presence["residual_pair_cosine_all"][layer, class_a, class_b]
                )
                axis_dot = float(presence["pair_axis_dot_all"][layer, class_a, class_b])
                residual_dot = float(
                    presence["pair_residual_dot_all"][layer, class_a, class_b]
                )
                rows["token_pairs"].append(
                    {
                        **base,
                        "layer": layer + 1,
                        "class_a": class_a,
                        "class_b": class_b,
                        "class_a_name": VOC_CLASS_NAMES[class_a],
                        "class_b_name": VOC_CLASS_NAMES[class_b],
                        "target_a": target_a,
                        "target_b": target_b,
                        "predicted_positive_a": bool(predicted[class_a]),
                        "predicted_positive_b": bool(predicted[class_b]),
                        "presence_pair_status": _pair_status(
                            target_a, target_b, positive_name="GT_positive"
                        ),
                        "prediction_pair_status": _pair_status(
                            bool(predicted[class_a]),
                            bool(predicted[class_b]),
                            positive_name="predicted_positive",
                        ),
                        "raw_pair_cosine": raw,
                        "residual_pair_cosine": residual,
                        "cosine_delta": residual - raw,
                        "axis_dot": axis_dot,
                        "residual_dot": residual_dot,
                        "reconstructed_dot": axis_dot + residual_dot,
                    }
                )
        rows["patch_axis"].append(
            {
                **base,
                "layer": layer + 1,
                "patch_coefficient_mean": float(
                    presence["patch_coefficient_mean"][layer]
                ),
                "patch_coefficient_std": float(
                    presence["patch_coefficient_std"][layer]
                ),
                "patch_axis_energy_mean": float(
                    presence["patch_axis_energy_mean"][layer]
                ),
                "patch_axis_energy_std": float(
                    presence["patch_axis_energy_std"][layer]
                ),
            }
        )

    variant_maps, timings = _variant_maps(presence, source)
    aligned_timing = aligned_normalized_maps(source)[1]
    for rho, region_key in ((0.5, "region_masks_rho05"), (0.7, "region_masks_rho07")):
        regions_all = np.asarray(source[region_key])
        for offset, class_id in enumerate(positive):
            for layer in range(EXPECTED_LAYERS):
                for variant in POSITIVE_PROBE_VARIANTS:
                    timing = (
                        aligned_timing[layer]
                        if variant == "norm_timing_aligned"
                        else timings[variant]
                    )
                    values = variant_maps[variant][layer, offset]
                    rows["probe_region"].append(
                        {
                            **base,
                            "layer": layer + 1,
                            "class_id": int(class_id),
                            "class_name": VOC_CLASS_NAMES[int(class_id)],
                            "class_logit": float(logits[class_id]),
                            "predicted_positive": bool(predicted[class_id]),
                            "variant": variant,
                            "representation_timing": timing,
                            "rho": rho,
                            **_region_values(
                                values,
                                regions_all[offset],
                                nonnegative_mass=variant == "attn_c2p_conditional",
                            ),
                        }
                    )
            # Explicit final-LayerNorm join.  It is an analysis-only control,
            # never called an additional native model stage.
            rows["probe_region"].append(
                {
                    **base,
                    "layer": 12,
                    "class_id": int(class_id),
                    "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "class_logit": float(logits[class_id]),
                    "predicted_positive": bool(predicted[class_id]),
                    "variant": "final_norm_analysis_only",
                    "representation_timing": "post_L12_final_LayerNorm_analysis_only",
                    "rho": rho,
                    **_region_values(
                        source["feature_final_norm_scores"][offset],
                        regions_all[offset],
                        nonnegative_mass=False,
                    ),
                }
            )

    positive_pairs = _positive_pair_indices(positive)
    counts = np.asarray(source["patch_label_counts"])
    target_valid_rho05 = {
        int(class_id): np.asarray(source["region_masks_rho05"])[offset] != REGION_VOID
        for offset, class_id in enumerate(positive)
    }
    pair_valid_rho05 = {
        (class_a, class_b): assign_pair_patch_regions_from_counts(
            counts,
            class_a,
            class_b,
            rho=0.5,
            grid_size=(28, 28),
        )["region_codes"].reshape(-1)
        != PAIR_REGION_VOID
        for _, _, class_a, class_b in positive_pairs
    }
    for layer in range(EXPECTED_LAYERS):
        for variant in POSITIVE_PROBE_VARIANTS:
            maps = variant_maps[variant][layer]
            timing = (
                aligned_timing[layer]
                if variant == "norm_timing_aligned"
                else timings[variant]
            )
            for offset_a, offset_b, class_a, class_b in positive_pairs:
                for ratio in TOPK_RATIO_VALUES:
                    overlap = eligible_map_overlap_metrics(
                        maps[offset_a],
                        maps[offset_b],
                        ratio=ratio,
                        eligible=pair_valid_rho05[(class_a, class_b)],
                    )
                    rows["positive_map_overlap"].append(
                        {
                            **base,
                            "layer": layer + 1,
                            "class_a": class_a,
                            "class_b": class_b,
                            "predicted_positive_a": bool(predicted[class_a]),
                            "predicted_positive_b": bool(predicted[class_b]),
                            "prediction_pair_status": _pair_status(
                                bool(predicted[class_a]),
                                bool(predicted[class_b]),
                                positive_name="predicted_positive",
                            ),
                            "variant": variant,
                            "representation_timing": timing,
                            "topk_ratio": ratio,
                            "spearman": _float_or_none(overlap["spearman"]),
                            "topk_jaccard": float(overlap["topk_jaccard"]),
                            "topk_overlap_coefficient": float(
                                overlap["topk_overlap_coefficient"]
                            ),
                        }
                    )
        for offset, class_id in enumerate(positive):
            for link, left_name, right_name in PROBE_LINKS:
                for ratio in TOPK_RATIO_VALUES:
                    overlap = eligible_map_overlap_metrics(
                        variant_maps[left_name][layer, offset],
                        variant_maps[right_name][layer, offset],
                        ratio=ratio,
                        eligible=target_valid_rho05[int(class_id)],
                    )
                    rows["probe_linkage"].append(
                        {
                            **base,
                            "layer": layer + 1,
                            "class_id": int(class_id),
                            "predicted_positive": bool(predicted[class_id]),
                            "link": link,
                            "topk_ratio": ratio,
                            "spearman": _float_or_none(overlap["spearman"]),
                            "topk_jaccard": float(overlap["topk_jaccard"]),
                            "topk_overlap_coefficient": float(
                                overlap["topk_overlap_coefficient"]
                            ),
                        }
                    )

    for rho in REGION_THRESHOLDS:
        pair_regions = {
            (class_a, class_b): assign_pair_patch_regions_from_counts(
                counts,
                class_a,
                class_b,
                rho=rho,
                grid_size=(28, 28),
            )["region_codes"].reshape(-1)
            for _, _, class_a, class_b in positive_pairs
        }
        for layer in range(EXPECTED_LAYERS):
            for variant in SHARED_OWNERSHIP_VARIANTS:
                maps = variant_maps[variant][layer]
                for offset_a, offset_b, class_a, class_b in positive_pairs:
                    regions = pair_regions[(class_a, class_b)]
                    for ratio in TOPK_RATIO_VALUES:
                        metrics = shared_support_metrics(
                            maps[offset_a], maps[offset_b], regions, ratio=ratio
                        )
                        rows["shared_ownership"].append(
                            {
                                **base,
                                "layer": layer + 1,
                                "class_a": class_a,
                                "class_b": class_b,
                                "variant": variant,
                                "representation_timing": (
                                    aligned_timing[layer]
                                    if variant == "norm_timing_aligned"
                                    else timings[variant]
                                ),
                                "rho": rho,
                                "topk_ratio": ratio,
                                "shared_set_size": int(metrics["shared_set_size"]),
                                "topk_jaccard": float(metrics["topk_jaccard"]),
                                "topk_overlap_coefficient": float(
                                    metrics["topk_overlap_coefficient"]
                                ),
                                "shared_target_a_fraction": _float_or_none(
                                    metrics["shared_target_a_fraction"]
                                ),
                                "shared_target_b_fraction": _float_or_none(
                                    metrics["shared_target_b_fraction"]
                                ),
                                "shared_other_fg_fraction": _float_or_none(
                                    metrics["shared_other_fg_fraction"]
                                ),
                                "shared_background_fraction": _float_or_none(
                                    metrics["shared_background_fraction"]
                                ),
                                "shared_mixed_void_fraction": _float_or_none(
                                    metrics["shared_mixed_void_fraction"]
                                ),
                                "shared_target_a_enrichment": _float_or_none(
                                    metrics["shared_target_a_enrichment"]
                                ),
                                "shared_target_b_enrichment": _float_or_none(
                                    metrics["shared_target_b_enrichment"]
                                ),
                                "shared_other_fg_enrichment": _float_or_none(
                                    metrics["shared_other_fg_enrichment"]
                                ),
                                "shared_background_enrichment": _float_or_none(
                                    metrics["shared_background_enrichment"]
                                ),
                            }
                        )

    # Reduced all-class control rows retain negative/prediction strata without
    # writing 16.5 million redundant per-pair map rows.
    valid_patch = counts[:, 21].astype(np.float64) / counts.sum(axis=1) <= 0.5
    control = _control_maps(presence)
    for control_offset, layer in enumerate(CONTROL_LAYERS):
        for variant, all_maps in control.items():
            geometry = pairwise_map_geometry(
                all_maps[control_offset], eligible=valid_patch
            )
            grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
            for class_a in range(20):
                for class_b in range(class_a + 1, 20):
                    key = (
                        _pair_status(
                            class_a in positive_set,
                            class_b in positive_set,
                            positive_name="GT_positive",
                        ),
                        _pair_status(
                            bool(predicted[class_a]),
                            bool(predicted[class_b]),
                            positive_name="predicted_positive",
                        ),
                    )
                    grouped.setdefault(key, []).append((class_a, class_b))
            for (gt_status, prediction_status), pairs in grouped.items():
                indices_a = np.asarray([a for a, _ in pairs], dtype=np.int64)
                indices_b = np.asarray([b for _, b in pairs], dtype=np.int64)
                spearman_values = geometry["spearman"][indices_a, indices_b]
                rows["all_class_control_map_strata"].append(
                    {
                        **base,
                        "layer": layer,
                        "variant": variant,
                        "representation_timing": CONTROL_TIMINGS[variant],
                        "presence_pair_status": gt_status,
                        "prediction_pair_status": prediction_status,
                        "num_pairs": len(pairs),
                        "spearman_mean": _float_or_none(np.nanmean(spearman_values))
                        if np.isfinite(spearman_values).any()
                        else None,
                        "top05_jaccard_mean": float(
                            geometry["top05_jaccard"][indices_a, indices_b].mean()
                        ),
                        "top10_jaccard_mean": float(
                            geometry["top10_jaccard"][indices_a, indices_b].mean()
                        ),
                        "top20_jaccard_mean": float(
                            geometry["top20_jaccard"][indices_a, indices_b].mean()
                        ),
                    }
                )
    return rows


def _verify_sources_unchanged(
    runs: Sequence[PresenceRun], source_metadata: Path, source_hash: str
) -> None:
    if sha256_file(source_metadata) != source_hash:
        raise RuntimeError("audit source_metadata changed during analysis")
    for run in runs:
        actual = {path.resolve() for path in (run.root / "signals").glob("*.npz")}
        expected = {entry.artifact_path for entry in run.entries}
        if actual != expected:
            raise RuntimeError(f"Presence source membership changed: {run.root}")
        for path, digest in run.watched_hashes.items():
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"Presence source changed during analysis: {path}")


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    restored = pd.read_csv(path, low_memory=False)
    if len(restored) != len(frame) or list(restored.columns) != list(frame.columns):
        raise RuntimeError(f"CSV round-trip validation failed: {path}")
    return {"path": str(path), "rows": len(frame), "sha256": sha256_file(path)}


def _identity_groups(frame: pd.DataFrame, columns: Sequence[str]):
    if not columns:
        yield (), frame
        return
    grouper: object = columns[0] if len(columns) == 1 else list(columns)
    yield from frame.groupby(grouper, sort=True, dropna=False)


def _strata(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    output = [("all", frame)]
    if "label_stratum" in frame:
        output.extend(
            (str(value), group)
            for value, group in frame.groupby("label_stratum", sort=True)
        )
    for column in (
        "presence_status",
        "prediction_status",
        "presence_pair_status",
        "prediction_pair_status",
    ):
        if column in frame:
            output.extend(
                (f"{column}:{value}", group)
                for value, group in frame.groupby(column, sort=True)
            )
    if "predicted_positive" in frame:
        output.extend(
            (
                f"prediction_status:{'predicted_positive' if bool(value) else 'predicted_negative'}",
                group,
            )
            for value, group in frame.groupby("predicted_positive", sort=True)
        )
    return output


def _clustered_summary(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws,
    include_strata: bool = True,
    class_col: str = "class_id",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for keys, group in _identity_groups(frame, group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_cols, keys))
        subsets = _strata(group) if include_strata else [("all", group)]
        for stratum, subset in subsets:
            if subset.empty:
                continue
            macro = class_col in subset.columns
            summary = summarize_clustered_means(
                subset,
                value_cols=value_cols,
                draws=draws,
                include_macro_class=macro,
                class_col=class_col,
                identity={**identity, "analysis_stratum": stratum},
            )
            rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _classwise_summary(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws,
    class_col: str = "class_id",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for class_id, class_frame in frame.groupby(class_col, sort=True):
        for keys, group in _identity_groups(class_frame, group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            identity = dict(zip(group_cols, keys))
            summary = summarize_clustered_means(
                group,
                value_cols=value_cols,
                draws=draws,
                include_macro_class=False,
                identity={
                    **identity,
                    "analysis_stratum": "all",
                    "class_id": int(class_id),
                    "class_name": VOC_CLASS_NAMES[int(class_id)],
                },
            )
            summary["aggregation"] = "classwise"
            rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _weighted_auc_samples(
    scores: np.ndarray,
    labels: np.ndarray,
    image_columns: np.ndarray,
    multiplicities: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[float, np.ndarray]:
    """Exact weighted ROC-AUC samples, including tied-score half credit."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    image_columns = np.asarray(image_columns, dtype=np.int64)
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or image_columns.shape != scores.shape
    ):
        raise ValueError("scores, labels, and image columns must be equal vectors")
    if not np.isfinite(scores).all() or not np.all(
        np.isin(labels, (0, 1, False, True))
    ):
        raise ValueError("invalid AUROC input")
    labels = labels.astype(bool)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_columns = image_columns[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    ends = np.r_[starts[1:] - 1, len(scores) - 1]
    group_for_row = np.repeat(np.arange(len(starts)), ends - starts + 1)
    previous_index = np.where(group_for_row == 0, -1, ends[group_for_row - 1])
    end_index = ends[group_for_row]

    def auc_from_weights(weights: np.ndarray) -> np.ndarray:
        row_weights = weights[:, sorted_columns].astype(np.float64, copy=False)
        negative = row_weights * (~sorted_labels)
        cumulative_negative = np.cumsum(negative, axis=1)
        before = np.zeros_like(row_weights)
        has_previous = previous_index >= 0
        before[:, has_previous] = cumulative_negative[:, previous_index[has_previous]]
        group_negative = cumulative_negative[:, end_index] - before
        positive = row_weights * sorted_labels
        numerator = np.sum(positive * (before + 0.5 * group_negative), axis=1)
        positive_total = positive.sum(axis=1)
        negative_total = negative.sum(axis=1)
        return np.divide(
            numerator,
            positive_total * negative_total,
            out=np.full(len(weights), np.nan),
            where=(positive_total > 0) & (negative_total > 0),
        )

    unit_weights = np.ones((1, multiplicities.shape[1]), dtype=np.float64)
    point = float(auc_from_weights(unit_weights)[0])
    samples = np.full(len(multiplicities), np.nan, dtype=np.float64)
    for start in range(0, len(samples), chunk_size):
        selected = multiplicities[start : start + chunk_size]
        samples[start : start + len(selected)] = auc_from_weights(selected)
    return point, samples


def clustered_projection_auc_summary(
    frame: pd.DataFrame,
    *,
    draws: ImageBootstrapDraws,
    identity: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """OOF projection AUROC with whole-image weights and class macro estimates."""

    required = {"image_id", "class_id", "target", "heldout_projection"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"OOF frame missing {sorted(required - set(frame.columns))}")
    lookup = {image_id: index for index, image_id in enumerate(draws.image_ids)}
    if not set(frame["image_id"]).issubset(lookup):
        raise ValueError("OOF frame contains images outside bootstrap draws")
    columns = np.asarray([lookup[value] for value in frame["image_id"]], dtype=np.int64)
    point, samples = _weighted_auc_samples(
        frame["heldout_projection"].to_numpy(),
        frame["target"].to_numpy(),
        columns,
        draws.multiplicities,
    )
    class_points: list[float] = []
    class_samples: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for class_id, group in frame.groupby("class_id", sort=True):
        group_columns = np.asarray(
            [lookup[value] for value in group["image_id"]], dtype=np.int64
        )
        class_point, samples_for_class = _weighted_auc_samples(
            group["heldout_projection"].to_numpy(),
            group["target"].to_numpy(),
            group_columns,
            draws.multiplicities,
        )
        class_points.append(class_point)
        class_samples.append(samples_for_class)
        finite_class = samples_for_class[np.isfinite(samples_for_class)]
        low, high = (
            np.quantile(finite_class, (0.025, 0.975))
            if len(finite_class)
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "aggregation": "classwise",
                "class_id": int(class_id),
                "class_name": VOC_CLASS_NAMES[int(class_id)],
                "metric": "presence_projection_auroc",
                "estimate": class_point,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_valid_repeats": int(np.isfinite(samples_for_class).sum()),
            }
        )
    stacked_class_samples = np.stack(class_samples)
    finite_count = np.isfinite(stacked_class_samples).sum(axis=0)
    macro_samples = np.divide(
        np.nansum(stacked_class_samples, axis=0),
        finite_count,
        out=np.full(stacked_class_samples.shape[1], np.nan),
        where=finite_count > 0,
    )
    finite_class_points = np.asarray(class_points, dtype=np.float64)
    finite_class_points = finite_class_points[np.isfinite(finite_class_points)]
    macro_point = (
        float(finite_class_points.mean()) if len(finite_class_points) else float("nan")
    )
    for aggregation, estimate, values in (
        ("micro", point, samples),
        ("macro_class", macro_point, macro_samples),
    ):
        finite_values = values[np.isfinite(values)]
        low, high = (
            np.quantile(finite_values, (0.025, 0.975))
            if len(finite_values)
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "aggregation": aggregation,
                "class_id": None,
                "class_name": None,
                "metric": "presence_projection_auroc",
                "estimate": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_valid_repeats": int(np.isfinite(values).sum()),
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["bootstrap_repeats"] = draws.repeats
    result["bootstrap_seed"] = draws.seed
    result["bootstrap_unit"] = "image"
    result["ci_method"] = "95% percentile"
    result["fit_uncertainty"] = "conditional_on_fixed_crossfit_directions"
    if identity:
        for key, value in identity.items():
            result[key] = value
    return result


def paired_ci_decision(ci_low: float, ci_high: float, *, expected: str) -> str:
    """Apply the frozen sign-only paired-CI rule without point selection."""

    low, high = float(ci_low), float(ci_high)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return "undefined"
    if expected == "decrease":
        return "supported" if high < 0.0 else "not_supported"
    if expected == "increase":
        return "supported" if low > 0.0 else "not_supported"
    raise ValueError("expected must be 'decrease' or 'increase'")


def validation_a_decision(
    *,
    token_delta_ci: tuple[float, float],
    map_delta_ci: tuple[float, float],
    linkage_delta_ci: tuple[float, float],
    projection_auc_ci: tuple[float, float],
    final_signed_alignment: float,
) -> Mapping[str, object]:
    conditions = {
        "token_pair_cosine_decreases": paired_ci_decision(
            *token_delta_ci, expected="decrease"
        )
        == "supported",
        "class_map_overlap_decreases": paired_ci_decision(
            *map_delta_ci, expected="decrease"
        )
        == "supported",
        "perp_norm_agreement_increases": paired_ci_decision(
            *linkage_delta_ci, expected="increase"
        )
        == "supported",
        "oof_projection_above_chance": float(projection_auc_ci[0])
        > PRESENCE_AUROC_NULL,
        "final_signed_alignment_ge_090": float(final_signed_alignment)
        >= FINAL_DIRECTION_ALIGNMENT_THRESHOLD,
    }
    if all(conditions.values()):
        decision = "strong_support"
    elif (
        conditions["token_pair_cosine_decreases"]
        and not conditions["class_map_overlap_decreases"]
    ):
        decision = "partial_support_token_geometry_only"
    elif (
        not conditions["token_pair_cosine_decreases"]
        and not conditions["class_map_overlap_decreases"]
    ):
        decision = "not_supported"
    else:
        # Map-only evidence, or core token/map evidence whose controls do not
        # close, is neither the plan's token-only partial case nor evidence
        # that both geometries stayed unchanged.
        decision = "mixed_or_indeterminate"
    return {"decision": decision, "conditions": conditions}


def _paired_variant_summary(
    frame: pd.DataFrame,
    *,
    system_col: str,
    baseline: str,
    comparison: str,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws,
    identity: Mapping[str, object],
    class_col: str = "class_id",
) -> pd.DataFrame:
    return paired_clustered_mean_summary(
        frame,
        system_col=system_col,
        baseline=baseline,
        comparison=comparison,
        key_cols=key_cols,
        value_cols=value_cols,
        draws=draws,
        include_macro_class=class_col in frame.columns,
        class_col=class_col,
        identity=identity,
    )


def _build_compact_tables(
    canonical: Path,
    runs: Sequence[PresenceRun],
    draws: ImageBootstrapDraws,
) -> Mapping[str, pd.DataFrame]:
    token = pd.read_parquet(canonical / "token_axis.parquet")
    patch = pd.read_parquet(canonical / "patch_axis.parquet")
    oof = pd.read_parquet(canonical / "oof_projection.parquet")
    token_metrics = _clustered_summary(
        token,
        group_cols=("model", "layer"),
        value_cols=(
            "class_coefficient",
            "class_axis_energy",
            "class_norm",
            "class_residual_norm",
            "shared_axis_energy",
        ),
        draws=draws,
    )
    token_classwise = _classwise_summary(
        token[token["layer"].isin(CONTROL_LAYERS)],
        group_cols=("model", "layer"),
        value_cols=("class_axis_energy", "shared_axis_energy"),
        draws=draws,
    )
    token_metrics = pd.concat(
        [token_metrics, token_classwise], ignore_index=True, sort=False
    )
    token_paired_parts = []
    for layer, group in token.groupby("layer", sort=True):
        paired = paired_clustered_mean_summary(
            group,
            system_col="model",
            baseline="mctformer",
            comparison="mctformer_plus",
            key_cols=("image_id", "class_id", "layer"),
            value_cols=("class_coefficient", "class_axis_energy", "shared_axis_energy"),
            draws=draws,
            include_macro_class=True,
            identity={
                "layer": int(layer),
                "analysis_stratum": "all",
                "comparison_scope": "paired_model",
            },
        )
        token_paired_parts.append(paired)
    token_metrics = pd.concat(
        [token_metrics, *token_paired_parts], ignore_index=True, sort=False
    )
    patch_metrics = _clustered_summary(
        patch,
        group_cols=("model", "layer"),
        value_cols=(
            "patch_coefficient_mean",
            "patch_coefficient_std",
            "patch_axis_energy_mean",
            "patch_axis_energy_std",
        ),
        draws=draws,
    )
    patch_paired_parts = []
    for layer, group in patch.groupby("layer", sort=True):
        patch_paired_parts.append(
            paired_clustered_mean_summary(
                group,
                system_col="model",
                baseline="mctformer",
                comparison="mctformer_plus",
                key_cols=("image_id", "layer"),
                value_cols=(
                    "patch_coefficient_mean",
                    "patch_coefficient_std",
                    "patch_axis_energy_mean",
                    "patch_axis_energy_std",
                ),
                draws=draws,
                include_macro_class=False,
                identity={
                    "layer": int(layer),
                    "analysis_stratum": "all",
                    "comparison_scope": "paired_model",
                },
            )
        )
    patch_metrics = pd.concat(
        [patch_metrics, *patch_paired_parts], ignore_index=True, sort=False
    )
    patch_metrics["table_scope"] = "patch_axis"
    token_metrics["table_scope"] = "class_token_axis"
    token_metrics = pd.concat(
        [token_metrics, patch_metrics], ignore_index=True, sort=False
    )

    pair_parts: list[pd.DataFrame] = []
    pair_columns = [
        "model",
        "image_id",
        "layer",
        "class_a",
        "class_b",
        "label_stratum",
        "presence_pair_status",
        "prediction_pair_status",
        "raw_pair_cosine",
        "residual_pair_cosine",
        "cosine_delta",
        "axis_dot",
        "residual_dot",
    ]
    late_frames: list[pd.DataFrame] = []
    for layer in range(1, 13):
        frame = pd.read_parquet(
            canonical / "token_pairs.parquet",
            columns=pair_columns,
            filters=[("layer", "=", layer)],
        )
        endpoint = pd.concat(
            [
                frame.assign(class_id=frame["class_a"]),
                frame.assign(class_id=frame["class_b"]),
            ],
            ignore_index=True,
        )
        pair_parts.append(
            _clustered_summary(
                endpoint,
                group_cols=("model", "layer"),
                value_cols=(
                    "raw_pair_cosine",
                    "residual_pair_cosine",
                    "cosine_delta",
                    "axis_dot",
                    "residual_dot",
                ),
                draws=draws,
                class_col="class_id",
            )
        )
        pair_parts.append(
            paired_clustered_mean_summary(
                endpoint,
                system_col="model",
                baseline="mctformer",
                comparison="mctformer_plus",
                key_cols=(
                    "image_id",
                    "class_a",
                    "class_b",
                    "class_id",
                    "layer",
                ),
                value_cols=(
                    "raw_pair_cosine",
                    "residual_pair_cosine",
                    "cosine_delta",
                ),
                draws=draws,
                include_macro_class=True,
                identity={
                    "layer": layer,
                    "analysis_stratum": "all",
                    "comparison_scope": "paired_model",
                },
            )
        )
        if layer in CONTROL_LAYERS:
            pair_parts.append(
                _classwise_summary(
                    endpoint,
                    group_cols=("model", "layer"),
                    value_cols=(
                        "raw_pair_cosine",
                        "residual_pair_cosine",
                        "cosine_delta",
                    ),
                    draws=draws,
                )
            )
        if layer in PRIMARY_LATE_LAYERS:
            late_frames.append(endpoint)
    late = pd.concat(late_frames, ignore_index=True)
    late_summary = _clustered_summary(
        late,
        group_cols=("model",),
        value_cols=("raw_pair_cosine", "residual_pair_cosine", "cosine_delta"),
        draws=draws,
        class_col="class_id",
    )
    late_summary["layer"] = "L10-L12_pooled_primary"
    pair_parts.append(late_summary)
    pair_metrics = pd.concat(pair_parts, ignore_index=True, sort=False)

    region = pd.read_parquet(canonical / "probe_region.parquet")
    region_values = (
        "target_hit",
        "target_top10_fraction",
        "other_fg_top10_fraction",
        "bg_top10_fraction",
        "target_tail_enrich_10",
        "other_fg_tail_enrich_10",
        "bg_tail_enrich_10",
        "auc_target_bg",
        "ap_target_bg",
        "auc_target_other",
        "ap_target_other",
        "conditional_bg_mass",
        "target_bg_mean_margin",
    )
    region_metrics = _clustered_summary(
        region,
        group_cols=("model", "variant", "layer", "rho"),
        value_cols=region_values,
        draws=draws,
    )
    region_focus = region[
        region["layer"].isin(CONTROL_LAYERS)
        & region["variant"].isin(
            (
                "raw",
                "both_removed",
                "norm_timing_aligned",
                "qk_mean",
                "attn_c2p_conditional",
            )
        )
    ]
    region_classwise = _classwise_summary(
        region_focus,
        group_cols=("model", "variant", "layer", "rho"),
        value_cols=(
            "auc_target_bg",
            "auc_target_other",
            "target_top10_fraction",
            "bg_top10_fraction",
        ),
        draws=draws,
    )
    region_metrics = pd.concat(
        [region_metrics, region_classwise], ignore_index=True, sort=False
    )
    region_paired_parts = []
    paired_region = region_focus[region_focus["rho"] == 0.5]
    for keys, group in paired_region.groupby(["variant", "layer", "rho"], sort=True):
        variant, layer, rho = keys
        region_paired_parts.append(
            paired_clustered_mean_summary(
                group,
                system_col="model",
                baseline="mctformer",
                comparison="mctformer_plus",
                key_cols=("image_id", "class_id", "layer", "rho"),
                value_cols=(
                    "auc_target_bg",
                    "auc_target_other",
                    "target_top10_fraction",
                    "bg_top10_fraction",
                ),
                draws=draws,
                include_macro_class=True,
                identity={
                    "variant": variant,
                    "layer": int(layer),
                    "rho": float(rho),
                    "analysis_stratum": "all",
                    "comparison_scope": "paired_model",
                },
            )
        )
    region_metrics = pd.concat(
        [region_metrics, *region_paired_parts], ignore_index=True, sort=False
    )

    overlap = pd.read_parquet(canonical / "positive_map_overlap.parquet")
    overlap_endpoints = pd.concat(
        [
            overlap.assign(class_id=overlap["class_a"]),
            overlap.assign(class_id=overlap["class_b"]),
        ],
        ignore_index=True,
    )
    overlap_metrics = _clustered_summary(
        overlap_endpoints,
        group_cols=("model", "variant", "layer", "topk_ratio"),
        value_cols=("spearman", "topk_jaccard", "topk_overlap_coefficient"),
        draws=draws,
        class_col="class_id",
    )
    overlap_paired_parts = []
    paired_overlap = overlap[
        (overlap["topk_ratio"] == 0.10)
        & overlap["variant"].isin(("raw", "both_removed"))
    ]
    for (model, layer), group in paired_overlap.groupby(["model", "layer"], sort=True):
        overlap_paired_parts.append(
            _paired_variant_summary(
                group,
                system_col="variant",
                baseline="raw",
                comparison="both_removed",
                key_cols=("image_id", "class_a", "class_b", "layer", "topk_ratio"),
                value_cols=("spearman", "topk_jaccard", "topk_overlap_coefficient"),
                draws=draws,
                identity={
                    "model": model,
                    "layer": int(layer),
                    "topk_ratio": 0.10,
                    "analysis_stratum": "all",
                    "comparison_scope": "paired_axis_removal",
                },
                class_col="unused",
            )
        )
    if overlap_paired_parts:
        paired_overlap_summary = pd.concat(
            overlap_paired_parts, ignore_index=True, sort=False
        )
        paired_overlap_summary["table_scope"] = "paired_axis_removal"
        overlap_metrics = pd.concat(
            [overlap_metrics, paired_overlap_summary], ignore_index=True, sort=False
        )
    shared = pd.read_parquet(canonical / "shared_ownership.parquet")
    shared_endpoints = pd.concat(
        [
            shared.assign(class_id=shared["class_a"]),
            shared.assign(class_id=shared["class_b"]),
        ],
        ignore_index=True,
    )
    shared_metrics = _clustered_summary(
        shared_endpoints,
        group_cols=(
            "model",
            "variant",
            "representation_timing",
            "layer",
            "rho",
            "topk_ratio",
        ),
        value_cols=(
            "shared_set_size",
            "shared_target_a_fraction",
            "shared_target_b_fraction",
            "shared_other_fg_fraction",
            "shared_background_fraction",
            "shared_target_a_enrichment",
            "shared_target_b_enrichment",
            "shared_background_enrichment",
        ),
        draws=draws,
        include_strata=True,
        class_col="class_id",
    )
    shared_metrics["table_scope"] = "positive_shared_ownership"
    overlap_metrics["table_scope"] = "positive_pair_map_overlap"
    control = pd.read_parquet(canonical / "all_class_control_map_strata.parquet")
    control_metrics = _clustered_summary(
        control,
        group_cols=(
            "model",
            "variant",
            "representation_timing",
            "layer",
            "presence_pair_status",
            "prediction_pair_status",
        ),
        value_cols=(
            "spearman_mean",
            "top05_jaccard_mean",
            "top10_jaccard_mean",
            "top20_jaccard_mean",
        ),
        draws=draws,
        include_strata=False,
        class_col="unused",
    )
    control_metrics["table_scope"] = "all_class_reduced_pair_strata"
    map_metrics = pd.concat(
        [overlap_metrics, shared_metrics, control_metrics],
        ignore_index=True,
        sort=False,
    )

    linkage = pd.read_parquet(canonical / "probe_linkage.parquet")
    linkage_metrics = _clustered_summary(
        linkage,
        group_cols=("model", "link", "layer", "topk_ratio"),
        value_cols=("spearman", "topk_jaccard", "topk_overlap_coefficient"),
        draws=draws,
    )
    linkage_paired_parts = []
    paired_links = linkage[
        (linkage["topk_ratio"] == 0.10)
        & linkage["link"].isin(
            (
                "raw_to_norm_timing_aligned",
                "both_removed_to_norm_timing_aligned",
            )
        )
    ]
    for (model, layer), group in paired_links.groupby(["model", "layer"], sort=True):
        linkage_paired_parts.append(
            _paired_variant_summary(
                group,
                system_col="link",
                baseline="raw_to_norm_timing_aligned",
                comparison="both_removed_to_norm_timing_aligned",
                key_cols=("image_id", "class_id", "layer", "topk_ratio"),
                value_cols=("spearman", "topk_jaccard", "topk_overlap_coefficient"),
                draws=draws,
                identity={
                    "model": model,
                    "layer": int(layer),
                    "topk_ratio": 0.10,
                    "analysis_stratum": "all",
                    "comparison_scope": "paired_perp_vs_raw_norm_linkage",
                },
            )
        )
    linkage_metrics = pd.concat(
        [linkage_metrics, *linkage_paired_parts], ignore_index=True, sort=False
    )

    direction_rows: list[dict[str, object]] = []
    fixed = np.ones(384, dtype=np.float64) / math.sqrt(384.0)
    for run in runs:
        directions = run.directions
        for fit_fold in (0, 1):
            for layer in range(12):
                direction_rows.append(
                    {
                        "row_type": "shared_direction",
                        "model": run.model,
                        "fit_fold": fit_fold,
                        "eval_fold": 1 - fit_fold,
                        "layer": layer + 1,
                        "class_id": None,
                        "class_name": None,
                        "metric": "signed_alignment_with_all_ones",
                        "estimate": float(
                            directions["shared_directions"][fit_fold, layer] @ fixed
                        ),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "analysis_stratum": "all",
                        "fit_uncertainty": "fit_fold_point_estimate",
                    }
                )
                for class_id in range(20):
                    direction_rows.extend(
                        [
                            {
                                "row_type": "class_direction_alignment",
                                "model": run.model,
                                "fit_fold": fit_fold,
                                "eval_fold": 1 - fit_fold,
                                "layer": layer + 1,
                                "class_id": class_id,
                                "class_name": VOC_CLASS_NAMES[class_id],
                                "metric": "class_alignment",
                                "estimate": float(
                                    directions["class_alignment"][
                                        fit_fold, layer, class_id
                                    ]
                                ),
                                "ci_low": np.nan,
                                "ci_high": np.nan,
                                "analysis_stratum": "all",
                                "fit_uncertainty": "fit_fold_point_estimate",
                            },
                            {
                                "row_type": "loo_class_direction_alignment",
                                "model": run.model,
                                "fit_fold": fit_fold,
                                "eval_fold": 1 - fit_fold,
                                "layer": layer + 1,
                                "class_id": class_id,
                                "class_name": VOC_CLASS_NAMES[class_id],
                                "metric": "loo_class_alignment",
                                "estimate": float(
                                    directions["loo_class_alignment"][
                                        fit_fold, layer, class_id
                                    ]
                                ),
                                "ci_low": np.nan,
                                "ci_high": np.nan,
                                "analysis_stratum": "all",
                                "fit_uncertainty": "fit_fold_point_estimate",
                            },
                        ]
                    )
    projection_parts: list[pd.DataFrame] = []
    for (model, layer), group in oof.groupby(["model", "layer"], sort=True):
        for stratum, subset in _strata(group):
            if subset["target"].nunique() < 2:
                continue
            projection_parts.append(
                clustered_projection_auc_summary(
                    subset,
                    draws=draws,
                    identity={
                        "row_type": "oof_projection",
                        "model": model,
                        "layer": int(layer),
                        "analysis_stratum": stratum,
                        "fit_fold": None,
                        "eval_fold": None,
                    },
                )
            )
        for eval_fold, subset in group.groupby("eval_fold", sort=True):
            projection_parts.append(
                clustered_projection_auc_summary(
                    subset,
                    draws=draws,
                    identity={
                        "row_type": "oof_projection_fold_stability",
                        "model": model,
                        "layer": int(layer),
                        "analysis_stratum": f"eval_fold_{int(eval_fold)}",
                        "fit_fold": 1 - int(eval_fold),
                        "eval_fold": int(eval_fold),
                    },
                )
            )
    direction = pd.concat(
        [pd.DataFrame.from_records(direction_rows), *projection_parts],
        ignore_index=True,
        sort=False,
    )
    return {
        "presence_axis_token_metrics.csv": token_metrics,
        "presence_axis_pair_metrics.csv": pair_metrics,
        "presence_axis_map_metrics.csv": map_metrics,
        "shared_presence_direction.csv": direction,
        "presence_axis_gt_region_metrics.csv": region_metrics,
        "presence_axis_probe_linkage.csv": linkage_metrics,
    }


def _extract_delta_row(
    summary: pd.DataFrame,
    *,
    metric: str,
    aggregation: str = "micro",
) -> pd.Series:
    selected = summary[
        (summary["metric"] == metric)
        & (summary["aggregation"] == aggregation)
        & summary["paired_delta"].astype(bool)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one paired delta for {metric}, got {len(selected)}"
        )
    return selected.iloc[0]


def _primary_decision(
    canonical: Path,
    tables: Mapping[str, pd.DataFrame],
    draws: ImageBootstrapDraws,
    *,
    allow_not_evaluable: bool,
) -> Mapping[str, object]:
    pair = pd.read_parquet(
        canonical / "token_pairs.parquet",
        filters=[
            ("layer", "in", list(PRIMARY_LATE_LAYERS)),
            ("model", "=", "mctformer_plus"),
        ],
    )
    pair = pair[pair["presence_pair_status"] == "both_GT_positive"]
    empty_stat = {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "bootstrap_valid_repeats": 0,
    }
    if pair.empty:
        if not allow_not_evaluable:
            raise RuntimeError("full Validation A has no GT-positive class-pair rows")
        return {
            "decision": "smoke_not_evaluable",
            "conditions": {
                "token_pair_cosine_decreases": None,
                "class_map_overlap_decreases": None,
                "perp_norm_agreement_increases": None,
                "oof_projection_above_chance": None,
                "final_signed_alignment_ge_090": None,
            },
            "primary_statistics": {
                "token_pair_residual_minus_raw": empty_stat,
                "map_top10_both_removed_minus_raw": empty_stat,
                "perp_norm_minus_raw_norm_spearman": empty_stat,
                "l12_oof_projection_auroc": empty_stat,
                "l12_min_signed_alignment_across_fit_folds": None,
            },
            "reason": "smoke prefix contains no multi-label positive class pair",
        }
    token_summary = summarize_clustered_means(
        pair,
        value_cols=("cosine_delta",),
        draws=draws,
        include_macro_class=False,
        identity={"scope": "MCTformer+ L10-L12 GT-positive pairs"},
    )
    token_row = token_summary[token_summary["aggregation"] == "micro"].iloc[0]

    overlap = pd.read_parquet(
        canonical / "positive_map_overlap.parquet",
        filters=[
            ("layer", "in", list(PRIMARY_LATE_LAYERS)),
            ("model", "=", "mctformer_plus"),
            ("variant", "in", ["raw", "both_removed"]),
        ],
    )
    overlap = overlap[np.isclose(overlap["topk_ratio"], 0.10)]
    map_summary = _paired_variant_summary(
        overlap,
        system_col="variant",
        baseline="raw",
        comparison="both_removed",
        key_cols=("image_id", "class_a", "class_b", "layer", "topk_ratio"),
        value_cols=("topk_jaccard",),
        draws=draws,
        identity={"scope": "MCTformer+ L10-L12 positive class pairs"},
        class_col="unused",
    )
    map_row = _extract_delta_row(map_summary, metric="topk_jaccard")

    linkage = pd.read_parquet(
        canonical / "probe_linkage.parquet",
        filters=[
            ("layer", "in", list(PRIMARY_LATE_LAYERS)),
            ("model", "=", "mctformer_plus"),
            (
                "link",
                "in",
                ["raw_to_norm_timing_aligned", "both_removed_to_norm_timing_aligned"],
            ),
        ],
    )
    linkage = linkage[np.isclose(linkage["topk_ratio"], 0.10)]
    link_summary = _paired_variant_summary(
        linkage,
        system_col="link",
        baseline="raw_to_norm_timing_aligned",
        comparison="both_removed_to_norm_timing_aligned",
        key_cols=("image_id", "class_id", "layer", "topk_ratio"),
        value_cols=("spearman",),
        draws=draws,
        identity={"scope": "MCTformer+ L10-L12 timing-aligned normalized probe"},
    )
    link_row = _extract_delta_row(link_summary, metric="spearman")

    direction = tables["shared_presence_direction.csv"]
    projection = direction[
        (direction["row_type"] == "oof_projection")
        & (direction["model"] == "mctformer_plus")
        & (direction["layer"] == 12)
        & (direction["analysis_stratum"] == "all")
        & (direction["aggregation"] == "micro")
    ]
    if len(projection) != 1:
        raise RuntimeError("missing unique MCTformer+ L12 OOF projection AUROC")
    projection_row = projection.iloc[0]
    alignment = direction[
        (direction["row_type"] == "shared_direction")
        & (direction["model"] == "mctformer_plus")
        & (direction["layer"] == 12)
    ]["estimate"]
    if len(alignment) != 2:
        raise RuntimeError("missing two MCTformer+ L12 fit-fold alignments")
    final_alignment = float(alignment.min())
    decision = validation_a_decision(
        token_delta_ci=(float(token_row.ci_low), float(token_row.ci_high)),
        map_delta_ci=(float(map_row.ci_low), float(map_row.ci_high)),
        linkage_delta_ci=(float(link_row.ci_low), float(link_row.ci_high)),
        projection_auc_ci=(float(projection_row.ci_low), float(projection_row.ci_high)),
        final_signed_alignment=final_alignment,
    )
    return _json_safe(
        {
            **decision,
            "primary_statistics": {
                "token_pair_residual_minus_raw": token_row.to_dict(),
                "map_top10_both_removed_minus_raw": map_row.to_dict(),
                "perp_norm_minus_raw_norm_spearman": link_row.to_dict(),
                "l12_oof_projection_auroc": projection_row.to_dict(),
                "l12_min_signed_alignment_across_fit_folds": final_alignment,
            },
        }
    )


def _plot_or_placeholder(path: Path, draw) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    try:
        draw(axis)
    except Exception as error:
        axis.text(0.5, 0.5, f"Plot unavailable\n{error}", ha="center", va="center")
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_plots(tables: Mapping[str, pd.DataFrame], output: Path) -> None:
    token = tables["presence_axis_token_metrics.csv"]
    pair = tables["presence_axis_pair_metrics.csv"]
    maps = tables["presence_axis_map_metrics.csv"]
    directions = tables["shared_presence_direction.csv"]
    region = tables["presence_axis_gt_region_metrics.csv"]
    linkage = tables["presence_axis_probe_linkage.csv"]

    def legend_if_any(ax, *, fontsize: int) -> None:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=fontsize)

    def axis_energy(ax):
        selected = token[
            (token["table_scope"] == "class_token_axis")
            & (token["metric"] == "class_axis_energy")
            & (token["aggregation"] == "micro")
            & token["analysis_stratum"].isin(
                ["presence_status:GT_positive", "presence_status:GT_negative"]
            )
        ]
        for (model, stratum), group in selected.groupby(["model", "analysis_stratum"]):
            group = group.sort_values("layer")
            ax.plot(group.layer, group.estimate, marker="o", label=f"{model} {stratum}")
        ax.set(
            xlabel="Layer", ylabel="Axis energy", title="Fixed-axis energy by GT status"
        )
        legend_if_any(ax, fontsize=7)

    def pair_cosine(ax):
        selected = pair[
            (pair["analysis_stratum"] == "presence_pair_status:both_GT_positive")
            & (pair["aggregation"] == "micro")
            & pair["metric"].isin(["raw_pair_cosine", "residual_pair_cosine"])
        ].copy()
        selected = selected[pd.to_numeric(selected["layer"], errors="coerce").notna()]
        for (model, metric), group in selected.groupby(["model", "metric"]):
            group = group.sort_values("layer")
            ax.plot(
                group.layer.astype(int),
                group.estimate,
                marker="o",
                label=f"{model} {metric}",
            )
        ax.set(
            xlabel="Layer", ylabel="Cosine", title="Positive class-token pair geometry"
        )
        legend_if_any(ax, fontsize=7)

    def map_overlap(ax):
        selected = maps[
            (maps["table_scope"] == "positive_pair_map_overlap")
            & (maps["metric"] == "topk_jaccard")
            & (maps["topk_ratio"] == 0.10)
            & (maps["aggregation"] == "micro")
            & (maps["analysis_stratum"] == "all")
            & maps["variant"].isin(["raw", "both_removed"])
        ]
        for (model, variant), group in selected.groupby(["model", "variant"]):
            group = group.sort_values("layer")
            ax.plot(group.layer, group.estimate, marker="o", label=f"{model} {variant}")
        ax.set(
            xlabel="Layer", ylabel="Top-10% Jaccard", title="Positive-class map overlap"
        )
        legend_if_any(ax, fontsize=7)

    def probe(ax):
        selected = linkage[
            (linkage["metric"] == "spearman")
            & (linkage["topk_ratio"] == 0.10)
            & (linkage["aggregation"] == "micro")
            & (linkage["analysis_stratum"] == "all")
            & linkage["link"].isin(
                [
                    "raw_to_norm_timing_aligned",
                    "both_removed_to_norm_timing_aligned",
                    "raw_to_qk",
                ]
            )
        ]
        for (model, link), group in selected.groupby(["model", "link"]):
            group = group.sort_values("layer")
            ax.plot(group.layer, group.estimate, marker="o", label=f"{model} {link}")
        ax.set(
            xlabel="Layer",
            ylabel="Spearman",
            title="Raw/perpendicular/normalized/QK linkage",
        )
        legend_if_any(ax, fontsize=6)

    def alignment(ax):
        selected = directions[directions["row_type"] == "shared_direction"]
        for (model, fold), group in selected.groupby(["model", "fit_fold"]):
            group = group.sort_values("layer")
            ax.plot(
                group.layer, group.estimate, marker="o", label=f"{model} fit{int(fold)}"
            )
        ax.axhline(
            FINAL_DIRECTION_ALIGNMENT_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.set(
            xlabel="Layer",
            ylabel="Signed cosine with all-ones",
            title="Cross-fit shared-direction alignment",
        )
        legend_if_any(ax, fontsize=7)

    def region_quality(ax):
        selected = region[
            (region["metric"] == "auc_target_bg")
            & (region["rho"] == 0.5)
            & (region["aggregation"] == "micro")
            & (region["analysis_stratum"] == "all")
            & region["variant"].isin(
                [
                    "raw",
                    "both_removed",
                    "norm_timing_aligned",
                    "qk_mean",
                    "attn_c2p_conditional",
                ]
            )
        ]
        for (model, variant), group in selected.groupby(["model", "variant"]):
            group = group.sort_values("layer")
            ax.plot(group.layer, group.estimate, marker="o", label=f"{model} {variant}")
        ax.set(
            xlabel="Layer",
            ylabel="Target-vs-BG AUROC",
            title="Positive-class region quality",
        )
        legend_if_any(ax, fontsize=6)

    specs = {
        "axis_energy_by_layer_and_status.png": axis_energy,
        "class_pair_cosine_raw_vs_residual.png": pair_cosine,
        "class_map_overlap_raw_vs_axis_removed.png": map_overlap,
        "raw_perp_norm_qk_probe_comparison.png": probe,
        "shared_presence_direction_alignment.png": alignment,
        "presence_axis_region_quality.png": region_quality,
    }
    for filename, draw in specs.items():
        _plot_or_placeholder(output / filename, draw)


def _format_stat(record: Mapping[str, object]) -> str:
    if any(record.get(name) is None for name in ("estimate", "ci_low", "ci_high")):
        return "not evaluable"
    return (
        f"{float(record['estimate']):.4f} "
        f"[95% CI {float(record['ci_low']):.4f}, {float(record['ci_high']):.4f}]"
    )


def _write_report(
    path: Path,
    *,
    run_kind: str,
    num_images: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    canonical_metadata: Mapping[str, object],
    decision: Mapping[str, object],
) -> None:
    stats = decision["primary_statistics"]
    conditions = decision["conditions"]
    smoke_warning = (
        "**SMOKE ONLY — not eligible for scientific conclusions.**"
        if run_kind == "smoke"
        else "Full paired VOC-val analysis."
    )
    alignment_value = stats["l12_min_signed_alignment_across_fit_folds"]
    alignment_text = (
        "not evaluable" if alignment_value is None else f"{float(alignment_value):.4f}"
    )
    text = f"""# Validation A — Presence-Axis Decomposition

{smoke_warning}

## Fact

- Inputs: paired MCTformer and MCTformer+ `{run_kind}` runs with {num_images:,} identical evaluation images.
- Frozen deterministic direction split: SHA-256 image-ID parity; every evaluation image uses the opposite fit fold.
- Bootstrap: exactly {bootstrap_repeats:,} whole-image multinomial draws with seed `{bootstrap_seed}`. Patches, classes, and class pairs from one image were never resampled independently.
- Exact L12 classifier contract: `x_cls.mean(-1) = <x_cls, 1/sqrt(D)>/sqrt(D)`; both source runs passed the `<1e-6` numerical gate.
- Post-block Lk is compared with `norm1` entering L(k+1). L12 uses final LayerNorm only as an explicitly analysis-only control, not as a native CAM stage.
- Semantic overlap estimand: {SEMANTIC_OVERLAP_ESTIMAND}.
- Canonical artifacts: {len(canonical_metadata)} Parquet tables, each typed, compressed, hashed, and row-count verified.

## Statistical inference

- MCTformer+ L10–L12 GT-positive token-pair residual-minus-raw cosine: {_format_stat(stats["token_pair_residual_minus_raw"])}.
- MCTformer+ L10–L12 positive-class top-10% map Jaccard, both-removed minus raw: {_format_stat(stats["map_top10_both_removed_minus_raw"])}.
- MCTformer+ L10–L12 timing-aligned normalized-feature agreement, perpendicular-minus-raw Spearman: {_format_stat(stats["perp_norm_minus_raw_norm_spearman"])}.
- MCTformer+ L12 cross-fitted presence projection AUROC: {_format_stat(stats["l12_oof_projection_auroc"])}. This interval is conditional on the two fixed fitted directions.
- Minimum signed L12 shared-direction/all-ones alignment across fit folds: {alignment_text}; frozen descriptive threshold: {FINAL_DIRECTION_ALIGNMENT_THRESHOLD:.2f}.

Frozen condition outcomes:

{chr(10).join(f"- `{name}`: **{str(value).lower()}**" for name, value in conditions.items())}

Production-frozen operational outcome: **{decision["decision"]}**. The plan
pre-registered the qualitative decision logic; numerical CI and alignment
cutoffs were operationalized before the full VOC analysis and are not claimed
as an original numerical preregistration.

Layer-specific estimates outside the pooled L10–L12 primary analysis are descriptive and were not used to select a favorable layer.

## Mechanistic interpretation

The supported statements concern representation geometry only: how much the fixed dimension-mean direction and a cross-fitted shared presence direction account for class-token similarity and shared spatial score support. A positive result can show that late raw-feature recoupling contains a shared presence component; it does not by itself establish how attention or CAM computation caused that component.

## Unsupported

- No claim of background leakage, causal shortcut, or lazy semantic assignment follows from Validation A alone.
- No attention-behavior or CAM-behavior claim is made here; those require Validations B/C and their native-output controls.
- Final LayerNorm is an analysis-only probe and is not an additional model stage.
- Cross-fitted projection uncertainty is conditional on two fixed fitted directions; it is not a nested refit bootstrap.
- No best layer, proposed method, retraining result, or causal intervention is reported.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-run-root", type=Path, required=True)
    parser.add_argument("--mctformer-plus-run-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.bootstrap_repeats != DEFAULT_BOOTSTRAP_REPEATS:
        raise ValueError(
            "Validation A production analysis requires exactly 5,000 bootstrap repeats"
        )
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap seed must be non-negative")
    return args


def run_analysis(args: argparse.Namespace) -> Mapping[str, object]:
    require_tgca_repro()
    source_path = args.source_metadata.expanduser().resolve()
    source = read_json(source_path)
    source_hash = sha256_file(source_path)
    if source.get("status") != "complete" or source.get("integrity_passed") is not True:
        raise RuntimeError("Experiment 3 source audit is not a completed PASS")
    dataset = source.get("dataset")
    signal_roots = source.get("signal_roots")
    if not isinstance(dataset, Mapping) or not isinstance(signal_roots, Mapping):
        raise TypeError("source_metadata lacks dataset/signal roots")
    if int(dataset.get("num_images", -1)) != EXPECTED_IMAGES:
        raise RuntimeError(
            "source audit is not the full 1,449-image VOC validation set"
        )
    split_ids = ordered_val_ids(Path(str(dataset["list_path"])).resolve())
    if len(split_ids) != EXPECTED_IMAGES:
        raise RuntimeError("VOC validation list does not contain 1,449 IDs")

    runs = (
        load_presence_run(
            "mctformer",
            args.mctformer_run_root,
            source_metadata_path=source_path,
            source_metadata_sha256=source_hash,
            expected_split_ids=split_ids,
            allow_smoke=bool(args.allow_smoke),
        ),
        load_presence_run(
            "mctformer_plus",
            args.mctformer_plus_run_root,
            source_metadata_path=source_path,
            source_metadata_sha256=source_hash,
            expected_split_ids=split_ids,
            allow_smoke=bool(args.allow_smoke),
        ),
    )
    if runs[0].run_kind != runs[1].run_kind:
        raise RuntimeError("MCTformer and MCTformer+ full/smoke run kinds differ")
    if runs[0].processed_images != runs[1].processed_images:
        raise RuntimeError("paired Presence-Axis image counts differ")
    if [entry.image_id for entry in runs[0].entries] != [
        entry.image_id for entry in runs[1].entries
    ]:
        raise RuntimeError("paired Presence-Axis image membership/order differs")
    if runs[0].split_assignments != runs[1].split_assignments:
        raise RuntimeError("paired Presence-Axis cross-fit splits differ")

    exp2_roots = {
        model: load_experiment2_signal_root(
            model, Path(str(signal_roots[model])).resolve(), require_full=True
        )
        for model in MODEL_ORDER
    }
    assert_experiment2_manifest_match(
        exp2_roots["mctformer"], exp2_roots["mctformer_plus"]
    )
    output = assert_new_output(
        args.output_dir,
        [
            runs[0].root,
            runs[1].root,
            *(root.root for root in exp2_roots.values()),
            source_path,
            Path(str(dataset["list_path"])).resolve(),
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    tables_dir = output / "tables"
    plots_dir = output / "plots"
    canonical_dir = output / "canonical"
    tables_dir.mkdir()
    plots_dir.mkdir()
    canonical_dir.mkdir()
    command = shlex.join([sys.executable, *sys.argv])
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    json_dump(output / "decision_rules.json", DECISION_RULES)
    log_path = output / "analysis.log"

    def log(message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    metadata: dict[str, object] = {
        "status": "running",
        "analysis": "experiment3_validation_a_presence_axis_offline",
        "run_kind": runs[0].run_kind,
        "processed_images": runs[0].processed_images,
        "source_metadata": str(source_path),
        "source_metadata_sha256": source_hash,
        "input_runs": {run.model: str(run.root) for run in runs},
        "experiment2_signal_roots": {
            model: str(root.root) for model, root in exp2_roots.items()
        },
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "image",
            "fit_uncertainty": "conditional_on_fixed_crossfit_directions",
        },
        "semantic_overlap_estimand": SEMANTIC_OVERLAP_ESTIMAND,
        "decision_rules": dict(DECISION_RULES),
        "started_at": timestamp(),
        "command": command,
    }
    json_dump(output / "metadata.json", metadata)
    writers = {
        name: AtomicParquetWriter(canonical_dir, name, schema)
        for name, schema in CANONICAL_SCHEMAS.items()
    }
    exp2_by_model = {
        model: {entry.image_id: entry for entry in root.entries}
        for model, root in exp2_roots.items()
    }
    used_exp2_hashes: dict[Path, str] = {}
    exp2_control_hashes = {
        path: sha256_file(path)
        for root in exp2_roots.values()
        for path in (root.metadata_path, root.completion_path, root.manifest_path)
    }
    started = time.perf_counter()
    try:
        log(
            "validated paired run metadata, split, directions, manifests, and source roots"
        )
        for run in runs:
            for index, entry in enumerate(run.entries, 1):
                exp2_entry = exp2_by_model[run.model][entry.image_id]
                source_artifact = load_experiment2_artifact(exp2_entry)
                presence_artifact = load_and_validate_presence_artifact(
                    entry,
                    experiment2_entry=exp2_entry,
                    experiment2_artifact=source_artifact,
                )
                rows = _canonical_rows_for_image(
                    run.model, entry, presence_artifact, source_artifact
                )
                for name, values in rows.items():
                    writers[name].append(values)
                writers["source_index"].append(
                    [
                        {
                            "model": run.model,
                            "image_id": entry.image_id,
                            "presence_artifact_path": str(entry.artifact_path),
                            "presence_artifact_sha256": entry.artifact_sha256,
                            "experiment2_artifact_path": str(exp2_entry.artifact_path),
                            "experiment2_artifact_sha256": exp2_entry.artifact_sha256,
                            "source_hash_link_verified": True,
                            "presence_schema_verified": True,
                        }
                    ]
                )
                used_exp2_hashes[exp2_entry.artifact_path] = exp2_entry.artifact_sha256
                if index % 25 == 0 or index == len(run.entries):
                    log(
                        f"canonicalized model={run.model} images={index}/{len(run.entries)}"
                    )
        canonical_metadata = {name: writer.close() for name, writer in writers.items()}
        json_dump(
            canonical_dir / "canonical_metadata.json",
            {
                "status": "complete",
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "run_kind": runs[0].run_kind,
                "tables": canonical_metadata,
                "completed_at": timestamp(),
            },
        )
        expected_ids = [entry.image_id for entry in runs[0].entries]
        draws = image_multinomial_draws(
            expected_ids, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed
        )
        products = _build_compact_tables(canonical_dir, runs, draws)
        table_metadata = {
            filename: _write_csv_atomic(frame, tables_dir / filename)
            for filename, frame in products.items()
        }
        decision = _primary_decision(
            canonical_dir,
            products,
            draws,
            allow_not_evaluable=runs[0].run_kind == "smoke",
        )
        decision = _json_safe(decision)
        json_dump(output / "validation_a_decision.json", decision)
        _make_plots(products, plots_dir)
        _write_report(
            output / "VALIDATION_A_PRESENCE_AXIS.md",
            run_kind=runs[0].run_kind,
            num_images=runs[0].processed_images,
            bootstrap_repeats=args.bootstrap_repeats,
            bootstrap_seed=args.bootstrap_seed,
            canonical_metadata=canonical_metadata,
            decision=decision,
        )
        _verify_sources_unchanged(runs, source_path, source_hash)
        for path, digest in used_exp2_hashes.items():
            if sha256_file(path) != digest:
                raise RuntimeError(
                    f"Experiment 2 source changed during analysis: {path}"
                )
        for path, digest in exp2_control_hashes.items():
            if sha256_file(path) != digest:
                raise RuntimeError(
                    f"Experiment 2 control metadata changed during analysis: {path}"
                )
        artifact_rows: list[dict[str, object]] = []
        for path in sorted(value for value in output.rglob("*") if value.is_file()):
            if path.name in {
                # The append-only execution log receives its final completion
                # line after immutable artifact inventory construction.  Keep
                # it as an exact command log, but do not claim a stale digest
                # for it in the immutable generated-artifact manifest.
                "analysis.log",
                "metadata.json",
                "completion.json",
                "artifact_manifest.csv",
            }:
                continue
            artifact_rows.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        with (output / "artifact_manifest.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("relative_path", "size_bytes", "sha256")
            )
            writer.writeheader()
            writer.writerows(artifact_rows)
        metadata.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "canonical_tables": canonical_metadata,
                "compact_tables": table_metadata,
                "decision": decision,
                "source_immutability_verified": True,
                "artifact_manifest": {
                    "path": str(output / "artifact_manifest.csv"),
                    "rows": len(artifact_rows),
                    "sha256": sha256_file(output / "artifact_manifest.csv"),
                    "excluded_mutable_log": "analysis.log",
                },
                "completed_at": timestamp(),
            }
        )
        json_dump(output / "metadata.json", metadata)
        completion = {
            "status": "complete",
            "analysis": metadata["analysis"],
            "run_kind": runs[0].run_kind,
            "num_images": runs[0].processed_images,
            "bootstrap_repeats": args.bootstrap_repeats,
            "source_immutability_verified": True,
            "report": "VALIDATION_A_PRESENCE_AXIS.md",
            "completed_at": metadata["completed_at"],
        }
        json_dump(output / "completion.json", completion)
        log(f"complete decision={decision['decision']}")
        return metadata
    except Exception as error:
        for writer in writers.values():
            if not writer.closed:
                writer.abort()
        metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        json_dump(output / "metadata.json", metadata)
        raise


def main() -> None:
    result = run_analysis(parse_args())
    print(
        json.dumps(
            {"status": result["status"], "decision": result["decision"]["decision"]}
        )
    )


if __name__ == "__main__":
    main()
