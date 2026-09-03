#!/usr/bin/env python3
"""Build immutable-source canonical Parquet tables for Experiment 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import (
    AnalysisLog,
    MODEL_ORDER,
    VOC_CLASS_NAMES,
    assert_output_outside_sources,
    json_dump,
    resolve_completed_result_root,
    timestamp,
)
from analysis.lazy_assignment.metrics_experiment1 import (
    cosine_similarity,
    score_map_metrics,
    spearman_correlation,
    topk_jaccard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-results", type=Path, required=True)
    parser.add_argument("--mctformer-plus-results", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_manifest(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        image_id = str(record["image_id"])
        if image_id in records:
            raise ValueError(f"duplicate image {image_id} at manifest line {line_number}")
        records[image_id] = record
    return records


def load_score_npz(path: Path, mapping: dict[str, str]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        return {
            "image_id": str(artifact[mapping["image_id"]].item()),
            "class_ids": np.asarray(
                artifact[mapping["positive_class_ids"]], dtype=np.int64
            ).reshape(-1),
            "scores": np.asarray(artifact[mapping["scores_raw"]], dtype=np.float32),
            "grid_h": int(artifact[mapping["grid_h"]].item()),
            "grid_w": int(artifact[mapping["grid_w"]].item()),
        }


def parquet_write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    reloaded = pd.read_parquet(path, engine="pyarrow")
    if len(reloaded) != len(frame) or list(reloaded.columns) != list(frame.columns):
        raise RuntimeError(f"Parquet round-trip shape/schema mismatch: {path}")


def map_record(
    model: str,
    image_id: str,
    class_id: int,
    layer: int,
    class_count: int,
    values: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> dict[str, object]:
    return {
        "model": model,
        "image_id": image_id,
        "class_id": int(class_id),
        "class_name": VOC_CLASS_NAMES[int(class_id)],
        "layer": int(layer),
        "block_index": int(layer - 1),
        "num_positive_classes": int(class_count),
        "is_multilabel": bool(class_count >= 2),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        **score_map_metrics(values, grid_h, grid_w),
    }


def build_tables(args: argparse.Namespace) -> dict[str, object]:
    mct_root, _ = resolve_completed_result_root(args.mctformer_results, "mctformer")
    plus_root, _ = resolve_completed_result_root(
        args.mctformer_plus_results, "mctformer_plus"
    )
    roots = {"mctformer": mct_root, "mctformer_plus": plus_root}
    output = args.output_dir.resolve()
    assert_output_outside_sources(output, tuple(roots.values()))
    output.mkdir(parents=True, exist_ok=False)
    log = AnalysisLog(output / "canonical.log")

    audit_report = json.loads(
        (args.audit_dir.resolve() / "integrity_report.json").read_text(encoding="utf-8")
    )
    if not audit_report.get("integrity_passed"):
        raise RuntimeError("canonical construction requires a passing integrity audit")
    mappings = audit_report["schema_mapping"]
    manifests = {model: read_manifest(root) for model, root in roots.items()}
    common_images = sorted(
        set(manifests["mctformer"]) & set(manifests["mctformer_plus"])
    )
    if len(common_images) != audit_report["common_pairs"]["common_images"]:
        raise RuntimeError("live manifests no longer match the audit common-image count")

    inventory = pd.read_csv(args.audit_dir.resolve() / "file_manifest.csv")
    hash_lookup = {
        (str(row.model), str(row.relative_path)): str(row.sha256_before)
        for row in inventory.itertuples(index=False)
    }

    map_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    multiclass_pair_rows: list[dict[str, object]] = []
    cross_model_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    log(f"Building canonical records for {len(common_images)} common images")

    for image_number, image_id in enumerate(common_images, start=1):
        loaded: dict[str, dict[str, Any]] = {}
        for model in MODEL_ORDER:
            record = manifests[model][image_id]
            relative_path = str(record["score_path"])
            score_path = (roots[model] / relative_path).resolve()
            item = load_score_npz(score_path, mappings[model])
            if item["image_id"] != image_id:
                raise RuntimeError(f"image ID changed after audit: {score_path}")
            loaded[model] = item
            class_ids = item["class_ids"]
            scores = item["scores"]
            num_layers, num_classes, num_patches = scores.shape
            if num_classes != len(class_ids):
                raise RuntimeError(f"class axis mismatch after audit: {score_path}")
            source_rows.append(
                {
                    "model": model,
                    "image_id": image_id,
                    "score_path": str(score_path),
                    "relative_score_path": relative_path,
                    "source_sha256_before": hash_lookup[(model, relative_path)],
                    "positive_class_ids_json": json.dumps(class_ids.tolist()),
                    "num_positive_classes": int(num_classes),
                    "num_layers": int(num_layers),
                    "num_patches": int(num_patches),
                    "grid_h": int(item["grid_h"]),
                    "grid_w": int(item["grid_w"]),
                    "score_dtype": str(scores.dtype),
                }
            )

            for class_index, class_id in enumerate(class_ids):
                class_maps = scores[:, class_index, :]
                first_map = class_maps[0]
                previous_map: np.ndarray | None = None
                for layer_index, values in enumerate(class_maps):
                    layer = layer_index + 1
                    map_rows.append(
                        map_record(
                            model,
                            image_id,
                            int(class_id),
                            layer,
                            num_classes,
                            values,
                            item["grid_h"],
                            item["grid_w"],
                        )
                    )
                    rank_rows.append(
                        {
                            "model": model,
                            "image_id": image_id,
                            "class_id": int(class_id),
                            "class_name": VOC_CLASS_NAMES[int(class_id)],
                            "layer": layer,
                            "block_index": layer_index,
                            "num_positive_classes": int(num_classes),
                            "consecutive_layer_spearman": (
                                spearman_correlation(previous_map, values)
                                if previous_map is not None
                                else np.nan
                            ),
                            "layer1_to_layer_spearman": (
                                1.0
                                if layer == 1
                                else spearman_correlation(first_map, values)
                            ),
                            "consecutive_layer_top10_jaccard": (
                                topk_jaccard(previous_map, values, 0.10)
                                if previous_map is not None
                                else np.nan
                            ),
                            "layer1_to_layer_top10_jaccard": (
                                1.0
                                if layer == 1
                                else topk_jaccard(first_map, values, 0.10)
                            ),
                        }
                    )
                    previous_map = values

            if num_classes >= 2:
                for left_index in range(num_classes - 1):
                    for right_index in range(left_index + 1, num_classes):
                        left_class = int(class_ids[left_index])
                        right_class = int(class_ids[right_index])
                        for layer_index in range(num_layers):
                            left_map = scores[layer_index, left_index]
                            right_map = scores[layer_index, right_index]
                            multiclass_pair_rows.append(
                                {
                                    "model": model,
                                    "image_id": image_id,
                                    "class_id_a": left_class,
                                    "class_name_a": VOC_CLASS_NAMES[left_class],
                                    "class_id_b": right_class,
                                    "class_name_b": VOC_CLASS_NAMES[right_class],
                                    "layer": layer_index + 1,
                                    "block_index": layer_index,
                                    "num_positive_classes": int(num_classes),
                                    "pairwise_class_spearman": spearman_correlation(
                                        left_map, right_map
                                    ),
                                    "pairwise_class_cosine": cosine_similarity(
                                        left_map, right_map
                                    ),
                                    "top05_class_map_jaccard": topk_jaccard(
                                        left_map, right_map, 0.05
                                    ),
                                    "top10_class_map_jaccard": topk_jaccard(
                                        left_map, right_map, 0.10
                                    ),
                                    "top20_class_map_jaccard": topk_jaccard(
                                        left_map, right_map, 0.20
                                    ),
                                }
                            )

        mct = loaded["mctformer"]
        plus = loaded["mctformer_plus"]
        common_classes = sorted(set(mct["class_ids"]) & set(plus["class_ids"]))
        mct_indices = {int(class_id): index for index, class_id in enumerate(mct["class_ids"])}
        plus_indices = {
            int(class_id): index for index, class_id in enumerate(plus["class_ids"])
        }
        for class_id in common_classes:
            for layer_index in range(mct["scores"].shape[0]):
                left = mct["scores"][layer_index, mct_indices[int(class_id)]]
                right = plus["scores"][layer_index, plus_indices[int(class_id)]]
                cross_model_rows.append(
                    {
                        "image_id": image_id,
                        "class_id": int(class_id),
                        "class_name": VOC_CLASS_NAMES[int(class_id)],
                        "layer": layer_index + 1,
                        "block_index": layer_index,
                        "cross_model_map_spearman": spearman_correlation(left, right),
                        "cross_model_map_cosine": cosine_similarity(left, right),
                        "cross_model_top05_jaccard": topk_jaccard(left, right, 0.05),
                        "cross_model_top10_jaccard": topk_jaccard(left, right, 0.10),
                        "cross_model_top20_jaccard": topk_jaccard(left, right, 0.20),
                    }
                )
        if image_number % 200 == 0 or image_number == len(common_images):
            log(f"Processed {image_number}/{len(common_images)} images")

    maps = pd.DataFrame(map_rows).sort_values(
        ["model", "image_id", "class_id", "layer"], ignore_index=True
    )
    ranks = pd.DataFrame(rank_rows).sort_values(
        ["model", "image_id", "class_id", "layer"], ignore_index=True
    )
    multiclass_pairs = pd.DataFrame(multiclass_pair_rows).sort_values(
        ["model", "image_id", "class_id_a", "class_id_b", "layer"],
        ignore_index=True,
    )
    cross_model = pd.DataFrame(cross_model_rows).sort_values(
        ["image_id", "class_id", "layer"], ignore_index=True
    )
    source_index = pd.DataFrame(source_rows).sort_values(
        ["model", "image_id"], ignore_index=True
    )

    key_cols = ["image_id", "class_id", "class_name", "layer", "block_index"]
    metadata_cols = ["num_positive_classes", "is_multilabel", "grid_h", "grid_w"]
    metric_cols = [
        column
        for column in maps.columns
        if column not in {"model", *key_cols, *metadata_cols}
    ]
    wide_parts: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        subset = maps[maps["model"] == model][key_cols + metadata_cols + metric_cols]
        rename = {
            column: f"{column}_{model}"
            for column in [*metadata_cols, *metric_cols]
        }
        wide_parts.append(subset.rename(columns=rename))
    paired = wide_parts[0].merge(wide_parts[1], on=key_cols, validate="one_to_one")
    for metric in metric_cols:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_mctformer_plus"] - paired[f"{metric}_mctformer"]
        )
    paired = paired.merge(cross_model, on=key_cols, validate="one_to_one")
    paired = paired.sort_values(["image_id", "class_id", "layer"], ignore_index=True)

    multiclass_image = (
        multiclass_pairs.groupby(["model", "image_id", "layer", "block_index"], sort=True)
        .agg(
            num_positive_classes=("num_positive_classes", "first"),
            num_class_pairs=("class_id_a", "size"),
            pairwise_class_spearman=("pairwise_class_spearman", "mean"),
            pairwise_class_cosine=("pairwise_class_cosine", "mean"),
            top05_class_map_jaccard=("top05_class_map_jaccard", "mean"),
            top10_class_map_jaccard=("top10_class_map_jaccard", "mean"),
            top20_class_map_jaccard=("top20_class_map_jaccard", "mean"),
        )
        .reset_index()
    )

    numeric_metrics = maps.select_dtypes(include=[np.number]).columns.difference(
        ["class_id", "layer", "block_index", "num_positive_classes", "grid_h", "grid_w"]
    )
    model_layer = (
        maps.groupby(["model", "layer", "block_index"], sort=True)
        .agg(
            num_images=("image_id", "nunique"),
            num_image_class_pairs=("image_id", "size"),
            **{f"mean_{column}": (column, "mean") for column in numeric_metrics},
        )
        .reset_index()
    )

    tables = {
        "per_image_class_layer.parquet": maps,
        "per_pair_layer.parquet": paired,
        "per_model_layer.parquet": model_layer,
        "source_index.parquet": source_index,
        "rank_stability.parquet": ranks,
        "multiclass_pair_layer.parquet": multiclass_pairs,
        "multiclass_image_layer.parquet": multiclass_image,
    }
    for filename, frame in tables.items():
        parquet_write(frame, output / filename)
        log(f"Wrote {filename}: {len(frame)} rows x {len(frame.columns)} columns")

    expected_map_rows = 2 * audit_report["common_pairs"]["common_image_class_pairs"] * 12
    if len(maps) != expected_map_rows:
        raise RuntimeError(f"expected {expected_map_rows} map rows, found {len(maps)}")
    if maps.duplicated(["model", "image_id", "class_id", "layer"]).any():
        raise RuntimeError("canonical map key is not unique")
    if paired.duplicated(["image_id", "class_id", "layer"]).any():
        raise RuntimeError("canonical paired key is not unique")

    metadata: dict[str, object] = {
        "generated_at": timestamp(),
        "source_roots": {model: str(root) for model, root in roots.items()},
        "audit_report": str((args.audit_dir.resolve() / "integrity_report.json")),
        "parquet_engine": "pyarrow",
        "parquet_compression": "zstd",
        "analysis_unit": "model x image_id x positive class_id x layer",
        "score_definition": "cosine(class token, patch token), post-block pre-final-norm",
        "topk_definition": "ceil(num_patches * ratio), exact-size stable descending sort; lower flat patch index wins ties",
        "entropy_definition": "normalized entropy of an auxiliary spatial softmax over cosine scores; not model attention entropy",
        "entropy_temperatures": [0.05, 0.10, 0.20],
        "layer_convention": "layer 1..12; block_index 0..11",
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "column_counts": {name: len(frame.columns) for name, frame in tables.items()},
        "primary_map_metrics": metric_cols,
        "source_npz_writes": 0,
    }
    json_dump(output / "canonical_metadata.json", metadata)
    log("Canonical construction and Parquet round-trip validation complete")
    return metadata


def main() -> None:
    build_tables(parse_args())


if __name__ == "__main__":
    main()
