#!/usr/bin/env python3
"""Verify canonical statistics against a deterministic sample of source NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import json_dump, timestamp
from analysis.lazy_assignment.metrics_experiment1 import QUANTILE_LEVELS


CHECK_COLUMNS = (
    "score_mean",
    "score_std",
    "score_max",
    "score_q01",
    "score_q05",
    "score_q10",
    "score_q25",
    "score_q50",
    "score_q75",
    "score_q90",
    "score_q95",
    "score_q99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def direct_statistics(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(flat, QUANTILE_LEVELS)
    result = {
        "score_mean": float(flat.mean()),
        "score_std": float(flat.std(ddof=0)),
        "score_max": float(flat.max()),
    }
    for level, value in zip(QUANTILE_LEVELS, quantiles):
        result[f"score_q{int(round(level * 100)):02d}"] = float(value)
    return result


def verify(args: argparse.Namespace) -> dict[str, object]:
    canonical = args.canonical_dir.resolve()
    audit = args.audit_dir.resolve()
    source_index = pd.read_parquet(canonical / "source_index.parquet")
    maps = pd.read_parquet(canonical / "per_image_class_layer.parquet")
    schema = json.loads((audit / "schema_mapping.json").read_text(encoding="utf-8"))
    if args.sample_size > len(source_index):
        raise ValueError("sample size exceeds source-index rows")
    sampled = source_index.sample(n=args.sample_size, random_state=args.seed).sort_values(
        ["model", "image_id"]
    )
    failures: list[dict[str, object]] = []
    comparisons = 0
    maximum_absolute_error = 0.0
    sampled_paths: list[str] = []
    for source in sampled.itertuples(index=False):
        path = Path(source.score_path)
        sampled_paths.append(str(path))
        mapping = schema[str(source.model)]
        with np.load(path, allow_pickle=False) as artifact:
            class_ids = np.asarray(
                artifact[mapping["positive_class_ids"]], dtype=np.int64
            ).reshape(-1)
            scores = np.asarray(artifact[mapping["scores_raw"]], dtype=np.float32)
        for class_index, class_id in enumerate(class_ids):
            for layer_index in range(scores.shape[0]):
                expected = direct_statistics(scores[layer_index, class_index])
                row = maps[
                    (maps["model"] == source.model)
                    & (maps["image_id"] == source.image_id)
                    & (maps["class_id"] == int(class_id))
                    & (maps["layer"] == layer_index + 1)
                ]
                if len(row) != 1:
                    failures.append(
                        {
                            "model": source.model,
                            "image_id": source.image_id,
                            "class_id": int(class_id),
                            "layer": layer_index + 1,
                            "metric": "canonical_key_count",
                            "expected": 1,
                            "actual": len(row),
                        }
                    )
                    continue
                canonical_row = row.iloc[0]
                for metric in CHECK_COLUMNS:
                    difference = abs(float(canonical_row[metric]) - expected[metric])
                    maximum_absolute_error = max(maximum_absolute_error, difference)
                    comparisons += 1
                    if difference > args.atol:
                        failures.append(
                            {
                                "model": source.model,
                                "image_id": source.image_id,
                                "class_id": int(class_id),
                                "layer": layer_index + 1,
                                "metric": metric,
                                "expected": expected[metric],
                                "actual": float(canonical_row[metric]),
                                "absolute_error": difference,
                            }
                        )
    report: dict[str, object] = {
        "generated_at": timestamp(),
        "passed": not failures,
        "sample_seed": args.seed,
        "sample_size_npz": args.sample_size,
        "sampled_paths": sampled_paths,
        "checked_metrics": list(CHECK_COLUMNS),
        "comparisons": comparisons,
        "absolute_tolerance": args.atol,
        "maximum_absolute_error": maximum_absolute_error,
        "failures": failures,
    }
    json_dump(args.output.resolve(), report)
    if failures:
        raise RuntimeError(f"canonical verification failed with {len(failures)} errors")
    return report


def main() -> None:
    verify(parse_args())


if __name__ == "__main__":
    main()
