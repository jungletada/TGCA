#!/usr/bin/env python3
"""Collect the matched MCTformer+ token-role seed-0 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--baseline-role-metrics", type=Path, required=True)
    parser.add_argument("--baseline-benchmark", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", default=[], metavar="MODE=RUN_DIR", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def final_classification_map(run_dir):
    text = (run_dir / "pipeline.log").read_text(encoding="utf-8")
    values = re.findall(r'"test_mAP":\s*([0-9.eE+-]+)', text)
    if not values:
        raise RuntimeError(f"No test_mAP values found in {run_dir / 'pipeline.log'}")
    return 100.0 * float(values[-1])


def averaged_role_value(role_metrics, stage, metric):
    values = [
        row["mean"]
        for row in role_metrics["aggregates"]
        if row["stage"] == stage and row["metric"] == metric
    ]
    if not values:
        raise RuntimeError(f"Missing {stage}/{metric} token-role metric")
    return sum(values) / len(values)


def collect_row(mode, run_dir, role_path, benchmark_path):
    raw = load_json(run_dir / "raw_cam_diagnostics" / "metrics.json")
    role = load_json(role_path)
    benchmark = load_json(benchmark_path)
    class_patch_pre = averaged_role_value(
        role, "pre_norm1", "class_patch_cosine"
    )
    class_patch_post = averaged_role_value(
        role, "post_norm1", "class_patch_cosine"
    )
    return {
        "mode": mode,
        "run_dir": str(run_dir.resolve()),
        "raw_cam_miou_percent": raw["mean_iou_percent"],
        "semantic_foreground_precision_percent": raw[
            "semantic_foreground_precision_percent"
        ],
        "semantic_foreground_recall_percent": raw[
            "semantic_foreground_recall_percent"
        ],
        "binary_foreground_precision_percent": raw[
            "binary_foreground_precision_percent"
        ],
        "binary_foreground_recall_percent": raw[
            "binary_foreground_recall_percent"
        ],
        "background_false_positive_rate_percent": raw[
            "background_false_positive_rate_percent"
        ],
        "final_classification_map_percent": final_classification_map(run_dir),
        "diagnostic_classification_map_percent": role[
            "classification_map_percent"
        ],
        "class_patch_cosine_pre_norm1": class_patch_pre,
        "class_patch_cosine_post_norm1": class_patch_post,
        "class_patch_cosine_norm_delta": class_patch_post - class_patch_pre,
        "total_parameters": benchmark["total_parameters"],
        "role_specialization_parameters": benchmark[
            "role_specialization_parameters"
        ],
        "latency_ms_mean": benchmark["latency_ms_mean"],
        "peak_allocated_memory_mb": benchmark["peak_allocated_memory_mb"],
    }


def main():
    args = parse_args()
    output_json = args.output_dir / "summary.json"
    output_csv = args.output_dir / "summary.csv"
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("Refusing to overwrite token-role pilot summary")

    rows = [
        collect_row(
            "shared",
            args.baseline_run,
            args.baseline_role_metrics,
            args.baseline_benchmark,
        )
    ]
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Expected MODE=RUN_DIR, got {spec!r}")
        mode, path = spec.split("=", 1)
        run_dir = Path(path)
        rows.append(
            collect_row(
                mode,
                run_dir,
                run_dir / "token_role_diagnostics" / "metrics.json",
                run_dir / "benchmark.json",
            )
        )

    baseline = rows[0]
    for row in rows:
        row["raw_cam_miou_delta_vs_shared"] = (
            row["raw_cam_miou_percent"] - baseline["raw_cam_miou_percent"]
        )
        row["classification_map_delta_vs_shared"] = (
            row["final_classification_map_percent"]
            - baseline["final_classification_map_percent"]
        )
        row["parameter_increase_percent"] = 100.0 * (
            row["total_parameters"] - baseline["total_parameters"]
        ) / baseline["total_parameters"]
        row["latency_increase_percent"] = 100.0 * (
            row["latency_ms_mean"] - baseline["latency_ms_mean"]
        ) / baseline["latency_ms_mean"]

    payload = {
        "experiment": "MCTformer+ class-token/patch-token specialization pilot",
        "dataset": "PASCAL VOC 2012",
        "seed": 0,
        "fixed_background_threshold": 0.45,
        "baseline_reused": True,
        "rows": rows,
    }
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
