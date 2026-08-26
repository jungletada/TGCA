#!/usr/bin/env python3
"""Audit completed MCTformer+ normalization runs and export one comparison."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path


CORE_MODES = ("vanilla", "split_11", "split_05", "tgca", "tgca_bias")
STAGE_PATTERN = re.compile(r"^(STAGE=[^ ]+|PIPELINE_COMPLETE) (?:started|finished)=([^ ]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_training(run_dir):
    log_paths = sorted((run_dir / "log_dir").glob("train-*.log"))
    if len(log_paths) != 1:
        raise ValueError(f"Expected one training log in {run_dir}, found {len(log_paths)}")
    epochs = []
    for line in log_paths[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            if "epoch" in record and "test_mAP" in record:
                epochs.append(record)
    if not epochs:
        raise ValueError(f"No epoch metrics found in {log_paths[0]}")
    epochs.sort(key=lambda item: item["epoch"])
    if epochs[-1]["epoch"] != 44 or len(epochs) != 45:
        raise ValueError(f"Training is incomplete in {log_paths[0]}")
    return {
        "classification_final_map_percent": 100.0 * float(epochs[-1]["test_mAP"]),
        "classification_max_map_percent": 100.0 * max(float(item["test_mAP"]) for item in epochs),
        "classification_max_map_epoch": max(epochs, key=lambda item: item["test_mAP"])["epoch"],
        "training_log": str(log_paths[0]),
    }


def parse_pipeline(run_dir):
    path = run_dir / "pipeline.log"
    text = path.read_text(encoding="utf-8")
    if "PIPELINE_COMPLETE" not in text:
        raise ValueError(f"Pipeline is incomplete in {path}")
    timestamps = {}
    for line in text.splitlines():
        match = STAGE_PATTERN.match(line)
        if match:
            timestamps[match.group(1)] = dt.datetime.fromisoformat(match.group(2))
    train_start = timestamps.get("STAGE=train")
    finish = timestamps.get("PIPELINE_COMPLETE")
    maximum_memory = max(
        (int(value) for value in re.findall(r"max mem: ([0-9]+)", text)),
        default=None,
    )
    return {
        "pipeline_wall_seconds": (
            None if train_start is None or finish is None else (finish - train_start).total_seconds()
        ),
        "peak_training_gpu_memory_mb": maximum_memory,
        "pipeline_log": str(path),
    }


def parse_manifest_digest(path):
    fields = path.read_text(encoding="utf-8").split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise ValueError(f"Invalid SHA-256 manifest: {path}")
    return fields[0]


def collect(run_dir):
    required = (
        "metrics.json", "config.json", "git_state.json", "checkpoint_manifest.txt",
        "pipeline.log", "attention_diagnostics/metrics.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}: missing {missing}")

    cam = load_json(run_dir / "metrics.json")
    config = load_json(run_dir / "config.json")
    git_state = load_json(run_dir / "git_state.json")
    attention = load_json(run_dir / "attention_diagnostics" / "metrics.json")
    mode = cam["normalization"]
    if mode not in CORE_MODES:
        raise ValueError(f"Unexpected mode {mode!r} in {run_dir}")
    if config["mode"] != mode or attention["normalization"] != mode:
        raise ValueError(f"Normalization provenance mismatch in {run_dir}")
    manifest_digest = parse_manifest_digest(run_dir / "checkpoint_manifest.txt")
    if cam["checkpoint_sha256"] != manifest_digest:
        raise ValueError(f"Checkpoint digest mismatch in {run_dir}")
    if cam["commit"] != git_state["commit"]:
        raise ValueError(f"Commit provenance mismatch in {run_dir}")
    if cam["background_threshold"] != config["fixed_background_threshold"]:
        raise ValueError(f"Threshold provenance mismatch in {run_dir}")
    if attention["num_images"] != cam["num_images"] or attention["resolutions"] != [224, 320, 448, 512]:
        raise ValueError(f"Attention diagnostic coverage mismatch in {run_dir}")

    row = {
        "run_id": run_dir.name,
        "normalization": mode,
        "seed": cam["seed"],
        "commit": cam["commit"],
        "background_threshold": cam["background_threshold"],
        "raw_cam_miou_percent": cam["raw_cam_miou_percent"],
        "checkpoint_sha256": cam["checkpoint_sha256"],
        "attention_row_sum_target": 2.0 if mode == "split_11" else 1.0,
        "maximum_attention_row_sum_error": attention["maximum_row_sum_error"],
        "mean_group_mass_variance": attention["mean_group_mass_variance"],
        "median_group_mass_variance": attention["median_group_mass_variance"],
        "attention_diagnostic_elapsed_seconds": attention["elapsed_seconds"],
    }
    for direction, summary in attention["directional_mass_slope"].items():
        row[f"{direction}_mean"] = summary["mean"]
        row[f"{direction}_ci95_low"] = summary["bootstrap_mean_ci95"][0]
        row[f"{direction}_ci95_high"] = summary["bootstrap_mean_ci95"][1]
    row.update(parse_training(run_dir))
    row.update(parse_pipeline(run_dir))

    raw_diagnostic_path = run_dir / "raw_cam_diagnostics" / "metrics.json"
    if raw_diagnostic_path.is_file():
        raw = load_json(raw_diagnostic_path)
        if abs(raw["mean_iou_percent"] - cam["raw_cam_miou_percent"]) > 0.011:
            raise ValueError(f"Raw CAM diagnostic mIoU does not reproduce primary metric in {run_dir}")
        for key in (
            "semantic_foreground_precision_percent",
            "semantic_foreground_recall_percent",
            "binary_foreground_precision_percent",
            "binary_foreground_recall_percent",
            "background_false_positive_rate_percent",
        ):
            row[key] = raw[key]
    efficiency_path = run_dir / "efficiency" / "metrics.json"
    if efficiency_path.is_file():
        efficiency = load_json(efficiency_path)
        if efficiency["normalization"] != mode:
            raise ValueError(f"Efficiency mode mismatch in {run_dir}")
        for key in (
            "total_parameters",
            "trainable_parameters",
            "relation_bias_parameters",
            "latency_ms_mean",
            "latency_ms_median",
            "latency_ms_p95",
            "throughput_images_per_second",
            "peak_allocated_memory_mb",
            "incremental_peak_allocated_memory_mb",
        ):
            row[key] = efficiency[key]
    scale_consistency_path = run_dir / "scale_consistency" / "metrics.json"
    if scale_consistency_path.is_file():
        scale_consistency = load_json(scale_consistency_path)
        if scale_consistency["normalization"] != mode:
            raise ValueError(f"Scale-consistency mode mismatch in {run_dir}")
        for resolution, metric_summaries in scale_consistency["summary"].items():
            for metric_name, summary in metric_summaries.items():
                prefix = f"scale_{resolution}_vs_448_{metric_name}"
                row[f"{prefix}_mean"] = summary["mean"]
                row[f"{prefix}_ci95_low"] = summary["ci95"][0]
                row[f"{prefix}_ci95_high"] = summary["ci95"][1]
    for resolution in (224, 320, 448, 512):
        scale_quality_path = run_dir / "scale_cam_metrics" / str(resolution) / "metrics.json"
        if scale_quality_path.is_file():
            scale_quality = load_json(scale_quality_path)
            row[f"scale_{resolution}_raw_cam_miou_percent"] = scale_quality["mean_iou_percent"]
            row[f"scale_{resolution}_semantic_foreground_precision_percent"] = scale_quality[
                "semantic_foreground_precision_percent"
            ]
            row[f"scale_{resolution}_semantic_foreground_recall_percent"] = scale_quality[
                "semantic_foreground_recall_percent"
            ]
            row[f"scale_{resolution}_background_false_positive_rate_percent"] = scale_quality[
                "background_false_positive_rate_percent"
            ]
    return row


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows = [collect(path.resolve()) for path in args.run_dir]
    modes = [row["normalization"] for row in rows]
    if len(set(modes)) != len(modes):
        raise ValueError(f"Duplicate normalization modes: {modes}")
    missing_modes = [mode for mode in CORE_MODES if mode not in modes]
    if args.require_all and missing_modes:
        raise ValueError(f"Missing required modes: {missing_modes}")
    rows.sort(key=lambda row: CORE_MODES.index(row["normalization"]))

    commits = {row["commit"] for row in rows}
    seeds = {row["seed"] for row in rows}
    thresholds = {row["background_threshold"] for row in rows}
    validity = {
        "all_runs_same_commit": len(commits) == 1,
        "all_runs_same_seed": len(seeds) == 1,
        "all_runs_same_threshold": len(thresholds) == 1,
        "unit_row_sum_modes_within_1e-4": all(
            row["maximum_attention_row_sum_error"] <= 1e-4
            for row in rows if row["normalization"] != "split_11"
        ),
        "split_11_row_sum_two_within_1e-4": all(
            row["maximum_attention_row_sum_error"] <= 1e-4
            for row in rows if row["normalization"] == "split_11"
        ),
        "integrated_vanilla_reproduces_69_50": all(
            abs(row["raw_cam_miou_percent"] - 69.50) <= 0.01
            for row in rows if row["normalization"] == "vanilla"
        ),
    }
    if not all(validity.values()):
        raise ValueError(f"Pilot validity checks failed: {validity}")

    args.output_dir.mkdir(parents=True)
    unavailable_metrics = ["FLOPs/MACs"]
    if not all("latency_ms_mean" in row for row in rows):
        unavailable_metrics.extend(["inference latency", "inference throughput"])
    payload = {
        "host": "MCTformer+",
        "dataset": "PASCAL VOC 2012 train",
        "core_mode_order": list(CORE_MODES),
        "missing_modes": missing_modes,
        "validity_checks": validity,
        "runs": rows,
        "unavailable_metrics": unavailable_metrics,
    }
    (args.output_dir / "pilot_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "pilot_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
