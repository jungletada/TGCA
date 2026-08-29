#!/usr/bin/env python3
"""Validate and summarize the seed-0 Persistent Semantic Phase 2 screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


VARIANTS = ("baseline", "read_only", "write_only", "read_write")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training(run_dir):
    logs = sorted((run_dir / "log_dir").glob("train-*.log"))
    if len(logs) != 1:
        raise ValueError(f"Expected one training log in {run_dir}, found {len(logs)}")
    epochs = []
    for line in logs[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            if "epoch" in record and "test_mAP" in record:
                epochs.append(record)
    if len(epochs) != 45 or epochs[-1]["epoch"] != 44:
        raise ValueError(f"Incomplete training log: {logs[0]}")
    return {
        "classification_final_map_percent": 100.0 * epochs[-1]["test_mAP"],
        "classification_max_map_percent": 100.0 * max(
            item["test_mAP"] for item in epochs),
        "classification_max_map_epoch": max(
            epochs, key=lambda item: item["test_mAP"])["epoch"],
        "training_log": str(logs[0].resolve()),
    }


def parse_peak_memory(run_dir):
    text = (run_dir / "pipeline.log").read_text(encoding="utf-8")
    values = [int(value) for value in re.findall(r"max mem: ([0-9]+)", text)]
    return max(values) if values else None


def collect_baseline(run_dir):
    raw = load_json(run_dir / "raw_cam_diagnostics" / "metrics.json")
    row = {
        "variant": "baseline",
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "commit": load_json(run_dir / "git_state.json")["commit"],
        "seed": 0,
        "raw_cam_miou_percent": raw["mean_iou_percent"],
        "semantic_foreground_precision_percent": raw[
            "semantic_foreground_precision_percent"],
        "semantic_foreground_recall_percent": raw[
            "semantic_foreground_recall_percent"],
        "background_false_positive_rate_percent": raw[
            "background_false_positive_rate_percent"],
        "checkpoint_sha256": file_sha256(run_dir / "mctformerplus_final.pth"),
        "peak_training_gpu_memory_mb": parse_peak_memory(run_dir),
    }
    row.update(parse_training(run_dir))
    return row


def collect_variant(run_dir):
    required = (
        "completion.json", "config.json", "git_state.json",
        "checkpoint_manifest.txt", "raw_cam_diagnostics/metrics.json",
        "relation_diagnostics/metrics.json", "relation_diagnostics/completion.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Phase 2 run {run_dir}: {missing}")
    config = load_json(run_dir / "config.json")
    raw = load_json(run_dir / "raw_cam_diagnostics" / "metrics.json")
    relation = load_json(run_dir / "relation_diagnostics" / "metrics.json")
    checkpoint = run_dir / "mctformerplus_final.pth"
    manifest_hash = (run_dir / "checkpoint_manifest.txt").read_text().split()[0]
    if manifest_hash != file_sha256(checkpoint):
        raise ValueError(f"Checkpoint hash mismatch in {run_dir}")
    if relation["checkpoint_sha256"] != manifest_hash:
        raise ValueError(f"Relation diagnostic checkpoint mismatch in {run_dir}")
    if config["variant"] != relation["variant"]:
        raise ValueError(f"Variant provenance mismatch in {run_dir}")
    if raw["background_threshold"] != config["fixed_background_threshold"]:
        raise ValueError(f"Threshold provenance mismatch in {run_dir}")
    completion = load_json(run_dir / "completion.json")
    if not completion.get("complete"):
        raise ValueError(f"Incomplete marker in {run_dir}")
    final_layer = max(relation["layer_metrics"], key=lambda item: item["layer"])
    row = {
        "variant": config["variant"],
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "commit": load_json(run_dir / "git_state.json")["commit"],
        "seed": config["seed"],
        "raw_cam_miou_percent": raw["mean_iou_percent"],
        "semantic_foreground_precision_percent": raw[
            "semantic_foreground_precision_percent"],
        "semantic_foreground_recall_percent": raw[
            "semantic_foreground_recall_percent"],
        "background_false_positive_rate_percent": raw[
            "background_false_positive_rate_percent"],
        "checkpoint_sha256": manifest_hash,
        "total_parameters": relation["total_parameters"],
        "semantic_interaction_parameters": relation[
            "semantic_interaction_parameters"],
        "relation_diagnostic_classification_map_percent": 100.0 * relation[
            "classification_map"],
        "relation_foreground_accuracy_percent": 100.0 * final_layer[
            "foreground_accuracy"],
        "relation_background_accuracy_percent": 100.0 * final_layer[
            "background_accuracy"],
        "relation_mean_iou_percent": 100.0 * final_layer["mean_iou"],
        "write_background_mass_percent": 100.0 * final_layer[
            "write_background_mass"],
        "write_gate_final": final_layer["write_gate"],
        "foreground_token_gap": relation["foreground_token_gap"],
        "background_token_gap": relation["background_token_gap"],
        "peak_training_gpu_memory_mb": parse_peak_memory(run_dir),
    }
    row.update(parse_training(run_dir))
    return row


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows = [collect_baseline(args.baseline_run_dir.resolve())]
    rows.extend(collect_variant(path.resolve()) for path in args.run_dir)
    by_variant = {row["variant"]: row for row in rows}
    if set(by_variant) != set(VARIANTS) or len(by_variant) != len(rows):
        raise ValueError(f"Expected exactly {VARIANTS}, got {tuple(by_variant)}")
    rows.sort(key=lambda row: VARIANTS.index(row["variant"]))
    commits = {row["commit"] for row in rows if row["variant"] != "baseline"}
    seeds = {row["seed"] for row in rows}
    if len(commits) != 1 or seeds != {0}:
        raise ValueError(f"Unmatched screen provenance: commits={commits}, seeds={seeds}")

    baseline = by_variant["baseline"]
    read_only = by_variant["read_only"]
    read_write = by_variant["read_write"]
    for row in rows:
        row["cam_delta_vs_baseline"] = (
            row["raw_cam_miou_percent"] - baseline["raw_cam_miou_percent"])
        row["classification_delta_vs_baseline"] = (
            row["classification_final_map_percent"]
            - baseline["classification_final_map_percent"])
    gates = {
        "read_write_cam_within_one_point_of_baseline": (
            read_write["cam_delta_vs_baseline"] >= -1.0),
        "read_write_classification_within_one_point_of_baseline": (
            read_write["classification_delta_vs_baseline"] >= -1.0),
        "read_write_exceeds_read_only_by_half_point": (
            read_write["raw_cam_miou_percent"]
            >= read_only["raw_cam_miou_percent"] + 0.5),
        "read_write_exceeds_baseline_by_half_point": (
            read_write["cam_delta_vs_baseline"] >= 0.5),
    }
    retained = (
        gates["read_write_cam_within_one_point_of_baseline"]
        and gates["read_write_classification_within_one_point_of_baseline"])
    positive_feedback = gates["read_write_exceeds_read_only_by_half_point"]
    if retained and gates["read_write_exceeds_baseline_by_half_point"]:
        decision = "strong_go"
    elif retained and positive_feedback:
        decision = "conditional_go_for_targeted_phase2_ablations"
    else:
        decision = "no_go_for_phase2_expansion"
    payload = {
        "phase": 2,
        "host": "MCTformer+",
        "dataset": "PASCAL VOC 2012",
        "variants": rows,
        "gates": gates,
        "decision": decision,
        "limitations": [
            "seed-0 screen only",
            "no paired bootstrap for class-aggregated CAM mIoU",
            "background latent receives no auxiliary pixel supervision",
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "comparison.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
