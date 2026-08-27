#!/usr/bin/env python3
"""Validate and summarize the six-run VOC BCSS minimum screen."""

import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("e0", "e1", "e2", "e4", "e5", "e6")
BACKGROUND_METRICS = {
    "e0": "background_cam_complement",
    "e1": "background_register_to_patch",
    "e2": "background_attention",
    "e4": "background_ownership",
    "e5": "background_ownership",
    "e6": "background_ownership",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def training_metrics(run_dir):
    logs = sorted((run_dir / "log_dir").glob("train-*.log"))
    if len(logs) != 1:
        raise RuntimeError(f"Expected one training log in {run_dir}, found {len(logs)}")
    records = []
    for line in logs[0].read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if "epoch" in record and "test_mAP" in record:
            records.append(record)
    if len(records) != 45 or records[-1]["epoch"] != 44:
        raise RuntimeError(f"Incomplete 45-epoch training log: {logs[0]}")
    best = max(records, key=lambda item: item["test_mAP"])
    return {
        "classification_final_map_percent": 100.0 * records[-1]["test_mAP"],
        "classification_best_map_percent": 100.0 * best["test_mAP"],
        "classification_best_epoch": best["epoch"],
    }


def metric_mean(payload, name):
    value = payload[name]
    return value["mean"] if isinstance(value, dict) else value


def collect(run_dir):
    required = (
        "config.json", "git_state.json", "checkpoint_manifest.txt", "pipeline.log",
        "raw_cam_diagnostics/metrics.json",
        "bcss_diagnostics/ownership_final_cam/metrics.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}: missing {missing}")
    if "PIPELINE_COMPLETE" not in (run_dir / "pipeline.log").read_text(encoding="utf-8"):
        raise RuntimeError(f"Pipeline has no completion marker: {run_dir}")
    config = load_json(run_dir / "config.json")
    variant = config["variant"]
    cam = load_json(run_dir / "raw_cam_diagnostics/metrics.json")
    ownership = load_json(
        run_dir / "bcss_diagnostics/ownership_final_cam/metrics.json")
    background = load_json(
        run_dir / "bcss_diagnostics" / BACKGROUND_METRICS[variant] / "metrics.json")
    row = {
        "variant": variant,
        "run_id": run_dir.name,
        "commit": load_json(run_dir / "git_state.json")["commit"],
        "seed": config["seed"],
        "tau": config["tau"],
        "beta": config["beta"],
        "raw_cam_miou_percent": cam["mean_iou_percent"],
        "semantic_foreground_precision_percent": cam["semantic_foreground_precision_percent"],
        "semantic_foreground_recall_percent": cam["semantic_foreground_recall_percent"],
        "binary_foreground_precision_percent": cam["binary_foreground_precision_percent"],
        "binary_foreground_recall_percent": cam["binary_foreground_recall_percent"],
        "cbl": metric_mean(ownership, "cbl"),
        "ccs_bg": metric_mean(ownership, "ccs_bg"),
        "ccs_fg": metric_mean(ownership, "ccs_fg"),
        "background_map": background["map_key"],
        "background_auprc": metric_mean(background, "background_auprc"),
        "background_iou": metric_mean(background, "background_iou"),
        "background_predicted_fraction": metric_mean(
            background, "predicted_background_fraction"),
        "background_entropy": metric_mean(background, "map_entropy"),
    }
    row.update(training_metrics(run_dir))
    counterfactual = run_dir / "bcss_diagnostics/counterfactual/metrics.json"
    if counterfactual.is_file():
        payload = load_json(counterfactual)
        row["crs"] = metric_mean(payload, "crs")
        row["ors"] = metric_mean(payload, "ors")
    return row


def relative_reduction(baseline, value):
    return (baseline - value) / baseline if baseline else None


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows = [collect(path.resolve()) for path in args.run_dir]
    by_variant = {row["variant"]: row for row in rows}
    if set(by_variant) != set(VARIANTS) or len(rows) != len(VARIANTS):
        raise RuntimeError(f"Expected exactly {VARIANTS}, got {tuple(by_variant)}")
    if len({row["commit"] for row in rows}) != 1 or len({row["seed"] for row in rows}) != 1:
        raise RuntimeError("BCSS screen is not matched by commit and seed")
    rows.sort(key=lambda row: VARIANTS.index(row["variant"]))
    baseline = by_variant["e0"]
    full = by_variant["e6"]
    register = by_variant["e1"]
    primary_checks = {
        "cam_miou_gain_at_least_1": full["raw_cam_miou_percent"] - baseline["raw_cam_miou_percent"] >= 1.0,
        "cbl_relative_reduction_at_least_15pct": relative_reduction(
            baseline["cbl"], full["cbl"]) >= 0.15,
        "ccs_bg_relative_reduction_at_least_20pct": relative_reduction(
            baseline["ccs_bg"], full["ccs_bg"]) >= 0.20,
        "semantic_precision_gain_at_least_2": (
            full["semantic_foreground_precision_percent"]
            - baseline["semantic_foreground_precision_percent"] >= 2.0),
        "semantic_recall_drop_above_minus_1_5": (
            full["semantic_foreground_recall_percent"]
            - baseline["semantic_foreground_recall_percent"] > -1.5),
        "classification_map_drop_above_minus_0_3": (
            full["classification_best_map_percent"]
            - baseline["classification_best_map_percent"] > -0.3),
    }
    register_checks = {
        "raw_cam_miou": full["raw_cam_miou_percent"] > register["raw_cam_miou_percent"],
        "cbl": full["cbl"] < register["cbl"],
        "background_auprc": full["background_auprc"] > register["background_auprc"],
    }
    if "crs" in full and "crs" in register:
        register_checks["crs"] = full["crs"] < register["crs"]
    payload = {
        "variants": rows,
        "primary_gate": {
            "checks": primary_checks,
            "passed": all(primary_checks.values()),
        },
        "register_interim_gate": {
            "checks": register_checks,
            "wins": sum(register_checks.values()),
            "available_metrics": len(register_checks),
            "passed_three_available": sum(register_checks.values()) >= 3,
            "pending_required_metrics": ["final_segmentation"] + (
                [] if "crs" in register_checks else ["crs"]),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "screen.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
