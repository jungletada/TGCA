#!/usr/bin/env python3
"""Create a coverage-aware scientific review of a completed Phase 0/1 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.semantic_relations import conservative_diagnostic_gates


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value):
    return float(value) if value else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics_path = args.analysis_dir / "metrics.json"
    sensitivity_path = args.analysis_dir / "phase1_threshold_sensitivity.csv"
    output = args.output or args.analysis_dir / "scientific_review.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metrics = json.loads(metrics_path.read_text())
    phase1 = metrics["phase1"]
    layers = phase1["layer_metrics"]
    bootstrap = phase1["bootstrap"]
    total_images = metrics["num_images"]
    best_pc_layer = max(
        layers,
        key=lambda row: row["pc_all_fg_accuracy"]
        if row["pc_all_fg_accuracy"] is not None else -1,
    )["layer"]
    best_pc_bootstrap = bootstrap[f"layer_{best_pc_layer}_pc_all_fg_accuracy"]
    required_region_images = max(30, (total_images + 19) // 20)
    region_candidates = []
    for layer in range(len(layers)):
        item = bootstrap[f"layer_{layer}_region_c_purity_difference"]
        if item["num_images"] >= required_region_images:
            region_candidates.append((layer, item))
    best_region = (
        max(region_candidates, key=lambda item: item[1]["mean"])
        if region_candidates else (None, {"ci95": [None, None], "num_images": 0})
    )
    maximum_recovery = max(
        row["macro_image_region_c_recovery_recall"] or 0.0 for row in layers
    )
    gates = conservative_diagnostic_gates(
        pc_accuracy_ci_lower=best_pc_bootstrap["ci95"][0],
        random_accuracy=phase1["uniform_20_class_random_accuracy"],
        maximum_recovery_recall=maximum_recovery,
        region_c_ci_lower=best_region[1]["ci95"][0],
        region_c_images=best_region[1]["num_images"],
        total_images=total_images,
    )
    sensitivity = []
    with sensitivity_path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["layer"]) == best_pc_layer:
                sensitivity.append({
                    "semantic_threshold": float(row["semantic_threshold"]),
                    "pc_all_fg_precision": optional_float(row["pc_all_fg_precision"]),
                    "pc_all_fg_recall": optional_float(row["pc_all_fg_recall"]),
                    "region_c_target_purity": optional_float(
                        row["region_c_target_purity"]
                    ),
                    "region_c_recovery_recall": optional_float(
                        row["region_c_recovery_recall"]
                    ),
                })
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    best_row = layers[best_pc_layer]
    review = {
        "format_version": 1,
        "run_id": metrics["run_id"],
        "review_commit": head,
        "source_metrics": str(metrics_path.resolve()),
        "source_metrics_sha256": sha256(metrics_path),
        "source_sensitivity": str(sensitivity_path.resolve()),
        "source_sensitivity_sha256": sha256(sensitivity_path),
        "corrected_conservative_gates": gates,
        "original_mechanical_gates": phase1["gates"],
        "strict_pc_all": {
            "best_layer_zero_based": best_pc_layer,
            "patch_weighted_accuracy": best_row["pc_all_fg_accuracy"],
            "foreground_restricted_miou": best_row["pc_all_fg_miou"],
            "macro_image_accuracy": best_pc_bootstrap["mean"],
            "macro_image_accuracy_ci95": best_pc_bootstrap["ci95"],
            "uniform_random_accuracy": phase1["uniform_20_class_random_accuracy"],
        },
        "primary_region_c": {
            "minimum_supported_images": required_region_images,
            "best_supported_layer_zero_based": best_region[0],
            "maximum_macro_recovery_recall": maximum_recovery,
            "interpretation": "insufficient primary-threshold coverage and recovery",
        },
        "best_strict_layer_threshold_sensitivity": sensitivity,
        "decision": {
            "intrinsic_patch_to_class_semantics": "go",
            "primary_threshold_region_c_complementarity": "no_go",
            "phase2_status": "reasonable_to_predeclare_but_not_implemented",
        },
    }
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    print(json.dumps(review["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
