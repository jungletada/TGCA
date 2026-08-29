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
    confirmatory_layer = phase1.get("confirmatory_layer_zero_based")
    selected_pc_layer = (
        int(confirmatory_layer)
        if confirmatory_layer is not None
        else max(
            layers,
            key=lambda row: row["pc_all_fg_accuracy"]
            if row["pc_all_fg_accuracy"] is not None else -1,
        )["layer"]
    )
    selected_pc_bootstrap = bootstrap[
        f"layer_{selected_pc_layer}_pc_all_fg_accuracy"
    ]
    required_region_images = max(30, (total_images + 19) // 20)
    region_candidates = []
    region_layers = (
        [selected_pc_layer] if confirmatory_layer is not None else range(len(layers))
    )
    for layer in region_layers:
        item = bootstrap[f"layer_{layer}_region_c_purity_difference"]
        if item["num_images"] >= required_region_images:
            region_candidates.append((layer, item))
    best_region = (
        max(region_candidates, key=lambda item: item[1]["mean"])
        if region_candidates else (None, {"ci95": [None, None], "num_images": 0})
    )
    maximum_recovery = max(
        row["macro_image_region_c_recovery_recall"] or 0.0
        for row in layers
        if row["layer"] in region_layers
    )
    gates = conservative_diagnostic_gates(
        pc_accuracy_ci_lower=selected_pc_bootstrap["ci95"][0],
        random_accuracy=phase1["uniform_20_class_random_accuracy"],
        maximum_recovery_recall=maximum_recovery,
        region_c_ci_lower=best_region[1]["ci95"][0],
        region_c_images=best_region[1]["num_images"],
        total_images=total_images,
    )
    majority_accuracy = phase1.get("foreground_majority_class_accuracy")
    majority_control = phase1.get("foreground_majority_class_control") or {}
    majority_advantage_ci = majority_control.get(
        "pc_all_minus_majority_macro_image_accuracy_ci95", [None, None]
    )
    permutation_control = phase1.get("class_identity_permutation_control") or {}
    gates["pc_all_above_foreground_majority_class"] = bool(
        majority_advantage_ci[0] is not None and majority_advantage_ci[0] > 0
    )
    gates["class_identity_permutation_p_below_0_01"] = bool(
        permutation_control.get("empirical_p_greater_equal", 1.0) < 0.01
    )
    semantic_confirmed = all((
        gates["pc_all_above_uniform_random"],
        gates["pc_all_above_foreground_majority_class"],
        gates["class_identity_permutation_p_below_0_01"],
    ))
    region_c_confirmed = gates["region_c_enriched_over_cp_low_reference"]
    sensitivity = []
    with sensitivity_path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["layer"]) == selected_pc_layer:
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
    selected_row = layers[selected_pc_layer]
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
            "selection_mode": phase1.get("selection_mode", "exploratory_best_layer"),
            "selected_layer_zero_based": selected_pc_layer,
            "patch_weighted_accuracy": selected_row["pc_all_fg_accuracy"],
            "foreground_restricted_miou": selected_row["pc_all_fg_miou"],
            "macro_image_accuracy": selected_pc_bootstrap["mean"],
            "macro_image_accuracy_ci95": selected_pc_bootstrap["ci95"],
            "uniform_random_accuracy": phase1["uniform_20_class_random_accuracy"],
            "foreground_majority_class_accuracy": phase1.get(
                "foreground_majority_class_accuracy"
            ),
            "foreground_majority_class_control": majority_control,
            "class_identity_permutation_control": phase1.get(
                "class_identity_permutation_control"
            ),
        },
        "primary_region_c": {
            "minimum_supported_images": required_region_images,
            "best_supported_layer_zero_based": best_region[0],
            "maximum_macro_recovery_recall": maximum_recovery,
            "interpretation": (
                "confirmed" if region_c_confirmed
                else "insufficient primary-threshold coverage, enrichment, or recovery"
            ),
        },
        "best_strict_layer_threshold_sensitivity": sensitivity,
        "decision": {
            "intrinsic_patch_to_class_semantics": (
                "go" if semantic_confirmed else "no_go"
            ),
            "primary_threshold_region_c_complementarity": (
                "go" if region_c_confirmed else "no_go"
            ),
            "phase2_status": "reasonable_to_predeclare_but_not_implemented",
        },
    }
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    print(json.dumps(review["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
