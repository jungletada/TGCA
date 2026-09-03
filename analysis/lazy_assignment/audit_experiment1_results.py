#!/usr/bin/env python3
"""Read-only integrity audit for completed Experiment 1 result roots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import (
    AnalysisLog,
    VOC_CLASS_NAMES,
    assert_output_outside_sources,
    json_dump,
    resolve_completed_result_root,
    sha256_file,
    timestamp,
)
from datasets_cam import load_image_label_list_from_npy_voc, load_img_name_list


SCHEMA_CANDIDATES = {
    "image_id": ("image_id", "name", "image_name"),
    "positive_class_ids": ("positive_class_ids", "positive_ids", "class_ids"),
    "scores_raw": ("scores_raw", "scores", "class_patch_scores"),
    "grid_h": ("grid_h", "height", "patch_grid_h"),
    "grid_w": ("grid_w", "width", "patch_grid_w"),
}
ISSUE_COLUMNS = ("model", "image_id", "severity", "issue", "details", "path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctformer-results", type=Path, required=True)
    parser.add_argument("--mctformer-plus-results", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def _kind(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.parts and relative.parts[0] == "scores" and path.suffix == ".npz":
        return "score_npz"
    if relative.parts and relative.parts[0] == "visualizations":
        return "visualization"
    if path.name in {"metadata.json", "manifest.jsonl", "completion.json"}:
        return path.stem
    if path.name == "summary_by_layer.csv":
        return "summary"
    if path.suffix in {".log", ".txt"}:
        return "log_or_provenance"
    return "other"


def inventory_files(model: str, root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "model": model,
                "result_root": str(root),
                "relative_path": str(path.relative_to(root)),
                "absolute_path": str(path),
                "kind": _kind(path, root),
                "size_bytes": int(stat.st_size),
                "mtime_ns_before": int(stat.st_mtime_ns),
                "sha256_before": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def schema_mapping(keys: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, candidates in SCHEMA_CANDIDATES.items():
        matches = [candidate for candidate in candidates if candidate in keys]
        if len(matches) != 1:
            raise KeyError(
                f"expected one field for {canonical}; keys={keys}, candidates={candidates}"
            )
        mapping[canonical] = matches[0]
    return mapping


def _issue(
    issues: list[dict[str, object]],
    model: str,
    image_id: str,
    issue: str,
    details: object,
    path: Path | str,
    severity: str = "error",
) -> None:
    issues.append(
        {
            "model": model,
            "image_id": image_id,
            "severity": severity,
            "issue": issue,
            "details": str(details),
            "path": str(path),
        }
    )


def audit_model(
    model: str,
    root: Path,
    expected_ids: list[str],
    expected_labels: list[np.ndarray],
    tolerance: float,
    issues: list[dict[str, object]],
) -> tuple[dict[str, Any], set[tuple[str, int]], dict[str, str]]:
    required = ("metadata.json", "manifest.jsonl", "summary_by_layer.csv")
    existence = {name: (root / name).is_file() for name in required}
    for name, present in existence.items():
        if not present:
            _issue(issues, model, "", "missing_required_file", name, root)

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    completion_path = root / "completion.json"
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.is_file()
        else {}
    )
    manifest_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            manifest_records.append(json.loads(line))
        except json.JSONDecodeError as error:
            _issue(
                issues,
                model,
                "",
                "invalid_manifest_json",
                f"line {line_number}: {error}",
                root / "manifest.jsonl",
            )

    expected_set = set(expected_ids)
    manifest_ids = [str(record.get("image_id", "")) for record in manifest_records]
    manifest_paths = [str(record.get("score_path", "")) for record in manifest_records]
    id_counts = Counter(manifest_ids)
    path_counts = Counter(manifest_paths)
    duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
    duplicate_paths = sorted(key for key, count in path_counts.items() if count > 1)
    result_set = set(manifest_ids)
    missing_ids = sorted(expected_set - result_set)
    extra_ids = sorted(result_set - expected_set)
    for image_id in duplicate_ids:
        _issue(issues, model, image_id, "duplicate_manifest_image_id", id_counts[image_id], root)
    for path in duplicate_paths:
        _issue(issues, model, "", "duplicate_manifest_score_path", path_counts[path], path)
    for image_id in missing_ids:
        _issue(issues, model, image_id, "missing_image_id", "absent from manifest", root)
    for image_id in extra_ids:
        _issue(issues, model, image_id, "extra_image_id", "not in val list", root)

    score_files = sorted((root / "scores").glob("*.npz"))
    manifest_absolute = {
        (root / record["score_path"]).resolve()
        for record in manifest_records
        if record.get("score_path")
    }
    score_absolute = {path.resolve() for path in score_files}
    manifest_without_file = sorted(str(path) for path in manifest_absolute - score_absolute)
    files_without_manifest = sorted(str(path) for path in score_absolute - manifest_absolute)
    for path in manifest_without_file:
        _issue(issues, model, "", "manifest_without_score_file", "", path)
    for path in files_without_manifest:
        _issue(issues, model, "", "score_file_without_manifest", "", path)

    label_by_id = {image_id: np.asarray(label) for image_id, label in zip(expected_ids, expected_labels)}
    mappings: list[dict[str, str]] = []
    pair_set: set[tuple[str, int]] = set()
    exact_label_matches = 0
    label_mismatches = 0
    missing_class_count = 0
    extra_class_count = 0
    nan_count = 0
    inf_count = 0
    below_range_count = 0
    above_range_count = 0
    small_overshoot_count = 0
    layer_counts: Counter[int] = Counter()
    grids: Counter[tuple[int, int]] = Counter()
    dtypes: Counter[str] = Counter()
    valid_files = 0

    for record in manifest_records:
        image_id = str(record.get("image_id", ""))
        score_path = (root / str(record.get("score_path", ""))).resolve()
        if not score_path.is_file():
            continue
        try:
            with np.load(score_path, allow_pickle=False) as artifact:
                keys = list(artifact.files)
                mapping = schema_mapping(keys)
                mappings.append(mapping)
                stored_image_id = str(artifact[mapping["image_id"]].item())
                positive_ids = np.asarray(
                    artifact[mapping["positive_class_ids"]], dtype=np.int64
                ).reshape(-1)
                scores = np.asarray(artifact[mapping["scores_raw"]])
                grid_h = int(artifact[mapping["grid_h"]].item())
                grid_w = int(artifact[mapping["grid_w"]].item())
        except Exception as error:
            _issue(issues, model, image_id, "npz_read_or_schema_error", repr(error), score_path)
            continue

        if stored_image_id != image_id:
            _issue(
                issues,
                model,
                image_id,
                "stored_image_id_mismatch",
                stored_image_id,
                score_path,
            )
        if scores.ndim != 3:
            _issue(issues, model, image_id, "invalid_score_rank", scores.shape, score_path)
            continue
        num_layers, num_positive, num_patches = scores.shape
        layer_counts[int(num_layers)] += 1
        grids[(grid_h, grid_w)] += 1
        dtypes[str(scores.dtype)] += 1
        if num_positive != len(positive_ids):
            _issue(
                issues,
                model,
                image_id,
                "positive_class_axis_mismatch",
                f"shape={scores.shape}, ids={positive_ids.tolist()}",
                score_path,
            )
        if grid_h * grid_w != num_patches:
            _issue(
                issues,
                model,
                image_id,
                "grid_patch_mismatch",
                f"{grid_h}x{grid_w} != {num_patches}",
                score_path,
            )
        if not len(positive_ids):
            _issue(issues, model, image_id, "empty_positive_class_ids", "", score_path)
        if len(set(positive_ids.tolist())) != len(positive_ids):
            _issue(
                issues, model, image_id, "duplicate_positive_class_ids", positive_ids, score_path
            )
        if np.any((positive_ids < 0) | (positive_ids >= len(VOC_CLASS_NAMES))):
            _issue(
                issues, model, image_id, "class_id_out_of_range", positive_ids, score_path
            )
        local_nan = int(np.isnan(scores).sum())
        local_inf = int(np.isinf(scores).sum())
        nan_count += local_nan
        inf_count += local_inf
        if local_nan:
            _issue(issues, model, image_id, "nan_scores", local_nan, score_path)
        if local_inf:
            _issue(issues, model, image_id, "inf_scores", local_inf, score_path)
        finite = scores[np.isfinite(scores)].astype(np.float64, copy=False)
        if finite.size:
            low = finite < -1.0
            high = finite > 1.0
            low_invalid = finite < -1.0 - tolerance
            high_invalid = finite > 1.0 + tolerance
            below_range_count += int(low_invalid.sum())
            above_range_count += int(high_invalid.sum())
            small_overshoot_count += int((low & ~low_invalid).sum() + (high & ~high_invalid).sum())
            if low_invalid.any() or high_invalid.any():
                _issue(
                    issues,
                    model,
                    image_id,
                    "cosine_range_violation",
                    f"min={finite.min()}, max={finite.max()}",
                    score_path,
                )

        expected_positive = (
            np.flatnonzero(label_by_id[image_id] > 0).astype(np.int64)
            if image_id in label_by_id
            else np.asarray([], dtype=np.int64)
        )
        missing_classes = sorted(set(expected_positive.tolist()) - set(positive_ids.tolist()))
        extra_classes = sorted(set(positive_ids.tolist()) - set(expected_positive.tolist()))
        if not missing_classes and not extra_classes:
            exact_label_matches += 1
        else:
            label_mismatches += 1
            missing_class_count += len(missing_classes)
            extra_class_count += len(extra_classes)
            _issue(
                issues,
                model,
                image_id,
                "positive_class_label_mismatch",
                f"missing={missing_classes}, extra={extra_classes}",
                score_path,
            )
        pair_set.update((image_id, int(class_id)) for class_id in positive_ids)
        valid_files += 1

    mapping_set = {json.dumps(mapping, sort_keys=True) for mapping in mappings}
    if len(mapping_set) != 1:
        _issue(issues, model, "", "inconsistent_npz_schema_mapping", mapping_set, root)
    mapping = json.loads(next(iter(mapping_set))) if mapping_set else {}

    visualizations = sorted((root / "visualizations").glob("*")) if (root / "visualizations").is_dir() else []
    failure_files = sorted(root.glob("failure*.json"))
    result = {
        "result_root": str(root),
        "required_file_exists": existence,
        "completion_status": completion.get("status"),
        "run_kind": completion.get("run_kind"),
        "expected_images": len(expected_ids),
        "manifest_rows": len(manifest_records),
        "unique_manifest_images": len(result_set),
        "score_files": len(score_files),
        "valid_score_files": valid_files,
        "visualization_files": len(visualizations),
        "failure_files": [str(path) for path in failure_files],
        "missing_image_ids": missing_ids,
        "extra_image_ids": extra_ids,
        "duplicate_image_ids": duplicate_ids,
        "duplicate_score_paths": duplicate_paths,
        "manifest_without_score_file": manifest_without_file,
        "score_file_without_manifest": files_without_manifest,
        "positive_label_exact_match_count": exact_label_matches,
        "positive_label_mismatch_count": label_mismatches,
        "missing_class_count": missing_class_count,
        "extra_class_count": extra_class_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "cosine_below_tolerance_count": below_range_count,
        "cosine_above_tolerance_count": above_range_count,
        "floating_point_overshoot_count": small_overshoot_count,
        "layer_count_distribution": {str(key): value for key, value in sorted(layer_counts.items())},
        "grid_distribution": {f"{key[0]}x{key[1]}": value for key, value in sorted(grids.items())},
        "dtype_distribution": dict(sorted(dtypes.items())),
        "image_class_pairs": len(pair_set),
        "checkpoint_sha256": metadata.get("checkpoint", {}).get("sha256"),
        "source_git_commit": metadata.get("git", {}).get("commit"),
        "metadata": metadata,
    }
    return result, pair_set, mapping


def compare_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_meta = left["metadata"]
    right_meta = right["metadata"]
    comparisons = {
        "dataset_split": (
            left_meta["dataset"].get("split"), right_meta["dataset"].get("split")
        ),
        "val_list_sha256": (
            left_meta["dataset"].get("list_sha256"),
            right_meta["dataset"].get("list_sha256"),
        ),
        "input_size": (left_meta["input"].get("size"), right_meta["input"].get("size")),
        "transform": (
            left_meta["input"].get("transform"), right_meta["input"].get("transform")
        ),
        "patch_size": (
            left_meta["model"].get("patch_size"), right_meta["model"].get("patch_size")
        ),
        "num_layers": (
            left_meta["model"].get("depth"), right_meta["model"].get("depth")
        ),
        "embedding_dimension": (
            left_meta["model"].get("embed_dim"), right_meta["model"].get("embed_dim")
        ),
        "score_definition": (left_meta.get("score"), right_meta.get("score")),
        "representation_point": (
            left_meta.get("representation"), right_meta.get("representation")
        ),
        "layer_indexing": (
            left_meta.get("layer_indexing"), right_meta.get("layer_indexing")
        ),
        "positive_class_filter": (
            left_meta.get("positive_class_filter"), right_meta.get("positive_class_filter")
        ),
        "checkpoint_sha256": (
            left_meta["checkpoint"].get("sha256"),
            right_meta["checkpoint"].get("sha256"),
        ),
        "git_commit": (
            left_meta["git"].get("commit"), right_meta["git"].get("commit")
        ),
    }
    required_equal = {
        "dataset_split",
        "val_list_sha256",
        "input_size",
        "transform",
        "patch_size",
        "num_layers",
        "score_definition",
        "representation_point",
        "layer_indexing",
        "positive_class_filter",
    }
    return {
        key: {
            "mctformer": value[0],
            "mctformer_plus": value[1],
            "equal": value[0] == value[1],
            "required_for_paired_comparison": key in required_equal,
        }
        for key, value in comparisons.items()
    }


def verify_inventory_unchanged(frame: pd.DataFrame) -> tuple[bool, list[dict[str, object]]]:
    changes: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        path = Path(row.absolute_path)
        if not path.is_file():
            changes.append({"path": str(path), "issue": "missing_after_audit"})
            continue
        current_hash = sha256_file(path)
        current_stat = path.stat()
        if current_hash != row.sha256_before or current_stat.st_size != row.size_bytes:
            changes.append(
                {
                    "path": str(path),
                    "issue": "content_or_size_changed",
                    "sha256_before": row.sha256_before,
                    "sha256_after": current_hash,
                    "size_before": int(row.size_bytes),
                    "size_after": int(current_stat.st_size),
                }
            )
    return not changes, changes


def inventory_markdown(
    report: dict[str, Any], inventory: pd.DataFrame, issues: pd.DataFrame
) -> str:
    lines = [
        "# Experiment 1 Result Inventory",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Both source result directories were opened read-only. All derived files are outside the source roots.",
        "",
    ]
    for model in ("mctformer", "mctformer_plus"):
        item = report["models"][model]
        model_files = inventory[inventory["model"] == model]
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Result root: `{item['result_root']}`",
                f"- Manifest/score files: {item['manifest_rows']} / {item['score_files']}",
                f"- Unique images: {item['unique_manifest_images']} / {item['expected_images']}",
                f"- Image–class pairs: {item['image_class_pairs']}",
                f"- Visualizations: {item['visualization_files']}",
                f"- Total bytes: {int(model_files['size_bytes'].sum())}",
                f"- NaN/Inf: {item['nan_count']} / {item['inf_count']}",
                f"- Positive-label mismatches: {item['positive_label_mismatch_count']}",
                f"- Checkpoint SHA256: `{item['checkpoint_sha256']}`",
                f"- Source Git commit: `{item['source_git_commit']}`",
                "",
            ]
        )
    common = report["common_pairs"]
    lines.extend(
        [
            "## Cross-model alignment",
            "",
            f"- Common images: {common['common_images']}",
            f"- Common image–class pairs: {common['common_image_class_pairs']}",
            f"- MCTformer-only pairs: {common['mctformer_only_pairs']}",
            f"- MCTformer+-only pairs: {common['mctformer_plus_only_pairs']}",
            f"- Common-pair coverage: {common['common_pair_coverage_percent']:.4f}%",
            "",
            "## Audit outcome",
            "",
            f"- Integrity passed: **{report['integrity_passed']}**",
            f"- Recorded issues: {len(issues)}",
            f"- Source hashes unchanged during audit: **{report['source_hashes_unchanged_during_audit']}**",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    mct_root, mct_discovery = resolve_completed_result_root(
        args.mctformer_results, "mctformer"
    )
    plus_root, plus_discovery = resolve_completed_result_root(
        args.mctformer_plus_results, "mctformer_plus"
    )
    output = args.output_dir.resolve()
    assert_output_outside_sources(output, (mct_root, plus_root))
    output.mkdir(parents=True, exist_ok=False)
    log = AnalysisLog(output / "audit.log")
    log(f"MCTformer source: {mct_root}")
    log(f"MCTformer+ source: {plus_root}")

    expected_ids = load_img_name_list(str(args.val_list.resolve()))
    expected_labels = load_image_label_list_from_npy_voc(
        str(args.voc_root.resolve()), expected_ids
    )
    inventory = pd.concat(
        (
            inventory_files("mctformer", mct_root),
            inventory_files("mctformer_plus", plus_root),
        ),
        ignore_index=True,
    ).sort_values(["model", "relative_path"], ignore_index=True)
    inventory.to_csv(output / "file_manifest.csv", index=False)
    log(f"Hashed {len(inventory)} source files before analysis")

    issues: list[dict[str, object]] = []
    mct, mct_pairs, mct_mapping = audit_model(
        "mctformer",
        mct_root,
        expected_ids,
        expected_labels,
        args.cosine_tolerance,
        issues,
    )
    plus, plus_pairs, plus_mapping = audit_model(
        "mctformer_plus",
        plus_root,
        expected_ids,
        expected_labels,
        args.cosine_tolerance,
        issues,
    )
    common_pairs = mct_pairs & plus_pairs
    union_pairs = mct_pairs | plus_pairs
    common_images = {image_id for image_id, _ in common_pairs}
    metadata_comparison = compare_metadata(mct, plus)
    required_metadata_match = all(
        entry["equal"]
        for entry in metadata_comparison.values()
        if entry["required_for_paired_comparison"]
    )

    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    issue_frame.to_csv(output / "missing_or_invalid_samples.csv", index=False)
    unchanged, changes = verify_inventory_unchanged(inventory)
    error_count = int((issue_frame["severity"] == "error").sum()) if len(issue_frame) else 0
    report: dict[str, Any] = {
        "generated_at": timestamp(),
        "cosine_tolerance": args.cosine_tolerance,
        "expected_val_images": len(expected_ids),
        "voc_root": str(args.voc_root.resolve()),
        "val_list": str(args.val_list.resolve()),
        "models": {"mctformer": mct, "mctformer_plus": plus},
        "discovery": {
            "mctformer": mct_discovery,
            "mctformer_plus": plus_discovery,
        },
        "schema_mapping": {
            "mctformer": mct_mapping,
            "mctformer_plus": plus_mapping,
        },
        "common_pairs": {
            "common_images": len(common_images),
            "common_image_class_pairs": len(common_pairs),
            "mctformer_only_pairs": len(mct_pairs - plus_pairs),
            "mctformer_plus_only_pairs": len(plus_pairs - mct_pairs),
            "common_pair_coverage_percent": (
                100.0 * len(common_pairs) / len(union_pairs) if union_pairs else 0.0
            ),
        },
        "metadata_comparison": metadata_comparison,
        "required_metadata_match": required_metadata_match,
        "issue_count": len(issue_frame),
        "error_count": error_count,
        "source_hashes_unchanged_during_audit": unchanged,
        "source_hash_changes_during_audit": changes,
    }
    report["integrity_passed"] = bool(
        error_count == 0
        and unchanged
        and required_metadata_match
        and len(common_pairs) / max(1, len(union_pairs)) >= 0.99
    )
    json_dump(output / "schema_mapping.json", report["schema_mapping"])
    json_dump(output / "integrity_report.json", report)
    (output / "RESULT_INVENTORY.md").write_text(
        inventory_markdown(report, inventory, issue_frame), encoding="utf-8"
    )
    log(
        f"Audit complete: integrity_passed={report['integrity_passed']}, "
        f"issues={len(issue_frame)}, common_pairs={len(common_pairs)}"
    )
    return report


def main() -> None:
    report = run_audit(parse_args())
    if not report["integrity_passed"]:
        raise SystemExit("Experiment 1 integrity audit failed; inspect audit outputs")


if __name__ == "__main__":
    main()

