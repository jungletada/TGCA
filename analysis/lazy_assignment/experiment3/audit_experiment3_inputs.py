#!/usr/bin/env python3
"""Create the immutable-input and Experiment 2 linkage audit for Experiment 3."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_LAYERS,
    EXPECTED_MULTILABEL_IMAGES,
    EXPECTED_PATCHES,
    EXPECTED_POSITIVE_PAIRS,
    MODEL_KEYS,
    assert_new_output,
    git_state,
    json_dump,
    load_image_labels,
    ordered_val_ids,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)


MANIFEST_FIELDS = (
    "source_group",
    "model",
    "root",
    "relative_path",
    "absolute_path",
    "size_bytes",
    "mtime_ns",
    "sha256",
)

EXPECTED_CANONICAL_TABLE_ROWS = {
    "per_class_token_pair_layer": 21_744,
    "per_image_cam_confusion": 1_278_018,
    "per_image_class_cam_stage": 60_116,
    "per_image_class_layer_signal": 881_452,
    "per_image_class_stage_transition": 1_004_796,
    "per_image_classification": 57_960,
    "per_multilabel_class_pair_layer_signal": 472_932,
    "per_shared_patch_ownership": 1_021_968,
    "source_index": 2_898,
}

EXPECTED_SIGNAL_KEYS = {
    "image_id",
    "positive_class_ids",
    "grid_h",
    "grid_w",
    "patch_label_counts",
    "region_masks_rho05",
    "region_masks_rho07",
    "feature_post_scores",
    "feature_norm_scores",
    "feature_final_norm_scores",
    "qk_mean_scores",
    "qk_head_std",
    "attn_c2p_raw",
    "attn_c2p_conditional",
    "attn_patch_mass",
    "patch_logits",
    "patch_cam",
    "attn_official_raw",
    "attn_official_conditional",
    "attn_mid3_raw",
    "attn_mid3_conditional",
    "c2p_cam",
    "final_cam",
    "class_logits",
    "patch_class_logits",
    "class_logits_all",
    "patch_class_logits_all",
    "raw_final_cam_confusion_t045",
    "class_token_pairwise_cosine",
    "patch_norms",
    "qk_head_region_mean_rho05",
    "qk_head_region_mean_rho07",
    "diagnostic_c2p_cam_l10",
    "diagnostic_c2p_cam_l11",
    "diagnostic_c2p_cam_l12",
    "diagnostic_c2p_cam_mid3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    parser.add_argument(
        "--expected-positive-pairs", type=int, default=EXPECTED_POSITIVE_PAIRS
    )
    parser.add_argument(
        "--expected-multilabel-images",
        type=int,
        default=EXPECTED_MULTILABEL_IMAGES,
    )
    return parser.parse_args()


def _manifest_record(
    group: str, model: str, root: Path, path: Path
) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_group": group,
        "model": model,
        "root": str(root),
        "relative_path": str(path.relative_to(root)) if path != root else path.name,
        "absolute_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _tree_records(group: str, model: str, root: Path) -> Iterable[dict[str, object]]:
    for path in sorted(value.resolve() for value in root.rglob("*") if value.is_file()):
        yield _manifest_record(group, model, root, path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows


def _expected_signal_shapes(num_positive: int) -> dict[str, tuple[int, ...]]:
    p = num_positive
    layer_map = (EXPECTED_LAYERS, p, EXPECTED_PATCHES)
    class_map = (p, EXPECTED_PATCHES)
    return {
        "image_id": (),
        "positive_class_ids": (p,),
        "grid_h": (),
        "grid_w": (),
        "patch_label_counts": (EXPECTED_PATCHES, EXPECTED_CLASSES + 2),
        "region_masks_rho05": class_map,
        "region_masks_rho07": class_map,
        "feature_post_scores": layer_map,
        "feature_norm_scores": layer_map,
        "feature_final_norm_scores": class_map,
        "qk_mean_scores": layer_map,
        "qk_head_std": layer_map,
        "attn_c2p_raw": layer_map,
        "attn_c2p_conditional": layer_map,
        "attn_patch_mass": (EXPECTED_LAYERS, p),
        "patch_logits": class_map,
        "patch_cam": class_map,
        "attn_official_raw": class_map,
        "attn_official_conditional": class_map,
        "attn_mid3_raw": class_map,
        "attn_mid3_conditional": class_map,
        "c2p_cam": class_map,
        "final_cam": class_map,
        "class_logits": (p,),
        "patch_class_logits": (p,),
        "class_logits_all": (EXPECTED_CLASSES,),
        "patch_class_logits_all": (EXPECTED_CLASSES,),
        "raw_final_cam_confusion_t045": (
            EXPECTED_CLASSES + 1,
            EXPECTED_CLASSES + 1,
        ),
        "class_token_pairwise_cosine": (EXPECTED_LAYERS, p, p),
        "patch_norms": (EXPECTED_LAYERS, EXPECTED_PATCHES),
        "qk_head_region_mean_rho05": (EXPECTED_LAYERS, 6, p, 3),
        "qk_head_region_mean_rho07": (EXPECTED_LAYERS, 6, p, 3),
        "diagnostic_c2p_cam_l10": class_map,
        "diagnostic_c2p_cam_l11": class_map,
        "diagnostic_c2p_cam_l12": class_map,
        "diagnostic_c2p_cam_mid3": class_map,
    }


def _validate_signal_artifact(
    artifact: np.lib.npyio.NpzFile,
    *,
    path: Path,
    image_id: str,
    positive: np.ndarray,
) -> None:
    keys = set(artifact.files)
    if keys != EXPECTED_SIGNAL_KEYS:
        missing = sorted(EXPECTED_SIGNAL_KEYS - keys)
        extra = sorted(keys - EXPECTED_SIGNAL_KEYS)
        raise RuntimeError(
            f"signal schema mismatch {path}: missing={missing}, extra={extra}"
        )
    for key, shape in _expected_signal_shapes(len(positive)).items():
        value = np.asarray(artifact[key])
        if value.shape != shape:
            raise RuntimeError(
                f"signal shape mismatch {path}:{key}: {value.shape} != {shape}"
            )
        if value.dtype.kind in "fc":
            # Empty semantic regions legitimately produce NaN in the two
            # pre-aggregated QK region summaries. No other stored numeric
            # signal may be non-finite, and infinity is never legitimate.
            if key.startswith("qk_head_region_mean_"):
                valid = not np.isinf(value).any()
            else:
                valid = np.isfinite(value).all()
            if not valid:
                raise RuntimeError(f"invalid floating values in {path}:{key}")
    if str(artifact["image_id"].item()) != image_id:
        raise RuntimeError(f"stored image ID mismatch: {path}")
    if not np.array_equal(np.asarray(artifact["positive_class_ids"]), positive):
        raise RuntimeError(f"stored positive classes mismatch: {path}")
    if int(artifact["grid_h"].item()) != 28 or int(artifact["grid_w"].item()) != 28:
        raise RuntimeError(f"stored grid mismatch: {path}")
    for key in ("region_masks_rho05", "region_masks_rho07"):
        if not np.isin(np.asarray(artifact[key]), (0, 1, 2, 3, 4)).all():
            raise RuntimeError(f"invalid semantic region code in {path}:{key}")
    conditional = np.asarray(artifact["attn_c2p_conditional"], dtype=np.float64)
    if np.max(np.abs(conditional.sum(axis=-1) - 1.0)) >= 1e-6:
        raise RuntimeError(f"conditional attention is not normalized: {path}")


def _audit_signal_root(
    root: Path,
    model: str,
    image_ids: list[str],
    labels: np.ndarray,
    *,
    expected_checkpoint_path: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    completion = read_json(root / "completion.json")
    metadata = read_json(root / "metadata.json")
    if completion.get("status") != "complete" or completion.get("run_kind") != "full":
        raise RuntimeError(f"incomplete Experiment 2 signal source: {root}")
    if completion.get("model") != model or metadata.get("model") != model:
        raise RuntimeError(f"signal source model identity mismatch: {root}")
    if metadata.get("status") != "complete" or metadata.get("run_kind") != "full":
        raise RuntimeError(f"signal metadata is not a completed full run: {root}")
    if (
        int(completion.get("num_images", -1)) != EXPECTED_IMAGES
        or int(metadata.get("processed_images", -1)) != EXPECTED_IMAGES
    ):
        raise RuntimeError(f"signal image count metadata mismatch: {root}")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"signal metadata lacks checkpoint record: {root}")
    if (
        Path(str(checkpoint.get("path"))).resolve() != expected_checkpoint_path
        or str(checkpoint.get("sha256")) != expected_checkpoint_sha256
    ):
        raise RuntimeError(f"signal/checkpoint linkage mismatch: {root}")
    records = _read_jsonl(root / "manifest.jsonl")
    manifest_ids = [str(row.get("image_id", "")) for row in records]
    if manifest_ids != image_ids:
        raise RuntimeError(f"signal manifest order/membership mismatch: {root}")
    schemas: Counter[tuple[str, ...]] = Counter()
    pairs = 0
    for index, record in enumerate(records):
        path = (root / str(record["signal_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != str(record.get("artifact_sha256", "")):
            raise RuntimeError(f"signal hash mismatch: {path}")
        if int(record.get("grid_h", -1)) != 28 or int(record.get("grid_w", -1)) != 28:
            raise RuntimeError(f"signal manifest grid mismatch: {path}")
        if (
            int(record.get("num_layers", -1)) != EXPECTED_LAYERS
            or int(record.get("num_patches", -1)) != EXPECTED_PATCHES
        ):
            raise RuntimeError(f"signal manifest geometry mismatch: {path}")
        expected = np.flatnonzero(labels[index] > 0).astype(np.int64)
        if record.get("positive_class_ids") != expected.tolist():
            raise RuntimeError(f"signal manifest label mismatch: {path}")
        with np.load(path, allow_pickle=False) as artifact:
            schemas[tuple(sorted(artifact.files))] += 1
            _validate_signal_artifact(
                artifact,
                path=path,
                image_id=image_ids[index],
                positive=expected,
            )
        pairs += len(expected)
    if pairs != EXPECTED_POSITIVE_PAIRS:
        raise RuntimeError(f"positive-pair count mismatch in {root}: {pairs}")
    if len(schemas) != 1:
        raise RuntimeError(f"heterogeneous signal schemas in {root}: {schemas}")
    keys = set(next(iter(schemas)))
    sufficiency = {
        "per_layer_head_mean_c2p": "attn_c2p_raw" in keys,
        "patch_cam": "patch_cam" in keys,
        "transformed_gt_patch_counts_and_regions": {
            "patch_label_counts",
            "region_masks_rho05",
            "region_masks_rho07",
        }.issubset(keys),
        "full_per_layer_class_tokens": "class_tokens" in keys,
        "full_per_layer_patch_tokens": "patch_tokens" in keys,
        "all_layer_patch_to_patch_sum": "patch_to_patch_sum" in keys,
        "arbitrary_layer_readout_possible_offline": "patch_to_patch_sum" in keys,
    }
    return {
        "root": str(root),
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "completion_sha256": sha256_file(root / "completion.json"),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "manifest_rows": len(records),
        "positive_image_class_pairs": pairs,
        "schema_distribution": {"|".join(key): value for key, value in schemas.items()},
        "signal_sufficiency": sufficiency,
        "upstream_numerics": {
            key: metadata.get(key)
            for key in (
                "experiment1_feature_post_max_abs_diff",
                "native_cam_max_abs_diff",
                "qk_attention_max_abs_diff",
                "attention_row_sum_max_abs_error",
            )
        },
    }


def execute(args: argparse.Namespace) -> None:
    require_tgca_repro()
    exp2 = args.experiment2_root.expanduser().resolve()
    output = assert_new_output(args.output_dir, [exp2])
    status = read_json(exp2 / "pipeline_status.json")
    pipeline = read_json(exp2 / "pipeline_metadata.json")
    canonical = read_json(exp2 / "canonical/canonical_metadata.json")
    analysis = read_json(exp2 / "analysis/analysis_metadata.json")
    if status.get("status") != "complete" or int(status.get("exit_code", -1)) != 0:
        raise RuntimeError(f"Experiment 2 source is not complete: {exp2}")
    output.mkdir(parents=True, exist_ok=False)

    dataset = pipeline.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("Experiment 2 pipeline metadata lacks dataset mapping")
    voc_root = Path(str(dataset["voc_root"])).resolve()
    val_list = Path(str(dataset["val_list"])).resolve()
    labels_path = voc_root / "ImageLabel/cls_labels.npy"
    image_ids = ordered_val_ids(val_list)
    labels = load_image_labels(labels_path, image_ids)
    positive_pairs = int(labels.sum())
    multilabel = int(np.count_nonzero(labels.sum(axis=1) >= 2))
    if len(image_ids) != args.expected_images:
        raise RuntimeError(f"VOC count {len(image_ids)} != {args.expected_images}")
    if positive_pairs != args.expected_positive_pairs:
        raise RuntimeError(
            f"positive-pair count {positive_pairs} != {args.expected_positive_pairs}"
        )
    if multilabel != args.expected_multilabel_images:
        raise RuntimeError(
            f"multi-label count {multilabel} != {args.expected_multilabel_images}"
        )

    checkpoints_raw = pipeline.get("checkpoints")
    if not isinstance(checkpoints_raw, Mapping):
        raise TypeError("pipeline metadata lacks checkpoints")
    checkpoint_verification: dict[str, object] = {}
    checkpoint_paths: dict[str, Path] = {}
    for model in MODEL_KEYS:
        record = checkpoints_raw.get(model)
        if not isinstance(record, Mapping):
            raise TypeError(f"missing checkpoint record for {model}")
        path = Path(str(record["path"])).resolve()
        actual = sha256_file(path)
        expected = str(record["sha256"])
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint_paths[model] = path
        checkpoint_verification[model] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": True,
        }

    roots_raw = canonical.get("source_roots")
    if not isinstance(roots_raw, Mapping):
        raise TypeError("canonical metadata lacks source_roots")
    signal_audits = {
        model: _audit_signal_root(
            Path(str(roots_raw[model])).resolve(),
            model,
            image_ids,
            labels,
            expected_checkpoint_path=checkpoint_paths[model],
            expected_checkpoint_sha256=str(
                checkpoint_verification[model]["actual_sha256"]  # type: ignore[index]
            ),
        )
        for model in MODEL_KEYS
    }

    table_checks: dict[str, object] = {}
    tables = canonical.get("tables")
    if not isinstance(tables, Mapping):
        raise TypeError("canonical metadata lacks tables")
    if set(tables) != set(EXPECTED_CANONICAL_TABLE_ROWS):
        raise RuntimeError(
            "canonical table inventory mismatch: "
            f"actual={sorted(tables)}, expected={sorted(EXPECTED_CANONICAL_TABLE_ROWS)}"
        )
    for name, raw in tables.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"invalid canonical table record: {name}")
        path = Path(str(raw["path"])).resolve()
        actual = sha256_file(path)
        expected = str(raw["sha256"])
        if actual != expected:
            raise RuntimeError(f"canonical table hash mismatch: {path}")
        expected_rows = EXPECTED_CANONICAL_TABLE_ROWS[str(name)]
        if int(raw.get("rows", -1)) != expected_rows:
            raise RuntimeError(
                f"canonical row count mismatch for {name}: "
                f"{raw.get('rows')} != {expected_rows}"
            )
        if raw.get("roundtrip_verified") is not True:
            raise RuntimeError(f"canonical round-trip not verified: {name}")
        columns = raw.get("columns")
        if not isinstance(columns, list) or not columns:
            raise RuntimeError(f"canonical table columns missing: {name}")
        table_checks[str(name)] = {
            "path": str(path),
            "sha256": actual,
            "rows": raw.get("rows"),
        }

    if (
        canonical.get("status") != "complete"
        or canonical.get("source_immutability_verified") is not True
        or canonical.get("source_manifests_exact_match") is not True
        or int(canonical.get("num_artifacts_processed", -1)) != 2 * EXPECTED_IMAGES
        or int(canonical.get("num_manifest_images_per_model", -1)) != EXPECTED_IMAGES
        or int(canonical.get("expected_layers", -1)) != EXPECTED_LAYERS
        or canonical.get("expected_grid") != [28, 28]
    ):
        raise RuntimeError("Experiment 2 canonical completion/integrity gate failed")
    tree_records = canonical.get("source_tree_before_after")
    if not isinstance(tree_records, Mapping):
        raise TypeError("canonical metadata lacks source tree verification")
    for model in MODEL_KEYS:
        tree = tree_records.get(model)
        if (
            not isinstance(tree, Mapping)
            or tree.get("tree_sha256_before") != tree.get("tree_sha256_after")
            or int(tree.get("num_files", -1)) != EXPECTED_IMAGES + 7
        ):
            raise RuntimeError(f"canonical source-tree integrity failed: {model}")

    bootstrap = analysis.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise TypeError("Experiment 2 analysis lacks bootstrap metadata")
    if (
        int(bootstrap.get("repeats", -1)) != 5000
        or bootstrap.get("unit") != "image_id cluster"
        or bootstrap.get("patches_or_image_class_pairs_treated_independent")
        is not False
        or bootstrap.get("same_image_draw_reused_within_analysis_family") is not True
    ):
        raise RuntimeError("Experiment 2 clustered-bootstrap integrity gate failed")

    linkage = {
        "status": "complete",
        "experiment2_root": str(exp2),
        "experiment2_pipeline_status_sha256": sha256_file(
            exp2 / "pipeline_status.json"
        ),
        "experiment2_pipeline_metadata_sha256": sha256_file(
            exp2 / "pipeline_metadata.json"
        ),
        "experiment2_canonical_metadata_sha256": sha256_file(
            exp2 / "canonical/canonical_metadata.json"
        ),
        "experiment2_analysis_metadata_sha256": sha256_file(
            exp2 / "analysis/analysis_metadata.json"
        ),
        "canonical_tables": table_checks,
        "signals": signal_audits,
        "source_immutability_verified_by_experiment2": bool(
            canonical.get("source_immutability_verified")
        ),
        "bootstrap": bootstrap,
    }
    json_dump(output / "experiment2_linkage.json", linkage)
    json_dump(output / "checkpoint_verification.json", checkpoint_verification)

    # Manifest every directly consumed source tree and exact VOC/checkpoint file.
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()

    def add_tree(group: str, model: str, root: Path) -> None:
        for row in _tree_records(group, model, root):
            path = Path(str(row["absolute_path"]))
            if path not in seen:
                seen.add(path)
                rows.append(row)

    add_tree("experiment2_result", "", exp2)
    exp1_inputs = pipeline.get("experiment1_inputs")
    if isinstance(exp1_inputs, Mapping):
        for name, value in exp1_inputs.items():
            root = Path(str(value)).resolve()
            if root.is_dir():
                add_tree("experiment1_upstream", str(name), root)
    for model, path in checkpoint_paths.items():
        if path not in seen:
            seen.add(path)
            rows.append(_manifest_record("checkpoint", model, path.parent, path))
    dataset_files = [val_list, labels_path]
    dataset_files.extend(
        voc_root / "JPEGImages" / f"{value}.jpg" for value in image_ids
    )
    dataset_files.extend(
        voc_root / "SegmentationClass" / f"{value}.png" for value in image_ids
    )
    for path in dataset_files:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in seen:
            seen.add(path)
            rows.append(_manifest_record("voc", "", voc_root, path))
    rows.sort(key=lambda row: str(row["absolute_path"]))
    manifest_count = _write_csv(output / "immutable_manifest_before.csv", rows)
    manifest_bytes = int(sum(int(row["size_bytes"]) for row in rows))

    source_metadata = {
        "status": "complete",
        "integrity_passed": True,
        "experiment2_root": str(exp2),
        "experiment2_linkage": str(output / "experiment2_linkage.json"),
        "checkpoints": checkpoint_verification,
        "signal_roots": {model: signal_audits[model]["root"] for model in MODEL_KEYS},
        "dataset": {
            "voc_root": str(voc_root),
            "list_path": str(val_list),
            "labels_path": str(labels_path),
            "input_size": 448,
            "patch_size": 16,
            "transform": dataset.get("transform"),
            "num_images": len(image_ids),
            "positive_image_class_pairs": positive_pairs,
            "multilabel_images": multilabel,
        },
        "layer_numbering": "reports use 1..12; model blocks use 0..11",
        "class_mapping": "image class 0..19 maps to semantic mask ID 1..20",
        "existing_signal_sufficiency": {
            model: signal_audits[model]["signal_sufficiency"] for model in MODEL_KEYS
        },
        "supplemental_forward_justification": (
            "Experiment 2 stores no per-layer token vectors and no all-layer P2P "
            "sum; frozen deterministic supplemental inference is required for "
            "axis removal, exact B4, and C2C intervention."
        ),
        "immutable_manifest": {
            "path": str(output / "immutable_manifest_before.csv"),
            "rows": manifest_count,
            "bytes": manifest_bytes,
            "sha256": sha256_file(output / "immutable_manifest_before.csv"),
        },
        "git": git_state(REPO_ROOT),
        "completed_at": timestamp(),
    }
    json_dump(output / "source_metadata.json", source_metadata)
    audit_lines = [
        "# Experiment 3 input audit",
        "",
        "- Status: **PASS**",
        f"- Experiment 2 root: `{exp2}`",
        f"- VOC val images: {len(image_ids):,}",
        f"- Positive image-class pairs: {positive_pairs:,}",
        f"- Multi-label images: {multilabel:,}",
        f"- Immutable files hashed: {manifest_count:,} ({manifest_bytes:,} bytes)",
        "- Checkpoints: exact SHA-256 match for MCTformer and MCTformer+.",
        "- Experiment 2: both 1,449-image signal manifests, all artifacts, and all canonical table hashes verified.",
        "- Existing-signal conclusion: per-layer token vectors and the reusable all-layer P2P sum are absent; a matched frozen supplemental forward is necessary.",
        "- No source file was written or modified by this audit.",
        "",
    ]
    (output / "INPUT_AUDIT.md").write_text("\n".join(audit_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    execute(args)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
