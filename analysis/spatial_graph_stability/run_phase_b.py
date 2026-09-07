#!/usr/bin/env python
"""Run full frozen PatchFinalLN local spatial-graph semantic validation."""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES

from .basis import generate_basis_transforms
from .frozen import build_mct_dataset, load_patch_final_mctformer
from .graph import (
    GRAPH_NAMES,
    LocalEdges,
    construct_local_graphs,
    local_edge_index,
    semantic_patch_labels,
)
from .graph_metrics import evaluate_boundary_metrics, evaluate_graph_metrics
from .provenance import (
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    EXPECTED_VOC_IMAGES,
    RunLog,
    command_line,
    create_output,
    csv_dump,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    text_dump,
    timestamp,
    write_environment_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mct-checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument(
        "--list-path",
        type=Path,
        default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-bins", type=int, default=2048)
    parser.add_argument("--test-log", type=Path)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).expanduser().resolve()


def _load_phase_a_rotations(phase_a_dir: Path) -> tuple[list[str], list[np.ndarray], dict[str, object]]:
    completion_path = phase_a_dir / "completion.json"
    metadata_path = phase_a_dir / "basis_transforms.json"
    archive_path = phase_a_dir / "raw_outputs" / "basis_matrices_d384.npz"
    for path in (completion_path, metadata_path, archive_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    import json

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("invariance_passed") is not True:
        raise RuntimeError("Phase A is not a completed passing input")
    expected_hash = metadata["transforms_by_dimension"]["384"]["matrix_archive_sha256"]
    actual_hash = sha256_file(archive_path)
    if actual_hash != expected_hash:
        raise RuntimeError("Phase A d384 basis archive hash mismatch")
    with np.load(archive_path, allow_pickle=False) as source:
        names = list(source.files)
        matrices = [np.asarray(source[name], dtype=np.float64) for name in names]
    expected_names = [item.name for item in generate_basis_transforms(384)]
    if names != expected_names:
        raise RuntimeError(f"Phase A basis names differ: {names} versus {expected_names}")
    return names, matrices, {
        "phase_a_dir": str(phase_a_dir),
        "completion_sha256": sha256_file(completion_path),
        "basis_metadata_sha256": sha256_file(metadata_path),
        "matrix_archive_sha256": actual_hash,
    }


def _extract_graph_cache(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    edges: LocalEdges,
    rotation_names: list[str],
    rotations: list[np.ndarray],
    log: RunLog,
) -> tuple[dict[str, object], dict[str, np.ndarray], list[str], np.ndarray, np.ndarray]:
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit)
    if not args.limit and len(dataset) != EXPECTED_VOC_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_VOC_IMAGES} VOC val images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model, model_metadata = load_patch_final_mctformer(args.mct_checkpoint)
    model.to(device).eval()
    image_count = len(dataset)
    feature_weights = np.empty((image_count, edges.count), dtype=np.float32)
    semantic_labels = np.empty((image_count, 784), dtype=np.int8)
    label_counts = np.empty(image_count, dtype=np.uint8)
    invariance: dict[str, dict[str, float]] = {}
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            _, patches, attentions, all_class, auxiliary = model.forward_features(
                images, return_aux=True
            )
            if (
                patches.shape[1:] != (784, 384)
                or auxiliary.get("patch_final_norm") is not True
                or auxiliary.get("final_norm") is not False
                or len(attentions) != 12
                or len(all_class) != 12
            ):
                raise RuntimeError("frozen PatchFinalLN feature contract failed")
            graphs = construct_local_graphs(patches, edges, sigma_s=1.0)
            batch_size = len(images)
            sl = slice(offset, offset + batch_size)
            feature_weights[sl] = graphs["B2_feature"].cpu().numpy().astype(np.float32)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()
            label_counts[sl] = (labels > 0).sum(axis=1).astype(np.uint8)
            for local_index in range(batch_size):
                semantic_labels[offset + local_index] = semantic_patch_labels(
                    masks[local_index], patch_size=16
                ).reshape(-1)
            if offset == 0:
                baseline = {name: value[:2].double() for name, value in graphs.items()}
                for name, rotation_array in zip(rotation_names, rotations):
                    rotation = torch.from_numpy(rotation_array).to(
                        device=device, dtype=torch.float64
                    )
                    transformed = construct_local_graphs(
                        patches[:2].double() @ rotation, edges, sigma_s=1.0
                    )
                    invariance[name] = {
                        graph_name: float(
                            (baseline[graph_name] - transformed[graph_name]).abs().max().item()
                        )
                        for graph_name in ("B2_feature", "B3_spatial_feature")
                    }
            offset += batch_size
            if batch_number == 1 or batch_number % 20 == 0 or offset == image_count:
                log(f"Phase B extracted frozen patches {offset}/{image_count}")
    if offset != image_count:
        raise RuntimeError(f"processed {offset} images, expected {image_count}")
    uniform = np.ones_like(feature_weights, dtype=np.float32)
    spatial_edge = np.exp(-edges.squared_distance.astype(np.float64) / 2.0).astype(np.float32)
    spatial = np.broadcast_to(spatial_edge[None, :], feature_weights.shape).copy()
    weights = {
        "B0_uniform": uniform,
        "B1_spatial": spatial,
        "B2_feature": feature_weights,
        "B3_spatial_feature": spatial * feature_weights,
    }
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in weights.values()):
        raise RuntimeError("graph cache contains invalid weights")
    cache_path = output_dir / "graph_analysis_cache.npz"
    np.savez_compressed(
        cache_path,
        image_ids=np.asarray(dataset.image_ids),
        image_label_counts=label_counts,
        semantic_labels=semantic_labels,
        edge_source=edges.source,
        edge_target=edges.target,
        edge_squared_distance=edges.squared_distance,
        b2_feature_weights=feature_weights,
    )
    basis_checks = {
        "schema_version": 1,
        "subset_rule": "first two images in deterministic VOC val order",
        "subset_image_ids": dataset.image_ids[:2],
        "tolerance": 1e-5,
        "per_transform_max_abs_edge_weight_error": invariance,
        "maximum_error": max(
            value
            for transform in invariance.values()
            for value in transform.values()
        ),
    }
    basis_checks["passed"] = basis_checks["maximum_error"] < basis_checks["tolerance"]
    json_dump(output_dir / "basis_invariance.json", basis_checks)
    if not basis_checks["passed"]:
        raise RuntimeError(f"Phase B basis-invariance regression failed: {basis_checks}")
    metadata = {
        "model": model_metadata,
        "dataset": {
            "name": "PASCAL VOC 2012 val",
            "image_count": image_count,
            "input_size": 448,
            "patch_size": 16,
            "patch_grid": [28, 28],
            "patch_count": 784,
            "label_strata_counts": {
                "single_label": int(np.sum(label_counts == 1)),
                "two_label": int(np.sum(label_counts == 2)),
                "three_plus_label": int(np.sum(label_counts >= 3)),
            },
        },
        "graph": {
            "candidate_set": "undirected 8-neighbor / 3x3 local neighborhood",
            "edge_count_per_image": edges.count,
            "same_candidate_edges_all_variants": True,
            "sigma_s": 1.0,
            "feature_affinity": "clamp((1 + cosine(p_i,p_j))/2, 0, 1)",
            "joint_affinity": "spatial * feature",
            "symmetrized": True,
            "gt_used_in_graph_construction": False,
            "graph_low_pass_performed": False,
        },
        "patch_gt": {
            "valid_fraction_minimum": 0.5,
            "semantic_majority_minimum": 0.5,
            "tie_policy": "mixed",
            "mixed_void_primary_metric_policy": "excluded",
        },
        "duration_seconds": time.perf_counter() - started,
        "cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "size_bytes": cache_path.stat().st_size,
            "large_raw_cache_committed": False,
        },
        "basis_invariance": basis_checks,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, weights, list(dataset.image_ids), semantic_labels, label_counts


def _unnormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _render_examples(
    *,
    output_dir: Path,
    dataset,
    image_ids: list[str],
    semantic_labels: np.ndarray,
    label_counts: np.ndarray,
    weights: dict[str, np.ndarray],
    edges: LocalEdges,
) -> list[dict[str, object]]:
    first = semantic_labels[:, edges.source]
    second = semantic_labels[:, edges.target]
    fg_bg_count = np.sum(
        ((first == 0) & (second > 0)) | ((second == 0) & (first > 0)), axis=1
    )
    selections: list[int] = []
    for predicate in (label_counts == 1, label_counts >= 2):
        eligible = np.flatnonzero(predicate & (fg_bg_count >= 20))
        selections.extend(eligible[:2].tolist())
    selections = list(dict.fromkeys(selections))[:4]
    records: list[dict[str, object]] = []
    source_xy = np.stack(
        (edges.source % 28 + 0.5, edges.source // 28 + 0.5), axis=-1
    )
    target_xy = np.stack(
        (edges.target % 28 + 0.5, edges.target // 28 + 0.5), axis=-1
    )
    segments = np.stack((source_xy, target_xy), axis=1)
    for image_index in selections:
        sample = dataset[image_index]
        rgb = _unnormalize(sample["image"])
        labels = semantic_labels[image_index].reshape(28, 28)
        figure, axes = plt.subplots(1, 5, figsize=(18, 3.8))
        axes[0].imshow(rgb, extent=(0, 28, 28, 0))
        axes[0].set_title("RGB")
        shown = labels.copy()
        shown[shown < 0] = 21
        axes[1].imshow(shown, cmap="tab20b", vmin=0, vmax=21)
        axes[1].set_title("Patch GT")
        edge_same = labels.reshape(-1)[edges.source] == labels.reshape(-1)[edges.target]
        edge_valid = (labels.reshape(-1)[edges.source] >= 0) & (
            labels.reshape(-1)[edges.target] >= 0
        )
        for axis, graph in zip(axes[2:], ("B1_spatial", "B2_feature", "B3_spatial_feature")):
            axis.imshow(rgb, extent=(0, 28, 28, 0), alpha=0.68)
            value = weights[graph][image_index]
            keep_count = max(1, int(math.ceil(0.08 * len(value))))
            order = np.lexsort((np.arange(len(value)), -value))[:keep_count]
            colors = np.where(edge_valid[order] & edge_same[order], "lime", "red")
            collection = LineCollection(
                segments[order], colors=colors.tolist(), linewidths=0.8, alpha=0.7
            )
            axis.add_collection(collection)
            axis.set_xlim(0, 28)
            axis.set_ylim(28, 0)
            axis.set_title(f"{graph}\ntop 8% edges")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{image_ids[image_index]} · image-label count {int(label_counts[image_index])} · green=same, red=cross"
        )
        figure.tight_layout()
        path = output_dir / f"{image_ids[image_index]}_local_graphs.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {
                "image_id": image_ids[image_index],
                "image_index": image_index,
                "image_label_count": int(label_counts[image_index]),
                "selection_rule": "first deterministic VOC ID in stratum with >=20 FG-BG edges",
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return records


def _lookup(
    rows: list[dict[str, object]],
    graph: str,
    metric: str,
    *,
    aggregation: str = "micro",
    stratum: str = "all",
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["graph"] == graph
        and row["metric"] == metric
        and row["aggregation"] == aggregation
        and row["stratum"] == stratum
    ]
    if len(matches) != 1:
        raise RuntimeError(f"metric lookup is ambiguous/missing: {graph}, {metric}")
    return matches[0]


def _delta_lookup(
    rows: list[dict[str, object]],
    comparison: str,
    metric: str,
    *,
    aggregation: str = "micro",
    stratum: str = "all",
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["comparison"] == comparison
        and row["metric"] == metric
        and row["aggregation"] == aggregation
        and row["stratum"] == stratum
    ]
    if len(matches) != 1:
        raise RuntimeError(f"delta lookup is ambiguous/missing: {comparison}, {metric}")
    return matches[0]


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _write_report(
    output_dir: Path,
    extraction: dict[str, object],
    metric_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    boundary_rows: list[dict[str, object]],
    statistical_audit: dict[str, object],
) -> None:
    lines = [
        "# Spatial Graph Validation Report",
        "",
        "## Frozen contract",
        "",
        f"- Model: matched seed-0 PatchFinalLN MCTformer+-Small, checkpoint SHA256 `{extraction['model']['checkpoint_sha256']}`.",
        f"- Data: full VOC 2012 val, `{extraction['dataset']['image_count']}` images, deterministic 448 input and 28×28 patch grid.",
        "- Every graph uses the identical 2,970-edge undirected 8-neighbor candidate set. GT was never passed to graph construction; mixed/void patches are excluded from primary metrics.",
        "- B0=uniform, B1=fixed σs=1 spatial Gaussian, B2=parameter-free cosine affinity, B3=B1×B2. No graph filtering or training was performed.",
        "",
        "## Primary full-val micro results",
        "",
        "| Graph | Purity ↑ | Boundary leakage ↓ | FG-only purity ↑ | FG–BG leakage ↓ | Edge AUROC ↑ | Edge AUPRC ↑ | FG-only AUROC ↑ | FG-only AUPRC ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for graph in GRAPH_NAMES:
        values = [
            _lookup(metric_rows, graph, metric)["estimate"]
            for metric in (
                "same_semantic_purity",
                "semantic_boundary_leakage",
                "foreground_only_purity",
                "foreground_background_leakage",
                "edge_auroc",
                "edge_auprc",
                "foreground_only_edge_auroc",
                "foreground_only_edge_auprc",
            )
        ]
        lines.append(f"| {graph} | " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Paired B3 deltas (whole-image clustered 95% CI)",
            "",
            "| Comparison | Metric | Δ | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in ("B3-B1", "B3-B0", "B3-B2"):
        for metric in (
            "foreground_only_purity",
            "foreground_background_leakage",
            "edge_auroc",
            "edge_auprc",
            "foreground_only_edge_auroc",
        ):
            row = _delta_lookup(delta_rows, comparison, metric)
            lines.append(
                f"| {comparison} | {metric} | {_fmt(row['estimate_delta'])} | [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] |"
            )
    purity = _delta_lookup(delta_rows, "B3-B1", "foreground_only_purity")
    leakage = _delta_lookup(delta_rows, "B3-B1", "foreground_background_leakage")
    auc_b2 = _delta_lookup(delta_rows, "B3-B2", "edge_auroc")
    auc_b1 = _delta_lookup(delta_rows, "B3-B1", "edge_auroc")
    strong = (
        purity["ci_low"] is not None
        and float(purity["ci_low"]) > 0
        and leakage["ci_high"] is not None
        and float(leakage["ci_high"]) < 0
        and (
            (auc_b2["ci_low"] is not None and float(auc_b2["ci_low"]) > 0)
            or (auc_b1["ci_low"] is not None and float(auc_b1["ci_low"]) > 0)
        )
    )
    partial = any(
        row["ci_low"] is not None and float(row["ci_low"]) > 0
        for row in (purity, auc_b1, auc_b2)
    ) or (leakage["ci_high"] is not None and float(leakage["ci_high"]) < 0)
    support = "strong support" if strong else "partial support" if partial else "not supported"
    lines.extend(
        [
            "",
            "## Boundary-focused weights",
            "",
            "| Graph | Same interior | Same boundary-near | Cross boundary | FG–BG boundary |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    boundary_lookup = {
        (row["graph"], row["edge_category"]): row for row in boundary_rows
    }
    categories = (
        "same_semantic_interior",
        "same_semantic_boundary_near",
        "cross_semantic_boundary",
        "foreground_background_boundary",
    )
    for graph in GRAPH_NAMES:
        lines.append(
            f"| {graph} | "
            + " | ".join(_fmt(boundary_lookup[(graph, key)]["mean_weight"]) for key in categories)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            f"By the plan's qualitative/statistical criteria, the frozen B3 graph receives **{support}**. This label is diagnostic, not a claim that graph smoothing or a new method improves CAMs.",
            "",
            f"All intervals use exactly `{statistical_audit['bootstrap_repeats']}` paired whole-image resamples with seed `{statistical_audit['bootstrap_seed']}`. Edge rank point estimates are exact; their clustered intervals use `{statistical_audit['histogram_bins']}` fixed bins, with maximum observed full-sample approximation error `{statistical_audit['maximum_full_sample_histogram_abs_error']:.3e}`. Edges from an image were never independently resampled.",
            "",
            "Per-class, macro-class, image-label strata, boundary distributions, and every paired delta are available in the accompanying CSV files. The analysis establishes only whether the frozen local affinity graph aligns with patch-level semantic structure.",
            "",
        ]
    )
    text_dump(output_dir / "SPATIAL_GRAPH_VALIDATION_REPORT.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    require_environment()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.output_dir = _resolve(args.repo_root, args.output_dir)
    args.phase_a_dir = _resolve(args.repo_root, args.phase_a_dir)
    args.mct_checkpoint = _resolve(args.repo_root, args.mct_checkpoint)
    args.voc_root = _resolve(args.repo_root, args.voc_root)
    args.list_path = _resolve(args.repo_root, args.list_path)
    if args.test_log:
        args.test_log = _resolve(args.repo_root, args.test_log)
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0:
        raise ValueError("invalid batch/worker/limit configuration")
    if args.bootstrap_repeats < 1 or args.bootstrap_seed < 0 or args.bootstrap_bins < 2:
        raise ValueError("invalid bootstrap configuration")
    if not args.limit and (
        args.bootstrap_repeats != BOOTSTRAP_REPEATS
        or args.bootstrap_seed != BOOTSTRAP_SEED
    ):
        raise ValueError("full Phase B requires exactly 5,000 resamples and seed 20260901")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full Phase B requires an available CUDA device")
    git = require_clean_tracked(args.repo_root)
    rotation_names, rotations, phase_a_linkage = _load_phase_a_rotations(args.phase_a_dir)
    output_dir = create_output(args.output_dir)
    visual_dir = output_dir / "selected_graph_visualizations"
    visual_dir.mkdir()
    log = RunLog(output_dir / "run.log")
    log(f"Phase B started: {command_line()}")
    before_hashes = {
        "patchfinal_checkpoint": sha256_file(args.mct_checkpoint),
        "voc_val_list": sha256_file(args.list_path),
        "phase_a_completion": phase_a_linkage["completion_sha256"],
        "phase_a_basis_archive": phase_a_linkage["matrix_archive_sha256"],
    }
    manifests = write_environment_manifests(output_dir)
    edges = local_edge_index((28, 28))
    extraction, weights, image_ids, semantic_labels, label_counts = _extract_graph_cache(
        args=args,
        output_dir=output_dir,
        device=device,
        edges=edges,
        rotation_names=rotation_names,
        rotations=rotations,
        log=log,
    )
    log("Phase B frozen feature extraction complete; computing clustered metrics")
    metric_rows, per_class_rows, delta_rows, statistical_audit = evaluate_graph_metrics(
        image_ids=image_ids,
        image_label_counts=label_counts,
        semantic_labels=semantic_labels,
        weights=weights,
        edges=edges,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
        bins=args.bootstrap_bins,
        bootstrap_device=args.device,
    )
    boundary_rows = evaluate_boundary_metrics(
        image_ids=image_ids,
        semantic_labels=semantic_labels,
        weights=weights,
        edges=edges,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    csv_dump(output_dir / "graph_metrics.csv", metric_rows)
    csv_dump(output_dir / "graph_metrics_per_class.csv", per_class_rows)
    csv_dump(output_dir / "graph_bootstrap_deltas.csv", delta_rows)
    csv_dump(output_dir / "boundary_metrics.csv", boundary_rows)
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit)
    visualizations = _render_examples(
        output_dir=visual_dir,
        dataset=dataset,
        image_ids=image_ids,
        semantic_labels=semantic_labels,
        label_counts=label_counts,
        weights=weights,
        edges=edges,
    )
    json_dump(visual_dir / "selection_manifest.json", {"examples": visualizations})
    _write_report(
        output_dir,
        extraction,
        metric_rows,
        delta_rows,
        boundary_rows,
        statistical_audit,
    )
    exact_commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        command_line(),
    ]
    if args.test_log:
        exact_commands.extend(
            ["", f"# Test record: {args.test_log}", f"# SHA256: {sha256_file(args.test_log)}"]
        )
    text_dump(output_dir / "exact_commands.sh", "\n".join(exact_commands) + "\n")
    (output_dir / "exact_commands.sh").chmod(0o755)
    after_hashes = {
        "patchfinal_checkpoint": sha256_file(args.mct_checkpoint),
        "voc_val_list": sha256_file(args.list_path),
        "phase_a_completion": sha256_file(args.phase_a_dir / "completion.json"),
        "phase_a_basis_archive": sha256_file(
            args.phase_a_dir / "raw_outputs" / "basis_matrices_d384.npz"
        ),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("an immutable Phase B input changed during analysis")
    metadata = {
        "schema_version": 1,
        "phase": "B",
        "status": "complete",
        "finished_at": timestamp(),
        "command": command_line(),
        "git": git,
        "environment": manifests,
        "test_log": (
            {"path": str(args.test_log), "sha256": sha256_file(args.test_log)}
            if args.test_log
            else None
        ),
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "source_immutable": before_hashes == after_hashes,
        "phase_a_linkage": phase_a_linkage,
        "extraction": extraction,
        "statistics": statistical_audit,
        "output_tables": {
            name: {"sha256": sha256_file(output_dir / name)}
            for name in (
                "graph_metrics.csv",
                "graph_metrics_per_class.csv",
                "graph_bootstrap_deltas.csv",
                "boundary_metrics.csv",
            )
        },
    }
    json_dump(output_dir / "run_metadata.json", metadata)
    json_dump(
        output_dir / "completion.json",
        {
            "status": "complete",
            "phase": "B",
            "finished_at": timestamp(),
            "image_count": len(image_ids),
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
            "basis_invariance_passed": extraction["basis_invariance"]["passed"],
        },
    )
    log("Phase B complete")


if __name__ == "__main__":
    main()
