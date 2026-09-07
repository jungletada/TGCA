#!/usr/bin/env python
"""Phase C: frozen spatial graph-stability diagnostics for PatchFinalLN."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import summarize_clustered
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES

from .basis import transform_conv_input_basis
from .frozen import build_mct_dataset, load_patch_final_mctformer
from .graph import LocalEdges, local_edge_index
from .lowpass import bounded_stability, graph_energy, graph_lowpass, normalized_edge_weights
from .phase_metrics import (
    class_overlap_rows,
    correlation_rows,
    map_region_rows,
    negative_class_rows,
    region_distribution_rows,
    summarize_region_rows,
    summarize_simple_rows,
    boundary_distances,
    local_map_variance,
)
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


GRAPHS = ("B1_spatial", "B2_feature", "B3_spatial_feature")
LAMBDA_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
# The hard numerical contract is the plan's <1e-5 final residual.  A modestly
# tighter CG stop threshold avoids a float32 sparse-scatter rounding plateau
# being misreported as a solver failure on otherwise valid systems.
SOLVER_TOLERANCE = 9e-6
REQUIRED_RESIDUAL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-b-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mct-checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument(
        "--list-path", type=Path,
        default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--limit", type=int, default=0, help="smoke-only deterministic prefix")
    parser.add_argument("--test-log", type=Path)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _load_phase_b(phase_b_dir: Path, limit: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    completion = phase_b_dir / "completion.json"
    cache_path = phase_b_dir / "graph_analysis_cache.npz"
    metadata_path = phase_b_dir / "run_metadata.json"
    if not completion.is_file() or not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Phase C requires the complete immutable Phase B output/cache")
    import json

    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    if completion_payload.get("status") != "complete" or completion_payload.get("phase") != "B":
        raise RuntimeError("Phase B completion marker is invalid")
    with np.load(cache_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    count = int(limit or len(payload["image_ids"]))
    if count > len(payload["image_ids"]):
        raise ValueError("Phase C limit exceeds Phase B cache length")
    payload = {name: value[:count] if value.ndim and value.shape[0] == len(payload["image_ids"]) else value for name, value in payload.items()}
    linkage = {
        "phase_b_dir": str(phase_b_dir),
        "completion_sha256": sha256_file(completion),
        "cache_sha256": sha256_file(cache_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    return payload, linkage


def _graph_weights(cache: Mapping[str, np.ndarray], edges: LocalEdges) -> dict[str, np.ndarray]:
    feature = np.asarray(cache["b2_feature_weights"], dtype=np.float32)
    expected = (len(cache["image_ids"]), edges.count)
    if feature.shape != expected:
        raise RuntimeError(f"Phase B B2 cache has shape {feature.shape}, expected {expected}")
    spatial_edge = np.exp(-edges.squared_distance.astype(np.float64) / 2.0).astype(np.float32)
    spatial = np.broadcast_to(spatial_edge[None, :], expected).copy()
    return {
        "B1_spatial": spatial,
        "B2_feature": feature,
        "B3_spatial_feature": spatial * feature,
    }


def _map_correlations(raw: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson and continuous-map Spearman in [B,N,C] layout."""

    def pearson(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_centered = left - left.mean(dim=1, keepdim=True)
        right_centered = right - right.mean(dim=1, keepdim=True)
        numerator = (left_centered * right_centered).sum(dim=1)
        denominator = torch.sqrt(left_centered.square().sum(dim=1) * right_centered.square().sum(dim=1))
        valid = denominator > 1e-12
        safe = torch.where(valid, denominator, torch.ones_like(denominator))
        return torch.where(valid, numerator / safe, torch.full_like(numerator, torch.nan))

    # M is continuous in the frozen model.  Stable ordinal ranks make this a
    # device-local Spearman computation without patch-level inference.
    raw_rank = torch.argsort(torch.argsort(raw, dim=1, stable=True), dim=1, stable=True).to(raw.dtype)
    smooth_rank = torch.argsort(torch.argsort(smooth, dim=1, stable=True), dim=1, stable=True).to(raw.dtype)
    return pearson(raw, smooth), pearson(raw_rank, smooth_rank)


def _degree_from_weights(weights: np.ndarray, edges: LocalEdges) -> np.ndarray:
    result = np.zeros((weights.shape[0], 784), dtype=np.float32)
    np.add.at(result, (slice(None), edges.source), weights)
    np.add.at(result, (slice(None), edges.target), weights)
    return result


def _extract_and_analyze(
    *, args: argparse.Namespace, output_dir: Path, cache: Mapping[str, np.ndarray],
    edges: LocalEdges, weights: Mapping[str, np.ndarray], device: torch.device, log: RunLog,
) -> tuple[dict[str, object], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]]]:
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit)
    if not args.limit and len(dataset) != EXPECTED_VOC_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_VOC_IMAGES} VOC val images, got {len(dataset)}")
    image_ids = np.asarray(dataset.image_ids)
    if not np.array_equal(image_ids, cache["image_ids"]):
        raise RuntimeError("Phase B cache image order differs from deterministic Phase C dataset")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", drop_last=False)
    model, model_metadata = load_patch_final_mctformer(args.mct_checkpoint)
    model.to(device).eval()
    count = len(dataset)
    raw_maps = np.empty((count, 20, 784), dtype=np.float32)
    positive = np.empty((count, 20), dtype=bool)
    primary_smooth = {graph: np.empty_like(raw_maps) for graph in GRAPHS}
    primary_residual = {graph: np.empty_like(raw_maps) for graph in GRAPHS}
    primary_stability = {graph: np.empty_like(raw_maps) for graph in GRAPHS}
    numerical: list[dict[str, object]] = []
    max_graph_difference = 0.0
    max_residual = 0.0
    nonfinite_count = 0
    internal_tolerance_unmet = 0
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            _, patches, attentions, all_classes, auxiliary = model.forward_features(images, return_aux=True)
            if patches.shape[1:] != (784, 384) or len(attentions) != 12 or len(all_classes) != 12:
                raise RuntimeError("PatchFinalLN frozen feature contract failed")
            if auxiliary.get("patch_final_norm") is not True or auxiliary.get("final_norm") is not False:
                raise RuntimeError("Phase C requires PatchFinalLN readout exactly")
            batch_count = images.shape[0]
            grid = patches.reshape(batch_count, 28, 28, 384).permute(0, 3, 1, 2).contiguous()
            raw = model.head(grid)[:, :20].flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
            if not torch.isfinite(raw).all():
                raise RuntimeError("raw pre-ReLU classifier maps contain NaN/Inf")
            sl = slice(offset, offset + batch_count)
            raw_maps[sl] = raw.transpose(1, 2).cpu().numpy().astype(np.float32)
            positive[sl] = batch["label"].numpy() > 0
            for graph in GRAPHS:
                local_weights = torch.from_numpy(weights[graph][sl]).to(device=device, dtype=raw.dtype)
                if graph == "B2_feature":
                    from .graph import construct_local_graphs
                    recomputed = construct_local_graphs(patches, edges)["B2_feature"]
                    max_graph_difference = max(max_graph_difference, float((recomputed - local_weights).abs().max().item()))
                normalized, _ = normalized_edge_weights(local_weights, edges, dtype=raw.dtype)
                energy_raw = graph_energy(raw, normalized, edges)
                for lambda_value in LAMBDA_VALUES:
                    solve_started = time.perf_counter()
                    result = graph_lowpass(raw, local_weights, edges, lambda_value=lambda_value,
                                           tolerance=SOLVER_TOLERANCE, max_iterations=256)
                    elapsed = time.perf_counter() - solve_started
                    smooth = result.values
                    residual, scale, stability = bounded_stability(raw, smooth)
                    energy_smooth = graph_energy(smooth, normalized, edges)
                    pearson, spearman = _map_correlations(raw, smooth)
                    fidelity = torch.sqrt((raw - smooth).square().sum(dim=1)) / torch.sqrt(raw.square().sum(dim=1)).clamp_min(1e-12)
                    energy_retention = energy_smooth / energy_raw.clamp_min(1e-12)
                    std_ratio = smooth.std(dim=1, unbiased=False) / raw.std(dim=1, unbiased=False).clamp_min(1e-12)
                    residual_np = result.relative_residual.cpu().numpy()
                    max_residual = max(max_residual, float(residual_np.max()))
                    nonfinite_count += int((~np.isfinite(residual_np)).sum())
                    internal_tolerance_unmet += int((~result.converged).sum().item())
                    if float(residual_np.max()) >= REQUIRED_RESIDUAL:
                        raise RuntimeError(f"low-pass residual failure for {graph}, lambda={lambda_value}: {float(residual_np.max())}")
                    for local_image in range(batch_count):
                        image_index = offset + local_image
                        for class_id in range(20):
                            numerical.append({
                                "image_id": str(image_ids[image_index]), "image_index": image_index,
                                "class_id": class_id, "class_name": VOC_CLASS_NAMES[class_id],
                                "num_positive_classes": int(positive[image_index].sum()),
                                "graph": graph, "lambda": lambda_value,
                                "fidelity": float(fidelity[local_image, class_id].item()),
                                "relative_l2_change": float(fidelity[local_image, class_id].item()),
                                "map_pearson": float(pearson[local_image, class_id].item()),
                                "map_spearman": float(spearman[local_image, class_id].item()),
                                "energy_raw": float(energy_raw[local_image, class_id].item()),
                                "energy_smoothed": float(energy_smooth[local_image, class_id].item()),
                                "energy_retention": float(energy_retention[local_image, class_id].item()),
                                "smoothed_std_over_raw": float(std_ratio[local_image, class_id].item()),
                                "linear_solve_relative_residual": float(residual_np[local_image, class_id]),
                                "cg_iterations": int(result.iterations),
                                "solve_runtime_ms_per_map": 1000.0 * elapsed / (batch_count * 20),
                            })
                    if lambda_value == 1.0:
                        primary_smooth[graph][sl] = smooth.transpose(1, 2).cpu().numpy().astype(np.float32)
                        primary_residual[graph][sl] = residual.transpose(1, 2).cpu().numpy().astype(np.float32)
                        primary_stability[graph][sl] = stability.transpose(1, 2).cpu().numpy().astype(np.float32)
            offset += batch_count
            if batch_number == 1 or batch_number % 10 == 0 or offset == count:
                log(f"Phase C extracted and low-pass filtered {offset}/{count} images")
    if offset != count:
        raise RuntimeError("Phase C image extraction did not cover the requested dataset")
    elapsed_total = time.perf_counter() - started
    cache_dir = output_dir / "raw_outputs"
    cache_dir.mkdir()
    cache_files: dict[str, Path] = {}
    for name, array in (("raw_class_maps", raw_maps), ("positive_labels", positive)):
        path = cache_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        cache_files[name] = path
    for graph in GRAPHS:
        for kind, arrays in (("smoothed", primary_smooth), ("residual", primary_residual), ("stability", primary_stability)):
            path = cache_dir / f"{kind}_{graph}.npy"
            np.save(path, arrays[graph], allow_pickle=False)
            cache_files[f"{kind}_{graph}"] = path
    np.save(cache_dir / "image_ids.npy", image_ids, allow_pickle=False)
    np.save(cache_dir / "semantic_labels.npy", cache["semantic_labels"], allow_pickle=False)
    np.save(cache_dir / "image_label_counts.npy", cache["image_label_counts"], allow_pickle=False)
    cache_files.update({
        "image_ids": cache_dir / "image_ids.npy", "semantic_labels": cache_dir / "semantic_labels.npy",
        "image_label_counts": cache_dir / "image_label_counts.npy",
    })
    cache_manifest = {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in cache_files.items()}
    json_dump(cache_dir / "cache_manifest.json", cache_manifest)
    extraction = {
        "model": model_metadata,
        "dataset": {"name": "PASCAL VOC 2012 val", "image_count": count, "input_size": 448, "patch_grid": [28, 28], "patch_count": 784},
        "graph": {"definitions": "exact immutable Phase B B1/B2/B3 weights", "edge_count": edges.count, "gt_used_in_graph_construction": False, "gt_used_in_smoothing": False},
        "solver": {"method": "batched sparse-edge conjugate gradient", "tolerance": SOLVER_TOLERANCE, "required_relative_residual": REQUIRED_RESIDUAL, "maximum_observed_relative_residual": max_residual, "nonfinite_residual_count": nonfinite_count, "internal_tolerance_unmet_rhs_count": internal_tolerance_unmet},
        "phase_b_feature_weight_max_abs_reproduction_error": max_graph_difference,
        "duration_seconds": elapsed_total,
        "raw_cache": {"files": cache_manifest, "large_raw_cache_committed": False},
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return extraction, raw_maps, primary_smooth, primary_stability, numerical


def _basis_invariance(
    *, args: argparse.Namespace, edges: LocalEdges, weights: Mapping[str, np.ndarray],
    raw_maps: np.ndarray, primary_smooth: Mapping[str, np.ndarray], primary_stability: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    phase_a = args.phase_b_dir.parent.parent / "phase_a_basis_dependence" / "20260907-phase-a-full-1a54d71"
    archive = phase_a / "raw_outputs" / "basis_matrices_d384.npz"
    if not archive.is_file():
        raise FileNotFoundError("Phase C basis regression requires the immutable full Phase A D=384 archive")
    with np.load(archive, allow_pickle=False) as data:
        names = list(data.files)
        matrices = [data[name] for name in names]
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=max(2, args.limit) if args.limit else 2)
    if len(dataset) < 2:
        raise ValueError("basis regression requires at least two deterministic VOC images")
    images = torch.stack([dataset[index]["image"] for index in range(2)]).to(device)
    model, _ = load_patch_final_mctformer(args.mct_checkpoint)
    model.to(device).eval()
    records: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        _, patches, _, _, _ = model.forward_features(images, return_aux=True)
        grid = patches.reshape(2, 28, 28, 384).permute(0, 3, 1, 2).contiguous()
        # Both sides of the exact reparameterization control must use the
        # same float64 convolution.  Casting a float32 baseline *after* its
        # convolution would create an identity-transform rounding artifact.
        base_m = F.conv2d(
            grid.double(), model.head.weight.detach().double(),
            model.head.bias.detach().double(), padding=1,
        )[:, :20].flatten(2).transpose(1, 2)
        for name, matrix in zip(names, matrices):
            rotation = torch.from_numpy(matrix).to(device=device, dtype=torch.float64)
            transformed_patches = patches[:2].double() @ rotation
            transformed_weight = transform_conv_input_basis(model.head.weight.detach().double(), rotation)
            transformed_grid = transformed_patches.reshape(2, 28, 28, 384).permute(0, 3, 1, 2).contiguous()
            transformed_m = F.conv2d(transformed_grid, transformed_weight, model.head.bias.detach().double(), padding=1)[:, :20].flatten(2).transpose(1, 2)
            record: dict[str, float] = {"max_abs_m_error": float((transformed_m - base_m).abs().max().item())}
            base_relevance = torch.relu(base_m)
            base_relevance = base_relevance / base_relevance.amax(dim=1, keepdim=True).clamp_min(1e-12)
            transformed_relevance = torch.relu(transformed_m)
            transformed_relevance = transformed_relevance / transformed_relevance.amax(dim=1, keepdim=True).clamp_min(1e-12)
            record["max_abs_relevance_error"] = float((transformed_relevance - base_relevance).abs().max().item())
            for graph in GRAPHS:
                local = torch.from_numpy(weights[graph][:2]).to(device=device, dtype=torch.float64)
                base = graph_lowpass(base_m, local, edges, lambda_value=1.0, tolerance=SOLVER_TOLERANCE, max_iterations=256)
                altered = graph_lowpass(transformed_m, local, edges, lambda_value=1.0, tolerance=SOLVER_TOLERANCE, max_iterations=256)
                _, _, base_s = bounded_stability(base_m, base.values)
                _, _, altered_s = bounded_stability(transformed_m, altered.values)
                record[f"max_abs_mbar_error_{graph}"] = float((altered.values - base.values).abs().max().item())
                record[f"max_abs_stability_error_{graph}"] = float((altered_s - base_s).abs().max().item())
                record[f"max_abs_relevance_stability_error_{graph}"] = float(
                    (transformed_relevance * altered_s - base_relevance * base_s).abs().max().item()
                )
            records[name] = record
    del model
    maximum = max(value for row in records.values() for value in row.values())
    result = {"schema_version": 1, "subset_rule": "first two deterministic VOC val images", "subset_image_ids": list(dataset.image_ids[:2]), "transforms": records, "tolerance": 1e-5, "maximum_error": maximum, "passed": maximum < 1e-5}
    if not result["passed"]:
        raise RuntimeError(f"Phase C basis invariance failed: {result}")
    return result


def _render_examples(*, output_dir: Path, dataset, raw_maps: np.ndarray, smooth: Mapping[str, np.ndarray], stability: Mapping[str, np.ndarray], semantic_labels: np.ndarray, positive: np.ndarray, label_counts: np.ndarray) -> list[dict[str, object]]:
    selected: list[tuple[int, int, str]] = []
    for name, predicate in (("single_label", label_counts == 1), ("exactly_2_labels", label_counts == 2), ("3plus_labels", label_counts >= 3)):
        eligible = np.flatnonzero(predicate)
        if eligible.size:
            index = int(eligible[0]); selected.append((index, int(np.flatnonzero(positive[index])[0]), name))
    # Deterministic extreme object-area controls, excluding duplicated images.
    areas = np.zeros(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        labels = semantic_labels[index]
        areas[index] = int((labels > 0).sum())
    for name, order in (("small_foreground", np.argsort(areas)), ("large_foreground", np.argsort(-areas))):
        for index in order:
            if areas[index] and all(index != previous[0] for previous in selected):
                selected.append((int(index), int(np.flatnonzero(positive[index])[0]), name)); break
    selected = selected[:5]
    manifest: list[dict[str, object]] = []
    for image_index, class_id, reason in selected:
        sample = dataset[image_index]
        rgb = sample["image"].cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb * np.asarray((0.229, 0.224, 0.225)) + np.asarray((0.485, 0.456, 0.406)), 0, 1)
        labels = semantic_labels[image_index].reshape(28, 28).copy(); labels[labels < 0] = 21
        raw = raw_maps[image_index, class_id].reshape(28, 28)
        smoothed = smooth["B2_feature"][image_index, class_id].reshape(28, 28)
        residual = np.abs(raw - smoothed)
        score = stability["B2_feature"][image_index, class_id].reshape(28, 28)
        relevance = np.maximum(raw, 0.0); relevance /= max(float(relevance.max()), 1e-12)
        combined = relevance * score
        panels = (("RGB", rgb), ("Patch GT", labels), ("raw M", raw), ("B2 M_bar", smoothed), ("|M-M_bar|", residual), ("Stability S", score), ("Relevance R", relevance), ("R×S", combined))
        figure, axes = plt.subplots(2, 4, figsize=(15, 7))
        for axis, (title, data) in zip(axes.flat, panels):
            axis.imshow(data, cmap=None if title == "RGB" else "magma")
            axis.set_title(title); axis.axis("off")
        figure.suptitle(f"{dataset.image_ids[image_index]} · {VOC_CLASS_NAMES[class_id]} · {reason}")
        figure.tight_layout()
        path = output_dir / f"{dataset.image_ids[image_index]}_{VOC_CLASS_NAMES[class_id]}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight"); plt.close(figure)
        manifest.append({"image_id": str(dataset.image_ids[image_index]), "class_id": class_id, "class_name": VOC_CLASS_NAMES[class_id], "selection_rule": reason, "path": str(path), "sha256": sha256_file(path)})
    return manifest


def _format(value: object) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.6f}"


def _lookup(rows: list[dict[str, object]], graph: str, lambda_value: float, metric: str) -> dict[str, object]:
    found = [row for row in rows if row.get("graph") == graph and row.get("lambda") == lambda_value and row.get("metric") == metric and row.get("aggregation") == "micro" and row.get("stratum") == "all" and row.get("scope") == "all_positive_classes"]
    if len(found) != 1:
        raise RuntimeError(f"ambiguous numerical result for {graph}/{lambda_value}/{metric}: {len(found)}")
    return found[0]


def _write_report(output_dir: Path, extraction: Mapping[str, object], numerical_summary: list[dict[str, object]], region_summary: list[dict[str, object]], basis: Mapping[str, object]) -> None:
    lines = [
        "# Graph Stability Diagnostic Report", "", "## Frozen contract", "",
        f"- Frozen matched PatchFinalLN MCTformer+-Small checkpoint SHA256 `{extraction['model']['checkpoint_sha256']}`.",
        f"- Full deterministic VOC 2012 val: `{extraction['dataset']['image_count']}` images, 448 input, 28×28 patches.",
        "- Raw pre-ReLU 3×3 classifier response M is the only low-pass signal. B1/B2/B3 graph weights are reproduced from immutable Phase B; GT is used only after smoothing for diagnostic labels.",
        "- Low-pass solves (I+lambda L_sym) X=M with batched sparse-edge conjugate gradients; no matrix inverse, filtering of graph edges, training, selector, or pooling was used.", "",
        "## Low-pass numerical and preservation checks", "",
        "| Graph | λ | Fidelity | Energy retention | Pearson(M,M_bar) | Spearman(M,M_bar) |", "|---|---:|---:|---:|---:|---:|",
    ]
    for graph in GRAPHS:
        for lambda_value in LAMBDA_VALUES:
            values = [_lookup(numerical_summary, graph, lambda_value, metric)["estimate"] for metric in ("fidelity", "energy_retention", "map_pearson", "map_spearman")]
            lines.append(f"| {graph} | {lambda_value:g} | " + " | ".join(_format(value) for value in values) + " |")
    primary = [row for row in region_summary if row.get("graph") == "B2_feature" and row.get("lambda") == 1.0 and row.get("score") == "stability" and row.get("aggregation") == "micro" and row.get("stratum") == "all" and row.get("scope") == "all_positive_classes"]
    lookup = {row["metric"]: row for row in primary}
    lines.extend(["", "## Stability-alone semantic diagnostic (B2, λ=1)", "", "| Target-vs-BG AUROC | Target-vs-other-FG AUROC | C-PiM target hit | BG tail enrichment@10% |", "|---:|---:|---:|---:|", "| " + " | ".join(_format(lookup.get(metric, {}).get("estimate")) for metric in ("auc_target_bg", "auc_target_other", "target_hit", "bg_tail_enrich_10")) + " |", "", "## Interpretation boundary", "", "Graph stability is a frozen structural diagnostic. Its semantic region scores do not establish a selector, CAM improvement, causal intervention, or proposed method. Phase D tests whether it is useful only after class-specific relevance is supplied.", "", f"The maximum basis-regression error was `{basis['maximum_error']:.3e}` (required <1e-5); the maximum final linear-solve residual was `{extraction['solver']['maximum_observed_relative_residual']:.3e}` (required <1e-5).", ""])
    text_dump(output_dir / "03_GRAPH_STABILITY_DIAGNOSTIC_REPORT.md", "\n".join(lines))


def main() -> None:
    args = parse_args(); require_environment()
    args.repo_root = args.repo_root.expanduser().resolve()
    for name in ("output_dir", "phase_b_dir", "mct_checkpoint", "voc_root", "list_path"):
        setattr(args, name, _resolve(args.repo_root, getattr(args, name)))
    if args.test_log:
        args.test_log = _resolve(args.repo_root, args.test_log)
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0 or args.bootstrap_repeats < 1:
        raise ValueError("invalid Phase C batch/bootstrap/limit argument")
    if not args.limit and (args.bootstrap_repeats != BOOTSTRAP_REPEATS or args.bootstrap_seed != BOOTSTRAP_SEED):
        raise ValueError("full Phase C requires exactly 5,000 resamples and seed 20260901")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase C requires CUDA")
    git = require_clean_tracked(args.repo_root)
    cache, phase_b_linkage = _load_phase_b(args.phase_b_dir, args.limit)
    output_dir = create_output(args.output_dir)
    visual_dir = output_dir / "selected_visualizations"; visual_dir.mkdir()
    log = RunLog(output_dir / "run.log"); log(f"Phase C started: {command_line()}")
    before = {"checkpoint": sha256_file(args.mct_checkpoint), "voc_val_list": sha256_file(args.list_path), "phase_b_cache": phase_b_linkage["cache_sha256"], "phase_b_completion": phase_b_linkage["completion_sha256"]}
    manifests = write_environment_manifests(output_dir)
    edges = local_edge_index((28, 28)); weights = _graph_weights(cache, edges)
    extraction, raw_maps, primary_smooth, primary_stability, numerical = _extract_and_analyze(args=args, output_dir=output_dir, cache=cache, edges=edges, weights=weights, device=device, log=log)
    log("Phase C low-pass solves complete; computing image-clustered semantic diagnostics")
    numerical_summary = summarize_simple_rows(numerical, value_cols=("fidelity", "relative_l2_change", "map_pearson", "map_spearman", "energy_raw", "energy_smoothed", "energy_retention", "smoothed_std_over_raw", "linear_solve_relative_residual", "cg_iterations", "solve_runtime_ms_per_map"), repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    # Keep fidelity and energy as separate required tables while preserving the complete lambda curve.
    fidelity_rows = [row for row in numerical_summary if row["metric"] in ("fidelity", "relative_l2_change", "map_pearson", "map_spearman", "smoothed_std_over_raw")]
    energy_rows = [row for row in numerical_summary if row["metric"] in ("energy_raw", "energy_smoothed", "energy_retention", "linear_solve_relative_residual", "cg_iterations", "solve_runtime_ms_per_map")]
    semantics = np.asarray(cache["semantic_labels"], dtype=np.int8)
    counts = np.asarray(cache["image_label_counts"], dtype=np.uint8)
    image_ids = [str(value) for value in cache["image_ids"]]
    bdist = boundary_distances(semantics, edges); variance = local_map_variance(raw_maps, edges)
    all_region_rows: list[dict[str, object]] = []; distributions: list[dict[str, object]] = []; boundary_deltas: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []; negatives: list[dict[str, object]] = []; correlations: list[dict[str, object]] = []
    for graph in GRAPHS:
        stable = primary_stability[graph]
        all_region_rows.extend(map_region_rows(image_ids=image_ids, positive_labels=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), semantic_labels=semantics, image_label_counts=counts, score_maps={"stability": stable}, graph=graph, lambda_value=1.0))
        dist_rows, diff_rows = region_distribution_rows(image_ids=image_ids, positive_labels=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), semantic_labels=semantics, image_label_counts=counts, scores=stable, graph=graph, lambda_value=1.0, edges=edges)
        distributions.extend(dist_rows); boundary_deltas.extend(diff_rows)
        overlaps.extend(class_overlap_rows(image_ids=image_ids, positive_labels=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), semantic_labels=semantics, image_label_counts=counts, scores=stable, graph=graph, lambda_value=1.0))
        negatives.extend(negative_class_rows(image_ids=image_ids, positive_labels=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), semantic_labels=semantics, image_label_counts=counts, scores=stable, graph=graph, lambda_value=1.0, score_name="stability"))
        correlations.extend(correlation_rows(image_ids=image_ids, positive_labels=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), semantic_labels=semantics, image_label_counts=counts, stability=stable, raw_maps=raw_maps, node_degree=_degree_from_weights(weights[graph], edges), boundary_distance=bdist, local_variance=variance, graph=graph, lambda_value=1.0, edges=edges))
    region_summary = summarize_region_rows(all_region_rows, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    distribution_summary = summarize_simple_rows(distributions, value_cols=("mean_stability", "median_stability", "q25_stability", "q75_stability"), repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, group_columns=("graph", "lambda", "region", "location"))
    delta_summary = summarize_simple_rows(boundary_deltas, value_cols=("estimate",), repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, group_columns=("graph", "lambda", "region", "metric"))
    for row in delta_summary:
        row["metric"] = f"{row['metric']}_summary"
    overlap_summary = summarize_simple_rows(overlaps, value_cols=("stability_spearman", "top10_jaccard", "top20_jaccard", "shared_target_either_fraction", "shared_background_fraction", "shared_other_foreground_fraction"), repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    negative_frame = pd.DataFrame(negatives); negative_summary: list[dict[str, object]] = []
    for keys, subset in negative_frame.groupby(["graph", "lambda", "score", "presence"], sort=True):
        negative_summary.extend(summarize_clustered(subset, value_cols=("mean_score", "top10_mean_score", "max_score", "spatial_entropy", "high_score_fraction"), identity={"graph": keys[0], "lambda": keys[1], "score": keys[2], "presence": keys[3]}, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed))
    correlation_frame = pd.DataFrame(correlations); correlation_summary: list[dict[str, object]] = []
    for keys, subset in correlation_frame.groupby(["graph", "lambda", "stratum", "quantity"], sort=True):
        correlation_summary.extend(summarize_clustered(subset, value_cols=("spearman_stability_vs_quantity",), identity={"graph": keys[0], "lambda": keys[1], "spatial_stratum": keys[2], "quantity": keys[3]}, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed))
    basis = _basis_invariance(args=args, edges=edges, weights=weights, raw_maps=raw_maps, primary_smooth=primary_smooth, primary_stability=primary_stability, device=device)
    json_dump(output_dir / "basis_invariance.json", basis)
    csv_dump(output_dir / "graph_lowpass_fidelity.csv", fidelity_rows)
    csv_dump(output_dir / "graph_energy_metrics.csv", energy_rows)
    csv_dump(output_dir / "lambda_sensitivity.csv", numerical_summary)
    csv_dump(output_dir / "stability_region_metrics.csv", region_summary)
    csv_dump(output_dir / "stability_boundary_metrics.csv", distribution_summary + delta_summary, fields=list(distribution_summary[0].keys()))
    csv_dump(output_dir / "stability_class_overlap.csv", overlap_summary)
    csv_dump(output_dir / "stability_negative_class_control.csv", negative_summary)
    csv_dump(output_dir / "stability_correlations.csv", correlation_summary)
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit)
    visualization_manifest = _render_examples(output_dir=visual_dir, dataset=dataset, raw_maps=raw_maps, smooth=primary_smooth, stability=primary_stability, semantic_labels=semantics, positive=np.asarray(np.load(output_dir / "raw_outputs" / "positive_labels.npy", allow_pickle=False)), label_counts=counts)
    json_dump(visual_dir / "selection_manifest.json", {"examples": visualization_manifest})
    _write_report(output_dir, extraction, numerical_summary, region_summary, basis)
    after = {"checkpoint": sha256_file(args.mct_checkpoint), "voc_val_list": sha256_file(args.list_path), "phase_b_cache": sha256_file(args.phase_b_dir / "graph_analysis_cache.npz"), "phase_b_completion": sha256_file(args.phase_b_dir / "completion.json")}
    if before != after:
        raise RuntimeError("an immutable Phase C input changed during analysis")
    metadata = {"schema_version": 1, "phase": "C", "status": "complete", "finished_at": timestamp(), "command": command_line(), "git": git, "environment": manifests, "test_log": ({"path": str(args.test_log), "sha256": sha256_file(args.test_log)} if args.test_log else None), "input_hashes_before": before, "input_hashes_after": after, "source_immutable": before == after, "phase_b_linkage": phase_b_linkage, "extraction": extraction, "basis_invariance": basis, "statistics": {"bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.bootstrap_seed, "bootstrap_unit": "whole image; all class rows belonging to a resampled image retained"}, "raw_cache": extraction["raw_cache"], "output_tables": {name: {"sha256": sha256_file(output_dir / name)} for name in ("graph_lowpass_fidelity.csv", "graph_energy_metrics.csv", "lambda_sensitivity.csv", "stability_region_metrics.csv", "stability_boundary_metrics.csv", "stability_class_overlap.csv", "stability_negative_class_control.csv", "stability_correlations.csv")}}
    json_dump(output_dir / "run_metadata.json", metadata)
    json_dump(output_dir / "completion.json", {"status": "complete", "phase": "C", "finished_at": timestamp(), "image_count": len(image_ids), "bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.bootstrap_seed, "basis_invariance_passed": basis["passed"], "maximum_solver_residual": extraction["solver"]["maximum_observed_relative_residual"]})
    text_dump(output_dir / "exact_commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\n" + command_line() + "\n")
    (output_dir / "exact_commands.sh").chmod(0o755)
    log("Phase C complete")


if __name__ == "__main__":
    main()
