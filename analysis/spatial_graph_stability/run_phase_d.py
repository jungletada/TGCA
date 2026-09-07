#!/usr/bin/env python
"""Phase D: frozen Relevance x Graph-Stability decomposition (no pooling)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import summarize_clustered
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES
from analysis.lazy_assignment.experiment2.metrics_region import jaccard, stable_topk_mask

from .frozen import build_mct_dataset
from .graph import LocalEdges, PATCH_LABEL_VOID, boundary_patch_mask, local_edge_index
from .phase_metrics import (
    PRIMARY_REGION_METRICS,
    class_overlap_rows,
    class_region_labels,
    map_region_rows,
    negative_class_rows,
    paired_ratio_difference,
    ratio_summary,
    relevance_maps,
    summarize_region_rows,
    summarize_simple_rows,
    valid_patch_mask,
    within_map_quintiles,
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
from .run_phase_c import GRAPHS, _render_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-c-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument("--list-path", type=Path, default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"))
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--limit", type=int, default=0, help="smoke-only deterministic prefix")
    parser.add_argument("--test-log", type=Path)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _load_phase_c(phase_c_dir: Path, limit: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    completion = phase_c_dir / "completion.json"; metadata = phase_c_dir / "run_metadata.json"; basis = phase_c_dir / "basis_invariance.json"
    raw_dir = phase_c_dir / "raw_outputs"; manifest = raw_dir / "cache_manifest.json"
    needed = [completion, metadata, basis, manifest, *(raw_dir / f"{name}.npy" for name in ("raw_class_maps", "positive_labels", "image_ids", "semantic_labels", "image_label_counts"))]
    needed += [raw_dir / f"stability_{graph}.npy" for graph in GRAPHS]
    if any(not path.is_file() for path in needed):
        raise FileNotFoundError("Phase D requires a complete immutable Phase C signal cache")
    marker = json.loads(completion.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("phase") != "C":
        raise RuntimeError("invalid Phase C completion marker")
    with manifest.open(encoding="utf-8") as stream:
        cache_manifest = json.load(stream)
    arrays: dict[str, np.ndarray] = {}
    for name in ("raw_class_maps", "positive_labels", "image_ids", "semantic_labels", "image_label_counts"):
        arrays[name] = np.load(raw_dir / f"{name}.npy", mmap_mode="r")
    for graph in GRAPHS:
        arrays[f"stability_{graph}"] = np.load(raw_dir / f"stability_{graph}.npy", mmap_mode="r")
    total = len(arrays["image_ids"]); count = int(limit or total)
    if count > total:
        raise ValueError("Phase D limit exceeds immutable Phase C cache")
    arrays = {name: value[:count] if value.ndim and value.shape[0] == total else value for name, value in arrays.items()}
    linkage = {"phase_c_dir": str(phase_c_dir), "completion_sha256": sha256_file(completion), "metadata_sha256": sha256_file(metadata), "basis_sha256": sha256_file(basis), "cache_manifest_sha256": sha256_file(manifest), "cache_manifest": cache_manifest}
    return arrays, linkage


def _finite_auc(values: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    include = positive | negative
    y = positive[include].astype(np.uint8); x = values[include]
    if y.size == 0 or y.min() == y.max() or not np.isfinite(x).all():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, x)), float(average_precision_score(y, x))


def _grid_rows(*, image_ids: Sequence[str], raw: np.ndarray, positive: np.ndarray, semantic: np.ndarray, counts: np.ndarray, stability: np.ndarray, graph: str) -> list[dict[str, object]]:
    relevance = relevance_maps(raw)
    rows: list[dict[str, object]] = []
    for image_index, class_id in zip(*np.nonzero(positive)):
        valid = valid_patch_mask(semantic[image_index])
        rq = within_map_quintiles(relevance[image_index, class_id], valid)
        sq = within_map_quintiles(stability[image_index, class_id], valid)
        regions = class_region_labels(semantic[image_index], int(class_id))
        for relevance_q in range(5):
            for stability_q in range(5):
                chosen = valid & (rq == relevance_q) & (sq == stability_q)
                rows.append({
                    "image_id": str(image_ids[image_index]), "image_index": int(image_index),
                    "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "num_positive_classes": int(counts[image_index]), "graph": graph,
                    "relevance_quintile": relevance_q, "stability_quintile": stability_q,
                    "num_patches": int(chosen.sum()), "target_count": int((chosen & (regions == 0)).sum()),
                    "other_fg_count": int((chosen & (regions == 1)).sum()), "background_count": int((chosen & (regions == 2)).sum()),
                })
    return rows


def _summarize_grid(rows: list[dict[str, object]], image_ids: Sequence[str], repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows); output: list[dict[str, object]] = []
    for (graph, rq, sq), subset in frame.groupby(["graph", "relevance_quintile", "stability_quintile"], sort=True):
        for label, numerator in (("target", "target_count"), ("other_fg", "other_fg_count"), ("background", "background_count")):
            output.append(ratio_summary(subset, image_ids=image_ids, numerator=numerator, denominator="num_patches", identity={"graph": graph, "relevance_quintile": int(rq), "stability_quintile": int(sq), "region": label, "metric": "region_probability"}, repeats=repeats, seed=seed))
    return output


def _conditional_rows(*, image_ids: Sequence[str], raw: np.ndarray, positive: np.ndarray, semantic: np.ndarray, counts: np.ndarray, stability: np.ndarray, graph: str) -> list[dict[str, object]]:
    relevance = relevance_maps(raw); rows: list[dict[str, object]] = []
    for image_index, class_id in zip(*np.nonzero(positive)):
        valid = valid_patch_mask(semantic[image_index]); regions = class_region_labels(semantic[image_index], int(class_id))
        rq = within_map_quintiles(relevance[image_index, class_id], valid); sq = within_map_quintiles(stability[image_index, class_id], valid)
        for relevance_q in range(5):
            band = valid & (rq == relevance_q); high = band & (sq == 4); low = band & (sq == 0)
            auc_bg, ap_bg = _finite_auc(stability[image_index, class_id][band], (regions[band] == 0), (regions[band] == 2))
            auc_other, ap_other = _finite_auc(stability[image_index, class_id][band], (regions[band] == 0), (regions[band] == 1))
            rows.append({
                "image_id": str(image_ids[image_index]), "image_index": int(image_index), "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                "num_positive_classes": int(counts[image_index]), "graph": graph, "relevance_quintile": relevance_q,
                "high_target": int((high & (regions == 0)).sum()), "high_count": int(high.sum()), "low_target": int((low & (regions == 0)).sum()), "low_count": int(low.sum()),
                "stability_auc_target_bg": auc_bg, "stability_ap_target_bg": ap_bg, "stability_auc_target_other": auc_other, "stability_ap_target_other": ap_other,
            })
    return rows


def _summarize_conditional(rows: list[dict[str, object]], image_ids: Sequence[str], repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows); result: list[dict[str, object]] = []
    for (graph, relevance_q), subset in frame.groupby(["graph", "relevance_quintile"], sort=True):
        for level, numerator, denominator in (("high", "high_target", "high_count"), ("low", "low_target", "low_count")):
            result.append(ratio_summary(subset, image_ids=image_ids, numerator=numerator, denominator=denominator, identity={"row_type": "target_rate", "graph": graph, "relevance_quintile": int(relevance_q), "stability_level": level, "metric": "target_probability"}, repeats=repeats, seed=seed))
        per_image = subset.groupby("image_id", sort=False)[["high_target", "high_count", "low_target", "low_count"]].sum().reindex(list(image_ids), fill_value=0.0)
        delta, low, high, finite = paired_ratio_difference(image_ids=image_ids, first_numerator=per_image["high_target"].to_numpy(), first_denominator=per_image["high_count"].to_numpy(), second_numerator=per_image["low_target"].to_numpy(), second_denominator=per_image["low_count"].to_numpy(), repeats=repeats, seed=seed)
        result.append({"row_type": "paired_high_minus_low", "graph": graph, "relevance_quintile": int(relevance_q), "stability_level": "high_minus_low", "metric": "target_probability", "estimate": delta, "ci_low": low, "ci_high": high, "num_images": int((per_image[["high_count", "low_count"]].sum(axis=1) > 0).sum()), "num_images_total": len(image_ids), "bootstrap_repeats": repeats, "bootstrap_valid_repeats": finite, "bootstrap_seed": seed, "bootstrap_unit": "whole image"})
        result.extend(summarize_clustered(subset, value_cols=("stability_auc_target_bg", "stability_ap_target_bg", "stability_auc_target_other", "stability_ap_target_other"), identity={"row_type": "within_relevance_band_auc", "graph": graph, "relevance_quintile": int(relevance_q), "stability_level": "continuous", "metric_group": "stability"}, repeats=repeats, seed=seed))
    return result


def _quadrant_rows(*, image_ids: Sequence[str], raw: np.ndarray, positive: np.ndarray, semantic: np.ndarray, counts: np.ndarray, stability: np.ndarray, graph: str, edges: LocalEdges) -> list[dict[str, object]]:
    relevance = relevance_maps(raw); rows: list[dict[str, object]] = []
    for image_index, class_id in zip(*np.nonzero(positive)):
        valid = valid_patch_mask(semantic[image_index]); regions = class_region_labels(semantic[image_index], int(class_id))
        boundary = boundary_patch_mask(semantic[image_index].reshape(28, 28), edges).reshape(-1)
        rq = within_map_quintiles(relevance[image_index, class_id], valid); sq = within_map_quintiles(stability[image_index, class_id], valid)
        quadrants = {"Q1_highR_highS": (rq == 4) & (sq == 4), "Q2_highR_lowS": (rq == 4) & (sq <= 1), "Q3_lowR_highS": (rq <= 1) & (sq == 4), "Q4_lowR_lowS": (rq <= 1) & (sq <= 1)}
        for quadrant, mask in quadrants.items():
            chosen = valid & mask
            rows.append({"image_id": str(image_ids[image_index]), "image_index": int(image_index), "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)], "num_positive_classes": int(counts[image_index]), "graph": graph, "quadrant": quadrant, "num_patches": int(chosen.sum()), "target_count": int((chosen & (regions == 0)).sum()), "other_fg_count": int((chosen & (regions == 1)).sum()), "background_count": int((chosen & (regions == 2)).sum()), "boundary_count": int((chosen & boundary).sum())})
    return rows


def _summarize_quadrants(rows: list[dict[str, object]], image_ids: Sequence[str], repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows); output: list[dict[str, object]] = []
    for (graph, quadrant), subset in frame.groupby(["graph", "quadrant"], sort=True):
        for label, numerator in (("target", "target_count"), ("other_fg", "other_fg_count"), ("background", "background_count"), ("boundary_near", "boundary_count")):
            output.append(ratio_summary(subset, image_ids=image_ids, numerator=numerator, denominator="num_patches", identity={"graph": graph, "quadrant": quadrant, "region": label, "metric": "quadrant_fraction"}, repeats=repeats, seed=seed))
    return output


def _suppression_rows(*, image_ids: Sequence[str], raw: np.ndarray, positive: np.ndarray, semantic: np.ndarray, counts: np.ndarray, stability: np.ndarray, graph: str, edges: LocalEdges) -> list[dict[str, object]]:
    relevance = relevance_maps(raw); combined = relevance * stability; rows: list[dict[str, object]] = []
    for image_index, class_id in zip(*np.nonzero(positive)):
        valid = valid_patch_mask(semantic[image_index]); regions = class_region_labels(semantic[image_index], int(class_id)); boundary = boundary_patch_mask(semantic[image_index].reshape(28, 28), edges).reshape(-1)
        r, q, s = relevance[image_index, class_id], combined[image_index, class_id], stability[image_index, class_id]
        masks = {"target": regions == 0, "other_foreground": regions == 1, "background": regions == 2, "target_interior": (regions == 0) & ~boundary, "target_boundary": (regions == 0) & boundary, "fg_bg_boundary": boundary & ((regions == 0) | (regions == 1) | (regions == 2))}
        base = {"image_id": str(image_ids[image_index]), "image_index": int(image_index), "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)], "num_positive_classes": int(counts[image_index]), "graph": graph}
        for region, mask in masks.items():
            chosen = valid & mask
            rows.append({**base, "row_type": "suppression", "region": region, "mean_score_delta_rs_minus_r": float((q[chosen] - r[chosen]).mean()) if chosen.any() else float("nan"), "mean_relative_suppression": float((1.0 - s[chosen]).mean()) if chosen.any() else float("nan")})
        for ratio in (0.05, 0.10, 0.20):
            r_top = stable_topk_mask(r, ratio, valid); q_top = stable_topk_mask(q, ratio, valid)
            target = regions == 0; background = regions == 2
            rows.append({**base, "row_type": "top_tail_transition", "region": f"top{int(ratio*100):02d}", "target_retained_fraction": float((r_top & q_top & target).sum() / max(1, (r_top & target).sum())), "background_removed_fraction": float((r_top & ~q_top & background).sum() / max(1, (r_top & background).sum()))})
    return rows


def _multi_label_rows(*, image_ids: Sequence[str], positive: np.ndarray, semantic: np.ndarray, counts: np.ndarray, score_maps: Mapping[str, np.ndarray], graph: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score, maps in score_maps.items():
        base_rows = class_overlap_rows(image_ids=image_ids, positive_labels=positive, semantic_labels=semantic, image_label_counts=counts, scores=maps, graph=graph, lambda_value=1.0)
        for row in base_rows:
            row["score"] = score
            first, second = int(row["class_id"]), int(row["other_class_id"])
            image_index = int(row["image_index"]); valid = valid_patch_mask(semantic[image_index]); shared = stable_topk_mask(maps[image_index, first], 0.10, valid) & stable_topk_mask(maps[image_index, second], 0.10, valid)
            labels = semantic[image_index]; n = int(shared.sum())
            first_count, second_count = int((shared & (labels == first + 1)).sum()), int((shared & (labels == second + 1)).sum())
            row["dominant_object_capture"] = float(max(first_count, second_count) / n) if n else float("nan")
            rows.append(row)
    return rows


def _paired_comparisons(region_rows: list[dict[str, object]], *, repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(region_rows); result: list[dict[str, object]] = []
    keys = ["image_id", "image_index", "class_id", "class_name", "num_positive_classes", "label_stratum", "graph", "lambda"]
    metrics = ("target_hit", "target_tail_enrich_05", "target_tail_enrich_10", "bg_tail_enrich_05", "auc_target_bg", "auc_target_other", "ap_target_bg", "ap_target_other")
    for graph, subset in frame.groupby("graph", sort=True):
        left = subset[subset["score"] == "relevance"][keys + list(metrics)].copy(); right = subset[subset["score"] == "relevance_x_stability"][keys + list(metrics)].copy()
        merged = left.merge(right, on=keys, suffixes=("_r", "_rs"), validate="one_to_one")
        for metric in metrics:
            right_values = pd.to_numeric(merged[f"{metric}_rs"], errors="coerce").to_numpy(dtype=np.float64)
            left_values = pd.to_numeric(merged[f"{metric}_r"], errors="coerce").to_numpy(dtype=np.float64)
            merged[metric] = right_values - left_values
        for stratum, scoped in (("all", merged), ("single_label", merged[merged["label_stratum"] == "single_label"]), ("exactly_2_labels", merged[merged["label_stratum"] == "exactly_2_labels"]), ("3plus_labels", merged[merged["label_stratum"] == "3plus_labels"])):
            result.extend(summarize_clustered(scoped, value_cols=metrics, identity={"comparison": "relevance_x_stability_minus_relevance", "graph": graph, "stratum": stratum}, repeats=repeats, seed=seed))
    return result


def _fmt(value: object) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.6f}"


def _find(rows: list[dict[str, object]], *, graph: str, score: str, metric: str) -> object:
    matches = [row for row in rows if row.get("graph") == graph and row.get("score") == score and row.get("metric") == metric and row.get("scope") == "all_positive_classes" and row.get("aggregation") == "micro" and row.get("stratum") == "all"]
    return matches[0]["estimate"] if len(matches) == 1 else float("nan")


def _write_report(output_dir: Path, linkage: Mapping[str, object], metrics: list[dict[str, object]], comparisons: list[dict[str, object]], basis: Mapping[str, object]) -> None:
    lines = ["# Relevance–Stability Ablation Report", "", "## Frozen contract", "", "- Phase D reuses the immutable Phase C raw M and λ=1 B1/B2/B3 stability caches; no checkpoint, graph weight, model parameter, selector, pooling operator, or loss was changed.", "- Relevance R=ReLU(M)/max(ReLU(M)); S is bounded graph stability; R×S uses the fixed exponents 1 and 1.", "- GT appears only in diagnostic evaluation. Every interval uses whole-image clustered bootstrap resampling, never independent patches.", "", "## Primary micro semantic diagnostics", "", "| Graph | Score | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |", "|---|---|---:|---:|---:|---:|---:|"]
    for graph in GRAPHS:
        for score in ("relevance", "stability", "relevance_x_stability"):
            values = [_find(metrics, graph=graph, score=score, metric=metric) for metric in ("auc_target_bg", "auc_target_other", "target_tail_enrich_10", "bg_tail_enrich_10", "target_hit")]
            lines.append(f"| {graph} | {score} | " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(["", "## Paired R×S − R effects", "", "| Graph | Metric | Δ | 95% CI |", "|---|---|---:|---:|"])
    selected = [row for row in comparisons if row.get("aggregation") == "micro" and row.get("stratum") == "all" and row.get("metric") in ("target_tail_enrich_10", "bg_tail_enrich_10", "auc_target_bg", "auc_target_other")]
    for row in selected:
        lines.append(f"| {row['graph']} | {row['metric']} | {_fmt(row['estimate'])} | [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] |")
    lines.extend(["", "## Interpretation boundary", "", "These are frozen representation-level and semantic-diagnostic results. They test whether stability conditionally modifies existing relevance; they do not establish a trained pooling method, attention intervention, CAM improvement, or causal localization mechanism.", "", f"Phase C/D basis regression passed with maximum error `{basis['maximum_error']:.3e}` (<1e-5).", ""])
    text_dump(output_dir / "04_RELEVANCE_STABILITY_ABLATION_REPORT.md", "\n".join(lines))


def main() -> None:
    args = parse_args(); require_environment(); args.repo_root = args.repo_root.expanduser().resolve()
    for name in ("output_dir", "phase_c_dir", "voc_root", "list_path"):
        setattr(args, name, _resolve(args.repo_root, getattr(args, name)))
    if args.test_log: args.test_log = _resolve(args.repo_root, args.test_log)
    if args.limit < 0 or args.bootstrap_repeats < 1:
        raise ValueError("invalid Phase D limit/bootstrap argument")
    if not args.limit and (args.bootstrap_repeats != BOOTSTRAP_REPEATS or args.bootstrap_seed != BOOTSTRAP_SEED):
        raise ValueError("full Phase D requires exactly 5,000 resamples and seed 20260901")
    git = require_clean_tracked(args.repo_root)
    arrays, linkage = _load_phase_c(args.phase_c_dir, args.limit)
    if not args.limit and len(arrays["image_ids"]) != EXPECTED_VOC_IMAGES: raise RuntimeError("full Phase D requires all VOC val images")
    output_dir = create_output(args.output_dir); visual_dir = output_dir / "selected_visualizations"; visual_dir.mkdir(); log = RunLog(output_dir / "run.log"); log(f"Phase D started: {command_line()}")
    before = {"phase_c_completion": linkage["completion_sha256"], "phase_c_basis": linkage["basis_sha256"], "phase_c_cache_manifest": linkage["cache_manifest_sha256"], "voc_val_list": sha256_file(args.list_path)}; manifests = write_environment_manifests(output_dir)
    raw = np.asarray(arrays["raw_class_maps"], dtype=np.float32); positive = np.asarray(arrays["positive_labels"], dtype=bool); semantic = np.asarray(arrays["semantic_labels"], dtype=np.int8); counts = np.asarray(arrays["image_label_counts"], dtype=np.uint8); image_ids = [str(value) for value in arrays["image_ids"]]
    relevance = relevance_maps(raw); region_rows: list[dict[str, object]] = []; grid_raw: list[dict[str, object]] = []; conditional_raw: list[dict[str, object]] = []; quadrant_raw: list[dict[str, object]] = []; suppression_raw: list[dict[str, object]] = []; multi_raw: list[dict[str, object]] = []; absent_raw: list[dict[str, object]] = []
    edges = local_edge_index((28, 28))
    for graph in GRAPHS:
        stability = np.asarray(arrays[f"stability_{graph}"], dtype=np.float32); combined = relevance * stability
        score_maps = {"relevance": relevance, "stability": stability, "relevance_x_stability": combined}
        region_rows.extend(map_region_rows(image_ids=image_ids, positive_labels=positive, semantic_labels=semantic, image_label_counts=counts, score_maps=score_maps, graph=graph, lambda_value=1.0))
        grid_raw.extend(_grid_rows(image_ids=image_ids, raw=raw, positive=positive, semantic=semantic, counts=counts, stability=stability, graph=graph))
        conditional_raw.extend(_conditional_rows(image_ids=image_ids, raw=raw, positive=positive, semantic=semantic, counts=counts, stability=stability, graph=graph))
        quadrant_raw.extend(_quadrant_rows(image_ids=image_ids, raw=raw, positive=positive, semantic=semantic, counts=counts, stability=stability, graph=graph, edges=edges))
        suppression_raw.extend(_suppression_rows(image_ids=image_ids, raw=raw, positive=positive, semantic=semantic, counts=counts, stability=stability, graph=graph, edges=edges))
        multi_raw.extend(_multi_label_rows(image_ids=image_ids, positive=positive, semantic=semantic, counts=counts, score_maps=score_maps, graph=graph))
        for score, maps in score_maps.items():
            absent_raw.extend(negative_class_rows(image_ids=image_ids, positive_labels=positive, semantic_labels=semantic, image_label_counts=counts, scores=maps, graph=graph, lambda_value=1.0, score_name=score))
    log("Phase D generated R/S/R×S map-level diagnostics; computing clustered summaries")
    metrics = summarize_region_rows(region_rows, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    grid = _summarize_grid(grid_raw, image_ids, args.bootstrap_repeats, args.bootstrap_seed)
    conditional = _summarize_conditional(conditional_raw, image_ids, args.bootstrap_repeats, args.bootstrap_seed)
    quadrants = _summarize_quadrants(quadrant_raw, image_ids, args.bootstrap_repeats, args.bootstrap_seed)
    suppression_frame = pd.DataFrame(suppression_raw); suppression: list[dict[str, object]] = []
    for (graph, row_type, region), subset in suppression_frame.groupby(["graph", "row_type", "region"], sort=True):
        columns = ("mean_score_delta_rs_minus_r", "mean_relative_suppression") if row_type == "suppression" else ("target_retained_fraction", "background_removed_fraction")
        suppression.extend(summarize_clustered(subset, value_cols=columns, identity={"graph": graph, "row_type": row_type, "region": region}, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed))
    multi_frame = pd.DataFrame(multi_raw); multi: list[dict[str, object]] = []
    for (graph, score), subset in multi_frame.groupby(["graph", "score"], sort=True):
        multi.extend(summarize_clustered(subset, value_cols=("stability_spearman", "top10_jaccard", "top20_jaccard", "shared_target_either_fraction", "shared_background_fraction", "shared_other_foreground_fraction", "dominant_object_capture"), identity={"graph": graph, "score": score}, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed))
    absent_frame = pd.DataFrame(absent_raw); absent: list[dict[str, object]] = []
    for (graph, score, presence), subset in absent_frame.groupby(["graph", "score", "presence"], sort=True):
        absent.extend(summarize_clustered(subset, value_cols=("mean_score", "top10_mean_score", "max_score", "spatial_entropy", "high_score_fraction"), identity={"graph": graph, "score": score, "presence": presence}, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed))
    comparison = _paired_comparisons(region_rows, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    phase_c_basis = json.loads((args.phase_c_dir / "basis_invariance.json").read_text(encoding="utf-8"))
    required = [key for row in phase_c_basis.get("transforms", {}).values() for key in row if "relevance" in key]
    if not required or not phase_c_basis.get("passed", False): raise RuntimeError("Phase D requires a passing Phase C relevance/stability basis regression")
    json_dump(output_dir / "basis_invariance.json", {"phase_c_basis_path": str(args.phase_c_dir / "basis_invariance.json"), "phase_c_basis_sha256": linkage["basis_sha256"], "checks": phase_c_basis, "passed": True, "maximum_error": phase_c_basis["maximum_error"], "tolerance": 1e-5})
    csv_dump(output_dir / "relevance_stability_metrics.csv", metrics); csv_dump(output_dir / "relevance_stability_5x5_grid.csv", grid); csv_dump(output_dir / "conditional_stability_uplift.csv", conditional); csv_dump(output_dir / "quadrant_region_composition.csv", quadrants); csv_dump(output_dir / "false_positive_suppression.csv", suppression); csv_dump(output_dir / "multi_label_overlap.csv", multi); csv_dump(output_dir / "absent_class_control.csv", absent); csv_dump(output_dir / "graph_comparison.csv", comparison)
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit); smooth = {graph: np.asarray(np.load(args.phase_c_dir / "raw_outputs" / f"smoothed_{graph}.npy", mmap_mode="r")[:len(image_ids)]) for graph in GRAPHS}; stability_maps = {graph: np.asarray(arrays[f"stability_{graph}"]) for graph in GRAPHS}; manifest = _render_examples(output_dir=visual_dir, dataset=dataset, raw_maps=raw, smooth=smooth, stability=stability_maps, semantic_labels=semantic, positive=positive, label_counts=counts); json_dump(visual_dir / "selection_manifest.json", {"examples": manifest})
    basis = json.loads((output_dir / "basis_invariance.json").read_text(encoding="utf-8")); _write_report(output_dir, linkage, metrics, comparison, basis)
    after = {"phase_c_completion": sha256_file(args.phase_c_dir / "completion.json"), "phase_c_basis": sha256_file(args.phase_c_dir / "basis_invariance.json"), "phase_c_cache_manifest": sha256_file(args.phase_c_dir / "raw_outputs" / "cache_manifest.json"), "voc_val_list": sha256_file(args.list_path)}
    if before != after: raise RuntimeError("immutable Phase C input changed during Phase D")
    metadata = {"schema_version": 1, "phase": "D", "status": "complete", "finished_at": timestamp(), "command": command_line(), "git": git, "environment": manifests, "test_log": ({"path": str(args.test_log), "sha256": sha256_file(args.test_log)} if args.test_log else None), "input_hashes_before": before, "input_hashes_after": after, "source_immutable": before == after, "phase_c_linkage": linkage, "statistics": {"bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.bootstrap_seed, "bootstrap_unit": "whole image; all class rows retained"}, "output_tables": {name: {"sha256": sha256_file(output_dir / name)} for name in ("relevance_stability_metrics.csv", "relevance_stability_5x5_grid.csv", "conditional_stability_uplift.csv", "quadrant_region_composition.csv", "false_positive_suppression.csv", "multi_label_overlap.csv", "absent_class_control.csv", "graph_comparison.csv")}}
    json_dump(output_dir / "run_metadata.json", metadata); json_dump(output_dir / "completion.json", {"status": "complete", "phase": "D", "finished_at": timestamp(), "image_count": len(image_ids), "bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.bootstrap_seed, "basis_invariance_passed": True}); text_dump(output_dir / "exact_commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\n" + command_line() + "\n"); (output_dir / "exact_commands.sh").chmod(0o755); log("Phase D complete")


if __name__ == "__main__":
    main()
