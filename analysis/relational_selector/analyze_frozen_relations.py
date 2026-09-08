#!/usr/bin/env python
"""Offline Task A/B analyses for a frozen multi-class relation score dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.lazy_assignment.bootstrap import cluster_bootstrap_means, derived_seed
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES
from analysis.lazy_assignment.experiment2.metrics_region import region_map_metrics, stable_topk_mask
from analysis.lazy_assignment.experiment2.patch_regions import (
    REGION_BACKGROUND,
    REGION_OTHER_FOREGROUND,
    REGION_TARGET,
    REGION_VOID,
    assign_patch_regions,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import VOCSemanticDataset
from analysis.relational_selector.task_a import label_stratum
from analysis.spatial_graph_stability.provenance import (
    command_line,
    csv_dump,
    git_metadata,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    text_dump,
    timestamp,
)


SCORES = ("S0", "S1", "S2", "S3", "S4", "S5")
PRIMARY_METRICS = (
    "auc_target_other", "ap_target_other", "auc_target_bg", "ap_target_bg",
    "auc_target_all", "ap_target_all", "target_top05_fraction",
    "other_fg_top05_fraction", "bg_top05_fraction", "pair_jaccard_top05",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument("--list-path", type=Path, default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"))
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-tracked-dirty", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _write_unique_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact {path}")
    if frame.empty:
        raise ValueError(f"refusing to write empty analysis table {path}")
    csv_dump(path, frame.to_dict("records"), list(frame.columns))


def _binary(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    include = np.logical_or(positive, negative)
    labels = positive[include].astype(np.int8)
    if len(labels) < 2 or labels.min() == labels.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores[include])), float(average_precision_score(labels, scores[include]))


def _all_target_metrics(scores: np.ndarray, regions: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    codes = np.asarray(regions).reshape(-1)
    target = codes == REGION_TARGET
    valid = codes != REGION_VOID
    all_non_target = np.logical_and(valid, ~target)
    auc, ap = _binary(values, target, all_non_target)
    return {"auc_target_all": auc, "ap_target_all": ap}


def _top_metrics(scores: np.ndarray, regions: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    codes = np.asarray(regions).reshape(-1)
    valid = codes != REGION_VOID
    output: dict[str, float] = {}
    for name, code in (("target", REGION_TARGET), ("other_fg", REGION_OTHER_FOREGROUND), ("bg", REGION_BACKGROUND)):
        output[f"{name}_top01_fraction"] = float(codes[np.flatnonzero(valid)[np.argmax(values[valid])]] == code) if valid.any() else float("nan")
    for ratio, suffix in ((0.05, "05"), (0.10, "10")):
        top = stable_topk_mask(values, ratio, valid).reshape(-1)
        denominator = max(1, int(top.sum()))
        for name, code in (("target", REGION_TARGET), ("other_fg", REGION_OTHER_FOREGROUND), ("bg", REGION_BACKGROUND)):
            output[f"{name}_top{suffix}_fraction"] = float(np.logical_and(top, codes == code).sum() / denominator)
    return output


def _pair_jaccards(score_maps: np.ndarray, active: np.ndarray, regions: np.ndarray) -> dict[str, float]:
    valid = np.asarray(regions).reshape(-1) != REGION_VOID
    positives = np.flatnonzero(active > 0)
    result = {"pair_jaccard_top01": float("nan"), "pair_jaccard_top05": float("nan"), "pair_jaccard_top10": float("nan")}
    if len(positives) < 2:
        return result
    for ratio, suffix in ((1.0 / score_maps.shape[-1], "01"), (0.05, "05"), (0.10, "10")):
        values: list[float] = []
        for offset, first in enumerate(positives[:-1]):
            first_top = stable_topk_mask(score_maps[first], ratio, valid).reshape(-1)
            for second in positives[offset + 1 :]:
                second_top = stable_topk_mask(score_maps[second], ratio, valid).reshape(-1)
                union = np.logical_or(first_top, second_top).sum()
                values.append(float(np.logical_and(first_top, second_top).sum() / union) if union else 1.0)
        result[f"pair_jaccard_top{suffix}"] = float(np.mean(values))
    return result


def _pca_table(mu: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mu.ndim != 3 or mu.shape[1:] != (12, 384):
        raise ValueError(f"mu_common must be [N,12,384], got {mu.shape}")
    pca_rows: list[dict[str, object]] = []
    components: list[np.ndarray] = []
    for layer in range(12):
        values = np.asarray(mu[:, layer], dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        eigen = np.square(singular)
        total = float(eigen.sum())
        fractions = eigen / total if total > 0 else np.zeros_like(eigen)
        cumulative = np.cumsum(fractions)
        nonzero = fractions[fractions > 0]
        pca_rows.append({
            "layer": layer + 1,
            "pc1_fraction": float(fractions[0]), "pc2_fraction": float(fractions[1]),
            "pc4_cumulative": float(cumulative[min(3, len(cumulative) - 1)]),
            "pc8_cumulative": float(cumulative[min(7, len(cumulative) - 1)]),
            "pc16_cumulative": float(cumulative[min(15, len(cumulative) - 1)]),
            "participation_rank": float(1.0 / np.square(fractions).sum()) if total > 0 else float("nan"),
            "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if len(nonzero) else float("nan"),
            "r90": int(np.searchsorted(cumulative, 0.90, side="left") + 1) if total > 0 else 0,
            "r95": int(np.searchsorted(cumulative, 0.95, side="left") + 1) if total > 0 else 0,
        })
        components.append(vh)
    adjacent: list[dict[str, object]] = []
    for layer in range(11):
        for rank in (1, 4, 8, 16):
            singular_values = np.linalg.svd(components[layer][:rank] @ components[layer + 1][:rank].T, compute_uv=False)
            singular_values = np.clip(singular_values, -1.0, 1.0)
            angles = np.degrees(np.arccos(singular_values))
            adjacent.append({
                "from_layer": layer + 1, "to_layer": layer + 2, "rank": rank,
                "mean_subspace_cosine": float(singular_values.mean()),
                "min_subspace_cosine": float(singular_values.min()),
                "mean_principal_angle_deg": float(angles.mean()),
                "max_principal_angle_deg": float(angles.max()),
            })
    return pd.DataFrame(pca_rows), pd.DataFrame(adjacent)


def _task_a_summary(task_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "raw_pair_corr", "residual_pair_corr", "raw_pair_jaccard_top05", "residual_pair_jaccard_top05",
        "raw_pair_jaccard_top10", "residual_pair_jaccard_top10", "common_r2", "common_fg_top05",
        "common_bg_top05", "common_positive_spearman", "common_positive_jaccard_top05",
    ]
    rows: list[dict[str, object]] = []
    for layer, frame in task_rows.groupby("layer", sort=True):
        row: dict[str, object] = {"layer": int(layer), "num_images": int(frame.image_id.nunique())}
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{metric}_n"] = int(np.isfinite(values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _selector_rows(dump: Path, data: VOCSemanticDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(dump / "index.csv")
    if len(index) != len(data):
        raise RuntimeError("score dump and VOC data have different image counts")
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for image_index in range(len(data)):
        sample = data[image_index]
        image_id = str(sample["name"])
        expected = str(index.iloc[image_index].image_id)
        if image_id != expected:
            raise RuntimeError(f"VOC/dump order mismatch at {image_index}: {image_id} != {expected}")
        with np.load(dump / "samples" / f"{image_id}.npz", allow_pickle=False) as artifact:
            labels = np.asarray(artifact["labels"], dtype=np.float32)
            grid = tuple(int(value) for value in artifact["grid"])
            if grid != (28, 28) or labels.shape != (20,):
                raise RuntimeError("sample score artifact geometry/label contract failed")
            maps = {name: np.asarray(artifact[name], dtype=np.float32) for name in SCORES}
        if not np.array_equal(labels.astype(np.uint8), np.asarray(sample["label"], dtype=np.uint8)):
            raise RuntimeError(f"label mismatch for {image_id}")
        active = np.flatnonzero(labels > 0)
        label_count = int(len(active))
        strata = label_stratum(label_count)
        for selector, score_maps in maps.items():
            if score_maps.shape != (20, 784) or not np.isfinite(score_maps).all():
                raise RuntimeError(f"{image_id} {selector} violates the saved score contract")
            pair_rows.append({
                "image_id": image_id, "image_index": image_index, "selector": selector,
                "label_count": label_count, "label_stratum": strata,
                **_pair_jaccards(score_maps, labels, np.asarray(assign_patch_regions(sample["mask"].numpy(), int(active[0]))["region_codes"])),
            })
            for class_id in active:
                regions = np.asarray(assign_patch_regions(sample["mask"].numpy(), int(class_id))["region_codes"]).reshape(-1)
                metric = region_map_metrics(score_maps[class_id], regions, grid_h=28, grid_w=28)
                output = {
                    "image_id": image_id, "image_index": image_index, "class_id": int(class_id),
                    "class_name": VOC_CLASS_NAMES[int(class_id)], "selector": selector,
                    "label_count": label_count, "label_stratum": strata,
                    **{key: metric.get(key, float("nan")) for key in (
                        "auc_target_bg", "ap_target_bg", "auc_target_other", "ap_target_other",
                        "target_bg_mean_margin", "target_other_mean_margin",
                    )},
                    **_all_target_metrics(score_maps[class_id], regions),
                    **_top_metrics(score_maps[class_id], regions),
                }
                rows.append(output)
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def _selector_summary(frame: pd.DataFrame, pair_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [column for column in frame.columns if column not in {"image_id", "image_index", "class_id", "class_name", "selector", "label_count", "label_stratum"}]
    summaries: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for selector, subset in frame.groupby("selector", sort=True):
        base: dict[str, object] = {"selector": selector, "aggregation": "image_positive_class_mean", "num_images": int(subset.image_id.nunique()), "num_image_class": int(len(subset))}
        for metric in metric_columns:
            values = subset[metric].to_numpy(dtype=float)
            base[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        pair_subset = pair_frame[pair_frame.selector == selector]
        for metric in ("pair_jaccard_top01", "pair_jaccard_top05", "pair_jaccard_top10"):
            values = pair_subset[metric].to_numpy(dtype=float)
            base[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        summaries.append(base)
        for class_id, class_subset in subset.groupby("class_id", sort=True):
            row = {"selector": selector, "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)], "num_image_class": int(len(class_subset))}
            for metric in metric_columns:
                values = class_subset[metric].to_numpy(dtype=float)
                row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            class_rows.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(class_rows)


def _bootstrap(frame: pd.DataFrame, pair_frame: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    # Each selector estimate remains clustered by image.  Paired deltas use the
    # same image-row unit after pivoting, so patches and sibling classes never
    # become independent observations.
    direct_metrics = ("auc_target_other", "ap_target_other", "auc_target_bg", "ap_target_bg", "target_top05_fraction")
    for metric in direct_metrics:
        pivot = frame.pivot(index=["image_id", "class_id"], columns="selector", values=metric).reset_index()
        estimates = cluster_bootstrap_means(pivot, "image_id", list(SCORES), repeats, derived_seed(seed, "direct", metric))
        for selector, value in estimates.items():
            output.append({"family": "direct", "selector": selector, "metric": metric, "contrast": "estimate", **value.__dict__, "bootstrap_repeats": repeats, "bootstrap_seed": seed})
    pair_pivot = pair_frame.pivot(index="image_id", columns="selector", values="pair_jaccard_top05").reset_index()
    pair_estimates = cluster_bootstrap_means(pair_pivot, "image_id", list(SCORES), repeats, derived_seed(seed, "direct", "pair_jaccard_top05"))
    for selector, value in pair_estimates.items():
        output.append({"family": "direct", "selector": selector, "metric": "pair_jaccard_top05", "contrast": "estimate", **value.__dict__, "bootstrap_repeats": repeats, "bootstrap_seed": seed})
    delta_metrics = ("auc_target_other", "ap_target_other", "target_top05_fraction")
    for metric in delta_metrics:
        pivot = frame.pivot(index=["image_id", "class_id"], columns="selector", values=metric).reset_index()
        values: list[str] = []
        for selector in SCORES[1:]:
            column = f"{selector}_minus_S0"
            pivot[column] = pivot[selector] - pivot["S0"]
            values.append(column)
        estimates = cluster_bootstrap_means(pivot, "image_id", values, repeats, derived_seed(seed, "paired_delta", metric))
        for column, value in estimates.items():
            output.append({"family": "paired_delta", "selector": column.split("_minus_")[0], "metric": metric, "contrast": "minus_S0", **value.__dict__, "bootstrap_repeats": repeats, "bootstrap_seed": seed})
    pivot = pair_pivot.copy()
    values = []
    for selector in SCORES[1:]:
        column = f"{selector}_minus_S0"
        pivot[column] = pivot[selector] - pivot["S0"]
        values.append(column)
    estimates = cluster_bootstrap_means(pivot, "image_id", values, repeats, derived_seed(seed, "paired_delta", "pair_jaccard_top05"))
    for column, value in estimates.items():
        output.append({"family": "paired_delta", "selector": column.split("_minus_")[0], "metric": "pair_jaccard_top05", "contrast": "minus_S0", **value.__dict__, "bootstrap_repeats": repeats, "bootstrap_seed": seed})
    return pd.DataFrame(output)


def _stratum_summary(frame: pd.DataFrame, pair_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stratum, subset in frame.groupby("label_stratum", sort=True):
        pair_subset_all = pair_frame[pair_frame.label_stratum == stratum]
        for selector, selector_subset in subset.groupby("selector", sort=True):
            row = {"label_stratum": stratum, "selector": selector, "num_image_class": int(len(selector_subset)), "num_images": int(selector_subset.image_id.nunique())}
            for metric in ("auc_target_other", "ap_target_other", "auc_target_bg", "ap_target_bg", "target_top05_fraction", "other_fg_top05_fraction", "bg_top05_fraction"):
                values = selector_subset[metric].to_numpy(dtype=float)
                row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            pairs = pair_subset_all[pair_subset_all.selector == selector]["pair_jaccard_top05"].to_numpy(dtype=float)
            row["pair_jaccard_top05"] = float(np.nanmean(pairs)) if np.isfinite(pairs).any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _report(task_a: pd.DataFrame, pca: pd.DataFrame, selector: pd.DataFrame, bootstrap: pd.DataFrame, metadata: dict[str, object]) -> str:
    late = task_a[task_a.layer.isin([10, 11, 12])]
    s0 = selector[selector.selector == "S0"].iloc[0]
    s5 = selector[selector.selector == "S5"].iloc[0]
    lines = [
        "# Frozen Multi-Class Token Relation Analysis",
        "",
        "## Scope",
        "",
        "Frozen native MCTformer+-Small, 448 single-scale VOC-val. Relations were computed from raw post-block tokens in float32. No training, gradients, attention intervention, new model component, or GT-conditioned score construction was used.",
        "",
        "## Task A: shared class component",
        "",
        f"Late-layer (L10--L12) mean raw pair correlation is {late.raw_pair_corr.mean():.4f}; residual pair correlation is {late.residual_pair_corr.mean():.4f}. This is a representation-level decomposition, not evidence of a semantic prototype or attention mechanism.",
        f"The L12 common-map R² is {task_a.loc[task_a.layer == 12, 'common_r2'].iloc[0]:.4f}; L12 PC1 explained fraction is {pca.loc[pca.layer == 12, 'pc1_fraction'].iloc[0]:.4f}, with r95={int(pca.loc[pca.layer == 12, 'r95'].iloc[0])}.",
        "",
        "## Task B: frozen selector comparison",
        "",
        f"Classifier-only S0 target-vs-other AUROC/AP: {s0.auc_target_other:.4f}/{s0.ap_target_other:.4f}. Frozen S5: {s5.auc_target_other:.4f}/{s5.ap_target_other:.4f}. These compare selector maps only; they do not establish a proposed method.",
        "",
        "## Inference boundary",
        "",
        "The analysis tests frozen representation geometry and GT ownership of precomputed maps. It does not identify causal attention behavior, background leakage, lazy semantic assignment, or a new localization method. Those claims require separate interventions or experiments.",
        "",
        "## Provenance",
        "",
        f"- Dump checkpoint SHA256: `{metadata['checkpoint_sha256']}`",
        f"- Dump Git commit: `{metadata['git']['commit']}`",
        f"- Bootstrap: image-clustered paired, {int(bootstrap.bootstrap_repeats.iloc[0])} repeats, seed {int(bootstrap.bootstrap_seed.iloc[0])}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    os.chdir(root)
    require_environment()
    if not args.allow_tracked_dirty:
        require_clean_tracked(root)
    dump = _resolve(root, args.dump_dir)
    metadata_path = dump / "metadata.json"
    if not metadata_path.is_file() or not (dump / "completion.json").is_file():
        raise FileNotFoundError("dump directory lacks immutable frozen score provenance")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "frozen_multi_class_token_relation_dump_v1":
        raise RuntimeError("unexpected frozen score-dump schema")
    if int(metadata["input"]["input_size"]) != 448:
        raise RuntimeError("analysis requires 448 score dump")
    if args.limit and int(metadata["input"]["num_images"]) != args.limit:
        raise RuntimeError("--limit does not match score dump")
    for name in ("task_a_per_image_layer.csv", "mu_common.npy", "index.csv", "samples"):
        if not (dump / name).exists():
            raise FileNotFoundError(dump / name)
    task_rows = pd.read_csv(dump / "task_a_per_image_layer.csv")
    mu = np.load(dump / "mu_common.npy", mmap_mode="r")
    task_summary = _task_a_summary(task_rows)
    pca, adjacent = _pca_table(mu)
    _write_unique_csv(dump / "task_a_layer_summary.csv", task_summary)
    _write_unique_csv(dump / "task_a_pca.csv", pca)
    _write_unique_csv(dump / "task_a_adjacent_subspaces.csv", adjacent)
    voc_root = _resolve(root, args.voc_root)
    list_path = _resolve(root, args.list_path)
    data = VOCSemanticDataset(voc_root, list_path, input_size=448, limit=args.limit)
    if len(data) != int(metadata["input"]["num_images"]):
        raise RuntimeError("VOC analysis data count differs from dump")
    selector_rows, pair_rows = _selector_rows(dump, data)
    selector_summary, class_summary = _selector_summary(selector_rows, pair_rows)
    strata = _stratum_summary(selector_rows, pair_rows)
    bootstrap = _bootstrap(selector_rows, pair_rows, args.bootstrap_repeats, args.bootstrap_seed)
    _write_unique_csv(dump / "selector_per_image_class.csv", selector_rows)
    _write_unique_csv(dump / "selector_pair_diversity.csv", pair_rows)
    _write_unique_csv(dump / "selector_summary.csv", selector_summary)
    _write_unique_csv(dump / "selector_classwise.csv", class_summary)
    _write_unique_csv(dump / "selector_label_strata.csv", strata)
    _write_unique_csv(dump / "selector_bootstrap.csv", bootstrap)
    report = _report(task_summary, pca, selector_summary, bootstrap, metadata)
    text_dump(dump / "FROZEN_MULTI_CLASS_TOKEN_RELATION_REPORT.md", report)
    analysis_metadata = {
        "schema": "frozen_multi_class_token_relation_analysis_v1", "created_at": timestamp(),
        "command": command_line(), "git": git_metadata(root), "dump_dir": str(dump),
        "dump_metadata_sha256": sha256_file(metadata_path), "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_seed": int(args.bootstrap_seed), "gt_used_only_in_evaluation": True,
        "files": ["task_a_layer_summary.csv", "task_a_pca.csv", "task_a_adjacent_subspaces.csv", "selector_per_image_class.csv", "selector_pair_diversity.csv", "selector_summary.csv", "selector_classwise.csv", "selector_label_strata.csv", "selector_bootstrap.csv", "FROZEN_MULTI_CLASS_TOKEN_RELATION_REPORT.md"],
    }
    json_dump(dump / "analysis_metadata.json", analysis_metadata)
    with (dump / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"[{timestamp()}] analysis complete: task_a={len(task_rows)} selector_rows={len(selector_rows)}\n")


if __name__ == "__main__":
    main()
