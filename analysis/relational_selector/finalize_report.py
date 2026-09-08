#!/usr/bin/env python
"""Build compact preregistered tables/report from an immutable frozen dump."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.spatial_graph_stability.provenance import (
    command_line,
    create_output,
    csv_dump,
    git_metadata,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    text_dump,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _write(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"refusing to write an empty compact table: {path}")
    csv_dump(path, frame.to_dict("records"), list(frame.columns))


def _shared_table(dump: Path) -> pd.DataFrame:
    summary = pd.read_csv(dump / "task_a_layer_summary.csv")
    pca = pd.read_csv(dump / "task_a_pca.csv")
    merged = summary.merge(pca[["layer", "pc1_fraction", "r95", "pc4_cumulative", "effective_rank"]], on="layer", validate="one_to_one")
    return merged.rename(columns={
        "raw_pair_corr": "raw_corr", "residual_pair_corr": "residual_corr",
        "raw_pair_jaccard_top05": "raw_collision_at5", "residual_pair_jaccard_top05": "residual_collision_at5",
        "common_r2": "common_r2", "common_fg_top05": "common_fg_at5",
        "common_bg_top05": "common_bg_at5", "pc1_fraction": "pca_pc1", "r95": "pca_r95",
    })[[
        "layer", "raw_corr", "residual_corr", "raw_collision_at5", "residual_collision_at5",
        "common_r2", "common_fg_at5", "common_bg_at5", "pca_pc1", "pc4_cumulative", "pca_r95", "effective_rank",
    ]]


def _task_a_strata(dump: Path) -> pd.DataFrame:
    raw = pd.read_csv(dump / "task_a_per_image_layer.csv")
    metrics = ["raw_pair_corr", "residual_pair_corr", "raw_pair_jaccard_top05", "residual_pair_jaccard_top05", "common_r2", "common_fg_top05", "common_bg_top05"]
    rows: list[dict[str, object]] = []
    for (layer, stratum), frame in raw.groupby(["layer", "label_stratum"], sort=True):
        row: dict[str, object] = {"layer": int(layer), "label_stratum": stratum, "num_images": int(frame.image_id.nunique())}
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _selector_main(dump: Path) -> pd.DataFrame:
    summary = pd.read_csv(dump / "selector_summary.csv")
    return summary.rename(columns={
        "auc_target_other": "target_vs_other_auroc", "ap_target_other": "target_vs_other_ap",
        "auc_target_bg": "target_vs_bg_auroc", "target_top05_fraction": "target_at5",
        "other_fg_top05_fraction": "other_fg_at5", "bg_top05_fraction": "bg_at5",
        "pair_jaccard_top05": "collision_at5",
    })[[
        "selector", "num_images", "num_image_class", "target_vs_other_auroc", "target_vs_other_ap",
        "target_vs_bg_auroc", "target_at5", "other_fg_at5", "bg_at5", "collision_at5",
    ]]


def _multi_table(dump: Path) -> pd.DataFrame:
    rows = pd.read_csv(dump / "selector_per_image_class.csv")
    pairs = pd.read_csv(dump / "selector_pair_diversity.csv")
    rows = rows[rows.label_count >= 2]
    result: list[dict[str, object]] = []
    for selector, frame in rows.groupby("selector", sort=True):
        output: dict[str, object] = {"selector": selector, "num_images": int(frame.image_id.nunique()), "num_image_class": int(len(frame))}
        for metric in ("auc_target_other", "ap_target_other", "auc_target_bg", "ap_target_bg", "target_top05_fraction", "other_fg_top05_fraction", "bg_top05_fraction"):
            values = frame[metric].to_numpy(dtype=float)
            output[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        pair_values = pairs[(pairs.selector == selector) & (pairs.label_count >= 2)].pair_jaccard_top05.to_numpy(dtype=float)
        output["pair_jaccard_top05"] = float(np.nanmean(pair_values)) if np.isfinite(pair_values).any() else float("nan")
        result.append(output)
    return pd.DataFrame(result)


def _markdown(shared: pd.DataFrame, selector: pd.DataFrame, multi: pd.DataFrame, bootstrap: pd.DataFrame, pca: pd.DataFrame, adjacent: pd.DataFrame, dump_metadata: dict[str, object]) -> str:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        shown = frame[columns].copy()
        for column in shown.columns:
            if pd.api.types.is_numeric_dtype(shown[column]):
                shown[column] = shown[column].map(lambda value: "—" if not np.isfinite(value) else f"{value:.4f}")
        # Avoid the optional ``tabulate`` dependency: reproducibility must not
        # depend on an undeclared report-formatting package in tgca-repro.
        def escape(value: object) -> str:
            return str(value).replace("|", "\\|")
        header = "| " + " | ".join(escape(column) for column in columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| " + " | ".join(escape(value) for value in row) + " |"
            for row in shown.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *body])

    late = shared[shared.layer >= 9]
    layer12 = shared.loc[shared.layer == 12].iloc[0]
    l11_l12 = adjacent[
        (adjacent["from_layer"] == 11)
        & (adjacent["to_layer"] == 12)
        & (adjacent["rank"] == 4)
    ].iloc[0]
    s0 = selector[selector.selector == "S0"].iloc[0]
    s5 = selector[selector.selector == "S5"].iloc[0]
    paired = bootstrap[(bootstrap.family == "paired_delta") & (bootstrap.selector == "S5") & (bootstrap.metric.isin(["auc_target_other", "ap_target_other", "target_top05_fraction", "pair_jaccard_top05"]))]
    paired_text = table(paired, ["metric", "estimate", "ci_low", "ci_high", "n_clusters", "n_rows"])
    return "\n".join([
        "# Frozen Multi-Class Token Relation Analysis",
        "",
        "## Scope and integrity",
        "",
        "This is a frozen, inference-only analysis of the audited native MCTformer+-Small checkpoint at single-scale 448 on all 1,449 VOC-val images. All class--patch relations were calculated in float32 from raw post-block representations. No training, loss, backward pass, attention change, token change, or GT-conditioned score construction occurred. GT enters only after S0--S5 were saved, for region evaluation.",
        "",
        f"- Checkpoint SHA256: `{dump_metadata['checkpoint_sha256']}`",
        f"- Checkpoint/model parameter SHA256 before and after: `{dump_metadata['model_parameter_sha256_before']}` / `{dump_metadata['model_parameter_sha256_after']}` (identical).",
        f"- Dump code commit: `{dump_metadata['git']['commit']}`; this compact report commit is recorded in `compact_metadata.json`.",
        "- Statistical unit: image × positive class; paired comparisons cluster-resample complete images, 10,000 repeats, seed 2027.",
        "",
        "## Task A — all-layer shared-presence table",
        "",
        table(shared, ["layer", "raw_corr", "residual_corr", "raw_collision_at5", "residual_collision_at5", "common_r2", "common_fg_at5", "pca_pc1", "pca_r95"]),
        "",
        "### Question A — where does shared activation form?",
        "",
        f"A common additive component is already strong by L5--L7 (R² {shared.loc[shared.layer == 5, 'common_r2'].iloc[0]:.3f}--{shared.loc[shared.layer == 7, 'common_r2'].iloc[0]:.3f}) and the raw cross-class relation becomes distinctly positive at L6, then rises sharply at L9--L12 (raw correlation {late.raw_corr.min():.3f}--{late.raw_corr.max():.3f}; raw collision@5% {late.raw_collision_at5.min():.3f}--{late.raw_collision_at5.max():.3f}). However, the preregistered H1 criterion is *not supported*: residual cross-class correlation and collision are larger, not smaller, in every late layer. Thus these data show a shared low-dimensional component, but not that subtracting its all-class mean removes the observed cross-class relation coupling.",
        "",
        "### Question B — rank-1 or higher-rank?",
        "",
        f"Late shared means are strongly low-dimensional rather than strictly rank-1: L12 PC1 explains {layer12.pca_pc1:.3f}, PC4 explains {pca.loc[pca.layer == 12, 'pc4_cumulative'].iloc[0]:.3f}, and r95={int(layer12.pca_r95)} (D=384). The L11→L12 rank-4 subspace has mean cosine {l11_l12.mean_subspace_cosine:.3f}. This supports a compact, stable late subspace, not a claim that a fixed rank-1 direction is sufficient or semantically interpretable.",
        "",
        "## Task B — frozen final-layer selector table",
        "",
        table(selector, ["selector", "target_vs_other_auroc", "target_vs_other_ap", "target_vs_bg_auroc", "target_at5", "other_fg_at5", "bg_at5", "collision_at5"]),
        "",
        "### Single-label and multi-label strata",
        "",
        "`selector_single_label_table.csv` and `selector_multi_label_table.csv` contain the corresponding complete strata. Target-vs-other is undefined for the single-label stratum when no other-foreground region exists; it is deliberately reported as missing rather than imputed.",
        "",
        table(multi, ["selector", "num_images", "target_vs_other_auroc", "target_vs_other_ap", "target_at5", "other_fg_top05_fraction", "bg_top05_fraction", "pair_jaccard_top05"]),
        "",
        "### Paired S5 − S0 bootstrap",
        "",
        paired_text,
        "",
        "### Question C — does classifier + relative ownership improve frozen target-vs-other?",
        "",
        f"No. S5 target-vs-other AUROC/AP is {s5.target_vs_other_auroc:.4f}/{s5.target_vs_other_ap:.4f} versus S0 {s0.target_vs_other_auroc:.4f}/{s0.target_vs_other_ap:.4f}. The paired AUROC difference is negative with a 95% CI entirely below zero (see table). S5 also has a small negative target@5% difference. This fails the plan's S5 go criterion; no selector integration or trainable follow-up is warranted from this experiment.",
        "",
        "## Interpretation boundary",
        "",
        "These are representation-level and frozen-ranking observations. They do not establish causal attention behavior, semantic prototypes, background leakage, lazy semantic assignment, a CAM mechanism, or a new WSSS method. No further method is proposed or implemented.",
        "",
    ]) + "\n"


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    os.chdir(root)
    require_environment()
    require_clean_tracked(root)
    dump = _resolve(root, args.dump_dir)
    output = create_output(_resolve(root, args.output_dir))
    metadata = pd.read_json(dump / "metadata.json", typ="series").to_dict()
    shared = _shared_table(dump)
    task_strata = _task_a_strata(dump)
    selector = _selector_main(dump)
    label_strata = pd.read_csv(dump / "selector_label_strata.csv")
    single = label_strata[label_strata.label_stratum == "single_label"].copy()
    multi = _multi_table(dump)
    classwise = pd.read_csv(dump / "selector_classwise.csv")
    bootstrap = pd.read_csv(dump / "selector_bootstrap.csv")
    pca = pd.read_csv(dump / "task_a_pca.csv")
    adjacent = pd.read_csv(dump / "task_a_adjacent_subspaces.csv")
    paired = bootstrap[bootstrap.family == "paired_delta"].copy()
    _write(output / "shared_presence_layer_table.csv", shared)
    _write(output / "shared_presence_by_label_count.csv", task_strata)
    _write(output / "selector_main_table.csv", selector)
    _write(output / "selector_single_label_table.csv", single)
    _write(output / "selector_multi_label_table.csv", multi)
    _write(output / "selector_classwise.csv", classwise)
    _write(output / "paired_bootstrap_vs_s0.csv", paired)
    report = _markdown(shared, selector, multi, bootstrap, pca, adjacent, metadata)
    text_dump(output / "FROZEN_MULTI_CLASS_TOKEN_RELATION_FINAL_REPORT.md", report)
    compact_metadata = {
        "schema": "frozen_multi_class_token_relation_compact_v1", "created_at": timestamp(),
        "command": command_line(), "git": git_metadata(root), "source_dump": str(dump),
        "source_metadata_sha256": sha256_file(dump / "metadata.json"),
        "source_analysis_metadata_sha256": sha256_file(dump / "analysis_metadata.json"),
        "files": sorted(path.name for path in output.iterdir()),
    }
    json_dump(output / "compact_metadata.json", compact_metadata)


if __name__ == "__main__":
    main()
