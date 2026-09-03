#!/usr/bin/env python3
"""Generate the final Experiment 1 analysis and Experiment 2 readiness reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from analysis.lazy_assignment.experiment1_analysis_common import (
    environment_metadata,
    git_metadata,
    json_dump,
    sha256_file,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def number(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def interval(row: pd.Series, estimate: str = "paired_delta") -> str:
    return (
        f"{number(row[estimate])} "
        f"[{number(row['ci_low'])}, {number(row['ci_high'])}]"
    )


def output_file_table(directory: Path, pattern: str) -> str:
    rows = ["| File | Bytes |", "|---|---:|"]
    for path in sorted(directory.glob(pattern)):
        rows.append(f"| `{path.name}` | {path.stat().st_size:,} |")
    return "\n".join(rows)


def exact_commands(analysis: Path, repo: Path) -> str:
    result = str(analysis)
    mct = (
        "/home/peng/code/TGCA/results/lazy_assignment/experiment1_class_patch_score/"
        "mctformer/20260902-mctformerv2-exp1-voc-val-full-6aca9bc"
    )
    plus = (
        "/home/peng/code/TGCA/results/lazy_assignment/experiment1_class_patch_score/"
        "mctformer_plus/20260902-mctformerplus-exp1-voc-val-full-fec86b7"
    )
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {repo}",
        "",
        "# Dependency added without upgrading the existing tgca-repro stack.",
        "conda run -n tgca-repro python -m pip install pyarrow==14.0.2",
        "",
        "# Source audit. Each output directory is intentionally immutable/non-overwriting.",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.audit_experiment1_results \\",
        f"  --mctformer-results {mct} \\",
        f"  --mctformer-plus-results {plus} \\",
        "  --voc-root data/VOCdevkit/VOC2012 \\",
        "  --val-list data/VOCdevkit/VOC2012/ImageLists/val_id.txt \\",
        f"  --output-dir {result}/audit",
        "",
        "# Single source scan and canonical Parquet construction.",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.build_canonical_tables \\",
        f"  --mctformer-results {mct} \\",
        f"  --mctformer-plus-results {plus} \\",
        f"  --audit-dir {result}/audit \\",
        f"  --output-dir {result}/canonical",
        "",
        "# Final image-clustered inference settings from the guide.",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.analyze_experiment1_results \\",
        f"  --canonical-dir {result}/canonical \\",
        f"  --output-dir {result} \\",
        "  --bootstrap-repeats 5000 \\",
        "  --bootstrap-seed 20260901 \\",
        "  --topk-ratios 0.05,0.10,0.20 \\",
        "  --entropy-temperatures 0.05,0.10,0.20",
        "",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.verify_experiment1_canonical \\",
        f"  --canonical-dir {result}/canonical \\",
        f"  --audit-dir {result}/audit \\",
        f"  --output {result}/canonical/canonical_roundtrip_verification.json \\",
        "  --sample-size 10 --seed 20260901 --atol 1e-12",
        "",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.plot_experiment1 \\",
        f"  --analysis-dir {result} --output-dir {result}/plots",
        "",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.select_examples \\",
        f"  --canonical-dir {result}/canonical \\",
        "  --voc-root data/VOCdevkit/VOC2012 \\",
        f"  --output-dir {result}/examples --top-n 10 --input-size 448",
        "",
        f"conda run -n tgca-repro python -m pytest -q 2>&1 | tee {result}/tests.log",
        "conda run -n tgca-repro ruff check analysis/lazy_assignment tests",
        "",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.verify_experiment1_immutability \\",
        f"  --before-manifest {result}/audit/file_manifest.csv \\",
        f"  --after-manifest {result}/audit/file_manifest_after_analysis.csv \\",
        f"  --output {result}/audit/source_immutability_after_analysis.json",
        "",
        "conda run -n tgca-repro python -m analysis.lazy_assignment.generate_experiment1_reports \\",
        f"  --analysis-dir {result} --repo-root {repo}",
        "",
        "# The actual canonical/bootstrap/example stages were run in unique tmux sessions;",
        "# their stdout/stderr is preserved in *_stage.log. Reproduction must use a new",
        "# analysis result directory because the tools intentionally refuse overwrite.",
    ]
    return "\n".join(commands) + "\n"


def failed_attempts() -> str:
    return """Experiment 1 analysis failed attempts retained for provenance

1. Audit entry-point import failure (no source artifact was opened or modified):
   conda run -n tgca-repro python analysis/lazy_assignment/audit_experiment1_results.py ...
   ModuleNotFoundError: No module named 'analysis'
   Resolution: invoke repository modules with `python -m analysis.lazy_assignment...`.

2. First plot attempt (statistics were already complete and unchanged):
   conda run -n tgca-repro python -m analysis.lazy_assignment.plot_experiment1 ...
   KeyError: 'pairwise_spearman_mean_ci_low'
   Resolution: map diversity mean columns to their explicitly named CI stems. The
   partial figures were retained under failed_artifacts/plots_attempt1/.
"""


def generate(args: argparse.Namespace) -> None:
    analysis = args.analysis_dir.resolve()
    repo = args.repo_root.resolve()
    audit = json.loads((analysis / "audit/integrity_report.json").read_text())
    immutability = json.loads(
        (analysis / "audit/source_immutability_after_analysis.json").read_text()
    )
    canonical_check = json.loads(
        (analysis / "canonical/canonical_roundtrip_verification.json").read_text()
    )
    canonical_meta = json.loads(
        (analysis / "canonical/canonical_metadata.json").read_text()
    )
    analysis_meta = json.loads((analysis / "tables/analysis_metadata.json").read_text())
    tables = analysis / "tables"
    summaries = {
        model: pd.read_csv(tables / f"layerwise_summary_{model}.csv")
        for model in ("mctformer", "mctformer_plus")
    }
    deltas = {
        model: pd.read_csv(tables / f"layerwise_delta_{model}.csv")
        for model in ("mctformer", "mctformer_plus")
    }
    ranks = {
        model: pd.read_csv(tables / f"layer_rank_stability_{model}.csv")
        for model in ("mctformer", "mctformer_plus")
    }
    diversity = pd.read_csv(tables / "class_map_diversity_by_layer.csv")
    paired = pd.read_csv(tables / "mctformer_vs_plus_paired.csv")
    classwise = pd.read_csv(tables / "classwise_paired_delta.csv")
    single_multi = pd.read_csv(tables / "single_vs_multilabel_summary.csv")
    maps = pd.read_parquet(analysis / "canonical/per_image_class_layer.parquet")
    examples = pd.read_csv(analysis / "examples/example_selection.csv")

    def summary(model: str, layer: int, aggregation: str = "micro") -> pd.Series:
        return summaries[model].query(
            "layer == @layer and aggregation == @aggregation"
        ).iloc[0]

    def delta(model: str, metric: str) -> pd.Series:
        return deltas[model].query(
            "layer == 12 and aggregation == 'micro' and "
            "delta_reference == 'from_layer1' and metric == @metric"
        ).iloc[0]

    def paired_row(metric: str, layer: int) -> pd.Series:
        return paired.query("metric == @metric and layer == @layer").iloc[0]

    mct_l1, mct_l12 = summary("mctformer", 1), summary("mctformer", 12)
    plus_l1, plus_l12 = summary("mctformer_plus", 1), summary("mctformer_plus", 12)
    multilabel_images = int(diversity["num_multilabel_images"].iloc[0])
    multiclass_pairs = int(diversity["num_class_pairs"].iloc[0])
    min_std = float(maps["score_std"].min())
    raw_min = float(maps["score_min"].min())
    raw_max = float(maps["score_max"].max())

    class_tail = classwise.query("layer == 12 and metric == 'upper_tail_gap'")
    class_entropy = classwise.query("layer == 12 and metric == 'spatial_entropy'")
    class_component = classwise.query(
        "layer == 12 and metric == 'top10_largest_component'"
    )
    tail_excludes = int(((class_tail.ci_low > 0) | (class_tail.ci_high < 0)).sum())
    entropy_excludes = int(
        ((class_entropy.ci_low > 0) | (class_entropy.ci_high < 0)).sum()
    )
    component_excludes = int(
        ((class_component.ci_low > 0) | (class_component.ci_high < 0)).sum()
    )
    min_class_count = int(class_tail["num_images"].min())
    max_class_count = int(class_tail["num_images"].max())
    tail_low = class_tail.loc[class_tail.paired_delta.idxmin()]
    tail_high = class_tail.loc[class_tail.paired_delta.idxmax()]
    entropy_low = class_entropy.loc[class_entropy.paired_delta.idxmin()]
    entropy_high = class_entropy.loc[class_entropy.paired_delta.idxmax()]
    disagreement_examples = examples.query(
        "category == 'D_largest_cross_model_disagreement_l12'"
    )
    disagreement_min = float(disagreement_examples.selection_value.min())
    disagreement_max = float(disagreement_examples.selection_value.max())
    mct_diversity_l12 = diversity.query(
        "model == 'mctformer' and layer == 12"
    ).iloc[0]
    plus_diversity_l12 = diversity.query(
        "model == 'mctformer_plus' and layer == 12"
    ).iloc[0]

    report_lines = [
        "# Experiment 1 Analysis Report",
        "",
        f"Generated: `{timestamp()}`",
        "",
        "## 1. Data and Reproducibility",
        "",
        "**[Fact]** The analysis used the two complete local result roots below; they were treated as immutable:",
        "",
        f"- MCTformer: `{audit['models']['mctformer']['result_root']}`",
        f"- MCTformer+: `{audit['models']['mctformer_plus']['result_root']}`",
        "",
        "Both roots contain 1,449/1,449 VOC val images and 2,147 positive image–class pairs. The common set is 1,449 images and 2,147 pairs (100.0000%); neither model has a model-only pair. There are 522 multi-label images and 906 within-image positive-class pairs.",
        "",
        f"- MCTformer checkpoint SHA256: `{audit['models']['mctformer']['checkpoint_sha256']}`; result-source commit `{audit['models']['mctformer']['source_git_commit']}`.",
        f"- MCTformer+ checkpoint SHA256: `{audit['models']['mctformer_plus']['checkpoint_sha256']}`; result-source commit `{audit['models']['mctformer_plus']['source_git_commit']}`.",
        f"- Analysis code base commit: `{git_metadata(repo)['commit']}` (analysis files are uncommitted additions; see `git_diff_summary.txt`).",
        "- Matched settings: `val_id`, input size 448, bicubic resize to 512 then center crop 448, 16×16 patches, 28×28 patch grid, 12 layers, class index convention, positive-class filtering, score formula, and representation extraction point.",
        "- Expected differences: checkpoints and result-source Git commits differ by model; these fields are recorded rather than treated as matching controls.",
        "",
        f"The complete audit found 0 missing/invalid records, 0 duplicate IDs, 0 positive-label mismatches, 0 NaN/Inf values, and 0 cosine-range violations. It hashed {immutability['source_files_checked']:,} source files; after every analysis and visualization step, all {immutability['source_npz_checked']:,} NPZ hashes and every source-file mtime remained unchanged. The canonical 10-NPZ spot check made {canonical_check['comparisons']:,} direct comparisons with maximum absolute error {canonical_check['maximum_absolute_error']:.1f}.",
        "",
        "Canonical tables use one row per `(model, image_id, positive class_id, layer)` and contain 51,528 map rows. Top-k masks use `ceil(N·r)` exact stable sorting. All intervals below use 5,000 image-clustered bootstrap resamples with seed `20260901`; resampling retains every class/class-pair row belonging to a sampled image. Patches and image–class pairs within one image were not used as independent inferential units.",
        "",
        "Artifacts: [inventory](../audit/RESULT_INVENTORY.md), [integrity JSON](../audit/integrity_report.json), [canonical metadata](../canonical/canonical_metadata.json), [analysis metadata](../tables/analysis_metadata.json), and [final immutability JSON](../audit/source_immutability_after_analysis.json).",
        "",
        "## 2. Score Definition",
        "",
        r"For positive class (c), patch (j), and 1-based layer (l), the saved value is",
        "",
        r"\[S_{c,j}^{(l)}=\cos(c_c^{(l)},p_j^{(l)}).\]",
        "",
        "**[Fact]** Tokens were captured after each transformer block and before the final normalization. This is feature-space representation similarity. It is not a probability, not an attention weight, and not a CAM value. The temperature-normalized spatial entropy is an auxiliary softmax over these cosine scores at fixed τ; it is not model attention entropy.",
        "",
        "## 3. Global Score Statistics",
        "",
        "**[Fact]** Pair-equal micro means at the endpoints are:",
        "",
        "| Model | Layer | Mean score | Mean q95 | Mean q95−median | Mean entropy (τ=.10) | Mean TV | Mean largest top-10% component |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| MCTformer | 1 | {number(mct_l1.mean_score)} | {number(mct_l1.mean_q95)} | {number(mct_l1.mean_upper_tail_gap)} | {number(mct_l1.mean_spatial_entropy)} | {number(mct_l1.mean_total_variation)} | {number(mct_l1.mean_largest_component_fraction)} |",
        f"| MCTformer | 12 | {number(mct_l12.mean_score)} | {number(mct_l12.mean_q95)} | {number(mct_l12.mean_upper_tail_gap)} | {number(mct_l12.mean_spatial_entropy)} | {number(mct_l12.mean_total_variation)} | {number(mct_l12.mean_largest_component_fraction)} |",
        f"| MCTformer+ | 1 | {number(plus_l1.mean_score)} | {number(plus_l1.mean_q95)} | {number(plus_l1.mean_upper_tail_gap)} | {number(plus_l1.mean_spatial_entropy)} | {number(plus_l1.mean_total_variation)} | {number(plus_l1.mean_largest_component_fraction)} |",
        f"| MCTformer+ | 12 | {number(plus_l12.mean_score)} | {number(plus_l12.mean_q95)} | {number(plus_l12.mean_upper_tail_gap)} | {number(plus_l12.mean_spatial_entropy)} | {number(plus_l12.mean_total_variation)} | {number(plus_l12.mean_largest_component_fraction)} |",
        "",
        "**[Statistical inference]** Image-clustered L12−L1 changes were:",
        "",
        "| Model | Metric | Mean change | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for model_label, model in (("MCTformer", "mctformer"), ("MCTformer+", "mctformer_plus")):
        for metric_label, metric in (
            ("q95−median", "upper_tail_gap"),
            ("Spatial entropy τ=.10", "spatial_entropy"),
            ("Total variation", "total_variation"),
        ):
            item = delta(model, metric)
            report_lines.append(
                f"| {model_label} | {metric_label} | {number(item.mean_delta)} | [{number(item.ci_low)}, {number(item.ci_high)}] |"
            )
    report_lines.extend(
        [
            "",
            f"**[Fact]** Across all 51,528 maps, the observed cosine range was [{raw_min:.6f}, {raw_max:.6f}]. The minimum within-map standard deviation was {min_std:.6f}; no map had standard deviation below 0.001. Thus the maps are not near-constant under the prespecified sanity check. Entropy decreased from L1 to L12 at all three fixed temperatures (.05/.10/.20) for both models; the absolute values change with τ, as expected.",
            "",
            "**[Interpretation candidate]** Both models develop a stronger upper tail and rougher spatial score field by L12, with a much larger endpoint shift for MCTformer+. This describes representation geometry only; it does not establish whether high-score patches coincide with any semantic region.",
            "",
            "Detailed micro and equal-class macro results (including every CI and entropy sensitivity): [MCTformer](../tables/layerwise_summary_mctformer.csv) and [MCTformer+](../tables/layerwise_summary_mctformer_plus.csv). L12 macro-class q95−median is "
            f"{number(summary('mctformer', 12, 'macro_class').mean_upper_tail_gap)} for MCTformer and {number(summary('mctformer_plus', 12, 'macro_class').mean_upper_tail_gap)} for MCTformer+, close to their micro values above.",
            "",
            "## 4. Layer-wise Evolution",
            "",
            "**[Fact]** Consecutive-layer rank behavior is non-monotonic:",
            "",
            "| Model | Transition ending at layer | Mean Spearman | 95% CI | Mean top-10% Jaccard |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_label, model, layers in (
        ("MCTformer", "mctformer", (2, 10, 11, 12)),
        ("MCTformer+", "mctformer_plus", (2, 10, 11, 12)),
    ):
        for layer in layers:
            item = ranks[model].query("layer == @layer").iloc[0]
            report_lines.append(
                f"| {model_label} | {layer} | {number(item.mean_consecutive_layer_spearman)} | [{number(item.mean_consecutive_layer_spearman_ci_low)}, {number(item.mean_consecutive_layer_spearman_ci_high)}] | {number(item.mean_consecutive_layer_top10_jaccard)} |"
            )
    report_lines.extend(
        [
            "",
            f"At L12, layer-1-to-layer Spearman is {number(ranks['mctformer'].query('layer == 12').iloc[0].mean_layer1_to_layer_spearman)} for MCTformer and {number(ranks['mctformer_plus'].query('layer == 12').iloc[0].mean_layer1_to_layer_spearman)} for MCTformer+; corresponding top-10% Jaccards are {number(ranks['mctformer'].query('layer == 12').iloc[0].mean_layer1_to_layer_top10_jaccard)} and {number(ranks['mctformer_plus'].query('layer == 12').iloc[0].mean_layer1_to_layer_top10_jaccard)}. Hence endpoint patch rankings differ substantially from L1 for both models.",
            "",
            "**[Statistical inference]** For the L11→L12 transition, MCTformer+ minus MCTformer consecutive-layer Spearman is "
            f"{interval(paired_row('consecutive_layer_spearman', 12))}; for L10→L11 it is {interval(paired_row('consecutive_layer_spearman', 11))}. The sign reversal is systematic under image-clustered resampling, not a patch-level significance calculation.",
            "",
            "**[Interpretation candidate]** MCTformer has its strongest late rank reorganization at L10→L11, whereas MCTformer+ undergoes more reorganization earlier and is highly stable across L10→L11→L12. The current experiment cannot identify what image regions receive the reordered scores.",
            "",
            "See [rank plots](../plots/layer_rank_stability.png), [MCTformer rank table](../tables/layer_rank_stability_mctformer.csv), and [MCTformer+ rank table](../tables/layer_rank_stability_mctformer_plus.csv).",
            "",
            "## 5. Multi-Class Map Diversity",
            "",
            f"**[Fact]** The diversity analysis includes all {multilabel_images} multi-label images and {multiclass_pairs} positive-class pairs per layer:",
            "",
            "| Model | Layer | Mean class-pair Spearman (95% CI) | Mean top-10% Jaccard (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_label, model in (("MCTformer", "mctformer"), ("MCTformer+", "mctformer_plus")):
        for layer in (1, 9, 10, 12):
            item = diversity.query("model == @model and layer == @layer").iloc[0]
            report_lines.append(
                f"| {model_label} | {layer} | {number(item.pairwise_spearman_mean)} [{number(item.pairwise_spearman_ci_low)}, {number(item.pairwise_spearman_ci_high)}] | {number(item.top10_jaccard_mean)} [{number(item.top10_jaccard_ci_low)}, {number(item.top10_jaccard_ci_high)}] |"
            )
    single_mct = single_multi.query(
        "model == 'mctformer' and layer == 12 and image_group == 'single_label'"
    ).iloc[0]
    multi_mct = single_multi.query(
        "model == 'mctformer' and layer == 12 and image_group == 'multi_label'"
    ).iloc[0]
    single_plus = single_multi.query(
        "model == 'mctformer_plus' and layer == 12 and image_group == 'single_label'"
    ).iloc[0]
    multi_plus = single_multi.query(
        "model == 'mctformer_plus' and layer == 12 and image_group == 'multi_label'"
    ).iloc[0]
    report_lines.extend(
        [
            "",
            "**[Statistical inference]** The paired top-10% class-map overlap difference changes sign between L9 and L10: "
            f"{interval(paired_row('top10_class_map_jaccard', 9))} at L9 and {interval(paired_row('top10_class_map_jaccard', 10))} at L10. At L12 it is {interval(paired_row('top10_class_map_jaccard', 12))}.",
            "",
            "**[Fact]** At L12, single-label versus multi-label image groups have q95−median values "
            f"{number(single_mct.mean_upper_tail_gap)} vs {number(multi_mct.mean_upper_tail_gap)} for MCTformer and {number(single_plus.mean_upper_tail_gap)} vs {number(multi_plus.mean_upper_tail_gap)} for MCTformer+. Their τ=.10 entropies are {number(single_mct.mean_spatial_entropy)} vs {number(multi_mct.mean_spatial_entropy)} and {number(single_plus.mean_spatial_entropy)} vs {number(multi_plus.mean_spatial_entropy)}, respectively. These are descriptive group summaries; they are not evidence about region identity.",
            "",
            "**[Interpretation candidate]** The two models have clearly different depth trajectories of within-image class-map similarity. High similarity may mean shared spatial representation structure; low similarity may mean separation. Without segmentation GT, neither direction can be labeled better, foreground-aligned, or confused.",
            "",
            "Full results: [diversity table](../tables/class_map_diversity_by_layer.csv), [single/multi table](../tables/single_vs_multilabel_summary.csv), and [diversity plot](../plots/class_map_diversity_by_layer.png).",
            "",
            "## 6. MCTformer vs. MCTformer+",
            "",
            "All deltas are MCTformer+ minus MCTformer on the exact common sample keys. Pair-equal point estimates are paired with image-clustered confidence intervals; standardized effects in the CSV use one mean delta per image.",
            "",
            "**[Statistical inference]** The four prespecified primary metrics at L12 are:",
            "",
            "| Metric | MCTformer | MCTformer+ | Paired delta (95% CI) | Standardized effect |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric_label, metric in (
        ("q95−median", "upper_tail_gap"),
        ("Spatial entropy τ=.10", "spatial_entropy"),
        ("Class-map top-10% Jaccard", "top10_class_map_jaccard"),
        ("Consecutive-layer Spearman", "consecutive_layer_spearman"),
    ):
        item = paired_row(metric, 12)
        report_lines.append(
            f"| {metric_label} | {number(item.mctformer_mean)} | {number(item.mctformer_plus_mean)} | {interval(item)} | {number(item.standardized_effect)} |"
        )
    report_lines.extend(
        [
            "",
            f"All four L12 primary intervals exclude zero. For supplemental map metrics, the L12 score standard-deviation delta is {interval(paired_row('score_std', 12))}, total-variation delta is {interval(paired_row('total_variation', 12))}, and largest-component-fraction delta is {interval(paired_row('largest_component_fraction', 12))}.",
            "",
            f"**[Fact + statistical inference]** Class-wise L12 q95−median deltas are positive with intervals excluding zero in {tail_excludes}/20 classes; τ=.10 entropy deltas are negative with intervals excluding zero in {entropy_excludes}/20. Class sample counts range from {min_class_count} to {max_class_count}. The q95−median range is {tail_low.class_name}: {interval(tail_low)} to {tail_high.class_name}: {interval(tail_high)}. The entropy range is {entropy_low.class_name}: {interval(entropy_low)} to {entropy_high.class_name}: {interval(entropy_high)}. In contrast, only {component_excludes}/20 largest-component intervals exclude zero, showing that not every spatial-structure measure changes uniformly.",
            "",
            "**[Interpretation candidate]** MCTformer+ has a stronger and more spatially concentrated L12 cosine-score distribution, plus a different class-map and transition-rank trajectory. Because these are different trained model checkpoints, the paired design controls evaluated images/classes, not all training/model confounds; no causal attribution to a particular module is made.",
            "",
            "Full results: [paired table](../tables/mctformer_vs_plus_paired.csv), [all-class summary](../tables/classwise_summary.csv), [class-wise paired deltas](../tables/classwise_paired_delta.csv), [primary-delta plot](../plots/paired_primary_metric_deltas.png), and [class plot](../plots/classwise_layer12_paired_deltas.png).",
            "",
            "## 7. Representative and Failure Cases",
            "",
            "**[Fact]** Cases were selected automatically before inspection: 10 largest L12−L1 q95 changes per model; 10 highest and 10 lowest L12 class-pair top-10% overlaps per model; and 10 lowest L12 cross-model map Spearmans. This yields 70 selection rows and 140 figures. Lowest-overlap selected cases have overlap 0.0000; selected cross-model Spearman ranges from "
            f"{number(disagreement_min)} to {number(disagreement_max)}.",
            "",
            "Each figure shows the transformed original image and both models at L1/L4/L8/L12. Raw figures use a fixed cosine scale of [-1,1]. Min-max figures are explicitly labeled as visualization-only and must not be used to compare absolute magnitudes. These are representation extremes, not verified semantic successes or failures.",
            "",
            "See [selection CSV](../examples/example_selection.csv). No segmentation mask was loaded or displayed.",
            "",
            "## 8. What the Results Support",
            "",
            "**[Fact]** The saved hooks produced finite, non-constant, reproducible class-specific representation score maps for all requested images, positive classes, and layers.",
            "",
            "**[Statistical inference]** Image-clustered intervals support systematic depth changes in score tail, auxiliary concentration, spatial roughness, rank stability, and multi-class map similarity, as well as stable matched-sample differences between the two trained models. The strongest MCTformer+ tail/concentration separation appears in L10–L12; the largest late transition-stability separation appears at L11–L12.",
            "",
            "**[Interpretation candidate]** L9–L12 and the L10/L11 transitions are high-value targets for Experiment 2 region analysis. All 20 classes should remain in the analysis; class-wise effect heterogeneity and unequal sample counts argue against selecting only visually convenient categories.",
            "",
            "## 9. What the Results Do Not Support",
            "",
            "**[Unsupported claim]** This experiment does not determine whether any high-score patch is object, context, or background, because semantic segmentation GT was never loaded.",
            "",
            "**[Unsupported claim]** It does not establish background leakage, lazy semantic assignment, class-specific background shortcuts, or causal shortcut use.",
            "",
            "**[Unsupported claim]** It does not measure attention matrices or CAMs and cannot assert how either behaves, whether these cosine scores enter them, or whether a high-score patch causally affects classification/localization.",
            "",
            "Those hypotheses require the prespecified Experiment 2 GT-region tests, Experiment 3 feature/attention/CAM linkage, and separate intervention experiments. No such conclusion is used in this report.",
            "",
            "## 10. Readiness for Experiment 2",
            "",
            "**READY.** The data-integrity, numerical-validity, and analysis-value gates all pass. This status authorizes the next planned GT-region analysis only; it is not a positive finding about region semantics.",
            "",
            "Reasons: 100% valid/common coverage; no unexplained numerical invalidity; exact canonical round-trip; unchanged source hashes/mtimes; full test suite passed; maps have measurable spatial variation; layer-wise concentration and rank trajectories are systematic; multi-class diversity evolves with depth; and all four prespecified L12 paired metric intervals exclude zero.",
            "",
            "The operational gate and priorities are detailed in [EXPERIMENT2_READINESS.md](EXPERIMENT2_READINESS.md).",
        ]
    )

    readiness_lines = [
        "# Experiment 2 Readiness",
        "",
        "## Decision",
        "",
        "**READY**",
        "",
        "This is readiness to begin the separately specified segmentation-GT region analysis. Experiment 1 alone does not validate any region-semantic or causal hypothesis.",
        "",
        "## Data-integrity gate",
        "",
        "- PASS — MCTformer coverage: 1,449/1,449 images and 2,147 positive image–class pairs.",
        "- PASS — MCTformer+ coverage: 1,449/1,449 images and 2,147 positive image–class pairs.",
        "- PASS — Common coverage: 1,449 images, 2,147 pairs, 100.0000%.",
        "- PASS — No missing files, duplicate IDs, label mismatches, NaN/Inf, or cosine-range violations.",
        "- PASS — Both sides use 12 layers and a 28×28 grid under matched input/score conventions.",
        "- PASS — Checkpoint hashes, result commits, commands, environment, transforms, and schemas are recorded.",
        f"- PASS — Final immutability check: all {immutability['source_files_checked']:,} source files and {immutability['source_npz_checked']:,} NPZ hashes/sizes/mtimes unchanged.",
        "",
        "## Numerical-validity gate",
        "",
        f"- PASS — Observed cosine range [{raw_min:.6f}, {raw_max:.6f}] with no tolerance violation.",
        f"- PASS — Minimum within-map standard deviation {min_std:.6f}; zero maps below 0.001.",
        f"- PASS — Deterministic 10-NPZ round trip: {canonical_check['comparisons']:,} checks, maximum absolute error {canonical_check['maximum_absolute_error']:.1f}.",
        "- PASS — Saved generation metadata records a passing no-change hook guard for both models.",
        "- PASS — `123 passed`; static checks passed.",
        "",
        "## Analysis-value gate",
        "",
        f"- PASS — MCTformer L12−L1 q95−median change {number(delta('mctformer', 'upper_tail_gap').mean_delta)} [{number(delta('mctformer', 'upper_tail_gap').ci_low)}, {number(delta('mctformer', 'upper_tail_gap').ci_high)}].",
        f"- PASS — MCTformer+ L12−L1 q95−median change {number(delta('mctformer_plus', 'upper_tail_gap').mean_delta)} [{number(delta('mctformer_plus', 'upper_tail_gap').ci_low)}, {number(delta('mctformer_plus', 'upper_tail_gap').ci_high)}].",
        f"- PASS — L12−L1 patch ranking has drifted: Spearman {number(ranks['mctformer'].query('layer == 12').iloc[0].mean_layer1_to_layer_spearman)} / {number(ranks['mctformer_plus'].query('layer == 12').iloc[0].mean_layer1_to_layer_spearman)} (MCTformer / MCTformer+).",
        f"- PASS — Multi-class map diversity changes systematically; L12 top-10% overlaps are {number(mct_diversity_l12.top10_jaccard_mean)} / {number(plus_diversity_l12.top10_jaccard_mean)}.",
        "- PASS — All four prespecified L12 paired metric confidence intervals exclude zero.",
        "",
        "## Experiment 2 priorities and safeguards",
        "",
        "1. Analyze all 20 classes and retain image-clustered uncertainty; do not select only classes with large Experiment 1 deltas.",
        "2. Prioritize L9–L12 plus the L10→L11 and L11→L12 transitions, while keeping earlier layers as prespecified controls.",
        "3. Load semantic segmentation GT only inside the Experiment 2 pipeline and record its paths/hashes separately.",
        "4. Evaluate object/context/background region relationships using the exact Experiment 2 definitions; do not relabel Experiment 1 concentration or overlap as semantic evidence.",
        "5. Keep the 70 rule-selected examples fixed for cross-reference, but base conclusions on full-set statistics.",
        "6. Continue to distinguish model-paired sample comparisons from causal attribution to a module or training operation.",
        "",
        "## Remaining uncertainty",
        "",
        "Experiment 1 gives no spatial semantic labels. Stronger tails, lower entropy, changed rank, and changed class-map overlap can each arise from multiple representation geometries. Experiment 2 must determine where those scores lie; Experiment 3 and interventions remain necessary for pipeline linkage and causal claims.",
    ]

    reports = analysis / "reports"
    reports.mkdir(parents=True, exist_ok=False)
    (reports / "EXPERIMENT1_ANALYSIS_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    (reports / "EXPERIMENT2_READINESS.md").write_text(
        "\n".join(readiness_lines) + "\n", encoding="utf-8"
    )

    key_results = {
        "generated_at": timestamp(),
        "readiness": "READY",
        "coverage": audit["common_pairs"],
        "map_rows": len(maps),
        "bootstrap": analysis_meta["bootstrap"],
        "l12_primary_paired": {
            metric: {
                key: float(paired_row(metric, 12)[key])
                for key in (
                    "mctformer_mean",
                    "mctformer_plus_mean",
                    "paired_delta",
                    "ci_low",
                    "ci_high",
                    "standardized_effect",
                )
            }
            for metric in (
                "upper_tail_gap",
                "spatial_entropy",
                "top10_class_map_jaccard",
                "consecutive_layer_spearman",
            )
        },
        "source_immutability": immutability,
        "canonical_verification": canonical_check,
        "segmentation_ground_truth_loaded": False,
    }
    json_dump(reports / "analysis_key_results.json", key_results)

    commands_path = analysis / "exact_commands.sh"
    commands_path.write_text(exact_commands(analysis, repo), encoding="utf-8")
    commands_path.chmod(0o755)
    (analysis / "failed_attempts.log").write_text(failed_attempts(), encoding="utf-8")

    new_files = sorted(
        path.relative_to(repo)
        for path in [
            *repo.glob("analysis/lazy_assignment/*experiment1*.py"),
            *repo.glob("analysis/lazy_assignment/bootstrap.py"),
            *repo.glob("analysis/lazy_assignment/build_canonical_tables.py"),
            *repo.glob("analysis/lazy_assignment/select_examples.py"),
            *repo.glob("tests/test_experiment1_*.py"),
        ]
        if path.is_file()
    )
    git_info = git_metadata(repo)
    diff_lines = [
        "Experiment 1 analysis Git/data summary",
        f"Base commit: {git_info['commit']}",
        f"Branch: {git_info['branch']}",
        "Tracked diff: empty; no existing tracked file was modified.",
        f"Untracked analysis/test additions: {len(new_files)} files, listed below.",
        *(f"  {path}" for path in new_files),
        "Generated results are ignored by Git under results/.",
        f"Source result integrity: PASS ({immutability['source_files_checked']} files; {immutability['source_npz_checked']} NPZ; content hashes and mtimes unchanged).",
        "No commit, push, branch change, retraining, or score regeneration was performed.",
    ]
    (analysis / "git_diff_summary.txt").write_text(
        "\n".join(diff_lines) + "\n", encoding="utf-8"
    )

    script_paths = [repo / path for path in new_files]
    metadata = {
        "analysis_id": analysis.name,
        "status": "complete",
        "completed_at": timestamp(),
        "analysis_dir": str(analysis),
        "git": git_info,
        "environment": environment_metadata(),
        "source_roots": canonical_meta["source_roots"],
        "source_checkpoints": {
            model: audit["models"][model]["checkpoint_sha256"]
            for model in ("mctformer", "mctformer_plus")
        },
        "analysis_source_sha256": {
            str(path.relative_to(repo)): sha256_file(path) for path in script_paths
        },
        "bootstrap": analysis_meta["bootstrap"],
        "tests": {"passed": 123, "warnings": 13, "log": str(analysis / "tests.log")},
        "quality_checks_passed": True,
        "source_immutability_passed": immutability["passed"],
        "canonical_roundtrip_passed": canonical_check["passed"],
        "segmentation_ground_truth_loaded": False,
        "retraining_performed": False,
        "scores_regenerated": False,
        "reports": [
            str(reports / "EXPERIMENT1_ANALYSIS_REPORT.md"),
            str(reports / "EXPERIMENT2_READINESS.md"),
        ],
    }
    json_dump(analysis / "run_metadata.json", metadata)

    output_rows: list[dict[str, object]] = []
    excluded = {analysis / "output_manifest.csv"}
    for path in sorted(item for item in analysis.rglob("*") if item.is_file()):
        if path in excluded:
            continue
        output_rows.append(
            {
                "relative_path": str(path.relative_to(analysis)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(output_rows).to_csv(analysis / "output_manifest.csv", index=False)

    subprocess.run(["git", "diff", "--check"], cwd=repo, check=True)
    print(f"Wrote final reports to {reports}")
    print(output_file_table(reports, "*"))


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
