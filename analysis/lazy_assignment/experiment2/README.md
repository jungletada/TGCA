# Experiment 2: Semantic Ownership

This directory implements the frozen-model Experiment 2 pipeline described in
`docs/Experiment_2_Semantic_Ownership_Feature_Attention_CAM_Codex_Plan.md`.
It measures the GT ownership of class-specific feature, attention, and native
CAM supports for the exact Experiment 1 MCTformer and MCTformer+ checkpoints.
It does not train or modify either model.

## Safety and statistical contracts

- Experiment 1 results, checkpoints, VOC images, and VOC semantic masks are
  immutable inputs. The audit and post-run verifier record every input SHA-256.
- Production signal generation requires a clean Git checkout and tracked
  runtime sources. `--allow-uncommitted-source` is restricted to limited smoke
  runs, and `--allow-smoke` is likewise restricted to non-report validation.
- RGB uses the exact Experiment 1 448 transform. The semantic mask receives the
  same resize/crop geometry with nearest-neighbor interpolation.
- Native patch CAM, last-three class-attention CAM, and final patch-affinity CAM
  are numerically checked against the official model output.
- Confidence intervals resample complete `image_id` clusters. Patches,
  image-class rows, and class-pair rows from one image are never independent
  bootstrap units. Paired effects are MCTformer+ minus MCTformer on exact common
  keys.
- Each aggregate row reports the metric-specific finite image/row denominator,
  the actual derived draw seed, and the fraction of finite bootstrap replicates.
- Scientific report generation requires full signal roots, exactly 1,449 images
  per model, exactly 5,000 bootstrap repeats, and matching table hashes.

## Stages

1. `audit_experiment2_inputs.py`: locate and hash the exact Experiment 1 roots,
   checkpoints, VOC inputs, and paired analysis.
2. `verify_experiment2_geometry.py`: deterministic 20-image geometry smoke.
3. `run_experiment2_signals.py`: one native forward per batch plus read-only
   hooks; saves reduced per-image signals and numerical-equivalence metadata.
4. `verify_experiment2_immutability.py`: re-hash all audited source inputs.
5. `build_experiment2_canonical.py`: stream nine Zstandard Parquet tables with
   schema, manifest, source-hash, and row-group round-trip checks, including
   all-class classification logits and per-image raw-CAM confusion matrices.
6. `analyze_experiment2.py`: layer/CAM ownership, shared/new support, feature to
   attention to CAM transitions, probe controls, failure-pattern prevalence,
   checkpoint classification mAP, fixed-threshold raw final-CAM mIoU,
   class/multi-label strata, and common-key paired image-cluster bootstrap CIs.
   Joint post-block-cosine/patch-L2-norm controls test whether high-score
   background support is concentrated in relatively high- or low-norm patches.
   Classification-conditioned subsets are model-specific and therefore are not
   mislabeled as paired comparisons; paired tables always use exact common keys.
   Before this stage, create and freeze `exact_commands.sh` and
   `pipeline_metadata.json` in the analysis output directory's parent. Their
   hashes, together with `analysis.log`, are bound into `analysis_metadata.json`
   and are required unchanged by report generation.
7. `plot_experiment2.py`: the thirteen pre-registered aggregate diagnostics.
8. `select_experiment2_examples.py`: deterministic GT-driven case manifest plus
   the fixed seventy Experiment 1 cases. Every selected multi-label Experiment 2
   image records at least two GT-positive display classes.
9. `render_experiment2_examples.py`: model-free seven-column panels from the
   immutable NPZ signals and VOC GT; fixed Experiment 1 figures are linked, not
   redrawn.
10. `generate_experiment2_report.py`: full-set-only scientific report and the
    uncertainty-aware Case A–G (or unresolved) next-experiment decision. The
    exact operational tests, all simultaneously satisfied cases, and the locked
    `G,F,E,A,B,C,D` primary-case precedence are versioned before the full run;
    the source plan did not prescribe an exact precedence.

`delivery_validation.py` makes reporting fail closed unless all registered plots,
their input tables, deterministic selections, rendered panels, immutable NPZ
links, and source metadata match their recorded SHA-256 values. Smoke outputs can
exercise these stages but cannot be promoted to a scientific report.

All commands expose `--help`. Long production runs must be launched in `tmux`
and their stdout/stderr captured. A production result root should contain the
audit, the two signal roots, canonical tables, analysis tables, plots, example
selection/rendering, reports, test logs, `exact_commands.sh`, and pipeline
metadata.

## Required environment

Use the independent TGCA/MCTformer+ Conda environment:

```bash
conda run -n tgca-repro python -m pytest -q tests
```

Do not install MoRe or CTI dependencies into this environment.
