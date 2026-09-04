# Experiment 3: three inference-only validations

This directory implements only the three diagnostics pre-registered in
`docs/Experiment_3_Three_Low_Cost_Validations_Codex_Plan.md`:

- **A — Presence axis:** fixed all-ones decomposition plus a deterministic,
  two-fold cross-fitted shared-presence direction.
- **B — CAM layer readout:** B0--B5 diagnostic layer readouts with each host's
  exact native CAM formula and its unchanged all-layer patch propagation.
- **C — late C2C intervention:** C0--C5 mass-preserving class-to-class
  self-rerouting, implemented only as exception-safe runtime hooks.

No entry point trains a model or writes to Experiment 1/2 results,
checkpoints, or VOC data. Every inference runner requires `tgca-repro`, a
completed input audit, batch size 8, the deterministic 448 transform, and a
new output directory. A full run additionally refuses dirty or untracked
runtime source. Smoke runs may use `--allow-uncommitted-source` only together
with a positive `--limit`.

## Production sequence

Run from a clean worktree at the tagged Experiment 3 implementation commit.
Replace the two shell variables below with new, non-existent paths.

```bash
EXP2_ROOT=/home/peng/code/TGCA/results/lazy_assignment/experiment2_semantic_ownership/20260904-exp2-semantic-ownership-voc-val-full-0d47db4-v1
EXP3_ROOT=/home/peng/code/TGCA/results/lazy_assignment/experiment3_three_validations/<run_id>
```

The recommended production entry point initializes the new run root and its
complete command/status ledger atomically, then executes the fail-closed queue
in the order shown below:

```bash
python analysis/lazy_assignment/experiment3/run_experiment3_pipeline.py \
  --experiment2-root "$EXP2_ROOT" \
  --run-root "$EXP3_ROOT" \
  --implementation-tag <annotated-production-tag> \
  --device cuda:0 --num-workers 4 \
  --context-document docs/Experiment_3_Three_Low_Cost_Validations_Codex_Plan.md \
  --context-document docs/Experiment1_Independent_Scientific_Analysis.md \
  --context-document docs/Experiment2_Independent_Detailed_Analysis.md
```

For auditability, the individual commands expanded by that queue are described
next. Do not execute this expansion in addition to a live queue. First audit
every immutable input:

```bash
python analysis/lazy_assignment/experiment3/audit_experiment3_inputs.py \
  --experiment2-root "$EXP2_ROOT" \
  --output-dir "$EXP3_ROOT/audit"
```

Run Validation A once per host and then analyze the paired runs:

```bash
python analysis/lazy_assignment/experiment3/run_presence_axis_analysis.py \
  --model mctformer_plus --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/presence_axis/mctformer_plus" \
  --batch-size 8 --num-workers 4 --device cuda:0
python analysis/lazy_assignment/experiment3/run_presence_axis_analysis.py \
  --model mctformer --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/presence_axis/mctformer" \
  --batch-size 8 --num-workers 4 --device cuda:0
python analysis/lazy_assignment/experiment3/analyze_presence_axis.py \
  --mctformer-run-root "$EXP3_ROOT/presence_axis/mctformer" \
  --mctformer-plus-run-root "$EXP3_ROOT/presence_axis/mctformer_plus" \
  --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/presence_axis/analysis" \
  --bootstrap-repeats 5000 --bootstrap-seed 20260901
```

Run and analyze Validation B:

```bash
python analysis/lazy_assignment/experiment3/run_cam_layer_intervention.py \
  --model mctformer_plus --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/cam_layer_intervention/mctformer_plus" \
  --batch-size 8 --num-workers 4 --device cuda:0
python analysis/lazy_assignment/experiment3/run_cam_layer_intervention.py \
  --model mctformer --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/cam_layer_intervention/mctformer" \
  --batch-size 8 --num-workers 4 --device cuda:0
python analysis/lazy_assignment/experiment3/analyze_cam_layer_readout.py \
  --mctformer-run-root "$EXP3_ROOT/cam_layer_intervention/mctformer" \
  --mctformer-plus-run-root "$EXP3_ROOT/cam_layer_intervention/mctformer_plus" \
  --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/cam_layer_intervention/analysis" \
  --bootstrap-repeats 5000 --bootstrap-seed 20260901
```

Validation C is primary on MCTformer+. Its 50-image smoke must pass before the
full command. The MCTformer C0/C3/C4 architecture control is conditional on a
clear C3/C4 MCTformer+ result and is not launched automatically.

```bash
python analysis/lazy_assignment/experiment3/run_c2c_intervention.py \
  --model mctformer_plus --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --output-dir "$EXP3_ROOT/c2c_intervention/mctformer_plus" \
  --batch-size 8 --num-workers 4 --device cuda:0
python analysis/lazy_assignment/experiment3/analyze_c2c_intervention.py \
  --mctformer-plus-run-root "$EXP3_ROOT/c2c_intervention/mctformer_plus" \
  --output-dir "$EXP3_ROOT/c2c_intervention/analysis" \
  --bootstrap-repeats 5000 --bootstrap-seed 20260901
```

Finally re-hash the complete pre-run source manifest:

```bash
python analysis/lazy_assignment/experiment3/verify_source_immutability.py \
  --before-manifest "$EXP3_ROOT/audit/immutable_manifest_before.csv" \
  --output-dir "$EXP3_ROOT/audit/final_immutability"
```

Render the three rule-selected example sets, then finalize the reports. The
renderer freezes selection to the maximum, lower median, and minimum primary
per-image paired delta with lexical image-ID tie-breaking. The finalizer
requires completed 1,449-image analyses, exactly 5,000 image-clustered
bootstrap draws, and a passing post-run immutability verification. It preserves
existing pipeline stage history and never edits `exact_commands.sh`.

```bash
python analysis/lazy_assignment/experiment3/render_experiment3_examples.py \
  --run-root "$EXP3_ROOT" \
  --validation-a-root "$EXP3_ROOT/presence_axis/analysis" \
  --validation-b-root "$EXP3_ROOT/cam_layer_intervention/analysis" \
  --validation-c-root "$EXP3_ROOT/c2c_intervention/analysis" \
  --source-metadata "$EXP3_ROOT/audit/source_metadata.json" \
  --dpi 120
python analysis/lazy_assignment/experiment3/generate_experiment3_report.py \
  --run-root "$EXP3_ROOT" \
  --validation-a-root "$EXP3_ROOT/presence_axis/analysis" \
  --validation-b-root "$EXP3_ROOT/cam_layer_intervention/analysis" \
  --validation-c-root "$EXP3_ROOT/c2c_intervention/analysis" \
  --source-verification "$EXP3_ROOT/audit/final_immutability/immutability_verification.json"
```

Each runner records its exact invoked command, environment manifests, source
hashes, immutable manifest linkage, numerical equivalence gates, per-artifact
SHA-256, and a completion marker. Analyses use exactly 5,000 whole-image
clustered paired bootstrap draws; patches and multiple classes from one image
are never resampled independently.
