# BCSS diagnostic and VOC screening protocol

This file is the executable companion to `docs/CHAT_HANDOFF-0827.md`.
It freezes the implementation choices for the diagnostic system and the
single-seed VOC minimum screen. The older TGCA runners remain available only
for reproducing completed TGCA evidence.

## Frozen variants

All variants use vanilla MCTformer+ attention, the same DeiT initialization,
45 classification epochs, seed 0, input size 448, CAM scales
`1.0,0.75,1.25`, and fixed background threshold 0.45.

| ID | Implemented change | CAM selection | Auxiliary loss |
|---|---|---|---|
| E0 | Original MCTformer+ | Original | None |
| E1 | Generic register in backbone self-attention | Original | None |
| E2 | Background-named token in backbone self-attention | Independent background-to-patch gate | None |
| E4 | Post-encoder class/background competition | Competitive ownership gate | Foreground anchor |
| E5 | E4 | Competitive ownership gate | Foreground anchor + background null |
| E6 | E5 + one zero-gated slot update | Competitive ownership gate | Foreground anchor + background null |

E2 is deliberately stronger than an untrained post-encoder query: its token is
inside backbone attention and receives gradients from the unchanged base
classification objectives. It is still not the final BCSS design because its
background map is generated independently and has neither class/background
competition nor null-class supervision. E4--E6 use the role-decoupled
post-encoder decoder and never write slot features into the patch stream.

The foreground semantic classifier reuses the spatial mean of the existing
MCTformer+ `3 x 3` patch-classifier kernel. No extra classification head is
introduced. Ownership calibration preserves the original class-to-patch row
mass; consequently beta 0 is exactly the E0 CAM path.

## Schedule

The final defaults are:

```text
tau = 0.5
beta = 0.5
lambda_fg = 0.5
lambda_bg = 0.1
background slots = 1
```

Epochs 0--2 use `tau=1`, `beta=0`, and zero refinement strength. Epochs 3--8
linearly move to the final values. E6 has a learnable scalar update gate
initialized to zero. Training RNG is reset after variant-specific parameter
initialization so all six variants use the same sampling, augmentation,
DropPath, and dropout random stream.

## Diagnostic data contract

`analysis.dump_attention` exports one compressed NPZ per image. Every variant
contains:

- patch-classification CAM;
- last-three-layer class-to-patch map;
- propagated final CAM;
- class logits and patch-feature norm.

E1 additionally contains register-to-patch and patch-to-register attention.
E2 contains both directions for its independent background token. E4--E6
contain raw background score, foreground/background ownership, and the full
patch-wise ownership distribution. `--save-layer-head` retains the directional
layer/head tensors; the default unified dump stores compact aggregate maps.

The metric entry points are:

```text
python -m analysis.ownership_metrics
python -m analysis.background_metrics
python -m analysis.counterfactual
python -m analysis.visualize_slots
python -m analysis.sweep_parameters
```

They produce JSON and CSV alongside plots. Background AUPRC and Spearman are
threshold-free. Any max scaling used for register/E2 IoU is written into the
metric JSON and must not be confused with the direct probability-valued BCSS
ownership map.

## Commands

Run diagnostics for one completed run:

```bash
source /home/peng/anaconda3/etc/profile.d/conda.sh
conda activate tgca-repro
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/diagnostics/run_bcss_voc_diagnostics.sh RUN_DIR VARIANT
```

Run one full VOC variant:

```bash
BCSS_GPU_ID=0 BCSS_SEED=0 \
  bash experiments/ablations/run_bcss_voc_variant.sh e6
```

Queue the complete minimum screen after the implementation is reviewed and the
tracked worktree is committed cleanly:

```bash
tmux new-session -d -s bcss-voc-screen \
  "bash -lc 'source /home/peng/anaconda3/etc/profile.d/conda.sh && \
  conda activate tgca-repro && cd /home/peng/code/TGCA && \
  BCSS_GPU_ID=0 bash experiments/ablations/run_bcss_voc_screen.sh'"
```

The queue refuses a tracked dirty worktree and refuses to overwrite queue, run,
diagnostic, sweep, or comparison directories. It runs E0, E1, E2, E4, E5, and
E6 sequentially, evaluates fixed-threshold CAM precision/recall, generates the
mechanism diagnostics, runs a 200-image inference-only tau/beta screen for E6,
and writes a validated comparison with the go/no-go checks.

Context-only/object-only evaluation is implemented but disabled in the minimum
screen because it is not one of the September 3 selection gates. Enable it for
a diagnostic run with `BCSS_RUN_COUNTERFACTUAL=1`; optionally limit the initial
audit using `BCSS_COUNTERFACTUAL_MAX_IMAGES`.

## Completed VOC screen

The complete seed-0 queue ran from `2026-08-27T19:09:11+09:00` to
`2026-08-28T03:28:12+09:00` at commit
`4147fc368fdef2698d3a0570c6c5b913527327ee`. All variants completed 45 epochs
and wrote final checkpoint hashes, CAM metrics, diagnostic dumps, and
completion markers. The queue log and validated comparison are:

```text
results/queues/4147fc3/bcss-voc-screen-20260827-190911.log
results/mctformerplus/voc/comparisons/
  bcss-screen-20260827-190911-s0-4147fc3/screen.json
```

Fixed-threshold VOC validation results (`threshold=0.45`) are:

| ID | CAM mIoU | Sem. precision | Sem. recall | Best cls. mAP | CBL | CCS-bg | BG AUPRC | Pred. BG fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 70.063 | 80.735 | 85.817 | 96.545 | 0.5785 | 0.4467 | 0.9673 | 0.6461 |
| E1 | 69.602 | 78.959 | 86.682 | 96.705 | 0.5826 | 0.4323 | 0.6115 | 0.0176 |
| E2 | 68.713 | 77.254 | 87.475 | 96.705 | 0.5832 | 0.4306 | 0.6115 | 0.0176 |
| E4 | 56.060 | 78.433 | 66.292 | 96.678 | 0.5965 | 0.3878 | 0.7338 | 0.9961 |
| E5 | 55.551 | 77.593 | 65.583 | 96.691 | 0.5977 | 0.4181 | 0.7349 | 0.9944 |
| E6 | 56.145 | 77.986 | 66.089 | 96.291 | 0.5925 | 0.3993 | 0.7392 | 0.9968 |

Background AUPRC values above do not all describe the same map: E0 uses the
CAM complement, E1/E2 use token attention, and E4--E6 use learned ownership.
They are useful counterfactual diagnostics, not interchangeable task scores.
The matched E0 for this queue is `70.063`; the older TGCA pilot's `69.50`
vanilla result came from commit `22427d6` and a different queue. Use E0, rather
than the historical run, for every BCSS delta.

### Go/no-go conclusion

E6 relative to E0 changes raw CAM mIoU by `-13.918` points, semantic precision
by `-2.749` points, semantic recall by `-19.727` points, CBL by `+2.41%`
(worse), and CCS-bg by `-10.61%`. Best classification mAP changes by `-0.255`
points. Therefore only the classification-retention condition passes; all five
task/mechanism improvement checks fail. E6 beats the generic register only on
the available background AUPRC check and loses raw CAM mIoU and CBL, so the
interim register gate also fails (`1/3` wins; CRS and final segmentation were
not run).

The ablation sequence gives the following conclusions:

- E1 shows that a generic register is not a semantic-background estimator. It
  costs `0.462` CAM mIoU and its predicted-background fraction is only `1.76%`.
- E1 and E2 checkpoints are bitwise identical after the token-role key rename.
  E2 therefore isolates CAM gating, which costs another `0.888` mIoU.
- E2 to E4 introduces competition and loses `12.653` mIoU, mainly through a
  `21.183`-point semantic-recall collapse; classification remains intact.
- E4 to E5 shows that the background-null loss optimizes successfully but does
  not learn semantic background. It loses another `0.508` mIoU and worsens both
  CBL and CCS-bg.
- E5 to E6 gives only a `+0.594`-point recovery. The learned E6 update gate is
  `-0.0141`, so the slot update is nearly unused and does not repair collapse.

### Failure mechanism

Across all `1,573,622` validation patch positions, E4, E5, and E6 put more than
`0.9` background ownership on `99.20%`, `99.00%`, and `99.30%` of patches,
respectively. Their mean total active-foreground ownership is only `0.00449`,
`0.00603`, and `0.00378`. Maximum ownership row-sum error is below `2.4e-7`, so
the competitive softmax is numerically correct.

The degeneracy is in spatial aggregation and supervision. Each foreground slot
is divided by its own spatial ownership sum before classification. A class slot
can therefore own only an epsilon amount at every patch, still produce a
normalized aggregate feature, and minimize the foreground anchor loss. The
background slot can simultaneously own nearly the entire image and satisfy the
null loss. Low auxiliary losses consequently do not imply meaningful ownership.

The map decomposition also localizes the task failure: patch-classifier maps
remain close to E0, while class-to-patch selection and subsequent affinity
propagation lose foreground coverage. Reducing CCS-bg with a stronger ownership
gate trades directly against recall rather than producing a better seed.

The E6 200-image inference-only sweep does not find a calibration fix. Its best
diagnostic point is `tau=0.35, beta=0.25`, but every tested setting still has a
predicted-background fraction above `99.7%`. Increasing beta reduces CCS-bg
while monotonically damaging localization. The sweep is stored at:

```text
results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e6-s0-4147fc3/
  bcss_diagnostics/parameter_sweep/sweep.json
```

This formulation is a no-go. Do not run three seeds, COCO, downstream
segmentation, or the independent host until the loss makes total foreground
ownership identifiable. The next implementation must first include a
deterministic case in which epsilon foreground ownership is penalized, then run
one E4-like seed-0 debug before restoring the full ablation queue.

## Mass-aware E4 debug

The only authorized follow-up to the failed screen is `e4_mass`, a seed-0
E4-like debug variant. Historical `e4`, `e5`, and `e6` behavior is unchanged.
`e4_mass` has the same model parameters, competitive ownership, normalized slot
aggregate, foreground anchor weight, tau/beta schedule, CAM gate, and evaluation
pipeline as E4. It changes only the logits consumed by the foreground anchor.

For class slot `c`, define its resolution-normalized total ownership as:

```text
m_c = (1 / N) sum_j O_cj
```

The debug foreground logits are:

```text
r_c_mass = m_c r_c
```

and the existing one-class cross-entropy is applied to `r_c_mass`. This retains
the magnitude of total ownership without setting a foreground/background area
target. As `m_c` approaches zero, all logits approach zero and the loss
approaches `log(N_c)`, so spatial normalization can no longer hide an epsilon
foreground slot from the anchor. Slot aggregation itself remains unchanged to
isolate the loss correction.

Deterministic verification requires:

- legacy foreground loss is identical for healthy and epsilon ownership when
  the normalized aggregate is held fixed;
- mass-aware loss is strictly larger for epsilon ownership and approaches
  `log(N_c)`;
- the loss has a finite nonzero gradient to ownership magnitude;
- E4 and E4-mass parameters, forward outputs, ownership, and aggregates are
  exactly equal before the auxiliary loss;
- CUDA FP16 forward/backward is finite.

All 44 repository tests pass. A read-only 50-image counterfactual on the
completed collapsed E4 checkpoint gives old foreground-anchor loss `0.00130`,
mass-aware loss `2.93545`, mean active foreground ownership `0.00840`, and
`log(20)=2.99573`. The new loss therefore detects the observed collapse rather
than only an artificial unit-test case.

Run exactly one full debug experiment:

```bash
BCSS_GPU_ID=0 BCSS_SEED=0 BCSS_FIXED_THRESHOLD=0.45 \
  BCSS_TAU=0.5 BCSS_BETA=0.5 BCSS_LAMBDA_FG=0.5 \
  bash experiments/ablations/run_bcss_voc_variant.sh e4_mass
```

Do not add E5/E6, tune tau/beta/loss weights, or launch another seed before
reviewing this run. The primary comparison remains matched E0 `70.063`; compare
the debug directly with historical E4 `56.060` to determine whether removing
the loss degeneracy restores foreground ownership and CAM recall.

## Implementation verification

Implementation verification on 2026-08-27:

- 39 deterministic BCSS/TGCA/host tests pass;
- existing E0 checkpoint loads strictly;
- E0 CAM matches the pre-BCSS equations exactly (`rtol=0`, `atol=0`);
- E6 CUDA FP16 forward/backward is finite;
- maximum observed patch-wise ownership row-sum error is `1.19e-7`;
- two-image unified dump, CBL, background AUPRC, PNG/PDF visualization, and a
  one-image context/object counterfactual smoke completed under `results/smoke/`.

These checks establish implementation integrity; they do not overturn the
negative scientific result above.
