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

## Current verification

Implementation verification on 2026-08-27:

- 39 deterministic BCSS/TGCA/host tests pass;
- existing E0 checkpoint loads strictly;
- E0 CAM matches the pre-BCSS equations exactly (`rtol=0`, `atol=0`);
- E6 CUDA FP16 forward/backward is finite;
- maximum observed patch-wise ownership row-sum error is `1.19e-7`;
- two-image unified dump, CBL, background AUPRC, PNG/PDF visualization, and a
  one-image context/object counterfactual smoke completed under `results/smoke/`.

These are implementation checks, not BCSS scientific results. No six-variant
screen has been launched from the dirty implementation worktree.
