# E1 detach and E4 channel aggregation execution

Source plan: `docs/MCTformerPlus_Detach_and_ChannelAgg_Plan.md` (read-only,
untracked user file; exact SHA256 saved in the run manifest).

## Audit and scope

The preceding four45-epoch runs and all180 checkpoint evaluations finished.
Existing epoch45 CAM results match the plan (GWRP70.063/68.632;
all-product72.315/62.241, seeds0/11). No retraining of these controls.
Source uses raw final class tokens and `.mean(-1)`; BCSS E0 has no BG token.
`docs/design.md` remains absent; it is not recreated.

Exactly four unconditional fresh trainings, sequential on the single A6000:

1. E1 V2 all-product, detached pooling weights, seed0.
2. E1 V2 all-product, detached pooling weights, seed11.
3. E4 A1-a on GWRP, shared384-channel softmax readout, seed0.
4. E4 A1-a on GWRP, shared384-channel softmax readout, seed11.

Preserve45 epochs,448, batch32,330 updates/epoch, original DeiT-S initialization,
optimizer/augmentation/loss recipe and CAM protocol. Save all180 new epoch
checkpoints, plus best/final. Old files/checkpoints remain read-only.

The plan's header, §2.6, §3.4 and §4 are ambiguous about the conditional two-run
budget when fallback and A1-b both trigger, and do not set a seed11 fallback
success criterion. A non-blocking question was sent to the user. The current
queue executes only the unambiguous first four, records gate outcomes and does
not automatically choose a fallback, launch A1-b, or extend to five seeds.
No register/Gram/COCO/affinity/A2–A4 experiment is included.

## Implementation

- `--detach-weights` defaults false. Only the normalized all-product pooling
  weights are detached; logits and the transformer stay differentiable.
- `--channel-agg` defaults false; disabled readout is exactly the original mean.
  Enabled readout is `mean(T)+sum((softmax(theta)-1/D)*T)`, mathematically the
  requested weighted sum, with bitwise-identical theta0 forward at initialization.
  This avoids a new reduction-order difference between mean and multiply/sum.
- theta is zero-initialized, excluded from weight decay and placed in its own
  optimizer group. A1-a uses multiplier1. Optional multiplier10 is implemented
  but NOT scheduled; the multiplier applies to warmup/cosine/minimum LR alike.
- Raw per-block class tokens and CCT remain unchanged. No patch initialization,
  classifier or CAM refinement changes. CAM/classification load flags are checked
  against checkpoint metadata, including theta for A1.
- Epoch20 FP32 macro-class AP warning runs in a separate inference process
  after saving that checkpoint, preserving parent RNG/model/optimizer state.
  `<80` emits a warning only, never early-stops or changes hyperparameters.

## Measurements

New checkpoints: classification val1449 (FP32 single448), native CAM train1464
(scales1/.75/1.25+flip, fixed.45 and original diagnostic threshold grid), every
epoch. Compare all8 method/seed trajectories using existing old metrics.

All8 trajectories get frozen val1449 diagnostics every epoch: normalized entropy
and top1 mass of all-product C2P weights (also on GWRP as counterfactual weights),
actual pooling-weight entropy/top1, and final raw-token positive-class-pair rho_cc.
Positive classes are averaged within each image, followed by equal-image means;
rho_cc excludes single-label images. M2/M4 reuse artifact_probe definitions at
epochs15/30/45 for all12 layers. No semantic masks are loaded by diagnostics.
A1 entropy, max/min ratio, L1-uniform distance and384 learned channel weights are
recorded. The requested L1 value is not total variation: TV=L1/2.

Gates use literal68.1/71.1 for E1 and+.5 for both E4 seeds. If both V2 seeds fall
below GWRP, the explicit stop rule takes precedence over overlapping table rows.
Entropy>.98 is only a near-uniform flag. Two-seed outcomes do not establish
causal feedback lock-in, prove all variation comes from C2P, prove the mean is
optimal, or establish literature novelty. Such claims are not adopted from the
plan as facts. Appendix in the report documents the GWRP gradient control.

## Execution and artifacts

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/peng/anaconda3/envs/tgca-repro/bin/python -u \
  -m experiments.ablations.run_detach_channel \
  --output results/detach_channel/20260917-voc-s0-s11
```

Tmux: `mct-detach-channel-20260917`.
Log: `results/detach_channel/20260917-voc-s0-s11.queue.log`.
Runner executes focused tests, two2-epoch training/CAM/classification/diagnostic
smokes, then the four full trainings with final evaluation, followed by all-epoch
evaluation and missing historical diagnostics. Final gates/report appear after
the fourth training and are updated when trajectories finish.

Outputs: `manifest.json`, per-run `commands.sh`, config/environment/test logs,
checkpoint SHA256 indices, `final_comparison.csv`, `epoch_metrics.csv`,
`diagnostic_trajectories.csv`, curve SVG/PNG, `decision.json`, and
`DETACH_AND_CHANNEL_AGG_REPORT.md`. `QUEUE_COMPLETE` means the four trainings and
all scheduled diagnostics/evaluations have actually finished.
Classification per-image predictions are deleted only after metric validation;
online CAM avoids per-image arrays. No checkpoint deletion. New storage ~50GB.
Expected runtime ~14–17h including all-epoch evaluations and missing diagnostics,
subject to measured inference speed; not a completion claim.
