# Operational handoff — 2026-09-18

## Current request — Head-axis / alpha plan (2026-09-18)

Plan read in full: `docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md`.
This new authorization is INFERENCE ONLY, two frozen seed0 hosts; no training,
learnable heads, register, Gram, COCO, extra seeds or automatic follow-up.
Previous FP32 repair validation r2 COMPLETE at `92ace1a`:65 tests, frozen probe,
two-epoch smoke, checkpoint verification, classification4/CAM2; all7 source
hashes unchanged. Actual smoke AdamW steps3/4, default scale65536->32768;
this does not establish that all AMP overflow or the seed gap is resolved.
At startup only user tmux `codex-wsss-0`; GPU idle. Three user plans preserved.

Step0 runner: `analysis.head_alpha_preflight`, output
`results/head_alpha/20260918-voc/preflight`. Full FP32 VOCval1449 epoch45
seed0 GWRP/all-product; isolate probability-cast error by FP16 round-trip of
the SAME FP32 forward probabilities. No autocast or TF32; no GT segmentation.
Before seeing results, interpret requested1e-4-order TV gate as BOTH image-mean
and pooled-positive-class mean TV <1e-3 for all-product. GWRP product-TV is
hypothetical, not its classification path. Stop if the prerequisite fails.
Code audit: native C2P=head mean then last3 mean; seed=sqrt(ReLU(M)*a);
P=sum all12 head means; sum upsampled/unflipped views then class min-max.
Later A/H scans are pending, not completed. `docs/design.md` remains absent.

Stage0 completed at `95447b4`, gate PASSED: all-product image TV0.0001501707,
pooled TV0.0001472097; no all-zero positive rows, all source hashes unchanged.
Implementation/pre-registration: `docs/Head_Alpha_Execution_Notes.md`.
Runner `analysis.head_alpha`; smoke then full under
`results/head_alpha/20260918-voc/{smoke,full}`. All forwards FP32/autocastOFF/
TF32OFF; no training. Core9 unit tests passed before integration launch.
Main reports will be `full/HEAD_ALPHA_REPORT.md`, `ALPHA_SWEEP_REPORT.md`,
`HEAD_AXIS_REPORT.md`, `NEXT_EXPERIMENT_DECISION.md` when full is complete.
Plan ambiguities (epsilon/native, crossfit weighting, rownorm multi-view effects,
unlabeled ranking direction and multiplicity limitations) are resolved explicitly
in the execution notes before inspecting scan outcomes.

Initial smoke at `49bdcd8` completed both hosts/all8 inference stages; official
native CAM max error0. Report rendering exposed a metadata key collision:
the numeric crossfit reference was overwritten by its description. This is
fixed without changing inference/statistics. Preserve original smoke results.
Retry paths: `smoke-r2`, then `full`; full did NOT start after failed smoke.

Smoke-r2 at `19be6f6` subsequently completed all stages/reports/integrity checks.
Full started but was explicitly stopped early in GWRP alpha (before head stage)
after a synthetic tiny-map test found FP32 L2-norm underflow in gamma. Fix:
mass-normalize before cosine (mathematically scale-invariant). No source data
changed. Keep partial `full`/log and cancellation marker; new clean run paths
`smoke-r3` then `full-r2`. No training or other experiment was stopped.

## Current request — all-product FP16 precision repair (2026-09-18)

Startup main HEAD: `3e7ae75`. Live state supersedes older launch notes below:
the four VOC baseline trainings/180 evaluations and detach/A1 trainings/180
evaluations plus360 frozen diagnostics are COMPLETE. No experiment was running;
only the user's `codex-wsss-0` tmux session remained. `docs/design.md` remains absent.
Two untracked user plans remain untouched. No full training is authorized by
this bounded repair task, and none is being launched.

Request: verify and repair the FP16 attention boundary in all-product pooling.
New opt-in `--c2p-pooling-fp32` reads head-mean C2P probabilities BEFORE the
normalizer casts to FP16; native attention/V, token updates, CCT and CAM retain
their original path. No engine/loss/scaler/seed changes. Legacy default is false;
checkpoint metadata and evaluation validation distinguish repaired runs.
65 focused unit tests passed before implementation commit.
Runner: `experiments.ablations.validate_c2p_fp32` (frozen seed0/11 epoch12 probe,
then64-image two-epoch smoke, checkpoint reload,4-image classification,2-image CAM).
Output: `results/c2p_fp32_validation/20260918-all-product-r2`.
Planned tmux: `mct-c2p-fp32-check-20260918`. No full matched retraining.
Completion/integrity evidence must be read from `VALIDATION_COMPLETE` and
`manifest.json`; the integration run is pending at this implementation commit.

Initial r1 at `9fe02e5` passed65 tests, the frozen probe, smoke training,
checkpoint verification and classification. CAM invocation failed because the
smoke list lacked `train` in its filename (required by the existing evaluator).
No model failure. Preserve r1 logs/results; r2 corrects only this list filename.
An additional user plan `docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md` appeared
during this task; it is untracked and untouched, and has not been executed.

Host: LHR. Repository: `/home/peng/code/TGCA`, branch `main`.
Environment: `/home/peng/anaconda3/envs/tgca-repro/bin/python`.
Implementation at cancellation: `dd4f370` (diagnostic probes: `5b66a3c`).

## Current request — E1 detach / E4 A1 (2026-09-17)

Previous VOC four-run training and180-epoch evaluation queues are COMPLETE;
last evaluation ended approximately05:57 JST. No experiment GPU process or
experiment tmux remained at new-task startup. Baseline files are read-only.

New plan read completely: `docs/MCTformerPlus_Detach_and_ChannelAgg_Plan.md`.
Scope: detach all-product s0/s11, then A1-a GWRP s0/s11,45epochs each; preserve
all epoch checkpoints; classification/CAM and frozen all-epoch diagnostics.
No automatic ambiguous conditional extension; non-blocking user question sent.
Execution rationale/command: `docs/Detach_Channel_Execution_Notes.md`.
Runner: `experiments.ablations.run_detach_channel`.
Planned tmux: `mct-detach-channel-20260917`.
Output: `results/detach_channel/20260917-voc-s0-s11`.
Log: output path plus `.queue.log`. Unit tests pass before local commit/launch;
two training/inference smoke gates precede full training. See live markers for
actual progress; no new full result is claimed at implementation time.
Both pre-existing untracked user plan documents are preserved, not committed.
Live launch verified at approximately10:24 JST: implementation `a77dda8`,
tmux active. Local broader suite60 passed; queue suite48 passed. Both detach/A1
2-epoch smokes passed training, archive verification, classification, CAM and
artifact diagnostics. Disposable per-image classification outputs were cleaned
with recorded hashes; smoke checkpoints retained. Full detach seed0 has begun;
epoch001 is saved and independently hash-verified. Other three trainings and
all full trajectory evaluations/diagnostics remain queued, not completed.

## Appended request — all-epoch evaluation (2026-09-16)

The user authorized classification and raw CAM evaluation of all180 archived
checkpoints after the four runs finish, followed by epoch curves and per-checkpoint
cleanup of disposable evaluation artifacts. Checkpoints remain retained.
No training/model code is changed for this task; the running queue is untouched.
Live inspection: first three runs finished, all-product seed11 at epoch36/45
(one-based) during preparation. Training completion must be read from live markers.

New CPU-waiting tmux: `mct-voc-epoch-eval-20260916`.
Runner: `experiments.ablations.evaluate_voc_epoch_trajectory`.
Output: `results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11`.
Log: output path plus `.queue.log`. Poll interval60 seconds; no GPU inference
until the training queue's `QUEUE_COMPLETE` exists. Then two small GPU checks,
followed by all180 evaluations and progressive CSV/SVG/PNG curves.
Protocol/command/cleanup scope: `docs/VOC_Epoch_Evaluation_Protocol.md`.
CPU tests12 passed. GPU smoke and full trajectory evaluation are pending,
not yet completed. Do not launch a duplicate watcher or restart training.

## Current request — VOC epoch archives (2026-09-16)

Starting from main `137b156`, the user requested four fresh matched VOC runs:
GWRP seed0 -> all-product seed0 -> GWRP seed11 -> all-product seed11.
Each uses the unchanged 45-epoch recipe, with a new opt-in
`--save-every-epoch` archive (180 full-run snapshots in total).
No beta1 sweep or old cancelled diagnostic queue is included.
Implementation/tests are committed before launching the queue; the manifest
records the exact implementation SHA. The pre-existing untracked
`docs/MCTformerPlus_A2_Beta1_Sweep.md` remains untouched and uncommitted.

Runner: `experiments.ablations.run_voc_epoch_checkpoints`.
Scheduled tmux: `mct-voc-epochs-s0-s11-20260916-r2`.
Output: `results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2`.
Queue log: same path plus `.queue.log`.
Protocol: `docs/VOC_Epoch_Checkpoint_Runs.md`.
Queue first repeats 33 focused tests, then two 2-epoch smoke runs, including
loading epoch002 for classification/CAM, and only then starts full training.
Consult live markers/logs to distinguish planned, active and completed stages.
Do not launch a duplicate queue. Source checkpoints/results remain immutable.
The initial non-r2 queue at code `215afce` passed tests and both smoke runs but
was intentionally terminated early in GWRP seed0 training: a read-only metadata
comparison revealed legacy model_spec files omit newer default pooling flags.
The runner now canonicalizes only those documented defaults before comparison.
Initial outputs/logs are preserved; r2 starts fresh with no checkpoint reuse.
Live verification at approximately16:54 JST: r2 launched with implementation
`dc52762` at16:51;33 tests passed and both smoke runs completed, including CAM
and classification from archived epoch002. GWRP seed0 completed epoch001 and
entered epoch002. The first full-run snapshot exists with model, optimizer,
scaler and CUDA RNG states; its SHA256 was independently verified. Remaining
three full runs are queued, not completed. Model/pretrained/optimizer metadata
match the historical GWRP recipe after canonicalizing known legacy defaults.

## New request — Artifact / affinity plan (2026-09-16 10:17 JST)

User explicitly requested execution of
`docs/MCTformerPlus_Artifact_and_Affinity_Repair_Plan.md` in order.
This authorizes only that new plan; the old Diagnostic Probe queue stays cancelled.
First stage: `analysis/artifact_probe.py`, tests `tests/test_artifact_probe.py`.
Full output: `results/artifact_probe/20260916-voc-s0`; tmux
`mct-artifact-20260916`. Execution details and pending later-stage clarifications:
`docs/Artifact_Affinity_Execution_Notes.md`.

Artifact full run completed successfully at code `b04df6b`, all source hashes
unchanged. GWRP passes all three gates; product and affinity fail overlap gate.
Next queue: `experiments.ablations.run_affinity_repair`, output
`results/affinity_repair/20260916-voc-s0`, tmux `mct-affinity-repair-20260916`.
Queue order: unit tests -> small A1/A2 checks -> full A1 (72 fixed configs) ->
A2/A3 (native and <=10 repair candidates) -> B only if fixed A2 mIoU beats the
unrounded historical all-product reference. If gate fails, no training launches.
The old Diagnostic Probe queue must NOT be restarted. New queue preserves all
source checkpoints/results, uses online CAM evaluation, and records failures.
Stage B, if reached, has minimal / A2-best / conservative, deduplicating identical
configs; each is fresh seed0 45epochs with unchanged native CAM evaluation.
Completion verified at14:38 JST: queue ended successfully at14:32:39, with
`QUEUE_COMPLETE`, no failure markers, no experiment tmux or GPU compute process.
A1 completed72 configs (47 passing,10 selected); A2/A3 completed. Local broad
tests71 passed; the queue's selected suite51 passed.
Stage B completed two distinct45-epoch runs: minimal and conservative; A2-best
aliases minimal, so no duplicate third training was performed. Both have final
checkpoint hashes verified, CAM1464 and classification1449 coverage. Fixed CAM
mIoU is71.782743% /71.781563%, below original all-product72.315445%.
IMPORTANT: the A2 gate used a strict greater-than against the historical native
value and admitted a numerical-scale difference. Minimal minus same-run native
is only0.0000042947 percentage points, with paired95% CI containing0; this is
not meaningful evidence that inference repair improves CAM. Stage B did run,
but the recorded `gate_passed=True` must not be interpreted as scientific success.
The root `ARTIFACT_AFFINITY_REPORT.md` is a short completion summary; detailed
numbers are in `stage_b_comparison.csv` and `a2/comparison.csv`. Existing reports
and decision JSONs were not rewritten during this read-only result audit.
Source hashes for queue/A1/A2 and both new checkpoints were reverified unchanged.

## Previous instruction: experiments stopped

At the user's explicit request on 2026-09-16, stop the current experiment and
all pending experiments, and remove intermediate large outputs. Do not restart
or schedule further experiments without a new user request.

The Diagnostic Probe queue process group `1026827` was terminated with SIGTERM
at approximately 10:09 JST. Its tmux session exited. GPU compute-process listing
was empty afterward; only the user's `codex-wsss-0` tmux session remains.
The monitor had already exited after handing off the queue at 08:00 JST.

Queue: `results/diagnostic_trajectory/20260916-voc-ab-s012`.

- GWRP seed 0: completed 45 epochs; checkpoint and diagnostic results retained.
- C2P all-product seed 0: interrupted during zero-based epoch 19 (20th epoch).
  Existing best checkpoint and prior completed diagnostics retained; NOT a
  completed matched experiment.
- Seeds 1 and 2 for both poolings: cancelled before starting.
- `QUEUE_CANCELLED`, `c2p_s0/RUN_CANCELLED` and monitor `HANDOFF_CANCELLED`
  document cancellation. Historical launch records remain unchanged.

## Completed COCO

`results/c2p_pooling/20260915-coco-all-product-s0-r2` completed at 07:43 JST.
Its `C2P_COCO_REPORT.md` and `comparison.csv` hold the final comparison.
All-product final checkpoint severely degraded: fixed raw CAM mIoU 1.3944%
versus GWRP 42.6674%; this is a completed run, not a positive method result.

## Authorized cleanup

Exact commands:

```bash
kill -TERM -- -1026827
/home/peng/anaconda3/envs/tgca-repro/bin/python -m tools.cleanup_results_20260916 plan
/home/peng/anaconda3/envs/tgca-repro/bin/python -m tools.cleanup_results_20260916 apply
```

Audit: `results/cleanup/20260916-stop-diagnostics/`.
Deleted 9 per-image classification prediction NPZs, 50,097,096 bytes (47.78 MiB).
969 retained files passed before/after SHA256 checks, including checkpoints,
reports, numeric CSV/JSON results, aggregate confusion arrays and logs.
No smoke directories or per-image CAM arrays remained at this cleanup:
older CAM arrays were already removed; the latest COCO used online evaluation.
No payload backup was made; deleted predictions require inference from retained
checkpoints to regenerate. Existing result manifests are historical and were
not rewritten to hide deletions. Compact diagnostic arrays and figures remain.

`docs/design.md` is absent from this checkout; it was not recreated. This
handoff records the current operational state rather than restoring old plans.
