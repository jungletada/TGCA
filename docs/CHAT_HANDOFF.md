# Operational handoff — 2026-09-16

Host: LHR. Repository: `/home/peng/code/TGCA`, branch `main`.
Environment: `/home/peng/anaconda3/envs/tgca-repro/bin/python`.
Implementation at cancellation: `dd4f370` (diagnostic probes: `5b66a3c`).

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
