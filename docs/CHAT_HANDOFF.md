# TGCA Operational Handoff

## 2026-09-13 patch-first order ablation

User requested exchanging concatenation order only, keeping everything else
unchanged. Plan: `docs/MCTformerPlus_Patch_First_Ablation.md`.
Entry point: `python -m experiments.ablations.run_patch_first --output results/patch_first/20260913-voc-s0`.
Active tmux: `mct-patch-first-voc-20260913`.
Verified 2026-09-13 20:22 JST: 32 tests passed, smoke training/CAM/evaluation
passed, full VOC epoch 0 is running with finite losses. Code SHA `638394d`.
Queue log: `results/patch_first/20260913-voc-s0.queue.log`.
Training log: `results/patch_first/20260913-voc-s0/VOC12/train.log`.
One VOC seed-0 run is active (COCO is not queued); exact completed default VOC
baseline settings, 45 epochs, 448, batch 32, DeiT-S, nominal LR 5e-4/min 1e-5.
Only new argument: `--patch-first`. All blocks see [patch,class]; CCT takes raw
tail class tokens. Semantic readout and CAM attention indices are restored
outside the blocks; original head/loss/CAM formulas are unchanged. Positions
follow tokens, so this is a permutation-equivalent control in exact arithmetic.
Tests precede smoke; smoke precedes full training and original raw CAM eval.
Queue never overwrites existing files. Do not relaunch if tmux/output exists.
Output `manifest.json` records actual code SHA; `commands.sh` and stage logs
reside in each run. Results will be `comparison.csv` and `PATCH_FIRST_REPORT.md`.
The unrelated existing results and checkpoints remain untouched.

## 2026-09-13 completion verified

The fresh default VOC / COCO queue below completed at 2026-09-13 13:11:42
JST (`QUEUE_COMPLETE`). Both 45-epoch runs, native CAM generation and raw CAM
evaluation have completed. No experiment tmux or GPU process remains active.
Read-only verification at 20:14 JST confirmed checkpoint SHA256 matches for
both datasets and exact CAM coverage: VOC 1,464 / COCO 82,783, no missing or
extra image files. Tests: 29 passed. Execution SHA remains `eba5069`.

Raw CAM train-split mIoU: VOC fixed 0.45 = 70.063%, best 0.48 = 70.340%;
COCO fixed 0.45 = 42.667%, best 0.44 = 42.688%. These are native three-scale
CAMs with image-level label gating, not validation-split or segmentation-model
results. No CRF or downstream segmentation was run. Compact results are in
`results/default_mctformerplus/20260912-voc-coco-s0-r2/RAW_CAM_REPORT.md` and
`cam_summary.csv`; each dataset has `raw_cam/metrics.json` and the complete
`raw_cam/threshold_curve.csv`. Do not restart the completed queue.

## 2026-09-12 fresh default VOC / COCO task

The user explicitly requested fresh default MCTformer+ training on both VOC
and COCO, stopping at raw CAM evaluation. This is NOT a restart of the stopped
decoupled queue below. Live main started at `f120611`; the GPU driver mismatch
seen earlier is resolved (580.178.04, CUDA available in `tgca-repro`).

New queue entry point: `python -m experiments.baselines.run_default_voc_coco`.
Current tmux: `mct-default-voc-coco-20260912-r2`.
Output: `results/default_mctformerplus/20260912-voc-coco-s0-r2`.
Verified live at 2026-09-12 17:19 JST: both smoke training/CAM/evaluation
pipelines passed; full VOC epoch 0 is active with finite losses and normal GPU
utilization. COCO full training is pending behind VOC training + raw CAM eval.
Execution code SHA: `eba5069` (full SHA in output `manifest.json`).
Queue log: `results/default_mctformerplus/20260912-voc-coco-s0-r2.queue.log`.
Training log: `<output>/VOC12/train.log`; later `<output>/COCO/train.log`.
The first attempt (same path without `-r2`, code `f759165`) passed both
smoke trainings and VOC CAM evaluation, then stopped at COCO CAM checkpoint
resolution: an old validator hardcoded 20 class tokens. Its logs/checkpoints
remain untouched. The follow-up fix passes the actual dataset class count
into that validator; it does not change model computation or training.
Launch only once; inspect this directory and tmux before doing anything else.
Both dataset smokes precede full VOC, then full COCO. Training uses the existing
MCTformer+ baseline recipe (not legacy `mcta` COCO script): Small, vanilla joint
attention, no experimental flags, 448, seed 0, batch 32, 45 epochs, nominal LR
5e-4, min LR 1e-5, AdamW/cosine and remaining original settings. Native LR batch
scaling is unchanged. Initialization is the DeiT-S file recorded below.
Training lists: VOC train_aug 10,582 / val 1,449; COCO train2014 82,783 /
val2014 40,504. All listed images/labels and CAM evaluation masks passed a
read-only existence audit. Raw CAM evaluation: final checkpoint, train split
(VOC 1,464; COCO 82,783), native scales 1,.75,1.25 and image-level label gating.
Threshold grid 0.00:0.01:0.59, fixed 0.45; global confusion accumulation includes
empty CAM predictions. No CRF/refinement/segmentation stages.

Tests: `python -m pytest -q tests/test_raw_cam_streaming.py tests/test_width_scaling_aggregation.py tests/test_mctformerplus_variants.py`
passed after the class-count fix (29 tests).
Queue repeats and saves these tests. Exact commands, environment
manifests, dataset hashes, code SHA, optimizer specs, logs and checkpoint hashes
are written under the new output. Final compact files: `cam_summary.csv`,
`RAW_CAM_REPORT.md`, and each dataset's `raw_cam/metrics.json` and
`raw_cam/threshold_curve.csv`. `QUEUE_COMPLETE` means both full runs finished;
`QUEUE_FAILED` means inspect logs. These full results do not exist yet at setup.

The historical state below is retained for provenance and is superseded by this
section for the current task.

Updated: 2026-09-09 21:13 JST

This file was absent at the start of the current task and was created to restore
the operational handoff required by `AGENTS.md`. It records the live server
state for the current experiment; older completed studies remain immutable
under `results/` and in Git history.

## Repository and environment

- Repository: `/home/peng/code/TGCA`
- Branch: `main`
- Experiment code commit: `b1b4830e3d4eaaf5356425de906f8d0a1664b708`
- Conda environment: `tgca-repro`
- Device: NVIDIA RTX A6000, GPU 0
- Dataset: `/home/peng/code/TGCA/data/VOCdevkit/VOC2012`
- Official DeiT-S initialization:
  `/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth`
- DeiT-S SHA256:
  `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f`

## Stopped experiment queue

- Former tmux session: `mct-decoupled-q-20260909` (stopped and absent)
- Run ID: `20260909-mctformerplus-decoupled-s0-b1b4830`
- Result root:
  `/home/peng/code/TGCA/results/decoupled_bidirectional/20260909-mctformerplus-decoupled-s0-b1b4830`
- Runner: `experiments/ablations/run_mctformerplus_decoupled_queue.sh`
- Pipeline log: `<result root>/pipeline.log`
- Exact commands: `<result root>/exact_commands.sh`

The 100-update, batch-32, 448 smoke passed with finite class/CCT/patch
losses, strict checkpoint audit, and four completed single-scale CAMs. At the
user's request, the complete queue was stopped at 2026-09-09 21:13 JST. No
queue, training, evaluation, or CAM process remains active on the GPU. Do not
restart or duplicate this queue without a new user request.

The serialized queue order is:

1. `full`
2. `no_p2c`
3. `no_c2c`
4. `no_p2c_no_c2c`
5. `c2p_update_off`
6. `p2p_update_off`
7. `p2c_middle`
8. `p2c_early`

The first four items completed 45-epoch seed-0 training, strict checkpoint
audit, 1449-image
single-scale classification evaluation with image bootstrap, native multiscale
CAM generation on the 1464-image train split, and the fixed 0.45 / common
threshold-grid evaluation. The existing original joint MCTformer+ result is
referenced read-only from
`results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8`.

Completed variants, each with `VARIANT_COMPLETE`, are `full`, `no_p2c`,
`no_c2c`, and `no_p2c_no_c2c`. `c2p_update_off` was interrupted after epoch 1;
its partial best checkpoint and logs are preserved but it has no final
checkpoint, audit, classification evaluation, CAM evaluation, or completion
marker and must not be reported as a completed experiment. `p2p_update_off`,
`p2c_middle`, and `p2c_early` were not started. The run root correctly has no
`EXPERIMENT_COMPLETE` or `PIPELINE_COMPLETE` marker.

## Validation completed before launch

- Relevant deterministic tests: 84 passed.
- The runner's committed test set: 69 passed.
- Exact 448, batch-32 AMP forward/backward: finite.
- Exact pretraining adaptation and strict load: passed.
- Preflight peak allocated CUDA memory: 15.13 GiB.
- Model parameter count: 22,050,836, identical to original MCTformer+-Small.

## Safe status checks

```bash
tail -n 120 results/decoupled_bidirectional/20260909-mctformerplus-decoupled-s0-b1b4830/pipeline.log
nvidia-smi
tmux ls
```

The overall queue is intentionally incomplete. Inspect the immutable
per-variant completion markers before using results. Do not delete completed
checkpoints, CAMs, or the preserved partial `c2p_update_off` files.
