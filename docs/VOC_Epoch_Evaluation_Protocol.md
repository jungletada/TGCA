# All-epoch VOC classification and raw CAM evaluation

User request: evaluate all 45 epoch checkpoints for GWRP/all-product, seeds0/11,
after all four training runs complete. Keep checkpoints, compact numeric results
and curves; remove disposable newly generated per-image evaluation outputs.

Source (read-only):
`results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2`.

Evaluation output:
`results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11`.

Tmux: `mct-voc-epoch-eval-20260916`.
Its log is the output path plus `.queue.log`.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/peng/anaconda3/envs/tgca-repro/bin/python -u \
  -m experiments.ablations.evaluate_voc_epoch_trajectory \
  --source results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2 \
  --output results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11 \
  --wait
```

The CPU-only watcher checks every60 seconds for training `QUEUE_COMPLETE`,
and rejects training failure/cancellation markers. It does not modify, restart,
or compete for GPU with the training queue. After completion, it checks the
45-per-run inventory and runs two tiny inference smoke checks before the180
full evaluations. It evaluates epoch1 across all four runs, then epoch2, etc.
Do not start a duplicate evaluation queue. A file lock also rejects duplicate
use of the same evaluation output. `--resume` with the same code SHA continues
an interrupted evaluation queue, retaining previous completed and failed attempts.

## Matched protocol

- Classification:1449 VOC validation images, FP32 single-scale448 deterministic
  evaluation; class-token and patch-head macro-class AP, validation losses and
  per-class AP. This is not the legacy mean-image AP printed during training.
- Raw CAM:1464 VOC training images, exact existing native multiscale
  1/.75/1.25 plus flip, image-level label gating and refinement. Online
  dataset-global confusion accumulation, no per-image CAM arrays on disk.
- Main CAM curve: mIoU at fixed threshold .45; also foreground precision/recall.
  Best-threshold mIoU and selected threshold use the existing0:.01:.59 diagnostic
  grid. They are not independent validation-based model/threshold selections.
- No retraining, new pooling, additional post-processing or segmentation.
- Each checkpoint SHA256 is checked before and after inference against its
  training archive index; expected coverage and checkpoint epoch are checked.

## Outputs and storage

`epoch_metrics.csv`:180 rows after completion, including hashes and metric values.
`epoch_curves.svg` and `.png`: eight panels, four separate method/seed curves.
X-axis is epoch, not measured wall-clock time. No confidence intervals inferred
from two seeds. `EPOCH_TRAJECTORY_REPORT.md` labels partial/completed coverage.

Each `<run>/epoch_NNN/attempt_NNN/` retains `result.json`, classification JSON,
per-class AP CSV, CAM metrics JSON, threshold curve CSV, small aggregate confusion
counts, exact commands/logs, completion markers and `cleanup.json`.

Only after validated result JSON has been saved, remove that attempt's named
`classification_predictions.npz`, `classification_per_image.csv`, and any
`bootstrap_samples.npz`. No recursive deletion or checkpoint deletion. These
disposable outputs are reproducible by inference from retained checkpoints;
there is no separate backup. Old training results, all180 checkpoints, best/final
files and unrelated user files remain untouched. This avoids accumulating large
CAMs in the first place. Smoke checks generate no new checkpoints.

Prelaunch CPU tests:12 passed, logs at
`results/epoch_checkpoint_validation/20260916-trajectory-code/tests.log`.
GPU smoke checks are queued after training, not claimed complete at launch.
Estimated full evaluation duration based on recent logs: approximately7–8 hours.
