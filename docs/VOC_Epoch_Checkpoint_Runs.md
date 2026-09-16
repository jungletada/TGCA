# VOC GWRP / all-product, seeds 0 and 11

Four fresh matched runs, in order: `gwrp_s0`, `all_product_s0`, `gwrp_s11`,
`all_product_s11`. No affinity repair or diagnostic intervention is enabled.
All-product is the existing C2P all-layer product pooling implementation.

Recipe: DeiT-S pretrained initialization, VOC train_aug 10582, validation 1449,
448 input, batch 32, 45 epochs, AdamW, nominal LR 5e-4 (scaled 3.125e-5),
minimum LR 1e-5, weight decay .05, cosine schedule and 5 warmup epochs.
Original initialization, augmentation, CCT and class branch remain unchanged.
The only differences between paired methods are patch-pooling options.

Launch from clean tracked `main` in `tgca-repro`, under tmux:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/peng/anaconda3/envs/tgca-repro/bin/python -u \
  -m experiments.ablations.run_voc_epoch_checkpoints \
  --output results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11 \
  --execute
```

The runner rejects an existing output directory. Without `--execute` it prints
the matrix only. Tests and two smoke runs must succeed before full training.
Exact subprocess commands are saved in each directory's `commands.sh`.

## Checkpoints

Each run retains best/final checkpoints and adds:

```text
epoch_checkpoints/mctformerplus_epoch_001.pth
...
epoch_checkpoints/mctformerplus_epoch_045.pth
epoch_checkpoints/index.jsonl
```

File numbering is one-based; the existing payload `epoch` remains zero-based.
Snapshots include model, optimizer, scheduler, AMP scaler, main-process RNG
states, args, metrics and code/command provenance. Files are atomically published
without replacement and indexed with SHA256. Saving does not consume training
RNG. Final snapshot/model equality and all hashes are checked after each run.
This does not add a resume implementation to the legacy trainer and does not
claim bitwise replay of dataloader worker RNG. Estimated new storage: 45–50 GB.

## Evaluation and evidence

Final checkpoints: FP32 class/patch macro-class AP and validation losses on
1449 images; native three-scale plus flip raw CAM on VOC train 1464, fixed
threshold .45 and the unchanged diagnostic threshold grid. No segmentation or
extra refinement, and no large per-image CAM dumps. Best-threshold results are
diagnostics on this split, not independently selected performance estimates.

`comparison.csv`, `manifest.json`, `VOC_EPOCH_CHECKPOINT_REPORT.md`, per-run
config/environment/commands/logs and SHA256 indices provide the final evidence.
`QUEUE_COMPLETE` means all four finished; `QUEUE_FAILED` means inspect the log.
At implementation commit time these are planned runs, not completed results.

Prelaunch validation: 32 tests passed (13 dependency warnings), saved at
`results/epoch_checkpoint_validation/20260916-code-v1/tests.log`.
