# MCTformer+ Tiny / Small / Base Width Study

This experiment is a matched backbone-width/capacity study, not a scaling-law
claim. Tiny, Small, and Base all use 12 transformer blocks, patch size 16, head
dimension 64, the native MCTformer+ loss and CAM equations, and the same VOC
recipe. Only embedding width, head count, and resulting capacity change.

## Registered variants

| Variant | timm model name | Width | Heads | Official non-distilled DeiT initialization |
|---|---|---:|---:|---|
| Tiny | `mctformerplus_tiny` | 192 | 3 | `deit_tiny_patch16_224-a1311bcf.pth` |
| Small | `mctformerplus` | 384 | 6 | `deit_small_patch16_224-cd65a155.pth` |
| Base | `mctformerplus_base` | 768 | 12 | `deit_base_patch16_224-b5f2ef4d.pth` |

The original `mctformerplus` name remains the canonical Small name. A legacy
checkpoint without `model_spec` is accepted only with that exact CLI name;
Tiny or Base is never guessed from an unspecified checkpoint.

## Fixed protocol

- PASCAL VOC 2012, seed 0, input size 448, 45 epochs.
- AdamW, nominal learning rate `5e-4`, effective-batch-scaled optimizer learning
  rate `3.125e-5`, weight decay `0.05`, five warm-up epochs, cosine schedule,
  minimum learning rate `1e-5`, drop `0`, drop-path `0.1`.
- Effective batch size 32. Gradient accumulation divides each micro-loss and
  steps, unscales, clips, updates the scaler, and clears gradients only at an
  accumulation boundary. Every epoch uses exactly `floor(N/32) * 32` samples.
- Vanilla global softmax, BCSS E0, PSL baseline, and CTI-BGT disabled.
- Native CAM: last-three-layer class-to-patch average and all-12-layer summed
  patch-to-patch propagation; scales `1.0,0.75,1.25`.
- Raw CAM thresholds `0.00..0.59` are evaluated exhaustively. Results report
  fixed `0.45`, one threshold calibrated on canonical Small train CAMs, and a
  clearly labelled per-model oracle sensitivity diagnostic.
- Classification uses dataset-level one-vs-rest class AP as the primary macro
  metric. Image-wise AP is retained only as a legacy diagnostic.
- Confidence intervals and paired differences resample complete images. Patches
  and image-class pairs from the same image are never treated as independent.

## Immutable Small reuse

The historical seed-0 Small source remains in its original result directory.
No checkpoint or CAM is copied, rewritten, or retrained. New evaluation products
are written below `results/mctformerplus_width_scaling/voc/small_reanalysis/`.
The reference directory contains a path/hash pointer, the read-only audit, and
the deterministic pre-change regression capture.

## Reproducible entry points

Capacity probe:

```bash
python tools/probe_mctformerplus_capacity.py \
  --variant base \
  --official-pretrained /home/peng/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth \
  --output results/mctformerplus_width_scaling/voc/references/base_capacity_probe.json
```

Small read-only reanalysis:

```bash
TGCA_SMALL_RUN_DIR=/absolute/path/to/canonical/small \
bash experiments/scaling/run_mctformerplus_width_voc.sh \
  --variant small --seed 0 --gpu 0 --stage reanalysis
```

Tiny and Base full runs:

```bash
bash experiments/scaling/run_mctformerplus_width_voc.sh \
  --variant tiny --seed 0 --gpu 0 --stage all

bash experiments/scaling/run_mctformerplus_width_voc.sh \
  --variant base --seed 0 --gpu 0 --stage all \
  --micro-batch M --accum-iter A
```

Aggregation requires four explicit source roots and refuses any mismatch in
dataset hashes, seed, effective batch, epochs, model method configuration, CAM
policy, threshold grid, or checkpoint policy:

```bash
python tools/aggregate_mctformerplus_width_scaling.py \
  --tiny-run /absolute/path/to/tiny \
  --small-run /absolute/path/to/immutable/small \
  --small-reanalysis /absolute/path/to/small/reanalysis \
  --base-run /absolute/path/to/base \
  --output-dir /absolute/path/to/aggregate
```

Scientific full runs require `main`, the `tgca-repro` environment, and a clean
tracked worktree. Every result directory is unique and every producer refuses
to overwrite its output.
