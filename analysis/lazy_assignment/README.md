# Experiment 1 — Class-specific Patch Score

This package implements a read-only representation diagnostic for native
MCTformer and MCTformer+.  For every post-block token sequence it computes

```text
score[b, class, patch] = cosine(class_token[b, class], patch_token[b, patch])
```

in float32.  The model forward path, attention, value path, logits, CAM code,
and checkpoint are not modified.  Output layer numbers are one-based
(`layer_01` is block 0 output); the representation is
`post_block_pre_final_norm`.

## Scientific boundary

This experiment only establishes that layer-wise class/patch semantic-alignment
maps can be extracted reproducibly.  It does not load segmentation masks and
does not measure foreground/background distributions, C-PiM, BG-Tail,
class-to-patch attention, CAM quality, or lazy semantic assignment.  Those are
separate experiments.

The score definition follows the official LAST-ViT representation-level Patch
Score at commit `cdeb884af65e7774f2da80f666d95cf09a76b717`, extended from one
CLS/pooled token to all class tokens.  LAST-ViT's frequency selector and
selective aggregation are intentionally not included.

## Checkpoints selected for Experiment 1

MCTformer+:

```text
results/mctformerplus/voc/
  20260826-mctformerplus-voc-vanilla-s0-22427d6/
  mctformerplus_final.pth

SHA-256:
0a0c304250aa448bdb2c6ab00a8cd3f7684fb3c2ccf237b5353176f2ad545660
```

This is the final checkpoint actually used by the run's fixed-threshold CAM
evaluation.  The analysis loader uses `strict=True` and records any explicit
removal of a uniform `module.` prefix.

MCTformer V2:

```text
results/mctformerv2/voc/
  20260901-mctformerv2-voc-mctplus-default-s0-e6389f2/
  mctformerv2_final.pth

SHA-256:
fafe0459f528233dc3ea86cecae91ef9ad3d2ebd5bd601bea0826889e4419ebc
```

This is likewise the final checkpoint recorded by that run's fixed-threshold
CAM evaluation manifest.

## Stage status on LHR (2026-09-02)

The implementation and both hosts' mechanical smoke and full VOC validation
runs are complete.  The result-critical implementation commit is
`fec86b719a62c55c0aaf15d9b45bb2d0f74d3e8e`.

MCTformer+:

```text
run ID: 20260902-mctformerplus-exp1-voc-val-full-fec86b7
VOC val images: 1449 / 1449
positive-class maps: 2147
saved score files / manifest rows: 1449 / 1449
visualization files: 24
checkpoint strict load: passed, no missing or unexpected keys
hook effect on forward_features: max absolute difference 0
layer-12 hook/final-token difference: 0 for class and patch tokens
saved raw-cosine range: [-0.9929999709, 0.9888190031]
NaN / Inf: 0 / 0
independent NPZ reload: passed
queue and tmux exit status: 0 / 0
```

Full result:

```text
results/lazy_assignment/experiment1_class_patch_score/mctformer_plus/
  20260902-mctformerplus-exp1-voc-val-full-fec86b7/
```

MCTformer V2:

```text
run ID: 20260902-mctformerv2-exp1-voc-val-full-6aca9bc
VOC val images: 1449 / 1449
positive-class maps: 2147
saved score files / manifest rows: 1449 / 1449
visualization files: 24
checkpoint strict load: passed, no missing or unexpected keys
hook effect on forward_features: max absolute difference 0
layer-12 hook/final-token difference: 0 for class and patch tokens
saved raw-cosine range: [-0.7154812217, 0.9059177637]
NaN / Inf: 0 / 0
independent NPZ reload: passed
queue and tmux exit status: 0 / 0
```

Full result:

```text
results/lazy_assignment/experiment1_class_patch_score/mctformer/
  20260902-mctformerv2-exp1-voc-val-full-6aca9bc/
```

The earlier MCTformer+ `micro1`/`smoke50` and MCTformer V2 `smoke50`
directories were removed from the repository result tree after their
corresponding full results passed independent audits, as requested.  They were
moved to the user's system Trash and were not used as scientific evidence.

## Mechanical smoke

Activate the required environment first:

```bash
source /home/peng/anaconda3/etc/profile.d/conda.sh
conda activate tgca-repro
cd /home/peng/code/TGCA
```

An uncommitted implementation may only be exercised as a finite smoke, with the
exception explicitly recorded:

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerplus \
  --checkpoint results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-s0-22427d6/mctformerplus_final.pth \
  --expected-checkpoint-sha256 0a0c304250aa448bdb2c6ab00a8cd3f7684fb3c2ccf237b5353176f2ad545660 \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 4 \
  --num-workers 4 \
  --limit 50 \
  --save-visualizations \
  --allow-uncommitted-source \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer_plus/RUN_ID
```

For MCTformer V2, use the same protocol with its pinned checkpoint:

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerv2 \
  --checkpoint results/mctformerv2/voc/20260901-mctformerv2-voc-mctplus-default-s0-e6389f2/mctformerv2_final.pth \
  --expected-checkpoint-sha256 fafe0459f528233dc3ea86cecae91ef9ad3d2ebd5bd601bea0826889e4419ebc \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 4 \
  --num-workers 4 \
  --limit 50 \
  --save-visualizations \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer/RUN_ID
```

## Full runs

A full run refuses untracked runtime source or any tracked Git diff.  After the
implementation has been reviewed and checkpointed in Git, omit both `--limit`
and `--allow-uncommitted-source`:

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerplus \
  --checkpoint results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-s0-22427d6/mctformerplus_final.pth \
  --expected-checkpoint-sha256 0a0c304250aa448bdb2c6ab00a8cd3f7684fb3c2ccf237b5353176f2ad545660 \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 8 \
  --num-workers 8 \
  --save-visualizations \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer_plus/RUN_ID
```

MCTformer V2:

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerv2 \
  --checkpoint results/mctformerv2/voc/20260901-mctformerv2-voc-mctplus-default-s0-e6389f2/mctformerv2_final.pth \
  --expected-checkpoint-sha256 fafe0459f528233dc3ea86cecae91ef9ad3d2ebd5bd601bea0826889e4419ebc \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 8 \
  --num-workers 8 \
  --save-visualizations \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer/RUN_ID
```

Every output directory is immutable and must not already exist.  A valid run
contains `completion.json`; failures retain `failure.json` and all partial
artifacts.

## Output contract

```text
metadata.json
command.txt
analysis.log
pip_freeze.txt
conda_explicit.txt
manifest.jsonl
summary_by_layer.csv
scores/<image_id>.npz
visualizations/*_raw_cosine.png
visualizations/*_minmax.png
completion.json
```

By default, each NPZ stores `[12, number_of_positive_classes, number_of_patches]`
float32 raw cosine scores.  `positive_class_ids` use VOC indices `0..19`.
`grid_h` and `grid_w` are inferred separately; a square grid is never assumed.
At completion, every NPZ is independently reopened and checked for shape,
dtype, finite values, cosine range, image ID, class IDs, and grid consistency.
