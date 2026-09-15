# COCO all-product matched experiment

User explicitly requested COCO transfer on 2026-09-15. Run one new ordinary
MCTformer+-Small baseline-init seed0 model with all12-layer C2P product pooling,
NO pooling P2P affinity. Reuse the completed GWRP COCO control read-only:
`results/default_mctformerplus/20260912-voc-coco-s0-r2/COCO`.
This is a pooling experiment, not evidence that TGCA mechanism/generality gates passed.

## Fixed recipe

Same DeiT-S initialization, 45 epochs, input448, effective batch32, seed0,
nominal LR5e-4/min1e-5 and original AdamW/cosine, augmentation, class/CCT/patch
losses and 3x3 head. Training82783 / validation40504 images, 80 class tokens.
Only pooling: `--patch-pooling c2p --c2p-pooling-layers all --c2p-pooling-reduction product`.
Heads averaged within layer; product across12, spatial normalization via
FP32 softmax(sum(log(a_l))) with existing numerical floor; gradients retained.
Do not change class-token initialization, norm, attention, or native CAM formulas.

## Evaluation and disk budget

Native CAM train82783 with image-level labels, scales1,.75,1.25/flip, unchanged
last3-mean C2P, sqrt, all-layer P2P. Fixed threshold .45, same0:.01:.59 grid;
best threshold diagnostic only. No CRF, segmentation or other variants.
Baseline full CAM dump uses249GiB; current free disk76GiB. New full run uses
native in-memory CAM dictionaries to accumulate the same dataset-global
confusion matrices, without persisting full raw CAMs. No prior data deleted.
Same native generator; only sink differs. Smoke independently runs both sinks
with the same checkpoint and requires EXACT all60-threshold confusion equality.
Keep compact threshold curves, aggregate confusion, checkpoint and provenance.
The online sink currently supports one GPU/train CAMs only.

Classification: same FP32 single448 transform, val40504, class-token/patch
macro-class AP and validation losses. Reuse baseline checkpoint, do not retrain
it. Existing baseline lacks this two-head macro measurement; save new frozen
baseline evaluation only under the NEW experiment root. COCO per-class labels
are explicitly named by existing label-vector index, not assumed category IDs.
Single seed; no statistical robustness or semantic-mechanism claims.

## Execution

Necessary tests -> smoke train/checkpoint/native CAM offline+online parity ->
smoke both new and old classification -> full all-product train/CAM -> two-head
classification for new and frozen control -> source hashes/config audit -> report.
63 tests passed before commit; queue repeats tests with log. Match dataset audit,
optimizer and pretrained-load reports; preserve all source checkpoints/results.

Command (tgca-repro, main, clean tracked checkout, unique output):

```bash
/home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.ablations.run_c2p_coco --output results/c2p_pooling/20260915-coco-all-product-s0
```

Output: `comparison.csv`, `C2P_COCO_REPORT.md`, command/config/environment/test
logs, checkpoint SHA256, full-run `COCO/raw_cam/`, classification directories.
QUEUE_COMPLETE requires all stages; QUEUE_FAILED means inspect and do not duplicate.
