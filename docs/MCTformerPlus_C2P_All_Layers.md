# C2P patch pooling: all Transformer layers

User-requested continuation of the matched C2P experiment. Only change:
pooling uses the actual A_c2p from all 12 blocks rather than the last 3.

`--patch-pooling c2p --c2p-pooling-layers all`

```
a = mean(A_c2p[L1:L12], dimensions=(layers, heads))
w = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
patch_logits = (w * raw_3x3_classifier_logits).sum(-1)
```

Average raw attention FIRST, then normalize over patch keys. No per-layer
conditionalization before averaging. No sorting, detach, activation on the
classifier logits, extra parameters or losses. Retain FP32 accumulation under
AMP. Initialization (including all existing CWP configuration/code), class
branch, CCT and backbone are unchanged. The default `last3` preserves previous
behavior and old checkpoint interpretation; layer choice is serialized in
model_spec and checked on classification/CAM loading.

CAM is NOT changed to all-layer C2P: native last-three C2P refinement, sqrt,
all-layer P2P propagation remain exactly as before. Same-weight CAM parity is
tested for both pooling layer choices, along with full-forward patch gating.

Fresh DeiT-S initialization, ordinary Small, class_token_init=baseline, seed 0,
448, batch 32, 45 epochs, nominal LR 5e-4/min 1e-5, same AdamW/cosine/augmentation/
loss weights as completed C2P-last3. No FinalLN, patch-first or other variants.
VOC train_aug 10582; classification val 1449; native three-scale raw CAM train
1464, fixed threshold .45, same diagnostic grid 0:.01:.59. No CRF/segmentation.

Reuse immutable GWRP and C2P-last3 results from
`results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12` and
`results/c2p_pooling/20260914-voc-s0`. Do not retrain either reference.

Runner: `python -m experiments.ablations.run_c2p_all_layers --output results/c2p_pooling/20260914-voc-all-layers-s0`.
Tests -> 64-image/one-epoch smoke -> smoke CAM/evaluation/classification ->
full matched training -> native CAM/evaluation -> two-head classification ->
three-row comparison.csv and C2P_ALL_LAYERS_REPORT.md.
Existing clean-tracked-worktree and no-overwrite gates are retained.
Save exact commands, environment, checkpoint/source hashes and all stage logs.
