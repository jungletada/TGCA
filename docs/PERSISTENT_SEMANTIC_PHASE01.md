# Persistent Semantic Latent Phase 0/1 Protocol

This protocol implements only Phase 0 and Phase 1 of
`docs/Persistent_Semantic_Latent_Codex_Plan.md`. It does not implement or train
the persistent semantic latent architecture.

## Frozen baseline

- Host: MCTformer+ E0, vanilla global softmax, 20 class tokens followed by patch tokens.
- Checkpoint: the matched seed-0 E0 final checkpoint with SHA-256
  `41ac9ce47f6a22875cba32edb92c31c150e804ae5ae19824c2585e4e3cda7a2a`.
- Trusted reproduction: raw CAM mIoU `70.06305830908285` at the prespecified
  background threshold `0.45` on VOC `train_id` (1464 images). The Phase 0/1
  diagnostic validates and references this immutable result rather than
  retraining or regenerating its multi-scale CAMs.
- Diagnostic input geometry: aspect-preserving resize of the short side to
  `256/224 * resolution`, followed by a square center crop. GT uses the same
  geometry and nearest-neighbor reduction to the patch grid.

## Phase 0 instrumentation

`tools/analyze_patch_to_class.py` registers forward hooks on each
`TokenGroupNormalizer`. The hook input is the exact pre-normalization
`QK^T/sqrt(d)` tensor and the hook output is the attention used by the model.
It records raw-logit, evidence, entropy, row-sum, and post-softmax group-mass
statistics for all 12 layers, 6 heads, four directions, and resolutions
`224,320,448,512`.

Complete raw and post-softmax matrices are stored at resolution 224 for two
fixed images. Remaining images use online aggregation. This retains direct
auditable tensors without producing terabytes of redundant quadratic dumps.

## Phase 1 definitions

For every head, class-to-patch logits are normalized over patch keys and
patch-to-class logits over all 20 class keys. Heads are averaged only after
their respective softmax. A second patch-to-class result masks absent classes
using image-level labels before softmax; it is reported separately.

- `cp`: per-class spatial min-max normalization, then fixed threshold `0.5`.
- `pc_all`: strict 20-class attribution. Semantic accuracy and mIoU are
  evaluated only at GT foreground patches because E0 has no background class.
- `pc_present`: image-label-masked attribution, also restricted to GT
  foreground. Single-class images necessarily assign every patch to that class;
  foreground precision/recall is therefore reported and must not be omitted.
- `mutual`: `sqrt(P(p|c) * P(c|p))`, diagnostic only, followed by the same
  spatial normalization and threshold as `cp`.
- Regions A/B/C/D: both relations use strict `> 0.5`. Patch-to-class uses the
  unmasked 20-class probability for this test. Region C is compared with the
  target-class purity of all class-to-patch-low patches in the same images.

Threshold `0.5` remains the primary disagreement definition. A single shared
patch-to-class sensitivity grid `0.05,0.10,0.25,0.50` is recorded for every
layer to distinguish an uninformative relation from a calibrated but diffuse
one. It is diagnostic only and must not be used to select a favorable threshold
for each method or layer.

All layer indices in machine-readable files are zero based. GT is used only
for analysis. No metric in this phase is a full semantic segmentation result
for patch-to-class because no background/dustbin latent exists.
