# Persistent Semantic Latent Phase 2 Protocol

This protocol covers only the first seed-0 architecture screen from Phase 2 of
`docs/Persistent_Semantic_Latent_Codex_Plan.md`. It does not authorize semantic
width reduction, dynamic initialization, multiple latents per class, OT,
hierarchical backbones, or extra background losses.

## Fixed baseline and host

- Host: plain DeiT-S/16 MCTformer+, PASCAL VOC 2012.
- Baseline: completed E0 seed-0 run
  `20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3`.
- Training: `train_aug_id`, 45 epochs, seed 0, input 448, batch 32, original
  classification losses and optimizer schedule.
- CAM evaluation: `train_id`, scales `1.0,0.75,1.25`, no CRF, fixed background
  threshold `0.45` selected by the baseline.
- Attention normalization: vanilla for every run.

The baseline is reused rather than retrained. Its checkpoint, training log,
CAMs, and fixed-threshold metrics are immutable inputs to the comparison.

## Minimal architecture

For every persistent-semantic variant:

- the patch stream contains only patch tokens and uses all 12 original ViT
  blocks with their pretrained patch projection, positional embedding,
  self-attention, MLP, and residual paths;
- 20 foreground semantic latents and one static background semantic latent
  remain outside patch self-attention;
- `D_c=D_p=D_r=384`;
- the first screen interacts only after zero-based block `11` (paper layer
  12), as prespecified from the strict Phase 1 confirmation;
- one shared class-patch relation supplies spatial-softmax Read attention and
  class-softmax Write attention;
- semantic Read executes before Write, and Write values use the updated
  image-conditioned semantic latents;
- every Write residual has a learned scalar gate initialized exactly to zero;
- relation Q/K/V and output projections are initialized from the corresponding
  pretrained backbone attention block after DeiT weights are loaded;
- semantic and patch inputs receive parameter-free per-token LayerNorm before
  the copied relation Q/K/V projections, preserving the pretrained pre-norm
  scale without adding a learned normalization specialization;
- no background loss, ownership loss, calibration schedule, or image-label
  masking is introduced.

All three variants instantiate the same relation projections and background
latent so their parameter counts match. Disabled paths remain unused rather
than being removed.

## Prespecified variants

| Variant | Read | Write |
|---|---:|---:|
| `read_only` | yes | no |
| `write_only` | no | yes |
| `read_write` | yes | yes |

`write_only` intentionally retains static semantic latents. No alternate
relation-pooled classifier is added to rescue it, because that would introduce
a second architectural difference.

## CAM and diagnostics

The persistent-semantic CAM replaces the unavailable joint-attention
class-to-patch block with the foreground Read distribution from the layer-12
interaction. It otherwise preserves the baseline CAM computation:

1. multiply Read attention by the ReLU patch classifier map;
2. take the square root;
3. refine with the sum of all 12 patch-only self-attention matrices;
4. apply the unchanged multi-scale, flip, per-class min-max, and fixed-threshold
   evaluation pipeline.

A fixed first-200-image `val_id` diagnostic reports relation entropy, semantic
foreground/background attribution, background Write mass, foreground and
background token gaps, learned Write gates, parameters, and classification mAP.
GT is used only in this diagnostic.

## Seed-0 decision rule

This is an early screen, not final statistical evidence.

- **Strong go:** `read_write` retains classification within 1 point, improves
  raw CAM mIoU over E0 by at least 0.5 point, and has no mechanical failure.
- **Conditional go:** `read_write` remains within 1 point of E0 in raw CAM and
  classification, and exceeds `read_only` CAM by at least 0.5 point.
- **No-go for expansion:** neither condition holds. Do not run ordering, depth,
  independent-relation, width, dynamic-initialization, or multi-seed studies.

Regardless of the mechanical decision, inspect background attribution,
per-class behavior, learned gate magnitudes, and training stability before
accepting a scientific conclusion.
