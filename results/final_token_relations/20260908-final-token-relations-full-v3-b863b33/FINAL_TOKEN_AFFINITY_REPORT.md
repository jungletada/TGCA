# Final-Token Affinity Report

## Frozen contract

- Native MCTformer+-Small checkpoint SHA256 `aced3d3bd69c57782c8e85f1d10abd7ff7ab02504df92c2f2f3898defcf7a65a`; raw post-Block-12 class and patch tokens only.
- No final LayerNorm, L2 normalization in S_last, projection, Q/K projection, attention-map analysis, CAM modification, selector, loss, or training was used.
- `S_last = c^T p / sqrt(D)` is final-token compatibility, not Transformer attention. GT is used only after relation construction; intervals resample whole images.

## E1 primary semantic diagnostics (micro)

| Relation | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |
|---|---:|---:|---:|---:|---:|
| s_dot | 0.561988 | 0.551990 | 1.702038 | 0.790343 | 0.405217 |
| s_last | 0.561988 | 0.551990 | 1.702038 | 0.790343 | 0.405217 |
| s_cosine | 0.541829 | 0.552372 | 1.433335 | 0.901807 | 0.360969 |
| patch_norm | 0.750786 | 0.503195 | 1.893349 | 0.888093 | 0.278528 |
| r_native_classifier | 0.911374 | 0.891195 | 4.804042 | 0.424890 | 0.826269 |

## Raw-product, direction, norm, and classifier-reference comparison (micro)

| `S_last` comparator | Map Spearman | Top-10% Jaccard | Target-vs-BG AUROC | Target-vs-other-FG AUROC |
|---|---:|---:|---:|---:|
| s_dot | 1.000000 | 1.000000 | 0.561988 | 0.551990 |
| s_cosine | 0.989588 | 0.768451 | 0.541829 | 0.552372 |
| patch_norm | -0.114037 | 0.195923 | 0.750786 | 0.503195 |
| r_native_classifier | 0.103277 | 0.077307 | 0.911374 | 0.891195 |

## Present / absent control for `S_last` (micro)

| Class status | Max score | Top-10% mean | Foreground soft mass | Background soft mass |
|---|---:|---:|---:|---:|
| positive | 2018.486221 | 1146.552648 | 0.553313 | 0.382630 |
| absent | 10878.683173 | 4070.369418 | 0.383321 | 0.577211 |

## Final positive-class map overlap (micro)

| Relation | Map Spearman | Top-10% Jaccard | Shared BG fraction | Dominant-object capture |
|---|---:|---:|---:|---:|
| s_last | 0.343397 | 0.549721 | 0.449963 | 0.405996 |
| s_pos | 0.522327 | 0.612540 | 0.454147 | 0.402359 |

## Required E1 answers

1. `S_last` target-vs-BG AUROC is 0.561988 [0.551303, 0.572572]. This quantifies a frozen representation-level target/background ordering only.
2. `S_last` target-vs-other-FG AUROC is 0.551990 [0.541591, 0.562881]; it is materially weaker than its target-vs-BG discrimination when the estimate is near 0.5.
3. Positive-class `S_last` maps have top-10% Jaccard 0.549721 [0.520627, 0.581058]. This is direct evidence about final-map overlap, not an attention or causal mechanism claim.
4. The direction-only `S_cosine` target-vs-other-FG AUROC is 0.552372 [0.542018, 0.563398], versus patch-norm 0.503195 [0.499599, 0.506926]. The table above separates directional and norm-associated information without attributing cause.
5. The native classifier reference has target-vs-other-FG AUROC 0.891195 [0.880671, 0.901029]; it is a frozen reference rather than a signal fused with `S_last`.
6. Full `S_last` passed the shared-orthogonal-basis regression: max error `6.185e-11` < `1e-5`; transformed mean-readout max error is `7.105e-15`.

## Basis regression

- Shared permutation/signed-permutation/five-Haar maximum `S_last` error: `6.185e-11` (required <1e-5).
- Equivalent transformed mean-readout maximum logit error: `7.105e-15`.
- CSV tables contain micro, macro-class, per-class, and registered label-stratum summaries; all confidence intervals use 5,000 whole-image clustered resamples.

## Interpretation boundary

These are frozen representation-level diagnostics. They establish neither attention behavior, CAM behavior, semantic leakage, causal shortcut use, nor an intervention effect.
