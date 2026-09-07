# Spatial Graph Validation Report

## Frozen contract

- Model: matched seed-0 PatchFinalLN MCTformer+-Small, checkpoint SHA256 `a2a67b293aec37baea859e80a1e09f17b9fbe72afc2b87cbde22907638cc3227`.
- Data: full VOC 2012 val, `1449` images, deterministic 448 input and 28×28 patch grid.
- Every graph uses the identical 2,970-edge undirected 8-neighbor candidate set. GT was never passed to graph construction; mixed/void patches are excluded from primary metrics.
- B0=uniform, B1=fixed σs=1 spatial Gaussian, B2=parameter-free cosine affinity, B3=B1×B2. No graph filtering or training was performed.

## Primary full-val micro results

| Graph | Purity ↑ | Boundary leakage ↓ | FG-only purity ↑ | FG–BG leakage ↓ | Edge AUROC ↑ | Edge AUPRC ↑ | FG-only AUROC ↑ | FG-only AUPRC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_uniform | 0.935736 | 0.064264 | 0.985494 | 0.059319 | 0.500000 | 0.935736 | 0.500000 | 0.985494 |
| B1_spatial | 0.939122 | 0.060878 | 0.986318 | 0.056202 | 0.557757 | 0.942783 | 0.559277 | 0.987194 |
| B2_feature | 0.939045 | 0.060955 | 0.987065 | 0.056482 | 0.742384 | 0.976234 | 0.844248 | 0.997033 |
| B3_spatial_feature | 0.942243 | 0.057757 | 0.987782 | 0.053523 | 0.670173 | 0.967007 | 0.727022 | 0.994198 |

## Paired B3 deltas (whole-image clustered 95% CI)

| Comparison | Metric | Δ | 95% CI |
|---|---|---:|---:|
| B3-B1 | foreground_only_purity | 0.001464 | [0.001276, 0.001662] |
| B3-B1 | foreground_background_leakage | -0.002679 | [-0.002812, -0.002548] |
| B3-B1 | edge_auroc | 0.112416 | [0.109737, 0.115123] |
| B3-B1 | edge_auprc | 0.024224 | [0.023376, 0.024990] |
| B3-B1 | foreground_only_edge_auroc | 0.167745 | [0.161323, 0.174056] |
| B3-B0 | foreground_only_purity | 0.002288 | [0.002009, 0.002583] |
| B3-B0 | foreground_background_leakage | -0.005796 | [-0.005987, -0.005615] |
| B3-B0 | edge_auroc | 0.170173 | [0.167349, 0.172939] |
| B3-B0 | edge_auprc | 0.031271 | [0.030246, 0.032221] |
| B3-B0 | foreground_only_edge_auroc | 0.227022 | [0.220105, 0.233744] |
| B3-B2 | foreground_only_purity | 0.000717 | [0.000627, 0.000807] |
| B3-B2 | foreground_background_leakage | -0.002959 | [-0.003041, -0.002881] |
| B3-B2 | edge_auroc | -0.072211 | [-0.074809, -0.069712] |
| B3-B2 | edge_auprc | -0.009227 | [-0.009587, -0.008812] |
| B3-B2 | foreground_only_edge_auroc | -0.117226 | [-0.123207, -0.111315] |

## Boundary-focused weights

| Graph | Same interior | Same boundary-near | Cross boundary | FG–BG boundary |
|---|---:|---:|---:|---:|
| B0_uniform | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| B1_spatial | 0.491826 | 0.489479 | 0.463801 | 0.463873 |
| B2_feature | 0.926276 | 0.924244 | 0.875117 | 0.878497 |
| B3_spatial_feature | 0.456335 | 0.453175 | 0.406748 | 0.408351 |

## Statistical interpretation

By the plan's qualitative/statistical criteria, the frozen B3 graph receives **strong support**. This label is diagnostic, not a claim that graph smoothing or a new method improves CAMs.

All intervals use exactly `5000` paired whole-image resamples with seed `20260901`. Edge rank point estimates are exact; their clustered intervals use `2048` fixed bins, with maximum observed full-sample approximation error `8.039e-04`. Edges from an image were never independently resampled.

Per-class, macro-class, image-label strata, boundary distributions, and every paired delta are available in the accompanying CSV files. The analysis establishes only whether the frozen local affinity graph aligns with patch-level semantic structure.
