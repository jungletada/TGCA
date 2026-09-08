# Positive-Channel Relation Report

## Frozen contract

- `S_pos = ReLU(c)^T p / sqrt(D)` uses coordinate signs defined by the native mean class-token readout. It is a coordinate-dependent diagnostic, not an intrinsic rotation-invariant relation or a proposed method.
- `S_last = S_pos - S_negmag` was numerically verified per forward batch; GT enters only after all maps are fixed.

## E2 semantic diagnostics (micro)

| Relation | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |
|---|---:|---:|---:|---:|---:|
| s_last | 0.561988 | 0.551990 | 1.702038 | 0.790343 | 0.405217 |
| s_pos | 0.561354 | 0.533688 | 1.630328 | 0.777795 | 0.401025 |
| s_negmag | 0.448998 | 0.428319 | 1.072513 | 1.090343 | 0.271076 |

## Paired S_pos − S_last effects

| Family | Metric | Δ | 95% CI |
|---|---|---:|---:|
| semantic | auc_target_bg | -0.000634 | [-0.003258, 0.001988] |
| semantic | auc_target_other | -0.018302 | [-0.024101, -0.013107] |
| semantic | target_tail_enrich_10 | -0.071710 | [-0.098080, -0.045422] |
| semantic | bg_tail_enrich_10 | -0.012548 | [-0.022718, -0.003459] |
| multilabel | top10_jaccard | 0.062819 | [0.051659, 0.074635] |

## Positive-coordinate structure: GT-positive versus absent classes (micro)

| Class status | Positive-channel fraction | Positive mass | Negative mass | Positive/negative ratio | Native logit |
|---|---:|---:|---:|---:|---:|
| positive | 0.608511 | 2582.250485 | 1208.325813 | 2.334866 | 3.577929 |
| absent | 0.383410 | 2453.794384 | 5162.979219 | 0.487601 | -7.055169 |

## Positive-mask/token specificity (micro)

| Comparison | Positive-mask Jaccard | Positive-mask cosine | ReLU-token cosine |
|---|---:|---:|---:|
| positive_classes_same_image | 0.520540 | 0.636236 | 0.599804 |
| same_class_other_images | 0.756137 | 0.825309 | 0.822653 |
| different_class_other_images | 0.675160 | 0.768200 | 0.768732 |

## Positive-coordinate patch distributions (micro mean)

| Signal | Target | Other-FG | Background | Target − BG | Target − other-FG |
|---|---:|---:|---:|---:|---:|
| u_pos | 0.114144 | 0.190222 | -0.580246 | 0.707280 | 0.231055 |
| c_sign | 0.526008 | 0.525391 | 0.472540 | 0.053283 | 0.019445 |
| s_pos | 133.927032 | 49.920082 | -86.025100 | 223.102456 | 128.035995 |

## Positive/negative decomposition by region (micro)

| Term | Target mean | Other-FG mean | BG mean | Target-vs-BG AUROC | Target-vs-other-FG AUROC |
|---|---:|---:|---:|---:|---:|
| s_pos | 133.927032 | 49.920082 | -86.025100 | 0.561354 | 0.533688 |
| s_negmag | -57.792886 | 18.899560 | -6.919134 | 0.448998 | 0.428319 |
| s_last | 191.719916 | 31.020524 | -79.105965 | 0.561988 | 0.551990 |

## `S_pos` confidence stratification (micro)

| Native-logit quintile | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% |
|---|---:|---:|---:|
| 0 | 0.496499 | 0.470083 | 1.480797 |
| 1 | 0.455211 | 0.453619 | 1.157492 |
| 2 | 0.578061 | 0.589232 | 1.793308 |
| 3 | 0.604364 | 0.593554 | 1.752632 |
| 4 | 0.653427 | 0.621879 | 1.923882 |

## Coordinate-basis dependence

| Transform | S_pos map Spearman | Top-10% Jaccard | Normalized L1 |
|---|---:|---:|---:|
| haar_00 | 0.941637 | 0.813900 | 0.560190 |
| haar_01 | 0.939426 | 0.818863 | 0.553672 |
| haar_02 | 0.938699 | 0.808062 | 0.572665 |
| haar_03 | 0.941188 | 0.816190 | 0.520038 |
| haar_04 | 0.943341 | 0.823993 | 0.530553 |
| identity | 1.000000 | 1.000000 | 0.000000 |
| permutation | 1.000000 | 1.000000 | 0.000000 |
| signed_permutation | 0.948924 | 0.853463 | 0.384474 |

## Required E2 interpretation

- Relative to `S_last`, `S_pos` changes target-vs-BG AUROC by -0.000634 [-0.003258, 0.001988] and target-vs-other-FG AUROC by -0.018302 [-0.024101, -0.013107] in paired whole-image resampling.
- Positive-mask Jaccard is 0.520540 [0.500504, 0.541938] for different positive classes within an image and 0.675160 [0.667659, 0.682478] for different classes across images. These are representation-coordinate similarities, not evidence of an attention or causal process.
- Ordinary permutations preserve `S_pos`; signed permutations and Haar rotations are recorded as coordinate-basis diagnostics, so any empirical utility remains classifier-coordinate-specific rather than intrinsic to the representation.
- CSV tables contain micro, macro-class, per-class, and registered label-stratum summaries; all confidence intervals use 5,000 whole-image clustered resamples.

## Interpretation boundary

Positive-coordinate utility, if observed, is only a classifier-coordinate mechanism because signed permutations and rotations may change ReLU-coordinate selection. No selector, aggregation, loss, CAM modification, or training is implied by this analysis.
