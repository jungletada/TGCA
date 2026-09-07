# Relevance–Stability Ablation Report

## Frozen contract

- Phase D reuses the immutable Phase C raw M and λ=1 B1/B2/B3 stability caches; no checkpoint, graph weight, model parameter, selector, pooling operator, or loss was changed.
- Relevance R=ReLU(M)/max(ReLU(M)); S is bounded graph stability; R×S uses the fixed exponents 1 and 1.
- GT appears only in diagnostic evaluation. Every interval uses whole-image clustered bootstrap resampling, never independent patches.

## Primary micro semantic diagnostics

| Graph | Score | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |
|---|---|---:|---:|---:|---:|---:|
| B1_spatial | relevance | 0.904683 | 0.897405 | 4.659674 | 0.446737 | 0.826269 |
| B1_spatial | stability | 0.373426 | 0.374948 | 0.687012 | 1.036746 | 0.210061 |
| B1_spatial | relevance_x_stability | 0.821984 | 0.866762 | 3.316662 | 0.603688 | 0.725664 |
| B2_feature | relevance | 0.904683 | 0.897405 | 4.659674 | 0.446737 | 0.826269 |
| B2_feature | stability | 0.380455 | 0.379566 | 0.698851 | 1.037382 | 0.217047 |
| B2_feature | relevance_x_stability | 0.823935 | 0.868145 | 3.356244 | 0.598427 | 0.741966 |
| B3_spatial_feature | relevance | 0.904683 | 0.897405 | 4.659674 | 0.446737 | 0.826269 |
| B3_spatial_feature | stability | 0.383463 | 0.380506 | 0.700661 | 1.033067 | 0.214718 |
| B3_spatial_feature | relevance_x_stability | 0.825592 | 0.868518 | 3.379001 | 0.593827 | 0.741034 |

## Paired R×S effects

| Comparison | Graph | Metric | Δ | 95% CI |
|---|---|---|---:|---:|
| relevance_x_stability_minus_relevance | B1_spatial | target_tail_enrich_10 | -1.343012 | [-1.441366, -1.246479] |
| relevance_x_stability_minus_relevance | B1_spatial | auc_target_bg | -0.082699 | [-0.087064, -0.078579] |
| relevance_x_stability_minus_relevance | B1_spatial | auc_target_other | -0.030643 | [-0.036745, -0.025628] |
| relevance_x_stability_minus_stability | B1_spatial | target_tail_enrich_10 | 2.629650 | [2.517890, 2.741509] |
| relevance_x_stability_minus_stability | B1_spatial | auc_target_bg | 0.448558 | [0.439288, 0.457966] |
| relevance_x_stability_minus_stability | B1_spatial | auc_target_other | 0.491814 | [0.475977, 0.507878] |
| relevance_x_stability_minus_relevance | B2_feature | target_tail_enrich_10 | -1.303429 | [-1.400818, -1.208570] |
| relevance_x_stability_minus_relevance | B2_feature | auc_target_bg | -0.080748 | [-0.085139, -0.076595] |
| relevance_x_stability_minus_relevance | B2_feature | auc_target_other | -0.029259 | [-0.035254, -0.024290] |
| relevance_x_stability_minus_stability | B2_feature | target_tail_enrich_10 | 2.657393 | [2.544119, 2.770420] |
| relevance_x_stability_minus_stability | B2_feature | auc_target_bg | 0.443480 | [0.434180, 0.452909] |
| relevance_x_stability_minus_stability | B2_feature | auc_target_other | 0.488579 | [0.472649, 0.504576] |
| relevance_x_stability_minus_relevance | B3_spatial_feature | target_tail_enrich_10 | -1.280673 | [-1.376099, -1.188036] |
| relevance_x_stability_minus_relevance | B3_spatial_feature | auc_target_bg | -0.079092 | [-0.083320, -0.075053] |
| relevance_x_stability_minus_relevance | B3_spatial_feature | auc_target_other | -0.028886 | [-0.034892, -0.023956] |
| relevance_x_stability_minus_stability | B3_spatial_feature | target_tail_enrich_10 | 2.678340 | [2.564117, 2.792388] |
| relevance_x_stability_minus_stability | B3_spatial_feature | auc_target_bg | 0.442128 | [0.432906, 0.451590] |
| relevance_x_stability_minus_stability | B3_spatial_feature | auc_target_other | 0.488012 | [0.471933, 0.504008] |

## Interpretation boundary

These are frozen representation-level and semantic-diagnostic results. They test whether stability conditionally modifies existing relevance; they do not establish a trained pooling method, attention intervention, CAM improvement, or causal localization mechanism.

Phase C/D basis regression passed with maximum error `3.588e-13` (<1e-5).
