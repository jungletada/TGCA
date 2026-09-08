# Frozen Multi-Class Token Relation Analysis

## Scope and integrity

This is a frozen, inference-only analysis of the audited native MCTformer+-Small checkpoint at single-scale 448 on all 1,449 VOC-val images. All class--patch relations were calculated in float32 from raw post-block representations. No training, loss, backward pass, attention change, token change, or GT-conditioned score construction occurred. GT enters only after S0--S5 were saved, for region evaluation.

- Checkpoint SHA256: `aced3d3bd69c57782c8e85f1d10abd7ff7ab02504df92c2f2f3898defcf7a65a`
- Checkpoint/model parameter SHA256 before and after: `5f9a4cf2aa1c56a448b421038826e8abd6ecb896162b30d1a6d3593564b06f36` / `5f9a4cf2aa1c56a448b421038826e8abd6ecb896162b30d1a6d3593564b06f36` (identical).
- Dump code commit: `2f5e16092f4c1f7f10db00b9c9154ef7475e8bc9`; this compact report commit is recorded in `compact_metadata.json`.
- Statistical unit: image × positive class; paired comparisons cluster-resample complete images, 10,000 repeats, seed 2027.

## Task A — all-layer shared-presence table

| layer | raw_corr | residual_corr | raw_collision_at5 | residual_collision_at5 | common_r2 | common_fg_at5 | pca_pc1 | pca_r95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | -0.1837 | -0.1236 | 0.1409 | 0.2172 | 0.7272 | 0.3179 | 0.3911 | 15.0000 |
| 2.0000 | -0.2083 | -0.1034 | 0.0940 | 0.1070 | 0.4165 | 0.4525 | 0.2017 | 37.0000 |
| 3.0000 | -0.1947 | -0.0434 | 0.0330 | 0.0436 | 0.3574 | 0.6084 | 0.3370 | 68.0000 |
| 4.0000 | -0.2372 | -0.0358 | 0.0643 | 0.0711 | 0.4557 | 0.3329 | 0.7649 | 32.0000 |
| 5.0000 | -0.0561 | 0.3018 | 0.0662 | 0.0939 | 0.7712 | 0.2870 | 0.8842 | 8.0000 |
| 6.0000 | 0.0987 | 0.5527 | 0.0812 | 0.2149 | 0.8419 | 0.2339 | 0.7678 | 6.0000 |
| 7.0000 | 0.1387 | 0.6024 | 0.0752 | 0.2169 | 0.8315 | 0.2236 | 0.7351 | 8.0000 |
| 8.0000 | 0.1993 | 0.6239 | 0.0789 | 0.2173 | 0.7978 | 0.1934 | 0.6316 | 27.0000 |
| 9.0000 | 0.3779 | 0.8350 | 0.2631 | 0.5477 | 0.8505 | 0.2070 | 0.4256 | 19.0000 |
| 10.0000 | 0.4439 | 0.9001 | 0.4645 | 0.8083 | 0.9614 | 0.2771 | 0.6792 | 9.0000 |
| 11.0000 | 0.4529 | 0.8934 | 0.5504 | 0.8644 | 0.9829 | 0.3130 | 0.7664 | 6.0000 |
| 12.0000 | 0.4517 | 0.8921 | 0.5896 | 0.8773 | 0.9866 | 0.3188 | 0.7937 | 6.0000 |

### Question A — where does shared activation form?

A common additive component is already strong by L5--L7 (R² 0.771--0.831) and the raw cross-class relation becomes distinctly positive at L6, then rises sharply at L9--L12 (raw correlation 0.378--0.453; raw collision@5% 0.263--0.590). However, the preregistered H1 criterion is *not supported*: residual cross-class correlation and collision are larger, not smaller, in every late layer. Thus these data show a shared low-dimensional component, but not that subtracting its all-class mean removes the observed cross-class relation coupling.

### Question B — rank-1 or higher-rank?

Late shared means are strongly low-dimensional rather than strictly rank-1: L12 PC1 explains 0.794, PC4 explains 0.941, and r95=6 (D=384). The L11→L12 rank-4 subspace has mean cosine 0.987. This supports a compact, stable late subspace, not a claim that a fixed rank-1 direction is sufficient or semantically interpretable.

## Task B — frozen final-layer selector table

| selector | target_vs_other_auroc | target_vs_other_ap | target_vs_bg_auroc | target_at5 | other_fg_at5 | bg_at5 | collision_at5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.9210 | 0.8704 | 0.9242 | 0.7431 | 0.0562 | 0.2000 | 0.0192 |
| S1 | 0.5524 | 0.5135 | 0.5418 | 0.3219 | 0.1335 | 0.5437 | 0.5168 |
| S2 | 0.5520 | 0.5133 | 0.5620 | 0.3680 | 0.1592 | 0.4717 | 0.5896 |
| S3 | 0.4997 | 0.4879 | 0.5477 | 0.3675 | 0.1874 | 0.4440 | 0.8773 |
| S4 | 0.5963 | 0.5575 | 0.5477 | 0.3852 | 0.0976 | 0.5165 | 0.0250 |
| S5 | 0.8988 | 0.8475 | 0.8918 | 0.7333 | 0.0552 | 0.2110 | 0.0184 |

### Single-label and multi-label strata

`selector_single_label_table.csv` and `selector_multi_label_table.csv` contain the corresponding complete strata. Target-vs-other is undefined for the single-label stratum when no other-foreground region exists; it is deliberately reported as missing rather than imputed.

| selector | num_images | auc_target_other | ap_target_other | target_top05_fraction | other_fg_top05_fraction | bg_top05_fraction | pair_jaccard_top05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 522.0000 | 0.9210 | 0.8704 | 0.6446 | 0.0989 | 0.2557 | 0.0192 |
| S1 | 522.0000 | 0.5524 | 0.5135 | 0.2329 | 0.2349 | 0.5310 | 0.5168 |
| S2 | 522.0000 | 0.5520 | 0.5133 | 0.2629 | 0.2802 | 0.4558 | 0.5896 |
| S3 | 522.0000 | 0.4997 | 0.4879 | 0.2536 | 0.3298 | 0.4152 | 0.8773 |
| S4 | 522.0000 | 0.5963 | 0.5575 | 0.2939 | 0.1718 | 0.5336 | 0.0250 |
| S5 | 522.0000 | 0.8988 | 0.8475 | 0.6342 | 0.0971 | 0.2680 | 0.0184 |

### Paired S5 − S0 bootstrap

| metric | estimate | ci_low | ci_high | n_clusters | n_rows |
| --- | --- | --- | --- | --- | --- |
| auc_target_other | -0.0221 | -0.0256 | -0.0187 | 1449.0000 | 2147.0000 |
| ap_target_other | -0.0229 | -0.0275 | -0.0185 | 1449.0000 | 2147.0000 |
| target_top05_fraction | -0.0098 | -0.0124 | -0.0072 | 1449.0000 | 2147.0000 |
| pair_jaccard_top05 | -0.0008 | -0.0022 | 0.0005 | 1449.0000 | 1449.0000 |

### Question C — does classifier + relative ownership improve frozen target-vs-other?

No. S5 target-vs-other AUROC/AP is 0.8988/0.8475 versus S0 0.9210/0.8704. The paired AUROC difference is negative with a 95% CI entirely below zero (see table). S5 also has a small negative target@5% difference. This fails the plan's S5 go criterion; no selector integration or trainable follow-up is warranted from this experiment.

## Interpretation boundary

These are representation-level and frozen-ranking observations. They do not establish causal attention behavior, semantic prototypes, background leakage, lazy semantic assignment, a CAM mechanism, or a new WSSS method. No further method is proposed or implemented.
