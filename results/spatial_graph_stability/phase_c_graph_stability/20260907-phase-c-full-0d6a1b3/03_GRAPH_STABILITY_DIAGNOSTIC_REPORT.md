# Graph Stability Diagnostic Report

## Frozen contract

- Frozen matched PatchFinalLN MCTformer+-Small checkpoint SHA256 `a2a67b293aec37baea859e80a1e09f17b9fbe72afc2b87cbde22907638cc3227`.
- Full deterministic VOC 2012 val: `1449` images, 448 input, 28×28 patches.
- Raw pre-ReLU 3×3 classifier response M is the only low-pass signal. B1/B2/B3 graph weights are reproduced from immutable Phase B; GT is used only after smoothing for diagnostic labels.
- Low-pass solves (I+lambda L_sym) X=M with batched sparse-edge conjugate gradients; no matrix inverse, filtering of graph edges, training, selector, or pooling was used.

## Low-pass numerical and preservation checks

| Graph | λ | Fidelity | Energy retention | Pearson(M,M_bar) | Spearman(M,M_bar) |
|---|---:|---:|---:|---:|---:|
| B1_spatial | 0.25 | 0.013623 | 0.734517 | 0.998646 | 0.997640 |
| B1_spatial | 0.5 | 0.023138 | 0.582764 | 0.996055 | 0.993388 |
| B1_spatial | 1 | 0.036178 | 0.415484 | 0.990281 | 0.984341 |
| B1_spatial | 2 | 0.051894 | 0.267809 | 0.979968 | 0.969009 |
| B1_spatial | 4 | 0.069020 | 0.159975 | 0.964777 | 0.947380 |
| B2_feature | 0.25 | 0.014591 | 0.703242 | 0.998110 | 0.996993 |
| B2_feature | 0.5 | 0.024453 | 0.542679 | 0.994626 | 0.991959 |
| B2_feature | 1 | 0.037544 | 0.374899 | 0.987262 | 0.982053 |
| B2_feature | 2 | 0.052765 | 0.235972 | 0.975081 | 0.966515 |
| B2_feature | 4 | 0.068901 | 0.140086 | 0.958584 | 0.946019 |
| B3_spatial_feature | 0.25 | 0.014451 | 0.711264 | 0.998126 | 0.996989 |
| B3_spatial_feature | 0.5 | 0.024319 | 0.551939 | 0.994631 | 0.991884 |
| B3_spatial_feature | 1 | 0.037506 | 0.383077 | 0.987163 | 0.981763 |
| B3_spatial_feature | 2 | 0.052910 | 0.241514 | 0.974714 | 0.965875 |
| B3_spatial_feature | 4 | 0.069259 | 0.143155 | 0.957858 | 0.945107 |

## Stability-alone semantic diagnostic (B2, λ=1)

| Target-vs-BG AUROC | Target-vs-other-FG AUROC | C-PiM target hit | BG tail enrichment@10% |
|---:|---:|---:|---:|
| 0.380455 | 0.379566 | 0.217047 | 1.037382 |

## Interpretation boundary

Graph stability is a frozen structural diagnostic. Its semantic region scores do not establish a selector, CAM improvement, causal intervention, or proposed method. Phase D tests whether it is useful only after class-specific relevance is supplied.

The maximum basis-regression error was `3.588e-13` (required <1e-5); the maximum final linear-solve residual was `9.055e-06` (required <1e-5).
