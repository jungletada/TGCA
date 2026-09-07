# PatchFinalLN MCTformer+ Basis-Control Report

## Frozen inputs and scope

- PatchFinalLN checkpoint: `/home/peng/code/TGCA/results/final_ln_ablation/20260906-mctformerplus-patch-final-ln-s0-64f2aa9/checkpoints/patch_final_ln/mctformerplus_final.pth`.
- Checkpoint SHA256: `a2a67b293aec37baea859e80a1e09f17b9fbe72afc2b87cbde22907638cc3227`.
- Full VOC val: `1449` images, deterministic 448 transform, 28×28 patches.
- The semantic 3×3 map is held invariant by transforming every input-channel kernel slice; GT is used only after selection for union-FG/BG composition.

## Selector results

| Basis | Vote Spearman | Top-10% Jaccard | FG vote mass (valid) | FG vote enrichment |
|---|---:|---:|---:|---:|
| permutation | 0.196698 | 0.189808 | 0.374985 | 1.289229 |
| signed_permutation | 0.181931 | 0.177744 | 0.369071 | 1.256465 |
| haar_00 | 0.176488 | 0.177521 | 0.373147 | 1.286239 |
| haar_01 | 0.184056 | 0.182654 | 0.373140 | 1.294187 |
| haar_02 | 0.181815 | 0.181130 | 0.365701 | 1.280616 |
| haar_03 | 0.172155 | 0.171365 | 0.375328 | 1.304225 |
| haar_04 | 0.172798 | 0.174137 | 0.356809 | 1.175402 |
| haar_05 | 0.182163 | 0.181253 | 0.376705 | 1.279982 |
| haar_06 | 0.182869 | 0.184091 | 0.372403 | 1.297455 |
| haar_07 | 0.164387 | 0.168896 | 0.372728 | 1.266224 |
| haar_08 | 0.172731 | 0.176541 | 0.375214 | 1.258554 |
| haar_09 | 0.182394 | 0.182286 | 0.371690 | 1.296587 |

## Invariance controls and interpretation

The reparameterized 3×3 classifier reproduced its semantic map with maximum float64 absolute error `2.722e-07`; pairwise cosine affinity remained invariant with maximum error `3.625e-09` (both required `<1e-5`).

Any vote-map differences therefore belong to the channel-index Fourier selector, while the spatial semantic map and cosine graph remain equivalent. This is a frozen representation-level diagnosis, not a trained-method or localization claim.

The official unclamped selector produced 54 non-finite ranking entries on this input set. Selected original patch tokens remained finite; the counts are preserved in run metadata rather than being hidden by a denominator clamp.
