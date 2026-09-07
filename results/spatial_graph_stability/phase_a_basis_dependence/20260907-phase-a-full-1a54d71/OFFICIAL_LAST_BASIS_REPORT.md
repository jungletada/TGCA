# Official LaST Basis-Dependence Report

## Frozen inputs and scope

- Official repository commit: `cdeb884af65e7774f2da80f666d95cf09a76b717`.
- Official checkpoint SHA256: `8d4593229dda2b8774a04210cb0f185efd3bd35e948ecb14036141ca7eed3253`.
- Natural-image fallback: full VOC val (`1449` images) with the official 224 validation transform.
- ImageNet val was unavailable; no ImageNet classification accuracy is reported.
- Selector is the official unclamped FFT/Gaussian/Top-1 implementation; no training occurred.

## Equivalent-basis results

| Basis | Vote Spearman | Top-10% vote Jaccard | LaST-logit cosine | Top-1 agreement |
|---|---:|---:|---:|---:|
| permutation | 0.625771 | 0.429872 | 0.983480 | 0.792271 |
| signed_permutation | 0.498750 | 0.353068 | -0.246198 | 0.692202 |
| haar_00 | 0.480224 | 0.354950 | -0.393766 | 0.684610 |
| haar_01 | 0.486551 | 0.356675 | -0.297736 | 0.679779 |
| haar_02 | 0.470519 | 0.345918 | -0.329307 | 0.663216 |
| haar_03 | 0.469124 | 0.342979 | -0.444877 | 0.677019 |
| haar_04 | 0.476380 | 0.346209 | -0.369913 | 0.673568 |
| haar_05 | 0.473839 | 0.349357 | -0.417915 | 0.685300 |
| haar_06 | 0.464533 | 0.331436 | -0.429335 | 0.687371 |
| haar_07 | 0.476464 | 0.348977 | -0.364194 | 0.683920 |
| haar_08 | 0.481453 | 0.352154 | -0.355776 | 0.680469 |
| haar_09 | 0.466806 | 0.344978 | -0.389610 | 0.685300 |

## Equivalence control and interpretation

The ordinary final CLS classifier remained equivalent with maximum float64 absolute logit error `1.132e-07` (required `<1e-5`). The table therefore measures selector/output sensitivity under representations of the same ordinary classifier function.

These are representation/selector diagnostics only. They do not establish accuracy changes, semantic leakage, or a causal mechanism.
