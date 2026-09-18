# Migration and scope

## Layout, not a model dependency

```python
from vitprobe import TokenLayout

multi_query = TokenLayout.mctformer_plus(n_class=20, grid=(28, 28))
register_model = TokenLayout.dinov3(grid=(28, 28))
plain = TokenLayout(n_cls=1, grid_hw=(28, 28))
patch_first = TokenLayout(n_cls=1, n_class=20, n_register=4,
                          grid_hw=(28, 32), class_first=False)
```

Extra tokens are ALWAYS ordered `cls -> class -> register`. `class_first`
places the complete extra block before/after patches, without reversing it.
The named factories are the user-approved exception to the name-search rule;
they construct metadata only. The register-model factory describes a layout,
not a model implementation or an assertion about every possible variant.

Use `layout.patch_tokens(tokens)` for `[B,N,D]`;
`layout.patch_grid(tokens)` for `[B,H,W,D]`;
`layout.patch_attention(attention)` for `[...,N,N]`;
`layout.query_patch_attention(attention, 'cls'|'class'|'register')` for query maps.
Slices preserve spatial order, dtype and gradients. Empty groups remain empty;
they are not reinterpreted as another token group. Rectangular cross-attention
is not implicitly sliced as self-attention: current graph inputs are `[B,H,T,T]`.

## Diagnostics in other tasks

| API | Input / interpretation |
| --- | --- |
| `stats.conditional_attention` | Spatial maps `[...,N]`; zero mass becomes NaN. |
| `stats.effective_support`, `gini`, `normalized_entropy` | Already conditional nonnegative distributions; concentration, not quality by itself. |
| `stats.weight_stats` | `[B,Q,N]`, `[B,Q]` query-selection mask; image-wise entropy/top1/minimum-floor/pair overlap. |
| `relations.class_token_affinity` | Arbitrary selected query vectors `[B,Q,D]`; name retained, no semantic labels required. |
| `relations.head_statistics`, `class_agnosticism` | `[cells,B,Q,N]`, query mask; gamma measures positive-query support similarity. |
| `relations.symptom_overlap` | Two patch descriptors `[B,N]`; top-support Jaccard. |
| `artifacts.patch_norm_stats` | Patch-only `[B,N,D]`; relative norm outliers. |
| `artifacts.attractor_stats` | Head-reduced `[B,N,N]`; incoming KEY mass, not outgoing query mass. |
| `artifacts.lowinfo_scores` | Unnormalized RGB and patch ratios; `layout=` enables rectangular grids. Not applicable to non-image inputs without a separate descriptor. |
| `spectral.spectrum_and_autocorrelation` | `[B,H,W,D]`, H/W>=28; token L2, spatial demeaning, Hann then FFT. |
| `spectral.radial_power_spectrum`, `high_freq_ratio`, `correlation_length` | Equal-weight radial-bin energy summaries and actual 2D autocorrelation; constant/invalid observations retain missingness. |
| `state.isolated_eval` | FP32 master model, device; exception-safe module/RNG/TF32 restoration. |

For a single query, gamma and positive-pair affinity are undefined, not zero.
The non-semantic smoke uses register queries as an arbitrary multi-query example;
this is not a claim that registers are semantic class tokens.

## Generic graph operators

`graph.aggregate_p2p(records, layout, ...)` preserves sequential accumulation,
layer/head reduction, optional row normalization, exponent and symmetry.
`graph.propagate_weights(w, P, ...)` preserves `w @ P.T`, existing floor/damping
and explicit fallback mask. Input FP64 stays FP64; lower precision accumulates
in FP32 with autocast off, exactly as in the source functions. This established
behavior does not resolve the separately pending NEW numerics module policy.

No model wiring, classifier pooling, repair configuration search, or loss was
migrated. These are callable matrix operators, not an enabled repair method.

## Streaming scans

```python
from vitprobe.sweep import ConfigSweep

sweep = ConfigSweep(configs=[{'setting': 0}, {'setting': 1}],
                    n_thresholds=3, n_classes=5)
# samples yields (inputs, integer_target).
# forward is called once per sample; readout must not mutate shared features.
# readout returns integer predictions [threshold,*target.shape].
# rows = sweep.run(samples, forward, readout)
rows = sweep.results()                 # ordinary records
# frame = sweep.results(dataframe=True) # optional vitprobe[dataframe]
```

Empty totals yield undefined ratios; no result is meaningful before observations.
For thresholded winner scores use `image_threshold_confusions` and
`sweep.observe_confusions` instead of retaining a dense prediction stack.
Strict score>threshold, class0 background, void255 default and tie behavior match
the source; class count and ignore index are explicit. Totals use int64.

Framework callbacks are newly introduced and NOT VALIDATED AGAINST TGCA RUNS;
their counting/one-forward semantics have synthetic tests. Domain-specific data
loading, augmentations, feature acquisition and masks are the caller's job.

## Two different paired estimands

```python
from vitprobe.bootstrap import paired_image_bootstrap, paired_confusion_bootstrap

# values: [2,images,...], already reduced within each image.
# point, ci, count = paired_image_bootstrap(values, seed=20260916)
# reference, variant: [images,C,C], same ordered image IDs, fixed threshold.
# ci = paired_confusion_bootstrap(variant, reference, seed=20260916)
```

Defaults: 5000 image draws, seed20260916; paired sample0/reference,
sample1/variant, delta=1-0. `paired_image_bootstrap` returns point `[3,...]`,
CI `[2,3,...]`, finite-intersection image counts `[...]`; its source default seed
was20260914, so exact historical reproduction must pass that seed explicitly.
Missing observations are intersected per metric across models before sampling.

`paired_confusion_bootstrap` returns CI `[2,3]` of dataset-global mean IoU:
resample complete image matrices THEN compute IoU. It preserves the source
percentile behavior, including NaN for undefined resampled statistics; it does
not average per-image IoU. No training-seed uncertainty, no selection correction,
and no crossfit implementation is implied by either API.

Keep paired image IDs in your manifest and retain per-image sufficient statistics
if CI is needed. Aggregate totals alone cannot reconstruct image bootstrap.

## Not migrated / not completed

- GWRP, training C2P pooling, the affinity repair workflow, CAM generation,
  VOC/COCO data pipelines, checkpoint loading and training runners remain in TGCA.
- New `numerics.py` algorithms are pending the user's separate policy decision;
  there are no placeholder implementations or silently skipped parity claims.
- Templates describe new runs. Historical metadata and evidence remain immutable;
  missing historical fields are not retroactively invented.

## Testing and provenance

Use `TGCA_PATH` to enable live original-source parity; without it those cases
explicitly skip. Tensor tolerance is atol1e-7/rtol0 (equal NaNs), never relaxed.
Image-bootstrap CIs use the same input, seed and default5000 draws as references.
`evidence_manifest.json` maps byte-identical compact copies to source paths/hashes;
implementation provenance and current commands are recorded in validation files.
Passing synthetic tests does not establish quality on a new host or dataset.
