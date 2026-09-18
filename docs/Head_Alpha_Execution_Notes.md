# Head-axis / alpha execution specification (2026-09-18)

Plan: `MCTformerPlus_HeadAxis_and_Alpha_Plan.md`, read in full. This queue is
inference-only: no training, learnable head weights, extra seeds, COCO or joint
head/alpha slice. Gates produce decisions, never automatically start a new method.

## Source and precision gate

Final epoch45 seed0 GWRP/all-product from
`results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2` are frozen.
Stage0 at code `95447b4` completed VOCval1449, no segmentation masks. Every
forward is FP32, autocast and TF32 disabled. A half round-trip of the same
probabilities isolates cast error; this is not a full AMP-backbone comparison.
Before measurement the1e-4-order gate was defined as means<1e-3. All-product
image-average TV=0.0001501707 (95% image CI0.0001474399..0.0001530152),
pooled-positive average=0.0001472097, max=0.0022649374. No positive row went
all-zero. GWRP hypothetical-product TV=0.0001240604; its actual GWRP branch
does not use these weights. Gate PASSED; not proof every training epoch is clean.

## Formula / protocol decisions made before scans

- Native: head mean then mean of L10–L12 C2P; `sqrt(ReLU(M)*a)`;
  all12 head-mean P2P sum multiplying the seed in the native direction.
- CAM uses native scales1,.75,1.25 plus flip and minbound448; spatial grids
  can be rectangular, not always784 patches. Upsample to original image size,
  unflip and sum scales, gate by image labels, then per-class min-max with1e-8.
- Existing threshold grid is60 values0..0.59, not the plan's rough50-bin
  storage estimate. Rank at.45; best threshold remains diagnostic.
- A has21 candidates plus an independently computed exact native reference.
  Endpoints return exact ReLU(M) / a, including zeros. Interior uses the plan's
  epsilon1e-8 before exponentiation. Clamped alpha.5 can differ from native:
  both are retained, and crossfit deltas are reported against both references.
- Hash-driven multilabel stratification fixes732/732 folds; every class count
  differs by at most1. SHA256 resolves ordering/ties. No masks/scores enter
  splitting. Select alpha on one half, report the other; main statistic is the
  size-weighted average of two held-out dataset mIoUs. Each5000 paired image
  bootstrap draw resamples within halves and RESELECTS alpha. Pooled held-out
  confusion mIoU is secondary. Native threshold crossfit is a separate diagnostic.
- H keeps alpha native and P native:72 cells+native. Kappa/Gini average over
  positive classes inside images; gamma averages positive pairs inside images,
  excludes single-label/undefined images rather than treating them as zero.
- Freeze ascending-kappa and ascending-gamma ranks on VOCval image labels,
  WITHOUT semantic masks, before head mIoU evaluation. Evaluate both prespecified
  top-m lists1,2,4,8,16,32,72. No mIoU-ranked oracles. m72 equals an all-layer,
  all-head MEAN C2P CAM readout, not the all-product training-pooling operation.
- Best/median/worst row-normalized head controls are selected only for testing
  scaling. Per-view row scalars need not cancel after multi-view summation;
  final min-max also has epsilon. Differences do not by themselves imply that
  the implementation lacks class-wise normalization.
- Head/alpha readouts share a per-image/per-view forward within each stage;
  no per-configuration forward and no attention/CAM dumps. A runs first, then
  H, then rownorm/top-m. Image-confusion statistics are retained for paired CIs.
- Cross-host same-cell signs are reported. A stricter head gate requires
  positive lower CIs in both hosts. This does not correct72-test multiplicity.
  In-sample maxima / best m are exploratory, not independent generalization.

## Commands and status

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/peng/anaconda3/envs/tgca-repro/bin/python -u -m analysis.head_alpha --preflight results/head_alpha/20260918-voc/preflight --output results/head_alpha/20260918-voc/smoke --limit 4
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/peng/anaconda3/envs/tgca-repro/bin/python -u -m analysis.head_alpha --preflight results/head_alpha/20260918-voc/preflight --output results/head_alpha/20260918-voc/full
```

The smoke must finish before full inference. Results are not complete until
the corresponding `SMOKE_COMPLETE` / `COMPLETE` markers exist and source hashes
match. Source checkpoints and user plans are immutable; only new result paths
are written. Tests and integration status are logged beside the results.
