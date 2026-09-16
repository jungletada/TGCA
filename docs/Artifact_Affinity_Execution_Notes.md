# Artifact / affinity execution record

2026-09-16, LHR, tgca-repro. User's new request supersedes the previous stop
instruction for this plan only; the old diagnostic queue remains cancelled.

## Stage 1 pre-registration

- Frozen final checkpoints: GWRP, all-product, all-product-affinity, exactly the
  three paths in the plan. VOC val1449 / multilabel522. No semantic masks.
- FP32, TF32/autocast off; 512 short-edge bicubic, center crop448.
- Actual API returns `(class_tokens, patch_tokens, attention_records, raw_classes)`.
  Read-only block hooks compute statistics on all12 raw post-block patch tokens.
  No model source changes in this stage. Prior train/eval modes restored.
- M2 threshold is 0.005 (0.5%); M4 is 0.05 (5%); M5 is0.2. Each checkpoint judged
  independently by full-set equal-image L12 means; multi-label subset supplementary.
- M6 uses unnormalized RGB grayscale gradient, not a claim of semantic background.
  Both top1%-norm set (reference code) and actual rho>3 set are reported; empty
  high-norm sets have missing M6 with explicit counts, not a zero observation.
- Top1% uses round(7.84)=8; independent random-set Jaccard is approximately0.0055,
  not0.01. Top10% in later A1 uses ceil(78.4)=79, as in prior diagnostics.
- 5000 paired image bootstrap, seed20260916. No patch/pair independence assumption.
- Retain only compact per-image metrics, histogram counts and six prespecified
  examples. No full attention dumps or per-image CAM arrays.

## Clarifications pending before later stages

The plan lists 72 A1 configurations but its full Cartesian product has648.
No exact72 list is supplied. Requested user clarification; any proposed reduced
design must be frozen and documented before A1 measurements.
Stage B has both a three-training instruction and a conditional gate in§3.7;
requested confirmation of the gate (default: A2 fixed-threshold >72.315%).

The DC explanation remains a hypothesis. Sum versus mean differs only by a
global factor and cancels under normalized propagation (except epsilon effects).
Row-dependent group mass, operator orientation, spatial structure and post-CAM
normalization must be distinguished. Inference screening alone does not prove
a causal training mechanism or independent generalization improvement.
