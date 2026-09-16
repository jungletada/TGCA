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

## Stage 1 completed

Full val1449, all three checkpoints, 5000 image-paired bootstrap completed.
`results/artifact_probe/20260916-voc-s0/decision.md`:
GWRP M2/M4/M5 = .013429/.081556/.323059 (all gates pass).
All-product = .005396/.063604/.007630; affinity = .016951/.088546/.066659.
Both product checkpoints fail the overlap gate. No register implemented.
Source checkpoint/data-list/label hashes unchanged. Ten probe tests passed.

## Stage A preregistration / explicit interpretation

The plan lists 72 A1 configurations but its full Cartesian product has648.
No exact72 list is supplied. With no replacement list received, use deterministic
72-point coverage: original setting, every single-factor deviation, conservative
arm, then greedy maximin Hamming-distance filling across648 (total Hamming
distance secondary, lexical config ID tertiary). This covers all listed variable
levels and contains both specified Stage B controls, but is NOT an exhaustive
factorial or an independently optimized experimental design.
`analysis.affinity_repair.preregistered_configs()` specifies the exact sequence.
`configs.json` is written before the first image is processed.
Select by mean image entropy[.35,.65] and DC<.5; rank distance to entropy.5,
then config ID. Retain at most10 including mandatory minimal/conservative
controls, explicitly label controls that fail the band. Native CAM is an
additional reference, not counted as a repair candidate.

Stage B has both a three-training instruction and a conditional gate in§3.7;
requested confirmation of the gate (default: A2 fixed-threshold exceeds the
unrounded historical all-product mIoU, reported as72.315%).

CAM interpretation: keep S=sqrt(ReLU(M)*native-last3 mean A_c2p). Compute P*S,
subtract min / positive centered mean across space. beta1 preserves resulting
raw CAM scale (no added mass normalization). For beta<1, mix normalized S and
normalized repaired response, then restore the repaired response's total mass.
This matches the pooling damping distribution while avoiding an unrequested
class-wise scale change for beta1. Zero-floor rows fall back to S and are counted.
Native reference uses exact original sum and matmul and is checked against
`get_cam`, then against historical full1464 mIoU (tolerance.0001 fraction).
All candidates share the exact same scales/flip/resize/minmax/gating. No CAM dumps.
The candidate search is a screening result, not an unbiased generalization claim.

The DC explanation remains a hypothesis. Sum versus mean differs only by a
global factor and cancels under normalized propagation (except epsilon effects).
Row-dependent group mass, operator orientation, spatial structure and post-CAM
normalization must be distinguished. Inference screening alone does not prove
a causal training mechanism or independent generalization improvement.
