# TGCA research handoff

Last updated: **2026-08-30 (Asia/Tokyo)**

> **Research direction status.** TGCA, BCSS, and token-role specialization are
> completed negative explorations retained for provenance. The active research
> plan is now `docs/Persistent_Semantic_Latent_Codex_Plan.md`. Its Phase 0
> frozen-baseline instrumentation and Phase 1 patch-to-class semantic diagnostic
> are complete on exploratory `train_id` and frozen confirmatory `val_id`.
> Phase 2 has now been separately predeclared and authorized as the minimal
> seed-0 Read/Write screen described below. Do not expand it to depth, width,
> initialization, multi-latent, OT, or multi-seed studies until this gate is
> reviewed.

## Persistent Semantic Latent Phase 2 result

The first Phase 2 screen is frozen in
`docs/PERSISTENT_SEMANTIC_PHASE2.md`. It reuses the completed E0 seed-0 baseline
and sequentially trains only `read_only`, `write_only`, and `read_write` under
the matched 45-epoch VOC schedule. The core configuration is:

```text
patch stream:             patch-only DeiT-S/16, 384 dimensions
semantic latents:         20 foreground + 1 static background, 384 dimensions
shared relation:          384 dimensions
interaction:              zero-based block 11 only (paper layer 12)
ordering:                 Read then Write
Write residual gate:      learned scalar, initialized exactly to 0
attention normalization:  vanilla
CAM threshold:            fixed 0.45
seed:                     0
```

Layer 12 is the prespecified first interaction point from the strict Phase 1
confirmation. Late-3 and other depth settings remain later ablations. The
foreground and background latents stay outside patch self-attention. Relation
Q/K/V and output projections copy the corresponding pretrained block, with
parameter-free per-token normalization restoring the scale expected by those
pretrained projections. All three active variants instantiate identical
parameter shapes.

The implementation adds:

```text
models/persistent_semantic.py
tests/test_persistent_semantic.py
tools/analyze_psl_relations.py
tools/collect_psl_phase2.py
experiments/ablations/run_psl_voc_variant.sh
experiments/ablations/run_psl_phase2_screen.sh
docs/PERSISTENT_SEMANTIC_PHASE2.md
```

Before launch, the full repository suite passes `62` tests. A full-size CUDA
forward/backward at 448 verifies exact zero-gate initialization, a finite gate
gradient, strict train-to-CAM state loading, and CAM shape `[1,20,28,28]`.
An actual one-epoch training smoke also completed with finite losses before the
interaction depth was frozen; that smoke is mechanical validation only and is
not a Phase 2 result.

The immutable queue identity is:

```text
tmux:      tgca-psl-phase2
screen ID: 20260830-012501
order:     read_only -> write_only -> read_write -> comparison
results:   results/persistent_semantic/phase2/voc/
queue log: results/queues/persistent-semantic/20260830-012501-psl-phase2-<launch-commit>.log
```

The queue ran from `2026-08-30T01:27:04+09:00` through
`2026-08-30T05:22:14+09:00` at clean commit
`2630473c16fc57a74ff3c4786cbc486d8dd04e64`. All three variants completed 45
epochs, generated exactly 1464 CAM files, passed checkpoint hash validation,
and completed the fixed 200-image relation diagnostic. No matching experiment
or tmux session remains active.

The fixed-threshold seed-0 screen is strongly negative:

| Variant | Raw CAM mIoU | Delta | Final cls mAP | Semantic P/R | Background FPR |
|---|---:|---:|---:|---:|---:|
| E0 baseline | 70.063 | 0.000 | 96.410 | 80.735/85.817 | 6.756 |
| Read-only | 43.592 | -26.471 | 95.722 | 47.907/60.581 | 23.341 |
| Write-only | 36.658 | -33.405 | 30.529 | 70.427/39.920 | 5.124 |
| Read then Write | 43.427 | -26.636 | 95.744 | 48.429/62.366 | 23.540 |

Read then Write preserves image classification within the predeclared one-point
band but is `0.165` CAM point below Read-only, so semantic feedback supplies no
localization benefit. Its learned Write gate is nonzero (`-0.01516`), but its
effect is not useful. The foreground/background error is systematic:
Read-then-Write decreases IoU for all 21 evaluated classes, while Read-only
improves only one class by a negligible amount.

The relation diagnostic identifies a mechanism failure rather than an
incomplete run. Read-then-Write patch-to-semantic foreground accuracy is only
`2.85%`, background accuracy is `0.58%`, and relation mIoU is `0.24%`. Its mean
background Write mass is only `0.54%`, despite VOC crops being predominantly
background. Relation-logit standard deviation grows to `12.88`, normalized
Write entropy falls to `0.097`, and the relation becomes overconfident without
learning semantic ownership. Read-only shows the same localization failure;
Write-only confirms that static latents cannot classify an image without first
reading visual evidence.

The machine-readable decision is `no_go_for_phase2_expansion`:

```text
results/persistent_semantic/phase2/voc/comparisons/20260830-012501-s0-2630473/comparison.json
results/persistent_semantic/phase2/voc/comparisons/20260830-012501-s0-2630473/comparison.csv
```

This rejects the current minimal Late-1 shared-relation formulation. It does
not by itself prove that every persistent-semantic-latent architecture is
impossible, but the predeclared gate does not support more seeds, a depth
sweep, width reduction, dynamic initialization, multi-latent variants, or OT.
Any follow-up must first isolate whether the loss comes from replacing joint
attention, using the Read relation directly as CAM, or unregularized relation
scale/background allocation; it must not be presented as a routine Phase 2
expansion.

## Persistent Semantic Latent Phase 0/1 implementation

The current implementation is non-invasive: it registers forward hooks on the
12 existing `TokenGroupNormalizer` modules, where hook inputs are the exact raw
`QK^T/sqrt(d)` logits and hook outputs are the attention matrices used by the
baseline. It does not add parameters, alter the model forward result, retrain a
checkpoint, or implement the proposed Phase 2 architecture.

The frozen host is the matched MCTformer+ E0 seed-0 checkpoint:

```text
checkpoint: results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3/mctformerplus_final.pth
sha256:     41ac9ce47f6a22875cba32edb92c31c150e804ae5ae19824c2585e4e3cda7a2a
baseline:   70.06305830908285 raw CAM mIoU at fixed threshold 0.45
split:      VOC train_id, 1464 images
```

Implementation and protocol:

```text
analysis/semantic_relations.py
tools/analyze_patch_to_class.py
tools/review_patch_to_class_result.py
tests/test_semantic_relations.py
experiments/diagnostics/run_persistent_semantic_phase01.sh
experiments/diagnostics/run_persistent_semantic_val_confirmatory.sh
docs/PERSISTENT_SEMANTIC_PHASE01.md
```

Phase 0 records raw-logit and post-softmax summaries at resolutions
`224,320,448,512`, per image/layer/head/direction. Two fixed 224-resolution
images retain complete 12-layer raw and post-softmax matrices. Phase 1 evaluates
per-layer `softmax_p(S_cp)`, strict 20-class `softmax_c(S_pc)`, an explicitly
separate image-label-masked version, mutual maps, foreground-only semantic
metrics, and A/B/C/D GT composition. Primary high/low thresholds are fixed at
`0.5`; a common `0.05,0.10,0.25,0.50` calibration sensitivity grid is
diagnostic only.

Validation before the full launch:

```text
full repository tests: 51 passed
GPU smoke:              5 VOC images, all four resolutions, complete
checkpoint load:        strict
maximum FP32 row error: 5.96e-7 in the earlier 2-image smoke
shell validation:       bash -n passed; shellcheck is not installed on LHR
```

The 5-image smoke showed that the output is informative enough to distinguish
diffuse calibration from random ranking, but it is not scientific evidence and
must not be reported as the Phase 1 result.

The full Phase 0/1 queue launched from clean result-critical commit `e222676`
at `2026-08-29T01:09:48+09:00` and completed normally at
`2026-08-29T01:13:05+09:00`:

```text
tmux:     tgca-psl-phase01
run ID:   20260829-010948-persistent-semantic-phase01-e222676
run dir:  results/persistent_semantic/voc/20260829-010948-persistent-semantic-phase01-e222676
queue log: results/queues/persistent-semantic/20260829-010948-persistent-semantic-phase01-e222676.log
PID:      385018 at launch
```

Both `analysis/completion.json` and the run-root `completion.json` are present;
the analysis processed all 1464 images in `194.38` seconds. The outer tmux
wrapper's cosmetic exit-status print was expanded early, but the inner runner
emitted `QUEUE_COMPLETE` and tmux exited. There is no active experiment now.

Result validation:

```text
checkpoint hash:       exact expected match
train_id hash:         aa623bd2c8ce4443a8aaae51c524c0eb165e8e44caf6aa9e3ad33d7b75a3ef20
Phase 0 image rows:    281088 = 1464 * 4 resolutions * 12 layers * 4 directions
Phase 0 head rows:     8064 = 4 * 12 * 6 * 4 * 7 metrics
Phase 1 image rows:    17568 = 1464 * 12
maximum FP32 row error: 9.5367431640625e-7
raw matrix dumps:      2, each [12,6,216,216] for raw and post attention
sample/preview dumps:  12 / 12
total result size:     99 MiB
```

### Phase 1 scientific result

The strict result normalizes `S_pc` over all 20 class keys and evaluates only
GT foreground patches. This avoids the strong single-label shortcut in the
image-label-masked diagnostic. The best strict layer is layer 12:

```text
patch-weighted foreground accuracy: 57.363%
foreground-restricted class mIoU:    43.506%
macro image accuracy:                60.399%
image-bootstrap 95% CI:              [58.781%, 62.011%]
uniform 20-class reference:          5.000%
```

Every class is above the 5% uniform reference; per-class accuracies range from
`16.77%` for chair to `86.91%` for cow. This is strong evidence that the
baseline patch-to-class raw relation contains non-random patch semantics. The
image-label-masked result peaks at `83.77%` patch-weighted accuracy in layer 10,
but is secondary because masking makes single-label images trivial and forces
all patches into a foreground class.

Direct per-layer relation maps are much weaker localizers than the actual
MCTformer+ pipeline: the best conditional class-to-patch CAM mIoU is `6.92%`,
best post-global-softmax class-to-patch mIoU is `15.70%`, and best mutual-map
mIoU is `19.49%`. These use a single 448 center crop at patch resolution and
must not be compared as if they were the trusted multi-scale final CAM result
of `70.063%`. The mutual result is diagnostic only.

The primary `0.5` Region C test does **not** establish complementarity. Layer 10
has non-empty Region C in only two images, with mean recovery of only
`0.0017%`; the original run's mechanical `region_c=true` flag therefore has
insufficient support. Post-result code adds a conservative guard requiring at
least `max(30, 5% of images)` and at least 1% recovery. Under that guard, the
primary-threshold Region C and recovery flags are false.

The shared threshold-sensitivity diagnostic is more encouraging but remains
secondary. At threshold `0.10`, layer 12 obtains `56.46%` binary foreground
precision, `65.70%` recall, Region C target purity `44.32%`, and Region C
recovery `59.30%`. At threshold `0.05`, Region C recovery is `78.07%` with
`35.31%` target purity. Thus the relation has useful ranking but diffuse class
probabilities; these thresholds were not used to replace the primary `0.5`
gate and do not yet prove a deployable foreground/background decision.

Overall Phase 1 is a **go for the intrinsic semantic-attribution hypothesis**,
because one prespecified criterion, clearly non-random `A_pc`, is strongly met.
It is not yet a go based on primary-threshold Region C complementarity. Phase 2
is scientifically reasonable to design next, but remains unimplemented and
must be predeclared before training.

The coverage-aware machine-readable post-review is generated without changing
the immutable raw metrics at:

```text
results/persistent_semantic/voc/20260829-010948-persistent-semantic-phase01-e222676/analysis/scientific_review.json
```

Its generator is `tools/review_patch_to_class_result.py`. Treat its corrected
conservative flags as the interpretation layer over the original measurements.

### Frozen VOC val confirmatory diagnostic

The confirmatory protocol was committed before the full validation result was
observed. It fixes paper layer 12 (zero-based 11), the six-head mean, strict
20-class `pc_all`, primary threshold `0.5`, and a secondary paper-head-6
diagnostic. It additionally requires superiority to both 5% uniform accuracy
and a paired global-majority-class predictor, plus a 10,000-sample fixed-seed
class-identity permutation test. No model weights or forward computation were
changed.

```text
implementation commit: cea7f9ca8a3400ae795b53a0b56e93b1e1b845ee
run ID: 20260829-134444-persistent-semantic-val-confirmatory-cea7f9c
run dir: results/persistent_semantic/voc/20260829-134444-persistent-semantic-val-confirmatory-cea7f9c
queue log: results/queues/persistent-semantic/20260829-134444-persistent-semantic-val-confirmatory-cea7f9c.log
val_id SHA-256: 6f8edc37993764f6e212237d39546fb595246244147e8a050813c520aac0ade1
images: 1449 complete; 1444 contain foreground after the fixed center crop
elapsed: 192.71 seconds
```

The fixed layer-12 result confirms the intrinsic semantic-attribution finding:

```text
patch-weighted foreground accuracy:       55.094%
foreground-restricted class mIoU:          41.700%
macro image accuracy:                      57.347%
macro image bootstrap 95% CI:              [55.654%, 59.029%]
majority class:                            person
majority macro image accuracy:             16.084%
paired accuracy advantage:                 41.263 points
paired advantage 95% CI:                   [38.797, 43.712] points
class-identity permutation mean / p-value: 5.015% / 0.00010
```

The exploratory train-to-confirmatory-val change is `-3.052` points in macro
image accuracy, `-2.269` points in patch-weighted accuracy, and `-1.806` points
in foreground-restricted mIoU. The macro accuracy stays within the predeclared
five-point retention band. Layer 12 also happens to remain the best validation
layer, but that fact was not used for selection. Secondary head 6 obtains
`61.178%` accuracy and `45.031%` foreground-restricted mIoU.

The primary-threshold Region C result remains a no-go. At `0.5`, no strict
`pc_all` probability passes the threshold, so Region C has zero support and
zero recovery. On the fixed sensitivity grid, threshold `0.25` gives `77.97%`
purity but only `5.72%` recovery; threshold `0.10` gives `45.16%` purity and
`57.06%` recovery. These secondary values confirm diffuse calibration, not the
predeclared foreground decision. The machine-readable decision is therefore
`go` for intrinsic patch-to-class semantics and `no_go` for primary-threshold
Region-C complementarity.

Artifact counts were checked: Phase 0 has `278208` image rows, Phase 1 has
`17388` image-layer rows, and the run contains 2 raw matrix dumps plus 12 sample
and preview dumps. Maximum FP32 attention row-sum error is `9.54e-7`. At fixed
layer 12, mean class-query-to-patch-key mass rises from `34.67%` at 224 to
`45.37%` at 512, while patch-query-to-patch-key mass rises from `43.38%` to
`58.29%`; the validation split therefore reproduces the expected vanilla
resolution/cardinality dependence. Both completion markers and
`scientific_review.json` are present. All 54 repository tests passed before
launch, and there is no active experiment now.

## Completed MCTformer+ token-role pilot

The seed-0 class-token/patch-token specialization pilot completed normally:

```text
queue ID:     20260828-152106
queue log:    results/queues/token-role/20260828-152106.log
result commit: d055da84197d1965d81176efd4785e05357822ba
finished:     2026-08-28T18:22:06+09:00
comparison:   results/mctformerplus/voc/comparisons/token-role-pilot-20260828-152106-s0-d055da8
```

It adapts the ICLR 2026 paper *Revisiting [CLS] and Patch Token Interaction in
Vision Transformers* to MCTformer+'s first 20 class tokens versus its spatial
patch tokens. This is an exploratory prior-art transfer, not a novelty claim.
The historical adaptation and decision rule are preserved in Git commit
`0f9a217` and the immutable result directories. The active-tree protocol file
was removed when this exploration was terminated:

```text
0f9a217 Add MCTformer+ token-role specialization pilots
```

The queue reused the completed E0 seed-0 run as `shared`, recomputed matched
200-image pre/post-`norm1` cosine and efficiency diagnostics for its frozen
checkpoint, then trained these two variants sequentially:

```text
norm      separate class/patch norm1 and norm2 in all 12 blocks
norm_qkv  norm plus separate class QKV in blocks 0--3
```

Both variants retained vanilla attention normalization, BCSS E0, shared MLP and
output projection, 45 epochs, seed 0, input 448, CAM scales
`1.0,0.75,1.25`, and fixed threshold `0.45`. The `norm` and `norm_qkv`
parameter increases are `0.0836%` and `8.1290%`. Specialized paths copy their
corresponding shared DeiT weights, making the modes equivalent at
initialization within numerical tolerance. All 52 repository tests pass.

The fixed-threshold results are:

| Mode | Raw CAM mIoU | Semantic P/R | Background FPR | Final cls mAP | Params | Latency |
|---|---:|---:|---:|---:|---:|---:|
| shared | 70.063 | 80.735/85.817 | 6.756 | 96.410 | 22.051M | 8.431 ms |
| norm | 68.960 | 80.121/84.657 | 7.001 | 96.503 | 22.069M | 8.803 ms |
| norm_qkv | 68.482 | 79.911/84.477 | 7.033 | 96.203 | 23.843M | 8.870 ms |

Paired 10,000-resample image bootstrap confirms that both CAM losses are
negative rather than sampling noise. Relative to shared, the mIoU change is
`-1.104` points with 95% CI `[-1.519,-0.688]` for `norm`, and `-1.581`
with CI `[-2.332,-0.805]` for `norm_qkv`. For `norm`, semantic and binary
precision and recall all decrease with intervals excluding zero, while
background FPR increases by `0.245` point with CI `[0.097,0.394]`.

The 200-image geometry diagnostic does not support a useful LayerNorm-only
mechanism. All-layer class-patch cosine before/after `norm1` is
`0.00745/-0.05755` for shared and `0.00994/-0.05811` for `norm`; their
normalization-induced deltas are nearly unchanged (`-0.06500` versus
`-0.06805`). `norm_qkv` makes the pre-normalization representation more
separated (`-0.03456`) but still degrades localization. Thus greater role
separation is not sufficient for MCTformer+'s multi-class-token WSSS.

This is a no-go for direct transfer of separate LayerNorm/QKV to MCTformer+.
Do not run more seeds, combine it with TGCA/BCSS, or expand it to COCO or an
independent host. Its active implementation, tests, and runners were deleted at
the user's request; the checkpoints, result directories, commits, and negative
conclusion remain available for provenance.

## BCSS VOC screen state on 2026-08-28

### Completed mass-aware E4 debug

The prespecified `e4_mass` follow-up completed normally:

```text
commit:       ade83ad7010820314d62867eb00d01ef3414b832
run ID:       20260828-135358-mctformerplus-voc-bcss-e4_mass-s0-ade83ad
run dir:      results/mctformerplus/voc/20260828-135358-mctformerplus-voc-bcss-e4_mass-s0-ade83ad
started:      2026-08-28T13:54:00+09:00
finished:     2026-08-28T15:17:39+09:00
checkpoint:   c9b8f1f5fdcde60658188ed934bbaaea164c920bace0e4c002336b66cc759cb1
```

The launch manifest records a clean worktree, seed `0`, tau/beta `0.5/0.5`,
lambda-fg `0.5`, fixed CAM threshold `0.45`, vanilla backbone attention, and
`foreground_anchor_mode=ownership_mass_scaled`. All 45 epochs, CAM generation,
fixed-threshold metrics, unified-map dump, and layer/head dump completed. There
is no active tmux session.

The anti-collapse objective worked mechanically but failed scientifically.
Mean foreground ownership remained `0.639` at epoch 44 instead of collapsing
to epsilon, and final classification mAP was `96.365%`. Nevertheless, raw CAM
mIoU was only `55.650`, versus `70.063` for matched E0 and `56.060` for E4.
Semantic precision/recall were `68.032/72.204`; background false-positive rate
rose to `11.463%`, versus `6.756%` for E0. Background ownership predicted only
`28.272%` of patches as background although the validation masks average
`72.769%` background. CBL was `0.5930`, CCS-bg `0.4061`, and background AUPRC
`0.7867`.

Thus retaining total ownership information removes the all-background escape
but does not recover useful competition. It shifts the failure toward excessive
foreground assignment and class/background miscalibration. This is a final
no-go for the current BCSS formulation: do not run more seeds, tune its
loss/tau/beta, combine it with TGCA, or expand it to COCO or another host.
Inspect the completed record with:

```bash
tail -n 120 results/mctformerplus/voc/20260828-135358-mctformerplus-voc-bcss-e4_mass-s0-ade83ad/pipeline.log
```

The diagnostic system and VOC minimum-screen implementation requested by
Section 11 of `docs/CHAT_HANDOFF-0827.md` now exists. Exact variant contracts,
commands, outputs, and verification are recorded in:

```text
docs/BCSS_EXPERIMENTS.md
```

Implemented experiment variants are `E0`, `E1`, `E2`, `E4`, `E5`, and `E6`.
The queue entry point is:

```text
experiments/ablations/run_bcss_voc_screen.sh
```

The full seed-0 VOC screen completed normally at `2026-08-28T03:28:12+09:00`
from commit `4147fc368fdef2698d3a0570c6c5b913527327ee`. All six variants reached 45
epochs, produced final checkpoints and diagnostics, and passed the queue's
completion and manifest checks. The comparison is:

```text
results/mctformerplus/voc/comparisons/
  bcss-screen-20260827-190911-s0-4147fc3/screen.json
```

The screen is a **no-go for the current BCSS formulation**. E6 obtains `56.145`
raw CAM mIoU versus `70.063` for E0, while semantic foreground recall falls
from `85.817` to `66.089`. CBL worsens by `2.41%`; CCS-bg falls by only `10.61%`,
short of the required `20%`; only the classification-retention constraint
passes. E4--E6 assign background ownership above `0.9` to more than `99%` of
patches. The ownership rows are correctly normalized, so this is an objective
degeneracy rather than a softmax or Split implementation error.

The anti-collapse tests, exact E4 pre-loss parity test, complete 44-test suite,
and CUDA FP16 backward passed before launch. The completed run demonstrates why
those implementation checks were necessary but insufficient: loss
identifiability was repaired, while semantic-background learning and CAM
localization remained poor.

This file transfers the scientific context, repository state, server setup, completed evidence, active experiments, and next decisions for the TGCA project. It is intended to be the first file read by a new Codex task on the `LHR` server.

## Resume instructions

On `LHR`, work only in:

```text
/home/peng/code/TGCA
```

Before doing anything else:

```bash
cd /home/peng/code/TGCA
git status --short --branch
git log -8 --oneline --decorate
tmux ls
tail -n 120 results/mctformerplus/voc/20260828-135358-mctformerplus-voc-bcss-e4_mass-s0-ade83ad/pipeline.log
```

Then read these files completely:

1. `docs/CHAT_HANDOFF.md` — this operational handoff;
2. `docs/design.md` — detailed TGCA method and experiment design;
3. `docs/RESEARCH_PLAN_FULL.md` — full paper rationale and go/no-go plan;
4. `docs/TIP_REVIEWS.md` — reviewer objections the new work must resolve.

`docs/MCTTA.pdf` is the rejected legacy manuscript. It is evidence and background, not a draft to compress or edit.

There is no active experiment queue. The completed Persistent Semantic Latent
Phase 0/1 run, token-role pilot, and `e4_mass` run must not be restarted or
duplicated.

## Historical TGCA objective

Prepare a focused ICASSP 2027 Computer Vision paper:

> **Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation**

The paper is not a shortened version of the rejected 16-page MCTTA manuscript. Its single contribution is a measurable failure mode of vanilla attention normalization over heterogeneous token groups and a minimal correction:

```text
s_ij^h = (q_i^h)^T k_j^h / sqrt(d_h)
s_tilde_ij^h = s_ij^h - log(N_{g(j)}) + b_{g(i),g(j)}^h
A_ij^h = softmax_j(s_tilde_ij^h)
```

The central hypothesis is that a joint softmax over a small class-token group and a much larger patch-token group mixes semantic evidence with group cardinality. Increasing patch count, for example through input resolution, may shift aggregate attention mass even when the evidence distribution has not meaningfully changed. TGCA subtracts the log key-group size before one global softmax.

## Non-negotiable scientific contract

TGCA must satisfy all of the following:

- every attention row sums to one;
- duplicating every key/value in one group leaves that group's aggregate mass and attention output unchanged, up to numerical tolerance;
- the count correction has no trainable parameters;
- optional relation bias is only a tiny per-head query-group/key-group matrix;
- the same formulation supports self-attention and rectangular cross-attention;
- value projection, output scaling, residual paths, and all unrelated training choices remain fixed.

The required normalization ablation is:

1. vanilla global softmax;
2. legacy split softmax `(1,1)` — row sum is two by design and this must be stated;
3. normalized split softmax `(0.5,0.5)` — row sum is one but group mass is forced to `0.5/0.5`;
4. TGCA without relation bias;
5. TGCA with relation bias.

Do not reinterpret the two split-softmax baselines as TGCA. They remove group-size effects by assigning fixed group mass, whereas TGCA retains evidence-driven competition between group means.

## Host and baseline roles

- **MCTformer+** is the primary TGCA implementation host.
- **Know Your Attention Maps: Class-specific Token Masking for WSSS** (ICCV 2025), official repository `https://github.com/HSG-AIML/TokenMasking-WSSS`, is the required independent implementation host. Its singleton register token must be handled as an explicit design choice and tested; do not silently merge it into class or patch tokens.
- **DiCLIP** (T-IP 2026), official repository `https://github.com/zwyang6/DiCLIP`, is a recent external comparison only. Do not patch TGCA into DiCLIP for the core generality claim.
- **MoRe** and **CTI** are optional supplementary hosts and currently exist as Git submodules.

Do not return to graph-adapter, CTP-convergence, hierarchical-fusion, pipeline-taxonomy, universal-adapter, or unmatched-SOTA claims. Do not use “Adapter” in the new title. Do not call the old frozen-classifier-plus-separate-segmentation procedure “single-stage.”

## Repository policy

The sole working code repository is:

```text
/home/peng/code/TGCA
```

Create or modify research code only within that directory. Do not recreate root-level `src/`, `experiments/`, or `results/` beside the TGCA repository. Keep legacy manuscript material read-only.

Repository remote:

```text
origin  https://github.com/jungletada/TGCA.git
```

Git branch state at this handoff:

```text
main                            7bd603e [origin/main] Document environments and add independent hosts
research/mctformerplus-baseline active research branch
```

The result-critical token-role implementation and launch manifest are:

```text
0f9a217 Add MCTformer+ token-role specialization pilots
d055da8 Queue MCTformer+ token-role VOC pilots
```

The completed TGCA pilot results were produced from result commit
`22427d60bff5d1f6c4cc9c5c33f8912502d5a4b0`. Commit `596fbc6` adds
post-analysis tooling and documentation without changing that pilot's
result-critical model, training, CAM-generation, or ablation-runner code.

Commit `4147fc3` contains the BCSS implementation used by the completed VOC
screen. Commit `ade83ad` records the negative result summary, mass-aware anchor,
anti-collapse tests, and single-debug runner support. Commit `522be66` records
the completed negative `e4_mass` result. Commit `0f9a217` adds the token-role
implementation, diagnostics, tests, and sequential VOC pilot. Commit `d055da8`
records the frozen queue and is the full commit in both run manifests; the
worktree was clean at launch. No push was performed.

Relevant commits:

```text
d055da8 Queue MCTformer+ token-role VOC pilots
0f9a217 Add MCTformer+ token-role specialization pilots
522be66 Record mass-aware BCSS debug result
ade83ad Add mass-aware BCSS foreground anchor debug
4147fc3 Add BCSS support to MCTformerPlus and update training and evaluation processes
596fbc6 Refactor VOC normalization process and enhance diagnostics for TGCA integration
22427d6 Implement TGCA diagnostics and VOC normalization pilots
cf44aa2 Make CRF optional for raw CAM generation
63d8877 Add reproducible MCTformer+ VOC baseline pipeline
d6a4a90 Fix vanilla MCTformer+ baseline plumbing
7bd603e Document environments and add independent hosts
```

Independent host submodules:

```text
hosts/CTI   1c6fdb4d14e6843e3d861ebd4580468e30598859
hosts/MoRe  d733d347e64f21425df245341e1f88900886b2bb
```

## Server and environment

SSH alias:

```bash
ssh LHR
```

Server project path:

```text
/home/peng/code/TGCA
```

Primary Conda environment:

```text
tgca-repro
Python 3.9.25
torch 2.1.0
torchvision 0.16.0
timm 0.4.12
CUDA runtime 11.8
```

Observed GPU for current runs:

```text
NVIDIA RTX A6000, 49140 MiB
driver 580.173.02
```

Activate the environment with:

```bash
source /home/peng/anaconda3/etc/profile.d/conda.sh
conda activate tgca-repro
cd /home/peng/code/TGCA
```

MoRe, CTI, and TGCA require separate Conda environments because their dependency stacks conflict. Do not merge them merely for convenience.

Codex CLI is installed at `/usr/local/bin/codex`, but it did not start at this snapshot because the server's default Node.js was `v12.22.9` and could not parse the installed CLI. Install/select a current Node.js LTS using an available environment or version manager, reinstall/update `@openai/codex`, and authenticate. Do not copy an entire `~/.codex` directory from another machine because it can contain credentials and machine-specific state.

## Dataset and pretrained assets

The data links are available:

```text
data/VOCdevkit -> /home/peng/data/VOCdevkit
data/MSCOCO    -> /home/peng/data/MSCOCO
```

The development dataset is PASCAL VOC 2012. Current baseline manifests record:

- training: VOC augmented training split;
- CAM generation and raw-CAM evaluation: 1,464 VOC train images;
- validation classification: 1,449 images;
- input training size: `448`;
- CAM scales: `1.0, 0.75, 1.25`;
- ImageNet/DeiT initialization: `$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth`.

Every result directory records dataset hashes, pretrained-weight hash, package freeze, Conda explicit package list, GPU, Git commit, command, seed, and checkpoint hash. Preserve that provenance contract.

Do not start COCO until the VOC mechanism and independent-host go/no-go gates pass.

## Implemented TGCA code

Commit `22427d6` added the initial implementation and experiment plumbing:

- `models/tgca.py` — shared `TokenGroupNormalizer` and functional normalization;
- `models/vit.py` — MCTformer+ attention integration;
- `train_model_v2.py` and `make_cam.py` — normalization-mode arguments;
- `tools/analyze_attention_groups.py` — per-image/layer/head/direction group-mass diagnostics at multiple resolutions;
- `tools/test_token_replication.py` — synthetic cardinality and replication experiment;
- `tests/test_tgca_normalization.py` — row sums, masks, relation bias, gradients, mixed precision;
- `tests/test_tgca_replication.py` — group-mass and output replication invariance;
- `tests/test_mctformerplus_attention.py` — host integration checks;
- `experiments/diagnostics/` — reproducible mechanism runners;
- `experiments/ablations/` — reproducible normalization pilots;
- `experiments/run_mctformerplus_next_experiments.sh` — completed historical TGCA queue runner.

Supported mode names in the current code are:

```text
vanilla
split_11
split_05
tgca
tgca_bias
tgca_gamma05
```

The five-mode core suite uses the first five except `tgca_gamma05`.

The legacy `split_11` operator deliberately has row sum two. The diagnostic tool therefore computes its reported “maximum row-sum error” relative to two, not relative to one. Never cite that metric as proof that `split_11` is normalized to unit mass.

## Trustworthy vanilla MCTformer+ baseline

The first completed reproduction is:

```text
run:       results/mctformerplus/voc/20260825-mctformerplus-voc-vanilla-s0-63d8877
dataset:   PASCAL VOC 2012 train, 1,464 images
metric:    raw CAM mIoU, no CRF
threshold: 0.45
result:    69.50%
seed:      0
epochs:    45
input:     448
scales:    1.0, 0.75, 1.25
```

The exact command is:

```bash
TGCA_GPU_ID=0 \
TGCA_RUN_ID=20260825-mctformerplus-voc-vanilla-s0-63d8877 \
bash experiments/baselines/run_mctformerplus_voc.sh
```

The final checkpoint SHA-256 is:

```text
f63dc438a2b2b2650aec5d7c4a0c2c2a92780893466138ed2b71bdc70be43cd7
```

The training commit was `63d8877`; CAM evaluation used `cf44aa2`; the referenced official MCTformer commit was `0acc27ada87a5582053efb14648442d8644168aa`.

Important interpretation: threshold `0.45` was selected by the existing `train_model_v2.py` tuning workflow. Keep the script, but label `0.45` as the vanilla-seed-0 selected threshold. All current normalization pilots use this same fixed threshold so that each method does not tune its own evaluation threshold.

The integrated vanilla run at commit `22427d6` reproduced the same result exactly:

```text
run:    results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-s0-22427d6
result: 69.50% raw CAM mIoU at fixed threshold 0.45
```

This is the main regression check that the instrumentation and mode plumbing preserve vanilla behavior.

## Completed mechanism evidence

### Synthetic token-replication test

Run:

```text
results/mechanism/synthetic/20260826-synthetic-replication-22427d6
```

Command:

```bash
bash experiments/diagnostics/run_synthetic_replication.sh
```

The grid contains 960 configurations over class counts `[1,20,80]`, head counts `[1,6]`, patch counts `[49,196,400,784,1024]`, four logit regimes, and replication factors `[1,2,4,8]`, with seed `2027`.

Observed maximum output change after within-group replication:

```text
vanilla: 0.4615142941
TGCA:    0.0000215769
```

Maximum TGCA row-sum error in this mixed-precision diagnostic:

```text
0.0000165701
```

This strongly confirms the mathematical replication property in the synthetic setting. It does not by itself establish better CAM quality.

### Full VOC resolution/group-mass diagnostic for vanilla

Run:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-scale-diagnostic-22427d6
```

It processed all 1,464 VOC train images at resolutions `224, 320, 448, 512`, corresponding to patch counts `196, 400, 784, 1024`.

Key results:

```text
maximum row-sum error:       1.430511474609375e-06
mean group-mass variance:    0.0046481917763
median group-mass variance:  0.0018502148937
```

Directional mean slopes versus log patch count, all layers:

```text
class query -> class key: -0.0994545
class query -> patch key: +0.0994545
patch query -> class key: -0.0539273
patch query -> patch key: +0.0539273
```

All four bootstrap 95% confidence intervals exclude zero by a wide margin. In the last three layers, the corresponding magnitudes are approximately `0.09546` for class queries and `0.08179` for patch queries.

Do not use the aggregate `mean_group_mass_slope_vs_log_patch_count` value, which is approximately zero because complementary class-key and patch-key directions cancel under indiscriminate averaging. The directional slopes are the interpretable mechanism result.

This is direct empirical evidence that vanilla attention reallocates aggregate mass toward patch keys as patch count/resolution rises. It still does not prove causality for CAM quality; the normalization ablation must establish that link.

## Completed primary experiment queue

Session:

```text
tgca-mctplus-next
```

Pane start command:

```bash
bash -lc 'source /home/peng/anaconda3/etc/profile.d/conda.sh && \
conda activate tgca-repro && \
cd /home/peng/code/TGCA && \
set -o pipefail && \
bash experiments/run_mctformerplus_next_experiments.sh 2>&1 | \
tee results/queues/22427d6/mctformerplus-next.log'
```

Queue log:

```text
results/queues/22427d6/mctformerplus-next.log
```

The queue performs:

1. a full vanilla resolution diagnostic;
2. five sequential 45-epoch normalization runs:
   `vanilla`, `split_11`, `split_05`, `tgca`, `tgca_bias`;
3. raw CAM generation and fixed-threshold evaluation after each run;
4. a full four-resolution attention diagnostic after each run.

Final status, completed **2026-08-27 01:19 JST**:

- vanilla: complete, `69.50%` raw CAM mIoU;
- split `(1,1)`: complete, `58.22%` raw CAM mIoU;
- split `(0.5,0.5)`: complete, `66.31%` raw CAM mIoU;
- count-only TGCA: complete, `69.05%` raw CAM mIoU;
- TGCA with relation bias: complete, `68.88%` raw CAM mIoU.

The completed split `(1,1)` run is:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-split_11-s0-22427d6
```

Its fixed-threshold raw CAM score is `58.22%`, substantially below vanilla. Its group-mass variance is essentially zero because this baseline forces each group to aggregate to mass one; total row mass is two. This result does not refute TGCA because `split_11` changes output scale and removes evidence-driven inter-group competition.

The completed TGCA-with-relation-bias run is:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-tgca_bias-s0-22427d6
```

Each mode uses:

```bash
TGCA_GPU_ID=0 TGCA_SEED=0 TGCA_FIXED_THRESHOLD=0.45 \
bash experiments/ablations/run_mctformerplus_voc_mode.sh MODE
```

Do not launch a duplicate primary queue. All five run directories contain `metrics.json`, the final checkpoint, CAM evaluation outputs, and `attention_diagnostics/metrics.json`.

## Completed post-pilot diagnostics

The original waiting session observed the primary `QUEUE_COMPLETE` marker and then exited at **2026-08-27 01:19 JST** because `HEAD` had advanced from result commit `22427d6` to analysis commit `596fbc6`. This was the intended provenance guard, not an experiment failure. The intervening commit added documentation and post-analysis tools only; `models/`, training, CAM generation, and the ablation runner are unchanged between the commits.

A provenance-explicit replacement session started at **2026-08-27 11:20 JST** and completed at **11:45 JST**:

```text
tgca-mctplus-post
```

It runs:

```bash
bash experiments/diagnostics/run_mctformerplus_post_pilot_resume.sh
```

and logs to:

```text
results/queues/22427d6/mctformerplus-post-pilot.log
```

The replacement records result commit `22427d6` and analysis commit `596fbc6` separately and verifies that result-critical code did not change. Its CUDA test stage passed `19/19`. There is no active tmux session after normal completion.

After the marker appears, it sequentially performs:

1. CUDA FP32/FP16/BF16 normalization and host tests on exact commit `22427d6`;
2. fixed-threshold raw-CAM precision, recall, and false-positive diagnostics for all five modes;
3. matched batch-1 inference latency, throughput, parameter, and peak-memory measurements;
4. flip-aggregated CAM generation at short-side resolutions `224`, `320`, `448`, and `512` for every mode;
5. fixed-threshold CAM quality at every resolution;
6. soft CAM cosine, class-mask IoU, foreground IoU, semantic-mask IoU, and pixel agreement relative to `448`;
7. an audited five-mode JSON/CSV comparison under:

```text
results/mctformerplus/voc/comparisons/pilot-s0-22427d6
```

The post-pilot queue stops after producing Gate 3 evidence. It does not automatically launch additional seeds, partial-gamma training, KYAM, or COCO. Those depend on reviewing the completed pilot. At queue creation, the filesystem had approximately `170 GiB` available; the four-resolution CAM outputs are expected to add roughly `30 GiB`.

### Post-pilot result summary

All five modes use seed `0`, result commit `22427d6`, and fixed background threshold `0.45`:

| Mode | Raw CAM mIoU | Semantic precision | Semantic recall | Background FPR | Mean mass variance | Latency at 448 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 69.50 | 81.85 | 83.87 | 6.04 | 0.004648 | 8.48 ms |
| split `(1,1)` | 58.22 | 61.72 | 76.58 | 16.36 | approximately 0 | 18.76 ms |
| split `(0.5,0.5)` | 66.31 | 82.33 | 78.22 | 5.38 | approximately 0 | 18.79 ms |
| TGCA | 69.05 | 78.73 | 86.40 | 7.65 | 0.000837 | 11.12 ms |
| TGCA + bias | 68.88 | 79.37 | 85.13 | 7.21 | 0.000837 | 12.35 ms |

TGCA reduces mean attention group-mass variance by approximately `82%`. Relative to vanilla, its all-layer patch-key mass slope magnitude falls by approximately `89%` for class queries and `75%` for patch queries. Thus the cardinality mechanism and correction are measurable.

The localization result does not pass the primary gain target. Count-only TGCA is `0.45` mIoU point below vanilla and TGCA+bias is `0.62` below. Both are below vanilla at every single-scale short-side resolution (`224`, `320`, `448`, `512`). TGCA raises recall but lowers precision and increases background false positives at the fixed threshold.

Cross-scale mask stability improves modestly at the largest resolution gap. For `224` versus `448`, paired image bootstrap analysis gives foreground-mask IoU differences of `+1.44` points for TGCA (95% CI `+0.81` to `+2.06`) and `+1.57` for TGCA+bias (`+0.99` to `+2.16`). At `512` versus `448`, the differences are small and the paired intervals include zero.

The current Python/group-mask implementation also does not yet satisfy the negligible-overhead gate: count-only TGCA latency is about `31%` above vanilla and TGCA+bias about `46%` above. Parameter count is unchanged for count-only TGCA; relation bias adds `288` parameters. FLOPs/MACs remain unmeasured.

Machine-readable comparison:

```text
results/mctformerplus/voc/comparisons/pilot-s0-22427d6/pilot_comparison.json
results/mctformerplus/voc/comparisons/pilot-s0-22427d6/pilot_comparison.csv
```

## How to evaluate the completed queue

For every mode, make a machine-readable comparison containing at least:

- raw CAM mIoU at the fixed vanilla threshold `0.45`;
- final and maximum classification mAP if recorded;
- checkpoint hash;
- maximum attention row-sum error, with the split `(1,1)` target explicitly recorded as two;
- directional group-mass slopes and their bootstrap intervals;
- mean/median group-mass variance over resolution;
- wall time, peak memory, parameters, and latency where available.

Then run these validity checks before drawing conclusions:

1. Vanilla in the integrated pipeline remains `69.50%`.
2. `split_05`, TGCA, and TGCA-bias rows sum to one numerically.
3. `split_11` rows sum to two and are never compared as if normalized.
4. TGCA directional mass is more stable than vanilla under real resolution changes.
5. TGCA retains evidence-driven group-mass variation rather than degenerating to the fixed `0.5/0.5` behavior of `split_05`.
6. Any CAM gain is not explained by a separately tuned background threshold, output rescaling, changed training schedule, or different checkpoint selection.
7. Runtime overhead is measured. The current Python/group-mask implementation may be inefficient and should not be treated as the final efficiency result without profiling and, if needed, a mathematically identical vectorized implementation.

Do not select a favorable method from only seed 0 and present it as final. After the pilot identifies viable modes, repeat the matched comparison with prespecified additional seeds and report variance.

## Immediate next sequence

1. Treat the minimal Late-1 shared-relation Phase 2 screen as a no-go for expansion.
2. Do not start more seeds, Late-3/depth, independent-relation, width, dynamic-initialization, multi-latent, OT, or extra-background-loss experiments from this result.
3. If the research direction is retained, first run a frozen-checkpoint diagnostic that separates patch-feature quality, Read-map quality, relation-scale collapse, and background allocation without retraining.
4. Treat Phase 1 as positive for intrinsic strict patch-to-class semantic attribution, not as proof that the new Phase 2 relation preserves that information.
5. Do not select a separate favorable threshold per layer or method to rescue the negative fixed-threshold result.

## Risks that can invalidate or weaken the hypothesis

- **Trained logits may compensate for token count.** Synthetic replication invariance can be mathematically true while the learned network already counteracts cardinality in practice.
- **Resolution is confounded.** Changing resolution changes image evidence, receptive fields, interpolation, positional embeddings, and object scale—not just patch count. Synthetic replication and real-scale tests must be reported together.
- **Stability may not improve localization.** More stable attention mass is useful only if CAM quality, cross-scale consistency, or errors improve.
- **Fixed group mass may be sufficient.** If `split_05` matches TGCA, the evidence-driven mean-competition story becomes weaker and the contribution may reduce to a normalization heuristic.
- **Output-scale confounding.** `split_11` doubles row mass and therefore changes residual/output magnitude. Its poor CAM result cannot be attributed solely to group competition.
- **Threshold sensitivity.** A gain appearing only after per-method threshold tuning is weak. Fixed-threshold results are primary; threshold curves are diagnostic.
- **Implementation overhead.** A slow mask/one-hot implementation could invalidate the negligible-overhead claim even when an optimized equivalent is possible.
- **Relation bias may dominate.** If gains occur only with learned bias, distinguish count calibration from added capacity using TGCA without bias and parameter-matched controls.
- **Single-host overfitting.** A positive MCTformer+ result without a positive independent Know Your Attention Maps result is insufficient for the generality claim.
- **Register-token ambiguity.** In Know Your Attention Maps, treating the singleton register token as class, patch, or a separate group changes the method. All choices must be explicit and ablated.
- **Training instability or seed variance.** Seed-0 improvements can disappear. Confirm with multiple matched seeds after the pilot.
- **Metric cancellation.** Averaging complementary attention directions can falsely imply zero cardinality effect. Retain directional layer/head statistics.
- **Post-processing confounds.** Keep raw CAM evaluation primary and hold downstream CRF/segmentation pipelines fixed.
- **Unmatched comparisons.** DiCLIP and other recent systems may use different pretraining, backbones, supervision, or post-processing. Author-reported and reproduced results must be separated.

## Paper decision gates

Proceed to the full paper only if all are satisfied:

- the directional cardinality phenomenon is measurable;
- TGCA stabilizes group mass under patch-count/resolution changes;
- TGCA improves or stabilizes CAMs beyond simple output rescaling and fixed group mass;
- at least one independent host shows a positive effect;
- optimized overhead is negligible;
- all claims are matched for backbone, pretraining, supervision, and post-processing.

If TGCA improves invariance but consistently damages CAM quality, do not force a positive paper claim. Determine whether cardinality bias is actually a useful inductive bias for WSSS, whether partial correction `gamma < 1` is scientifically justified, or whether the hypothesis should be rejected.

## Provenance and reporting rules

Never report a result without:

- repository URL and commit;
- exact command and config;
- Conda/pip environment;
- dataset and list hashes;
- pretrained and trained checkpoint hashes;
- seed;
- raw machine-readable metrics;
- explicit threshold and post-processing;
- hardware and timing.

Generated metrics belong under `results/` as JSON/CSV in addition to paper tables. Use Git checkpoints before and after substantive code changes. Do not fabricate missing values, citations, repository behavior, or experimental outcomes.

## Documentation inventory

The complete documentation copied for this handoff is:

```text
docs/CHAT_HANDOFF.md
docs/MCTTA.pdf
docs/RESEARCH_PLAN_FULL.md
docs/TIP_REVIEWS.md
```

The server repository already contains the tracked file:

```text
docs/design.md
```

At the time of transfer, the local and server copies of `design.md` had matching SHA-256:

```text
4086ac6250aa7acd91d928f4f67b05d5139898dc6637792732d492d52671f5e0
```
