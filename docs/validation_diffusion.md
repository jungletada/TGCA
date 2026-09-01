# Minimal Validation Plan for Diffusion and Flow-Matching WSSS

**Plan date:** 2026-08-30

**Status:** pre-implementation experiment design. No experiment in this
document has been launched, completed, or authorized by the existence of this
file.

**Related survey:** [survey_diffusion.md](survey_diffusion.md)

## 1. Decision objective

Before rebuilding the repository around a diffusion or Flow Matching WSSS
framework, answer three sequential questions:

1. **Teacher signal:** Does a frozen diffusion model provide localization
   evidence that is useful and complementary to strong CAM/CLIP teachers on
   the same VOC images?
2. **Generative objective:** With the teacher signals, pseudo-targets,
   conditioning features, model capacity, training data, and evaluation fixed,
   does diffusion or Flow Matching outperform an ordinary deterministic
   refiner?
3. **Uncertain endpoint:** Does a structured distribution of multi-teacher,
   multi-view weak masks provide a better Flow Matching endpoint than one hard
   pseudo-mask, beyond ordinary soft-label learning and unstructured-noise
   controls?

These are hierarchical gates. Do not implement or run Experiment 2 unless
Experiment 1 passes. Do not run Experiment 3 unless a generative variant passes
Experiment 2. A failed gate is a scientific stopping result, not permission to
tune thresholds, prompts, teachers, losses, or architectures until one becomes
positive.

## 2. Hypotheses

### H1: independent diffusion evidence

On fixed VOC images, at least one frozen diffusion signal has non-random class
localization and makes errors that are sufficiently different from the current
MCTformer+ and CLIP signals that a prespecified, non-learned fusion improves
expected localization.

### H2: value of the generative training objective

Under identical weak targets and conditions, a diffusion or Flow Matching
refiner improves fixed-rule validation prediction over both a deterministic
refiner and a noise-augmented deterministic denoiser. A gain from additional
parameters, training updates, teachers, or test-time samples does not pass H2.

### H3: value of structured uncertainty

A flow trained against complete-mask samples from a calibrated multi-teacher,
multi-view endpoint distribution improves expected segmentation and
uncertainty calibration over:

- the same flow trained against one hard endpoint;
- deterministic learning from the endpoint mean;
- a confidence-matched but spatially unstructured uncertainty control.

## 3. Fixed scope and non-goals

### 3.1 Development dataset

Use PASCAL VOC 2012 only.

- `train_aug`: source of image-level-supervised training images and cached weak
  teacher outputs for learned refiners.
- `train_id`: exploratory audit split, following the repository's existing
  1,464-image raw-CAM protocol.
- `val_id`: frozen confirmatory split of 1,449 images.
- VOC test and MS COCO are prohibited before all three gates pass.

Training code must not read VOC segmentation masks. Pixel masks may be opened
only by standalone evaluation processes after checkpoints and fixed inference
rules have been written to the run manifest.

### 3.2 Primary task

The primary task is 21-state semantic seed prediction:

```text
20 VOC foreground classes + 1 explicit background class
```

The headline result is raw seed/CAM quality before CRF, IRN, PSA, SAM, or a
separately trained downstream segmentation network. A single fixed downstream
pipeline is permitted only after a validation gate passes.

### 3.3 Excluded expansions

The minimal validation does not include:

- COCO training;
- open-vocabulary evaluation;
- multiple diffusion backbones selected after seeing VOC masks;
- prompt search using pixel metrics;
- class-specific thresholds;
- per-method background thresholds;
- SAM-based correction;
- CRF/IRN/PSA rescue experiments;
- best-of-K sample selection using ground truth;
- a new MCTformer+/TGCA/BCSS/persistent-latent hybrid.

## 4. Frozen baseline and provenance anchors

The existing discriminative anchor is the matched MCTformer+ E0 seed-0 run:

```text
checkpoint:
  results/mctformerplus/voc/
  20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3/
  mctformerplus_final.pth
sha256:
  41ac9ce47f6a22875cba32edb92c31c150e804ae5ae19824c2585e4e3cda7a2a
raw CAM mIoU:
  70.06305830908285 at fixed background threshold 0.45
training input:
  448
CAM scales:
  1.0, 0.75, 1.25
```

This baseline is an evaluation anchor, not a requirement that the new refiner
reuse MCTformer+'s architecture. Every imported teacher must record:

```text
paper and official repository URL
repository commit
model/checkpoint identifier and SHA-256
model license and code license
environment manifest
pretraining data and supervision
prompt set
diffusion timesteps and layers
native preprocessing and output normalization
exact extraction command
```

## 5. Environment and execution policy

Create a separate Conda environment, provisionally named:

```text
diffusion-wsss-repro
```

Do not upgrade or merge packages into `tgca-repro`. Official DiCLIP,
DiffSegmenter/iSeg, and any selected flow implementation must first be tested
in isolated environments if their dependency requirements conflict. Freeze a
Conda explicit manifest and `pip freeze` before the first result-producing run.

Long runs use unique tmux queues and immutable result directories. Before each
launch:

1. verify clean tracked result-critical state;
2. check `tmux ls`, GPU processes, free GPU memory, and disk space;
3. refuse a duplicate run ID or existing output directory;
4. save the command, config, Git state, environment, dataset hashes, and
   checkpoint hashes;
5. run independent GPU-heavy stages sequentially.

The current completed experiments must not be stopped, restarted, or
overwritten to make room for this work.

## 6. Common representation and evaluation contract

### 6.1 Semantic score representation

Every teacher output is converted into a categorical probability tensor:

```text
q(x) in [0, 1]^(21 x H x W)
sum_c q_c(x, u) = 1 for every pixel u
```

Rules:

- Classes absent from the image-level label vector receive exactly zero
  foreground probability before the final categorical normalization.
- Background is an explicit state. It must not disappear because foreground
  classes are normalized independently.
- The conversion from a native CAM/attention map to categorical probabilities
  is specified before confirmatory metrics are observed.
- Numerical normalization uses float32 and preserves finite, unit-sum rows.
- Maps are retained both at native resolution and in original-image
  coordinates.

### 6.2 Calibration without pixel-label leakage

No per-pixel VOC ground truth may calibrate a teacher used for training. Any
teacher temperature is fit only through image-level classification likelihood
or fixed to the official value. The main evaluation uses one common conversion
and the existing fixed background decision `0.45` after the documented CAM
normalization.

A common threshold grid may be reported as a diagnostic:

```text
0.05, 0.10, 0.15, ..., 0.75
```

The grid must not select a separate headline threshold for each teacher or
model. Threshold-free metrics are required so the decision does not depend on
one operating point.

### 6.3 Common image views

Teacher audit views are fixed to:

```text
short-side resolutions: 224, 320, 448, 512
horizontal flip:         off and on
crop/padding rule:       one shared deterministic implementation
reference coordinates:  original VOC image
```

If a diffusion implementation accepts only a native 512 canvas, record the
letterbox/crop mapping and map its output back to the same original-image
coordinates. Do not silently compare a center crop with a full-image CAM.

### 6.4 Primary metrics

For every model and fixed inference rule, save:

- 21-class raw seed mIoU;
- per-class IoU;
- semantic foreground precision and recall;
- binary foreground precision and recall;
- background false-positive rate;
- class-conflict rate: pixels assigned to a wrong present foreground class;
- absent-class activation mass before image-label masking, as a diagnostic;
- foreground and per-class pixel average precision;
- threshold-free soft intersection/union and Brier score;
- cross-scale soft cosine and fixed-threshold semantic-mask IoU;
- image-level classification mAP where the model contains a classifier;
- parameters, peak memory, training time, and inference latency.

### 6.5 Statistics

- Use paired image-level bootstrap confidence intervals with 10,000 resamples
  and fixed bootstrap seed `2027`.
- Treat images, not pixels or stochastic samples, as independent units.
- Report effect size and 95% confidence interval for every primary comparison.
- The hierarchical gates define one primary comparison at each stage; secondary
  teacher, scale, class, and sample-count analyses are descriptive.
- A seed-0 training screen can reject a variant. It cannot establish a final
  positive claim. Passing learned variants must be confirmed with seeds
  `0, 1, 2` before a paper claim.

## 7. Experiment 1: frozen teacher and complementarity audit

### 7.1 Scientific question

Does a pretrained diffusion model contribute spatial/class evidence that is
not already captured by the existing MCTformer+ CAM or a strong frozen CLIP
teacher?

### 7.2 No-training candidate signals

Freeze the following signal registry before processing full VOC splits:

| ID | Signal | Intended implementation | External information |
|---|---|---|---|
| `T0_mct` | Current MCTformer+ E0 multi-scale CAM | Existing pinned checkpoint and CAM pipeline | ImageNet/DeiT pretraining |
| `T1_clip` | CLIP-only class localization | CLIP branch from one pinned official WSSS implementation, preferably the DiCLIP baseline path | CLIP image-text pretraining and fixed VOC prompts |
| `T2_sd_attn` | Diffusion cross-attention mask | One pinned DiffSegmenter or iSeg extraction path | Stable Diffusion and fixed VOC prompts |
| `T3_sd_corr` | Diffusion feature/correlation map | Frozen diffusion-only correlation path exposed from pinned DiCLIP/DiG-compatible code | Diffusion pretraining; no CLIP feature fused into this row |

`T2_sd_attn` and `T3_sd_corr` must remain separate. The first tests explicit
text-to-spatial grounding; the second tests spatial correlation/locality in
diffusion features. If an official implementation cannot expose a pure branch
without learned CLIP fusion, mark that row unavailable rather than inventing
an ad hoc extractor after seeing results.

### 7.3 Fixed prompts

Use one prompt policy for all text-conditioned teachers:

```text
primary:   "a photo of a {voc_class}"
synonyms:  one checked VOC class-name mapping only where the official class
           name is not natural English, fixed before inference
negative:  none in the primary audit
```

An official prompt ensemble may be reported as a secondary reproduction, but
it cannot replace the primary prompt after metrics are inspected.

### 7.4 Fusion controls

Evaluate the individual teachers and these prespecified, parameter-free rules:

| ID | Fusion rule | Purpose |
|---|---|---|
| `F0_best_single` | Best single teacher selected on exploratory `train_id` and frozen for `val_id` | Strong individual anchor |
| `F1_mean` | Uniform arithmetic mean of categorical teacher probabilities | Simple ensemble control |
| `F2_entropy` | Per-pixel inverse-normalized-entropy weighting, formula frozen before `val_id` | ComCD-like confidence control |
| `F3_mct_diff` | Fixed 0.5/0.5 mean of `T0_mct` and the strongest available pure diffusion row | Direct complementarity test |

Do not train a fusion network in Experiment 1. Do not select weights on
`val_id`. If one teacher fails mechanically or is unavailable, keep the
remaining registry and document the omission before seeing its scientific
metric.

### 7.5 Complementarity diagnostics

For every teacher pair and image, report:

- pixel error correlation;
- foreground true-positive intersection and exclusive regions;
- Region A: both correct;
- Region B: first teacher only correct;
- Region C: second teacher only correct;
- Region D: neither correct;
- Region C support, target-class purity, and recoverable foreground fraction;
- oracle union mIoU as a non-deployable ceiling;
- actual fixed-rule fusion mIoU.

The oracle union is diagnostic only. It cannot pass the gate without a
deployable fixed fusion improvement.

### 7.6 Execution order

1. Run a 20-image mechanical smoke with no scientific interpretation.
2. Freeze checkpoint hashes, prompts, timesteps/layers, preprocessing,
   categorical conversion, and fusion formulas.
3. Run the complete 1,464-image `train_id` audit.
4. Select only `F0_best_single` and one strongest diffusion row according to
   the predeclared primary metric.
5. Write the selection and exact configuration to an immutable JSON decision.
6. Run the complete 1,449-image `val_id` confirmation once.

### 7.7 Experiment 1 pass criteria

Pass H1 only if the frozen `val_id` result satisfies all of the following:

1. The selected pure diffusion signal has class-aware foreground pixel AP
   above the uniform/present-class reference and nonzero IoU for at least 15 of
   20 foreground classes.
2. Its exclusive-correct Region C is non-empty in at least
   `max(30, 5% of eligible images)`, with target-class purity above the
   present-class random reference and at least 1% foreground recovery.
3. One prespecified parameter-free fusion improves raw seed mIoU over its best
   constituent by at least `0.75` point, with the paired 95% bootstrap interval
   lower bound above zero.
4. The fusion does not increase background false-positive rate by more than
   `0.5` point or reduce semantic precision by more than `1.0` point.

Failure means diffusion may still be a usable standalone external baseline,
but there is no evidence to build a new generative WSSS framework around its
complementarity. Stop before Experiment 2.

### 7.8 Artifacts

```text
results/diffusion_wsss/voc/teacher_audit/<run_id>/
  config.yaml
  command.txt
  git_state.json
  environment.txt
  dataset_manifest.json
  teacher_manifest.json
  per_image_metrics.csv
  per_class_metrics.csv
  complementarity.csv
  scale_consistency.csv
  metrics.json
  decision.json
  completion.json
```

## 8. Experiment 2: deterministic versus generative refinement

### 8.1 Scientific question

When input signal, pseudo-target, condition, architecture size, training data,
updates, and evaluation are fixed, is a generative denoising/transport
objective better than ordinary deterministic refinement?

### 8.2 Frozen data products

Use the passing Experiment 1 teachers to generate and cache, for every
`train_aug` image:

```text
q_source:  strongest single weak teacher at the canonical view
q_target:  fixed parameter-free fused categorical probability map
features:  frozen image-conditioning features
labels:    image-level 20-class vector
```

The target is not ground truth. It must be named `weak_target` throughout code
and manifests. Cache hashes are part of every training run. Regenerating the
cache after model metrics are observed creates a new experiment version.

### 8.3 Common spatial and semantic representation

The minimal screen operates at the DeiT patch grid rather than full image
resolution:

```text
canonical crop:       448 x 448
mask/logit grid:      28 x 28
semantic channels:   21
image condition:     frozen 384-channel E0 patch feature or one predeclared
                     shared feature tensor
```

This isolates semantic refinement and makes matched generative training
tractable. It is not a boundary-complete segmentation system. Full-resolution
decoding is forbidden until the patch-grid gate passes.

Convert probabilities to a shared centered logit representation with one
fixed epsilon. The same representation and inverse transform are used by all
learned variants.

### 8.4 Shared refiner architecture

Implement one time-conditioned convolutional refiner whose parameterized core
is shared across objectives:

```text
inputs:
  corrupted/current 21-channel mask state
  q_source
  frozen image feature projection
  broadcast image-level labels
  scalar time embedding
core:
  fixed-width residual U-Net/refiner at 28 x 28
output:
  21-channel logits, noise/velocity, according to objective
```

Architecture width, depth, normalization, activation, condition projection,
and parameter budget are identical. Objective-specific output semantics are
allowed; adding extra attention blocks or a larger encoder to only one variant
is not.

### 8.5 Required variants

| ID | Objective | Inference steps | Purpose |
|---|---|---:|---|
| `R0_source` | No learned refinement | 0 | Strongest individual teacher |
| `R1_target` | No learned refinement | 0 | Fixed fused weak-target ceiling/anchor |
| `R2_det` | Deterministic KL/CE to `q_target` | 1 | Ordinary soft-label refiner |
| `R3_det_noise` | Deterministic denoising with the same sampled corruption levels used by generative variants | 1 | Noise-augmentation control |
| `R4_diffusion` | Conditional Gaussian/logit diffusion with fixed prediction parameterization | 20 primary; 4-step diagnostic | Diffusion objective |
| `R5_flow` | Conditional straight-path Flow Matching/Rectified Flow in the same logit space | 4 primary; 1-step diagnostic | Flow objective |

`R4` and `R5` receive the same endpoint `q_target`. Experiment 2 deliberately
does not use a distribution of teacher masks; otherwise teacher uncertainty and
the generative objective would be confounded.

### 8.6 Matched training recipe

Freeze before seed-0 launch:

- initialization seed;
- optimizer and learning-rate schedule;
- batch size and gradient accumulation;
- total optimizer updates;
- augmentations;
- EMA policy;
- weight decay and gradient clipping;
- image-condition checkpoint;
- weak-target cache;
- checkpoint selection rule based only on weak training/held-out image-level
  objectives, never VOC pixel masks.

Every variant sees the same image order and number of optimizer updates. Record
both matched-update results and actual wall-clock cost. Do not grant generative
variants more training data or teacher views.

### 8.7 Fixed inference rule

- `R2` and `R3`: one forward pass.
- `R4`: probability mean over four fixed stochastic samples, 20 steps each.
- `R5`: probability mean over four fixed stochastic samples, four ODE steps
  each.
- Sampling seeds are fixed per VOC image in the manifest.
- Report single-sample and step-count sensitivity as secondary diagnostics.
- Never choose the best sample using ground truth or confidence after seeing
  validation results.

Because sample averaging adds compute, also compare generative mean prediction
against a four-member deterministic ensemble trained with the same aggregate
number of seeds/updates if `R4` or `R5` appears positive. That ensemble is
required before confirming H2.

### 8.8 Mechanical tests before training

1. All probability outputs are finite and sum to one.
2. Absent foreground classes receive exactly zero final probability.
3. Background remains available for every image and cannot be masked out.
4. Constant all-background and all-foreground synthetic predictions incur
   nonzero loss under mixed foreground/background fixtures.
5. Diffusion and flow forward/backward gradients are finite in FP32 and AMP.
6. At time endpoints, interpolation/noising equations match their analytical
   values.
7. With a zero vector field, the ODE solver produces the expected identity.
8. Fixed seeds reproduce identical cached corruptions and inference samples.
9. The training dataset object raises an error if asked to open a VOC
   segmentation-mask path.
10. Parameter-count differences stay within 1% across `R2` through `R5`, or a
    parameter-matched projection is added before any scientific run.

### 8.9 Execution order

1. One-batch CPU and CUDA tests.
2. One-epoch smoke for `R2` through `R5`; discard outputs as mechanical only.
3. Freeze the seed-0 screen config and commit result-critical code.
4. Train sequentially: `R2_det`, `R3_det_noise`, `R4_diffusion`, `R5_flow`.
5. Generate `val_id` predictions once using the fixed inference rules.
6. Produce one machine-readable comparison and gate decision.
7. Only a passing generative variant is repeated for seeds `1` and `2`, along
   with its strongest deterministic control.

### 8.10 Experiment 2 pass criteria

Select the better of `R4_diffusion` and `R5_flow` using the seed-0 screen, but
do not claim success until the three-seed confirmation. H2 passes only if:

1. Three-seed mean raw seed mIoU is at least `1.0` point above both `R2_det`
   and `R3_det_noise` under the fixed inference rule.
2. The paired image bootstrap interval for the pooled/per-seed-preserving
   generative-minus-best-deterministic difference has a lower bound above zero.
3. Semantic precision does not decrease by more than `1.0` point, background
   FPR does not increase by more than `0.5` point, and image-level class support
   remains valid.
4. The gain remains positive against the compute-matched deterministic
   ensemble.
5. The learned result improves over `R1_target`; merely reproducing the fused
   pseudo-target more accurately does not demonstrate correction of weak-label
   error.

If no generative variant passes, stop. The rational framework is then a
deterministic teacher-fusion/refinement system; Flow Matching is not justified.

### 8.11 Artifacts

```text
results/diffusion_wsss/voc/objective_screen/<run_id>/
  config.yaml
  command.txt
  git_state.json
  environment.txt
  dataset_manifest.json
  weak_target_manifest.json
  train.log
  checkpoint_manifest.json
  metrics.json
  per_image_metrics.csv
  per_class_metrics.csv
  uncertainty_metrics.json
  efficiency.json
  decision.json
  completion.json
```

## 9. Experiment 3: hard endpoint versus structured uncertainty

### 9.1 Scientific question

Does Flow Matching benefit specifically from a coherent distribution of weak
mask hypotheses, rather than from ordinary soft labels, random corruption, or
test-time ensembling?

Run this experiment only with the generative family that passed Experiment 2.
If diffusion passes and Flow Matching does not, replace "flow" in the variant
names below with the passing diffusion objective and weaken any Flow Matching
research claim accordingly.

### 9.2 Endpoint construction

For each training image, retain complete categorical mask samples indexed by:

```text
teacher x resolution x horizontal flip
```

Do not independently sample every pixel from the marginal mean; that would
destroy object coherence. Each endpoint sample is a complete teacher/view mask
in original-image coordinates, transformed to the common 28 x 28 grid.

Teacher/view reliability weights use only image-level classification
consistency, augmentation equivariance, and predeclared entropy statistics.
They do not use VOC pixel masks. The endpoint distribution is:

```text
p_weak(M | image, image_labels)
  = categorical mixture over complete teacher/view masks
```

The hard endpoint is the pixelwise argmax of the fixed mixture mean. The soft
endpoint is the mixture mean itself.

### 9.3 Required variants

| ID | Training target/objective | Purpose |
|---|---|---|
| `U0_det_soft` | Deterministic KL to mixture mean | Ordinary soft-label control |
| `U1_flow_hard` | Flow to the single hard endpoint | Tests Flow Matching without endpoint uncertainty |
| `U2_flow_structured` | Flow to sampled complete teacher/view endpoints | Proposed structured uncertainty mechanism |
| `U3_flow_unstructured` | Flow to confidence-matched endpoints with spatial teacher identity shuffled in fixed blocks | Controls for entropy/noise without coherent hypotheses |
| `U4_flow_single_teacher` | Flow to views from the best single teacher only | Tests whether gains are simply multi-view augmentation |

All variants use the same condition, refiner core, updates, augmentations, and
inference sample count. `U3` must preserve per-image class support, approximate
class mass, and the marginal confidence histogram while breaking coherent
teacher-level spatial alternatives. Its construction is frozen and unit-tested
before validation.

### 9.4 Shared auxiliary constraints

If used, the following constraints must be identical for `U1` through `U4` and
must also be available to `U0` where mathematically applicable:

- image-level absent-class suppression;
- foreground/background mass regularization;
- augmentation equivariance;
- local feature-affinity smoothness;
- EMA teacher consistency;
- boundary regularization.

Do not add a boundary, affinity, CLIP, or reconstruction loss only to
`U2_flow_structured`. Otherwise the uncertainty contribution is not isolated.

### 9.5 Inference and uncertainty evaluation

The primary prediction is the mean categorical probability over exactly eight
fixed samples. Report `K = 1, 4, 8` as a cost/quality diagnostic, with `K=8`
frozen before confirmatory evaluation. No best-of-K score is allowed.

Evaluate:

- raw seed metrics from Section 6;
- categorical Brier score;
- negative log likelihood with fixed clipping epsilon;
- expected calibration error with fixed bins;
- entropy/error AUROC;
- risk-coverage curve and area under that curve;
- uncertainty near true semantic boundaries versus interior regions;
- sample diversity, pairwise mask IoU, and per-class area variance;
- correlation between learned uncertainty and original teacher disagreement;
- inference cost per expected prediction.

High sample diversity is not intrinsically good. It is useful only when
uncertainty aligns with errors/ambiguity and the mean prediction improves.

### 9.6 Anti-collapse and shortcut diagnostics

Log by image and class:

- predicted background fraction versus weak-teacher background fraction;
- foreground area and class-mass distribution;
- absent-class probability before and after masking;
- number of classes present in generated samples;
- all-background and all-foreground sample frequency;
- sample dependence on image condition, measured by condition shuffling;
- sample dependence on image-level labels, measured by label shuffling;
- teacher identity recoverability from generated masks.

A model that ignores the image, copies a single teacher, or encodes teacher
identity rather than semantic uncertainty fails even if one aggregate mIoU is
positive.

### 9.7 Experiment 3 pass criteria

H3 passes only after seeds `0, 1, 2` if all are satisfied:

1. `U2_flow_structured` exceeds `U1_flow_hard` and `U0_det_soft` by at least
   `1.0` raw seed mIoU point on average.
2. Both paired 95% bootstrap intervals have lower bounds above zero.
3. `U2` exceeds `U3_flow_unstructured`; otherwise entropy/noise, not structured
   uncertainty, explains the result.
4. Brier score and risk-coverage area improve over both `U1` and `U0`, with no
   material background-FPR or semantic-precision regression.
5. The mean prediction, not an oracle or best sample, supplies the gain.
6. Condition-shuffling diagnostics show material degradation, confirming that
   the model uses image content and labels.

Passing H3 authorizes design of a full WSSS framework. It does not authorize a
state-of-the-art claim or COCO automatically; those require a separate plan.

### 9.8 Artifacts

```text
results/diffusion_wsss/voc/uncertainty_screen/<run_id>/
  config.yaml
  command.txt
  git_state.json
  environment.txt
  endpoint_manifest.json
  train.log
  checkpoint_manifest.json
  expected_prediction_metrics.json
  per_image_metrics.csv
  per_class_metrics.csv
  calibration.csv
  risk_coverage.csv
  sample_statistics.csv
  efficiency.json
  decision.json
  completion.json
```

## 10. Overall decision matrix

| Experiment 1 | Experiment 2 | Experiment 3 | Decision |
|---:|---:|---:|---|
| Fail | Not run | Not run | Do not build a diffusion/flow WSSS framework from these teachers |
| Pass | Fail | Not run | Retain diffusion as a frozen teacher; use deterministic fusion/refinement |
| Pass | Diffusion passes, FM fails | Conditional | A diffusion objective may be viable, but no Flow Matching claim |
| Pass | FM passes | Fail | FM may refine weak masks, but structured-uncertainty contribution is unsupported |
| Pass | FM passes | Pass | Proceed to a full uncertainty-conditioned Flow Matching WSSS design |

No branch permits selecting a new threshold, prompt, or teacher after the
confirmatory result and rerunning under the same experiment identity.

## 11. Estimated execution stages and resource controls

Exact duration and storage must be measured by smoke runs because diffusion
backbone choice and caching format dominate cost. Before launch, record an
estimate based on measured images/second and bytes/image.

Suggested queue stages:

```text
V0  environment, checkpoint, and 20-image mechanical validation
V1  full train_id frozen-teacher audit
V1c frozen val_id confirmation
V2  weak-target cache generation for train_aug
V2s seed-0 matched objective screen
V2c passing objective plus deterministic control, seeds 1 and 2
V3s seed-0 endpoint-distribution screen
V3c passing uncertainty variants, seeds 1 and 2
```

Each stage checks free disk before writing. Cache teacher logits in float16 only
after verifying that categorical normalization and metrics match float32 within
a predeclared tolerance. Never delete prior results or checkpoints to recover
space without explicit approval.

## 12. Planned code boundaries

If implementation is authorized, keep the new direction isolated:

```text
models/diffusion_wsss/
  representation.py
  refiner.py
  objectives.py
  samplers.py
experiments/diffusion_wsss/
  configs/
  run_teacher_audit.sh
  run_objective_screen.sh
  run_uncertainty_screen.sh
tools/diffusion_wsss/
  cache_teacher_outputs.py
  evaluate_predictions.py
  collect_comparison.py
tests/diffusion_wsss/
results/diffusion_wsss/
```

Do not modify the frozen E0 checkpoint or overwrite existing CAMs. Reusable
evaluation utilities may call current code, but every new result must retain a
separate run namespace and provenance manifest.

## 13. Required tests for an authorized implementation

### Representation

- categorical rows sum to one in FP32/FP16/BF16;
- absent classes are zero;
- background is present and finite;
- resize/flip round trips preserve alignment;
- native teacher maps are not mutated;
- invalid/padded pixels remain masked.

### Objectives

- deterministic, diffusion, and flow endpoint losses match analytical toy
  cases;
- time interpolation is correct at `t=0` and `t=1`;
- gradients are finite;
- constant foreground/background predictions are penalized;
- objective reduction is identical across batch partitioning.

### Sampling

- fixed seeds reproduce samples;
- ODE solver convergence is checked on a known vector field;
- single-step and multi-step output shapes/dtypes match;
- probability mean is used instead of best-of-K selection;
- sampling never introduces absent classes.

### Data and provenance

- training cannot open segmentation masks;
- split hashes match manifests;
- cached teacher outputs match image IDs and transforms;
- checkpoint and teacher hashes are verified before inference;
- existing result directories cause a hard refusal to launch.

## 14. Paper-level evidence required after all gates

Only after all minimal gates pass should a full research plan add:

- three-seed VOC core results;
- fixed-threshold raw seed and downstream segmentation results;
- matched deterministic, diffusion, and Flow Matching ablations;
- uncertainty calibration and risk-coverage analysis;
- DiG, DiCLIP, ComCD, BRNF, medical conditional diffusion, lung RF,
  SymmFlow, and FlowSDF positioning;
- parameter, memory, training-time, sampling-step, and latency comparisons;
- COCO and an independent architecture only after a separate compute gate.

The central claim must remain narrow enough to answer the T-IP reviewer
concerns: one measurable failure or missing capability, one isolated mechanism,
matched controls, and quantitative evidence explaining why localization
changes.
