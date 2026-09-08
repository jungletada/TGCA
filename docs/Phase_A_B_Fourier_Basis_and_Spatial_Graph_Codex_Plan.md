# Phase A–B Codex Execution Plan
## LaST Channel-Fourier Basis Dependence + Spatial Graph Semantic Validation

> **Repository:** `~/code/TGCA`  
> **Target repo:** https://github.com/jungletada/TGCA  
> **This plan contains diagnosis only. No training and no new method implementation.**

---

# 0. Which model should be used?

## Phase A uses two frozen models

### A1. Primary model: official supervised LaST-ViT

Use the official LaST-ViT supervised ViT-B/16 implementation/checkpoint from:

```text
https://github.com/ChengShiest/LAST-ViT
```

Reason:

> Phase A questions a property of LaST's own channel-index Fourier selector.  
> The strongest test should therefore be performed on the original LaST architecture itself.

Use the official final normalized patch tokens:

```text
12 Transformer blocks
→ final encoder LayerNorm
→ patch tokens
→ LaST selector
```

If ImageNet validation data is available on the server, use ImageNet val.

If ImageNet val is not available, do **not** block the experiment:
use a fixed natural-image subset already available on the server (VOC val is acceptable)
for the basis-equivalence diagnostic only. In that fallback case, do not report ImageNet
classification accuracy; only report representation/selector/logit invariance diagnostics.

---

### A2. Target-architecture control: frozen PatchFinalLN MCTformer+

Use the **matched seed-0 PatchFinalLN MCTformer+ checkpoint** already produced in the current project.

Reason:

> This is the exact representation on which the proposed Spatial Graph Stability will later operate.

Use:

\[
P_n = LN(P^{12})
\]

from the frozen PatchFinalLN model.

No training.

This control tests whether the same basis-dependence problem remains when LaST's
channel-Fourier selector is applied to the MCTformer+ patch representation.

---

## Phase B uses only one frozen model

Use the same **PatchFinalLN MCTformer+ seed-0 checkpoint**.

Do not use:

```text
original MCTformer+
failed pool-before-classifier Last-MCT
Class-Stable LaST trained checkpoint
```

as the primary Phase-B model.

Reason:

> Phase B asks whether the graph constructed from the representation that will actually be
> used by the future method has semantic meaning. PatchFinalLN is the clean pre-method baseline.

Use full VOC 2012 val with segmentation masks for analysis only.

---

# 1. General rules

Both Phase A and Phase B are frozen-checkpoint analyses.

Do not:

- train or fine-tune any model;
- modify checkpoints;
- introduce Graph Stability into training;
- add losses;
- alter CAM inference;
- tune graph parameters using VOC GT;
- implement later phases.

GT segmentation masks are analysis-only.

Create:

```text
analysis/spatial_graph_stability/
```

with separate Phase-A and Phase-B scripts/tests.

Suggested result root:

```text
results/spatial_graph_stability/
├── phase_a_basis_dependence/<run_id>/
└── phase_b_graph_validation/<run_id>/
```

---

# 2. Phase A — Basis dependence of LaST channel Fourier

# 2.1 Main question

Test:

\[
\boxed{
\text{Does the LaST selector change under an equivalent orthogonal change
of embedding basis, even when the represented semantic function is unchanged?}
}
\]

The important control is to construct a mathematically equivalent basis reparameterization.

---

# 2.2 A1 — Official LaST-ViT test

## Frozen representation

Let the official LaST encoder produce:

\[
X=[q_{\mathrm{CLS}},P]
\]

after the official final encoder LayerNorm, with:

\[
P\in\mathbb R^{N\times D}.
\]

Generate orthogonal matrices:

\[
R^\top R=I.
\]

Use:

```text
Identity
random channel permutation
random signed permutation
10 fixed-seed random Haar orthogonal matrices
```

Transform all final representations:

\[
q' = qR,
\qquad
P'=PR.
\]

For the linear classifier:

\[
z=xW^\top+b,
\]

transform classifier weights so the ordinary non-LaST classifier remains equivalent:

\[
W'=WR.
\]

Verify:

\[
q'W'^\top+b
=
qW^\top+b.
\]

Required numerical check:

```text
max_abs_logit_error < 1e-5
```

before interpreting the LaST selector.

---

# 2.3 Run the LaST selector in each equivalent basis

Use the official repository logic:

```text
FFT over embedding dimension
Gaussian low-pass
repo stability score
channel-wise Top-K over patches
K=1
synthetic pooled token
transformed linear classifier
```

Do not change LaST's selector definition.

For each equivalent basis save:

```text
selected_indices
patch vote counts
pooled synthetic token
LaST logits
```

---

# 2.4 A1 metrics

The dense Haar rotation changes the meaning of individual channel indices, so do not use
raw "same-channel selected index" as the main Haar metric.

Primary basis-comparison metrics:

### Patch vote-map agreement

Aggregate channel selections into a spatial vote map:

\[
v_i=
\sum_d \mathbf 1[i=i_d^*].
\]

Compare each transformed basis with identity using:

```text
Spearman(v_identity, v_basis)
top-10% vote-map Jaccard
normalized L1 distance
vote-map entropy
```

### LaST output sensitivity

Compare:

```text
logit cosine similarity
logit L1 / L2 difference
top-1 prediction agreement
top-5 set agreement
```

For signed permutations / ordinary permutations, additionally report per-channel
selected-index agreement because channel correspondence remains explicit.

### Ordinary classifier control

The standard final representation + transformed classifier must remain invariant.

---

# 2.5 A2 — PatchFinalLN MCTformer+ basis control

Use:

\[
P_n\in\mathbb R^{B\times N\times D}.
\]

For each orthogonal \(R\):

\[
P_n'=P_nR.
\]

The MCTformer+ spatial classifier is a 3×3 Conv:

\[
M=H(P_n).
\]

For every spatial kernel offset \(\delta\), transform its input-channel weight matrix:

\[
W_\delta' = W_\delta R.
\]

Bias remains unchanged.

Verify:

\[
\boxed{
H'(P_n') = H(P_n)=M
}
\]

with:

```text
max_abs_CAM_logit_error < 1e-5
```

This is mandatory.

---

# 2.6 Apply the LaST channel selector to equivalent MCT bases

Apply the same LaST channel-Fourier selector to:

```text
P_n
P_n R
```

without training.

Save patch vote maps.

Primary metrics:

```text
vote-map Spearman
top-10% Jaccard
normalized L1 distance
vote entropy
```

Because VOC GT is available, additionally evaluate the **class-agnostic union foreground mask**:

```text
foreground vote mass
background vote mass
foreground vote enrichment
top-10% foreground/background composition
```

This GT use is analysis-only.

The semantic class map \(M\) is identical by construction; only the channel-Fourier selector is allowed to change.

---

# 2.7 Basis-invariant graph control

Using the same PatchFinalLN features, construct the feature-affinity term intended for Phase B:

\[
F_{ij}
=
\frac{1+\cos(p_i,p_j)}{2}.
\]

Repeat after:

\[
P_n'=P_nR.
\]

Verify:

\[
F'=F
\]

within:

```text
max_abs_error < 1e-5
```

This provides a direct contrast:

```text
channel-Fourier selector: potentially basis-dependent
pairwise cosine graph: orthogonal-basis invariant
```

---

# 2.8 Phase-A outputs

```text
phase_a_basis_dependence/
├── OFFICIAL_LAST_BASIS_REPORT.md
├── MCT_BASIS_REPORT.md
├── basis_transforms.json
├── last_vote_metrics.csv
├── mct_vote_metrics.csv
├── logit_sensitivity.csv
├── invariance_checks.json
├── selected_plots/
└── exact_commands.sh
```

Important plots:

```text
identity vs rotation LaST vote maps
distribution of vote-map Spearman across Haar rotations
distribution of top-10% Jaccard
LaST logit sensitivity under equivalent bases
MCT selector sensitivity while M remains invariant
```

Stop Phase A conclusions at diagnosis.

---

# 3. Phase B — Does the patch graph encode local semantic structure?

# 3.1 Frozen model

Use only:

```text
PatchFinalLN MCTformer+ matched seed-0 checkpoint
```

Dataset:

```text
PASCAL VOC 2012 val
1,449 images
```

Extract:

\[
P_n = LN(P^{12})
\]

at 448 resolution:

```text
28 × 28 = 784 patches
D = 384 for MCTformer+-Small
```

Use the same image transform and GT geometry used in Experiments 1–2.

---

# 3.2 Patch GT labels

Reuse Experiment-2 patch labeling logic.

For each 16×16 patch:

- ignore void pixels;
- require at least 50% valid pixels;
- assign a semantic label only if one valid semantic class/background occupies at least 50%;
- otherwise mark `mixed`;
- `void` and `mixed` are excluded from the primary edge-semantic metrics.

Patch semantic labels:

```text
background = 0
VOC foreground classes = 1..20
mixed
void
```

Do not use these labels to construct the graph.

---

# 3.3 Local candidate edge set

Do not use a fully connected graph.

Primary candidate graph:

```text
8-neighbor / 3×3 local spatial neighborhood
```

Each undirected edge is counted once.

All graph variants must use the same candidate edge set.

---

# 3.4 Graph candidates

Use fixed, non-tuned definitions in Phase B.

## B0 — Uniform local graph

\[
W^{U}_{ij}
=
\mathbf 1[j\in\mathcal N(i)].
\]

This is the simplest locality control.

---

## B1 — Spatial Gaussian graph

Let patch coordinates be in patch units.

\[
W^S_{ij}
=
\mathbf 1[j\in\mathcal N(i)]
\exp
\left(
-\frac{\|r_i-r_j\|^2}{2}
\right).
\]

Use fixed:

```text
sigma_s = 1 patch
```

No tuning.

---

## B2 — Local feature graph

Normalize patch features:

\[
\hat p_i=\frac{p_i}{\|p_i\|_2}.
\]

Use a parameter-free nonnegative affinity:

\[
W^F_{ij}
=
\mathbf 1[j\in\mathcal N(i)]
\frac{1+\hat p_i^\top\hat p_j}{2}.
\]

Clamp to `[0,1]` for numerical safety.

This avoids selecting a feature temperature using GT.

---

## B3 — Joint spatial + feature graph

\[
\boxed{
W^{SF}_{ij}
=
W^S_{ij}W^F_{ij}
}
\]

Symmetrize every weighted graph:

\[
W\leftarrow\frac{W+W^\top}{2}.
\]

No graph low-pass is performed in Phase B.

---

# 3.5 Primary graph-semantic metrics

The goal is not to show that nearby pixels are usually similar; it is to test whether feature affinity improves the semantic quality of local graph edges beyond pure spatial proximity.

For all valid edges:

## Weighted same-semantic-label purity

\[
Purity(W)
=
\frac{
\sum_{(i,j)}W_{ij}\mathbf 1[y_i=y_j]
}{
\sum_{(i,j)}W_{ij}
}.
\]

---

## Weighted semantic boundary leakage

\[
Leak(W)
=
\frac{
\sum_{(i,j)}W_{ij}\mathbf 1[y_i\neq y_j]
}{
\sum_{(i,j)}W_{ij}
}.
\]

For valid labels:

\[
Leak = 1-Purity.
\]

Report both for readability.

---

## Foreground-only purity

Only include edges whose endpoints are both foreground-valid.

Measure whether the graph connects patches of the same foreground semantic class.

This prevents background-dominated images from inflating overall purity.

---

## Foreground–background leakage

Measure:

\[
\frac{
\sum W_{ij}\mathbf 1[
(y_i=0,y_j>0)\lor(y_j=0,y_i>0)
]
}{
\sum W_{ij}
}.
\]

---

## Edge semantic discrimination AUROC / AUPRC

Among the common local candidate edges, use \(W_{ij}\) as a score for:

```text
same semantic label vs different semantic label
```

Report:

```text
overall edge AUROC/AUPRC
foreground-only edge AUROC/AUPRC
```

Uniform B0 has no ranking information and acts as the trivial control.

---

# 3.6 Macro / stratified metrics

Report:

```text
micro edge-weighted result
macro over 20 foreground classes
per-class result
single-label images
two-label images
3+-label images
```

This is important because VOC background area is large.

Do not report only a single overall edge purity.

---

# 3.7 Boundary-focused analysis

Define GT semantic boundary patches from the downsampled/patch-level GT.

Report graph weight distributions for:

```text
same-semantic interior edges
same-semantic boundary-near edges
cross-semantic boundary edges
FG-BG boundary edges
```

The key question is:

> Does \(W^{SF}\) suppress cross-boundary edges more strongly than ordinary spatial smoothing while retaining same-region edges?

---

# 3.8 Phase-B success criteria

Do not use a hard performance threshold to tune the graph.

Interpret qualitatively/statistically:

### Strong support

If B3 `spatial + feature` shows:

```text
higher foreground-only semantic purity than B0/B1
lower FG-BG leakage than B0/B1
higher edge AUROC/AUPRC than B2 or at least clear benefit over spatial-only
```

with consistent image-clustered confidence intervals.

### Partial support

If feature affinity helps only some classes / image strata.

### Not supported

If B3 is no better than spatial-only locality or substantially disconnects valid same-class regions.

---

# 3.9 Statistics

Use whole-image clustered bootstrap:

```text
5,000 resamples
fixed seed
95% percentile CI
```

Do not treat edges as independent statistical samples.

For paired graph comparisons, resample the same image IDs.

Primary paired deltas:

```text
B3 - B1
B3 - B0
B3 - B2
```

---

# 3.10 Basis-invariance regression test

Reuse Phase-A orthogonal transforms on a small fixed image subset.

Verify for B2/B3:

```text
max_abs edge-weight error < 1e-5
```

under orthogonal channel rotations.

This should be a unit/integration test, not the main Phase-B analysis.

---

# 3.11 Phase-B outputs

```text
phase_b_graph_validation/
├── SPATIAL_GRAPH_VALIDATION_REPORT.md
├── graph_metrics.csv
├── graph_metrics_per_class.csv
├── graph_bootstrap_deltas.csv
├── boundary_metrics.csv
├── basis_invariance.json
├── selected_graph_visualizations/
└── exact_commands.sh
```

Suggested visualizations:

```text
RGB image
patch-level GT
local graph edge overlay for B1/B2/B3
high-weight cross-boundary edges
high-weight same-semantic edges
```

Only a small rule-selected set of examples is needed.

---

# 4. Implementation order

Codex should execute in this order:

```text
Step 0  Audit repository / locate exact PatchFinalLN checkpoint and metadata
Step 1  Add shared frozen-feature extraction utility
Step 2  Implement Phase-A basis transforms and tests
Step 3  Run A1 official LaST analysis
Step 4  Run A2 PatchFinalLN MCT analysis
Step 5  Generate Phase-A reports
Step 6  Implement Phase-B local graphs and GT edge metrics
Step 7  Unit test graph symmetry / basis invariance / GT patch geometry
Step 8  Run full VOC Phase-B analysis
Step 9  Generate Phase-B report
Step 10 Stop
```

No training.

---

# 5. Minimal tests

## Phase A

- orthogonal matrix test: \(R^\top R\approx I\);
- standard LaST classifier reparameterization preserves logits;
- transformed MCT 3×3 classifier preserves \(M\);
- LaST selector implementation matches official reference;
- cosine affinity is invariant under orthogonal basis change.

## Phase B

- local edge set is identical across graph variants;
- graph weights are finite and nonnegative;
- graph symmetry;
- no GT information enters graph construction;
- patch GT geometry matches Experiment 2 convention;
- basis invariance of B2/B3;
- bootstrap clusters by image.

---

# 6. Git / execution authorization

Codex is authorized to:

- modify/add analysis code in `~/code/TGCA`;
- download the official LaST checkpoint if absent;
- inspect existing dataset locations;
- run frozen inference on GPU;
- run the full VOC-val Phase-A/Phase-B analyses;
- run unit tests;
- commit analysis code and compact reports/tables without additional confirmation.

Do not:

```text
git push
git reset --hard
force-push
delete unrelated worktree changes
commit model checkpoints
commit large raw feature caches
```

Suggested commit:

```text
Add Fourier basis and spatial graph diagnostics
```

Optional second compact-results commit:

```text
Record Phase A-B graph diagnostic results
```

---

# 7. Codex execution prompt

```text
Work directly in ~/code/TGCA on the current main checkout.

You are authorized to inspect and modify analysis code, run GPU inference,
download the official LaST-ViT supervised checkpoint if necessary, run unit
tests, execute the full frozen Phase-A and Phase-B analyses, and git commit
the implementation plus compact reports/results without asking me again.

Do not git push. Do not use destructive git operations. Preserve unrelated
dirty-worktree changes. Do not train or fine-tune any model.

PHASE A MODEL CHOICE
--------------------
Run Phase A on two frozen models:

A1) Primary: official supervised LaST-ViT ViT-B/16 from
    https://github.com/ChengShiest/LAST-ViT
    using its official final-normalized patch representation and official
    channel-Fourier selector.

A2) Target-architecture control: the existing matched seed-0 PatchFinalLN
    MCTformer+ checkpoint from this project, using P_n = LN(P^12).

For A1, if ImageNet val is available, use it. If not, use a fixed natural-image
subset already available on the server for basis-equivalence diagnostics and
do not report ImageNet accuracy.

PHASE A TASK
------------
Test whether LaST's channel-index Fourier selector changes under mathematically
equivalent embedding-basis transformations.

Use:
- identity;
- random channel permutation;
- random signed permutation;
- 10 fixed-seed Haar-random orthogonal matrices.

For official LaST:
- transform q' = qR and P' = PR;
- transform linear classifier W' = WR;
- verify the ordinary non-LaST classifier logits are invariant (<1e-5);
- then run the official LaST selector in each basis;
- compare patch vote maps and LaST logits.

For PatchFinalLN MCTformer+:
- transform P_n' = P_n R;
- for every 3x3 Conv spatial offset transform the input-channel matrix
  W_delta' = W_delta R;
- verify H'(P_n') == H(P_n) with max error <1e-5;
- run the LaST channel selector on P_n and P_nR;
- compare spatial vote maps;
- because VOC GT is available, report union-foreground/background vote
  composition and enrichment.

Also verify that pairwise cosine patch affinity is invariant under the same
orthogonal basis changes.

Primary Phase-A metrics:
- vote-map Spearman;
- top-10% vote-map Jaccard;
- normalized L1 distance;
- vote entropy;
- LaST logit cosine/L1/L2 differences;
- top-1 and top-5 prediction agreement;
- all equivalence/invariance errors.

Generate separate:
OFFICIAL_LAST_BASIS_REPORT.md
MCT_BASIS_REPORT.md

PHASE B MODEL CHOICE
--------------------
Use ONLY the same frozen matched seed-0 PatchFinalLN MCTformer+ checkpoint.
Do not use the failed Last-MCT or Class-Stable-LaST trained checkpoints.

Use full PASCAL VOC 2012 val (1,449 images), 448 input, and the same patch/GT
geometry as Experiments 1-2. Extract P_n = LN(P^12). No training.

PHASE B GRAPH
-------------
Use the same 8-neighbor local candidate edge set for every graph.

Construct:

B0 Uniform:
    W_ij = 1 on local edges

B1 Spatial Gaussian:
    W_ij = exp(-||r_i-r_j||^2 / 2)
    with sigma_s = 1 patch

B2 Local feature:
    normalize p_i;
    W_ij = clamp((1 + cos(p_i,p_j))/2, 0, 1)

B3 Spatial + feature:
    W_ij = W_spatial * W_feature

Symmetrize weighted graphs.

Do NOT perform graph low-pass yet.
Do NOT use GT to build or tune the graph.

PHASE B GT ANALYSIS
-------------------
Reuse Experiment-2 patch-label geometry:
- valid pixels >= 50%;
- semantic majority >= 50%;
- mixed/void excluded from primary edge metrics.

Evaluate:
- weighted same-semantic-label purity;
- weighted semantic boundary leakage;
- foreground-only same-class purity;
- foreground-background leakage;
- edge AUROC/AUPRC for same-vs-different semantic label;
- foreground-only edge AUROC/AUPRC;
- boundary-focused weight distributions;
- micro, macro-class, per-class;
- single-label / 2-label / 3+-label strata.

Use 5,000 whole-image clustered paired bootstrap resamples.
Primary paired comparisons:
B3-B1, B3-B0, B3-B2.

Also regression-test B2/B3 basis invariance using Phase-A rotations.

Generate:
SPATIAL_GRAPH_VALIDATION_REPORT.md
plus compact CSVs and selected visualizations.

EXECUTION ORDER
---------------
1. Audit and locate exact frozen checkpoints/data.
2. Implement/test Phase A.
3. Run A1 and A2.
4. Generate Phase-A reports.
5. Implement/test Phase B.
6. Run full VOC Phase B.
7. Generate Phase-B report.
8. Commit compact code/results if appropriate.
9. Stop.

Do not implement graph low-pass, selector scores, attention anchors, soft pooling,
new losses, or any training phase in this task.
```
