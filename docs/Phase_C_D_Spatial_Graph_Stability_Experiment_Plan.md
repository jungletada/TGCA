# Phase C–D Preliminary Experiment Design
## Spatial Graph Stability Diagnostics + Relevance/Stability Decomposition

### Core hypothesis

\[
\boxed{
\text{A class response is reliable when it is simultaneously discriminative
and stable under a structure-preserving graph low-pass perturbation.}
}
\]

This document covers only:

- **Phase C:** study Spatial Graph Stability itself; no selector, no pooling change, no training.
- **Phase D:** explicitly separate semantic Relevance and Graph Stability; still no pooling change and no training.

The purpose is to determine whether Graph Stability contains useful **conditional reliability information** before introducing \(A_{c2p}\) or soft selective aggregation.

---

# 0. What Phase A–B already established

## Phase A

The equivalent-basis experiment showed that the LaST channel-Fourier selector is strongly basis dependent even when the semantic function is preserved.

For PatchFinalLN MCTformer+:

\[
H'(P_nR)=H(P_n)=M
\]

was preserved to approximately:

\[
2.7\times 10^{-7},
\]

and pairwise cosine patch affinity was invariant to approximately:

\[
3.6\times10^{-9}.
\]

However, under Haar rotations, the LaST-style patch-vote selector changed strongly:

\[
\text{vote Spearman}\approx0.177,
\qquad
\text{top-10\% Jaccard}\approx0.178.
\]

Therefore the next stability probe should be built from **basis-invariant patch geometry**, not channel index.

---

## Phase B

The PatchFinalLN local graph contains real semantic structure.

Important observations:

- Uniform local graph already has high same-semantic purity because locality itself is strong.
- Feature-only local affinity showed the strongest semantic edge discrimination:

\[
AUROC_{\text{edge}}\approx0.742,
\qquad
AUROC_{\text{FG-only}}\approx0.844.
\]

- Spatial × feature graph had the lowest FG–BG leakage:

\[
\text{FG-BG leakage}\approx0.0535,
\]

and the highest weighted purity, but its edge-ranking AUROC was lower than feature-only.

Therefore:

\[
\boxed{
\text{feature affinity is the stronger semantic discriminator,}
}
\]

while

\[
\boxed{
\text{spatial × feature is the more conservative propagation graph.}
}
\]

This distinction must be preserved in Phase C–D. Do not prematurely collapse B2 and B3 into one graph.

---

# 1. Frozen model and dataset for Phase C–D

Use exactly the frozen matched:

```text
PatchFinalLN MCTformer+ Small
seed 0
PASCAL VOC 2012
input 448
```

No retraining.

Patch representation:

\[
P_n=LN(P^{12})
\in\mathbb R^{B\times N\times D},
\]

with:

\[
N=28\times28=784,
\qquad
D=384.
\]

Spatial class map:

\[
M=H_{3\times3}(P_n)
\in\mathbb R^{B\times C\times28\times28}.
\]

Use the **pre-ReLU raw classifier response \(M\)** as the graph signal in Phase C.

Reason:

- graph low-pass is a linear signal operation;
- ReLU would destroy negative-response information before stability is measured;
- Relevance will be introduced separately in Phase D.

Use VOC segmentation GT only for diagnostic evaluation.

No GT information enters graph construction or graph smoothing.

---

# 2. Graphs carried forward from Phase B

Use the exact graph definitions from Phase B. Do not change their weights in Phase C.

## C1 / B1 — Spatial graph

\[
W^S_{ij}
=
\mathbf1[j\in\mathcal N(i)]
\exp\left(
-\frac{\|r_i-r_j\|^2}{2}
\right).
\]

## C2 / B2 — Feature graph

\[
W^F_{ij}
=
\mathbf1[j\in\mathcal N(i)]
\operatorname{clamp}
\left(
\frac{1+\cos(p_i,p_j)}{2},
0,1
\right).
\]

## C3 / B3 — Spatial × feature graph

\[
W^{SF}_{ij}=W^S_{ij}W^F_{ij}.
\]

All use the same local 8-neighbor candidate edge set and are symmetrized.

### Why retain all three?

- \(W^S\): simple spatial-smoothing control.
- \(W^F\): strongest semantic edge-ranking graph from Phase B.
- \(W^{SF}\): most conservative boundary-preserving graph from Phase B.

Phase C should determine which graph produces the most useful **stability signal**, which is not necessarily the same graph that gives the best edge AUROC.

---

# 3. Phase C — Spatial Graph Stability only

## 3.1 Scientific questions

Phase C asks:

1. Does graph low-pass behave as a meaningful structure-preserving perturbation?
2. Is the resulting stability signal basis invariant?
3. What does high Graph Stability actually correspond to spatially?
4. Does Graph Stability alone distinguish target / other foreground / background?
5. Is it mostly measuring semantic consistency, object interiorness, or merely local smoothness?
6. Do the B2 and B3 graphs lead to different stability behavior, consistent with their Phase-B roles?

No selector is built in this phase.

---

# 4. Graph low-pass definition

For each image construct the symmetric normalized Laplacian:

\[
L
=
I-D^{-1/2}WD^{-1/2}.
\]

For each class-response signal:

\[
m_c\in\mathbb R^N,
\]

define:

\[
\boxed{
\bar m_c
=
(I+\lambda L)^{-1}m_c.
}
\]

This is equivalent to:

\[
\bar m_c
=
\arg\min_u
\left[
\|u-m_c\|_2^2
+
\lambda u^\top Lu
\right].
\]

The spectral transfer function is:

\[
h_\lambda(\mu)
=
\frac{1}{1+\lambda\mu},
\qquad
\mu\in[0,2].
\]

Thus it is a genuine graph low-pass filter.

---

# 5. Lambda policy

Use:

\[
\boxed{\lambda=1}
\]

as the pre-registered primary setting.

Reason: with normalized Laplacian spectrum \([0,2]\),

\[
h_1(0)=1,
\qquad
h_1(1)=0.5,
\qquad
h_1(2)=\frac13,
\]

which gives moderate smoothing without an extreme cutoff.

Run an analysis-only sensitivity check:

\[
\lambda\in\{0.25,0.5,1,2,4\}.
\]

Important:

- report the entire curve;
- do not choose a “best lambda” from VOC GT;
- Phase-D primary results should continue to use \(\lambda=1\) unless Phase C reveals a numerical failure.

The purpose of the grid is to understand robustness, not tune the method.

---

# 6. Numerical implementation check

Do not explicitly invert the matrix.

For each image/graph/\(\lambda\), solve:

\[
(I+\lambda L)X=M^\top
\]

for all class maps simultaneously.

The matrix is symmetric positive definite.

Use a stable sparse solver / conjugate-gradient implementation.

Record:

```text
relative linear-solve residual
number of iterations
NaN / Inf count
runtime
```

Required numerical residual:

\[
\frac{\|(I+\lambda L)X-M^\top\|_F}
{\|M^\top\|_F+\epsilon}
<10^{-5}.
\]

---

# 7. Stability definition

First compute graph residual:

\[
e_{c,i}
=
|m_{c,i}-\bar m_{c,i}|.
\]

Do **not** use the LaST ratio:

\[
\frac{|M|}{|M-\bar M|+\epsilon}.
\]

That formulation has already shown denominator-pathology risk and confounds Relevance with Stability.

Instead define a robust within-map scale:

\[
\rho_c
=
\operatorname{median}_i e_{c,i}+\epsilon.
\]

Then define bounded stability:

\[
\boxed{
s_{c,i}
=
\frac{1}{
1+e_{c,i}/\rho_c
}.
}
\]

Properties:

\[
0<s_{c,i}\le1.
\]

The score is scale-normalized within each image/class and has no singularity.

Also save the raw residual \(e\). The raw residual is needed to understand what the bounded transformation is hiding.

---

# 8. Phase C-A — Verify that graph low-pass actually performs the intended perturbation

For every graph and lambda, compute:

## Fidelity

\[
F_c
=
\frac{
\|m_c-\bar m_c\|_2
}{
\|m_c\|_2+\epsilon
}.
\]

## Graph Dirichlet energy

\[
E_G(m_c)
=
\frac{
m_c^\top Lm_c
}{
\|m_c\|_2^2+\epsilon
}.
\]

Compare:

\[
E_G(\bar m_c)
\quad\text{vs}\quad
E_G(m_c).
\]

Report energy-retention ratio:

\[
R_E
=
\frac{
E_G(\bar m_c)
}{
E_G(m_c)+\epsilon
}.
\]

## Map preservation

Report:

```text
Pearson(M, M_bar)
Spearman(M, M_bar)
relative L2 change
```

Expected behavior:

- increasing lambda should monotonically reduce graph energy;
- fidelity loss should increase smoothly;
- maps should not collapse to constants at lambda=1.

If this fails, do not interpret semantic stability metrics.

---

# 9. Phase C-B — Basis-invariance regression

Repeat the Phase-A equivalent orthogonal basis transformation on a fixed subset.

Because:

\[
M'=M
\]

and cosine-based graph weights satisfy:

\[
W'=W,
\]

the complete graph-stability computation should satisfy:

\[
\bar M'=\bar M,
\qquad
S'=S.
\]

Required checks:

```text
max_abs graph-weight error
max_abs M_bar error
max_abs stability error
```

Target:

\[
<10^{-5}.
\]

This is an important theoretical contrast with LaST channel Fourier.

---

# 10. Phase C-C — What spatial phenomenon does Stability measure?

This is a mechanism analysis, not a performance test.

For every positive image-class pair, divide valid patches into:

```text
target foreground
other foreground
background
mixed
void
```

Reuse Experiment-2 patch labeling rules.

Additionally divide valid semantic patches into:

```text
semantic interior
semantic boundary-near
```

using patch-level GT boundaries.

Then report stability distributions for:

1. target interior;
2. target boundary;
3. other-FG interior;
4. other-FG boundary;
5. BG interior;
6. BG boundary.

Metrics:

```text
mean
median
25/75 percentile
paired image-level differences
```

This directly tests whether Graph Stability is primarily:

- object-interior consistency;
- generic interiorness;
- class-specific semantic reliability;
- or simple boundary distance.

---

# 11. Phase C-D — Semantic diagnostic of Stability alone

For positive class \(c\), use \(s_{c,i}\) itself as the score.

Compute the same style of region diagnostics as Experiment 2:

## Target vs background

```text
AUROC
AUPRC
```

## Target vs other foreground

```text
AUROC
AUPRC
```

## Tail composition

For:

```text
top 5%
top 10%
top 20%
```

of stability:

```text
target fraction
other-FG fraction
BG fraction
```

## Enrichment

```text
Target-TailEnrich
BG-TailEnrich
OtherFG-TailEnrich
```

## C-PiM-style region ownership

Record which semantic region contains the highest-stability patch.

The purpose is **not** to expect Stability alone to be a good classifier.

The purpose is to measure exactly how far:

\[
\boxed{
\text{stability} \neq \text{semantic correctness}
}
\]

holds in the spatial-graph setting.

---

# 12. Phase C-E — Class specificity of Stability

In multi-label images, compare stability maps for all positive classes.

Compute:

```text
positive-class pair Spearman
top-10% Jaccard
top-20% Jaccard
shared-support ownership
```

Interpretation:

- very high overlap means Stability is mostly class-agnostic “interiorness”;
- lower overlap with target-consistent tails suggests the class response \(M_c\) is meaningfully interacting with graph structure.

This is important because Stability is derived from a class-specific signal \(m_c\), but the graph itself is class-agnostic.

---

# 13. Phase C-F — Negative-class control

For classes absent from the image, compute the same stability statistics.

Compare positive vs absent classes:

```text
mean stability
top-10% stability
max stability
stability entropy
```

If absent classes also contain many highly stable patches, that is not a failure of the graph. It directly demonstrates:

\[
\boxed{
\text{Stability cannot serve as semantic Relevance by itself.}
}
\]

This motivates Phase D.

---

# 14. Phase C-G — Correlation / redundancy analysis

For each positive class map measure relationships among:

```text
Stability S
raw class response M
ReLU(M)
distance to GT semantic boundary
local feature affinity / graph degree
local class-map variance
```

Report:

```text
Spearman correlations
partial/stratified summaries
```

Key questions:

1. Is \(S\) nearly a monotonic copy of \(|M|\)?
2. Is \(S\) mostly boundary distance?
3. Does \(S\) contain information not explained by class-response magnitude?

Do not claim complementarity before this analysis.

---

# 15. Phase C-H — Compare graph choices

For \(W^S\), \(W^F\), and \(W^{SF}\), compare:

```text
low-pass fidelity
energy reduction
target-vs-BG stability AUROC
target-vs-other-FG stability AUROC
target interior vs BG interior separation
boundary suppression
class-pair stability overlap
```

Interpret Phase-B results explicitly:

### Feature graph \(W^F\)

Expected strength:

```text
semantic affinity / target-other separation
```

because it had the strongest edge-semantic AUROC in Phase B.

### Spatial × feature graph \(W^{SF}\)

Expected strength:

```text
boundary preservation / reduced FG-BG leakage
```

because it was the most conservative graph in Phase B.

Do not force a single winner.

If \(W^F\) and \(W^{SF}\) show complementary behavior, carry both to Phase D.

---

# 16. Phase C statistical protocol

Use whole image as the bootstrap cluster.

```text
5,000 paired bootstrap resamples
95% percentile CI
fixed seed
same image IDs across graph variants
```

Report:

```text
micro
macro-class
per-class
single-label
2-label
3+-label
```

Also save per-image/per-class summaries so later Phase D can reuse them.

---

# 17. Phase C decision gate

Proceed to Phase D only if:

1. graph low-pass is numerically stable;
2. it consistently reduces graph-frequency energy;
3. stability is basis invariant;
4. stability is not merely a pathological function of tiny residuals;
5. at least one graph yields interpretable structure-related behavior.

Important:

Phase C does **not** require Stability alone to have high semantic AUROC.

A result such as:

```text
high stability for both target interior and background interior
```

would actually be useful evidence that semantic Relevance must be modeled separately.

---

# 18. Phase D — Separate Relevance and Stability

## 18.1 Scientific question

Phase D tests:

\[
\boxed{
\text{Does Graph Stability provide useful conditional information
once semantic Relevance is already known?}
}
\]

This is stronger than asking whether Stability alone predicts target pixels.

No selector and no pooling change yet.

---

# 19. Relevance definition

Use a CAM-aligned class-specific relevance score:

\[
r_{c,i}
=
ReLU(M_{c,i}).
\]

Normalize within each image/class:

\[
\boxed{
a_{c,i}
=
\frac{
r_{c,i}
}{
\max_j r_{c,j}+\epsilon
}.
}
\]

Thus:

\[
a_{c,i}\in[0,1].
\]

If all responses are non-positive, set the relevance map to zero.

Why this definition:

- it is directly aligned with the existing MCTformer+ raw-CAM branch;
- it is class-specific;
- it introduces no learned parameter or temperature;
- it avoids mixing \(A_{c2p}\) into Phase D.

Do not use \(A_{c2p}\) until Phase E.

---

# 20. Three core Phase-D score maps

For each graph-stability map \(s\), construct:

## D0 — Relevance only

\[
q^R_{c,i}=a_{c,i}.
\]

## D1 — Stability only

\[
q^S_{c,i}=s_{c,i}.
\]

## D2 — Relevance × Stability

\[
\boxed{
q^{RS}_{c,i}
=
a_{c,i}s_{c,i}.
}
\]

No exponents:

\[
\alpha=\beta=1.
\]

No additive combination and no learned fusion in the first pass.

The goal is to test the simplest independent-factor hypothesis.

---

# 21. Phase D-A — Standard semantic diagnostics

For:

```text
R
S
R×S
```

run the same region metrics:

```text
target-vs-BG AUROC/AUPRC
target-vs-other-FG AUROC/AUPRC
top-5/10/20% target/other/BG composition
Target-TailEnrich
BG-TailEnrich
OtherFG-TailEnrich
C-PiM-style ownership
```

Primary paired comparisons:

\[
(R\times S)-R
\]

and:

\[
(R\times S)-S.
\]

The key quantity is not whether \(R\times S\) beats \(S\). It should.

The key question is:

\[
\boxed{
\text{Does }R\times S\text{ improve over }R\text{?}
}
\]

---

# 22. Phase D-B — 2D Relevance–Stability decomposition

This is the most important Phase-D analysis.

For every positive image-class pair, convert:

```text
Relevance percentile
Stability percentile
```

into 5 within-map quintiles.

Build a \(5\times5\) grid.

For each cell report:

```text
P(target)
P(other foreground)
P(background)
number of patches
```

The desired pattern is:

- target probability increases with Relevance;
- within a fixed Relevance band, higher Stability further increases target probability;
- low-Relevance/high-Stability cells remain mostly non-target.

This directly tests the statement:

\[
\boxed{
\text{Reliability requires both discrimination and stability.}
}
\]

---

# 23. Phase D-C — Conditional utility of Stability

To avoid a misleading result caused only by correlation between \(R\) and \(S\), evaluate Stability **within matched Relevance bands**.

For each relevance quintile \(Q_r\), compare:

\[
P(target\mid Q_r,\text{high }S)
\]

with:

\[
P(target\mid Q_r,\text{low }S).
\]

Also compare:

```text
target-vs-BG AUC within relevance bands
target-vs-other-FG AUC within relevance bands
```

This is a non-parametric test of whether Stability contributes information beyond activation strength.

Report image-clustered bootstrap CIs for the high-S minus low-S target-rate difference.

---

# 24. Phase D-D — Four-quadrant failure-mode analysis

Define within-map thresholds:

```text
High Relevance = top 20%
High Stability = top 20%
Low Relevance = bottom 50%
Low Stability = bottom 50%
```

Analyze:

## Q1 — High R, High S

Expected:

```text
strong target enrichment
```

## Q2 — High R, Low S

Potential interpretation:

```text
isolated discriminative peaks
boundary artifacts
false-positive spikes
```

## Q3 — Low R, High S

Expected:

```text
stable but semantically irrelevant background/interior regions
```

This quadrant is the direct empirical demonstration that:

\[
\text{stability}\neq\text{semantic correctness}.
\]

## Q4 — Low R, Low S

Expected mostly irrelevant/noisy responses.

For each quadrant report target/other/BG composition and boundary proximity.

---

# 25. Phase D-E — Does Stability suppress false positives without deleting target evidence?

Use \(R\) as the reference score.

Measure how multiplication by \(S\) changes scores:

\[
\Delta q_{c,i}
=
q^{RS}_{c,i}-q^R_{c,i}.
\]

Stratify by:

```text
target
other foreground
background
target boundary
target interior
FG-BG boundary
```

Report:

- average relative suppression;
- fraction of top-R target patches retained in top-RS;
- fraction of top-R background false positives removed from top-RS;
- target retention vs false-positive removal curve.

This directly tests the intended role of Stability as a **reliability modifier**, not a semantic detector.

---

# 26. Phase D-F — Multi-label foreground competition

Because earlier Experiment 2 showed that the main difficulty is often target-vs-other-foreground rather than generic BG separation, evaluate:

```text
exactly-2-label images
3+-label images
```

For positive class pairs compute:

```text
top-10% Jaccard for R
top-10% Jaccard for S
top-10% Jaccard for R×S
shared-support ownership
dominant-object capture
```

Also compare:

\[
AUROC_{target-other}
\]

for \(R\) vs \(R\times S\).

If Stability improves BG precision but worsens target-other separation, this must be recorded explicitly before adding \(A_{c2p}\) in Phase E.

---

# 27. Phase D-G — Absent-class control

For classes absent from an image compare:

```text
R
S
R×S
```

in terms of:

```text
top-10% mean score
max score
spatial entropy
fraction of high-score patches
```

Expected:

- \(S\) may remain high because background can be structurally stable;
- \(R\) should suppress absent classes;
- \(R\times S\) should inherit semantic suppression from \(R\).

This is another direct test that Stability must remain a modifier, not the semantic anchor.

---

# 28. Phase D-H — Graph comparison using the combined reliability map

Carry at least:

```text
feature graph W^F
spatial×feature graph W^SF
```

through the full Phase-D analysis.

Keep spatial-only \(W^S\) as the control.

Key paired comparisons:

### \(R\times S_F\) vs \(R\)

Does the semantically discriminative feature graph improve:

```text
target-vs-other foreground
multi-label class specificity
```

?

### \(R\times S_{SF}\) vs \(R\)

Does the conservative joint graph improve:

```text
FG-BG precision
boundary false-positive suppression
```

?

This directly follows the Phase-B finding that B2 and B3 have different strengths.

---

# 29. Phase D-I — Basis-invariance regression

On the Phase-A transformed subset verify:

\[
R'=R,
\qquad
S'=S,
\qquad
(RS)'=RS
\]

under equivalent orthogonal basis reparameterization.

Target numerical error:

\[
<10^{-5}.
\]

This should become a permanent regression test for the proposed reliability formulation.

---

# 30. Phase D statistical protocol

Use the same whole-image paired bootstrap:

```text
5,000 resamples
95% percentile CI
fixed seed
```

Report:

```text
micro
macro-class
per-class
single-label
2-label
3+-label
```

For conditional/quadrant analyses, preserve image/class grouping.

Do not treat patches as independent samples for confidence intervals.

---

# 31. Phase D decision gate

Proceed to Phase E (\(A_{c2p}\) semantic anchor) only if at least one graph satisfies the following:

1. \(R\times S\) provides a consistent gain over \(R\) in at least one reliability-relevant metric such as:
   - target tail purity;
   - BG false-positive suppression;
   - target-vs-BG AUC;
   - boundary false-positive reduction.

2. The gain does not come with a severe collapse in:
   - target retention;
   - target-vs-other-FG discrimination.

3. Conditional analysis shows that Stability adds information within fixed Relevance bands.

If these conditions fail, do **not** move immediately to \(A_{c2p}\) or soft pooling.

Instead revisit:

```text
graph definition
low-pass strength
stability transformation
```

because adding more modules would obscure the failure source.

---

# 32. Required outputs

## Phase C

```text
03_GRAPH_STABILITY_DIAGNOSTIC_REPORT.md
graph_lowpass_fidelity.csv
graph_energy_metrics.csv
stability_region_metrics.csv
stability_boundary_metrics.csv
stability_class_overlap.csv
stability_negative_class_control.csv
stability_correlations.csv
lambda_sensitivity.csv
basis_invariance.json
selected_visualizations/
```

## Phase D

```text
04_RELEVANCE_STABILITY_ABLATION_REPORT.md
relevance_stability_metrics.csv
relevance_stability_5x5_grid.csv
conditional_stability_uplift.csv
quadrant_region_composition.csv
false_positive_suppression.csv
multi_label_overlap.csv
absent_class_control.csv
graph_comparison.csv
basis_invariance.json
selected_visualizations/
```

---

# 33. Recommended visualization set

Use a small fixed image set selected by rule, not manually cherry-picked.

For each image/class show:

```text
RGB
GT
raw M
graph-smoothed M_bar
graph residual |M-M_bar|
Stability S
Relevance R
R×S
```

Include examples from:

```text
single-label
2-label
3+-label
large object
small object
strong FG-BG boundary
known false-positive case from Experiment 2
```

These visualizations are diagnostic only; quantitative results remain primary.

---

# 34. Main claims Phase C–D are allowed to support

If the experiments succeed, the intended progression is:

### From Phase A

\[
\text{channel-index Fourier stability is basis dependent.}
\]

### From Phase B

\[
\text{basis-invariant local patch geometry contains semantic structure.}
\]

### From Phase C

\[
\text{graph low-pass residual provides a meaningful structural-stability signal,
but stability alone is not semantic correctness.}
\]

### From Phase D

\[
\boxed{
\text{graph stability is useful as a modifier of class-discriminative evidence,
rather than as a standalone selector.}
}
\]

Only after these statements are empirically supported should the project move to:

\[
A_{c2p}
\]

and soft reliability-weighted aggregation.
