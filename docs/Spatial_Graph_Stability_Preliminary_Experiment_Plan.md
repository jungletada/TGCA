# Preliminary Experiment Design: Spatial Graph Stability for Selective Aggregation in MCTformer+

## Working hypothesis

> **A class response is reliable when it is simultaneously discriminative and stable under structure-preserving graph low-pass perturbation.**

The goal is not to transplant LaST-ViT's channel-index Fourier heuristic. The transferable paradigm is

\[
\boxed{\text{Patch Features}\rightarrow\text{Reliability / Relevance Score}\rightarrow\text{Selective Pooling}\rightarrow\text{Image-level Prediction}}
\]

For WSSS, the spatial class score map must remain directly supervised:

\[
P^{12}\rightarrow LN\rightarrow H_{3\times3}\rightarrow M\rightarrow\text{Selective Aggregation}\rightarrow z^{patch}.
\]

The raw CAM remains

\[
M=H_{3\times3}(P),
\]

and graph smoothing is used only as a **reliability probe**, not as the final CAM.

---

# 1. Progressive experimental strategy

Implement and analyze in this order:

1. **Phase A — Diagnose LaST channel-Fourier basis dependence**
2. **Phase B — Validate the spatial/semantic graph itself**
3. **Phase C — Validate Spatial Graph Stability as a reliability signal**
4. **Phase D — Separate semantic relevance and graph stability**
5. **Phase E — Add \(A_{c2p}(c,i)\) as a semantic anchor**
6. **Phase F — Replace hard selection with soft reliability-weighted pooling**
7. **Phase G — Progressive matched training ablations**

Each phase should generate a compact report before the next phase is implemented. VOC pixel GT is analysis-only and must never be used for training.

---

# 2. Locked baseline and protocol

Use the already validated **PatchFinalLN MCTformer+** as the direct baseline.

\[
P_n=LN(P^{12}),
\]

\[
M=H_{3\times3}(P_n)\in\mathbb R^{B\times C\times H\times W},
\]

\[
z_c^{GWRP}=GWRP(M_c).
\]

Keep unchanged: class-token branch, CCT, 3×3 classifier, native last-three \(A_{c2p}\), all-layer \(A_{p2p}\), CAM refinement, 448 input, VOC train/val protocol, and matched seed/checkpoint where applicable.

Reuse Experiment 2 patch-region definitions:

```text
target
other foreground
background
mixed
void
```

Use the same valid-patch and majority rules as Experiment 2.

---

# 3. Phase A — Test the basis dependence of LaST channel Fourier

## 3.1 Question

Does LaST's channel-wise Fourier stability depend strongly on an arbitrary embedding basis even when the underlying spatial class score map is mathematically unchanged?

This phase requires **no training**.

## 3.2 Orthogonal-basis diagnostic

Take PatchFinalLN features

\[
P_n\in\mathbb R^{N\times D}.
\]

Generate an orthogonal transform

\[
R^\top R=I,
\]

and transform the feature basis:

\[
P_n'=P_nR.
\]

Transform the input-channel basis of every 3×3 classifier kernel so that

\[
H'(P_n')=H(P_n)=M.
\]

For each spatial kernel offset \(\delta\), if the original weight matrix is \(W_\delta\), use

\[
W_\delta'=W_\delta R.
\]

Verify

\[
\max|M'-M|<10^{-5}.
\]

Evaluate at least:

```text
identity basis
random channel permutation
random sign-permutation
5–10 random Haar orthogonal bases
```

## 3.3 Compare LaST selector under equivalent bases

For each basis compute LaST channel stability \(R^{Fourier}_{i,d}\). Record:

- per-channel selected-patch agreement;
- patch vote-map Spearman correlation;
- top-10% vote-map Jaccard;
- target/other-FG/background vote enrichment;
- maximum and median variation across equivalent bases.

The classifier map \(M\) must remain identical by construction.

## 3.4 Graph invariance control

For the graph defined later from cosine feature affinity, verify numerically that the graph and graph-stability maps remain invariant under the same orthogonal basis changes.

**Output:** `01_FOURIER_BASIS_DEPENDENCE_REPORT.md`

---

# 4. Phase B — Validate the patch graph

## 4.1 Question

Does the graph actually encode the probability that nearby patches belong to the same local semantic structure?

Use normalized patch features

\[
\hat p_i=\frac{p_i}{\|p_i\|_2}
\]

and normalized grid coordinates \(r_i=(x_i,y_i)\).

Do not use a fully connected graph initially. Restrict edges to a local spatial neighborhood \(\mathcal N_s(i)\), initially an 8-neighbor / local 3×3 patch neighborhood.

## 4.2 Graph candidates

### G-S: spatial-only

\[
W^S_{ij}=\mathbf1[j\in\mathcal N_s(i)]\exp\left(-\frac{\|r_i-r_j\|^2}{2\sigma_s^2}\right).
\]

### G-F: local feature-only

\[
W^F_{ij}=\mathbf1[j\in\mathcal N_s(i)]\exp\left(\frac{\hat p_i^\top\hat p_j-1}{\tau_f}\right).
\]

### G-SF: spatial + feature

\[
W^{SF}_{ij}=\mathbf1[j\in\mathcal N_s(i)]\exp\left(\frac{\hat p_i^\top\hat p_j-1}{\tau_f}-\frac{\|r_i-r_j\|^2}{2\sigma_s^2}\right).
\]

Symmetrize:

\[
W\leftarrow\frac{W+W^\top}{2}.
\]

Use normalized Laplacian

\[
L=I-D^{-1/2}WD^{-1/2}.
\]

## 4.3 Graph-quality analysis with VOC GT

GT is analysis-only. Compute:

### Weighted semantic edge purity

\[
Purity=\frac{\sum_{ij}W_{ij}\mathbf1[y_i=y_j]}{\sum_{ij}W_{ij}}.
\]

### Boundary leakage

\[
Leak=\frac{\sum_{ij}W_{ij}\mathbf1[y_i\neq y_j]}{\sum_{ij}W_{ij}}.
\]

Also report foreground-region internal edge retention and compare against:

```text
uniform 8-neighbor graph
2D Gaussian spatial smoothing
G-S
G-F
G-SF
```

The purpose is to test whether semantic feature affinity adds something beyond ordinary spatial blur.

**Output:** `02_SPATIAL_GRAPH_VALIDATION_REPORT.md`

---

# 5. Phase C — Spatial Graph Stability as a reliability probe

## 5.1 Graph low-pass

For each class map \(m_c\in\mathbb R^N\), define

\[
\bar m_c=(I+\lambda L)^{-1}m_c.
\]

Do not form the inverse explicitly; use a stable linear solver / conjugate gradient.

Primary offline value:

```text
lambda = 1.0
```

Small analysis-only sensitivity:

```text
lambda ∈ {0.5, 1.0, 2.0}
```

## 5.2 Graph residual

\[
e_{c,i}=|m_{c,i}-\bar m_{c,i}|.
\]

Interpretation: how strongly the original class response violates local graph structure.

## 5.3 Bounded stability score

Do not reuse LaST's unbounded ratio. Define a robust per-image/per-class scale

\[
\rho_c=\operatorname{median}_i e_{c,i}+\epsilon
\]

and

\[
\boxed{s_{c,i}=\frac{1}{1+e_{c,i}/\rho_c}}.
\]

Thus \(s_{c,i}\in(0,1]\).

## 5.4 Diagnose stability alone

Borrow Experiment 2 metrics:

- C-PiM / top-1 region;
- target-vs-BG AUROC/AUPRC;
- target-vs-other-FG AUROC/AUPRC;
- top-5/10/20% region composition;
- BG-TailEnrich / Target-TailEnrich;
- positive-class-pair top-10% Jaccard;
- shared-support ownership.

Explicitly test whether stability alone prefers background/context.

## 5.5 Compare low-pass families

Compare:

```text
LaST channel-Fourier stability
plain 2D Gaussian stability
G-S graph stability
G-F graph stability
G-SF graph stability
```

Graph stability should only be considered meaningful if it improves semantic-region statistics over simple spatial blur.

**Output:** `03_GRAPH_STABILITY_DIAGNOSTIC_REPORT.md`

---

# 6. Phase D — Explicitly separate relevance and stability

## 6.1 Relevance

Primary classifier relevance:

\[
\boxed{a^M_{c,i}=\sigma(M_{c,i})}.
\]

This remains defined for both present and absent classes.

## 6.2 Stability

Use \(s_{c,i}\) from Phase C.

## 6.3 Selector ablations

### D0 — relevance only

\[
q^M_{c,i}=a^M_{c,i}.
\]

### D1 — stability only

\[
q^S_{c,i}=s_{c,i}.
\]

### D2 — relevance × stability

\[
\boxed{q^{MS}_{c,i}=a^M_{c,i}s_{c,i}}.
\]

Use unit exponents only in the first experiment.

## 6.4 Analysis

Use the Experiment 2 GT-region metrics. Key questions:

1. Does stability alone over-select background?
2. Does \(M\times S\) reduce BG enrichment relative to \(M\)?
3. Does \(M\times S\) preserve target-vs-other-FG discrimination?
4. Does \(M\times S\) preserve target coverage?

Also compare map correlation/top-tail overlap among:

```text
M
S
M×S
native A_c2p
final CAM
```

**Output:** `04_RELEVANCE_STABILITY_ABLATION_REPORT.md`

---

# 7. Phase E — Add A_c2p(c,i) as a semantic anchor

## 7.1 Attention source

Use the existing **native last-three \(A_{c2p}\)** as the primary anchor to remain aligned with MCTformer+.

Do not tune the attention layer source in the first experiment.

## 7.2 Patch-conditional attention

\[
A^{sp}_{c,i}=\frac{A_{c2p}(c,i)}{\sum_jA_{c2p}(c,j)+\epsilon}.
\]

Rescale per class:

\[
a^A_{c,i}=\frac{A^{sp}_{c,i}}{\max_jA^{sp}_{c,j}+\epsilon}.
\]

## 7.3 Ablations

\[
q^M=a^M
\]

\[
q^{MS}=a^Ms
\]

\[
q^{MA}=a^Ma^A
\]

\[
\boxed{q^{MAS}=a^Ma^As}.
\]

All exponents are fixed to 1.

## 7.4 Key metrics

Emphasize:

- target-vs-other-FG AUROC/AUPRC;
- positive-class-pair top-10% Jaccard;
- shared-support ownership;
- dominant-object capture;
- BG-TailEnrich;
- weighted target/other/BG mass.

Interpret each factor separately:

```text
M      → classifier evidence
A_c2p  → class-specific semantic routing
S      → structural reliability
```

**Output:** `05_ATTENTION_ANCHOR_ABLATION_REPORT.md`

---

# 8. Phase F — Soft selective aggregation

Hard semantic-class-wise \(K=1\) is explicitly excluded.

## 8.1 Reliability-weighted pooling

For any nonnegative reliability score \(q_{c,i}\), define

\[
\boxed{w_{c,i}=\frac{q_{c,i}+\epsilon}{\sum_j(q_{c,j}+\epsilon)}}
\]

and

\[
\boxed{z_c^{RWP}=\sum_i w_{c,i}M_{c,i}}.
\]

Do not introduce a softmax temperature in the first version.

## 8.2 Effective support

\[
N_{eff}=\frac{1}{\sum_iw_{c,i}^2}.
\]

Report mean, median, and 5/25/75/95 percentiles.

Compare:

```text
hard K=1
GWRP effective support
relevance-only RWP
full reliability RWP
```

## 8.3 Weighted semantic ownership

Compute:

\[
Mass_{target}=\sum_{i\in target}w_{c,i},
\]

\[
Mass_{other}=\sum_{i\in otherFG}w_{c,i},
\]

\[
Mass_{bg}=\sum_{i\in bg}w_{c,i}.
\]

These are primary for soft aggregation. Retain top-10% metrics only for comparability with Experiments 1–2.

## 8.4 Frozen-checkpoint screening

Using the existing PatchFinalLN checkpoint, evaluate image-level logits produced by:

```text
F0: GWRP baseline
F1: relevance-only RWP
F2: relevance × graph-stability RWP
F3: relevance × attention RWP
F4: relevance × attention × graph-stability RWP
```

Record:

```text
classification AP
weight target/other/BG mass
N_eff
weight entropy
multi-label shared support
```

Raw CAM remains exactly \(M\) in this frozen screening.

**Output:** `06_SOFT_AGGREGATION_SCREENING_REPORT.md`

---

# 9. Phase G — Progressive matched training ablations

Only start after Phases A–F are analyzed.

All variants use identical:

```text
DeiT-S initialization
PatchFinalLN
3×3 classifier
seed
optimizer
LR
epochs
effective batch
augmentation
VOC split
class-token branch
CCT
CAM pipeline
```

No pixel GT is used for training.

## G0 — locked baseline

\[
M\rightarrow GWRP.
\]

Reuse existing PatchFinalLN result if configuration is identical.

## G1 — relevance-only soft pooling

\[
q=a^M.
\]

Purpose: isolate replacing GWRP rank pooling with dense relevance-weighted pooling.

## G2 — relevance × graph stability

\[
q=a^Ms.
\]

Purpose: isolate Spatial Graph Stability over relevance-only pooling.

## G3 — relevance × attention anchor

\[
q=a^Ma^A.
\]

Purpose: isolate the semantic contribution of \(A_{c2p}\).

## G4 — full reliability

\[
\boxed{q=a^Ma^As}.
\]

Purpose: test the complete reliability hypothesis.

---

# 10. First-pass gradient policy

For the first matched training pass use **stop-gradient reliability weights**:

```python
w = reliability.detach()
z = (w * M).sum(...)
```

while \(M\) remains fully differentiable.

This isolates whether the reliability weighting itself is useful and prevents the network from trivially gaming the graph/stability probe.

Do not make graph construction or the graph linear solve end-to-end trainable in the first pass.

---

# 11. Training evaluation

For G0–G4 record:

## Classification

```text
class-token macro mAP
patch-head macro mAP
class loss
patch loss
```

## CAM

```text
fixed-threshold raw CAM mIoU
best-threshold raw CAM mIoU
best threshold
foreground precision
foreground recall
```

## Reliability

```text
N_eff
weight entropy
target weighted mass
other-FG weighted mass
BG weighted mass
```

## Attention / ownership

Reuse selected Experiment-2 metrics:

```text
target-vs-BG AUROC
target-vs-other-FG AUROC
positive-class-pair top10 Jaccard
shared-support ownership
```

---

# 12. Statistics

Use the Experiment 2/3 convention:

```text
whole image = bootstrap cluster
5,000 paired bootstrap resamples
same sampled image IDs for paired variants
95% percentile CI
```

Report:

```text
micro
macro-class
per-class
single-label
exactly-2-label
3+-label
```

Do not treat patches or class pairs from the same image as independent samples.

---

# 13. Core ablation table

| ID | Spatial classifier | Relevance M | Graph stability S | Attention anchor A | Aggregation |
|---|---|---:|---:|---:|---|
| G0 | 3×3 | implicit rank | ✗ | ✗ | GWRP |
| G1 | 3×3 | ✓ | ✗ | ✗ | soft RWP |
| G2 | 3×3 | ✓ | ✓ | ✗ | soft RWP |
| G3 | 3×3 | ✓ | ✗ | ✓ | soft RWP |
| G4 | 3×3 | ✓ | ✓ | ✓ | soft RWP |

This is the main method ablation.

---

# 14. Interpretation gates

- If Phase A shows strong selector changes under equivalent orthogonal bases while \(M\) is invariant, the basis-dependence criticism of channel Fourier is supported.
- If G-SF has higher semantic edge purity and lower boundary leakage than plain spatial smoothing, the graph captures meaningful local structure beyond ordinary blur.
- If stability-only is weak but \(M\times S\) improves over \(M\), stability acts as a reliability modifier rather than semantic evidence.
- If \(M\times A\) mainly improves target-vs-other-FG, \(A_{c2p}\) is functioning as the intended semantic anchor.
- If \(M\times A\times S\) improves target mass / BG mass / precision while preserving support size, the complete hypothesis is supported.
- If soft RWP keeps large \(N_{eff}\) and outperforms semantic-class-wise hard K=1, the previous over-sparsity diagnosis is supported.

---

# 15. Required reports in order

```text
01_FOURIER_BASIS_DEPENDENCE_REPORT.md
02_SPATIAL_GRAPH_VALIDATION_REPORT.md
03_GRAPH_STABILITY_DIAGNOSTIC_REPORT.md
04_RELEVANCE_STABILITY_ABLATION_REPORT.md
05_ATTENTION_ANCHOR_ABLATION_REPORT.md
06_SOFT_AGGREGATION_SCREENING_REPORT.md
07_MATCHED_TRAINING_ABLATION_REPORT.md
```

Do not jump directly to Report 07.

---

# 16. Core claim to test

\[
\boxed{\text{A class response is reliable when it is simultaneously semantically discriminative and stable under a structure-preserving graph low-pass perturbation.}}
\]

Operationally:

\[
\boxed{\text{Reliability}=\underbrace{\text{classifier relevance}}_M\times\underbrace{\text{class-token semantic anchor}}_{A_{c2p}}\times\underbrace{\text{spatial-graph stability}}_S.}
\]

The graph-smoothed map \(\bar M\) is never the final prediction. It is only a counterfactual reliability probe.

The final raw CAM remains the original spatial class map \(M\).
