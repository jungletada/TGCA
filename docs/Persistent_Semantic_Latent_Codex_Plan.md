# Persistent Semantic Latent for MCTformer
## Plain-ViT Research & Codex Execution Plan

> **Goal:** Re-define MCTformer class tokens as **persistent semantic latents** that are structurally decoupled from the ViT patch-token stream, while keeping the backbone as a **plain ViT** in the first research stage.
>
> This document is intended to be executable by Codex/Claude Code as a research-engineering plan. The priority is **mechanism validation**, not immediately maximizing final WSSS mIoU.

---

# 1. Research Motivation

MCTformer and MCTformer+ treat class tokens and patch tokens as homogeneous tokens:

\[
X=[C;P],
\]

where

\[
C\in\mathbb{R}^{N_{cls}\times D},\qquad
P\in\mathbb{R}^{N_{patch}\times D}.
\]

The concatenated sequence is passed through the same self-attention:

\[
A=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}\right).
\]

This formulation implicitly makes several assumptions:

1. class tokens and patch tokens should share the same embedding width;
2. class tokens must belong to the same token sequence as visual patches;
3. class-class, class-patch, patch-class, and patch-patch relations should share the same normalization mechanism;
4. class-to-patch and patch-to-class relations are sufficiently represented by blocks extracted from full self-attention;
5. the class-token representation should be static at initialization and progressively updated only through the same Transformer blocks.

These assumptions are convenient for implementation, but they are not theoretically required for WSSS.

The new research direction is:

> **Class tokens should instead be treated as persistent semantic latents that are decoupled from backbone spatial resolution and channel width. They should interact with visual patch tokens explicitly through bidirectional semantic read/write relations.**

For the first stage, **do not introduce a hierarchical backbone**. Keep the original plain ViT/DeiT setting so that the effect of semantic-latent modeling can be isolated.

---

# 2. Core Research Questions

## RQ1 — Does patch-to-class attention contain useful semantic information?

MCTformer explicitly uses:

- class-to-patch attention for localization;
- patch-to-patch attention for affinity refinement.

However, patch-to-class attention is largely unused.

We want to test whether:

\[
A_{p\rightarrow c}
\]

contains meaningful patch-level semantic attribution.

Specifically:

\[
P(c|p)
\]

may complement the conventional class-conditioned localization:

\[
P(p|c).
\]

## RQ2 — Should class and patch tokens be modeled as homogeneous tokens?

Instead of:

\[
[C;P]\rightarrow \text{joint self-attention},
\]

test whether it is better to separate:

- visual representation learning;
- semantic representation learning;
- semantic-to-visual interaction.

The key hypothesis is:

> class semantics and visual context are heterogeneous information streams and should not necessarily compete inside the same softmax normalization.

## RQ3 — Can a shared class-patch compatibility matrix support two semantic views?

Construct a class-patch relation:

\[
R_{cp}\in\mathbb{R}^{C\times N}.
\]

From the same relation, derive:

### Class-conditioned localization

\[
A^{read}_{c,p}=\operatorname{Softmax}_{p}(R_{cp}),
\]

interpreted as:

\[
P(p|c).
\]

### Patch-conditioned semantic attribution

\[
A^{write}_{p,c}=\operatorname{Softmax}_{c}(R_{cp}^{\top}),
\]

interpreted as:

\[
P(c|p).
\]

The two relations answer different questions:

- \(P(p|c)\): **Where is class \(c\)?**
- \(P(c|p)\): **What semantic class explains patch \(p\)?**

## RQ4 — Should semantic latents first read image evidence before writing semantics back to patches?

Test whether the following sequence is necessary:

\[
P\rightarrow C\rightarrow P.
\]

Interpretation:

1. **Semantic Read:** visual features update semantic latents;
2. **Semantic Write:** image-conditioned semantic latents feed class semantics back into patch features.

This should be compared against read-only and write-only variants.

## RQ5 — Is class-token width really required to equal patch-token width?

After validating the interaction formulation, decouple:

\[
D_c \neq D_p.
\]

Test whether semantic representation width can be substantially smaller than the backbone embedding width without degrading performance.

## RQ6 — Is one wide semantic latent better than multiple narrow semantic latents?

Only after the single-latent formulation is stable, compare:

\[
K\times d
\]

under approximately fixed total semantic capacity.

Example:

| Latents per class \(K\) | Width \(d\) | Total width \(K d\) |
|---:|---:|---:|
| 1 | 384 | 384 |
| 2 | 192 | 384 |
| 3 | 128 | 384 |
| 4 | 96 | 384 |
| 6 | 64 | 384 |

This directly tests:

> **representation width vs. representation multiplicity**

without conflating the result with parameter count.

---

# 3. Scope of the First Research Stage

## Keep

- plain ViT / DeiT backbone;
- original image resolution and patch resolution;
- original WSSS dataset and evaluation protocol;
- original classification supervision;
- original training schedule wherever possible;
- original patch positional embeddings;
- original pretrained weights.

## Do NOT introduce yet

- hierarchical ViT / Swin / PVT / MiT;
- Optimal Transport;
- Sinkhorn normalization;
- multi-scale fusion;
- dynamic prototype number;
- diversity losses for multiple semantic slots;
- complex background modeling;
- new heavy CNN/MLP token generators;
- new segmentation decoder;
- downstream architecture changes unrelated to class-patch interaction.

The purpose is to isolate the contribution of:

\[
\boxed{\text{semantic/visual decoupling + bidirectional relation}}
\]

---

# 4. Phase 0 — Reproduce and Instrument the Baseline

Before modifying the architecture, establish a reproducible MCTformer/MCTformer+ baseline.

## 4.1 Required outputs

For every Transformer block \(l\), record raw attention logits before softmax:

\[
S^l=\frac{Q^l(K^l)^\top}{\sqrt d}.
\]

Partition into:

\[
S^l=
\begin{bmatrix}
S_{cc}^l & S_{cp}^l\\
S_{pc}^l & S_{pp}^l
\end{bmatrix}.
\]

Record post-softmax blocks:

\[
A_{cc}^l,\;A_{cp}^l,\;A_{pc}^l,\;A_{pp}^l.
\]

Also record group attention mass for class queries and patch queries, per:

- layer;
- head;
- dataset split;
- image resolution.

Suggested statistics:

\[
m_{c\rightarrow c},\quad m_{c\rightarrow p},\quad m_{p\rightarrow c},\quad m_{p\rightarrow p}.
\]

---

# 5. Phase 1 — Patch-to-Class Semantic Diagnostic

This phase should require minimal or no retraining.

## 5.1 Raw-logit semantic normalization

Take:

\[
S_{pc}^l.
\]

Do **not** use the full-sequence softmax block directly.

Instead normalize only over class keys:

\[
\hat A_{pc}^l=\operatorname{Softmax}_{c}(S_{pc}^l).
\]

This produces a proper patch-conditioned semantic distribution.

If no background class is available yet, evaluate first on **GT foreground pixels only**.

## 5.2 Compare against class-to-patch relation

For:

\[
S_{cp}^l,
\]

compute:

\[
\hat A_{cp}^l=\operatorname{Softmax}_{p}(S_{cp}^l).
\]

Evaluate:

1. conventional class-to-patch localization;
2. patch-to-class semantic attribution;
3. their overlap;
4. their disagreement.

## 5.3 Required diagnostic metrics

Per layer:

- CAM mIoU from \(A_{cp}\);
- semantic mIoU from \(A_{pc}^{\top}\);
- foreground patch classification accuracy;
- per-class semantic accuracy;
- precision/recall of foreground regions;
- overlap IoU between \(A_{cp}\) and \(A_{pc}^{\top}\).

Also evaluate:

\[
M^{mutual}_{c,p}=\sqrt{\hat A_{cp}(c,p)\cdot\hat A_{pc}(p,c)}.
\]

This is diagnostic only; do not claim novelty for dual normalization itself.

## 5.4 Four-region disagreement analysis

For each class and patch, divide regions into:

### Region A
\[
A_{cp}\text{ high},\quad A_{pc}\text{ high}
\]

Interpretation: reliable semantic foreground.

### Region B
\[
A_{cp}\text{ high},\quad A_{pc}\text{ low}
\]

Possible interpretation: discriminative/class-attended but semantically uncertain region or context false positive.

### Region C
\[
A_{cp}\text{ low},\quad A_{pc}\text{ high}
\]

This is the most important region.

Hypothesis:

> non-discriminative object regions missed by class-conditioned localization but still semantically attributable from the patch side.

### Region D
\[
A_{cp}\text{ low},\quad A_{pc}\text{ low}
\]

Likely background/irrelevant/uncertain regions.

For each region, compute GT composition.

## 5.5 Go/No-Go criterion for RQ1

Proceed if at least one of the following holds:

- \(A_{pc}\) achieves clearly non-random semantic attribution;
- \(A_{pc}\) recovers meaningful GT foreground pixels missed by \(A_{cp}\);
- Region C contains significantly more correct foreground than random/background;
- \(A_{cp}\) and \(A_{pc}\) show consistent but complementary layer-wise behavior.

If none holds, do not build the full bidirectional architecture yet.

---

# 6. Phase 2 — Minimal Persistent Semantic Latent Architecture

For the first implementation:

\[
D_c=D_p=384.
\]

This isolates interaction structure from dimensionality.

## 6.1 Backbone

Keep the pretrained patch stream unchanged:

\[
\tilde P^l=\operatorname{ViTBlock}_l(P^{l-1}).
\]

Semantic latents are **not concatenated** into the patch sequence.

## 6.2 Semantic latent initialization

Use:

\[
C^0=E,
\]

where:

\[
E\in\mathbb{R}^{C\times D_c}
\]

is a learnable static semantic embedding.

For this phase:

- one latent per foreground class;
- no multi-prototype slots;
- no image-conditioned initializer yet.

## 6.3 Shared class-patch relation

For selected Transformer layers:

\[
Q_c^l=C^{l-1}W_c^l,
\]

\[
K_p^l=\tilde P^lW_p^l.
\]

Compute:

\[
R^l=\frac{Q_c^l(K_p^l)^\top}{\sqrt{D_r}}.
\]

First implementation:

\[
D_r=384
\]

to minimize confounding.

Later \(D_r\) can be reduced.

---

# 7. Semantic Read

Normalize over spatial patches:

\[
A_R^l=\operatorname{Softmax}_{p}(R^l).
\]

Interpretation:

\[
A_R^l=P(p|c).
\]

Update semantic memory:

\[
\Delta C^l=A_R^l V_p^l,
\]

\[
C^l=C^{l-1}+W_R^l\Delta C^l.
\]

Optional stabilization:

\[
C^l=C^{l-1}+\alpha_l W_R^l\Delta C^l,
\]

where \(\alpha_l\) may be learnable.

---

# 8. Semantic Write

Use the same underlying compatibility matrix:

\[
A_W^l=\operatorname{Softmax}_{c}((R^l)^\top).
\]

Interpretation:

\[
A_W^l=P(c|p).
\]

Generate semantic message:

\[
\Delta P^l=A_W^lV_c^l.
\]

Patch update:

\[
P^l=\tilde P^l+\beta_lW_W^l\Delta P^l.
\]

## Critical implementation rule

Initialize:

\[
\boxed{\beta_l=0}
\]

or a very small value.

This preserves pretrained visual features at the beginning of training and prevents randomly initialized semantic latents from corrupting the patch stream.

---

# 9. Background Handling

Patch-conditioned semantic attribution requires a background option.

Without background:

\[
\sum_{c=1}^{C}P(c|p)=1
\]

would force every patch into a foreground category.

## First minimal implementation

Preferred first implementation: include one background latent:

\[
C^0\in\mathbb{R}^{(C+1)\times D_c}.
\]

Alternative minimal dustbin formulation:

\[
r_{p,bg}=b.
\]

Do not introduce a complex image-conditioned background generator in the first iteration.

---

# 10. First Architecture Ablation: Read vs Write

Run the following four variants.

| Variant | Semantic Read \(P\to C\) | Semantic Write \(C\to P\) |
|---|---:|---:|
| Baseline | ✗ | ✗ |
| Read-only | ✓ | ✗ |
| Write-only | ✗ | ✓ |
| Bidirectional | ✓ | ✓ |

## Interpretations

### Baseline
Original patch stream / original MCTformer reference.

### Read-only
Tests whether semantic latents can improve localization without affecting patch representations.

### Write-only
Tests whether static semantic priors can directly improve patch features.

### Read → Write
Tests the key hypothesis:

> semantic latents should first become image-conditioned by reading visual evidence before writing class semantics back to patches.

---

# 11. Important Additional Ordering Ablation

Test:

1. Read → Write;
2. Write → Read;
3. simultaneous independent Read/Write.

Expected best candidate:

\[
\boxed{\text{Read}\rightarrow\text{Write}}
\]

because semantic latents should first adapt to the current image.

---

# 12. Shared Relation vs Independent Relations

## Shared relation

\[
R_{cp}
\]

is computed once.

Then:

\[
P(p|c)=Softmax_p(R),
\]

\[
P(c|p)=Softmax_c(R^\top).
\]

## Independent relations

Compute:

\[
R^{read}=Q_cK_p^\top,
\]

\[
R^{write}=Q_pK_c^\top.
\]

Compare both.

### Research preference

The shared-relation formulation is theoretically cleaner:

> localization and semantic attribution are two conditional views of the same class-patch compatibility.

Do not assume it will necessarily perform better; verify empirically.

---

# 13. Interaction Depth Study

Do not automatically insert semantic Read/Write into all Transformer layers.

Test:

| Configuration | Read/Write layers |
|---|---|
| Late-1 | 12 |
| Late-3 | 10–12 |
| Late-4 | 9–12 |
| Mid-late | 7–12 |
| Full | 1–12 |

Measure:

- CAM mIoU;
- classification mAP;
- pseudo-label mIoU;
- final WSSS segmentation if affordable;
- FLOPs;
- parameters;
- training stability.

Hypothesis:

> early ViT blocks should primarily build visual representations, while semantic read/write may be more useful in middle/late blocks.

---

# 14. Phase 3 — Dimension Decoupling

Only start this phase if Phase 2 validates the new interaction architecture.

Fix:

\[
D_p=384.
\]

Test:

\[
D_c\in\{64,128,192,256,384\}.
\]

Use a common relation dimension:

\[
D_r=128
\]

or:

\[
D_r=256.
\]

Relation:

\[
Q_c=CW_c,\qquad K_p=PW_p,
\]

with:

\[
W_c\in\mathbb{R}^{D_c\times D_r},
\]

\[
W_p\in\mathbb{R}^{384\times D_r}.
\]

---

# 15. Dimension-Decoupling Metrics

For every \(D_c\), report:

- total parameters;
- added parameters relative to baseline;
- FLOPs;
- classification mAP;
- CAM mIoU;
- pseudo-mask mIoU;
- final segmentation mIoU if feasible;
- \(A_{cp}\) localization quality;
- \(A_{pc}\) semantic attribution quality;
- class-token/foreground-feature similarity;
- effective rank of semantic latents.

---

# 16. Semantic Latent Effective Rank

For each class \(c\), collect image-conditioned semantic states:

\[
C_c(I_1),\dots,C_c(I_M).
\]

Build:

\[
X_c\in\mathbb{R}^{M\times D_c}.
\]

Perform SVD:

\[
X_c=U\Sigma V^\top.
\]

Compute normalized singular values:

\[
p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
\]

Effective rank:

\[
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right).
\]

Analyze whether:

\[
r_{\mathrm{eff}}\ll D_c.
\]

If yes, this supports the hypothesis that the original class-token width is at least partly an architectural compatibility choice rather than an empirically justified semantic capacity.

Do not claim "over-parameterized" without this evidence and controlled width experiments.

---

# 17. Phase 4 — Image-Conditioned Semantic Initialization

Only after static persistent latents work.

Current initialization:

\[
C_c^0=E_c.
\]

Replace with:

\[
C_c^0(I)=E_c+\gamma\Delta C_c(I).
\]

Use zero-initialized:

\[
\gamma=0.
\]

## 17.1 Minimal image-conditioned initializer

Use a pre-read:

\[
R^0=\phi_c(E)\phi_p(P^0)^\top.
\]

Then:

\[
A^0=Softmax_p(R^0),
\]

\[
\Delta C^0=A^0V_p^0.
\]

Finally:

\[
C^0(I)=E+\gamma W\Delta C^0.
\]

This avoids adding a separate heavy token prediction network.

---

# 18. Critical Controls for Dynamic Initialization

The objective is to prove that improvement comes from **image specificity**, not extra parameters.

Run:

| Variant | Same parameter count? | Correct image-conditioned input? |
|---|---:|---:|
| Static latent | baseline | ✗ |
| Parameter-matched static MLP | ✓ | ✗ |
| Dynamic latent | ✓ | ✓ |
| Dynamic + shuffled image feature | ✓ | wrong image |
| Dynamic + detached feature | ✓ | ✓, no backward path |
| Dynamic + random/noise feature | ✓ | ✗ |

The most important experiment:

\[
\boxed{\text{correct image}>\text{shuffled image}}
\]

If shuffled-image initialization falls back toward static performance, this strongly supports image-specific semantic adaptation.

---

# 19. Phase 5 — Width vs Multiplicity

Do not begin until:

- persistent semantic latent architecture is stable;
- dimension decoupling is understood;
- image-conditioned initialization has been evaluated.

Use approximately fixed total semantic scalar capacity.

Example:

| \(K\) semantic latents/class | latent width \(d\) | \(K d\) |
|---:|---:|---:|
| 1 | 384 | 384 |
| 2 | 192 | 384 |
| 3 | 128 | 384 |
| 4 | 96 | 384 |
| 6 | 64 | 384 |

Semantic latent tensor:

\[
C\in\mathbb{R}^{C_{cls}\times K\times d}.
\]

---

# 20. Multi-Latent Read

Each semantic slot can read patches:

\[
R_{c,k,p}=\phi_c(C_{c,k})^\top\phi_p(P_p).
\]

Then:

\[
A_{c,k,p}=Softmax_p(R_{c,k,p}).
\]

Class map fusion candidates:

### Max

\[
M_{c,p}=\max_k A_{c,k,p}.
\]

### LogSumExp

\[
M_{c,p}=\tau\log\sum_k\exp(A_{c,k,p}/\tau).
\]

Do not add learned slot weighting until the simple variants are understood.

---

# 21. Do NOT Add Diversity Regularization Initially

First observe whether slots naturally specialize.

Measure pairwise similarity and spatial overlap:

\[
Sim(A_{c,k},A_{c,k'}).
\]

Only if clear slot collapse occurs should later experiments consider:

- orthogonality loss;
- diversity loss;
- coverage loss;
- OT-based allocation;
- entropy regularization.

These are not part of the first implementation.

---

# 22. CAM Construction

For the proposed model, record several candidate CAMs.

## Read CAM

\[
M^{read}=A_R.
\]

## Write semantic map

\[
M^{write}=A_W^\top.
\]

## Mutual CAM

\[
M^{mutual}_{c,p}=\sqrt{M^{read}_{c,p}M^{write}_{c,p}}.
\]

## Union-style fusion

\[
M^{union}=1-(1-M^{read})(1-M^{write}).
\]

Do not use learnable fusion until simple rules have been evaluated.

The objective is to determine whether read and write maps are complementary.

---

# 23. Patch-Patch Relation: Do Not Replace Yet

In the first plain-ViT experiments, keep the original patch-patch relation available.

Do **not** immediately replace:

\[
A_{pp}
\]

with:

\[
A_{pc}A_{cp}.
\]

First validate semantic Read/Write independently.

Later diagnostic experiment:

\[
R_{pp}^{sem}=A_{pc}A_{cp}.
\]

Compare against:

\[
R_{pp}^{visual}=A_{pp}.
\]

Possible refinement:

\[
M'=R_{pp}^{sem}M.
\]

Measure whether class-mediated semantic affinity can replace or complement raw visual affinity.

This is a secondary research branch.

---

# 24. Semantic Low-Rank Affinity Diagnostic

Given:

\[
A_{pc}\in\mathbb{R}^{N\times C},
\]

\[
A_{cp}\in\mathbb{R}^{C\times N},
\]

construct:

\[
R_{pp}^{sem}=A_{pc}A_{cp}.
\]

Then:

\[
\operatorname{rank}(R_{pp}^{sem})\le C.
\]

For \(K\) semantic slots per class:

\[
\operatorname{rank}(R_{pp}^{sem})\le CK.
\]

Interpretation:

> semantic latents may act as a low-rank semantic basis for patch-patch relations.

This could later become important for hierarchical extension, but it is not the main Phase-1 claim.

---

# 25. Required Metrics

Every major experiment should log:

## Classification
- image-level mAP;
- per-class AP.

## Localization
- raw CAM mIoU;
- foreground IoU;
- background IoU where applicable;
- precision;
- recall.

## Pseudo labels
- pseudo-mask mIoU;
- seed precision;
- seed recall.

## Final WSSS
When justified:
- val mIoU;
- test mIoU if relevant.

## Attention/relation diagnostics
- class-key mass;
- patch-key mass;
- \(A_{cp}\) entropy;
- \(A_{pc}\) entropy;
- per-layer semantic accuracy;
- per-layer localization mIoU;
- agreement/disagreement between read/write maps.

## Representation diagnostics
- semantic latent norm;
- patch norm;
- cosine similarity;
- token gap;
- effective rank;
- class-to-class similarity.

## Efficiency
- trainable parameters;
- total parameters;
- FLOPs;
- GPU memory;
- training throughput;
- inference latency if meaningful.

---

# 26. Token-Gap Diagnostic

Using GT only for analysis, define foreground centroid:

\[
\mu_c^l=\frac{1}{|\Omega_c|}\sum_{p\in\Omega_c}P_p^l.
\]

Semantic-latent gap:

\[
d_c^l=1-\cos(C_c^l,\mu_c^l).
\]

Track:

\[
d_c^1,\dots,d_c^L.
\]

Compare:

- original MCTformer class token;
- persistent semantic latent;
- dynamic semantic latent;
- OTPL-style proxy if available as reference.

Hypothesis:

\[
d_c^l
\]

should decrease while classification capability remains stable.

---

# 27. Parameter-Matching Rules

Avoid attributing improvements to extra capacity.

Whenever possible:

1. report exact added parameters;
2. create parameter-matched static controls;
3. use shared projection layers where possible;
4. compare equal-width and equal-budget variants separately;
5. do not compare \(K>1\) against \(K=1\) without reporting total semantic parameters.

---

# 28. Implementation Structure Suggested for Codex

Adapt to the existing repository, but aim for modular components.

Suggested modules:

```text
models/
  semantic_latent/
    latent_bank.py
    class_patch_relation.py
    semantic_read.py
    semantic_write.py
    background_latent.py

analysis/
  dump_attention_blocks.py
  analyze_patch_to_class.py
  analyze_relation_disagreement.py
  analyze_token_gap.py
  analyze_effective_rank.py
  analyze_attention_mass.py

configs/
  psl/
    baseline.yaml
    read_only.yaml
    write_only.yaml
    read_write.yaml
    dimension_sweep/
    depth_sweep/
    dynamic_init/
    multiplicity_sweep/

scripts/
  run_psl_ablation.sh
  run_p2c_diagnostic.sh
  run_dimension_sweep.sh
```

---

# 29. Recommended API Design

A semantic relation module should expose raw relations explicitly.

Example conceptual API:

```python
relation = semantic_relation(
    semantic_latents,
    patch_tokens,
)

read_attn = softmax(relation, dim="patch")
write_attn = softmax(relation.transpose(-1, -2), dim="class")

semantic_latents = semantic_read(
    semantic_latents,
    patch_tokens,
    read_attn,
)

patch_tokens = semantic_write(
    patch_tokens,
    semantic_latents,
    write_attn,
)
```

Do not hide the relation tensors inside a generic attention block.

They must remain accessible for analysis.

---

# 30. Logging Requirements

Every run should save:

```text
run_id/
  config.yaml
  metrics.json
  model_summary.txt
  param_count.json
  attention_stats.npz
  relation_stats.npz
  cam_metrics.json
  checkpoint_best.pth
```

For diagnostic runs, save a small fixed visualization set across all methods.

Use the same image IDs for:

- baseline;
- read-only;
- write-only;
- bidirectional;
- dynamic initialization;
- multi-latent variants.

---

# 31. Visualization Requirements

For selected images, produce:

1. input image;
2. GT;
3. baseline CAM;
4. read map \(P(p|c)\);
5. write map \(P(c|p)\);
6. mutual map;
7. disagreement map;
8. baseline patch-patch affinity refinement;
9. semantic-latent version.

Important cases:

- large objects;
- small objects;
- multiple instances;
- cluttered background;
- occlusion;
- non-discriminative body regions;
- co-occurring classes.

---

# 32. Statistical Stability

Do not accept a method solely from one run.

For promising variants:

- run at least 3 seeds;
- report mean ± std;
- use identical data splits;
- use identical preprocessing;
- use identical downstream pseudo-label pipeline.

For early screening, one seed is acceptable.

---

# 33. Go/No-Go Decision Tree

## Gate A — Patch-to-class semantics

Proceed if patch-to-class relation contains meaningful semantic information.

If not:
- inspect logits;
- inspect background handling;
- inspect layer choice;
- stop bidirectional modeling if evidence remains weak.

## Gate B — Relation decoupling

Proceed if:

\[
\text{Read/Write model}
\]

matches or exceeds MCTformer with similar parameter budget.

Strong result:

\[
\text{Read+Write} > \text{MCTformer}.
\]

Even stronger:

\[
\text{Read+Write} > \text{Read-only}.
\]

This demonstrates useful semantic feedback.

## Gate C — Semantic width decoupling

Strong result if:

\[
D_c < D_p
\]

matches or exceeds:

\[
D_c=D_p.
\]

Example:

\[
D_c=128/192\ge D_c=384.
\]

This supports semantic/visual channel decoupling.

## Gate D — Image conditioning

Strong evidence requires:

\[
Dynamic(correct)>Dynamic(shuffled).
\]

Best case:

\[
Dynamic(shuffled)\approx Static.
\]

## Gate E — Semantic multiplicity

Strong evidence if, with comparable semantic capacity:

\[
K>1>K=1
\]

in performance terms; operationally require stable improvement of \(K>1\) over \(K=1\) under matched capacity.

Only then pursue multi-prototype/multi-slot modeling.

---

# 34. Priority Order

## P0 — Immediate

1. reproduce baseline;
2. extract raw \(S_{cp},S_{pc}\);
3. patch-to-class semantic diagnostic;
4. layer-wise \(A_{cp}\)/\(A_{pc}\) evaluation;
5. disagreement analysis.

## P1 — Core architecture

6. persistent semantic latent module;
7. Read-only;
8. Write-only;
9. Read → Write;
10. interaction-depth sweep;
11. shared vs independent relation.

## P2 — Core research extensions

12. semantic-width sweep;
13. token-gap analysis;
14. effective-rank analysis;
15. image-conditioned initialization;
16. shuffled-image control.

## P3 — Later

17. width-vs-multiplicity study;
18. semantic low-rank patch affinity;
19. multi-latent specialization;
20. OT only if a concrete allocation problem appears.

## Out of scope for now

21. hierarchical backbone;
22. multi-scale fusion;
23. Swin/PVT/MiT replacement.

---

# 35. Claims That Are Allowed Only After Evidence

Do not prematurely claim:

- class tokens are over-parameterized;
- 384 dimensions are unnecessary;
- patch-to-class attention is superior to class-to-patch attention;
- multi-latent representation is better;
- semantic feedback improves all layers;
- class-mediated affinity replaces patch-patch attention;
- the method solves token imbalance;
- OT is necessary;
- the architecture is inherently hierarchical.

These must be supported by the experiments above.

---

# 36. Preferred Paper-Level Hypothesis

If the main experiments succeed, the paper should center on:

> **Multi-class-token WSSS methods unnecessarily treat class representations as homogeneous members of the visual token sequence. We instead formulate them as persistent semantic latents and explicitly model bidirectional semantic–visual communication through two conditional views of a shared class–patch relation.**

The central conceptual shift is:

\[
\boxed{\text{class token}\rightarrow\text{persistent semantic latent}}
\]

and:

\[
\boxed{\text{joint self-attention}\rightarrow\text{explicit semantic Read/Write}}
\]

---

# 37. Potential Method Naming — Working Only

Do not commit to a final name yet.

Candidates:

- Persistent Semantic Latent Transformer (PSLT)
- Semantic Read-Write Transformer (SRWT)
- Bidirectional Semantic Latent Transformer (BSLT)
- Persistent Class Latent Transformer (PCLT)
- Semantic Latent MCTformer
- Bidirectional Class-Patch Relation Transformer

Naming should be postponed until the dominant mechanism is experimentally established.

---

# 38. Expected Scientific Contributions if Successful

A strong final version could eventually support the following contributions:

1. **Intrinsic finding:** patch-to-class relation carries semantic attribution complementary to class-to-patch localization.
2. **Representation reformulation:** class representations are modeled as persistent semantic latents rather than homogeneous ViT tokens.
3. **Bidirectional relation modeling:** a shared class-patch compatibility yields class-conditioned localization \(P(p|c)\) and patch-conditioned semantics \(P(c|p)\).
4. **Semantic read/write mechanism:** image evidence first updates semantic latents, which then provide semantic feedback to patch features.
5. **Capacity analysis:** semantic latent width is empirically decoupled from visual embedding width.
6. **Optional later finding:** semantic multiplicity may be more useful than simply increasing embedding width.

---

# 39. Codex Execution Checklist

## Step 1
- [ ] Identify baseline MCTformer/MCTformer+ attention implementation.
- [ ] Confirm token ordering.
- [ ] Confirm raw Q/K/V tensor shapes.
- [ ] Confirm class-token count.
- [ ] Confirm patch-token count.
- [ ] Confirm attention head aggregation used for CAM.

## Step 2
- [ ] Add non-invasive hooks for raw attention logits.
- [ ] Dump \(S_{cc},S_{cp},S_{pc},S_{pp}\).
- [ ] Add layer/head statistics.

## Step 3
- [ ] Implement patch-to-class class-only softmax diagnostic.
- [ ] Implement class-to-patch patch-only softmax diagnostic.
- [ ] Evaluate with GT only for analysis.
- [ ] Generate disagreement maps.

## Step 4
- [ ] Implement semantic latent bank.
- [ ] Remove class tokens from patch self-attention for the new branch.
- [ ] Keep patch ViT backbone unchanged.
- [ ] Implement shared class-patch relation.

## Step 5
- [ ] Implement Read-only.
- [ ] Implement Write-only.
- [ ] Implement Read→Write.
- [ ] Zero-init write gate.

## Step 6
- [ ] Add background latent/dustbin.
- [ ] Validate normalization.

## Step 7
- [ ] Run interaction-depth sweep.
- [ ] Run shared-vs-independent relation ablation.

## Step 8
- [ ] Run \(D_c\) sweep.
- [ ] Add parameter-matched controls.

## Step 9
- [ ] Add dynamic pre-read initialization.
- [ ] Run shuffled-image control.

## Step 10
- [ ] Only after positive results: run \(K\times d\) multiplicity study.

---

# 40. Suggested Result Tables

## Table A — Intrinsic relation diagnostic

| Layer | c→p CAM mIoU | p→c semantic mIoU | p→c FG accuracy | Mutual CAM mIoU | Region-C FG purity |
|---|---:|---:|---:|---:|---:|
| 1 | | | | | |
| ... | | | | | |
| 12 | | | | | |

## Table B — Read/Write mechanism

| Variant | Read | Write | Params | CAM | Pseudo Mask | Final mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | ✗ | ✗ | | | | |
| Read-only | ✓ | ✗ | | | | |
| Write-only | ✗ | ✓ | | | | |
| Read→Write | ✓ | ✓ | | | | |

## Table C — Semantic width

| \(D_c\) | \(D_r\) | Params | Eff. Rank | CAM | Pseudo | Final |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | | | | | | |
| 128 | | | | | | |
| 192 | | | | | | |
| 256 | | | | | | |
| 384 | | | | | | |

## Table D — Dynamic initialization

| Variant | Image-specific | Params matched | CAM | Pseudo | Final |
|---|---:|---:|---:|---:|---:|
| Static | ✗ | baseline | | | |
| Static + matched MLP | ✗ | ✓ | | | |
| Dynamic | ✓ | ✓ | | | |
| Shuffled Dynamic | wrong image | ✓ | | | |
| Detached Dynamic | ✓ | ✓ | | | |

## Table E — Width vs multiplicity

| \(K\) | \(d\) | \(Kd\) | Params | Slot overlap | CAM | Final |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 384 | 384 | | | | |
| 2 | 192 | 384 | | | | |
| 3 | 128 | 384 | | | | |
| 4 | 96 | 384 | | | | |
| 6 | 64 | 384 | | | | |

---

# 41. Final Principle for Implementation

Every architectural change must correspond to a specific research question.

Do not add modules simply because they are plausible.

The intended progression is:

\[
\boxed{\text{observe}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{form a mechanism-level hypothesis}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{implement the minimum change required to test it}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{measure representation/attention behavior}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{only then increase model complexity}}
\]

For the current stage, the main target is **not hierarchical modeling**.

The immediate target is to determine whether:

\[
\boxed{\text{Persistent Semantic Latents + Explicit Bidirectional Class–Patch Relations}}
\]

provide a better foundation for WSSS than the original MCTformer formulation:

\[
\boxed{[C;P]\rightarrow\text{homogeneous self-attention}}.
\]
