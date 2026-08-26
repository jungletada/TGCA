# AGENTS.md

> **Project:** ICASSP 2027 paper derived from the MCTTA research line  
> **Target track/scope:** Computer Vision; potentially Image, Video & Multidimensional Signal Processing  
> **Working paper:** **Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation**  
> **Status date:** 2026-08-08  
> **Official full-paper deadline:** 2026-09-16  
> **Paper budget:** four pages for technical content, figures, tables, and references, plus an optional fifth page containing references only

## 1. Mission

The goal is **not** to compress the rejected 16-page MCTTA journal manuscript into a four-page conference paper. The goal is to extract one scientifically focused observation from that work, reformulate it rigorously, validate it across multiple class-token WSSS architectures, and submit it as an original ICASSP paper.

The selected direction is:

> **Vanilla softmax confounds semantic relevance with token-group cardinality when class tokens and patch tokens are normalized together. This causes class–patch attention allocation to depend on the number of patch tokens and therefore on input resolution. Token-Group Calibrated Attention (TGCA) removes this cardinality effect while retaining evidence-driven attention.**

The new paper must be organized around this single failure mode and its correction. MCTTA may appear as one host architecture in the experiments, but it is no longer the paper’s main contribution.

## 2. Repository Reality and Source of Truth

This archive currently contains the old paper package rather than model-training code:

- `main.tex`: rejected T-IP journal manuscript.
- `references.bib`: references used by the old manuscript.
- `figures/`: old journal figures, mostly vector PDFs.
- `IEEEtran.cls`: journal class, not the ICASSP conference template.
- No WSSS training or evaluation implementation is included in this archive.

Therefore:

1. Treat `main.tex` as a **technical record**, not as the new paper skeleton.
2. Do not overwrite the old manuscript.
3. Create a separate directory such as `icassp2027/` for the new paper.
4. Obtain the experiment code from the original MCTformer+/MCTTA development repository or rebuild TGCA from an official class-token WSSS implementation.
5. Record the upstream repository URL, commit hash, environment, and local modifications before reporting any result.
6. Use the official ICASSP 2027 template rather than the included journal class.

## 3. Non-Negotiable Research Pivot

### 3.1 What the new paper is about

The paper is about **attention normalization for heterogeneous token groups** in class-token WSSS.

The primary host is MCTformer+ because it concatenates class tokens and patch tokens in self-attention and directly uses class-to-patch and patch-to-patch attention for CAM refinement.

The method must be tested as a plug-and-play attention replacement in at least one additional independent class-token architecture. Preferred hosts are:

1. MCTformer+ — primary baseline and cleanest test bed.
2. MoRe or CTI — minimum independent generality test.
3. Hierarchical MCTTA — preferred third host if its code is available and reproducible.

### 3.2 What the new paper is not about

Do **not** present any of the following as the central contribution:

- A graph-based adapter.
- Spatial Prior Grapher superiority.
- Class Token Projection convergence speed.
- Hierarchical feature fusion borrowed from FSSS.
- A new direct/single-stage/multi-stage family of pipelines.
- “MCTTA” as a universal adapter.

Do not use **Adapter** in the new title unless the method is genuinely demonstrated as parameter-efficient adaptation across multiple frozen backbones. TGCA is an attention operator, not an adapter.

Do not call the old frozen-classifier-plus-separate-segmentation training procedure “single-stage.” Avoid that taxonomy entirely in the new submission.

## 4. Reviewer Critiques That the New Work Must Resolve

The T-IP decision identified four central problems: insufficient novelty, unclear technical explanations, outdated comparisons, and inadequate evidence. The individual reviews add several concrete requirements.

### Reviewer 1: narrative and current baselines

Required actions:

- Begin with one clearly measurable failure mode rather than broad claims about all WSSS architectures.
- Establish the gap using class-token attention literature.
- Update comparisons through 2025–2026.
- Use vector figures.
- Remove redundant background and preliminaries.

### Reviewer 2: scientific mechanism rather than engineering aggregation

Required actions:

- Provide a mathematical explanation of the token-group cardinality effect.
- Quantify why attention and CAMs improve, not only final mIoU.
- Include controlled alternatives to show that the improvement is not caused only by doubling the attention output scale.
- Report attention-group mass, scale sensitivity, precision/recall behavior, and CAM consistency.

### Reviewer 3: novelty, generality, terminology, and unsupported motivation

Required actions:

- Demonstrate TGCA in multiple host architectures.
- Remove the misleading adapter positioning.
- Do not use CTP convergence as a central motivation.
- Do not make an ambiguous single-stage claim.
- Support every claimed failure mode with either a derivation, a direct measurement, or prior work.

## 5. Scientific Hypothesis

Let an attention row contain keys belonging to semantically different groups:

- `G_c`: class tokens, with cardinality `N_c`.
- `G_p`: patch tokens, with cardinality `N_p`.

For attention logits `s_ij`, vanilla attention is

\[
A_{ij}=\frac{\exp(s_{ij})}{\sum_k\exp(s_{ik})}.
\]

The mass assigned to group `g` is

\[
m_{i,g}=\sum_{j\in G_g} A_{ij}.
\]

When logits are equal, or exchangeable in expectation, vanilla softmax allocates expected group mass in proportion to group cardinality:

\[
\mathbb{E}[m_{i,g}] = \frac{N_g}{\sum_r N_r}.
\]

For a 448×448 image with patch size 16:

\[
N_p=28\times28=784.
\]

For PASCAL VOC:

\[
N_c=20,\qquad \frac{N_c}{N_c+N_p}\approx2.49\%.
\]

For MS COCO:

\[
N_c=80,\qquad \frac{N_c}{N_c+N_p}\approx9.26\%.
\]

Thus the attention allocation changes with class count, patch size, and input resolution even when semantic evidence is otherwise comparable.

The empirical hypothesis is:

> Reducing this token-group cardinality dependence improves class–patch interaction, raw CAM quality, and cross-scale consistency without requiring additional supervision or a larger backbone.

Do not state this as proven until the scale and host-generalization experiments support it.

## 6. Proposed Method: Token-Group Calibrated Attention

### 6.1 Required formulation

For attention head `h`, let

\[
s_{ij}^{h}=\frac{(q_i^h)^\top k_j^h}{\sqrt{d_h}}.
\]

Let `g(i)` and `g(j)` denote the query and key token groups. Define group-calibrated logits as

\[
\widetilde{s}_{ij}^{h}
=
 s_{ij}^{h}
-
\log N_{g(j)}
+
b_{g(i),g(j)}^{h},
\]

and compute one standard row-normalized softmax:

\[
A_{ij}^{h}
=
\frac{\exp(\widetilde{s}_{ij}^{h})}
{\sum_k\exp(\widetilde{s}_{ik}^{h})}.
\]

The relation bias `b` is optional but recommended. It should be:

- Defined per layer and per head.
- Indexed by query-group/key-group relation.
- Zero-initialized.
- Tiny in parameter count.

For two groups, a self-attention head uses a 2×2 relation table:

\[
B^h=
\begin{bmatrix}
 b_{c\rightarrow c}^{h} & b_{c\rightarrow p}^{h}\\
 b_{p\rightarrow c}^{h} & b_{p\rightarrow p}^{h}
\end{bmatrix}.
\]

For DeiT-S with 12 layers and 6 heads, this is only `12 × 6 × 4 = 288` scalar parameters if one table is used per layer and head.

### 6.2 Equivalent hierarchical interpretation

TGCA can be interpreted as a two-level probability model.

First compute mean evidence for each key group:

\[
e_{i,g}^{h}
=
\operatorname{LogSumExp}_{j\in G_g}(s_{ij}^{h})
-
\log N_g.
\]

Then compute evidence-driven group mass:

\[
\pi_{i,g}^{h}
=
\operatorname{Softmax}_{g}
\left(e_{i,g}^{h}+b_{g(i),g}^{h}\right).
\]

Within each group, compute the conditional token distribution:

\[
\rho_{i,j\mid g}^{h}
=
\operatorname{Softmax}_{j\in G_g}(s_{ij}^{h}).
\]

The final attention is

\[
A_{ij}^{h}
=
\pi_{i,g(j)}^{h}\rho_{i,j\mid g(j)}^{h}.
\]

This interpretation is useful in the paper because it distinguishes TGCA from fixed split normalization:

- Vanilla softmax uses group **sum evidence**, which grows with token count.
- Fixed normalized split softmax forces a constant group mass such as 0.5/0.5 regardless of evidence.
- TGCA uses group **mean evidence**, allowing semantic evidence and relation bias to determine the group mass.

### 6.3 Required properties

The implementation and paper must verify the following.

#### Row normalization

\[
\sum_j A_{ij}^{h}=1.
\]

#### Within-group token-replication invariance

If every key/value in one group is duplicated `r` times, its group evidence remains unchanged:

\[
\frac{1}{rN_g}
\sum_{j=1}^{rN_g}\exp(s_{ij})
=
\frac{1}{N_g}
\sum_{j=1}^{N_g}\exp(s_{ij}).
\]

With exact duplicated values, the resulting attention output should also remain unchanged up to numerical precision.

#### Negligible overhead

The cardinality correction adds no learned parameters. Relation biases add only a few hundred parameters for DeiT-S and no meaningful FLOP increase.

### 6.4 Relationship to the old Split Weighted Softmax

The old manuscript defines separate softmax operations:

\[
A_c'=\alpha_c\operatorname{Softmax}(S_c),\qquad
A_p'=\alpha_p\operatorname{Softmax}(S_p).
\]

This formulation must not be reused without correction.

- With `(α_c, α_p)=(1,1)`, the full attention row sums to 2 rather than 1.
- With `(0.5,0.5)`, the row is normalized but each token group is forced to receive exactly 0.5 total mass, independent of semantic evidence.
- The old improvement may therefore contain an output-scale effect and a fixed group-prior effect.

The ICASSP paper must explicitly separate these alternatives experimentally.

## 7. Minimal Implementation Contract

Implement TGCA as a reusable attention-normalization function. It must accept attention logits shaped like `[B, H, Nq, Nk]`, query-group IDs, key-group IDs, and an optional relation-bias table.

Conceptual pseudocode:

```python
def tgca_softmax(logits, query_groups, key_groups, relation_bias=None):
    # logits: [B, H, Nq, Nk]
    # query_groups: [Nq]
    # key_groups: [Nk]

    counts = bincount(key_groups).clamp_min(1)
    correction = log(counts[key_groups])
    corrected = logits - correction[None, None, None, :]

    if relation_bias is not None:
        # relation_bias: [H, num_query_groups, num_key_groups]
        bias = relation_bias[:, query_groups[:, None], key_groups[None, :]]
        corrected = corrected + bias[None, :, :, :]

    return softmax(corrected, dim=-1)
```

Implementation requirements:

- Support `float32`, AMP `float16`, and `bfloat16` without NaNs.
- Compute cardinality logs in a numerically safe dtype when needed.
- Preserve masks for padded or invalid tokens.
- Keep the existing attention dropout behavior after softmax.
- Allow TGCA to be toggled by configuration without changing checkpoints unrelated to the new bias parameters.
- Initialize relation bias to zero so that the initial behavior is count-only TGCA.
- Log the exact token-group counts used at every resolution.

## 8. Required Unit and Diagnostic Tests

Before full training, add tests for:

1. **Row sum:** every unmasked attention row sums to 1 within tolerance.
2. **One-group reduction:** with only one key group, TGCA equals vanilla softmax.
3. **Correction-off reduction:** disabling the cardinality term and relation bias exactly reproduces vanilla softmax.
4. **Replication invariance:** duplicating all keys and values in one group does not change the output beyond numerical tolerance.
5. **Gradient stability:** forward and backward passes contain no NaNs or infinities.
6. **Mask correctness:** masked tokens remain zero after normalization.
7. **Device/dtype coverage:** CPU/GPU and full/mixed precision where available.
8. **Checkpoint compatibility:** an old checkpoint loads with only expected missing TGCA bias parameters.

Create a standalone synthetic script such as:

- `tools/test_token_replication.py`
- `tools/plot_group_mass_vs_token_count.py`

These diagnostics should produce the mechanism figure before expensive training begins.

## 9. Experiment Plan

### 9.1 Stage 0 — Reproduce a trustworthy baseline

Minimum requirement:

- Reproduce MCTformer+ using its official or previously verified implementation.
- Match the original training protocol as closely as possible.
- Record the exact code commit, pretrained checkpoint, dataset split, crop size, augmentation, optimizer, learning rate, batch size, and seed.

The old manuscript reports a VOC raw CAM/seed mIoU of 68.8 for MCTformer+ and 73.9 for full MCTTA under its setup. Treat these numbers only as reference points. Do not copy them into the new paper unless they are reproduced under the final codebase.

A baseline must be within a predeclared tolerance, preferably ±0.3 mIoU of the official/reported result, before method comparisons are considered valid.

### 9.2 Stage 1 — Verify the cardinality phenomenon

Using one trained vanilla checkpoint, evaluate at multiple resolutions without retraining:

\[
224,\ 320,\ 448,\ 512.
\]

For every layer and head, log:

- Class-key group mass.
- Patch-key group mass.
- Attention entropy.
- Class-to-patch and patch-to-class mass separately.
- CAM mIoU at each scale.
- Classification performance at each scale.

Produce:

- Layer-wise group-mass plots.
- Group mass versus patch count.
- A synthetic replication plot where patch tokens are duplicated while logits/values are controlled.

This is the main go/no-go experiment. Do not proceed with the full paper if the measured behavior does not support the stated problem.

### 9.3 Stage 2 — Core normalization ablation

Run the following under identical training conditions:

| Variant | Full row sum | Count-aware | Evidence-driven group mass | Learned relation prior |
|---|---:|---:|---:|---:|
| Vanilla softmax | 1 | No | Yes, but cardinality-confounded | No |
| Old split `(1,1)` | 2 | Partial heuristic | No | Fixed |
| Normalized split `(0.5,0.5)` | 1 | Partial heuristic | No | Fixed |
| TGCA without bias | 1 | Yes | Yes | No |
| TGCA with relation bias | 1 | Yes | Yes | Yes |

Report at least:

- Raw CAM mIoU.
- Pseudo-mask mIoU if the standard refinement is retained.
- Classification mAP/F1 as appropriate.
- Pixel precision and recall under one fixed, documented threshold protocol.
- Confusion ratio using the same definition as the compared class-patch regularization literature.
- Cross-scale CAM consistency.
- Mean and standard deviation over at least three seeds on VOC for the core ablation.

Do not select a different threshold separately for each method on the test set.

### 9.4 Stage 3 — Resolution and cardinality stress test

The main robustness experiment should use a checkpoint trained at the standard resolution and evaluated at several resolutions.

Define a cross-scale CAM consistency metric before inspecting final outcomes. One acceptable option is:

\[
C_{\mathrm{scale}}
=
\frac{1}{|\mathcal{S}|-1}
\sum_{s\neq 1}
\operatorname{IoU}
\left(
\operatorname{Resize}(M_s),
M_1
\right),
\]

where all thresholds and resizing rules are fixed in advance.

Also report the variance of class-group mass across scales:

\[
\operatorname{Var}_{s\in\mathcal S}(m_c^s).
\]

The desired mechanism result is:

\[
\operatorname{Var}_{s}(m_c^s)_{\mathrm{TGCA}}
<
\operatorname{Var}_{s}(m_c^s)_{\mathrm{vanilla}}.
\]

Optional controlled tests:

- Change patch size while keeping the image content fixed.
- Artificially duplicate patch keys/values.
- Evaluate VOC-style and COCO-style class-token counts in the same synthetic setup.

### 9.5 Stage 4 — Plug-and-play generality

Minimum publication requirement:

- Positive result on MCTformer+.
- Positive result on one independent class-token WSSS host such as MoRe or CTI.

Preferred:

- Add the hierarchical MCTTA host as a third architecture.

Use the same reporting pattern:

| Host architecture | Vanilla | TGCA | Gain | Added params | Added latency |
|---|---:|---:|---:|---:|---:|
| MCTformer+ | TBD | TBD | TBD | TBD | TBD |
| MoRe or CTI | TBD | TBD | TBD | TBD | TBD |
| Hierarchical MCTTA | TBD | TBD | TBD | TBD | TBD |

If TGCA works only in full MCTTA, do not claim a general attention principle.

### 9.6 Stage 5 — Datasets and downstream validation

Use:

- PASCAL VOC 2012 for rapid iteration, complete ablations, and three-seed reporting.
- MS COCO 2014 for large-scale validation after the method is stable.

Keep one standardized downstream WSSS pipeline only:

1. Train the image-level classifier.
2. Generate CAMs with the same multi-scale protocol across variants.
3. Apply one fixed refinement/pseudo-mask pipeline.
4. Train one fixed segmentation network with identical settings.

The paper should prioritize raw CAM evidence. Final segmentation mIoU is secondary validation and must not obscure the mechanism.

## 10. Efficiency Reporting

Report:

- Additional parameters.
- FLOPs or MACs.
- Training throughput.
- Inference latency at the standard input size.
- Peak memory if readily available.

Expected TGCA overhead should be approximately zero for the count correction and negligible for relation biases. Measure rather than assert.

## 11. Literature Map for 2025–2026

The related-work section must be short and organized by the failure mode each family addresses.

### 11.1 Class-token and class–patch interaction

- **MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation**, AAAI 2025. Treats erroneous class–patch relations through explicit regularization.
- **Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation**, ICCV 2025. Addresses the semantic gap between classification-oriented class tokens and patch tokens using adaptive proxies and optimal transport.
- **Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation**, ICCV 2025. Uses multiple class-specific tokens, random masking, and attention-head selection.

Required differentiation:

> TGCA does not add semantic regularization, construct proxies, or mask tokens. It changes how heterogeneous token groups are normalized so that group size does not automatically dominate group evidence.

### 11.2 Prototype and region expansion methods

- **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation**, CVPR 2025.
- **Weakly Supervised Semantic Segmentation via Progressive Confidence Region Expansion**, CVPR 2025.
- **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation**, CVPR 2025.

Required differentiation:

> TGCA targets the attention operator before region expansion or prototype transport and can be combined with these strategies.

### 11.3 Frequency and boundary/detail correction

- **FFR: Frequency Feature Rectification for Weakly Supervised Semantic Segmentation**, CVPR 2025.
- **Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation**, CVPR 2026.

Required differentiation:

> TGCA does not recover high-frequency features or explicitly model boundaries; it targets token-group probability allocation.

### 11.4 Foundation-model-assisted WSSS

- **Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation**, CVPR 2025.
- **SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Semantic Segmentation**, AAAI 2026.
- **Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation**, CVPR 2026.
- **Beyond Text: Visual Description Assembly by Probabilistic Model for CLIP-based Weakly Supervised Semantic Segmentation**, CVPR 2026.
- **DiCLIP: Diffusion Model Enhances CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation**, arXiv 2026 unless a final venue is verified before submission.

Required comparison policy:

- Separate ImageNet-only/class-token methods from CLIP-, language-, SAM-, or diffusion-assisted methods.
- Always list backbone, pretraining data, and additional supervision/priors.
- Do not claim overall SOTA by mixing incomparable settings.

### 11.5 Literature verification rule

Before camera-ready submission, verify every paper’s title, author list, venue, year, page numbers, DOI when available, and final publication status using primary sources. Do not cite a 2026 arXiv paper as a conference paper unless the venue has been confirmed.

## 12. Working Title and Contribution Statements

Preferred title:

> **Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation**

Alternative:

> **Calibrating Class–Patch Attention for Scale-Robust Weakly Supervised Semantic Segmentation**

Search for naming conflicts before finalizing the acronym TGCA.

Target contribution statements:

1. **We identify a token-group cardinality bias in class-token Transformers for WSSS, where vanilla softmax entangles semantic relevance with the unequal numbers of class and patch tokens, leading to resolution-sensitive attention allocation.**
2. **We propose Token-Group Calibrated Attention, which replaces group sum evidence with cardinality-normalized mean evidence and optionally learns lightweight relation-specific priors while preserving normalized attention.**
3. **Experiments across multiple class-token WSSS architectures, datasets, and input resolutions evaluate CAM quality, cross-scale consistency, downstream segmentation, and computational overhead.**

Do not write “state of the art” in the contribution list unless the final matched-setting table unambiguously supports it.

## 13. ICASSP Four-Page Narrative

### Page 1 — Problem, observation, and positioning

Include:

- One concise paragraph defining image-label WSSS and class-token CAM generation.
- One paragraph on class–patch attention work from 2024–2026.
- The cardinality-bias observation and a simple VOC numerical example.
- One mechanism figure showing vanilla group mass versus TGCA under changing patch count/resolution.
- Three contribution statements.

Do not include a standalone long Related Work section.

### Page 2 — Method

Include:

- Vanilla group-mass formulation.
- TGCA equation.
- Two-level group/within-group interpretation.
- Replication-invariance statement or compact proposition.
- Integration into self-attention and, optionally, cross-attention.
- Parameter and complexity statement.

### Page 3 — Main experiments

Include:

- Matched-setting VOC and COCO results.
- Core normalization ablation.
- Efficiency.

### Page 4 — Mechanism and generality

Include:

- Resolution/cardinality stress test.
- Plug-and-play host table.
- Precision/recall/confusion or scale-consistency analysis.
- Short limitations paragraph and conclusion.

### Optional Page 5

References only. Do not place method text, figures, tables, acknowledgements, or experimental discussion there unless the official ICASSP instructions explicitly permit it at submission time.

## 14. Figure and Table Budget

Target no more than:

- **Figure 1:** problem and method intuition, including group-mass drift.
- **Figure 2:** scale/cardinality diagnostic or selected qualitative CAMs.
- **Table 1:** matched main comparison plus pretraining/supervision columns.
- **Table 2:** compact ablation and host generality.

All architecture diagrams, plots, and table graphics must be vector PDF/SVG. Raster images are acceptable only for input photographs, CAMs, and segmentation visualizations.

## 15. Claims Allowed and Claims Forbidden

### Allowed when supported

- Vanilla joint softmax produces cardinality-dependent group mass under equal or exchangeable logits.
- TGCA is row-normalized.
- TGCA is invariant to exact within-group token replication.
- TGCA adds negligible parameters and computation.
- TGCA improves CAM quality and scale stability in the tested hosts.
- TGCA is complementary to semantic regularization, prototypes, and foundation-model priors.

### Forbidden without new evidence

- “Graph modules are inherently better than CNN or Transformer blocks for WSSS.”
- “Rapid class-token loss convergence is critical to final segmentation.”
- “MCTTA is a universal adapter.”
- “The method is single-stage” when it uses a separately pretrained frozen classifier to generate pseudo labels.
- “Overall state of the art” across unmatched backbones, pretraining, language supervision, SAM, CLIP, or diffusion priors.
- “Resolution invariant” unless the measured performance and attention allocation support that exact claim. Prefer “more scale-robust” or “less resolution-sensitive.”

## 16. Known Defects in the Old Manuscript That Must Not Be Carried Forward

1. The cross-attention concatenation order in Equation (4) conflicts with the output slicing description.
2. The residual variable `X` in Equation (5) is undefined.
3. The old Split Weighted Softmax formula yields row sum 2 at `(1,1)`, while the accompanying figure suggests a total of 1.
4. Equation (8) contains a textual typo referring to `A_c` twice instead of class and patch groups.
5. COCO MCTTA-D is 43.3 in Table IV but is described as 44.0 in the text.
6. The implementation section says MCTTA-D is trained for 20K/80K segmentation iterations; this appears to mean MCTTA-S.
7. The Fig. 10 caption describes CAM processing stages, while the surrounding text describes direct/single-stage/multi-stage variants.
8. The `OneHot` operation used for multiple positive categories is actually multi-hot thresholding.
9. The old paper does not isolate Cross Attention and hierarchical fusion in a complete additive ablation.
10. The old graph-versus-non-graph evidence is insufficient for a strong graph-specific claim.

Every equation, tensor order, table value, and caption in the new paper must be checked against executable code and raw logs.

## 17. Reproducibility Rules

For every experiment, save:

- Configuration file.
- Random seed.
- Git commit hash and uncommitted diff.
- Environment lock file or package list.
- Training log.
- Best-checkpoint selection rule.
- Evaluation command.
- Raw metrics in machine-readable JSON/CSV.
- Attention diagnostics by layer/head.

Use one canonical script to produce each paper table. Do not manually copy numbers from terminal output.

Recommended tools/scripts:

- `tools/analyze_attention_groups.py`
- `tools/test_token_replication.py`
- `tools/evaluate_scale_consistency.py`
- `tools/collect_cam_metrics.py`
- `tools/export_paper_tables.py`

## 18. Practical Go/No-Go Gates

### Gate A — Phenomenon exists

Proceed only if vanilla attention allocation measurably changes with patch count/resolution and this change is visible in either CAM quality or scale consistency.

### Gate B — Method works beyond output rescaling

TGCA must outperform or meaningfully stabilize results relative to:

- Vanilla softmax.
- Old unnormalized split `(1,1)`.
- Normalized fixed split `(0.5,0.5)`.

### Gate C — Generality

Preferred acceptance target:

- Approximately +0.8 to +1.0 raw CAM mIoU on MCTformer+, or a smaller but consistent improvement accompanied by a strong mechanism result.
- Positive improvement in at least one additional host.
- Clearly reduced cross-scale group-mass variance.
- No material classification degradation.
- Negligible parameter and latency overhead.

These are planning thresholds, not claims to be written before experiments.

### Gate D — Submission decision

Do not submit the TGCA framing if:

- It works only in full MCTTA.
- It improves mIoU only because the old split attention doubles output magnitude.
- It has no measurable scale/cardinality effect.
- Results depend on a different threshold or training schedule for each variant.

If count correction is too strong, a controlled fallback is

\[
\widetilde{s}_{ij}=s_{ij}-\gamma\log N_{g(j)}+b_{g(i),g(j)},
\]

with a small predeclared ablation such as `γ ∈ {0, 0.5, 1}`. Do not perform an extensive search that turns the method into another tuned heuristic.

## 19. Work Schedule

### 2026-08-08 to 2026-08-12 — Pilot and mechanism verification

- Recover or obtain the model code.
- Reproduce MCTformer+.
- Add attention-group logging.
- Implement count-only TGCA.
- Complete synthetic replication tests.
- Run 224/320/448/512 evaluation on VOC.
- Make the go/no-go decision.

### 2026-08-13 to 2026-08-21 — VOC core experiments

- Complete all normalization variants.
- Run three seeds for the core ablation.
- Measure precision, recall, confusion, scale consistency, and efficiency.
- Implement relation bias only after count-only TGCA is validated.

### 2026-08-22 to 2026-08-31 — Generality and COCO

- Integrate TGCA into MoRe or CTI.
- Add hierarchical MCTTA as a third host if feasible.
- Run COCO classification/CAM experiments.
- Run one standardized downstream segmentation pipeline.

### 2026-09-01 to 2026-09-07 — Draft

- Create the ICASSP project with the official template.
- Write a completely new introduction and method section.
- Generate figures from scripts.
- Build tables automatically from raw result files.

### 2026-09-08 to 2026-09-12 — Internal review

- Check technical correctness line by line.
- Verify all recent references and comparison settings.
- Confirm page budget and legibility at 100% zoom.
- Ask one reader unfamiliar with MCTTA to explain the paper’s single core claim after reading Page 1.

### 2026-09-13 to 2026-09-15 — Finalization

- Run final number/caption/equation audit.
- Check similarity and rewrite copied prose.
- Verify PDF fonts, vector figures, anonymity policy, and metadata.
- Submit at least one day before the official deadline.

## 20. Writing and Editing Rules

- Write the paper in English.
- Use concise, falsifiable claims.
- Define every tensor and token order before use.
- Keep notation identical between equations and code.
- Do not reuse long passages from the rejected manuscript.
- Reuse prior results only after reproducing and verifying them.
- Do not cite the unpublished rejected manuscript as prior art unless it becomes publicly available.
- Cite MCTformer+, MoRe, CTI, and recent WSSS work directly.
- State limitations honestly, especially the restriction to architectures with identifiable token groups.
- Avoid “obviously,” “clearly,” and causal language unsupported by controlled experiments.

## 21. Definition of Done

The project is ready for ICASSP submission only when all items below are complete:

- [ ] Official ICASSP 2027 template is used.
- [ ] Four-page technical limit is satisfied.
- [ ] Baseline is reproduced under a documented environment.
- [ ] TGCA unit tests pass.
- [ ] Synthetic replication invariance is demonstrated.
- [ ] Vanilla cardinality/scale sensitivity is measured.
- [ ] Old split, normalized split, count-only TGCA, and TGCA+bias are compared fairly.
- [ ] VOC core experiments include multiple seeds.
- [ ] COCO validation is complete or the paper explicitly narrows its scope.
- [ ] At least two independent host architectures are evaluated.
- [ ] Raw CAM metrics and mechanism diagnostics are reported.
- [ ] One standardized downstream segmentation result is reported.
- [ ] Efficiency is measured.
- [ ] 2025–2026 related work is verified from primary sources.
- [ ] All figures are vector except image/CAM examples.
- [ ] No “adapter,” “single-stage,” graph-superiority, or unmatched-SOTA claim remains.
- [ ] All equations, captions, and numeric values are checked against code and raw logs.

## 22. Primary External References

- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)
- [ICASSP 2027 Publishing and Paper Presentation Options](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)
- [MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/33018)
- [POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.html)
- [Weakly Supervised Semantic Segmentation via Progressive Confidence Region Expansion](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Weakly_Supervised_Semantic_Segmentation_via_Progressive_Confidence_Region_Expansion_CVPR_2025_paper.html)
- [Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.html)
- [FFR: Frequency Feature Rectification for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FFR_Frequency_Feature_Rectification_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
- [Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Exploring_CLIPs_Dense_Knowledge_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.html)
- [Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Class_Token_as_Proxy_Optimal_Transport-assisted_Proxy_Learning_for_Weakly_ICCV_2025_paper.html)
- [Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Hanna_Know_Your_Attention_Maps_Class-specific_Token_Masking_for_Weakly_Supervised_ICCV_2025_paper.html)
- [Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Frequency-Aware_Affinity_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2026_paper.html)
- [Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Leveraging_Class_Distributions_in_CLIP_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2026_paper.html)
- [Beyond Text: Visual Description Assembly by Probabilistic Model for CLIP-based Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Beyond_Text_Visual_Description_Assembly_by_Probabilistic_Model_for_CLIP-based_CVPR_2026_paper.html)
- [DiCLIP: Diffusion Model Enhances CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2605.04593)
