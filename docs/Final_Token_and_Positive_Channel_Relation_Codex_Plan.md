# Final-Token Semantic Relation & Positive-Channel Relation
## Codex 执行计划（Native MCTformer+）

> 仓库：`~/code/TGCA`  
> 模型：**原始 / Native MCTformer+ Small**  
> 数据：PASCAL VOC 2012 val，448 输入  
> 任务性质：**冻结 checkpoint 的 representation diagnosis**  
> 本轮明确：**不使用 PatchFinalLN，不训练，不改 loss，不实现 selector / aggregation。**

---

# 0. 研究目标

Phase A–D 后，本轮不再研究 LaST channel Fourier、Graph low-pass Stability 或 PatchFinalLN，而直接研究 MCTformer+ 自身最核心的 multi-class token。

只研究两个关系：

\[
\boxed{
S^{last}_{c,i}=c_c^\top p_i
}
\]

以及

\[
\boxed{
S^{pos}_{c,i}=\operatorname{ReLU}(c_c)^\top p_i.
}
\]

其中：

- \(c_c\)：最后 Block-12 的第 \(c\) 个 raw multi-class token；
- \(p_i\)：最后 Block-12 的第 \(i\) 个 raw patch token；
- \(C=20\)，\(D=384\)，448 输入时 \(N=28\times28=784\)。

本任务**不分析 Transformer 每层内部的 \(A_{c2p}^{(l)}\)**。那是后续单独讨论的问题。

---

# 1. 锁定模型

使用此前 Experiment 1 / 2 使用的 matched native MCTformer+ Small seed-0 checkpoint。

必须满足：

```text
final_norm = false
patch_final_norm = false
last_mct = false
MCTformer+ Small
embed_dim = 384
depth = 12
num_heads = 6
BCSS E0
PSL baseline
CTI-BGT disabled
vanilla attention
```

禁止使用：

```text
PatchFinalLN checkpoint
Full FinalLN checkpoint
Last-MCT checkpoint
Class-Stable LaST checkpoint
Graph Stability 相关训练 checkpoint
```

Codex 第一步必须 audit checkpoint/config。若不是 native baseline，直接报错，不要继续。

---

# 2. Token 提取

使用原始 Block-12 post-block residual tokens：

\[
C^{12}
=
[c_1,\ldots,c_C]
\in
\mathbb R^{B\times C\times D}
\]

\[
P^{12}
=
[p_1,\ldots,p_N]
\in
\mathbb R^{B\times N\times D}.
\]

不允许额外：

```text
Final LayerNorm
L2 normalization
projection layer
Q/K projection
```

Primary token source必须是 raw final tokens。

验证原始 class-token readout：

\[
z_c^{cls}
=
\frac1D\sum_{d=1}^{D} c_{c,d}.
\]

要求：

```text
max_abs(extracted_x_cls.mean(-1) - native_class_logits) < 1e-6
```

---

# 3. GT 分析协议

使用：

```text
VOC 2012 val
1,449 images
448 input
```

复用 Experiment 2 的 patch-level GT geometry，将 patch 标记为：

```text
target
other foreground
background
mixed
void
```

Primary semantic metrics 排除 mixed / void。

GT 只允许用于分析，不允许进入 relation 的构造。

---

# 4. Experiment E1 — Final-token product relation

## 4.1 定义

Primary raw compatibility：

\[
\boxed{
S^{last}_{c,i}
=
\frac{c_c^\top p_i}{\sqrt D}.
}
\]

实现：

```python
s_last = torch.einsum(
    "bcd,bnd->bcn",
    x_cls_raw,
    x_patch_raw,
) / math.sqrt(D)
```

Shape：

```text
[B, C, N]
```

注意：

> \(S^{last}\) 不是 Transformer attention。

报告中使用：

```text
final-token affinity
final-token compatibility
S_last
```

不要把 raw product 称为 attention weights。

---

# 5. Spatial normalized view

仅为了 spatial probability-mass diagnostics，定义：

\[
A^{last}_{c,i}
=
\operatorname{softmax}_{i}(S^{last}_{c,i}).
\]

不加入温度参数，不做调参。

规则：

- ranking / AUROC：使用 \(S^{last}\)；
- weighted spatial mass：使用 \(A^{last}\)。

不要用 \(A^{last}\) 替换模型 CAM。

---

# 6. E1：验证最终 token relation 是否具有强 semantic patch relation

对所有 positive image-class pairs 计算：

## Target vs Background

```text
AUROC
AUPRC
```

## Target vs Other Foreground

```text
AUROC
AUPRC
```

这一项是 multi-label WSSS 的重点。

## Top-tail

对：

```text
top 5%
top 10%
top 20%
```

统计：

```text
target fraction
other-FG fraction
background fraction
```

并计算：

```text
Target-TailEnrich
OtherFG-TailEnrich
BG-TailEnrich
```

## Maximum ownership

记录最大 \(S^{last}\) patch 属于：

```text
target
other-FG
background
```

即 C-PiM-style ownership。

## Soft mass

使用 \(A^{last}\) 统计：

\[
Mass_{target},
\quad
Mass_{other},
\quad
Mass_{bg}.
\]

---

# 7. E1：Multi-label class specificity

对于同一图像中的所有 positive-class pairs，计算：

```text
map Spearman
top-5% Jaccard
top-10% Jaccard
top-20% Jaccard
shared-support ownership
dominant-object capture
```

分层：

```text
single-label
exactly 2 labels
3+ labels
```

核心问题：

\[
\boxed{
S^{last}
\text{ 是 class-specific spatial relation，还是 shared foreground/presence relation？}
}
\]

---

# 8. E1：Present / absent class control

对于 absent classes 计算：

```text
max S_last
top-10% mean S_last
A_last spatial entropy
A_last foreground mass
A_last background mass
```

与 GT-positive classes 比较。

如果 relation 真正具有 class semantics，应当体现明显的 class-presence dependence。

---

# 9. E1：Dot product 的 norm / direction 分解

因为：

\[
c_c^\top p_i
=
\|c_c\|
\|p_i\|
\cos(c_c,p_i),
\]

对同一个 image/class，\(\|c_c\|\) 对所有 patches 是常数。

因此同时计算：

```text
S_dot = c^T p
S_cos = cosine(c,p)
P_norm = ||p||
```

分别评估：

```text
target-vs-BG AUROC
target-vs-other AUROC
top-10% region composition
```

并比较：

```text
map Spearman
top-10% overlap
```

目的：

> 判断 raw product 相比此前研究过的 cosine 是否获得额外 semantic information，以及这种差异是否主要来自 patch norm。

这是解释性对照，不是新方法。

---

# 10. E1：与 native 3×3 classifier relation 对比

只把 native patch classifier 作为 reference：

\[
M=H_{3\times3}(P^{12}).
\]

构造：

\[
R^M_{c,i}
=
\frac{
ReLU(M_{c,i})
}{
\max_j ReLU(M_{c,j})+\epsilon
}.
\]

比较：

```text
S_last vs R_M
```

指标：

```text
map Spearman
top-10% Jaccard
target-vs-BG AUROC
target-vs-other AUROC
shared-support region composition
```

禁止在本任务中融合两个信号。

禁止读取每层内部 \(A_{c2p}^{(l)}\)。

---

# 11. E1：Basis-invariance regression

对 shared orthogonal transform：

\[
c'_c=c_cQ,
\qquad
p'_i=p_iQ,
\qquad
Q^\top Q=I,
\]

验证：

\[
(c'_c)^\top p'_i=c_c^\top p_i.
\]

测试：

```text
random permutation
random signed permutation
5 fixed-seed Haar orthogonal matrices
```

要求：

\[
\max |S^{last\prime}-S^{last}|<10^{-5}.
\]

这作为与 LaST channel Fourier Phase-A 结果的 permanent contrast。

---

# 12. E1 最终必须回答的问题

报告必须明确回答：

1. \(S^{last}\) 的 target-vs-BG semantic quality 有多强？
2. \(S^{last}\) 能否区分 target 和 other foreground？
3. final positive classes 的 maps 是否重新 collapse / overlap？
4. raw dot 的行为主要来自 direction 还是 patch norm？
5. 与 native classifier relevance \(R^M\) 相比，final-token relation 提供什么不同的信息？
6. full relation 是否满足预期的 orthogonal-basis invariance？

完成 E1 report 后再进入 E2。

---

# 13. Experiment E2 — Positive-channel relation

Native class-token classification：

\[
\boxed{
z_c^{cls}
=
\frac1D
\sum_{d=1}^{D}c_{c,d}.
}
\]

因此第 \(d\) 个 coordinate 对 logit 的直接 additive contribution 是：

\[
q_{c,d}
=
\frac{c_{c,d}}D.
\]

定义 positive channels：

\[
\mathcal P_c
=
\{d:c_{c,d}>0\}.
\]

以及：

\[
c_c^+=ReLU(c_c).
\]

Primary relation：

\[
\boxed{
S^{pos}_{c,i}
=
\frac{
(c_c^+)^\top p_i
}{
\sqrt D
}.
}
\]

实现：

```python
c_pos = torch.relu(x_cls_raw)

s_pos = torch.einsum(
    "bcd,bnd->bcn",
    c_pos,
    x_patch_raw,
) / math.sqrt(D)
```

同样全部使用 raw post-L12 tokens。

---

# 14. E2-A：先研究 class token 的 positive-channel 结构

在把 positive channels 当作 semantic signal 之前，先分析它们本身。

## Positive-channel fraction

\[
f_c^+
=
\frac{|\mathcal P_c|}{D}.
\]

## Positive contribution mass

\[
m_c^+
=
\sum_d ReLU(c_{c,d}).
\]

## Negative contribution mass

\[
m_c^-
=
\sum_d ReLU(-c_{c,d}).
\]

数值验证：

\[
\boxed{
D z_c^{cls}
=
m_c^+-m_c^-.
}
\]

要求误差 `<1e-6`。

比较 GT-positive vs absent classes：

```text
positive-channel fraction
positive contribution mass
negative contribution mass
positive / negative mass ratio
```

目的：

> 判断 image-level class presence 是否表现为 coherent positive-coordinate shift。

---

# 15. E2-B：positive-channel mask 是否 class-specific？

对于同一图像中的 positive class pairs：

```text
binary positive-mask Jaccard
binary-mask cosine
cosine(ReLU(c_c), ReLU(c_c'))
```

然后跨图像比较：

### Same class across different images

```text
positive-channel Jaccard
ReLU-token cosine
```

### Different classes across different images

同样的指标。

关键比较：

\[
\boxed{
\text{within-class consistency}
>
\text{between-class consistency}?
}
\]

如果不同 positive classes 的 masks 高度共享，则 positive channels 更可能表示 shared presence，而不是 class-specific semantic coordinates。

---

# 16. E2-C：\(S^{pos}\) 的 semantic patch relation

对 \(S^{pos}\) 使用和 E1 完全相同的指标：

```text
target-vs-BG AUROC/AUPRC
target-vs-other AUROC/AUPRC
top-5/10/20% region composition
Target/OtherFG/BG TailEnrich
C-PiM-style ownership
multi-label map overlap
present-vs-absent controls
```

直接进行 paired comparison：

\[
S^{pos}
\quad vs\quad
S^{last}.
\]

重点比较：

```text
Δ target-vs-BG AUROC
Δ target-vs-other AUROC
Δ Target-TailEnrich
Δ BG-TailEnrich
Δ positive-class top-10% Jaccard
```

核心问题：

> 只保留真正对 class-token logit 有正贡献的 coordinates，是否能够提高 spatial class specificity？

---

# 17. E2-D：研究 positive channels 在 patch tokens 中的分布

这是本实验的核心 mechanism analysis。

对每个 positive image-class pair：

## Unweighted positive-coordinate patch mean

\[
U^{pos}_{c,i}
=
\frac1{|\mathcal P_c|}
\sum_{d\in\mathcal P_c}p_{i,d}.
\]

## Weighted positive-coordinate relation

\[
S^{pos}_{c,i}
=
\sum_{d\in\mathcal P_c}
c_{c,d}p_{i,d}.
\]

## Positive-sign concordance

\[
C^{sign}_{c,i}
=
\frac1{|\mathcal P_c|}
\sum_{d\in\mathcal P_c}
\mathbf1[p_{i,d}>0].
\]

分别统计：

```text
target
other foreground
background
mixed
```

的：

```text
mean
median
25/75 percentile
paired image-level differences
```

需要回答：

> target patches 是否比 background / other foreground 更强地表达那些对 class token 分类有正贡献的 coordinates？

---

# 18. E2-E：分解 full final-token relation

定义：

\[
c_c
=
c_c^+-c_c^-,
\]

其中：

\[
c_c^+=ReLU(c_c),
\qquad
c_c^-=ReLU(-c_c).
\]

于是：

\[
\boxed{
S^{last}_{c,i}
=
S^{pos}_{c,i}
-
S^{negmag}_{c,i}
}
\]

其中：

\[
S^{negmag}_{c,i}
=
(c_c^-)^\top p_i.
\]

必须做数值 identity test。

然后分别分析：

```text
S_pos
S_negmag
S_last
```

在：

```text
target
other foreground
background
```

中的分布和 AUROC。

目的：

> 判断最终 semantic relation 主要来自 positive class-token channels，negative channels，还是二者的 cancellation。

---

# 19. E2-F：Positive-channel relation 的 basis dependence

这一点必须明确分析，而不能默认 positive channels 是 intrinsic semantic dimensions。

Full inner product 对共同 orthogonal transform 不变：

\[
S^{last\prime}=S^{last}.
\]

但是：

\[
ReLU(cQ)
\]

一般不会等价于：

\[
ReLU(c)Q.
\]

为了做数学等价的 class-token readout reparameterization，将 mean classifier 写成：

\[
z_c=c_c^\top w,
\qquad
w=\frac1D\mathbf1.
\]

在 basis transform 后：

\[
c'=cQ,
\qquad
p'=pQ,
\qquad
w'=Q^\top w.
\]

验证：

\[
c'w'=cw.
\]

然后研究：

\[
S^{pos\prime}
=
ReLU(cQ)^\top(pQ).
\]

测试：

```text
ordinary channel permutation
signed channel permutation
5 fixed-seed Haar orthogonal matrices
```

预期必须区分：

- 普通 permutation：\(S^{pos}\) 应数值保持；
- signed permutation：不保证；
- general rotation：不保证。

记录：

```text
map Spearman
top-10% Jaccard
normalized L1
semantic AUROC change
```

结论应区分：

```text
empirically useful coordinate mechanism
```

和：

```text
intrinsic representation relation
```

不要混为一谈。

---

# 20. E2-G：class-token confidence stratification

对 GT-positive classes 按 native：

\[
z_c^{cls}
\]

分成 confidence quintiles。

分别报告：

```text
S_pos target-vs-BG AUROC
S_pos target-vs-other AUROC
Target-TailEnrich
```

目的：

> 判断 positive-channel spatial relation 是否主要在 class-token branch 自身分类正确/高置信时才具有 semantic meaning。

---

# 21. 统计协议

统一使用：

```text
whole image = bootstrap cluster
5,000 paired bootstrap resamples
fixed seed
95% percentile CI
```

报告：

```text
micro
macro-class
per-class
single-label
exactly-2-label
3+-label
```

不能把同一图像中的 patches / class pairs 当作独立 bootstrap samples。

---

# 22. 可视化

固定规则选取样本，不允许人工 cherry-pick。

每个 image/class 至少画：

```text
RGB
GT
S_last
S_pos
native classifier relevance R_M
```

E2 额外：

```text
U_pos
C_sign
```

覆盖：

```text
single-label
2-label
3+-label
small object
large object
dominant-object case
classification-success
classification-failure
known late-layer class-overlap case
```

---

# 23. 输出目录

```text
results/final_token_relations/<run_id>/
├── FINAL_TOKEN_AFFINITY_REPORT.md
├── POSITIVE_CHANNEL_RELATION_REPORT.md
├── final_token_semantic_metrics.csv
├── final_token_multilabel_overlap.csv
├── final_token_norm_cosine_decomposition.csv
├── final_token_present_absent.csv
├── positive_channel_statistics.csv
├── positive_channel_class_specificity.csv
├── positive_channel_patch_distributions.csv
├── positive_channel_semantic_metrics.csv
├── positive_channel_decomposition.csv
├── positive_channel_basis_dependence.csv
├── class_confidence_stratification.csv
├── bootstrap_deltas.csv
├── basis_invariance_checks.json
├── selected_visualizations/
├── config.json
├── git_metadata.json
└── exact_commands.sh
```

---

# 24. Interpretation matrix

## 若 \(S^{last}\) target-vs-BG 和 target-vs-other 都很强

说明：

> final persistent semantic tokens 已直接建立 class-specific spatial relation。

## 若 \(S^{last}\) target-vs-BG 强，但 target-vs-other 弱

说明：

> final-token relation 更接近 shared foreground / presence relation，class ownership 不够强。

## 若 \(S^{pos}>S^{last}\)，尤其 target-vs-other 明显提升

说明：

> 对 class-token classification 有正贡献的 coordinates 包含额外的 class-specific spatial structure。

## 若 positive-channel masks 在不同 positive classes 间高度重叠，且 \(S^{pos}\) 没改善

说明：

> positive coordinates 更可能是 distributed/shared presence evidence，而非 class-specific semantic dimensions。

## 若 \(S^{pos}\) semantic quality 很好，但对 rotation 高度敏感

说明：

> positive-channel relation 可以是有用的 classifier-coordinate mechanism，但不能称为 intrinsic representation relation。

---

# 25. 执行顺序

```text
Step 0  audit native checkpoint/config
Step 1  实现 frozen raw final-token extraction
Step 2  验证 class-token mean-logit equivalence
Step 3  实现 S_last + basis-invariance tests
Step 4  运行完整 VOC E1 semantic diagnostics
Step 5  生成 FINAL_TOKEN_AFFINITY_REPORT.md
Step 6  实现 positive-channel statistics + S_pos
Step 7  运行 positive-channel class-specificity
Step 8  运行 S_pos semantic diagnostics
Step 9  运行 positive-channel patch-distribution analysis
Step 10 运行 positive/negative contribution decomposition
Step 11 运行 S_pos basis-dependence diagnostics
Step 12 运行 class-confidence stratification
Step 13 生成 POSITIVE_CHANNEL_RELATION_REPORT.md
Step 14 提交 compact code/results（如合适）
Step 15 STOP
```

禁止继续：

```text
per-layer A_c2p
new selector
new aggregation
new loss
training
CAM modification
```

---

# 26. 必要测试

1. checkpoint 必须是 native MCTformer+，不能是 FinalLN / PatchFinalLN / Last-MCT。
2. `x_cls.mean(-1)` 必须匹配 native class logits。
3. 448 输入 shape 必须为 `[B,20,384]` 和 `[B,784,384]`。
4. \(S^{last}\) einsum 与直接 matrix multiplication 一致。
5. \(S^{last}\) 在 shared orthogonal transform 下不变。
6. 
   \[
   Dz_c=m_c^+-m_c^-.
   \]
7. 
   \[
   S^{last}=S^{pos}-S^{negmag}.
   \]
8. 普通 channel permutation 保持 \(S^{pos}\)。
9. signed/general rotation 作为 basis-dependence diagnostic 正确执行。
10. GT 不进入 relation construction。
11. bootstrap cluster unit 必须是 image。

---

# 27. Codex 权限

Codex 获得权限：

- 修改 `~/code/TGCA` 中必要的 analysis code；
- 查找并读取 native MCTformer+ checkpoint；
- 使用 GPU 对完整 VOC val 做 frozen inference；
- 新增必要 tests；
- 生成 compact CSV / JSON / Markdown；
- tests 通过后可以直接 `git commit`；
- 完整分析无需再次询问用户确认。

禁止：

```text
training / fine-tuning
git push
git reset --hard
force-push
destructive rebase
删除无关 dirty-worktree changes
提交 checkpoint
提交大型 raw token cache
```

建议 commit：

```text
Add final-token and positive-channel relation diagnostics
```

结果 commit 可选：

```text
Record final-token relation analysis
```

---

# 28. Codex 执行 Prompt

```text
Work directly in ~/code/TGCA on the current main checkout.

You are explicitly authorized to:
- inspect and modify analysis code;
- locate/load the existing native MCTformer+ Small seed-0 checkpoint;
- run frozen full-VOC-val GPU inference;
- add tests;
- generate compact reports/tables;
- git commit the analysis implementation and compact results without asking me again.

Do NOT train or fine-tune any model.
Do NOT git push.
Do NOT use destructive git operations.
Preserve unrelated dirty-worktree changes.

This task explicitly abandons PatchFinalLN.

Use only native/original MCTformer+ raw post-Block-12 tokens:

    C_raw: [B,20,384]
    P_raw: [B,784,384] at 448 input

Required config:
    final_norm=false
    patch_final_norm=false
    last_mct=false
    MCTformer+ Small
    BCSS E0
    PSL baseline
    CTI-BGT disabled
    vanilla attention

Abort if the checkpoint is not the native matched baseline.

EXPERIMENT E1 — FINAL-TOKEN AFFINITY

Define:

    S_last[c,i] = c_c^T p_i / sqrt(D)

using raw final class and patch tokens.

This is final-token compatibility, not Transformer attention.

Evaluate on VOC val with the same GT patch-region protocol as Experiment 2:

- target-vs-BG AUROC/AUPRC;
- target-vs-other-FG AUROC/AUPRC;
- top-5/10/20% target/other/BG composition;
- Target/OtherFG/BG TailEnrich;
- maximum-patch ownership;
- spatial-softmax A_last target/other/BG mass;
- multi-label positive-class map Spearman/top-tail Jaccard;
- shared-support ownership/dominant-object capture;
- present-vs-absent controls.

Also compare:
    raw dot
    cosine(c,p)
    patch norm

to determine whether dot-product semantic behavior comes from direction or norm.

Use the native 3x3 classifier relevance R_M only as a frozen reference.
Do not fuse it with S_last.
Do not inspect per-layer Transformer A_c2p in this task.

Verify S_last invariance under:
- channel permutation;
- signed permutation;
- 5 Haar orthogonal transforms.

Generate:
    FINAL_TOKEN_AFFINITY_REPORT.md

EXPERIMENT E2 — POSITIVE-CHANNEL RELATION

Native class-token classification is:

    z_cls[c] = mean_d(c_c[d])

Define:

    P_c = {d | c_c[d] > 0}
    c_pos = ReLU(c_c)

    S_pos[c,i] = ReLU(c_c)^T p_i / sqrt(D)

Before evaluating S_pos, analyze:

- fraction of positive channels;
- positive contribution mass;
- negative contribution mass;
- verify D*z = positive_mass - negative_mass;
- GT-positive vs absent classes;
- positive-channel mask overlap across positive classes;
- same-class cross-image consistency vs different-class consistency.

Evaluate S_pos with the same semantic metrics as S_last.

Directly compare S_pos vs S_last for:
- target-vs-BG;
- target-vs-other;
- Target-TailEnrich;
- BG-TailEnrich;
- positive-class top-10% Jaccard.

Also analyze patch-token values on positive class-token coordinates:

    U_pos[c,i] =
        mean_{d in P_c} p_i[d]

    S_pos[c,i] =
        sum_{d in P_c} c_c[d] * p_i[d]

    C_sign[c,i] =
        fraction_{d in P_c}(p_i[d] > 0)

Compare target / other-FG / background distributions.

Decompose:

    c = ReLU(c) - ReLU(-c)

    S_last = S_pos - S_negmag

and analyze all three terms by GT region.

Quantify S_pos basis dependence.

Under:
    c'=cQ
    p'=pQ

rewrite the fixed mean readout as:
    z = c^T w
    w = ones(D)/D
    w' = Q^T w

Verify class logits remain unchanged and S_last remains unchanged.

Then measure S_pos changes under:
- ordinary channel permutation;
- signed permutation;
- 5 Haar orthogonal transforms.

Ordinary permutation should preserve S_pos; signed/general rotations need not.

Also stratify GT-positive classes by native class-token confidence and test
whether S_pos is more semantic when the class-token classifier is confident.

STATISTICS

Use:
    whole image = bootstrap cluster
    5,000 paired resamples
    fixed seed
    95% percentile CI

Report:
    micro
    macro-class
    per-class
    single-label
    exactly-2-label
    3+-label

EXECUTION ORDER

1. Audit native checkpoint.
2. Implement/test raw-final-token extraction.
3. Run E1.
4. Write E1 report.
5. Implement E2 channel diagnostics.
6. Run E2.
7. Write E2 report.
8. Commit compact code/results if appropriate.
9. Stop.

Do not implement:
- PatchFinalLN;
- graph low-pass;
- new selector;
- new aggregation;
- new loss;
- training;
- per-layer A_c2p analysis.

Stop after these two frozen representation diagnostics.
```
