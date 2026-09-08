# Frozen Multi-Class Token Relation Analysis

## 1. 研究目标

本实验只研究 **原始 native MCTformer+ 已经学到的 multi-class token / patch token relation**。

已知前提：

1. LaST-ViT 的 channel Fourier + low-pass selector 在 MCTformer+ 上已经验证为负面或无效。
2. `stable patch ≠ positive-semantic patch`，尤其在 multi-label WSSS 中，intrinsic patch stability 无法判断 patch 属于哪个 positive class。
3. 已观察到不同 class tokens 会共同高激活一部分 patches，说明 multi-class tokens 中存在明显的 shared-presence/common component。
4. 这个 shared-presence 现象不能只研究最后一层，需要分析 MCTformer+ 全部 12 个 Transformer blocks 的演化。
5. 当前阶段不训练新模型、不改变 MCTformer+、不增加 loss、不修改 attention。
6. 完全放弃 PatchFinalLN。
7. 本轮不使用 LaST Fourier / low-pass。

因此本实验分成两个任务：

\[
\boxed{
\text{Task A: 12-layer Shared-Presence Analysis}
}
\]

和

\[
\boxed{
\text{Task B: Frozen Final-Layer Relation/Selector Comparison}
}
\]

二者使用同一个 frozen native MCTformer+ checkpoint。

---

# 2. Hard Constraints

Codex 必须严格遵守以下约束。

### 模型

使用当前已经训练好的原始 MCTformer+-Small checkpoint。

模型必须为：

```text
MCTformer+-Small
embed_dim = 384
depth = 12
num_classes = 20
input = 448
native joint-token path
```

必须确认：

```text
final_norm = False
patch_final_norm = False
last_mct = False
BCSS = E0 / baseline
PSL = baseline
CTI-BGT = disabled
attention = native/vanilla configuration of the checkpoint
```

禁止为了本实验重新训练 checkpoint。

---

### 推理

必须：

```python
model.eval()

with torch.no_grad():
    ...
```

所有 relation calculation 强制转换到 `float32`。

禁止：

- optimizer；
- backward；
- new loss；
- learnable projection；
- prototype MLP；
- new token；
- fine-tuning；
- LaST FFT；
- low-pass；
- PatchFinalLN。

---

### 数据

沿用之前 Experiment 1 完全相同的：

- PASCAL VOC split；
- checkpoint；
- image list；
- preprocessing；
- input size 448；
- segmentation GT；
- image-level labels。

不要创建新的 evaluation subset。

保存 checkpoint SHA256、git commit 和完整运行配置。

---

# 3. 表征定义

对于第 \(l\) 层 Transformer block：

\[
l=1,\ldots,12,
\]

获取该 block 的原始 post-block output：

\[
X^{(l)}
=
[
C^{(l)},P^{(l)}
].
\]

其中：

\[
C^{(l)}
\in
\mathbb R^{B\times20\times384},
\]

\[
P^{(l)}
\in
\mathbb R^{B\times784\times384}.
\]

注意：

> 使用 block 输出的 raw representation。

不能额外使用：

\[
LN(C^{(l)})
\]

或者：

\[
LN(P^{(l)}).
\]

尤其最终层必须是：

\[
C^{12},P^{12}
\]

而不是 PatchFinalLN representation。

---

# 4. Task A — 全 12 层 Shared-Presence Analysis

## 4.1 核心研究问题

我们已经观察到：

> 不同 class tokens 会共同选择/激活一些相同 patches。

现在需要回答：

### Q1

shared activation 从第几层开始出现？

### Q2

它是否随着网络加深越来越强？

### Q3

这种 shared relation 是否主要来自一个简单 common vector：

\[
\mu_C^{(l)}
=
\frac1C
\sum_{c=1}^{C}
c_c^{(l)}
\]

还是存在更高维的 shared presence subspace？

### Q4

shared patches 更倾向于：

- target foreground；
- other positive-class foreground；
- general foreground；
- background/context？

### Q5

去除 simple common mode 后：

\[
c_c-\mu_C
\]

是否已经显著降低不同 class maps 的 collision？

---

# 5. Layer-wise Relation Decomposition

首先定义 raw dot relation：

\[
R_{c,i}^{(l)}
=
\frac{
(c_c^{(l)})^\top p_i^{(l)}
}{
\sqrt D
}.
\]

其中：

\[
D=384.
\]

对于每张图，在每一层计算：

\[
\mu_C^{(l)}
=
\frac1{20}
\sum_{c=1}^{20}
c_c^{(l)}.
\]

定义 shared/common patch map：

\[
\boxed{
G_i^{(l)}
=
\frac{
(\mu_C^{(l)})^\top p_i^{(l)}
}{
\sqrt D
}
}
\]

以及 class-specific residual relation：

\[
\boxed{
D_{c,i}^{(l)}
=
R_{c,i}^{(l)}
-
G_i^{(l)}
}
\]

因为：

\[
c_c
=
\mu_C
+
(c_c-\mu_C),
\]

所以：

\[
R_{c,i}
=
G_i
+
D_{c,i}.
\]

这是一个 **严格的 additive relation decomposition**。

不要把 \(G\) 直接称为最终的“presence subspace”。

当前只称：

```text
common-mode relation
shared relation component
```

后面的 PCA 才研究它是否构成低维 subspace。

---

# 6. Layer-wise Shared-Presence Metrics

每层必须至少计算以下指标。

## 6.1 Cross-Class Relation Correlation

对于同一 image 中所有 GT-positive classes：

\[
c,k\in\mathcal Y^+,
\qquad
c\neq k,
\]

计算：

\[
Corr
(
R_c^{(l)},
R_k^{(l)}
).
\]

统计：

```text
Raw Pairwise Correlation
```

然后对 residual relation：

\[
Corr
(
D_c^{(l)},
D_k^{(l)}
)
\]

统计：

```text
Centered Pairwise Correlation
```

关键结果：

\[
Corr(R)
\quad vs\quad
Corr(D).
\]

如果：

\[
Corr(D)\ll Corr(R),
\]

说明 shared common component 确实解释了相当一部分 cross-class activation collision。

---

# 7. Common-Map Explanation Ratio

对于每个 class relation map：

\[
R_c^{(l)}
\]

用 shared map：

\[
G^{(l)}
\]

进行一维线性 regression：

\[
R_c
=
a_cG+b_c+\epsilon.
\]

保存：

\[
R^2_{\text{common}}.
\]

这一指标回答：

> 一个 class-specific patch relation map 中，有多少 spatial variation 可以仅通过所有 class tokens 的 common component 解释？

对每层统计：

\[
\operatorname{mean}_{x,c}
R^2_{\text{common}}.
\]

绘制：

```text
Layer 1 → Layer 12
```

的变化。

---

# 8. Top-K Cross-Class Collision

对于每个 positive class：

\[
T_c^{(l)}(q)
=
Top_q(R_c^{(l)}),
\]

其中固定：

\[
q\in
\{1\%,5\%,10\%\}.
\]

计算两个 positive classes 的：

\[
Jaccard
=
\frac{
|T_c\cap T_k|
}{
|T_c\cup T_k|
}.
\]

得到：

```text
Raw Top-K Collision
```

再对 residual：

\[
Top_q(D_c)
\]

计算：

```text
Residual Top-K Collision
```

重点比较：

\[
Collision_R^{(l)}
\quad vs\quad
Collision_D^{(l)}.
\]

不要调 K。

三个固定比例全部报告。

---

# 9. Shared Map 到底激活什么？

对于：

\[
G^{(l)}
\]

取 Top 5% patches。

根据 VOC segmentation GT 分成：

### Background

\[
GT_i=0
\]

### Any Foreground

\[
GT_i>0
\]

进一步在 multi-label image 中报告：

```text
foreground ratio
background ratio
```

不要人为把 common map 指派给某一个 class。

这个分析回答：

> shared component 更像 object/presence signal，还是 background/context signal？

---

# 10. Positive-Class Shared Map

另外计算一个 **仅用于 diagnostic** 的量：

\[
\mu_+^{(l)}
=
\frac1{|\mathcal Y^+|}
\sum_{c\in\mathcal Y^+}
c_c^{(l)}.
\]

以及：

\[
G_+^{(l)}
=
(\mu_+^{(l)})^\top P^{(l)}.
\]

只在：

\[
|\mathcal Y^+|\ge2
\]

的图像中分析。

重要：

> \(G_+\) 使用 GT label，因此绝对不能作为 selector 或未来 inference module。

这里只用于回答：

> shared component 是否在当前真正存在的多个 classes 之间尤其明显？

最终比较：

\[
G_{\text{all}}
\quad vs\quad
G_+.
\]

---

# 11. Presence Subspace PCA

对于每张图片、每层保存：

\[
\mu_C^{(l)}(x)
\in\mathbb R^{384}.
\]

构造：

\[
M^{(l)}
=
[
\mu_C^{(l)}(x_1);
\ldots;
\mu_C^{(l)}(x_N)
].
\]

对每层独立做 PCA/SVD。

报告：

```text
PC1 explained variance
PC2 explained variance
PC4 cumulative variance
PC8 cumulative variance
PC16 cumulative variance
r90
r95
participation ratio / effective rank
```

这一步才真正研究：

\[
\boxed{
\text{shared presence 是否形成低维 subspace}
}
\]

---

# 12. Shared Subspace 的 Layer Evolution

对于每层得到：

\[
U_r^{(l)}.
\]

如果 PCA 显示存在明显低维结构，再计算相邻层：

\[
U_r^{(l)}
\quad vs\quad
U_r^{(l+1)}
\]

的 principal-angle / subspace similarity。

目标是回答：

> shared presence subspace 是早期形成后保持稳定，还是在后几层重新组织？

至少报告：

```text
layer 1 → 2
2 → 3
...
11 → 12
```

的 subspace similarity。

本轮不能因为 PCA 结果而自动创建新的 selector。

**PCA/SVD 在 Task A 中只用于诊断。**

---

# 13. Task B — 最终层 Frozen Relation/Selector Comparison

Task B 只使用：

\[
C^{12},P^{12}.
\]

严格比较 **6 个** relation/selector。

这里不再使用 shared-subspace PCA projection。

原因：

> 在 Task A 完整确认 shared presence 的层间结构之前，不应该提前假定存在一个固定可删除的 low-rank subspace。

---

# 14. S0 — Classifier-Only

使用已经完成的 Classifier-first experiment 中完全相同的 spatial classifier score：

\[
\boxed{
S^{(0)}_{c,i}=M_{c,i}
}
\]

不要重新定义 classifier score。

直接复用原实验中送入 selector 之前的：

```text
class-specific spatial patch score
```

S0 是 reference baseline。

---

# 15. S1 — Raw Cosine Relation

使用已有：

```python
class_specific_patch_score()
```

的定义：

\[
\boxed{
S^{(1)}_{c,i}
=
\cos(c_c^{12},p_i^{12})
}
\]

即：

\[
\frac{
c_c^\top p_i
}{
\|c_c\|_2\|p_i\|_2
}.
\]

不加 LN。

---

# 16. S2 — Raw Token Product

对应我们之前定义的：

\[
A_{c2p}^{last}.
\]

定义：

\[
\boxed{
S^{(2)}_{c,i}
=
\frac{
c_c^\top p_i
}{
\sqrt D
}
}
\]

其中：

\[
D=384.
\]

这是 direct final-token product。

注意：

> 这里不是 Transformer 内部 attention matrix 中的 \(A_{c2p}\)。

它纯粹是最终输出 token representation 的直接 relation。

---

# 17. S3 — Class-Token Centered Relation

定义所有 class tokens 的 image-conditioned mean：

\[
\mu_C
=
\frac1{20}
\sum_{k=1}^{20}
c_k.
\]

然后：

\[
\boxed{
S^{(3)}_{c,i}
=
\frac{
(c_c-\mu_C)^\top p_i
}{
\sqrt D
}
}
\]

这个实验只回答一个问题：

> 删除简单 additive common mode 是否能够提高 class specificity？

它不是 learned decomposition。

没有新参数。

---

# 18. S4 — Relative Class Ownership

首先使用 S2 的 raw dot relation：

\[
q_{c,i}
=
S^{(2)}_{c,i}.
\]

定义：

\[
\boxed{
S^{(4)}_{c,i}
=
q_{c,i}
-
\log
\left[
\frac1{C-1}
\sum_{k\neq c}
e^{q_{k,i}}
\right]
}
\]

即 temperature 固定：

\[
\tau=1.
\]

必须使用稳定的 `logsumexp` implementation。

不做 temperature sweep。

不使用 GT label。

denominator 使用全部其他 19 个 class tokens。

这个 selector 测试：

> relative cross-class relation 能否通过 common-mode rejection 自动削弱 shared presence。

---

# 19. S5 — Classifier + Relative Ownership

S5 测试：

> multi-class-token ownership 是否能够补充已经更可靠的 classifier-first spatial evidence？

首先分别对 patch dimension 做 per-class spatial min-max：

\[
\tilde M_{c,i}
=
MinMax_i(M_{c,i}),
\]

\[
\tilde O_{c,i}
=
MinMax_i(S^{(4)}_{c,i}).
\]

复用：

```python
analysis.semantic_relations.spatial_minmax
```

然后：

\[
\boxed{
S^{(5)}_{c,i}
=
\tilde M_{c,i}
+
\tilde O_{c,i}
}
\]

不增加 \(\beta\)。

不 sweep mixing coefficient。

两项固定 equal contribution。

本轮只评价：

\[
S^{(5)}
\]

作为 selector ranking。

---

# 20. 为什么只有这 6 项

最终固定：

| ID | Relation / Selector | 使用 multi-class token | common-mode mechanism |
|---|---|---:|---|
| S0 | Classifier only | No new use | No |
| S1 | Cosine \(c_c,p_i\) | Yes | No |
| S2 | Dot \(c_c^\top p_i\) | Yes | No |
| S3 | Centered dot | Yes | Mean removal |
| S4 | Relative ownership | Yes | Cross-class cancellation |
| S5 | Classifier + ownership | Yes | Cross-class cancellation + spatial evidence |

暂时不要加入：

```text
LaST stability
Fourier
low-pass
graph stability
shared-subspace projection selector
learned prototype
OT/Sinkhorn
contrastive loss
positive-channel relation
new class-token loss
```

其中 shared-subspace projection 必须等待 Task A 结果。

Positive-channel relation 保留为之后独立 diagnostic，不扩大本轮 6 项主矩阵。

---

# 21. Ground-Truth Region Definition

对于每个 image-level GT-positive class \(c\)，patch 分成三类：

## Target Foreground

\[
\mathcal P_c
=
\{i:GT_i=c\}
\]

## Other Positive Foreground

\[
\mathcal P_{\text{other}}
=
\{i:GT_i>0,\ GT_i\neq c\}
\]

## Background

\[
\mathcal P_{\text{bg}}
=
\{i:GT_i=0\}.
\]

Ignore：

\[
GT_i=255.
\]

segmentation GT 必须 nearest-neighbor resize 到：

\[
28\times28.
\]

---

# 22. 最关键的数据切分

必须分别统计：

### Single-label images

\[
|\mathcal Y^+|=1
\]

### Multi-label images

\[
|\mathcal Y^+|\ge2.
\]

本实验的最主要结果是：

\[
\boxed{
\text{multi-label subset}
}
\]

因为真正需要解决的是：

\[
\boxed{
target\ class
\quad vs\quad
other\ positive\ class
}
\]

而不是简单：

\[
foreground
\quad vs\quad
background.
\]

---

# 23. Primary Selector Metrics

对于每个 selector \(S^{(j)}\)，必须计算：

## Target vs Other Positive

\[
AUROC_
{
target/other
}
\]

和：

\[
AP_
{
target/other
}.
\]

这是 **最重要指标**。

只在：

\[
\mathcal P_{\text{other}}\neq\emptyset
\]

时计算。

---

## Target vs Background

计算：

\[
AUROC_{target/bg},
\]

\[
AP_{target/bg}.
\]

---

## Target vs All Non-target

\[
\mathcal P_{\neg c}
=
\mathcal P_{\text{other}}
\cup
\mathcal P_{\text{bg}}.
\]

计算：

\[
AUROC_{target/all},
\]

\[
AP_{target/all}.
\]

---

# 24. Score Separation

保存：

\[
\Delta_{\text{other}}
=
E[S|\mathcal P_c]
-
E[S|\mathcal P_{\text{other}}],
\]

以及：

\[
\Delta_{\text{bg}}
=
E[S|\mathcal P_c]
-
E[S|\mathcal P_{\text{bg}}].
\]

其中：

\[
\boxed{
\Delta_{\text{other}}
}
\]

优先级高于：

\[
\Delta_{\text{bg}}.
\]

因为这是 LaST intrinsic selector 无法解决的问题。

---

# 25. Top-K Semantic Purity

固定：

\[
q=
1\%,5\%,10\%.
\]

对每个 target class：

\[
T_c(q)=Top_q(S_c).
\]

分别计算：

### Target Precision

\[
P_{target}@q
=
\frac{
|T_c\cap\mathcal P_c|
}{
|T_c|
}.
\]

### Other-positive contamination

\[
P_{other}@q
=
\frac{
|T_c\cap\mathcal P_{\text{other}}|
}{
|T_c|
}.
\]

### Background contamination

\[
P_{bg}@q
=
\frac{
|T_c\cap\mathcal P_{\text{bg}}|
}{
|T_c|
}.
\]

必须满足：

\[
P_{target}
+
P_{other}
+
P_{bg}
\approx1
\]

忽略 void patches 后。

---

# 26. Cross-Class Collision

在 multi-label images 中，对于两个 positive classes：

\[
c,k\in\mathcal Y^+
\]

计算：

\[
Jaccard
(
Top_q(S_c),
Top_q(S_k)
).
\]

目标：

\[
\boxed{
\text{一个好的 ownership selector 应降低不同 positive classes 对同一 patch set 的重复占有}
}
\]

重点比较：

```text
S1 cosine
S2 dot
S3 centered dot
S4 relative ownership
S5 fused
```

---

# 27. Aggregation 方式

所有 dataset-level metrics 以：

```text
image × positive-class
```

为基本 observation。

主报告使用 macro average。

不要把所有 patches 直接 pool 成一个 micro metric，因为：

- 大 object；
- 大图片；
- patch 数较多的类别

不能在统计上拥有更高权重。

同时保存：

```text
per-image-class.csv
```

便于之后 bootstrap。

---

# 28. Statistical Testing

复用已有：

```text
analysis/lazy_assignment/bootstrap.py
```

做 paired bootstrap。

固定：

```text
seed = 2027
resamples = 10000
```

所有 selector 都和 S0 做 paired comparison。

关键报告：

\[
\Delta AUROC_{target/other}
\]

\[
\Delta AP_{target/other}
\]

\[
\Delta P_{target}@5\%
\]

\[
\Delta Collision@5\%.
\]

报告：

```text
mean delta
95% confidence interval
```

---

# 29. Task A 的实现方式

不要 dump 全部 12 层 patch tokens 到磁盘。

数据量没有必要。

现有 `BlockTokenCollector` 已经通过 forward hook 获取每个 block 的 post-block token。

新建独立模块：

```text
analysis/relational_selector/
```

不要破坏已有 Experiment 1。

建议文件：

```text
analysis/relational_selector/
    __init__.py
    relations.py
    layer_collector.py
    dump_frozen_relations.py
    shared_presence_metrics.py
    selector_metrics.py
    analyze_shared_presence.py
    analyze_selectors.py
    generate_report.py
    test_relations.py
```

---

# 30. Collector 设计

参考现有：

```text
analysis/lazy_assignment/token_collector.py
```

创建新的：

```python
LayerRelationCollector
```

每层 hook 只需要即时计算：

\[
R^{dot}
\]

\[
R^{cos}
\]

\[
G
\]

\[
D
\]

以及：

\[
\mu_C.
\]

不要把：

\[
[B,784,384]\times12
\]

完整长期保存。

每个 batch 消费完 layer statistics 后立即释放。

---

# 31. 最终层额外保存

第 12 层保存：

```text
S0
S1
S2
S3
S4
S5
```

对应每个 image：

\[
[6,20,28,28].
\]

建议存成：

```text
float32
```

的 per-image compressed `.npz`。

同时保存：

```text
image_id
image-level label
grid_size
```

不要在 dump 阶段保存 segmentation GT。

GT matching 放在 analysis 阶段进行。

---

# 32. PCA 数据

对于每层、每图，只保存：

\[
\mu_C^{(l)}
\]

因此数据只有：

\[
N\times12\times384.
\]

完全足够进行 shared-presence PCA。

不要为了 PCA dump 全部 patch tokens。

---

# 33. Unit Tests

在真正跑 VOC 前，必须实现 synthetic tests。

### Test 1 — Shape

确认：

```text
C: [B,20,384]
P: [B,784,384]

S0-S5:
[B,20,784]
```

---

### Test 2 — Common-Shift Invariance of Centering

人工：

\[
c'_c=c_c+g.
\]

验证：

\[
(c'_c-\mu'_C)^\top p
=
(c_c-\mu_C)^\top p.
\]

误差：

```text
< 1e-5
```

---

### Test 3 — Common-Shift Invariance of Relative Ownership

人工：

\[
q'_{c,i}
=
q_{c,i}
+
g^\top p_i.
\]

必须验证：

\[
Ownership(q')
=
Ownership(q).
\]

误差：

```text
< 1e-5
```

这是 S4 最关键的数学 sanity check。

---

### Test 4 — No GT Leakage

S0–S5 的 score construction API 中：

```text
不得出现 segmentation GT
不得出现 image-level GT label
```

GT 只能进入 metrics。

---

### Test 5 — No Gradient

forward 完成后确认：

```python
all(p.grad is None for p in model.parameters())
```

且 checkpoint parameters before/after hash 一致。

---

# 34. Smoke Run

先运行：

```text
8–16 images
```

检查：

- 12 layers 全部被 hook；
- C=20；
- P=784；
- D=384；
- S0–S5 无 NaN/Inf；
- shared metrics 正常；
- segmentation resize 正确；
- multi-label region mask 正确；
- outputs 可重复。

固定同一 seed 连续运行两次。

关键 CSV 必须 bitwise/deterministically 一致，允许 GPU kernel 导致的极小浮点误差时至少在 `1e-6` 内一致。

---

# 35. Full Run

Smoke 通过后运行完整 VOC evaluation split。

必须一次 forward 同时生成 Task A 和 Task B 所需数据。

禁止为了 6 个 selector 分别跑 6 次 model forward。

---

# 36. 输出目录

建议：

```text
experiments/frozen_relational_selector/
```

结构：

```text
run_metadata.json
checkpoint.json

shared_presence/
    layer_summary.csv
    common_map_metrics.csv
    collision_by_layer.csv
    pca_summary.csv
    subspace_similarity.csv
    by_label_count.csv

selectors/
    per_image_class.csv
    summary.csv
    multi_label_summary.csv
    single_label_summary.csv
    by_class.csv
    topk_summary.csv
    collision_summary.csv
    bootstrap.json

samples/
    *.npz

report.md
```

---

# 37. Task A 最终必须生成的 Layer Table

核心表格格式：

| Layer | Raw corr | Residual corr | Raw collision@5 | Residual collision@5 | Common \(R^2\) | Common FG@5 | PCA PC1 | PCA r95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|

必须包含 Layer 1–12。

这个表是研究 shared presence evolution 的主要结果。

---

# 38. Task B 最终主表

必须生成：

| Selector | T-vs-Other AUROC | T-vs-Other AP | T-vs-BG AUROC | Target@5 | Other@5 | BG@5 | Collision@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 Classifier | | | | | | | |
| S1 Cosine | | | | | | | |
| S2 Dot | | | | | | | |
| S3 Centered Dot | | | | | | | |
| S4 Relative Ownership | | | | | | | |
| S5 Classifier + Ownership | | | | | | | |

其中主排序依据不是 BG suppression。

首先看：

\[
\boxed{
Target\ vs\ Other\ Positive
}
\]

其次才看：

\[
Target\ vs\ Background.
\]

---

# 39. 结果解释规则

Codex 不要只输出数字，需要根据以下 hypothesis 自动写 `report.md`。

## H1 — Shared presence 是否真实存在

支持条件：

\[
Corr(R)>Corr(D)
\]

且：

\[
Collision(R)>Collision(D)
\]

在多个后层稳定出现。

---

## H2 — shared presence 是否逐层增强

检查：

```text
common R²
cross-class correlation
Top-K collision
```

从 Layer 1 到 Layer 12 的变化。

不要强制要求单调。

重点寻找：

```text
early / middle / late layer transition
```

---

## H3 — shared presence 是否低维

如果：

```text
PC1/PC4 explained variance 高
r95 明显小于 D=384
adjacent-layer principal subspaces 相对稳定
```

则下一阶段可以研究：

\[
\text{explicit shared-subspace residualization}.
\]

否则不要假设存在固定的低秩 presence basis。

---

## H4 — Centering 是否足够

比较：

\[
S2
\quad vs\quad
S3.
\]

如果 S3 显著提高：

\[
Target\ vs\ Other
\]

并降低 cross-class collision：

说明 simple common vector removal 已经有效。

---

## H5 — Relative ownership 是否比显式 centering 更好

比较：

\[
S3
\quad vs\quad
S4.
\]

如果：

\[
S4>S3,
\]

尤其是在 multi-label subset：

说明：

\[
\boxed{
cross-class relative calibration
}
\]

比简单 mean subtraction 更适合 multi-class token geometry。

---

## H6 — Multi-class token 是否真正补充 classifier-first evidence

比较：

\[
S0
\quad vs\quad
S5.
\]

最关键结果是：

\[
Target\ vs\ Other
\]

改善，同时：

\[
Target\ vs\ BG
\]

不能明显恶化。

如果成立，则说明：

\[
\boxed{
multi-class token relation
}
\]

提供了 classifier spatial score 中没有的：

\[
\boxed{
semantic ownership information.
}
\]

---

# 40. Go / No-Go Criteria

下一阶段只有在以下结果出现时才设计新的 trainable selector。

## Go: Relative Ownership

如果 S4 相比 S1/S2：

- target-vs-other AUROC/AP 提升；
- cross-class Top-K collision 降低；
- multi-label improvement 明显强于 single-label；

则继续研究 relative semantic ownership。

---

## Go: Classifier + Ownership

如果 S5 相比 S0：

\[
\Delta AUROC_{target/other}>0
\]

且 paired-bootstrap 95% CI 不跨 0，同时：

\[
AUROC_{target/bg}
\]

没有明显下降，则进入真正 selector integration 实验。

---

## Go: Explicit Presence Subspace

只有 Task A 显示：

- presence PCA 明显低秩；
- 多层具有稳定 subspace；
- simple centering 确实改善 class specificity；

才能继续研究：

\[
P_{\perp U_{presence}}c_c
\]

这一类显式 residualized relation。

---

# 41. 当前阶段禁止得出的结论

即使结果很好，也不能直接声称：

```text
multi-class tokens are semantic prototypes
```

目前更准确的说法应该是：

```text
Multi-class tokens contain class-conditioned relational information,
but their raw patch relations are partially contaminated by a
shared/common presence component.
```

如果 S4/S5 有效，可以进一步写：

```text
Relative cross-class calibration suppresses common-mode activation
and improves patch-level semantic ownership.
```

只有后续训练和完整 WSSS segmentation 实验完成后，才能进一步讨论新 selector 方法。

---

# 42. Codex 最终交付物

Codex 完成任务后必须提供：

1. 所有新增/修改文件列表；
2. git diff summary；
3. unit-test 输出；
4. smoke-run command 和结果；
5. full-run command；
6. checkpoint SHA256；
7. Layer 1–12 shared-presence 主表；
8. S0–S5 主比较表；
9. single-label / multi-label 分表；
10. paired-bootstrap 结果；
11. `report.md`；
12. 对以下三个问题给出明确结论：

### Question A

shared activation 在 MCTformer+ 的哪几层开始明显形成？

### Question B

它更像一个简单 rank-1/common component，还是明显的 higher-rank shared presence subspace？

### Question C

在完全冻结模型的情况下：

\[
\boxed{
classifier
+
multi\text{-}class\ relative\ ownership
}
\]

是否比：

\[
classifier-only
\]

更能区分：

\[
\boxed{
target\ positive\ class
\quad vs\quad
other\ positive\ classes?
}
\]

这是本轮实验的最终核心问题。