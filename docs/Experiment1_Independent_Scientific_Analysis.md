# Experiment 1 独立科学分析
## MCTformer / MCTformer+ Class-specific Patch Score

**分析对象：** Codex 整理的 `20260902-mctformer-paired-full-f05d15b` 结果包  
**数据：** PASCAL VOC 2012 val，1,449 张图像，2,147 个正类 image–class pairs  
**Score：**

\[
S^{(l)}_{c,j}=\cos(c^{(l)}_c,p^{(l)}_j)
\]

其中 hidden states 取自每个 Transformer block 之后、最终 LayerNorm 之前。

---

## 1. 总结判断

Experiment 1 的工程质量很高，可以进入 Experiment 2：

- 两个模型均覆盖 1,449/1,449 张 VOC val 图像和 2,147/2,147 个正类 pair；
- common pair 覆盖率 100%；
- 没有缺失、重复、label mismatch、NaN/Inf 或 cosine 越界；
- 2,898 个 NPZ 在分析前后 hash 不变；
- canonical spot check 共 2,016 项，最大误差为 0；
- 123 个测试通过；
- 置信区间使用 image-clustered paired bootstrap，而不是把 patch 或同一图像中的多个类别错误地视为独立样本。

科学上，Experiment 1 **尚未证明 lazy semantic assignment**，但揭示了两个值得继续验证的强现象：

1. **MCTformer+ 在 L9→L10 出现明显的表示相变，之后 L10–L12 基本固定高分 patch 排名并持续放大 score 对比。**
2. **MCTformer+ 的不同正类别 maps 在早中层高度分离，却在 L9–L12 重新耦合到共同的高分 patch support；L12 的 top-10% overlap 非常高。**

这比“单纯更集中”更有研究价值。下一阶段真正需要回答的是：

> MCTformer+ 后三层重新共享的高分 patches，究竟属于各自正确目标、某一个占主导的共现对象，还是 background/context？

---

## 2. 对 Codex 报告的一处重要修正

Codex 报告将 MCTformer+ 概括为“更强 upper tail、更加集中、更加 rough”。其中第一部分在**绝对数值尺度**上成立，但不能全部视为独立的空间结构结论。

L12：

| 指标 | MCTformer | MCTformer+ | 倍数 |
|---|---:|---:|---:|
| Score std | 0.1068 | 0.3489 | 3.27× |
| q95−median | 0.1644 | 0.4307 | 2.62× |
| Total variation | 0.0761 | 0.2477 | 3.26× |

但是按 score spread 粗略归一化：

\[
\frac{q95-\mathrm{median}}{\mathrm{std}}
=1.539\quad\text{(MCTformer)},
\]

\[
\frac{q95-\mathrm{median}}{\mathrm{std}}
=1.235\quad\text{(MCTformer+)},
\]

以及：

\[
\frac{TV}{\mathrm{std}}
=0.712\quad\text{(MCTformer)},
\]

\[
\frac{TV}{\mathrm{std}}
=0.710\quad\text{(MCTformer+)}.
\]

L12 neighbor Spearman 也接近：

\[
0.4961\quad\text{vs.}\quad0.4818.
\]

因此更准确的说法是：

> **MCTformer+ 的 raw cosine maps 在后层发生强烈的幅度极化/方差膨胀；现有结果尚不能证明其标准化 upper-tail shape 更重，或空间拓扑本身更粗糙。**

固定温度 \(\tau=0.1\) 的 softmax entropy 同样会随着 score variance 增大而自动下降。因此 entropy 下降不能被单独当成“定位更集中”的独立证据。

建议补充：

- `(q95−median)/(std+eps)`；
- `(q95−q50)/(q75−q25+eps)`；
- `TV/(std+eps)`；
- 对每张 map 先 z-score 后再计算固定温度 entropy；
- rank-based spatial autocorrelation。

---

## 3. 主要发现一：MCTformer+ 在 L9→L10 出现后层相变

MCTformer+：

| Layer | Score std | q95−median | Entropy τ=.10 | TV |
|---:|---:|---:|---:|---:|
| 8 | 0.1102 | 0.1662 | 0.8809 | 0.0727 |
| 9 | 0.1143 | 0.1611 | 0.8771 | 0.0782 |
| 10 | 0.2239 | 0.2830 | 0.8168 | 0.1593 |
| 11 | 0.3047 | 0.3716 | 0.7583 | 0.2168 |
| 12 | 0.3489 | 0.4307 | 0.7387 | 0.2477 |

真正的断点在：

\[
\boxed{L9\rightarrow L10}
\]

Rank dynamics：

\[
\rho(S^{(9)},S^{(10)})=0.5267,
\]

\[
\rho(S^{(10)},S^{(11)})=0.8765,
\]

\[
\rho(S^{(11)},S^{(12)})=0.9879.
\]

对应 top-10% Jaccard：

\[
0.2750,\quad0.5442,\quad0.8510.
\]

这意味着：

1. L9→L10 仍发生明显 patch support 重选；
2. 到 L10 后，高分 patch 排名大体确定；
3. L11–L12 主要继续放大/极化同一批 patches，而不是重新寻找完全不同的位置。

Experiment 2 重点层应为：

\[
L4/5,\quad L9,\quad L10,\quad L11,\quad L12.
\]

---

## 4. 主要发现二：不同 class maps 在后层重新耦合

### MCTformer

| Layer | Pairwise Spearman | Top-10% Jaccard |
|---:|---:|---:|
| 1 | 0.9960 | 0.8920 |
| 8 | 0.8421 | 0.5282 |
| 9 | 0.4828 | 0.3279 |
| 10 | 0.0872 | 0.1563 |
| 11 | 0.2583 | 0.2191 |
| 12 | 0.3454 | 0.3533 |

MCTformer 的类别 maps 在早层几乎相同，L9–L10 才明显分离，L11–L12 又有一定回升。

### MCTformer+

| Layer | Pairwise Spearman | Top-10% Jaccard |
|---:|---:|---:|
| 1 | -0.0054 | 0.2085 |
| 4 | -0.0984 | 0.0588 |
| 5 | -0.0850 | 0.0547 |
| 9 | 0.2278 | 0.1988 |
| 10 | 0.3900 | 0.4048 |
| 11 | 0.4337 | 0.5084 |
| 12 | 0.4370 | 0.5458 |

对于 28×28 共 784 个 patches，每张 map 取 top 79 patches，两个随机独立集合的期望 Jaccard 约为：

\[
0.0531.
\]

所以 MCTformer+ L4–L5 的类别高分 support 接近随机独立；但 L12：

\[
0.5458
\]

约为随机基线的 10.3 倍。按集合大小反推，L12 两个正类别的 top-79 patches 平均约共享：

\[
55.8\text{ 个 patches},
\]

即每个 top set 的约 70.6%。MCTformer L12 约共享 41.2/79，即 52.2%。

因此 Experiment 1 最强、最不依赖 absolute score scale 的结果是：

\[
\boxed{
\text{MCTformer+ 先分离 class-specific supports，
后在 L9–L12 强烈重新共享同一批高分 patches。}
}
\]

这种 re-coupling 可能来自：

1. 合理共享前景区域；
2. 一个 dominant co-occurring object；
3. context/background semantic hubs；
4. class tokens 自身后层 collapse。

当前数据不能区分这些解释。

---

## 5. 主要发现三：高分区域有空间结构，但 Plus 不比 Base 更连贯

L12 top-10% mask：

| 指标 | MCTformer | MCTformer+ |
|---|---:|---:|
| Connected components | 30.73 | 31.07 |
| Largest component fraction | 0.2722 | 0.2589 |
| Neighbor Spearman | 0.4961 | 0.4818 |

两种模型的高分 patches 都明显不是随机散点；但 MCTformer+ 没有表现出更少 components、更大主连通块或更高邻域相关性。

因此不能写：

> MCTformer+ produces more spatially coherent regions.

更准确的是：

> MCTformer+ produces more polarized raw cosine scores, while the normalized spatial connectivity of its high-score support is broadly comparable to MCTformer.

---

## 6. Class-wise 异质性暴露了 raw cosine probe 的局限

MCTformer+ L12：

| Class | q95 | 估计 median | q95−median |
|---|---:|---:|---:|
| train | 0.495 | -0.144 | 0.639 |
| bottle | 0.567 | -0.021 | 0.588 |
| person | 0.733 | 0.329 | 0.404 |
| aeroplane | 0.008 | -0.353 | 0.361 |
| bird | -0.502 | -0.843 | 0.342 |

`bird` 连 q95 都是负值，而 `person` median 约为正 0.33。这说明：

\[
\boxed{
\text{raw cosine 的绝对正负不能跨类别统一解释为“语义强度”。}
}
\]

这与模型训练方式一致：

- class-token logit 使用 `x_cls.mean(-1)`；
- patch logit 使用独立 `Conv2d` head；
- class token 没有被训练成 patch feature 的 cosine prototype；
- 实际 attention 使用 LayerNorm 后的 learned Q/K projection。

所以 LaST 的 CLS–patch cosine在 MCTformer 中目前只是探索性 representation probe。

Experiment 2 应以 rank-based C-PiM、top-k region composition 和 signed AUROC 为主，不应使用固定 raw-cosine threshold。

若某些类别的 target-vs-BG AUROC < 0.5，不能立刻解释为背景泄漏；也可能说明 raw cosine probe 对该类别方向相反或无效。Experiment 3 的 QK logits / \(A_{c2p}\) 是必要判别证据。

---

## 7. 定性案例的意义

自动选择的案例显示：

- `train` 的 MCTformer+ L12 raw cosine 可能在 train body 上为强负，而在植被、天空或轨道周围为正；
- 某些 `person` 正类图像中，高分区域覆盖大型 aeroplane/地面，而 person 可能很小；
- `chair` 与 `tvmonitor` 的后层 maps 可以几乎相同。

它们与 other-foreground/context/background alignment 假设一致，但没有 GT overlay，因此只能作为 Experiment 2 的待验证样本。

禁止把红色直接解释成前景、蓝色直接解释成背景；raw cosine orientation 显著 class-dependent。

---

## 8. 当前对 lazy semantic assignment 的支持边界

### 已支持

1. 两模型的 class–patch representation geometry 随 depth 系统变化；
2. MCTformer+ 在 L9→L10 存在明显后层相变；
3. L10 后高分 patch ranking 固定，L11–L12 主要继续极化；
4. 多标签图像中，不同正类别的高分 patch supports 在 MCTformer+ 后层显著重新耦合；
5. 这些差异不是数据缺失、数值异常或抽样错误造成的。

### 尚未支持

1. 共享 patches 是 background；
2. 它们属于 other foreground；
3. 它们是 high-norm artifacts；
4. 它们进入 \(A_{c2p}\)；
5. 它们进入 CAM；
6. 分类因果依赖这些 patches；
7. MCTformer+ 的 lazy semantic assignment 比 MCTformer 更严重。

当前最合适的表述：

> Experiment 1 reveals a late-layer representational phase transition and cross-class re-coupling in MCTformer+, motivating a semantic-region analysis of the shared high-score patch supports.

---

## 9. Experiment 2 的最高优先级

### 9.1 基础 GT-region 指标

对全部 12 层、全部 20 类：

- target / other-FG / background distributions；
- C-PiM：target / other-FG / BG / mixed/void；
- top 5/10/20% patch region composition；
- area-normalized BG-Tail enrichment；
- target-vs-BG signed AUROC；
- target-vs-other signed AUROC。

### 9.2 Shared Top-Tail Ownership（最重要）

对 multi-label image 的正类 pair \((c_a,c_b)\)：

1. 取两张 map 的 top-10%；
2. 取交集；
3. 用 GT 把 shared patches 分成：class A、class B、other FG、background、mixed/void；
4. 逐层报告 ownership composition。

核心问题：

\[
\boxed{
\text{L10–L12 新增的 shared support 到底属于谁？}
}
\]

### 9.3 必须新增的控制

- 按 positive-class classification correctness / logit 分层；
- 逐层 class-token pairwise cosine，区分 token collapse 与 shared patch support；
- patch norm，区分 register-style artifact 与 low-norm semantic shortcut；
- relative class score：
  \[
  S_{c,j}-\max_{c'\neq c}S_{c',j}
  \]
  以及 active-class softmax；
- 两个 checkpoint 的 classification mAP 与 raw CAM mIoU；
- scale-normalized map-shape metrics。

---

## 10. 最终结论

Experiment 1 可以评为：

\[
\boxed{
\text{技术上可靠，科学上有明显新信号，适合进入 Experiment 2。}
}
\]

但它目前最强的信号不是“背景已经被证明获得 class semantics”，而是：

\[
\boxed{
\text{MCTformer+ 在 L9–L12 将早中层分离的 class-specific patch rankings
重新耦合到高度共享的高分 patch support。}
}
\]

这为 Background Semantic Ownership 提供了一个更具体、可检验的问题：

> 在 multi-label coarse supervision 下，后层 multi-class tokens 为什么共同指向同一批 patches？这些 patches 的真实 semantic owner 是目标类别、其他共现类别，还是 background？

Experiment 2 应首先用 GT 回答这一问题。
