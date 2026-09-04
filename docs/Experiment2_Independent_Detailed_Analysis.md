# Experiment 2 独立详细分析与研究讨论

## 1. 一句话结论

这组结果**不支持**把 MCTformer+ 的主要问题概括为“普遍的 background lazy semantic assignment”。更准确的结论是：

\[
\boxed{
\text{LaST-style raw hidden cosine 在 MCTformer+ 中不是实际 attention routing 的可靠代理；}
}
\]

同时：

\[
\boxed{
\text{MCTformer+ 的实际 }QK/A_{c2p}\text{ 在 L10 最具类别特异性，随后在 L11–L12 发生明显的跨类别重新耦合。}
}
\]

而 GT 语义归属分析表明，这些后层共享 support **总体并未对背景过度富集**；它们更多落在两个真实正类别中的某一个，且往往被其中一个 dominant target 所主导。因此，当前最有证据的瓶颈是：

\[
\boxed{
\text{late-layer cross-class semantic ownership ambiguity / dominant-object capture}
}
\]

而不是一个普遍的 foreground-vs-background failure。

Background 仍然是必要的候选 owner 和困难子集问题，但不应再是唯一主语。

---

## 2. 数据与实现可信度

结果覆盖 VOC val 全部 1,449 张图像、2,147 个正类 image–class pairs，以及 522 张多标签图像中的 906 个正类别 pairs。两模型覆盖率为 100%；没有缺失、重复、label mismatch、NaN/Inf 或 cosine 越界。Experiment 1 feature score 复算最大误差为 0；QK 重建 attention 最大误差为 0；外部拆分的 native CAM 与模型输出最大误差分别为 \(9.54\times10^{-7}\) 和 \(4.77\times10^{-7}\)。全流程检查 6,071 个不可变输入文件，无 hash 变化。统计使用 5,000 次 image-clustered bootstrap。

需要记录的边界情况：

- 一张图存在 raw mask/image-level label mismatch；
- 77 个正类 pairs 在 matched crop 后没有 target pixels；
- 116/121 个 pairs 在 \(\rho=.5/.7\) 下没有 target-dominant patch；
- 报告同时提供 target-visible control，关键 AUC 与主结果几乎相同。

因此，主要结论不是由 crop-invisible pairs 驱动的。

---

## 3. Experiment 1 的核心解释需要修正

Experiment 1 发现 MCTformer+ 的 raw post-block cosine maps：

\[
S^{feat}_{c,j}=\cos(c_c,p_j)
\]

在后层明显极化，并且不同正类别的 top support 重新重叠。Experiment 2 显示，这个现象不能直接解释成实际 class attention 或背景泄漏。

MCTformer+ L12：

| Probe | Target-vs-BG AUROC | C-PiM | BG enrichment@10 |
|---|---:|---:|---:|
| Raw post-block cosine | 0.592 | 0.362 | 0.805 |
| Pre-attention normalized feature | 0.623 | 0.393 | 0.783 |
| Mean QK energy | 0.853 | 0.453 | 0.562 |
| Conditional \(A_{c2p}\) | 0.855 | 0.409 | 0.571 |
| Final CAM | 0.959 | 0.867 | 0.357 |

从 raw feature 进入 Q/K projection 后，target-vs-background discrimination 大幅提高。更直接的 linkage：

\[
\rho(\text{feature}_{norm},QK)=0.148,
\]

\[
\rho(QK,A_{c2p})=0.979.
\]

Top-10% Jaccard 分别为：

\[
0.199,\qquad0.755.
\]

这说明模型实际 class-to-patch routing 几乎完全由 learned Q/K geometry 决定，而不是 raw hidden cosine。Raw cosine 的用途应降级为：

> **representation-geometry probe**

而不能叫 attention proxy、class localization map 或 semantic ownership posterior。

“bird”案例非常直观：raw cosine 在鸟体上整体呈强负值，而 \(A_{c2p}\)、Patch CAM、C2P CAM 和 Final CAM 都准确覆盖鸟。由此可见，raw cosine 的正负方向不能跨类别统一解释。

---

## 4. 真正的后层现象发生在实际 \(A_{c2p}\)

MCTformer+ 的 attention region quality：

| Layer | C-PiM | Target-vs-BG AUC | Target-vs-Other AUC | BG enrich@10 | Conditional BG mass |
|---:|---:|---:|---:|---:|---:|
| L9 | 0.411 | 0.892 | 0.791 | 0.570 | 0.404 |
| L10 | **0.766** | **0.913** | **0.780** | **0.415** | **0.288** |
| L11 | 0.574 | 0.894 | 0.725 | 0.440 | 0.327 |
| L12 | 0.409 | 0.855 | 0.608 | 0.571 | 0.397 |

L10 是非常明确的最佳 attention layer。L11、L12 的退化有两个特点：

1. Target-vs-BG 仍然较强，L12 AUC 仍有 0.855；
2. Target-vs-Other 从 0.780 快速跌到 0.608，C-PiM也从 0.766 跌到 0.409。

所以后层主要丢失的是：

\[
\boxed{
\text{foreground class specificity}
}
\]

而不是首先失去 foreground/background distinction。

多类别 attention top-10% overlap 同样支持这一点：

\[
0.200\ (L10)
\rightarrow
0.306\ (L11)
\rightarrow
0.467\ (L12).
\]

因此，Experiment 1 的“后层重新耦合”在 actual attention 中确实存在，但它更准确地发生在：

\[
\boxed{
L10\rightarrow L12
}
\]

而不是把 raw-feature 的 L9→L10 变化直接等同于 attention routing。

---

## 5. Shared patches 的真实 owner：不是普遍背景，而是以真实目标为主

MCTformer+ L12 top-10% shared support：

| Signal/stage | Pair targets | Dominant target | Other FG | Background | BG enrichment |
|---|---:|---:|---:|---:|---:|
| Raw feature | 0.465 | 0.413 | 0.102 | 0.432 | 0.752 |
| QK | 0.543 | 0.466 | 0.125 | 0.331 | 0.586 |
| \(A_{c2p}\) | 0.543 | 0.462 | 0.131 | 0.325 | 0.591 |
| Native last3 \(A_{c2p}\) | 0.556 | 0.485 | 0.108 | 0.335 | 0.672 |
| Patch CAM | 0.600 | 0.563 | 0.038 | 0.359 | 0.626 |
| C2P CAM | 0.639 | 0.598 | 0.043 | 0.316 | 0.553 |
| Final CAM | **0.672** | **0.618** | **0.039** | **0.288** | **0.499** |

背景在 raw feature shared set 中占 43.2%，看起来很高；但背景本来占据大部分图像面积。BG enrichment 只有 0.752，显著小于 1，说明它在 shared high-score set 中其实**低于面积基线**。

随着 pipeline 进入 QK、attention 和 CAM：

- 背景比例与 enrichment 持续下降；
- 两个真实目标的合计比例持续提高；
- `dominant target` 几乎接近 `pair targets`。

例如 final CAM：

\[
\frac{0.618}{0.672}\approx92\%.
\]

这表示 shared target support 的绝大部分通常集中在 pair 中的一个主导对象，而不是平均分布在两个目标上。

最合理的阶段性解释是：

\[
\boxed{
\text{late shared support 更像 dominant-object capture / cross-class foreground collision，}
}
\]

而不是：

\[
\boxed{
\text{所有类别共同吸附背景。}
}
\]

---

## 6. 新进入 shared set 的 patches 进一步排除了“普遍背景驱动”

MCTformer+ actual attention 中，新进入共享集合的 patch ownership：

### L9→L10

\[
\text{pair target}=0.622,\quad
\text{other FG}=0.121,\quad
\text{BG}=0.256.
\]

### L10→L11

\[
\text{pair target}=0.577,\quad
\text{other FG}=0.131,\quad
\text{BG}=0.292.
\]

### L11→L12

\[
\text{pair target}=0.492,\quad
\text{other FG}=0.135,\quad
\text{BG}=0.371.
\]

因此 actual attention 在 L9→L10、L10→L11 新增的共享 patches 主要属于 pair targets。到 L11→L12 背景成分上升，但仍不是绝对主导。

Raw post-block feature 在 L9→L10 的新增 shared set 中 BG 比例为 0.541，然而 normalized feature、QK 和 attention 均不支持同样结论。这再次表明 raw cosine 的空间语义解释不稳定。

---

## 7. Class-token cosine 与 feature-map overlap 揭示了一个“共同存在轴”

MCTformer+ L12：

\[
\text{positive class-token pair cosine}=0.356.
\]

但按分类正确性分层后：

### 两个类别在 class-token head 和 patch head 中都判为 positive

\[
\text{class-token cosine}=0.866,
\]

\[
\text{feature-map top10 Jaccard}=0.825.
\]

### 至少一个类别未正确判为 positive

\[
\text{class-token cosine}=-0.224,
\]

\[
\text{feature-map Jaccard}=0.231.
\]

MCTformer+ 的 class-token logit 是 embedding dimensions 的均值。因而所有正类别 token 被推向一个共享的“positive/present”方向，是一个非常合理的解释。由此：

\[
\boxed{
\text{raw class-token cosine 高，不必等于类别身份 collapse；}
}
\]

它可能同时编码：

- 类别 identity；
- 图像中“存在”的共同正向轴。

Q/K projection 在 L9–L11 明显解耦这种 token-vector similarity 与 routing overlap：

- L10 token cosine vs feature overlap correlation约 0.877；
- token cosine vs QK/attention overlap约 -0.089/-0.118；
- 到 L12 才回升到 0.427/0.457。

这意味着 learned Q/K 在大部分后层能够抵消 raw token common-mode geometry，但最后一层这种解耦能力减弱。

这比“背景 token”更接近一个可研究的机制：

\[
\boxed{
\text{late class-query recoupling after projection}
}
\]

---

## 8. Head-wise 结果：背景区分并不难，其他前景才难

MCTformer+ L12 六个 QK heads 的：

\[
target-BG
\]

margin 全部为正，约在 1.29–2.06 之间。

但：

\[
target-otherFG
\]

margin 为：

| Head | Target − Other FG |
|---:|---:|
| 0 | +0.186 |
| 1 | +0.306 |
| 2 | -0.083 |
| 3 | -0.219 |
| 4 | -0.242 |
| 5 | **-1.096** |

也就是说，六个 heads 都能把目标与背景分开，但其中四个 heads 平均更偏向另一个前景类别，head 5 尤其严重。

因此，当前最有证据的 attention failure 是：

\[
\boxed{
\text{inter-foreground class routing ambiguity}
}
\]

而不是所有 heads 都无法识别背景。

这也提示，简单增加 background token 可能无法触及主要问题；更关键的是 class-specific head/query 的 ownership。

---

## 9. 为什么官方使用 last-three attention 仍然合理

单层 attention：

\[
L10\text{ 最好},\quad
L11\text{ 次之},\quad
L12\text{ 最差}.
\]

官方 last3：

\[
C\text{-}PiM=0.653,\quad
AUC_{target-BG}=0.903,\quad
BG\ enrich=0.453.
\]

它显著优于 L12，并接近 L10 的 foreground/background discrimination，但 C-PiM仍低于 L10。

这表明 last-three aggregation 的作用更像：

> 以 L10 的高质量 routing 为锚，平均掉单层不稳定性；

而不是证明 L10–L12 每层都同样可靠。

但是将不同单层 attention 代入 CAM 后：

| CAM attention source | C-PiM | Target-vs-BG AUC |
|---|---:|---:|
| L10 | 0.867 | 0.959 |
| L11 | 0.868 | 0.957 |
| L12 | 0.851 | 0.957 |
| Native last3 | 0.867 | 0.959 |
| Mid3 | 0.845 | 0.953 |

差异已经很小。原因是 patch CAM 本身很强，class-attention filter 只承担部分作用。

因此下一步在提出“换掉 last-three”之前，必须补：

- L10/L11/L12/native last3 的完整 fixed-threshold raw CAM mIoU；
- class-wise confusion；
- multi-label strata；
- 不能只根据 C-PiM 选择层。

---

## 10. CAM pipeline 的作用：每一步总体都在改善排序

MCTformer+：

| Stage | C-PiM | Target-vs-BG AUC | BG enrich@10 | Conditional BG mass |
|---|---:|---:|---:|---:|
| Patch CAM | 0.840 | 0.916 | 0.409 | 0.339 |
| C2P CAM | 0.856 | 0.923 | 0.387 | 0.292 |
| Final CAM | **0.867** | **0.959** | **0.357** | 0.505 |

Class-attention filtering：

- 提高 C-PiM 和 AUC；
- 降低 high-score background enrichment；
- 降低 conditional BG mass。

\(A_{p2p}\) propagation 后：

- C-PiM/AUC进一步提升；
- high-score background enrichment进一步下降；
- 但 total conditional BG mass 从 0.292 升到 0.505。

这不是直接矛盾。Propagation 将低/中强度 activation 扩散到更大空间，所以总体 mass进入更多背景；但 top ranking 和 target-vs-BG ordering 反而变好。

因此以后不能只用：

\[
\sum_{j\in BG}M_j
\]

评价 background leakage。应同时区分：

1. integrated BG mass；
2. high-score BG enrichment；
3. thresholded FP；
4. AUROC/AUPRC；
5. raw CAM mIoU。

在当前结果里，\(A_{p2p}\) 总体是有益的，不支持优先削弱 patch-to-patch propagation。

---

## 11. MCTformer+ 的提升是真正 localization 提升，而不是 classification 提升

两个 checkpoint 的 classification mAP 几乎相同：

\[
0.930\quad\text{vs.}\quad0.931
\]

（class-token head），patch-head mAP 也都约为 0.933。

但固定 448 crop、BG threshold 0.45 的 raw final-CAM mIoU：

\[
0.531\quad\text{MCTformer}
\]

\[
0.678\quad\text{MCTformer+},
\]

paired delta：

\[
+0.147\ [0.134,0.161].
\]

因此 MCTformer+ 的主要优势确实来自 localization pipeline，而不是更高分类精度。

---

## 12. Multi-label 难度上升，但不是主要由背景区分崩溃造成

MCTformer+ raw final-CAM mIoU：

\[
0.708\quad\text{single-label},
\]

\[
0.639\quad\text{exactly two labels},
\]

\[
0.591\quad\text{three or more labels}.
\]

但 L12 attention target-vs-BG AUC只从：

\[
0.860\rightarrow0.854\rightarrow0.843
\]

轻微下降。

这说明 multi-label degradation 不能主要归因于“越来越不会区分背景”。更可能涉及：

- 类别间 ownership；
- object size；
- occlusion；
- dominant-object capture；
- 一个 top-1 patch 落在另一个真实前景对象上。

因此 Competitive Semantic Class-Slot Attention 的出发点仍然有潜力，但它应从：

> foreground vs background competition

扩展成：

> active foreground classes + background 的 soft ownership，其中 foreground class competition 是核心。

---

## 13. Background failure 主要集中在分类失败子集

当 class-token 与 patch-head 对正类别都判为 positive：

\[
BG\ enrich_{feature}=0.769,
\]

\[
BG\ enrich_{attn}=0.532,
\]

\[
BG\ enrich_{finalCAM}=0.235,
\]

\[
AUC_{finalCAM}=0.969.
\]

当至少一个 head 未判为 positive：

\[
BG\ enrich_{feature}=1.013,
\]

\[
BG\ enrich_{attn}=0.797,
\]

\[
BG\ enrich_{finalCAM}=1.062,
\]

\[
AUC_{finalCAM}=0.874.
\]

所以真正有害的 background over-enrichment 主要出现在：

\[
\boxed{
\text{class-presence representation 本身失败的样本}
}
\]

而不是正常识别样本中的普遍结构性问题。

这对 background-token 方案是一个限制：一个 BG branch未必能修复“正类 token/patch head 根本没有激活”的问题。

---

## 14. Register / high-norm artifact 假设目前也没有被直接支持

MCTformer+ L12 background patches 中：

\[
corr(S^{feat},\|p\|_2)=-0.459.
\]

但 top-10% feature-background patches 的平均 norm是普通 background 的 1.154 倍，且 44.6% 位于 valid-patch norm 的 top quartile。

这个组合不是简单的：

> high norm 越大，class similarity 越高。

更合理的结论是：

- 部分高分背景 patch 与高 norm重叠；
- 但整体 score–norm 关系为负；
- register-style artifact 与 semantic alignment 可能只部分重叠。

所以现在没有证据把 generic register 设为主要解决方案。

---

## 15. Codex 的 “Case UNRESOLVED” 应如何理解

Codex 的预注册决策规则要求某一个 Case A–G 的**全部必要 CI 条件**同时落在规定方向，因此最终选择 `UNRESOLVED`。

这不等于“没有发现”。

已经高度稳定的 descriptive evidence 包括：

1. raw cosine 和 actual QK/attention geometry差异极大；
2. L10 attention最好，L11/L12 class specificity退化；
3. actual shared support不背景富集；
4. shared support多数属于两个正类别，且常由一个 dominant target主导；
5. final pipeline总体过滤背景并提高 target ranking；
6. 背景 over-enrichment集中在分类失败样本。

`UNRESOLVED` 的含义只是：

> 当前没有因果干预能够从预设 Cases 中唯一识别训练机制。

这是合理的。

---

## 16. 对当前研究主题的影响

原来的宽泛命题：

> MCTformer+ 很容易产生 background-specific lazy semantic assignment。

不再适合作为中心 claim。全局数据事实上显示：

- MCTformer+ 比 MCTformer 有更好的 target-vs-BG discrimination；
- L12 feature shared BG fraction更低；
- attention/final CAM 的 BG enrichment明显低于 1；
- MCTformer+ full-pipeline background failure并不更高。

更有证据的主题是：

\[
\boxed{
\text{Semantic Ownership under Multi-Label Supervision}
}
\]

其中背景是一个必要 owner，但主要异常是：

\[
\boxed{
\text{late-layer cross-class routing recoupling}
}
\]

和：

\[
\boxed{
\text{dominant-foreground ownership capture}.
}
\]

可考虑的论文问题表述：

> Multi-class tokens become separable in representation space, yet their class-to-patch routing loses foreground-class specificity in the final blocks and increasingly shares support with another present category.

这比“背景噪声”更贴合当前证据。

---

## 17. 我建议的下一组因果实验

### 优先级 1：补齐 single-layer CAM mIoU

使用完全相同 patch CAM 和 \(A_{p2p}\)，只更换：

- L10 \(A_{c2p}\)；
- L11；
- L12；
- native last3。

报告：

- raw CAM mIoU；
- per-class IoU；
- single / 2-label / 3+；
- cross-class confusion。

这是最便宜的必要补充。

### 优先级 2：Late class-to-class mixing intervention

actual attention 在 L10最好、L12退化，且不同 class attention maps后层重合。最直接的因果对象不是 P2C，而是：

\[
A_{c\rightarrow c}.
\]

进行 inference-time intervention：

1. 仅在 L10、L11、L12 分别屏蔽 off-diagonal class-to-class attention；
2. 保留 class self、class-to-patch、patch相关路径；
3. 对剩余 key重新归一化；
4. 分别测：
   - class-token pair cosine；
   - \(A_{c2p}\) top10 overlap；
   - target-vs-other AUC；
   - C-PiM；
   - raw CAM mIoU；
   - classification mAP。

如果屏蔽后：

\[
\text{class-map overlap}\downarrow,
\quad
\text{target-vs-other}\uparrow,
\quad
mIoU\uparrow,
\]

而 classification基本不变，则可以因果支持 late class-token mixing导致 ownership ambiguity。

### 优先级 3：Head ablation

因为 heads 2–5 的 target-other margin 为负，做逐 head disable / leave-one-head-out：

- 不重新训练；
- 分层 L10/L11/L12；
- 检查 head 5 是否稳定拉低 class specificity；
- 同时防止只因单头幅值不同而误判。

### 优先级 4：Context counterfactual

只有在希望保留 background shortcut作为论文重要支线时，再做：

- object-only；
- context-only；
- background swap。

当前数据不支持把它排在 class-mixing causal test 之前。

---

## 18. 最终讨论结论

这套实验把研究方向推进了一步，但方向与最初假设有所变化：

### 被削弱的假设

\[
\text{Background is the universal dominant failure.}
\]

### 被强化的假设

\[
\boxed{
\text{The last Transformer blocks lose foreground-class ownership specificity.}
}
\]

MCTformer+ 的 QK/attention 在 L10 已经非常好；L11/L12 并不是进一步稳定提升，而是在某种程度上重新混合不同 positive classes。Native last3 aggregation可以缓冲 L12 的退化，patch CAM 和 \(A_{p2p}\) 又进一步纠正它，因此最终 CAM仍然很强。

这意味着真正值得研究的不是“怎样再做一次背景抑制”，而是：

> **怎样在不破坏 patch CAM completeness 和合法共享区域的前提下，保持 late-layer class-query routing 的 semantic ownership。**

Background slot 仍可保留，但更适合作为：

\[
\text{none-of-the-active-classes alternative},
\]

而不应成为唯一创新中心。

---

## 19. 目前最精确的阶段性表述

\[
\boxed{
\begin{aligned}
&\text{Raw class–patch cosine recoupling is largely decoupled from actual routing by learned Q/K projections;}\\
&\text{however, class-to-patch attention itself loses inter-foreground specificity from L10 to L12;}\\
&\text{the resulting shared supports are target-enriched, often dominated by one present class, rather than background-enriched.}
\end{aligned}
}
\]

中文：

> Raw class–patch cosine 的后层重新耦合，并不直接等于 attention 背景泄漏；Q/K 投影在很大程度上重新组织了路由。但 \(A_{c2p}\) 本身从 L10 到 L12 确实逐渐丢失前景类别特异性。后层共享的高分区域总体富集于真实正类别，常被其中一个主导对象占据，而不是普遍富集于背景。
