新的 ICLR 方案应当与旧 MCTTA 的解释路线彻底断开：

- 不再讨论 Split Weighted Softmax；
- 不再解释旧 MCTTA 中 Split 为什么有效；
- 不再安排 self-attention / cross-attention 的 Split 因果实验；
- 不再使用 SPG、CTP、Cross Attention、Hierarchical Fusion 作为新论文贡献；
- 不再保留 direct、single-stage、multi-stage 三套方法叙事；
- MCTformer+ 仅作为 class-token WSSS 的基础编码器和强基线。

现有 `AGENTS.md` 中以 TGCA、token-group calibration 为中心的部分应标记为废止，由下面的方案完全替代。

这样也更直接地回应了此前审稿意见：新论文不再堆叠模块，而是围绕一个明确的表示学习问题、一个直接对应的方法以及一套机制性实验展开。此前审稿人最希望看到的正是“为什么 CAM 变好”的定量分析，而不只是更高的最终 mIoU。

# 一、最终研究主题

## 暂定题目

**Background-Aware Competitive Semantic Slots for Weakly Supervised Semantic Segmentation**

更有记忆点的标题可以是：

**Who Owns the Background? Competitive Semantic Slots for Weakly Supervised Semantic Segmentation**

方法简称建议：

\[
\boxed{\text{BCSS}}
\]

即：

> **Background-Aware Competitive Semantic Slots**

## 核心问题

在 multi-class-token WSSS 中，每个 class token 都能独立定位 patches，但模型没有显式回答：

\[
\boxed{\text{每个 patch 在语义上究竟属于哪个前景类别，还是属于背景？}}
\]

标准 class-to-patch attention 更接近：

\[
P(\text{patch }j\mid\text{class }c),
\]

不同类别分别在空间维度归一化。于是多个类别可以同时高响应于同一个上下文 patch，例如：

- boat 与 water；
- train 与 railway；
- cow 与 grass；
- person 与 road；
- aeroplane 与 sky。

这会产生两种问题：

1. **Background semantic leakage**

   前景 class token 在背景或共现上下文上产生较大响应。

2. **Cross-class semantic collision**

   多个 class tokens 同时“认领”同一个背景区域。

MCTformer+ 的 CAM 流程本身就同时使用 class-to-patch attention 和 patch-to-patch affinity，因此我们只保留这一基础编码和 CAM pipeline，不改变其 backbone self-attention。

---

# 二、与已有 background/register 工作的区别

这一部分必须从一开始就定位准确。

Register token 已被用于吸收 ViT 中出现在低信息背景区域的高范数 artifact tokens。 2025 年的 WSSS 工作 Know Your Attention Maps 也已经在多个 class tokens 后加入一个 `[REG]` token，用来捕获 general context、减轻 class-token attention 污染，因此“在 WSSS 中加入一个 register/background token”本身不能作为新论文的主要 novelty。

此外，后续分析指出，register token 虽然可能产生更干净的 attention map，但干净的 attention map并不一定真实反映 global representation 是如何从局部 patches 形成的；register 甚至可能主导 global representation，使局部与全局特征解耦。

2026 年的相关研究进一步指出，在 image-level coarse supervision 下，全局 attention 容易让前景语义扩散到背景 patches，从而形成“用背景捷径完成分类”的行为。

因此，BCSS 不能把贡献写成：

> We introduce a background token.

而应写成：

> We explicitly formulate foreground–background separation as a label-anchored semantic ownership problem, where image-present class slots and a background slot compete for every patch.

Slot Attention 已经提出让多个 slots 在输入维度上竞争并聚合对象表示。 BCSS 与普通 Slot Attention 的区别是：

| 普通 Slot Attention | BCSS |
|---|---|
| Slots 通常是 exchangeable 的 | 每个 foreground slot 绑定一个确定类别 |
| 主要用于无监督对象发现 | 使用 image-level labels 进行语义锚定 |
| 没有固定 foreground/background 语义 | 显式设置 background slot |
| 通常依赖重建或对象级监督 | 使用 WSSS classification objective |
| 输出对象 slots | 输出 class/background semantic ownership maps |

所以真正的研究贡献是三者的组合：

\[
\boxed{
\text{class-anchored slots}
+
\text{explicit background alternative}
+
\text{per-patch semantic competition}
}
\]

---

# 三、研究假设

整篇论文建议围绕三个假设展开。

## H1：Class-token WSSS 存在可量化的背景语义泄漏

对于存在类别 \(c\)，其 class map 在真实背景区域上仍然包含显著质量：

\[
\operatorname{BLR}_c
=
\frac{
\sum_{j\in\Omega_{\mathrm{bg}}}M_{c,j}
}{
\sum_jM_{c,j}+\epsilon
}.
\]

并且在多标签图像中，不同类别 maps 在背景区域上的重叠高于合理水平。

## H2：Generic register 不等同于 semantic background

Register 可以吸收内部计算信息，但：

\[
A_{\mathrm{reg}\rightarrow p}
\]

不一定与真实背景掩码对齐。

因此需要分别测量：

- register 是否捕获高范数 artifact；
- register 是否对应真实 background；
- register map 是否只是看起来更平滑；
- background slot 是否真正提高 foreground/background ownership accuracy。

## H3：显式 class/background competition 可以减少 leakage

对于每个 patch \(j\)，让 active class slots 和 background slot 竞争：

\[
P(s\mid j),
\qquad
s\in\mathcal Y^+\cup\{\mathrm{bg}\}.
\]

这不强制前景和背景各占固定面积，也不指定固定 attention mass。每个 patch 的归属完全由当前图像证据决定。

---

# 四、完整方法：Background-Aware Competitive Semantic Slots

## 4.1 基础编码器

使用原始 MCTformer+，保持其 vanilla self-attention 不变。

输入图像经过 encoder 后得到：

\[
C\in\mathbb R^{N_c\times D},
\]

\[
P\in\mathbb R^{N\times D},
\]

其中：

- \(C\) 为 multi-class tokens；
- \(P\) 为 patch tokens；
- \(N_c\) 为数据集类别数；
- \(N\) 为 patch 数量。

同时保留基线产生的：

\[
H\in\mathbb R^{N_c\times N},
\]

即 patch-classification CAM，以及：

\[
A^{c2p}\in\mathbb R^{N_c\times N},
\qquad
A^{p2p}\in\mathbb R^{N\times N}.
\]

BCSS 不修改 \(A^{p2p}\)，避免影响原有 patch affinity propagation。

---

## 4.2 Semantic slot bank

为每张图像构造 semantic slots：

\[
S^0=
[C_{\mathcal A};B^0],
\]

其中：

- \(C_{\mathcal A}\) 是当前图像的 active class tokens；
- \(B^0\in\mathbb R^{K_b\times D}\) 是 learnable background slots；
- 默认：

\[
K_b=1.
\]

训练时 active class set 来自 image-level labels：

\[
\mathcal A_{\mathrm{train}}
=
\{c\mid y_c=1\}.
\]

背景 slot 始终有效。

验证和测试时安排两种设置：

### Label-known localization

使用 image-level labels 过滤类别，用于标准 CAM 和 pseudo-mask 评价。

### Label-free inference

通过 class-token logits 预测 active classes：

\[
\mathcal A_{\mathrm{test}}
=
\{c\mid \sigma(z_c)>\delta_{\mathrm{cls}}\}.
\]

若没有类别超过阈值，则至少保留 top-1 类别。

这两个设置必须分开报告，不能混在一起。

---

## 4.3 Role-decoupled slot–patch interaction

BCSS 不把 background slot 加入原始 backbone self-attention，而是在 encoder 后增加一个轻量 semantic slot decoder。

Semantic slots 作为 queries，patch tokens 作为 keys 和 values：

\[
Q_s
=
\operatorname{LN}(S^t)W_q,
\]

\[
K_p
=
\operatorname{LN}(P)W_k,
\]

\[
V_p
=
\operatorname{LN}(P)W_v.
\]

相似度矩阵为：

\[
E^t
=
\frac{Q_sK_p^\top}{\sqrt d}
\in
\mathbb R^{(|\mathcal A|+K_b)\times N}.
\]

这里采取的是单向交互：

\[
\boxed{
\text{semantic slots read patches}
}
\]

而不是：

\[
\boxed{
\text{patches read semantic slots}
}
\]

因此不会把 class/background global semantics 写回 patch stream。

---

## 4.4 Competitive Semantic Ownership

对每一个 patch，在 active foreground classes 和 background slots 之间做 softmax：

\[
O^t_{s,j}
=
\frac{
\exp(E^t_{s,j}/\tau)
}{
\sum_{r\in\mathcal A\cup\mathcal B}
\exp(E^t_{r,j}/\tau)
}.
\]

它满足：

\[
\sum_s O^t_{s,j}=1.
\]

其语义是：

\[
O^t_{s,j}
\approx
P(\text{slot }s\mid\text{patch }j).
\]

因此：

\[
O^t_{\mathrm{bg},j}
\]

可以直接作为 background semantic ownership map。

这个机制不会规定：

\[
\sum_jO_{\mathrm{bg},j}
\]

应该是多少，因此不会引入固定 foreground/background 面积比例。

---

## 4.5 Slot aggregation

竞争完成后，对每个 slot 在空间维度重新归一化：

\[
\bar O^t_{s,j}
=
\frac{
O^t_{s,j}
}{
\sum_kO^t_{s,k}+\epsilon
}.
\]

然后聚合 patch features：

\[
U^t_s
=
\sum_j
\bar O^t_{s,j}V_{p,j}.
\]

注意两个归一化分别回答不同问题：

\[
O_{s,j}
:
\quad
\text{这个 patch 属于哪个 slot？}
\]

\[
\bar O_{s,j}
:
\quad
\text{这个 slot 应从哪些 patches 聚合表示？}
\]

---

## 4.6 One-step semantic slot refinement

建议完整模型只做一次 slot update：

\[
S^{t+1}
=
S^t+
\gamma
\operatorname{MLP}
\left(
\operatorname{LN}(U^t)
\right),
\]

其中 \(\gamma\) 为可学习 residual gate，初始值设为：

\[
\gamma=0.
\]

然后使用更新后的 slots 重新计算一次：

\[
E^1,\quad O^1.
\]

最终 ownership map 使用：

\[
O=O^1.
\]

默认只进行：

\[
T=1
\]

次 refinement，避免将方法扩展成复杂 iterative Slot Attention 系统。

---

## 4.7 Ownership-calibrated class-to-patch attention

不直接丢弃 MCTformer+ 已有的 class-to-patch map，而是使用 semantic ownership 对它进行重新加权：

\[
\widetilde A^{c2p}_{c,j}
=
\frac{
A^{c2p}_{c,j}
\left(O_{c,j}+\epsilon\right)^\beta
}{
\sum_k
A^{c2p}_{c,k}
\left(O_{c,k}+\epsilon\right)^\beta
+\epsilon
}.
\]

其中：

\[
\beta\in[0,1]
\]

控制竞争强度。

- \(\beta=0\)：退化为原始 MCTformer+；
- \(\beta=1\)：完全使用 semantic ownership；
- 初始建议：

\[
\beta=0.5.
\]

这样可以降低背景 false positives，同时避免过强竞争导致 object recall 下降。

---

## 4.8 CAM 生成

保持原来的 patch CAM：

\[
H_c
=
\operatorname{ReLU}
\left(
W_c^\top P
\right).
\]

先进行 ownership-calibrated class selection：

\[
M_c^u
=
H_c
\odot
\widetilde A^{c2p}_c.
\]

再使用原始 patch affinity：

\[
M_c
=
A^{p2p}M_c^u.
\]

因此新方法仅替换：

\[
\text{class-to-patch semantic selection}
\]

而不修改：

\[
\text{patch-to-patch spatial propagation}.
\]

背景 map 则直接定义为：

\[
M_{\mathrm{bg}}
=
O_{\mathrm{bg}}.
\]

在 pseudo-label generation 中可以比较两种方式：

### Threshold mode

\[
\hat y_j=
\begin{cases}
\mathrm{bg},
&
M_{\mathrm{bg},j}>\delta_{\mathrm{bg}},
\\
\arg\max_c M_{c,j},
&
\text{otherwise}.
\end{cases}
\]

### Ownership argmax mode

\[
\hat y_j
=
\arg\max_{s\in\mathcal A\cup\{\mathrm{bg}\}}
M_{s,j}.
\]

低置信度区域仍设为 ignore label。

---

# 五、训练目标

新方法必须保持 loss 数量很少，避免再次成为模块和 loss 堆叠。

## 5.1 Baseline classification loss

保留 MCTformer+ 原始分类目标：

\[
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{token}}
+
\lambda_p\mathcal L_{\mathrm{patch}}
+
\lambda_{\mathrm{CCT}}\mathcal L_{\mathrm{CCT}}.
\]

---

## 5.2 Foreground semantic anchoring

将竞争聚合后的 foreground slot feature \(U_c\) 投影到所有类别：

\[
r_c
=
W_{\mathrm{cls}}^\top U_c
\in
\mathbb R^{N_c}.
\]

因为 slot \(c\) 明确对应类别 \(c\)，使用 one-class semantic anchoring：

\[
\mathcal L_{\mathrm{fg}}
=
-\frac{1}{|\mathcal A|}
\sum_{c\in\mathcal A}
\log
\frac{
\exp(r_{c,c}/\tau_s)
}{
\sum_k\exp(r_{c,k}/\tau_s)
}.
\]

它要求：

> 第 \(c\) 个 class slot 聚合到的 patches，应能被识别为类别 \(c\)，而不是其他共现类别。

---

## 5.3 Background null-class loss

将 background slot 聚合特征通过同一个分类器：

\[
r_{\mathrm{bg}}
=
W_{\mathrm{cls}}^\top U_{\mathrm{bg}}.
\]

背景不应包含任何 foreground class evidence：

\[
\mathcal L_{\mathrm{bg}}
=
\frac{1}{N_c}
\sum_{k=1}^{N_c}
\operatorname{softplus}
\left(
r_{\mathrm{bg},k}
\right).
\]

这使 background slot 与 generic register 有根本区别：

- generic register 没有明确语义；
- background slot 被训练为不携带任何前景类别证据。

---

## 5.4 总目标

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{base}}
+
\lambda_{\mathrm{fg}}\mathcal L_{\mathrm{fg}}
+
\lambda_{\mathrm{bg}}\mathcal L_{\mathrm{bg}}
}
\]

第一轮建议：

\[
\lambda_{\mathrm{fg}}=0.5,
\qquad
\lambda_{\mathrm{bg}}=0.1.
\]

只进行一个小范围搜索：

\[
\lambda_{\mathrm{fg}}
\in
\{0.25,0.5,1.0\},
\]

\[
\lambda_{\mathrm{bg}}
\in
\{0.05,0.1,0.25\}.
\]

初版不加入：

- CAM pseudo-background supervision；
- foreground/background 面积先验；
- entropy minimization；
- optimal transport；
- contrastive memory bank；
- graph regularization；
- cross-view consistency。

这些都只能在核心机制验证后作为可选扩展。

---

# 六、训练策略

## Warm-up

前 3 个 epochs：

\[
\beta=0,
\qquad
\gamma=0,
\]

先让 backbone 建立基本 class representations。

之后在 3–8 epochs 中：

\[
\beta:
0\rightarrow\beta_{\max},
\]

\[
\gamma:
0\rightarrow1.
\]

Competition temperature 从较平滑的：

\[
\tau=1.0
\]

逐步下降至：

\[
\tau=0.5.
\]

这样可以避免训练初期 background slot 吸收全部 patches。

## 最终训练公平性

Baseline、register baseline 和 BCSS 必须使用完全相同的：

- ImageNet pretrained weights；
- 数据增强；
- optimizer；
- learning rate；
- batch size；
- epochs；
- random seed；
- CAM generation；
- patch affinity refinement；
- downstream pseudo-mask pipeline。

最终结果从相同 ImageNet 权重重新训练，而不是仅从现有 MCTformer+ checkpoint fine-tune。

---

# 七、完整实验问题

## RQ1：Baseline 是否真的存在 semantic background leakage？

在 VOC val 和 COCO val 上使用 pixel labels，仅用于分析，不参与训练。

必须输出：

- class-to-patch maps；
- patch CAM；
- class/background ownership；
- patch feature norm；
- generic register map；
- predicted segmentation；
- GT segmentation。

### Background Leakage Ratio

\[
\operatorname{BLR}
=
\frac{
\sum_{c\in\mathcal Y^+}
\sum_{j\in\Omega_{\mathrm{bg}}}
M_{c,j}
}{
\sum_{c\in\mathcal Y^+}
\sum_jM_{c,j}
+\epsilon
}.
\]

### Conditional Background Leakage

为了排除不同 map 总幅值的影响，对每个 class map 单独归一化：

\[
\bar M_{c,j}
=
\frac{M_{c,j}}
{\sum_kM_{c,k}+\epsilon},
\]

\[
\operatorname{CBL}
=
\frac{1}{|\mathcal Y^+|}
\sum_{c\in\mathcal Y^+}
\sum_{j\in\Omega_{\mathrm{bg}}}
\bar M_{c,j}.
\]

CBL 是论文的核心 leakage 指标。

### Cross-Class Collision

在多标签图像中：

\[
\operatorname{CCS}_{\mathrm{bg}}
=
\frac{2}{m(m-1)}
\sum_{c<c'}
\frac{
\sum_{j\in\Omega_{\mathrm{bg}}}
\min(M_{c,j},M_{c',j})
}{
\sum_{j\in\Omega_{\mathrm{bg}}}
\max(M_{c,j},M_{c',j})
+\epsilon
}.
\]

预期：

\[
\operatorname{CCS}_{\mathrm{bg}}
\gg
\operatorname{CCS}_{\mathrm{fg}}
\]

或者至少 BCSS 对背景 collision 的下降更明显。

---

## RQ2：Generic register 与 semantic background slot 是否不同？

比较：

\[
A_{\mathrm{reg}\rightarrow p},
\]

\[
A_{\mathrm{bg}\rightarrow p},
\]

\[
O_{\mathrm{bg},p}.
\]

报告：

- Background IoU；
- Background AUPRC；
- Background balanced accuracy；
- 与 GT background 的 Spearman correlation；
- 与 patch feature norm 的 correlation；
- register/background map entropy；
- foreground class score 对 top-attended patches 的依赖。

如果 generic register 主要对应少量高范数 patches，而 \(O_{\mathrm{bg}}\) 覆盖真实背景区域，就能定量区分：

\[
\text{computational sink}
\quad\text{与}\quad
\text{semantic background}.
\]

只提供更漂亮的 background-token heat map 不够，因为 register map 的可解释性本身可能具有误导性。

---

## RQ3：竞争是否比简单添加 background token 更有效？

核心组件表：

| ID | Generic REG | BG slot | Patch-wise competition | BG null loss | Slot update |
|---|---:|---:|---:|---:|---:|
| E0 |  |  |  |  |  |
| E1 | ✓ |  |  |  |  |
| E2 |  | ✓ |  |  |  |
| E3 |  |  | Class only |  |  |
| E4 |  | ✓ | Class + BG |  |  |
| E5 |  | ✓ | Class + BG | ✓ |  |
| E6 |  | ✓ | Class + BG | ✓ | ✓ |
| E7 | 参数匹配 dummy token |  |  |  |  |

含义：

- **E0**：原始 MCTformer+；
- **E1**：普通 register baseline；
- **E2**：只增加 background token，独立关注 patches；
- **E3**：foreground classes 竞争，但没有 background；
- **E4**：完整 class/background competition；
- **E5**：增加 semantic background supervision；
- **E6**：完整 BCSS；
- **E7**：证明增益不是增加参数带来的。

最关键的对比是：

\[
E1\quad\text{vs.}\quad E2,
\]

\[
E2\quad\text{vs.}\quad E4,
\]

\[
E4\quad\text{vs.}\quad E5,
\]

\[
E5\quad\text{vs.}\quad E6.
\]

其中 E1 必须包含，因为已有 WSSS 工作已经使用 `[REG]` 捕获 general context。

---

## RQ4：BCSS 是否只是 CAM complement？

必须加入两个非常强的简单 baseline。

### CAM complement

\[
M_{\mathrm{bg}}^{\mathrm{comp}}
=
1-
\max_{c\in\mathcal Y^+}
\operatorname{Norm}(M_c).
\]

然后使用该 background map 进行同样的 CAM gating。

### Confidence entropy

根据 class-map entropy 或最大类置信度生成背景：

\[
M_{\mathrm{bg}}^{\mathrm{ent}}
=
1-
\max_cP(c\mid j).
\]

如果 BCSS 不能明显超过这些无需学习的背景估计，那么 learnable background slot 的价值不足。

还可以设置 evaluation-only oracle：

\[
M_c^{\mathrm{oracle}}
=
M_c
\odot
(1-G_{\mathrm{bg}}),
\]

其中 \(G_{\mathrm{bg}}\) 是 GT background。该结果只作为背景抑制的理论上限，不参与训练。

---

## RQ5：竞争方式如何影响 precision 和 recall？

竞争机制容易提升 precision，但可能降低 recall，因此必须单独报告：

\[
\text{CAM Precision},
\qquad
\text{CAM Recall},
\qquad
\text{CAM F1}.
\]

归一化消融：

| 方式 | 类别间竞争 | 显式背景 |
|---|---:|---:|
| Independent row softmax |  |  |
| Sigmoid class maps |  |  |
| Class-only column softmax | ✓ |  |
| Class + background softmax | ✓ | ✓ |
| Class + background + residual ownership gate | ✓ | ✓ |
| Hard argmax ownership | ✓ | ✓ |

主方法保持 soft assignment。

若 recall 下降，优先降低：

\[
\beta
\]

或使用 residual ownership：

\[
G_{c,j}
=
(1-\lambda_o)
+
\lambda_oO_{c,j}.
\]

不应首先增加新的 loss。

---

## RQ6：Background slot 的数量是否重要？

默认：

\[
K_b=1.
\]

消融：

\[
K_b\in\{1,2,4\}.
\]

当 \(K_b>1\) 时，不能简单让多个 background slots 共同参与普通 softmax，否则 background 因 slot 数量增加而获得额外概率。

应先聚合 background evidence：

\[
E_{\mathrm{bg},j}
=
\log
\left[
\frac{1}{K_b}
\sum_{b=1}^{K_b}
\exp(E_{b,j})
\right].
\]

然后在：

\[
\{\text{active classes},\text{aggregated background}\}
\]

之间竞争。

最后再在 background group 内分配：

\[
P(b\mid\mathrm{bg},j).
\]

这样 \(K_b\) 的变化不会自动改变 foreground/background group prior。

---

## RQ7：方法是否具有跨架构泛化能力？

最低要求是两个 host：

1. MCTformer+；
2. 一个外部 multi-class-token WSSS architecture。

优先顺序建议：

\[
\text{MoRe}
>
\text{CTI}
>
\text{MCTformer}.
\]

最终表格：

| Host | Baseline Seed | +REG | +BCSS | CBL↓ | BG IoU↑ |
|---|---:|---:|---:|---:|---:|
| MCTformer+ |  |  |  |  |  |
| External host |  |  |  |  |  |

新论文中不需要出现 MCTTA host，也不需要解释旧 MCTTA 的任何结果。

---

# 八、反事实实验

这部分会明显提升论文的 ICLR 风格，因为它直接验证模型是否依赖背景捷径。

## 8.1 Context-only

使用 GT mask 移除目标物体，保留背景：

\[
x_{\mathrm{ctx}}
=
x\odot(1-G_c)
+
\mu G_c.
\]

其中 \(\mu\) 可以是：

- dataset mean；
- Gaussian blur；
- patch shuffle。

定义：

\[
\operatorname{CRS}_c
=
\frac{
\sigma(z_c(x_{\mathrm{ctx}}))
}{
\sigma(z_c(x))+\epsilon
}.
\]

较低的 CRS 表明模型不再依赖上下文完成类别预测。

## 8.2 Object-only

移除背景，仅保留对象：

\[
x_{\mathrm{obj}}
=
x\odot G_c
+
\mu(1-G_c).
\]

定义：

\[
\operatorname{ORS}_c
=
\frac{
\sigma(z_c(x_{\mathrm{obj}}))
}{
\sigma(z_c(x))+\epsilon
}.
\]

理想结果：

\[
\operatorname{CRS}\downarrow,
\]

同时：

\[
\operatorname{ORS}
\]

保持或上升。

## 8.3 Background swap

将目标对象粘贴到另一张图像的背景上，报告：

- class score stability；
- object-region CAM consistency；
- new-background false positives；
- background ownership adaptability。

这项可以放 appendix，但 context-only 和 object-only 必须放主文。

---

# 九、数据集和最终评价流程

## PASCAL VOC 2012

用于：

- 快速方法筛选；
- 全部机制指标；
- 三随机种子；
- counterfactual test；
- pseudo masks；
- downstream segmentation。

## MS COCO 2014

必须完成，因为：

- 类别更多；
- 多标签共现更复杂；
- 背景种类更多；
- class collision 问题更明显。

## 可选：COCO-Stuff

仅用于分析 background slots 是否分化出：

- sky；
- grass；
- road；
- wall；
- water。

不使用其 pixel labels 进行训练。

## 主指标

任务性能：

\[
\text{Classification mAP},
\]

\[
\text{Raw CAM mIoU},
\]

\[
\text{Pseudo-mask mIoU},
\]

\[
\text{Final segmentation mIoU}.
\]

机制指标：

\[
\text{CBL},
\]

\[
\text{CCS}_{\mathrm{bg}},
\]

\[
\text{Background IoU/AUPRC},
\]

\[
\text{Semantic ownership purity},
\]

\[
\text{CRS},
\quad
\text{ORS}.
\]

所有 VOC 核心实验报告：

\[
\text{mean}\pm\text{std}
\]

基于 3 个随机种子。

---

# 十、可视化规划

## Figure 1：问题与方法

左侧：

- 多个 class tokens 分别产生 maps；
- boat 和 person 同时关注 water；
- 没有 background ownership；
- 背景区域产生 semantic collision。

右侧：

- active class slots 与 background slot 竞争；
- 每个 patch 有 ownership distribution；
- 背景 slot 吸收 context patches；
- class maps 更少泄漏。

## Figure 2：Register 与 Background Slot

列为：

1. Input；
2. GT；
3. class-to-patch；
4. register query → patches；
5. patches → register；
6. background slot raw score；
7. background ownership \(O_{\mathrm{bg}}\)；
8. final CAM。

必须区分：

\[
A_{\mathrm{reg}\rightarrow p}
\]

和：

\[
A_{p\rightarrow\mathrm{reg}},
\]

不能将两个 attention 方向混为一谈。

## Figure 3：Semantic Competition

选择多类别共现图像，展示：

- baseline class maps；
- foreground-only competition；
- foreground+background competition；
- class ownership；
- background ownership；
- final maps。

## Figure 4：Counterfactual

展示原图、context-only、object-only，以及：

- class score；
- class CAM；
- background ownership。

## 可视化规范

- 同一图像、同一指标使用相同色标；
- 不对每种方法单独 min-max；
- 同时展示随机样例和 highest-leakage 样例；
- 主文展示成功与失败案例；
- 附录展示更多 layer/head maps；
- 所有图使用 vector layout。

---

# 十一、实验执行顺序

## 8 月 27–30 日：诊断系统

完成：

- baseline attention dump；
- generic register baseline；
- class/background map visualization；
- CBL、CCS、Background IoU/AUPRC；
- context-only/object-only generator；
- VOC baseline 三类 map 的统一导出。

输出目录建议：

```text
analysis/
├── ownership_metrics.py
├── background_metrics.py
├── counterfactual.py
├── dump_attention.py
└── visualize_slots.py
```

## 8 月 30 日–9 月 3 日：VOC 最小筛选

只训练：

\[
E0,\ E1,\ E2,\ E4,\ E5,\ E6.
\]

单 seed、完整 classification schedule。

9 月 3 日必须确定：

- class+background competition 是否优于 register；
- 是否需要 slot update；
- 是否需要 background null loss；
- \(\tau\) 和 \(\beta\) 的大致范围；
- background slot 是否发生 collapse。

## 9 月 4–9 日：VOC 完整实验

- 主配置 3 seeds；
- 核心组件 ablation；
- CAM complement baseline；
- precision/recall；
- counterfactual test；
- IRN/pseudo-mask；
- downstream segmentation；
- 主图初稿。

## 9 月 6–13 日：COCO 并行

只使用 VOC 锁定后的主配置。

GPU 安排：

| GPU | 工作 |
|---|---|
| A6000-1 | VOC seeds、消融、counterfactual |
| A6000-2 | COCO baseline、register、BCSS |

不在 COCO 上做大规模超参数搜索。

## 9 月 9–15 日：第二 host

优先复现：

- MoRe；
- 或 CTI。

只运行：

- host baseline；
- host + BCSS。

## 9 月 13–17 日：锁定摘要和主表

ICLR 2027 摘要截止为 **2026 年 9 月 18 日 AoE**，全文截止为 **9 月 25 日 AoE**；主文最多 9 页且采用 double-blind submission。

9 月 17 日前必须确定：

- 最终方法名；
- 三条 contribution；
- VOC main result；
- COCO preliminary result；
- CBL/CCS 机制结论；
- register vs background 结论；
- abstract 中所有数字。

## 9 月 18–23 日：论文与 appendix

主文建议：

| 部分 | 页数 |
|---|---:|
| Introduction | 1.25 |
| Related Work | 0.75 |
| Problem Diagnosis | 1.0 |
| Method | 2.0 |
| Main Experiments | 2.0 |
| Mechanism Analysis | 1.5 |
| Conclusion & Limitations | 0.5 |

## 9 月 24 日：冻结

只处理：

- 数值一致性；
- 排版；
- anonymization；
- supplementary；
- code cleanup；
- disclosure。

ICLR 2027 要求作者披露生成式 AI 在研究假设、方法设计、实验设计、代码实现和结果解释等环节的使用；当前这类方法与实验规划属于需要披露的范围。

---

# 十二、Go/No-Go 标准

## 第一阶段：VOC 单 seed

BCSS 相比 MCTformer+ 至少满足四项：

\[
\Delta\mathrm{CAM\ mIoU}\geq1.0,
\]

\[
\mathrm{CBL\ relative\ reduction}\geq15\%,
\]

\[
\mathrm{CCS}_{bg}
\text{ relative reduction}\geq20\%,
\]

\[
\Delta\mathrm{CAM\ precision}\geq2.0,
\]

同时：

\[
\Delta\mathrm{CAM\ recall}>-1.5,
\]

\[
\Delta\mathrm{classification\ mAP}>-0.3.
\]

## 第二阶段：必须超过 register

至少在以下四项中的三项超过 generic register：

- raw CAM mIoU；
- CBL；
- Background IoU/AUPRC；
- CRS；
- final segmentation。

如果只比无 register baseline 好，但与 register 无明显区别，则论文 novelty 不足。

## 第三阶段：泛化

必须满足：

- COCO 上正增益；
- 第二 host 上正增益；
- 三个 seeds 方向一致；
- mechanism metrics 与 task metrics 同向；
- context reliance 降低；
- object retention 不明显下降。

---

# 十三、这篇论文最后应声称什么

不应声称：

> Background tokens are new.

也不应声称：

> Attention maps directly explain model decisions.

更稳妥的三条贡献是：

> **1.** We identify and quantify semantic background leakage in class-token-based weakly supervised segmentation, showing that independently generated class maps frequently collide on contextual background regions.

> **2.** We introduce Background-Aware Competitive Semantic Slots, which formulate patch localization as label-anchored semantic ownership among image-present foreground classes and an explicit background alternative.

> **3.** Through register controls, background-alignment metrics, cross-class collision analysis, and counterfactual context tests, we demonstrate that the learned background slot captures semantic background rather than merely serving as a generic attention sink.

整篇论文的核心可以压缩成一句：

\[
\boxed{
\text{Do not merely suppress background; give it explicit semantic ownership.}
}
\]

这条路线已经完全删除旧 MCTTA 结果解释，MCTformer+ 只承担基础编码器和公平 baseline 的角色。新论文的成败将由 **background ownership 是否真实、竞争是否减少语义泄漏、以及是否稳定改善 CAM 与 downstream segmentation** 决定，而不是由旧模块的重新组合决定。