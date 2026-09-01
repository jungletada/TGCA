可以。我这次按你的要求，**不先设计新方法**，而是先把 LaST-ViT 的源码里“怎么验证 lazy aggregation”拆清楚，再严格映射到 **MCTformer / MCTformer+**，目标只有一个：

$$
\boxed{\text{先验证 MCTformer 系列到底有没有更严重的 lazy semantic assignment}}
$$

我读完 `ChengShiest/LAST-ViT` 当前公开源码后，觉得这个验证非常值得做，而且 MCTformer 比普通单 `[CLS]` ViT 其实能做出**更强的 class-specific 版本验证**。

---

# 1. LaST-ViT 源码真正公开了哪些验证工具

当前 repo 主要分成：

```text
LAST-ViT/
├── cls_pretrain/
│   └── conf.py
└── visualization/
    ├── patch_score.py
    ├── evaluate_patch_hit.py
    ├── visualize_patch_score_distribution.py
    └── visualize_token_selection.py
```

官方 README 明确把核心问题定义成：

> ViT uses semantically irrelevant background patches as shortcuts to represent global semantics, driven by global attention and coarse-grained semantic supervision.

而 LaST 的 solution 是 selectively integrate patch features into global representation。

对我们来说，现在**先不要搬它的 solution**。最重要的是搬它的 diagnosis。

---

# 2. LaST 的第一个关键量不是 attention，而是 Patch Score

这是最容易误解的地方。

`visualization/patch_score.py` 里，它没有使用：

$$
A_{\mathrm{CLS}\rightarrow p}.
$$

而是使用最终层 feature：

$$
q_{\mathrm{CLS}}
$$

与每一个 patch feature：

$$
p_j
$$

之间的 cosine similarity：

$$
\boxed{
S_j
=
\cos(p_j,q_{\mathrm{CLS}})
}
$$

代码就是：

```python
similarity = torch.cosine_similarity(
    patch_tokens,
    cls_token_expanded.expand(-1, num_patches, -1),
    dim=-1
)
```

所以 LaST 在问的不是：

> CLS 当前 attention 到哪里？

而是：

> **哪些 patch representation 已经变得最像 global semantic representation？**

这是一个 representation-level diagnosis。

---

# 3. 这一点搬到 MCTformer 后会非常有意思

普通 ViT 只有：

$$
q_{\mathrm{CLS}}.
$$

但 MCTformer 有：

$$
C=
[c_1,c_2,\ldots,c_K].
$$

因此 LaST 的 Patch Score 可以自然升级成：

$$
\boxed{
S_{c,j}^{(l)}
=
\cos
\left(
c_c^{(l)},
p_j^{(l)}
\right)
}
$$

其中：

* \(c\)：类别；
* \(j\)：patch；
* \(l\)：Transformer layer。

这其实比 LaST 原来的定义更有信息。

LaST 只能问：

> 哪些 background patches 变得像“整张图”？

我们可以问：

> 哪些 background patches 变得像 **dog class token**？

> 哪些 background patches 变得像 **train class token**？

> 哪些 background patches 变得像 **boat class token**？

所以普通 LaST 的：

$$
\text{background}\rightarrow\text{global semantics}
$$

到了 MCTformer，就可以变成：

$$
\boxed{
\text{background}
\rightarrow
\text{class-specific global semantics}
}
$$

这正是你说的核心问题。

---

# 4. LaST 的第二个验证：最高 Patch Score 是否真的落在物体上

源码 `evaluate_patch_hit.py` 很简单。

对：

$$
S_j
$$

取：

$$
j^*=\arg\max_jS_j.
$$

然后检查：

$$
j^*
$$

是否在 GT bounding box 中。

源码就是：

```python
top1 = scores.argmax(dim=1)
```

然后：

```python
if int(top1[i]) in patch_set:
    hit += 1
```

最终报告：

```text
Top-1 Patch in BBox
```

这个指标实际上就是论文里的 Point-in-Box 思想。

---

# 5. 在 MCTformer 上，我们可以定义更强的 Class-specific Point-in-Mask

我们没必要用 bbox。

VOC / COCO val 本来就有 segmentation GT，所以反而比 LaST 的 ImageNet evaluation 条件更好。

对图像中真实存在的类别：

$$
c\in Y^+,
$$

计算：

$$
j_c^*
=
\arg\max_j
S_{c,j}.
$$

然后判断：

$$
GT(j_c^*)=c.
$$

定义：

$$
\boxed{
\mathrm{C\text{-}PiM}
=
\frac{
\sum_{i,c\in Y_i^+}
\mathbf1[
GT_i(j_c^*)=c
]
}{
\sum_i|Y_i^+|
}
}
$$

即：

> **Class-specific Point-in-Mask。**

这个指标我觉得应该是第一优先级。

因为它直接回答：

> `dog` class token 最像的 patch，到底是不是 dog？

---

# 6. 但只做二分类 FG/BG 还不够

MCTformer 是多类别模型。

所以对每个正类别：

$$
c,
$$

我们应该把 patches 分成三组，而不是 LaST 的两组：

$$
\Omega_c
=
\{\text{target-class patches}\},
$$

$$
\Omega_{\mathrm{other}}
=
\{\text{other foreground-class patches}\},
$$

$$
\Omega_{\mathrm{bg}}
=
\{\text{background patches}\}.
$$

例如一张：

```text
person + bicycle + road
```

的图。

对于：

$$
c=\mathrm{person},
$$

三个集合分别是：

```text
person patches
bicycle patches
road/background patches
```

这样我们可以区分两种完全不同的问题：

$$
\boxed{
\text{Foreground class confusion}
}
$$

和：

$$
\boxed{
\text{Background semantic leakage}
}
$$

这比 LaST 原来的 foreground/background 二分更加适合 WSSS。

---

# 7. LaST 源码已经有 FG/BG distribution 的完整实现

`visualize_patch_score_distribution.py` 后来专门加入了 Figure 2 风格的分析。

它使用：

* ImageNet bbox；
* SAM2 从 bbox refinement 出前景 mask；
* patch majority overlap > 0.5 判定前景；
* 剩下全部 patch 作为 background。

源码随后把：

```python
foreground_scores
background_scores
```

分别收集起来，再统计：

* mean；
* median；
* q90；
* density histogram。

我们在 VOC 上完全不需要 SAM2。

直接：

$$
GT_{\mathrm{VOC}}
$$

下采样到：

$$
28\times28
$$

即可。

而且可以比 LaST 更干净地统计：

$$
P(S_{c,j}\mid j\in\Omega_c),
$$

$$
P(S_{c,j}\mid j\in\Omega_{\mathrm{other}}),
$$

$$
P(S_{c,j}\mid j\in\Omega_{\mathrm{bg}}).
$$

---

# 8. 第一组我认为最重要的实验

针对 **MCTformer 与 MCTformer+ 都做**：

$$
S_{c,j}^{(L)}
=
\cos
(c_c^{(L)},p_j^{(L)}).
$$

报告三个 distribution：

| Region            | Score                                   |
| ----------------- | --------------------------------------- |
| Target foreground | \(S_{c,j},j\in\Omega_c\)                |
| Other foreground  | \(S_{c,j},j\in\Omega_{\mathrm{other}}\) |
| Background        | \(S_{c,j},j\in\Omega_{\mathrm{bg}}\)    |

除了 mean，至少看：

$$
Q_{50},
\quad
Q_{90},
\quad
Q_{95}.
$$

因为 lazy assignment 最可能出现在 background distribution 的 **right tail**，均值未必明显。

尤其可以定义：

$$
\boxed{
\mathrm{BG\text{-}Tail@q}
=
\frac{
|\operatorname{Top}_q(S_c)\cap\Omega_{\mathrm{bg}}|
}{
|\operatorname{Top}_q(S_c)|
}
}
$$

例如：

$$
q=5\%,10\%.
$$

如果：

$$
\mathrm{BG\text{-}Tail@10\%}
$$

非常大，那就是很强的证据。

---

# 9. Point-in-Mask 应该进一步拆成三种 outcome

不要只报告：

$$
C\text{-}PiM.
$$

对于：

$$
j^*=\arg\max_jS_{c,j},
$$

报告：

$$
P(j^*\in\Omega_c),
$$

$$
P(j^*\in\Omega_{\mathrm{other}}),
$$

$$
P(j^*\in\Omega_{\mathrm{bg}}).
$$

也就是：

| Outcome           | Meaning                        |
| ----------------- | ------------------------------ |
| Target hit ↑      | 正常 class semantic localization |
| Other-class hit ↓ | class confusion                |
| Background hit ↓  | lazy background assignment     |

这样我们第一次就可以把：

$$
\boxed{
\text{class confusion}
}
$$

和：

$$
\boxed{
\text{background leakage}
}
$$

真正分离开。

---

# 10. 我尤其建议不要只分析最后一层

这一点是 MCTformer 相比 LaST 原实验最值得扩展的地方。

对所有层：

$$
l=1,\ldots,12
$$

提取：

$$
c_c^{(l)},
\qquad
p_j^{(l)}.
$$

然后：

$$
S_{c,j}^{(l)}
=
\cos(c_c^{(l)},p_j^{(l)}).
$$

画：

$$
C\text{-}PiM(l),
$$

$$
BG\text{-}Tail(l),
$$

$$
TargetMean(l),
$$

$$
BGMean(l).
$$

这样我们就可以回答：

> **Lazy semantic assignment 是什么时候形成的？**

如果出现：

$$
l=1\sim4:
\quad
S_{bg}\ll S_{fg},
$$

但：

$$
l=8\sim12:
\quad
S_{bg}\uparrow,
$$

那就是非常漂亮的 evidence：

$$
\boxed{
\text{background semantics 是在 global Transformer interaction 中逐层形成的，}
}
$$

而不是 patch embedding 一开始就有。

---

# 11. 而且 MCTformer 有一个 LaST 没有的优势：Attention Matrix

这是非常关键的地方。

LaST 的 Patch Score 是：

$$
S_{c,j}^{feat}
=
\cos(c_c,p_j).
$$

但 MCTformer 同时直接产生：

$$
A_{c2p}(c,j).
$$

而 MCTformer+ 的 CAM 本来就是用 class-to-patch attention 和 patch-to-patch attention refine；你现有稿件也明确写出了 \(A_{c2p}\) 和 \(A_{p2p}\) 的使用方式。

因此我们应该**同时测三个东西**：

$$
\boxed{
S_{c,j}^{feat}
=
\cos(c_c,p_j)
}
$$

representation semantic alignment；

$$
\boxed{
S_{c,j}^{attn}
=
A_{c2p}(c,j)
}
$$

attention routing；

以及：

$$
\boxed{
S_{c,j}^{cam}
=
CAM_c(j)
}
$$

最终 localization。

这三个不要混在一起。

---

# 12. 这会产生非常有价值的四种结果

假设某个 background patch：

### 情况 A

$$
S^{feat}_{bg}\uparrow,
\qquad
A_{c2p,bg}\uparrow.
$$

说明：

> patch 本身已经获得 class semantics，而且 class token 又主动读取它。

这是最严重的 semantic leakage。

---

### 情况 B

$$
S^{feat}_{bg}\uparrow,
\qquad
A_{c2p,bg}\downarrow.
$$

说明：

> patch representation 已经被污染，但 class attention 暂时还能抑制它。

这是 representation pollution。

---

### 情况 C

$$
S^{feat}_{bg}\downarrow,
\qquad
A_{c2p,bg}\uparrow.
$$

说明：

> attention routing 错了，但 patch representation 本身还正常。

这更接近 MoRe 的问题。

---

### 情况 D

两者都低。

正常。

这会把我们前面讨论的：

$$
\text{storage pollution}
$$

$$
\text{aggregation pollution}
$$

$$
\text{semantic assignment pollution}
$$

第一次真正变成可测量东西。

---

# 13. 如何验证它是不是“class-specific global semantics”而不仅仅是 generic global semantics

这是我觉得最重要的新实验之一。

对 class token \(c\)，比较：

$$
S_{c,j}
$$

在两种图像中的 background patches：

### 图像中 class \(c\) 存在

$$
y_c=1.
$$

### 图像中 class \(c\) 不存在

$$
y_c=0.
$$

定义：

$$
\boxed{
\Delta_{\mathrm{presence}}(c)
=
\mathbb E[
S_{c,j}
\mid
j\in BG,y_c=1
]
-
\mathbb E[
S_{c,j}
\mid
j\in BG,y_c=0
].
}
$$

如果：

$$
\Delta_{\mathrm{presence}}(c)\gg0,
$$

说明同一个 `car` class token：

> 只有在 car 真正出现在图像中时，background patches 才开始变得像 car。

这比单纯说 background 有 global semantics 强得多。

它真正说明：

$$
\boxed{
\text{foreground class semantics 已经扩散进 background patch representation。}
}
$$

也就是：

$$
\boxed{
\text{background}\rightarrow
\text{class-specific global semantics}.
}
$$

---

# 14. MCTformer 和 MCTformer+ 的比较特别有意义

MCTformer+ 相比 MCTformer 增加了更强的 class-token discrimination，例如 CCT，以及改进后的 classification/CAM generation。你现有稿件也把 MCTformer+ 的 class-token discrimination 作为基础。

这里存在两个相反的可能性。

### Hypothesis A

CCT 让：

$$
c_{\mathrm{dog}},
c_{\mathrm{cat}}
$$

更可分，因此：

$$
\text{other-class confusion}\downarrow.
$$

这是好事。

但同时更强 class semantics 可能通过 global self-attention 更容易传播到 patches：

$$
\text{BG semantic leakage}\uparrow.
$$

也就是：

$$
\boxed{
\text{class specificity improved,
but semantic diffusion worsened}.
}
$$

---

### Hypothesis B

更好的 class-token representation 同时提高：

$$
C\text{-}PiM
$$

并降低 background confusion。

到底是哪一种不能猜。

这正是实验应该回答的问题。

如果结果是 A，我觉得会非常有研究价值。

---

# 15. 第二个关键验证：High-score patch 是否真的对分类有贡献？

LaST 论文一个很强的 causal test 是：

> 删除 Patch Score 最高的 patches，classification 几乎不下降；删除低分 patches 反而掉得厉害。

值得注意的是，我检查了当前公开 repo：目前公开了 Patch Score、Point-in-BBox、FG/BG score distribution 和 token-selection visualization；**没有找到论文中这个 masking accuracy experiment 的现成脚本**。所以这部分我们要自己实现。

但搬到 MCTformer 后反而可以做得更细。

---

# 16. Class-specific causal masking

对一个正类别：

$$
c,
$$

先在原图计算：

$$
S_{c,j}.
$$

然后固定这些 patch indices，不重新排序。

分别删除：

$$
\operatorname{TopK}_{BG}(S_c),
$$

$$
\operatorname{Random}_{BG},
$$

$$
\operatorname{TopK}_{FG}(S_c),
$$

$$
\operatorname{Random}_{FG}.
$$

删除方式可以先统一使用：

* ImageNet mean；
* 或 zero after normalization。

然后重新 forward。

测：

$$
\Delta z_c
=
z_c(x)-z_c(x_{\mathrm{masked}}).
$$

---

# 17. 这组实验的解释比 LaST 还能更细

假设某个 background patch：

$$
S_{c,j}
$$

非常高。

可能有两种情况。

### 类型 1：Representational leakage

高 score background 被删除以后：

$$
\Delta z_c\approx0.
$$

说明它虽然“长得像 class token”，但分类不真正依赖它。

这接近 LaST 原来的现象：

$$
\boxed{
\text{semantic information 被懒惰地写进了 background，}
}
$$

但它只是冗余 representation。

---

### 类型 2：Causal background shortcut

高 score background 被删除后：

$$
\Delta z_c\gg0.
$$

说明：

> 模型不仅在 background 中编码了 class semantics，而且真的依赖它分类。

这其实比 LaST 原始现象更危险：

$$
\boxed{
\text{background is a decision shortcut}.
}
$$

所以我们不应该预设 masking high-score background 一定“不影响分类”。

两种结果都有研究意义。

---

# 18. 我建议正式把这两个概念分开

这是我觉得比照搬 LaST 更好的地方。

### Representational Lazy Assignment

$$
\boxed{
\text{Background patch becomes class-semantic}
}
$$

测：

$$
S_{c,j}^{feat},
\quad
C\text{-}PiM,
\quad
BG\text{-}Tail.
$$

### Decision Shortcut

$$
\boxed{
\text{Classification depends on background semantics}
}
$$

测：

$$
\Delta z_c
$$

以及 background removal/context swap。

这样论文不会把：

$$
\text{semantic alignment}
$$

和：

$$
\text{causal importance}
$$

混成一个概念。

这也正好规避 Register/[CLS] decoupling 工作对 attention faithfulness 的批评。

---

# 19. 第三个实验：把 LaST 的“global dependency”验证搬过来

这一步对 MCTformer 特别关键。

标准 MCTformer attention 实际包含四个区域：

$$
A=
\begin{bmatrix}
A_{c2c} & A_{c2p}\\
A_{p2c} & A_{p2p}
\end{bmatrix}.
$$

你之前的 MCTTA 稿件 Fig. 2 也清楚画出了这四部分。

这里：

$$
A_{c2p}
$$

表示：

> class token 读取 patches；

而：

$$
A_{p2c}
$$

表示：

> patch token 读取 class tokens。

这第二条路径非常值得怀疑。

---

# 20. MCTformer 的“语义写回”路径

如果：

$$
p_j
$$

通过：

$$
A_{p2c}
$$

大量读取：

$$
c_{\mathrm{dog}},
$$

那么 patch feature 更新后：

$$
p'_j
$$

自然会越来越像：

$$
c_{\mathrm{dog}}.
$$

即使 patch \(j\) 本身是 grass/background。

所以：

$$
\boxed{
A_{p2c}
}
$$

实际上提供了一条非常直接的：

$$
\boxed{
\text{class semantic}
\rightarrow
\text{patch feature}
}
$$

写入通道。

普通单 `[CLS]` ViT 只有一个 global token；

MCTformer 有：

$$
20/80
$$

个 class-specific global tokens。

因此这可能就是：

$$
\boxed{
\text{MCTformer 更容易发生 class-specific lazy assignment}
}
$$

的结构原因之一。

---

# 21. 可以做一个很干净的 inference-time causal intervention

先完全不训练新模型。

只在 forward 时修改 attention。

### Original

$$
A_{p2c}
$$

保持原样。

### Block-P2C

设置：

$$
A_{p2c}=0,
$$

然后对每个 patch query 剩余的：

$$
A_{p2p}
$$

重新归一化：

$$
A'_{p2p}
=
\frac{A_{p2p}}
{\sum_kA_{p2p}(j,k)}.
$$

其他：

$$
A_{c2p},
A_{c2c}
$$

全部不变。

也就是说：

$$
\boxed{
\text{class token can still read patches,}
}
$$

但：

$$
\boxed{
\text{patches cannot read class tokens.}
}
$$

这正对应你说的：

> **直接保护 patch local feature。**

---

# 22. 如果这个 intervention 出现以下结果，会非常漂亮

例如：

$$
\text{classification mAP}
\approx
\text{unchanged},
$$

但：

$$
BG\text{-}Tail\downarrow,
$$

$$
C\text{-}PiM\uparrow,
$$

$$
CAM\ mIoU\uparrow.
$$

这将提供非常强的机制证据：

> MCTformer 的 class tokens 并不需要反向写入 patches 才能完成 image classification，但这种写入导致 dense patch representation 获得 class-specific background semantics。

这几乎直接支持你的第 2 点：

$$
\boxed{
\text{Patch local feature 需要直接保护。}
}
$$

---

# 23. 还可以按 layer 做 P2C blocking

不用一次全 block。

分别：

$$
L=1\sim4,
$$

$$
5\sim8,
$$

$$
9\sim12.
$$

或者逐层：

$$
block(l).
$$

然后重新计算：

$$
S_{c,j}^{feat}.
$$

这样可以回答：

> class-specific semantic leakage 是从哪一阶段开始写入 patch stream 的？

如果例如：

$$
block\ A_{p2c}^{9:12}
$$

就能明显降低 background similarity，那么说明 leakage 主要来自 late semantic blocks。

如果：

$$
block\ A_{p2c}^{1:4}
$$

影响最大，则说明很早就发生了 token-role contamination。

---

# 24. 第四个实验：Global Patch-to-Patch Dependency

LaST 的理论里另一半是：

$$
\text{global attention}.
$$

即使不经过 class token：

$$
p_i
\rightarrow p_j
$$

也能把 foreground semantics 扩散到 distant background。

因此我们还可以单独处理：

$$
A_{p2p}.
$$

不是立即训练 window Transformer。

先在 inference 时设置 locality mask：

$$
A_{p2p}(i,j)=0
$$

如果：

$$
d(i,j)>r.
$$

再重新归一化。

分别测试：

$$
r=1,2,4,\infty.
$$

同时 class-to-patch：

$$
A_{c2p}
$$

保持 global。

这能区分：

$$
\boxed{
\text{class-token semantic writing}
}
$$

与：

$$
\boxed{
\text{patch-to-patch global diffusion}
}
$$

谁是主要来源。

---

# 25. 于是可以得到一个非常干净的 2×2 diagnosis

|                   | Global \(P\rightarrow P\) |      Local \(P\rightarrow P\) |
| ----------------- | ------------------------: | ----------------------------: |
| Patches 可读 class  |                  baseline |         isolate P2P diffusion |
| Patches 不可读 class | isolate P2C contamination | strongest locality protection |

不需要先提出新方法。

这只是 mechanism intervention。

如果：

$$
\text{Block P2C}
$$

已经解决大部分 leakage，那么问题主要来自：

$$
class\rightarrow patch
$$

semantic contamination。

如果 local P2P 影响更大，则更接近 LaST：

$$
global dependency
\rightarrow
lazy aggregation.
$$

---

# 26. 第五个实验：Registers 能不能解决 MCTformer 的问题？

这一步也值得做，但应该在确认 baseline 现象之后。

Registers 的作用主要是：

$$
\boxed{
\text{提供 explicit scratch space。}
}
$$

因此在 MCTformer+ 中加：

$$
K=4
$$

generic registers。

然后比较：

### High-norm artifact

$$
\|p_j\|_2.
$$

### Class-specific semantic leakage

$$
S_{c,j}.
$$

如果出现：

$$
\text{high-norm outlier}\downarrow,
$$

但：

$$
BG\text{-}Tail
\approx
\text{unchanged},
$$

$$
C\text{-}PiM
\approx
\text{unchanged},
$$

那我们就几乎在 MCTformer+ 上复现了 LaST 的核心观点：

$$
\boxed{
\text{Register solves storage artifact, not semantic shortcut.}
}
$$

这对后面的研究定位非常重要。

---

# 27. 还可以联合 Patch Norm 与 Class Semantic Score

这是判断：

$$
\text{Registers 问题}
$$

和：

$$
\text{LaST 问题}
$$

是否为同一批 patches 的最好方式。

对每一个 background patch 计算：

$$
N_j=\|p_j\|_2,
$$

和：

$$
G_j
=
\max_{c\in Y^+}
S_{c,j}.
$$

然后把 patches 分成四类：

| Norm | Class similarity | Interpretation                |
| ---- | ---------------- | ----------------------------- |
| 高    | 高                | scratchpad + semantic leakage |
| 高    | 低                | computational artifact        |
| 低    | 高                | **pure semantic shortcut**    |
| 低    | 低                | normal background             |

我觉得特别值得看：

$$
\boxed{
\text{Low-norm / High-class-similarity background patches}
}
$$

有多少。

如果大量存在，那么：

> Register 从理论上就不可能完全解决 MCTformer background semantics。

---

# 28. MCTformer 的一个特别关键指标：P2C mass

因为我们已经可以拿到 attention matrix，所以每层可以统计：

$$
m_{p\rightarrow c}^{(l)}
=
\frac1N
\sum_i
\sum_{c}
A_{p2c}^{(l)}(i,c).
$$

然后和：

$$
BG\text{-}Tail^{(l)}
$$

做相关性：

$$
\rho
\left(
m_{p\rightarrow c}^{(l)},
BG\text{-}Tail^{(l)}
\right).
$$

如果随着 layer：

$$
m_{p\rightarrow c}\uparrow
$$

同时：

$$
BG\text{-}Tail\uparrow,
$$

然后 P2C blocking 又能让 BG-Tail 降低，那么我们就从：

$$
\text{correlation}
$$

走到了：

$$
\text{intervention}.
$$

这就非常有说服力。

---

# 29. 还有一个我认为很强的 class-specific context experiment

以 VOC 为例。

对于：

$$
boat,
train,
cow,
aeroplane
$$

这些典型 context-biased classes。

分别统计：

$$
S_{\mathrm{boat},j},
\quad j\in water,
$$

$$
S_{\mathrm{train},j},
\quad j\in railway/background,
$$

$$
S_{\mathrm{cow},j},
\quad j\in grass,
$$

等等。

VOC 没有 stuff label，不能直接知道 water/grass，但可以用：

* GT object mask 的 complement；
* 再按图像类别分组。

例如：

$$
\text{boat-present images}
$$

中所有非-boat patch。

比较：

$$
\text{boat absent images}
$$

中的 background。

如果 boat-present 图的 background 明显更像：

$$
c_{\mathrm{boat}},
$$

这就是 class-specific context imprinting。

COCO 更适合，因为类别共现更复杂。

---

# 30. Context-only / Object-only 可以作为更强的因果验证

这不是 LaST repo 当前直接提供的，但和其理论非常一致。

利用 VOC val GT：

### Object-only

保留 class \(c\) 对象：

$$
x_{\mathrm{obj}}.
$$

### Context-only

移除 class \(c\) 对象：

$$
x_{\mathrm{ctx}}.
$$

然后分别观察：

$$
z_c,
$$

以及：

$$
S_{c,j}^{bg}.
$$

最值得看的不是只有 classification score。

而是：

> 移除 foreground object 后，background patch 是否仍然保持对 class token 的高 similarity？

如果：

$$
S_{c,bg}(x_{\mathrm{ctx}})
$$

仍然非常高，那么 background semantics 更可能是：

$$
\boxed{\text{context-based class representation}}
$$

而不只是 foreground semantic diffusion。

如果移除对象后迅速消失，则说明它更可能是：

$$
\boxed{\text{foreground semantics 经 global attention 扩散进 background。}}
$$

这两个机制也是不一样的。

---

# 31. 所以“Lazy Semantic Assignment”最好先不要定义成一个单一 score

我目前不建议马上人为构造：

$$
LSA=\alpha A+\beta B+\gamma C.
$$

先保留几个物理意义清楚的原始指标：

| Metric               | 问题                                      |
| -------------------- | --------------------------------------- |
| C-PiM ↑              | class token 最相关 patch 是否属于正确类别？         |
| Other-FG Hit ↓       | 是否混淆其他前景类别？                             |
| BG Hit ↓             | 最相关 patch 是否落在背景？                       |
| BG-Tail@10 ↓         | 高 class-semantic patches 中有多少背景？        |
| FG–BG Margin ↑       | target patches 与背景的 semantic separation |
| \(m_{p2c}\)          | patches 读取多少 class semantics？           |
| Patch Norm           | 是否出现 scratchpad artifact？               |
| Masked Logit Drop    | high-score patch 是否有因果作用？               |
| Context-only Score ↓ | 是否依赖 context shortcut？                  |

等看到结果后，再决定是否需要一个统一的 Lazy Semantic Assignment Index。

---

# 32. 第一阶段我会怎么排列实验优先级

先只做**不改训练方法**的验证，我建议顺序就是下面这一组：

1. **MCTformer 与 MCTformer+ layer-wise Class-Patch Score**

   $$
   S_{c,j}^{(l)}
   =
   \cos(c_c^{(l)},p_j^{(l)}).
   $$

   输出 target / other-FG / BG 三分布，以及 C-PiM、BG-Hit、BG-Tail。

2. **Feature score vs Attention score vs CAM**

   同时对比：

   $$
   \cos(c_c,p_j),
   \quad
   A_{c2p},
   \quad
   CAM_c.
   $$

3. **Patch Norm × Class Similarity**

   区分：

   $$
   \text{register-style artifact}
   $$

   和：

   $$
   \text{semantic shortcut}.
   $$

4. **Class-specific causal masking**

   分别删除：

   $$
   TopBG,\ RandomBG,\ TopFG,\ RandomFG.
   $$

5. **P2C blocking**

   只禁止：

   $$
   patch\rightarrow class
   $$

   读取，观察 patch semantic leakage 是否下降。

6. **Global P2P locality intervention**

   判断：

   $$
   class\rightarrow patch
   $$

   与：

   $$
   patch\rightarrow patch
   $$

   哪条 global pathway 是主要污染源。

这六组实验已经足够判断：

$$
\boxed{
\text{MCTformer(+)
是否真的比普通 ViT 更容易形成 lazy semantic assignment}
}
$$

而且几乎全都是 **evaluation/inference-time analysis**，前四项不需要重新训练模型。

---

# 33. 什么结果才能支持你的核心假设？

我会认为下面这样的证据链最强：

$$
\boxed{
\textbf{Observation 1}
}
$$

MCTformer+ 中：

$$
S_{c,bg}
$$

在深层逐渐升高，并且：

$$
BG\text{-}Tail
$$

高于普通 ViT 或 MCTformer。

然后：

$$
\boxed{
\textbf{Observation 2}
}
$$

这些 background patches 与**当前存在的 class token**高度相关，而不是 generic CLS/global representation。

然后：

$$
\boxed{
\textbf{Observation 3}
}
$$

它们并不只是 high-norm artifacts：

$$
\text{大量 Low-Norm / High-Class-Sim BG patches 存在}.
$$

然后：

$$
\boxed{
\textbf{Observation 4}
}
$$

禁止：

$$
A_{p2c}
$$

或限制：

$$
A_{p2p}
$$

global propagation 后：

$$
S_{c,bg}\downarrow,
$$

$$
C\text{-}PiM\uparrow,
$$

而 classification 基本保持。

最后：

$$
\boxed{
\textbf{Observation 5}
}
$$

generic registers 可以消除 high-norm artifacts，却不能显著改善：

$$
BG\text{-}Tail
$$

或：

$$
C\text{-}PiM.
$$

如果真的出现这五项，我觉得就可以比较有把握地说：

> **MCTformer 的 coarse image-level supervision 与 multi-class global interaction 导致了一种比普通 ViT 更细粒度的 lazy semantic assignment：class-specific semantics 被写入与目标类别无关的 patches，尤其是背景 patches。**

这会比单纯说：

> WSSS 有 background noise

强很多。

---

# 34. 最关键的研究概念我会暂时这样定义

LaST 的：

$$
\boxed{
\text{Lazy Aggregation}
}
$$

重点是：

> global representation 懒惰地从 semantic shortcut patches 聚合信息。

而我们真正要验证的可能是：

$$
\boxed{
\text{Lazy Semantic Assignment}
}
$$

即：

> Under coarse image-level supervision, multi-class global interaction allows class-specific semantics to be assigned to spatial patches without respecting their true semantic ownership.

这里的“assigned”不一定意味着模型最终 classification 依赖这些 patches。

它首先意味着：

$$
\boxed{
p_j^{bg}
\approx
c_c
}
$$

即 patch representation 本身获得了不属于它的 class semantics。

如果进一步：

$$
z_c
$$

也依赖它，那就是更强的：

$$
\boxed{
\text{Class-specific Background Shortcut}.
}
$$

我觉得这两个层级最好从一开始就区分开。

---

另外还有一个很重要的源码细节：LaST 当前发布的 `visualize_patch_score_distribution.py` 其实同时保留了 `"repo"` 和 `"paper"` 两种 stability-score formula，实际训练代码 `conf.py` 使用的是 `patch / |lowpass-patch|`，并且按 **embedding channel** 在 patch 维度做 `topk`，再把选中的 patch values 汇聚为 global token。  这意味着后面如果真要把 LaST 的 **solution** 搬到 MCTformer+，必须先解决“一个 CLS 的 channel-wise selection 如何变成 C 个 class-specific tokens 的 selection”这个问题；但目前验证 lazy semantic assignment 完全不需要碰这一部分。

如果按实验优先级，我建议下一步先只实现 **1–3：class-specific patch score、三区域 distribution、layer-wise C-PiM/BG-Tail，以及 feature score vs \(A_{c2p}\) vs CAM**。这些结果出来以后，再决定是否值得做 P2C causal blocking。这样最省训练时间，也最先回答我们的核心假设。
