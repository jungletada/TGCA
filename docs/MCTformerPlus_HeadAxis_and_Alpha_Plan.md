# MCTformer+ 实验计划：逐头 CAM 质量热图（H 轴）与两分支 α 扫描

> 状态：计划稿，未实施。
> 宿主：MCTformer+ Small（DeiT-S，$L=12$ 层，$H=6$ 头，$C=20$ 类别 token，$D=384$，patch 16，448 输入 → $N=784$）。
> **两个实验都是推理端 only，零训练。**
> 建议落盘：`docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md`
> 输出：`results/head_alpha/20260918-voc/`
> **授权边界**：本队列只含 H 与 A 两个推理扫描。不授权任何训练、不授权可学习头权重、不授权 register / Gram / COCO / 额外 seed。后续由 §2.6 / §3.6 的门决定，需另开队列。

---

## 0. 前置条件

### 0.1 数值精度

鉴于已定位的 FP16 下溢路径（`models/tgca.py:129` 的 attention 转换、`models/mctformer_plus.py:1312` 的 pooling 转换），本队列的**全部前向与聚合强制 FP32，禁用 autocast，TF32 关闭**。两个实验都涉及注意力的逐元素幂次与逐头切片，对小值极其敏感——单头切片后的行质量比六头平均更小，下溢风险更高。

在 `manifest.json` 中显式记录 `precision: fp32` 与 `autocast: false`，并在报告首段声明。

### 0.2 checkpoint 选择与其污染状态

| checkpoint | 用途 | 已知污染程度 |
|---|---|---|
| `gwrp` s0（70.063） | 主宿主 | 不走 C2P 路径，未受影响 |
| `all_product` s0（72.315） | 对照宿主 | ep12 正类 TV 距离均值 0.000178，可忽略 |
| `all_product` s11 | **不使用** | 已确认受下溢污染 |


### 0.3 待确认的实现细节

动手前在代码里核对三件事，写进 manifest：

1. **`a_native` 的确切定义**——是 $\{L10,L11,L12\}$ 三层、跨 head 平均后的 $A_{c\to p}$ 切片，还是别的层集；
2. **$S$ 的确切形式**——按已有文档是 $S[c,j]=\sqrt{\mathrm{ReLU}(M[c,j])\cdot a_{\text{native}}[c,j]}$，即几何平均，**对应 §3 中 $\alpha=0.5$**；若实际是 $A_{cp}\odot M$（算术乘积），则当前点是 $\alpha=1$ 的另一种参数化，§3 的网格要相应改写；
3. **CAM 归一化方式**——min-max 还是仅 max。A3 的零结果暗示是 min-max，但要在代码里确认，因为它决定 §3 中哪些变换会被吸收。

---

## 1. 共同设置

沿用既有 CAM 评估协议，保证与七变体表逐项可比。

| 项目 | 设置 |
|---|---|
| 数据 | VOC train 1,464（CAM），VOC val 1,449（仅用于 §2.5 的无标注指标） |
| 尺度 | 原生 scales {1.0, 0.75, 1.25} + flip |
| 排序指标 | `fixed_mean_iou_percent` @ 0.45；best-threshold 仅作诊断 |
| 后处理 | 无 CRF、无 IRN |
| bootstrap | 图像聚类配对，5,000 次，seed 0 |
| 精度 | FP32 全程（§0.1） |

**两个实验共用同一个执行框架**：逐图前向一次，在内层循环里遍历所有配置，每个配置维护自己的混淆矩阵累加器。前向是唯一的大开销，配置数量几乎不增加成本。

---

## 2. 实验 H：逐 $(l,h)$ 的 CAM 质量热图

### 2.1 动机

MCTformer+ 把 $A\in\mathbb{R}^{12\times 6\times 20\times 784}$ 沿 $H$ 轴**直接平均**。DINO 的核心观察正是不同 head 关注不同物体与部件，跨 head 平均会把这种分工抹平。

与 $D$ 轴（通道均值）相比，$H$ 轴有两个优势：**先验更强**（head 分工是实测过的，通道分工是推测的）、**成本更低**（可学习权重只需 72 个参数）。

而你已经证明 $L$ 轴有影响（last3 与 all 的差异）——那么「同一层内不同 head 之间有没有影响」是紧接着的问题，而且没人问过。

### 2.2 设计：只替换 C2P 这一侧

CAM 生成里注意力进入两处：$a_{\text{native}}$（进 $S$）与 $P$（patch-patch 精修）。**本实验只替换前者，$P$ 保持原生不变**，以隔离 $H$ 轴对 C2P 的影响。

对每个 $(l,h)\in\{0..11\}\times\{0..5\}$：

$$a^{(l,h)}[c,j] = A^{(l)}[h,\,c,\,n_{\text{class}}+j]$$

代入原公式生成 CAM，其余流程一字不改。**72 个单元 + 1 个原生基线 = 73 配置。**

**归一化对照**：单头切片后各行的 patch 质量不等。由于 CAM 最终按类别归一化，逐类标量会被吸收，理论上无影响；但仍加 3 个「行归一化后」的控制单元（取热图中的最优、中位、最差三格）验证这一点。若出现差异，说明归一化不是逐类的，需回到 §0.3 第 3 条重新确认。

**可选第二张热图**：把 $P$ 换成单 $(l,h)$、$a_{\text{native}}$ 保持原生。它测的是 patch-patch 侧的 head 分工，与主实验正交。**先不做**，等主热图结果再定。

### 2.3 每格记录的量

| 类别 | 指标 | 需要标注 |
|---|---|---|
| 因变量 | fixed mIoU@0.45、best mIoU、FG precision / recall | 是 |
| 无标注 A | $\kappa^{(l,h)}$：$a^{(l,h)}$ 的有效支撑比（归一化困惑度） | 否 |
| 无标注 B | Gini | 否 |
| **无标注 C** | $\gamma^{(l,h)}$：**类无关度** | 否 |

$\gamma$ 的定义是本实验新增的量：

$$\gamma^{(l,h)} = \frac{1}{|\mathcal{P}|}\sum_{(c,c')\in\mathcal{P}} \cos\big(a^{(l,h)}[c,:],\, a^{(l,h)}[c',:]\big)$$

$\mathcal{P}$ 取图像中共现的正类对。**$\gamma\to 1$ 表示该 head 对所有类别给出同一张图**——这样的 head 即使 $\kappa$ 看起来漂亮，对类特异定位也是无用的。这正是 LaSt-ViT 的稳定性判据在多标签设定下失效的那一点：它是类无关的。$\gamma$ 把这件事变成一个可测量的量，而且**它是为 head 这个对象量身定的判据，不是硬搬**。

### 2.4 参考代码

```python
# analysis/head_heatmap.py
import torch, torch.nn.functional as F

@torch.no_grad()
def c2p_single_head(attn_layers, n_class: int, l: int, h: int,
                    row_normalize: bool = False, eps: float = 1e-8):
    """返回 (B, C, N) 的单层单头 class-to-patch 注意力。全程 FP32。"""
    a = attn_layers[l].float()[:, h, :n_class, n_class:]      # (B, C, N)
    if row_normalize:
        a = a / a.sum(-1, keepdim=True).clamp_min(eps)
    return a


@torch.no_grad()
def effective_support(a, eps: float = 1e-12):
    """kappa: 归一化困惑度, 值域 [1/N, 1]。a 先按 N 维归一化。"""
    N = a.shape[-1]
    p = (a / a.sum(-1, keepdim=True).clamp_min(eps)).clamp_min(eps)
    return (-(p * p.log()).sum(-1)).exp() / N                  # (B, C)


@torch.no_grad()
def class_agnosticism(a, y, eps: float = 1e-12):
    """gamma: 共现正类对之间注意力图的平均余弦。a:(B,C,N), y:(B,C) 0/1"""
    ah = F.normalize(a.clamp_min(0), dim=-1)
    S = ah @ ah.transpose(-1, -2)                              # (B, C, C)
    C = S.shape[-1]
    eye = torch.eye(C, device=S.device, dtype=torch.bool)
    pair = (y.unsqueeze(2) * y.unsqueeze(1)).bool() & ~eye
    denom = pair.sum(dim=(1, 2)).clamp_min(1)
    return (S * pair).sum(dim=(1, 2)) / denom                  # (B,)
```

**执行结构**（关键，决定成本）：

```
for image in dataset:                    # 唯一的大开销
    for scale, flip in views:
        tokens, attn = model.forward_features(x)   # FP32
        M = patch_branch_logits(tokens)
        P = aggregate_p2p_native(attn)             # 原生, 不变
        for (l, h) in cells + [NATIVE]:
            a = c2p_single_head(attn, n_class, l, h)
            S = build_S(M, a)                      # 原生公式
            cam = S @ P.transpose(-1, -2)
            accumulate(cell_id, cam)               # 混淆矩阵累加
```

存储：73 配置 × 50 阈值档 × 21×21 × 8 B ≈ 12.9 MB。可忽略。

### 2.5 三个层次的分析

**第一层：热图本身。** $12\times 6$ 的 mIoU 热图，与原生基线并排。三种读法：

| 热图形态 | 结论 | 动作 |
|---|---|---|
| 存在单格 > 原生基线 | **$H$ 轴聚合有损** | 可学习头权重值得做，热图直接给初始化先验 |
| 极差 < 2 pp，无结构 | **六个 head 无分工** | $H$ 轴作废，省下两周 |
| 有结构但最优单格 < 原生 | **分工存在但互补** | 该做的是**加权**而非**选择** |

**第二层：无标注判据的相关性。** 跨 72 格算 $\kappa$、$\gamma$ 与 mIoU 的 Spearman。若某个判据 $|\rho_s|\ge 0.7$，就存在一个**不需要任何标注的 head 选择准则**。这是本实验真正的回报。

**第三层：按无标注判据的 top-$m$ 平均。** 用 $\kappa$（或 $\gamma$，或两者的组合）对 72 格排序，取前 $m$ 格平均生成 CAM，$m=1,2,4,8,16,32,72$（$m=72$ 且层集为全部时应退化为一个已知配置，作为自检）。

**这一层必须用无标注判据排序，不得用 mIoU 排序**——后者是用 GT 选超参，正是我们在批评别人的那种泄漏。若报告中同时给出 mIoU 排序的 top-$m$ 曲线，必须明确标为 oracle 上界，不得作为方法结果。

### 2.6 预先注册的判定门

记 $m^\ast$ 为 72 格中的最大 fixed mIoU，$m_0$ 为该宿主的原生基线（gwrp 70.063 / all_product 72.315）。

| 条件 | 结论 | 动作 |
|---|---|---|
| $m^\ast > m_0$ 且配对 bootstrap 95% CI 下界 > 0，**且在两个宿主上符号一致** | $H$ 轴有损，成立 | 开可学习头权重队列（$\boldsymbol\pi=\mathrm{softmax}(\boldsymbol\theta)$，$\boldsymbol\theta\in\mathbb{R}^{12\times 6}$，零初始化 → 与均匀精确等价） |
| 热图极差 < 2 pp | 无分工 | $H$ 轴封盘 |
| 第三层中存在 $m$ 使无标注 top-$m$ > $m_0$ | **无标注 head 选择准则成立** | 这比可学习权重更值钱，优先写 |
| 以上皆否 | 记录为负结果 | 转 §3 |

多重比较：72 格在 95% 下期望约 3.6 个假阳性。**因此第一行强制要求在 `gwrp` 与 `all_product` 两个宿主上符号一致**，以复现代替 Bonferroni，与 β1 扫描采用同一套纪律。

---

## 3. 实验 A：两分支融合指数 α 的推理端扫描

### 3.1 动机

MCTformer 的 $S$ 把 class-patch 注意力与 patch 分支 CAM 以**固定的等权几何平均**融合。而 CLIP-ES / WeCLIP 那条线用的是 $\big(\frac{R_{\text{nor}}+R_{\text{nor}}^\top}{2}\big)^{\alpha}\cdot M$，$\alpha=2$——**说明这个指数本来就是可调的，只是 MCTformer 系列没人调过。**

### 3.2 参数化

$$S_\alpha[c,j] = \mathrm{ReLU}(M[c,j])^{\,1-\alpha}\cdot a_{\text{native}}[c,j]^{\,\alpha}$$

- $\alpha = 0.5$ → 当前实现（几何平均）
- $\alpha \to 0$ → 纯 patch 分支 CAM
- $\alpha \to 1$ → 纯注意力

**网格**：$\alpha \in \{0, 0.05, \dots, 1.0\}$，21 个点。

若 §0.3 第 2 条核对发现实现是算术乘积 $A_{cp}\odot M$ 而非几何平均，则改写为 $M^{1-\alpha}a^{\alpha}$ 后当前点落在 $\alpha=0.5$ 之外，网格中心相应平移；**这一步必须在开跑前定稿并写入 manifest**。

数值保护：两个底数都 `clamp_min(eps)` 后再取幂，eps = 1e-8，FP32。$\alpha$ 接近 0 或 1 时另一侧的幂次接近 0，$x^0=1$ 会把整图抬成常数——**$\alpha=0$ 与 $\alpha=1$ 两个端点必须单独验证其输出与「纯 $M$」「纯 $a$」逐值一致**，否则说明 eps 或 clamp 引入了偏差。

### 3.3 交叉拟合：避免用 GT 选 α

在 VOC train 1,464 上用 GT 扫 $\alpha$ 再报同一集合的结果，**就是我们批评 WSOL 领域的那种泄漏**。

做法：把 1,464 张按图像 ID 哈希**固定二分**为 A / B 两半（比例 1:1，分层保证类别分布相近）。

- 在 A 上选最优 $\alpha_A$，报 B 上的 mIoU；
- 在 B 上选最优 $\alpha_B$，报 A 上的 mIoU；
- **主报告数字 = 两个留出半的加权平均**。

这不增加任何计算——所有 $\alpha$ 对所有图像都已算出，只是换一种聚合方式。同时报告 $\alpha_A$ 与 $\alpha_B$ 是否一致：**若两半选出的 $\alpha$ 相差大于一个网格步长，说明这个超参本身不稳定，不应作为方法的一部分。**

同样的交叉拟合原则**建议顺带应用到阈值 0.45 上**，作为报告里的一个附注——这会引出「现有 WSSS 的阈值选择同样有泄漏」这个观察，是论文里一句有分量的话。

### 3.4 参考代码

```python
# analysis/alpha_sweep.py
import torch

ALPHAS = [round(0.05 * k, 2) for k in range(21)]

@torch.no_grad()
def blend(M: torch.Tensor, a: torch.Tensor, alpha: float, eps: float = 1e-8):
    """M, a: (B, C, N) FP32。alpha=0.5 复现当前的几何平均。"""
    m = M.clamp_min(0).clamp_min(eps)
    aa = a.clamp_min(eps)
    if alpha == 0.0:
        return m
    if alpha == 1.0:
        return aa
    return m.pow(1.0 - alpha) * aa.pow(alpha)
```

端点自检（`tests/test_alpha_blend.py`）：

- `blend(M, a, 0.5)` 与现有 `sqrt(relu(M) * a)` 在 `clamp` 之外的区域 `allclose`（atol=1e-6）；
- `blend(M, a, 0.0)` 与 `M.clamp_min(0)` 在非零区域一致；
- `blend(M, a, 1.0)` 与 `a` 一致；
- 单调性抽查：固定像素，$\alpha$ 从 0 到 1 时 $S_\alpha$ 在 $M>a$ 与 $M<a$ 两种情形下单调方向相反。

### 3.5 记录的量

每个 $\alpha$ × 每个宿主：fixed / best mIoU、FG precision / recall、**前景面积占比**（t=0.45 下判为任意前景的像素比例）。

前景面积占比是这个扫描的关键诊断：它直接显示 $\alpha$ 是在**扩张**还是**收缩**激活区域，从而把「哪一支主导」这件事从 mIoU 的单一数字里拆出来。

### 3.6 预先注册的判定门

| 条件 | 结论 |
|---|---|
| 交叉拟合结果 > 当前 $\alpha=0.5$，且 95% CI 下界 > 0，**且两个宿主符号一致，且 $\|\alpha_A-\alpha_B\|\le 0.05$** | $\alpha$ 是有效的可调量，进消融表 |
| 曲线在 $\alpha\in[0.4,0.6]$ 内平坦（极差 < 0.3 pp） | 当前选择已近最优，**干净的负结果**，一句话带过 |
| $\alpha_A$ 与 $\alpha_B$ 相差 > 0.05 | 超参不稳定，不作为方法；只作为敏感性分析报告 |
| 最优 $\alpha$ 落在端点（0 或 1） | 说明某一支完全主导，**这本身是关于两分支设计的实质发现**，优先写 |

最后一行值得展开：如果最优 $\alpha \to 1$（纯注意力），那意味着 patch 分支的 CAM 在融合中是多余的；如果 $\alpha\to 0$，那意味着 class-patch 注意力对最终 CAM 没有贡献——**任何一端都会动摇 MCTformer 双分支设计的前提**，比中间某个值涨 0.5 pp 有意思得多。

---

## 4. 排期

| 序 | 动作 | 成本 | 门 |
|---|---|---|---|
| 0 | §0.2 的 ep45 × 全 val TV 对照；§0.3 三项代码核对 | 半天 | **TV 仍在 1e-4 量级** |
| 1 | 写 `head_heatmap.py` / `alpha_sweep.py` + 单测 | 半天 | 端点自检与几何平均等价性通过 |
| 2 | **实验 A**（21 配置 × 2 宿主） | **1 小时** | §3.6 |
| 3 | **实验 H**（73 配置 × 2 宿主） | **2–3 小时** | §2.6 |
| 4 | 第三层 top-$m$ 分析 | 1 小时 | §2.6 第三行 |

**实验 A 先跑**：配置少、代码简单、可能直接涨点，同时它会先把执行框架跑通，实验 H 只是换一个内层循环。

全部不占训练队列，可与任何训练并行。

---

## 5. 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| 单头切片后行质量过小，FP32 下仍有精度问题 | 低 | §0.1 全程 FP32；另记录每格 $a$ 的最小非零值，低于 1e-30 时告警 |
| 72 格的多重比较产生假阳性 | **中高** | 两宿主符号复现（§2.6） |
| $\alpha$ 扫描用 GT 选超参被指为泄漏 | 中 | §3.3 交叉拟合，从设计上消除 |
| top-$m$ 分析不慎用 mIoU 排序 | 中 | §2.5 第三层已明确禁止；oracle 曲线必须单独标注 |
| $a_{\text{native}}$ / $S$ / 归一化的实现与假设不符 | **中** | §0.3 三项核对是硬前置，未完成不得开跑 |
| 两个实验的结论互相依赖（α 的最优值随 head 选择而变） | 中 | 主线保持正交（A 用原生 $a$，H 用原生 $\alpha$）；仅在两者都通过门时，跑一个 5 头 × 21 α 的小联合切片 |

---

## 6. 与论文的接口

这两个实验补的是同一个框架里的两块：

> 多类别 token transformer 通过四次约简把 $A\in\mathbb{R}^{L\times H\times C\times N}$ 与 $\mathbf{T}_{cls}\in\mathbb{R}^{C\times D}$ 压成 $C$ 个类别 logit。
>
> | 轴 | 当前 | 状态 |
> |---|---|---|
> | $L$ 层 | mean / product，last3 或 all | 已探索 |
> | $N$ patch | GWRP → C2P | 已探索（+2.25 pp） |
> | **$H$ 头** | **均匀平均** | **本实验 H** |
> | **$D$ 通道** | **均匀平均** | A1 已跑，$\mathbf{a}$ 未偏离均匀，待 A1-b |
>
> 此外，两分支的融合指数固定为等权几何平均，同样未被消融（本实验 A）。

**四个轴、两个已知、两个新填**，加上直流基底那节分析、传播的推理/训练不对称、以及 FP16 下溢那个数值稳定性发现——这是一篇结构完整的论文，而且不依赖任何单点结果成立。

即使 H 与 A 全负，「我们系统地检查了全部四个约简轴与融合指数，其中两个是可改进的、两个已接近最优」仍然是一个完整的陈述，比现在的「我们换了个池化权重」强得多。
