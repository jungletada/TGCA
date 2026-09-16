# MCTformer+ 实验计划：Artifact 探针 与 Affinity 修复

> 状态：计划稿，未实施。撰写日期基准 2026-09-16。
> 宿主：`jungletada/TGCA`，MCTformer+ Small（DeiT-S，12 blocks，6 heads，dim 384，20 class tokens，patch 16，448 输入 → 28×28=784 patch，class-first 拼接）。
> 对照基准：`docs/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md` 的七组 seed-0 结果。
> 建议落盘：`docs/MCTformerPlus_Artifact_and_Affinity_Repair_Plan.md`

---

## 0. 范围

本计划只覆盖两件事：

- **实验 1（Artifact 探针）**：不训练、不改模型，用已有 checkpoint 判断 register token 值不值得做。
- **实验 2（Affinity 修复）**：先做零训练的离线筛选，再只重训被筛出的少数配置。

不覆盖：register token 的实现、Gram 锚定、多 seed 补跑、COCO。那些取决于本计划的结论。

---



## 1. 共同约定


| 项目    | 设置                                                                           | 理由                                       |
| ----- | ---------------------------------------------------------------------------- | ---------------------------------------- |
| 探针集   | VOC val 1,449 张（与现有权重诊断同一集合）                                                 | 与 `weight_diagnostics/summary.csv` 可直接并排 |
| 多标签子集 | 522 张（现有 `multi` stratum）                                                    | Jaccard 类指标沿用                            |
| 变换    | 短边 resize 512（bicubic）→ center crop 448，无增强                                  | 与现有分类复评一致                                |
| 精度    | FP32，TF32 off，`autocast` 关闭                                                  | 二阶统计与 Gram 类量在 FP16 下不可靠                 |
| 模式    | `model.eval()`，`torch.no_grad()`，探针结束断言 `model.training` 恢复                  | drop_path 会污染 attention                  |
| 随机性   | seed=20260916，bootstrap 5,000 次图像级配对                                         | 与现有诊断同口径                                 |
| 产出    | `results/` 下独立目录 + `manifest.json` + `commands.sh` + `checkpoint_sha256.txt` | 沿用仓库规范                                   |


**参与诊断的 checkpoint（三个，全部已存在）**：


| 标签                     | 目录                                                                          | 为什么要它        |
| ---------------------- | --------------------------------------------------------------------------- | ------------ |
| `gwrp`                 | `results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12`               | 未接触 C2P 的对照  |
| `all_product`          | `results/c2p_pooling/20260914-voc-product-s0/all_product`                   | 当前最佳（72.315） |
| `all_product_affinity` | `results/c2p_pooling/20260914-voc-product-affinity-s0/all_product_affinity` | 崩塌组（69.406）  |


三个一起跑，成本几乎不增，但能回答一个额外问题：**通过 P 反向传播训练，会不会反而制造更多 artifact？**

---



## 2. 实验 1：Artifact 探针



### 2.1 原理与动机

Darcet 等人的观察是：ViT 会把语义无关的低信息背景 patch 征用为全局信息的缓存，表现为少数 patch 的输出范数异常高，并在注意力图上成为「全局吸引子」——所有 query 都高度关注它们。register token 的作用是提供无监督的草稿纸，把这部分信息吸走。

**为什么这在 MCTformer+ 上特别有理由**：模型已经有 20 个额外 token，但它们**全部被分类损失监督**，因此没有自由充当草稿纸。等于说 CLS 的位置被占满了，却没有留缓存空间。

**为什么这与实验 2 直接相关**：如果存在吸引子 patch，$P$ 中就有若干列质量异常大，一次传播必然把权重摊到它们身上。**artifact 可能正是 affinity 传播失败的物理原因之一**，两个实验因此必须一起读。

### 2.2 指标

设 L12 patch token 输出 $X\in\mathbb{R}^{N\times d}$（$N=784$），逐层 head 平均后的 patch-patch 块 $\bar A_l\in\mathbb{R}^{N\times N}$。


| 编号     | 指标      | 定义                                                                | artifact 存在的表现           |
| ------ | ------- | ----------------------------------------------------------------- | ------------------------ |
| **M1** | 范数比     | $\rho_j=\lVert X_j\rVert_2 / \mathrm{median}_k\lVert X_k\rVert_2$ | 出现 $\rho\gg 1$ 的长尾       |
| **M2** | 高范数占比   | $\mathrm{frac}_{>3}=\frac{1}{N}\lvertj:\rho_j>3\rvert$            | 0.5%–5% 之间               |
| **M3** | 逐层范数演化  | 每层的 $p_{99}/p_{50}$                                               | 在某个中间层突然抬升               |
| **M4** | 吸引子得分   | $g_j^{(l)}=\sum_i \bar A_l[i,j]$，归一化后取 Gini 与 top-1% 质量份额         | top-1% patch 吸走远超 1% 的质量 |
| **M5** | 两种症状的重合 | top-1% by $\rho$ 与 top-1% by $g^{(12)}$ 的 Jaccard                 | 显著高于随机（≈0.01）            |
| **M6** | 位置是否低信息 | 高范数 patch 处的图像梯度幅值 vs 全图中位数                                       | 显著偏低（落在平坦背景）             |


M6 用图像梯度作为「低信息」的代理，**不需要任何 GT**，符合无标注诊断的定位。

### 2.3 判定规则（预先写定，避免事后挑选）

**register 值得做**，当且仅当以下三条同时成立：

1. $\mathrm{frac}_{>3} \ge 0.5$（存在高范数群体，而非单峰分布）；
2. M4 的 top-1% 质量份额 $\ge 5$（即吸引子效应至少是均匀情形的 5 倍）；
3. M5 的 Jaccard $\ge 0.2$（两种症状确实指向同一批 patch，而不是两个不相干现象）。

**任一不成立 → register 立刻止损**，把资源全部投给实验 2 和后续的 Gram 方向。

附加读数（不参与判定，但要记录）：`all_product_affinity` 的 M2/M4 是否高于 `all_product`。若明显更高，说明**通过 $P$ 训练会放大 artifact**，这本身是论文里一句有分量的话。

### 2.4 参考代码

```python
# analysis/artifact_probe.py
import math, torch

@torch.no_grad()
def patch_norm_stats(patch_tokens: torch.Tensor, hi: float = 3.0) -> dict:
    """patch_tokens: (B, N, D) — L12 的 patch token 输出（FP32）。"""
    norms = patch_tokens.float().norm(dim=-1)                     # (B, N)
    med = norms.median(dim=-1, keepdim=True).values.clamp_min(1e-8)
    ratio = norms / med                                           # (B, N)
    return {
        "ratio": ratio,                                           # 留给直方图
        "frac_hi": (ratio > hi).float().mean(dim=-1),             # M2  (B,)
        "p99_over_p50": torch.quantile(norms, 0.99, dim=-1) / med.squeeze(-1),
    }


@torch.no_grad()
def attractor_stats(attn_pp: torch.Tensor, topk_frac: float = 0.01) -> dict:
    """attn_pp: (B, N, N) — 某一层 head 平均后的 patch-patch 块.
    g[j] = sum_i A[i, j] : patch j 作为 key 被接收的总质量."""
    g = attn_pp.float().sum(dim=1)                                # (B, N)
    g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    B, N = g.shape

    # Gini, 排序法 O(N log N)
    x, _ = g.sort(dim=-1)
    idx = torch.arange(1, N + 1, device=g.device, dtype=g.dtype)
    gini = ((2 * idx - N - 1) * x).sum(-1) / (N * x.sum(-1).clamp_min(1e-12))

    k = max(1, int(round(topk_frac * N)))
    top_share = g.topk(k, dim=-1).values.sum(-1)                  # M4
    return {"recv": g, "gini": gini, "top_share": top_share, "k": k}


@torch.no_grad()
def symptom_overlap(ratio: torch.Tensor, recv: torch.Tensor,
                    topk_frac: float = 0.01) -> torch.Tensor:
    """M5: 高范数集合 与 高接收质量集合 的 Jaccard. 随机基线 ≈ topk_frac."""
    B, N = ratio.shape
    k = max(1, int(round(topk_frac * N)))
    a = torch.zeros_like(ratio, dtype=torch.bool)
    b = torch.zeros_like(recv, dtype=torch.bool)
    a.scatter_(1, ratio.topk(k, dim=-1).indices, True)
    b.scatter_(1, recv.topk(k, dim=-1).indices, True)
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float().clamp_min(1.0)
    return inter / union


@torch.no_grad()
def lowinfo_score(images: torch.Tensor, ratio: torch.Tensor,
                  grid: int = 28, topk_frac: float = 0.01) -> torch.Tensor:
    """M6: 高范数 patch 处的图像梯度幅值 / 全图中位数. <1 表示落在平坦区域."""
    x = images.float().mean(dim=1, keepdim=True)                  # (B,1,H,W) 灰度
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    g = (gx[..., :-1, :].pow(2) + gy[..., :, :-1].pow(2)).sqrt()
    g = torch.nn.functional.adaptive_avg_pool2d(g, grid).flatten(1)   # (B, N)

    B, N = g.shape
    k = max(1, int(round(topk_frac * N)))
    sel = ratio.topk(k, dim=-1).indices
    hi = g.gather(1, sel).mean(-1)
    med = g.median(dim=-1).values.clamp_min(1e-8)
    return hi / med
```

**接入方式**：不要改 `models/mctformer_plus.py`。在 `analysis/artifact_probe.py` 里直接调用模型的 `forward_features`，取回 `(tokens, attn_layers)`，用 `n_class=20` 切片：

```python
patch_tokens = tokens[:, n_class:, :]                  # (B, N, D)
attn_pp_l = attn_layers[l].mean(dim=1)[:, n_class:, n_class:]   # (B, N, N)
```



### 2.5 成本与产出

1,449 张、batch 16、DeiT-S@448、仅前向，三个 checkpoint 合计约 **20–30 分钟**（含 12 层 attention 的切片与统计）。若显存紧张，M3 的逐层统计可只取 L4/L8/L12 三层。

落盘 `results/artifact_probe/20260916-voc-s0/`：

- `summary.csv`：三个 checkpoint × 六个指标的均值与 bootstrap CI
- `norm_hist_{ckpt}.png`：M1 直方图（log y 轴）
- `attractor_map_{ckpt}_{img_id}.png`：若干张 $g^{(12)}$ 的 28×28 热力图
- `decision.md`：三条判定规则逐条打勾/打叉，给出 register 做/不做的结论

---



## 3. 实验 2：Affinity 修复



### 3.1 原理：直流基底（DC floor）

这是整个修复的核心论证。

把每层 head 平均后的 patch-patch 行分解为「均匀分量 + 结构分量」：

$$\bar A_l[i,j] = \frac{f_l[i]}{N} + Q_l[i,j],\qquad \sum_j Q_l[i,j]\approx 0$$

浅层的自注意力接近弥散，$Q_l$ 幅值很小，$f_l[i]$ 接近该行的全部 patch 质量。你当前的定义是**跨 12 层直接求和且不做任何归一化**：

$$P[i,j]=\sum_{l=1}^{12}\bar A_l[i,j]=\frac{f[i]}{N}+Q[i,j],\qquad f[i]=\sum_l f_l[i]\approx 10\text{–}12$$

传播后（注意 $w[c,\cdot]$ 是和为 1 的概率）：

$$r[c,i]=\sum_j P[i,j]w[c,j]=\frac{f[i]}{N}+(Qw)[c,i]$$

**第一项与 $c$ 无关、与 $w$ 的结构无关**——它是一个纯粹的常数基底。归一化后：

$$w_{\text{aff}}[c,i]=\frac{f[i]/N+(Qw)[c,i]}{f+\sum_i (Qw)[c,i]}$$

当 $f \gg \sum_i (Qw)$ 时，$w_{\text{aff}}\to 1/N$，即**退化为 GAP**。

你的诊断数字与这个预测完全吻合：


|                        | 传播前   | 传播后       | 均匀      |
| ---------------------- | ----- | --------- | ------- |
| all-product 归一化熵       | 0.294 | **0.972** | 1.0     |
| all-product top-1 mass | 0.465 | **0.013** | 0.00128 |


**结论：崩塌不是「P2P 无用」的证据，而是「12 层未归一化求和引入了一个压倒性的直流基底」的证据。**

### 3.2 一个必须澄清的疑问：为什么推理端的同一个 $P$ 没崩？

原生 CAM 用的是完全相同的 $P$：

$$\mathrm{CAM}[c,i]=\sum_j P[i,j]S[c,j],\quad S[c,j]=\sqrt{\mathrm{ReLU}(M[c,j])\cdot a_{\text{native}}[c,j]}$$

同样会得到 $\frac{f[i]}{N}\sum_j S[c,j] + (QS)[c,i]$，**基底一样存在**。它没有致命，可能的原因有二：$S$ 经过 ReLU 与 $\sqrt{\cdot}$ 后本身极稀疏尖锐，结构项的相对幅度更大；以及后续的逐类归一化 + 固定阈值对加性偏移的容忍度，高于「和为 1 的加权平均」对它的容忍度。

**但这只是解释假设，没有被测过。** 所以下面的 Stage A3 把同一套去基底操作也施加到推理端——**那是一个零训练、可能直接白拿 mIoU 的实验**。

### 3.3 设计变量


| 变量         | 取值                                              | 作用                                                                                |
| ---------- | ----------------------------------------------- | --------------------------------------------------------------------------------- |
| `p_layers` | `last3`(L10-12) / `last6`(L7-12) / `all`(L1-12) | 直接决定基底大小                                                                          |
| `p_reduce` | `sum`（当前） / `mean` / `rownorm_mean`（每层先行归一化再平均） | 消除层数带来的尺度                                                                         |
| `p_alpha`  | 1.0（当前） / 2.0 / 4.0                             | 逐元素幂次锐化，压低基底                                                                      |
| `p_sym`    | off（当前） / on：$(P+P^\top)/2$                     | 对齐 CLIP-ES 那条线的标准做法                                                               |
| `floor`    | `none`（当前） / `min` / `mean`                     | **显式去直流**，最直接的修复                                                                  |
| `beta`     | 1.0（当前） / 0.5 / 0.3 / 0.1                       | 阻尼：$w_{\text{aff}}=\mathrm{norm}\big((1-\beta)w+\beta\mathrm{norm}(wP^\top)\big)$ |


当前失败配置 = `(all, sum, α=1, sym=off, floor=none, β=1.0)`，**六个变量全部取在最容易塌的一端**。这本身就说明之前那次实验是在检验一个极端点，而不是在检验 P2P 这个机制。

### 3.4 Stage A：零训练离线筛选

**关键洞察：崩塌是 $P$ 和 $w$ 的纯函数，不依赖训练。** 用已训好的 `all_product` checkpoint（它从未用 affinity 训练过，$w$ 干净）就能把全部配置筛一遍。

**实现要点**：不要缓存 $P$（FP32 下 $12\times784^2$ ≈ 29 MB/图，1,449 张会爆盘）。改为**逐图前向一次，在内层循环里遍历所有配置**——每个配置只是几次 $784\times784$ 的矩阵运算，开销可忽略。

**A1 — 权重统计筛选**。对每个配置计算：

- 归一化熵 $H(w_{\text{aff}})/\log N$
- top-1 mass
- **DC 份额** $N\cdot\min_i w_{\text{aff}}[c,i]$（新指标：越接近 1 越接近均匀，直接量化基底）
- 正类对 top-10% Jaccard（沿用现有口径，$\lceil 0.1\times784\rceil=79$）

**筛选带**（预先写定，作为先验而非结论）：熵 $\in[0.35,0.65]$ 且 DC 份额 $<0.5$。依据是传播的目的是从判别性峰值扩散到完整物体，而不是抹平——参照 all-product 传播前是 0.294、last3-product 是 0.602，两者 mIoU 分别为 72.315 / 71.665。

**A2 — 真正的把关：把修复后的传播搬到推理端 CAM，直接测 mIoU。**

用 `all_product` 的权重，把 CAM 公式里的 $P$ 换成修复版，重跑 raw CAM 评估（scales 1.0/0.75/1.25 + flip，固定阈值 0.45，VOC train 1,464）。**这不需要重训，但给出的是真正关心的指标。**

**A3 — 顺手的副产品**：对 `gwrp` 基线做同样的事。如果去基底能把 70.063 抬上去，那是一个**完全免费的 baseline 提升**，而且说明这个问题不只存在于你的 C2P 分支，而是 MCTformer 原生 CAM 流程里就有的——那会是论文里更有分量的一句话。

### 3.5 Stage B：重训

从 Stage A 里选 **3 个**配置重训（seed=0，其余协议与七组完全一致）：

1. **最小改动组**：`(all, sum, α=1, sym=off, floor=min, β=1.0)` — 只加去基底，隔离单因素；
2. **A2 最佳组**：Stage A2 里 mIoU 最高的那一组；
3. **保守阻尼组**：`(last3, rownorm_mean, α=2, sym=on, floor=min, β=0.3)` — 全部变量取保守端。

三次训练按现有配置（45 epochs、448、有效 batch 32、330 updates/epoch）各约与之前一组相同的耗时。

### 3.6 参考代码

在 `models/mctformer_plus.py` 中**新增**下面两个函数，并让 `c2p_affinity_weights` 调用它们；保留原实现作为 `floor="none", beta=1.0, p_reduce="sum", p_layers="all"` 的默认路径，**保证旧结果可复现**。

```python
# models/mctformer_plus.py  (新增)
import torch

_P_LAYER_SETS = {
    "last3": (9, 10, 11),           # 0-indexed: L10-L12
    "last6": (6, 7, 8, 9, 10, 11),
    "all":   tuple(range(12)),
}


def aggregate_p2p(attn_layers, n_class: int,
                  p_layers: str = "all",
                  p_reduce: str = "sum",
                  p_alpha: float = 1.0,
                  p_sym: bool = False,
                  eps: float = 1e-8) -> torch.Tensor:
    """把逐层 joint attention 聚合成 patch-patch 传播算子 P.

    attn_layers: 长度 12 的序列, 每个 (B, H, T, T), class-first, T = n_class + N
    返回 P: (B, N, N), P[b, i, j] = query patch i 从 key patch j 接收的质量
    """
    idx = _P_LAYER_SETS[p_layers]
    mats = []
    for l in idx:
        a = attn_layers[l].float().mean(dim=1)              # (B, T, T) head 平均
        a = a[:, n_class:, n_class:]                        # (B, N, N) 切 patch 块
        if p_reduce == "rownorm_mean":
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
        mats.append(a)
    P = torch.stack(mats, dim=0)                            # (L, B, N, N)

    if p_reduce == "sum":
        P = P.sum(dim=0)
    else:                                                    # mean / rownorm_mean
        P = P.mean(dim=0)

    if p_alpha != 1.0:
        # 幂次锐化前先行归一化, 避免 alpha 同时改变尺度
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        P = P.clamp_min(0).pow(p_alpha)

    if p_sym:
        P = 0.5 * (P + P.transpose(-1, -2))

    return P


def propagate_c2p_weights(w: torch.Tensor, P: torch.Tensor,
                          floor: str = "none",
                          beta: float = 1.0,
                          eps: float = 1e-8) -> torch.Tensor:
    """w: (B, C, N), 已在 N 维和为 1;  P: (B, N, N).
    r[c,i] = sum_j P[i,j] w[c,j]   —— 与推理端 CAM 同向, 勿改。
    """
    r = torch.einsum("bij,bcj->bci", P, w)                  # (B, C, N)

    # --- 去直流基底: 本修复的核心 ---
    if floor == "min":
        r = r - r.amin(dim=-1, keepdim=True)
    elif floor == "mean":
        r = (r - r.mean(dim=-1, keepdim=True)).clamp_min(0.0)
    elif floor != "none":
        raise ValueError(f"unknown floor: {floor}")

    r = r / r.sum(dim=-1, keepdim=True).clamp_min(eps)

    # --- 阻尼: 保留一部分未传播的原始选择性 ---
    if beta < 1.0:
        w = (1.0 - beta) * w + beta * r
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(eps)
    else:
        w = r
    return w
```

诊断量（Stage A 用，也可挂进训练日志）：

```python
# analysis/weight_stats.py
import math, torch

@torch.no_grad()
def weight_stats(w: torch.Tensor, eps: float = 1e-12) -> dict:
    """w: (B, C, N), 每行和为 1. 只在图像的正类上调用。"""
    N = w.shape[-1]
    p = w.clamp_min(eps)
    return {
        "entropy":  -(p * p.log()).sum(-1) / math.log(N),   # 归一化熵
        "top1":     w.amax(-1),                             # top-1 mass
        "dc_share": w.amin(-1) * N,                         # 新: 直流份额, →1 即均匀
    }
```

**必须配套的单元测试**（`tests/test_affinity_repair.py`）：

- 旧路径等价性：`floor="none", beta=1.0, p_reduce="sum", p_layers="all"` 的输出与重构前的 `c2p_affinity_weights` **逐值一致**（`torch.allclose`，atol=1e-6）。**这条最重要**，否则七组历史结果失去可比性。
- `beta=0` 时输出恒等于输入 `w`。
- 人造均匀 $P$：`floor="none"` 输出为均匀分布；`floor="min"` 输出退化为全零后被 clamp，需明确约定此边界行为（建议：`r.sum()` 低于阈值时回退为原 `w` 并计数告警）。
- `p_sym=True` 时 $P$ 对称。
- 幂次 $\alpha=2$ 时熵单调不增。



### 3.7 判定


| 结果                              | 结论                  | 下一步                         |
| ------------------------------- | ------------------- | --------------------------- |
| A2 中某配置的 raw CAM > 72.315（当前最佳） | **P2P 有效，之前是配置问题**  | 进 Stage B，affinity 成为主方法之一  |
| A2 最佳落在 70.1–72.3 之间            | 修复有效但不超越 product    | 作为论文的诊断章节，不作主方法             |
| A2 全部 ≤ 70.063（GWRP 水平）         | P2P 在此框架内确实无增益      | 保留为完整的负结果 + 机理分析，转向 Gram 方向 |
| **A3 中 GWRP 基线被抬高**             | **原生 CAM 流程本身有此缺陷** | 单独成节，价值高于上面任何一条             |


---



## 4. 顺序与时间


| 序   | 内容                                     | 成本        | 门                             |
| --- | -------------------------------------- | --------- | ----------------------------- |
| 1   | 写 `analysis/artifact_probe.py` + 单测    | 半天        | 单测通过                          |
| 2   | **实验 1 三个 checkpoint 全跑**              | **30 分钟** | **§2.3 三条判定 → register 做/不做** |
| 3   | 重构 `c2p_affinity_weights` + 等价性单测      | 半天        | **旧路径逐值一致**                   |
| 4   | Stage A1 权重统计筛选（72 配置）                 | 2 小时      | 筛出 ≤10 个候选                    |
| 5   | **Stage A2/A3 推理端 mIoU**（候选 + GWRP 基线） | 半天        | **§3.7 判定**                   |
| 6   | Stage B 三组重训                           | 3 组训练     | —                             |


**第 2 步和第 5 步是两个真正的决策点**，前面的都是为它们做准备。整个计划在**两天内**能走到第 5 步，也就是说两天后你就知道 affinity 这条线还要不要继续。

---



## 5. 风险


| 风险                                                    | 概率     | 缓解                                             |
| ----------------------------------------------------- | ------ | ---------------------------------------------- |
| DeiT-S 在 45 epoch 微调下未发育出 artifact，实验 1 直接否定 register | **中高** | 这正是花 30 分钟先做它的原因；否定就是省下两周                      |
| 重构破坏旧路径，历史七组结果失去可比性                                   | 中      | §3.6 的等价性单测是硬门，未通过不得继续                         |
| `floor="min"` 在某些图上把 $r$ 压成全零                         | 中      | 约定回退策略并计数告警，不要静默                               |
| Stage A2 的推理端替换与训练分布不匹配，结论不能外推到 Stage B               | 中      | A2 只用于**排序与筛选**，最终结论以 Stage B 的重训为准；这一点要在报告里写明 |
| 所有结论仍是单 seed                                          | 高      | 沿用现有免责声明；Stage B 的胜出配置必须补到 3 seed 才能进论文        |
| 幂次 $\alpha$ 与 `floor` 功能重叠，消融不可归因                     | 中      | Stage B 的第 1 组刻意只加 `floor`，隔离单因素               |


---



## 6. 与论文的接口

如果 §3.7 走到第一行或第四行，论文骨架是现成的：

> **正结果**：C2P 取代 GWRP，+2.0 pp，四个变体一致。
> **负结果**：朴素 P2P 传播 −2.9 pp，权重熵 0.29→0.97。
> **诊断**：未归一化的跨层求和引入压倒性直流基底，解析式与实测吻合。
> **修复**：去基底 + 阻尼 + 锐化，把 −2.9 变为 +X；同一修复施加到原生 CAM 流程上同样有效。

这个结构不需要 register 和 Gram 也能独立成篇。那两者是加分项，不是必需项。