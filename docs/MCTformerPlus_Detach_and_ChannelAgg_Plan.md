# MCTformer+ 实验计划：C2P 池化权重的 stop-gradient，与通道聚合 A1

> 状态：计划稿，未实施。
> 宿主：`jungletada/TGCA`，MCTformer+ Small（DeiT-S，12 blocks，6 heads，dim 384，20 class tokens，patch 16，448 输入 → 28×28 = 784 patch）。
> 触发依据：`results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2/` 的双 seed 轨迹显示 C2P 的核心主张在 seed 11 上符号翻转。
> 建议落盘：`docs/MCTformerPlus_Detach_and_ChannelAgg_Plan.md`
> **授权边界**：本队列只包含 E1（4 次训练）与 E4（2 次训练）。不授权 register、Gram、COCO、affinity 后续、A2/A3/A4 通道变体，也不授权 5-seed 扩展；扩展由 §2.5 / §3.6 的门决定，需另开队列。

---

## 0. 触发这份计划的事实

| | seed 0 | seed 11 | 极差 |
|---|---:|---:|---:|
| all_product raw CAM mIoU@0.45 | 72.315 | **62.241** | **10.074** |
| gwrp raw CAM mIoU@0.45 | 70.063 | 68.632 | 1.431 |
| **C2P − GWRP** | **+2.252** | **−6.391** | 符号翻转 |

配套读数（epoch 45）：

| run | class mAP | patch mAP | FG precision | FG recall |
|---|---:|---:|---:|---:|
| all_product s0 | 92.442 | 92.864 | 83.318 | 84.654 |
| **all_product s11** | **89.580** | 91.942 | 81.793 | **68.607** |
| gwrp s0 | 92.906 | 93.258 | 80.735 | 85.817 |
| gwrp s11 | 92.849 | 93.135 | 81.171 | 81.990 |

精度基本持平而召回掉 16.0 pp——**严重欠激活**，不是随机噪声。GWRP 两个 seed 只差 1.43，说明方差全部来自 C2P 这一支。

轨迹上 all_product s11 的分类始终追不上（ep20 时 77.41，其余三条 84.13–91.11），CAM 在 ep21 甚至回落到 57.99 再恢复到 ~62 后走平。这是**优化被锁死在欠激活解**的形态。

---

## 1. 共同设置

沿用现有协议，不做任何改动，以保证与已有四条轨迹逐项可比。

| 项目 | 设置 |
|---|---|
| 训练 | 45 epochs，448 crop，有效 batch 32，330 updates/epoch，其余超参与现有 run 完全一致 |
| seeds | **0 与 11**（与既有轨迹对齐，便于直接并排） |
| checkpoint | 每 epoch 保存，复用 `voc_epoch_checkpoints` 那套基础设施 |
| 分类评估 | VOC val 1449，FP32，单尺度 448 |
| CAM 评估 | VOC train 1464，原生 scales {1.0, 0.75, 1.25} + flip，固定阈值 0.45，无 CRF |
| 排序指标 | `fixed_mean_iou_percent`；best-threshold 仅作诊断 |
| 产出 | `results/` 独立目录 + `manifest.json` + `commands.sh` + `checkpoint_sha256.txt` |

**早期告警（节省算力，不作为判定）**：epoch 20 的 class mAP < 80 时标记该 run 为疑似失稳。依据是四条既有轨迹在 ep20 的取值为 90.10 / **77.41** / 84.13 / 91.11。**注意 ep15 不具判别力**——gwrp s0 在 ep15 只有 74.91，最终仍收敛到 92.906，所以不要把检查点提前到 15。

---

## 2. 实验 1：C2P 池化权重的 stop-gradient

### 2.1 原理：GWRP 本来就是 detached 的

这是整个实验的核心论证，也是它不该被视作「工程 trick」的理由。

GWRP 的权重是 $w^{\text{GWRP}}_j = d^{\,\text{rank}(j)} / \sum_k d^{\,\text{rank}(k)}$，其中 rank 来自对 patch logits 的 `argsort`。**`argsort` 不可微**，因此在

$$z[c] = \sum_j w^{\text{GWRP}}[c,j]\; M[c,j]$$

中，梯度只经由 $M$ 回传，**$w$ 这条路径的梯度恒为零**。也就是说，GWRP 在实现上已经等价于一个 stop-gradient 的权重。

而当前的 C2P 实现把注意力直接当权重，$w^{\text{C2P}}$ 是可微的。于是分类损失多出一条 GWRP 从未有过的梯度通路：

$$\frac{\partial \mathcal{L}}{\partial A} \;\longleftarrow\; \frac{\partial \mathcal{L}}{\partial w^{\text{C2P}}}\cdot\frac{\partial w^{\text{C2P}}}{\partial A}$$

**后果有两层。**

**(a) 现有的 C2P vs GWRP 比较是混杂的。** 它同时改变了两件事：用哪种权重（我们想测的），以及权重是否在梯度路径上（我们没打算测的）。`detach` 不是补丁，**它是那个本该做的对照**。

**(b) 这条通路构成自强化环路。** 注意力决定哪些 patch 参与池化 → 分类损失经权重回传 → 塑形注意力 → 注意力更集中到已经高响应处。seed 0 上环路正向自举（+2.25），seed 11 上早期注意力起点差，环路把模型锁在欠激活解里——分类与定位一起卡住，正是 §0 的观测形态。

### 2.2 三个变体

| ID | 权重 | 权重在梯度路径上 | 说明 |
|---|---|---|---|
| **V0** | GWRP | 否（argsort 阻断） | 基线，已有 s0/s11 数据，**不重跑** |
| **V1** | C2P（all-product） | **是** | 当前实现，已有 s0/s11 数据，**不重跑** |
| **V2** | C2P（all-product） | **否（detach）** | **本实验，seed 0 + 11** |

只需 2 次训练。V0/V1 的四条轨迹直接复用。

**推理端完全不受影响**：`detach` 只在训练时改变梯度流，前向数值不变；原生 CAM 生成本就在 `no_grad` 下，因此 V2 的 CAM 流程与 V1 逐位一致。这一点要在报告里写明，与 Stage B 的既有措辞保持一致。

### 2.3 参考代码

在 `models/mctformer_plus.py` 的 C2P 池化路径上加一个开关，**默认 `False` 以保证既有结果可复现**：

```python
# models/mctformer_plus.py
def c2p_pool(patch_logits: torch.Tensor,
             attn_layers,
             n_class: int,
             layers: str = "all",
             mode: str = "product",
             detach_weights: bool = False,      # 新增
             eps: float = 1e-8) -> torch.Tensor:
    """patch_logits: (B, C, N)  —— patch branch 的逐 patch 类别响应
    返回 z: (B, C) 图像级 logit
    """
    w = c2p_spatial_weights(attn_layers, n_class,
                            layers=layers, mode=mode)      # (B, C, N), 和为 1

    if detach_weights:
        w = w.detach()          # 与 GWRP 的 argsort 阻断等价

    return (w * patch_logits).sum(dim=-1)
```

**必须配套的单元测试**（`tests/test_detach_pooling.py`）：

1. **前向等价**：`detach_weights=True` 与 `False` 的前向输出 `torch.allclose`（atol=1e-7）。
2. **梯度差异**：构造 `attn_layers` 为 leaf tensor，反传后 `detach=True` 时其 `.grad` 为 `None` 或全零，`detach=False` 时非零。
3. **GWRP 对照**：对 GWRP 路径做同样的梯度检查，确认其权重路径的梯度本来就是零——**这条是 §2.1 论证的实证，必须进论文附录**。
4. **默认值回归**：不传 `detach_weights` 时行为与重构前逐位一致。

### 2.4 每个 epoch 记录的诊断量

除既有的 class/patch mAP、val loss、CAM mIoU、FG precision/recall 外，复用已有工具追加：

| 量 | 来源 | 期望的区分力 |
|---|---|---|
| $\kappa$：C2P 权重的归一化熵 | `analysis/weight_stats.py` | V1-s11 应显著低于其它（权重过度集中 → 欠激活） |
| top-1 mass | 同上 | 同上 |
| $\rho_{cc}$：共现类对的 class token 余弦 | `analysis/diagnostics/metrics.py` | 检查是否伴随类别塌缩 |
| M2 / M4：artifact 占比与吸引子集中度 | `analysis/artifact_probe.py` | 只在 ep 15/30/45 三点测，不必每轮 |

**这组量的作用是把「失稳」从一个结果变成一个可描述的过程。** 如果 V1-s11 的 $\kappa$ 在 ep10–20 急剧下降而 V2-s11 不降，自强化环路的假设就得到了直接证据，而不只是推断。

### 2.5 预先注册的判定门

记 $m(\cdot)$ 为 epoch 45 的 `fixed_mean_iou_percent`。已知 V0: s0 = 70.063, s11 = 68.632；V1: s0 = 72.315, s11 = 62.241。

| 条件 | 结论 | 动作 |
|---|---|---|
| $m(\text{V2-s11}) \ge 68.1$ **且** $m(\text{V2-s0}) \ge 71.1$ | **detach 是修复**：失稳源于额外梯度通路，且收益不依赖它 | 开 5-seed 队列（V0 与 V2 各 5 seed），进主表 |
| $m(\text{V2-s11}) \ge 68.1$ **但** $m(\text{V2-s0}) < 71.1$ | 稳定了，但**收益本身来自那条环路** | C2P 是彩票不是方法；转为分析章节，见 §6 |
| $m(\text{V2-s11}) < 68.1$ | 失稳与梯度通路无关 | 走 §2.6 的回退变体 |
| 两个 seed 都低于 V0 | detach 破坏了 C2P 的价值 | 记录并停止本方向 |

门槛的取法：`68.1 = V0-s11 − 0.5`（不再显著低于基线即可），`71.1 = V0-s0 + 1.0`（保留有意义的增益）。两者都在开跑前固定，**结果出来后不得调整**。

### 2.6 回退变体（仅在第三种分支触发）

按成本排序，每个先只跑 seed 11（失败的那个），过了再补 seed 0：

1. **课程**：前 $E_0$ epoch 用 GWRP，之后切 C2P。$E_0 = 10$，依据是四条轨迹的 class mAP 在 ep10 前后才起来。
2. **混合**：$w = (1-\lambda_t)\,w^{\text{GWRP}} + \lambda_t\, w^{\text{C2P}}$，$\lambda_t$ 在 ep 5→20 线性由 0 升到 1。
3. **熵正则**：对 $\kappa$ 加下界惩罚，阻止权重过早塌缩。你已经有这个量，只需加一项损失。

三者都不改推理端。

---

## 3. 实验 4：通道聚合 A1

### 3.1 前置核对（10 分钟，先做）

打开 `models/mctformer_plus.py`，确认类别 logit 的实际算法。CTI 描述的是 MCTformer（CVPR'22）的做法——把 $\mathbf{T}^L_{cls}\in\mathbb{R}^{C\times D}$ 在**嵌入维度**上池化得到 $y\in\mathbb{R}^C$。**MCTformer+（TPAMI'24）可能已经改成线性头。**

- 若确实是通道均值 → 继续；
- 若已是线性头 → **本实验前提不成立，整节作废**，把资源全部给 E1。

同时确认是否存在 BG token（CTI 引入了，MCTformer+ Small 未必有），因为它会影响 §3.3 的归一化范围。

### 3.2 原理

若当前实现为

$$y[c]=\frac{1}{D}\sum_{j=1}^{D}\mathbf{T}^L_{cls}[c,j]$$

这等价于一个权重全部固定为 $1/D$ 的线性头，有三个后果：

1. **零通道判别能力**——无法区分通道重要性；
2. **梯度在通道维完全均匀**——$\partial y[c]/\partial \mathbf{T}[c,j]=1/D$ 对所有 $j$ 相同，class token 自身不受任何通道选择性压力；
3. **目标方向被钉死**——class token 必须朝 $\mathbf{1}/\sqrt{D}$ 对齐才能得到高 logit，而注意力用的 query 是 $W_q\mathbf{T}_{cls}$；两者是同一向量的不同线性泛函，**没有任何理由要求它们的方向一致**。

与已有工作的对称性（这是它在论文里的位置）：

| 聚合 | 维度 | 当前 | 状态 |
|---|---|---|---|
| 空间 | patch 维 $N$ | GWRP（rank 衰减） | 已替换为 C2P（稳定性待 E1 确认） |
| **通道** | 嵌入维 $D$ | **均匀平均** | **从未被质疑** |

同一原理作用在两个正交维度上。CTI 唯一相关的数据点是用 FC 层拆分**输入端**的 class token，**输出端的通道聚合在文献中没有任何消融**。

### 3.3 参数化：在初始化处精确等价于基线

$$y[c]=\sum_{j=1}^{D} a_j\, \mathbf{T}^L_{cls}[c,j],\qquad \mathbf{a}=\mathrm{softmax}(\boldsymbol\theta),\quad \boldsymbol\theta\in\mathbb{R}^{D}$$

四个性质，每一个都是刻意选的：

- $\boldsymbol\theta$ 初始化为 **0** → $a_j = 1/D$ → **前向输出与基线逐位相同**，run 从基线出发，任何偏离都是学出来的；
- $\sum_j a_j = 1$ → **logit 尺度不变**，不会与 multi-label soft margin loss 的尺度敏感性纠缠；
- $a_j > 0$ → 不引入符号翻转这个额外自由度；
- $\mathbf{a}$ **跨类别共享** → 类别对称性完全保持，不存在 A2（每类线性头）那种「把类别 $c$ 的信息路由到 token $c'$」的退化风险。

参数量：**384**（约为 DeiT-S 的 0.0017%）。

### 3.4 参考代码

```python
# models/mctformer_plus.py
class ChannelAggregator(nn.Module):
    """class token -> class logit 的通道聚合。
    theta=0 时精确等价于通道均值基线。"""
    def __init__(self, dim: int, enabled: bool = False, tau: float = 1.0):
        super().__init__()
        self.enabled, self.tau = enabled, tau
        self.theta = nn.Parameter(torch.zeros(dim)) if enabled else None

    def weights(self) -> torch.Tensor | None:
        if not self.enabled:
            return None
        return torch.softmax(self.theta / self.tau, dim=0)      # (D,), 和为 1

    def forward(self, t_cls: torch.Tensor) -> torch.Tensor:
        # t_cls: (B, C, D)
        if not self.enabled:
            return t_cls.mean(dim=-1)                            # 基线
        return (t_cls * self.weights()).sum(dim=-1)              # (B, C)
```

优化器分组：$\boldsymbol\theta$ **不加 weight decay**（它是 softmax 的 logit，衰减会把它拉回均匀，等于抵消实验本身），并设一个独立的 lr 倍率作为次要变体。

- **A1-a**：lr 与 backbone 相同；
- **A1-b**：lr × 10。

先跑 A1-a；只有当 §3.5 显示 $\mathbf{a}$ 几乎没动时才跑 A1-b，用来区分「模型不想要非均匀权重」与「学习率不够它动」。

单元测试（`tests/test_channel_agg.py`）：`enabled=False` 与 `enabled=True, theta=0` 的输出 `torch.allclose`（atol=1e-7）；`weights()` 和为 1 且全正；`theta` 不在 weight-decay 参数组内。

### 3.5 学到的 $\mathbf{a}$ 本身就是诊断

这是这个实验设计得好的地方：**参数本身回答问题**。每个 epoch 记录

- 归一化熵 $H(\mathbf{a})/\log D$
- $\max_j a_j \,/\, \min_j a_j$
- $\lVert \mathbf{a}-\tfrac{1}{D}\mathbf{1}\rVert_1$（与均匀的总变差距离）

三种结局都有明确读法：

| $\mathbf{a}$ | mIoU | 结论 |
|---|---|---|
| 仍接近均匀（$H/\log D > 0.98$） | 无变化 | **通道均值本来就接近最优**，干净的负结果，一段话写完 |
| 明显非均匀 | 提升 | **主结果**：均匀通道加权确实有害 |
| 明显非均匀 | 无提升或下降 | 容量存在但无用——写进分析，指向「约束反而起了正则作用」 |

### 3.6 判定门与宿主选择

**宿主必须是 GWRP，不是 all_product。** 理由：all_product 目前处于失稳状态（§0），在它之上加 A1 会把两个效应混在一起。**等 E1 确认 C2P 稳定之后，再考虑 A1 + C2P 的组合。**

对照：`gwrp` s0 = 70.063，s11 = 68.632。

| 条件 | 结论 |
|---|---|
| 两个 seed 的 $\Delta$ 均 > 0 且 $\ge +0.5$ | 进入主表，开 5-seed 队列 |
| 两个 seed 符号一致但 $\Delta < 0.5$ | 记为无效改动，按 §3.5 第一行处理 |
| 两个 seed 符号不一致 | 方差过大，需更多 seed 才能判断；本队列不授权，记录后暂停 |
| 两个 seed 的 $\Delta$ 均 < 0 | 通道均值优于学习权重，负结果 |

---

## 4. 排期与并行

E1 与 E4 **完全独立**（不同分支、不同宿主、不同代码路径），可以同时挂上去。

| 序 | 动作 | 训练次数 | 门 |
|---|---|---|---|
| 0 | §3.1 代码核对；两个模块的单元测试 | 0 | **单测全绿，尤其是 GWRP 零梯度那条** |
| 1 | **E1**：V2 detach，seed 0 + 11 | 2 | §2.5 |
| 2 | **E4**：A1-a on GWRP，seed 0 + 11 | 2 | §3.6 |
| 3 | 条件触发：E1 回退变体 或 A1-b | ≤2 | — |

合计 4–6 次训练。按现有 45-epoch 单 run 的耗时估算即可。

**epoch 级 checkpoint 全部保留**，因为 §2.4 的诊断需要逐轮读取；这也让第二条轨迹图可以直接与既有的四条并排。

---

## 5. 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| **detach 稳定了但增益也没了**（V1 的 +2.25 本来就来自那条环路） | **中高** | 这是 §2.5 第二行，已预先写好读法；即使如此，「+2.25 来自一个不稳定的梯度环路」本身是有价值的发现 |
| seed 11 只是一次偶发失败，detach 的「修复」其实是运气 | 中 | 门通过后必须扩到 5 seed 才写进主表；本队列不授权扩展正是为此 |
| §3.1 核对发现 MCTformer+ 已用线性头 | 中 | 10 分钟即可确认，作废成本极低 |
| A1 的 $\boldsymbol\theta$ 被 weight decay 拉回均匀，实验变成空跑 | 中 | 优化器分组里显式排除；单测检查参数组 |
| 改动破坏既有结果可复现性 | 低但致命 | 两个开关默认 `False`；前向等价性单测是硬门 |
| epoch-checkpoint 占盘 | 低 | 沿用既有清理策略，只保留 metrics JSON 与 checkpoint |

---

## 6. 与论文的接口

E1 的结果决定论文的形态，两种都能写：

**若 detach 是修复**（§2.5 第一行）：

> 空间池化用 C2P 注意力替代 GWRP 可提升 seed 质量，**但必须切断权重的梯度通路**；否则注意力与池化权重之间形成自强化环路，导致训练在部分随机种子上锁死于欠激活解（seed 11：召回 68.6 vs 84.7，mIoU 62.2 vs 72.3）。GWRP 因 `argsort` 不可微而天然免疫，这也说明此前的 C2P/GWRP 比较未控制该因素。

**若增益随 detach 一起消失**（第二行）：

> C2P 的增益来自池化权重的梯度通路，而该通路同时是训练失稳的来源；在两个种子上它一次带来 +2.25、一次带来 −6.39。我们据此认为把注意力直接用作可微池化权重在多类别 token 架构中不是一个可靠的设计。

第二种是负结果，但它有机制、有对照、有可复现的失败案例，**配合直流基底那节分析，仍然构成一篇完整的分析型论文**。

E4 无论结果如何都是独立的一节：通道聚合在 MCTformer 系列中从未被消融，384 个参数的对照给出了这个空白的答案。

---

## 7. 一句话总结

先花 10 分钟核对通道聚合的实现，再把两个默认关闭的开关和它们的等价性单测写好，然后并行跑四次训练。**E1 决定 C2P 这条线是方法还是彩票，E4 填一个文献里没人碰过的空白**；两者都在 §2.5 / §3.6 预先写死了读法，跑完不需要再讨论怎么解释。
