# MCTformer+ 训练动力学诊断探针：设计与实施计划

> 状态：计划稿，未实施。本文档遵循仓库约定——代码存在不代表有效性已验证，结论以对应实验报告为准。
> 建议落盘路径：`docs/MCTformerPlus_Diagnostic_Probe_Plan.md`

---

## 0. 定位与动机

### 0.1 本计划不做什么

- **不修改训练目标**，不增加任何损失项，不改变优化器与调度。
- **不影响任何已有结果**。探针在 `torch.no_grad()` + `model.eval()` 下于独立 probe 集上前向，与训练 batch 完全解耦。
- **不替代** `make_cam.py` / `eval_cam_crf.py` 的正式评估流程。

### 0.2 本计划要回答什么

当前主线是用 class-to-patch attention（C2P）替换 MCTformer+ patch branch 的 GWRP 权重，已观测到提点。但目前缺两样东西：

1. **机制解释**。C2P 为什么比 GWRP 好？现有说法只能是「用了真实的 attention 权重」，这在审稿时是描述而非解释。
2. **无标注的 checkpoint 选择依据**。当前（和 WSSS 领域普遍做法一样）依赖带 GT mask 的评估来选 epoch 和阈值，这是一种被默认容忍的监督泄漏。

本探针提供一组**只依赖模型自身输出、不依赖任何像素标注**的诊断量，用来：

- (a) 检验「密集特征质量在分类指标仍在上升时已经见顶回落」这一现象在 MCTformer+ 上是否成立；
- (b) 检验这些诊断量能否**预测** raw CAM mIoU 的峰值 epoch；
- (c) 给 C2P 相对 GWRP 的增益提供可量化的机制描述；
- (d) 作为 C2P 层选择（last-3 / all-layers）的设计依据，而非事后消融。

### 0.3 理论来源

DINOv3（arXiv:2508.10104）§4.1 的诊断：ViT 长训练中全局指标单调上升而密集指标在约 200k 迭代后下降，机制是 `[CLS]` 与 patch 输出的余弦相似度持续上升 → patch 局部性流失。其修复手段 Gram anchoring 约束的是 patch 特征的二阶结构（Gram 矩阵）而非特征本身。

MCTformer 把单个 `[CLS]` 换成 C 个类别 token，因此**同一对矛盾变成三方竞争**：类别 token 之间、类别 token 与 patch、patch 与 patch。这既给出更多可测自由度，也引入新的失效模式。

---

## 1. 假设与判定标准

| 编号 | 假设 | 证据形态 | 若成立的含义 |
|---|---|---|---|
| **H0** | 训练窗口内不存在密集质量早峰，mIoU 与 mAP 同步单调 | mini-CAM-mIoU 在最后 epoch 取最大 | 现象不存在于当前训练长度；转入 §9.1 回退方案（延长训练诱发） |
| **H1** | multi-class token 结构本身抑制了「全局吞噬局部」 | MCTformer+ 的 κ 衰减显著慢于单 CLS baseline | 为 MCT 架构找到一条此前未被论证的理论优势 |
| **H2** | 全局压力未消失，只是转移到类间同化 | ρ_cc 快速上升，且与 mIoU 下降同步 | 定位了一个新的、可被针对性修复的失效模式 |
| **H3** | C2P 优于 GWRP 的机制是延缓了上述退化 | C2P 组的 κ 衰减 / ρ_cc 上升慢于 GWRP 组 | 给当前主线结果提供机制解释 |

**H1 与 H2 不互斥**，可以同时部分成立（不同层表现不同）。

**核心判定指标**：诊断量与 mini-CAM-mIoU 在 epoch 维度上的 Spearman 相关系数 $\rho_s$，以及**峰值 epoch 一致性** $|\arg\max_e \text{mIoU} - \arg\text{opt}_e \text{diag}|$。

判定门槛（预注册，避免事后挑选）：

- 至少一个诊断量满足 $|\rho_s| \ge 0.7$，且在 **3 个 seed 上符号一致**；
- 峰值 epoch 偏差 $\le 2$ epoch；
- 该诊断量在 GWRP 与 C2P 两组上都成立（否则只是 C2P 的伴随现象，不是通用诊断）。

---

## 2. 诊断量定义

记：

- 类别 token 输出 $\mathbf{T}\in\mathbb{R}^{C\times d}$，$\ell_2$ 归一化后为 $\hat{\mathbf{T}}$；
- patch token 输出 $\mathbf{X}\in\mathbb{R}^{N\times d}$，归一化后 $\hat{\mathbf{X}}$，空间网格 $h\times w$，$N=hw$；
- 第 $l$ 层、跨 head 平均的注意力块 $A^{(l)}\in\mathbb{R}^{(C+N)\times(C+N)}$，切出 $A^{(l)}_{c\to p}\in\mathbb{R}^{C\times N}$ 与 $A^{(l)}_{p\to p}\in\mathbb{R}^{N\times N}$；
- 图像级标签 $y\in\{0,1\}^C$，present 类集合 $\mathcal{C}^+=\{c: y_c=1\}$。

### D1. 类间同化 ρ_cc

$$\rho_{\text{cc}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\hat{\mathbf{t}}_i^\top\hat{\mathbf{t}}_j,\qquad \mathcal{P}=\{(i,j): i\neq j,\ y_i=y_j=1\}$$

**只在同图共现的类别对上统计**。理由：跨图不共现的类别对（如 aeroplane–sofa）相似度低是平凡的；共现对（boat–water 语境、person–horse）才是共现误激活的来源。

同时记录**全局版本** $\rho_{\text{cc}}^{\text{all}}$（所有 $i\neq j$，跨整个 probe 集聚合 token 后计算）作为对照——MCTformer 的类别 token 是全局参数，其两两相似度本身就是一个不依赖输入的量，值得单独追踪。

- 失效方向：↑
- 对应病症：共现误激活

### D2. 类内完整性 κ（有效支撑比）

对 $c\in\mathcal{C}^+$，把 $A^{(l)}_{c\to p}[c,:]$ 在 patch 维归一化为分布 $a_c$：

$$\kappa_c^{(l)}=\frac{\exp\big(H(a_c)\big)}{N},\qquad H(a_c)=-\sum_{p}a_{cp}\log a_{cp}$$

即困惑度归一化到 $[1/N, 1]$。$\kappa=1$ 表示注意力均匀铺满（无定位能力），$\kappa\to 1/N$ 表示塌缩到单个 patch。

**这个量不是越大越好**，存在最优区间——这与 D3 配合读取。经验上物体占图像面积的 10–40%，因此 $\kappa\approx0.1\text{–}0.4$ 是合理带。

- 失效方向：↓（过度聚焦）或 ↑（失去类特异性）
- 对应病症：只激活判别性区域 / 过度平滑

### D3. 注意力集中度 σ（Gini）

$$\sigma_c^{(l)}=\text{Gini}\big(a_c\big)=\frac{\sum_{p}\sum_{q}|a_{cp}-a_{cq}|}{2N\sum_p a_{cp}}$$

与 D2 的区别：κ 对分布的「有效宽度」敏感，σ 对「头部集中程度」敏感。**两者分离能区分两种不同的退化**：

| κ | σ | 诊断 |
|---|---|---|
| ↓ | ↑ | 塌缩到判别性区域（典型 CAM 病症） |
| ↑ | ↑ | 长尾弥散 + 少数尖峰（噪声化） |
| ↑ | ↓ | 过度平滑，类特异性丧失 |

### D4. patch 空间相关长度 ξ（谱诊断）

把 $\hat{\mathbf{X}}$ 重排成网格 $F\in\mathbb{R}^{h\times w\times d}$，逐通道去均值、加 Hann 窗，计算径向平均功率谱：

$$P(k)=\big\langle \textstyle\sum_d |\hat F_d(\mathbf{k})|^2 \big\rangle_{\|\mathbf{k}\|\in[k,k+\Delta k)}$$

由 Wiener–Khinchin，$P(k)$ 是 Gram 矩阵在空间平稳假设下的一维约化：

$$G(\Delta)=\sum_d\text{Autocorr}_d(\Delta)\ \xrightarrow{\ \mathcal{F}\ }\ P(\mathbf{k})$$

相关长度由自相关函数的指数拟合得到：

$$\hat G(r)=\hat G(0)\exp(-r/\xi)\ \Rightarrow\ \xi = -\big[\text{slope of }\log \hat G(r)\text{ vs }r\big]^{-1}$$

同时记录**高频能量占比** $E_{\text{hi}}=\sum_{k>k_{1/2}}P(k)\big/\sum_k P(k)$ 作为更稳健的代理量（不需要拟合，对噪声不敏感）。

**重要限制**：patch 特征场并不空间平稳（前景/背景统计不同），因此 ξ **只适合作为「每个 checkpoint 一个标量」的训练动力学监控，不适合做单图质量评估**。这一点必须写进论文的方法局限。

- 失效方向：ξ ↓ / $E_{\text{hi}}$ ↑
- 对应病症：相似度图变噪、边界模糊

### D5. GWRP–C2P 权重背离度 δ（本仓库专属）

这是与当前主线直接耦合的诊断，其他工作没有。

对每个 $c\in\mathcal{C}^+$：

- $w^{\text{GWRP}}_c$：按 patch branch 的类别 $c$ logit 排序后的几何衰减权重（$d^{\text{rank}}$，$d$ 取实际配置值）；
- $w^{\text{C2P}}_c$：归一化的 $A_{c\to p}[c,:]$（按当前实验配置的层聚合方式：last-3 / all-layers，mean / product）。

两个量：

$$\delta^{\text{rank}}_c = 1-\text{Spearman}\big(w^{\text{GWRP}}_c,\ w^{\text{C2P}}_c\big),\qquad
\delta^{\text{JS}}_c = \text{JS}\big(w^{\text{GWRP}}_c\,\|\,w^{\text{C2P}}_c\big)$$

**无论结果朝哪个方向都是有内容的发现**：

- δ 随训练**下降** → 两种权重后期趋同，C2P 的收益集中在训练早期 → 提示可以做「早期 C2P、后期 GWRP」的调度，或直接缩短训练；
- δ 随训练**上升** → C2P 在做结构上不同的事，且差异随训练放大 → 支持 C2P 是独立机制而非 GWRP 的平滑近似；
- δ 与 mIoU 增量正相关 → 直接把「背离程度」变成增益的解释变量。

### D6. 逐层剖面（不是新指标，是新的读取维度）

D2 / D3 / D5 都**逐层计算并保存**，得到 `layer × epoch` 的矩阵。

这是把诊断从「分析工具」升级为「设计工具」的关键：你正在对比 last-3 layers 与 all layers 的 C2P 聚合。逐层 κ / σ 剖面可以**先验地**告诉你哪些层的 attention 定位质量最好，从而把层选择从盲目消融变成有依据的设计。如果剖面显示 layer 8–10 最优而 11–12 已经退化，那 last-3 的选择本身就需要修正。

---

## 3. 模块设计

### 3.1 目录结构

```
analysis/
  diagnostics/
    __init__.py
    probe.py          # TokenProbe：主类，负责挂载、前向、指标计算
    metrics.py        # D1–D3, D5 的纯函数实现（无状态、可单测）
    spectral.py       # D4：DFT / 径向谱 / 相关长度
    probe_set.py      # 固定 probe 集的构建与缓存
    writer.py         # CSV / JSON 落盘，manifest 兼容
    plot_trajectory.py# 轨迹图与相关性分析
experiments/
  diag_trajectory_voc.sh
  diag_trajectory_coco.sh
tests/
  test_diagnostics.py
docs/
  MCTformerPlus_Diagnostic_Probe_Plan.md   # 本文件
```

### 3.2 核心接口

```python
# analysis/diagnostics/probe.py

@dataclass
class ProbeConfig:
    enabled: bool = False
    every_n_epochs: int = 1
    probe_set_path: str = "data/probe/voc_probe_256.txt"
    probe_batch_size: int = 32
    # 双分辨率：语义类指标用训练分辨率，谱指标用高分辨率
    res_semantic: int = 224          # 与训练一致
    res_spectral: int = 448          # 必须使网格边长 >= 28，见 §3.4
    layers: tuple = (6, 7, 8, 9, 10, 11)   # 0-indexed；全层可设 tuple(range(12))
    gwrp_decay: float = 0.996        # 必须与 patch branch 实际配置一致
    c2p_agg: str = "mean"            # "mean" | "product"，与实验配置对齐
    n_radial_bins: int = 16
    dump_raw: bool = False           # 是否保存原始 attention（调试用，体积大）
    out_dir: str = "results/{run_id}/diagnostics"


class TokenProbe:
    def __init__(self, model, cfg: ProbeConfig, device): ...

    @torch.no_grad()
    def run(self, epoch: int, global_step: int) -> dict:
        """在 probe 集上跑一遍，返回扁平化的 metric dict，并追加写 CSV。"""

    def close(self): ...
```

### 3.3 挂载点

两处改动，都在 if 分支内，默认关闭：

**`train_model_v2.py`** —— 构造与调用：

```python
from analysis.diagnostics import TokenProbe, ProbeConfig

probe = TokenProbe(model, ProbeConfig(**args.probe), device) if args.probe.enabled else None

for epoch in range(args.start_epoch, args.epochs):
    train_one_epoch(...)                      # engine.py，不动
    if probe is not None and epoch % probe.cfg.every_n_epochs == 0:
        probe.run(epoch=epoch, global_step=global_step)
    # 已有的 checkpoint 保存逻辑不动
```

**不要改 `engine.py`**。诊断需要的一切都能从独立前向拿到，把探针塞进 `train_one_epoch` 只会污染训练循环并引入 AMP/DDP 的耦合问题。

### 3.4 关键实现约束（这一节的每一条都是踩过的坑）

| # | 约束 | 原因 |
|---|---|---|
| 1 | **固定 probe 集**，不用训练 batch | 随机 batch 的构成方差会淹没 epoch 间的趋势。取 VOC `train_aug` 中固定 256 张（分层采样保证每类至少 8 张），顺序固定，`seed=0` 生成后写入 txt 并纳入 manifest |
| 2 | **无数据增强**，只 resize + center crop + normalize | 增强的随机性会进入所有二阶统计量 |
| 3 | `model.eval()` + `torch.no_grad()`，**探针结束后恢复 `model.train()`** | drop_path（MCTformer 默认 0.1）会给 attention 加噪；忘记恢复会静默毁掉后续训练 |
| 4 | **全程 fp32**，探针内禁用 autocast | cosine、FFT、Gini 在 fp16 下数值不可靠，尤其 $\hat{\mathbf{X}}\hat{\mathbf{X}}^\top$ 的小值区 |
| 5 | **DDP 下只在 rank 0 执行**，或 all-gather 后由 rank 0 写盘 | 避免重复计算与 CSV 竞写 |
| 6 | 谱诊断的网格边长 $h \ge 28$ | 14×14 网格做径向平均只有约 7 个频段，拟合 ξ 无意义。patch=16 时需输入 ≥ 448 |
| 7 | 谱前向用**独立分辨率**，依赖位置编码插值 | MCTformer 的 pos-embed 支持插值（CAM 推理本来就多尺度）。但插值本身有伪影，所以 D1–D3 用训练分辨率、D4 用 448，**两者不混用** |
| 8 | FFT 前逐通道去均值 + Hann 窗 | 否则 DC 分量主导 $P(0)$；小网格上周期延拓的边界伪影不可忽略 |
| 9 | attention 取**跨 head 平均**，同时另存**逐 head 的 κ 方差** | head 之间的分工是 MCTformer 的已知性质，平均会掩盖它；head 间方差本身可能是一个有用的第六指标 |
| 10 | 每个指标同时保存 **mean / std / per-class**，不要只存均值 | 类别间差异（大物体 vs 小物体类）极可能是主要信号来源 |

### 3.5 开销估算

probe 集 256 张、batch 32、双分辨率各一次前向：

- ViT-S/16 @224（14×14）：8 个 batch，约 1–2 秒
- ViT-S/16 @448（28×28，序列 784+C）：8 个 batch，约 4–6 秒
- Gram/affinity $N^2$：$784^2\times4\text{B}\approx2.5$ MB/图，batch 内逐图计算后立即归约，峰值显存 < 1 GB
- 指标计算（CPU 侧 Gini/Spearman 可用 GPU 实现）：< 2 秒

**每 epoch 总计 < 15 秒**，相对 VOC 一个 epoch 的训练时间可忽略。COCO 同理（probe 集大小不变）。

---

## 4. 参考实现（关键函数）

以下是 D2 / D3 / D4 / D5 的核心逻辑，不含工程外壳。

```python
# analysis/diagnostics/metrics.py
import torch

def effective_support(attn_cp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D2. attn_cp: (B, C, N) 已按 N 维归一化. 返回 (B, C), 值域 [1/N, 1]."""
    N = attn_cp.shape[-1]
    p = attn_cp.clamp_min(eps)
    H = -(p * p.log()).sum(-1)
    return H.exp() / N


def gini(attn_cp: torch.Tensor) -> torch.Tensor:
    """D3. 排序法 O(N log N) 实现, 避免 O(N^2) 的两两绝对差."""
    B, C, N = attn_cp.shape
    x, _ = attn_cp.sort(dim=-1)                      # 升序
    idx = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    num = (2 * idx - N - 1) * x
    return num.sum(-1) / (N * x.sum(-1).clamp_min(1e-12))


def class_token_affinity(T: torch.Tensor, y: torch.Tensor) -> dict:
    """D1. T: (B, C, d) 未归一化; y: (B, C) 0/1 图像级标签."""
    Th = torch.nn.functional.normalize(T, dim=-1)
    S = Th @ Th.transpose(-1, -2)                    # (B, C, C)
    C = S.shape[-1]
    eye = torch.eye(C, device=S.device, dtype=torch.bool)
    cooc = (y.unsqueeze(2) * y.unsqueeze(1)).bool() & ~eye   # (B, C, C)
    denom = cooc.sum(dim=(1, 2)).clamp_min(1)
    rho_cooc = (S * cooc).sum(dim=(1, 2)) / denom     # (B,)
    rho_all = S.masked_fill(eye, 0).sum(dim=(1, 2)) / (C * (C - 1))
    return {"rho_cc": rho_cooc, "rho_cc_all": rho_all}


def gwrp_weights(logits_map: torch.Tensor, decay: float) -> torch.Tensor:
    """D5 左半. logits_map: (B, C, N) patch branch 的逐 patch 类别响应."""
    order = logits_map.argsort(dim=-1, descending=True)
    N = logits_map.shape[-1]
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(N, device=order.device)
                   .expand_as(order))
    w = decay ** ranks.to(logits_map.dtype)
    return w / w.sum(-1, keepdim=True)


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """按最后一维计算 Spearman. 无 tie 处理(attention 值几乎不重复)."""
    def rank(x):
        idx = x.argsort(-1)
        r = torch.empty_like(idx)
        r.scatter_(-1, idx, torch.arange(x.shape[-1], device=x.device)
                   .expand_as(idx))
        return r.to(x.dtype)
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)
```

```python
# analysis/diagnostics/spectral.py
import torch

def radial_power_spectrum(F: torch.Tensor, n_bins: int = 16):
    """D4. F: (B, h, w, d) 未归一化的 patch 特征网格.
    返回 P: (B, n_bins), 频率中心 kcent: (n_bins,)"""
    B, h, w, d = F.shape
    F = F - F.mean(dim=(1, 2), keepdim=True)          # 逐通道去均值 (约束 8)

    # Hann 窗, 抑制小网格的周期延拓伪影
    wy = torch.hann_window(h, periodic=False, device=F.device)
    wx = torch.hann_window(w, periodic=False, device=F.device)
    F = F * (wy[None, :, None, None] * wx[None, None, :, None])

    spec = torch.fft.fftshift(torch.fft.fft2(F, dim=(1, 2)), dim=(1, 2))
    power = (spec.real ** 2 + spec.imag ** 2).sum(-1)  # (B, h, w), 通道求和

    cy, cx = h // 2, w // 2
    yy = torch.arange(h, device=F.device) - cy
    xx = torch.arange(w, device=F.device) - cx
    r = (yy[:, None] ** 2 + xx[None, :] ** 2).sqrt()
    rmax = r.max()
    bins = (r / rmax * (n_bins - 1)).round().long().flatten()

    flat = power.reshape(B, -1)
    P = torch.zeros(B, n_bins, device=F.device, dtype=flat.dtype)
    P.scatter_add_(1, bins.expand(B, -1), flat)
    cnt = torch.zeros(n_bins, device=F.device).scatter_add_(
        0, bins, torch.ones_like(bins, dtype=P.dtype))
    P = P / cnt.clamp_min(1)
    kcent = torch.linspace(0, 1, n_bins, device=F.device) * rmax
    return P, kcent


def high_freq_ratio(P: torch.Tensor) -> torch.Tensor:
    """E_hi: 高于中位频率的能量占比. 比 xi 稳健, 优先作为主指标."""
    half = P.shape[-1] // 2
    return P[..., half:].sum(-1) / P.sum(-1).clamp_min(1e-12)


def correlation_length(P: torch.Tensor, kcent: torch.Tensor,
                       fit_range=(1, 6)) -> torch.Tensor:
    """xi: 由径向谱逆变换得到的自相关, 拟合 log G(r) ~ -r/xi.
    注意: 依赖空间平稳假设, 只用于 checkpoint 级趋势监控 (见 2.D4 限制)."""
    G = torch.fft.irfft(P, dim=-1)                    # 近似自相关
    G = G / G[..., :1].clamp_min(1e-12)
    lo, hi = fit_range
    r = torch.arange(lo, hi, device=P.device, dtype=P.dtype)
    logG = G[..., lo:hi].clamp_min(1e-6).log()
    rc = r - r.mean()
    slope = (rc * (logG - logG.mean(-1, keepdim=True))).sum(-1) / \
            (rc ** 2).sum().clamp_min(1e-12)
    return (-1.0 / slope.clamp_max(-1e-6))
```

**关于 `correlation_length` 的诚实说明**：这个实现是一维径向谱的近似逆变换，不是严格的二维自相关。如果拟合质量差（$R^2 < 0.8$），直接退回 `high_freq_ratio`——后者不需要任何拟合，对本计划的判定目的完全够用。**不要为了让 ξ 好看而调 `fit_range`**，那是事后挑选。

---

## 5. 因变量：mini-CAM-mIoU

诊断量是自变量，需要一个能**逐 epoch 廉价计算**的密集质量因变量。

正式的 `make_cam.py` + `eval_cam_crf.py` 全流程太慢，不适合每 epoch 跑。方案：

- **固定子集**：VOC `train_aug` 中有 GT mask 的 500 张（与 probe 集**不重叠**，避免自证）。
- **单尺度 raw CAM**，不做 CRF、不做多尺度 TTA、不做 IRN/PSA。
- **固定阈值**用当前报告里已确定的那个值，同时另存 **common-threshold 曲线的最优点**（你们已有这个评估口径）。
- 输出：`mIoU@fixed`、`mIoU@best`、前景 precision/recall——与现有报告口径一致，便于交叉验证。

估算：500 张单尺度前向 + 评估约 30–60 秒/epoch，可接受。

**这一步必须复用现有评估代码路径**，不要另写一份 mIoU，否则和正式报告对不上就失去意义。建议在 `tools/` 下加一个瘦封装，内部调用现有函数。

---

## 6. 实验矩阵

| ID | 模型 | pooling | 数据集 | seeds | 目的 |
|---|---|---|---|---|---|
| `DIAG-A` | MCTformer+ | GWRP（基线） | VOC | 3 | 建立退化曲线基准，判定 H0 |
| `DIAG-B` | MCTformer+ | C2P（当前最佳配置） | VOC | 3 | 判定 H3，对比 A |
| `DIAG-C` | DeiT-S + 单 CLS CAM | — | VOC | 1 | 判定 H1（multi-token 是否天然抑制退化） |
| `DIAG-D` | MCTformer+ | C2P | COCO | 1 | 检验现象是否跨数据集（COCO 共现更密集，H2 应更明显） |
| `DIAG-E` | `DIAG-B` 延长至 3× epoch | C2P | VOC | 1 | 若 H0 成立的回退实验（§9.1） |

**3 个 seed 是判定门槛的硬要求**——你们 README 自己写了「单 seed 结果不代表稳定收益」，这条在诊断实验上同样适用，而且诊断量的方差可能比 mIoU 更大。

**DIAG-C 的实现注意**：单 CLS baseline 需要与 MCTformer+ 共享 backbone、初始化、augmentation、优化配置，只改 token 结构与 CAM 生成方式。否则 H1/H2 的判定会被混淆因素污染。这是整个矩阵里最容易做错的一格。

---

## 7. 产出与分析

### 7.1 落盘格式

```
results/{run_id}/diagnostics/
  metrics.csv          # 长表: epoch, metric, layer, class_id, mean, std
  metrics_wide.csv     # 宽表, 供绘图直接读取
  spectrum_{epoch}.npy # 径向谱, 供后续重新拟合而不必重跑
  manifest.json        # 与仓库现有规范一致: 代码 SHA, 配置, probe 集 hash, ckpt SHA256
  commands.sh
```

**长表优先**，因为逐层 × 逐类的组合会让宽表爆炸。`probe 集 hash` 必须入 manifest——换了 probe 集的两次运行不可比。

### 7.2 三张核心图

1. **轨迹图**：x = epoch，双 y 轴，左轴 mAP（class-token / patch-head）与 mini-CAM-mIoU，右轴各诊断量（标准化到 z-score）。GWRP 与 C2P 两组同图，用线型区分。
2. **逐层热图**：`layer × epoch` 的 κ 与 σ，GWRP / C2P 各一张。这张图直接支撑 C2P 的层选择决策。
3. **相关性表**：每个诊断量 vs mini-CAM-mIoU 的 Spearman $\rho_s$（跨 epoch，按 seed 分别计算后报均值±std），以及峰值 epoch 偏差。

### 7.3 结果的三种可能与对应动作

| 情形 | 动作 |
|---|---|
| 存在早峰 + 至少一个诊断量 $\rho_s\ge0.7$ | **主结果成立**。写成「无标注 checkpoint 选择准则」，这是独立于 C2P 的方法论贡献，可以单独成节 |
| 存在早峰但诊断量都不相关 | 现象成立、解释失败。仍可报告现象本身（对领域有价值），但不能宣称可预测。转而尝试 D5 或逐层量作为预测子 |
| 不存在早峰（H0） | 见 §9.1 |

---

## 8. 与仓库规范的对接

- **默认关闭**。`ProbeConfig.enabled=False`，所有现有脚本行为零变化。已有的 `results/` 目录不受影响。
- **独立输出目录**，不覆盖已有结果，符合 README 的「新运行使用独立输出目录」。
- **manifest 兼容**：复用现有 `manifest.json` 字段，新增 `probe_set_sha256`、`probe_config`。
- **长时间运行走 tmux**，与现有约定一致。
- **`tests/test_diagnostics.py`** 必须包含：
  - 合成输入的数值正确性：均匀 attention → $\kappa=1$、Gini$=0$；one-hot → $\kappa=1/N$、Gini$\to1$；
  - 白噪声特征场 → $E_{\text{hi}}\approx0.5$，$\xi\approx1$；
  - 常数特征场 → 全部能量在 DC；
  - `gwrp_weights` 与 patch branch 现有实现的**逐值一致性**（这条最重要，权重定义对不上整个 D5 就是错的）；
  - 探针前后 `model.training` 状态一致（约束 3 的回归测试）；
  - AMP 开启时探针内部仍为 fp32。
- **AGENTS.md** 中补一条：诊断模块不得写入 `results/` 之外的路径，不得修改 `engine.py`。

---

## 9. 风险与回退

### 9.1 主要风险：H0（不存在早峰）

MCTformer 在 VOC 上的训练长度（数十 epoch）比 DINOv3 的 100 万迭代短三个数量级。**很可能在当前训练窗口内根本观察不到退化**，因为模型还没走到那个阶段。

回退方案（`DIAG-E`）：**主动延长训练到 3× epoch**，看现象是否被诱发。

- 若延长后出现退化：论文叙述改为「该退化在更长训练下必然出现，本诊断给出其早期征兆」——**仍然成立，只是需要延长实验作为支撑**。
- 若延长后仍无退化：H0 真成立，multi-class token + 图像级分类这个组合不受该问题困扰。**这本身是一个值得报告的负结果**，且直接支持 H1（架构优势）。此时第一节转为 §2 中 D5（GWRP–C2P 背离度）的单独分析，仍能服务于主线的机制解释。

**这条回退路径意味着整个计划没有「完全白做」的分支**，这是决定先做它的主要理由。

### 9.2 次要风险

| 风险 | 缓解 |
|---|---|
| 诊断量方差大于 epoch 间变化 | 固定 probe 集（约束 1）+ 无增强（约束 2）；若仍然过大，probe 集扩到 512 张 |
| ξ 拟合不稳定 | 主指标用 $E_{\text{hi}}$，ξ 降为辅助 |
| pos-embed 插值伪影污染谱诊断 | D4 单独在 448 下跑；另做一次「训练即 448」的对照确认伪影量级 |
| C2P 与 GWRP 的权重定义对不上导致 D5 无意义 | 单测强制逐值一致（§8） |
| 逐层 × 逐类存储爆炸 | 长表 + 只存 present 类；COCO（C=80）下 per-class 只存 top-20 频次类 |

---

## 10. 时间表与决策门

| 阶段 | 内容 | 预估 | 决策门 |
|---|---|---|---|
| S0 | probe 集构建 + `metrics.py` + 单测 | 1 天 | 单测全绿 |
| S1 | `spectral.py` + 分辨率验证 | 0.5 天 | 合成场验证通过；若 ξ 不稳则锁定 $E_{\text{hi}}$ |
| S2 | `probe.py` 挂载 + mini-CAM-mIoU 封装 | 1 天 | 探针前后训练结果逐位一致（关键回归） |
| S3 | `DIAG-A` / `DIAG-B` × 3 seeds | 2–3 天（可并行） | **是否存在早峰 → 决定走主线还是 §9.1** |
| S4 | `DIAG-C` 单 CLS 对照 | 1 天 | H1 / H2 判定 |
| S5 | 逐层热图 + 相关性分析 + 报告 | 1 天 | 产出 `docs/MCTformerPlus_Diagnostic_Probe_Results.md` |
| S6 | （条件触发）`DIAG-D` COCO / `DIAG-E` 延长训练 | 3–5 天 | — |

**S3 是生死门**。S0–S2 的基础设施投入约 2.5 天，即使 S3 判定 H0，这套探针在后续的 Gram 约束实验（若推进）中仍然是必需的仪表盘——**调 $\lambda$ 的时候靠的就是 ρ_cc 不失控**。所以基础设施的投入不随 S3 结果沉没。

---

## 11. 与后续工作的接口

本探针是两条后续路线的前置：

1. **Gram 约束**（在 patch 端锚定二阶结构、在 class token 端去相关）：D1 是 class-token 去相关项的直接监控量，D2/D3 是过度平滑风险的报警器。没有这套仪表盘，$\lambda_p$ / $\lambda_t$ 只能盲调。
2. **C2P 层选择的理论化**：D6 逐层剖面把 last-3 / all-layers 的选择从消融变成设计。

此外，在推进 Gram 方向之前**必须先做一轮 prior-art 定位**，重点核对：ToCo（CVPR 2023，Token Contrast，用中间层 token 监督末层 token——与「早期快照锚定后期」精神相近）、AFA（CVPR 2022，从 attention 学亲和）、MCTformer+ 自身是否已含类别 token 对比项（若已含，则 §2.D1 对应的方法项需要重新定位差异）。这一步不做，后面的方法工作有撞车风险。

---

## 附：一句话总结

先花 2.5 天把仪表盘装上，再花 2–3 天看 MCTformer+ 的密集质量到底是「一路上升」还是「早峰回落」——这个问题的答案决定后续所有方法工作的前提，而且无论答案是哪个，仪表盘本身都要留下来用。