# A2-β1：固定 β=1.0 的四变量推理端扫描

> 状态：计划稿，未实施。
> 目的：补上 A2 筛选中 **β 与其余四个变量完全混淆** 的设计缺陷，给出 `p_layers / p_reduce / p_alpha / p_sym` 在唯一可用的 β 下的干净测量。
> 前置：`results/affinity_repair/20260916-voc-s0/a2/AFFINITY_SCREENING_REPORT.md`、`.../stage_b_comparison.csv`、`results/artifact_probe/20260916-voc-s0/decision.md`
> 建议落盘：`docs/MCTformerPlus_A2_Beta1_Sweep.md`，输出 `results/affinity_repair/20260917-voc-s0-beta1/`
> 授权边界：**推理端 only，不训练。** 本队列不授权 register、Gram、COCO、额外 seed，也不授权任何 Stage B 重训；是否重训由 §6 的门决定，需另行开队列。

---

## 1. 为什么必须补这一轮

### 1.1 缺陷

A2 的 22 行里只有 4 行取 β=1.0（两个 model × {native, floor=min}），其余 18 行全部是 β ∈ {0.1, 0.3}。而 β<1 在推理端独立致命：

| config | fixed mIoU | precision | recall |
|---|---:|---:|---:|
| `all_sum_1.0_false_none_1.0`（native） | 0.7232 | 0.833 | 0.847 |
| `all_sum_1.0_false_none_0.1` | 0.4997 | 0.925 | **0.488** |
| `last3_mean_4.0_true_mean_0.3` | 0.0596 | 0.894 | **0.020** |

precision 全部升到 0.92+ 而 recall 崩塌——这是「只剩判别性区域」的签名。推理端的传播**本来就是用来把 CAM 从判别性峰值扩散到完整物体的**，掺回未传播的权重等于取消扩散。

**后果**：`p_layers`、`p_reduce`、`p_alpha`、`p_sym` 从未在 β=1.0 下被测过。现有的 18 行全部无法用于这四个变量的归因。Stage B 的 `conservative` 组同样把 α=2、sym、last3、rownorm 与 β=0.3 捆在一起，也无法归因。

### 1.2 这一轮要回答的唯一问题

> 在 β=1.0（推理端唯一可用的传播强度）下，`p_layers × p_reduce × p_alpha × p_sym` 的任何组合能否超过 native 的 0.72316？

回答「能」→ 开 Stage B 队列。回答「不能」→ affinity 线封盘，转为论文的分析章节。

---

## 2. 两处可以先行化简的地方

### 2.1 `floor` 退出变量集（降为控制单元）

A3 已证明 floor=min 相对 native 的 delta 是 **4.29e-08 / 7.94e-09**——纯浮点噪声。原因是推理端 CAM 流程做 min-max 归一化，加性常数与正标量都被吸收。

按当前实现（`floor` 在 `sum→1` 归一化之前作用），这个吸收对任意 α、sym 都成立。**因此本轮固定 `floor=none`**，但保留 3 个 `floor=min` 控制单元（取 α≠1 的配置），确认「无操作」性质在 α≠1 下依然成立。若控制单元出现非零 delta，说明实现里有未预期的路径，需先查清再继续。

### 2.2 `p_reduce` 的 `sum` 与 `mean` 在推理端数学等价

按参考实现：

- **α = 1**：`sum` 与 `mean` 相差常数因子 $L$；下游 min-max 归一化吸收正标量 → 等价。
- **α ≠ 1**：幂次前先做 `P = P / P.sum(-1, keepdim)` 行归一化，$L$ 因子在这一步被消掉 → 等价。
- `p_sym` 的对称化是线性运算，不破坏上述任一条。

**所以 `p_reduce` 的有效取值只有两个：`sum`（= `mean`）与 `rownorm_mean`**。后者逐层先行归一化再平均，改变的是**各层之间的相对权重**（浅层行质量大，rownorm 后其相对贡献被抬高），这才是真正不同的算子。

本轮因此取 `p_reduce ∈ {sum, rownorm_mean}`，另加 3 个 `mean` 单元作为**数值恒等性检查**。这个检查同时验证 harness 本身：若 `sum` 与 `mean` 的 fixed mIoU 不在 1e-6 内相等，说明流程里存在对绝对尺度敏感的阈值或 fallback，必须先定位。

---

## 3. 网格

| 变量 | 取值 | 数量 |
|---|---|---|
| `p_layers` | `last3`(L10-12) / `last6`(L7-12) / `all`(L1-12) | 3 |
| `p_reduce` | `sum` / `rownorm_mean` | 2 |
| `p_alpha` | 1.0 / 2.0 / 4.0 | 3 |
| `p_sym` | false / true | 2 |
| `floor` | **none**（固定） | 1 |
| `beta` | **1.0**（固定） | 1 |

主网格 **36 单元/model**，加 3 个 `mean` 恒等检查 + 3 个 `floor=min` 控制 = **42 单元/model**。

Model：`all_product`（主，native 0.72316）与 `gwrp`（复现校验，native 0.70063）。**共 84 单元。**

`all_sum_1.0_false_none_1.0` 即 native，自然落在网格内，作为两个 model 的 delta 基准（与 A2 一致）。

**可选扩展（不进主网格，视 §6 结果再定）**：`p_layers=last1`（仅 L12），用于回答「最后一层单独是否够」。加它只增 14 单元，成本可忽略，但它引入第四个层档位会改变多重比较的计数——如果做，须在开跑前写进本文件，不得事后追加。

---

## 4. 实现与成本

### 4.1 关键优化：前向与配置解耦

CAM 生成的开销全在前向（VOC train 1,464 张 × 3 scales × flip = 6 次前向/图），而**前向结果与配置无关**——变的只有 $P$ 的聚合方式与 $S\!\otimes\!P$ 的乘积。

因此：**逐图前向一次，在内层循环里遍历 42 个配置**，不要为每个配置重跑数据集。

- 不缓存 $P$：35×35=1225 网格下，12 层 FP32 是 72 MB/图/scale，跨图缓存会爆盘；
- 每个配置维护自己的**混淆矩阵累加器**（21×21），以及阈值网格（步长 0.02，约 50 档）的混淆矩阵组：42 × 50 × 441 × 8 B ≈ 7.4 MB，可忽略；
- 逐图的 per-config 预测不落盘（沿用已有的清理策略）。

**预计成本**：约等于 1 个配置的完整 CAM 评估 + 42 次廉价矩阵运算，两个 model 合计 **1.5–2.5 小时**。

### 4.2 配置枚举

```python
# experiments/ablations/enumerate_beta1_grid.py
from itertools import product

P_LAYERS = ("last3", "last6", "all")
P_REDUCE = ("sum", "rownorm_mean")
P_ALPHA  = (1.0, 2.0, 4.0)
P_SYM    = (False, True)

def cid(layers, reduce, alpha, sym, floor, beta):
    return f"{layers}_{reduce}_{alpha}_{str(sym).lower()}_{floor}_{beta}"

def main_grid():
    for l, r, a, s in product(P_LAYERS, P_REDUCE, P_ALPHA, P_SYM):
        yield dict(p_layers=l, p_reduce=r, p_alpha=a, p_sym=s,
                   floor="none", beta=1.0, role="main")

def identity_checks():
    """sum ≡ mean 的数值恒等检查; 同时验证 harness 对绝对尺度不敏感。"""
    for l, a, s in (("all", 1.0, False), ("all", 2.0, True), ("last3", 4.0, False)):
        yield dict(p_layers=l, p_reduce="mean", p_alpha=a, p_sym=s,
                   floor="none", beta=1.0, role="identity")

def floor_controls():
    """确认 floor 在 alpha != 1 下依然是无操作。"""
    for l, a, s in (("all", 2.0, False), ("last6", 4.0, True), ("last3", 2.0, True)):
        yield dict(p_layers=l, p_reduce="sum", p_alpha=a, p_sym=s,
                   floor="min", beta=1.0, role="control")

CONFIGS = list(main_grid()) + list(identity_checks()) + list(floor_controls())
assert len(CONFIGS) == 42
```

### 4.3 新增的诊断列

A2 的输出列全部保留，另加三列，使**负结果也可解释**：

| 列 | 定义 | 作用 |
|---|---|---|
| `fg_area_frac` | t=0.45 下被判为任意前景的像素占比 | 直接区分「过扩散」与「欠扩散」 |
| `cam_entropy` | min-max 归一化后逐类 CAM 的归一化空间熵，正类平均 | 与训练端 §3.1 的熵口径对齐，可跨两侧比较 |
| `fallback_class_rows` | 沿用 A2 定义 | floor 控制单元必须为 0；主网格应全为 0 |

`fallback_class_rows` 在主网格里**必须恒为 0**（因为 `floor=none`）。若不为 0，说明存在另一条未记录的回退路径，须查清。

---

## 5. 预先声明的预期（写在开跑之前）

写下这一节的目的是让任何结果都不可事后重新叙述。

| 变量 | 方向 | 理由 |
|---|---|---|
| `p_layers`：last3/last6 vs all | **预期变差** | 去掉弥散的浅层 = 减少扩散，与 β<1 的失败方向相同 |
| `p_alpha` > 1 | **预期变差** | 锐化 $P$ = 减少扩散，同上 |
| `p_sym` = true | **可能变好** | $(P+P^\top)/2$ 是 CLIP-ES / MCTformer 那条线的标准做法，是四个变量里唯一有正向先验的 |
| `p_reduce` = rownorm_mean | **方向不定** | 抬高浅层相对权重 → 扩散更强；可能过扩散，也可能正好补足 |

**综合预期：主网格 36 个单元中最多只有 `sym=true` 一支可能超过 native，且概率不高。** 若结果符合预期，本轮的价值在于把「β 混淆」这个可被审稿人一眼看穿的缺陷补成一张完整的、可辩护的消融表。

---

## 6. 预先注册的判定门

采用与 artifact 探针相同的纪律：门写在跑之前，**结果出来后不得修改门**。

某配置**合格**，当且仅当同时满足：

1. **主判**：在 `all_product` 上 `delta_fixed > 0`，且图像聚类配对 bootstrap（5,000 次，seed 20260916）的 **95% CI 下界 > 0**；
2. **复现判**：在 `gwrp` 上 `delta_fixed > 0`（仅要求符号一致，不要求 CI 排除 0）。

第 2 条替代 Bonferroni 校正。理由：36 个单元在 95% 下期望约 1.8 个假阳性，而要求在**另一个独立 checkpoint 上符号复现**，比把 CI 拉到 99.86% 更可解释，也更贴近「这个算子是否真的更好」这个问题。两条都写进报告。

**分支**：

| 结果 | 结论 | 动作 |
|---|---|---|
| ≥1 个配置合格 | β=1 下存在更优算子 | **另开 Stage B 队列**，只重训合格配置中 `delta_fixed` 最大的 1 个；本队列不授权 |
| 0 个合格，但恒等检查与 floor 控制通过 | **四变量在 β=1 下均无增益** | affinity 线封盘，转为论文分析章节；不再投入训练资源 |
| 恒等检查或 floor 控制未通过 | harness 存在未知路径 | **停止**，先定位，本轮全部结果作废 |

---

## 7. 产出

`results/affinity_repair/20260917-voc-s0-beta1/`

```
manifest.json              # 代码 SHA、环境、checkpoint SHA256、bootstrap seed、网格定义
commands.sh
beta1_sweep.csv            # 84 行；A2 全部列 + §4.3 三列 + role 列
BETA1_SWEEP_REPORT.md      # §6 两条门逐配置打勾/打叉；恒等与控制单元单列
heatmaps/                  # p_layers × p_alpha 的 fixed mIoU 热图，sym/reduce 各一张
```

`BETA1_SWEEP_REPORT.md` 的措辞沿用既有规范：

- 明确「Inference-only screening, NOT retraining results or causal proof.」
- 明确固定阈值 0.45 是排序指标，best threshold 仅作诊断；
- 明确单 seed，bootstrap 不含训练种子不确定性；
- 明确本队列未授权 register / Gram / COCO / 额外 seed / Stage B。

---

## 8. 这一轮在论文里的位置

无论结果落在 §6 的哪一支，它填的都是同一个洞。

论文目前的分析链条是：

> C2P > GWRP（+2.25 pp，4/4 变体） → 朴素 P2P 传播 −2.91 pp → 直流基底解析诊断（熵 0.294→0.972，与解析式吻合） → 去基底回收 81.7%（69.41→71.78）但仍 −0.53 → **传播在推理端必需、在训练 pooling 中有害，成因是两侧归一化不同（A3 的 delta ≈ 4e-08 直接证明推理端已隐式去基底）**

这条链条缺的一环是：**「你们只试了一种 $P$ 聚合方式，换一种是不是就好了？」** 本轮给出的是一张 36 格的完整答复表。如果全负，那一句「我们在 β=1 下穷举了层范围、层归一化、锐化与对称化四个维度，均无增益」比现在的 22 行混淆表强得多，而成本只有两小时。

---

## 9. 与其它队列的关系

本轮**可与多 seed 补跑并行**：它是纯推理，只读 checkpoint，不占训练队列。

当前阻塞项仍是 **GWRP / all-mean / all-product 的多 seed 重跑**（论文核心主张 +2.25 pp 目前是 seed-0 单次）。建议先启动那三组训练，在其运行期间完成本轮与 `conservative` 组参数生效性核对（约 10 分钟）。
