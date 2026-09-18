# 归档计划：从 TGCA 中提取可迁移的 ViT 诊断工具集

> **执行者**：Claude Code，在 `jungletada/TGCA` 工作副本上运行。
> **目标**：把与 WSSS / MCTformer 无关的诊断能力、数值方法、扫描框架与方法论沉淀成一个独立、可安装、有回归测试保护的包，供后续任意课题复用。
> **性质**：**纯增量**。不删除、不重命名、不修改 TGCA 现有任何文件的行为。
> **建议产物**：新仓库 `vitprobe`（名字可改），或 TGCA 同级目录下的独立 Python 包。

---

## 0. 给执行者的硬性约束

以下六条在整个任务期间始终有效，违反任何一条应立即停止并报告。

1. **不修改 TGCA 的既有行为。** 允许新增文件、新增测试；不允许改动 `models/`、`experiments/`、`dataloaders/` 中任何已有函数的实现或签名。若提取过程中发现必须重构才能解耦，**停下来报告，不要自行重构**。
2. **不复制 checkpoint。** `results/` 下的 `.pth` 一律不进归档仓库，也不要 `git add -f`。只复制 markdown 报告、CSV、JSON 指标文件，单文件上限 1 MB。
3. **不改写结论。** 文档里的实验数字全部照抄来源文件，**不重新计算、不四舍五入、不"修正"**。若发现来源之间数字矛盾，记录矛盾本身，不做裁决。
4. **本规格可能与仓库实际状态不符。** 本文件基于对话记录撰写，其中部分模块可能只存在于计划文档中而从未实现。**Phase 0 的清点结果优先于本规格的任何描述。**
5. **每个 Phase 结束写一份 `ARCHIVE_LOG.md` 追加条目**：做了什么、遇到什么不符、跳过了什么。
6. **遇到需要判断的分叉就停下报告**，不要替用户做选择。

---

## Phase 0：清点（先做，不写任何产物代码）

对下列每一项，确认**实际存在**、**只在计划文档中**、还是**不存在**。输出到 `ARCHIVE_INVENTORY.md`，一行一项，带文件路径与行号。

### 0.1 诊断函数

| 期望的能力 | 对话中提到的位置 | 状态 |
|---|---|---|
| `effective_support` / $\kappa$（归一化困惑度） | 计划文档 `analysis/diagnostics/metrics.py` 或 `analysis/weight_stats.py` | ? |
| `gini`（排序法实现） | 同上 | ? |
| `weight_stats`（entropy / top1 / dc_share） | `analysis/weight_stats.py` | ? |
| `class_token_affinity` / $\rho_{cc}$ | 计划文档 | ? |
| `patch_norm_stats` / `attractor_stats` / `symptom_overlap` / `lowinfo_score` | `analysis/artifact_probe.py` | ? |
| `class_agnosticism` / $\gamma$ | 最新计划，可能未实现 | ? |
| 径向功率谱 / 相关长度 | `analysis/diagnostics/spectral.py`，可能未实现 | ? |

### 0.2 数值与聚合

| 能力 | 位置 | 状态 |
|---|---|---|
| `aggregate_p2p`（层集 / reduce / alpha / sym） | `models/mctformer_plus.py` | ? |
| `propagate_c2p_weights`（floor / beta） | 同上 | ? |
| C2P 空间权重（mean / product，层集） | `c2p_spatial_weights` | ? |
| 精度转换点 | `models/tgca.py:129`、`models/mctformer_plus.py:1312` | ? |
| 对数域连乘 / 逐层重归一化 | **很可能不存在**，需新写 | ? |

### 0.3 扫描与统计框架

| 能力 | 位置 | 状态 |
|---|---|---|
| 「逐图前向一次、内层遍历配置」的执行结构 | `experiments/ablations/` 下 A2 筛选相关脚本 | ? |
| 多阈值混淆矩阵累加器 | 同上 | ? |
| 图像聚类配对 bootstrap（5000 次） | 某处，A2/artifact 报告都用了 | ? |
| manifest.json 的字段集合 | `results/*/manifest.json` 任一实例 | ? |

### 0.4 结果文件（作为回归测试的黄金值来源）

确认以下文件存在并记录其 SHA256：

```
results/artifact_probe/20260916-voc-s0/decision.md
results/affinity_repair/20260916-voc-s0/ARTIFACT_AFFINITY_REPORT.md
results/affinity_repair/20260916-voc-s0/a2/AFFINITY_SCREENING_REPORT.md
results/affinity_repair/20260916-voc-s0/stage_b_comparison.csv
results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2/**/epoch_metrics.csv
docs/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md
```

**Phase 0 验收**：`ARCHIVE_INVENTORY.md` 完成，且对每个「?」给出了明确结论。**在用户确认清点结果之前不要进入 Phase 1。**

---

## Phase 1：目标包骨架

创建独立包，**不依赖 torch 以外的任何 TGCA 代码**，不依赖 VOC / COCO / MCTformer。

```
vitprobe/
  pyproject.toml            # 依赖仅 torch>=2.0, numpy；scipy 可选
  README.md
  src/vitprobe/
    __init__.py
    layout.py               # TokenLayout —— 解耦的关键
    stats.py                # kappa, gini, entropy, top1, dc_share
    relations.py            # rho_cc, gamma, symptom_overlap
    artifacts.py            # patch norms, attractor score, lowinfo
    spectral.py             # 径向谱, high_freq_ratio, correlation_length
    numerics.py             # 对数域聚合, 逐层重归一化, 下溢遥测
    graph.py                # [可选] aggregate, propagate(floor/damping)
    sweep.py                # ConfigSweep, ConfusionAccumulator
    bootstrap.py            # 图像聚类配对 bootstrap
  tests/
  docs/
    METHODOLOGY.md
    FINDINGS.md
    MIGRATION.md
  templates/
    gates.yaml
    manifest.schema.json
```

### 1.1 `layout.py` —— 这是整个归档里最重要的一个设计决定

所有诊断量当前都硬编码了 MCTformer 的 token 布局（class-first，`n_class=20`，切片 `[:, n_class:, n_class:]`）。**这是唯一真正阻碍迁移的耦合。**

引入一个描述对象，让所有函数接受它而不是裸的 `n_class`：

```python
@dataclass(frozen=True)
class TokenLayout:
    """描述一个 ViT 的 token 序列布局。"""
    n_class: int = 0          # MCTformer: 20;  DINOv3: 0
    n_cls: int = 0            # MCTformer: 0;   DINOv3: 1
    n_register: int = 0       # MCTformer: 0;   DINOv3: 4
    grid_hw: tuple[int, int] = (0, 0)   # (28, 28)
    class_first: bool = True  # 额外 token 在序列前端

    @property
    def n_patch(self) -> int: ...
    @property
    def patch_slice(self) -> slice: ...
    @property
    def class_slice(self) -> slice: ...
    @property
    def register_slice(self) -> slice: ...

    @classmethod
    def mctformer_plus(cls, n_class=20, grid=(28, 28)): ...
    @classmethod
    def dinov3(cls, grid=(28, 28)): ...
```

**验收**：`layout.mctformer_plus().patch_slice` 产生的切片与 TGCA 现有代码中的硬编码切片在 `n_class=20`、448 输入下**完全一致**（写成断言测试）。

---

## Phase 2：提取诊断函数

对 Phase 0 中标为「实际存在」的每一项：

1. 复制实现到目标模块，**逐字保留数学**；
2. 把 `n_class` 参数替换为 `layout: TokenLayout`；
3. 强制 FP32：函数入口 `.float()`，并在 docstring 写明理由（二阶统计与 FFT 在 FP16 下不可靠）；
4. 写 docstring：**这个量测什么、变差的方向、典型取值范围、判定阈值的来源**。

对标为「只在计划文档中」的项：按计划文档中的参考实现新写，**在 docstring 里标注 `# NOT VALIDATED AGAINST TGCA RUNS`**。

### 2.1 已知的取值参照（写进 docstring，便于将来判读）

这些数字来自实际运行，是判读新数据时的锚点：

| 量 | 观测到的取值 | 来源 |
|---|---|---|
| $\kappa$ 传播前 | all-product 0.294；last3-product 0.602 | C2P 设计文档 |
| $\kappa$ 传播后（崩塌） | 0.972（均匀 = 1.0） | 同上 |
| top-1 mass 传播前/后 | 0.465 → 0.013（均匀 = 0.00128，$N=784$） | 同上 |
| M2 高范数占比 | gwrp 1.343%；all_product 0.540%；affinity 1.695% | artifact 探针 |
| M4 top-1% 吸引质量 | 8.156% / 6.360% / 8.855%（均匀 = 1%） | 同上 |
| M5 两症状 Jaccard | 0.323 / 0.0076 / 0.0667（随机基线 0.00547，$k=8$） | 同上 |

注意 M5 的随机基线：**$k=\mathrm{round}(0.01\times 784)=8$ 时，两个独立随机集合的期望 Jaccard 约为 0.00547，不是 0.01。** 这个细节要保留在 docstring 里。

### 2.2 验收：黄金值回归（本 Phase 最重要的一条）

对每个提取出的函数，写一个测试：**用固定随机种子构造的合成张量，同时调用 TGCA 中的原实现与 `vitprobe` 中的新实现，断言 `torch.allclose(atol=1e-7)`。**

- 测试放在 `vitprobe/tests/test_parity_tgca.py`，通过环境变量 `TGCA_PATH` 定位原仓库；
- 若原实现不存在（Phase 0 标为「只在计划文档中」），跳过该测试并在报告中列出；
- **任何一项 parity 测试失败，停止并报告，不要自行调整容差。**

此外补充行为测试（不依赖 TGCA）：

- 均匀注意力 → $\kappa = 1$、Gini $= 0$；
- one-hot → $\kappa = 1/N$、Gini $\to 1$；
- 白噪声特征场 → `high_freq_ratio` $\approx 0.5$；
- 常数特征场 → 全部能量在 DC；
- 探针前后 `model.training` 状态一致。

---

## Phase 3：数值模块（`numerics.py`，主要是新写）

这部分是本次归档中**唯一有实质新代码**的地方，来源是已定位的 FP16 下溢缺陷及其分析。

### 3.1 要实现的三件事

```python
def renorm_chain_product(mats, eps=1e-8):
    """逐层重归一化的连乘。与 norm(prod(mats)) 逐值等价，但中间值始终在 ~1/N 量级。"""

def log_domain_aggregate(mats, weights=None, tau=1.0, eps=1e-30):
    """w = softmax( (1/tau) * sum_l pi_l * log a_l )。
    tau=1, pi=1 → 归一化连乘;  tau=L, pi=1 → 几何平均。
    softmax 内部减最大值, 完全免疫下溢。"""

def underflow_telemetry(mats, layout):
    """返回逐层逐类的 patch 质量 mu_l[c]、最小非零值、clamp 命中数、整行归零计数。"""
```

**必须写进 `log_domain_aggregate` 的 docstring**：head 维的算术平均要在**概率域**做（值约 $10^{-3}$，绝对安全），放到对数域会变成几何平均，语义就变了。**只有跨层聚合需要对数域。**

### 3.2 等价性与安全性测试

- `renorm_chain_product` 与朴素 `prod` 在 FP64 下 `allclose`；
- `log_domain_aggregate(tau=1)` 与 `renorm_chain_product` 在 FP32 下 `allclose`；
- **下溢压力测试**：构造 12 层、每层最大值 $10^{-3}$ 的注意力，断言朴素 FP32 `prod` 产生零而两个新实现不产生零；
- `underflow_telemetry` 能在人造的「整行归零」输入上正确计数。

### 3.3 记录进 `FINDINGS.md` 的边界

$T=804$ 个 token，patch 上典型值约 $10^{-3}$，12 层连乘得 $10^{-36}$；FP32 最小正规数 $1.18\times10^{-38}$，FP16 最小次正规数 $5.96\times10^{-8}$。**结论：多层 softmax 连乘在 FP32 下也只有约两个数量级的余量，在 FP16 下从第 3 层起就不可用。**

---

## Phase 4：扫描框架（`sweep.py`）

把 A2 筛选里的执行结构抽象出来，这是所有推理端扫描的通用骨架。

```python
class ConfigSweep:
    """逐样本前向一次, 内层遍历 N 个配置。
    前向是唯一的大开销; 配置数量几乎不增加成本。"""
    def __init__(self, configs, n_thresholds, n_classes): ...
    def observe(self, config_id, prediction, target): ...
    def results(self) -> "pandas.DataFrame": ...
```

`ConfusionAccumulator`：每配置 × 每阈值一个 $K\times K$ 累加器。内存参照：**73 配置 × 50 阈值 × 21² × 8 B ≈ 12.9 MB**，可忽略——这个估算要写进 docstring，因为它是「为什么不必缓存中间张量」的依据。

`bootstrap.py`：图像聚类配对 bootstrap，默认 5,000 次重采样，seed 可配（TGCA 中用的是 20260916）。**必须与 TGCA 现有实现做 parity 测试**（同 seed 同输入 → 同 CI）。

---

## Phase 5：方法论文档与模板

`docs/METHODOLOGY.md`，把这几周实际生效过的做法写成可复用的规程，每条都带一个真实案例：

| 规程 | 真实案例 |
|---|---|
| **预注册判定门** | artifact 探针的三条门（M2 ≥ 0.5%、M4 ≥ 5%、M5 ≥ 0.2）在结果出来前写定，直接产出 NO-GO，省下两周 |
| **双宿主符号复现代替 Bonferroni** | 36–72 格网格下，要求在 `gwrp` 与 `all_product` 上符号一致，比把 CI 拉到 99.86% 更可解释 |
| **交叉拟合避免超参泄漏** | α 扫描按图像 ID 固定二分，在 A 半选、B 半报，反之亦然；不增加任何计算 |
| **逐值等价性单测** | 任何重构必须证明旧路径逐值不变，否则历史结果失去可比性 |
| **零训练先导** | 崩塌是 $P$ 和 $w$ 的纯函数，用现成 checkpoint 就能筛完 72 个配置 |
| **oracle 上界与方法结果分列** | top-$m$ 分析必须用无标注判据排序；用 GT 排序的曲线只能标为 oracle |
| **负结果的可读性设计** | 指标设计成「参数本身回答问题」（A1 的 $\mathbf{a}$ 熵 0.99999 直接说明模型没动它） |

`templates/gates.yaml`：判定门的结构化模板（条件、阈值、阈值的来源、触发后的动作）。
`templates/manifest.schema.json`：从 TGCA 现有 manifest 实例反推字段集合，加上 `precision`、`autocast`、`probe_set_sha256`。

---

## Phase 6：`FINDINGS.md` —— 与领域无关的结论

只收录**换了领域仍然成立**的条目。每条：现象、机制、证据、可迁移性。

1. **直流基底**。跨层未归一化求和引入 $f[i]/N$ 的常数基底；和为 1 的归一化保留它（权重退化为 GAP，熵 0.294 → 0.972），min-max 归一化吸收它（实测 delta 4.3e-08）。**同一个算子在两种归一化下命运相反。**
2. **多层 softmax 连乘的下溢边界**（Phase 3.3）。
3. **高范数 artifact 与注意力吸引子在小模型上会解耦**：gwrp 上两者重合（Jaccard 0.323，约随机的 59 倍），C2P 训练后降到随机水平（0.0076）。与「高范数只在更大模型的中后期出现」的公开观察一致。
4. **分类指标与密集质量的相关性是设定依赖的**。VOC 多标签 + ImageNet 预训练微调下，epoch ≥ 10 时 class mAP 与 CAM mIoU 的 Spearman 为 **+0.65 ~ +0.95**（四条轨迹），与 WSOL 文献中 CUB 单物体设定下的「分类涨、定位跌」相反。**不要跨设定搬运这类结论。**
5. **单 seed 结果不可信的一个具体代价**：同一配置两个 seed 相差 10.07 pp，且与基线的差值符号翻转；事后定位为 FP16 下溢而非方法失稳。**双 seed 是发现它的最低成本。**

第 5 条要写清楚：**根因是数值缺陷，不是优化动力学**；当时基于「注意力进入梯度路径形成自强化环路」的推断是错的。保留这个更正，因为它本身就是「先测量再推断」的案例。

---

## Phase 7：迁移文档与收尾

`docs/MIGRATION.md`：

- 三个 `TokenLayout` 实例（MCTformer+、DINOv3、朴素 ViT）的构造示例；
- 每个诊断量在非 WSSS 场景下的用法（比如 $\gamma$ 用于任何多 query 架构的 query 特异性检查）；
- 明确列出**没有迁移过来的东西**：GWRP、C2P pooling、affinity 修复、CAM 生成、VOC/COCO 数据管线——这些是 WSSS 方法代码，留在 TGCA。

`ARCHIVE_LOG.md` 汇总：每个 Phase 的完成情况、跳过项、发现的不符、未通过的 parity 测试。

**最终验收**：

1. `pip install -e .` 后 `pytest` 全绿（跳过的 parity 测试单独列出）；
2. 包内**零处** `import` 指向 TGCA；
3. 全文搜索 `voc`、`coco`、`mctformer`、`cam` 只应命中 docstring 与文档，不应命中函数签名或逻辑；
4. 一个端到端 smoke test：用随机张量 + `TokenLayout.dinov3()` 跑通全部诊断量，证明包在非 MCTformer 布局下可用。

---

## 附：Phase 依赖与建议顺序

```
Phase 0 (清点) ── 用户确认 ──> Phase 1 (骨架)
                                  ├─> Phase 2 (诊断) ──> Phase 7
                                  ├─> Phase 3 (数值) ──> Phase 7
                                  └─> Phase 4 (框架) ──> Phase 7
                              Phase 5 / 6 (文档) 可随时并行
```

Phase 2/3/4 互不依赖，可并行。**Phase 0 之后必须停下等用户确认**——因为本规格中相当一部分模块可能从未实现，清点结果会显著改变后续工作量。
