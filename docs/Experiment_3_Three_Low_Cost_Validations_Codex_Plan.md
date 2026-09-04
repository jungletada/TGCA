# Experiment 3：三组低成本机制验证  
## Presence Axis、Last-Three CAM Readout 与 Late Inter-Class Mixing（Codex 执行计划）

> **目标仓库：** https://github.com/jungletada/TGCA  
> **服务器目录：** `~/code/TGCA`  
> **执行环境：** 复用 Experiment 1/2 的 `tgca-repro` 环境  
> **主模型：** MCTformer+  
> **架构对照：** MCTformer  
> **数据：** PASCAL VOC 2012 val，单尺度 448×448，无 flip、无 CRF  
> **已有 checkpoints、Experiment 1 和 Experiment 2 结果均视为只读输入**  
> **本阶段性质：** 机制验证与 inference-time intervention，不训练新模型、不提出新模块

---

# 0. 背景与动机

Experiment 2 已经确认以下事实。

## 0.1 Raw feature cosine 与真实 attention geometry 明显不同

LaST-style score：

\[
S^{feat,post}_{c,j}
=
\cos(c_c,p_j)
\]

在 MCTformer+ L12 的 target-vs-background AUROC 约为：

\[
0.592.
\]

但真实 QK energy 与 \(A_{c2p}\) 的 AUROC 已达到约：

\[
0.853,\quad0.855.
\]

同时：

\[
\rho(S^{feat,norm},E^{QK})\approx0.148,
\]

而：

\[
\rho(E^{QK},A_{c2p})\approx0.979.
\]

因此 raw residual-stream cosine 中存在的 late recoupling，并不能直接视为实际 attention routing。

---

## 0.2 MCTformer+ 的 class-token readout 存在一个明确公共方向

当前源码中：

```python
x_cls_logits = x_cls.mean(-1)
```

因此类别 \(c\) 的 logit 是：

\[
z_c
=
\frac1D\sum_{d=1}^{D}c_{c,d}.
\]

定义：

\[
u
=
\frac{1}{\sqrt D}
[1,\ldots,1]^\top,
\]

则：

\[
z_c
=
\frac{1}{\sqrt D}u^\top c_c.
\]

所以 final class token 可以精确分解成：

\[
c_c
=
a_cu+r_c,
\qquad
a_c=u^\top c_c,
\qquad
r_c\perp u.
\]

其中 \(a_cu\) 是候选 **class-presence component**，\(r_c\) 是与该固定 readout direction 正交的 residual representation。

MCTformer 和 MCTformer+ 的 class-token logits都采用同样的 dimension-mean readout。源码：

- `models/mctformer.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mctformer.py
- `models/mctformer_plus.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mctformer_plus.py

---

## 0.3 Multi-label supervision 可能使所有 present class tokens 共享正向 presence component

对一张多标签图像：

\[
y_{c_1}=1,\quad y_{c_2}=1,
\]

multi-label classification loss 同时要求：

\[
z_{c_1}>0,\quad z_{c_2}>0.
\]

由于两者使用同一个 \(u\) readout direction，两个 present class tokens 都会被推动到 \(+u\) 方向。

Experiment 2 的旁证是：MCTformer+ L12 中，两个正类别均被 class-token head 和 patch-head 正确判断为 positive 时，class-token pair cosine 约为：

\[
0.866;
\]

若至少一个正类别未被正确判断，则约为：

\[
-0.224.
\]

该现象可能表示：

\[
\boxed{
\text{present class tokens share a common class-presence axis}
}
\]

但仍需通过显式分解验证。

---

## 0.4 Actual \(A_{c2p}\) 的类别特异性在 L10 最好，L11–L12 退化

MCTformer+：

| Layer | C-PiM | Target-vs-BG AUC | Target-vs-Other AUC | Class-map top10 Jaccard |
|---:|---:|---:|---:|---:|
| L10 | 0.766 | 0.913 | 0.780 | 0.200 |
| L11 | 0.574 | 0.894 | 0.725 | 0.306 |
| L12 | 0.409 | 0.855 | 0.608 | 0.467 |

而原生 MCTformer+ CAM 使用最后三层 attention：

\[
A^{native}_{c2p}
=
\frac13
\left(
A^{(10)}_{c2p}
+
A^{(11)}_{c2p}
+
A^{(12)}_{c2p}
\right).
\]

源码：

- `models/mctformer_plus.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mctformer_plus.py

因此需要验证：

1. L11/L12 的 routing recoupling 是否真的降低 CAM mIoU；
2. late inter-class token mixing 是否是这种 recoupling 的因果来源。

---

# 1. 本阶段的三个验证

本阶段只执行以下三组。

## Validation A：Presence-Axis Decomposition

验证：

> Experiment 1 中 raw feature recoupling 是否主要由共同的 class-presence component 造成。

## Validation B：L10 / L11 / L12 / Last3 CAM Readout Intervention

验证：

> L10 更好的 \(A_{c2p}\) region quality 是否真正转化成更好的 raw CAM mIoU，还是被 patch CAM 与 \(A_{p2p}\) 抵消。

## Validation C：Late Inter-Class Mixing Causal Intervention

验证：

> L10–L12 的 class-to-class off-diagonal value mixing 是否因果导致后续 class-to-patch routing 丢失类别特异性。

---

# 2. 全局执行纪律

## 2.1 禁止事项

本阶段禁止：

- 重新训练 MCTformer/MCTformer+；
- 修改 checkpoint；
- 加入 BG token、register token；
- 加入 competitive slot；
- 修改 loss；
- P2C blocking；
- head pruning；
- LaST FFT/selective aggregation；
- 根据 VOC val 最佳结果立即包装“新方法”；
- 自动进入下一阶段。

## 2.2 源结果不可修改

以下目录和 checkpoints 均只读：

```text
Experiment 1 result roots
Experiment 1 paired analysis root
Experiment 2 signal/canonical/analysis roots
MCTformer checkpoint
MCTformer+ checkpoint
VOC images/masks/labels
```

派生结果写入：

```text
results/lazy_assignment/experiment3_three_validations/<run_id>/
```

## 2.3 首先提交现有分析代码

Experiment 2 报告显示部分 analysis/test files 在当时可能尚未提交。开始本阶段前：

1. 检查 `git status`；
2. review Experiment 1/2 analysis code；
3. 提交到独立 commit；
4. 创建 tag；
5. 在本阶段 metadata 中记录 commit/tag。

不要在 untracked analysis code 基础上继续累积。

---

# 3. 推荐代码目录

```text
analysis/
└── lazy_assignment/
    └── experiment3/
        ├── README.md
        ├── audit_experiment3_inputs.py
        ├── presence_axis.py
        ├── run_presence_axis_analysis.py
        ├── cam_layer_intervention.py
        ├── run_cam_layer_intervention.py
        ├── c2c_intervention.py
        ├── run_c2c_intervention.py
        ├── build_experiment3_canonical.py
        ├── analyze_experiment3.py
        ├── bootstrap_experiment3.py
        ├── plot_experiment3.py
        ├── render_experiment3_examples.py
        ├── generate_experiment3_report.py
        └── tests/
            ├── test_presence_axis_decomposition.py
            ├── test_presence_direction_estimation.py
            ├── test_cam_source_equivalence.py
            ├── test_c2c_self_reroute.py
            ├── test_c2c_baseline_equivalence.py
            ├── test_l12_negative_control.py
            ├── test_source_immutability.py
            └── test_bootstrap_pairing.py
```

---

# 4. Phase 0：Input Audit

先生成：

```text
audit/
├── INPUT_AUDIT.md
├── source_metadata.json
├── checkpoint_verification.json
├── experiment2_linkage.json
└── immutable_manifest_before.csv
```

必须核对：

1. Experiment 2 source root；
2. 两个 checkpoint SHA256；
3. 1,449 VOC val IDs；
4. 2,147 positive image–class pairs；
5. 522 multi-label images；
6. image/mask transform；
7. layer numbering；
8. class index 0–19 与 mask ID 1–20；
9. Experiment 2 canonical tables是否完整；
10. existing signal NPZ 是否已经包含本阶段所需 tokens/maps。

优先复用 Experiment 2 signal artifacts。只有缺少必要 full class/patch token 时，才重新执行 deterministic forward。

---

# 5. Validation A：Presence-Axis Decomposition

## 5.1 核心假设

### H-A1：Final token 存在精确的固定 presence axis

因为：

\[
z_c=\operatorname{mean}(c_c),
\]

final output 上：

\[
u=\mathbf1/\sqrt D
\]

是精确 classifier direction。

### H-A2：Raw class-token pair cosine 的高值主要来自 \(u\) 分量

若移除 \(u\) 后：

\[
\cos(c_a^\perp,c_b^\perp)
\ll
\cos(c_a,c_b),
\]

尤其在 `both_positive` pairs 中明显下降，则 shared presence component 成立。

### H-A3：Experiment 1 的 raw feature-map recoupling部分来自 presence axis

若移除 \(u\) 后：

\[
\operatorname{Jaccard}
(T_{10}(S^\perp_{c_a}),
T_{10}(S^\perp_{c_b}))
\]

显著低于 raw score-map overlap，则后层 recoupling 不完全是 class semantic ownership failure。

### H-A4：Pre-attention LayerNorm 去除了该 component

若：

\[
S^\perp
\]

与 Experiment 2 的：

\[
S^{feat,norm}
\]

显著比 raw score 更一致，则支持 LayerNorm 正在抑制 presence component。

---

## 5.2 精确 final-layer decomposition

对每个 class token：

\[
u=\frac{\mathbf1}{\sqrt D},
\]

\[
a_{i,c}=u^\top c_{i,c},
\]

\[
c^\perp_{i,c}=c_{i,c}-a_{i,c}u.
\]

验证：

\[
\boxed{
z_{i,c}=\frac{a_{i,c}}{\sqrt D}
}
\]

与模型 `x_cls.mean(-1)` 数值一致：

```text
max_abs_diff < 1e-6
```

对 patch token：

\[
b_{i,j}=u^\top p_{i,j},
\]

\[
p^\perp_{i,j}=p_{i,j}-b_{i,j}u.
\]

---

## 5.3 四种 feature score control

对 L1–L12 都计算；但只有 L12 的 \(u\) 可以被称为精确 final classifier axis，早层只能称为 mean-direction control。

### V0：Raw

\[
S^{raw}_{c,j}
=
\cos(c_c,p_j).
\]

### V1：Class-only axis removed

\[
S^{c\perp}_{c,j}
=
\cos(c_c^\perp,p_j).
\]

### V2：Patch-only axis removed

\[
S^{p\perp}_{c,j}
=
\cos(c_c,p_j^\perp).
\]

### V3：Both removed

\[
\boxed{
S^{\perp}_{c,j}
=
\cos(c_c^\perp,p_j^\perp)
}
\]

### V4：Actual pre-attention normalized feature score

复用 Experiment 2：

\[
S^{norm}_{c,j}
=
\cos(
\operatorname{norm1}(c),
\operatorname{norm1}(p)
).
\]

---

## 5.4 Class-token representation metrics

按层、按 image 中 class pair 计算：

```text
raw pair cosine
axis-removed pair cosine
axis contribution to each token norm
axis dot-product contribution
residual dot-product contribution
```

定义：

\[
AxisEnergy(c)
=
\frac{a_c^2}{\|c_c\|_2^2+\epsilon}.
\]

对 class pair：

\[
Dot_{axis}(a,b)=a_aa_b,
\]

\[
Dot_{resid}(a,b)=r_a^\top r_b.
\]

不要用：

\[
Dot_{axis}/Dot_{total}
\]

作为唯一指标，因为当 total dot 接近 0 时不稳定。

---

## 5.5 Presence strata

利用所有 20 个 classes，而不只正类，分成：

```text
GT positive / GT negative
predicted positive / predicted negative
both-positive class pair
positive-negative pair
negative-negative pair
```

主要比较：

\[
a_c\mid y_c=1
\]

和：

\[
a_c\mid y_c=0.
\]

该 separation 本身与 classification logit 等价，因此不作为新发现；真正重要的是：

- AxisEnergy；
- 去轴前后 pair cosine；
- 去轴前后 map overlap；
- class-specific residual 是否仍保持类别区分。

---

## 5.6 Data-derived shared presence direction

固定 all-ones axis只在 final output 是精确 classifier direction。为了测试早层是否存在变换后的 shared presence subspace，增加一个低成本、cross-fitted control。

### Step 1：按类别去除固定 identity mean

对 layer \(l\)：

\[
\bar c_c^{(l)}
=
\mathbb E_i[c_{i,c}^{(l)}],
\]

\[
\tilde c_{i,c}^{(l)}
=
c_{i,c}^{(l)}-\bar c_c^{(l)}.
\]

### Step 2：用一半图像估计 shared presence direction

按 image ID hash 固定分成 `fit` 与 `eval` 两半。

\[
d_c^{(l)}
=
\mathbb E[
\tilde c_{i,c}^{(l)}\mid y_{i,c}=1
]
-
\mathbb E[
\tilde c_{i,c}^{(l)}\mid y_{i,c}=0
].
\]

\[
u_{shared}^{(l)}
=
\operatorname{normalize}
\left(
\frac1C\sum_cd_c^{(l)}
\right).
\]

### Step 3：只在 held-out images 上评价

报告：

- projection 对 presence label 的 AUROC；
- 各类别 \(d_c\) 与 \(u_{shared}\) cosine；
- \(u_{shared}^{(12)}\) 与 all-ones \(u\) cosine；
- 去除 \(u_{shared}^{(l)}\) 后的 class-map overlap。

交换 fit/eval halves，再合并 cross-fitted结果。

禁止在同一图像上同时估计方向并评价。

---

## 5.7 Presence-axis 对 score maps 的影响

对 V0–V4 计算：

- class-pair map Spearman；
- top-5/10/20% Jaccard；
- Experiment 2 的 target/other/BG C-PiM；
- target-vs-BG AUC；
- target-vs-other AUC；
- shared support ownership；
- raw vs normalized/QK map correlation。

重点层：

\[
L4,L5,L9,L10,L11,L12.
\]

---

## 5.8 Validation A 主输出

```text
tables/
├── presence_axis_token_metrics.csv
├── presence_axis_pair_metrics.csv
├── presence_axis_map_metrics.csv
├── shared_presence_direction.csv
├── presence_axis_gt_region_metrics.csv
└── presence_axis_probe_linkage.csv
```

主图：

```text
plots/
├── axis_energy_by_layer_and_status.png
├── class_pair_cosine_raw_vs_residual.png
├── class_map_overlap_raw_vs_axis_removed.png
├── raw_perp_norm_qk_probe_comparison.png
├── shared_presence_direction_alignment.png
└── presence_axis_region_quality.png
```

---

## 5.9 Validation A 判定

### Strong support

同时满足：

1. `both_positive` class-pair cosine 在去轴后显著下降；
2. L10–L12 raw class-map top10 Jaccard 在去轴后显著下降；
3. \(S^\perp\) 与 \(S^{norm}\) 的相关性明显高于 \(S^{raw}\) 与 \(S^{norm}\)；
4. held-out `u_shared` 能跨类别预测 presence；
5. final `u_shared` 与 all-ones axis 高度对齐。

### Partial support

只解释 token pair cosine，但不解释 spatial map recoupling。

### Not supported

去除 axis 后 token/map geometry 基本不变。

---

# 6. Validation B：L10 / L11 / L12 / Last3 CAM Readout Intervention

## 6.1 目标

Experiment 2 显示 L10 单层 \(A_{c2p}\) 的 region quality 最好，但 native CAM 使用 L10–L12。

本验证回答：

> 后层 routing ambiguity 是否真实降低最终 raw CAM segmentation，而不是只影响 attention-level diagnostic metrics？

---

## 6.2 优先复用 Existing Experiment 2 Signals

先检查 signal NPZ 是否已有：

```text
per-layer head-mean A_c2p
patch CAM
all-layer A_p2p
VOC transformed GT
```

若已有，直接离线生成所有 CAM variants，不重新跑模型。

若缺少任一必要 signal，才使用完全相同 checkpoint、transform 和 signal collector补充。

---

## 6.3 Primary CAM variants

### B0：Native Last3

MCTformer+：

\[
A^{B0}
=
\frac13(A^{10}+A^{11}+A^{12}).
\]

MCTformer：

\[
A^{B0}
=
A^{10}+A^{11}+A^{12}.
\]

各自保持 native formula。

### B1：L10 only

\[
A^{B1}=A^{10}.
\]

### B2：L11 only

\[
A^{B2}=A^{11}.
\]

### B3：L12 only

\[
A^{B3}=A^{12}.
\]

### B4：L10–L11

MCTformer+：

\[
A^{B4}=\frac12(A^{10}+A^{11}).
\]

MCTformer使用与其 native scale convention 对应的 sum，但最终 per-class normalization前保留原始公式。

### B5：Mid3 diagnosis control

\[
A^{B5}
=
\operatorname{aggregate}(A^4,A^5,A^6).
\]

这只是诊断 control。

---

## 6.4 CAM construction

### MCTformer+

保持原生 patch CAM 和 \(A_{p2p}\)：

\[
CAM^{c2p,Bk}
=
\sqrt{
A^{Bk}_{c2p}
\odot
CAM^{patch}
},
\]

\[
CAM^{final,Bk}
=
A_{p2p}^{all}
CAM^{c2p,Bk}.
\]

### MCTformer

\[
CAM^{c2p,Bk}
=
A^{Bk}_{c2p}
\odot
CAM^{patch},
\]

\[
CAM^{final,Bk}
=
A_{p2p}^{all}
CAM^{c2p,Bk}.
\]

不得为了“公平”把两模型公式改成一样；它们各自保持官方实现。

---

## 6.5 Primary segmentation evaluation

复用 Experiment 2 的 raw CAM mIoU pipeline。

主设置：

```text
single-scale 448 crop
GT positive-class gating
per-class CAM normalization与Experiment 2完全一致
background threshold = 0.45
```

报告：

- mIoU；
- foreground precision；
- foreground recall；
- per-class IoU；
- single-label；
- exactly 2 labels；
- 3+ labels；
- object-size strata（若现有 GT 工具支持）。

---

## 6.6 Threshold robustness

固定 0.45 是 primary endpoint。

辅助做统一 threshold sweep：

\[
\delta\in[0.20,0.60],
\quad step=0.01.
\]

对所有 variants 使用相同网格，报告：

- mIoU-vs-threshold curve；
- curve AUC；
- best mIoU；
- best threshold；
- 在 native-best threshold 下各变体结果。

不得为每个 variant 单独精调后只报告最好数字。

---

## 6.7 Attention-level 和 CAM-level联合评价

每个 B0–B5 同时报告：

### Attention

- C-PiM；
- target-vs-BG AUC/AP；
- target-vs-other AUC/AP；
- BG-TailEnrich@10；
- class-pair top10 Jaccard；
- shared-support ownership。

### CAM

- patch CAM不变；
- c2p CAM；
- final CAM；
- stage-wise background introduced/removed；
- raw CAM mIoU。

---

## 6.8 Paired significance

以 image 为 cluster，5,000 次 paired bootstrap。

主比较：

\[
B1-B0
\quad
\text{（L10 vs native last3）}.
\]

其次：

\[
B4-B0.
\]

输出：

```text
delta mIoU
95% CI
delta precision
delta recall
delta target-other AUC
delta class-map Jaccard
```

---

## 6.9 Validation B 判定

### Late layers are practically harmful

若 L10 或 L10–L11：

- raw CAM mIoU显著高于 native last3；
- target-vs-other AUC提高；
- class-map overlap下降；
- recall没有明显损失。

### Late ambiguity is largely compensated

若 attention指标改善明显，但 CAM mIoU近似不变，则 patch CAM 和 \(A_{p2p}\) 已抵消该 ambiguity。

### Native aggregation is beneficial

若 native last3 优于所有 single-layer variants，则最后三层融合虽包含 recoupling，但提供了互补信息或稳定性。

---

# 7. Validation C：Late Inter-Class Mixing Causal Intervention

## 7.1 核心假设

MCTformer+ 在 L10 形成较好的 class-specific routing，但 L11–L12 退化。

候选机制：

\[
\boxed{
\text{late off-diagonal class-to-class value mixing}
}
\]

使不同 class tokens 在后续层共同读取相似 patch support。

---

## 7.2 为什么不采用简单 pre-softmax mask 作为主干预

若只把 class-query→other-class-key logits设为 \(-\infty\) 并重新 softmax：

- class-to-patch group mass会改变；
- attention output scale和组成改变；
- 同层 patch spatial ranking本身不变，只是整体缩放；
- 容易把“去除 class mixing”和“重新分配 attention mass”混为一谈。

因此主干预采用 **mass-preserving self-reroute**。

---

## 7.3 Primary intervention：C2C Self-Reroute

原 attention 对 class query \(c\)：

\[
o_c
=
\sum_{c'}A_{c,c'}v_{c'}
+
\sum_jA_{c,j}v_j.
\]

定义 off-diagonal class mass：

\[
m_c^{off}
=
\sum_{c'\neq c}A_{c,c'}.
\]

干预后：

\[
\tilde A_{c,c'}=0,
\quad c'\neq c,
\]

\[
\tilde A_{c,c}
=
A_{c,c}+m_c^{off},
\]

\[
\tilde A_{c,j}
=
A_{c,j}.
\]

因此：

\[
\sum_k\tilde A_{c,k}=1.
\]

并且：

- class→patch weights完全不变；
- class-group总质量完全不变；
- row sum不变；
- 只把“读取其他 class value”替换为“读取自己的 class value”；
- patch-query rows完全不变。

这是最接近“只删除 inter-class semantic content mixing”的干预。

---

## 7.4 Implementation strategy

在 `models/vit.py` 的 `Attention` 中增加默认关闭、非持久化的 analysis-only intervention，或在 experiment3 中提供等价 wrapper。

推荐接口：

```python
with C2CIntervention(
    model,
    layers=[9],              # 0-based block index
    mode="self_reroute",
):
    outputs = model(...)
```

不要把 intervention config 保存进 checkpoint state_dict。

### Default path

```text
intervention = None
```

时必须与原始模型：

```text
bitwise-equivalent 或 max_abs_diff < 1e-7
```

---

## 7.5 Intervention variants

使用 1-based layer naming。

### C0：Baseline

无干预。

### C1：L12 only

只干预 L12。

这是重要的 structural negative control：

- L12 没有后续 block；
- self-reroute 不改变当层 class→patch weights；
- patch-query rows不变；
- 因此 final patch tokens 和 CAM 理论上应保持不变；
- class logits可能改变。

### C2：L11 only

检验 L11 class mixing 是否导致 L12 routing退化。

### C3：L10 only

检验 L10 输出中的 inter-class mixing 是否导致 L11/L12 recoupling。

### C4：L10–L11

保护进入 L11/L12 的 class semantics。

### C5：L10–L12

完整 late-block intervention。L12 部分主要影响最终 class-token logits，是 C4 的附加 control。

可选但非首轮：

```text
L9–L11
```

仅当 C3/C4 有明显效果后再运行。

---

## 7.6 Secondary intervention controls

只在 primary self-reroute 显示效果后运行。

### Hard mask + renormalization

对 off-diagonal C2C logits设 \(-\infty\)。

### Zero off-diagonal without renormalization

用于检测 output-magnitude confound，不作为主结果。

### Soft self-reroute dose

将比例 \(\lambda\) 的 offdiag mass reroute到自身：

\[
\lambda\in\{0.25,0.5,0.75,1.0\}.
\]

用于 dose-response。

---

## 7.7 Structural invariants

对被干预层必须验证：

### Patch rows unchanged

\[
\tilde A_{p,\cdot}=A_{p,\cdot}.
\]

### Class-to-patch weights unchanged

\[
\tilde A_{c,p}=A_{c,p}.
\]

### Class group mass unchanged

\[
\sum_{c'}\tilde A_{c,c'}
=
\sum_{c'}A_{c,c'}.
\]

### Row sum unchanged

\[
\sum_k\tilde A_{c,k}=1.
\]

### Offdiag zero

\[
\tilde A_{c,c'}=0,\quad c'\ne c.
\]

### Diagonal receives exact mass

\[
\tilde A_{c,c}
=
A_{c,c}
+
\sum_{c'\ne c}A_{c,c'}.
\]

---

## 7.8 L12-only negative-control invariant

C1 应满足：

```text
final patch tokens unchanged
patch-head logits unchanged
all A_c2p spatial maps unchanged
native CAM unchanged
```

容差：

```text
max_abs_diff < 1e-6
```

但 final class tokens / class logits可改变。

若 C1 改变 CAM，说明 implementation 修改了不该修改的路径，必须停止并修复。

---

## 7.9 评价指标

### Representation

- class-token pair cosine；
- axis-removed class-token cosine；
- class-map top10 Jaccard；
- shared support ownership。

### Attention

- L10/L11/L12 C-PiM；
- target-vs-other AUC/AP；
- target-vs-BG AUC/AP；
- class-pair top10 Jaccard；
- conditional BG mass；
- head-wise target-other margin；
- C2C offdiag mass。

### Classification

- class-token mAP；
- patch-head mAP；
- positive-class recall；
- single/2/3+ label strata。

### CAM

- fixed-threshold raw CAM mIoU；
- threshold-sweep curve；
- precision/recall；
- per-class IoU；
- shared-support ownership；
- stage-wise target retention/background removal。

---

## 7.10 Primary endpoints

Validation C 预注册三个 primary endpoints：

1. MCTformer+ L12:
   \[
   AUC_{target-other}
   \]
2. MCTformer+ L12:
   \[
   \text{positive-class-pair top10 Jaccard}
   \]
3. Final raw CAM:
   \[
   mIoU_{\delta=0.45}
   \]

Classification mAP 是 non-inferiority constraint：

\[
\Delta mAP > -0.003
\]

即不下降超过 0.3 percentage points。

---

## 7.11 MCTformer architecture control

首轮以 MCTformer+ 为主。

若 MCTformer+ C3/C4 有明确效果，再对 MCTformer 运行：

```text
C0
C3
C4
```

用于判断该机制是否：

- multi-class-token architecture普遍存在；
- 或主要由 MCTformer+ 训练/表示方式引起。

---

# 8. 统计方法

所有验证统一：

```text
5,000 image-clustered bootstrap repeats
95% percentile CI
same sampled image IDs for paired comparisons
```

## 8.1 Unit of inference

- 单类别指标：image cluster；
- class-pair指标：image cluster；
- 不把 patches 当独立样本；
- 不把同图多个类别当完全独立样本。

## 8.2 Aggregations

报告：

- micro；
- macro-class；
- per-class；
- single-label；
- exactly 2 labels；
- 3+ labels；
- classification-correctness strata。

---

# 9. 输出目录

```text
results/
└── lazy_assignment/
    └── experiment3_three_validations/
        └── <run_id>/
            ├── audit/
            ├── presence_axis/
            │   ├── canonical/
            │   ├── tables/
            │   ├── plots/
            │   └── examples/
            ├── cam_layer_intervention/
            │   ├── cams/
            │   ├── tables/
            │   ├── plots/
            │   └── examples/
            ├── c2c_intervention/
            │   ├── signals/
            │   ├── tables/
            │   ├── plots/
            │   └── examples/
            ├── reports/
            ├── logs/
            ├── exact_commands.sh
            ├── pipeline_metadata.json
            └── pipeline_status.json
```

---

# 10. 推荐执行顺序

## Step 0：Audit + commit hygiene

- 提交 Experiment 1/2 analysis code；
- 创建 tag；
- 建 immutable source manifest。

## Step 1：Validation A smoke test

```text
MCTformer+
50 images
L10–L12
```

验证 decomposition、exact logit identity 和 map metrics。

## Step 2：Validation A full run

两模型、全 VOC val、全 12 层。

先完成报告再继续。

## Step 3：Validation B offline smoke test

从 Experiment 2 signals 生成 20 张 CAM variants，验证 native B0 与原始 CAM完全一致。

## Step 4：Validation B full run

两模型、B0–B5、全 VOC val。

## Step 5：Validation C synthetic/unit tests

不跑数据，确保 self-reroute invariants。

## Step 6：Validation C 50-image smoke test

只跑 MCTformer+：

```text
C0
C1 L12
C2 L11
C3 L10
C4 L10–L11
C5 L10–L12
```

重点验证 L12 negative control。

## Step 7：Validation C full VOC val

若 smoke test通过，运行 MCTformer+ 全集。

## Step 8：MCTformer control

只有 MCTformer+ C3/C4 存在清晰结果时运行。

## Step 9：Independent reports

先分别生成三份报告，再生成总决策。

---

# 11. Go / No-Go 决策矩阵

## Outcome 1：Presence axis解释 raw recoupling，C2C blocking也改善 routing/CAM

支持：

\[
\boxed{
\text{Presence is shared, semantics should remain class-specific.}
}
\]

候选研究方向：

- presence/semantic representation decoupling；
- late class interaction control；
- soft semantic ownership。

## Outcome 2：Presence axis解释 raw recoupling，但 C2C blocking无效

说明：

- Experiment 1 主要是 probe artifact/representation geometry；
- actual routing ambiguity有其他来源；
- 不应围绕 class-token presence axis设计主方法。

## Outcome 3：Presence axis不成立，但 C2C blocking有效

说明：

- 真实问题是 late class-to-class interaction；
- 与 fixed all-ones axis无关；
- 研究主线应转向 inter-class routing / semantic ownership。

## Outcome 4：L10 CAM显著优于 native last3，C2C blocking也有效

提供最完整证据链：

\[
\text{late C2C mixing}
\rightarrow
\text{L11/L12 routing recoupling}
\rightarrow
\text{CAM degradation}.
\]

## Outcome 5：L10 attention更好，但 CAM mIoU无变化

说明 patch CAM 和 \(A_{p2p}\) 补偿了 routing ambiguity；不值得仅以 layer selection 为方法。

## Outcome 6：C2C blocking提高 target-vs-other，但 BG指标变差

说明 foreground competition/decoupling有效，但缺少 background alternative。

这将直接支持未来：

\[
\text{foreground class competition}
+
\text{semantic BG slot}.
\]

## Outcome 7：C2C blocking损害 classification和 CAM

说明 late class-token mixing可能承载有益共现信息；未来方法不能 hard block，需要：

- gated interaction；
- content-aware routing；
- uncertainty-aware ownership。

---

# 12. 必须添加的 Tests

## 12.1 Presence axis exactness

\[
z_c
=
a_c/\sqrt D
\]

误差：

```text
< 1e-6
```

## 12.2 Orthogonality

\[
(c_c^\perp)^\top u=0
\]

误差：

```text
< 1e-6
```

## 12.3 Decomposition reconstruction

\[
c_c=a_cu+c_c^\perp
\]

误差：

```text
< 1e-6
```

## 12.4 Cross-fit leakage prevention

Presence direction fit/eval image IDs完全不重叠。

## 12.5 Native CAM source equivalence

B0 与原生 CAM：

```text
max_abs_diff < 1e-6
```

## 12.6 C2C row invariants

逐项验证第 7.7 节。

## 12.7 L12 negative control

C1 CAM 与 C0 CAM：

```text
max_abs_diff < 1e-6
```

## 12.8 Intervention cleanup

退出 context manager 后，模型重新 forward 与 baseline一致。

## 12.9 Checkpoint/source immutability

前后 hashes 一致。

---

# 13. 最终报告

生成：

```text
reports/
├── VALIDATION_A_PRESENCE_AXIS.md
├── VALIDATION_B_CAM_LAYER_READOUT.md
├── VALIDATION_C_LATE_C2C_CAUSAL.md
├── EXPERIMENT3_COMBINED_REPORT.md
└── NEXT_METHOD_DECISION.md
```

每个结论必须标注：

```text
[Fact]
[Statistical inference]
[Mechanistic interpretation]
[Unsupported]
```

---

# 14. Codex 最终交付物

```text
1. Experiment 3 全部代码
2. tests 与测试日志
3. input audit 与 source hashes
4. presence-axis canonical tables
5. learned shared-presence directions
6. raw / axis-removed / norm / QK comparisons
7. CAM layer-source variants与完整 mIoU表
8. C2C intervention signals和指标
9. paired bootstrap tables
10. class-wise与multi-label strata
11. rule-selected examples
12. 三份分项报告
13. combined report
14. NEXT_METHOD_DECISION.md
15. exact_commands.sh
16. git diff summary
```

大型 per-image artifacts不提交 Git，只提交：

- code；
- tests；
- compact CSV；
- metadata；
- reports；
- selected plots。

---

# 15. 可直接交给 Codex 的任务说明

```text
Read this plan together with the Experiment 1 and Experiment 2 reports before
editing code.

Execute exactly three inference/analysis-only validations:

A. Presence-Axis Decomposition:
   Verify the exact final class-logit direction induced by x_cls.mean(-1),
   decompose class and patch tokens into the all-ones direction and its
   orthogonal residual, and determine how much of the late raw-feature
   class-token/map recoupling disappears after axis removal. Add a cross-fitted
   data-derived shared presence direction for intermediate-layer confirmation.
   Compare raw, class-only removed, patch-only removed, both removed, actual
   pre-attention normalized features, QK energy, and A_c2p.

B. CAM Layer Readout:
   Reuse Experiment 2 signals to construct native-last3, L10-only, L11-only,
   L12-only, L10-L11, and L4-L6 control CAMs while preserving the exact native
   MCTformer/MCTformer+ CAM formulas and the same A_p2p. Evaluate fixed-threshold
   raw CAM mIoU, threshold curves, precision/recall, per-class and multi-label
   strata. Do not describe the best layer as a proposed method.

C. Late C2C Causal Intervention:
   Implement the mass-preserving class-to-class self-reroute intervention.
   For each class-query row, move all off-diagonal class-key attention mass to
   the matching diagonal class key, while leaving class-to-patch weights,
   class-group mass, row sum, and every patch-query row unchanged. Test L12,
   L11, L10, L10-L11, and L10-L12. L12-only must leave patch tokens and CAM
   numerically unchanged and acts as a required negative control.

Use the exact existing checkpoints, deterministic 448 VOC-val pipeline, GT
region definitions, and 5,000 image-clustered paired bootstrap. Treat all
source files as immutable. Do not train a model, add a background/register
token, implement slots, prune heads, or begin a proposed solution.

Stop after producing the three validation reports, the combined report, and
NEXT_METHOD_DECISION.md.
```

---

# 16. 一句话目标

\[
\boxed{
\text{Determine whether late recoupling is caused by a shared presence component,
whether it harms actual CAMs, and whether late inter-class value mixing is its
causal source.}
}
