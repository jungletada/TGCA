# C2P product + P2P affinity pooling：待确认实验大纲

状态：2026-09-14 用户已批准第3节主方案，两组独立匹配训练；第4节备选不执行。
代码核对基于 main / `9d1fd37`，pooling 实现提交 `d7090fe`。
不修改或覆盖已有模型、checkpoints、results。

## 1. 实验问题与已有证据

在已有跨层 C2P 连乘 pooling 上，加入实际 Transformer patch-to-patch
attention，是否能改善分类和 CAM 定位，而不改变原始 CAM 生成公式？

当前固定阈值 .45 的原生多尺度 VOC train CAM mIoU：

| 已完成方法 | mIoU (%) |
|---|---:|
| GWRP | 70.063 |
| C2P last3 mean | 72.040 |
| C2P all mean | 72.084 |
| C2P last3 product | 71.665 |
| C2P all product | 72.315 |

来源：`results/c2p_pooling/20260914-voc-product-s0/comparison.csv`。
这些是单 seed 结果；不能预设连乘普遍更优，也不能预设 affinity 一定改善定位。

## 2. 先明确原生 CAM 与本次训练 pooling 的区别

普通 MCTformer+ 的 `MCTformerPlusCam.get_cam` 实际使用：

```
M_plus = ReLU(original 3x3 patch classifier logits)
a_native = mean_L10_L12(mean_heads(A_c2p))
S = sqrt(a_native * M_plus)                # element-wise multiplication
P = sum_L1_L12(mean_heads(A_p2p))           # [B,N,N]
CAM = P @ S_column                        # matrix multiplication over patches
```

P 的行是 patch query i，列是 patch key j；输出位置 i 收集 sum_j P[i,j] S[j]。
原生 P 是各层求和，不是跨层矩阵连乘，也没有在截取 patch 子矩阵后再次做行归一化。
完整 self-attention 的行归一化不代表截取的 P2P 子矩阵每行仍为 1。

本次建议借用同样来源、同样方向的 P 来传播 pooling 权重，但不直接复制
`ReLU(M)` 和 `sqrt` 到分类分支：那会让 pooled logits 非负，破坏负类别所需的
负 logit 表达能力，并同时引入非线性/尺度变化，不再是只增加 affinity 的对照。

因此以下主方案是“受 CAM affinity 传播启发的训练 pooling”，
不是声称逐步复刻完整 native CAM refinement。

## 3. 建议主方案：传播权重，再聚合原始 logits

统一记：M 为原始 3x3 classifier logits，[B,C,N]；448 输入时 N=784。
每层 head 平均后的 attention 为 a_l，[B,C,N]。
层集合 T 分别为 {10,11,12} 和 {1,...,12}。

### 3.1 保留已有跨层连乘

```
u[c,j] = product_{l in T} a_l[c,j]
w[c,j] = u[c,j] / sum_k u[c,k]
```

数值实现沿用已有 `softmax(sum_l(log(a_l)), dim=patches)`，包括 machine-tiny
数值零处理。不是几何平均，不额外开方或加入温度；每层 heads 仍平均。

### 3.2 新增一次 P2P affinity 传播

```
P = sum_{l=1..12} mean_heads(actual A_p2p_l)
r[c,i] = sum_j P[i,j] * w[c,j]
```

P2P 固定全部 12 层：last3/all 只指 C2P 层范围，避免同时改变两个因素。
对 `[B,C,N]` 的行向量存储，等价张量方向为 `r = w @ P.transpose(-1,-2)`。
不能误写为 `w @ P`；P 通常不对称。
不额外对 P 做 row-softmax、row-normalization、对称化、加自环或多次传播。

### 3.3 归一化传播后的权重并聚合

```
w_aff = r / r.sum(dim=-1, keepdim=True).clamp_min(1e-8)
z_patch = (w_aff * M).sum(dim=-1)           # [B,C]
```

输出仍是对原始正负 logits 的加权和，不对 M 做 ReLU、sigmoid、sqrt 或 CAM min-max。
P 的统一标量因子会在最终归一化中抵消，但逐行归一化一般不会抵消，不能混为一谈。
由于使用原生未条件化的 P2P 子矩阵，传播也携带 patch-query 的 patch-group mass；
不把 P 擅自描述为纯几何相似度或行随机矩阵。

梯度同时经过 M、C2P、P2P；不用带 no_grad 的 CAM helper，不 detach。
它改变训练读出及其梯度，不修改 Transformer attention 本身。

## 4. 不应混淆的另一种位置安排（仅供用户选择，不自动执行）

若用户实际希望“先 w × M，再 P2P”，则公式不同：

```
E = w * M
E_refined = P @ E_column
```

它更接近原生 CAM 的先乘后传播顺序，但不能简单认为后接全局求和/平均就能
保留完整的空间传播效果。令 s_j = sum_i P[i,j]：

```
sum_i E_refined[i] = sum_j s_j * w[j] * M[j]
```

也就是说，纯线性传播后全局汇聚只通过 P 的列和 s_j 进入最终分类值。
即使用 `sum_i(P @ E)_i / sum_i(P @ w)_i` 消除整体尺度，仍等价于
`normalize(s * w)` 对 M 加权，不等价于主方案的 `normalize(P @ w)`。
如果 P 恰好列和为 1，此归一化版本精确回到原 product pooling。
如果只做 GAP，还会额外改变原 weighted-sum logits 的尺度。

这不是说该备选毫无作用，而是其效果不能宣称来自完整空间关系的传播；
应命名为 column-mass-reweighted pooling，并单独确定最终聚合/归一化。
不同时运行主方案和此备选，不自动扩展成四组训练。

## 5. 最小实验矩阵（主方案获批时）

| 状态 | 名称 | C2P 范围 | C2P 跨层聚合 | 训练 pooling 中的 P2P |
|---|---|---|---|---|
| 复用 | GWRP | — | — | 无 |
| 复用 | last3-mean | L10–L12 | 平均 | 无 |
| 复用 | all-mean | L1–L12 | 平均 | 无 |
| 复用 | last3-product | L10–L12 | 连乘 | 无 |
| 复用 | all-product | L1–L12 | 连乘 | 无 |
| 待批准 | last3-product-affinity | L10–L12 | 连乘 | 全12层，传播权重一次 |
| 待批准 | all-product-affinity | L1–L12 | 连乘 | 全12层，传播权重一次 |

主比较是每组 product-affinity 对其对应 product；mean/GWRP 作为已有参考。
这些“无 P2P”仅指训练 pooling；所有旧方法的最终 CAM 都已有原生 P2P refinement。
最终 CAM 生成不再新增第二次 affinity 传播，仍采用原生公式。

## 6. 固定内容与执行顺序（须先批准）

- class_token_init、所有初始化模块和参数保持现状；当前匹配配置为 baseline。
- 原始 L12 class-token mean 分支、所有层 raw class-token CCT、3x3 patch head、
  原 patch multilabel loss 均不变；不引入新参数或辅助 loss。
- 原生 CAM 仍使用 last3-mean C2P、ReLU(M)、sqrt、全层 P2P；只有训练后权重可能不同。
- 两组均从相同 DeiT-S 初始化独立训练，不给旧 product checkpoint 切换开关冒充新训练。
- VOC seed0、45 epochs、448、batch32，原 AdamW/cosine/LR/augmentation/数据划分完全匹配。
- 不加新的归一化变体、BG tokens、额外 losses、attention interventions 或参数扫描。
- 获批后：必要测试 -> 两组 smoke -> last3 训练与评估 -> all 训练与评估 -> 紧凑报告。

## 7. 最小验证与评估

实现前后必要测试：shape [B,C]、空间权重和为 1、P=I 回到原 product pooling、
常数正 P 得到 GAP、非对称 P 确认传播方向、M/C2P/P2P 梯度、AMP 有限性、
禁用 affinity 时数值兼容、checkpoint 配置/gating 一致、同权重 native CAM 不变。
先切片/平均再进行小规模矩阵运算，避免训练时为本读出再次堆叠所有完整 NxN heads。

报告沿用现有指标：两头 macro-class mAP、两头验证 loss、固定 .45 raw CAM mIoU、
同一0:.01:.59阈值曲线与最佳点、FG precision/recall。分类 val1449/FP32单尺度448；
CAM train1464/原生三尺度，无CRF或后续segmentation。最佳阈值只作诊断。

附最小读出检查：比较传播前后权重熵、top-1质量，以及多标签正类别对的
top-10% Jaccard，检查是适度扩展还是趋于共同/均匀响应；样例在看结果前固定。
若报告配对置信区间，使用5000次图像级配对 bootstrap，不把patch/类别对当独立样本。
无GT语义归属分析时，不声称背景泄漏得到改善；单seed不声称训练稳健性。

输出精确命令、配置、代码与checkpoint SHA、环境/测试记录、comparison.csv 和紧凑报告。

## 8. 本次等待的用户决定

推荐批准主方案：传播 C2P product 权重，再加权原始 logits，只新增两组训练。
若希望严格先乘 M 再传播，请先选择第4节备选并确认读出，而不是混用两种公式。
本文件只是大纲；此次未实施上述模型/评估代码，也未启动任何新实验。
