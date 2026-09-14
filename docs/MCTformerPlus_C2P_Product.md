# C2P 跨层逐元素连乘：两组匹配实验

按用户要求，仅将 patch pooling 的跨层 arithmetic mean 改为 element-wise product。
层范围分别保持 last3 (L10–L12) 和 all (L1–L12)。每层 heads 仍求平均，不跨 heads 连乘。

```
a_l = mean_heads(actual Transformer A_c2p_l)   # [B,C,N]
u = product_layers(a_l)                        # same spatial indices
w = u / sum_patches(u)
z_patch = sum_patches(w * raw_3x3_classifier_logits)
```

这是 attention 之间的逐点乘积，不是 attention rollout 矩阵乘法。
原生 CAM 的乘法是 attention 与 ReLU 后 patch CAM 相乘；不宣称原生 CAM 已做跨层连乘。
本次 CAM 公式仍为 last3-mean C2P × ReLU(M) -> sqrt -> all-layer P2P。
分类 gating 使用 product pooled logits。初始化、class branch、CCT、patch classifier、
loss、训练配置、CAM 评估流程均不改动；没有新参数、loss 或其他方法。

## 数值实现

12 个小概率直接连乘可能在 FP32/AMP 下溢；也不能将极小乘积的分母硬截为 1e-8，
否则空间权重和不再为 1。使用归一化乘积的等价形式：

```
w = softmax(sum_layers(log(a_l)), dim=patches)
```

沿用 FP32 累积（双精度测试保留 FP64）。log 仅以累计 dtype 的 finfo.tiny 处理数值零；
不是可调 eps。对于高于 machine tiny 的正 attention，数学上等于直接连乘后归一化。
不取几何平均，不除以层数，不开方，不加 temperature，不 detach。
如某些 heads 在 AMP 下舍入为零，使用转换到 FP32 后的 head mean；数值零 floor 的
梯度遵循 clamp，属于明确记录的有限精度处理，不是额外实验分支。

CLI: `--patch-pooling c2p --c2p-pooling-layers last3|all --c2p-pooling-reduction product`。
新增 reduction 默认 mean，旧模型/命令保持原样；checkpoint 缺失该字段解释为 mean。
模型构建、训练、分类验证和 CAM checkpoint/gating 同步该配置。

## 队列与复现

仅训练两组：last3-product -> all-product。两组烟测都通过后，顺序完成 full train + eval。
每组独立从相同 DeiT-S 初始化开始：VOC、seed0、45epochs、448、batch32、
原 AdamW/cosine/名义 LR5e-4/min1e-5/augmentation/CCT，class_token_init=baseline。
不能只给旧训练 checkpoint 切换开关，也不重训已有 GWRP、last3-mean、all-mean。

分类：VOC val1449，FP32单尺度448；原生三尺度CAM：VOC train1464；固定阈值.45，
同一诊断阈值网格0:.01:.59，无 CRF/后续 segmentation。单 seed，不作稳健性保证。

输出：`results/c2p_pooling/20260914-voc-product-s0`。
Runner: `python -m experiments.ablations.run_c2p_product --output results/c2p_pooling/20260914-voc-product-s0`。
每组完成后更新 comparison.csv 和 C2P_PRODUCT_REPORT.md；只在两组均完成后写 QUEUE_COMPLETE。
保存 exact commands、config、环境、测试日志、代码 SHA、checkpoint/source SHA256。
保留 clean-worktree/no-overwrite gate，不修改任何旧结果和 checkpoint。
