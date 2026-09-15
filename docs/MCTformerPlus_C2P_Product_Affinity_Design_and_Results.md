# MCTformer+ C2P Product / Affinity：已完成实验设计与结果对比

整理日期：2026-09-15。仓库：`/home/peng/code/TGCA`，执行主机 LHR，环境 `tgca-repro`。

本文仅汇总已完成的 VOC seed-0 实验，不启动新实验、不修改模型或已有结果。COCO all-product 尚在训练，不纳入本表。数值以本地评估 JSON 和 [七组汇总 CSV](../results/c2p_pooling/20260914-voc-product-affinity-s0/comparison.csv) 为准；表内百分数保留三位小数，差值单位为百分点（pp）。

## 1. 核心结论

- **all-product 的固定阈值 raw CAM mIoU 最高：72.315%**，比 GWRP 高 2.252 pp，比 all-mean 高 0.231 pp；但其 class / patch macro mAP 分别低于 all-mean 0.677 / 0.625 pp。
- last3-product 为 71.665%，比 last3-mean 低 0.374 pp。因此，“连乘优于平均”并不是两种层范围均成立的结论。
- 加入训练 pooling 的 P2P affinity 后，last3 / all 的固定阈值 mIoU 分别降至 70.531% / 69.406%，相对各自 product 对照下降 1.134 / 2.909 pp。分类 mAP 没有同步下降。
- affinity 权重诊断显示空间权重明显变平，正类对 top-10% 支持集重叠反而减少。这是同一 affinity checkpoint 内传播前后的读出观察，不能直接证明 CAM 下降的原因，也不能据此判断背景泄漏。
- 所有主结果仅来自一个匹配的 seed-0 训练；没有跨 seed 置信区间，不应把 0.231 pp 的差距描述为稳定收益。

## 2. 实验保持不变的部分

本组宿主是普通 **MCTformer+ Small**，不是此前的 PatchFinalLN 或 LaST 变体：12 blocks、6 heads、embedding 384、20 个 class tokens、patch size 16；448 输入对应 28×28=784 个 patch。

保留 class-first 拼接和原始联合 self-attention；`class_token_init=baseline`，使用同一 DeiT class token 重复初始化各类。`final_norm=False`、`patch_final_norm=False`，不切换 CWP/residual CWP，不增加参数或辅助 loss。

class 分支仍为 raw L12 class tokens → `mean(-1)` → 原分类 loss；CCT 仍读取全部 12 层 raw post-block class tokens。patch 分支仍以 raw L12 patch tokens 经原有 3×3 Conv 得到原始 logits，再计算相同 multilabel soft-margin loss。唯一研究变量是 patch logits 的空间 pooling。

## 3. Pooling 的精确定义

忽略 batch 下标，令原 3×3 classifier 输出为 `M[c,j]`，其中 `j` 为原空间顺序的 patch 位置。`A_l` 是 `forward_features` 返回的实际 Transformer self-attention，而非最终 token 点积、初始化模块 attention 或 no-grad CAM helper 的输出。

定义每层 head 平均后的 class-to-patch 响应：

```text
a_l[c,j] = mean_h A_l[h, class_c, patch_j]
T_last3 = {10, 11, 12}
T_all   = {1, ..., 12}
```

### 3.1 对照：GWRP 与 C2P mean

GWRP 对每类原始 logits 降序排序，以原几何衰减系数 `d=0.996` 加权：

```text
z_gwrp[c] = sum_r d^(r-1) M_sorted[c,r] / sum_r d^(r-1)
```

C2P mean 保留空间顺序，先平均所选层及 heads，再在 patch 维归一化：

```text
a[c,j] = mean_{l in T} a_l[c,j]
w[c,j] = a[c,j] / clamp_min(sum_k a[c,k], 1e-8)
z_patch[c] = sum_j w[c,j] M[c,j]
```

这里平均的是实际 attention 响应，不是先将每层分别条件归一化后再平均。

### 3.2 last3-product / all-product

只将层间算术平均替换为逐元素连乘；heads 仍然平均：

```text
u[c,j] = product_{l in T} a_l[c,j]
w[c,j] = u[c,j] / sum_k u[c,k]
z_patch[c] = sum_j w[c,j] M[c,j]
```

实现采用等价的 log-space 计算，防止 12 层小概率连乘下溢：

```python
log_product = a_layers.clamp_min(torch.finfo(a_layers.dtype).tiny).log().sum(dim=0)
w = log_product.softmax(dim=-1)
```

FP16/BF16 输入先转 FP32 累积。使用 **sum(log)**，不是 mean(log)，没有开几何根、温度调节或额外 sqrt。零值按 dtype 的最小正 normal 值截断。

可理解为加强跨层共同高响应位置的相对权重，但“共同高响应”不自动等同于语义目标。两组都不排序、不使用 GWRP 衰减，`M` 不做 ReLU、sigmoid 或 CAM normalization。attention 和 `M` 均保留梯度。

### 3.3 两组 product-affinity

先构造上述 product 权重，再用所有 12 层的 patch-to-patch affinity 传播一次，最后归一化并聚合原始 `M`：

```text
P[i,j] = sum_{l=1..12} mean_h A_l[h, patch_i, patch_j]
r[c,i] = sum_j P[i,j] w[c,j]
w_aff[c,i] = r[c,i] / clamp_min(sum_k r[c,k], 1e-8)
z_patch[c] = sum_i w_aff[c,i] M[c,i]
```

`P[i,j]` 表示 query i 从 key j 接收信息。采用行向量表示时为 `r = w @ P.T`，**不是 `w @ P`**；P 通常不对称。last3-product-affinity 的 C2P 用最后三层，但其 P2P 与 all-product-affinity 一样，均用全部 12 层。

不对 P 再做 row-softmax、条件归一化、对称化或重复传播；原联合 attention 截取 patch keys 后，patch-group mass 仍保留其中。实现逐层切片/head 平均后累加，传播在关闭 autocast 的 FP32 中进行。梯度通过 C2P、P2P 和原始 classifier logits 回传。

这是“传播 pooling 权重后读取 M”，不是“先计算 w×M，再传播该乘积并全局平均”。后者不是本次执行的变体。

| 方法 | C2P 层范围 | 层间融合 | 训练 pooling 中 P2P |
| --- | --- | --- | --- |
| GWRP | 不使用 | 不使用 | 不使用 |
| last3-mean | L10–L12 | 算术平均 | 不使用 |
| all-mean | L1–L12 | 算术平均 | 不使用 |
| last3-product | L10–L12 | 逐元素连乘 | 不使用 |
| all-product | L1–L12 | 逐元素连乘 | 不使用 |
| last3-product-affinity | L10–L12 | 逐元素连乘 | 全部 12 层，传播一次 |
| all-product-affinity | L1–L12 | 逐元素连乘 | 全部 12 层，传播一次 |

### 3.4 所有实验的原生 CAM 公式均未改动

```text
a_native = mean_{l=10..12} mean_h A_c2p_l
S[c,j] = sqrt(ReLU(M[c,j]) * a_native[c,j])
CAM[c,i] = sum_j P[i,j] S[c,j]
```

随后沿用原有 resize、多尺度/翻转融合、归一化及标签 gating。`forward_with_label` 的 patch 分类 gating 使用对应的实际 pooled logits。

所以 **“训练 pooling 没有 affinity”不代表最终 CAM 没有 P2P refinement**；七种方法生成 CAM 时本来都保留原生 P2P。product/affinity 仅改变训练 patch readout，未把 CAM 的 ReLU/sqrt 搬入分类 logits。权重相同的 CAM 公式一致，不表示不同训练所得 checkpoint 的 CAM 数值相同。

## 4. 匹配训练与评估协议

| 项目 | 共同设置 |
| --- | --- |
| 数据 | VOC train_aug 10,582 张训练；val 1,449 张分类验证；train 1,464 张 raw CAM 评估 |
| 训练 | seed=0；45 epochs；448 输入；有效 batch=32；单进程，无梯度累积 |
| 实际更新 | drop_last；每 epoch 使用 10,560 样本、330 次更新 |
| 优化 | AdamW；weight decay=0.05；epsilon=1e-8；cosine；warmup=5 epochs |
| 学习率 | CLI nominal LR=5e-4；按 batch/512 缩放后的 optimizer LR=3.125e-5；min LR=1e-5 |
| 初始化 | 同一个 DeiT-S pretrained checkpoint；每个新变体重新训练，不是旧模型直接切换 pooling |
| 其他 | 原有 augmentation、CCT、分类/patch loss、loss weights、checkpoint policy 不变 |
| checkpoint | 统一 final，即完成 45 epochs 的 epoch=44 checkpoint，不按最佳分类指标挑选 |
| 分类复评 | 冻结 FP32，单尺度 448、无翻转；短边 resize 512（bicubic）后 center crop 448；macro-class AP |
| CAM 复评 | 原生 scales=1.0,0.75,1.25 + flip；图像正标签 gating；无 CRF / 下游 segmentation |
| CAM 指标 | dataset-global confusion；21 类含背景 mIoU；固定阈值 0.45；同一 0.00–0.59、步长 0.01 曲线 |

本表分类 mAP 是逐语义类在整个 val 集上计算 AP，再对 20 类求宏平均，**不是训练日志中的 mean-image AP**。分类与 CAM 使用不同 split，不要将本文 CAM 数值写为 VOC val mIoU。

下表 foreground precision/recall 为固定阈值下的 **semantic foreground** 指标，要求语义类预测正确，不是仅区分前景/背景的 binary 指标。best-threshold 数值仅用于共同阈值曲线的诊断，主要比较仍使用固定 0.45。

## 5. 完整结果对比

### 5.1 分类验证（VOC val，1,449 张）

| 方法 | Class macro mAP (%) | Patch macro mAP (%) | Class val loss | Patch val loss |
| --- | ---: | ---: | ---: | ---: |
| GWRP | 92.906 | 93.258 | 0.050633 | 0.049780 |
| last3-mean | 93.079 | 93.382 | 0.049564 | 0.048957 |
| all-mean | 93.119 | 93.489 | 0.049246 | 0.048168 |
| last3-product | 93.221 | 93.322 | 0.048733 | 0.049447 |
| all-product | 92.442 | 92.864 | 0.052682 | 0.050362 |
| last3-product-affinity | 93.247 | 93.359 | 0.049439 | 0.049337 |
| all-product-affinity | 93.114 | 93.169 | 0.049628 | 0.050013 |

### 5.2 原生 raw CAM（VOC train，1,464 张）

| 方法 | mIoU@0.45 (%) | Best mIoU (%) | Best threshold | FG precision@0.45 (%) | FG recall@0.45 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GWRP | 70.063 | 70.340 | 0.48 | 80.735 | 85.817 |
| last3-mean | 72.040 | 72.165 | 0.43 | 84.125 | 83.141 |
| all-mean | 72.084 | 72.102 | 0.44 | 83.946 | 83.563 |
| last3-product | 71.665 | 71.670 | 0.44 | 82.880 | 83.550 |
| all-product | 72.315 | 72.315 | 0.45 | 83.318 | 84.654 |
| last3-product-affinity | 70.531 | 70.531 | 0.45 | 81.903 | 84.034 |
| all-product-affinity | 69.406 | 69.700 | 0.48 | 79.835 | 85.655 |

### 5.3 单变量差值

差值为“前者 − 对应对照”，均由未舍入原数值计算。

| 比较 | Δ Class mAP (pp) | Δ Patch mAP (pp) | Δ mIoU@0.45 (pp) | Δ Best mIoU (pp) |
| --- | ---: | ---: | ---: | ---: |
| last3-product − last3-mean | +0.142 | −0.060 | −0.374 | −0.495 |
| all-product − all-mean | −0.677 | −0.625 | +0.231 | +0.213 |
| last3-product-affinity − last3-product | +0.026 | +0.037 | −1.134 | −1.139 |
| all-product-affinity − all-product | +0.672 | +0.305 | −2.909 | −2.615 |

all-product 相对 GWRP 的 fixed / best mIoU 分别高 2.252 / 1.976 pp。all-product-affinity 则比 GWRP 的固定 mIoU 低 0.657 pp。

affinity 的变化不是仅靠换阈值就消失：两组 best mIoU 相对 product 仍下降。固定阈值下，last3 affinity 的 precision 下降 0.978 pp、recall 上升 0.484 pp；all affinity 的 precision 下降 3.483 pp、recall 上升 1.000 pp。当前数据呈现精度与召回的取舍，以及分类表现与定位表现并不同步。

## 6. 已有 affinity 空间权重诊断

该诊断在每个**已经训练好的 affinity checkpoint 内**，比较同一图像、同一正类的 product 权重 `w` 与传播后的 `w_aff`。不是旧 product checkpoint 对新 affinity checkpoint 的比较，也不是训练前后对比。

使用 VOC val 1,449 张、FP32、TF32 off、匹配的 448 transform；不读取 semantic GT。先对每张图的正类或正类对求平均，再等权平均图像。归一化熵为 `H(w)/log(784)`；top-1 mass 为最大单位置权重；正类对 Jaccard 使用每类 top-10% 位置，即 `ceil(0.1×784)=79` 个 patch，不是 top-10 个 patch。

| 模型 / 指标 | 传播前 | 传播后 | Δ（后−前）及 95% paired image bootstrap CI |
| --- | ---: | ---: | --- |
| last3-product-affinity / 归一化熵 | 0.6020 | 0.9776 | +0.3756 [0.3695, 0.3819] |
| last3-product-affinity / top-1 mass | 0.1341 | 0.0066 | −0.1276 [−0.1333, −0.1220] |
| last3-product-affinity / 正类对 top-10% Jaccard | 0.2763 | 0.1012 | −0.1751 [−0.1897, −0.1616] |
| all-product-affinity / 归一化熵 | 0.2938 | 0.9719 | +0.6781 [0.6724, 0.6838] |
| all-product-affinity / top-1 mass | 0.4651 | 0.0129 | −0.4522 [−0.4623, −0.4420] |
| all-product-affinity / 正类对 top-10% Jaccard | 0.2858 | 0.1238 | −0.1619 [−0.1779, −0.1456] |

熵/top-1 使用 `all` stratum（1,449 图），Jaccard 使用 `multi` stratum（522 张多标签图）。CI 来自 5,000 次图像级配对 bootstrap，seed=20260914；同一图像的 patches/正类对不视为独立观测，且 CI 不包含训练 seed 不确定性。分类/CAM 主结果表没有该 bootstrap CI。

可以支持的观察是：affinity 后权重更分散，跨正类高权重支持集的重叠下降。不能直接说“所有正类越来越选择同一 patch”；也不能把重叠减少解释为定位一定变好。传播后最大权重仍高于均匀分布的 `1/784≈0.001276`，因此不是严格退化为 GAP。

“权重变平可能削弱定位选择性”只能作为解释假设；本诊断没有 GT 区域归属，不能据此确认背景泄漏、语义混淆或 CAM 下降的因果机制。

诊断紧凑文件与已有示例热力图：

- [last3 affinity summary.csv](../results/c2p_pooling/20260914-voc-product-affinity-s0/last3_product_affinity/weight_diagnostics/summary.csv)、[诊断目录](../results/c2p_pooling/20260914-voc-product-affinity-s0/last3_product_affinity/weight_diagnostics/)。
- [all affinity summary.csv](../results/c2p_pooling/20260914-voc-product-affinity-s0/all_product_affinity/weight_diagnostics/summary.csv)、[诊断目录](../results/c2p_pooling/20260914-voc-product-affinity-s0/all_product_affinity/weight_diagnostics/)。

## 7. 复现记录与结果路径

实现位于 [models/mctformer_plus.py](../models/mctformer_plus.py)，关键函数为 `gwrp`、`c2p_spatial_weights`、`c2p_affinity_weights`、`c2p_pool`。产品实验代码固定于 `d7090feb63c433e450eed6604a7b62f3c011cdb8`；affinity 固定于 `04aa6b870f14a807c5ffdb37c273d218cd5b9414`。本次整理时 HEAD 为 `ce45181`，不要把整理时 SHA 当成历史训练 SHA。

历史 runner 命令来自各自 manifest（记录用途；本次没有重新执行）：

```bash
/home/peng/anaconda3/envs/tgca-repro/bin/python -m experiments.ablations.run_c2p_product --output results/c2p_pooling/20260914-voc-product-s0
/home/peng/anaconda3/envs/tgca-repro/bin/python -m experiments.ablations.run_c2p_affinity --output results/c2p_pooling/20260914-voc-product-affinity-s0
```

pooling 参数对应关系如下；没有显式传入的字段使用当时版本的默认值，完整实际命令以每组 `commands.sh` 为准。

```text
last3-product:          --patch-pooling c2p --c2p-pooling-layers last3 --c2p-pooling-reduction product
all-product:            --patch-pooling c2p --c2p-pooling-layers all   --c2p-pooling-reduction product
last3-product-affinity: --patch-pooling c2p --c2p-pooling-layers last3 --c2p-pooling-reduction product --c2p-pooling-affinity
all-product-affinity:   --patch-pooling c2p --c2p-pooling-layers all   --c2p-pooling-reduction product --c2p-pooling-affinity
```

所有相对路径均以仓库根目录为基准：

| 方法 | 训练结果目录 | 分类复评目录 |
| --- | --- | --- |
| GWRP | `results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12` | `results/c2p_pooling/20260914-voc-s0/baseline_classification` |
| last3-mean | `results/c2p_pooling/20260914-voc-s0/VOC12` | `results/c2p_pooling/20260914-voc-s0/c2p_classification` |
| all-mean | `results/c2p_pooling/20260914-voc-all-layers-s0/VOC12` | `results/c2p_pooling/20260914-voc-all-layers-s0/c2p_classification` |
| last3-product | `results/c2p_pooling/20260914-voc-product-s0/last3_product` | 该目录下 `classification/` |
| all-product | `results/c2p_pooling/20260914-voc-product-s0/all_product` | 该目录下 `classification/` |
| last3-product-affinity | `results/c2p_pooling/20260914-voc-product-affinity-s0/last3_product_affinity` | 该目录下 `classification/` |
| all-product-affinity | `results/c2p_pooling/20260914-voc-product-affinity-s0/all_product_affinity` | 该目录下 `classification/` |

各训练目录下 `raw_cam/metrics.json` 与 `raw_cam/threshold_curve.csv` 保存 CAM 评估；四个新变体目录还保存 `commands.sh`、`optimizer_spec.json`、`model_spec.json`、`pretrained_load_report.json`、`checkpoint_sha256.txt`、`VARIANT_COMPLETE`。根目录 manifest 保存精确 Git SHA、数据列表哈希、源文件实验前后哈希；环境清单和测试日志也位于各实验根目录。

已有紧凑汇总：

- [Product 报告](../results/c2p_pooling/20260914-voc-product-s0/C2P_PRODUCT_REPORT.md)、[manifest](../results/c2p_pooling/20260914-voc-product-s0/manifest.json)。
- [Affinity 报告](../results/c2p_pooling/20260914-voc-product-affinity-s0/C2P_AFFINITY_REPORT.md)、[manifest](../results/c2p_pooling/20260914-voc-product-affinity-s0/manifest.json)、[七组对比 CSV](../results/c2p_pooling/20260914-voc-product-affinity-s0/comparison.csv)。

统一 pretrained 文件：`/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth`，SHA256：

```text
cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f
```

四个新变体均使用其目录中的 `mctformerplus_final.pth`，SHA256：

```text
last3-product          b2f125afc9b750f04979ab1065f9e712c7979d667d3fe79ec79c18abcb368fd2
all-product            ce7bde924fe6a1780151db6e4541eb8937b5f595078483cbbae222ba14f9ef0b
last3-product-affinity  0cdb5a93faf201b2b0b5887f23148f3ffcdacb7891218a2ca85b426c209f4847
all-product-affinity    431a0574282fbbfb3f611de3a05215881cb7a9abe29895f07e107ce48d8374e2
```

## 8. 完成状态、测试与数据完整性

两组 product 和两组 affinity 均有 `VARIANT_COMPLETE`，各队列均有 `QUEUE_COMPLETE`。历史测试记录为 product **59 passed**、affinity **66 passed**，详见 [product tests.log](../results/c2p_pooling/20260914-voc-product-s0/tests.log) 与 [affinity tests.log](../results/c2p_pooling/20260914-voc-product-affinity-s0/tests.log)。这是历史实验的测试状态，本次文档整理未重复训练或运行 GPU 测试。

本次只读检查确认四个新变体分类评估为 1,449 图、CAM 为 1,464 图，九项分类/CAM 汇总字段逐项与原 JSON 一致；两个 manifest 的 before/after 源哈希一致。对其记录及新 checkpoint 合并去重后的 **34 个现存文件**重新计算 SHA256，全部与记录一致。

此前按用户授权清理了 smoke 目录和逐图 CAM 等大中间文件；非 smoke checkpoints、最终数值、阈值曲线和上述紧凑诊断仍保留。`metrics.json` 中的历史 `cam_dir` 不代表逐图 NPY 现在仍存在；不能声称当前保留了完整逐图 CAM。清理记录见 [summary_after.json](../results/cleanup/20260915-keep-checkpoints-final/summary_after.json)。本文可从保留的 JSON/CSV 复核，但若要从逐图 CAM 重新评估，需另行重新生成，本文没有执行该操作。

本次仅新增本总结文档，保留工作区已有文档删除，不恢复旧文档、不改动结果/checkpoint、不干扰正在运行的 COCO 任务。
