# Experiment 2：Semantic Ownership Across Feature, Attention, and CAM  
## MCTformer / MCTformer+ 诊断实验执行计划（Codex 版）

> **目标仓库：** https://github.com/jungletada/TGCA  
> **运行位置：** LHR，`~/code/TGCA`  
> **环境：** 复用 Experiment 1 的 `tgca-repro` 环境  
> **模型：** 已训练完成的 PASCAL VOC 2012 MCTformer 与 MCTformer+ checkpoints  
> **上游结果：**
>
> - MCTformer Experiment 1：
>   `/home/peng/code/TGCA/results/lazy_assignment/experiment1_class_patch_score/mctformer/20260902-mctformerv2-exp1-voc-val-full-6aca9bc`
> - MCTformer+ Experiment 1：
>   `/home/peng/code/TGCA/results/lazy_assignment/experiment1_class_patch_score/mctformer_plus/20260902-mctformerplus-exp1-voc-val-full-fec86b7`
> - Paired Experiment 1 analysis：
>   `/home/peng/code/TGCA/results/lazy_assignment/experiment1_analysis/20260902-mctformer-paired-full-f05d15b`
>
> 实际执行时应从上游 `run_metadata.json` 自动读取 source roots、checkpoint SHA256 和 transform metadata；路径变化时通过 CLI 覆盖，不在代码里硬编码。
>
> **本阶段性质：** evaluation/diagnosis only。  
> **禁止：** 重新训练、修改 attention、加入 BG/register token、实现新 loss、P2C blocking、LaST selective aggregation、语义槽竞争或任何 proposed method。

---

# 1. Experiment 2 的核心目标

Experiment 1 已经发现：

1. MCTformer+ 在 **L9→L10** 出现显著 late-stage representation transition；
2. MCTformer+ 的 class-specific score maps 在 L4–L6 高度分离；
3. 从 L9 开始，不同正类别 maps 的高分 patch support 快速重新耦合；
4. MCTformer+ 的 class-map top-10% Jaccard：
   \[
   0.1988\ (L9)
   \rightarrow
   0.4048\ (L10)
   \rightarrow
   0.5084\ (L11)
   \rightarrow
   0.5458\ (L12);
   \]
5. L11→L12 的 patch ranking Spearman 已达到：
   \[
   0.9879,
   \]
   表明最后一层主要继续放大已经锁定的 spatial support；
6. 但是 Experiment 1 没有加载 segmentation GT，也没有分析 attention 或 CAM，因此不能判断 shared high-score patches 的真实语义。

Experiment 2 要把三个层面的量放在同一坐标系中：

\[
\boxed{
S^{feat}_{c,j}
=
\cos(c_c,p_j)
}
\]

表示 **representation alignment**；

\[
\boxed{
S^{attn}_{c,j}
=
A_{c2p}(c,j)
}
\]

表示 **attention routing**；

\[
\boxed{
S^{cam}_{c,j}
=
CAM_c(j)
}
\]

表示 **最终 localization**。

并利用 PASCAL VOC semantic GT 回答：

> 从 feature similarity、到 class-to-patch attention、再到 CAM，哪些 target、other-foreground 或 background patches 被保留、过滤、引入或放大？

---

# 2. Research Questions

## RQ1：Class-specific feature score 的高分 patches 属于谁？

对于正类别 \(c\)，将 patch 划分为：

\[
\Omega_c
\quad\text{target class},
\]

\[
\Omega_{\mathrm{other}}
\quad\text{other foreground classes},
\]

\[
\Omega_{\mathrm{bg}}
\quad\text{background},
\]

\[
\Omega_{\mathrm{mixed/void}}
\quad\text{ambiguous or void}.
\]

检查每层：

\[
S^{feat,(l)}_{c,j}
\]

在上述区域中的分布、排序和 high-score tail composition。

---

## RQ2：MCTformer+ 后三层重新共享的 high-score patches 属于谁？

对 multi-label image 中两个正类别：

\[
(c_a,c_b)
\]

计算各自 top-\(k\) patch sets：

\[
T_k(c_a),\quad T_k(c_b),
\]

共享 support：

\[
T_k^{shared}
=
T_k(c_a)\cap T_k(c_b).
\]

用 GT 判断 shared patches 属于：

- class \(c_a\)；
- class \(c_b\)；
- another foreground class；
- background；
- mixed/void。

这是 Experiment 2 的**第一优先级问题**。

---

## RQ3：Feature-level high-score patches 是否进入 \(A_{c2p}\)？

比较：

\[
S^{feat,(l)}_{c,\cdot}
\]

和：

\[
A^{(l)}_{c2p}(c,\cdot).
\]

检查：

- Spearman correlation；
- top-\(k\) overlap；
- target / other-FG / BG composition；
- feature-level background tail 是否被 attention 保留或过滤；
- attention 是否引入 feature score 中原本不高的背景 patches。

---

## RQ4：MCTformer+ 使用最后三层 \(A_{c2p}\) 是否放大后层 shared support？

MCTformer+ 的官方 CAM 使用最后三层 class-to-patch attention：

\[
A_{c2p}^{official}
=
\frac{1}{3}
\sum_{l=10}^{12}
A_{c2p}^{(l)}
\]

（1-based layer indexing）。

由于 Experiment 1 已发现 L10–L12 是 class-map recoupling 最强的阶段，必须比较：

- \(A_{c2p}^{(10)}\)；
- \(A_{c2p}^{(11)}\)；
- \(A_{c2p}^{(12)}\)；
- official last-three aggregation；
- mid-layer control：
  \[
  \frac{1}{3}(A^{(4)}+A^{(5)}+A^{(6)}).
  \]

该 control 只用于 diagnosis，不改变官方方法，也不作为新的推理配置宣称性能。

---

## RQ5：错误是在 patch CAM、class-attention filtering，还是 \(A_{p2p}\) propagation 阶段产生？

将官方 CAM 分成三个阶段：

\[
CAM^{patch}
\]

由 patch classification head 直接生成；

\[
CAM^{c2p}
\]

由 \(CAM^{patch}\) 与 official \(A_{c2p}\) 融合；

\[
CAM^{final}
\]

再经过 \(A_{p2p}\) propagation。

逐阶段检查 target / other-FG / BG 区域：

- 新增错误；
- 保留错误；
- 被过滤错误；
- 被 propagation 放大的错误。

---

## RQ6：LaST-style raw cosine 是否是 MCTformer 的有效 semantic probe？

Experiment 1 使用：

\[
S^{feat,post}_{c,j}
=
\cos(c_c^{post},p_j^{post}),
\]

但模型真实 attention 使用：

\[
Q=W_Q\operatorname{LN}(X),\quad
K=W_K\operatorname{LN}(X).
\]

所以 Experiment 2 至少加入两个 control：

1. **Pre-attention normalized feature cosine**
   \[
   S^{norm}_{c,j,l}
   =
   \cos(
   \operatorname{LN}_l(c^{pre}_c),
   \operatorname{LN}_l(p^{pre}_j)
   );
   \]
2. **QK energy**
   \[
   E^{qk}_{c,j,l,h}
   =
   \frac{
   q_{c,l,h}^{\top}k_{j,l,h}
   }{\sqrt{d_h}}.
   \]

主论文式三层仍然是：

\[
Feature\rightarrow Attention\rightarrow CAM.
\]

`norm-feature` 和 `QK energy` 是 probe-validity controls，不应被包装成第四项方法贡献。

---

# 3. 源码依据与当前官方 pipeline

Codex 开始前必须重新阅读并记录下列文件的实际版本和 Git SHA。

## 3.1 Native MCTformer

- https://github.com/jungletada/TGCA/blob/main/models/mctformer.py

当前 `MCTformerV2Cam.forward_attention()` 的官方流程：

1. attention heads 求均值；
2. 最后三层 attention 求和；
3. 得到：
   \[
   A^{official}_{c2p};
   \]
4. patch head 输出 ReLU：
   \[
   CAM^{patch};
   \]
5. 融合：
   \[
   CAM^{c2p}
   =
   A^{official}_{c2p}\odot CAM^{patch};
   \]
6. 所有层 \(A_{p2p}\) 求和并传播：
   \[
   CAM^{final}
   =
   A^{official}_{p2p}CAM^{c2p}.
   \]

## 3.2 Native MCTformer+

- https://github.com/jungletada/TGCA/blob/main/models/mctformer_plus.py

当前 `MCTformerPlusCam.get_cam()`：

1. attention heads 求均值；
2. 最后三层 attention 求均值；
3. patch head ReLU；
4. 融合并开平方：
   \[
   CAM^{c2p}
   =
   \sqrt{
   A^{official}_{c2p}
   \odot
   CAM^{patch}
   };
   \]
5. 所有层 \(A_{p2p}\) 求和传播：
   \[
   CAM^{final}
   =
   A^{official}_{p2p}CAM^{c2p}.
   \]

## 3.3 Shared ViT attention

- https://github.com/jungletada/TGCA/blob/main/models/vit.py

`Attention.forward()` 返回 softmax 后的：

```python
weights
```

shape：

```text
[B, num_heads, num_tokens, num_tokens]
```

`Block.forward()` 返回：

```python
(tokens_after_block, attention_weights)
```

## 3.4 Dataset / transforms

- https://github.com/jungletada/TGCA/blob/main/datasets_cam.py

Experiment 1 的 deterministic transform：

```text
Resize short side to 512
CenterCrop 448
ToTensor
ImageNet normalization
```

Experiment 2 的 RGB image 必须与 Experiment 1 完全一致；semantic mask 使用相同 resize/crop geometry，但 nearest-neighbor interpolation。

## 3.5 CAM model factory / loading

- https://github.com/jungletada/TGCA/blob/main/utils.py
- https://github.com/jungletada/TGCA/blob/main/make_cam.py

复用：

```python
create_cam_model(args)
```

以及相同 checkpoint 解包规则。

## 3.6 LaST-ViT 诊断参考

- Repository  
  https://github.com/ChengShiest/LAST-ViT
- Patch Score  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/patch_score.py
- Point hit  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/evaluate_patch_hit.py
- FG/BG distribution  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/visualize_patch_score_distribution.py

本实验借鉴 LaST-ViT 的 representation-level score 和 region-hit diagnosis，不移植其 FFT/stability-based selective aggregation。

---

# 4. 实验范围

## 4.1 数据

主数据：

```text
PASCAL VOC 2012 val
1,449 images
20 foreground classes
semantic masks from SegmentationClass
```

只使用 val GT 做分析，不参与训练。

## 4.2 模型

```text
MCTformer
MCTformer+
```

使用 Experiment 1 完全相同的 checkpoints。

## 4.3 推理设置

```text
input size = 448
single scale = 1.0
no horizontal flip
no CRF
no IRN/PSA
model.eval()
torch.inference_mode()
one GPU
```

多尺度/flip 会改变 patch grid 和 map aggregation，不进入机制主分析。

## 4.4 预注册重点层

所有 12 层都保存和统计。

主文优先层：

\[
L1,\ L4,\ L5,\ L8,\ L9,\ L10,\ L11,\ L12.
\]

重点 transition：

\[
L9\rightarrow L10,
\]

\[
L10\rightarrow L11,
\]

\[
L11\rightarrow L12.
\]

---

# 5. 实现结构

建议新增：

```text
analysis/
└── lazy_assignment/
    ├── experiment2/
    │   ├── README.md
    │   ├── audit_experiment2_inputs.py
    │   ├── voc_semantic_dataset.py
    │   ├── patch_regions.py
    │   ├── signal_collector.py
    │   ├── native_cam_stages.py
    │   ├── run_experiment2_signals.py
    │   ├── build_experiment2_canonical.py
    │   ├── metrics_region.py
    │   ├── metrics_stage_linkage.py
    │   ├── metrics_shared_ownership.py
    │   ├── bootstrap_experiment2.py
    │   ├── analyze_experiment2.py
    │   ├── select_experiment2_examples.py
    │   ├── plot_experiment2.py
    │   └── generate_experiment2_report.py
    └── tests/
        ├── test_voc_joint_transform.py
        ├── test_patch_region_assignment.py
        ├── test_attention_signal_capture.py
        ├── test_qk_energy_capture.py
        ├── test_native_cam_stage_equivalence.py
        ├── test_experiment1_score_reproduction.py
        ├── test_shared_ownership_metrics.py
        ├── test_stage_transition_metrics.py
        └── test_experiment2_immutability.py
```

Experiment 1 源结果和 paired analysis 目录保持只读。

---

# 6. Phase A：输入审计

首先生成：

```text
results/lazy_assignment/experiment2/.../audit/
├── INPUT_AUDIT.md
├── source_metadata.json
├── gt_manifest.csv
├── checkpoint_verification.json
└── experiment1_linkage.json
```

必须确认：

1. 两个 Experiment 1 source roots 均可读；
2. checkpoints SHA256 与 Experiment 1 metadata 完全一致；
3. VOC val IDs 为相同 1,449 images；
4. image-level positive labels 与 Experiment 1 一致；
5. semantic mask 对每张图存在；
6. VOC class mapping：
   ```text
   image-level class 0...19
   ↔ mask class 1...20
   ```
7. mask 中只出现：
   ```text
   0, 1...20, 255
   ```
8. 448×448 transformed image 与 Experiment 1 transform 数值一致；
9. patch grid 为 28×28；
10. 所有输入文件 paths、sizes 和 hashes 被记录。

---

# 7. RGB 与 GT mask 的严格几何对齐

## 7.1 Joint deterministic transform

必须实现 joint transform：

```python
image = Resize(512, bicubic)(image)
mask  = Resize(512, nearest)(mask)

image = CenterCrop(448)(image)
mask  = CenterCrop(448)(mask)

image = ToTensor + Normalize
mask  = integer tensor
```

这里 `Resize(512)` 必须保持 torchvision 的“短边缩放到 512”的语义，而不是强制变成 512×512。

## 7.2 Transform equivalence test

对固定 20 张图，比较：

```python
joint_transform(image)[0]
```

与 Experiment 1 使用的：

```python
build_transform(is_train=False, make_cam=False)
```

输出。

要求：

```text
max_abs_diff < 1e-6
```

如果 PIL / torchvision version 导致极小差异，必须解释并保存误差，不允许直接放宽到很大容差。

## 7.3 方向检查

人工检查至少 20 张：

- image；
- transformed mask；
- patch-region overlay；
- Experiment 1 score map。

确认没有：

- H/W 交换；
- transpose；
- horizontal flip；
- crop offset；
- class index off-by-one。

---

# 8. Patch Semantic Region Assignment

## 8.1 基础像素集合

VOC mask：

```text
0 = background
1...20 = foreground classes
255 = void
```

对目标 class index \(c\in[0,19]\)，对应 mask ID：

\[
g=c+1.
\]

## 8.2 每个 16×16 patch 的统计

对 patch \(j\)：

- `target_count`：mask == \(g\)；
- `other_fg_count`：mask in 1...20 且 != \(g\)；
- `bg_count`：mask == 0；
- `void_count`：mask == 255；
- `valid_count = 256 - void_count`。

## 8.3 主规则

首先要求：

\[
\frac{valid\_count}{256}\ge 0.5.
\]

否则：

```text
region = void
```

在 valid pixels 中计算：

\[
r_t,\quad r_o,\quad r_b.
\]

若：

\[
\max(r_t,r_o,r_b)\ge\rho,
\]

则分配给最大类别；否则：

```text
region = mixed
```

主设置：

\[
\rho=0.5.
\]

Sensitivity：

\[
\rho=0.7.
\]

只需重新聚合统计，无需重新执行模型。

## 8.4 Pair-specific shared ownership

对正类别 pair：

\[
(c_a,c_b)
\]

patch region 细分为：

```text
target_a
target_b
other_foreground
background
mixed
void
```

不要把 `target_b` 简单并入 `other_foreground`，否则无法判断共享 support 是由 pair 中哪一个对象主导。

---

# 9. Signal Collection

对每个模型、图像、layer 收集以下 signals。

---

## 9.1 Signal F1：Existing post-block feature score

复用 Experiment 1：

\[
S^{feat,post,(l)}_{c,j}
=
\cos(c^{post,(l)}_c,p^{post,(l)}_j).
\]

不得覆盖或重新写 Experiment 1 NPZ。

Experiment 2 新 forward 计算出的同一 score 必须与 Experiment 1 source 做一致性验证。

---

## 9.2 Signal F2：Pre-attention normalized feature score

对 block \(l\) 的输入：

\[
X^{pre,(l)}
\]

应用该 block 的真实：

\[
\operatorname{norm1}_l.
\]

切分 class/patch tokens，计算：

\[
S^{feat,norm,(l)}_{c,j}
=
\cos(
\operatorname{norm1}_l(c^{pre}_c),
\operatorname{norm1}_l(p^{pre}_j)
).
\]

这是 actual Q/K projection 之前的 feature-space control。

---

## 9.3 Signal QK：Pre-softmax class-to-patch energy

从：

```python
block.attn.qkv(block.norm1(x_pre))
```

提取 q、k。

对每个 head：

\[
E^{qk,(l,h)}_{c,j}
=
\frac{
q_{c}^{(l,h)\top}
k_{j}^{(l,h)}
}{
\sqrt{d_h}
}.
\]

保存：

- head mean；
- head std；
- 必要的 head-wise region summary。

主 map：

\[
E^{qk,(l)}_{c,j}
=
\frac{1}{H}
\sum_h E^{qk,(l,h)}_{c,j}.
\]

不需要长期保存完整所有 heads 的每张图 tensor，除非磁盘允许；至少保存 head-mean map和 per-head summary。

---

## 9.4 Signal A：Class-to-patch attention

每层：

\[
A^{(l,h)}_{c2p}
=
A^{(l,h)}[:, :, :C, C:].
\]

主 map：

\[
A^{(l)}_{c2p}
=
\frac1H\sum_h A^{(l,h)}_{c2p}.
\]

保存两种形式：

### Raw patch attention

\[
A^{raw}_{c,j}=A_{c2p}(c,j).
\]

同时保存 patch-group mass：

\[
m^{(l)}_{c\rightarrow p}
=
\sum_j A^{raw,(l)}_{c,j}.
\]

### Conditional spatial attention

\[
\boxed{
\bar A^{(l)}_{c,j}
=
\frac{
A^{raw,(l)}_{c,j}
}{
\sum_kA^{raw,(l)}_{c,k}+\epsilon
}
}
\]

区域 attention mass 和 background leakage 使用：

\[
\bar A.
\]

这样不会把“读取 patch group 的总质量”和“在 patch grid 上读哪里”混为一谈。

Top-k ranking 对 raw 与 conditional 相同，只保存一份 ranking 即可。

---

## 9.5 Signal A-official：官方 class-to-patch 聚合

### MCTformer

严格复用源码：

\[
A_{c2p}^{official}
=
\sum_{l=10}^{12} A^{(l)}_{c2p}.
\]

### MCTformer+

严格复用源码：

\[
A_{c2p}^{official}
=
\frac13\sum_{l=10}^{12} A^{(l)}_{c2p}.
\]

同时构造 diagnosis-only aggregates：

```text
L10 only
L11 only
L12 only
mid3 = mean/sum of L4–L6 using a scale-normalized form
last3 conditional map
```

对 attention region ranking，sum/mean 只有常数差异；对 CAM 必须保留各模型官方公式。

---

## 9.6 Signal C0：Patch-head CAM

最终 patch tokens reshape 为 2D，经过 native head：

\[
Z^{patch}
=
head(P^{(L)}).
\]

定义：

\[
CAM^{patch}
=
\operatorname{ReLU}(Z^{patch}).
\]

shape：

\[
[B,C,H_p,W_p].
\]

同时保存 pre-ReLU logits，仅用于检查，但主 localization 使用 ReLU 后 map。

---

## 9.7 Signal C1：Class-attention refined CAM

### MCTformer

\[
CAM^{c2p}
=
A^{official}_{c2p}
\odot
CAM^{patch}.
\]

### MCTformer+

\[
CAM^{c2p}
=
\sqrt{
A^{official}_{c2p}
\odot
CAM^{patch}
}.
\]

必须严格匹配 native implementation。

---

## 9.8 Signal C2：Final CAM

使用所有层的 head-mean \(A_{p2p}\) 求和：

\[
A_{p2p}^{official}
=
\sum_{l=1}^{12}A_{p2p}^{(l)}.
\]

然后：

\[
CAM^{final}
=
A_{p2p}^{official}
CAM^{c2p}.
\]

分析脚本输出的 final CAM 必须与：

```python
model(x)
```

或 native `get_cam` / `forward_attention` 完全一致。

---

# 10. 数据保存策略

## 10.1 不保存完整 attention square matrix

禁止长期保存：

\[
[B,H,(C+N),(C+N)]
\]

的所有层矩阵。

forward 后立即切出：

- \(A_{c2p}\)；
- 必要时 \(A_{p2p}\) 用于当前 batch 计算 final CAM；
- region/head summaries。

## 10.2 推荐 per-image artifact

```text
signals/<model>/<image_id>.npz
```

包含：

```text
image_id
positive_class_ids
grid_h
grid_w
region_masks_rho05
region_masks_rho07
feature_post_scores
feature_norm_scores
qk_mean_scores
attn_c2p_raw
attn_c2p_conditional
attn_patch_mass
patch_cam
c2p_cam
final_cam
class_logits
patch_logits
class_token_pairwise_cosine
patch_norms
```

为了控制大小：

- 所有 maps 只保存 positive classes；
- qk/head full tensors可选；
- default 使用 float32；
- 可选 float16 archive 只能在验证误差后使用，canonical统计必须 float32。

---

# 11. Experiment 1 一致性回归

在正式执行前，从两模型各随机抽取至少 100 张图，比较新 pipeline 计算的：

\[
S^{feat,post}
\]

与 Experiment 1 `.npz`。

要求：

```text
same image IDs
same positive class IDs
same layer order
same patch order
max_abs_diff < 1e-6
```

理想情况下对完整 1,449 images 做 streaming comparison，仅保存最大误差，不额外复制数据。

如果不一致，停止全部 Experiment 2 分析，先修复：

- transform；
- token extraction；
- checkpoint；
- class indexing；
- layer indexing；
- H/W patch order。

---

# 12. Region Metrics：每个 signal 使用统一的 ranking 分析

对以下 signals：

```text
feature_post_l
feature_norm_l
qk_l
attn_l
attn_official
patch_cam
c2p_cam
final_cam
```

计算相同的 rank-based semantic metrics。

---

## 12.1 C-PiM / Top-1 ownership

对 valid patches：

\[
j^*
=
\arg\max_j S_{c,j}.
\]

记录：

```text
target_hit
other_fg_hit
background_hit
mixed_hit
void_hit
```

定义：

\[
C\text{-}PiM=P(target\_hit).
\]

对 all-zero CAM，标记：

```text
degenerate_map
```

不任意选第 0 个 patch。

---

## 12.2 Top-k region composition

对：

\[
k\in\{5\%,10\%,20\%\}
\]

取 stable top-k patches，统计：

\[
P(target\mid topk),
\]

\[
P(other\_fg\mid topk),
\]

\[
P(bg\mid topk).
\]

---

## 12.3 Area-normalized enrichment

由于 background 面积通常更大，必须计算：

\[
BGFrac
=
\frac{|\Omega_{bg}|}{|\Omega_{valid}|}.
\]

\[
\boxed{
BG\text{-}TailEnrich@k
=
\frac{
P(bg\mid topk)
}{
BGFrac+\epsilon
}
}
\]

同理：

\[
Target\text{-}TailEnrich@k,
\]

\[
OtherFG\text{-}TailEnrich@k.
\]

解释：

- `>1`：该 region 在 high-score tail 中超面积比例富集；
- `≈1`：接近 area-matched random selection；
- `<1`：被 high-score tail 排斥。

---

## 12.4 AUROC 与 AUPRC

分别计算：

\[
AUROC_{target-vs-bg},
\]

\[
AUPRC_{target-vs-bg},
\]

\[
AUROC_{target-vs-other},
\]

\[
AUPRC_{target-vs-other}.
\]

不允许把 AUC < 0.5 自动翻转成 > 0.5。

同时可报告：

```text
orientation = sign(AUC - 0.5)
separability = 2 * abs(AUC - 0.5)
```

但原始 signed AUC 必须保留。

---

## 12.5 Region score summaries

对 raw feature/QK：

```text
target mean / median / q90 / q95
other-FG mean / median / q90 / q95
BG mean / median / q90 / q95
target-BG margin
target-other margin
```

对 attention/CAM，主 region mass使用空间归一化后的非负 map。

---

## 12.6 Conditional background leakage

### Attention

\[
CBL^{attn}
=
\sum_{j\in\Omega_{bg}}
\bar A_{c,j}.
\]

### CAM

对每张非负 CAM 做 spatial L1 normalization：

\[
\bar M_{c,j}
=
\frac{
M_{c,j}
}{
\sum_kM_{c,k}+\epsilon
}.
\]

\[
CBL^{cam}
=
\sum_{j\in\Omega_{bg}}\bar M_{c,j}.
\]

Feature cosine不定义 mass leakage，避免将负值任意变成概率；feature 使用 tail/enrichment/AUC。

---

# 13. Shared Top-Tail Semantic Ownership

这是 Experiment 2 的核心新增分析。

只使用 multi-label images 和正类别 pairs。

---

## 13.1 每个 signal 的 shared support

对 signal \(X\)：

\[
T_k^X(c_a),
\quad
T_k^X(c_b).
\]

共享集合：

\[
T_{shared,k}^X
=
T_k^X(c_a)\cap T_k^X(c_b).
\]

主设置：

\[
k=10\%.
\]

Sensitivity：

\[
k=5\%,20\%.
\]

---

## 13.2 Shared support composition

用 pair-specific GT region mask统计：

\[
P(c_a\mid T_{shared}),
\]

\[
P(c_b\mid T_{shared}),
\]

\[
P(otherFG\mid T_{shared}),
\]

\[
P(bg\mid T_{shared}),
\]

\[
P(mixed/void\mid T_{shared}).
\]

同时计算 area enrichment：

\[
SharedBGEnrich
=
\frac{
P(bg\mid T_{shared})
}{
BGFrac+\epsilon
}.
\]

---

## 13.3 L9→L12 新增共享 patches

不仅分析每层 shared set，还要计算：

\[
T^{new}_{shared,L10}
=
T_{shared,L10}
\setminus
T_{shared,L9},
\]

同理：

\[
T^{new}_{shared,L11},
\quad
T^{new}_{shared,L12}.
\]

用 GT 判断：

> 后层重新耦合时新加入的 shared patches 是 target、other-FG 还是 background？

这是直接解释 Experiment 1 transition 的最关键指标。

---

## 13.4 Class-token collapse control

逐层计算同一图像两个正 class tokens 的：

\[
\cos(c_a^{(l)},c_b^{(l)}).
\]

并与：

\[
Jaccard(T_k(c_a),T_k(c_b))
\]

做相关分析。

可能情况：

### Token collapse

\[
\cos(c_a,c_b)\uparrow
\]

同时 map overlap 上升。

### Shared-patch attraction

class token cosine 保持低，但 map overlap 上升。

后者更直接支持“不同 class identities 共同认领同一 spatial support”。

---

# 14. Feature → Attention → CAM Linkage

---

## 14.1 Map-level Spearman

对每个 image–class pair：

\[
\rho(
S^{feat,post,(l)},
A^{(l)}_{c2p}
),
\]

\[
\rho(
S^{feat,norm,(l)},
E^{qk,(l)}
),
\]

\[
\rho(
E^{qk,(l)},
A^{(l)}_{c2p}
),
\]

以及最后阶段：

\[
\rho(
S^{feat,post,(12)},
CAM^{patch}
),
\]

\[
\rho(
A^{official}_{c2p},
CAM^{c2p}
),
\]

\[
\rho(
CAM^{c2p},
CAM^{final}
).
\]

注意：

- block \(l\) 的 attention 使用 pre-block input；
- Experiment 1 feature score 是 post-block；
- 报告中必须明确这个时序偏移；
- `feature_norm_l ↔ QK_l` 是更严格的同阶段 comparison。

---

## 14.2 Top-k overlap

对各 signal top-10% set 计算：

```text
feature ↔ attention
feature ↔ patch CAM
attention ↔ c2p CAM
c2p CAM ↔ final CAM
```

Jaccard 和 overlap coefficient 都保存。

---

## 14.3 Region-conditioned survival

对 feature top-k 中的背景 patches：

\[
B^{feat}_k
=
T_k^{feat}\cap\Omega_{bg}.
\]

定义：

\[
Survive_{feat\rightarrow attn}^{bg}
=
\frac{
|B_k^{feat}\cap T_k^{attn}|
}{
|B_k^{feat}|+\epsilon
}.
\]

同理：

\[
Survive_{attn\rightarrow c2pCAM}^{bg},
\]

\[
Survive_{c2pCAM\rightarrow finalCAM}^{bg}.
\]

对 target patches 也计算 retention：

\[
Retain^{target}.
\]

---

## 14.4 Stage-introduced and stage-removed patches

对于 stage \(X\rightarrow Y\)：

### Introduced background

\[
Intro^{bg}_{X\rightarrow Y}
=
\frac{
|(T_k^Y\setminus T_k^X)\cap\Omega_{bg}|
}{
|T_k^Y|+\epsilon
}.
\]

### Removed background

\[
Remove^{bg}_{X\rightarrow Y}
=
\frac{
|(T_k^X\setminus T_k^Y)\cap\Omega_{bg}|
}{
|T_k^X|+\epsilon
}.
\]

Target / other-FG 同样计算。

由此可判断：

- attention 是过滤 feature-level background，还是引入新 background；
- c2p multiplication 是否改善 ownership；
- \(A_{p2p}\) propagation 是否扩散 background。

---

## 14.5 四种主要 failure pattern

### Type A：Representation leakage，被 attention 过滤

```text
feature BG enrichment high
attention BG enrichment low
CAM BG leakage low
```

### Type B：Attention routing error

```text
feature target separation acceptable
A_c2p BG enrichment high
```

### Type C：Patch-head localization error

```text
feature/attention acceptable
patch CAM BG leakage high
```

### Type D：Propagation amplification

```text
c2p CAM acceptable
final CAM BG leakage high
```

### Type E：Full pipeline leakage

```text
feature high-BG
→ attention preserves BG
→ CAM preserves/amplifies BG
```

报告应按 full-set统计归类 image–class pairs，而不是只展示案例。

---

# 15. Official Last-Three-Layer Attention Analysis

由于 MCTformer+ 的 native CAM 正好使用 L10–L12，必须单独输出：

```text
L10 A_c2p
L11 A_c2p
L12 A_c2p
native last3 A_c2p
mid3 L4–L6 control
```

## 15.1 Region quality

每种 aggregation 比较：

- C-PiM；
- target / other-FG / BG top-k composition；
- BG-TailEnrich；
- target-vs-BG AUROC/AP；
- conditional background leakage。

## 15.2 Shared support

比较：

- L10 class-pair shared support；
- L11；
- L12；
- last3 aggregate；
- mid3 aggregate。

重点回答：

> Native last3 aggregation 是平均掉单层噪声，还是把三层已经重合的 shared support 进一步稳定下来？

## 15.3 CAM consequence

分别构造 analysis-only：

```text
CAM using L10 A_c2p
CAM using L11 A_c2p
CAM using L12 A_c2p
CAM using native last3
CAM using mid3 control
```

保留 native patch CAM 和 \(A_{p2p}\)，只改变用于 diagnosis 的 c2p map。

这些结果只能用于理解 layer selection，不得冒充新的方法或调参后主结果。

---

# 16. Classification Correctness Stratification

保存：

```text
class-token logits
patch-head logits
```

对每个 GT positive class 标记：

```text
class_token_positive = logit > 0
patch_head_positive = logit > 0
both_positive
either_negative
```

分层分析：

- correctly detected positives；
- class-token false negatives；
- patch-head false negatives；
- both false negatives。

原因：

一个没有正确分类的 class token，其 Patch Score / attention map 不应与正确分类样本混为同一种 representation failure。

主分析仍报告全体 positive GT classes；正确分类 subset 作为关键 control。

---

# 17. Statistics

## 17.1 基本单位

单类别 metrics：

```text
image cluster
```

一张图的多个类别不可视为独立。

class-pair / shared support metrics仍以 image为 cluster。

## 17.2 Bootstrap

```text
5,000 repeats
seed = 20260901 或新的固定 seed，并记录
95% percentile CI
```

两模型 paired comparison 使用相同 sampled image IDs。

## 17.3 Aggregation

同时报告：

- micro：所有 image–class pairs 等权；
- macro-class：先按 20 类求均值；
- class-wise；
- single-label；
- exactly 2 labels；
- 3+ labels。

## 17.4 Model comparison

所有 delta 定义：

\[
\Delta
=
MCTformer+ - MCTformer.
\]

只在 common image–class 或 image–class-pair keys 上计算。

---

# 18. Canonical Tables

建议建立：

```text
canonical/
├── per_image_class_layer_signal.parquet
├── per_image_class_cam_stage.parquet
├── per_image_class_stage_transition.parquet
├── per_multilabel_class_pair_layer_signal.parquet
├── per_shared_patch_ownership.parquet
├── per_class_token_pair_layer.parquet
└── source_index.parquet
```

## 18.1 `per_image_class_layer_signal`

字段至少包括：

```text
model
image_id
class_id
layer
signal
rho
num_target
num_other_fg
num_bg
num_mixed
num_void
top1_region
target_top05_fraction
other_top05_fraction
bg_top05_fraction
target_top10_fraction
other_top10_fraction
bg_top10_fraction
target_top20_fraction
other_top20_fraction
bg_top20_fraction
target_tail_enrich_10
other_tail_enrich_10
bg_tail_enrich_10
auc_target_bg
ap_target_bg
auc_target_other
ap_target_other
conditional_bg_mass
degenerate_map
classification_status
```

## 18.2 `per_image_class_cam_stage`

```text
patch_cam
c2p_cam
final_cam
```

保存相同 region metrics。

## 18.3 `per_shared_patch_ownership`

```text
model
image_id
class_a
class_b
layer_or_stage
signal
topk_ratio
shared_set_size
shared_target_a_fraction
shared_target_b_fraction
shared_other_fg_fraction
shared_bg_fraction
shared_mixed_void_fraction
shared_bg_enrichment
new_shared_from_previous_layer
```

---

# 19. 输出图表

主图先生成 diagnostic 版本，不做论文最终排版。

```text
plots/
├── feature_region_metrics_by_layer.png
├── attention_region_metrics_by_layer.png
├── feature_vs_attention_c_pim.png
├── feature_vs_attention_bg_tail_enrichment.png
├── shared_support_ownership_by_layer.png
├── new_shared_support_l9_l12.png
├── last3_attention_aggregation_analysis.png
├── cam_stage_background_leakage.png
├── stage_transition_background_survival.png
├── target_retention_vs_bg_removal.png
├── class_token_similarity_vs_map_overlap.png
├── probe_validity_raw_norm_qk_attn.png
└── classwise_l12_semantic_ownership.png
```

---

# 20. 案例选择

继续保留 Experiment 1 固定的 70 个规则选择案例，并新增基于 GT 的自动案例。

禁止手工挑图。

新增类别：

1. `shared support mostly background`；
2. `shared support mostly target_a`；
3. `shared support mostly target_b`；
4. `feature BG high but attention filters it`；
5. `attention introduces BG`;
6. `p2p propagation introduces BG`;
7. `raw cosine fails but A_c2p/CAM succeeds`;
8. `all three stages fail`;
9. `train` class representative cases；
10. `bird` negative-cosine control cases。

每张图展示：

```text
Input
GT
Feature score
A_c2p
Patch CAM
C2P CAM
Final CAM
```

多标签图像应同时展示至少两个正类别 maps，并用统一 patch grid 和明确 class label。

---

# 21. 必须添加的 Tests

## 21.1 Joint transform

- image output 与 Experiment 1 transform一致；
- mask geometry aligned；
- nearest-neighbor 保留合法 labels；
- center crop 坐标一致。

## 21.2 Patch region assignment

使用 synthetic 32×32 或 64×64 masks 构造：

- pure target；
- pure other foreground；
- pure background；
- mixed boundary；
- void-heavy patch；

验证 \(\rho=0.5/0.7\)。

## 21.3 Attention extraction

验证：

```text
A shape = [B,H,C+N,C+N]
A_c2p shape = [B,H,C,N]
A_pp shape = [B,H,N,N]
```

softmax row sum：

\[
\sum_k A_{q,k}=1
\]

数值容差内成立。

## 21.4 Conditional attention

验证：

\[
\sum_j\bar A_{c,j}=1
\]

对 patch mass 非零的 rows 成立。

## 21.5 QK reproduction

由 q/k 重新计算：

\[
softmax(QK^\top/\sqrt d)
\]

与模型返回 `weights` 逐元素一致：

```text
max_abs_diff < 1e-6
```

eval/no-dropout 条件下执行。

## 21.6 Native CAM equivalence

外部 stage decomposition 得到的：

```text
final_cam
```

与 native：

```python
model(x)
```

一致：

```text
max_abs_diff < 1e-6
```

MCTformer 和 MCTformer+ 都要通过。

## 21.7 Experiment 1 score reproduction

新 pipeline 的 `feature_post` 与原 NPZ：

```text
max_abs_diff < 1e-6
```

## 21.8 Shared ownership

用小型 synthetic region masks 和 top-k sets 验证 shared set composition 及 enrichment。

## 21.9 No numerical change

analysis hooks 不改变：

- final class tokens；
- patch tokens；
- attention；
- CAM。

## 21.10 Source immutability

分析前后：

- Experiment 1 source files hash 不变；
- checkpoints hash 不变；
- VOC GT 不被修改。

---

# 22. 执行顺序

## Step 0：提交 Experiment 1 分析代码

Experiment 1 报告显示 15 个 analysis/test files 尚未提交。开始 Experiment 2 前：

1. review；
2. commit；
3. tag；
4. 记录 commit。

避免后续无法复现上游分析。

## Step 1：Audit only

不跑模型，检查：

- result roots；
- checkpoints；
- VOC GT；
- transforms；
- class mapping；
- output locations。

## Step 2：20-image geometry smoke test

只检查：

- image/mask alignment；
- patch region labels；
- visualization orientation。

## Step 3：50-image MCTformer+ signal smoke test

生成：

- feature；
- qk；
- \(A_{c2p}\)；
- patch CAM；
- c2p CAM；
- final CAM；
- GT regions。

运行全部 equivalence tests。

## Step 4：50-image MCTformer paired smoke test

确认两个模型结果 schema 完全一致。

## Step 5：Full VOC val signal generation

顺序：

1. MCTformer+；
2. MCTformer；
3. immutable audit；
4. canonical build。

## Step 6：Core region analysis

先完成：

- target / other-FG / BG；
- C-PiM；
- BG-TailEnrich；
- AUROC/AP；
- class-wise；
- model-paired comparison。

## Step 7：Shared top-tail ownership

优先分析 L4/L5、L9–L12。

## Step 8：Feature→Attention→CAM linkage

生成 stage-survival / introduced / removed metrics。

## Step 9：Last-three aggregation diagnosis

比较 single layers、native last3 和 mid3 control。

## Step 10：Report generation

只根据 full-set tables 和 bootstrap 结果写结论。

---

# 23. Go / No-Go 决策矩阵

Experiment 2 完成后，根据结果选择下一项研究，不要自动继续改模型。

## Case A：Feature、Attention、CAM 都出现 BG enrichment

条件示例：

```text
feature BG-TailEnrich > 1
A_c2p conditional BG mass high
final CAM BG leakage high
```

结论候选：

> class-specific background semantics are present in the representation and preserved through localization.

下一步才考虑：

- semantic ownership；
- P2C blocking；
- register comparison；
- BG competitor。

## Case B：Feature BG high，但 attention/CAM 过滤

结论候选：

> representation pollution exists, but it is not the main localization bottleneck.

此时不应直接做 background CAM suppression；应研究 patch representation protection 或 causal relevance。

## Case C：Feature正常，\(A_{c2p}\) 出错

结论候选：

> the dominant issue is attention routing rather than feature semantics.

下一步更接近：

- attention regularization；
- class competition；
- head/layer selection。

## Case D：Feature和 attention正常，final CAM变差

结论候选：

> patch-affinity propagation amplifies incorrect regions.

下一步聚焦 \(A_{p2p}\)，不是 background token。

## Case E：Shared support主要属于其中一个真实前景类别

结论候选：

> late layers exhibit dominant-object capture / cross-class ownership collision.

需要 foreground class competition，而不仅是 BG modeling。

## Case F：Shared support主要属于 background

结论候选：

> late cross-class recoupling is driven by common background/context patches.

这是 Background Semantic Ownership 最直接的支持。

## Case G：Raw cosine错，但 QK/\(A_{c2p}\)/CAM正确

结论候选：

> LaST-style raw hidden cosine is not a valid semantic probe for this multi-class-token architecture.

此时应降低 Experiment 1 在论文中的地位，改用 actual attention geometry。

---

# 24. 最终报告结构

生成：

```text
reports/EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md
reports/NEXT_EXPERIMENT_DECISION.md
```

主报告结构：

## 1. Data and Integrity

## 2. Signals and Exact Native Pipelines

## 3. GT Patch Region Definition

## 4. Feature-level Semantic Ownership

## 5. Attention-level Semantic Ownership

## 6. Shared Top-Tail Ownership

## 7. Official Last-Three Attention Analysis

## 8. Patch CAM → C2P CAM → Final CAM

## 9. Feature–Attention–CAM Linkage

## 10. Probe Validity: Raw Cosine vs Norm/QK/Attention

## 11. MCTformer vs MCTformer+

## 12. Class-wise and Multi-label Analysis

## 13. What the Results Support

## 14. What the Results Do Not Support

## 15. Decision for the Next Causal Experiment

每个结论必须标记：

```text
[Fact]
[Statistical inference]
[Interpretation candidate]
[Unsupported]
```

---

# 25. Codex 最终交付物

```text
1. Experiment 2 全部代码
2. 所有 tests 和测试日志
3. INPUT_AUDIT.md
4. checkpoint / result / GT hashes
5. signal-generation metadata
6. canonical Parquet tables
7. layer-wise region metrics CSV
8. shared-support ownership CSV
9. feature-attention-CAM linkage CSV
10. last-three aggregation analysis CSV
11. class-wise results
12. plots
13. rule-selected examples
14. EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md
15. NEXT_EXPERIMENT_DECISION.md
16. exact_commands.sh
17. git diff summary
```

大型 per-image signal NPZ 和 Parquet 不提交 Git；提交：

- code；
- tests；
- metadata；
- compact tables；
- report；
- selected plots。

---

# 26. 可直接交给 Codex 的任务说明

```text
Read this Experiment 2 plan, the original Experiment 1 execution plan, and the
Experiment 1 paired analysis report before editing code.

The objective is to determine the semantic ownership of the late-layer shared
high-score patch supports observed in MCTformer+, and to trace them through three
levels: class–patch feature similarity, class-to-patch attention, and CAM.

Do not retrain or modify either model. Use the exact MCTformer and MCTformer+
checkpoints, single-scale 448 evaluation, and the same deterministic image
transform as Experiment 1. Load VOC semantic masks only in this new pipeline,
apply the exactly matched geometric transform with nearest-neighbor mask
interpolation, and construct target / other-foreground / background / mixed /
void patch regions.

Reuse the existing Experiment 1 feature scores and verify full numerical
reproduction. Extract per-layer A_c2p, conditionalize it over patch keys for
region-mass analysis, reproduce the exact native patch-CAM, class-attention-CAM,
and final A_p2p-propagated CAM stages, and test numerical equivalence with the
model’s official output.

Analyze all 12 layers, with pre-registered emphasis on L4, L5, and L9–L12. The
highest-priority analysis is the GT composition of patch supports shared by
different positive classes, especially patches newly entering the shared set
from L9 to L10, L10 to L11, and L11 to L12.

Compute C-PiM, target/other/BG top-k composition, area-normalized enrichment,
target-vs-BG and target-vs-other AUROC/AUPRC, conditional BG mass for attention
and CAM, class-token-pair similarity, and feature→attention→CAM survival,
introduction, and removal metrics. Use image-clustered paired bootstrap
confidence intervals and report micro, macro-class, class-wise, single-label,
2-label, and 3+-label results.

Add final-LayerNorm/pre-attention and QK-energy controls because raw post-block
cosine is not guaranteed to be the model’s actual class-localization geometry.

Treat all source results and checkpoints as immutable. Run all tests and stop
after producing EXPERIMENT2_SEMANTIC_OWNERSHIP_REPORT.md and
NEXT_EXPERIMENT_DECISION.md. Do not implement a solution or proposed method.
```

---

# 27. Experiment 2 的一句话目标

\[
\boxed{
\text{Identify who owns the late-layer shared patches, and determine whether
feature-level alignment is filtered, preserved, or amplified by }A_{c2p}
\text{ and CAM.}
}
