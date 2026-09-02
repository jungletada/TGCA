# Experiment 1 结果分析指南  
## Class-specific Patch Score：MCTformer / MCTformer+（Codex 执行版）

> **目标仓库：** `https://github.com/jungletada/TGCA`  
> **实验对象：** 已完成的 PASCAL VOC 2012 MCTformer 与 MCTformer+ `Experiment 1: Class-specific Patch Score` 结果  
> **当前任务：** 整理、校验、聚合和分析已有结果，不重新训练模型，不修改原始结果，不进入 GT 区域分析  
> **上游执行计划：** `Experiment_1_Class_Specific_Patch_Score_Codex_Plan.md`

---

# 0. 核心原则

本阶段处理的是：

\[
S_{i,c,j}^{(l)}
=
\cos\left(c_{i,c}^{(l)},p_{i,j}^{(l)}\right),
\]

其中：

- \(i\)：图像；
- \(c\)：图像级正类别；
- \(j\)：patch；
- \(l\)：Transformer layer。

Experiment 1 只提供 **class token 与 patch token 的 representation-level alignment**。如果当前结果尚未关联 VOC segmentation GT，则本阶段禁止直接声称：

- 高分 patch 是背景；
- 已经证明 lazy semantic assignment；
- MCTformer+ 比 MCTformer 有更严重的 background shortcut；
- score map 等于模型的因果决策依据；
- class-token similarity 等于 \(A_{c2p}\) 或 CAM。

本阶段的任务是先回答：

1. 结果文件是否完整、可重现、彼此对齐？
2. Class-specific Patch Score 的数值和空间结构是否合理？
3. Score 随网络深度如何演化？
4. 同一图像中的不同 class score maps 是否逐渐区分，还是趋于相似？
5. MCTformer 与 MCTformer+ 在相同 image–class pairs 上是否存在稳定差异？
6. 当前结果是否具备进入 Experiment 2（结合 GT 的 target / other-FG / background 分析）的条件？

---

# 1. 开始前：只读保护与结果定位

## 1.1 不得覆盖原始结果

Codex 必须把已有结果目录视为只读数据源。

禁止：

- 重写 `.npz`；
- 修改 `manifest.jsonl`；
- 修改原始 `metadata.json`；
- 删除失败样本；
- 把重新计算结果写回原目录；
- 为“方便”而改变 score dtype 或压缩格式。

所有派生结果写入新目录：

```text
results/
└── lazy_assignment/
    └── experiment1_analysis/
        ├── audit/
        ├── canonical/
        ├── tables/
        ├── plots/
        ├── examples/
        └── reports/
```

## 1.2 自动发现结果目录

优先寻找：

```text
results/lazy_assignment/experiment1_class_patch_score/mctformer/
results/lazy_assignment/experiment1_class_patch_score/mctformer_plus/
```

但不得假设实际目录一定完全相同。增加 CLI 参数：

```text
--mctformer-results
--mctformer-plus-results
--output-dir
```

若目录结构与原计划不同，先记录实际结构，再适配分析器，不要移动原始文件。

## 1.3 保存分析运行信息

每次分析都创建：

```text
run_metadata.json
analysis.log
```

至少记录：

```text
TGCA git commit
analysis script git commit
source result directories
source checkpoint SHA256
source metadata
Python version
PyTorch version
NumPy / pandas / scipy / sklearn versions
host name
analysis timestamp
CLI command
```

---

# 2. Phase A：结果文件清点与完整性审计

先完成审计，再做任何统计或绘图。

输出：

```text
audit/RESULT_INVENTORY.md
audit/integrity_report.json
audit/file_manifest.csv
audit/missing_or_invalid_samples.csv
```

---

## 2.1 文件清点

分别对 MCTformer 和 MCTformer+ 统计：

```text
result root
metadata.json 是否存在
manifest.jsonl 是否存在
summary_by_layer.csv 是否存在
score file 数量
visualization 数量
日志文件
失败记录
总磁盘大小
```

若预期使用 VOC val：

```text
expected_images = 1449
```

但不要只用文件数量判断完整性。必须以实际 `val_id.txt` 为基准做集合比较：

\[
\mathcal I_{\mathrm{expected}}
\quad\text{vs.}\quad
\mathcal I_{\mathrm{result}}.
\]

输出：

```text
missing image IDs
extra image IDs
duplicate image IDs
duplicate score paths
manifest entry without score file
score file without manifest entry
```

---

## 2.2 NPZ schema 检查

逐文件检查以下字段，兼容实际实现中的等价命名：

```text
image_id
positive_class_ids
scores_raw
grid_h
grid_w
```

若字段名不同，建立 schema mapping，并记录在：

```text
audit/schema_mapping.json
```

主 score tensor 预期：

```text
scores_raw.shape = [num_layers, num_positive_classes, num_patches]
dtype = float32 或可安全转成 float32
```

VOC / DeiT-S 常见预期：

```text
num_layers = 12
num_classes = 20
input_size = 448
patch_size = 16
grid_h = 28
grid_w = 28
num_patches = 784
```

这些值用于 sanity check，不能硬编码为唯一合法情况。最终以每个模型的 metadata 和实际 shape 为准。

---

## 2.3 数值检查

每个 score file 检查：

```text
NaN count
Inf count
score < -1 - tolerance
score > 1 + tolerance
empty positive_class_ids
duplicate positive_class_ids
class id outside [0, 19]
grid_h * grid_w != num_patches
num layers inconsistent
score dtype inconsistent
```

容差建议：

```text
cosine tolerance = 1e-5
```

对超过 \([-1,1]\) 的值：

- 若超出小于容差，记录为 floating-point overshoot；
- 若明显超出，标为 invalid，不得静默 clip。

---

## 2.4 Positive class 对齐检查

从 VOC `ImageLabel/cls_labels.npy` 重新读取 image-level labels，与每个结果文件中的：

```text
positive_class_ids
```

比较。

必须报告：

```text
exact match count
mismatch count
missing class count
extra class count
```

若分析结果只保存正类别，那么这个检查是后续 paired analysis 的必要前提。

---

## 2.5 两模型共同样本集合

构建：

\[
\mathcal P_M
=
\{(i,c)\}_{\mathrm{MCTformer}},
\]

\[
\mathcal P_{M+}
=
\{(i,c)\}_{\mathrm{MCTformer+}}.
\]

定义：

\[
\mathcal P_{\mathrm{common}}
=
\mathcal P_M\cap\mathcal P_{M+}.
\]

所有模型间 paired comparison 必须只使用：

\[
\mathcal P_{\mathrm{common}}.
\]

报告：

```text
number of common images
number of common image–class pairs
pairs only in MCTformer
pairs only in MCTformer+
```

若 common pair 比例低于 99%，先解释原因，再做比较。

---

## 2.6 Metadata 可比性检查

对两个模型的 metadata 比较：

```text
dataset split
input size
transform
patch size
number of layers
embedding dimension
class order
score definition
representation point
checkpoint hash
git commit
```

其中以下项目必须一致，才允许直接 paired compare：

```text
VOC val image list
input transform
input size
class-index convention
layer indexing convention
score formula
positive-class filtering
```

模型结构、checkpoint 和 embedding dimension可以不同，但必须明确记录。

---

# 3. Phase B：建立统一分析数据集

不要直接对数千个 `.npz` 文件反复扫描。先建立 canonical tables。

输出：

```text
canonical/per_pair_layer.parquet
canonical/per_image_class_layer.parquet
canonical/per_model_layer.parquet
canonical/source_index.parquet
```

---

## 3.1 分析单位

最小统计单位：

\[
(i,c,l)
\]

即：

```text
image_id
class_id
layer
model
```

每个单位对应一张：

\[
H_p\times W_p
\]

的 class-specific score map。

不得把所有 patch 当作独立样本进行显著性检验，因为同一图像、同一类别和同一层内的 patches 强相关。

---

## 3.2 每个 score map 的基础统计

对每个 `(model, image_id, class_id, layer)` 计算：

```text
num_patches
score_min
score_max
score_mean
score_std
score_q01
score_q05
score_q10
score_q25
score_q50
score_q75
score_q90
score_q95
score_q99
positive_score_fraction
negative_score_fraction
top_01_mean
top_05_mean
top_10_mean
bottom_10_mean
dynamic_range = max - min
iqr = q75 - q25
upper_tail_gap = q95 - q50
```

这些指标的目的：

- 检查 score 是否随 depth 整体漂移；
- 检查 map 是否逐渐变尖锐；
- 判断高分 tail 是否形成；
- 检查某些层是否出现近常数 score map。

不要把 `score_mean` 上升自动解释为 semantic quality 提升。

---

## 3.3 Score-map concentration

为了比较不同层的空间集中程度，使用两个互补指标。

### A. Rank-based top-tail concentration

\[
C_{10}
=
\operatorname{mean}(\operatorname{Top}_{10\%} S)
-
\operatorname{median}(S).
\]

该指标不需要把 cosine 转为概率。

### B. Temperature-normalized spatial entropy

仅作为辅助，将 score 转成空间概率：

\[
P_j
=
\frac{
\exp(S_j/\tau)
}{
\sum_k\exp(S_k/\tau)
},
\]

使用固定：

\[
\tau=0.1
\]

和敏感性：

\[
\tau\in\{0.05,0.2\}.
\]

定义归一化 entropy：

\[
H_{\mathrm{spatial}}
=
\frac{-\sum_jP_j\log P_j}
{\log N_p}.
\]

解释：

- 接近 1：score 空间分布平坦；
- 较低：score 集中于少数 patches。

必须明确：这是从 feature cosine 构造的辅助 concentration 指标，不是模型原始 attention entropy。

---

## 3.4 空间粗糙度与连通性

在 2D patch grid 上计算：

### Total Variation

\[
TV(S)
=
\frac{1}{|\mathcal E|}
\sum_{(u,v)\in\mathcal E}
|S_u-S_v|.
\]

### Neighbor correlation

计算水平和垂直相邻 patches 的 Pearson/Spearman correlation。

### Top-tail component count

对 top 10% score mask 计算 4-neighbor connected components：

```text
num_components_top10
largest_component_fraction
```

目的：

- 判断 score map 是形成连续语义区域，还是零散高分点；
- 为后续 GT-region 分析做 sanity check。

不能根据连续性直接判定 foreground/background。

---

# 4. Phase C：Layer-wise Representation Evolution

这是 Experiment 1 的主分析。

---

## 4.1 Layer convention

统一输出：

```text
layer = 1 ... 12
```

并保留：

```text
block_index = 0 ... 11
```

不得混用。

---

## 4.2 Layer-wise 汇总

分别对每个模型生成：

```text
tables/layerwise_summary_<model>.csv
```

主表至少包含：

```text
layer
num_image_class_pairs
mean_score
median_score
mean_max_score
mean_q95
mean_upper_tail_gap
mean_score_std
mean_spatial_entropy
mean_total_variation
mean_num_components_top10
mean_largest_component_fraction
```

同时生成：

- micro average：所有 image–class pairs 等权；
- macro-class average：先按类别求均值，再对 20 类等权；
- 95% cluster-bootstrap CI。

---

## 4.3 层间变化量

对于每个 `(image,class)`，计算：

\[
\Delta^{(l)}
=
m^{(l)}-m^{(1)}
\]

和：

\[
\delta^{(l)}
=
m^{(l)}-m^{(l-1)}.
\]

主要关注：

```text
max score
q95
upper-tail gap
spatial entropy
TV
top10 largest component
```

输出：

```text
tables/layerwise_delta_<model>.csv
```

这能识别：

- 哪一层开始形成高分 tail；
- 哪一层发生 score map 急剧集中；
- MCTformer+ 的层间变化是否比 MCTformer 更早或更强。

---

## 4.4 Rank stability across layers

对同一 `(image,class)` 的 score maps 计算：

### Consecutive-layer Spearman

\[
\rho
\left(
S^{(l-1)},S^{(l)}
\right).
\]

### Layer-1 to layer-l Spearman

\[
\rho
\left(
S^{(1)},S^{(l)}
\right).
\]

### Top-10% Jaccard

\[
J_{l-1,l}
=
\frac{
|T_{10}^{(l-1)}\cap T_{10}^{(l)}|
}{
|T_{10}^{(l-1)}\cup T_{10}^{(l)}|
}.
\]

输出：

```text
tables/layer_rank_stability_<model>.csv
```

解释：

- 高 Spearman / 高 Jaccard：高分 patch 排名较早固定；
- 中后层突然下降：语义对齐区域在该阶段重新分配；
- 这里只描述 representation evolution，不判断重新分配到前景还是背景。

---

# 5. Phase D：Multi-Class Map Diversity

MCTformer 的核心是多个 class tokens，因此必须分析同一图像中不同正类别 score maps 是否真正不同。

只在至少有两个正类别的图像上计算。

---

## 5.1 Pairwise map correlation

对同一图像中所有正类别对：

\[
(c_a,c_b)
\]

计算：

\[
\rho_{\mathrm{class}}
=
\operatorname{Spearman}
(S_{c_a},S_{c_b}).
\]

同时计算 cosine similarity（将每张 map flatten 后 L2 normalize）。

输出：

```text
mean_pairwise_class_spearman
mean_pairwise_class_cosine
```

较高的相关性可能表示不同 class tokens 产生相似空间 semantic-alignment maps，但不能在无 GT 时直接认定为 class confusion。

---

## 5.2 Top-tail overlap

对每个 class map 的 top 10% patches：

\[
T_{10}(c).
\]

类别对 overlap：

\[
J(c_a,c_b)
=
\frac{
|T_{10}(c_a)\cap T_{10}(c_b)|
}{
|T_{10}(c_a)\cup T_{10}(c_b)|
}.
\]

同时报告：

```text
top05_jaccard
top10_jaccard
top20_jaccard
```

---

## 5.3 Layer-wise class-map separation

生成：

```text
tables/class_map_diversity_by_layer.csv
```

字段：

```text
model
layer
num_multilabel_images
num_class_pairs
pairwise_spearman_mean
pairwise_spearman_ci
top10_jaccard_mean
top10_jaccard_ci
```

可能的结果解释：

- 相关性随层数下降：不同类别 spatial alignment 逐渐分化；
- 相关性随层数上升：不同 class maps 趋于共享空间模式；
- MCTformer+ 比 MCTformer 更低：增强 class-token discrimination 可能改善 map diversity；
- MCTformer+ 比 MCTformer 更高：更强 class semantics 可能同时使多个 tokens 聚焦共享 context。

最后一种解释仍然只是候选，必须由 Experiment 2 的 GT region 验证。

---

# 6. Phase E：Class-wise Analysis

不同 VOC 类别的上下文偏差差异很大，因此不能只做总体平均。

---

## 6.1 每类别汇总

输出：

```text
tables/classwise_summary.csv
```

每个模型、类别、层至少包含：

```text
num_images
mean_max_score
mean_q95
mean_upper_tail_gap
mean_spatial_entropy
mean_total_variation
mean_top10_largest_component
```

优先查看：

```text
aeroplane
boat
train
cow
horse
person
bicycle
chair
diningtable
sofa
```

但不得只挑预期符合假设的类别写结论。

---

## 6.2 类别样本数控制

报告每类 image count。类别样本少时：

- 使用 bootstrap CI；
- 不根据单一 class point estimate 下结论；
- 不能将类别排序直接解释成 intrinsic context bias。

---

# 7. Phase F：MCTformer vs. MCTformer+ Paired Comparison

必须使用 common `(image,class)` pairs，并按 image 做 cluster bootstrap。

---

## 7.1 比较指标

逐层比较：

```text
score_mean
score_max
q95
upper_tail_gap
score_std
spatial_entropy
total_variation
largest_component_fraction
class-map pairwise correlation
top10 class-map overlap
```

差值统一定义：

\[
\Delta
=
\mathrm{MCTformer+}
-
\mathrm{MCTformer}.
\]

输出：

```text
tables/mctformer_vs_plus_paired.csv
```

字段：

```text
metric
layer
n_images
n_image_class_pairs
mctformer_mean
mctformer_plus_mean
paired_delta
ci_low
ci_high
standardized_effect
```

---

## 7.2 Bootstrap 方法

由于一张图可能包含多个正类别，不能把 image–class pairs 当成完全独立。

推荐：

1. 以 image ID 为 cluster；
2. bootstrap sample images with replacement；
3. 每次保留该图全部正类别；
4. 重新计算模型均值和 paired delta；
5. 1000 次用于初步；
6. 最终报告使用 5000 次；
7. 固定随机 seed。

同时报告 standardized paired effect：

\[
d_z
=
\frac{
\operatorname{mean}(\Delta_i)
}{
\operatorname{std}(\Delta_i)+\epsilon
}.
\]

不要只输出 p-value。

---

## 7.3 多层多指标的解释约束

本分析会产生很多 layer × metric 比较。禁止根据某一个偶然 CI 排除 0 的结果选择性下结论。

主指标预先固定为：

1. `upper_tail_gap`
2. `spatial_entropy`
3. `top10 class-map Jaccard`
4. `consecutive-layer Spearman`

其他指标作为补充。

---

# 8. Phase G：异常样本与案例选择

案例图必须通过预定义规则选取，不能手工挑“最漂亮”的图。

---

## 8.1 自动选择四类样本

分别输出每类 top 10：

### A. 最大层间变化

\[
\Delta q95
=
q95^{(12)}-q95^{(1)}.
\]

### B. 最高 class-map overlap

多标签图像中 layer 12 的 top10 Jaccard 最大。

### C. 最低 class-map overlap

多标签图像中 layer 12 的 top10 Jaccard 最小。

### D. 最大模型分歧

MCTformer 和 MCTformer+ layer 12 score-map Spearman 最低，或关键指标差值最大。

输出：

```text
examples/example_selection.csv
```

---

## 8.2 每个案例图

展示：

```text
Original image
positive class name
MCTformer: L1 / L4 / L8 / L12
MCTformer+: L1 / L4 / L8 / L12
```

保存两版：

### Raw cosine

固定色标：

```text
vmin = -1
vmax = 1
```

### Per-map min-max

仅辅助观察空间排名，标题必须标：

```text
Min-max visualization only
```

不得根据 min-max 图比较模型之间的 absolute score magnitude。

---

# 9. Phase H：Experiment 1 能回答什么

## 9.1 可以回答

- 是否成功获得逐层 class-specific semantic-alignment maps；
- score 数值是否稳定、是否形成高分 tail；
- semantic alignment map 随 depth 是否逐渐集中或重新排列；
- 同一图像中不同 class tokens 的空间 map 是否分化；
- MCTformer+ 相对 MCTformer 在 representation geometry 上有何稳定差异；
- 哪些 layer 和类别值得在 Experiment 2 中优先研究。

## 9.2 不能回答

- top score 是否落在目标物体；
- 高分 patch 是否为 background；
- 是否存在 class-specific background shortcut；
- feature score 是否进入 \(A_{c2p}\)；
- score 是否进入最终 CAM；
- high-score patches 是否对 classification 有因果作用；
- register 或 P2C blocking 是否能解决问题。

这些问题分别属于：

```text
Experiment 2: GT region / C-PiM / BG-Tail
Experiment 3: feature vs A_c2p vs CAM
Causal experiments: masking / P2C blocking / locality intervention
```

---

# 10. Experiment 2 Readiness Gate

完成分析后输出：

```text
reports/EXPERIMENT2_READINESS.md
```

只有以下条件满足，才进入下一阶段：

## 数据完整性

- 两模型有效 image 覆盖率 ≥ 99%；
- 两模型 common image–class pair 覆盖率 ≥ 99%；
- 无未解释的 NaN/Inf；
- score shape、layer 数和 grid size 一致；
- positive class IDs 与 VOC labels 一致；
- checkpoint 和 metadata 完整。

## 数值有效性

- score maps 不是近常数；
- 不存在大面积超过 cosine 范围；
- layer-wise statistics 可复现；
- hook 没有改变模型；
- 至少部分层存在可分析的 spatial variation。

## 分析价值

满足至少一项：

- layer-wise top-tail concentration 有清晰变化；
- score-map ranking 在某些层明显重组；
- 多标签图像的 class-map diversity 随 depth 有系统变化；
- MCTformer 与 MCTformer+ 存在稳定 paired difference。

即使没有明显模型差异，只要结果可信，也可以继续 Experiment 2，因为 GT-region 分析可能揭示总体统计无法看到的背景偏差。

---

# 11. 最终报告结构

Codex 生成：

```text
reports/EXPERIMENT1_ANALYSIS_REPORT.md
```

严格采用以下结构。

## 1. Data and Reproducibility

- 源目录；
- checkpoints；
- commits；
- image/pair coverage；
- schema 和 integrity；
- transform；
- 缺失与异常。

## 2. Score Definition

明确：

\[
S_{c,j}^{(l)}
=
\cos(c_c^{(l)},p_j^{(l)}).
\]

说明这是 representation similarity，不是 attention/CAM。

## 3. Global Score Statistics

- layer-wise score scale；
- tail；
- concentration；
- spatial roughness。

## 4. Layer-wise Evolution

- consecutive-layer rank stability；
- layer-1 to layer-l drift；
- top-tail movement。

## 5. Multi-Class Map Diversity

- pairwise map correlation；
- top-k overlap；
- 单标签 vs 多标签图像。

## 6. MCTformer vs. MCTformer+

- paired deltas；
- cluster-bootstrap CI；
- class-wise differences。

## 7. Representative and Failure Cases

只使用预定义 selection rules。

## 8. What the Results Support

仅陈述数据直接支持的内容。

## 9. What the Results Do Not Support

明确禁止的背景/因果结论。

## 10. Readiness for Experiment 2

- `READY`
- `READY WITH FIXES`
- `NOT READY`

并列出原因。

---

# 12. 推荐脚本结构

```text
analysis/lazy_assignment/
├── analyze_experiment1_results.py
├── audit_experiment1_results.py
├── build_canonical_tables.py
├── metrics_experiment1.py
├── bootstrap.py
├── select_examples.py
├── plot_experiment1.py
└── tests/
    ├── test_npz_schema.py
    ├── test_canonical_aggregation.py
    ├── test_cluster_bootstrap.py
    ├── test_map_metrics.py
    └── test_result_immutability.py
```

---

# 13. 推荐 CLI

## 审计

```bash
python analysis/lazy_assignment/audit_experiment1_results.py \
  --mctformer-results /ABS/PATH/TO/mctformer \
  --mctformer-plus-results /ABS/PATH/TO/mctformer_plus \
  --voc-root data/VOCdevkit/VOC2012 \
  --val-list data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --output-dir results/lazy_assignment/experiment1_analysis/audit
```

## 建 canonical table

```bash
python analysis/lazy_assignment/build_canonical_tables.py \
  --mctformer-results /ABS/PATH/TO/mctformer \
  --mctformer-plus-results /ABS/PATH/TO/mctformer_plus \
  --output-dir results/lazy_assignment/experiment1_analysis/canonical
```

## 全部分析

```bash
python analysis/lazy_assignment/analyze_experiment1_results.py \
  --canonical-dir results/lazy_assignment/experiment1_analysis/canonical \
  --output-dir results/lazy_assignment/experiment1_analysis \
  --bootstrap-repeats 5000 \
  --bootstrap-seed 20260901 \
  --topk-ratios 0.05,0.10,0.20 \
  --entropy-temperatures 0.05,0.10,0.20
```

## 绘图

```bash
python analysis/lazy_assignment/plot_experiment1.py \
  --analysis-dir results/lazy_assignment/experiment1_analysis \
  --output-dir results/lazy_assignment/experiment1_analysis/plots
```

---

# 14. 测试要求

## 14.1 结果不可变性

分析前后对所有原始 `.npz` 生成 SHA256。

测试要求：

```text
before hashes == after hashes
```

## 14.2 手工合成 map

创建 4×4 synthetic score maps：

- constant；
- one sharp peak；
- two components；
- smooth gradient。

验证：

```text
entropy
TV
component count
top-k mask
rank correlation
```

符合预期。

## 14.3 Cluster bootstrap

使用小型 synthetic dataset 验证：

- 同一 image 的多个 classes 始终一起抽样；
- paired models 使用同一 bootstrap image indices；
- 固定 seed 可复现。

## 14.4 Canonical table round trip

随机抽取 10 个 source `.npz`，确认 canonical table 中：

```text
mean
std
quantiles
max
```

与直接 NumPy 计算一致。

---

# 15. Codex 的输出纪律

Codex 在最终总结中必须区分：

### Fact

直接来自文件或计算结果，例如：

```text
Layer-12 median q95 was X.
```

### Statistical inference

例如：

```text
Paired bootstrap 95% CI for the delta was [a,b].
```

### Interpretation candidate

例如：

```text
The increase in map overlap may indicate reduced class-specific spatial diversity.
```

### Unsupported claim

例如：

```text
The model attends to background shortcuts.
```

如果没有 GT/causal evidence，最后一类必须明确标记为不支持，不能写入结论。

---

# 16. Codex 最终交付物

```text
1. audit/RESULT_INVENTORY.md
2. audit/integrity_report.json
3. audit/file_manifest.csv
4. canonical/*.parquet
5. tables/*.csv
6. plots/*.png 或 *.pdf
7. examples/example_selection.csv
8. reports/EXPERIMENT1_ANALYSIS_REPORT.md
9. reports/EXPERIMENT2_READINESS.md
10. analysis.log
11. exact_commands.sh
12. tests 及测试日志
13. git diff summary
```

大型原始 score files 和派生 Parquet 是否提交 Git，应遵循仓库 `.gitignore` 与文件大小策略。一般只提交：

- 分析代码；
- 文档；
- 小型汇总 CSV；
- 图；
- metadata；

不提交所有 per-image `.npz` 和超大 Parquet。

---

# 17. 可直接发送给 Codex 的任务说明

```text
Read this document and the original Experiment 1 execution plan first.

The Experiment 1 result files already exist locally and may be numerous. Treat all
source result directories as immutable. Do not retrain models, regenerate scores,
or modify any source NPZ/manifest/metadata files.

First locate the actual MCTformer and MCTformer+ result roots and produce a complete
integrity audit. Then build canonical Parquet tables, perform layer-wise score-scale,
tail-concentration, spatial-structure, rank-stability, multi-class map-diversity,
class-wise, and paired model analyses exactly as specified here.

Use image-clustered paired bootstrap confidence intervals. Do not treat patches or
image-class pairs from the same image as independent observations.

At this stage, do not load semantic segmentation GT and do not claim background
leakage, lazy semantic assignment, attention behavior, CAM behavior, or causal
shortcut. Clearly distinguish representation-level observations from hypotheses
that require Experiments 2–3.

Run all tests, preserve exact commands and logs, and end with:
1. EXPERIMENT1_ANALYSIS_REPORT.md
2. EXPERIMENT2_READINESS.md
3. a concise git diff and data-integrity summary.
```

---

# 18. 人工审阅检查表

在接受 Codex 结果前，人工确认：

- [ ] 它没有覆盖原始结果；
- [ ] 它没有用 min-max map 做模型 absolute-score 比较；
- [ ] 它没有把 patches 当作独立统计样本；
- [ ] 它只在 common image–class pairs 上做 paired comparison；
- [ ] 它区分 micro 与 macro-class 平均；
- [ ] 它没有因看到热图就声称 background leakage；
- [ ] 它没有选择性只展示符合假设的样本；
- [ ] 它记录了 source checkpoint hash 和 Git commit；
- [ ] 所有图都能追溯到 canonical table；
- [ ] 每个结论都能指向一个明确表格或统计文件；
- [ ] `EXPERIMENT2_READINESS.md` 给出了清晰的下一步条件。
