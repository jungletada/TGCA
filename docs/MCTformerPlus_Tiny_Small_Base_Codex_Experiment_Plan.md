# MCTformer+-Tiny / Small / Base 宽度扩展实验计划（Codex 执行版）

> **目标仓库：** `https://github.com/jungletada/TGCA`  
> **目标分支：** `main`  
> **LHR 路径：** `/home/peng/code/TGCA`  
> **数据集：** PASCAL VOC 2012  
> **核心对象：** 原生、Vanilla Attention、BCSS-E0、PSL-baseline、CTI-BGT-off 的 MCTformer+  
> **现有状态：** MCTformer+-Small（原始 `mctformerplus`）的 seed-0 训练已完成；必须优先复用，不得覆盖或默认重跑。  
> **计划性质：** 这是一个 **backbone width / capacity scaling study**，不是严格意义上的 scaling law 证明。Tiny、Small、Base 都是 12 层，变化的是 embedding width、attention heads 和参数量。

---

# 0. 交给 Codex 时附上的授权语句

由于仓库 `AGENTS.md` 要求完整科学运行必须绑定到干净、已跟踪的 Git 状态，而本任务需要修改代码，建议把下面这段与本计划一起交给 Codex：

> 授权你在 TGCA 仓库的 `main` 分支完成本计划所需的代码修改、测试、只读审计和实验执行；在测试全部通过后，授权创建一个本地 Git commit，用于固定完整实验对应的代码版本。禁止 push、merge、rebase、强制 reset、删除或覆盖已有 checkpoints/results，也禁止清理任何已有 tracked/untracked 用户文件。长时间实验使用 tmux，且不得中断已有任务。

---

# 1. 研究问题与结论边界

## 1.1 核心研究问题

固定 MCTformer+ 的数据、训练轮数、有效 batch size、优化器、CAM 生成方法和评估协议，只改变 DeiT backbone 的宽度：

\[
\text{Tiny}\;(D=192)
\rightarrow
\text{Small}\;(D=384)
\rightarrow
\text{Base}\;(D=768),
\]

研究以下问题：

1. **分类能力是否随模型容量单调提升？**
2. **Raw CAM localization 是否同步提升？**
3. **分类与定位是否出现 scaling decoupling？**
4. **更大模型是否产生更干净的 class-token semantic ownership，还是仅提升分类但保留甚至加重背景语义泄漏？**
5. **性能提升是否值得相应的参数、显存、延迟和训练时间成本？**

## 1.2 必须使用的表述

正文、表格和报告中使用：

- `MCTformer+ width scaling`
- `model-capacity scaling behavior`
- `width-scaling trend`
- `classification–localization scaling gap`

不得仅凭 3 个模型、1 个训练 seed 使用：

- “MCTformer+ follows a scaling law”
- “power law has been established”
- “performance always scales predictably with parameters”

可以拟合：

\[
y=a+b\log_{10}P
\]

作为**描述性趋势线**，但必须注明：只有 3 个架构点，不能据此证明 power law。

## 1.3 预注册的结果解释

### 情况 A：分类与 CAM 都单调提高

\[
\mathrm{mAP}_{T}<\mathrm{mAP}_{S}<\mathrm{mAP}_{B},
\qquad
\mathrm{mIoU}_{T}<\mathrm{mIoU}_{S}<\mathrm{mIoU}_{B}.
\]

解释为：在固定 MCTformer+ recipe 下，模型容量扩展同时改善 recognition 与 localization。

### 情况 B：分类提高，但 CAM 饱和或下降

\[
\mathrm{mAP}\uparrow,
\qquad
\mathrm{CAM\ mIoU}\approx\text{flat}\ \text{or}\ \downarrow.
\]

解释为：MCTformer+ 存在 **classification–localization scaling decoupling**；更大 backbone 并不会自动解决 weak multi-label supervision 下的 semantic ownership 问题。

### 情况 C：CAM 提高，但 semantic ownership 指标变差

例如：

\[
\mathrm{CAM\ mIoU}\uparrow,\quad
\mathrm{C\text{-}PiM}\downarrow,\quad
\mathrm{BG\text{-}Tail}\uparrow.
\]

解释为：最终 CAM 性能提高不等于 class-token/patch 表征变得更语义纯净，可能存在后处理、patch-CAM 分支或 P2P refinement 的补偿作用。

### 情况 D：Tiny / Small / Base 非单调

解释为：在当前数据量与原始 recipe 下，没有得到单调 width scaling 证据；需要区分容量瓶颈、优化失配和训练方差，不能强行宣称 scaling。

---

# 2. 当前 `main` 分支的代码审计结论

Codex 开始修改前必须重新读取当前 live worktree；下面是制定计划时对 GitHub `main` 的审计结果。

## 2.1 当前模型只注册了 Small

当前 `models/mctformer_plus.py` 只有：

```python
@register_model
def mctformerplus(...):
    model = MCTformerPlus(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        ...
    )
```

且 `pretrained=True` 路径写死为：

```text
deit_small_patch16_224-cd65a155.pth
```

因此必须添加 Tiny 和 Base factory，同时保留原始 `mctformerplus` 名称和 Small state-dict 兼容性。

## 2.2 训练入口的预训练权重默认值写死为 Small

`train_model_v2.py` 当前：

```python
--finetune
https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
```

新 runner 必须显式传入每个变体自己的官方权重，不能依赖默认值。

## 2.3 CAM 路径当前固定构造 384 维 Small

`utils.create_cam_model()` 对任何包含 `mctformerplus` 的名称都直接构造：

```python
MCTformerPlusCam(...)
```

其默认父类宽度为 Small。  
因此只添加 `timm` 注册函数是不够的；`make_cam.py`、benchmark、scale-CAM、lazy-assignment analysis 都需要改成由 checkpoint/model spec 解析准确变体。

## 2.4 当前 CAM 的 layer policy 必须保持不变

当前 MCTformer+ CAM 不是简单平均全部 class-to-patch layers：

- class-to-patch attention：最后 **3 层**平均；
- patch-to-patch refinement：对全部 **12 层**的 P2P attention 求和；
- Tiny / Small / Base 都保持 depth=12，因此无需修改 layer policy。

主实验禁止改变：

```python
self.n_layers = 3
```

否则会把 backbone scaling 与 CAM readout 改动混在一起。

## 2.5 当前训练日志中的 `mAP` 不是标准 VOC class-wise mAP

当前 `engine.compute_mAP()` 对每张图像分别在 20 个标签之间计算 AP，再对图像求平均。它更接近 image-wise label-ranking AP，不是通常的：

\[
\frac{1}{20}\sum_{c=1}^{20}\mathrm{AP}_c.
\]

处理原则：

1. 保留现有日志值，命名为：
   ```text
   legacy_imagewise_AP
   ```
   用于验证训练流程没有变化；
2. 新增完整 validation-set evaluator，计算：
   - class-token branch macro class AP；
   - patch-GWRP branch macro class AP；
   - micro AP；
   - 每类别 AP；
   - legacy image-wise AP；
3. 论文中的分类主指标使用 dataset-level macro class AP。

## 2.6 Small baseline runner已具备较好的 provenance 结构

当前 `experiments/baselines/run_mctformerplus_voc.sh` 已记录：

- Git commit / branch / dirty state；
- Conda / pip / CUDA / GPU；
- 数据列表和预训练权重 SHA-256；
- seed、训练命令；
- final checkpoint；
- multi-scale CAM；
- raw-CAM threshold sweep；
- `PIPELINE_COMPLETE`。

新 scaling runner应复用这些安全原则，但不得覆盖或改写旧 Small runner。

---

# 3. 模型矩阵

使用 Facebook/Meta 官方 **non-distilled DeiT ImageNet-1K 224×224** 权重，不使用 distilled checkpoint，不使用 DeiT III。

| 变体 | 注册名 | Depth | Embed dim \(D\) | Heads | Head dim | Patch | MLP ratio | 预期参数量约值 | 官方预训练文件 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| MCTformer+-Tiny | `mctformerplus_tiny` | 12 | 192 | 3 | 64 | 16 | 4 | 约 5.7M | `deit_tiny_patch16_224-a1311bcf.pth` | 新训练 |
| MCTformer+-Small | `mctformerplus` | 12 | 384 | 6 | 64 | 16 | 4 | 约 22.1M | `deit_small_patch16_224-cd65a155.pth` | 已完成，复用 |
| MCTformer+-Base | `mctformerplus_base` | 12 | 768 | 12 | 64 | 16 | 4 | 约 86.6M | `deit_base_patch16_224-b5f2ef4d.pth` | 新训练 |

官方 URL：

```text
https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
```

上表参数量仅用于预估。最终表格必须使用当前代码实例化后得到的实际参数量，不能直接照抄官方 DeiT 参数量。

---

# 4. 主实验范围

## 4.1 必做核心实验

1. 实现 MCTformer+-Tiny / Small / Base 统一架构 registry。
2. 完成官方 DeiT-Tiny / Base 权重适配和严格审计。
3. 复用已有 Small seed-0 结果。
4. 训练 Tiny seed-0。
5. 训练 Base seed-0。
6. 对三个模型统一评估：
   - 正确的 VOC classification macro AP；
   - raw CAM on VOC train；
   - raw CAM on VOC val；
   - 固定阈值 mIoU；
   - 完整阈值曲线；
   - 参数、显存、延迟、训练时间。
7. 运行 class-token semantic ownership 的轻量机制分析：
   - C-PiM；
   - target / other-FG / background top-1 outcome；
   - BG-Tail@5 / BG-Tail@10；
   - target-vs-background score margin；
   - layer-wise trend。
8. 生成机器可读汇总、图表和研究结论报告。

## 4.2 初始阶段不做

- 不更换到 DeiT III；
- 不改变 depth；
- 不改变 patch size；
- 不引入 distilled token；
- 不加入 BG token / register / TGCA / split softmax；
- 不启用 BCSS、PSL、CTI-BGT；
- 不修改 CAM layer readout；
- 不针对 Tiny 或 Base 单独搜索有利阈值并将其作为主结果；
- 不在看完 Tiny/Base 结果后再更改主指标；
- 不直接声称 power-law scaling。

## 4.3 条件触发实验

### 多 seed

初始只完成与现有 Small 对齐的 seed-0。  
准备发表结论时，统一追加：

```text
seed = 1, 2
```

且三个模型全部补齐，不得只对表现最好或最差的模型补 seed。

### 下游 segmentation

只有 raw CAM / semantic ownership 得到值得继续的趋势后，再对 T/S/B 使用完全相同的 pseudo-label refinement 与 segmentation pipeline，报告最终 val/test segmentation mIoU。

---

# 5. 不可变控制变量

三个模型必须保持下列设置一致。

## 5.1 模型与方法

```text
dataset                  VOC12
num_classes              20 foreground classes
input_size               448
patch_size               16
depth                    12
mlp_ratio                4
qkv_bias                 True
LayerNorm eps            1e-6
attention_normalization  vanilla
attention_gamma          1.0
bcss_variant             e0
psl_variant              baseline
cti_bgt                  False
CAM C2P layers           last 3
CAM P2P layers           all 12
```

## 5.2 训练 recipe

与已完成 Small runner 对齐：

```text
epochs                    45
effective batch size      32
seed                      0
optimizer                 AdamW
nominal --lr              5e-4
min_lr                    1e-5
weight_decay              0.05
warmup_epochs             5
scheduler                 cosine
drop                      0.0
drop_path                 0.1
input interpolation       bicubic
training augmentation     保持 train_model_v2.py 当前逻辑
train list                train_aug_id.txt
val list                  val_id.txt
```

当前代码会按：

\[
\mathrm{optimizer\ LR}
=
\mathrm{nominal\ LR}
\times
\frac{\mathrm{effective\ batch}}{512}
\]

缩放。有效 batch=32 时：

\[
5\times10^{-4}\times\frac{32}{512}
=
3.125\times10^{-5}.
\]

必须同时记录 nominal LR 和最终 optimizer LR。

## 5.3 CAM protocol

```text
checkpoint                final epoch checkpoint
input_size                448
scales                    1.0, 0.75, 1.25
horizontal flip           保持当前 make_cam.py 行为
normalization             per-image per-class min-max
train CAM list            train_id.txt
val CAM list              val_id.txt
CRF                       raw-CAM 主结果不使用
```

---

# 6. 代码实现设计

# 6.1 单一模型规格源

不得在 `models/`、shell runner、CAM tool 和 analysis tool 中分别手写多份维度表。

在 `models/mctformer_plus.py` 或一个小型专用模块中建立唯一 source of truth，例如：

```python
MCTFORMERPLUS_VARIANTS = {
    "tiny": {
        "model_name": "mctformerplus_tiny",
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "patch_size": 16,
        "mlp_ratio": 4,
        "pretrained_url": (
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_tiny_patch16_224-a1311bcf.pth"
        ),
    },
    "small": {
        "model_name": "mctformerplus",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 16,
        "mlp_ratio": 4,
        "pretrained_url": (
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_small_patch16_224-cd65a155.pth"
        ),
    },
    "base": {
        "model_name": "mctformerplus_base",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
        "mlp_ratio": 4,
        "pretrained_url": (
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_base_patch16_224-b5f2ef4d.pth"
        ),
    },
}
```

需要提供：

```python
resolve_mctformerplus_variant(model_name: str) -> str
get_mctformerplus_spec(variant_or_model_name: str) -> Mapping
build_mctformerplus(variant: str, cam: bool, **kwargs)
model_spec_from_instance(model) -> dict
```

注册：

```python
@register_model
def mctformerplus_tiny(...)

@register_model
def mctformerplus(...)       # 原始 Small，名称保持不变

@register_model
def mctformerplus_base(...)
```

不建议把原始 Small 改名为 `mctformerplus_small`。如确实添加别名，旧 `mctformerplus` 必须仍是 canonical legacy name。

# 6.2 预训练权重适配必须统一

把当前散落的 DeiT-to-MCTformer+ 适配逻辑整理为单一函数，例如：

```python
adapt_deit_checkpoint_for_mctformerplus(
    checkpoint,
    model,
    num_classes,
) -> tuple[state_dict, load_report]
```

必须保持原始 MCTformer+ 初始化语义：

1. 从官方 checkpoint 读取单个 DeiT CLS token；
2. repeat 为 20 个 class tokens；
3. 将 DeiT 224 的 14×14 positional patch embedding bicubic 插值到 448 的 28×28；
4. 将单个 CLS positional embedding repeat 为 20 个 class positional embeddings；
5. 不加载官方 1000-class linear head；
6. MCTformer+ 的 3×3 convolutional patch head随机初始化；
7. backbone patch embedding、12 个 blocks 和 norm全部加载；
8. 不静默吞掉未知 missing/unexpected keys。

每次加载输出 `pretrained_load_report.json`，至少包含：

```json
{
  "variant": "tiny",
  "source_url": "...",
  "cache_path": "...",
  "source_sha256": "...",
  "source_embed_dim": 192,
  "target_embed_dim": 192,
  "source_depth": 12,
  "target_depth": 12,
  "loaded_key_count": 0,
  "loaded_numel": 0,
  "randomly_initialized_keys": [
    "head.weight",
    "head.bias"
  ],
  "unexpected_keys": [],
  "shape_mismatches": [],
  "passed": true
}
```

如果随机初始化 key 集合不是预注册集合，立即失败，不得继续训练。

# 6.3 Checkpoint 必须携带架构信息

在 best/final checkpoint 中新增：

```python
"model_spec": {
    "family": "MCTformer+",
    "variant": "tiny",
    "model_name": "mctformerplus_tiny",
    "patch_size": [16, 16],
    "embed_dim": 192,
    "depth": 12,
    "num_heads": 3,
    "head_dim": 64,
    "mlp_ratio": 4,
    "cam_class_to_patch_layers": 3,
    "cam_patch_to_patch_layers": 12,
},
"pretrained": {
    "url": "...",
    "filename": "...",
    "sha256": "..."
},
"training_spec": {
    "seed": 0,
    "micro_batch_size": 32,
    "accum_iter": 1,
    "effective_batch_size": 32,
    "nominal_lr": 0.0005,
    "optimizer_lr": 0.00003125,
    "epochs": 45
}
```

旧 Small checkpoint没有 `model_spec` 时：

- 只有 `--model mctformerplus` 才允许按 legacy Small 解析；
- 在新分析目录写 `legacy_small_import_manifest.json`；
- 不修改原 checkpoint；
- 不允许把缺少 spec 的 checkpoint猜成 Tiny/Base。

# 6.4 CAM / analysis 必须从 checkpoint解析变体

修改 `make_cam.py` 的顺序：

1. 先加载 checkpoint metadata；
2. 解析 CLI model 与 checkpoint `model_spec`；
3. 二者必须一致；
4. 再创建正确宽度的 CAM model；
5. strict load state dict；
6. 校验 method configuration；
7. 生成 CAM。

更新：

- `utils.create_cam_model`
- `tools/benchmark_mctformerplus.py`
- `tools/generate_mctformerplus_scale_cams.py`
- `analysis/lazy_assignment/run_class_specific_patch_score.py`
- 任何直接写死 `MCTformerPlusCam(...)` 的脚本

禁止用字符串包含关系把 Tiny/Base都解析成默认 Small。

# 6.5 Base 的有效 batch size与梯度累积

Small 已使用：

```text
micro batch = 32
accum = 1
effective batch = 32
```

Base 在 448 输入下很可能无法使用 micro batch 32。新增：

```text
--accum-iter
--val-batch-size
```

有效 batch定义：

\[
B_{\mathrm{eff}}
=
B_{\mathrm{micro}}
\times
N_{\mathrm{GPU}}
\times
N_{\mathrm{accum}}.
\]

主实验必须保持：

\[
B_{\mathrm{eff}}=32.
\]

Base preflight依次测试：

```text
micro=32, accum=1
micro=16, accum=2
micro=8,  accum=4
micro=4,  accum=8
micro=2,  accum=16
```

选择能够完成 forward、backward、optimizer step且留有合理显存余量的最大 micro batch。

梯度累积实现要求：

1. loss 在 backward 前除以 `accum_iter`；
2. 只在 accumulation boundary 执行：
   - gradient unscale；
   - gradient clipping；
   - optimizer step；
   - GradScaler update；
   - zero_grad；
3. 日志记录原始未除的 loss；
4. LR scaling 使用 effective batch，不使用 micro batch；
5. 每 epoch optimizer update 数与 Small 一致：

\[
U
=
\left\lfloor
\frac{N_{\mathrm{train}}}{32}
\right\rfloor.
\]

为了与 Small 的 `drop_last=True` 对齐，micro-batch runner 每 epoch只消费：

\[
U\times 32
\]

个样本，不允许因为 micro batch更小而额外消费尾部样本。

Tiny优先使用：

```text
micro=32, accum=1
```

# 6.6 Small 数值回归保护

在任何代码修改前：

1. 找到用户指定的已完成 Small final checkpoint；
2. 选择固定 VOC 图像；
3. 在 no-AMP、`eval()`、固定输入下运行：
   - class logits；
   - patch logits；
   - final CAM；
   - 可选 class-to-patch；
4. 保存：
   ```text
   small_prechange_regression.npz
   small_prechange_regression.json
   ```
5. 记录 checkpoint SHA-256、图像 ID、输入 tensor hash、代码 commit。

修改后用同一环境重新运行，要求：

```text
state-dict keys/shapes identical
strict checkpoint load passes
output shapes identical
max_abs_diff <= 1e-6
```

若不能通过，禁止启动 Tiny/Base full run，必须先定位 Small 行为改变的原因。

# 6.7 新增正确的分类 evaluator

建议新增：

```text
tools/evaluate_mctformerplus_classification.py
```

输入：

```text
--checkpoint
--model
--voc-root
--list-path val_id.txt
--input-size 448
--batch-size
--output-dir
```

输出：

```text
classification_metrics.json
classification_per_class.csv
classification_predictions.npz
```

至少计算：

```text
class_token_macro_class_AP
class_token_micro_AP
patch_branch_macro_class_AP
patch_branch_micro_AP
legacy_imagewise_AP
per-class AP
classification loss
```

所有 AP 均保存 0–1 原值和 0–100 百分比，字段名必须明确。

# 6.8 新增完整 CAM threshold evaluator

不要依赖当前可能提前停止的 legacy sweep作为唯一结果。

建议新增：

```text
tools/evaluate_cam_threshold_grid.py
```

完整评估：

```text
threshold = 0.00, 0.01, ..., 0.59
```

输出：

```text
threshold_curve.csv
threshold_curve.json
per_image_confusion.npz
best_threshold.json
```

tie-break规则预注册为：

1. mIoU 最大；
2. 若完全相同，选最小 threshold。

还应计算：

- threshold-curve AUC；
- 最佳 mIoU；
- 最佳 threshold；
- 距最佳结果 0.5 mIoU point以内的 threshold plateau width。

# 6.9 新增统一 aggregator

建议新增：

```text
tools/aggregate_mctformerplus_width_scaling.py
```

必须由用户显式传入三个结果目录：

```text
--tiny-run
--small-run
--base-run
--output-dir
```

禁止仅凭“最新目录”自动混合结果。

聚合前必须校验：

- dataset/list SHA-256一致；
- 20 类顺序一致；
- input size一致；
- scales一致；
- seed一致；
- epochs一致；
- effective batch一致；
- attention/BCSS/PSL/CTI configuration一致；
- CAM layer policy一致；
- checkpoint类型都是 final；
- 每个 stage存在完成标记；
- Tiny < Small < Base实际参数量成立；
- Small checkpoint/CAM来自用户确认的 canonical run。

---

# 7. 预计修改和新增的文件

Codex应先以最小修改原则核对，不要机械创建重复工具。

## 7.1 预计修改

```text
models/mctformer_plus.py
utils.py
train_model_v2.py
engine.py                         # 仅梯度累积确有需要时
make_cam.py
analysis/lazy_assignment/run_class_specific_patch_score.py
tools/benchmark_mctformerplus.py
tools/generate_mctformerplus_scale_cams.py
```

## 7.2 建议新增

```text
experiments/scaling/run_mctformerplus_width_voc.sh
tools/audit_mctformerplus_variant.py
tools/audit_completed_small_run.py
tools/evaluate_mctformerplus_classification.py
tools/evaluate_cam_threshold_grid.py
tools/aggregate_mctformerplus_width_scaling.py
tests/test_mctformerplus_variants.py
tests/test_mctformerplus_pretrained_loading.py
tests/test_gradient_accumulation.py
tests/test_width_scaling_aggregation.py
docs/MCTformerPlus_Tiny_Small_Base_Width_Scaling.md
```

不要修改原始：

```text
experiments/baselines/run_mctformerplus_voc.sh
```

除非只是修复一个经过证明会影响已完成 Small 可解释性的严重错误；这种情况必须单独报告，不能悄悄改动。

---

# 8. 单元测试与集成测试

# 8.1 架构测试

对 T/S/B分别断言：

```text
depth       12
patch       16
D           192 / 384 / 768
heads       3 / 6 / 12
head_dim    64
num classes 20
```

断言：

\[
P_T<P_S<P_B.
\]

记录实际：

```text
total_parameters
trainable_parameters
parameters_receiving_gradient
checkpoint_size_bytes
```

# 8.2 Forward shape测试

训练模型，输入小尺寸 smoke tensor：

```text
B = 2
input = 224 x 224
```

输出应为：

```text
class logits       [B, 20]
class embeddings   [12, B, 20, D]
patch logits       [B, 20]
```

CAM 模型：

```text
input 224 -> CAM [B, 20, 14, 14]
input 448 -> CAM [B, 20, 28, 28]
```

Base 的 448 test可用 `B=1`。

# 8.3 Attention与CAM策略测试

对三个变体断言：

```text
CAM class-to-patch uses last 3 layers
CAM patch-to-patch uses all 12 layers
attention head average behavior unchanged
```

Tiny/Small/Base不得出现不同的 layer count。

# 8.4 预训练加载测试

对每个官方 checkpoint：

- embed dim一致；
- depth一致；
- 12 blocks全部有对应权重；
- positional interpolation输出为 `[1, 784, D]`；
- class token为 `[1, 20, D]`；
- 随机初始化 key只包含允许集合；
- 无 unexpected key；
- 无 silent shape mismatch；
- tensor均为 finite。

# 8.5 Checkpoint round-trip

对每个变体：

1. 随机初始化；
2. 保存带 `model_spec` checkpoint；
3. 从 checkpoint重新构造；
4. strict load；
5. 固定输入；
6. 输出数值一致。

# 8.6 Variant mismatch 必须失败

以下情况必须显式报错：

```text
Tiny checkpoint + Small CLI
Base checkpoint + Tiny CLI
Small checkpoint + Base CLI
checkpoint embed_dim与spec不一致
checkpoint缺spec但CLI声称Tiny/Base
```

# 8.7 梯度累积测试

构建小型、无 dropout、无 AMP 或可控 AMP 的 deterministic test：

- batch 4、accum1；
- micro batch 2、accum2；
- 使用同一初始权重、同一 4 个样本；
- 比较一次 optimizer update后的参数。

在合理数值容差内一致。

还要断言：

```text
optimizer step count一致
scheduler step count一致
zero_grad boundary正确
reported effective batch正确
LR scaling使用effective batch
```

# 8.8 Small regression test

必须完成第 6.6 节 pre/post 数值对照。

# 8.9 运行测试顺序

```bash
python -m pytest -q tests/test_mctformerplus_variants.py
python -m pytest -q tests/test_mctformerplus_pretrained_loading.py
python -m pytest -q tests/test_gradient_accumulation.py
python -m pytest -q tests/test_width_scaling_aggregation.py
python -m pytest -q tests/test_mctformerplus_attention.py
python -m pytest -q
```

完整测试失败时，不得通过跳过测试进入 full training。

---

# 9. 已完成 Small 的审计和复用

由用户或运行环境设置：

```bash
export TGCA_SMALL_RUN_DIR=/absolute/path/to/completed/small/run
```

不得把占位符直接执行。

建议 `tools/audit_completed_small_run.py` 检查：

```text
mctformerplus_final.pth
checkpoint_manifest.txt
dataset_manifest.txt
git_state.json
environment.txt
hardware.txt
pipeline.log
metrics.json
cam_train/
```

并核对：

1. `pipeline.log` 包含 `PIPELINE_COMPLETE`；
2. checkpoint SHA-256 与 manifest一致；
3. `cam_train/*.npy` 数量等于 `train_id.txt` 非空 ID 数；
4. dataset/list hash一致；
5. seed=0；
6. input=448；
7. epochs=45；
8. attention=vanilla；
9. BCSS=e0；
10. PSL=baseline；
11. CTI-BGT=False；
12. checkpoint为 final；
13. model state对应 D=384、depth=12、heads=6。

输出：

```text
small_run_audit.json
small_run_pointer.json
```

`small_run_pointer.json` 只记录绝对路径和 hashes，不复制、不移动、不改写 Small checkpoint/results。

如果存在多个候选 Small run：

- 列出候选；
- 不自动取最新；
- 只有 `TGCA_SMALL_RUN_DIR` 指定的目录才进入正式聚合。

如果 canonical Small 缺少必要 provenance：

- 不覆盖它；
- 将其标记为 `legacy_incomplete_provenance`；
- 只有在无法建立数值兼容性时才安排 matched Small rerun。

---

# 10. 实验执行阶段

# Phase 0：只读启动审计

在 LHR：

```bash
cd /home/peng/code/TGCA
pwd
git status --short --branch
git log -8 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
tmux ls
nvidia-smi
df -h .
```

要求：

- 当前工作分支为 `main`；
- 不清理已有文件；
- 不停止已有 tmux/job；
- 检查远端 `main` 与本地状态；
- 只有在没有运行任务依赖当前 worktree且可安全 fast-forward 时，才更新本地；
- 记录 live commit，不能把本计划撰写时的 commit当成最终实验 commit。

读取：

```text
AGENTS.md
docs/design.md
```

`AGENTS.md` 提到但 live worktree中不存在的文档只记录缺失，不得擅自重建旧文件。

# Phase 1：Small pre-change regression与canonical run审计

完成第 6.6 和第 9 节。

Go/No-Go：

```text
Small checkpoint strict load：PASS
Small CAM count：PASS
Small provenance：PASS 或明确 legacy标记
Small pre-change output artifact：已保存
```

# Phase 2：代码实现

按以下顺序：

1. variant registry；
2. Tiny/Base timm factories；
3. shared DeiT checkpoint adapter；
4. checkpoint `model_spec`；
5. variant-aware CAM construction；
6. gradient accumulation；
7. classification evaluator；
8. exhaustive threshold evaluator；
9. aggregator；
10. analysis/benchmark工具参数化；
11. runner；
12. tests。

禁止先启动训练再补 metadata/evaluator。

# Phase 3：测试、本地代码快照和 Small post-change regression

1. 运行全部测试；
2. 完成 Small post-change regression；
3. 检查 diff；
4. 确认无无关改动；
5. 创建本地 commit以固定 full-run code；
6. 记录完整 commit SHA。

禁止 push。

# Phase 4：机械 smoke test

Tiny和Base都做：

```text
2–5 optimizer updates
固定小数据子集
checkpoint保存
checkpoint strict reload
1–4 张图 CAM
classification evaluator
threshold evaluator
无 NaN/Inf
```

smoke 目录名称必须包含 `smoke`，不得进入正式结果表。

Base 同时完成 micro-batch capacity probe并选定 accumulation配置。

# Phase 5：正式 Tiny seed-0

建议目标 runner接口：

```bash
bash experiments/scaling/run_mctformerplus_width_voc.sh \
  --variant tiny \
  --seed 0 \
  --gpu 0 \
  --stage all
```

runner应自动：

1. preflight；
2. train；
3. evaluate classification；
4. generate train CAM；
5. generate val CAM；
6. exhaustive threshold eval；
7. fixed-threshold metrics；
8. benchmark；
9. hash/audit；
10. 写 `PIPELINE_COMPLETE`。

# Phase 6：正式 Base seed-0

```bash
bash experiments/scaling/run_mctformerplus_width_voc.sh \
  --variant base \
  --seed 0 \
  --gpu 0 \
  --micro-batch <preflight-result> \
  --accum-iter <preflight-result> \
  --stage all
```

有效 batch必须为32。

Tiny先运行的原因：成本低，能够先暴露 model registry、checkpoint、CAM 和 evaluator 的通用错误。

# Phase 7：对已有 Small 做统一新评估

不重训练 Small，使用 canonical final checkpoint：

1. 运行正确的 classification evaluator；
2. 复用已有 `cam_train`；
3. 生成 `cam_val`；
4. 运行 exhaustive threshold grid；
5. 运行固定阈值指标；
6. 在当前同一 GPU上重新 benchmark；
7. 运行 semantic ownership diagnostics；
8. 将所有新结果写入独立的 `small_reanalysis` 目录，不写入旧 run。

# Phase 8：机制分析

参数化已有 lazy-assignment Experiment 1/2工具，使其支持 T/S/B。

统一使用：

```text
VOC val
input 448
no random augmentation
all 12 blocks
same list hash
same checkpoint type
```

输出主指标：

\[
\mathrm{C\text{-}PiM}
=
P(j_c^*\in\Omega_c),
\]

\[
P(j_c^*\in\Omega_{\mathrm{other}}),
\qquad
P(j_c^*\in\Omega_{\mathrm{bg}}),
\]

\[
\mathrm{BG\text{-}Tail@5},
\qquad
\mathrm{BG\text{-}Tail@10},
\]

\[
\Delta_{\mathrm{target-bg}}
=
E[S_{c,j}\mid j\in\Omega_c]
-
E[S_{c,j}\mid j\in\Omega_{\mathrm{bg}}].
\]

同时输出 layer-wise curves：

```text
C-PiM(l)
BG top-1 hit(l)
BG-Tail@5(l)
BG-Tail@10(l)
target score mean/median/q90(l)
other-FG score mean/median/q90(l)
background score mean/median/q90/q95(l)
```

# Phase 9：聚合与报告

目标接口：

```bash
python tools/aggregate_mctformerplus_width_scaling.py \
  --tiny-run  /absolute/path/to/tiny/run \
  --small-run "$TGCA_SMALL_RUN_DIR" \
  --small-reanalysis /absolute/path/to/small/reanalysis \
  --base-run  /absolute/path/to/base/run \
  --output-dir /absolute/path/to/aggregate/result
```

只有全部输入审计通过后才能生成 final table。

---

# 11. 分类评估协议

# 11.1 主 checkpoint

为保持与 CAM 一致，主比较使用：

```text
final epoch checkpoint
```

同时可报告：

```text
best-validation checkpoint
```

但不得让不同模型分别选对自己最有利的 checkpoint作为唯一主结果。

# 11.2 主指标

VOC val上报告：

| 指标 | 分支 | 角色 |
|---|---|---|
| Macro class AP | class-token logits | 分类主指标 |
| Macro class AP | patch-GWRP logits | 分支诊断 |
| Micro AP | class-token logits | 辅助 |
| Legacy image-wise AP | class-token logits | 与旧日志回归 |
| Per-class AP | 两个分支 | 类别级分析 |

# 11.3 置信区间

使用 image bootstrap：

```text
resamples = 10000
seed = 2027
```

三个模型共享相同 resampling indices，生成 paired differences：

```text
Tiny - Small
Base - Small
Base - Tiny
```

注意：image bootstrap只反映 evaluation-set sampling uncertainty，不能替代 training-seed variance。

---

# 12. Raw CAM 评估协议

# 12.1 两个数据 split

## VOC train

作用：

- 与现有 Small结果和原始 WSSS pseudo-label generation对齐；
- 在 Small train上确定一个 calibration threshold。

## VOC val

作用：

- 作为跨模型 localization generalization主比较；
- 避免只在用于优化的训练图像上解释 scaling。

# 12.2 三类阈值结果

## A. 固定 0.45

所有模型、所有 split都报告：

```text
mIoU@0.45
foreground mIoU@0.45
precision/recall@0.45
background false-positive rate@0.45
```

这是最简单、完全统一的跨模型阈值。

## B. Small-calibrated threshold \(\tau_S\)

1. 在 canonical Small `cam_train` 上做完整 0.00–0.59 sweep；
2. 选择 \(\tau_S\)；
3. 在看到 Tiny/Base val最佳阈值前冻结；
4. 将同一个 \(\tau_S\) 应用于 T/S/B train和val。

这是主 generalization threshold之一。

## C. 每模型最佳 threshold

在完整 curve上报告：

```text
best_grid_mIoU
best_grid_threshold
```

但明确标记为：

```text
oracle / sensitivity diagnostic
```

不能将各模型各自调优后的最佳值作为唯一 scaling证据。

# 12.3 主 CAM 表

至少生成：

| Variant | Params | Train mIoU@0.45 | Train mIoU@\(\tau_S\) | Val mIoU@0.45 | Val mIoU@\(\tau_S\) | Val oracle best | Oracle threshold |
|---|---:|---:|---:|---:|---:|---:|---:|

同时报告：

- foreground mIoU；
- semantic FG precision；
- semantic FG recall；
- binary FG precision；
- binary FG recall；
- background false-positive rate；
- threshold-curve AUC；
- plateau width。

# 12.4 Paired bootstrap

为每张图保存 21×21 confusion counts。  
使用同一 image bootstrap indices计算：

\[
\Delta\mathrm{mIoU}_{B-S},
\quad
\Delta\mathrm{mIoU}_{S-T},
\quad
\Delta\mathrm{mIoU}_{B-T}.
\]

报告 95% CI。

---

# 13. 效率与资源评估

在同一 GPU、同一软件环境、同一输入设置下重新 benchmark三个 checkpoint：

```text
input 448
batch 1
warmup 20
measured iterations 100
AMP设置固定
```

记录：

```text
actual total params
actual trainable params
checkpoint size
peak training allocated memory
peak training reserved memory
training images/s
optimizer updates/s
mean epoch time
total training wall time
CAM latency mean/median/p95
CAM throughput
inference peak memory
```

如果 Small历史训练使用不同 GPU，训练速度只作带注释的历史值；推理 benchmark必须在同一当前 GPU重测。

FLOPs/MACs：

- 优先使用当前环境已有、不会改变依赖栈的可靠方法；
- 若工具对 attention matmul或自定义 CAM漏计，必须标注；
- 不为了一个不可靠 FLOPs数字升级工作环境；
- 实测 latency/memory是强制项。

---

# 14. 结果目录设计

已有 Small保持原位置不变。

建议新目录：

```text
results/
└── mctformerplus_width_scaling/
    └── voc/
        ├── references/
        │   ├── small_run_pointer.json
        │   ├── small_run_audit.json
        │   └── small_prechange_regression.npz
        ├── tiny/
        │   └── <run_id>/
        ├── base/
        │   └── <run_id>/
        ├── small_reanalysis/
        │   └── <analysis_id>/
        └── aggregate/
            └── <analysis_id>/
```

每个正式 run至少包含：

```text
command.txt
git_state.json
environment.txt
pip_freeze.txt
conda_explicit.txt
hardware.txt
dataset_manifest.txt
model_spec.json
pretrained_manifest.txt
pretrained_load_report.json
optimizer_spec.json
training.log
classification_metrics.json
classification_per_class.csv
checkpoint_manifest.txt
threshold_curve_train.csv
threshold_curve_val.csv
fixed_threshold_metrics/
benchmark.json
audit_report.json
PIPELINE_COMPLETE
```

大型 CAM和 checkpoint目录也必须有自己的完成标记，避免把部分生成结果当作完整结果。

---

# 15. 聚合输出

`aggregate/` 必须生成：

```text
scaling_summary.csv
scaling_summary.json
classification_by_class.csv
cam_threshold_summary.csv
paired_bootstrap_differences.csv
efficiency_summary.csv
semantic_ownership_summary.csv
audit_report.json
REPORT.md
```

图表：

```text
01_params_vs_classification_macro_ap.pdf/png
02_params_vs_val_cam_miou_fixed_threshold.pdf/png
03_classification_vs_localization.pdf/png
04_params_vs_c_pim.pdf/png
05_params_vs_bg_tail.pdf/png
06_layerwise_c_pim.pdf/png
07_layerwise_bg_tail.pdf/png
08_accuracy_efficiency_frontier.pdf/png
09_cam_threshold_curves.pdf/png
```

横轴参数量使用：

\[
\log_{10}(\text{actual parameters}).
\]

所有图同时保存矢量 PDF和便于查看的 PNG。

---

# 16. Go / No-Go gates

## Gate 1：实现正确性

进入 full training前必须全部满足：

```text
T/S/B architecture tests PASS
official pretrain audit PASS
Tiny/Base smoke PASS
Small pre/post numerical regression PASS
full pytest PASS
no uncommitted tracked source in scientific run
```

## Gate 2：结果完整性

某一模型进入聚合前必须满足：

```text
final checkpoint exists and hashes match
strict checkpoint reload PASS
classification output complete
train CAM count complete
val CAM count complete
60 thresholds complete
fixed-threshold metrics complete
no NaN/Inf
benchmark complete
PIPELINE_COMPLETE exists
```

## Gate 3：跨模型可比性

以下任一不一致时，aggregator必须失败：

```text
dataset/list hash
seed
effective batch
epochs
input size
CAM scales
attention mode
method variants
CAM layer policy
final-vs-best checkpoint policy
class ordering
```

## Gate 4：是否追加 3 seeds

满足下列任一情况，就值得补齐 seeds 1/2：

1. `|Base - Small|` 的 val raw-CAM mIoU@固定阈值 ≥ 0.5 point；
2. classification与localization方向相反；
3. Base分类提升但 CAM不提升；
4. semantic ownership指标与 CAM方向相反；
5. Tiny意外接近或超过 Small；
6. 单 seed结果将被写入论文主结论。

追加时必须是完整的：

```text
T/S/B × seeds 0,1,2
```

## Gate 5：是否运行下游 segmentation

满足以下条件之一：

- raw CAM存在清晰单调趋势；
- 分类–定位 decoupling明显；
- Base CAM不升但 semantic ownership显著变化；
- 审稿叙事需要证明 raw seed变化是否传递到最终 mask。

下游必须固定同一 pseudo-label、refinement、segmentation recipe和threshold policy。

---

# 17. 可选优化 sanity-check

主结果必须首先使用原始 Small recipe。

只有 Base出现下列工程异常时才做优化 sanity-check：

```text
loss diverges
NaN/Inf
classification macro AP明显低于Tiny/Small
45 epoch仍明显未收敛
```

可预注册一个独立、明确标记的 secondary study：

```text
nominal lr: 2.5e-4, 5e-4, 1e-3
drop_path: 0.1, 0.2
```

先做短 pilot，所有 tuned结果必须与 matched-recipe结果分表报告。  
不得用 Base tuned结果对比 Tiny/Small untuned结果后声称纯架构 scaling。

---

# 18. Codex 最终交付报告模板

Codex完成后必须给用户报告：

## 18.1 代码

```text
live main starting commit
scientific-run commit
changed files
new files
tests and exact commands
Small numerical regression max_abs_diff
```

## 18.2 Small复用

```text
canonical Small run path
checkpoint SHA-256
original commit / dirty state
audit status
whether retraining was avoided
```

## 18.3 每个新模型

```text
run ID
variant/model name
actual params
micro batch / accum / effective batch
optimizer LR
pretrain URL and SHA-256
checkpoint SHA-256
training completion
classification metrics
train/val CAM metrics
benchmark
```

## 18.4 汇总

必须给出 T/S/B主表及：

```text
Tiny -> Small delta
Small -> Base delta
paired 95% CI
classification–localization relationship
semantic ownership relationship
efficiency cost
```

## 18.5 结论

必须明确区分：

- 已完成、经过审计的证据；
- 单 seed观察；
- image bootstrap uncertainty；
- 尚未完成的 training-seed variance；
- 可支持的 claim；
- 不可支持的 scaling-law claim。

---

# 19. Codex 执行清单

## 启动

- [ ] 位于 `/home/peng/code/TGCA`
- [ ] 当前分支是 `main`
- [ ] 读取 `AGENTS.md`
- [ ] 读取 `docs/design.md`
- [ ] 检查 Git、tmux、GPU、磁盘
- [ ] 不干扰已有运行
- [ ] 设置 `TGCA_SMALL_RUN_DIR`

## Small保护

- [ ] 审计已有 Small run
- [ ] 保存 pre-change numerical artifact
- [ ] 不覆盖 Small结果
- [ ] 不默认重训 Small

## 实现

- [ ] 添加统一 variant registry
- [ ] 添加 Tiny/Base factory
- [ ] 统一预训练适配
- [ ] 添加 `model_spec`
- [ ] CAM/benchmark/analysis变体化
- [ ] 添加有效 batch梯度累积
- [ ] 添加正确 class-wise AP evaluator
- [ ] 添加 exhaustive threshold evaluator
- [ ] 添加显式三目录 aggregator
- [ ] 添加测试

## 验证

- [ ] 全部 tests PASS
- [ ] Small post-change numerical regression PASS
- [ ] 本地 commit固定科学运行代码
- [ ] Tiny smoke PASS
- [ ] Base smoke PASS
- [ ] Base micro-batch probe完成

## 正式运行

- [ ] Tiny seed-0 complete
- [ ] Base seed-0 complete
- [ ] Small统一 reanalysis complete
- [ ] 三模型 train CAM complete
- [ ] 三模型 val CAM complete
- [ ] 三模型 classification complete
- [ ] 三模型 threshold grid complete
- [ ] 三模型 fixed-threshold metrics complete
- [ ] 三模型 benchmark complete
- [ ] 三模型 semantic ownership diagnostics complete

## 聚合

- [ ] 所有 hashes/config一致
- [ ] paired bootstrap完成
- [ ] summary CSV/JSON完成
- [ ] vector plots完成
- [ ] `REPORT.md`完成
- [ ] 结论没有越过证据边界

---

# 20. 最终验收标准

本任务只有同时满足以下条件才算完成：

1. 原始 `mctformerplus` Small checkpoint在新代码中严格加载，并通过数值回归；
2. Tiny、Small、Base构造和 checkpoint解析没有宽度硬编码；
3. Tiny/Base使用各自正确的官方 non-distilled ImageNet-1K DeiT权重；
4. Base使用梯度累积时保持 effective batch=32、optimizer update数和 LR policy与Small一致；
5. Tiny和Base各自完成45 epochs seed-0训练；
6. 已完成 Small不被覆盖，并完成统一新 evaluator；
7. 三者在同一 train/val lists、同一 CAM protocol下完成 raw CAM；
8. 主结果至少包含固定 0.45 和 Small-calibrated \(\tau_S\)，而非仅各模型oracle threshold；
9. 分类主指标使用真正的 dataset-level class-wise macro AP；
10. 所有结果可追溯到 commit、命令、环境、数据 hash、预训练 hash和 checkpoint hash；
11. 生成性能、机制和效率三个层面的统一报告；
12. 最终结论使用“width scaling behavior/trend”，不把三个点包装成严格 scaling law。

---

# 21. 参考入口

## TGCA

```text
https://github.com/jungletada/TGCA
https://github.com/jungletada/TGCA/blob/main/AGENTS.md
https://github.com/jungletada/TGCA/blob/main/models/mctformer_plus.py
https://github.com/jungletada/TGCA/blob/main/train_model_v2.py
https://github.com/jungletada/TGCA/blob/main/make_cam.py
https://github.com/jungletada/TGCA/blob/main/utils.py
https://github.com/jungletada/TGCA/blob/main/engine.py
https://github.com/jungletada/TGCA/blob/main/experiments/baselines/run_mctformerplus_voc.sh
```

## Official DeiT

```text
https://github.com/facebookresearch/deit
https://github.com/facebookresearch/deit/blob/main/models.py
https://github.com/facebookresearch/deit/blob/main/README_deit.md
```
