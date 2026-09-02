# Experiment 1：Class-specific Patch Score  
## LaST-ViT → MCTformer / MCTformer+ 诊断实验执行计划（Codex 版）

> **目标仓库：** https://github.com/jungletada/TGCA  
> **目标分支：** 以执行时的 `main` 或用户指定工作分支为准  
> **实验对象：** 已训练好的 PASCAL VOC 2012 MCTformer 与 MCTformer+ checkpoints  
> **本阶段范围：** 只实现和验证 **Experiment 1：Class-specific Patch Score**  
> **明确不做：** 三区域 distribution、C-PiM、BG-Tail、\(A_{c2p}\)、CAM 对比、register、P2C blocking、LaST selective aggregation、新训练或新方法。

---

# 1. 研究目的

本实验的唯一目标是建立一个**不改变模型输出的分析通路**，从 MCTformer 和 MCTformer+ 的每个 Transformer block 中提取：

\[
C^{(l)}
=
[c^{(l)}_1,\ldots,c^{(l)}_{N_c}]
\in
\mathbb{R}^{B\times N_c\times D},
\]

\[
P^{(l)}
=
[p^{(l)}_1,\ldots,p^{(l)}_{N_p}]
\in
\mathbb{R}^{B\times N_p\times D},
\]

并计算每个类别 token 与每个 patch token 的 cosine similarity：

\[
\boxed{
S^{(l)}_{b,c,j}
=
\cos
\left(
c^{(l)}_{b,c},
p^{(l)}_{b,j}
\right)
}
\]

得到：

\[
S^{(l)}
\in
\mathbb{R}^{B\times N_c\times N_p}.
\]

本阶段只回答以下工程和测量问题：

1. 能否从 MCTformer 和 MCTformer+ 的全部 12 个 blocks 中稳定提取 class/patch hidden states？
2. Class-specific Patch Score 是否按预期产生：
   \[
   [B,N_c,N_p]
   \]
   的 score tensor？
3. 分析 hook 是否完全不改变原模型的 logits、CAM 或 hidden state？
4. 对 VOC 正类别，是否可以生成每层 class-specific patch-score map，并保存为后续实验可直接使用的数据？

本阶段**不据此宣称 lazy semantic assignment 已被证明**。只有后续结合 GT region、C-PiM、BG-Tail、\(A_{c2p}\) 和 CAM 后，才能判断高分 patch 是否真正落在背景。

---

# 2. LaST-ViT 的简单解读

## 2.1 LaST-ViT 的研究思想

LaST-ViT 的核心观察是：

> 在 coarse-grained supervision 和 global self-attention 下，ViT 可能让本来缺乏判别意义的背景 patch representation 变得与全局语义表示高度相似，从而形成 lazy aggregation / background shortcut。

LaST-ViT 没有首先用 attention weight 定义 patch 重要性，而是定义 representation-level **Patch Score**：

\[
S_j
=
\cos
\left(
p_j,
q_{\mathrm{CLS}}
\right),
\]

其中：

- \(p_j\) 是最终 patch token；
- \(q_{\mathrm{CLS}}\) 是最终 CLS token；
- score 越高，表示该 patch representation 与 global representation 越接近。

LaST-ViT 再检查高分 patch 是否落在目标区域，从而分析 global semantics 是否被错误分配给背景 patch。

## 2.2 LaST-ViT 源码入口

Codex 开始实现前，应先阅读以下官方源码：

### 仓库与总说明

- Repository  
  https://github.com/ChengShiest/LAST-ViT
- README  
  https://github.com/ChengShiest/LAST-ViT/blob/main/README.md

### Patch Score

- `visualization/patch_score.py`  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/patch_score.py

核心代码思想：

```python
patch_scores = torch.cosine_similarity(
    patch_tokens,
    cls_token.unsqueeze(1).expand_as(patch_tokens),
    dim=-1,
)
```

### Point-in-BBox 验证

- `visualization/evaluate_patch_hit.py`  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/evaluate_patch_hit.py

其逻辑是对 Patch Score 取 top-1：

```python
top1 = scores.argmax(dim=1)
```

再判断该 patch 是否位于目标 bbox。

### FG/BG 分布可视化

- `visualization/visualize_patch_score_distribution.py`  
  https://github.com/ChengShiest/LAST-ViT/blob/main/visualization/visualize_patch_score_distribution.py

该文件负责按 foreground/background mask 收集 Patch Score。**本阶段不搬它的 region statistics，只借鉴其结果保存和 visualization 组织方式。**

### LaST-ViT selective aggregation

- `cls_pretrain/conf.py`  
  https://github.com/ChengShiest/LAST-ViT/blob/main/cls_pretrain/conf.py

该文件包含 FFT、Gaussian low-pass、stability score 和 channel-wise top-k selection。**本阶段禁止移植这一部分。**我们现在只验证 Patch Score，不测试 LaST-ViT 的解决方案。

---

# 3. 如何迁移到 MCTformer

普通 ViT 只有一个：

\[
q_{\mathrm{CLS}}.
\]

MCTformer / MCTformer+ 有 \(N_c\) 个 class tokens：

\[
C=
[c_1,\ldots,c_{N_c}].
\]

因此把 LaST-ViT 的单 CLS Patch Score 扩展为：

\[
\boxed{
S^{(l)}_{c,j}
=
\cos
\left(
c^{(l)}_c,
p^{(l)}_j
\right)
}
\]

这不是 generic global similarity，而是**类别特异的 patch semantic alignment**。

例如：

\[
S^{(l)}_{\mathrm{boat},j}
\]

表示第 \(l\) 层 patch \(j\) 的 representation 与 `boat` class token 的相似程度。

此扩展允许后续分别研究：

- 正确目标 patch 是否与对应 class token 对齐；
- 其他前景 patch 是否产生 class confusion；
- background patch 是否获得 class-specific semantics；
- 该现象是否随 Transformer depth 增强。

但这些 region-level 判断都留到 Experiment 2。本阶段只生成可信、可复现的 \(S^{(l)}_{c,j}\)。

---

# 4. 当前 TGCA 仓库的相关代码结构

目标仓库：

- https://github.com/jungletada/TGCA

Codex 必须先读以下文件：

## 4.1 Native MCTformer

- `models/mctformer.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mctformer.py

当前 `MCTformerV2.forward_features()`：

1. 生成 patch tokens；
2. 将 \(N_c\) 个 class tokens 放在 patch tokens 前面；
3. 依次执行 `self.blocks`；
4. 最终按：
   ```python
   x[:, :self.num_classes]
   x[:, self.num_classes:]
   ```
   切分 class tokens 和 patch tokens；
5. 返回最终 tokens 与 attention weights。

因此 token layout 是：

```text
[class_0, ..., class_(C-1), patch_0, ..., patch_(N-1)]
```

## 4.2 Native MCTformer+

- `models/mctformer_plus.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mctformer_plus.py

当前 `MCTformerPlus.forward_features()` 同样使用：

```text
[class tokens, patch tokens]
```

并已保存每一层的：

```python
all_x_cls.append(x[:, :self.num_classes])
```

但它**没有保存每一层 patch tokens**。因此仅使用现有 `return_token=True` 不足以计算 layer-wise Class-specific Patch Score。

## 4.3 Shared native ViT

- `models/vit.py`  
  https://github.com/jungletada/TGCA/blob/main/models/vit.py

`MCTformerV2` 和 `MCTformerPlus` 都从该文件导入：

```python
from models.vit import VisionTransformer
```

其中：

- `Attention.forward()` 返回：
  ```python
  return x, weights
  ```
- `Block.forward()` 返回：
  ```python
  return x, weights
  ```

因此对：

```python
model.blocks[l]
```

注册 forward hook 时，block output 是 tuple：

```python
output[0]  # post-block token sequence
output[1]  # attention weights
```

## 4.4 重要警告：不要使用 `models/mct_vit.py`

仓库中另有：

- `models/mct_vit.py`  
  https://github.com/jungletada/TGCA/blob/main/models/mct_vit.py

该文件包含 Split Weighted Softmax 相关实现，不是当前 native MCTformer / MCTformer+ 所使用的共享 backbone。

Experiment 1 必须分析：

```text
models/mctformer.py
models/mctformer_plus.py
models/vit.py
```

禁止误切到：

```text
models/mct_vit.py
```

## 4.5 Model factory

- `utils.py`  
  https://github.com/jungletada/TGCA/blob/main/utils.py

当前 `create_cam_model(args)` 中：

```text
mctformerv2    → MCTformerV2Cam
mctformerplus  → MCTformerPlusCam
```

分析脚本应复用这个 factory，避免建立第二套 model construction logic。

## 4.6 Checkpoint loading

- `make_cam.py`  
  https://github.com/jungletada/TGCA/blob/main/make_cam.py

当前 CAM 代码使用：

```python
model_dict = torch.load(args.checkpoint)
if "model" in model_dict:
    model_dict = model_dict["model"]
model.load_state_dict(model_dict)
```

Experiment 1 应复用同样的 checkpoint 解析逻辑，并默认 `strict=True`。如果 checkpoint 存在 `module.` prefix，先记录并显式处理，不要静默 `strict=False`。

## 4.7 VOC 数据与 transform

- `datasets_cam.py`  
  https://github.com/jungletada/TGCA/blob/main/datasets_cam.py

其中已有：

- `VOC12Dataset`
- image-level class labels；
- deterministic validation transform；
- ImageNet mean/std。

Experiment 1 只需要 image、image-level label 和 image ID，不需要 segmentation GT。

---

# 5. 实现原则

## 5.1 优先使用 forward hook，不修改模型数学路径

推荐新增一个 analysis-only collector：

```python
class BlockTokenCollector:
    ...
```

对：

```python
model.blocks
```

注册 `register_forward_hook`。

Hook 逻辑：

```python
def hook(module, inputs, output):
    tokens = output[0]  # because Block returns (x, weights)
    ...
```

这样可以统一支持：

- `MCTformerV2Cam`
- `MCTformerPlusCam`

且无需修改两个模型的 `forward_features()`。

## 5.2 不允许 hook 改写 output

Hook 只能：

- read；
- detach；
- compute score；
- store result。

禁止：

- inplace 修改 token；
- return replacement output；
- 修改 attention；
- 调用 dropout；
- 写回 model state。

## 5.3 单 GPU、eval、inference mode

本阶段使用：

```python
model.eval()
with torch.inference_mode():
    ...
```

不要使用 DDP/DataParallel，避免 hook 在 replica 上产生复杂行为。

推荐：

```text
GPU count = 1
batch size = 4 or 8
```

---

# 6. 推荐代码结构

新增：

```text
analysis/
└── lazy_assignment/
    ├── README.md
    ├── run_class_specific_patch_score.py
    ├── token_collector.py
    ├── score_utils.py
    ├── voc_score_dataset.py
    ├── visualize_patch_score.py
    └── tests/
        ├── test_token_collector.py
        ├── test_class_patch_score.py
        ├── test_checkpoint_loader.py
        └── test_no_numerical_change.py
```

本阶段不创建 distribution、C-PiM、BG-Tail 或 CAM comparison 文件。

---

# 7. Data Loader

## 7.1 分析 split

默认：

```text
PASCAL VOC 2012 validation set
```

推荐参数：

```text
voc_root = data/VOCdevkit/VOC2012
list_path = data/VOCdevkit/VOC2012/ImageLists/val_id.txt
input_size = 448
num_classes = 20
```

实际路径由 Codex 在服务器中确认，不硬编码到源码。

## 7.2 Analysis-specific dataset

当前 `VOC12Dataset.__getitem__()` 只返回：

```python
img, label
```

为了保存 per-image 结果，应新建一个轻量 wrapper，返回：

```python
{
    "name": image_id,
    "image": image_tensor,
    "label": image_level_label,
}
```

该 dataset 应复用 `datasets_cam.py` 中：

- `load_img_name_list`
- `load_image_label_list_from_npy_voc`
- `build_transform`

不要重新发明 normalization。

## 7.3 固定 transform

第一阶段只用：

```text
scale = 1.0
flip = False
random augmentation = False
center crop = 448
```

调用：

```python
build_transform(
    is_train=False,
    make_cam=False,
    args=args,
)
```

这样会使用当前 repo 的 deterministic：

```text
Resize → CenterCrop → ToTensor → Normalize
```

---

# 8. Model Loading

分析脚本参数至少包括：

```text
--model
--checkpoint
--voc-root
--list-path
--input-size
--batch-size
--num-workers
--device
--output-dir
--limit
--save-all-classes
--save-visualizations
```

模型名称：

```text
mctformerv2
mctformerplus
```

通过：

```python
model = create_cam_model(args)
```

创建。

Checkpoint：

```python
payload = torch.load(checkpoint, map_location="cpu")
state_dict = payload["model"] if "model" in payload else payload
model.load_state_dict(state_dict, strict=True)
```

必须打印：

```text
model class
checkpoint path
checkpoint SHA256
num_classes
input_size
patch_size
depth
embed_dim
number of parameters
```

---

# 9. BlockTokenCollector

## 9.1 Hook registration

```python
class BlockTokenCollector:
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.handles = []
        self.scores = []

    def register(self):
        for layer_idx, block in enumerate(self.model.blocks):
            handle = block.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.handles.append(handle)
```

## 9.2 Hook operation

推荐在 hook 内直接计算 score，而不是保存全部 hidden states：

```python
def _hook(module, inputs, output):
    tokens = output[0]
    cls_tokens = tokens[:, :num_classes]
    patch_tokens = tokens[:, num_classes:]

    cls_norm = F.normalize(cls_tokens.float(), dim=-1)
    patch_norm = F.normalize(patch_tokens.float(), dim=-1)

    score = torch.einsum(
        "bcd,bnd->bcn",
        cls_norm,
        patch_norm,
    )

    self.scores[layer_idx] = score.detach().cpu()
```

这样避免保存：

\[
12\times B\times(C+N)\times D
\]

的大量 hidden states。

## 9.3 Precision

即使模型在 AMP 下运行，cosine 计算也强制转换为：

```python
float32
```

防止半精度归一化误差。

## 9.4 Hook cleanup

必须使用 context manager 或 `try/finally`：

```python
collector.remove()
```

避免同一模型多次运行时重复注册 hook。

---

# 10. Class-specific Patch Score 定义

对于 block \(l\) 的 post-block token：

\[
X^{(l)}
\in
\mathbb{R}^{B\times(N_c+N_p)\times D},
\]

切分：

\[
C^{(l)}
=
X^{(l)}[:,:N_c,:],
\]

\[
P^{(l)}
=
X^{(l)}[:,N_c:,:].
\]

L2 normalize：

\[
\bar C^{(l)}_{b,c}
=
\frac{C^{(l)}_{b,c}}
{\|C^{(l)}_{b,c}\|_2+\epsilon},
\]

\[
\bar P^{(l)}_{b,j}
=
\frac{P^{(l)}_{b,j}}
{\|P^{(l)}_{b,j}\|_2+\epsilon}.
\]

Score：

\[
\boxed{
S^{(l)}_{b,c,j}
=
\bar C^{(l)}_{b,c}
\cdot
\bar P^{(l)}_{b,j}
}
\]

PyTorch reference：

```python
cls_tokens = F.normalize(cls_tokens.float(), dim=-1)
patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
scores = torch.einsum("bcd,bnd->bcn", cls_tokens, patch_tokens)
```

Expected range：

\[
S^{(l)}_{b,c,j}\in[-1,1].
\]

---

# 11. Layer convention

使用所有 12 个 Transformer blocks 的 **post-block output**：

```text
layer_01 = output after block 0
...
layer_12 = output after block 11
```

不要将输入 token embedding 记作 layer 1。

当前 `MCTformerV2.forward_features()` 和 `MCTformerPlus.forward_features()` 在自定义 override 中没有对最终 tokens 额外执行 `self.norm` 后再返回。因此主分析必须使用它们实际 forward path 中的 post-block tokens，不要人为给最后一层再套 `model.norm`。

Metadata 必须记录：

```json
{
  "representation": "post_block_pre_final_norm",
  "layer_indexing": "1-based in outputs, block index 0-based in code"
}
```

---

# 12. Positive-class filtering

模型内部先计算所有 20 类：

\[
S^{(l)}
\in
\mathbb{R}^{B\times20\times N_p}.
\]

保存时默认只保存 image-level label 中存在的 classes：

```python
positive_class_ids = torch.nonzero(label > 0)
```

原因：

- 后续 class-specific GT region 分析首先针对 positive classes；
- 显著降低磁盘开销；
- 避免当前阶段混入 absent-class interpretation。

提供可选参数：

```text
--save-all-classes
```

默认关闭。

VOC image-level class index：

```text
0 ... 19
```

后续与 semantic mask 对接时才映射为：

```text
1 ... 20
```

本阶段不加载 segmentation mask。

---

# 13. Patch grid

不要硬编码：

```text
28 × 28
```

从输入和 patch token 数量推导：

```python
num_patches = scores.shape[-1]
grid_h = image.shape[-2] // model.patch_embed.patch_size[0]
grid_w = image.shape[-1] // model.patch_embed.patch_size[1]
assert grid_h * grid_w == num_patches
```

在 448、patch size 16 下预期：

\[
N_p=28\times28=784.
\]

若输入为非正方形，也必须按：

```text
grid_h × grid_w
```

保存，不能用 `sqrt(N)`。

---

# 14. Forward path

主分析直接调用：

```python
model.forward_features(images)
```

而不是：

```python
model(images)
```

原因：

- 两个 CAM model 的 `forward()` 返回内容不同；
- 本实验只需要执行 encoder blocks；
- hook 已在 blocks 中获取 score；
- 避免不必要的 patch head、CAM 与 patch-affinity计算。

伪代码：

```python
model.eval()

with BlockTokenCollector(model, num_classes=20) as collector:
    for batch in loader:
        collector.clear()

        images = batch["image"].to(device)
        labels = batch["label"]

        with torch.inference_mode():
            _ = model.forward_features(images)

        scores = collector.stack()
        # [L, B, C, N]

        save_batch(scores, batch)
```

必须确认：

```text
len(collector.scores) == model depth == 12
```

---

# 15. 结果保存格式

输出目录：

```text
results/
└── lazy_assignment/
    └── experiment1_class_patch_score/
        ├── mctformer/
        │   ├── metadata.json
        │   ├── manifest.jsonl
        │   ├── scores/
        │   │   ├── 2007_000033.npz
        │   │   └── ...
        │   ├── summary_by_layer.csv
        │   └── visualizations/
        └── mctformer_plus/
            └── ...
```

## 15.1 Per-image NPZ

每张图保存一个压缩文件：

```python
np.savez_compressed(
    path,
    image_id=image_id,
    positive_class_ids=positive_ids,
    scores_raw=scores_positive.float().numpy(),
    grid_h=grid_h,
    grid_w=grid_w,
)
```

其中：

```text
scores_raw shape = [12, num_positive_classes, num_patches]
dtype = float32
```

不要保存全部 hidden states。

## 15.2 Manifest

每行一张图：

```json
{
  "image_id": "2007_000033",
  "score_path": ".../2007_000033.npz",
  "positive_class_ids": [0, 14],
  "num_layers": 12,
  "num_patches": 784,
  "grid_h": 28,
  "grid_w": 28
}
```

## 15.3 Metadata

必须包括：

```text
repository URL
git commit
model name
model class
checkpoint path
checkpoint SHA256
checkpoint format
input size
patch size
num classes
depth
embed dim
dataset root
list path
number of samples
transform
torch version
CUDA version
GPU model
analysis timestamp
```

---

# 16. Summary statistics

Experiment 1 不做 GT region statistics，但需要保存基本 sanity summary。

每个 model/layer：

```text
num_images
num_positive_class_maps
score_min
score_max
score_mean
score_std
score_q05
score_q50
score_q95
nan_count
inf_count
```

输出：

```text
summary_by_layer.csv
```

这些数值只用于检查 score 是否合理，不用于声称 background leakage。

---

# 17. Diagnostic visualization

本阶段只生成少量 score-map sanity visualization。

## 17.1 默认样本

对 smoke-test 中固定的 10 张图：

- 每个图选第一个 positive class；
- 另可选所有 positive classes，但最多 3 个。

## 17.2 展示层

只展示：

```text
layer 1
layer 4
layer 8
layer 12
```

布局：

```text
Original | L1 | L4 | L8 | L12
```

## 17.3 两种显示方式

### Raw cosine

统一色标：

```text
vmin = -1
vmax = 1
```

### Per-map min-max

仅作为视觉辅助：

\[
\tilde S
=
\frac{S-\min S}{\max S-\min S+\epsilon}.
\]

文件名必须标清：

```text
*_raw_cosine.png
*_minmax.png
```

不得把 min-max 图称为 raw Patch Score。

---

# 18. 单元测试

## 18.1 `test_class_patch_score.py`

构造小 tensor，验证：

```python
score = einsum(normalize(cls), normalize(patch))
```

与逐元素 `F.cosine_similarity` 一致：

```text
max_abs_diff < 1e-6
```

验证 range：

```text
-1 - eps <= score <= 1 + eps
```

---

## 18.2 `test_token_collector.py`

分别实例化最小 MCTformer/MCTformer+ 或使用真实模型小输入，验证：

```text
12 layers captured
output[0] correctly interpreted as tokens
class slice = [:, :C]
patch slice = [:, C:]
score shape = [12, B, C, N]
```

---

## 18.3 Final-layer equivalence

Hook 捕获的 layer 12 tokens 应与 `forward_features()` 返回的最终：

```text
x_cls
x_patch
```

一致。

测试：

```text
max_abs_diff(cls_hook, x_cls_returned) < 1e-6
max_abs_diff(patch_hook, x_patch_returned) < 1e-6
```

MCTformer 和 MCTformer+ 都必须通过。

这是最关键的 token-slicing regression test。

---

## 18.4 No numerical change

同一 checkpoint、同一 input：

### 无 hook

```python
baseline = model.forward_features(x)
```

### 有 hook

```python
with collector:
    instrumented = model.forward_features(x)
```

比较最终 class tokens、patch tokens 和 attention weights：

```text
max_abs_diff < 1e-6
```

Hook 不得改变 forward。

---

## 18.5 Checkpoint strict load

两个 VOC checkpoints 均要求：

```text
missing_keys = 0
unexpected_keys = 0
```

除非 checkpoint 外层只有 `model` wrapper。

禁止通过 `strict=False` 掩盖结构问题。

---

# 19. Smoke Test

先只跑：

```text
model = mctformerplus
limit = 50
batch_size = 4
```

建议命令：

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerplus \
  --checkpoint /ABSOLUTE/PATH/TO/MCTFORMER_PLUS_VOC.pth \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 4 \
  --num-workers 4 \
  --limit 50 \
  --save-visualizations \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer_plus
```

Smoke test 必须人工检查：

1. checkpoint strict load 成功；
2. 12 个 layers 全部被捕获；
3. 每张图 score shape 正确；
4. 448 输入下 patch grid 为 28×28；
5. positive class IDs 与 image-level labels 一致；
6. raw score 无 NaN/Inf；
7. hook 与无 hook 输出一致；
8. 生成的 score map 与图像网格方向一致；
9. 图像没有转置、左右翻转或 H/W 交换；
10. 结果文件可重新读取。

---

# 20. Full Run

Smoke test 通过后：

## MCTformer+

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerplus \
  --checkpoint /ABSOLUTE/PATH/TO/MCTFORMER_PLUS_VOC.pth \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 8 \
  --num-workers 8 \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer_plus
```

## MCTformer

```bash
python analysis/lazy_assignment/run_class_specific_patch_score.py \
  --model mctformerv2 \
  --checkpoint /ABSOLUTE/PATH/TO/MCTFORMER_VOC.pth \
  --voc-root data/VOCdevkit/VOC2012 \
  --list-path data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --input-size 448 \
  --batch-size 8 \
  --num-workers 8 \
  --output-dir results/lazy_assignment/experiment1_class_patch_score/mctformer
```

若显存不足，只降低 batch size；不能改变：

- input size；
- transform；
- model；
- checkpoint；
- score definition。

---

# 21. Experiment 1 验收标准

只有满足全部条件，Experiment 1 才算完成。

## 21.1 代码

- [ ] MCTformer 与 MCTformer+ 共用同一 collector 和 score implementation；
- [ ] 不修改训练路径；
- [ ] 不使用 `models/mct_vit.py`；
- [ ] 所有 hooks 被可靠移除；
- [ ] checkpoint strict load；
- [ ] 测试全部通过。

## 21.2 数值

- [ ] 每个模型捕获 12 层；
- [ ] score shape 为 `[12, B, 20, N]`；
- [ ] 保存后的 positive-class score shape 正确；
- [ ] raw score 无 NaN/Inf；
- [ ] cosine score 位于 `[-1,1]` 数值容差内；
- [ ] layer-12 hook token 与 `forward_features()` final token 一致；
- [ ] hook 对模型输出的最大影响 `< 1e-6`。

## 21.3 数据

- [ ] VOC val 所有样本均处理或明确记录失败原因；
- [ ] 每张图有 image ID、positive classes、grid size；
- [ ] checkpoint SHA256、Git commit 和环境信息完整记录；
- [ ] `.npz` 可由独立读取脚本重新加载。

## 21.4 可视检查

- [ ] 至少 10 张图生成 L1/L4/L8/L12 score maps；
- [ ] patch map 与输入图方向一致；
- [ ] raw-cosine 与 min-max visualization 明确区分；
- [ ] 没有在本阶段加入 GT background interpretation。

---

# 22. Codex 最终交付物

Codex 完成后只交付：

```text
1. analysis/lazy_assignment/run_class_specific_patch_score.py
2. analysis/lazy_assignment/token_collector.py
3. analysis/lazy_assignment/score_utils.py
4. analysis/lazy_assignment/voc_score_dataset.py
5. analysis/lazy_assignment/visualize_patch_score.py
6. analysis/lazy_assignment/tests/*
7. Experiment 1 README
8. 两个模型的 smoke-test commands/logs
9. 两个模型的 full-run commands
10. MCTformer 与 MCTformer+ 的 metadata.json
11. summary_by_layer.csv
12. 结果 manifest 和 per-image score files
13. git diff summary
```

---

# 23. Codex 当前阶段禁止事项

本阶段禁止：

- 实现 target/other-FG/background region assignment；
- 计算 C-PiM；
- 计算 BG-Tail；
- 分析 GT segmentation；
- 提取或比较 \(A_{c2p}\)；
- 生成或比较 CAM；
- 加 register/BG token；
- 搬 LaST FFT/stability selector；
- 改 attention；
- 重新训练 checkpoint；
- 根据 score map 提出新方法；
- 直接宣称 lazy semantic assignment 存在。

Experiment 1 的最终结论只能写成：

> We successfully extended LaST-ViT’s representation-level Patch Score from a single CLS token to multiple class-specific tokens and extracted layer-wise class–patch semantic-alignment maps from MCTformer and MCTformer+ without altering their numerical behavior.

是否存在 background lazy semantic assignment，要等 Experiment 2 的 GT-region 验证后再判断。
