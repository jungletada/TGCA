# MCTformer+-FinalLN：干净对照实验计划（Codex 执行版）

> **目标仓库：** https://github.com/jungletada/TGCA  
> **目标：** 只验证在 MCTformer+ 的第 12 个 Transformer block 完整输出后加入标准 `Final LayerNorm` 的影响。  
> **本实验不加入 LaST 的 FFT / Low-pass / Top-K，不改 GWRP，不改 attention，不加 BG token，不加新的 loss。**

---

# 1. 实验目的

LaST-ViT 使用的 ViT encoder 在最后一个 Transformer block 后执行一次 final LayerNorm，再将归一化后的 patch tokens送入其后续 aggregation / classification path。

当前 MCTformer+ 在最后一个 block 后直接切分 class tokens 和 patch tokens，没有使用已有的 `self.norm`。

本实验只做一个结构改动：

```python
x_norm = self.norm(x)

x_cls = x_norm[:, :num_classes]
x_patch = x_norm[:, num_classes:]
```

随后继续执行原始 MCTformer+ 的训练与 CAM 流程。

实验名称：

```text
MCTformer+-FinalLN
```

这只是 LaST 前的结构对齐实验，不叫 Last-MCT。

---

# 2. 代码修改原则

## 2.1 修改 `forward_features()`

目标文件：

```text
models/mctformer_plus.py
```

Baseline 当前逻辑近似为：

```python
for i, blk in enumerate(self.blocks):
    x, weights_i = blk(x)
    attn_weights.append(weights_i)
    all_x_cls.append(x[:, :self.num_classes])

x_cls = x[:, :self.num_classes]
x_patch = x[:, self.num_classes:]

return x_cls, x_patch, attn_weights, all_x_cls
```

FinalLN variant 改为：

```python
for i, blk in enumerate(self.blocks):
    x, weights_i = blk(x)
    attn_weights.append(weights_i)

    # CCT 仍保存原始 post-block class tokens
    all_x_cls.append(x[:, :self.num_classes])

# only new operation
x_norm = self.norm(x)

x_cls = x_norm[:, :self.num_classes]
x_patch = x_norm[:, self.num_classes:]

return x_cls, x_patch, attn_weights, all_x_cls
```

如果当前仓库已经加入 BGT / BCSS / PSL 等可选结构，不要破坏现有 token slicing helper。应使用当前 baseline 的 foreground/patch slice helper，而不是硬编码 `:num_classes`，例如：

```python
x_norm = self.norm(x)

x_cls = x_norm[:, self._foreground_slice()]
x_patch = x_norm[:, self._patch_slice(patch_count)]
```

只要保证 baseline 模式下等价于：

```text
[class tokens, patch tokens]
```

即可。

---

# 3. CCT 保持原样

不要把 `all_x_cls` 改成 FinalLN 后的 tokens。

继续在每个 Transformer block 完成后保存：

```python
all_x_cls.append(raw_post_block_class_tokens)
```

因此 CCT loss 的定义完全不变。

本实验只改变最终两个 classification branches 的输入：

```text
Final class-token classification branch
Final patch classification branch
```

---

# 4. Class-token branch

FinalLN 后：

```python
x_cls_logits = x_cls.mean(dim=-1)
```

保持原版 MCTformer+ 的 readout。

不增加新的 classifier，不改变 class-token loss。

---

# 5. Patch branch

FinalLN 后的：

```python
x_patch
```

继续完全走原来的 patch classifier：

```text
reshape to 2D
→ existing 3×3 Conv head
→ existing GWRP
→ patch logits
→ original patch classification loss
```

禁止：

- 换成 1×1 classifier；
- 换成 Linear head；
- 换 GWRP；
- 加 FFT；
- 加 low-pass；
- 加 top-k aggregation。

---

# 6. CAM inference 必须同步使用 FinalLN patch tokens

`MCTformerPlusCam` 的 CAM path 也必须与训练一致。

确保：

```text
12 Transformer blocks
→ self.norm(x)
→ split normalized class / patch tokens
→ existing patch head
→ existing CAM refinement
```

保持以下部分完全不变：

```text
last-3 class-to-patch attention
all-layer patch-to-patch attention
sqrt refinement
CAM normalization / evaluation
```

禁止出现：

```text
training uses normalized patch tokens
but CAM inference uses raw patch tokens
```

---

# 7. Attention 本身不改

Final LayerNorm 位于 Block 12 完整输出之后，因此本实验不直接修改：

```text
A_c2p
A_p2p
Q/K/V
attention normalization
attention heads
```

不要为了本实验添加任何 attention ablation。

---

# 8. 预训练初始化

FinalLN variant 从与 baseline 完全相同的 DeiT-Small pretrained initialization 重新训练。

要求：

```text
same pretrained checkpoint
same model width/depth
same input size
same optimizer
same LR
same epochs
same batch size
same augmentation
same seed
same loss weights
same training list
same validation list
```

已有的 `self.norm.weight` / `self.norm.bias` 应正常从 DeiT pretrained checkpoint加载，并从 epoch 0 开始参与 WSSS finetuning。

不要从已经训练好的 MCTformer+ checkpoint直接打开 FinalLN 后评估。

---

# 9. Variant 开关

建议增加一个简单参数，例如：

```text
--final-norm
```

或模型参数：

```python
final_norm=False
```

要求：

### Baseline

```python
final_norm=False
```

必须与当前 MCTformer+ 数值行为保持一致。

### FinalLN

```python
final_norm=True
```

才执行：

```python
x = self.norm(x)
```

这样方便同一套代码完成严格 matched comparison。

---

# 10. 基本单元测试

只需要必要测试，不需要重新做 Experiment 1–3 的大量诊断。

## Test 1：Baseline equivalence

`final_norm=False` 时，与修改前 baseline：

```text
logits
patch logits
attention
CAM
```

数值一致。

目标：

```text
max_abs_diff < 1e-6
```

## Test 2：FinalLN shape

确认：

```text
x_cls shape unchanged
x_patch shape unchanged
attention shapes unchanged
CAM shapes unchanged
```

## Test 3：Norm gradient

训练一个 mini-batch 后确认：

```text
self.norm.weight.grad is not None
self.norm.bias.grad is not None
```

并且 gradient norm 非零。

## Test 4：Train / CAM path consistency

确认 FinalLN variant 的 CAM model 使用的 patch tokens 与 training model 的 normalized patch tokens一致。

---

# 11. 实验执行

至少训练：

```text
Baseline MCTformer+
MCTformer+-FinalLN
```

使用完全相同设置和同一个 seed。

如果现有 baseline checkpoint 已由完全相同当前训练配置产生，可直接复用 baseline结果；否则重跑 matched baseline。

---

# 12. 建议记录的结果

Codex 最后只整理以下结果，不做扩展机制分析。

## 12.1 Classification

记录：

```text
class-token mAP
patch-head mAP
validation classification loss
```

表格：

| Model | Class-token mAP | Patch-head mAP |
|---|---:|---:|
| MCTformer+ | | |
| MCTformer+-FinalLN | | |

---

## 12.2 Raw CAM

记录：

```text
raw CAM mIoU at the same default background threshold
best raw CAM mIoU from the same threshold sweep
best threshold
foreground precision
foreground recall
```

表格：

| Model | Fixed-threshold CAM mIoU | Best CAM mIoU | Best threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| MCTformer+ | | | | | |
| MCTformer+-FinalLN | | | | | |

---

## 12.3 Attention

只记录 L10、L11、L12 和 native last3。

指标：

```text
C-PiM
target-vs-background AUROC
target-vs-other-foreground AUROC
positive-class-pair top10 Jaccard
```

不需要重新生成 Experiment 2 的全部分析。

---

## 12.4 Representation

只记录 FinalLN 前后最后一层 class-token representation：

```text
positive class-token pair cosine
GT-positive vs GT-negative shared-presence projection AUROC
fixed all-ones-axis energy
learned shared-presence direction vs all-ones alignment
```

目的只是观察 FinalLN 是否明显改变 Experiment 3 中发现的 class-token representation geometry。

不需要重新执行完整 Presence-Axis Experiment。

---

# 13. 最终输出文件

Codex 最终整理：

```text
results/final_ln_ablation/<run_id>/
├── config_baseline.json
├── config_final_ln.json
├── training_logs/
├── checkpoints/
├── classification_results.csv
├── cam_results.csv
├── attention_results.csv
├── representation_results.csv
├── comparison_summary.csv
├── exact_commands.sh
└── FINAL_LN_EXPERIMENT_REPORT.md
```

报告只包含：

1. 代码改动；
2. 训练设置；
3. 上述四组结果；
4. Baseline vs FinalLN 的数值差值；
5. 运行/测试是否成功。

不要在报告中设计新的 LaST variant、competition、BG token 或后续方法。

---

# 14. 可直接给 Codex 的任务说明

```text
Implement a single clean MCTformer+ ablation called MCTformer+-FinalLN.

After all 12 Transformer blocks, apply the model's existing final LayerNorm to
the complete token sequence. Then split the normalized sequence into class
tokens and patch tokens.

The normalized class tokens must continue through the original
x_cls.mean(dim=-1) classification path.

The normalized patch tokens must continue through the original patch path:
reshape -> existing 3x3 Conv classifier -> existing GWRP -> patch loss.

Keep the per-block raw class tokens used by the CCT loss unchanged. Do not apply
the new final LayerNorm to all_x_cls.

The CAM model must use the same normalized final patch tokens before the existing
3x3 patch head. Keep the original last-three class-to-patch attention, all-layer
patch-to-patch propagation, sqrt refinement, loss weights, optimizer, training
schedule, augmentation, input size, and all other MCTformer+ behavior unchanged.

Add a simple final_norm flag so final_norm=False is numerically identical to the
current baseline.

Train MCTformer+-FinalLN from the same DeiT-S pretrained initialization as the
baseline using exactly the same training configuration and seed. Do not evaluate
an old trained MCTformer+ checkpoint by merely enabling FinalLN.

Run only the necessary unit tests and the matched experiment.

At the end, summarize only:
1. class-token mAP and patch-head mAP;
2. fixed-threshold and best-threshold raw CAM mIoU, plus precision/recall;
3. L10/L11/L12/native-last3 C-PiM, target-vs-BG AUROC,
   target-vs-other-FG AUROC, and positive-class-pair top10 Jaccard;
4. final-layer positive class-token pair cosine, shared-presence projection
   AUROC, fixed all-ones-axis energy, and learned shared-direction/all-ones
   alignment.

Do not add FFT, low-pass filtering, LaST aggregation, new classifiers,
background tokens, semantic competition, attention interventions, or any other
new method in this experiment.
```
