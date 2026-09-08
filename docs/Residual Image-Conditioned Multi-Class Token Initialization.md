# Residual Image-Conditioned Multi-Class Token Initialization for MCTformer+

## 1. 实验目的

当前 MCTformer+ 在进入第一个 Transformer block 之前，使用 DeiT 预训练 CLS token 初始化多个 class tokens。

保留这一 pretrained prior，不再像上一版实验那样完全删除 duplicated CLS。

新的初始化定义为：

\[
\boxed{
T^{(0)}(x)
=
\mathbf 1_C e_{\mathrm{cls}}^\top
+
\alpha T_{\mathrm{img}}(x)
+
E_{\mathrm{cls}}
}
\]

其中：

- \(C=20\)：VOC foreground classes；
- \(D=384\)：MCTformer+-Small embedding dimension；
- \(e_{\mathrm{cls}}\in\mathbb R^D\)：DeiT-S pretrained CLS token；
- \(E_{\mathrm{cls}}\in\mathbb R^{C\times D}\)：当前 MCTformer+ 已有 `pos_embed_cls`；
- \(T_{\mathrm{img}}(x)\in\mathbb R^{C\times D}\)：从当前图像 patch features 生成的 image-conditioned multi-class tokens；
- \(\alpha\)：控制 image-conditioned residual 强度的 learnable scalar。

因此，新方法不是替代 DeiT CLS prior，而是：

\[
\boxed{
\text{Pretrained CLS Prior}
+
\text{Image-conditioned Class Residual}
}
\]

---

# 2. Class-wise Weighted Pooling

使用进入第一个 ViT block 之前的 patch tokens：

\[
P(x)
=
[p_1,\ldots,p_N]
\in
\mathbb R^{N\times D}.
\]

对于 448×448 输入和 patch size 16：

\[
N=28\times28=784.
\]

引入一组 learnable class pooling queries：

\[
Q
=
[q_1,\ldots,q_C]
\in
\mathbb R^{C\times D}.
\]

计算 class-to-patch pooling logits：

\[
L
=
\frac{QP^\top}{\sqrt D}
\in
\mathbb R^{C\times N}.
\]

即：

\[
L_{c,i}
=
\frac{q_c^\top p_i}{\sqrt D}.
\]

沿 patch dimension 做 softmax：

\[
A_{c,i}
=
\frac{
\exp(L_{c,i})
}{
\sum_{j=1}^{N}\exp(L_{c,j})
}.
\]

因此：

\[
A
\in
\mathbb R^{C\times N}.
\]

然后：

\[
\boxed{
T_{\mathrm{img}}(x)
=
AP
}
\]

即：

\[
t_c^{img}
=
\sum_{i=1}^{N}A_{c,i}p_i.
\]

最终：

\[
\boxed{
t_c^{(0)}
=
e_{\mathrm{cls}}
+
\alpha t_c^{img}
+
e_c^{cls}.
}
\]

---

# 3. 模块实现

新增：

```text
models/class_token_pooling.py
```

实现：

```python
class ClassWiseWeightedPooling(nn.Module):
```

需要的参数只有：

```python
class_queries  # [C, D]
alpha          # scalar
```

其中：

```python
class_queries
```

使用：

```python
trunc_normal_(..., std=.02)
```

初始化。

建议：

```python
alpha = nn.Parameter(torch.tensor(0.1))
```

即：

\[
\alpha_0=0.1.
\]

这样训练开始时仍主要依赖 DeiT pretrained CLS prior，同时 image-conditioned branch 从第一步开始就有正常梯度。

第一轮只使用一个 global scalar \(\alpha\)，不要使用 per-class alpha。

---

# 4. Pooling forward

核心实现可以保持非常简单：

```python
def forward(self, patch_tokens):
    # patch_tokens: [B, N, D]

    q = self.class_queries.unsqueeze(0).expand(
        patch_tokens.shape[0], -1, -1
    )

    logits = torch.einsum(
        "bcd,bnd->bcn",
        q,
        patch_tokens,
    ) / math.sqrt(self.embed_dim)

    attention = torch.softmax(logits, dim=-1)

    image_tokens = torch.einsum(
        "bcn,bnd->bcd",
        attention,
        patch_tokens,
    )

    return image_tokens, attention
```

输出：

```text
image_tokens: [B, C, D]
attention:    [B, C, N]
```

第一轮不要增加：

- \(W_Q\)
- \(W_K\)
- \(W_V\)
- MLP
- LayerNorm
- multi-head pooling
- Top-K pooling

保持模块尽量简单。

---

# 5. 接入 MCTformer+

当前 MCTformer+ native path 中已经存在：

```python
x = self.patch_embed(x)
x = x + pos_embed_pat

cls_tokens = self.cls_token.expand(B, -1, -1)
cls_tokens = cls_tokens + self.pos_embed_cls
```

修改为：

```python
x = self.patch_embed(x)
x = x + pos_embed_pat

pretrained_cls = self.cls_token.expand(B, -1, -1)

image_cls, pooling_attention = self.class_token_pooler(x)

cls_tokens = (
    pretrained_cls
    + self.class_token_pooler.alpha * image_cls
    + self.pos_embed_cls
)
```

之后继续：

```python
x = torch.cat([cls_tokens, x], dim=1)
x = self.pos_drop(x)
```

然后完全沿用原来的：

```text
Transformer Block 1
...
Transformer Block 12
```

---

# 6. 重要原则

这次实验只修改：

\[
\boxed{
\text{initial multi-class token construction}
}
\]

不要修改 MCTformer+ 后续训练机制。

保持当前原始 MCTformer+ 的：

- DeiT-S pretrained backbone；
- 12 个 Transformer blocks；
- multi-class token classification；
- CCT；
- patch classifier；
- GWRP；
- classification losses；
- CAM generation；
- class-to-patch attention；
- patch-to-patch refinement；
- optimizer；
- learning rate；
- epoch；
- augmentation；
- dataset；
- evaluation pipeline。

即：

\[
\boxed{
\text{training logic 与原 MCTformer+ 完全一致}
}
\]

只多出：

```text
ClassWiseWeightedPooling
```

这个模块。

---

# 7. DeiT pretrained initialization

仍然按照原始 MCTformer+ 的方式使用 DeiT-S pretrained weights。

尤其保留：

\[
e_{\mathrm{cls}}
\]

并继续：

```python
source_cls_token.repeat(1, num_classes, 1)
```

初始化：

```python
self.cls_token
```

所以：

\[
\mathbf1_Ce_{\mathrm{cls}}^\top
\]

与 baseline 完全一致。

新增的：

```text
class_token_pooler.class_queries
class_token_pooler.alpha
```

随机初始化。

因此 pretrained adaptation 只需要允许这两个新增参数保持 model initialization，不需要改变其他 DeiT weight loading 行为。

---

# 8. 建议增加一个简单配置

增加：

```text
class_token_init
```

支持：

```text
baseline
residual_cwp
```

默认：

```text
baseline
```

对应：

### baseline

\[
T^{(0)}
=
\mathbf1_Ce_{\mathrm{cls}}^\top
+
E_{\mathrm{cls}}.
\]

### residual_cwp

\[
T^{(0)}
=
\mathbf1_Ce_{\mathrm{cls}}^\top
+
\alpha T_{\mathrm{img}}(x)
+
E_{\mathrm{cls}}.
\]

这样方便直接复用当前 MCTformer+。

---

# 9. 第一轮实验

只比较：

| Experiment | Initial class tokens |
|---|---|
| B0 | Original MCTformer+ |
| E1 | Residual CWP |

其中：

\[
B0:
T^{(0)}
=
\mathbf1_Ce_{\mathrm{cls}}^\top+E_{\mathrm{cls}}
\]

\[
E1:
T^{(0)}
=
\mathbf1_Ce_{\mathrm{cls}}^\top
+\alpha AP
+E_{\mathrm{cls}}
\]

第一轮不要 sweep：

- alpha；
- query dimension；
- pooling temperature；
- pooling heads；
- projection matrices。

固定：

\[
\alpha_0=0.1
\]

且 alpha learnable。

---

# 10. 简单测试

实现后只需要确认基本功能正确。

### Shape

输入：

```text
[B, 784, 384]
```

得到：

```text
image_tokens:
[B, 20, 384]

pooling_attention:
[B, 20, 784]
```

### Attention normalization

确认：

\[
\sum_i A_{c,i}=1.
\]

### Gradient

跑一次 backward，确认：

```text
class_queries.grad != None
alpha.grad != None
```

### Baseline compatibility

`class_token_init=baseline` 时原模型行为保持不变。

### Training smoke

跑一个短训练，确认：

- loss 正常；
- alpha 在更新；
- class queries 在更新；
- checkpoint 可保存；
- CAM evaluation 能正常运行。

不需要为这个实验增加复杂的 architecture protection logic。

---

# 11. 正式训练

E1 使用与当前 canonical MCTformer+-Small baseline **完全相同**的训练命令和参数。

唯一差异：

```text
--class-token-init residual_cwp
```

从同一个 official DeiT-S pretrained checkpoint 开始训练。

不要从已训练好的 MCTformer+ checkpoint fine-tune。

---

# 12. 训练期间简单记录

额外记录三个诊断即可。

## Alpha

保存每个 epoch：

\[
\alpha.
\]

观察 image-conditioned residual 最终被模型使用到什么强度。

---

## Pooling entropy

\[
H_c
=
-\frac{
\sum_iA_{c,i}\log A_{c,i}
}{
\log N
}.
\]

记录：

```text
mean pooling entropy
```

---

## Inter-class attention similarity

计算：

\[
\operatorname{mean}_{c\neq k}
\cos(A_c,A_k).
\]

观察不同 class queries 是否最终学习不同的 spatial pooling。

只记录，不添加任何额外 loss。

---

# 13. Evaluation

训练完成后完全使用当前 MCTformer+ 的标准 evaluation pipeline。

至少比较：

- classification performance；
- patch branch；
- CAM seed；
- final CAM mIoU；
- pseudo-mask / segmentation performance。

最终主表：

| Model | Classification | Patch | CAM | Final CAM / Mask |
|---|---:|---:|---:|---:|
| MCTformer+ | | | | |
| + Residual CWP | | | | |

指标名称直接使用当前仓库已有输出。

---

# 14. 重新运行现有 shared-relation analysis

训练完成后，也用之前已经实现好的 frozen relation analysis 对新 checkpoint 跑一次。

不要新增新的分析方法。

只直接比较原始 MCTformer+ 与 Residual CWP 的：

\[
L1\rightarrow L12
\]

结果：

- positive-class relation correlation；
- Top-5% cross-class collision；
- common \(R^2\)；
- PCA / effective rank。

此外可以给 Residual CWP 额外记录：

\[
L0
\]

也就是进入 Block 1 前：

\[
T^{(0)}
\]

和：

\[
P^{(0)}
\]

的 relation。

这样可以直接观察：

\[
\boxed{
\text{image-conditioned initialization 是否改变了 class-token relation trajectory}
}
\]

---

# 15. Codex 最终输出

完成后提供：

1. 修改文件列表；
2. 简单实现说明；
3. 新增参数量；
4. unit test / smoke test 结果；
5. exact training command；
6. training log；
7. alpha evolution；
8. pooling entropy；
9. inter-class pooling similarity；
10. baseline vs Residual CWP WSSS 主结果表；
11. L0/L1–L12 shared-relation comparison；
12. 一个简短 `report.md` 总结实验结果。

完成这一版以后停止，不要自行继续设计其他 CWP variants。

---

# 可直接交给 Codex 的 Prompt

请基于当前 `jungletada/TGCA` 最新 `main` 分支，实现一个新的 MCTformer+ class-token initialization variant：

```text
class_token_init = residual_cwp
```

目标公式：

\[
T^{(0)}(x)
=
\mathbf1_Ce_{\mathrm{cls}}^\top
+
\alpha T_{\mathrm{img}}(x)
+
E_{\mathrm{cls}}.
\]

必须保留原始 MCTformer+ 的 duplicated DeiT-S CLS token 作为 pretrained prior。

新增一个简单的：

```python
ClassWiseWeightedPooling
```

输入 Block 1 前的 patch features：

\[
P\in\mathbb R^{B\times N\times D}.
\]

使用 learnable class queries：

\[
Q\in\mathbb R^{C\times D}
\]

计算：

\[
A
=
softmax_N
\left(
QP^\top/\sqrt D
\right),
\]

然后：

\[
T_{\mathrm{img}}
=
AP.
\]

最终：

```python
pretrained_cls = self.cls_token.expand(B, -1, -1)

image_cls, pooling_attention = self.class_token_pooler(patches)

cls_tokens = (
    pretrained_cls
    + self.class_token_pooler.alpha * image_cls
    + self.pos_embed_cls
)
```

其中：

```text
C = 20
D = 384
N = 784
```

`class_queries` 使用 `trunc_normal_(std=.02)` 初始化。

使用一个 learnable scalar：

```python
alpha = nn.Parameter(torch.tensor(0.1))
```

第一轮不要增加 Q/K/V projection、MLP、LayerNorm、multi-head、Top-K 或额外 loss。

除了 class-token initialization 之外，MCTformer+ 的所有训练和 evaluation 逻辑保持原样，包括：

```text
DeiT-S pretrained weights
12 Transformer blocks
CCT
class-token loss
patch classifier
GWRP
losses
CAM pipeline
optimizer
LR
epochs
augmentation
dataset
```

DeiT pretrained loading 时继续像原 MCTformer+ 一样把 source CLS token repeat 成 20 个 `self.cls_token`。

新增的：

```text
class_token_pooler.class_queries
class_token_pooler.alpha
```

保持随机初始化即可。

实现一个简单配置：

```text
class_token_init = baseline | residual_cwp
```

默认 `baseline`。

先完成基本 shape、attention normalization、gradient 和 smoke-training 检查，然后使用当前 canonical MCTformer+-Small baseline 的原训练配置完整训练 `residual_cwp`。

训练过程中额外记录：

```text
alpha
mean pooling entropy
mean inter-class pooling-attention cosine similarity
```

训练完成后运行当前标准 MCTformer+ WSSS evaluation，和已有 baseline 比较。

然后用现有 frozen shared-relation analysis 分析 Residual CWP checkpoint，直接比较 baseline 与新模型的 L1–L12：

```text
positive-class relation correlation
Top-5% cross-class collision
common R²
PCA/effective rank
```

Residual CWP 另外保存一个 Layer-0 relation，即 Block 1 前的 \(T^{(0)}\) 与 \(P^{(0)}\)。

最后输出代码修改、训练命令、主要 WSSS 结果和 shared-relation 比较报告。

完成这一版后停止，不要自动实现其他 variants。