# MCTformer+ Class-wise Weighted Pooling 实验执行计划

## 1. 实验目标

在原始 MCTformer+ 上，只修改进入第一个 Transformer block 之前的 multi-class token initialization。

原始 MCTformer+：

\[
T_{\text{base}}^{(0)}
=
\operatorname{Repeat}(e_{\mathrm{CLS}},C)
+
E_{\mathrm{cls}},
\]

其中：

\[
e_{\mathrm{CLS}}\in\mathbb R^D,
\qquad
C=20,
\qquad
D=384.
\]

本实验彻底删除 duplicated DeiT CLS content，改为：

\[
\boxed{
T^{(0)}(x)
=
\Phi_\theta(P^{(0)}(x))
+
E_{\mathrm{cls}}
}
\]

其中：

\[
P^{(0)}
\in
\mathbb R^{B\times N\times D}
\]

为 patch embedding 加 spatial positional embedding 之后、进入 Transformer block 之前的 patch features。

对于 VOC / MCTformer+-Small：

\[
N=28\times28=784,
\qquad
C=20,
\qquad
D=384.
\]

目标是让：

\[
\boxed{
T^{(0)}
\in
\mathbb R^{B\times20\times384}
}
\]

从 Block 0 开始就是 image-conditioned multi-class representation。

---

# 2. 第一版只实现最简单的 Class-wise Weighted Pooling

暂时不要加入：

- \(W_Q/W_K/W_V\) projection；
- MLP；
- LayerNorm；
- multi-head cross-attention；
- Top-K；
- Fourier；
- low-pass；
- class competition；
- auxiliary loss；
- prototype loss；
- orthogonality loss。

第一轮必须尽可能只验证：

> **把 static duplicated CLS initialization 换成 image-conditioned class-wise pooling 本身是否有效。**

---

# 3. 数学定义

对于一张图片，patch embedding + positional embedding 得到：

\[
P
=
[p_1,\ldots,p_N]
\in
\mathbb R^{N\times D}.
\]

新增一组 learnable class pooling queries：

\[
Q
=
[q_1,\ldots,q_C]
\in
\mathbb R^{C\times D}.
\]

这里的 \(Q\)：

- 不是输入 ViT 的 class token；
- 不直接加入 token sequence；
- 只用于决定每一个 class 如何从 image patches 中做 weighted pooling。

计算：

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
\frac{
q_c^\top p_i
}{
\sqrt D
}.
\]

然后只沿 **patch dimension** 做 softmax：

\[
\boxed{
A_{c,i}
=
\frac{
\exp(L_{c,i})
}{
\sum_{j=1}^{N}\exp(L_{c,j})
}
}
\]

因此：

\[
A
\in
\mathbb R^{C\times N},
\]

且：

\[
\sum_i A_{c,i}=1.
\]

注意：

\[
\boxed{
\text{绝对不能沿 class dimension 做 softmax。}
}
\]

这是 multi-label task，不应该在初始化阶段强迫不同 classes 对 patch 做互斥 ownership。

---

# 4. 生成 image-conditioned multi-class tokens

定义：

\[
\boxed{
\Phi_\theta(P)
=
AP
}
\]

即：

\[
\Phi_\theta(P)
\in
\mathbb R^{C\times D}.
\]

对于每一个 class：

\[
\boxed{
\phi_c(P)
=
\sum_{i=1}^{N}
A_{c,i}p_i.
}
\]

最终：

\[
\boxed{
t_c^{(0)}
=
\phi_c(P)+e_c^{cls}.
}
\]

矩阵形式：

\[
\boxed{
T^{(0)}
=
AP+E_{\mathrm{cls}}.
}
\]

这就是本实验唯一允许使用的 CWP formulation。

---

# 5. 不使用 DeiT CLS token

CWP variant 中：

\[
\boxed{
e_{\mathrm{CLS}}
\text{ 完全不进入模型 forward。}
}
\]

禁止：

\[
T^{(0)}
=
Repeat(e_{\mathrm{CLS}})
+
AP
+
E_{\mathrm{cls}}.
\]

禁止 residual CLS。

禁止 gated CLS。

禁止把 DeiT CLS 当 query。

CWP 模型最终应该不存在一个参与 forward 的：

```python
self.cls_token
```

parameter。

最好在 CWP variant 中：

```python
self.cls_token = None
```

使其不出现在：

```python
model.named_parameters()
model.state_dict()
```

中。

baseline variant 保持原实现。

---

# 6. 保留 \(E_{\mathrm{cls}}\)

这里：

\[
E_{\mathrm{cls}}
\]

继续使用当前 MCTformer+ 的：

```python
self.pos_embed_cls
```

即：

\[
E_{\mathrm{cls}}
\in
\mathbb R^{1\times C\times D}.
\]

第一轮不要改变它的初始化方式。

因此从 official DeiT-S 初始化时：

- source CLS token content：**丢弃**
- source CLS positional embedding：继续按照 baseline 方式复制到 `pos_embed_cls`
- patch positional embedding：保持原实现
- patch embedding：保持 pretrained
- Transformer blocks：保持 pretrained

这样唯一改变的是：

\[
\boxed{
\text{class token content source}
}
\]

而不是同时修改 class positional prior。

---

# 7. 新增模块

建议新增：

```text
models/class_token_pooling.py
```

实现：

```python
class ClassWiseWeightedPooling(nn.Module):
```

核心参数只有：

```python
self.class_queries
```

shape：

```text
[1, C, D]
```

或者：

```text
[C, D]
```

均可，但 API 最终必须输出：

```text
pooled_tokens: [B, C, D]
attention:     [B, C, N]
```

建议参数初始化：

```python
trunc_normal_(self.class_queries, std=.02)
```

第一版参数量只有：

\[
C\times D
=
20\times384
=
\boxed{7680}
\]

个参数。

不要增加其他 learnable matrices。

---

# 8. 推荐 forward 实现

逻辑：

```python
def forward(self, patch_tokens, return_attention=False):
    # patch_tokens: [B, N, D]

    q = self.class_queries.expand(B, -1, -1)

    logits = torch.einsum(
        "bcd,bnd->bcn",
        q.float(),
        patch_tokens.float(),
    ) / math.sqrt(self.embed_dim)

    attention = torch.softmax(logits, dim=-1)

    pooled = torch.einsum(
        "bcn,bnd->bcd",
        attention,
        patch_tokens.float(),
    )

    pooled = pooled.to(patch_tokens.dtype)

    if return_attention:
        return pooled, attention
    return pooled
```

softmax/logits 使用 float32，保证 AMP 条件下稳定。

gradient 必须完整流过：

\[
T^{(0)}
\rightarrow A
\rightarrow Q.
\]

---

# 9. MCTformer+ 接入位置

当前 native pipeline：

\[
x
\rightarrow PatchEmbed
\rightarrow P
\rightarrow +PosEmbed
\]

然后原始代码构造：

```python
cls_tokens = self.cls_token.expand(B, -1, -1)
cls_tokens = cls_tokens + self.pos_embed_cls
```

CWP variant 改为：

```python
dynamic_cls_tokens, pooling_attention = \
    self.class_token_pooler(patches, return_attention=True)

cls_tokens = dynamic_cls_tokens + self.pos_embed_cls
```

然后保持：

```python
x = torch.cat([cls_tokens, patches], dim=1)
x = self.pos_drop(x)
```

完全不改后续：

```text
Block 1
...
Block 12
```

---

# 10. CWP 必须使用哪一个 patch representation

必须使用：

\[
\boxed{
P
=
PatchEmbed(x)+PosEmbed_{patch}
}
\]

即 **positional embedding 已加入，但尚未经过任何 Transformer block** 的 patches。

不要用：

```text
raw PatchEmbed without position
```

也不要用：

```text
Block 1 output
```

因为目标就是：

\[
\boxed{
\text{before Block 1}
}
\]

生成 image-conditioned class tokens。

训练时使用：

```python
patches + self.pos_embed_pat
```

evaluation 时使用当前已有：

```python
interpolate_pos_encoding()
```

逻辑。

---

# 11. 模型配置

新增参数：

```text
class_token_init
```

取值：

```text
baseline
cwp
```

默认：

```text
baseline
```

必须保证：

```text
class_token_init=baseline
```

时现有 MCTformer+ 行为完全不变。

第一轮实验固定：

```text
MCTformer+-Small
embed_dim = 384
depth = 12
heads = 6
input = 448
class_token_init = cwp
```

并且：

```text
final_norm = False
patch_final_norm = False
last_mct = False
class_stable_last = False
BCSS = E0
PSL = baseline
CTI-BGT = False
```

不要同时测试其他改进模块。

---

# 12. DeiT-S pretrained adaptation

必须修改：

```python
adapt_deit_checkpoint_for_mctformerplus()
```

但不能破坏 baseline。

## baseline

保持现在行为：

```text
DeiT cls_token
→ repeat ×20
→ MCTformer+ cls_token
```

## CWP

source：

```python
source["cls_token"]
```

只允许作为 DeiT checkpoint provenance / source validation 存在。

绝对不能加载进 CWP model。

CWP 新参数：

```text
class_token_pooler.class_queries
```

随机初始化。

`pos_embed_cls`：

继续来自：

```python
source["pos_embed"][:, :1]
```

repeat 到 20。

`pos_embed_pat`：

继续 bicubic interpolate。

Transformer block weights：

全部按原方案加载。

classifier head：

保持随机初始化。

---

# 13. Pretrained load report

对于 CWP，必须明确记录：

```json
{
  "class_token_init": "cwp",
  "deit_cls_token_used": false,
  "deit_cls_token_policy": "discarded",
  "class_pooling": "class-wise weighted pooling",
  "pooling_softmax_axis": "patch",
  "class_query_shape": [20, 384],
  "class_query_initialization": "trunc_normal_std_0.02"
}
```

并把：

```text
class_token_pooler.class_queries
head.weight
head.bias
```

记录为 randomly initialized parameters。

---

# 14. Checkpoint metadata

CWP checkpoint 必须保存：

```text
class_token_init = "cwp"
```

并写入：

```python
model_spec_from_instance()
```

或者 equivalent experiment metadata。

禁止让 baseline checkpoint 被误认为 CWP checkpoint。

禁止让 CWP checkpoint 在未开启：

```text
class_token_init=cwp
```

时 silent load。

---

# 15. Checkpoint architecture resolver

当前代码通过：

```text
cls_token
```

shape 推断 class token count。

CWP 没有这个 parameter，因此 resolver 必须支持两种情况：

### baseline

```text
cls_token.shape[1]
```

### cwp

```text
class_token_pooler.class_queries.shape[-2]
```

CWP checkpoint 必须具有明确 metadata：

```text
class_token_init = cwp
```

不要对 CWP 做 legacy fuzzy inference。

---

# 16. 保持 MCTformer+ 后续训练机制完全不变

CWP 只改变：

\[
T^{(0)}.
\]

以下全部保持原始 MCTformer+：

\[
T^{(1)},\ldots,T^{(12)}
\]

的 self-attention 更新方式；

class-token classification：

\[
x_{\mathrm{cls}}^{12}.mean(-1)
\]

保持不变；

patch classifier：

```text
3×3 Conv
```

保持不变；

GWRP：

保持不变；

CCT：

保持不变；

CAM：

\[
CAM
=
Class\text{-}Patch\ Attention
\times
Patch\ Class\ Feature
\]

保持不变；

最终 WSSS evaluation pipeline 保持不变。

---

# 17. 一个重要的梯度性质

由于：

\[
A_{c,i}
=
softmax_i(q_c^\top p_i),
\]

并且：

\[
t_c^{(0)}
=
\sum_iA_{c,i}p_i,
\]

原始 MCTformer+ 的 class-token classification loss 会直接产生：

\[
\frac{\partial\mathcal L}
{\partial q_c}
\neq0.
\]

因此 class pooling queries 可以在 image-level supervision 下学习：

> 第 \(c\) 个 class token 应该从哪些 patch features 初始化自己。

本轮不增加任何专门用于 \(Q\) 的额外 loss。

---

# 18. Unit Tests

必须新增完整测试。

## Test A — Shape

输入：

```text
[B, 784, 384]
```

输出：

```text
tokens    [B, 20, 384]
attention [B, 20, 784]
```

---

## Test B — Spatial normalization

必须满足：

\[
\sum_iA_{c,i}=1
\]

误差：

```text
< 1e-6
```

---

## Test C — No class competition

确认 softmax：

```python
dim=-1
```

即 patch dimension。

禁止：

```python
dim=1
```

---

## Test D — Image conditioning

固定同一个 class query bank。

输入两组不同 patch features：

\[
P_a\neq P_b.
\]

确认：

\[
T^{(0)}(P_a)
\neq
T^{(0)}(P_b).
\]

---

## Test E — Query differentiation

人为令：

\[
q_1=q_2.
\]

确认：

\[
A_1=A_2
\]

以及：

\[
\phi_1(P)=\phi_2(P).
\]

这样验证 class difference 确实来自 class query。

---

## Test F — Gradient

完成一次 dummy loss backward。

确认：

```python
class_token_pooler.class_queries.grad is not None
```

且：

```text
grad norm > 0
```

---

## Test G — No duplicated CLS

CWP model：

```python
assert "cls_token" not in dict(model.named_parameters())
```

或者至少确认：

```python
model.cls_token is None
```

且不出现在 state dict。

同时确认 forward 中没有读取 source/pretrained CLS content。

---

## Test H — Baseline regression

`class_token_init=baseline` 时：

- state keys；
- forward shapes；
- logits；
- CAM；

与修改前代码保持一致。

这是必须通过的 regression test。

---

# 19. Smoke Training

正式训练前先跑：

```text
100–500 iterations
```

检查：

1. loss finite；
2. class query gradient finite；
3. class query 参数在更新；
4. pooling attention 无 NaN；
5. attention row sums = 1；
6. CAM forward 正常；
7. checkpoint save/load 正常；
8. resume training 正常；
9. make_cam 正常加载 CWP checkpoint。

---

# 20. CWP 专属训练诊断

每个 epoch 或固定 interval 记录：

## Pooling entropy

\[
H_c
=
-
\frac{
\sum_iA_{c,i}\log A_{c,i}
}{
\log N
}.
\]

范围：

\[
0\le H_c\le1.
\]

如果：

\[
H\approx1
\]

表示接近 uniform pooling。

记录：

```text
mean
std
min
max
```

---

## Inter-class pooling similarity

对于：

\[
A_c,A_k
\]

计算 pairwise cosine：

\[
Sim_A
=
\operatorname{mean}_{c\neq k}
\cos(A_c,A_k).
\]

用于判断 20 个 class queries 是否最终仍然在做几乎相同的 pooling。

这里只做 diagnostic，不加 loss。

---

## Initial-token similarity

计算：

\[
Sim_T^{(0)}
=
\operatorname{mean}_{c\neq k}
\cos(t_c^{(0)},t_k^{(0)}).
\]

记录训练过程。

同样只观察，不参与优化。

---

# 21. 正式实验矩阵

第一轮只需要：

| ID | Initialization |
|---|---|
| B0 | Original MCTformer+ duplicated DeiT CLS |
| E1 | CWP: \(T^{(0)}=AP+E_{\mathrm{cls}}\) |

禁止第一轮增加更多 CWP variants。

B0 可以复用现有严格匹配的 baseline result，不必因为代码重构而重复长时间训练；但必须用 regression test 确认 baseline path 没有发生变化。

E1：

从 official DeiT-S pretrained checkpoint **重新完整训练**。

不能从训练好的 MCTformer+ baseline checkpoint 开始。

---

# 22. 训练配置

E1 必须完全继承 B0：

- optimizer；
- LR；
- weight decay；
- batch size；
- epochs；
- augmentation；
- input 448；
- losses；
- loss weights；
- random seed；
- dataset split；
- checkpoint policy。

唯一 architecture difference：

```text
class_token_init:
baseline → cwp
```

---

# 23. 最终 WSSS Evaluation

完成训练后，沿用当前 MCTformer+ 完整 evaluation pipeline。

至少报告与 baseline 完全一致的：

```text
image-level / classification metrics
patch classification metrics
CAM seed mIoU
final CAM mIoU
pseudo-mask / segmentation metrics
```

具体指标名称以仓库当前 baseline 输出为准，不要自行创造另一套 evaluation protocol。

---

# 24. 必须额外重新运行 shared-presence analysis

因为 CWP 的主要动机之一就是改变 class token 的 formation trajectory。

训练完成后，对 E1 checkpoint 重新运行现有 frozen relation analysis。

必须和原始 MCTformer+ checkpoint 使用：

- 同一个 VOC-val；
- 同一个 1449 images；
- 同一个 resize；
- 同一个 metrics implementation。

额外增加：

\[
\boxed{Layer\ 0}
\]

即：

\[
T^{(0)}
\]

进入 Block 1 之前的 relation。

---

# 25. Layer-0 分析

对：

\[
T^{(0)}\in\mathbb R^{C\times D}
\]

和：

\[
P^{(0)}\in\mathbb R^{N\times D}
\]

计算：

\[
R_{c,i}^{(0)}
=
\frac{
(t_c^{(0)})^\top p_i^{(0)}
}{
\sqrt D
}.
\]

然后报告：

```text
pairwise positive-class relation correlation
Top-5% collision
class-token pairwise cosine
```

之后继续原有：

```text
Layer 1
...
Layer 12
```

分析。

---

# 26. Shared-presence 主比较表

最终生成：

| Layer | Baseline Corr | CWP Corr | Baseline Collision@5 | CWP Collision@5 |
|---:|---:|---:|---:|---:|
| 0 | N/A | | N/A | |
| 1 | | | | |
| ... | | | | |
| 12 | | | | |

以及：

| Layer | Baseline common \(R^2\) | CWP common \(R^2\) | Baseline effective rank | CWP effective rank |
|---:|---:|---:|---:|---:|

不要因为 CWP 结果自动扩展新实验。

---

# 27. 最终需要回答的问题

实验报告只需要回答以下问题。

### Q1

完全删除 duplicated DeiT CLS 后，CWP 能否正常训练并达到至少接近 baseline 的 WSSS performance？

### Q2

\[
T^{(0)}(x)=AP+E_{\mathrm{cls}}
\]

是否真的产生 image-conditioned、class-dependent initial tokens？

### Q3

CWP 的 class-wise attention 是否从近似 uniform pooling 逐渐形成 class-dependent spatial weighting？

### Q4

相比 duplicated CLS initialization，CWP 是否改变了：

\[
L0\rightarrow L12
\]

multi-class token shared-relation 的形成过程？

### Q5

最终 segmentation/CAM performance 的变化是多少？

本轮不要自动设计下一代 CWP。