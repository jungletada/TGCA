# Decoupled Bidirectional Multi-Class/Patch Alternating Transformer for MCTformer+

## 1. 实验目标

本实验停止继续优化 Residual CWP。Residual CWP、CWP、FinalLN、LaST、TGCA、BCSS、PSL 和 CTI-BGT 均不与本实验组合。

本实验只研究一个问题：MCTformer+ 将 multi-class tokens 与 patch tokens 拼接后送入同一个全局 self-attention，使两类 token 共享同一 attention normalization；将它们改为不拼接的 role-specific self/cross attention，同时保留双向信息交换，是否能改善分类与定位。

原始 MCTformer+ 每层为：

\[
X^{l-1}=[C^{l-1};P^{l-1}],
\qquad
X^l=\operatorname{ViTBlock}_l(X^{l-1}).
\]

新的固定顺序为：

\[
\boxed{
P\text{-Self Attention}
\rightarrow
C\leftarrow P\text{ Cross Attention}
\rightarrow
C\text{-Self Attention}
\rightarrow
P\leftarrow C\text{ Cross Attention}
}
\]

其中两个 stream 均能读取对方的信息，但任何一次 attention 都不拼接两类 token。Patch tokens 先完成自身空间更新，class tokens 再读取更新后的 patch tokens并完成 class self-attention，最后 patch tokens 读取本层已经更新的 class tokens：

\[
\widehat P^l=f_l(P^{l-1}),
\]
\[
\widetilde C^l=g_l(C^{l-1},\widehat P^l),
\qquad
C^l=h_l(\widetilde C^l),
\qquad
P^l=r_l(\widehat P^l,C^l).
\]

因此这不是切断 class-to-patch 信息，而是把原 joint self-attention 中的四种 token relation 拆成显式的 `P2P`、`C2P`、`C2C` 和 `P2C` operation。

首轮方法暂命名为：

```text
MCTformer+-DecoupledBidirectional
```

配置名固定为：

```text
token_interaction = joint | decoupled_bidirectional
```

默认值为 `joint`，必须保持当前 MCTformer+ 数值行为不变。

---

## 2. 模型核心代码先行

实现时首先完成模型核心模块、单元测试和 official DeiT 权重映射。训练 runner 和 CAM runner 只能在核心模块测试通过后接入。

首轮不复制四套独立 attention，也不引入新的 projection、gate 或 loss。每个 layer 继续只保留原 DeiT block 的一套：

```text
norm1
attn.qkv
attn.proj
norm2
mlp
drop_path
```

这套参数在四种 attention operation 中共享。其目的有三点：

1. block 参数量与原始 MCTformer+ 完全相同；
2. official DeiT-S block 权重能够逐键、严格加载；
3. 结果主要反映 token interaction topology 的改变，而不是容量增加。

### 2.1 输入 token

完全保留原始 MCTformer+-Small 的 patch embedding：

```python
patch_tokens = self.patch_embed(images)
patch_tokens = patch_tokens + pos_embed_pat
```

完全保留原始 multi-class token 初始化：

```python
class_tokens = self.cls_token.expand(batch_size, -1, -1)
class_tokens = class_tokens + self.pos_embed_cls
```

其中 official DeiT-S singleton CLS token 仍然 repeat 为 20 个 `self.cls_token`。不得使用 CWP 或 Residual CWP。

位置编码加入一次后，两个 stream 分开保存：

```python
class_tokens = self.pos_drop(class_tokens)
patch_tokens = self.pos_drop(patch_tokens)
```

不得再执行：

```python
torch.cat([class_tokens, patch_tokens], dim=1)
```

### 2.2 共享投影的 attention 核心

保留原 block 的 `qkv: Linear(D,3D)` 和 `proj: Linear(D,D)` 参数命名。核心接口设计为：

```python
class SharedRoleAttention(nn.Module):
    def project_qkv(self, tokens):
        ...

    def self_attention(self, tokens):
        # tokens: [B, T, D]
        # returns output [B,T,D], weights [B,H,T,T]
        ...

    def cross_attention(self, query_tokens, key_value_tokens):
        # Supports C<-P and P<-C without concatenating the streams.
        # returns output shaped like query_tokens and weights [B,H,Tq,Tkv]
        ...
```

Self-attention 使用原始计算：

```python
qkv = self.qkv(tokens)
q, k, v = split_heads(qkv)
weights = softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
output = self.proj(merge_heads(self.attn_drop(weights) @ v))
```

Cross-attention 使用同一个 `qkv.weight` 和 `qkv.bias` 的对应切片，而不是新增投影参数：

```python
q = F.linear(
    query_tokens,
    self.qkv.weight[:D],
    None if self.qkv.bias is None else self.qkv.bias[:D],
)
k = F.linear(
    key_value_tokens,
    self.qkv.weight[D:2 * D],
    None if self.qkv.bias is None else self.qkv.bias[D:2 * D],
)
v = F.linear(
    key_value_tokens,
    self.qkv.weight[2 * D:],
    None if self.qkv.bias is None else self.qkv.bias[2 * D:],
)
weights = softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
output = self.proj(merge_heads(self.attn_drop(weights) @ v))
```

同一个接口分别计算两个方向。Class-to-patch 的 query 是 multi-class tokens，key/value 只来自 patch tokens：

\[
A_{c2p}^l
=
\operatorname{softmax}_{N}
\left(
Q(C)K(P)^\top/\sqrt{d_h}
\right).
\]

Patch-to-class 则反向使用 patch queries 和 class-only key/value：

\[
A_{p2c}^l
=
\operatorname{softmax}_{C}
\left(
Q(P)K(C)^\top/\sqrt{d_h}
\right).
\]

两个方向都必须保持 query stream 与 key/value stream 分离；不得在 cross-attention 内重新拼接 token。

### 2.3 每层的固定更新顺序

每层只使用一次 patch MLP 和一次 class MLP。两个 cross-attention 后都不额外增加 MLP，以避免任一 stream 每层拥有重复的 FFN update。

核心 forward 固定为：

```python
class DecoupledAlternatingBlock(nn.Module):
    def forward(self, class_tokens, patch_tokens):
        # Part 1: patch self-attention, then the original shared MLP.
        patch_delta, attn_p2p = self.attn.self_attention(
            self.norm1(patch_tokens)
        )
        patch_tokens = patch_tokens + self.drop_path(patch_delta)
        patch_tokens = patch_tokens + self.drop_path(
            self.mlp(self.norm2(patch_tokens))
        )

        # Part 2: class queries read the already-updated patch stream.
        cross_delta, attn_c2p = self.attn.cross_attention(
            self.norm1(class_tokens),
            self.norm1(patch_tokens),
        )
        class_tokens = class_tokens + self.drop_path(cross_delta)

        # Part 3: class-token self-attention, then the shared MLP.
        class_delta, attn_c2c = self.attn.self_attention(
            self.norm1(class_tokens)
        )
        class_tokens = class_tokens + self.drop_path(class_delta)
        class_tokens = class_tokens + self.drop_path(
            self.mlp(self.norm2(class_tokens))
        )

        # Part 4: patch queries read the fully updated class stream.
        patch_cross_delta, attn_p2c = self.attn.cross_attention(
            self.norm1(patch_tokens),
            self.norm1(class_tokens),
        )
        patch_tokens = patch_tokens + self.drop_path(patch_cross_delta)

        attention = {
            "patch_to_patch": attn_p2p,  # [B,H,N,N]
            "class_to_patch": attn_c2p,  # [B,H,C,N]
            "class_to_class": attn_c2c,  # [B,H,C,C]
            "patch_to_class": attn_p2c,  # [B,H,N,C]
        }
        return class_tokens, patch_tokens, attention
```

这里 `norm1`、`norm2`、`attn`、`mlp` 和 `drop_path` 均是同一个 layer 内从 original DeiT block 继承的一套模块，不是四份独立参数。第四步不改变 class tokens，所以该层的 CCT class-token snapshot 可以在第三步之后或整个 block 返回时记录；两者必须数值相同。

### 2.4 12 层主干

```python
all_x_cls = []
attention_records = []

for block in self.blocks:
    class_tokens, patch_tokens, attention = block(
        class_tokens, patch_tokens
    )
    all_x_cls.append(class_tokens)
    attention_records.append(attention)

return class_tokens, patch_tokens, attention_records, all_x_cls
```

最终不应用 FinalLN 或 PatchFinalLN：

```text
final_norm       = false
patch_final_norm = false
```

这与当前 canonical original MCTformer+ readout 保持一致。

---

## 3. 必须保持不变的训练分支

### 3.1 Class-token classification

保持：

```python
class_logits = class_tokens.mean(dim=-1)
class_loss = F.multilabel_soft_margin_loss(class_logits, targets)
```

不得增加新的 class classifier。

### 3.2 CCT

`all_x_cls` 保存每层 class stream 完整更新后的 raw class tokens；记录发生在整个四阶段 block 返回时：

```text
P-SA -> patch MLP -> C2P cross-attention -> C-SA -> class MLP
     -> P2C cross-attention
```

CCT 的公式、层数、权重和调用位置全部沿用 MCTformer+。不得对 `all_x_cls` 应用 FinalLN，不得增加辅助 relation loss。

### 3.3 Patch classification

最终 raw patch tokens 继续走：

```text
reshape
-> original Conv2d(D,20,kernel_size=3,padding=1)
-> original GWRP
-> original patch multilabel loss
```

Patch head、GWRP decay、patch loss 权重均不改变。

### 3.4 总损失

保持当前 MCTformer+：

\[
\mathcal L
=
\mathcal L_{class}
+
\mathcal L_{CCT}
+
\mathcal L_{patch}.
\]

不增加 cross-attention supervision、consistency、contrastive、orthogonality、entropy 或 background loss。

---

## 4. CAM 生成策略保持 MCTformer+ 不变

不得把 class self-attention 当作 CAM，也不得设计新的 attention fusion。

### 4.1 Native patch CAM

最终 patch tokens 经过原始 3×3 patch classifier：

\[
M=H_{3\times3}(P^{12}),
\qquad
M^+=\operatorname{ReLU}(M).
\]

### 4.2 Last-three class-to-patch refinement

原始 MCTformer+ 从 joint attention 中截取最后三层 `A_c2p`。新模型直接使用最后三层 cross-attention 权重，但聚合公式不变：

\[
\bar A_{c2p}
=
\frac{1}{3H}
\sum_{l=10}^{12}
\sum_{h=1}^{H}
A_{c2p}^{l,h}.
\]

然后保持：

\[
M_{att}=\bar A_{c2p}\odot M^+,
\qquad
M_{sqrt}=\sqrt{M_{att}}.
\]

### 4.3 All-layer patch-to-patch propagation

原始 MCTformer+ 从全部 12 层 joint attention 中截取 `A_p2p` 并求和。新模型直接使用每层 patch self-attention：

\[
\bar A_{p2p}
=
\sum_{l=1}^{12}
\frac{1}{H}
\sum_{h=1}^{H}
A_{p2p}^{l,h}.
\]

最终仍为：

\[
CAM_c
=
\bar A_{p2p}M_{sqrt,c}.
\]

保留 `forward_with_label` 中原始 class-logit 与 GWRP patch-logit gating，保留 `make_cam.py` 的 multi-scale、flip、resize、per-class normalization 和保存格式。

新 attention 的 key group 是单一同质集合，因此 `A_c2p`、`A_p2c`、`A_p2p` 和 `A_c2c` 的每个 row 都自然和为 1。不得为了模拟原 joint-softmax group mass 再增加人工缩放；这属于新拓扑的内在归一化，不是修改 CAM 公式。

`A_p2c` 参与 patch-token feature update，但不加入 CAM refinement。原生 MCTformer+ CAM 只读取 `A_c2p` 和 `A_p2p`，这一 readout contract 保持不变。

不得伪造一个 `[C+N,C+N]` joint attention matrix。模型应返回结构化 attention：

```python
{
    "class_to_patch": ...,
    "patch_to_patch": ...,
    "class_to_class": ...,
    "patch_to_class": ...,
}
```

CAM 和分析代码从对应字段读取真实 attention。

---

## 5. DeiT-S 预训练映射

首轮只支持普通 MCTformer+-Small：

```text
embed_dim = 384
depth = 12
heads = 6
patch size = 16
```

每个 decoupled block 保持以下 state-dict key 与 original DeiT 一致：

```text
blocks.L.norm1.*
blocks.L.attn.qkv.*
blocks.L.attn.proj.*
blocks.L.norm2.*
blocks.L.mlp.*
```

因此同一个 pretrained block 参数由四个 operation 共享调用，不进行复制。`patch_embed`、12 个 blocks 和位置编码继续使用同一个 official DeiT-S checkpoint。

CLS token 与当前 MCTformer+ 相同：

```python
self.cls_token = source_cls_token.repeat(1, 20, 1)
self.pos_embed_cls = source_cls_position.repeat(1, 20, 1)
```

3×3 patch classifier 保持随机初始化。除原 MCTformer+ 已有随机参数外不得增加新参数。

要求审计：

```text
trainable_parameter_count(decoupled) == trainable_parameter_count(baseline)
```

虽然参数量相同，四个 operation 会重复调用共享的 QKV/output projection。四张 attention score matrix 的元素总数满足

\[
N^2+NC+C^2+CN=(N+C)^2,
\]

但 projection 计算量高于一次 joint attention，因此必须实测训练显存、吞吐和推理延迟，不得仅依据 attention matrix 元素总数声称零开销。

---

## 6. 首轮兼容范围

只支持：

```text
MCTformer+ Small
class_token_init = baseline
vanilla attention
BCSS E0
PSL baseline
CTI-BGT disabled
final_norm = false
patch_final_norm = false
last_mct = false
class_stable_last = false
```

遇到其他可选 variant 时显式拒绝，不扩展范围。

首轮不实现：

- Residual CWP 或 CWP；
- TGCA/split softmax；
- 独立的 patch/class/cross 参数；
- cross-attention gate；
- 必需的双向 cross-attention 之外的第三种 cross/gating 路径；
- background/register tokens；
- 额外 LayerNorm、MLP 或 classifier；
- layer-order、层数或超参数 sweep。

---

## 7. 必要测试

### 7.1 核心 attention

1. Patch self-attention 输出 `[B,N,D]`，权重 `[B,H,N,N]`；
2. Class-to-patch cross-attention 输出 `[B,C,D]`，权重 `[B,H,C,N]`；
3. Class self-attention 输出 `[B,C,D]`，权重 `[B,H,C,C]`；
4. Patch-to-class cross-attention 输出 `[B,N,D]`，权重 `[B,H,N,C]`；
5. 四种 attention 的 row sum 均为 1；
6. 两个 cross-attention 均与显式 `F.linear` reference 数值一致；
7. 所有 operation 使用同一个 `qkv` 和 `proj` 参数对象。

### 7.2 解耦不变量

固定 patch input，只扰动初始 class tokens：

```text
Part 1 的 patch-self 输出必须不变；完整 block 的 final patch tokens 必须变化
```

固定 class tokens、扰动 patch input：

```text
class tokens 必须发生变化
```

检查双向 autograd：final patch-token scalar backward 必须能向 `cls_token` 传播有限非零梯度；final class-token scalar backward 必须能向 patch input/embedding 传播有限非零梯度。

另外验证 relation topology：`A_c2p` 的 key 轴严格只有 `N` 个 patch tokens，`A_p2c` 的 key 轴严格只有 `C` 个 class tokens，任何 cross operation 内都不存在 `[C;P]` 拼接。

### 7.3 顺序与训练输出

1. Hook 验证每层顺序严格为 P-SA、C2P、C-SA、P2C；
2. `all_x_cls` 恰好包含 12 层完成更新后的 raw class tokens；
3. class logits、patch logits、loss shapes 与 baseline 一致；
4. CCT 接收新 `all_x_cls`，其公式未变；
5. backward 后 qkv、MLP、class token、patch embedding 和 patch head 均有有限梯度；
6. AMP forward/backward 有限。

### 7.4 Pretrained 与 baseline compatibility

1. official DeiT-S checkpoint 能严格加载；
2. pretrained mapping 没有 missing/unexpected key；
3. decoupled 与 baseline block 参数量完全相同；
4. `token_interaction=joint` 与当前 checkpoint/code 数值完全一致；
5. checkpoint metadata 必须记录 interaction mode，避免 joint/decoupled 混载。

### 7.5 Native CAM

用小张量手算并验证：

```text
ReLU patch CAM
-> last-three A_c2p multiply
-> sqrt
-> all-layer A_p2p propagation
```

训练模型和 CAM 模型必须严格加载同一 checkpoint，`forward_with_label` 必须仍使用 class logits 与 GWRP patch logits。

---

## 8. Smoke 与正式训练

### 8.1 Smoke

使用 `tgca-repro`、VOC 448、原始训练入口 `train_model_v2.py`：

```text
seed = 0
100 optimizer steps
batch size = 32
accum_iter = 1
```

确认：

- 三项原始 loss 有限；
- 12 层四种 attention 有限且 row sum 正确；
- class 与 patch perturbation test 证明双向信息交换均有效；
- checkpoint 严格加载；
- 4 张 smoke CAM 能生成；
- 不增加额外 loss。

### 8.2 Full matched training

只运行一个首轮训练：

```text
dataset: VOC train_aug
input: 448
model: MCTformer+-Small
seed: 0
epochs: 45
batch size: 32
effective batch: 32
optimizer: AdamW
nominal LR: 5e-4
minimum LR: 1e-5
weight decay: 0.05
schedule: cosine
warmup: 5 epochs
drop path: 0.1
pretrained: official DeiT-S
```

除 `--token-interaction decoupled_bidirectional` 外，命令与 canonical original MCTformer+ seed-0 run 一致。从 official DeiT-S 初始化开始，不从任何训练好的 MCTformer+、CWP 或 Residual-CWP checkpoint fine-tune。

原始 MCTformer+ 结果已存在，不重训 baseline。

---

## 9. Evaluation

### 9.1 标准 MCTformer+ 指标

使用与 baseline 相同的 transform、split、single checkpoint policy 和 CAM threshold grid，报告：

- class-token macro mAP；
- patch-head macro mAP；
- class-token validation loss；
- patch validation loss；
- fixed threshold 0.45 raw/native CAM mIoU；
- 同一 `[0.00,0.59]` grid 上的 best CAM mIoU 与 threshold；
- foreground precision / recall；
- parameter count、训练显存、吞吐和 inference latency。

### 9.2 解耦机制指标

复用 Experiment 2 的 VOC semantic ownership 定义和 deterministic 448 pipeline，不在训练中加载 segmentation GT。

重点比较 baseline 与 decoupled model 的 L4、L5、L9–L12：

- class–patch feature C-PiM；
- target-vs-background AUROC/AUPRC；
- target-vs-other-foreground AUROC/AUPRC；
- positive-class-pair Top-10 Jaccard；
- target / other-FG / background top-k composition；
- `A_c2p` 的 conditional region mass；
- `A_p2c` 对 present/absent class keys 的 conditional mass、entropy 和 top-1 class 命中率；
- raw patch CAM、class-attention CAM 和 final propagated CAM 的语义所有权；
- patch-token class-similarity trajectory。

新模型的 `A_p2c` 是独立归一化的真实 cross-attention。与 baseline 比较时，baseline joint attention 的 patch-query/class-key slice 必须先在 class-key 轴上重新条件归一化；不得把 baseline 未归一化的 group mass 与新模型 row-sum-one 的 `A_p2c` 直接比较。

所有不确定性分析以 image 为 cluster 做 paired bootstrap；不得把同图 patch 或 image-class pair 当成独立样本。

### 9.3 解释边界

如果 CAM 提升且 patch semantic ownership 改善，只能说明 role-specific、双向、非拼接 attention 与结果一致。由于 operation 被串行化且每层调用共享 projection 多次，单个训练对照不能把增益唯一归因于取消 concatenation 或某一个 relation。

如果 localization 下降，即使某些语义所有权指标改善，也必须报告 joint normalization 或并行 relation mixing 可能是有用的 inductive bias，不得只强调纯度指标。

如果仅 best threshold 改善而 fixed threshold 不改善，应优先解释为 calibration 改变。

---

## 10. 后续消融候选（不混入首版实现）

首版代码和首次训练固定使用完整四阶段顺序，不加入组合式开关。确认主模型可以稳定训练后，再按下面顺序做消融；所有消融必须保持初始化、seed、优化器、训练长度、loss、patch head 和 CAM threshold protocol 一致。

### 10.1 Relation necessity：第一优先级

先比较四个最小模型：

| Variant | P2P | C2P | C2C | P2C | 目的 |
| --- | --- | --- | --- | --- | --- |
| Full | 更新 | 更新 | 更新 | 更新 | 完整双向模型 |
| No-P2C | 更新 | 更新 | 更新 | 删除 | 判断 class-to-patch 回写是否必要 |
| No-C2C | 更新 | 更新 | 删除 | 更新 | 判断 class tokens 内部通信是否必要 |
| No-P2C-No-C2C | 更新 | 更新 | 删除 | 删除 | 最小 patch encoder + class readout |

这里“删除”表示不计算该 attention 且不执行 residual update。`P2C` 和 `C2C` 不参与原生 CAM，能够干净删除。

`C2P` 和 `P2P` 是 MCTformer+ CAM 的必要 readout：last-three `A_c2p` 与 all-layer `A_p2p`。因此若研究它们对 feature update 的作用，必须继续计算 attention weights，仅关闭对应的 residual value update：

```text
C2P-update-off: compute A_c2p for CAM, do not add C2P value output to C
P2P-update-off: compute A_p2p for CAM, do not add P2P value output to P
```

不得把“关闭 residual update”和“完全删除 attention 计算”混为同一个消融。完全删除 `C2P` 或 `P2P` 后已经无法维持原生 MCTformer+ CAM，不能放入同一主表做严格比较。

### 10.2 P2C 插入时机：第二优先级

不枚举全部 `4!` 排列，只比较三个有明确语义的 P2C 位置：

1. `P2P → C2P → C2C → P2C`：主模型，patch 读取本层 fully updated class；
2. `P2P → C2P → P2C → C2C`：patch 读取 C2P 后、C2C 前的 class；
3. `P2P → P2C → C2P → C2C`：patch 先读取上一状态 class，class 随后读取已回写的 patch。

Patch MLP 始终跟随 P2P，class MLP 始终跟随 C2C，不随 P2C 位置移动。这样 order ablation 只改变跨流信息的新旧程度，不同时改变 FFN 次数。

如果这三种顺序差异很小，不继续扩展到其余排列。如果差异显著，可增加一个同步 cross control：从同一对 snapshot 同时计算 C2P/P2C delta，再同时 residual update，用于去除先后方向偏置。

### 10.3 次级消融

仅在 relation necessity 和 order 已有明确结果后考虑：

- decoupling layer scope：只改 L1–L4、L5–L8 或 L9–L12；
- 共享 QKV/proj 对比独立 projection；后者增加参数，必须单独报告容量差异；
- shared LayerNorm 对比 stream-specific LayerNorm；后者同样增加参数；
- cross residual 固定缩放，但不得在首轮搜索可学习 gate；
- synchronous cross 对比 sequential cross。

首轮不做独立参数、LayerNorm、gate 或 layer-scope 消融，避免把“取消 concatenation”的结果与新增容量混合。

---

## 11. Go / No-Go

首轮判定为值得继续，至少需要同时满足：

1. fixed-threshold CAM mIoU 不低于 baseline，并有实际正增益；
2. target-vs-BG 或 C-PiM 在预注册 late layers 稳定改善；
3. class-token 与 patch-head classification 没有明显下降；
4. 改善不是只来自 method-specific threshold；
5. 开销仍可接受。

若 fixed CAM、best CAM 和 classification 均不改善，则停止该方向，不自动实现独立参数、gate、额外 cross-attention 或其他 ordering。

---

## 12. 结果与复现文件

最终保存：

```text
exact_commands.sh
config.json
git_state.json
environment.txt
tests.txt
smoke_summary.json
checkpoint_manifest.txt
classification_metrics.json
cam_evaluation/metrics.json
semantic_ownership_summary.csv
efficiency.json
DECOUPLED_ALTERNATING_REPORT.md
summary.json
```

报告必须明确区分：

- 已完成的训练和测量；
- baseline 复用来源；
- architecture-level observation；
- 尚未由因果实验证明的解释。

大 checkpoint、raw CAM 和大日志不提交 Git；只在训练完成后提交紧凑报告、表格与 metadata。

完成该首轮实验后停止，不自行增加其他 decoupling variants。
