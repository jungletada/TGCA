# Decoupled Multi-Class/Patch Alternating Transformer for MCTformer+

## 1. 实验目标

本实验停止继续优化 Residual CWP。Residual CWP、CWP、FinalLN、LaST、TGCA、BCSS、PSL 和 CTI-BGT 均不与本实验组合。

本实验只研究一个问题：MCTformer+ 将 multi-class tokens 与 patch tokens 拼接后送入同一个全局 self-attention，是否会让 patch stream 反向读取 class-token information，并损害 patch 的空间语义所有权。

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
}
\]

其中 patch tokens 从不读取 class tokens：

\[
P^l=f_l(P^{l-1}),
\]

而 class tokens 先读取当前层已经更新的 patch tokens，再进行 class-token 内部交互：

\[
\widetilde C^l=g_l(C^{l-1},P^l),
\qquad
C^l=h_l(\widetilde C^l).
\]

首轮方法暂命名为：

```text
MCTformer+-DecoupledAlternating
```

配置名固定为：

```text
token_interaction = joint | decoupled_alternating
```

默认值为 `joint`，必须保持当前 MCTformer+ 数值行为不变。

---

## 2. 模型核心代码先行

实现时首先完成模型核心模块、单元测试和 official DeiT 权重映射。训练 runner 和 CAM runner 只能在核心模块测试通过后接入。

首轮不复制三套独立 attention，也不引入新的 projection、gate 或 loss。每个 layer 继续只保留原 DeiT block 的一套：

```text
norm1
attn.qkv
attn.proj
norm2
mlp
drop_path
```

这套参数在三种 attention operation 中共享。其目的有三点：

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
        # query_tokens:     [B,C,D]
        # key_value_tokens: [B,N,D]
        # returns output [B,C,D], weights [B,H,C,N]
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

这样 cross-attention 的 query 是 multi-class tokens，key/value 只来自 patch tokens：

\[
A_{c2p}^l
=
\operatorname{softmax}_{N}
\left(
Q(C)K(P)^\top/\sqrt{d_h}
\right).
\]

不得让 cross-attention 的 key/value 包含 class tokens。

### 2.3 每层的固定更新顺序

每层只使用一次 patch MLP 和一次 class MLP。Cross-attention 后不额外增加 MLP，以避免 class stream 每层拥有两套 FFN update。

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

        attention = {
            "patch_to_patch": attn_p2p,  # [B,H,N,N]
            "class_to_patch": attn_c2p,  # [B,H,C,N]
            "class_to_class": attn_c2c,  # [B,H,C,C]
        }
        return class_tokens, patch_tokens, attention
```

这里 `norm1`、`norm2`、`attn`、`mlp` 和 `drop_path` 均是同一个 layer 内从 original DeiT block 继承的一套模块，不是三份独立参数。

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

`all_x_cls` 保存每层完整完成以下三步后的 raw class tokens：

```text
P-SA -> C2P cross-attention -> C-SA -> class MLP
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

新 attention 的 key group 是单一同质集合，因此每个 `A_c2p` 和 `A_p2p` row 自然各自和为 1。不得为了模拟原 joint-softmax group mass 再增加人工缩放；这属于新拓扑的内在归一化，不是修改 CAM 公式。

不得伪造一个 `[C+N,C+N]` joint attention matrix。模型应返回结构化 attention：

```python
{
    "class_to_patch": ...,
    "patch_to_patch": ...,
    "class_to_class": ...,
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

因此同一个 pretrained block 参数由三个 operation 共享调用，不进行复制。`patch_embed`、12 个 blocks 和位置编码继续使用同一个 official DeiT-S checkpoint。

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

虽然参数量相同，cross-attention 和第二次 class-side attention 会增加投影调用，因此必须实测训练显存、吞吐和推理延迟，不得仅依据 attention matrix 大小声称零开销。

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
- reverse patch-to-class cross-attention；
- background/register tokens；
- 额外 LayerNorm、MLP 或 classifier；
- layer-order、层数或超参数 sweep。

---

## 7. 必要测试

### 7.1 核心 attention

1. Patch self-attention 输出 `[B,N,D]`，权重 `[B,H,N,N]`；
2. Class-to-patch cross-attention 输出 `[B,C,D]`，权重 `[B,H,C,N]`；
3. Class self-attention 输出 `[B,C,D]`，权重 `[B,H,C,C]`；
4. 三种 attention 的 row sum 均为 1；
5. cross-attention 与显式 `F.linear` reference 数值一致；
6. 所有 operation 使用同一个 `qkv` 和 `proj` 参数对象。

### 7.2 解耦不变量

固定 patch input，只扰动初始 class tokens：

```text
final patch tokens 必须 bit-exact 不变
```

固定 class tokens、扰动 patch input：

```text
class tokens 必须发生变化
```

检查 patch stream 的 autograd graph 不依赖 `cls_token`：仅对最终 patch-token scalar backward 时，`cls_token.grad` 必须为 `None` 或严格为零。

### 7.3 顺序与训练输出

1. Hook 验证每层顺序严格为 P-SA、C2P、C-SA；
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
- 12 层三种 attention 有限且 row sum 正确；
- patch-output 对 class-token perturbation 不变；
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

除 `--token-interaction decoupled_alternating` 外，命令与 canonical original MCTformer+ seed-0 run 一致。从 official DeiT-S 初始化开始，不从任何训练好的 MCTformer+、CWP 或 Residual-CWP checkpoint fine-tune。

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
- raw patch CAM、class-attention CAM 和 final propagated CAM 的语义所有权；
- patch-token class-similarity trajectory。

新模型不存在 `A_p2c`，因此该方向报告为“architecturally absent”，不得填零后与 baseline 当作连续数值比较。

所有不确定性分析以 image 为 cluster 做 paired bootstrap；不得把同图 patch 或 image-class pair 当成独立样本。

### 9.3 解释边界

如果 CAM 提升且 patch semantic ownership 改善，可以说明禁止 patch 读取 class tokens 与结果一致，但单个训练对照仍不能把所有变化唯一归因于某一层。

如果 localization 下降，即使 patch leakage 指标改善，也必须报告 decoupling 删除了可能有用的 class-conditioned contextualization，不得只强调纯度指标。

如果仅 best threshold 改善而 fixed threshold 不改善，应优先解释为 calibration 改变。

---

## 10. Go / No-Go

首轮判定为值得继续，至少需要同时满足：

1. fixed-threshold CAM mIoU 不低于 baseline，并有实际正增益；
2. target-vs-BG 或 C-PiM 在预注册 late layers 稳定改善；
3. class-token 与 patch-head classification 没有明显下降；
4. 改善不是只来自 method-specific threshold；
5. 开销仍可接受。

若 fixed CAM、best CAM 和 classification 均不改善，则停止该方向，不自动实现独立参数、gate、reverse cross-attention 或其他 ordering。

---

## 11. 结果与复现文件

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
