# Last-MCT：LaST-ViT Patch Aggregation 迁移到 PatchFinalLN MCTformer+ 的 Codex 执行计划

> **目标仓库：** `https://github.com/jungletada/TGCA`  
> **服务器仓库：** `~/code/TGCA`  
> **目标分支：** 当前 `main` 工作树  
> **直接基线：** 已完成的 **PatchFinalLN MCTformer+**  
> **本轮目标：** 只实现并训练一个 LaST-faithful patch-classification branch。  
> **禁止加入：** semantic competition、BG token、BCSS/PSL/CTI 新变体、attention intervention、额外 loss、额外 tuning。

---

# 1. Codex 权限

本任务已获得以下明确授权：

- 可以修改当前 TGCA 仓库中的代码；
- 可以新增必要的测试和紧凑结果脚本；
- 可以运行单元测试、smoke test 和完整训练；
- 可以使用服务器 GPU 启动完整 VOC 训练；
- **测试通过后可以直接执行 `git commit`，无需再次向用户确认；**
- **完整训练可以直接启动，无需再次向用户确认；**
- 训练结束后可以再提交 compact metrics / report；
- 不提交大 checkpoint、cache、原始大日志或大规模 per-image artifacts；
- **不要 `git push`，除非用户之后明确要求；**
- 不允许 `git reset --hard`、强制覆盖、rebase/force-push；
- 若工作树已有与本任务无关的用户修改，必须保留，不得删除或覆盖。

建议实现 commit message：

```text
Add LaST-style patch aggregation to MCTformer+
```

若训练结果另作一次 compact commit：

```text
Record Last-MCT matched VOC results
```

---

# 2. 本轮唯一方法定义

保持 MCTformer+ 的 class-token branch 完全不变：

```text
raw L12 class tokens
→ mean(dim=-1)
→ class-token logits
→ original class-token multi-label loss
```

CCT 仍使用每个 block 的 raw post-block class tokens。

只修改 patch branch。

## 当前 PatchFinalLN baseline

```text
raw L12 patch tokens
→ Final LayerNorm
→ 3×3 Conv classifier
→ GWRP
→ patch logits
```

## Last-MCT

```text
raw L12 patch tokens
→ Final LayerNorm
→ LaST FFT / Gaussian low-pass
→ stability score
→ channel-wise Top-K patch selection
→ pooled global patch token
→ 1×1 Conv / Linear-equivalent classifier
→ patch logits
```

CAM 使用与 pooled-token classification **同一组 1×1 classifier weights**：

```text
FinalLN patch feature map
→ same 1×1 classifier
→ patch CAM
→ original MCTformer+ last-3 A_c2p refinement
→ sqrt
→ original all-layer A_p2p propagation
→ final CAM
```

---

# 3. LaST 官方核心逻辑

实现以 LaST 官方 repo 的实际 `repo` 公式为准：

```python
# P: [B, N, D]

low = torch.fft.fft(P, dim=-1)
low = torch.fft.fftshift(low, dim=-1)
low = low * gaussian_kernel
low = torch.fft.ifftshift(low, dim=-1)
low = torch.fft.ifft(low, dim=-1).real

score = P / abs(low - P)

_, indices = torch.topk(
    score,
    k=K,
    dim=1,
    largest=True,
)

selected = torch.gather(
    P,
    dim=1,
    index=indices,
)

pooled = selected.mean(dim=1)
```

第一轮固定：

```text
K = 1
sigma = sqrt(embed_dim)
eps = 1e-6
score_formula = repo-safe
```

其中：

```python
score = P / (low - P).abs().clamp_min(eps)
```

不要实现 paper-formula 作为第一轮主路径。

---

# 4. 代码修改

主要文件：

```text
models/mctformer_plus.py
```

如命令行需要暴露开关，可同步修改当前训练参数文件，但只加入本任务必要参数。

建议新增：

```python
last_mct=False
last_topk=1
last_sigma=None
last_eps=1e-6
```

第一轮只支持：

```text
MCTformer+ Small
BCSS E0
PSL baseline
CTI-BGT disabled
vanilla attention
```

如果其他实验变体与 Last-MCT 发生接口冲突，直接显式报错，不要为了兼容所有旧实验扩大实现范围。

---

# 5. PatchFinalLN 保持原实现

在 12 个 Transformer blocks 后：

```python
# CCT still records raw post-block class tokens in the loop.

x_cls = x[:, foreground_slice]
x_patch_raw = x[:, patch_slice]
x_patch = self.norm(x_patch_raw)
```

要求：

- `x_cls` 不经过 final norm；
- `x_patch` 经过现有 `self.norm`；
- attention matrices保持原样；
- `all_x_cls`保持 raw post-block tokens。

---

# 6. 新增 LaST kernel

建议实现：

```python
def _get_last_kernel(self, patch_tokens):
    D = patch_tokens.shape[-1]

    if self._last_kernel.numel() != D:
        sigma = (
            self.last_sigma
            if self.last_sigma is not None
            else D ** 0.5
        )

        positions = torch.arange(
            -D // 2 + 1,
            D // 2 + 1,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

        kernel = torch.exp(
            -0.5 * (positions / sigma) ** 2
        )
        kernel = kernel / kernel.max()

        self._last_kernel = kernel

    return self._last_kernel.view(1, 1, D).to(
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    )
```

buffer：

```python
self.register_buffer(
    "_last_kernel",
    torch.empty(0),
    persistent=False,
)
```

不得硬编码 `768` 或 `384`。

---

# 7. 新增 low-pass

```python
def last_low_pass(self, patch_tokens):
    original_dtype = patch_tokens.dtype

    if patch_tokens.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        x = patch_tokens.float()
    else:
        x = patch_tokens

    kernel = self._get_last_kernel(x)

    x = torch.fft.fft(x, dim=-1)
    x = torch.fft.fftshift(x, dim=-1)
    x = x * kernel
    x = torch.fft.ifftshift(x, dim=-1)
    x = torch.fft.ifft(x, dim=-1).real

    return x.to(original_dtype)
```

FFT 必须沿：

```python
dim=-1
```

即 embedding/channel dimension。

---

# 8. 新增 stability score

```python
def last_stability_score(
    self,
    patch_tokens,
    low_pass_tokens,
):
    denominator = (
        low_pass_tokens - patch_tokens
    ).abs().clamp_min(self.last_eps)

    return patch_tokens / denominator
```

要求：

```text
input/output shape = [B, N, D]
finite values only
```

---

# 9. 新增 channel-wise Top-K aggregation

```python
def last_aggregate(self, patch_tokens):
    low_pass = self.last_low_pass(
        patch_tokens
    )

    stability = self.last_stability_score(
        patch_tokens,
        low_pass,
    )

    k = min(
        self.last_topk,
        patch_tokens.shape[1],
    )

    _, indices = torch.topk(
        stability,
        k=k,
        dim=1,
        largest=True,
    )

    selected = torch.gather(
        patch_tokens,
        dim=1,
        index=indices,
    )

    pooled = selected.mean(dim=1)

    return pooled, indices, stability
```

必须确认：

```text
patch_tokens: [B, N, D]
indices:      [B, K, D]
selected:     [B, K, D]
pooled:       [B, D]
```

注意：**每个 feature channel 独立选择 patch index。**

不要错误实现成：

```text
选 K 个完整 patch vectors
```

---

# 10. Patch classifier 改为 1×1 Conv

Last-MCT 模式下：

```python
self.head = nn.Conv2d(
    self.embed_dim,
    self.num_classes,
    kernel_size=1,
    stride=1,
    padding=0,
)
```

第一轮在 `CTI-BGT=False` 下工作。

初始化继续使用当前 `_init_weights`。

不要保留 Last-MCT 的 3×3 patch classifier。

---

# 11. 用同一 classifier 做 pooled classification

```python
def last_classify(self, pooled):
    weight = self.head.weight[:, :, 0, 0]

    return F.linear(
        pooled,
        weight,
        self.head.bias,
    )
```

因此 pooled-token classification 和 spatial patch CAM 共享同一组 class weights。

---

# 12. 修改 `forward()`

Last-MCT 模式核心：

```python
x_cls, x_patch, attentions, all_x_cls, auxiliary =     self.forward_features(
        x,
        active_labels=active_labels,
        return_aux=True,
    )

x_cls_logits = x_cls.mean(dim=-1)

last_token, _, _ = self.last_aggregate(
    x_patch
)

x_patch_logits = self.last_classify(
    last_token
)

output = [
    x_cls_logits,
    torch.stack(all_x_cls),
    x_patch_logits,
]

return output
```

训练 loop 的输出接口必须保持与当前 MCTformer+ 一致。

不要修改：

```text
class-token loss
CCT loss
patch loss type
loss weights
```

Patch loss仍为：

```python
F.multilabel_soft_margin_loss(
    x_patch_logits,
    targets,
)
```

---

# 13. CAM path

Last-MCT 模式下先把 FinalLN patch tokens恢复为空间 feature map：

```python
B, N, D = x_patch.shape

hp = H // patch_size
wp = W // patch_size

feature_map = x_patch.reshape(
    B, hp, wp, D
).permute(
    0, 3, 1, 2
).contiguous()
```

再：

```python
patch_cam = self.head(feature_map)
```

之后保持当前 MCTformer+ CAM pipeline 不变：

```text
ReLU patch CAM
× native last-3 class-to-patch attention
sqrt
all-layer patch-to-patch propagation
```

不要修改：

```text
A_c2p aggregation
A_p2p aggregation
sqrt
CAM post-processing
```

---

# 14. `forward_with_label()` / CAM classification gating

当前 patch label 若来自 GWRP，Last-MCT 模式必须替换为 LaST patch logits：

```python
last_token, _, _ = self.last_aggregate(
    x_patch
)

patch_logits = self.last_classify(
    last_token
)

patch_label = (
    patch_logits > 0
).to(x.dtype)
```

class-token prediction仍保持：

```python
cls_logits = x_cls.mean(dim=-1)
```

CAM 本体继续来自同一组 classifier weights：

```python
patch_cam = self.head(feature_map)
```

---

# 15. 不允许改变的部分

本轮禁止改动：

```text
Transformer blocks
class-token count
class-token readout
CCT implementation
attention normalization
last-three A_c2p
all-layer A_p2p
sqrt CAM refinement
training epochs
optimizer
learning rate
batch/effective batch
augmentation
VOC train/val split
seed
loss weights
```

不得加入：

```text
semantic ownership
BG token
background loss
class competition
new attention loss
new regularizer
new LaST hyperparameter sweep
```

第一轮只跑：

```text
K=1
sigma=sqrt(D)
eps=1e-6
repo-safe formula
```

---

# 16. 必要测试

只写实现所需的测试：

- Aggregator shape；
- 与直接 LaST reference implementation 的数值一致性；
- Top-K 确认沿 patch dimension，且每个 channel 可以选不同 patch；
- pooled classifier 与同一 1×1 Conv 权重的 `F.linear` 一致；
- 单个 spatial patch 的 1×1 Conv score 与同一 linear score 一致；
- patch loss backward 后 `self.head.weight.grad`、`self.norm.weight.grad` 非空，selected patch values 获得梯度；
- AMP forward/backward 无 NaN/Inf。

---

# 17. Smoke training

测试通过后先执行短 smoke run。

确认：

```text
forward/backward 正常
loss finite
patch loss能够下降
class-token branch正常
checkpoint可保存/加载
CAM可生成
```

Smoke run只用于发现实现错误，不用于调参。

---

# 18. 完整训练

Smoke通过后直接启动完整 VOC 训练，无需再次确认。

训练配置必须复用最近的 **PatchFinalLN matched seed-0** 实验的完整配置与命令。

唯一方法差异：

```text
PatchFinalLN:
FinalLN → 3×3 Conv → GWRP

Last-MCT:
FinalLN → LaST K=1 → 1×1/Linear classifier
```

如果现有结果目录中已有 PatchFinalLN 的 exact command / config / metadata，优先直接复用。

不要重新调：

```text
LR
epoch
sigma
K
threshold
```

---

# 19. 训练结束后的验证

只整理：

```text
class-token macro mAP
Last-MCT patch-head macro mAP
class-token validation loss
patch validation loss
fixed/default threshold raw CAM mIoU
best raw CAM mIoU from the same threshold sweep
best threshold
foreground precision
foreground recall
```

最终比较表：

| Model | Class-token mAP | Patch-head mAP | Fixed CAM mIoU | Best CAM mIoU | Best threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original MCTformer+ | existing | existing | existing | existing | existing | existing | existing |
| PatchFinalLN MCTformer+ | existing | existing | existing | existing | existing | existing | existing |
| **Last-MCT** | new | new | new | new | new | new | new |

不要重新训练已有两个 baseline。

---

# 20. 输出与 Git

建议输出：

```text
results/last_mct/<run_id>/
├── config.json
├── command.txt
├── training_log.txt
├── checkpoint_final.pth
├── classification_results.csv
├── cam_results.csv
├── comparison_summary.csv
├── tests.txt
├── git_metadata.json
└── LAST_MCT_REPORT.md
```

Git 只提交：

```text
implementation
tests
compact CSV
compact report
metadata / exact command
```

不要提交大 checkpoint 或大日志。

---

# 21. Codex 最终执行 Prompt

```text
Work directly in ~/code/TGCA on the current main checkout.

You are explicitly authorized to:
- inspect and modify repository code;
- add the minimal tests required for this task;
- run unit tests and smoke training;
- use the available GPU(s);
- launch the full matched VOC training without asking for additional confirmation;
- create git commit(s) for your implementation and compact result/report files.

Do NOT push to the remote repository unless I explicitly ask later.
Do NOT use git reset --hard, force-push, rebase away unrelated work, or delete
unrelated user changes. Preserve any unrelated dirty-worktree changes.

Implement one method only: Last-MCT, built on the already validated patch-only
FinalLN MCTformer+.

Required architecture:

1. Keep the MCTformer+ class-token branch unchanged:
   raw L12 class tokens -> mean(dim=-1) -> original class-token multilabel loss.

2. Keep CCT unchanged:
   it must continue to use raw post-block class tokens from all 12 blocks.

3. Keep the current patch-only final LayerNorm:
   raw L12 patch tokens -> self.norm -> normalized patch tokens.

4. Replace the PatchFinalLN baseline patch classifier/aggregation
   (3x3 Conv -> GWRP) with the LaST-ViT repository logic:
   normalized patch tokens
   -> FFT along embedding dimension
   -> Gaussian low-pass filter
   -> repo stability score
   -> channel-wise Top-K over the patch dimension
   -> mean selected values across K
   -> one pooled D-dimensional token
   -> shared 20-class classifier.

5. First run is fixed:
   K = 1
   sigma = sqrt(embed_dim)
   eps = 1e-6
   score formula = repo-safe:
       P / abs(P_lowpass - P).clamp_min(eps)

6. Top-K semantics must exactly match LaST:
   input [B,N,D];
   torch.topk(..., dim=1);
   indices [B,K,D];
   each embedding channel independently chooses its own patch index.

7. Replace the Last-MCT patch classifier with Conv2d(D,20,kernel_size=1),
   mathematically equivalent to Linear(D,20).
   Use the SAME classifier weights for:
   a) pooled Last-MCT classification via F.linear;
   b) spatial patch CAM generation via the 1x1 Conv.

8. Keep the original MCTformer+ CAM refinement unchanged after patch CAM:
   ReLU
   -> native last-three class-to-patch attention
   -> multiplication
   -> sqrt
   -> original all-layer patch-to-patch propagation.

9. Update forward_with_label / CAM gating so patch classification labels
   come from Last-MCT pooled-token logits, not GWRP.

10. Do not add semantic competition, BG tokens, background losses, BCSS/PSL/CTI
    methods, attention interventions, new regularizers, or hyperparameter sweeps.

11. First-round support only needs ordinary MCTformer+ Small:
    vanilla attention, BCSS E0, PSL baseline, CTI-BGT disabled.
    Explicitly reject incompatible optional variants rather than expanding scope.

12. Add only necessary tests:
    - aggregator shapes;
    - equivalence against a direct LaST reference implementation;
    - correct Top-K dimension and per-channel selection;
    - pooled classifier == F.linear with 1x1 Conv weights;
    - spatial 1x1 score == same linear classifier score;
    - gradient flow through norm, selected patch values, and classifier;
    - finite AMP forward/backward.

13. After tests pass, commit the implementation.
    Suggested commit message:
        Add LaST-style patch aggregation to MCTformer+

14. Run a short smoke training. If healthy, immediately launch the full VOC
    training using the exact same settings, seed, initialization, effective
    batch, optimizer, LR, epochs, augmentations, dataset split, and evaluation
    protocol as the most recent matched PatchFinalLN seed-0 run.
    Do not ask me again before starting full training.

15. Do not tune K, sigma, LR, epochs, or any other hyperparameter.

16. After training, evaluate and report only:
    - class-token macro mAP;
    - Last-MCT patch-head macro mAP;
    - class-token and patch validation losses;
    - fixed-threshold raw CAM mIoU;
    - best raw CAM mIoU on the same threshold sweep;
    - best threshold;
    - foreground precision and recall.

17. Build one compact table containing:
    Original MCTformer+ (existing result),
    PatchFinalLN MCTformer+ (existing matched result),
    Last-MCT (new result).
    Do not retrain the first two if matched results already exist.

18. Save exact commands, config, git SHA, test status, checkpoint SHA256,
    compact CSV results, and LAST_MCT_REPORT.md.

19. After results are organized, you are authorized to make a second commit
    containing only compact result tables/report/metadata if appropriate.
    Do not commit the large checkpoint or raw large logs.

Stop after the matched Last-MCT training, evaluation, and compact result report.
Do not propose or implement any next research direction.
```
