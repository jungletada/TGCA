# Class-Stable LaST Pooling for MCTformer+：Codex 执行计划

> **目标仓库：** `https://github.com/jungletada/TGCA`  
> **服务器仓库：** `~/code/TGCA`  
> **直接基线：** 已验证的 **PatchFinalLN MCTformer+**  
> **本轮目标：** 只替换 patch branch 的 pooling rule：  
> `GWRP → class-response frequency-stability Top-K pooling`  
> **保持 spatial classifier 在 pooling 之前。**

---

# 1. 方法定义

## 1.1 PatchFinalLN baseline

```text
raw L12 patch tokens
→ Final LayerNorm
→ 3×3 Conv classifier
→ class-specific spatial map M
→ GWRP
→ patch logits
→ patch multilabel loss
```

其中：

\[
P_n = LN(P^{12})
\]

\[
M = H_{3\times3}(P_n)
\]

\[
z^{patch} = GWRP(M)
\]

---

## 1.2 本轮方法：Class-Stable LaST Pooling

保持：

```text
FinalLN
3×3 Conv classifier
class-token branch
CCT
attention
CAM refinement
```

全部不变。

新增一条 low-pass response 分支：

\[
\tilde P_n = LP(P_n)
\]

使用**同一个** `3×3 Conv classifier`：

\[
\tilde M = H_{3\times3}(\tilde P_n)
\]

然后定义 class-response stability：

\[
R_{c,j}
=
\frac{
M_{c,j}
}{
|\tilde M_{c,j}-M_{c,j}|+\epsilon
}
\]

对每个 semantic class 独立沿空间 patch 维选 Top-K：

\[
I_c
=
TopK_j(R_{c,j})
\]

最终 image-level patch logit：

\[
z_c^{patch}
=
\frac1K
\sum_{j\in I_c}
M_{c,j}
\]

只对：

\[
z^{patch}
\]

计算原来的 `MultiLabelSoftMarginLoss`。

---

# 2. 不增加 \tilde{M} 的额外 loss

本轮明确禁止：

```text
BCE(tilde_M)
consistency loss between M and tilde_M
KL loss
MSE loss
contrastive loss
auxiliary low-pass classification loss
```

\(\tilde M\) 只用于生成 stability score：

\[
R_{c,j}
\]

并决定 Top-K index。

最终被 pooling 的数值必须来自原始：

\[
M
\]

而不是：

\[
\tilde M.
\]

这与 LaST 官方逻辑一致：low-pass 分支用于 selection，最终选中的值来自原始 feature/response。

---

# 3. 频率处理位置

FFT 仍然必须发生在 patch embedding channel 维：

```python
dim=-1
```

即：

\[
P_n \in \mathbb R^{B\times N\times D}
\]

沿：

\[
D
\]

做 FFT。

禁止直接对：

```text
num_class = 20
```

个 semantic class channels 做 FFT。

---

# 4. FinalLN

沿用已经验证过的 PatchFinalLN：

```python
x_cls = x[:, foreground_slice]

x_patch_raw = x[:, patch_slice]

x_patch = self.norm(x_patch_raw)
```

要求：

- class tokens 不经过 final norm；
- patch tokens 经过 `self.norm`；
- CCT 继续使用 raw post-block class tokens。

---

# 5. 3×3 classifier 保持不变

当前：

```python
self.head = nn.Conv2d(
    self.embed_dim,
    self.num_classes,
    kernel_size=3,
    stride=1,
    padding=1,
)
```

本轮必须继续使用原有 `3×3 Conv`。

不要改成：

```text
1×1 Conv
Linear
MLP head
```

这一点是本实验的核心约束。

---

# 6. Low-pass 实现

新增或复用 LaST-style Gaussian low-pass：

```python
def _get_last_kernel(self, patch_tokens):
    D = patch_tokens.shape[-1]

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

    return kernel.view(1, 1, D)
```

```python
def last_low_pass(self, patch_tokens):
    original_dtype = patch_tokens.dtype

    x = (
        patch_tokens.float()
        if patch_tokens.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        else patch_tokens
    )

    kernel = self._get_last_kernel(x)

    x = torch.fft.fft(x, dim=-1)
    x = torch.fft.fftshift(x, dim=-1)
    x = x * kernel
    x = torch.fft.ifftshift(x, dim=-1)
    x = torch.fft.ifft(x, dim=-1).real

    return x.to(original_dtype)
```

第一轮固定：

```text
sigma = sqrt(embed_dim)
eps = 1e-6
```

不要 sweep。

---

# 7. Spatial classifier maps

对 normalized patch feature：

```python
P_n
```

和 low-pass patch feature：

```python
P_lp
```

使用同一个 `self.head`：

```python
M = self.head(
    reshape_to_2d(P_n)
)

M_lp = self.head(
    reshape_to_2d(P_lp)
)
```

要求：

```text
M.shape == M_lp.shape == [B, C, H, W]
```

其中：

```text
C = num_classes
```

---

# 8. Class-response stability

建议实现：

```python
def class_stability_score(
    self,
    class_map,
    class_map_lowpass,
):
    denominator = (
        class_map_lowpass - class_map
    ).abs().clamp_min(self.last_eps)

    return class_map / denominator
```

shape：

```text
[B, C, H, W]
```

然后 flatten spatial dimension：

```python
stability = stability.flatten(2)
class_map_flat = class_map.flatten(2)
```

得到：

```text
[B, C, N]
```

---

# 9. Class-wise Top-K pooling

对每个 class channel 独立做：

```python
_, indices = torch.topk(
    stability,
    k=K,
    dim=-1,
    largest=True,
)
```

其中：

```text
indices.shape = [B, C, K]
```

然后必须从原始 class map：

```python
class_map_flat
```

中 gather：

```python
selected = torch.gather(
    class_map_flat,
    dim=-1,
    index=indices,
)
```

最后：

```python
patch_logits = selected.mean(dim=-1)
```

shape：

```text
[B, C]
```

第一轮固定：

```text
K = 1
```

---

# 10. Patch loss

保持原来的：

```python
F.multilabel_soft_margin_loss(
    patch_logits,
    targets,
)
```

总训练 loss 仍然是：

```text
class-token multilabel loss
+ CCT
+ patch multilabel loss
```

不增加任何新 loss 项。

---

# 11. CAM path

CAM 必须继续直接使用：

\[
M = H_{3\times3}(P_n)
\]

即原始、未 low-pass 的 class-specific spatial map。

CAM path：

```text
M
→ ReLU
→ native last-three A_c2p
→ multiply
→ sqrt
→ all-layer A_p2p propagation
→ final CAM
```

禁止使用：

```text
M_lp
stability map
Top-K mask
```

直接替换 raw CAM。

它们只参与 image-level patch pooling。

---

# 12. forward() 目标结构

核心逻辑：

```python
x_cls, x_patch, attentions, all_x_cls, auxiliary = \
    self.forward_features(
        x,
        active_labels=active_labels,
        return_aux=True,
    )

# class-token branch unchanged
x_cls_logits = x_cls.mean(dim=-1)

# normalized patch features
feature_map = reshape_to_2d(x_patch)

# original spatial class map
class_map = self.head(feature_map)

# low-pass patch features
x_patch_lp = self.last_low_pass(x_patch)

feature_map_lp = reshape_to_2d(x_patch_lp)

# same classifier
class_map_lp = self.head(feature_map_lp)

# stability-based pooling
stability = self.class_stability_score(
    class_map,
    class_map_lp,
)

patch_logits = self.class_stable_topk_pool(
    class_map,
    stability,
)

return [
    x_cls_logits,
    torch.stack(all_x_cls),
    patch_logits,
]
```

---

# 13. forward_with_label() / CAM gating

Patch label不再使用 GWRP。

必须使用新的 stability pooled logits：

```python
patch_logits = self.class_stable_topk_pool(
    class_map,
    stability,
)

patch_label = (
    patch_logits > 0
).to(x.dtype)
```

class-token label仍然：

```python
cls_logits = x_cls.mean(dim=-1)
```

CAM 本体：

```python
outputs = self.get_cam(
    class_map,
    attn_weights,
    auxiliary,
)
```

其中 `class_map` 必须是：

```text
H(P_n)
```

不是：

```text
H(P_lp)
```

---

# 14. Variant 开关

建议新增：

```python
class_stable_last=False
last_topk=1
last_sigma=None
last_eps=1e-6
```

支持：

```text
class_stable_last=False
```

时保持现有 PatchFinalLN + GWRP baseline 数值行为。

支持：

```text
class_stable_last=True
```

时替换：

```text
GWRP
```

为：

```text
Class-Stable LaST Top-K Pooling
```

---

# 15. 第一轮兼容范围

只支持：

```text
MCTformer+ Small
PatchFinalLN enabled
vanilla attention
BCSS E0
PSL baseline
CTI-BGT disabled
```

其他变体直接显式拒绝。

不要扩大范围。

---

# 16. 必要测试

只添加以下测试。

## Test A：baseline equivalence

```text
class_stable_last=False
```

时：

```text
class logits
patch logits
CAM
attention
```

与现有 PatchFinalLN baseline数值一致。

---

## Test B：low-pass shape / finite

```text
input  [B,N,D]
output [B,N,D]
```

无 NaN/Inf。

---

## Test C：shared classifier

确认：

```text
M = H(P_n)
M_lp = H(P_lp)
```

两者使用同一 `self.head` 参数对象。

---

## Test D：stability shape

```text
M        [B,C,H,W]
M_lp     [B,C,H,W]
R        [B,C,H,W]
```

全部一致。

---

## Test E：Top-K semantics

确认：

```text
indices [B,C,K]
```

且每个 class 独立沿：

```text
spatial dimension
```

选择 patch。

---

## Test F：gather source

必须确认最终：

```python
selected = gather(M, indices)
```

而不是：

```python
gather(M_lp, indices)
```

---

## Test G：No auxiliary loss

确认 training loss 中没有：

```text
loss(M_lp)
consistency(M, M_lp)
```

---

## Test H：gradient flow

一个 mini-batch backward 后确认：

```text
self.head receives gradient
self.norm receives gradient
selected M locations receive gradient
```

不要求 Top-K indices 可导。

---

# 17. Smoke training

测试通过后运行短 smoke：

```text
forward/backward正常
loss finite
class loss正常
patch loss正常
checkpoint可保存/load
CAM可生成
```

不做调参。

---

# 18. 完整训练

Smoke通过后直接运行完整 VOC matched training。

复用最新 PatchFinalLN matched seed-0 的：

```text
same DeiT-S initialization
same seed
same LR
same optimizer
same epochs
same effective batch
same augmentation
same VOC split
same input size
same attention setup
same loss weights
```

唯一方法差异：

```text
PatchFinalLN:
3×3 Conv → GWRP

Class-Stable LaST:
3×3 Conv
+ low-pass shared classifier response
→ class-wise stability Top-K
```

第一轮固定：

```text
K=1
sigma=sqrt(D)
eps=1e-6
```

禁止 sweep。

---

# 19. 结果整理

训练结束后只整理：

```text
class-token macro mAP
patch-head macro mAP
class-token validation loss
patch validation loss
fixed/default threshold raw CAM mIoU
best raw CAM mIoU from same threshold sweep
best threshold
foreground precision
foreground recall
```

比较表：

| Model | Class-token mAP | Patch mAP | Fixed CAM mIoU | Best CAM mIoU | Best threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original MCTformer+ | existing | existing | existing | existing | existing | existing | existing |
| PatchFinalLN | existing | existing | existing | existing | existing | existing | existing |
| LaST-faithful pool-before-classifier | existing | existing | existing | existing | existing | existing | existing |
| **Class-Stable LaST Pooling** | new | new | new | new | new | new | new |

已有实验不要重跑。

---

# 20. Git 与权限

Codex 获得以下权限：

- 可直接修改 `~/code/TGCA`；
- 可新增必要 tests；
- 可运行 unit tests；
- 可运行 smoke training；
- 可使用服务器 GPU；
- smoke健康后可直接启动完整 VOC training；
- 测试通过后可直接 `git commit`；
- 训练结束后可提交 compact result/report；
- **不需要再次询问用户确认这些操作。**

禁止：

```text
git push
git reset --hard
force push
rebase away unrelated work
delete unrelated dirty-worktree changes
commit large checkpoints
commit large raw logs
```

建议 implementation commit：

```text
Add class-stable LaST pooling for MCTformer+
```

结果 commit：

```text
Record class-stable LaST VOC results
```

---

# 21. Codex 执行 Prompt

```text
Work directly in ~/code/TGCA on the current main checkout.

You are explicitly authorized to:
- inspect and modify the repository;
- add the minimal tests needed for this task;
- run tests and smoke training;
- use the available GPU(s);
- launch the full matched VOC training without asking me again;
- git commit the implementation after tests pass;
- git commit compact result tables/report after training if appropriate.

Do not git push unless I explicitly request it later.
Do not use git reset --hard, force-push, destructive rebase, or delete unrelated
dirty-worktree changes.

Implement exactly one new patch-pooling variant built on the validated
PatchFinalLN MCTformer+.

The design is:

1. Keep the class-token branch unchanged:
   raw L12 class tokens -> mean(dim=-1) -> original multilabel class loss.

2. Keep CCT unchanged and using raw post-block class tokens.

3. Keep patch-only Final LayerNorm:
   raw L12 patch tokens -> self.norm -> P_n.

4. Keep the original MCTformer+ 3x3 Conv classifier BEFORE pooling.

5. Compute the original class-specific spatial map:
       M = H_3x3(P_n)

6. Apply LaST-style Gaussian low-pass to P_n along the embedding dimension only:
       P_lp = LP(P_n)
   with:
       sigma = sqrt(embed_dim)
       eps = 1e-6
   Do not FFT over the class dimension.

7. Pass P_lp through the SAME 3x3 Conv classifier:
       M_lp = H_3x3(P_lp)

8. Define class-response stability:
       R = M / abs(M_lp - M).clamp_min(eps)

9. Flatten only the spatial dimensions and, for each semantic class independently,
   select Top-K spatial locations:
       indices = topk(R, k=K, dim=spatial_dim)
   First run:
       K = 1

10. Gather the selected values from the ORIGINAL M, not M_lp:
       selected = gather(M, indices)
       patch_logits = mean(selected over K)

11. Patch multilabel loss remains exactly the existing
    F.multilabel_soft_margin_loss(patch_logits, targets).

12. Do not add any loss on M_lp.
    Do not add consistency, KL, MSE, contrastive, or auxiliary low-pass losses.
    M_lp is only a selector signal.

13. CAM generation must remain based on the original M:
       M -> ReLU
         -> native last-three A_c2p refinement
         -> sqrt
         -> native all-layer A_p2p propagation
    Do not replace raw CAM with M_lp, R, or a Top-K mask.

14. Update forward_with_label so patch classification gating uses the new
    class-stable pooled logits instead of GWRP.

15. Add a clean variant flag, e.g. class_stable_last=False.
    When false, PatchFinalLN + original GWRP behavior must remain numerically
    equivalent to the current baseline.

16. First-round compatibility only needs:
    MCTformer+ Small,
    PatchFinalLN,
    vanilla attention,
    BCSS E0,
    PSL baseline,
    CTI-BGT disabled.
    Explicitly reject incompatible optional variants.

17. Add only necessary tests:
    - baseline equivalence when the variant is disabled;
    - low-pass shape and finite values;
    - M and M_lp use the same classifier;
    - class-response stability shape;
    - Top-K is class-wise over spatial locations;
    - selected values are gathered from original M;
    - no auxiliary loss is added for M_lp;
    - gradient flow through norm, classifier, and selected M values.

18. After tests pass, commit the implementation:
       Add class-stable LaST pooling for MCTformer+

19. Run a short smoke training. If healthy, immediately launch the full matched
    VOC training using exactly the same initialization, seed, optimizer, LR,
    epochs, effective batch, augmentations, split, input size, loss weights, and
    evaluation protocol as the latest matched PatchFinalLN seed-0 experiment.
    Do not ask me again before starting.

20. Do not sweep K, sigma, LR, threshold, or any other hyperparameter.

21. After training, evaluate and report only:
    - class-token macro mAP;
    - patch-head macro mAP;
    - class-token validation loss;
    - patch validation loss;
    - fixed/default threshold raw CAM mIoU;
    - best raw CAM mIoU from the same threshold sweep;
    - best threshold;
    - foreground precision;
    - foreground recall.

22. Build one compact comparison table containing:
    Original MCTformer+,
    PatchFinalLN,
    previous LaST-faithful pool-before-classifier run,
    new Class-Stable LaST Pooling run.
    Reuse existing results for the first three; do not retrain them.

23. Save exact command, config, git SHA, tests, checkpoint SHA256, compact CSVs,
    and a concise CLASS_STABLE_LAST_REPORT.md.

24. You are authorized to make a second commit containing only compact result
    tables/report/metadata. Do not commit large checkpoints or raw large logs.

Stop after the matched training, evaluation, and compact report.
Do not propose or implement any next research direction.
```
