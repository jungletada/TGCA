# Token-Group Calibrated Attention (TGCA): Design and Validation Plan

**Status:** implementation-ready research design; no result in this document is claimed as observed

**Target:** ICASSP 2027 Computer Vision

**Last updated:** 2026-08-10
**Working title:** *Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation*

## 1. Purpose

This document is the implementation and validation source of truth for Token-Group Calibrated Attention (TGCA). It translates the research hypothesis into:

- a precise mathematical operator;
- a common software interface for every normalization baseline;
- integration plans for MCTformer+ and Know Your Attention Maps (KYAM);
- deterministic unit and mechanism tests;
- a staged VOC-first experiment program;
- machine-readable logging and reproducibility requirements;
- explicit falsification criteria and stopping gates.

TGCA code, tests, experiment configurations, and generated metrics must remain inside this `TGCA/` repository. The rejected MCTTA manuscript and its legacy files are records only and must not be overwritten.

## 2. Decisions fixed before implementation

### 2.1 Method and host roles

| Method | Role in the TGCA project | TGCA implemented? |
|---|---|---:|
| MCTformer+ | Primary host and cleanest mechanism test | Yes |
| Know Your Attention Maps, ICCV 2025 | Required independent host | Yes |
| DiCLIP, IEEE T-IP 2026 | Recent external comparison baseline | No |
| MoRe | Optional supplementary host | Optional |
| CTI | Optional supplementary host | Optional |
| Hierarchical MCTTA | Optional diagnostic host for self- and cross-attention | Optional |

Primary external sources:

- [MCTformer+ paper](https://arxiv.org/abs/2308.03005) and [official repository](https://github.com/xulianuwa/MCTformer)
- [Know Your Attention Maps paper](https://openaccess.thecvf.com/content/ICCV2025/html/Hanna_Know_Your_Attention_Maps_Class-specific_Token_Masking_for_Weakly_Supervised_ICCV_2025_paper.html) and [official repository](https://github.com/HSG-AIML/TokenMasking-WSSS)
- [DiCLIP paper](https://doi.org/10.1109/TIP.2026.3692055) and [official repository](https://github.com/zwyang6/DiCLIP)

### 2.2 Required normalization variants

Every core comparison must use the same attention implementation, Q/K/V projections, value path, output projection, dropout placement, training schedule, and CAM pipeline. Only the normalization mode may change.

1. Vanilla joint softmax.
2. Original split weighted softmax `(1, 1)`.
3. Normalized split weighted softmax `(0.5, 0.5)`.
4. Count-only TGCA (`gamma=1`, no relation bias).
5. TGCA with zero-initialized relation bias.

The partial correction `gamma=0.5` is a contingency ablation, not a default method and not a hyperparameter to tune extensively.

### 2.3 Dataset and execution order

1. Unit and synthetic tests.
2. PASCAL VOC 2012 baseline reproduction.
3. VOC mechanism and one-seed pilot experiments.
4. VOC three-seed core ablation.
5. KYAM independent-host validation.
6. MS COCO only after all mechanism and generality gates pass.

### 2.4 Claims that are out of scope

TGCA is not an adapter, graph module, CAM post-processor, boundary method, prototype method, or foundation-model contribution. Do not claim:

- global state of the art across unmatched supervision or pretraining;
- that MCTTA is a universal adapter;
- that graph blocks are inherently superior;
- that rapid classification-loss convergence proves localization quality;
- that the old MCTTA pipeline is single-stage;
- exact resolution invariance from finite empirical tests.

The strongest permissible empirical phrasing is “less resolution-sensitive” or “more scale-robust,” and only after the planned measurements support it.

## 3. Research questions and falsifiable hypotheses

### RQ1 — Does joint softmax mix semantic evidence with token-group cardinality?

**H1:** In trained class-token attention, the aggregate class/patch group mass under vanilla softmax changes measurably when patch count changes, even when the semantic content is held as constant as practical.

**Evidence required:**

- exact synthetic token-replication behavior;
- group mass by layer and head at input resolutions `224`, `320`, `448`, and `512`;
- a paired image-level association between patch count and group mass;
- corresponding CAM or cross-scale consistency changes.

**Hypothesis failure:** the trained model's semantic logit changes fully cancel the count effect, or observed scale changes are unrelated to group mass and are explained by positional interpolation, receptive field, or other scale effects.

### RQ2 — Does TGCA remove the intended count effect without changing the attention contract?

**H2:** TGCA preserves unit row sums and is invariant to exact replication of all key/value pairs within one group.

**Evidence required:** deterministic unit tests in full and mixed precision, including forward output and backward gradients.

**Hypothesis failure:** replication changes aggregate group mass or the attention output beyond numerical tolerance, or masks/dtypes produce unstable rows.

### RQ3 — Does calibration improve useful WSSS behavior rather than only output magnitude?

**H3:** TGCA improves or stabilizes raw CAM quality and cross-scale consistency beyond both unnormalized split `(1,1)` and normalized fixed split `(0.5,0.5)`.

**Evidence required:** matched Q/K/V and value paths, fixed thresholds, raw CAM metrics, group-mass diagnostics, and classification metrics.

**Hypothesis failure:** the gain is reproduced by row-mass doubling, fixed 50/50 group allocation, per-method threshold selection, or changed CAM post-processing.

### RQ4 — Is the effect independent of the original MCTTA engineering stack?

**H4:** TGCA produces a positive mechanism or CAM effect in both MCTformer+ and KYAM.

**Hypothesis failure:** the effect occurs only in MCTTA, only with its cross-attention, or only after unrelated host modifications.

## 4. Notation and token layouts

For attention head `h`:

\[
s_{ij}^{h}=\frac{(q_i^h)^\top k_j^h}{\sqrt{d_h}},
\]

where `i` indexes queries, `j` indexes keys, and `g_q(i)` and `g_k(j)` identify query and key groups.

Group IDs are semantic roles, not token positions. Implementations may use integer IDs internally, but logs and configurations must use stable names.

### 4.1 MCTformer+ self-attention

Token order is:

```text
[class_1, ..., class_C, patch_1, ..., patch_P]
```

Groups are:

```text
0: class
1: patch
```

Both query and key layouts use these groups. VOC normally has `C=20`; COCO normally has `C=80`. With patch size 16:

| Input | Patch grid | `P` |
|---:|---:|---:|
| 224 | 14 × 14 | 196 |
| 320 | 20 × 20 | 400 |
| 448 | 28 × 28 | 784 |
| 512 | 32 × 32 | 1024 |

### 4.2 MCTTA cross-attention

MCTTA is not a required publication host, but the reusable operator must support it.

Typical query order:

```text
[class tokens, spatial-query tokens]
```

Typical key/value order:

```text
[class tokens, patch tokens]
```

The count correction always uses **key-group cardinality**. Query groups are needed only for relation-bias lookup and diagnostics.

### 4.3 KYAM self-attention

The audited official implementation uses:

```text
[class_1, ..., class_C, patch_1, ..., patch_P, register]
```

The primary, predeclared grouping is:

```text
0: global = class tokens + singleton register token
1: patch
```

Rationale:

- it keeps the primary method as a two-group normalization;
- class and register tokens are both global/non-spatial tokens;
- making the singleton register a third TGCA group would grant it a full evidence-driven group prior and could over-weight one token;
- merging the register into the patch group would mix a global token with spatial evidence.

This choice must not remain untested. Run one predeclared register-sensitivity comparison on the best count-only configuration:

1. `global`: register grouped with class tokens — primary;
2. `patch`: register grouped with patch tokens;
3. `singleton`: register is a third group, without relation bias.

Do not remove the register token, because that changes the host architecture. If the TGCA conclusion depends strongly on register grouping, report this as a limitation and weaken the generality claim.

## 5. Mathematical design

### 5.1 Vanilla group mass

Vanilla attention is:

\[
A_{ij}^{h}=\frac{\exp(s_{ij}^{h})}{\sum_k\exp(s_{ik}^{h})}.
\]

The aggregate mass assigned by query `i` to key group `g` is:

\[
m_{i,g}^{h}=\sum_{j:g_k(j)=g}A_{ij}^{h}.
\]

Define group sum evidence:

\[
Z_{i,g}^{h}=\sum_{j:g_k(j)=g}\exp(s_{ij}^{h}).
\]

Then:

\[
m_{i,g}^{h}=\frac{Z_{i,g}^{h}}{\sum_r Z_{i,r}^{h}}.
\]

Under equal logits, vanilla group mass is exactly proportional to cardinality:

\[
m_{i,g}^{h}=\frac{N_g}{\sum_r N_r}.
\]

For VOC with 20 class tokens, the equal-logit vanilla class-group mass decreases from approximately `0.0926` at 224 input to `0.0192` at 512 input.

This equal-logit example establishes a mechanism, not an empirical claim about trained networks. Trained logits may compensate for, amplify, or ignore the effect; that is why direct measurement is mandatory.

### 5.2 TGCA formulation

TGCA corrects each key logit by its valid key-group count:

\[
\widetilde{s}_{ij}^{h}
=s_{ij}^{h}
-\gamma\log N_{g_k(j)}
+b_{g_q(i),g_k(j)}^{h}.
\]

The primary method fixes `gamma=1`. Attention is one ordinary row softmax:

\[
A_{ij}^{h}
=\frac{\exp(\widetilde{s}_{ij}^{h})}
{\sum_k\exp(\widetilde{s}_{ik}^{h})}.
\]

Count-only TGCA uses `b=0` and adds no learned parameters. TGCA+bias uses one small relation table per attention layer and head.

### 5.3 Hierarchical interpretation

Define group mean evidence in log space:

\[
e_{i,g}^{h}
=\operatorname{LogSumExp}_{j:g_k(j)=g}(s_{ij}^{h})-\log N_g.
\]

TGCA group mass is:

\[
\pi_{i,g}^{h}
=\operatorname{Softmax}_{g}\left(e_{i,g}^{h}+b_{g_q(i),g}^{h}\right).
\]

Within group `g`, the conditional token distribution is:

\[
\rho_{i,j\mid g}^{h}
=\operatorname{Softmax}_{j:g_k(j)=g}(s_{ij}^{h}).
\]

The final probability factorizes as:

\[
A_{ij}^{h}=\pi_{i,g_k(j)}^{h}\rho_{i,j\mid g_k(j)}^{h}.
\]

This makes the comparison precise:

- vanilla softmax uses group **sum evidence**;
- normalized split `(0.5,0.5)` fixes group mass independently of evidence;
- TGCA uses group **mean evidence**, then allocates within-group mass normally.

### 5.4 Replication-invariance proposition

**Proposition.** If every key/value pair in one group `g` is duplicated exactly `r` times, count-only TGCA leaves aggregate group mass and the attention output unchanged, up to floating-point error.

**Proof sketch.** After replication, the corrected unnormalized weight for each duplicate is divided by `rN_g` instead of `N_g`. Summing the `r` identical copies cancels the factor `r`:

\[
\sum_{t=1}^{r}\frac{\exp(s_{ij})}{rN_g}
=\frac{\exp(s_{ij})}{N_g}.
\]

Therefore the group contribution to the global denominator is unchanged. Each duplicated value receives `1/r` of its original token weight, so the `r` identical value contributions sum to the original attention output.

This property applies to **exact duplicates**. It does not claim invariance when higher resolution introduces distinct patches, new positional encodings, different receptive fields, or genuinely new evidence.

### 5.5 Relation bias

For two groups, each head has:

\[
B^h=\begin{bmatrix}
b_{0\rightarrow0}^{h} & b_{0\rightarrow1}^{h}\\
b_{1\rightarrow0}^{h} & b_{1\rightarrow1}^{h}
\end{bmatrix}.
\]

Design rules:

- shape per attention module: `[num_heads, num_query_groups, num_key_groups]`;
- initialize every element to zero;
- apply before the one global softmax;
- do not rescale values or attention output;
- report learned biases after centering over key groups for each query group, because adding one constant to a full query-group row is softmax-unidentifiable;
- add no bias regularizer unless a failure is observed and a new ablation is approved.

For DeiT-S with 12 layers, 6 heads, and 2×2 relations, the total is 288 scalar parameters.

## 6. Baseline semantics

| Mode | Operation | Pre-dropout row sum | Count-corrected? | Evidence-driven group mass? |
|---|---|---:|---:|---:|
| `vanilla` | one joint softmax | 1 | No | Yes, sum evidence |
| `split_11` | per-group softmax × `(1,1)` | 2 | Heuristic | No, fixed `(1,1)` |
| `split_05` | per-group softmax × `(0.5,0.5)` | 1 | Heuristic | No, fixed `(0.5,0.5)` |
| `tgca` | subtract `log N_g`, one softmax | 1 | Yes | Yes, mean evidence |
| `tgca_bias` | TGCA + relation bias | 1 | Yes | Yes, mean evidence + learned prior |
| `tgca_gamma05` | subtract `0.5 log N_g` | 1 | Partial | Yes |

`tgca_gamma05` is permitted only if full correction clearly over-corrects. The only planned gamma values are `{0, 0.5, 1}`, where `0` is vanilla. Do not tune gamma per dataset, host, resolution, or seed.

The split baselines are defined only for two groups. KYAM register-sensitivity mode `singleton` is evaluated only with vanilla or TGCA, not split softmax.

## 7. Software architecture

### 7.1 Current repository facts

The current code is an MCTTA/MCTG-derived repository, not a finished TGCA implementation.

Relevant existing normalization sites include:

- `models/vit.py::Attention` — vanilla joint softmax used by `models/mctformer_plus.py`;
- `models/mct_vit.py::Attention` — old split `(1,1)` self-attention;
- `models/adapter_modules.py` — two old split cross-attention paths;
- `models/modules.py` — one active split cross-attention path and other vanilla paths.

Do not compare variants by switching between these files. They contain other architectural differences. The core ablation must select normalization modes inside one shared attention path.

KYAM is not currently present under `hosts/`. Before implementation, pin its official repository commit and record it in the baseline registry. The repository currently has no declared license; do not redistribute a modified copy until licensing or author permission is resolved.

### 7.2 Planned files

```text
TGCA/
├── docs/
│   └── design.md
├── models/
│   └── tgca.py
├── experiments/
│   ├── configs/
│   │   ├── mctformerplus/
│   │   └── kyam/
│   ├── hosts/
│   │   ├── mctformerplus.py
│   │   └── kyam.py
│   └── run_experiment.py
├── tests/
│   ├── test_tgca_normalization.py
│   ├── test_tgca_replication.py
│   ├── test_tgca_masks_and_dtypes.py
│   ├── test_mctformerplus_attention.py
│   └── test_kyam_attention.py
├── tools/
│   ├── analyze_attention_groups.py
│   ├── test_token_replication.py
│   ├── evaluate_scale_consistency.py
│   ├── collect_cam_metrics.py
│   └── export_paper_tables.py
└── results/
    └── <host>/<dataset>/<run_id>/...
```

These paths are planned, not authorization to implement them before the design is approved.

### 7.3 Pure function and module API

Implement one pure normalization function and one `nn.Module` wrapper.

Conceptual API:

```python
def token_group_normalize(
    logits,                    # [B, H, Nq, Nk]
    key_group_ids,             # [Nk] or [B, Nk]
    query_group_ids=None,      # [Nq] or [B, Nq]
    key_valid_mask=None,       # broadcastable to [B, H, Nq, Nk]
    mode="tgca",
    gamma=1.0,
    split_weights=None,
    relation_bias=None,        # [H, Gq, Gk]
):
    """Return pre-dropout attention probabilities."""
```

```python
class TokenGroupNormalizer(nn.Module):
    def __init__(
        self,
        num_heads,
        num_query_groups,
        num_key_groups,
        mode="vanilla",
        gamma=1.0,
        split_weights=(1.0, 1.0),
        learn_relation_bias=False,
    ):
        ...
```

The module must return only the probability tensor by default so it can replace `nn.Softmax(dim=-1)` without changing host call signatures. Diagnostics should be collected with hooks or a separate stateless helper, not a mutable `last_stats` field that is unsafe under distributed execution.

### 7.4 Group layout builders

Token positions are host-specific. Keep them outside the mathematical operator:

```python
build_mctformer_groups(num_classes, num_patches)
build_mctta_cross_groups(num_classes, num_spatial, num_patches)
build_kyam_groups(num_classes, num_patches, register_policy="global")
```

Each builder returns named query/key group layouts and validates total sequence length. Never infer semantic groups from tensor length alone when a host exposes explicit token metadata.

### 7.5 Numerical behavior

Requirements:

1. Compute group counts from valid keys, not nominal padded length.
2. Require at least one valid key per attention row; fail loudly on fully masked rows.
3. Form count logs in float32.
4. For FP16/BF16, perform correction and softmax accumulation in float32, then cast probabilities back to the intended attention dtype.
5. Apply invalid-key masks after correction and bias, immediately before softmax.
6. Preserve exact zero probability for invalid keys.
7. Apply attention dropout **after** normalization, as in the host baseline.
8. Measure row sums before dropout; dropout intentionally changes the realized row sum during training.
9. Do not use `nan_to_num` to conceal invalid rows.
10. Do not clamp logits except where the unchanged host already does so.

### 7.6 Masked group counts

With per-sample valid mask `M_bj`, use:

\[
N_{b,g}=\sum_j M_{bj}\mathbf{1}[g_k(j)=g].
\]

The correction for key `j` in sample `b` is `log N_{b,g_k(j)}`. A group with zero valid keys contributes no logits to softmax. The implementation must never take `log(0)` for an active key.

### 7.7 Checkpoint compatibility

- `vanilla`, `split_11`, `split_05`, and count-only `tgca` add no parameters.
- `tgca_bias` adds only relation-bias parameters.
- Old checkpoints must load in count-only mode with no missing or unexpected keys.
- In bias mode, only the documented relation-bias keys may be missing, and they must be zero-initialized.
- Save normalization configuration inside every checkpoint and result manifest; a checkpoint filename is not sufficient provenance.

### 7.8 Configuration contract

Example:

```yaml
attention_normalization:
  mode: tgca
  gamma: 1.0
  relation_bias: false
  split_weights: [1.0, 1.0]
  key_groups: class_patch
  kyam_register_policy: global
  diagnostics:
    enabled: true
    save_full_attention: false
    aggregate_by_layer_head: true
```

Unknown modes or group policies must raise configuration errors. Do not silently fall back to vanilla.

## 8. Host integration design

### 8.1 MCTformer+

Primary path:

```text
models/mctformer_plus.py
    -> imports VisionTransformer from models/vit.py
    -> models/vit.py::Attention
```

Implementation sequence:

1. Reproduce the existing vanilla path before editing.
2. Add `TokenGroupNormalizer` to `models/vit.py::Attention` with default `mode="vanilla"`.
3. Build the two-group layout from `self.num_classes` and runtime sequence length.
4. Route all five normalization modes through this same class.
5. Preserve returned `weights` as pre-dropout normalized weights.
6. Preserve Q/K/V, `attn_drop`, value aggregation, output projection, and residual paths exactly.
7. Add a host smoke test showing default vanilla output and gradients match the pre-change implementation within strict tolerance under a fixed seed.

CAM extraction in `models/mctformer_plus.py` consumes class-to-patch and patch-to-patch attention. Log both the probabilities used for CAMs and their group aggregates. Do not add CAM rescaling that differs by normalization mode.

### 8.2 Know Your Attention Maps (KYAM)

Initial audited official commit:

```text
3daaec734700a4c9578dd8ce7bedef7f917aed66
```

Re-verify and pin the commit immediately before baseline reproduction.

The official `Attention` class stores `self.attend = nn.Softmax(dim=-1)`, and its recorder attaches hooks to `module.attend`. Preserve this interface by injecting a `TokenGroupNormalizer` module in place of `self.attend`; do not replace it with an unhookable inline function.

Prefer a parent-repository integration wrapper under `experiments/hosts/kyam.py` that:

1. imports the pinned official host;
2. finds each KYAM attention module;
3. replaces only its `attend` normalizer;
4. derives `[global, patch]` groups using class count, patch count, and final register position;
5. leaves all Q/K/V and CAM logic in the host untouched;
6. exposes inserted bias parameters to the optimizer and checkpoint state.

This overlay approach minimizes source redistribution while the upstream repository has no declared license.

KYAM contains optional post-softmax Concrete head gating. The clean normalization experiment uses `prune=False`, because post-softmax gating can change row sums and confound the operator contract. Use two tracks if official baseline parity requires pruning:

- `KYAM-core`: `prune=False`; primary TGCA generality result;
- `KYAM-full`: official pruning setting; optional robustness result with vanilla and TGCA compared under identical gating.

For `KYAM-full`, row-normalization claims apply to the pre-gate attention returned by the normalizer, not the gated output.

KYAM's label-driven class-token dropout is applied after the Transformer output in the audited code, so it does not change attention key counts. Confirm this again at the pinned commit.

### 8.3 MCTTA self- and cross-attention

MCTTA is optional and must be integrated only after the two required hosts pass.

Potential sites:

- `models/mct_vit.py::Attention` for self-attention;
- split paths in `models/adapter_modules.py`;
- split/vanilla paths in `models/modules.py`.

Before any MCTTA result, resolve the legacy cross-attention token-order and output-slicing inconsistency against executable shapes. The integration must use explicit query and key layouts rather than assuming both axes have the same group order.

### 8.4 DiCLIP comparison policy

DiCLIP is not a TGCA host. Its custom attention combines independently normalized attention maps with unequal coefficients and injects diffusion-derived patch affinity. This prevents a clean one-variable normalization comparison.

Use DiCLIP in the recent-method comparison table with:

- venue/year: IEEE T-IP 2026;
- backbone and pretraining;
- image-level supervision and external diffusion prior;
- author-reported versus locally reproduced status;
- VOC/COCO result definition and post-processing.

Do not imply that a DiCLIP number is directly matched to ImageNet-only class-token hosts.

## 9. Attention diagnostics

### 9.1 Quantities collected before dropout

For each layer, head, query group, and key group:

- mean aggregate group mass;
- standard deviation across query rows and images;
- group mean log evidence `e_{i,g}`;
- attention entropy;
- maximum row-sum error;
- valid key count per group;
- relation bias, when enabled.

Required directional summaries include:

- class-query → class-key mass;
- class-query → patch-key mass;
- patch-query → class-key mass;
- patch-query → patch-key mass.

For KYAM, use `global` in the primary logs and additionally separate class and register query rows in diagnostic summaries even though they share one correction group.

### 9.2 Online aggregation

Full attention storage scales quadratically with token count and should be disabled by default. Aggregate within the forward hook and save sufficient statistics.

Required CSV columns:

```text
run_id,dataset,split,image_id,resolution,patch_count,class_count,
layer,head,query_group,key_group,mean_mass,std_mass,
mean_entropy,max_row_sum_error,mean_group_log_evidence
```

For synthetic tests, full matrices may be saved because sequence lengths are small.

### 9.3 Distributed behavior

- each rank writes a rank-specific temporary file;
- aggregation occurs once after a barrier;
- sort final rows deterministically;
- include world size and rank count in the run manifest;
- never average head IDs or layer IDs together before raw per-head data is saved.

## 10. Unit and synthetic validation

### 10.1 Required deterministic unit tests

1. **Vanilla parity:** `mode=vanilla` matches `torch.softmax(logits, dim=-1)`.
2. **One-group reduction:** TGCA equals vanilla when all valid keys share one group.
3. **Row sum:** vanilla, normalized split, TGCA, and TGCA+bias rows sum to one before dropout.
4. **Old split row sum:** `split_11` rows sum to two and `split_05` rows sum to one.
5. **Group replication:** duplicating one entire key group preserves TGCA aggregate mass.
6. **Output replication:** duplicating the corresponding keys and values preserves `A @ V`.
7. **Vanilla non-invariance:** the same replication changes vanilla aggregate group mass in a controlled equal-logit case.
8. **Relation lookup:** each query/key group pair receives the intended bias.
9. **Zero-bias reduction:** TGCA+bias with a zero table equals count-only TGCA.
10. **Mask correctness:** invalid keys receive exactly zero probability and do not contribute to counts.
11. **Batch-specific masks:** two samples with different valid counts obtain the correct per-sample correction.
12. **Rectangular attention:** support `Nq != Nk` for MCTTA cross-attention.
13. **Gradient check:** finite gradients for logits, Q/K/V-derived inputs, and relation bias.
14. **Device/dtype:** CPU FP32; CUDA FP32/FP16/BF16 when supported.
15. **Checkpoint loading:** only expected relation-bias keys differ.
16. **No input mutation:** logits, values, masks, and group-ID tensors remain unchanged.

Suggested FP32 tolerances:

```text
row sum max error <= 1e-6
replicated group mass max error <= 1e-6
replicated output max absolute error <= 1e-5
vanilla parity max absolute error <= 1e-7
```

Mixed-precision tolerances must be defined by dtype in the tests and must not be loosened after inspecting method outcomes.

### 10.2 Synthetic mechanism grid

Use controlled Gaussian logits plus exact duplicates:

```text
class counts:  [1, 20, 80]
patch counts:  [49, 196, 400, 784, 1024]
heads:         [1, 6]
logit regimes: equal, iid normal, class-favored, patch-favored
replication:   [1, 2, 4, 8]
```

Produce:

- group mass versus patch count;
- group mass versus replication factor;
- attention-output error versus replication factor;
- row-sum error by dtype;
- gradient-norm distributions.

The first mechanism figure should be generated from this script, not manually drawn.

### 10.3 Host smoke tests

For both required hosts:

- construct the smallest supported model;
- run one forward/backward batch;
- verify sequence layout and group IDs;
- verify pre-dropout row sums;
- verify attention and CAM shapes;
- load the vanilla checkpoint;
- confirm default vanilla logits/CAMs are unchanged after the integration wrapper is added.

## 11. Experiment program

### Stage A — Baseline registry and environment freeze

Before modifying host behavior, create a baseline manifest containing:

```text
host repository URL
host commit SHA
TGCA repository commit SHA and diff status
environment name and package lock
Python/PyTorch/CUDA/cuDNN/GPU versions
dataset roots and file-list hashes
pretrained-weight URL and SHA256
training/evaluation commands
seed
expected and reproduced metrics
```

Environment policy after adding KYAM:

- `tgca-repro`: MCTformer+/current TGCA code;
- `kyam-repro`: pinned KYAM reproduction;
- `diclip-repro`: only if DiCLIP is locally reproduced;
- `more-repro` and `cti-repro`: optional and remain independent.

Do not force hosts into one environment before each official baseline is reproduced. The existing README environment section will need a separate update before KYAM execution.

Baseline acceptance targets:

- MCTformer+: within `±0.3` raw CAM mIoU of the pinned official protocol;
- KYAM: within `±0.5` pseudo-mask/raw-attention mIoU of the pinned official protocol, unless an upstream issue documents a wider reproducibility range.

If the target is missed, diagnose the baseline; do not compensate by tuning TGCA.

### Stage B — Mechanism-only evaluation on a vanilla checkpoint

Use a reproduced vanilla MCTformer+ checkpoint. Evaluate the same images without retraining at:

```text
224, 320, 448, 512
```

Collect all attention diagnostics, classification metrics, raw CAM metrics, and cross-scale consistency. This stage determines whether the proposed failure mode is present in trained attention.

Separate two experiments:

1. **Resolution stress:** normal image resizing and positional interpolation; realistic but confounded.
2. **Exact token replication:** fixed synthetic logits/keys/values; isolates the mathematical property.

Do not describe the realistic resolution test as a pure cardinality intervention.

### Stage C — One-seed normalization pilot

On VOC, train or fine-tune one seed for:

```text
vanilla
split_11
split_05
tgca
tgca_bias
```

Use one shared configuration template with only the normalization fields changed. The pilot answers:

- does count-only TGCA train stably?
- does it change classification quality materially?
- does it improve raw CAM or scale stability?
- does relation bias add value beyond count correction?

Do not proceed to three seeds if count-only TGCA fails unit tests, materially degrades classification, or shows no mechanism effect.

### Stage D — VOC core experiment

Canonical seeds:

```text
0, 1, 2
```

If an official host requires a specific historical seed, reproduce it separately as `official_seed`; do not substitute it for the three canonical seeds.

Run the five required modes with identical:

- initialization and pretrained checkpoint;
- image lists and augmentation;
- optimizer and schedule;
- number of updates and gradient accumulation;
- checkpoint selection rule;
- multi-scale CAM generation;
- thresholds and post-processing.

Report mean, standard deviation, and per-seed values.

### Stage E — Resolution and patch-count stress

Evaluate the selected checkpoint from every core mode at all four resolutions. The checkpoint-selection rule must be fixed before scale evaluation.

Also run, if compute permits:

- patch-size change with fixed image content;
- controlled patch subsampling;
- exact patch-key/value replication at an isolated attention layer.

Patch subsampling and patch-size changes are secondary because they modify information content or positional structure.

### Stage F — KYAM generality

1. Pin and reproduce official KYAM.
2. Run `KYAM-core` with `prune=False` for vanilla and count-only TGCA.
3. If positive, add `tgca_bias` and the normalized split baseline.
4. Run the register-policy sensitivity comparison.
5. Run the four-resolution stress test.
6. If the official best setting uses pruning, optionally repeat vanilla and best TGCA in `KYAM-full` with identical gating.
7. Expand to three seeds if the one-seed mechanism and CAM results are positive.

KYAM counts as a positive independent host only if the improvement is observed without changing its loss, token masking, CAM construction, backbone, or evaluation pipeline.

### Stage G — COCO and downstream segmentation

Proceed only after the VOC mechanism and KYAM gates pass.

Use one fixed downstream pipeline:

1. train the image-level classifier;
2. generate CAMs using identical scales;
3. apply one fixed refinement/pseudo-label process;
4. train one fixed segmentation network;
5. evaluate with the same code and checkpoint rule.

Raw CAM evidence remains primary. Downstream segmentation is validation, not a substitute for mechanism evidence.

### Stage H — Recent external comparison

Include DiCLIP and verified 2025–2026 methods in a setting-aware table. Required columns:

```text
method, venue/year, backbone, pretraining, language supervision,
foundation/diffusion prior, image-level labels, post-processing,
VOC val/test, COCO val, author-reported/reproduced
```

Do not rank incomparable rows as one undifferentiated leaderboard.

## 12. Metrics and evaluation definitions

### 12.1 Raw CAM and pseudo-mask metrics

Report:

- raw CAM/seed mIoU;
- foreground pixel precision and recall;
- false-positive rate or a precisely defined confusion ratio;
- pseudo-mask mIoU after one fixed refinement pipeline;
- downstream segmentation mIoU.

If using MoRe's “confusion ratio,” reproduce its exact published definition and cite it. Otherwise use an explicitly defined false-positive diagnostic rather than reusing the name.

### 12.2 Threshold policy

Threshold selection can invalidate the comparison. Use this policy:

1. choose the background/CAM threshold on a fixed development split using vanilla only;
2. freeze it for every normalization mode, seed, and resolution;
3. report a threshold sweep as a diagnostic, not as the headline result;
4. add a threshold-free consistency metric so robustness is not determined by one binarization point;
5. never select thresholds on the test set.

### 12.3 Classification metrics

Use the host's standard multi-label classification metric and add mAP where feasible. Report any change in class-token classification independently of CAM changes.

### 12.4 Group-mass stability

For each image, layer, and head, compute variance over scales:

\[
V_{i,l,h,g}=\operatorname{Var}_{s\in\{224,320,448,512\}}m_{i,l,h,g}^{(s)}.
\]

Report:

- mean and median variance;
- interquartile range;
- last-layer and all-layer summaries;
- per-head heatmaps;
- slope of group mass versus `log(patch_count)`.

### 12.5 Cross-scale CAM consistency

Resize all CAMs to the original image coordinates. For scale `s` and reference scale `448`:

\[
C_{\mathrm{soft}}(s)=
\frac{\langle \widehat M_s,\widehat M_{448}\rangle}
{\|\widehat M_s\|_2\|\widehat M_{448}\|_2+\epsilon}.
\]

Also report binarized IoU at the frozen threshold. Normalize CAMs using one common rule before comparison and do not normalize differently by method.

### 12.6 Efficiency

Measure at the native host resolution and batch size 1 plus one representative training batch:

- parameters;
- MACs/FLOPs;
- images per second;
- inference latency after warm-up;
- peak GPU memory;
- training iteration time.

Count correction should add no parameters, but its runtime overhead must be measured rather than asserted.

## 13. Statistical analysis

- Core VOC ablations use three seeds.
- Report mean ± standard deviation and every individual seed.
- Use paired image-level bootstrap confidence intervals for CAM and consistency differences, with 10,000 resamples and a fixed bootstrap seed.
- Treat images, not pixels, as bootstrap units.
- Do not claim significance from head-level samples because heads within one model are not independent experimental replicates.
- Report effect sizes even when confidence intervals overlap.
- Do not discard failed seeds unless a host-independent infrastructure failure is documented before metrics are inspected.

## 14. Result and provenance layout

Each run writes:

```text
results/<host>/<dataset>/<run_id>/
├── config.yaml
├── command.txt
├── git_state.json
├── environment.txt
├── dataset_manifest.json
├── train.log
├── metrics.json
├── metrics_by_class.csv
├── attention_group_mass.csv
├── scale_consistency.csv
├── efficiency.json
└── checkpoint_manifest.json
```

Large checkpoints and datasets must not be committed. `checkpoint_manifest.json` stores the path/URL, SHA256, selection metric, epoch/step, and producing run ID.

Suggested run ID:

```text
<date>-<host>-<dataset>-<mode>-s<seed>-<short_commit>
```

Every paper table must be generated from these files by script. Do not manually transcribe terminal numbers.

## 15. Go/no-go gates

### Gate 0 — Baseline integrity

Pass only when the host baseline is within the predeclared tolerance and all provenance fields are recorded.

### Gate 1 — Operator correctness

Pass only when all unit tests, replication invariance, mask tests, and gradient tests succeed. Row normalization is non-negotiable for vanilla, normalized split, and TGCA variants.

### Gate 2 — Phenomenon exists in trained attention

Pass when:

- the paired bootstrap interval for the vanilla group-mass slope across patch counts excludes zero in the CAM-relevant layers or heads; and
- group-mass changes co-occur with a measurable CAM or cross-scale consistency change.

Synthetic vanilla drift alone is insufficient to pass this gate.

### Gate 3 — TGCA is more than rescaling

Pass when count-only TGCA improves raw CAM quality or scale consistency relative to both vanilla and normalized split, while:

- retaining unit row sums;
- avoiding material classification degradation;
- using the same thresholds and CAM pipeline;
- showing lower group-mass slope or variance.

The practical target remains roughly `+0.8 to +1.0` raw CAM mIoU on MCTformer+, but a smaller gain is acceptable when the invariance and scale-stability evidence is unusually strong.

### Gate 4 — Independent host

Pass only when KYAM shows a positive effect under the clean `KYAM-core` comparison. MoRe, CTI, or MCTTA cannot replace this required result without revising the approved research plan.

### Gate 5 — COCO authorization

Start expensive COCO training only after Gates 0–4 pass, overhead is negligible, and no unresolved threshold or checkpoint confound remains.

### Stop or reframe

Stop the TGCA paper framing if:

- trained attention shows no meaningful cardinality association;
- TGCA helps only because another baseline has non-unit output scale;
- benefits disappear under a fixed threshold;
- relation bias, rather than count correction, accounts for all gains;
- results occur only in MCTTA;
- KYAM results depend on an arbitrary register grouping;
- classification degrades enough to explain the CAM tradeoff;
- different post-processing is required per method.

## 16. Risks and confounds

### 16.1 Scientific risks

1. **Exchangeable-logit assumption may be irrelevant.** Trained logits can learn offsets that compensate for group size.
2. **More tokens may represent more evidence.** Correcting cardinality can over-compensate when additional distinct patches genuinely improve coverage.
3. **Effective sample size differs from token count.** Nearby patches are correlated; `N_g` may not equal the number of independent evidence units.
4. **Log-mean-exp is not invariant to new distinct samples.** TGCA guarantees exact-duplication invariance, not invariance to every resolution change.
5. **Residual networks can absorb normalization changes.** Later layers may learn around either vanilla or TGCA, hiding the intended mechanism.
6. **Benefits may come from regularization.** Any CAM gain must be tied to group-mass behavior rather than merely a changed optimization landscape.

### 16.2 Resolution confounds

Changing input size also changes:

- positional-embedding interpolation;
- apparent object scale;
- boundary sampling;
- patch receptive fields;
- augmentation statistics;
- memory pressure and possibly batch size.

Therefore, resolution stress is realistic evidence but not a pure causal intervention. Exact replication is the pure operator test.

### 16.3 Host confounds

- KYAM includes a register token with ambiguous semantic grouping.
- KYAM's optional post-softmax head gate can break unit row sums.
- KYAM's code release lacks a declared license and complete environment specification.
- MCTTA contains several duplicated attention implementations and known token-order inconsistencies.
- Existing MoRe/CTI environments differ substantially and must not be used to “fix” a failed primary baseline.

### 16.4 Measurement confounds

- Per-method CAM normalization or thresholds can create artificial gains.
- Averaging heads/layers too early can hide opposite effects.
- Saving attention after dropout or gating can misstate the normalized distribution.
- Full attention logging can cause OOM and alter batch size or throughput.
- Choosing only visually favorable scales, heads, or images creates selection bias.

### 16.5 Relation-bias risks

- relation bias may learn back a cardinality prior;
- bias parameters are identifiable only up to a row-wise additive constant;
- bias can dominate count correction with little parameter cost;
- a positive result only with bias weakens the claim that cardinality correction is sufficient.

Always report count-only TGCA as the primary mechanistic variant.

## 17. Implementation sequence

### Phase 1 — Infrastructure

- [ ] Create `models/tgca.py` with pure function and module wrapper.
- [ ] Add all deterministic unit tests.
- [ ] Add synthetic replication and plotting tools.
- [ ] Add result manifests and config validation.
- [ ] Confirm no source file outside `TGCA/` is modified.

### Phase 2 — MCTformer+ baseline and instrumentation

- [ ] Pin official repository/checkpoint references.
- [ ] Reproduce vanilla VOC result.
- [ ] Add normalization module in default-vanilla mode.
- [ ] Prove pre/post integration vanilla parity.
- [ ] Add online group-mass diagnostics.
- [ ] Run four-resolution phenomenon test.

### Phase 3 — MCTformer+ ablation

- [ ] Run one-seed five-mode pilot.
- [ ] Evaluate all metrics with frozen threshold policy.
- [ ] Make Gate 3 decision.
- [ ] Run three seeds for passing variants.
- [ ] Measure efficiency.

### Phase 4 — KYAM generality

- [ ] Resolve license/redistribution approach.
- [ ] Pin official code and construct `kyam-repro` environment.
- [ ] Reproduce vanilla baseline.
- [ ] Inject hook-compatible normalizer overlay.
- [ ] Validate primary register grouping.
- [ ] Run one-seed vanilla/TGCA pilot and scale test.
- [ ] Run register-policy sensitivity.
- [ ] Expand positive result to required ablations/seeds.

### Phase 5 — COCO and recent comparison

- [ ] Approve COCO only after all gates pass.
- [ ] Run the selected primary variants under one fixed pipeline.
- [ ] Verify DiCLIP publication metadata and comparison settings.
- [ ] Separate author-reported and reproduced external results.
- [ ] Generate setting-aware paper tables from raw files.

## 18. Definition of done

The TGCA experimental package is complete only when:

- [ ] vanilla baseline reproduction is documented for both required hosts;
- [ ] all normalization modes share one Q/K/V and value path;
- [ ] row-sum, mask, dtype, gradient, and replication tests pass;
- [ ] trained vanilla cardinality sensitivity is measured;
- [ ] count-only TGCA reduces the intended sensitivity;
- [ ] old split `(1,1)` and normalized split `(0.5,0.5)` are fairly isolated;
- [ ] core VOC results include three seeds;
- [ ] KYAM provides a positive independent-host result;
- [ ] register-token handling is reported and does not determine the conclusion;
- [ ] raw CAM, precision, recall, false-positive, classification, and consistency metrics are saved;
- [ ] downstream segmentation uses one fixed pipeline;
- [ ] parameters, FLOPs, latency, memory, and throughput are measured;
- [ ] DiCLIP appears only as a transparent external comparison;
- [ ] every result has a config, command, seed, checkpoint hash, environment, and commit;
- [ ] every paper number and plot is generated from machine-readable results;
- [ ] limitations and negative results are retained rather than hidden.

## 19. Immediate next action after approval

Implement only the pure `models/tgca.py` operator and its deterministic tests. Do not modify MCTformer+, download KYAM, or begin training until the operator tests and the group-layout decisions in this document are approved.
