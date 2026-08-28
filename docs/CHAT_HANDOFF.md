# TGCA research handoff

Last updated: **2026-08-28 (Asia/Tokyo)**

> **Superseded research direction.** `docs/CHAT_HANDOFF-0827.md` is now the
> canonical scientific and execution plan. It replaces the TGCA paper plan
> below with Background-Aware Competitive Semantic Slots (BCSS) for ICLR 2027.
> The historical TGCA state in this file is retained for provenance only.

## BCSS VOC screen state on 2026-08-28

The diagnostic system and VOC minimum-screen implementation requested by
Section 11 of `docs/CHAT_HANDOFF-0827.md` now exists. Exact variant contracts,
commands, outputs, and verification are recorded in:

```text
docs/BCSS_EXPERIMENTS.md
```

Implemented experiment variants are `E0`, `E1`, `E2`, `E4`, `E5`, and `E6`.
The queue entry point is:

```text
experiments/ablations/run_bcss_voc_screen.sh
```

The full seed-0 VOC screen completed normally at `2026-08-28T03:28:12+09:00`
from commit `4147fc368fdef2698d3a0570c6c5b913527327ee`. All six variants reached 45
epochs, produced final checkpoints and diagnostics, and passed the queue's
completion and manifest checks. The comparison is:

```text
results/mctformerplus/voc/comparisons/
  bcss-screen-20260827-190911-s0-4147fc3/screen.json
```

The screen is a **no-go for the current BCSS formulation**. E6 obtains `56.145`
raw CAM mIoU versus `70.063` for E0, while semantic foreground recall falls
from `85.817` to `66.089`. CBL worsens by `2.41%`; CCS-bg falls by only `10.61%`,
short of the required `20%`; only the classification-retention constraint
passes. E4--E6 assign background ownership above `0.9` to more than `99%` of
patches. The ownership rows are correctly normalized, so this is an objective
degeneracy rather than a softmax or Split implementation error.

Do not expand this formulation to three seeds, COCO, downstream segmentation,
or an independent host. The only authorized follow-up is the `e4_mass` seed-0
debug documented in `docs/BCSS_EXPERIMENTS.md`. It multiplies each class slot's
semantic logits by its mean patch ownership, causing epsilon ownership to
approach the `log(N_c)` no-information loss instead of near-zero loss. It does
not change historical E4--E6 behavior, ownership normalization, slot
aggregation, CAM gating, or the training schedule.

The anti-collapse tests, exact E4 pre-loss parity test, complete 44-test suite,
and CUDA FP16 backward all pass. On 50 images from the collapsed E4 checkpoint, the
old and mass-aware foreground losses are `0.00130` and `2.93545`, respectively,
with `log(20)=2.99573`. Launch only this one debug after creating the required
clean Git checkpoint; do not start a multi-variant queue.

This file transfers the scientific context, repository state, server setup, completed evidence, active experiments, and next decisions for the TGCA project. It is intended to be the first file read by a new Codex task on the `LHR` server.

## Resume instructions

On `LHR`, work only in:

```text
/home/peng/code/TGCA
```

Before doing anything else:

```bash
cd /home/peng/code/TGCA
git status --short --branch
git log -8 --oneline --decorate
tmux ls
tail -n 120 results/queues/4147fc3/bcss-voc-screen-20260827-190911.log
```

Then read these files completely:

1. `docs/CHAT_HANDOFF.md` — this operational handoff;
2. `docs/design.md` — detailed TGCA method and experiment design;
3. `docs/RESEARCH_PLAN_FULL.md` — full paper rationale and go/no-go plan;
4. `docs/TIP_REVIEWS.md` — reviewer objections the new work must resolve.

`docs/MCTTA.pdf` is the rejected legacy manuscript. It is evidence and background, not a draft to compress or edit.

No tmux experiment is active as of this update. When a future queue is active,
inspect it read-only and do not stop or restart it merely for status checks.

## Project objective

Prepare a focused ICASSP 2027 Computer Vision paper:

> **Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation**

The paper is not a shortened version of the rejected 16-page MCTTA manuscript. Its single contribution is a measurable failure mode of vanilla attention normalization over heterogeneous token groups and a minimal correction:

```text
s_ij^h = (q_i^h)^T k_j^h / sqrt(d_h)
s_tilde_ij^h = s_ij^h - log(N_{g(j)}) + b_{g(i),g(j)}^h
A_ij^h = softmax_j(s_tilde_ij^h)
```

The central hypothesis is that a joint softmax over a small class-token group and a much larger patch-token group mixes semantic evidence with group cardinality. Increasing patch count, for example through input resolution, may shift aggregate attention mass even when the evidence distribution has not meaningfully changed. TGCA subtracts the log key-group size before one global softmax.

## Non-negotiable scientific contract

TGCA must satisfy all of the following:

- every attention row sums to one;
- duplicating every key/value in one group leaves that group's aggregate mass and attention output unchanged, up to numerical tolerance;
- the count correction has no trainable parameters;
- optional relation bias is only a tiny per-head query-group/key-group matrix;
- the same formulation supports self-attention and rectangular cross-attention;
- value projection, output scaling, residual paths, and all unrelated training choices remain fixed.

The required normalization ablation is:

1. vanilla global softmax;
2. legacy split softmax `(1,1)` — row sum is two by design and this must be stated;
3. normalized split softmax `(0.5,0.5)` — row sum is one but group mass is forced to `0.5/0.5`;
4. TGCA without relation bias;
5. TGCA with relation bias.

Do not reinterpret the two split-softmax baselines as TGCA. They remove group-size effects by assigning fixed group mass, whereas TGCA retains evidence-driven competition between group means.

## Host and baseline roles

- **MCTformer+** is the primary TGCA implementation host.
- **Know Your Attention Maps: Class-specific Token Masking for WSSS** (ICCV 2025), official repository `https://github.com/HSG-AIML/TokenMasking-WSSS`, is the required independent implementation host. Its singleton register token must be handled as an explicit design choice and tested; do not silently merge it into class or patch tokens.
- **DiCLIP** (T-IP 2026), official repository `https://github.com/zwyang6/DiCLIP`, is a recent external comparison only. Do not patch TGCA into DiCLIP for the core generality claim.
- **MoRe** and **CTI** are optional supplementary hosts and currently exist as Git submodules.

Do not return to graph-adapter, CTP-convergence, hierarchical-fusion, pipeline-taxonomy, universal-adapter, or unmatched-SOTA claims. Do not use “Adapter” in the new title. Do not call the old frozen-classifier-plus-separate-segmentation procedure “single-stage.”

## Repository policy

The sole working code repository is:

```text
/home/peng/code/TGCA
```

Create or modify research code only within that directory. Do not recreate root-level `src/`, `experiments/`, or `results/` beside the TGCA repository. Keep legacy manuscript material read-only.

Repository remote:

```text
origin  https://github.com/jungletada/TGCA.git
```

Git branch state at this handoff:

```text
main                            7bd603e [origin/main] Document environments and add independent hosts
research/mctformerplus-baseline 4147fc3 [origin/research/mctformerplus-baseline] Add BCSS support to MCTformerPlus and update training and evaluation processes
```

The active server checkout is `research/mctformerplus-baseline` at commit:

```text
4147fc368fdef2698d3a0570c6c5b913527327ee
```

The completed TGCA pilot results were produced from result commit
`22427d60bff5d1f6c4cc9c5c33f8912502d5a4b0`. Commit `596fbc6` adds
post-analysis tooling and documentation without changing that pilot's
result-critical model, training, CAM-generation, or ablation-runner code.

Commit `4147fc3` contains the BCSS implementation used by the completed VOC
screen. The tracked worktree was clean when that queue started and when its
results were audited. This 2026-08-28 result-summary update modifies only
`docs/CHAT_HANDOFF.md` and `docs/BCSS_EXPERIMENTS.md`; it is not committed.
Do not commit, push, or merge it without explicit user approval.

Relevant commits:

```text
4147fc3 Add BCSS support to MCTformerPlus and update training and evaluation processes
596fbc6 Refactor VOC normalization process and enhance diagnostics for TGCA integration
22427d6 Implement TGCA diagnostics and VOC normalization pilots
cf44aa2 Make CRF optional for raw CAM generation
63d8877 Add reproducible MCTformer+ VOC baseline pipeline
d6a4a90 Fix vanilla MCTformer+ baseline plumbing
7bd603e Document environments and add independent hosts
```

Independent host submodules:

```text
hosts/CTI   1c6fdb4d14e6843e3d861ebd4580468e30598859
hosts/MoRe  d733d347e64f21425df245341e1f88900886b2bb
```

## Server and environment

SSH alias:

```bash
ssh LHR
```

Server project path:

```text
/home/peng/code/TGCA
```

Primary Conda environment:

```text
tgca-repro
Python 3.9.25
torch 2.1.0
torchvision 0.16.0
timm 0.4.12
CUDA runtime 11.8
```

Observed GPU for current runs:

```text
NVIDIA RTX A6000, 49140 MiB
driver 580.173.02
```

Activate the environment with:

```bash
source /home/peng/anaconda3/etc/profile.d/conda.sh
conda activate tgca-repro
cd /home/peng/code/TGCA
```

MoRe, CTI, and TGCA require separate Conda environments because their dependency stacks conflict. Do not merge them merely for convenience.

Codex CLI is installed at `/usr/local/bin/codex`, but it did not start at this snapshot because the server's default Node.js was `v12.22.9` and could not parse the installed CLI. Install/select a current Node.js LTS using an available environment or version manager, reinstall/update `@openai/codex`, and authenticate. Do not copy an entire `~/.codex` directory from another machine because it can contain credentials and machine-specific state.

## Dataset and pretrained assets

The data links are available:

```text
data/VOCdevkit -> /home/peng/data/VOCdevkit
data/MSCOCO    -> /home/peng/data/MSCOCO
```

The development dataset is PASCAL VOC 2012. Current baseline manifests record:

- training: VOC augmented training split;
- CAM generation and raw-CAM evaluation: 1,464 VOC train images;
- validation classification: 1,449 images;
- input training size: `448`;
- CAM scales: `1.0, 0.75, 1.25`;
- ImageNet/DeiT initialization: `$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth`.

Every result directory records dataset hashes, pretrained-weight hash, package freeze, Conda explicit package list, GPU, Git commit, command, seed, and checkpoint hash. Preserve that provenance contract.

Do not start COCO until the VOC mechanism and independent-host go/no-go gates pass.

## Implemented TGCA code

Commit `22427d6` added the initial implementation and experiment plumbing:

- `models/tgca.py` — shared `TokenGroupNormalizer` and functional normalization;
- `models/vit.py` — MCTformer+ attention integration;
- `train_model_v2.py` and `make_cam.py` — normalization-mode arguments;
- `tools/analyze_attention_groups.py` — per-image/layer/head/direction group-mass diagnostics at multiple resolutions;
- `tools/test_token_replication.py` — synthetic cardinality and replication experiment;
- `tests/test_tgca_normalization.py` — row sums, masks, relation bias, gradients, mixed precision;
- `tests/test_tgca_replication.py` — group-mass and output replication invariance;
- `tests/test_mctformerplus_attention.py` — host integration checks;
- `experiments/diagnostics/` — reproducible mechanism runners;
- `experiments/ablations/` — reproducible normalization pilots;
- `experiments/run_mctformerplus_next_experiments.sh` — active sequential queue.

Supported mode names in the current code are:

```text
vanilla
split_11
split_05
tgca
tgca_bias
tgca_gamma05
```

The five-mode core suite uses the first five except `tgca_gamma05`.

The legacy `split_11` operator deliberately has row sum two. The diagnostic tool therefore computes its reported “maximum row-sum error” relative to two, not relative to one. Never cite that metric as proof that `split_11` is normalized to unit mass.

## Trustworthy vanilla MCTformer+ baseline

The first completed reproduction is:

```text
run:       results/mctformerplus/voc/20260825-mctformerplus-voc-vanilla-s0-63d8877
dataset:   PASCAL VOC 2012 train, 1,464 images
metric:    raw CAM mIoU, no CRF
threshold: 0.45
result:    69.50%
seed:      0
epochs:    45
input:     448
scales:    1.0, 0.75, 1.25
```

The exact command is:

```bash
TGCA_GPU_ID=0 \
TGCA_RUN_ID=20260825-mctformerplus-voc-vanilla-s0-63d8877 \
bash experiments/baselines/run_mctformerplus_voc.sh
```

The final checkpoint SHA-256 is:

```text
f63dc438a2b2b2650aec5d7c4a0c2c2a92780893466138ed2b71bdc70be43cd7
```

The training commit was `63d8877`; CAM evaluation used `cf44aa2`; the referenced official MCTformer commit was `0acc27ada87a5582053efb14648442d8644168aa`.

Important interpretation: threshold `0.45` was selected by the existing `train_model_v2.py` tuning workflow. Keep the script, but label `0.45` as the vanilla-seed-0 selected threshold. All current normalization pilots use this same fixed threshold so that each method does not tune its own evaluation threshold.

The integrated vanilla run at commit `22427d6` reproduced the same result exactly:

```text
run:    results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-s0-22427d6
result: 69.50% raw CAM mIoU at fixed threshold 0.45
```

This is the main regression check that the instrumentation and mode plumbing preserve vanilla behavior.

## Completed mechanism evidence

### Synthetic token-replication test

Run:

```text
results/mechanism/synthetic/20260826-synthetic-replication-22427d6
```

Command:

```bash
bash experiments/diagnostics/run_synthetic_replication.sh
```

The grid contains 960 configurations over class counts `[1,20,80]`, head counts `[1,6]`, patch counts `[49,196,400,784,1024]`, four logit regimes, and replication factors `[1,2,4,8]`, with seed `2027`.

Observed maximum output change after within-group replication:

```text
vanilla: 0.4615142941
TGCA:    0.0000215769
```

Maximum TGCA row-sum error in this mixed-precision diagnostic:

```text
0.0000165701
```

This strongly confirms the mathematical replication property in the synthetic setting. It does not by itself establish better CAM quality.

### Full VOC resolution/group-mass diagnostic for vanilla

Run:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-vanilla-scale-diagnostic-22427d6
```

It processed all 1,464 VOC train images at resolutions `224, 320, 448, 512`, corresponding to patch counts `196, 400, 784, 1024`.

Key results:

```text
maximum row-sum error:       1.430511474609375e-06
mean group-mass variance:    0.0046481917763
median group-mass variance:  0.0018502148937
```

Directional mean slopes versus log patch count, all layers:

```text
class query -> class key: -0.0994545
class query -> patch key: +0.0994545
patch query -> class key: -0.0539273
patch query -> patch key: +0.0539273
```

All four bootstrap 95% confidence intervals exclude zero by a wide margin. In the last three layers, the corresponding magnitudes are approximately `0.09546` for class queries and `0.08179` for patch queries.

Do not use the aggregate `mean_group_mass_slope_vs_log_patch_count` value, which is approximately zero because complementary class-key and patch-key directions cancel under indiscriminate averaging. The directional slopes are the interpretable mechanism result.

This is direct empirical evidence that vanilla attention reallocates aggregate mass toward patch keys as patch count/resolution rises. It still does not prove causality for CAM quality; the normalization ablation must establish that link.

## Completed primary experiment queue

Session:

```text
tgca-mctplus-next
```

Pane start command:

```bash
bash -lc 'source /home/peng/anaconda3/etc/profile.d/conda.sh && \
conda activate tgca-repro && \
cd /home/peng/code/TGCA && \
set -o pipefail && \
bash experiments/run_mctformerplus_next_experiments.sh 2>&1 | \
tee results/queues/22427d6/mctformerplus-next.log'
```

Queue log:

```text
results/queues/22427d6/mctformerplus-next.log
```

The queue performs:

1. a full vanilla resolution diagnostic;
2. five sequential 45-epoch normalization runs:
   `vanilla`, `split_11`, `split_05`, `tgca`, `tgca_bias`;
3. raw CAM generation and fixed-threshold evaluation after each run;
4. a full four-resolution attention diagnostic after each run.

Final status, completed **2026-08-27 01:19 JST**:

- vanilla: complete, `69.50%` raw CAM mIoU;
- split `(1,1)`: complete, `58.22%` raw CAM mIoU;
- split `(0.5,0.5)`: complete, `66.31%` raw CAM mIoU;
- count-only TGCA: complete, `69.05%` raw CAM mIoU;
- TGCA with relation bias: complete, `68.88%` raw CAM mIoU.

The completed split `(1,1)` run is:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-split_11-s0-22427d6
```

Its fixed-threshold raw CAM score is `58.22%`, substantially below vanilla. Its group-mass variance is essentially zero because this baseline forces each group to aggregate to mass one; total row mass is two. This result does not refute TGCA because `split_11` changes output scale and removes evidence-driven inter-group competition.

The completed TGCA-with-relation-bias run is:

```text
results/mctformerplus/voc/20260826-mctformerplus-voc-tgca_bias-s0-22427d6
```

Each mode uses:

```bash
TGCA_GPU_ID=0 TGCA_SEED=0 TGCA_FIXED_THRESHOLD=0.45 \
bash experiments/ablations/run_mctformerplus_voc_mode.sh MODE
```

Do not launch a duplicate primary queue. All five run directories contain `metrics.json`, the final checkpoint, CAM evaluation outputs, and `attention_diagnostics/metrics.json`.

## Completed post-pilot diagnostics

The original waiting session observed the primary `QUEUE_COMPLETE` marker and then exited at **2026-08-27 01:19 JST** because `HEAD` had advanced from result commit `22427d6` to analysis commit `596fbc6`. This was the intended provenance guard, not an experiment failure. The intervening commit added documentation and post-analysis tools only; `models/`, training, CAM generation, and the ablation runner are unchanged between the commits.

A provenance-explicit replacement session started at **2026-08-27 11:20 JST** and completed at **11:45 JST**:

```text
tgca-mctplus-post
```

It runs:

```bash
bash experiments/diagnostics/run_mctformerplus_post_pilot_resume.sh
```

and logs to:

```text
results/queues/22427d6/mctformerplus-post-pilot.log
```

The replacement records result commit `22427d6` and analysis commit `596fbc6` separately and verifies that result-critical code did not change. Its CUDA test stage passed `19/19`. There is no active tmux session after normal completion.

After the marker appears, it sequentially performs:

1. CUDA FP32/FP16/BF16 normalization and host tests on exact commit `22427d6`;
2. fixed-threshold raw-CAM precision, recall, and false-positive diagnostics for all five modes;
3. matched batch-1 inference latency, throughput, parameter, and peak-memory measurements;
4. flip-aggregated CAM generation at short-side resolutions `224`, `320`, `448`, and `512` for every mode;
5. fixed-threshold CAM quality at every resolution;
6. soft CAM cosine, class-mask IoU, foreground IoU, semantic-mask IoU, and pixel agreement relative to `448`;
7. an audited five-mode JSON/CSV comparison under:

```text
results/mctformerplus/voc/comparisons/pilot-s0-22427d6
```

The post-pilot queue stops after producing Gate 3 evidence. It does not automatically launch additional seeds, partial-gamma training, KYAM, or COCO. Those depend on reviewing the completed pilot. At queue creation, the filesystem had approximately `170 GiB` available; the four-resolution CAM outputs are expected to add roughly `30 GiB`.

### Post-pilot result summary

All five modes use seed `0`, result commit `22427d6`, and fixed background threshold `0.45`:

| Mode | Raw CAM mIoU | Semantic precision | Semantic recall | Background FPR | Mean mass variance | Latency at 448 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 69.50 | 81.85 | 83.87 | 6.04 | 0.004648 | 8.48 ms |
| split `(1,1)` | 58.22 | 61.72 | 76.58 | 16.36 | approximately 0 | 18.76 ms |
| split `(0.5,0.5)` | 66.31 | 82.33 | 78.22 | 5.38 | approximately 0 | 18.79 ms |
| TGCA | 69.05 | 78.73 | 86.40 | 7.65 | 0.000837 | 11.12 ms |
| TGCA + bias | 68.88 | 79.37 | 85.13 | 7.21 | 0.000837 | 12.35 ms |

TGCA reduces mean attention group-mass variance by approximately `82%`. Relative to vanilla, its all-layer patch-key mass slope magnitude falls by approximately `89%` for class queries and `75%` for patch queries. Thus the cardinality mechanism and correction are measurable.

The localization result does not pass the primary gain target. Count-only TGCA is `0.45` mIoU point below vanilla and TGCA+bias is `0.62` below. Both are below vanilla at every single-scale short-side resolution (`224`, `320`, `448`, `512`). TGCA raises recall but lowers precision and increases background false positives at the fixed threshold.

Cross-scale mask stability improves modestly at the largest resolution gap. For `224` versus `448`, paired image bootstrap analysis gives foreground-mask IoU differences of `+1.44` points for TGCA (95% CI `+0.81` to `+2.06`) and `+1.57` for TGCA+bias (`+0.99` to `+2.16`). At `512` versus `448`, the differences are small and the paired intervals include zero.

The current Python/group-mask implementation also does not yet satisfy the negligible-overhead gate: count-only TGCA latency is about `31%` above vanilla and TGCA+bias about `46%` above. Parameter count is unchanged for count-only TGCA; relation bias adds `288` parameters. FLOPs/MACs remain unmeasured.

Machine-readable comparison:

```text
results/mctformerplus/voc/comparisons/pilot-s0-22427d6/pilot_comparison.json
results/mctformerplus/voc/comparisons/pilot-s0-22427d6/pilot_comparison.csv
```

## How to evaluate the completed queue

For every mode, make a machine-readable comparison containing at least:

- raw CAM mIoU at the fixed vanilla threshold `0.45`;
- final and maximum classification mAP if recorded;
- checkpoint hash;
- maximum attention row-sum error, with the split `(1,1)` target explicitly recorded as two;
- directional group-mass slopes and their bootstrap intervals;
- mean/median group-mass variance over resolution;
- wall time, peak memory, parameters, and latency where available.

Then run these validity checks before drawing conclusions:

1. Vanilla in the integrated pipeline remains `69.50%`.
2. `split_05`, TGCA, and TGCA-bias rows sum to one numerically.
3. `split_11` rows sum to two and are never compared as if normalized.
4. TGCA directional mass is more stable than vanilla under real resolution changes.
5. TGCA retains evidence-driven group-mass variation rather than degenerating to the fixed `0.5/0.5` behavior of `split_05`.
6. Any CAM gain is not explained by a separately tuned background threshold, output rescaling, changed training schedule, or different checkpoint selection.
7. Runtime overhead is measured. The current Python/group-mask implementation may be inefficient and should not be treated as the final efficiency result without profiling and, if needed, a mathematically identical vectorized implementation.

Do not select a favorable method from only seed 0 and present it as final. After the pilot identifies viable modes, repeat the matched comparison with prespecified additional seeds and report variance.

## Immediate next sequence

1. Treat full TGCA as mechanism-positive but localization-negative on the seed-0 primary-host pilot; do not launch additional full-TGCA seeds yet.
2. Run the single prespecified contingency `tgca_gamma05` at seed `0`, fixed threshold `0.45`, with identical training and evaluation. Do not tune gamma beyond `{0, 0.5, 1}`.
3. Apply the same post-pilot attention, error, scale, and efficiency diagnostics to `tgca_gamma05`.
4. If partial correction still loses CAM quality, stop primary-host expansion and reassess or reject the WSSS benefit hypothesis rather than proceeding to KYAM or COCO.
5. If partial correction retains meaningful stability and recovers or improves localization, predeclare additional matched seeds before running them.
6. Optimize the mathematically identical count-correction implementation and measure FLOPs/MACs before claiming negligible overhead.
7. Reproduce vanilla Know Your Attention Maps only after the primary-host decision, then specify and ablate its singleton register-token grouping.
8. Use DiCLIP only as a transparent external comparison. Start COCO only if the VOC mechanism and independent-host gates pass.

## Risks that can invalidate or weaken the hypothesis

- **Trained logits may compensate for token count.** Synthetic replication invariance can be mathematically true while the learned network already counteracts cardinality in practice.
- **Resolution is confounded.** Changing resolution changes image evidence, receptive fields, interpolation, positional embeddings, and object scale—not just patch count. Synthetic replication and real-scale tests must be reported together.
- **Stability may not improve localization.** More stable attention mass is useful only if CAM quality, cross-scale consistency, or errors improve.
- **Fixed group mass may be sufficient.** If `split_05` matches TGCA, the evidence-driven mean-competition story becomes weaker and the contribution may reduce to a normalization heuristic.
- **Output-scale confounding.** `split_11` doubles row mass and therefore changes residual/output magnitude. Its poor CAM result cannot be attributed solely to group competition.
- **Threshold sensitivity.** A gain appearing only after per-method threshold tuning is weak. Fixed-threshold results are primary; threshold curves are diagnostic.
- **Implementation overhead.** A slow mask/one-hot implementation could invalidate the negligible-overhead claim even when an optimized equivalent is possible.
- **Relation bias may dominate.** If gains occur only with learned bias, distinguish count calibration from added capacity using TGCA without bias and parameter-matched controls.
- **Single-host overfitting.** A positive MCTformer+ result without a positive independent Know Your Attention Maps result is insufficient for the generality claim.
- **Register-token ambiguity.** In Know Your Attention Maps, treating the singleton register token as class, patch, or a separate group changes the method. All choices must be explicit and ablated.
- **Training instability or seed variance.** Seed-0 improvements can disappear. Confirm with multiple matched seeds after the pilot.
- **Metric cancellation.** Averaging complementary attention directions can falsely imply zero cardinality effect. Retain directional layer/head statistics.
- **Post-processing confounds.** Keep raw CAM evaluation primary and hold downstream CRF/segmentation pipelines fixed.
- **Unmatched comparisons.** DiCLIP and other recent systems may use different pretraining, backbones, supervision, or post-processing. Author-reported and reproduced results must be separated.

## Paper decision gates

Proceed to the full paper only if all are satisfied:

- the directional cardinality phenomenon is measurable;
- TGCA stabilizes group mass under patch-count/resolution changes;
- TGCA improves or stabilizes CAMs beyond simple output rescaling and fixed group mass;
- at least one independent host shows a positive effect;
- optimized overhead is negligible;
- all claims are matched for backbone, pretraining, supervision, and post-processing.

If TGCA improves invariance but consistently damages CAM quality, do not force a positive paper claim. Determine whether cardinality bias is actually a useful inductive bias for WSSS, whether partial correction `gamma < 1` is scientifically justified, or whether the hypothesis should be rejected.

## Provenance and reporting rules

Never report a result without:

- repository URL and commit;
- exact command and config;
- Conda/pip environment;
- dataset and list hashes;
- pretrained and trained checkpoint hashes;
- seed;
- raw machine-readable metrics;
- explicit threshold and post-processing;
- hardware and timing.

Generated metrics belong under `results/` as JSON/CSV in addition to paper tables. Use Git checkpoints before and after substantive code changes. Do not fabricate missing values, citations, repository behavior, or experimental outcomes.

## Documentation inventory

The complete documentation copied for this handoff is:

```text
docs/CHAT_HANDOFF.md
docs/MCTTA.pdf
docs/RESEARCH_PLAN_FULL.md
docs/TIP_REVIEWS.md
```

The server repository already contains the tracked file:

```text
docs/design.md
```

At the time of transfer, the local and server copies of `design.md` had matching SHA-256:

```text
4086ac6250aa7acd91d928f4f67b05d5139898dc6637792732d492d52671f5e0
```
