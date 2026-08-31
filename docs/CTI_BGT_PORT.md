# CTI BGT-only baseline on MCTformer+

Implemented on LHR in `/home/peng/code/TGCA`, branch
`research/mctformerplus-cti-bgt`, from `5a689926f86b6fe344b00631d86cd1f42c946842`.
This is an optional baseline, **not a full CTI reproduction or a new method**.
Default execution and old foreground-only checkpoints remain unchanged.

## Reference and mapping

Official source: [yoon307/CTI](https://github.com/yoon307/CTI/tree/1c6fdb4d14e6843e3d861ebd4580468e30598859),
read-only local submodule `hosts/CTI` at that same commit.
[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html).

| Official CTI code | This repository |
|---|---|
| `networks/mctformer.py`: `bg_token`, C+1 conv head, BG-first order | `models/mctformer_plus.py`: optional `bg_token`, `pos_embed_bg`, `num_class_tokens`, explicit FG/patch slices |
| `forward`: foreground-only `cls`/`pcls` | C-dimensional class logits, intermediate class regularizer and GWRP patch logits; no image-level BG label |
| `forward`: `mtatt`, `fcams`, `attn`, `rcams` | `models/cti_bgt.py::cti_bgt_maps` |
| `models/model_CTI.py::update`: masked FG union and `loss_bg` | `cti_bcam_loss`; added to `engine.py::train_one_epoch_mctplus` |
| `models/model_CTI.py::max_norm` | `cti_max_norm`, preserving ReLU/min/max/epsilon semantics |
| `train_trm.py`: `W[0]=0.1` | `--cti-bgt-weight 0.1` |

No intra-/cross-image infusion, token swapping, tokenizer MLP, positive-image
branch, memory bank, extra classification losses or CTI training schedule is
ported. Existing BCSS, PSL and non-vanilla attention combinations are rejected
when BGT is enabled, so this baseline isolates the requested mechanism.

## Mechanism and corrections to the earlier interpretation

Tokens are `[BG, FG_1, ..., FG_C, patch_1, ..., patch_P]`. BG participates in
all ordinary joint self-attention layers; it is not a read-only slot. The BG
parameter and its position embedding start at zero. Foreground token parameters
and positions retain the host's existing parameterization. CTI itself also adds
a tokenizer-generated component, so its *effective* initial BG embedding is not
simply zero; deliberately omitting that tokenizer is a host-preserving adaptation.

With attention averaged over heads, define

```text
A_cp = sum of final 6 layers, rows 0:C+1, columns C+1:C+1+P
A_pp = sum of layers 4:12 (zero-based), rows/columns C+1:C+1+P
X    = ReLU(patch_head) * A_cp
F    = A_pp @ max_c(X_c * image_label_c), c in foreground
B    = A_pp @ X_BG
L_bcam = mean(abs((1 - norm(F)) - norm(B)))
L_total = L_original_MCTformer+ + lambda_bgt * L_bcam
```

The union is formed **before** propagation. Both foreground and background,
including attention and affinity, receive gradients; no foreground detach is
used (official VOC behavior). `max_norm` is actually min–max normalization:

```text
z = ReLU(z)
norm(z) = ReLU(z - spatial_min(z) - 1e-5)
          / (spatial_max(z) - spatial_min(z) + 1e-5)
```

Constant maps normalize to zero. Empty-label/all-zero maps remain finite and
incur loss 1 when both normalized maps are zero; the implementation does not hide
this degeneracy. BGT is complement consistency, not per-patch class/BG softmax
competition. It requires no pixel-level labels.

The official VOC caller passes `swap_idx=3` even for the unswapped forward, which
restricts patch affinity to layers 4:12. This port makes that selection explicit
as `cti_bgt_affinity_start=4`, without introducing any swapping logic. Using 0
would instead match CTI's no-`swap_idx` forward; it is a separately recorded
configuration. The default class-to-patch window remains the final six layers.

Propagation and normalization use FP32 with autocast disabled, instead of CTI's
FP64 propagation. Tests compare values and gradients against an independent FP64
reference. CTI's COCO-specific foreground detach/warmup is not included; all class
counts use the documented VOC gradient policy. No COCO experiment was launched.

## Host behavior retained

Normal image classification remains foreground-only, including MCTformer+'s
intermediate token regularizer. GWRP is retained instead of CTI's average pool.
The standard CAM API still returns `[B,C,H,W]` using the original MCTformer+
last-three-layer mean, square root and all-layer patch-affinity recipe. Its
indexing excludes BG. The learned BG channel does **not** replace the existing
background threshold in pseudo-label generation.

CTI training maps use their own exact product without the host square root.
Inspect all C+1 refined maps, normalized FG union/BG, and BG attention through
`MCTformerPlusCam(..., cti_bgt=True)(images, active_labels=labels,
return_diagnostics=True)['cti_bgt']`. Without labels, diagnostics omit the FG
union; ordinary inference never needs ground-truth image labels for BGT.

## Usage

Use the existing `tgca-repro` environment and supported training entry point
`train_model_v2.py` (the legacy `train_model.py` entry point is not extended).

```bash
source /home/peng/anaconda3/etc/profile.d/conda.sh
conda activate tgca-repro
cd /home/peng/code/TGCA

# Append these options to the matched MCTformer+ training command:
#   --cti-bgt --cti-bgt-weight 0.1
# Architecture-only ablation: --cti-bgt --cti-bgt-weight 0
# Original baseline: omit --cti-bgt

python train_model_v2.py --model mctformerplus --dataset VOC12 \
  --cti-bgt --cti-bgt-weight 0.1 --input-size 448 \
  --train_list data/VOCdevkit/VOC2012/ImageLists/train_aug_id.txt \
  --val_list data/VOCdevkit/VOC2012/ImageLists/val_id.txt \
  --epochs 45 --batch_size 32 --seed 0 --lr 5e-4 --min-lr 1e-5 \
  --work_space results/mctformerplus/voc/CHOOSE_A_NEW_RUN_ID

python make_cam.py --model mctformerplus --dataset VOC12 \
  --cti-bgt --cti-bgt-weight 0.1 --input_size 448 --scales 1.0,0.75,1.25 \
  --train_list data/VOCdevkit/VOC2012/ImageLists/train_id.txt \
  --work_space results/mctformerplus/voc/CHOOSE_A_NEW_RUN_ID \
  --checkpoint results/mctformerplus/voc/CHOOSE_A_NEW_RUN_ID/mctformerplus_final.pth

OMP_NUM_THREADS=1 python -m pytest -q tests
```

These training commands are examples, **not executed experiments**. Choose an
unused run directory, checkpoint the intended code and record normal manifests
before a research run. Existing runner cleanliness/overwrite guards are unchanged.
`--cti-bgt-n-layers` (6) and `--cti-bgt-affinity-start` (4) must match training
when exporting CAMs, as must BGT enablement/weight metadata. CLI/checkpoint
mismatches fail explicitly, and CAM state loading remains strict.

`--finetune` with BGT supports DeiT URL/local checkpoints and foreground-only
MCTformer+ checkpoints. Foreground host head rows move to indices 1:C+1 without
being discarded; the new BG row retains its initialization. BG checkpoints keep
all BG parameters. This initializes weights only, not optimizer state/resumption.

## Verification and diff summary

The original suite passed 62 tests before editing. The final suite passed **99 tests** (including CUDA FP32/FP16/BF16);
13 warnings come from existing matplotlib/pyparsing dependencies. Regression compares against
the immutable pre-port source at `5a68992`: default parameters, outputs, gradients
and CAMs must match exactly. Other tests cover BG-first indexing, C=1/20/80,
rectangular/multi-scale CAMs, strict checkpoint loading, foreground-only
supervision, masked union order, official normalization, FP64-reference loss and
gradients, zero/constant/empty maps, loss weight zero, the real training loop,
invalid combinations, and FP32/FP16/BF16 CUDA backward.

Changed existing files: `models/mctformer_plus.py` (optional token/head and slicing),
`engine.py` (weighted loss/log), `train_model_v2.py` (flags, initialization,
checkpoint metadata), `make_cam.py` and `utils.py` (flags and safe CAM loading),
`docs/CHAT_HANDOFF.md` (operational note). New files: `models/cti_bgt.py`,
`tests/test_cti_bgt.py`, and this note. The official CTI submodule is unchanged.

Pre-existing user changes are preserved: five deleted historical documents
(`BCSS_EXPERIMENTS`, `PERSISTENT_SEMANTIC_PHASE01`, `PERSISTENT_SEMANTIC_PHASE2`,
`Persistent_Semantic_Latent_Codex_Plan`, `RESEARCH_PLAN_FULL`) and the untracked
`docs/survey_diffusion.md`, `docs/validation_diffusion.md`. The new branch started
at the existing HEAD; these unrelated changes were not stashed, restored, or
committed. No commit, push or full training run is part of this port.


### Full-size verification record

- Environment: `tgca-repro`, torch 2.1.0, timm 0.4.12, CUDA 11.8, RTX A6000.
- Two real VOC training images at 448, DeiT initialization, seed 0; eight
  repeated-batch AMP attempts, **seven successful optimizer updates**.
- First attempt triggered global GradScaler overflow detection and was skipped;
  scale decreased from 65536 to 32768. The initial one-attempt smoke had failed
  its nonzero-BG-gradient assertion (the clipped BG gradient was zero, not proof
  of a persistent nonfinite BG gradient). Seven subsequent BG gradients were
  finite and nonzero. The original scaler and training code were not changed.
- Strict training-to-CAM loading passed. Inputs 224², 448², 320x448 and 512²
  produced finite C-channel host CAMs and C+1 CTI diagnostic maps.
- The actual `make_cam.py` training-split CLI exported 20 nonempty, finite `.npy` files at scales
  `1.0,0.75,1.25` from the smoke checkpoint with BGT enabled.
- The existing training-CAM progress logger divides by `len(dataset)//20`;
  a two-image training-list check hit this pre-existing divide-by-zero. It is
  unchanged to preserve host scope; use at least 20 images per GPU for that
  legacy CLI. The earlier test-list check returned empty dictionaries because
  the nearly untrained classifier predicted no classes; no localization claim
  is made from it. Both preliminary logs are retained.
- This is mechanical verification only: no full training, mIoU evaluation,
  threshold selection, or scientific performance claim.

Artifacts (relative to repository root):

```text
results/cti_bgt/validation/20260831-120656-bgt-port/pytest.log
results/cti_bgt/validation/20260831-120656-bgt-port/smoke.log  # initial failed check
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/smoke.py
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/command.txt
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/smoke.json
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/train20_cam_export.log
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/implementation.patch
results/cti_bgt/validation/20260831-120656-bgt-port-amp-check/DIFF_SUMMARY.md
```


## Authorized full VOC validation (2026-08-31)

After the port, the user explicitly requested commit, push, then full BGT
validation. Runner: `experiments/baselines/run_cti_bgt_voc.sh`.
It requires a clean checkout, the exact pushed commit and a new run directory;
it repeats the tests, trains 45 epochs at 448/batch 32/seed 0, evaluates image
classification each epoch on 1449 validation images, exports 1464 train-split
CAMs at scales 1.0/0.75/1.25, and evaluates at fixed threshold 0.45 without CRF.
It writes JSON/CSV mIoU, foreground precision/recall, background FPR and a matched
comparison against completed E0 run
`20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3` (no baseline retraining).
This is one complete seed-0 VOC screen, not multi-seed, COCO or downstream
segmentation training. BGT weight remains 0.1; no new tuning is introduced.

The parent checkout's unrelated document changes remain untouched. Execution
uses a linked clean worktree inside `results/cti_bgt/checkouts/` in the same
LHR repository, with output directed to the parent's `results/mctformerplus/voc/`.
All launches use tmux and record source commit, environment, datasets, pretrained
and trained checkpoint hashes, commands and stage/exit/completion markers.
The user's explicit push authorization covers the implementation and this runner.
