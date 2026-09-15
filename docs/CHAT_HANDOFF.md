# TGCA Operational Handoff

## 2026-09-15 COCO smoke metadata fix, preserved first attempt

Initial code `85b1d06`, run `results/c2p_pooling/20260915-coco-all-product-s0`,
passed63 tests, smoke training/nativeCAM, and exact online/offline confusion
parity at all60 thresholds. It then failed AFTER COCO classification inference,
when provenance hashing used the VOC filename ImageLabel/cls_labels.npy.
The correct file is ImageLabel/COCO_cls_labels.npy. No full training started.
Initial queue is stopped with QUEUE_FAILED; all files/logs are preserved.
Fixed only dataset-aware provenance path and added an end-to-end80-class
classification test that writes metrics/metadata to catch this failure.
Retry uses a fresh root `results/c2p_pooling/20260915-coco-all-product-s0-r2`
and tmux `mct-c2p-coco-all-product-20260915-r2`; no source files overwritten.

## 2026-09-15 COCO all-product setup

User requested one COCO all-product run, without affinity. Plan:
`docs/MCTformerPlus_C2P_All_Product_COCO.md`. Model computation unchanged;
reuse existing all12 product helper with 80 classes. Match completed COCO GWRP
baseline at `results/default_mctformerplus/20260912-voc-coco-s0-r2/COCO`:
seed0, fresh DeiT-S, 45 epochs, 448, batch32, original optimizer/loss/augmentation.
Necessary tests: 63 passed, including80-class AMP/backward/CAM and online/offline
CAM evaluator equality. Environment tgca-repro; RTX A6000 idle,48505MiB free.
Disk76GiB free versus baseline CAM dump249GiB. Do NOT delete source data;
new full CAM pipeline accumulates confusion online with the SAME native generator.
Smoke must confirm exact all60-threshold confusion parity against disk output.
COCO classification evaluator now supports80 labels with same deterministic448
transform; both frozen baseline and new macro-class AP measured under new root.
Queue: tests -> smoke train/CAM parity/classification -> full train/nativeCAM ->
classification/control comparison -> compact report. No refinement/segmentation.
Command: `/home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.ablations.run_c2p_coco --output results/c2p_pooling/20260915-coco-all-product-s0`.
At this setup checkpoint, queue not yet launched. Preserve clean checkout gate.

## 2026-09-15 C2P affinity completion and comparison verified

Queue completed 2026-09-15 02:29:52 JST; both full variants completed 45 epochs,
1464 native CAMs, 1449 classification images and 1449-image before/after weight
diagnostics (six PNG/PDF examples each). QUEUE_COMPLETE/VARIANT_COMPLETE exist,
no QUEUE_FAILED; experiment tmux ended and GPU is idle. Code SHA `04aa6b8`.
Read-only audit verified all 32 recorded source hashes unchanged, both new
checkpoint hashes, optimizer/pretraining equality and model-spec difference
limited to affinity=true. Test log: 66 passed. Results were not regenerated.
Last3 checkpoint: `0cdb5a93faf201b2b0b5887f23148f3ffcdacb7891218a2ca85b426c209f4847`.
All checkpoint: `431a0574282fbbfb3f611de3a05215881cb7a9abe29895f07e107ce48d8374e2`.

Compared with corresponding product without affinity:
- Last3 class/patch macro AP 93.2467/93.3587% (+.0262/+.0367 pp),
  fixed .45 CAM 70.5312% (-1.1341 pp), best 70.5312% at .45;
  fixed FG P/R 81.9028/84.0339% (-.9776/+.4840 pp).
- All class/patch macro AP 93.1141/93.1688% (+.6721/+.3048 pp),
  fixed .45 CAM 69.4062% (-2.9092 pp), best 69.7004% at .48;
  fixed FG P/R 79.8355/85.6549% (-3.4826/+1.0005 pp).
Thus both new runs worsen CAM despite comparable/improved classification.
Best threshold does not remove the deficit. Existing all-product remains
highest observed fixed CAM (72.3154%) among the seven seed0 runs.

Within each NEW checkpoint, before -> after affinity readout (all val images,
equal image weights): entropy .6020 -> .9776 (last3), .2938 -> .9719 (all);
top1 mass .1341 -> .00656 and .4651 -> .01292. Positive-pair top10% Jaccard
on 522 multi-label images decreases .2763 -> .1012 and .2858 -> .1238.
Hence substantial spatial flattening, NOT increased top-support overlap.
This is not a before/after-training comparison and does not prove semantic
background leakage or a causal explanation for CAM degradation. Single seed;
5000 paired-image readout bootstrap does not measure training-seed uncertainty.
Compact sources: `results/c2p_pooling/20260914-voc-product-affinity-s0/comparison.csv`,
`C2P_AFFINITY_REPORT.md`, each variant's `weight_diagnostics/summary.csv`.
No experiments launched/restarted or source results changed in this status check.

## 2026-09-14 22:28 JST C2P affinity queue active

Implementation SHA: `04aa6b870f14a807c5ffdb37c273d218cd5b9414` (local only).
tmux: `mct-c2p-affinity-voc-20260914`, launched 22:26:57 JST.
Run root: `results/c2p_pooling/20260914-voc-product-affinity-s0`.
Queue log: `results/c2p_pooling/20260914-voc-product-affinity-s0.queue.log`.
Runner's clean-checkout test repeat: **66 passed**, recorded in `tests.log`.
Both `last3_product_affinity_smoke` and `all_product_affinity_smoke` finished
training, checkpoint save/load, native CAM evaluation, classification and readout diagnostics.
Smoke model specs match corresponding prior product runs except affinity=true.
Full `last3_product_affinity` started at 22:27:38 JST; observed epoch0 step80/330,
finite loss decreasing (running loss 4.4596), GPU peak allocation ~19065 MiB.
Its optimizer and pretrained-load reports exactly match prior last3-product.
`all_product_affinity` is queued after last3 train/eval/readout completion, not running yet.
Do not restart/duplicate. The queue automatically saves classification/CAM results,
readout bootstrap/plots, source integrity audit and C2P_AFFINITY_REPORT.md.
Full results are **pending**, not completed; check QUEUE_COMPLETE/QUEUE_FAILED and
per-variant VARIANT_COMPLETE before reporting final values. Current handoff-only
commit does not change the implementation SHA recorded by the runner.

## 2026-09-14 C2P product + P2P affinity approved and implemented

User approved exactly two runs from `docs/MCTformerPlus_C2P_Product_P2P_Outline.md`,
section 3: propagate product C2P weights with all-12-layer head-mean P2P, normalize,
then weight raw patch logits. The alternative propagate-weighted-logits is NOT selected.
New `--c2p-pooling-affinity` defaults off; requires c2p/product. All existing
initialization modules, class/CCT branches, patch classifier and native CAM stay unchanged.
Training and frozen classification/gating share the same differentiable helper.
Necessary tests: 66 passed in tgca-repro (including CUDA AMP); runner repeats and logs them.
Command: `/home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.ablations.run_c2p_affinity --output results/c2p_pooling/20260914-voc-product-affinity-s0`.
Queue will run both smokes (training/checkpoint/CAM/classification/readout audit), then
last3-product-affinity full train/eval/audit, then all-product-affinity full train/eval/audit.
Matched seed0/45 epochs/448/batch32/DeiT-S recipe. Five existing results read-only;
source checkpoint/result hashes rechecked, matched model/optimizer/pretrain metadata audited.
Compact outputs: comparison.csv, C2P_AFFINITY_REPORT.md, per-variant classification/,
raw_cam/, weight_diagnostics/ (image-level CSV, 5000 paired bootstrap, six prespecified examples).
Preflight: RTX A6000 idle, 48505 MiB free, disk 80 GB free; no active experiment session.
At this implementation checkpoint the queue has NOT yet been launched.

## 2026-09-14 C2P product pooling completion verified

Both product variants and the queue completed at 2026-09-14 21:40:50 JST.
Run: `results/c2p_pooling/20260914-voc-product-s0`; code SHA `d7090fe`.
QUEUE_COMPLETE and both VARIANT_COMPLETE markers exist; no failure marker or
active experiment tmux/GPU process. Read-only audit verified 45 epochs per
variant, exactly 1464 CAM files and 1449 classification rows per variant,
matching optimizer/pretraining reports, unchanged original source hashes,
and both new final checkpoint hashes. Tests: 59 passed.
Final SHA256 last3_product:
`b2f125afc9b750f04979ab1065f9e712c7979d667d3fe79ec79c18abcb368fd2`;
all_product:
`ce7bde924fe6a1780151db6e4541eb8937b5f595078483cbbae222ba14f9ef0b`.

Last3-product: class/patch macro AP 93.2205/93.3221%; fixed .45 CAM mIoU
71.6654%, best 71.6700% at .44; fixed FG P/R 82.8805/83.5499%.
All-product: class/patch macro AP 92.4420/92.8639%; fixed .45 CAM mIoU
72.3154%, also best-grid at .45; fixed FG P/R 83.3181/84.6544%.
Compared to corresponding means, fixed CAM changes are -0.3744 pp (last3)
and +0.2315 pp (all). All-product lowers class/patch AP by .6775/.6255 pp
versus all-mean while increasing recall by 1.0912 pp and reducing precision
by .6281 pp. No uniform product benefit; all-product has highest observed
fixed CAM among the five runs, but the small extra gain is single-seed only.
Do not infer a semantic or causal mechanism from these metrics alone.
Compact outputs: `comparison.csv`, `C2P_PRODUCT_REPORT.md`; each variant has
`commands.sh`, `classification/`, `raw_cam/` and checkpoint_sha256.txt.
No additional experiments were launched during this completion check.

## 2026-09-14 C2P product pooling two-run queue active

User requested element-wise multiplication across layers instead of mean for
both last3 and all C2P patch pooling. Plan: `docs/MCTformerPlus_C2P_Product.md`.
New CLI: `--c2p-pooling-reduction product` with `--patch-pooling c2p` and
`--c2p-pooling-layers last3|all`. Default mean preserves previous checkpoints.
Heads are averaged within each layer; layers are multiplied pointwise and
normalized over patches. Stable implementation uses FP32 softmax(sum(log(a_l)))
with machine-tiny log floor; no geometric root/temperature/detach/new loss.
Raw patch logits and native CAM formulas are unchanged (CAM still last3 mean).
Class initialization and other training settings are unchanged.

59 relevant tests passed, including FP64 direct-product forward/backward
equivalence, underflow/normalization, all selected layer gradients, finite AMP,
mean/GWRP compatibility, checkpoint config and native CAM/gating parity.
Runner:
`python -m experiments.ablations.run_c2p_product --output results/c2p_pooling/20260914-voc-product-s0`.
Active tmux: `mct-c2p-product-voc-20260914`.
Queue log: `results/c2p_pooling/20260914-voc-product-s0.queue.log`.
Both last3/all product smokes precede full last3_product -> all_product,
sequentially including classification and CAM evaluation for each.
Each is a fresh matched seed0 DeiT-S / VOC / 45 epochs / 448 / batch32 run.
Existing GWRP and both mean controls are reused read-only, not retrained.
Root comparison.csv/C2P_PRODUCT_REPORT.md update after each completed variant;
QUEUE_COMPLETE means both runs finished. Check live state before any action;
never duplicate a running queue. Existing source results/checkpoints are immutable.

Verified 2026-09-14 18:44 JST: runner repeated 59 passing tests. Both product
smokes (64 images/two batches, 448/batch32) completed training, strict checkpoint
load, native CAM/raw evaluation and classification. Smoke checkpoint hashes
match; layer/reduction metadata is correct and CAM remains last3 mean.
Full last3_product is active at epoch0 past batch40/330, with finite decreasing
losses; all_product full training is queued, NOT yet started. Code SHA `d7090fe`
(full SHA in manifest). Optimizer specs and pretraining load reports already
match all three existing controls exactly. Full comparison metrics are pending.
Active log: `results/c2p_pooling/20260914-voc-product-s0/last3_product/train.log`.
Later log: `results/c2p_pooling/20260914-voc-product-s0/all_product/train.log`.
The original GradScaler is unchanged; the historical optimizer_updates log
counts attempted batch boundaries, not necessarily non-skipped AMP steps.

## 2026-09-14 C2P all-layer pooling completion verified

The all-layer queue completed at 2026-09-14 18:07:40 JST. All 45 epochs,
native CAM generation/evaluation and two-head classification completed;
QUEUE_COMPLETE and all stage markers exist, with no QUEUE_FAILED or active
experiment tmux/GPU process. Execution code SHA `5b2c618`.
Run: `results/c2p_pooling/20260914-voc-all-layers-s0`.

Read-only completion audit: 49 tests passed; exactly 1464 CAM image files;
1449 classification evaluation rows; final checkpoint SHA256 matches
`fb8df1df2fa051b0158391df1758dcc9702168354ac0ce0b6e288ac909ae026e`.
All source hashes still match the before/after manifest. Optimizer specs and
pretraining load reports match both GWRP and C2P-last3 exactly.

GWRP / C2P-last3 / C2P-all:
class macro AP 92.9063 / 93.0787 / 93.1194%; patch macro AP
93.2577 / 93.3819 / 93.4894%. Raw train CAM mIoU at fixed .45:
70.0631 / 72.0397 / 72.0840%; best-grid 70.3397 / 72.1650 / 72.1024%
at thresholds .48 / .43 / .44. All vs last3 fixed mIoU +0.0443 pp,
best-grid -0.0626 pp: essentially similar point estimates, not evidence of
an additional robust gain. This is one seed; no equivalence/significance
claim is justified. All vs last3 FG precision -0.1785 pp, recall +0.4225 pp.
Reports: `C2P_ALL_LAYERS_REPORT.md`, `comparison.csv`; commands under
`commands.sh` and `VOC12/commands.sh`. No new runs launched by this status check.

## 2026-09-14 C2P all-layer pooling experiment active

User requested changing actual pooling from last-three to all-layer A_c2p.
Plan: `docs/MCTformerPlus_C2P_All_Layers.md`.
CLI: `--patch-pooling c2p --c2p-pooling-layers all`.
Raw attention is averaged over all 12 layers and 6 heads BEFORE patch-key
normalization. Existing last3 default/checkpoints are preserved. No initializer,
class/CCT branch, loss, optimizer, schedule or CAM formula changes. Native CAM
still uses last3 C2P, sqrt, all-layer P2P. No additional variants or COCO run.
Tests: 49 passed, including all-layer formula/gradients, original last3/GWRP,
same-weight CAM parity, checkpoint selection and finite CUDA AMP backward.

Runner:
`python -m experiments.ablations.run_c2p_all_layers --output results/c2p_pooling/20260914-voc-all-layers-s0`.
Active tmux: `mct-c2p-all-layers-voc-20260914`.
Queue log: `results/c2p_pooling/20260914-voc-all-layers-s0.queue.log`.
Tests -> smoke train/save/load/CAM/classification -> full 45-epoch seed-0 VOC
training -> native CAM evaluation -> two-head classification -> report.
Same 448/batch32/DeiT-S/AdamW/cosine/nominal5e-4/min1e-5 recipe as C2P-last3.
Existing GWRP and last3 results are reused read-only, not retrained.
Final files: `comparison.csv` (three methods), `C2P_ALL_LAYERS_REPORT.md`;
commands/config/environment/Git SHA/source hashes and stage logs in the run root.
Check live tmux/logs and QUEUE_COMPLETE/QUEUE_FAILED before taking any action.

Verified live 2026-09-14 16:35 JST: 49 runner tests passed; smoke train (2
updates at actual 448/batch32), checkpoint save/load, native three-scale CAM,
raw CAM evaluation and two-head classification all completed. Full training
started at 16:35:08, epoch 0 past update 90/330 with finite, decreasing losses.
Execution SHA `5b2c618` (full SHA in manifest); training max allocated memory
16,675 MiB so far. Full metrics are pending, not completed evidence.
Read-only three-way comparison confirmed optimizer_spec and pretraining load
reports exactly match GWRP and C2P-last3. Only model-spec difference versus
last3 is c2p_pooling_layers=all. Source checkpoints/results are untouched.
Training log: `results/c2p_pooling/20260914-voc-all-layers-s0/VOC12/train.log`.
Do not duplicate or restart the running job. Queue automatically evaluates and
writes the three-way comparison/report after full training.

## 2026-09-14 paired layer-wise GWRP/C2P attention audit completed

Read-only frozen FP32 evaluation: all 1,449 VOC val images, deterministic
resize-short-side 512 / center-crop 448, actual 6-head-mean A_c2p for L1-L12.
522 multi-label images (375 two-label, 147 three-plus); class-pair statistics
averaged within image, then equally over images. 5,000 paired image bootstrap.
Output: `results/c2p_pooling_attention/20260914-voc-val-s0` (`COMPLETE`).
Code commits: extraction/tests `469d2d5`, native-last3 postprocessing `020e817`.
Tests: 5 passed. Both strict loads passed; reconstructed native-last3 spatial
weights agree with the model helper (max absolute errors 2.98e-8 / 5.96e-8).
Both source checkpoints and audited root metadata hashes unchanged. No model
or training edits, no segmentation GT, no running analysis tmux left at completion.

Main finding: sharing varies by layer, not a universal collapse. C2P has lower
top10%-support Jaccard/Pearson at L10/L11 but higher at L12. L12 pair top1
coincidence 30.98% -> 36.25%, top10% Jaccard .419 -> .551. All-positive-classes
same top1 occurs on 26.82% -> 31.80% of multi-label images (not all images).
Native-last3 must not be equated with L12: pair top1 8.30% -> 5.96% (paired CI
includes zero), top10% Jaccard .255 -> .275, Pearson .450 -> .409.
No semantic ownership or causal CAM-improvement claim is established.
Reports: `ATTENTION_COMPARISON_REPORT.md`, `FIGURES.md`; six label-prespecified
paired image heatmaps with all 12 layers, both raw and conditional versions.
`layer_summary.csv`, `native_last3_summary.csv`, per-image metrics and manifests
are compact. Positive-only raw NPZ maps are optional and excluded from download bundle.

## 2026-09-14 C2P pooling completion verified

The C2P pooling queue finished at 2026-09-14 12:03:11 JST. `QUEUE_COMPLETE`
exists under `results/c2p_pooling/20260914-voc-s0`, with no failure marker or
active experiment tmux/GPU job. Both final checkpoints' SHA256 hashes match;
all 1,464 C2P CAM files exist without missing/extra IDs. Optimizer specs and
pretraining load reports match the GWRP baseline. Both classification evaluations
cover the same 1,449 VOC val images and retain `class_token_init=baseline`.
Tests: 43 passed; execution SHA `18c490e`.

GWRP -> C2P: class macro AP 92.9063 -> 93.0787%; patch macro AP 93.2577 ->
93.3819%; class val loss .0506326 -> .0495640; patch val loss .0497795 ->
.0489571. Raw train CAM mIoU at fixed .45: 70.0631 -> 72.0397% (+1.9767 pp).
Best grid mIoU 70.3397% at .48 -> 72.1650% at .43 (+1.8253 pp); best-grid
values are diagnostic, not independently selected thresholds. Fixed-threshold
semantic FG precision 80.7353 -> 84.1247%, recall 85.8169 -> 83.1407%.
This is a single-seed result, not evidence of statistical robustness or a
specific semantic mechanism. No additional variants were launched.
Compact output: `C2P_POOLING_REPORT.md`, `comparison.csv`; exact training
command: `VOC12/commands.sh`, all relative to the result root above.

## 2026-09-14 C2P patch pooling experiment

New user task: ONLY replace original patch GWRP pooling with last-three-layer,
all-head actual Transformer C2P spatial attention pooling, preserving gradients
and raw 3x3 classifier logits. No changes to class-token initialization code or
configuration. Current matched default remains `class_token_init=baseline`;
CWP/residual CWP options and parameters are untouched and tested for identical
initialization. CLI: `--patch-pooling gwrp|c2p` (default gwrp).

Plan: `docs/MCTformerPlus_C2P_Pooling.md`.
Runner: `python -m experiments.ablations.run_c2p_pooling --output results/c2p_pooling/20260914-voc-s0`.
Active tmux: `mct-c2p-pooling-voc-20260914`.
Verified 2026-09-14 10:39 JST: smoke train/checkpoint/CAM/classification passed;
full c2p VOC epoch 0 is running with finite losses. Execution SHA `18c490e`.
Queue log: `results/c2p_pooling/20260914-voc-s0.queue.log`.
Full training log: `results/c2p_pooling/20260914-voc-s0/VOC12/train.log`.
Read-only comparison confirmed identical optimizer specs and pretraining load
reports versus the GWRP baseline. New `patch_first=False` metadata merely
records the unchanged default; only `patch_pooling=c2p` changes computation.
Frozen GWRP baseline classification completed: macro class-token AP 92.9063%,
patch-head AP 93.2577%, class loss .0506326, patch loss .0497795. Its historical
mean-image AP is 96.4100% (class) / 96.7885% (patch), a different metric.
Full c2p classification and CAM results are pending, not completed evidence.
Tests passed: 43 (C2P formula/GAP/gradient/GWRP parity/initializer parity,
checkpoint/CAM, AMP, existing variants and evaluators). No model weights or
source results overwritten. Baseline is the completed ordinary VOC GWRP run
under `results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12`, NOT the
patch-first run. Only c2p is trained, using the identical 45-epoch seed-0 recipe.
Order: tests -> smoke train/CAM/classification -> frozen baseline classification
-> full c2p train/CAM -> frozen c2p classification -> compact comparison/report.
No new COCO run, refinement or segmentation. Results pending at setup.
Final outputs: `comparison.csv`, `C2P_POOLING_REPORT.md`; exact stage commands,
tests, environment, config and checkpoint hashes are saved under the new root.
Both heads' classification is measured with the same frozen FP32 448 evaluator;
training's historical mean-image AP must not be labeled macro-class AP.
Do not duplicate a live tmux/output; inspect `QUEUE_COMPLETE`/`QUEUE_FAILED`.

## 2026-09-14 patch-first completion verified

The patch-first VOC queue completed at 2026-09-13 21:44:35 JST.
`results/patch_first/20260913-voc-s0/QUEUE_COMPLETE` exists; no failure marker
or active experiment tmux/GPU process. Read-only verification confirmed all
1,464 CAM files (no missing/extra), final checkpoint SHA256, and optimizer
configuration equality against the default VOC baseline. Tests: 32 passed.
Final validation class-token mAP 96.426%; raw train CAM mIoU at .45 68.745%
versus baseline 70.063% (-1.318 pp). Best-grid mIoU 68.901% at .47 versus
baseline 70.340% at .48 (-1.439 pp). This single seed shows no improvement;
it does not establish a structural order effect for a permutation-equivalent
model. Code SHA `638394d`; reports: `PATCH_FIRST_REPORT.md`, `comparison.csv`
under the run root. No COCO patch-first run was launched. Do not restart.

## 2026-09-13 patch-first order ablation

User requested exchanging concatenation order only, keeping everything else
unchanged. Plan: `docs/MCTformerPlus_Patch_First_Ablation.md`.
Entry point: `python -m experiments.ablations.run_patch_first --output results/patch_first/20260913-voc-s0`.
Active tmux: `mct-patch-first-voc-20260913`.
Verified 2026-09-13 20:22 JST: 32 tests passed, smoke training/CAM/evaluation
passed, full VOC epoch 0 is running with finite losses. Code SHA `638394d`.
Queue log: `results/patch_first/20260913-voc-s0.queue.log`.
Training log: `results/patch_first/20260913-voc-s0/VOC12/train.log`.
One VOC seed-0 run is active (COCO is not queued); exact completed default VOC
baseline settings, 45 epochs, 448, batch 32, DeiT-S, nominal LR 5e-4/min 1e-5.
Only new argument: `--patch-first`. All blocks see [patch,class]; CCT takes raw
tail class tokens. Semantic readout and CAM attention indices are restored
outside the blocks; original head/loss/CAM formulas are unchanged. Positions
follow tokens, so this is a permutation-equivalent control in exact arithmetic.
Tests precede smoke; smoke precedes full training and original raw CAM eval.
Queue never overwrites existing files. Do not relaunch if tmux/output exists.
Output `manifest.json` records actual code SHA; `commands.sh` and stage logs
reside in each run. Results will be `comparison.csv` and `PATCH_FIRST_REPORT.md`.
The unrelated existing results and checkpoints remain untouched.

## 2026-09-13 completion verified

The fresh default VOC / COCO queue below completed at 2026-09-13 13:11:42
JST (`QUEUE_COMPLETE`). Both 45-epoch runs, native CAM generation and raw CAM
evaluation have completed. No experiment tmux or GPU process remains active.
Read-only verification at 20:14 JST confirmed checkpoint SHA256 matches for
both datasets and exact CAM coverage: VOC 1,464 / COCO 82,783, no missing or
extra image files. Tests: 29 passed. Execution SHA remains `eba5069`.

Raw CAM train-split mIoU: VOC fixed 0.45 = 70.063%, best 0.48 = 70.340%;
COCO fixed 0.45 = 42.667%, best 0.44 = 42.688%. These are native three-scale
CAMs with image-level label gating, not validation-split or segmentation-model
results. No CRF or downstream segmentation was run. Compact results are in
`results/default_mctformerplus/20260912-voc-coco-s0-r2/RAW_CAM_REPORT.md` and
`cam_summary.csv`; each dataset has `raw_cam/metrics.json` and the complete
`raw_cam/threshold_curve.csv`. Do not restart the completed queue.

## 2026-09-12 fresh default VOC / COCO task

The user explicitly requested fresh default MCTformer+ training on both VOC
and COCO, stopping at raw CAM evaluation. This is NOT a restart of the stopped
decoupled queue below. Live main started at `f120611`; the GPU driver mismatch
seen earlier is resolved (580.178.04, CUDA available in `tgca-repro`).

New queue entry point: `python -m experiments.baselines.run_default_voc_coco`.
Current tmux: `mct-default-voc-coco-20260912-r2`.
Output: `results/default_mctformerplus/20260912-voc-coco-s0-r2`.
Verified live at 2026-09-12 17:19 JST: both smoke training/CAM/evaluation
pipelines passed; full VOC epoch 0 is active with finite losses and normal GPU
utilization. COCO full training is pending behind VOC training + raw CAM eval.
Execution code SHA: `eba5069` (full SHA in output `manifest.json`).
Queue log: `results/default_mctformerplus/20260912-voc-coco-s0-r2.queue.log`.
Training log: `<output>/VOC12/train.log`; later `<output>/COCO/train.log`.
The first attempt (same path without `-r2`, code `f759165`) passed both
smoke trainings and VOC CAM evaluation, then stopped at COCO CAM checkpoint
resolution: an old validator hardcoded 20 class tokens. Its logs/checkpoints
remain untouched. The follow-up fix passes the actual dataset class count
into that validator; it does not change model computation or training.
Launch only once; inspect this directory and tmux before doing anything else.
Both dataset smokes precede full VOC, then full COCO. Training uses the existing
MCTformer+ baseline recipe (not legacy `mcta` COCO script): Small, vanilla joint
attention, no experimental flags, 448, seed 0, batch 32, 45 epochs, nominal LR
5e-4, min LR 1e-5, AdamW/cosine and remaining original settings. Native LR batch
scaling is unchanged. Initialization is the DeiT-S file recorded below.
Training lists: VOC train_aug 10,582 / val 1,449; COCO train2014 82,783 /
val2014 40,504. All listed images/labels and CAM evaluation masks passed a
read-only existence audit. Raw CAM evaluation: final checkpoint, train split
(VOC 1,464; COCO 82,783), native scales 1,.75,1.25 and image-level label gating.
Threshold grid 0.00:0.01:0.59, fixed 0.45; global confusion accumulation includes
empty CAM predictions. No CRF/refinement/segmentation stages.

Tests: `python -m pytest -q tests/test_raw_cam_streaming.py tests/test_width_scaling_aggregation.py tests/test_mctformerplus_variants.py`
passed after the class-count fix (29 tests).
Queue repeats and saves these tests. Exact commands, environment
manifests, dataset hashes, code SHA, optimizer specs, logs and checkpoint hashes
are written under the new output. Final compact files: `cam_summary.csv`,
`RAW_CAM_REPORT.md`, and each dataset's `raw_cam/metrics.json` and
`raw_cam/threshold_curve.csv`. `QUEUE_COMPLETE` means both full runs finished;
`QUEUE_FAILED` means inspect logs. These full results do not exist yet at setup.

The historical state below is retained for provenance and is superseded by this
section for the current task.

Updated: 2026-09-09 21:13 JST

This file was absent at the start of the current task and was created to restore
the operational handoff required by `AGENTS.md`. It records the live server
state for the current experiment; older completed studies remain immutable
under `results/` and in Git history.

## Repository and environment

- Repository: `/home/peng/code/TGCA`
- Branch: `main`
- Experiment code commit: `b1b4830e3d4eaaf5356425de906f8d0a1664b708`
- Conda environment: `tgca-repro`
- Device: NVIDIA RTX A6000, GPU 0
- Dataset: `/home/peng/code/TGCA/data/VOCdevkit/VOC2012`
- Official DeiT-S initialization:
  `/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth`
- DeiT-S SHA256:
  `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f`

## Stopped experiment queue

- Former tmux session: `mct-decoupled-q-20260909` (stopped and absent)
- Run ID: `20260909-mctformerplus-decoupled-s0-b1b4830`
- Result root:
  `/home/peng/code/TGCA/results/decoupled_bidirectional/20260909-mctformerplus-decoupled-s0-b1b4830`
- Runner: `experiments/ablations/run_mctformerplus_decoupled_queue.sh`
- Pipeline log: `<result root>/pipeline.log`
- Exact commands: `<result root>/exact_commands.sh`

The 100-update, batch-32, 448 smoke passed with finite class/CCT/patch
losses, strict checkpoint audit, and four completed single-scale CAMs. At the
user's request, the complete queue was stopped at 2026-09-09 21:13 JST. No
queue, training, evaluation, or CAM process remains active on the GPU. Do not
restart or duplicate this queue without a new user request.

The serialized queue order is:

1. `full`
2. `no_p2c`
3. `no_c2c`
4. `no_p2c_no_c2c`
5. `c2p_update_off`
6. `p2p_update_off`
7. `p2c_middle`
8. `p2c_early`

The first four items completed 45-epoch seed-0 training, strict checkpoint
audit, 1449-image
single-scale classification evaluation with image bootstrap, native multiscale
CAM generation on the 1464-image train split, and the fixed 0.45 / common
threshold-grid evaluation. The existing original joint MCTformer+ result is
referenced read-only from
`results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8`.

Completed variants, each with `VARIANT_COMPLETE`, are `full`, `no_p2c`,
`no_c2c`, and `no_p2c_no_c2c`. `c2p_update_off` was interrupted after epoch 1;
its partial best checkpoint and logs are preserved but it has no final
checkpoint, audit, classification evaluation, CAM evaluation, or completion
marker and must not be reported as a completed experiment. `p2p_update_off`,
`p2c_middle`, and `p2c_early` were not started. The run root correctly has no
`EXPERIMENT_COMPLETE` or `PIPELINE_COMPLETE` marker.

## Validation completed before launch

- Relevant deterministic tests: 84 passed.
- The runner's committed test set: 69 passed.
- Exact 448, batch-32 AMP forward/backward: finite.
- Exact pretraining adaptation and strict load: passed.
- Preflight peak allocated CUDA memory: 15.13 GiB.
- Model parameter count: 22,050,836, identical to original MCTformer+-Small.

## Safe status checks

```bash
tail -n 120 results/decoupled_bidirectional/20260909-mctformerplus-decoupled-s0-b1b4830/pipeline.log
nvidia-smi
tmux ls
```

The overall queue is intentionally incomplete. Inspect the immutable
per-variant completion markers before using results. Do not delete completed
checkpoints, CAMs, or the preserved partial `c2p_update_off` files.
