# MCTformer+ C2P patch pooling

Single change: `--patch-pooling gwrp|c2p` (default `gwrp`).
Class-token initialization code, options, parameters and random draws are
unchanged. No CWP/residual CWP module is added, deleted, or switched.

For c2p, slice actual `forward_features` attention records from L10–L12,
average their heads and layers, then normalize over patch keys with denominator
clamped at 1e-8. Weight the original 3x3 classifier logits in their spatial
order. No sorting, geometric decay, ReLU, sigmoid or CAM normalization in
pooling. No detach, extra parameters or loss. FP32 accumulation under AMP
prevents the specified 1e-8 denominator clamp underflowing in FP16.

Training `forward` and frozen two-head classification both execute this branch.
`forward_with_label` uses its pooled patch logits for gating. Native CAM
generation remains unchanged: original head logits -> ReLU -> native C2P
refinement -> sqrt -> all-layer P2P. Same-weight CAM equivalence is tested.
Checkpoint `model_spec.patch_pooling` records the choice; missing legacy
metadata means gwrp. CLI must match for CAM and classification loading.

## Matched experiment

VOC only, same ordinary MCTformer+-Small baseline (not patch-first):
`class_token_init=baseline`, joint vanilla attention, no FinalLN, BCSS E0,
PSL baseline, no CTI-BGT or LaST. All existing initialization configurations
remain available, tested for identical state initialization across pooling.

Baseline: `results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12`.
New: `results/c2p_pooling/20260914-voc-s0`.
Seed 0, 448, batch 32, 45 epochs, same official DeiT-S weights, AdamW/cosine,
nominal LR 5e-4, min LR 1e-5, original augmentation/CCT/losses/splits.
Existing baseline weights and CAM results remain immutable. Reuse CAM metrics;
evaluate both final checkpoints with the same two-head classification tool.

Training's historical `mAP` is mean image AP, not dataset macro-class AP.
Report both branches' macro-class AP on VOC val (1449), FP32 448 no flip,
and validation losses. Raw CAM: native scales 1,.75,1.25, train 1464,
fixed .45 and complete grid 0:.01:.59. No CRF or downstream segmentation.
This is one seed and provides no training-seed uncertainty estimate.

Runner: `python -m experiments.ablations.run_c2p_pooling --output results/c2p_pooling/20260914-voc-s0`.
Tests -> short train/checkpoint/CAM/classification smoke -> existing baseline
classification -> full c2p train/CAM -> c2p classification -> compact report.
Final files: `comparison.csv`, `C2P_POOLING_REPORT.md`, exact commands, config,
environment/test logs, code and checkpoint hashes. No additional variants.
