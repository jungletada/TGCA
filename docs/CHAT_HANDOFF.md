# TGCA Operational Handoff

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
