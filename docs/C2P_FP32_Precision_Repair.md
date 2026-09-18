# All-product pre-cast FP32 pooling repair

## Scope

This repairs an AMP precision boundary, not the pooling formula or training recipe.
The old path was FP32 softmax -> FP16 probability -> FP32 head/layer pooling.
Casting back to FP32 cannot recover probabilities already rounded to zero, and
the backward pass also crosses a half-precision probability-gradient node.

The new opt-in path branches at the existing FP32 softmax: head-mean C2P values
are retained in FP32 with gradients, followed by the unchanged log-product and
spatial softmax. The native probability cast, attention times V, residuals,
raw CCT tokens, classifier and raw CAM equations are unchanged. QK still uses
the original AMP arithmetic. No new parameters/losses or GradScaler changes.

Enable consistently in training, classification evaluation and CAM commands:

```sh
--patch-pooling c2p --c2p-pooling-layers all --c2p-pooling-reduction product --c2p-pooling-fp32
```

Default false deliberately reproduces historical checkpoints. The new flag is
recorded in `model_spec`; evaluation rejects a flag/metadata mismatch. Frozen
old-checkpoint probes toggle it only in memory for precision diagnostics, not
to claim a newly trained method's performance. GWRP is untouched.

## Validation command

```sh
OMP_NUM_THREADS=2 /home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.ablations.validate_c2p_fp32 --output results/c2p_fp32_validation/20260918-all-product-r1
```

The runner requires tracked-clean Git and a new output directory. It logs exact
commands, environment, implementation SHA, source hashes, tests and checkpoint
metadata. It runs only a bounded frozen first8-val-image probe and two-epoch
64-image smoke, not the full45-epoch seed comparison. CAM evaluation is online;
no per-image CAM dumps are needed. Old source results/checkpoints are read-only.

## Evidence status

Before implementation commit:65 focused tests passed, including a reproducible
synthetic probability-underflow case and FP16 probability-gradient overflow case,
FP32-reference pooling and gradients, exact native AMP token/attention/CCT/CAM
and RNG equivalence, legacy behavior, checkpoint metadata and save/load.
Real-checkpoint probe and training integration status will be appended after
completion. Numerical repair alone cannot establish the cause of the large
seed0/11 performance gap or promise its removal.
