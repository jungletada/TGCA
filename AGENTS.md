# AGENTS.md

## Scope and execution host

This file governs the TGCA Git repository. On the LHR server, the repository root is:

```text
/home/peng/code/TGCA
```

All research-code development, testing, training, evaluation, and result generation must occur on `LHR` inside this repository. Treat paths in this file as relative to the repository root; do not create a nested `TGCA/` directory.

Before starting work, read `docs/CHAT_HANDOFF.md` and `docs/design.md` completely. The handoff records the current Git commit, Conda environment, datasets, checkpoints, completed measurements, active tmux session, result paths, and immediate next actions. Consult `docs/RESEARCH_PLAN_FULL.md` for the full rationale and go/no-go plan, and `docs/TIP_REVIEWS.md` for the reviewer concerns that the new work must resolve.

## Startup protocol on LHR

At the beginning of every new task:

1. Confirm the repository and instruction scope with `pwd`, `git status --short --branch`, and `git log -8 --oneline --decorate`.
2. Read `docs/CHAT_HANDOFF.md` and `docs/design.md` before proposing or changing research code.
3. Inspect `tmux ls` and the queue/log paths named in the handoff.
4. Do not stop, restart, duplicate, or replace a running experiment merely to inspect it.
5. Preserve all pre-existing tracked and untracked user files. Do not clean the worktree or discard changes without explicit approval.
6. Use the Conda environment specified for the method being run. TGCA, MoRe, and CTI use three independent environments; do not merge their dependency stacks.

The previously requested repository audit has already been completed and approved. Do not restart the project from the old no-edit audit gate. Resume from the handoff and current server state. If the handoff and live server state differ, treat the live Git/tmux/result state as operational truth, document the difference, and continue safely.

## Project mission

Prepare an original ICASSP 2027 Computer Vision paper derived from the MCTTA research line. Do **not** compress the rejected 16-page T-IP manuscript into four pages. Extract and validate one focused scientific contribution:

> Vanilla softmax mixes semantic evidence with token-group cardinality when class and patch tokens are normalized together. This can make class–patch attention depend on patch count and input resolution. Token-Group Calibrated Attention (TGCA) removes this group-size effect while preserving normalized, evidence-driven attention.

Working title: **Token-Group Calibrated Attention for Weakly Supervised Semantic Segmentation**.

The repository began as a clone of the existing MCTTA/MCTG codebase. It now contains an initial TGCA implementation, diagnostics, tests, and experiment runners, but TGCA must not be described as validated until the mechanism, matched normalization ablations, and independent-host tests pass.

## Source-of-truth files

- `docs/CHAT_HANDOFF.md`: operational state, reproducibility details, live experiment context, and next actions.
- `docs/design.md`: detailed TGCA method and experiment design.
- `docs/MCTTA.pdf`: rejected MCTTA manuscript; legacy evidence only.
- `docs/TIP_REVIEWS.md`: T-IP decision letter and reviewer comments.
- `docs/RESEARCH_PLAN_FULL.md`: full ICASSP/TGCA research plan.
- `models/tgca.py`: shared TGCA and normalization-ablation implementation.
- `experiments/`: reproducible experiment entry points.
- `tests/`: deterministic normalization and host-integration tests.
- `results/`: generated machine-readable metrics, logs, manifests, and checkpoints.

Never overwrite the legacy manuscript. Do not recreate absent legacy trees on the server. If paper writing is requested, create the new paper under `paper/` using the official ICASSP 2027 template.

## Non-negotiable research positioning

The new paper is about **attention normalization for heterogeneous token groups** in class-token WSSS.

- **MCTformer+** is the primary TGCA implementation host.
- **Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation** (ICCV 2025) is the required independent TGCA host.
- **DiCLIP** (T-IP 2026) is a recent external comparison baseline only; it is not a TGCA host.
- **MoRe**, **CTI**, and hierarchical MCTTA are optional supplementary hosts if time and compute permit.

Do not present the following as the main contribution:

- graph-based adapter or Spatial Prior Grapher superiority;
- Class Token Projection convergence speed;
- hierarchical feature fusion borrowed from FSSS;
- direct/single-stage/multi-stage pipeline taxonomy;
- MCTTA as a universal adapter;
- global SOTA claims across unmatched backbones, pretraining, supervision, or post-processing.

Do not use “Adapter” in the new title. Do not call the old frozen-classifier-plus-separate-segmentation procedure “single-stage.”

## Core method contract

For attention head `h`, let

```text
s_ij^h = (q_i^h)^T k_j^h / sqrt(d_h)
```

and let `g(j)` identify the key-token group, initially `{class, patch}`. TGCA is

```text
s_tilde_ij^h = s_ij^h - log(N_{g(j)}) + b_{g(i),g(j)}^h
A_ij^h = softmax_j(s_tilde_ij^h)
```

Required properties:

1. attention rows sum to one;
2. duplicating every key/value in one group does not change that group’s aggregate mass or the attention output, within numerical tolerance;
3. the count correction adds no trainable parameters;
4. optional relation bias is tiny, such as a per-head `2 x 2` matrix;
5. implementation works for self-attention and rectangular cross-attention;
6. no hidden output rescaling, changed value projection, altered residual path, or unrelated optimization change confounds the comparison.

The original split weighted softmax is a baseline, not the final method. Explicitly compare:

- vanilla global softmax;
- original split softmax `(1,1)`;
- normalized split softmax `(0.5,0.5)`;
- TGCA without relation bias;
- TGCA with relation bias.

The original split `(1,1)` assigns unit mass to each group and therefore has total row mass two. Never describe it as unit-normalized. If a diagnostic reports a small row-sum error for this mode, verify whether the implementation measures error relative to two. Normalized split `(0.5,0.5)` has unit row mass but fixes group mass rather than preserving evidence-driven inter-group competition.

## Required mechanism validation

Before expensive or broad experiments:

1. preserve a trustworthy vanilla MCTformer+ reproduction;
2. log class-group and patch-group attention mass by image, resolution, layer, head, query group, and key group;
3. test input resolutions `224, 320, 448, 512`;
4. perform a synthetic within-group patch-token replication test;
5. verify that vanilla group mass changes with patch count and TGCA remains stable;
6. confirm all unit-normalized attention rows sum to one numerically;
7. show that stability translates to CAM quality or cross-scale consistency rather than merely changing output scale.

Do not average complementary attention directions into a cancellation-prone scalar. Report class-query-to-class-key, class-query-to-patch-key, patch-query-to-class-key, and patch-query-to-patch-key effects separately, with layer/head detail and uncertainty estimates.

Do not proceed to expensive COCO experiments until the VOC mechanism, implementation tests, primary-host ablation, and independent-host gate are satisfied.

## Host and comparison roles

Keep implementation hosts separate from external comparison baselines:

1. **MCTformer+ — primary host.** Preserve the vanilla baseline and run the full normalization ablation and mechanism diagnostics under matched data, checkpoint policy, training, and evaluation.
2. **Know Your Attention Maps — required independent host.** Use the official repository at `https://github.com/HSG-AIML/TokenMasking-WSSS`. Reproduce its vanilla baseline first, then test TGCA under the same data, checkpoint, training, and evaluation pipeline. Its token sequence includes class tokens, patch tokens, and a singleton register token. Explicitly document and ablate whether the register is a separate group or handled another way. Do not silently merge it into another group or change unrelated components.
3. **DiCLIP — external comparison only.** Use the official repository at `https://github.com/zwyang6/DiCLIP`. Report author-provided and reproduced results separately, with transparent backbone, pretraining, supervision, and post-processing. Do not patch TGCA into DiCLIP for the core generality claim because its CLIP/diffusion pipeline and composite normalization create confounds.
4. **MoRe and CTI — optional supplementary hosts.** They do not replace the required Know Your Attention Maps experiment.

## Experiment and metric requirements

Use PASCAL VOC 2012 for development and ablation. Run MS COCO only after the VOC mechanism and generality gates pass.

Report at least:

- raw CAM/seed mIoU at a prespecified fixed threshold;
- threshold sensitivity as a diagnostic, without selecting a separate favorable threshold for every method;
- precision and recall;
- confusion ratio or an equivalent false-positive diagnostic;
- class-token classification performance;
- attention group mass by layer, head, and direction;
- variance and slope of directional group mass across input scales;
- cross-scale CAM consistency;
- downstream segmentation mIoU under one fixed pipeline;
- parameters, FLOPs, memory, wall time, and latency overhead;
- results across prespecified matched seeds after pilot selection.

Record repository URL, commit hash, environment, pretrained weights, dataset preparation, exact command, config, seed, checkpoint, threshold, and post-processing for every number. Separate author-reported and reproduced values.

## Go/no-go gates

Proceed only when all are satisfied:

- the directional group-cardinality phenomenon is measurable;
- TGCA improves or stabilizes CAM quality beyond simple output rescaling and fixed group mass;
- attention mass is substantially more stable across resolution/token-count changes;
- at least one independent host shows a positive effect;
- optimized overhead is negligible;
- claims remain matched for backbone, pretraining, supervision, and post-processing.

A practical target is roughly `+0.8 to +1.0` raw seed mIoU on the primary host, but a smaller gain may be publishable if the mechanism, invariance, and cross-scale stability evidence are exceptionally clear. If invariance improves while localization consistently worsens, do not force a positive conclusion: test whether cardinality bias is a useful inductive bias, whether a prespecified partial correction is justified, or whether the hypothesis should be rejected.

## Risks that must remain visible

- Trained logits may already compensate for token count.
- Resolution changes image evidence, object scale, interpolation, positional embeddings, and receptive fields in addition to patch count.
- Synthetic invariance does not establish better localization.
- Fixed group mass may perform as well as evidence-driven TGCA.
- Split `(1,1)` changes output scale and residual magnitude.
- Method-specific threshold tuning may create artificial gains.
- Relation bias may supply the gain rather than count correction.
- Seed variance may erase a seed-0 improvement.
- A Python mask/one-hot implementation may obscure the negligible-overhead claim; profile an optimized but mathematically identical version before reporting efficiency.
- A result confined to MCTformer+/MCTTA is insufficiently general.
- Register-token grouping in Know Your Attention Maps can materially change the result.
- Post-processing and unmatched pretraining/backbones can invalidate comparisons.

## Reviewer issues that must be resolved

The T-IP decision identified insufficient novelty, unclear technical explanation, outdated comparisons, and inadequate evidence. Therefore:

- organize the paper around one measurable failure mode;
- give rigorous equations, invariance reasoning, and implementation details;
- isolate every claimed component with additive ablations;
- include relevant 2025–2026 class-token, prototype/region, frequency/detail, and foundation-model WSSS work;
- compare only under transparent backbone, pretraining, supervision, and post-processing;
- quantify why CAMs improve instead of relying on qualitative examples;
- use vector figures.

## Known legacy defects: do not copy forward

- Cross-attention concatenation order conflicts with output slicing in the old equations.
- The residual input `X` in the old cross-attention block is undefined.
- Original split softmax `(1,1)` gives total row mass two despite the old figure implying one.
- The old manuscript has inconsistent MCTTA-D COCO values (`43.3` versus `44.0`).
- “MCTTA-D is trained for 20K/80K iterations” is almost certainly a naming error.
- The Fig. 10 caption and body describe different column meanings.
- The old `OneHot` operation is actually multi-hot thresholding.
- “rapid convergence is critical” is not established by MCTformer+ and must not be repeated.

## Repository and coding rules

- Treat this directory as the sole code Git repository and use its Git history for checkpoints.
- Put reusable model code in `models/`, experiment entry points and configs in `experiments/`, deterministic tests in `tests/`, generated metrics in `results/`, documentation in `docs/`, and the paper in `paper/` if requested.
- Do not create a nested `TGCA/` directory.
- Keep legacy material read-only.
- Preserve unrelated user changes and untracked files.
- Save machine-readable results (`json`, `csv`) in addition to human-readable tables and figures.
- Add deterministic unit tests for normalization, replication invariance, shape handling, mixed precision, and backward gradients.
- Use float32 for count corrections and normalization accumulation when needed for mixed-precision stability; preserve the intended output dtype.
- Never report a result without a command, config, seed, checkpoint, and commit.
- Use Git checkpoints before and after substantive changes, but do not commit, push, merge, rebase, or change branches unless the user requests or approves it.
- Do not fabricate citations, baselines, numbers, implementation behavior, or completed runs.
- Never delete result directories or checkpoints merely to recover disk space without explicit approval.

The current experiment runners refuse to start from tracked dirty state and refuse to overwrite an existing run directory. Preserve those safeguards.

## Conda and dependency rules

Use three independent Conda environments:

1. TGCA/MCTformer+ environment for this repository's primary development;
2. MoRe environment for `hosts/MoRe`;
3. CTI environment for `hosts/CTI`.

Follow the environment-building links in `README.md` and the host repositories. Record explicit Conda and pip manifests with each reproduced baseline. Do not upgrade working experiment dependencies mid-queue. If a dependency change is required, finish or safely stop the relevant queue, checkpoint Git, record the old environment, and create a new environment or explicit manifest.

## Running experiments safely

- Use `tmux` for long-running server experiments.
- Give every queue and run a unique descriptive name and immutable result directory.
- Capture stdout/stderr to logs and record stage boundaries.
- Check available GPU memory and disk space before launching.
- Run independent stages sequentially when they contend for the same GPU or result paths.
- Inspect a running job read-only through tmux, process listings, logs, and GPU monitoring.
- Do not launch a duplicate experiment when a matching tmux session or result directory exists.
- When a job finishes or fails, inspect exit status, completion markers, metrics, checkpoint hashes, and Git state before starting dependent work.

## Paper constraints

Target an ICASSP 2027 regular paper: four pages of technical content, with an optional fifth page for references and permitted acknowledgements only. Planned structure:

1. problem and cardinality-bias observation;
2. TGCA formulation and invariance;
3. main VOC/COCO and normalization ablations;
4. resolution stress test, host generality, efficiency, and conclusion.

Prefer one mechanism figure, one main comparison table, one normalization/generalization table, and one compact diagnostic plot.

## Completion standard for each task

Before reporting completion:

1. verify the requested artifact or result exists on LHR;
2. run tests or read-only validation proportional to the change;
3. report exact files, commands, commits, run IDs, and remaining uncertainties;
4. distinguish completed evidence from active, pending, failed, or merely planned work;
5. update `docs/CHAT_HANDOFF.md` when the operational research state materially changes.

