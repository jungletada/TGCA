# MCTformer+ Class/Patch Token-Role Specialization Pilot

## Status and scope

**Completed 2026-08-28; final seed-0 decision: no-go for direct transfer.**

This is an exploratory MCTformer+ seed-0 pilot motivated by the ICLR 2026 paper
[Revisiting [CLS] and Patch Token Interaction in Vision Transformers](https://arxiv.org/abs/2602.08626).
It does not establish a novel contribution: separate normalization and selective
QKV specialization are prior art. The purpose of this pilot is to test whether
the same mechanism is relevant to multi-class-token WSSS before choosing a new
research direction.

The paper specializes LayerNorm and LayerScale in every block and obtains an
average segmentation improvement from `64.5` to `65.6` mIoU in its ViT-L
DINOv2 setting. Adding separate QKV projections in the first third of blocks
raises the average to `66.6`, with an `8.3%` parameter increase. QKV-only
specialization is reported to be close to baseline, and MLP specialization is
not beneficial. These numbers are context, not MCTformer+ baselines.

## MCTformer+ adaptation

MCTformer+ has 20 learned class tokens rather than one `[CLS]` token. The
adapted roles are therefore:

```text
class role: the first 20 class tokens
patch role: all spatial patch tokens
```

The E0 host has no register or background token. The pilot does not combine
role specialization with BCSS or TGCA. MCTformer+'s blocks have LayerNorm but
no LayerScale, so only `norm1` and `norm2` are specialized. The attention
normalization, output projection, MLP, residual path, losses, optimizer, data,
and CAM construction remain shared and unchanged.

## Prespecified modes

| Mode | LayerNorm | QKV | MLP |
|---|---|---|---|
| `shared` | shared in all blocks | shared | shared |
| `norm` | class/patch paths in all 12 blocks | shared | shared |
| `norm_qkv` | class/patch paths in all 12 blocks | separate in blocks 0-3 | shared |

Each new class-role path is initialized by copying the corresponding shared
DeiT weight. Consequently, `norm` and `norm_qkv` are equivalent to `shared` at
initialization within numerical tolerance. The patch path retains the original
state-dict names.

## Matched VOC protocol

- PASCAL VOC 2012 augmented training split;
- MCTformer+ / DeiT-S initialization;
- input size `448`, 45 epochs, seed `0`;
- vanilla global attention normalization and BCSS `E0`;
- CAM scales `1.0, 0.75, 1.25`;
- fixed background threshold `0.45`, selected previously by vanilla seed 0;
- raw CAM evaluation on 1,464 VOC train images, without CRF.

The completed E0 run
`20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3` is reused as the
`shared` baseline because the new default code path has no new parameters or
state-dict keys and is covered by an exact-equation compatibility test.

The sequential entry point is:

```bash
bash experiments/ablations/run_token_role_voc_pilot.sh
```

Every specialized run records raw CAM mIoU, semantic and binary foreground
precision/recall, background false-positive rate, final classification mAP,
parameter count, peak memory, latency, and per-image/per-layer class-patch
cosine similarity immediately before and after `norm1`. The same cosine and
efficiency diagnostics are recomputed for the frozen shared checkpoint.

## Decision rule

Do not expand to multiple seeds, COCO, an independent host, or combinations
with TGCA/BCSS from this pilot alone. A useful seed-0 signal should improve raw
CAM mIoU while retaining classification and should improve at least one
localization error measure, rather than only increasing class/patch feature
separation. Compare `norm` against `norm_qkv` to determine whether the gain, if
any, comes from the lightweight normalization split or the 8% QKV capacity.

If feature separation increases while CAM localization degrades, the ICLR 2026
mechanism does not transfer directly to MCTformer+'s multi-class-token WSSS
objective. If both modes improve, the next research discussion must identify a
WSSS-specific mechanism beyond simply reusing separate LayerNorm/QKV.

## Completed result

The sequential queue completed normally at `2026-08-28T18:22:06+09:00` from
commit `d055da84197d1965d81176efd4785e05357822ba`. Machine-readable outputs are
under:

```text
results/mctformerplus/voc/comparisons/
  token-role-pilot-20260828-152106-s0-d055da8/
```

| Mode | Raw CAM mIoU | Semantic P/R | Background FPR | Final cls mAP |
|---|---:|---:|---:|---:|
| shared | 70.063 | 80.735/85.817 | 6.756 | 96.410 |
| norm | 68.960 | 80.121/84.657 | 7.001 | 96.503 |
| norm_qkv | 68.482 | 79.911/84.477 | 7.033 | 96.203 |

Relative to shared, paired image bootstrap gives a raw-CAM mIoU change of
`-1.104` (95% CI `[-1.519,-0.688]`) for `norm` and `-1.581`
(`[-2.332,-0.805]`) for `norm_qkv`. Classification is retained, so the
localization decline is not explained by a failed classifier. Both semantic
precision and recall decline, and background false positives increase.

The LayerNorm-only path also does not create a materially different all-layer
class-patch cosine response: shared changes from `0.00745` before `norm1` to
`-0.05755` after it, while `norm` changes from `0.00994` to `-0.05811`.
Separate early QKV moves the pre-normalization cosine to `-0.03456`, but that
stronger representation separation coincides with worse CAMs and costs an
`8.129%` parameter increase.

The prespecified decision rule therefore fails. Do not expand these modes to
additional seeds, datasets, hosts, or combinations. The result rejects this
direct MCTformer+ transfer; it does not prove that class and patch roles are
identical or rule out a different WSSS-specific mechanism.
