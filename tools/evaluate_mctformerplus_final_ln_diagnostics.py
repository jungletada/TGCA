#!/usr/bin/env python3
"""Focused attention and representation diagnostics for MCTformer+-FinalLN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    map_overlap_metrics,
    region_map_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    PAIR_REGION_VOID,
    assign_pair_patch_regions_from_counts,
    assign_patch_regions,
    patch_label_counts,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    VOCSemanticDataset,
)
from analysis.lazy_assignment.experiment3.presence_axis import (  # noqa: E402
    TwoFoldPresenceAccumulator,
    heldout_centered_projections,
    normalized_all_ones_direction,
    presence_projection_auroc,
    sha256_two_fold,
)
from models.mctformer_plus import (  # noqa: E402
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_final_norm_checkpoint,
)


STAGES = ('L10', 'L11', 'L12', 'native_last3')
ATTENTION_METRICS = (
    'c_pim',
    'target_vs_bg_auroc',
    'target_vs_other_fg_auroc',
    'positive_class_pair_top10_jaccard',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--list-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    final_norm_group = parser.add_mutually_exclusive_group()
    final_norm_group.add_argument('--final-norm', action='store_true')
    final_norm_group.add_argument('--patch-final-norm', action='store_true')
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bootstrap-resamples', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=20270906)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _clustered_mean(sums, counts):
    denominator = float(np.asarray(counts, dtype=np.float64).sum())
    if denominator <= 0:
        return float('nan')
    return float(np.asarray(sums, dtype=np.float64).sum() / denominator)


def _bootstrap_mean_intervals(sums, counts, draws):
    """Bootstrap one or more ratio-of-sums estimands by whole image."""

    values = np.asarray(sums, dtype=np.float64)
    denominators = np.asarray(counts, dtype=np.float64)
    if values.shape != denominators.shape or values.shape[0] != draws.shape[1]:
        raise ValueError('clustered sufficient-statistic shape mismatch')
    flat_values = values.reshape(values.shape[0], -1)
    flat_counts = denominators.reshape(denominators.shape[0], -1)
    numerator = draws @ flat_values
    denominator = draws @ flat_counts
    samples = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    low, high = np.nanquantile(samples, (0.025, 0.975), axis=0)
    return low.reshape(values.shape[1:]), high.reshape(values.shape[1:])


def _weighted_binary_auroc_samples(labels, scores, draws, batch_size=100):
    """Exact tie-aware AUROC for image-clustered multinomial weights."""

    target = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if target.shape != values.shape or target.ndim != 2:
        raise ValueError('presence labels/scores must have matching [image,class] shape')
    image_index = np.repeat(np.arange(target.shape[0]), target.shape[1])
    target_flat = target.reshape(-1)
    score_flat = values.reshape(-1)
    order = np.argsort(score_flat, kind='mergesort')
    score_sorted = score_flat[order]
    target_sorted = target_flat[order]
    image_sorted = image_index[order]
    starts = np.r_[0, np.flatnonzero(score_sorted[1:] != score_sorted[:-1]) + 1]
    output = np.full(len(draws), np.nan, dtype=np.float64)
    for begin in range(0, len(draws), batch_size):
        end = min(begin + batch_size, len(draws))
        weights = draws[begin:end, image_sorted].astype(np.float64, copy=False)
        positive = weights * target_sorted[None, :]
        negative = weights * (~target_sorted)[None, :]
        positive_group = np.add.reduceat(positive, starts, axis=1)
        negative_group = np.add.reduceat(negative, starts, axis=1)
        negative_before = np.cumsum(negative_group, axis=1) - negative_group
        numerator = np.sum(
            positive_group * (negative_before + 0.5 * negative_group), axis=1
        )
        denominator = positive_group.sum(axis=1) * negative_group.sum(axis=1)
        output[begin:end] = np.divide(
            numerator,
            denominator,
            out=np.full(end - begin, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    return output


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'cannot write empty table {path}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    if args.input_size != 448:
        raise ValueError('FinalLN diagnostics are fixed at input size 448')
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0:
        raise ValueError('invalid batch/worker/limit configuration')
    if args.bootstrap_resamples < 0:
        raise ValueError('bootstrap resamples must be non-negative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    for path in (args.checkpoint, args.voc_root, args.list_path):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, 'mctformerplus'
    )
    if resolution['variant'] != 'small':
        raise ValueError('MCTformer+-FinalLN is fixed to DeiT-Small')
    validate_mctformerplus_final_norm_checkpoint(
        checkpoint, bool(args.final_norm), bool(args.patch_final_norm)
    )
    attention_config = checkpoint.get('attention_normalization', {})
    bcss = checkpoint.get('bcss', {'variant': 'e0'})
    psl = checkpoint.get('psl', {'variant': 'baseline'})
    cti = checkpoint.get('cti_bgt', {'enabled': False})
    if (
        attention_config.get('mode', 'vanilla') != 'vanilla'
        or bcss.get('variant', 'e0') != 'e0'
        or psl.get('variant', 'baseline') != 'baseline'
        or cti.get('enabled', False)
    ):
        raise ValueError('FinalLN diagnostics require vanilla/E0/native MCTformer+')

    model = build_mctformerplus(
        'small', cam=True, num_classes=20, input_size=448,
        attention_normalization='vanilla', attention_gamma=1.0,
        bcss_variant='e0', psl_variant='baseline', cti_bgt=False,
        final_norm=bool(args.final_norm),
        patch_final_norm=bool(args.patch_final_norm),
    )
    state = checkpoint.get('model', checkpoint)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f'Unexpected strict-load result: {incompatibility}')
    model.to(device).eval()

    dataset = VOCSemanticDataset(
        args.voc_root, args.list_path, input_size=448, limit=args.limit
    )
    if not args.limit and len(dataset) != 1449:
        raise RuntimeError(f'Expected 1449 VOC val images, got {len(dataset)}')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    image_ids = list(dataset.image_ids)
    num_images = len(image_ids)
    attention_sums = np.zeros(
        (num_images, len(STAGES), len(ATTENTION_METRICS)), dtype=np.float64
    )
    attention_counts = np.zeros_like(attention_sums)
    pair_cosine_sum = np.zeros(num_images, dtype=np.float64)
    pair_cosine_count = np.zeros(num_images, dtype=np.float64)
    axis_energy_sum = np.zeros(num_images, dtype=np.float64)
    axis_energy_count = np.zeros(num_images, dtype=np.float64)
    labels_all = np.zeros((num_images, 20), dtype=np.uint8)
    class_tokens_all = np.zeros((num_images, 20, 384), dtype=np.float32)
    presence_accumulator = TwoFoldPresenceAccumulator(1, 20, 384)
    ones = normalized_all_ones_direction(384, device=device, dtype=torch.float32)

    started = time.perf_counter()
    image_index = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            class_tokens, _patch_tokens, attention_heads, all_x_cls, auxiliary = (
                model.forward_features(images, return_aux=True)
            )
            if auxiliary.get('final_norm') is not bool(args.final_norm):
                raise RuntimeError('runtime FinalLN state differs from requested state')
            if auxiliary.get('patch_final_norm') is not bool(
                    args.patch_final_norm):
                raise RuntimeError(
                    'runtime patch FinalLN state differs from requested state'
                )
            if len(attention_heads) != 12 or len(all_x_cls) != 12:
                raise RuntimeError('expected exactly 12 Transformer blocks')
            if class_tokens.shape[1:] != (20, 384):
                raise RuntimeError(f'unexpected class-token shape {class_tokens.shape}')

            late_raw = []
            for layer in (9, 10, 11):
                values = attention_heads[layer]
                if values.shape[1:] != (6, 804, 804):
                    raise RuntimeError(
                        f'unexpected L{layer + 1} attention shape {values.shape}'
                    )
                late_raw.append(values.float().mean(dim=1)[:, :20, 20:804])
            stage_maps = torch.stack((
                late_raw[0], late_raw[1], late_raw[2],
                torch.stack(late_raw).mean(dim=0),
            ), dim=1)
            stage_maps = stage_maps / stage_maps.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

            labels = batch['label'].to(dtype=torch.uint8)
            unit_tokens = torch.nn.functional.normalize(
                class_tokens.float(), p=2, dim=-1, eps=1e-12
            )
            pairwise = torch.einsum('bcd,bed->bce', unit_tokens, unit_tokens)
            coefficients = torch.einsum('bcd,d->bc', class_tokens.float(), ones)
            axis_energy = coefficients.square() / class_tokens.float().norm(
                dim=-1
            ).square().clamp_min(1e-12)

            maps_np = stage_maps.cpu().numpy()
            tokens_np = class_tokens.float().cpu().numpy()
            labels_np = labels.numpy()
            pairwise_np = pairwise.cpu().numpy()
            axis_np = axis_energy.cpu().numpy()
            masks_np = batch['mask'].numpy()

            for local, image_id in enumerate(batch['name']):
                global_index = image_index + local
                positives = np.flatnonzero(labels_np[local]).tolist()
                if not positives:
                    raise RuntimeError(f'{image_id} has no positive class')
                labels_all[global_index] = labels_np[local]
                class_tokens_all[global_index] = tokens_np[local]
                presence_accumulator.update(
                    image_id, tokens_np[local][None, ...], labels_np[local]
                )

                axis_energy_sum[global_index] = float(axis_np[local, positives].sum())
                axis_energy_count[global_index] = len(positives)
                for class_a, class_b in itertools.combinations(positives, 2):
                    pair_cosine_sum[global_index] += float(
                        pairwise_np[local, class_a, class_b]
                    )
                    pair_cosine_count[global_index] += 1

                counts = patch_label_counts(masks_np[local], patch_size=16)
                for class_id in positives:
                    regions = assign_patch_regions(
                        masks_np[local], class_id, patch_size=16,
                        rho=0.5, valid_fraction=0.5,
                    )['region_codes'].reshape(-1)
                    for stage_index in range(len(STAGES)):
                        result = region_map_metrics(
                            maps_np[local, stage_index, class_id], regions,
                            grid_h=28, grid_w=28, nonnegative_mass=True,
                        )
                        values = (
                            float(result['target_hit']),
                            float(result['auc_target_bg']),
                            float(result['auc_target_other']),
                        )
                        for metric_index, value in enumerate(values):
                            if np.isfinite(value):
                                attention_sums[
                                    global_index, stage_index, metric_index
                                ] += value
                                attention_counts[
                                    global_index, stage_index, metric_index
                                ] += 1

                for class_a, class_b in itertools.combinations(positives, 2):
                    pair_regions = assign_pair_patch_regions_from_counts(
                        counts, class_a, class_b, rho=0.5,
                        valid_fraction=0.5, grid_size=(28, 28),
                    )['region_codes'].reshape(-1)
                    eligible = pair_regions != PAIR_REGION_VOID
                    for stage_index in range(len(STAGES)):
                        overlap = map_overlap_metrics(
                            maps_np[local, stage_index, class_a],
                            maps_np[local, stage_index, class_b],
                            ratio=0.10, eligible=eligible,
                        )
                        attention_sums[global_index, stage_index, 3] += float(
                            overlap['topk_jaccard']
                        )
                        attention_counts[global_index, stage_index, 3] += 1

            image_index += len(batch['name'])
            if image_index % 100 == 0 or image_index == num_images:
                print(f'images={image_index}/{num_images}', flush=True)
            del attention_heads, stage_maps, class_tokens

    if image_index != num_images:
        raise RuntimeError(f'processed {image_index}, expected {num_images}')

    registry = presence_accumulator.finalize()
    oof_scores = np.empty((num_images, 20), dtype=np.float64)
    for index, image_id in enumerate(image_ids):
        oof_scores[index] = heldout_centered_projections(
            class_tokens_all[index][None, ...],
            eval_fold=sha256_two_fold(image_id), registry=registry,
        )[0]
    presence = presence_projection_auroc(oof_scores, labels_all)
    alignments = np.einsum(
        'fd,d->f',
        registry.shared_directions[:, 0],
        np.full(384, 1.0 / np.sqrt(384), dtype=np.float64),
    )

    if args.bootstrap_resamples:
        generator = np.random.default_rng(args.bootstrap_seed)
        draws = generator.multinomial(
            num_images,
            np.full(num_images, 1.0 / num_images),
            size=args.bootstrap_resamples,
        ).astype(np.float64)
        attn_low, attn_high = _bootstrap_mean_intervals(
            attention_sums, attention_counts, draws
        )
        pair_low, pair_high = _bootstrap_mean_intervals(
            pair_cosine_sum[:, None], pair_cosine_count[:, None], draws
        )
        energy_low, energy_high = _bootstrap_mean_intervals(
            axis_energy_sum[:, None], axis_energy_count[:, None], draws
        )
        auroc_samples = _weighted_binary_auroc_samples(
            labels_all, oof_scores, draws
        )
        auroc_low, auroc_high = np.nanquantile(
            auroc_samples, (0.025, 0.975)
        )
    else:
        draws = None
        attn_low = attn_high = np.full(
            (len(STAGES), len(ATTENTION_METRICS)), np.nan
        )
        pair_low = pair_high = energy_low = energy_high = np.asarray([np.nan])
        auroc_low = auroc_high = np.nan

    attention_rows = []
    for stage_index, stage in enumerate(STAGES):
        for metric_index, metric in enumerate(ATTENTION_METRICS):
            attention_rows.append({
                'stage': stage,
                'metric': metric,
                'estimate': _clustered_mean(
                    attention_sums[:, stage_index, metric_index],
                    attention_counts[:, stage_index, metric_index],
                ),
                'ci95_low': float(attn_low[stage_index, metric_index]),
                'ci95_high': float(attn_high[stage_index, metric_index]),
                'num_images': int(np.count_nonzero(
                    attention_counts[:, stage_index, metric_index]
                )),
                'num_observations': int(
                    attention_counts[:, stage_index, metric_index].sum()
                ),
                'bootstrap_unit': 'image',
            })

    representation_rows = [
        {
            'metric': 'positive_class_token_pair_cosine',
            'estimate': _clustered_mean(pair_cosine_sum, pair_cosine_count),
            'ci95_low': float(pair_low[0]),
            'ci95_high': float(pair_high[0]),
            'num_images': int(np.count_nonzero(pair_cosine_count)),
            'num_observations': int(pair_cosine_count.sum()),
            'aggregation': 'micro_positive_pairs',
        },
        {
            'metric': 'shared_presence_projection_auroc',
            'estimate': float(presence.micro),
            'ci95_low': float(auroc_low),
            'ci95_high': float(auroc_high),
            'num_images': num_images,
            'num_observations': int(labels_all.size),
            'aggregation': 'micro_image_class',
        },
        {
            'metric': 'fixed_all_ones_axis_energy',
            'estimate': _clustered_mean(axis_energy_sum, axis_energy_count),
            'ci95_low': float(energy_low[0]),
            'ci95_high': float(energy_high[0]),
            'num_images': num_images,
            'num_observations': int(axis_energy_count.sum()),
            'aggregation': 'micro_gt_positive_tokens',
        },
        {
            'metric': 'learned_shared_direction_all_ones_alignment',
            'estimate': float(alignments.min()),
            'ci95_low': float('nan'),
            'ci95_high': float('nan'),
            'num_images': num_images,
            'num_observations': 2,
            'aggregation': 'minimum_signed_alignment_across_fit_folds',
        },
    ]

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'command.txt').write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    _write_csv(args.output_dir / 'attention_results.csv', attention_rows)
    _write_csv(args.output_dir / 'representation_results.csv', representation_rows)
    np.savez_compressed(
        args.output_dir / 'per_image_sufficient_statistics.npz',
        image_ids=np.asarray(image_ids), labels=labels_all,
        attention_sums=attention_sums, attention_counts=attention_counts,
        pair_cosine_sum=pair_cosine_sum,
        pair_cosine_count=pair_cosine_count,
        axis_energy_sum=axis_energy_sum,
        axis_energy_count=axis_energy_count,
        oof_presence_scores=oof_scores,
    )
    np.savez_compressed(
        args.output_dir / 'shared_presence_directions.npz',
        fit_means=registry.fit_means,
        class_deltas=registry.class_deltas,
        shared_directions=registry.shared_directions,
        class_alignment=registry.class_alignment,
        total_counts=registry.total_counts,
        positive_counts=registry.positive_counts,
        negative_counts=registry.negative_counts,
        signed_alignment_with_all_ones=alignments,
    )
    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'MCTformer+-FinalLN focused diagnostics',
        'final_norm': bool(args.final_norm),
        'patch_final_norm': bool(args.patch_final_norm),
        'model_spec': model_spec_from_instance(model),
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': sha256_file(args.checkpoint),
            'epoch': checkpoint.get('epoch'),
        },
        'dataset': {
            'name': 'PASCAL VOC 2012 val',
            'num_images': num_images,
            'list_path': str(args.list_path.resolve()),
            'list_sha256': sha256_file(args.list_path),
            'input_size': 448,
            'transform': 'Experiment2JointTransform bicubic RGB / nearest GT',
        },
        'attention_definition': {
            'per_layer': 'head-mean native class-to-patch softmax attention',
            'native_last3': 'arithmetic mean of raw head-mean L10/L11/L12 attention before patch-conditional normalization',
            'region_rho': 0.5,
            'minimum_valid_fraction': 0.5,
            'top10_ratio': 0.10,
            'pair_top10_void_policy': 'exclude pair-region void patches',
        },
        'representation_definition': {
            'tokens': 'actual final class-token branch input',
            'presence_folds': 'SHA-256 image-ID parity; opposite fold used for evaluation',
            'presence_projection': 'fit-fold per-class centered, shared presence direction',
            'fixed_axis_energy_stratum': 'GT-positive class tokens',
            'alignment_primary': 'minimum signed alignment across two fit folds',
            'signed_alignment_by_fit_fold': [float(value) for value in alignments],
            'signed_alignment_mean': float(alignments.mean()),
            'presence_projection_macro_class_auroc': _finite_or_none(
                presence.macro_class
            ),
        },
        'bootstrap': {
            'unit': 'image',
            'resamples': args.bootstrap_resamples,
            'seed': args.bootstrap_seed,
            'patches_image_class_pairs_or_class_pairs_independent': False,
            'presence_fit_uncertainty': 'conditional on two fixed fitted directions',
        },
        'attention_results': attention_rows,
        'representation_results': [
            {key: (_finite_or_none(value) if key.startswith('ci95') else value)
             for key, value in row.items()}
            for row in representation_rows
        ],
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': str(device),
            'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
            'commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
            ).strip(),
        },
        'elapsed_seconds': time.perf_counter() - started,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'ATTENTION_REPRESENTATION_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
