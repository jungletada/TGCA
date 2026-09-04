#!/usr/bin/env python3
"""Layer-wise semantic ownership of MCTformer+ class/patch feature scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.metrics_region import (  # noqa: E402
    region_map_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    assign_patch_regions,
)
from analysis.lazy_assignment.experiment2.signal_collector import (  # noqa: E402
    SignalCollector,
    assert_no_change,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    VOCSemanticDataset,
)
from models.mctformer_plus import (  # noqa: E402
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
)


METRICS = (
    'target_hit', 'other_fg_hit', 'background_hit', 'mixed_hit',
    'target_top05_fraction', 'other_fg_top05_fraction', 'bg_top05_fraction',
    'target_top10_fraction', 'other_fg_top10_fraction', 'bg_top10_fraction',
    'target_tail_enrich_05', 'other_fg_tail_enrich_05', 'bg_tail_enrich_05',
    'target_tail_enrich_10', 'other_fg_tail_enrich_10', 'bg_tail_enrich_10',
    'target_mean', 'target_median', 'target_q90',
    'other_fg_mean', 'other_fg_median', 'other_fg_q90',
    'bg_mean', 'bg_median', 'bg_q90', 'bg_q95',
    'target_bg_mean_margin', 'target_other_mean_margin',
    'auc_target_bg', 'ap_target_bg', 'auc_target_other', 'ap_target_other',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--model', choices=(
            'mctformerplus_tiny', 'mctformerplus', 'mctformerplus_base'),
        required=True,
    )
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--list-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--minimum-valid-fraction', type=float, default=0.5)
    parser.add_argument('--bootstrap-resamples', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=2027)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    return float(value)


def _bootstrap_all(sums, counts, resamples, seed, batch_size=100):
    if resamples < 1:
        return None
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(sums), 1.0 / len(sums))
    samples = np.empty((resamples, *sums.shape[1:]), dtype=np.float64)
    flat_sums = sums.reshape(len(sums), -1)
    flat_counts = counts.reshape(len(counts), -1)
    for begin in range(0, resamples, batch_size):
        end = min(begin + batch_size, resamples)
        weights = generator.multinomial(
            len(sums), probabilities, size=end - begin
        )
        numerator = weights @ flat_sums
        denominator = weights @ flat_counts
        values = np.divide(
            numerator, denominator,
            out=np.full_like(numerator, np.nan), where=denominator != 0,
        )
        samples[begin:end] = values.reshape(end - begin, *sums.shape[1:])
    return {
        'low': np.nanquantile(samples, 0.025, axis=0),
        'high': np.nanquantile(samples, 0.975, axis=0),
        'finite_resamples': np.isfinite(samples).sum(axis=0),
    }


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    if args.input_size != 448:
        raise ValueError('Width semantic ownership is pre-registered at 448')
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0:
        raise ValueError('Invalid batch/worker/limit configuration')
    if args.bootstrap_resamples < 0:
        raise ValueError('bootstrap resamples must be non-negative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, args.model
    )
    attention = checkpoint.get('attention_normalization', {})
    bcss = checkpoint.get('bcss', {'variant': 'e0'})
    psl = checkpoint.get('psl', {'variant': 'baseline'})
    cti = checkpoint.get('cti_bgt', {'enabled': False})
    if (
        attention.get('mode', 'vanilla') != 'vanilla'
        or bcss.get('variant', 'e0') != 'e0'
        or psl.get('variant', 'baseline') != 'baseline'
        or cti.get('enabled', False)
    ):
        raise ValueError('Semantic ownership requires native vanilla/E0 baseline')
    model = build_mctformerplus(
        resolution['variant'], cam=True, num_classes=20, input_size=448,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
    )
    state = checkpoint.get('model', checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'Unexpected strict-load result: {incompatible}')
    model.to(device).eval()

    dataset = VOCSemanticDataset(
        args.voc_root, args.list_path, input_size=448, limit=args.limit
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    first = dataset[0]['image'].unsqueeze(0).to(device)
    with torch.inference_mode():
        plain = model(first)
        with SignalCollector(model, num_classes=20) as guard:
            guard.clear(expected_num_patches=784)
            hooked = model(first)
            guard_capture = guard.consume()
    no_change = assert_no_change(plain, hooked, tolerance=0.0)
    guard_metrics = {
        'native_cam_max_abs_diff': no_change,
        'qk_attention_max_abs_diff': float(
            guard_capture.qk_attention_max_abs_diff.max()
        ),
        'attention_row_sum_max_abs_error': float(
            guard_capture.attention_row_sum_max_abs_error.max()
        ),
        'pre_norm_input_max_abs_diff': float(
            guard_capture.pre_norm_input_max_abs_diff.max()
        ),
        'norm_qkv_input_max_abs_diff': float(
            guard_capture.norm_qkv_input_max_abs_diff.max()
        ),
    }
    if guard_metrics['qk_attention_max_abs_diff'] >= 1e-6:
        raise RuntimeError(f'QK reconstruction guard failed: {guard_metrics}')
    del first, plain, hooked, guard_capture

    image_ids = list(dataset.image_ids)
    metric_sums = np.zeros(
        (len(image_ids), 12, len(METRICS)), dtype=np.float64
    )
    metric_counts = np.zeros_like(metric_sums)
    rows = []
    image_index = 0
    maximum_qk_difference = 0.0
    maximum_row_error = 0.0
    started = time.perf_counter()
    with SignalCollector(model, num_classes=20) as collector:
        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(device, non_blocking=True)
                collector.clear(expected_num_patches=784)
                model(images)
                capture = collector.consume()
                maximum_qk_difference = max(
                    maximum_qk_difference,
                    float(capture.qk_attention_max_abs_diff.max()),
                )
                maximum_row_error = max(
                    maximum_row_error,
                    float(capture.attention_row_sum_max_abs_error.max()),
                )
                scores = capture.feature_post_scores.cpu().numpy()
                labels = batch['label'].numpy()
                masks = batch['mask'].numpy()
                for local_index, image_id in enumerate(batch['name']):
                    positives = np.flatnonzero(labels[local_index]).tolist()
                    stratum = (
                        'single_label' if len(positives) == 1 else
                        ('two_label' if len(positives) == 2 else 'three_plus_label')
                    )
                    for class_id in positives:
                        assignment = assign_patch_regions(
                            masks[local_index], class_id, patch_size=16,
                            rho=args.rho,
                            valid_fraction=args.minimum_valid_fraction,
                        )
                        regions = assignment['region_codes'].reshape(-1)
                        for layer in range(12):
                            result = region_map_metrics(
                                scores[layer, local_index, class_id],
                                regions,
                                grid_h=28,
                                grid_w=28,
                                nonnegative_mass=False,
                            )
                            row = {
                                'image_id': image_id,
                                'class_id': class_id,
                                'positive_class_count': len(positives),
                                'stratum': stratum,
                                'layer': layer + 1,
                                'num_target': result['num_target'],
                                'num_other_fg': result['num_other_fg'],
                                'num_bg': result['num_bg'],
                                'num_mixed': result['num_mixed'],
                                'num_void': result['num_void'],
                            }
                            for metric_index, name in enumerate(METRICS):
                                value = _numeric(result[name])
                                row[name] = value
                                if np.isfinite(value):
                                    metric_sums[
                                        image_index, layer, metric_index
                                    ] += value
                                    metric_counts[
                                        image_index, layer, metric_index
                                    ] += 1
                            rows.append(row)
                    image_index += 1
                if image_index % 100 == 0 or image_index == len(image_ids):
                    print(f'images={image_index}/{len(image_ids)}', flush=True)
    if image_index != len(image_ids):
        raise RuntimeError(f'Processed {image_index}, expected {len(image_ids)}')
    if maximum_qk_difference >= 1e-6 or maximum_row_error > 2e-6:
        raise RuntimeError(
            f'Runtime numerical guard failed: qk={maximum_qk_difference}, '
            f'row={maximum_row_error}'
        )

    bootstrap = _bootstrap_all(
        metric_sums, metric_counts, args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    layer_rows = []
    for layer in range(12):
        row = {'layer': layer + 1}
        for metric_index, name in enumerate(METRICS):
            numerator = metric_sums[:, layer, metric_index]
            denominator = metric_counts[:, layer, metric_index]
            total_count = denominator.sum()
            point = (
                float(numerator.sum() / total_count)
                if total_count > 0 else np.nan
            )
            row[name] = point
            row[f'{name}_ci95_low'] = (
                float(bootstrap['low'][layer, metric_index])
                if bootstrap is not None else np.nan
            )
            row[f'{name}_ci95_high'] = (
                float(bootstrap['high'][layer, metric_index])
                if bootstrap is not None else np.nan
            )
        layer_rows.append(row)

    class_rows = []
    for class_id in range(20):
        selected = [row for row in rows if row['class_id'] == class_id]
        for layer in range(1, 13):
            layer_values = [row for row in selected if row['layer'] == layer]
            output = {
                'class_id': class_id,
                'layer': layer,
                'image_class_count': len(layer_values),
            }
            for name in METRICS:
                values = np.asarray([row[name] for row in layer_values])
                output[name] = float(np.nanmean(values))
            class_rows.append(output)

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'command.txt').write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    with (args.output_dir / 'per_image_class_layer.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'layerwise_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    with (args.output_dir / 'classwise_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    np.savez_compressed(
        args.output_dir / 'per_image_metric_sufficient_statistics.npz',
        image_ids=np.asarray(image_ids),
        layers=np.arange(1, 13, dtype=np.int16),
        metric_names=np.asarray(METRICS),
        metric_sums=metric_sums,
        metric_counts=metric_counts,
    )

    summary = {
        'schema_version': 1,
        'status': 'complete',
        'analysis': 'MCTformer+ width semantic ownership of raw post-block cosine scores',
        'model_spec': model_spec_from_instance(model),
        'variant_resolution': resolution,
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': sha256_file(args.checkpoint),
            'epoch': checkpoint.get('epoch'),
        },
        'dataset': {
            'name': 'PASCAL VOC 2012 val',
            'voc_root': str(args.voc_root.resolve()),
            'list_path': str(args.list_path.resolve()),
            'list_sha256': sha256_file(args.list_path),
            'num_images': len(image_ids),
            'input_size': 448,
            'transform': 'Experiment2JointTransform bicubic RGB / nearest GT',
        },
        'region_definition': {
            'rho': args.rho,
            'minimum_valid_fraction': args.minimum_valid_fraction,
            'regions': ['target', 'other_fg', 'background', 'mixed', 'void'],
        },
        'bootstrap': {
            'unit': 'image',
            'resamples': args.bootstrap_resamples,
            'seed': args.bootstrap_seed,
            'patches_or_image_class_pairs_independent': False,
            'training_seed_uncertainty_included': False,
        },
        'primary_layer12': {
            name: (
                float(layer_rows[-1][name])
                if np.isfinite(layer_rows[-1][name]) else None
            )
            for name in METRICS
        },
        'numerical_guards': {
            **guard_metrics,
            'maximum_qk_attention_diff_full': maximum_qk_difference,
            'maximum_attention_row_error_full': maximum_row_error,
        },
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
    (args.output_dir / 'semantic_ownership_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'SEMANTIC_OWNERSHIP_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
