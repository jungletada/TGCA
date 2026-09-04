#!/usr/bin/env python3
"""Audit and aggregate Tiny/Small/Base MCTformer+ width experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_mctformerplus_classification import (  # noqa: E402
    CLASS_NAMES,
    bootstrap_metrics,
    classification_metrics,
)
from tools.evaluate_cam_threshold_grid import confusion_metrics  # noqa: E402


VARIANTS = ('tiny', 'small', 'base')
EXPECTED_MODELS = {
    'tiny': 'mctformerplus_tiny',
    'small': 'mctformerplus',
    'base': 'mctformerplus_base',
}
EXPECTED_PARAMETERS = {
    'tiny': 5717012,
    'small': 22050836,
    'base': 86568980,
}
PAIR_ORDER = (
    ('tiny', 'small'),
    ('base', 'small'),
    ('base', 'tiny'),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiny-run', type=Path, required=True)
    parser.add_argument('--small-run', type=Path, required=True)
    parser.add_argument('--small-reanalysis', type=Path, required=True)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2027)
    parser.add_argument('--semantic-bootstrap-resamples', type=int, default=5000)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _manifest_hashes(path):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 64:
            result[Path(fields[-1]).name] = fields[0]
    return result


def load_bundle(variant, root, small_source=None):
    root = root.expanduser().resolve()
    required = (
        'PIPELINE_COMPLETE', 'config.json', 'git_state.json', 'model_spec.json',
        'dataset_manifest.txt', 'audit_report.json', 'benchmark.json',
        'training_runtime.json',
        'classification/classification_metrics.json',
        'classification/classification_predictions.npz',
        'cam_evaluation_train/metrics.json',
        'cam_evaluation_train/per_image_confusions.npz',
        'cam_evaluation_val/metrics.json',
        'cam_evaluation_val/per_image_confusions.npz',
        'semantic_ownership/semantic_ownership_summary.json',
        'semantic_ownership/per_image_metric_sufficient_statistics.npz',
        'semantic_ownership/layerwise_summary.csv',
    )
    for relative in required:
        _require((root / relative).is_file(), f'{variant} missing {relative}')
    config = load_json(root / 'config.json')
    spec = load_json(root / 'model_spec.json')
    audit = load_json(root / 'audit_report.json')
    classification = load_json(root / 'classification/classification_metrics.json')
    cam = {
        split: load_json(root / f'cam_evaluation_{split}/metrics.json')
        for split in ('train', 'val')
    }
    semantic = load_json(
        root / 'semantic_ownership/semantic_ownership_summary.json'
    )
    benchmark = load_json(root / 'benchmark.json')
    training_runtime = load_json(root / 'training_runtime.json')
    _require(audit.get('passed'), f'{variant} checkpoint audit did not pass')
    _require(
        all(audit.get('pipeline_checks', {}).values()),
        f'{variant} pipeline audit did not pass',
    )
    _require(spec.get('variant') == variant, f'{variant} model_spec variant mismatch')
    _require(
        spec.get('model_name') == EXPECTED_MODELS[variant],
        f'{variant} model name mismatch',
    )
    _require(
        audit['parameters']['total'] == EXPECTED_PARAMETERS[variant],
        f'{variant} parameter count mismatch',
    )
    checkpoint = Path(audit['checkpoint']['path']).resolve()
    _require(checkpoint.is_file(), f'{variant} checkpoint is absent')
    _require(
        sha256_file(checkpoint) == audit['checkpoint']['sha256'],
        f'{variant} checkpoint hash changed',
    )
    if variant == 'small':
        _require(small_source is not None, 'Small source root was not provided')
        expected = (small_source / 'mctformerplus_final.pth').resolve()
        _require(checkpoint == expected, 'Small reanalysis checkpoint linkage differs')
    else:
        manifest = root / 'checkpoint_manifest.txt'
        _require(manifest.is_file(), f'{variant} checkpoint manifest absent')
        _require(
            manifest.read_text().split()[0] == sha256_file(checkpoint),
            f'{variant} checkpoint manifest mismatch',
        )
    return {
        'variant': variant,
        'root': root,
        'config': config,
        'spec': spec,
        'audit': audit,
        'classification': classification,
        'cam': cam,
        'semantic': semantic,
        'benchmark': benchmark,
        'training_runtime': training_runtime,
        'checkpoint': checkpoint,
        'dataset_hashes': _manifest_hashes(root / 'dataset_manifest.txt'),
    }


def audit_comparability(bundles):
    checks = []

    def check(name, passed, observed):
        checks.append({'name': name, 'passed': bool(passed), 'observed': observed})

    expected_config = {
        'seed': 0,
        'epochs': 45,
        'input_size': 448,
        'effective_batch_size': 32,
        'optimizer': 'adamw',
        'nominal_lr': 0.0005,
        'optimizer_lr': 0.00003125,
        'minimum_lr': 0.00001,
        'weight_decay': 0.05,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'drop': 0.0,
        'drop_path': 0.1,
        'train_interpolation': 'bicubic',
        'attention_normalization': 'vanilla',
        'attention_gamma': 1.0,
        'bcss_variant': 'e0',
        'psl_variant': 'baseline',
        'cti_bgt': False,
        'cam_scales': [1.0, 0.75, 1.25],
        'cam_class_to_patch_layers': 3,
        'cam_patch_to_patch_layers': 12,
        'checkpoint_policy': 'final',
    }
    for key, expected in expected_config.items():
        observed = {variant: bundle['config'].get(key) for variant, bundle in bundles.items()}
        check(f'config:{key}', all(value == expected for value in observed.values()), observed)
    for filename in (
            'train_aug_id.txt', 'train_id.txt', 'val_id.txt', 'cls_labels.npy'):
        observed = {
            variant: bundle['dataset_hashes'].get(filename)
            for variant, bundle in bundles.items()
        }
        check(
            f'dataset_hash:{filename}',
            len(set(observed.values())) == 1 and None not in observed.values(),
            observed,
        )
    parameter_counts = [bundles[value]['audit']['parameters']['total'] for value in VARIANTS]
    check('parameter_order', parameter_counts[0] < parameter_counts[1] < parameter_counts[2], parameter_counts)
    for variant, bundle in bundles.items():
        check(
            f'{variant}:classification_count',
            bundle['classification']['num_images'] == 1449,
            bundle['classification']['num_images'],
        )
        check(
            f'{variant}:class_order',
            bundle['classification'].get('class_names') == list(CLASS_NAMES),
            bundle['classification'].get('class_names'),
        )
        check(
            f'{variant}:final_checkpoint_name',
            bundle['checkpoint'].name
            == f'{EXPECTED_MODELS[variant]}_final.pth',
            bundle['checkpoint'].name,
        )
        check(
            f'{variant}:cam_counts',
            bundle['cam']['train']['num_images'] == 1464
            and bundle['cam']['val']['num_images'] == 1449,
            {
                split: bundle['cam'][split]['num_images']
                for split in ('train', 'val')
            },
        )
        check(
            f'{variant}:threshold_grid',
            len(bundle['cam']['train']['threshold_grid']) == 60
            and bundle['cam']['train']['threshold_grid']
            == bundle['cam']['val']['threshold_grid'],
            bundle['cam']['train']['threshold_grid'],
        )
        check(
            f'{variant}:semantic_count',
            bundle['semantic']['dataset']['num_images'] == 1449,
            bundle['semantic']['dataset']['num_images'],
        )
    small_tau = bundles['small']['cam']['train']['oracle_selection']['threshold']
    observed_tau = {'small': small_tau}
    for variant in ('tiny', 'base'):
        observed_tau[variant] = bundles[variant]['cam']['val'][
            'selected_metrics'
        ]['small_calibrated']['threshold']
    check(
        'small_calibrated_threshold_shared',
        len(set(observed_tau.values())) == 1,
        observed_tau,
    )
    failures = [item for item in checks if not item['passed']]
    if failures:
        raise RuntimeError(f'Cross-model comparability gate failed: {failures}')
    return {'passed': True, 'checks': checks, 'small_calibrated_threshold': small_tau}


def classification_bootstrap(bundles, resamples, seed):
    merged_scores = {}
    point = {}
    reference_ids = None
    reference_labels = None
    class_rows = []
    for variant, bundle in bundles.items():
        path = bundle['root'] / 'classification/classification_predictions.npz'
        with np.load(path, allow_pickle=False) as data:
            ids = data['image_ids']
            labels = data['labels']
            scores = {
                'class_token': data['class_token_scores'],
                'patch_gwrp': data['patch_gwrp_scores'],
            }
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        else:
            _require(np.array_equal(ids, reference_ids), 'Classification image order differs')
            _require(np.array_equal(labels, reference_labels), 'Classification labels differ')
        for branch, values in scores.items():
            name = f'{variant}__{branch}'
            merged_scores[name] = values
            point[name] = classification_metrics(labels, values)
        per_class_path = bundle['root'] / 'classification/classification_per_class.csv'
        with per_class_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                class_rows.append({'variant': variant, **row})
    assert reference_labels is not None
    _, samples = bootstrap_metrics(
        reference_labels, merged_scores, resamples, seed
    )
    rows = []
    for left, right in PAIR_ORDER:
        for branch in ('class_token', 'patch_gwrp'):
            for metric in ('macro_class_ap', 'micro_ap', 'legacy_mean_image_ap'):
                left_key = f'{left}__{branch}'
                right_key = f'{right}__{branch}'
                differences = (
                    samples[f'{left_key}_{metric}']
                    - samples[f'{right_key}_{metric}']
                )
                delta = point[left_key][metric] - point[right_key][metric]
                rows.append({
                    'domain': 'classification',
                    'split': 'val',
                    'selection': branch,
                    'metric': metric,
                    'contrast': f'{left}-{right}',
                    'delta': 100.0 * float(delta),
                    'ci95_low': 100.0 * float(np.quantile(differences, 0.025)),
                    'ci95_high': 100.0 * float(np.quantile(differences, 0.975)),
                    'unit': 'percentage_points',
                    'bootstrap_unit': 'image',
                    'bootstrap_resamples': resamples,
                    'bootstrap_seed': seed,
                })
    return point, rows, class_rows


def _confusion_sufficient_statistics(confusions):
    target = confusions.sum(axis=3)
    predicted = confusions.sum(axis=2)
    true_positive = np.diagonal(confusions, axis1=2, axis2=3)
    return true_positive.astype(np.float64), (
        target + predicted - true_positive
    ).astype(np.float64)


def cam_bootstrap(bundles, small_tau, resamples, seed):
    rows = []
    point_rows = []
    for split_index, split in enumerate(('train', 'val')):
        loaded = {}
        reference_ids = None
        reference_thresholds = None
        for variant, bundle in bundles.items():
            path = bundle['root'] / f'cam_evaluation_{split}/per_image_confusions.npz'
            with np.load(path, allow_pickle=False) as data:
                ids = data['image_ids']
                thresholds = data['thresholds'].astype(np.float64)
                confusions = data['confusions']
            if reference_ids is None:
                reference_ids, reference_thresholds = ids, thresholds
            else:
                _require(np.array_equal(ids, reference_ids), f'{split} CAM IDs differ')
                _require(np.array_equal(thresholds, reference_thresholds), f'{split} grids differ')
            loaded[variant] = confusions
        assert reference_ids is not None and reference_thresholds is not None
        selections = {'fixed_0.45': 0.45, 'small_calibrated': small_tau}
        generator = np.random.default_rng(seed + split_index)
        probabilities = np.full(len(reference_ids), 1.0 / len(reference_ids))
        weights_batches = []
        batch_size = 100
        for begin in range(0, resamples, batch_size):
            end = min(begin + batch_size, resamples)
            weights_batches.append(generator.multinomial(
                len(reference_ids), probabilities, size=end - begin
            ).astype(np.float64))
        for selection, threshold in selections.items():
            indices = np.flatnonzero(np.isclose(
                reference_thresholds, threshold, rtol=0, atol=1e-6
            ))
            _require(len(indices) == 1, f'{selection} threshold missing from grid')
            threshold_index = int(indices[0])
            samples = {}
            points = {}
            for variant in VARIANTS:
                selected = loaded[variant][:, threshold_index]
                point_metric = confusion_metrics(selected.sum(axis=0))['mean_iou']
                points[variant] = float(point_metric)
                tp, union = _confusion_sufficient_statistics(
                    loaded[variant][:, threshold_index:threshold_index + 1]
                )
                tp = tp[:, 0]
                union = union[:, 0]
                values = []
                for weights in weights_batches:
                    sampled_tp = weights @ tp
                    sampled_union = weights @ union
                    iou = np.divide(
                        sampled_tp, sampled_union,
                        out=np.full_like(sampled_tp, np.nan),
                        where=sampled_union != 0,
                    )
                    values.append(np.nanmean(iou, axis=1))
                samples[variant] = np.concatenate(values)
                point_rows.append({
                    'variant': variant,
                    'split': split,
                    'selection': selection,
                    'threshold': threshold,
                    'mean_iou_percent': 100.0 * points[variant],
                })
            for left, right in PAIR_ORDER:
                differences = samples[left] - samples[right]
                rows.append({
                    'domain': 'raw_cam',
                    'split': split,
                    'selection': selection,
                    'metric': 'mean_iou',
                    'contrast': f'{left}-{right}',
                    'delta': 100.0 * (points[left] - points[right]),
                    'ci95_low': 100.0 * float(np.quantile(differences, 0.025)),
                    'ci95_high': 100.0 * float(np.quantile(differences, 0.975)),
                    'unit': 'percentage_points',
                    'bootstrap_unit': 'image',
                    'bootstrap_resamples': resamples,
                    'bootstrap_seed': seed + split_index,
                })
    return rows, point_rows


def semantic_bootstrap(bundles, resamples, seed):
    selected_metrics = (
        'target_hit', 'other_fg_hit', 'background_hit', 'mixed_hit',
        'bg_top05_fraction', 'bg_top10_fraction',
        'target_mean', 'target_median', 'target_q90',
        'other_fg_mean', 'other_fg_median', 'other_fg_q90',
        'bg_mean', 'bg_median', 'bg_q90', 'bg_q95',
        'target_bg_mean_margin',
    )
    data = {}
    reference_ids = None
    reference_names = None
    for variant, bundle in bundles.items():
        path = bundle['root'] / (
            'semantic_ownership/per_image_metric_sufficient_statistics.npz'
        )
        with np.load(path, allow_pickle=False) as values:
            ids = values['image_ids']
            names = values['metric_names']
            sums = values['metric_sums']
            counts = values['metric_counts']
        if reference_ids is None:
            reference_ids, reference_names = ids, names
        else:
            _require(np.array_equal(ids, reference_ids), 'Semantic image IDs differ')
            _require(np.array_equal(names, reference_names), 'Semantic metric schema differs')
        data[variant] = (sums, counts)
    assert reference_ids is not None and reference_names is not None
    name_to_index = {str(name): index for index, name in enumerate(reference_names)}
    indices = [name_to_index[name] for name in selected_metrics]
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(reference_ids), 1.0 / len(reference_ids))
    weights_batches = []
    for begin in range(0, resamples, 100):
        end = min(begin + 100, resamples)
        weights_batches.append(generator.multinomial(
            len(reference_ids), probabilities, size=end - begin
        ).astype(np.float64))
    samples = {}
    points = {}
    for variant in VARIANTS:
        sums, counts = data[variant]
        numerator = sums[:, :, indices]
        denominator = counts[:, :, indices]
        point_denominator = denominator.sum(axis=0)
        points[variant] = np.divide(
            numerator.sum(axis=0),
            point_denominator,
            out=np.full_like(point_denominator, np.nan),
            where=point_denominator != 0,
        )
        flat_num = numerator.reshape(len(reference_ids), -1)
        flat_den = denominator.reshape(len(reference_ids), -1)
        chunks = []
        for weights in weights_batches:
            sample_num = weights @ flat_num
            sample_den = weights @ flat_den
            chunks.append(np.divide(
                sample_num,
                sample_den,
                out=np.full_like(sample_num, np.nan),
                where=sample_den != 0,
            ).reshape(len(weights), 12, len(selected_metrics)))
        samples[variant] = np.concatenate(chunks)

    rows = []
    paired = []
    probability_metrics = {
        'target_hit', 'other_fg_hit', 'background_hit', 'mixed_hit',
        'bg_top05_fraction', 'bg_top10_fraction',
    }
    for metric_index, metric in enumerate(selected_metrics):
        for variant in VARIANTS:
            for layer in range(12):
                rows.append({
                    'variant': variant,
                    'layer': layer + 1,
                    'metric': metric,
                    'value': float(points[variant][layer, metric_index]),
                    'ci95_low': float(np.nanquantile(
                        samples[variant][:, layer, metric_index], 0.025
                    )),
                    'ci95_high': float(np.nanquantile(
                        samples[variant][:, layer, metric_index], 0.975
                    )),
                    'unit': (
                        'fraction' if metric in probability_metrics
                        else 'raw_cosine_score'
                    ),
                })
        for left, right in PAIR_ORDER:
            for layer in range(12):
                differences = (
                    samples[left][:, layer, metric_index]
                    - samples[right][:, layer, metric_index]
                )
                scale = 100.0 if metric in probability_metrics else 1.0
                paired.append({
                    'domain': 'semantic_ownership',
                    'split': 'val',
                    'selection': f'layer_{layer + 1}',
                    'metric': metric,
                    'contrast': f'{left}-{right}',
                    'delta': scale * float(
                        points[left][layer, metric_index]
                        - points[right][layer, metric_index]
                    ),
                    'ci95_low': scale * float(
                        np.nanquantile(differences, 0.025)
                    ),
                    'ci95_high': scale * float(
                        np.nanquantile(differences, 0.975)
                    ),
                    'unit': (
                        'percentage_points' if scale == 100.0
                        else 'raw_cosine_score'
                    ),
                    'bootstrap_unit': 'image',
                    'bootstrap_resamples': resamples,
                    'bootstrap_seed': seed,
                })
    return rows, paired


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot write empty table: {path}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(output_dir, stem, figure):
    figure.tight_layout()
    figure.savefig(output_dir / f'{stem}.pdf')
    figure.savefig(output_dir / f'{stem}.png', dpi=180)
    plt.close(figure)


def make_plots(output_dir, summaries, bundles, semantic_rows):
    x = np.log10([summaries[v]['parameters'] for v in VARIANTS])
    labels = ['Tiny', 'Small', 'Base']

    def line_plot(stem, y, ylabel):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(x, y, marker='o')
        ax.set_xticks(x, labels)
        ax.set_xlabel('Model width / log10(parameters)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        _save_figure(output_dir, stem, fig)

    line_plot(
        '01_params_vs_classification_macro_ap',
        [summaries[v]['classification_macro_ap_percent'] for v in VARIANTS],
        'VOC val macro class AP (%)',
    )
    line_plot(
        '02_params_vs_val_cam_miou_fixed_threshold',
        [summaries[v]['val_miou_t045_percent'] for v in VARIANTS],
        'VOC val raw-CAM mIoU@0.45 (%)',
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, variant in zip(labels, VARIANTS):
        ax.scatter(
            summaries[variant]['classification_macro_ap_percent'],
            summaries[variant]['val_miou_t045_percent'], label=label,
        )
    ax.set_xlabel('Macro class AP (%)')
    ax.set_ylabel('Raw-CAM mIoU@0.45 (%)')
    ax.legend()
    ax.grid(alpha=.25)
    _save_figure(output_dir, '03_classification_vs_localization', fig)
    line_plot(
        '04_params_vs_c_pim',
        [summaries[v]['layer12_c_pim_percent'] for v in VARIANTS],
        'Layer-12 C-PiM (%)',
    )
    line_plot(
        '05_params_vs_bg_tail',
        [summaries[v]['layer12_bg_tail10_percent'] for v in VARIANTS],
        'Layer-12 BG-Tail@10 (%)',
    )
    for stem, metric, ylabel in (
        ('06_layerwise_c_pim', 'target_hit', 'C-PiM (%)'),
        ('07_layerwise_bg_tail', 'bg_top10_fraction', 'BG-Tail@10 (%)'),
    ):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for label, variant in zip(labels, VARIANTS):
            values = [
                row['value'] * 100 for row in semantic_rows
                if row['variant'] == variant and row['metric'] == metric
            ]
            ax.plot(range(1, 13), values, marker='o', ms=3, label=label)
        ax.set_xlabel('Transformer layer')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(alpha=.25)
        _save_figure(output_dir, stem, fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, variant in zip(labels, VARIANTS):
        ax.scatter(
            summaries[variant]['cam_latency_ms_mean'],
            summaries[variant]['val_miou_t045_percent'], label=label,
        )
    ax.set_xlabel('CAM latency, batch 1 (ms)')
    ax.set_ylabel('Raw-CAM mIoU@0.45 (%)')
    ax.legend()
    ax.grid(alpha=.25)
    _save_figure(output_dir, '08_accuracy_efficiency_frontier', fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, variant in zip(labels, VARIANTS):
        curve_path = bundles[variant]['root'] / 'cam_evaluation_val/threshold_curve.csv'
        with curve_path.open(newline='', encoding='utf-8') as stream:
            curve = list(csv.DictReader(stream))
        ax.plot(
            [float(row['threshold']) for row in curve],
            [float(row['mean_iou_percent']) for row in curve],
            label=label,
        )
    ax.set_xlabel('Background threshold')
    ax.set_ylabel('VOC val raw-CAM mIoU (%)')
    ax.legend()
    ax.grid(alpha=.25)
    _save_figure(output_dir, '09_cam_threshold_curves', fig)


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    if args.bootstrap_resamples < 1 or args.semantic_bootstrap_resamples < 1:
        raise ValueError('Bootstrap resample counts must be positive')
    small_source = args.small_run.expanduser().resolve()
    bundles = {
        'tiny': load_bundle('tiny', args.tiny_run),
        'small': load_bundle(
            'small', args.small_reanalysis, small_source=small_source
        ),
        'base': load_bundle('base', args.base_run),
    }
    audit = audit_comparability(bundles)
    small_tau = audit['small_calibrated_threshold']
    classification_point, classification_pairs, class_rows = (
        classification_bootstrap(
            bundles, args.bootstrap_resamples, args.bootstrap_seed
        )
    )
    cam_pairs, cam_points = cam_bootstrap(
        bundles, small_tau, args.bootstrap_resamples, args.bootstrap_seed
    )
    semantic_rows, semantic_pairs = semantic_bootstrap(
        bundles, args.semantic_bootstrap_resamples, args.bootstrap_seed
    )
    paired_rows = classification_pairs + cam_pairs + semantic_pairs

    cam_point_index = {
        (row['variant'], row['split'], row['selection']): row
        for row in cam_points
    }
    semantic_index = {
        (row['variant'], row['layer'], row['metric']): row
        for row in semantic_rows
    }
    summaries = {}
    summary_rows = []
    efficiency_rows = []
    cam_threshold_rows = []
    for variant, bundle in bundles.items():
        class_values = classification_point[f'{variant}__class_token']
        patch_values = classification_point[f'{variant}__patch_gwrp']
        train_fixed = cam_point_index[(variant, 'train', 'fixed_0.45')]
        train_cal = cam_point_index[(variant, 'train', 'small_calibrated')]
        val_fixed = cam_point_index[(variant, 'val', 'fixed_0.45')]
        val_cal = cam_point_index[(variant, 'val', 'small_calibrated')]
        summary = {
            'variant': variant,
            'model_name': EXPECTED_MODELS[variant],
            'parameters': bundle['audit']['parameters']['total'],
            'classification_macro_ap_percent': 100 * class_values['macro_class_ap'],
            'classification_micro_ap_percent': 100 * class_values['micro_ap'],
            'classification_legacy_image_ap_percent': 100 * class_values['legacy_mean_image_ap'],
            'patch_macro_ap_percent': 100 * patch_values['macro_class_ap'],
            'train_miou_t045_percent': train_fixed['mean_iou_percent'],
            'train_miou_small_calibrated_percent': train_cal['mean_iou_percent'],
            'val_miou_t045_percent': val_fixed['mean_iou_percent'],
            'val_miou_small_calibrated_percent': val_cal['mean_iou_percent'],
            'val_oracle_miou_percent': bundle['cam']['val']['oracle_selection']['mean_iou_percent'],
            'val_oracle_threshold': bundle['cam']['val']['oracle_selection']['threshold'],
            'layer12_c_pim_percent': 100 * semantic_index[(variant, 12, 'target_hit')]['value'],
            'layer12_bg_tail10_percent': 100 * semantic_index[(variant, 12, 'bg_top10_fraction')]['value'],
            'layer12_target_bg_margin': semantic_index[(variant, 12, 'target_bg_mean_margin')]['value'],
            'cam_latency_ms_mean': bundle['benchmark']['latency_ms_mean'],
            'cam_throughput_images_per_second': bundle['benchmark']['throughput_images_per_second'],
            'inference_peak_allocated_memory_mb': bundle['benchmark']['peak_allocated_memory_mb'],
            'training_runtime_available': bundle['training_runtime']['available'],
            'training_wall_seconds': bundle['training_runtime'].get(
                'wall_seconds_train_and_validation'
            ),
            'training_images_per_second': bundle['training_runtime'].get(
                'training_images_per_second'
            ),
            'optimizer_updates_per_second': bundle['training_runtime'].get(
                'optimizer_updates_per_second'
            ),
            'training_peak_allocated_memory_mb': (
                bundle['training_runtime'].get(
                    'training_peak_allocated_bytes', 0
                ) / (1024 ** 2)
                if bundle['training_runtime']['available'] else None
            ),
            'training_peak_reserved_memory_mb': (
                bundle['training_runtime'].get(
                    'training_peak_reserved_bytes', 0
                ) / (1024 ** 2)
                if bundle['training_runtime']['available'] else None
            ),
            'checkpoint_size_bytes': bundle['audit']['checkpoint']['size_bytes'],
            'micro_batch_size': bundle['config']['micro_batch_size'],
            'accum_iter': bundle['config']['accum_iter'],
            'effective_batch_size': bundle['config']['effective_batch_size'],
            'checkpoint_sha256': bundle['audit']['checkpoint']['sha256'],
            'run_dir': str(bundle['root']),
        }
        summaries[variant] = summary
        summary_rows.append(summary)
        efficiency_rows.append({
            key: summary[key] for key in (
                'variant', 'parameters', 'checkpoint_size_bytes',
                'cam_latency_ms_mean', 'cam_throughput_images_per_second',
                'inference_peak_allocated_memory_mb', 'micro_batch_size',
                'accum_iter', 'effective_batch_size',
                'training_runtime_available', 'training_wall_seconds',
                'training_images_per_second', 'optimizer_updates_per_second',
                'training_peak_allocated_memory_mb',
                'training_peak_reserved_memory_mb',
            )
        })
        for split in ('train', 'val'):
            metrics = bundle['cam'][split]
            cam_threshold_rows.append({
                'variant': variant,
                'split': split,
                'miou_t045_percent': cam_point_index[
                    (variant, split, 'fixed_0.45')
                ]['mean_iou_percent'],
                'miou_small_calibrated_percent': cam_point_index[
                    (variant, split, 'small_calibrated')
                ]['mean_iou_percent'],
                'small_calibrated_threshold': small_tau,
                'oracle_miou_percent': metrics['oracle_selection']['mean_iou_percent'],
                'oracle_threshold': metrics['oracle_selection']['threshold'],
                'curve_auc_normalized': metrics['threshold_curve_auc_normalized'],
                'plateau_width': metrics['plateau']['width'],
            })

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'command.txt').write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    _write_csv(args.output_dir / 'scaling_summary.csv', summary_rows)
    _write_csv(args.output_dir / 'classification_by_class.csv', class_rows)
    _write_csv(args.output_dir / 'cam_threshold_summary.csv', cam_threshold_rows)
    _write_csv(args.output_dir / 'paired_bootstrap_differences.csv', paired_rows)
    _write_csv(args.output_dir / 'efficiency_summary.csv', efficiency_rows)
    _write_csv(args.output_dir / 'semantic_ownership_summary.csv', semantic_rows)
    (args.output_dir / 'scaling_summary.json').write_text(
        json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    audit.update({
        'source_roots': {
            'tiny': str(bundles['tiny']['root']),
            'small_immutable': str(small_source),
            'small_reanalysis': str(bundles['small']['root']),
            'base': str(bundles['base']['root']),
        },
        'source_checkpoint_sha256': {
            variant: bundle['audit']['checkpoint']['sha256']
            for variant, bundle in bundles.items()
        },
        'bootstrap': {
            'classification_and_cam': {
                'unit': 'image', 'resamples': args.bootstrap_resamples,
                'seed': args.bootstrap_seed,
            },
            'semantic_ownership': {
                'unit': 'image',
                'resamples': args.semantic_bootstrap_resamples,
                'seed': args.bootstrap_seed,
            },
            'patches_or_image_class_pairs_independent': False,
            'training_seed_uncertainty_included': False,
        },
        'source_inputs_modified': False,
    })
    (args.output_dir / 'audit_report.json').write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    make_plots(args.output_dir, summaries, bundles, semantic_rows)

    def delta(domain, contrast, selection, split, metric):
        matches = [
            row for row in paired_rows
            if row['domain'] == domain and row['contrast'] == contrast
            and row['selection'] == selection and row['split'] == split
            and row['metric'] == metric
        ]
        return matches[0] if len(matches) == 1 else None

    report = [
        '# MCTformer+ Tiny / Small / Base Width-Scaling Report',
        '',
        '## Scope and integrity',
        '',
        '- All three hosts use 12 blocks, patch size 16, vanilla attention, BCSS E0, PSL baseline, CTI-BGT disabled, seed 0, effective batch 32, 45 epochs, and the final checkpoint.',
        '- The canonical Small checkpoint and CAM tree were read only; Small was not retrained.',
        '- Confidence intervals resample whole images. They quantify evaluation-set sampling uncertainty, not training-seed variance.',
        '- These three width points support only a width/capacity trend analysis; they do not establish a scaling law.',
        '',
        '## Primary results',
        '',
        '| Variant | Params | Macro AP | Train mIoU@.45 | Val mIoU@.45 | Val mIoU@tau_S | Val oracle | L12 C-PiM | L12 BG-tail@10 | CAM ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        value = summaries[variant]
        report.append(
            f"| {variant.title()} | {value['parameters']:,} | "
            f"{value['classification_macro_ap_percent']:.2f} | "
            f"{value['train_miou_t045_percent']:.2f} | "
            f"{value['val_miou_t045_percent']:.2f} | "
            f"{value['val_miou_small_calibrated_percent']:.2f} | "
            f"{value['val_oracle_miou_percent']:.2f} @ {value['val_oracle_threshold']:.2f} | "
            f"{value['layer12_c_pim_percent']:.2f} | "
            f"{value['layer12_bg_tail10_percent']:.2f} | "
            f"{value['cam_latency_ms_mean']:.2f} |"
        )
    report.extend([
        '',
        f"Small-calibrated threshold: `{small_tau:.2f}` (selected exhaustively on canonical Small VOC train CAMs before Tiny/Base val comparison).",
        '',
        '## Paired uncertainty',
        '',
    ])
    for contrast in ('tiny-small', 'base-small', 'base-tiny'):
        ap = delta('classification', contrast, 'class_token', 'val', 'macro_class_ap')
        cam = delta('raw_cam', contrast, 'fixed_0.45', 'val', 'mean_iou')
        if ap and cam:
            report.append(
                f"- {contrast}: macro AP delta {ap['delta']:.2f} "
                f"(95% CI {ap['ci95_low']:.2f}, {ap['ci95_high']:.2f}); "
                f"val CAM mIoU@.45 delta {cam['delta']:.2f} "
                f"(95% CI {cam['ci95_low']:.2f}, {cam['ci95_high']:.2f})."
            )
    report.extend([
        '',
        '## Interpretation boundary',
        '',
        'Classification, localization, feature-score semantic ownership, and efficiency are reported as separate measured outcomes. Any relationship among them is descriptive for these frozen seed-0 checkpoints. A paper-level claim requires the pre-registered seed-1/2 gate and cannot be inferred from image bootstrap alone.',
        '',
    ])
    (args.output_dir / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    (args.output_dir / 'AGGREGATION_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(summaries, sort_keys=True))
    return summaries, audit


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
