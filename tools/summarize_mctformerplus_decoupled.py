#!/usr/bin/env python3
"""Create compact matched results for the decoupled MCTformer+ queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VARIANTS = (
    'full',
    'no_p2c',
    'no_c2c',
    'no_p2c_no_c2c',
    'c2p_update_off',
    'p2p_update_off',
    'p2c_middle',
    'p2c_early',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _row(name, topology, training_root, evaluation_root, audit_path):
    classification_path = (
        evaluation_root / 'classification' / 'classification_metrics.json'
    )
    cam_path = evaluation_root / 'cam_evaluation' / 'metrics.json'
    runtime_path = training_root / 'training_runtime.json'
    checkpoint_path = training_root / 'mctformerplus_final.pth'
    required = (
        classification_path, cam_path, runtime_path, checkpoint_path, audit_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing completed result files: {missing}')
    classification = _load(classification_path)
    cam = _load(cam_path)
    runtime = _load(runtime_path)
    audit = _load(audit_path)
    if not classification.get('finite') or not cam.get('finite'):
        raise ValueError(f'Non-finite result recorded for {name}')
    if not audit.get('passed'):
        raise ValueError(f'Checkpoint audit did not pass for {name}')
    fixed = cam['selected_metrics']['fixed_0.45']
    oracle = cam['selected_metrics']['oracle']
    class_metrics = classification['metrics_percent']['class_token']
    patch_metrics = classification['metrics_percent']['patch_gwrp']
    losses = classification['classification_loss']
    return {
        'variant': name,
        'token_interaction': topology,
        'class_token_macro_map_percent': class_metrics['macro_class_ap'],
        'patch_head_macro_map_percent': patch_metrics['macro_class_ap'],
        'class_token_validation_loss': (
            losses['class_token_multilabel_soft_margin_mean']
        ),
        'patch_validation_loss': (
            losses['patch_gwrp_multilabel_soft_margin_mean']
        ),
        'fixed_0.45_raw_cam_miou_percent': fixed['mean_iou_percent'],
        'best_raw_cam_miou_percent': oracle['mean_iou_percent'],
        'best_threshold': oracle['threshold'],
        'fixed_binary_fg_precision_percent': (
            fixed['binary_foreground_precision_percent']
        ),
        'fixed_binary_fg_recall_percent': (
            fixed['binary_foreground_recall_percent']
        ),
        'parameters': audit['parameters']['total'],
        'training_seconds': runtime['training_seconds'],
        'training_images_per_second': runtime['training_images_per_second'],
        'training_peak_allocated_gib': (
            runtime['training_peak_allocated_bytes'] / (1024 ** 3)
        ),
        'checkpoint_sha256': _sha256(checkpoint_path),
    }


def execute(args):
    run_root = args.run_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_csv = run_root / 'decoupled_results.csv'
    output_summary = run_root / 'summary.json'
    output_report = run_root / 'DECOUPLED_ALTERNATING_REPORT.md'
    completion = run_root / 'EXPERIMENT_COMPLETE'
    for output in (output_csv, output_summary, output_report, completion):
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')

    rows = [
        _row(
            'original_joint_baseline',
            'joint',
            baseline_root / 'checkpoints' / 'baseline',
            baseline_root / 'evaluations' / 'baseline',
            baseline_root / 'audit' / 'baseline.json',
        )
    ]
    for variant in VARIANTS:
        root = run_root / 'variants' / variant
        rows.append(_row(
            variant,
            'decoupled_bidirectional',
            root / 'checkpoints',
            root / 'evaluations',
            root / 'audit.json',
        ))

    with output_csv.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'schema_version': 1,
        'status': 'complete',
        'dataset': 'PASCAL VOC 2012',
        'model': 'MCTformer+-Small',
        'seed': 0,
        'matched_variants': list(VARIANTS),
        'baseline_root': str(baseline_root),
        'run_root': str(run_root),
        'rows': rows,
        'interpretation_status': (
            'measurements_complete; scientific interpretation not automated'
        ),
    }
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )

    columns = (
        ('variant', 'Variant'),
        ('class_token_macro_map_percent', 'Class mAP'),
        ('patch_head_macro_map_percent', 'Patch mAP'),
        ('fixed_0.45_raw_cam_miou_percent', 'CAM@0.45'),
        ('best_raw_cam_miou_percent', 'Best CAM'),
        ('best_threshold', 'Best th.'),
        ('fixed_binary_fg_precision_percent', 'FG Prec.'),
        ('fixed_binary_fg_recall_percent', 'FG Rec.'),
    )
    lines = [
        '# Decoupled Alternating MCTformer+ Results',
        '',
        'All rows use the matched VOC seed-0 protocol. The joint baseline is '
        'read-only and was not retrained by this queue.',
        '',
        '| ' + ' | '.join(label for _, label in columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(value if isinstance(value, str) else f'{value:.4f}')
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.extend((
        '',
        'This report records measurements only. It does not assign a causal '
        'interpretation to concatenation removal or any individual relation.',
        '',
    ))
    output_report.write_text('\n'.join(lines), encoding='utf-8')
    completion.write_text('complete\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
