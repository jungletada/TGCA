#!/usr/bin/env python3
"""Build the compact four-way Class-Stable LaST matched result report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-root', type=Path, required=True)
    parser.add_argument('--patch-run-root', type=Path, required=True)
    parser.add_argument('--last-run-root', type=Path, required=True)
    parser.add_argument('--class-stable-run-root', type=Path, required=True)
    parser.add_argument('--expected-source-summary-sha256', required=True)
    parser.add_argument('--expected-patch-summary-sha256', required=True)
    parser.add_argument('--expected-last-summary-sha256', required=True)
    return parser.parse_args()


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require_files(paths):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def _write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f'cannot write empty table {path}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference(root, expected_hash, checkpoint_selector):
    _require_files((
        root / 'PIPELINE_COMPLETE',
        root / 'EXPERIMENT_COMPLETE',
        root / 'summary.json',
    ))
    if _sha256(root / 'summary.json') != expected_hash:
        raise RuntimeError(f'immutable source summary hash mismatch: {root}')
    summary = _read_json(root / 'summary.json')
    if summary.get('status') != 'complete':
        raise RuntimeError(f'immutable source is incomplete: {root}')
    checkpoint = checkpoint_selector(summary)
    checkpoint_path = Path(checkpoint['path'])
    if (
        not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != checkpoint['sha256']
    ):
        raise RuntimeError(
            f'immutable source checkpoint hash mismatch: {checkpoint_path}'
        )
    return summary, checkpoint


def _classification_row(label, metrics, patch_branch):
    values = metrics['metrics_percent']
    losses = metrics['classification_loss']
    return {
        'model': label,
        'class_token_macro_map_percent': values[
            'class_token'
        ]['macro_class_ap'],
        'patch_head_macro_map_percent': values[
            patch_branch
        ]['macro_class_ap'],
        'class_token_validation_loss': losses[
            'class_token_multilabel_soft_margin_mean'
        ],
        'patch_head_validation_loss': losses[
            f'{patch_branch}_multilabel_soft_margin_mean'
        ],
    }


def _cam_row(label, metrics):
    fixed = metrics['selected_metrics']['fixed_0.45']
    best = metrics['selected_metrics']['oracle']
    return {
        'model': label,
        'fixed_threshold': fixed['threshold'],
        'fixed_raw_cam_miou_percent': fixed['mean_iou_percent'],
        'best_raw_cam_miou_percent': best['mean_iou_percent'],
        'best_threshold': best['threshold'],
        'foreground_precision_percent_at_fixed_threshold': fixed[
            'semantic_foreground_precision_percent'
        ],
        'foreground_recall_percent_at_fixed_threshold': fixed[
            'semantic_foreground_recall_percent'
        ],
    }


def _validate_new_run(root, patch_root):
    checkpoint = (
        root / 'checkpoints/class_stable_last/mctformerplus_final.pth'
    )
    audit_path = root / 'audit/class_stable_last.json'
    classification_path = (
        root
        / 'evaluations/class_stable_last/classification/classification_metrics.json'
    )
    cam_path = (
        root / 'evaluations/class_stable_last/cam_evaluation/metrics.json'
    )
    _require_files((
        checkpoint,
        audit_path,
        classification_path,
        cam_path,
        root / 'smoke/SMOKE_COMPLETE',
        root / 'training_logs/tests_complete.txt',
        root / 'evaluations/class_stable_last/classification/CLASSIFICATION_COMPLETE',
        root / 'evaluations/class_stable_last/cam_train/CAM_COMPLETE',
        root / 'evaluations/class_stable_last/cam_evaluation/THRESHOLD_EVALUATION_COMPLETE',
    ))
    audit = _read_json(audit_path)
    if not audit.get('passed'):
        raise RuntimeError('Class-Stable LaST checkpoint audit failed')
    expected_configuration = {
        'enabled': True,
        'topk': 1,
        'sigma': math.sqrt(384),
        'eps': 1e-6,
        'score_formula': 'M / abs(M_lowpass - M).clamp_min(eps)',
        'fft_dimension': 'embedding',
        'topk_dimension': 'spatial',
        'selection_semantics': 'independent spatial index per semantic class',
        'gathered_values': 'original M',
        'lowpass_role': 'selector only',
        'patch_final_norm': True,
        'classifier': 'shared Conv2d(D, 20, kernel_size=3, padding=1)',
        'classification_map': 'original M',
        'cam_map': 'original M',
    }
    method = audit['method_configuration']
    if (
        method.get('final_norm')
        or not method.get('patch_final_norm')
        or method.get('last_mct')
        or not method.get('class_stable_last')
        or method.get('class_stable_last_configuration')
        != expected_configuration
    ):
        raise RuntimeError('Class-Stable LaST method configuration mismatch')
    if audit['checkpoint']['sha256'] != _sha256(checkpoint):
        raise RuntimeError('Class-Stable LaST checkpoint audit hash mismatch')
    classification = _read_json(classification_path)
    if (
        classification['protocol'].get('patch_branch')
        != 'patch_class_stable_last'
    ):
        raise RuntimeError('Class-Stable classification branch mismatch')

    patch_audit = _read_json(patch_root / 'audit/patch_final_ln.json')
    contract_keys = (
        'seed', 'micro_batch_size', 'accum_iter', 'world_size',
        'effective_batch_size', 'nominal_lr', 'optimizer_lr', 'epochs',
        'optimizer_updates_per_epoch', 'consumed_samples_per_epoch',
        'train_dataset_size', 'val_batch_size',
    )
    mismatches = {
        key: {
            'patch_final_ln': patch_audit['training_spec'].get(key),
            'class_stable_last': audit['training_spec'].get(key),
        }
        for key in contract_keys
        if patch_audit['training_spec'].get(key)
        != audit['training_spec'].get(key)
    }
    if mismatches:
        raise RuntimeError(f'matched training contract mismatch: {mismatches}')
    if (
        patch_audit['official_pretrained']['source_sha256']
        != audit['official_pretrained']['source_sha256']
    ):
        raise RuntimeError('matched pretrained initialization hash mismatch')

    patch_optimizer = _read_json(
        patch_root / 'checkpoints/patch_final_ln/optimizer_spec.json'
    )
    new_optimizer = _read_json(
        root / 'checkpoints/class_stable_last/optimizer_spec.json'
    )
    optimizer_keys = (
        'optimizer', 'weight_decay', 'epsilon', 'betas', 'schedule',
        'warmup_epochs', 'minimum_lr', *contract_keys,
    )
    optimizer_mismatches = {
        key: {
            'patch_final_ln': patch_optimizer.get(key),
            'class_stable_last': new_optimizer.get(key),
        }
        for key in optimizer_keys
        if patch_optimizer.get(key) != new_optimizer.get(key)
    }
    if optimizer_mismatches:
        raise RuntimeError(
            f'matched optimizer/config mismatch: {optimizer_mismatches}'
        )
    return checkpoint, audit, classification, _read_json(cam_path)


def execute(args):
    source = args.source_run_root.resolve()
    patch = args.patch_run_root.resolve()
    previous_last = args.last_run_root.resolve()
    root = args.class_stable_run_root.resolve()
    if len({source, patch, previous_last, root}) != 4:
        raise ValueError('all four run roots must differ')
    targets = (
        'classification_results.csv', 'cam_results.csv',
        'comparison_summary.csv', 'CLASS_STABLE_LAST_REPORT.md',
        'summary.json', 'EXPERIMENT_COMPLETE',
    )
    existing = [name for name in targets if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f'Refusing to overwrite summary artifacts: {existing}'
        )

    source_summary, original_checkpoint = _reference(
        source,
        args.expected_source_summary_sha256,
        lambda summary: summary['source_checkpoints']['baseline'],
    )
    patch_summary, patch_checkpoint = _reference(
        patch,
        args.expected_patch_summary_sha256,
        lambda summary: summary['patch_checkpoint'],
    )
    last_summary, last_checkpoint = _reference(
        previous_last,
        args.expected_last_summary_sha256,
        lambda summary: summary['last_checkpoint'],
    )
    new_checkpoint, new_audit, new_classification, new_cam = (
        _validate_new_run(root, patch)
    )

    models = (
        (
            'Original MCTformer+',
            _read_json(source / 'evaluations/baseline/classification/classification_metrics.json'),
            'patch_gwrp',
            _read_json(source / 'evaluations/baseline/cam_evaluation/metrics.json'),
        ),
        (
            'PatchFinalLN MCTformer+',
            _read_json(patch / 'evaluations/patch_final_ln/classification/classification_metrics.json'),
            'patch_gwrp',
            _read_json(patch / 'evaluations/patch_final_ln/cam_evaluation/metrics.json'),
        ),
        (
            'Last-MCT (pool before classifier)',
            _read_json(previous_last / 'evaluations/last_mct/classification/classification_metrics.json'),
            'patch_last_mct',
            _read_json(previous_last / 'evaluations/last_mct/cam_evaluation/metrics.json'),
        ),
        (
            'Class-Stable LaST Pooling',
            new_classification,
            'patch_class_stable_last',
            new_cam,
        ),
    )
    classification_rows = tuple(
        _classification_row(label, classification, branch)
        for label, classification, branch, _ in models
    )
    cam_rows = tuple(
        _cam_row(label, cam) for label, _, _, cam in models
    )
    cam_by_model = {row['model']: row for row in cam_rows}
    comparison_rows = tuple({
        **classification,
        **{
            key: value
            for key, value in cam_by_model[classification['model']].items()
            if key != 'model'
        },
    } for classification in classification_rows)
    _write_csv(root / 'classification_results.csv', classification_rows)
    _write_csv(root / 'cam_results.csv', cam_rows)
    _write_csv(root / 'comparison_summary.csv', comparison_rows)

    def fmt(value):
        return f'{float(value):.6f}'

    table = [
        '| Model | Class mAP (%) | Patch mAP (%) | Class val loss | Patch val loss | Fixed@0.45 CAM mIoU (%) | Best CAM mIoU (%) | Best threshold | FG precision@0.45 (%) | FG recall@0.45 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in comparison_rows:
        table.append(
            f"| {row['model']} | "
            f"{fmt(row['class_token_macro_map_percent'])} | "
            f"{fmt(row['patch_head_macro_map_percent'])} | "
            f"{fmt(row['class_token_validation_loss'])} | "
            f"{fmt(row['patch_head_validation_loss'])} | "
            f"{fmt(row['fixed_raw_cam_miou_percent'])} | "
            f"{fmt(row['best_raw_cam_miou_percent'])} | "
            f"{fmt(row['best_threshold'])} | "
            f"{fmt(row['foreground_precision_percent_at_fixed_threshold'])} | "
            f"{fmt(row['foreground_recall_percent_at_fixed_threshold'])} |"
        )
    checkpoint_hash = _sha256(new_checkpoint)
    git_commit = _read_json(root / 'git_state.json')['commit']
    report = '\n'.join((
        '# Class-Stable LaST Pooling matched VOC result',
        '',
        *table,
        '',
        'The best threshold is reported from the same `0.00:0.01:0.59` '
        'sensitivity sweep. Precision and recall use the prespecified fixed '
        'threshold `0.45`.',
        '',
        '## Reproducibility',
        '',
        f'- Git commit: `{git_commit}`',
        f'- Class-Stable LaST checkpoint SHA256: `{checkpoint_hash}`',
        '- Configuration: `config.json`',
        '- Exact commands: `exact_commands.sh`',
        '- Test status: `training_logs/tests_complete.txt`',
        f'- Original summary SHA256: `{args.expected_source_summary_sha256}`',
        f'- PatchFinalLN summary SHA256: `{args.expected_patch_summary_sha256}`',
        f'- Previous Last-MCT summary SHA256: `{args.expected_last_summary_sha256}`',
        '',
    ))
    (root / 'CLASS_STABLE_LAST_REPORT.md').write_text(
        report, encoding='utf-8'
    )
    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'Class-Stable LaST Pooling matched VOC seed-0',
        'method_configuration': new_audit['method_configuration'],
        'classification_results': list(classification_rows),
        'cam_results': list(cam_rows),
        'comparison': list(comparison_rows),
        'class_stable_checkpoint': {
            'path': str(new_checkpoint.resolve()),
            'sha256': checkpoint_hash,
        },
        'immutable_references': {
            'original': {
                'run_root': str(source),
                'summary_sha256': args.expected_source_summary_sha256,
                'checkpoint': original_checkpoint,
            },
            'patch_final_ln': {
                'run_root': str(patch),
                'summary_sha256': args.expected_patch_summary_sha256,
                'checkpoint': patch_checkpoint,
            },
            'last_mct': {
                'run_root': str(previous_last),
                'summary_sha256': args.expected_last_summary_sha256,
                'checkpoint': last_checkpoint,
            },
        },
        'matched_training_spec': new_audit['training_spec'],
        'command': shlex.join([sys.executable] + sys.argv),
        'source_status': {
            'original': source_summary['status'],
            'patch_final_ln': patch_summary['status'],
            'last_mct': last_summary['status'],
        },
    }
    (root / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (root / 'EXPERIMENT_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
