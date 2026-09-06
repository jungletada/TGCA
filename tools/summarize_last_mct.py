#!/usr/bin/env python3
"""Build the compact matched Original/PatchFinalLN/Last-MCT result report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path


MODELS = (
    ('Original MCTformer+', 'baseline'),
    ('PatchFinalLN MCTformer+', 'patch_final_ln'),
    ('Last-MCT', 'last_mct'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-root', type=Path, required=True)
    parser.add_argument('--patch-run-root', type=Path, required=True)
    parser.add_argument('--last-run-root', type=Path, required=True)
    parser.add_argument('--expected-source-summary-sha256', required=True)
    parser.add_argument('--expected-patch-summary-sha256', required=True)
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


def _write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f'cannot write empty table {path}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _require_complete(root, markers):
    for marker in markers:
        path = root / marker
        if not path.is_file():
            raise FileNotFoundError(path)


def _validate_reference(root, expected_hash, checkpoint_key=None):
    _require_complete(root, ('PIPELINE_COMPLETE', 'EXPERIMENT_COMPLETE'))
    summary_path = root / 'summary.json'
    observed_hash = _sha256(summary_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f'immutable source summary hash mismatch: {summary_path}'
        )
    summary = _read_json(summary_path)
    if summary.get('status') != 'complete':
        raise RuntimeError(f'incomplete immutable source: {root}')
    checkpoint_info = (
        summary['source_checkpoints'][checkpoint_key]
        if checkpoint_key is not None else summary['patch_checkpoint']
    )
    checkpoint = Path(checkpoint_info['path'])
    if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_info['sha256']:
        raise RuntimeError(f'immutable source checkpoint hash mismatch: {checkpoint}')
    return summary, checkpoint_info


def _validate_last(root, patch_root):
    checkpoint = root / 'checkpoints/last_mct/mctformerplus_final.pth'
    audit_path = root / 'audit/last_mct.json'
    classification_path = (
        root / 'evaluations/last_mct/classification/classification_metrics.json'
    )
    cam_path = root / 'evaluations/last_mct/cam_evaluation/metrics.json'
    required = (
        checkpoint,
        audit_path,
        classification_path,
        cam_path,
        root / 'evaluations/last_mct/classification/CLASSIFICATION_COMPLETE',
        root / 'evaluations/last_mct/cam_train/CAM_COMPLETE',
        root / 'evaluations/last_mct/cam_evaluation/THRESHOLD_EVALUATION_COMPLETE',
        root / 'smoke/SMOKE_COMPLETE',
        root / 'training_logs/tests_complete.txt',
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = _read_json(audit_path)
    if not audit.get('passed'):
        raise RuntimeError('Last-MCT checkpoint audit failed')
    method = audit['method_configuration']
    expected_configuration = {
        'enabled': True,
        'topk': 1,
        'sigma': math.sqrt(384),
        'eps': 1e-6,
        'score_formula': 'P / abs(P_lowpass - P).clamp_min(eps)',
        'fft_dimension': 'embedding',
        'topk_dimension': 'patch',
        'selection_semantics': 'independent patch index per embedding channel',
        'patch_final_norm': True,
        'classifier': 'shared Conv2d(D, 20, kernel_size=1) / F.linear',
    }
    if (
        method.get('final_norm')
        or not method.get('patch_final_norm')
        or not method.get('last_mct')
        or method.get('last_mct_configuration') != expected_configuration
    ):
        raise RuntimeError('Last-MCT method configuration mismatch')
    if audit['checkpoint']['sha256'] != _sha256(checkpoint):
        raise RuntimeError('Last-MCT checkpoint audit hash mismatch')
    if (
        _read_json(classification_path)['protocol'].get('patch_branch')
        != 'patch_last_mct'
    ):
        raise RuntimeError('Last-MCT classification branch metadata mismatch')

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
            'last_mct': audit['training_spec'].get(key),
        }
        for key in contract_keys
        if patch_audit['training_spec'].get(key) != audit['training_spec'].get(key)
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
    last_optimizer = _read_json(root / 'checkpoints/last_mct/optimizer_spec.json')
    optimizer_keys = (
        'optimizer', 'weight_decay', 'epsilon', 'betas', 'schedule',
        'warmup_epochs', 'minimum_lr', *contract_keys,
    )
    optimizer_mismatches = {
        key: {
            'patch_final_ln': patch_optimizer.get(key),
            'last_mct': last_optimizer.get(key),
        }
        for key in optimizer_keys
        if patch_optimizer.get(key) != last_optimizer.get(key)
    }
    if optimizer_mismatches:
        raise RuntimeError(
            f'matched optimizer/config mismatch: {optimizer_mismatches}'
        )
    return (
        checkpoint,
        audit,
        _read_json(classification_path),
        _read_json(cam_path),
    )


def _classification_row(label, classification, patch_branch):
    metrics = classification['metrics_percent']
    losses = classification['classification_loss']
    return {
        'model': label,
        'class_token_macro_map_percent': metrics['class_token']['macro_class_ap'],
        'patch_head_macro_map_percent': metrics[patch_branch]['macro_class_ap'],
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


def execute(args):
    source = args.source_run_root.resolve()
    patch = args.patch_run_root.resolve()
    root = args.last_run_root.resolve()
    if len({source, patch, root}) != 3:
        raise ValueError('source, PatchFinalLN, and Last-MCT roots must differ')
    targets = (
        'classification_results.csv', 'cam_results.csv',
        'comparison_summary.csv', 'LAST_MCT_REPORT.md', 'summary.json',
        'EXPERIMENT_COMPLETE',
    )
    existing = [name for name in targets if (root / name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite summary artifacts: {existing}')

    source_summary, baseline_checkpoint = _validate_reference(
        source, args.expected_source_summary_sha256, checkpoint_key='baseline'
    )
    patch_summary, patch_checkpoint = _validate_reference(
        patch, args.expected_patch_summary_sha256
    )
    last_checkpoint, last_audit, last_classification, last_cam = (
        _validate_last(root, patch)
    )

    baseline_classification = _read_json(
        source / 'evaluations/baseline/classification/classification_metrics.json'
    )
    patch_classification = _read_json(
        patch / 'evaluations/patch_final_ln/classification/classification_metrics.json'
    )
    baseline_cam = _read_json(
        source / 'evaluations/baseline/cam_evaluation/metrics.json'
    )
    patch_cam = _read_json(
        patch / 'evaluations/patch_final_ln/cam_evaluation/metrics.json'
    )

    classification_rows = (
        _classification_row(
            MODELS[0][0], baseline_classification, 'patch_gwrp'
        ),
        _classification_row(
            MODELS[1][0], patch_classification, 'patch_gwrp'
        ),
        _classification_row(
            MODELS[2][0], last_classification, 'patch_last_mct'
        ),
    )
    cam_rows = (
        _cam_row(MODELS[0][0], baseline_cam),
        _cam_row(MODELS[1][0], patch_cam),
        _cam_row(MODELS[2][0], last_cam),
    )
    cam_by_model = {row['model']: row for row in cam_rows}
    comparison_rows = tuple({
        **classification,
        **{
            key: value for key, value in cam_by_model[classification['model']].items()
            if key != 'model'
        },
    } for classification in classification_rows)

    _write_csv(root / 'classification_results.csv', classification_rows)
    _write_csv(root / 'cam_results.csv', cam_rows)
    _write_csv(root / 'comparison_summary.csv', comparison_rows)

    def fmt(value):
        return f'{float(value):.6f}'

    table_lines = [
        '| Model | Class mAP (%) | Patch mAP (%) | Class val loss | Patch val loss | Fixed@0.45 CAM mIoU (%) | Best CAM mIoU (%) | Best threshold | FG precision@0.45 (%) | FG recall@0.45 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in comparison_rows:
        table_lines.append(
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

    checkpoint_hash = _sha256(last_checkpoint)
    git_commit = _read_json(root / 'git_state.json')['commit']
    report = '\n'.join([
        '# Last-MCT matched VOC result',
        '',
        *table_lines,
        '',
        'The best threshold is reported only from the same fixed '
        '`0.00:0.01:0.59` sensitivity sweep. Precision and recall use the '
        'prespecified fixed threshold `0.45`.',
        '',
        '## Reproducibility',
        '',
        f'- Git commit: `{git_commit}`',
        f'- Last-MCT checkpoint SHA256: `{checkpoint_hash}`',
        '- Configuration: `config.json`',
        '- Exact commands: `exact_commands.sh`',
        '- Test status: `training_logs/tests_complete.txt`',
        f'- Original source summary SHA256: `{args.expected_source_summary_sha256}`',
        f'- PatchFinalLN source summary SHA256: `{args.expected_patch_summary_sha256}`',
        '',
    ])
    (root / 'LAST_MCT_REPORT.md').write_text(report, encoding='utf-8')

    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'Last-MCT matched VOC seed-0',
        'method_configuration': last_audit['method_configuration'],
        'classification_results': list(classification_rows),
        'cam_results': list(cam_rows),
        'comparison': list(comparison_rows),
        'last_checkpoint': {
            'path': str(last_checkpoint.resolve()),
            'sha256': checkpoint_hash,
        },
        'immutable_references': {
            'original': {
                'run_root': str(source),
                'summary_sha256': args.expected_source_summary_sha256,
                'checkpoint': baseline_checkpoint,
            },
            'patch_final_ln': {
                'run_root': str(patch),
                'summary_sha256': args.expected_patch_summary_sha256,
                'checkpoint': patch_checkpoint,
            },
        },
        'matched_training_spec': last_audit['training_spec'],
        'commands': shlex.join([sys.executable] + sys.argv),
        'source_status': {
            'original': source_summary['status'],
            'patch_final_ln': patch_summary['status'],
        },
    }
    (root / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (root / 'EXPERIMENT_COMPLETE').write_text('complete\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
