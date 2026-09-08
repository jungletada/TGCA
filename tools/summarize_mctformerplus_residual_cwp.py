#!/usr/bin/env python3
"""Summarize matched Residual-CWP runs across the fixed epoch schedules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


EPOCHS = (25, 30, 35, 45)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--expected-baseline-summary-sha256', required=True)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def cam_values(payload):
    fixed = payload['selected_metrics']['fixed_0.45']
    best = payload['selected_metrics']['oracle']
    return {
        'fixed_threshold': fixed['threshold'],
        'fixed_native_cam_miou_percent': fixed['mean_iou_percent'],
        'fixed_foreground_precision_percent': (
            fixed['binary_foreground_precision_percent']
        ),
        'fixed_foreground_recall_percent': (
            fixed['binary_foreground_recall_percent']
        ),
        'best_threshold': best['threshold'],
        'best_native_cam_miou_percent': best['mean_iou_percent'],
        'best_foreground_precision_percent': (
            best['binary_foreground_precision_percent']
        ),
        'best_foreground_recall_percent': (
            best['binary_foreground_recall_percent']
        ),
    }


def performance_row(method, epochs, classification, cam):
    losses = classification['classification_loss']
    return {
        'method': method,
        'epochs': epochs,
        'class_token_macro_map_percent': classification[
            'metrics_percent'
        ]['class_token']['macro_class_ap'],
        'patch_head_macro_map_percent': classification[
            'metrics_percent'
        ]['patch_gwrp']['macro_class_ap'],
        'class_token_validation_loss': losses[
            'class_token_multilabel_soft_margin_mean'
        ],
        'patch_validation_loss': losses[
            'patch_gwrp_multilabel_soft_margin_mean'
        ],
        **cam_values(cam),
        'pseudo_mask_metric': 'unavailable_in_matched_B0_pipeline',
        'downstream_segmentation_metric': 'unavailable_in_matched_B0_pipeline',
    }


def training_diagnostics(path: Path, epochs: int):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('{"epoch"'):
            continue
        payload = json.loads(line)
        row = {
            'training_epochs': epochs,
            'epoch': payload['epoch'],
            'alpha_batch_mean': payload.get('train_residual_cwp_alpha'),
            'alpha_end': payload.get('train_residual_cwp_alpha_end'),
            'pooling_entropy_mean': payload.get('train_cwp_entropy_mean'),
            'interclass_pooling_attention_cosine': payload.get(
                'train_cwp_attention_interclass_cosine'
            ),
            'attention_row_sum_max_error': payload.get(
                'train_cwp_attention_row_sum_max_error'
            ),
        }
        if any(value is None for value in row.values()):
            raise RuntimeError(
                f'missing Residual-CWP diagnostics at epoch {payload["epoch"]}'
            )
        rows.append(row)
    if len(rows) != epochs:
        raise RuntimeError(
            f'expected {epochs} diagnostic rows in {path}, found {len(rows)}'
        )
    return rows


def fmt(value, digits=3):
    return f'{float(value):.{digits}f}'


def main():
    args = parse_args()
    baseline = args.baseline_run_root.resolve()
    root = args.run_root.resolve()
    outputs = {
        'performance': root / 'comparison_summary.csv',
        'diagnostics': root / 'residual_cwp_training_diagnostics.csv',
        'relations': root / 'shared_relation_comparison.csv',
        'report': root / 'RESIDUAL_CWP_REPORT.md',
        'summary': root / 'summary.json',
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('refusing to overwrite Residual-CWP summary')
    baseline_summary_path = baseline / 'summary.json'
    if sha256(baseline_summary_path) != args.expected_baseline_summary_sha256:
        raise RuntimeError('immutable baseline summary SHA-256 mismatch')
    baseline_classification = read_json(
        baseline / 'evaluations/baseline/classification/classification_metrics.json'
    )
    baseline_cam = read_json(
        baseline / 'evaluations/baseline/cam_evaluation/metrics.json'
    )
    rows = [performance_row(
        'Original MCTformer+', 45, baseline_classification, baseline_cam
    )]
    diagnostic_rows = []
    relation_frames = []
    checkpoint_rows = []
    parameter_totals = set()
    for epochs in EPOCHS:
        epoch_root = root / f'epochs_{epochs}'
        paths = {
            'classification': epoch_root / (
                'evaluations/residual_cwp/classification/'
                'classification_metrics.json'
            ),
            'cam': epoch_root / (
                'evaluations/residual_cwp/cam_evaluation/metrics.json'
            ),
            'audit': epoch_root / 'audit/residual_cwp.json',
            'completion': epoch_root / 'shared_presence/completion.json',
            'relations': epoch_root / (
                'shared_presence/'
                'baseline_vs_residual_cwp_layer_comparison.csv'
            ),
            'pooling': epoch_root / (
                'shared_presence/residual_cwp_layer0_pooling_summary.json'
            ),
            'training_log': epoch_root / 'training_logs/residual_cwp.log',
            'checkpoint': epoch_root / (
                'checkpoints/residual_cwp/mctformerplus_final.pth'
            ),
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        audit = read_json(paths['audit'])
        completion = read_json(paths['completion'])
        if not audit.get('passed') or completion.get('status') != 'complete':
            raise RuntimeError(f'epoch-{epochs} audit or analysis incomplete')
        rows.append(performance_row(
            'MCTformer+ Residual CWP', epochs,
            read_json(paths['classification']), read_json(paths['cam']),
        ))
        diagnostic_rows.extend(training_diagnostics(
            paths['training_log'], epochs
        ))
        relations = pd.read_csv(paths['relations'])
        relations.insert(0, 'training_epochs', epochs)
        relation_frames.append(relations)
        pooling = read_json(paths['pooling'])[f'trained_residual_cwp']
        parameter_totals.add(audit['parameters']['trainable'])
        checkpoint_rows.append({
            'epochs': epochs,
            'path': str(paths['checkpoint']),
            'sha256': sha256(paths['checkpoint']),
            'alpha_final': diagnostic_rows[-1]['alpha_end'],
            'pooling_entropy_full_val': pooling['normalized_entropy_mean'],
            'interclass_attention_cosine_full_val': (
                pooling['interclass_attention_cosine_mean']
            ),
        })
    if len(parameter_totals) != 1:
        raise RuntimeError('Residual-CWP parameter counts differ across runs')

    with outputs['performance'].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(diagnostic_rows).to_csv(outputs['diagnostics'], index=False)
    pd.concat(relation_frames, ignore_index=True).to_csv(
        outputs['relations'], index=False
    )

    baseline_row = rows[0]
    table_lines = []
    for row in rows:
        table_lines.append(
            f"| {row['method']} | {row['epochs']} | "
            f"{fmt(row['class_token_macro_map_percent'])} | "
            f"{fmt(row['patch_head_macro_map_percent'])} | "
            f"{fmt(row['class_token_validation_loss'], 5)} | "
            f"{fmt(row['patch_validation_loss'], 5)} | "
            f"{fmt(row['fixed_native_cam_miou_percent'])} | "
            f"{fmt(row['best_native_cam_miou_percent'])} "
            f"({fmt(row['best_threshold'], 2)}) | "
            f"{fmt(row['fixed_foreground_precision_percent'])} / "
            f"{fmt(row['fixed_foreground_recall_percent'])} |"
        )
    alpha_lines = [
        f"| {item['epochs']} | 0.100000 | "
        f"{item['alpha_final']:.6f} | "
        f"{item['pooling_entropy_full_val']:.6f} | "
        f"{item['interclass_attention_cosine_full_val']:.6f} |"
        for item in checkpoint_rows
    ]
    report = f"""# Residual Image-Conditioned Multi-Class Token Initialization

The four runs differ only in the independently configured total epoch count. All use seed 0, the same official DeiT-S initialization, deterministic 448 VOC pipeline, original class/CCT/patch/GWRP losses, and native MCTformer+ CAM generation. Residual CWP adds 7,681 trainable parameters (20×384 class queries plus one scalar alpha) while retaining the original repeated pretrained CLS token.

| Method | Epochs | Class-token macro mAP | Patch-head macro mAP | Class val loss | Patch val loss | Fixed 0.45 CAM mIoU | Best CAM mIoU (threshold) | FG precision / recall at 0.45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

The matched B0 pipeline has no pseudo-mask or downstream segmentation result, so those values remain explicitly unavailable rather than being inferred.

| Epoch schedule | Initial alpha | Final-epoch alpha | Full-val pooling entropy | Full-val inter-class attention cosine |
|---:|---:|---:|---:|---:|
{chr(10).join(alpha_lines)}

The complete per-epoch alpha/entropy/similarity trajectory is in `residual_cwp_training_diagnostics.csv`. Layer-0 through Layer-12 positive-class relation correlation, Top-5% collision, common R², and effective rank are in `shared_relation_comparison.csv` for every schedule. Baseline Layer 0 remains unavailable because the frozen baseline source analysis starts at Block 1; Residual-CWP Layer 0 is measured immediately before Block 1.
"""
    outputs['report'].write_text(report, encoding='utf-8')
    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'Residual Image-Conditioned Multi-Class Token Initialization',
        'epoch_schedules': list(EPOCHS),
        'added_trainable_parameters': 20 * 384 + 1,
        'residual_cwp_trainable_parameters': next(iter(parameter_totals)),
        'performance': rows,
        'checkpoints': checkpoint_rows,
        'baseline_reference': {
            'summary_path': str(baseline_summary_path),
            'summary_sha256': sha256(baseline_summary_path),
            'performance': baseline_row,
        },
        'compact_artifacts': [str(path) for path in outputs.values()],
    }
    outputs['summary'].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
