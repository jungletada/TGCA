#!/usr/bin/env python
"""Build compact matched results and the five-question CWP report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run-root', type=Path, required=True)
    parser.add_argument('--cwp-run-root', type=Path, required=True)
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


def cam_metrics(payload):
    fixed = payload['selected_metrics']['fixed_0.45']
    best = payload['selected_metrics']['oracle']
    return {
        'fixed_threshold': fixed['threshold'],
        'fixed_native_cam_miou_percent': fixed['mean_iou_percent'],
        'fixed_foreground_precision_percent': fixed['binary_foreground_precision_percent'],
        'fixed_foreground_recall_percent': fixed['binary_foreground_recall_percent'],
        'best_threshold': best['threshold'],
        'best_native_cam_miou_percent': best['mean_iou_percent'],
        'best_foreground_precision_percent': best['binary_foreground_precision_percent'],
        'best_foreground_recall_percent': best['binary_foreground_recall_percent'],
    }


def performance_row(name, classification, cam):
    patch = classification['metrics_percent']['patch_gwrp']
    losses = classification['classification_loss']
    return {
        'method': name,
        'class_token_macro_map_percent': classification['metrics_percent']['class_token']['macro_class_ap'],
        'patch_head_macro_map_percent': patch['macro_class_ap'],
        'class_token_validation_loss': losses['class_token_multilabel_soft_margin_mean'],
        'patch_validation_loss': losses['patch_gwrp_multilabel_soft_margin_mean'],
        **cam_metrics(cam),
        'pseudo_mask_metric': 'unavailable_in_matched_B0_pipeline',
        'downstream_segmentation_metric': 'unavailable_in_matched_B0_pipeline',
    }


def training_diagnostics(log_path: Path):
    rows = []
    for line in log_path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('{"epoch"'):
            continue
        payload = json.loads(line)
        row = {'epoch': payload['epoch']}
        for key, value in payload.items():
            if key.startswith('train_cwp_'):
                row[key.removeprefix('train_')] = value
        if len(row) > 1:
            rows.append(row)
    if not rows:
        raise RuntimeError('training log has no CWP diagnostics')
    return pd.DataFrame(rows)


def fmt(value, digits=3):
    return 'N/A' if pd.isna(value) else f'{float(value):.{digits}f}'


def main():
    args = parse_args()
    baseline = args.baseline_run_root.resolve()
    root = args.cwp_run_root.resolve()
    outputs = (
        root / 'comparison_summary.csv', root / 'cwp_training_diagnostics.csv',
        root / 'CWP_REPORT.md', root / 'summary.json',
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError('refusing to overwrite an existing CWP summary')
    baseline_summary = baseline / 'summary.json'
    if sha256(baseline_summary) != args.expected_baseline_summary_sha256:
        raise RuntimeError('immutable baseline summary SHA-256 mismatch')
    paths = {
        'baseline_classification': baseline / 'evaluations/baseline/classification/classification_metrics.json',
        'baseline_cam': baseline / 'evaluations/baseline/cam_evaluation/metrics.json',
        'cwp_classification': root / 'evaluations/cwp/classification/classification_metrics.json',
        'cwp_cam': root / 'evaluations/cwp/cam_evaluation/metrics.json',
        'cwp_audit': root / 'audit/cwp.json',
        'shared_completion': root / 'shared_presence/completion.json',
        'shared_comparison': root / 'shared_presence/baseline_vs_cwp_layer_comparison.csv',
        'pooling': root / 'shared_presence/cwp_layer0_pooling_summary.json',
        'training_log': root / 'training_logs/cwp.log',
        'checkpoint': root / 'checkpoints/cwp/mctformerplus_final.pth',
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = read_json(paths['cwp_audit'])
    completion = read_json(paths['shared_completion'])
    if not audit.get('passed') or completion.get('status') != 'complete':
        raise RuntimeError('CWP audit or shared-presence analysis is incomplete')
    rows = [
        performance_row(
            'Original MCTformer+', read_json(paths['baseline_classification']),
            read_json(paths['baseline_cam']),
        ),
        performance_row(
            'MCTformer+ CWP', read_json(paths['cwp_classification']),
            read_json(paths['cwp_cam']),
        ),
    ]
    with outputs[0].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = training_diagnostics(paths['training_log'])
    diagnostics.to_csv(outputs[1], index=False)
    shared = pd.read_csv(paths['shared_comparison']).sort_values('layer')
    layer0 = shared[shared.layer == 0].iloc[0]
    layer12 = shared[shared.layer == 12].iloc[0]
    pooling = read_json(paths['pooling'])
    initial = pooling['official_deit_initialization']
    trained = pooling['trained_cwp']
    base, cwp = rows
    delta_class = cwp['class_token_macro_map_percent'] - base['class_token_macro_map_percent']
    delta_patch = cwp['patch_head_macro_map_percent'] - base['patch_head_macro_map_percent']
    delta_fixed = cwp['fixed_native_cam_miou_percent'] - base['fixed_native_cam_miou_percent']
    delta_best = cwp['best_native_cam_miou_percent'] - base['best_native_cam_miou_percent']
    report = f"""# MCTformer+ Class-wise Weighted Pooling Report

## Q1. Can CWP train without duplicated DeiT CLS content, and how does it perform?

The strict checkpoint audit passed. The CWP state has `class_token_pooler.class_queries` and no `cls_token`. Training used the ordinary MCTformer+ AMP/GradScaler path without a CWP-specific abort guard.

| Method | Class-token macro mAP | Patch-head macro mAP | Class val loss | Patch val loss | Fixed 0.45 native CAM mIoU | Best native CAM mIoU (threshold) | FG precision / recall at 0.45 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original MCTformer+ | {fmt(base['class_token_macro_map_percent'])} | {fmt(base['patch_head_macro_map_percent'])} | {fmt(base['class_token_validation_loss'], 5)} | {fmt(base['patch_validation_loss'], 5)} | {fmt(base['fixed_native_cam_miou_percent'])} | {fmt(base['best_native_cam_miou_percent'])} ({fmt(base['best_threshold'], 2)}) | {fmt(base['fixed_foreground_precision_percent'])} / {fmt(base['fixed_foreground_recall_percent'])} |
| CWP | {fmt(cwp['class_token_macro_map_percent'])} | {fmt(cwp['patch_head_macro_map_percent'])} | {fmt(cwp['class_token_validation_loss'], 5)} | {fmt(cwp['patch_validation_loss'], 5)} | {fmt(cwp['fixed_native_cam_miou_percent'])} | {fmt(cwp['best_native_cam_miou_percent'])} ({fmt(cwp['best_threshold'], 2)}) | {fmt(cwp['fixed_foreground_precision_percent'])} / {fmt(cwp['fixed_foreground_recall_percent'])} |

Matched deltas (CWP − baseline) are {delta_class:+.3f} class-mAP points, {delta_patch:+.3f} patch-mAP points, {delta_fixed:+.3f} fixed-threshold CAM mIoU points, and {delta_best:+.3f} best-grid CAM mIoU points. The matched B0 pipeline has no pseudo-mask or downstream segmentation metric, so neither is fabricated here.

## Q2. Is Layer-0 initialization image-conditioned and class-dependent?

Yes at the representation level: after training, the mean attention standard deviation across images is {trained['mean_attention_std_across_images']:.8f}, while mean within-image inter-class attention cosine is {trained['interclass_attention_cosine_mean']:.6f} (1 would mean identical class maps). Layer-0 positive-class token-pair cosine is {fmt(pd.read_csv(root / 'shared_presence/cwp_layer_summary.csv').query('layer == 0').iloc[0]['initial_positive_class_token_pair_cosine'], 6)}. All pooling rows sum to one within a maximum error of {trained['max_attention_row_sum_error']:.3e}.

## Q3. Does pooling evolve from near-uniform toward class-dependent weighting?

At the exact official-DeiT initialization, normalized pooling entropy is {initial['normalized_entropy_mean']:.6f}, mean absolute deviation from uniform is {initial['mean_absolute_deviation_from_uniform']:.8f}, and inter-class attention cosine is {initial['interclass_attention_cosine_mean']:.6f}. After training they are {trained['normalized_entropy_mean']:.6f}, {trained['mean_absolute_deviation_from_uniform']:.8f}, and {trained['interclass_attention_cosine_mean']:.6f}, respectively. Per-epoch diagnostics are preserved in `cwp_training_diagnostics.csv`.

## Q4. How does shared relation form from Layer 0 to Layer 12?

At CWP Layer 0, positive-class relation correlation is {fmt(layer0['raw_pair_corr_cwp'], 6)}, Top-5% collision is {fmt(layer0['raw_pair_jaccard_top05_cwp'], 6)}, common-component R² is {fmt(layer0['common_r2_cwp'], 6)}, and shared-mean effective rank is {fmt(layer0['effective_rank_cwp'], 6)}. At Layer 12, the corresponding CWP values are {fmt(layer12['raw_pair_corr_cwp'], 6)}, {fmt(layer12['raw_pair_jaccard_top05_cwp'], 6)}, {fmt(layer12['common_r2_cwp'], 6)}, and {fmt(layer12['effective_rank_cwp'], 6)}. The complete matched Layer 0–12 trajectory is in `shared_presence/baseline_vs_cwp_layer_comparison.csv`; baseline Layer 0 is explicitly N/A as preregistered.

## Q5. Does changing only initialization change WSSS performance?

The directly matched deltas are the Q1 values above. They quantify the effect of replacing duplicated DeiT CLS initialization with CWP under the unchanged 12-block transformer, losses, patch head/GWRP, and native CAM pipeline. No causal claim beyond this controlled architecture comparison is made.
"""
    outputs[2].write_text(report, encoding='utf-8')
    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'MCTformer+ Class-wise Weighted Pooling',
        'performance': rows,
        'deltas_cwp_minus_baseline': {
            'class_token_macro_map_percent': delta_class,
            'patch_head_macro_map_percent': delta_patch,
            'fixed_native_cam_miou_percent': delta_fixed,
            'best_native_cam_miou_percent': delta_best,
        },
        'checkpoint': {
            'path': str(paths['checkpoint']),
            'sha256': sha256(paths['checkpoint']),
            'git_commit': audit.get('variant_resolution', {}).get('git_commit'),
        },
        'source_baseline_summary': {
            'path': str(baseline_summary),
            'sha256': sha256(baseline_summary),
        },
        'compact_artifacts': [str(path) for path in outputs[:3]],
    }
    outputs[3].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
