from __future__ import annotations

import argparse
import csv
import hashlib
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from tools.evaluate_mctformerplus_final_ln_diagnostics import (
    ATTENTION_METRICS,
    STAGES,
    _bootstrap_mean_intervals,
    _weighted_binary_auroc_samples,
)
from tools.summarize_mctformerplus_final_ln import (
    REPRESENTATION_METRICS,
    execute as summarize,
)
from tools.summarize_mctformerplus_patch_final_ln import (
    execute as summarize_patch_final_ln,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_weighted_presence_auroc_matches_sklearn_image_weights():
    labels = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.uint8)
    scores = np.asarray([
        [0.8, 0.2, 0.1],
        [0.3, 0.7, 0.1],
        [0.8, 0.4, 0.9],
        [0.2, 0.6, 0.9],
    ])
    draws = np.asarray([
        [1, 1, 1, 1],
        [2, 0, 1, 1],
        [0, 2, 0, 2],
    ], dtype=np.float64)
    observed = _weighted_binary_auroc_samples(labels, scores, draws, batch_size=2)
    expected = np.asarray([
        roc_auc_score(
            labels.reshape(-1), scores.reshape(-1),
            sample_weight=np.repeat(weights, labels.shape[1]),
        )
        for weights in draws
    ])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)


def test_clustered_mean_bootstrap_uses_image_sufficient_statistics():
    sums = np.asarray([[[2.0]], [[6.0]], [[1.0]]])
    counts = np.asarray([[[2.0]], [[3.0]], [[1.0]]])
    draws = np.asarray([[1, 1, 1], [3, 0, 0], [0, 3, 0]], dtype=np.float64)
    low, high = _bootstrap_mean_intervals(sums, counts, draws)
    samples = np.asarray([(2 + 6 + 1) / 6, 1.0, 2.0])
    np.testing.assert_allclose(low, np.quantile(samples, 0.025))
    np.testing.assert_allclose(high, np.quantile(samples, 0.975))


def test_final_ln_summary_emits_only_requested_result_families(tmp_path):
    root = tmp_path / 'run'
    for key, enabled, offset in (
        ('baseline', False, 0.0), ('final_ln', True, 0.1)
    ):
        checkpoint = root / 'checkpoints' / key / 'mctformerplus_final.pth'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f'{key}-checkpoint'.encode())
        audit = root / 'audit' / f'{key}.json'
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({'passed': True}))

        evaluation = root / 'evaluations' / key
        (evaluation / 'classification').mkdir(parents=True)
        (evaluation / 'classification/CLASSIFICATION_COMPLETE').write_text('complete\n')
        (evaluation / 'classification/classification_metrics.json').write_text(
            json.dumps({
                'metrics_percent': {
                    'class_token': {'macro_class_ap': 90.0 + offset},
                    'patch_gwrp': {'macro_class_ap': 91.0 + offset},
                }
            })
        )
        (evaluation / 'cam_train').mkdir()
        (evaluation / 'cam_train/CAM_COMPLETE').write_text('complete\n')
        (evaluation / 'cam_evaluation').mkdir()
        (evaluation / 'cam_evaluation/THRESHOLD_EVALUATION_COMPLETE').write_text(
            'complete\n'
        )
        point = {
            'threshold': 0.45,
            'mean_iou_percent': 60.0 + offset,
            'semantic_foreground_precision_percent': 70.0 + offset,
            'semantic_foreground_recall_percent': 80.0 + offset,
        }
        oracle = dict(point)
        oracle.update({'threshold': 0.47, 'mean_iou_percent': 61.0 + offset})
        (evaluation / 'cam_evaluation/metrics.json').write_text(json.dumps({
            'selected_metrics': {'fixed_0.45': point, 'oracle': oracle}
        }))

        diagnostics = evaluation / 'diagnostics'
        diagnostics.mkdir()
        (diagnostics / 'ATTENTION_REPRESENTATION_COMPLETE').write_text('complete\n')
        (diagnostics / 'summary.json').write_text(json.dumps({
            'final_norm': enabled
        }))
        _write_csv(diagnostics / 'attention_results.csv', [
            {
                'stage': stage, 'metric': metric,
                'estimate': 0.5 + offset, 'ci95_low': 0.4,
                'ci95_high': 0.6, 'num_images': 3,
                'num_observations': 4, 'bootstrap_unit': 'image',
            }
            for stage in STAGES for metric in ATTENTION_METRICS
        ])
        _write_csv(diagnostics / 'representation_results.csv', [
            {
                'metric': metric, 'estimate': 0.6 + offset,
                'ci95_low': 0.5, 'ci95_high': 0.7,
                'num_images': 3, 'num_observations': 4,
                'aggregation': 'test',
            }
            for metric in REPRESENTATION_METRICS
        ])

    result = summarize(argparse.Namespace(run_root=root))
    assert result['status'] == 'complete'
    assert (root / 'EXPERIMENT_COMPLETE').is_file()
    assert (root / 'FINAL_LN_EXPERIMENT_REPORT.md').is_file()
    comparison = list(csv.DictReader((root / 'comparison_summary.csv').open()))
    assert {row['domain'] for row in comparison} == {
        'classification', 'raw_cam', 'attention', 'representation'
    }
    assert all(
        abs(float(row['delta_final_ln_minus_baseline']) - 0.1) < 1e-9
        for row in comparison
        if row['metric'] != 'best_threshold'
    )


def test_patch_final_ln_summary_reuses_complete_source_read_only(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'PIPELINE_COMPLETE').write_text('complete\n')
    (source / 'EXPERIMENT_COMPLETE').write_text('complete\n')
    common_training = {
        'seed': 0, 'epochs': 45, 'micro_batch_size': 32, 'accum_iter': 1,
        'effective_batch_size': 32, 'nominal_lr': 0.0005,
        'optimizer_lr': 0.00003125, 'optimizer_updates_per_epoch': 330,
        'consumed_samples_per_epoch': 10560, 'train_dataset_size': 10582,
        'val_batch_size': 32, 'world_size': 1,
    }
    source_checkpoints = {}
    for key, enabled in (('baseline', False), ('final_ln', True)):
        checkpoint = source / 'checkpoints' / key / 'mctformerplus_final.pth'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(key.encode())
        source_checkpoints[key] = {
            'path': str(checkpoint),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'final_norm': enabled,
        }
        audit = source / 'audit' / f'{key}.json'
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({
            'passed': True,
            'method_configuration': {'final_norm': enabled},
            'training_spec': common_training,
            'official_pretrained': {'source_sha256': 'pretrained-hash'},
            'model_spec': {'variant': 'small'},
        }))
    (source / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'source_checkpoints': source_checkpoints,
    }))
    _write_csv(source / 'classification_results.csv', [
        {
            'model': label, 'final_norm': str(enabled).lower(),
            'class_token_map_percent': 90.0 + offset,
            'patch_head_map_percent': 91.0 + offset,
        }
        for label, enabled, offset in (
            ('MCTformer+', False, 0.0),
            ('MCTformer+-FinalLN', True, 1.0),
        )
    ])
    _write_csv(source / 'cam_results.csv', [
        {
            'model': label, 'final_norm': str(enabled).lower(),
            'fixed_threshold': 0.45,
            'fixed_raw_cam_miou_percent': 60.0 + offset,
            'fixed_semantic_fg_precision_percent': 70.0 + offset,
            'fixed_semantic_fg_recall_percent': 80.0 + offset,
            'best_threshold': 0.47,
            'best_raw_cam_miou_percent': 61.0 + offset,
            'best_semantic_fg_precision_percent': 71.0 + offset,
            'best_semantic_fg_recall_percent': 81.0 + offset,
        }
        for label, enabled, offset in (
            ('MCTformer+', False, 0.0),
            ('MCTformer+-FinalLN', True, 1.0),
        )
    ])
    _write_csv(source / 'attention_results.csv', [
        {
            'model': label, 'final_norm': str(enabled).lower(),
            'stage': stage, 'metric': metric, 'estimate': 0.5 + offset,
            'ci95_low': 0.4, 'ci95_high': 0.6,
            'num_images': 3, 'num_observations': 4,
            'bootstrap_unit': 'image',
        }
        for label, enabled, offset in (
            ('MCTformer+', False, 0.0),
            ('MCTformer+-FinalLN', True, 0.1),
        )
        for stage in STAGES for metric in ATTENTION_METRICS
    ])
    _write_csv(source / 'representation_results.csv', [
        {
            'model': label, 'final_norm': str(enabled).lower(),
            'metric': metric, 'estimate': 0.6 + offset,
            'ci95_low': 0.5, 'ci95_high': 0.7,
            'num_images': 3, 'num_observations': 4,
            'aggregation': 'test',
        }
        for label, enabled, offset in (
            ('MCTformer+', False, 0.0),
            ('MCTformer+-FinalLN', True, 0.1),
        )
        for metric in REPRESENTATION_METRICS
    ])

    root = tmp_path / 'patch'
    checkpoint = root / 'checkpoints/patch_final_ln/mctformerplus_final.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'patch-final-ln')
    audit = root / 'audit/patch_final_ln.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({
        'passed': True,
        'method_configuration': {
            'final_norm': False, 'patch_final_norm': True,
        },
        'training_spec': {
            **common_training, 'final_norm': False, 'patch_final_norm': True,
        },
        'official_pretrained': {'source_sha256': 'pretrained-hash'},
        'model_spec': {'variant': 'small'},
    }))
    evaluation = root / 'evaluations/patch_final_ln'
    (evaluation / 'classification').mkdir(parents=True)
    (evaluation / 'classification/CLASSIFICATION_COMPLETE').write_text('complete\n')
    (evaluation / 'classification/classification_metrics.json').write_text(
        json.dumps({'metrics_percent': {
            'class_token': {'macro_class_ap': 92.0},
            'patch_gwrp': {'macro_class_ap': 93.0},
        }})
    )
    (evaluation / 'cam_train').mkdir()
    (evaluation / 'cam_train/CAM_COMPLETE').write_text('complete\n')
    (evaluation / 'cam_evaluation').mkdir()
    (evaluation / 'cam_evaluation/THRESHOLD_EVALUATION_COMPLETE').write_text(
        'complete\n'
    )
    fixed = {
        'threshold': 0.45, 'mean_iou_percent': 62.0,
        'semantic_foreground_precision_percent': 72.0,
        'semantic_foreground_recall_percent': 82.0,
    }
    oracle = {
        **fixed, 'threshold': 0.48, 'mean_iou_percent': 63.0,
        'semantic_foreground_precision_percent': 73.0,
        'semantic_foreground_recall_percent': 83.0,
    }
    (evaluation / 'cam_evaluation/metrics.json').write_text(json.dumps({
        'selected_metrics': {'fixed_0.45': fixed, 'oracle': oracle},
    }))
    diagnostics = evaluation / 'diagnostics'
    diagnostics.mkdir()
    (diagnostics / 'ATTENTION_REPRESENTATION_COMPLETE').write_text('complete\n')
    (diagnostics / 'summary.json').write_text(json.dumps({
        'final_norm': False, 'patch_final_norm': True,
    }))
    _write_csv(diagnostics / 'attention_results.csv', [
        {
            'stage': stage, 'metric': metric, 'estimate': 0.7,
            'ci95_low': 0.6, 'ci95_high': 0.8,
            'num_images': 3, 'num_observations': 4,
            'bootstrap_unit': 'image',
        }
        for stage in STAGES for metric in ATTENTION_METRICS
    ])
    _write_csv(diagnostics / 'representation_results.csv', [
        {
            'metric': metric, 'estimate': 0.8,
            'ci95_low': 0.7, 'ci95_high': 0.9,
            'num_images': 3, 'num_observations': 4,
            'aggregation': 'test',
        }
        for metric in REPRESENTATION_METRICS
    ])

    result = summarize_patch_final_ln(argparse.Namespace(
        source_run_root=source, patch_run_root=root,
    ))
    assert result['status'] == 'complete'
    assert (root / 'EXPERIMENT_COMPLETE').is_file()
    classification = list(csv.DictReader(
        (root / 'classification_results.csv').open()
    ))
    assert [row['normalization_scope'] for row in classification] == [
        'none', 'all_tokens', 'patch_tokens'
    ]
    comparisons = list(csv.DictReader(
        (root / 'comparison_summary.csv').open()
    ))
    assert {row['reference_model'] for row in comparisons} == {
        'MCTformer+', 'MCTformer+-FinalLN'
    }
