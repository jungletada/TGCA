from __future__ import annotations

import argparse
import csv
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
