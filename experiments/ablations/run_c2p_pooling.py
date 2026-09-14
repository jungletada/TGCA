"""One matched VOC C2P-vs-GWRP patch-pooling experiment; no initializer changes."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.baselines.run_default_voc_coco import (
    REPO, PRETRAIN, audit, dataset_spec, experiment, run,
)
from tools.evaluate_cam_threshold_grid import sha256_file


def classification(checkpoint, output, spec, pooling, stage, limit=0, c2p_pooling_layers='last3'):
    run([sys.executable, '-u', 'tools/evaluate_mctformerplus_classification.py',
         '--checkpoint', checkpoint, '--model', 'mctformerplus',
         '--voc-root', spec['root'], '--list-path', spec['val'],
         '--input-size', '448', '--batch-size', '16', '--num-workers', '8',
         '--patch-pooling', pooling, '--bootstrap-resamples', '0',
         '--c2p-pooling-layers', c2p_pooling_layers,
         '--limit', str(limit), '--output-dir', output / stage], output, stage)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.chdir(REPO)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Tracked worktree must be clean')
    if subprocess.check_output(['git', 'branch', '--show-current']).decode().strip() != 'main':
        raise RuntimeError('Expected main')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = REPO / 'results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12'
    assert (baseline / 'RUN_COMPLETE').is_file()
    spec = dataset_spec('VOC12')
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'command': shlex.join([sys.executable, '-m', 'experiments.ablations.run_c2p_pooling', *sys.argv[1:]]),
        'baseline': str(baseline), 'baseline_metrics_sha256': sha256_file(baseline / 'raw_cam/metrics.json'),
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'only_change': 'patch_pooling gwrp -> c2p', 'class_token_init': 'baseline',
        'dataset': audit(spec), 'seed': 0, 'epochs': 45, 'batch_size': 32,
        'input_size': 448, 'nominal_lr': 5e-4, 'min_lr': 1e-5,
        'checkpoint_policy': 'final', 'cam_scales': [1, .75, 1.25],
        'cam_thresholds': '0:.01:.59; fixed .45',
        'classification': 'VOC val 1449, single-scale 448, FP32, no flip; macro-class AP and legacy mean-image AP',
        'training_legacy_validation': 'unchanged AMP class-token mean-image AP; not macro-class AP',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_mctformerplus_c2p_pooling.py',
             'tests/test_mctformerplus_variants.py', 'tests/test_mctformerplus_patch_first.py',
             'tests/test_raw_cam_streaming.py', 'tests/test_width_scaling_aggregation.py'], output, 'tests')
        experiment(spec, output / 'VOC12_smoke', True, patch_pooling='c2p')
        classification(output / 'VOC12_smoke/mctformerplus_final.pth', output, spec,
                       'c2p', 'smoke_classification', limit=4)
        # Existing baseline weights/results remain immutable; this only adds a
        # matched two-head classification measurement under the new run root.
        classification(baseline / 'mctformerplus_final.pth', output, spec,
                       'gwrp', 'baseline_classification')
        experiment(spec, output / 'VOC12', False, patch_pooling='c2p')
        classification(output / 'VOC12/mctformerplus_final.pth', output, spec,
                       'c2p', 'c2p_classification')
        baseline_config = json.loads((baseline / 'optimizer_spec.json').read_text())
        new_config = json.loads((output / 'VOC12/optimizer_spec.json').read_text())
        assert baseline_config == new_config, 'Training settings must be matched'
        rows = []
        for pooling, directory in [('gwrp', baseline), ('c2p', output / 'VOC12')]:
            cam = json.loads((directory / 'raw_cam/metrics.json').read_text())
            stage = 'baseline_classification' if pooling == 'gwrp' else 'c2p_classification'
            cls = json.loads((output / stage / 'classification_metrics.json').read_text())
            branch = 'patch_' + pooling
            rows.append({
                'pooling': pooling,
                'class_macro_mAP_percent': cls['metrics_percent']['class_token']['macro_class_ap'],
                'patch_macro_mAP_percent': cls['metrics_percent'][branch]['macro_class_ap'],
                'class_val_loss': cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
                'patch_val_loss': cls['classification_loss'][branch + '_multilabel_soft_margin_mean'],
                'fixed_t045_cam_mIoU_percent': 100 * cam['fixed']['mean_iou'],
                'best_cam_mIoU_percent': 100 * cam['best']['mean_iou'],
                'best_threshold': cam['best']['threshold'],
                'fixed_semantic_FG_precision_percent': 100 * cam['fixed']['semantic_foreground_precision'],
                'fixed_semantic_FG_recall_percent': 100 * cam['fixed']['semantic_foreground_recall'],
            })
        with (output / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        report = ['# C2P patch pooling vs original GWRP', '', 'Code: ' + manifest['git_sha'], '',
                  'Only pooling changes. Existing class_token_init=baseline is retained in both runs.',
                  'No new parameters/loss; attention gradients retained; original raw 3x3 logits.',
                  'Training recipe and native CAM formulas unchanged. Final checkpoint for both.',
                  'Classification: VOC val 1449, FP32 single-scale 448, macro-class AP.',
                  'CAM: VOC train 1464, native three-scale, no CRF/segmentation; best threshold diagnostic only.',
                  'Exact commands: manifest.json, commands.sh, VOC12/commands.sh; tests: tests.log.', '',
                  '| Pooling | Class mAP | Patch mAP | Class loss | Patch loss | CAM @.45 | Best CAM | Best t | FG precision @.45 | FG recall @.45 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            report.append('| ' + ' | '.join(str(v) if isinstance(v, str) else f'{v:.4f}' for v in row.values()) + ' |')
        (output / 'C2P_POOLING_REPORT.md').write_text('\n'.join(report) + '\n')
        (output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
