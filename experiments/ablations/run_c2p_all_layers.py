"""One matched VOC run: all-layer actual C2P pooling; native CAM unchanged."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.ablations.run_c2p_pooling import classification
from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, audit, dataset_spec, experiment, run
from tools.evaluate_cam_threshold_grid import sha256_file


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
    previous = REPO / 'results/c2p_pooling/20260914-voc-s0'
    assert (baseline / 'RUN_COMPLETE').is_file() and (previous / 'QUEUE_COMPLETE').is_file()
    sources = [previous / 'comparison.csv', previous / 'manifest.json']
    for directory in [baseline, previous / 'VOC12']:
        sources.extend(directory / name for name in [
            'mctformerplus_final.pth', 'optimizer_spec.json', 'model_spec.json',
            'pretrained_load_report.json', 'raw_cam/metrics.json'])
    for stage in ['baseline_classification', 'c2p_classification']:
        sources.append(previous / stage / 'classification_metrics.json')
    hashes = {str(p): sha256_file(p) for p in sources}
    spec = dataset_spec('VOC12')
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'command': shlex.join([sys.executable, '-m', 'experiments.ablations.run_c2p_all_layers', *sys.argv[1:]]),
        'only_change_vs_previous': 'c2p_pooling_layers last3 -> all (L1-L12)',
        'patch_pooling': 'c2p', 'c2p_pooling_layers': 'all', 'class_token_init': 'baseline',
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'source_sha256_before': hashes, 'dataset': audit(spec),
        'seed': 0, 'epochs': 45, 'batch_size': 32, 'input_size': 448,
        'nominal_lr': 5e-4, 'min_lr': 1e-5, 'checkpoint_policy': 'final',
        'cam_class_to_patch_layers': 3, 'cam_patch_to_patch_layers': 12,
        'cam_scales': [1, .75, 1.25], 'cam_thresholds': '0:.01:.59; fixed .45',
        'classification': '1449 VOC val, single-scale FP32 448, two-head macro-class AP',
        'cam_evaluation': '1464 VOC train, native multi-scale, no CRF/segmentation',
        'comparison': 'reuse existing GWRP and C2P-last3 results; train only C2P-all',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_mctformerplus_c2p_pooling.py',
             'tests/test_mctformerplus_variants.py', 'tests/test_mctformerplus_patch_first.py',
             'tests/test_raw_cam_streaming.py', 'tests/test_width_scaling_aggregation.py'], output, 'tests')
        experiment(spec, output / 'VOC12_smoke', True, patch_pooling='c2p', c2p_pooling_layers='all')
        classification(output / 'VOC12_smoke/mctformerplus_final.pth', output, spec, 'c2p',
                       'smoke_classification', limit=4, c2p_pooling_layers='all')
        # Smoke and both native evaluators must complete before matched full training.
        experiment(spec, output / 'VOC12', False, patch_pooling='c2p', c2p_pooling_layers='all')
        classification(output / 'VOC12/mctformerplus_final.pth', output, spec, 'c2p',
                       'c2p_classification', c2p_pooling_layers='all')
        for name in ['optimizer_spec.json', 'pretrained_load_report.json']:
            assert json.loads((output / 'VOC12' / name).read_text()) == json.loads((baseline / name).read_text()), name
        rows = list(csv.DictReader((previous / 'comparison.csv').open()))
        for row in rows:
            if row['pooling'] == 'c2p':
                row['pooling'] = 'c2p-last3'
        cam = json.loads((output / 'VOC12/raw_cam/metrics.json').read_text())
        cls = json.loads((output / 'c2p_classification/classification_metrics.json').read_text())
        rows.append({
            'pooling': 'c2p-all',
            'class_macro_mAP_percent': cls['metrics_percent']['class_token']['macro_class_ap'],
            'patch_macro_mAP_percent': cls['metrics_percent']['patch_c2p']['macro_class_ap'],
            'class_val_loss': cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
            'patch_val_loss': cls['classification_loss']['patch_c2p_multilabel_soft_margin_mean'],
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
        after = {p: sha256_file(Path(p)) for p in hashes}
        assert after == hashes, 'Source results/checkpoints changed'
        manifest['source_sha256_after'] = after
        manifest['source_integrity_unchanged'] = True
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        report = ['# C2P all-layer patch pooling', '', 'Code: ' + manifest['git_sha'], '',
                  'Only pooling attention layer range changes from L10-L12 to L1-L12.',
                  'Average raw attention over layers/heads, then normalize over patch keys.',
                  'Raw classifier logits, attention gradients, class initialization, CCT, losses and recipe unchanged.',
                  'Fresh matched DeiT-S initialization, seed 0, 45 epochs; not an old checkpoint with a new flag.',
                  'Native CAM still uses last-three C2P and all-layer P2P. No CRF or segmentation.',
                  'Classification: 1449 VOC val, FP32 448; CAM: 1464 VOC train, native three scales.',
                  'Best threshold is a common-grid diagnostic, not a prespecified fixed-threshold result.',
                  'One seed only. Existing GWRP/C2P-last3 results reused without retraining or source edits.',
                  'Commands/config/SHA: manifest.json, commands.sh, VOC12/commands.sh; tests: tests.log.', '',
                  '| Pooling | Class mAP | Patch mAP | Class loss | Patch loss | CAM @.45 | Best CAM | Best t | FG precision @.45 | FG recall @.45 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            report.append('| ' + row['pooling'] + ' | ' + ' | '.join(
                f'{float(row[k]):.4f}' for k in rows[0] if k != 'pooling') + ' |')
        (output / 'C2P_ALL_LAYERS_REPORT.md').write_text('\n'.join(report) + '\n')
        (output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
