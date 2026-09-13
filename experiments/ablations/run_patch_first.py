"""Order-only VOC ablation, matched to the completed default seed-0 run."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.baselines.run_default_voc_coco import (
    PRETRAIN, REPO, audit, dataset_spec, experiment, run,
)
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
    spec = dataset_spec('VOC12')
    baseline = REPO / 'results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12'
    assert (baseline / 'RUN_COMPLETE').is_file()
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'command': shlex.join([sys.executable, '-m', 'experiments.ablations.run_patch_first', *sys.argv[1:]]),
        'baseline': str(baseline),
        'baseline_metrics_sha256': sha256_file(baseline / 'raw_cam/metrics.json'),
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'patch_first': True, 'only_change': 'concatenate [patch, class]; positions follow tokens',
        'dataset': audit(spec), 'seed': 0, 'epochs': 45, 'batch_size': 32,
        'input_size': 448, 'nominal_lr': 5e-4, 'min_lr': 1e-5,
        'checkpoint_policy': 'final epoch', 'cam_scales': [1, .75, 1.25],
        'cam_thresholds': '0:.01:.59, fixed .45', 'postprocessing': 'native CAM only',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_mctformerplus_patch_first.py',
             'tests/test_mctformerplus_variants.py', 'tests/test_raw_cam_streaming.py'], output, 'tests')
        experiment(spec, output / 'VOC12_smoke', True, patch_first=True)
        experiment(spec, output / 'VOC12', False, patch_first=True)
        rows = []
        for name, directory in [('class_first', baseline), ('patch_first', output / 'VOC12')]:
            metrics = json.loads((directory / 'raw_cam/metrics.json').read_text())
            for policy in ('fixed', 'best'):
                rows.append({'variant': name, 'policy': policy, **metrics[policy]})
        with (output / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        report = ['# Patch-first token order control', '', 'Code: ' + manifest['git_sha'], '',
                  'Only concatenation order changed. Positions retain semantic identities.',
                  'Both use the same DeiT-S initialization, seed 0, 448, batch 32 and 45 epochs.',
                  'Permutation-equivalent in exact arithmetic; finite precision can change training trajectories.',
                  'VOC train 1464 native raw CAM; no CRF or segmentation. Best threshold is diagnostic only.', '',
                  '| Order | Policy | Threshold | mIoU (%) | Semantic FG precision (%) | Semantic FG recall (%) |',
                  '|---|---|---:|---:|---:|---:|']
        for row in rows:
            report.append(f"| {row['variant']} | {row['policy']} | {row['threshold']:.2f} | "
                          f"{row['mean_iou']*100:.3f} | {row['semantic_foreground_precision']*100:.3f} | "
                          f"{row['semantic_foreground_recall']*100:.3f} |")
        (output / 'PATCH_FIRST_REPORT.md').write_text('\n'.join(report) + '\n')
        (output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
