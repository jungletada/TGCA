"""Sequential matched VOC C2P products: last3 then all; reuse mean controls."""
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


def result_row(name, directory):
    cam = json.loads((directory / 'raw_cam/metrics.json').read_text())
    cls = json.loads((directory / 'classification/classification_metrics.json').read_text())
    return {
        'pooling': name,
        'class_macro_mAP_percent': cls['metrics_percent']['class_token']['macro_class_ap'],
        'patch_macro_mAP_percent': cls['metrics_percent']['patch_c2p']['macro_class_ap'],
        'class_val_loss': cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
        'patch_val_loss': cls['classification_loss']['patch_c2p_multilabel_soft_margin_mean'],
        'fixed_t045_cam_mIoU_percent': 100 * cam['fixed']['mean_iou'],
        'best_cam_mIoU_percent': 100 * cam['best']['mean_iou'],
        'best_threshold': cam['best']['threshold'],
        'fixed_semantic_FG_precision_percent': 100 * cam['fixed']['semantic_foreground_precision'],
        'fixed_semantic_FG_recall_percent': 100 * cam['fixed']['semantic_foreground_recall'],
    }


def report(output, rows, manifest):
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# C2P layer product versus arithmetic mean', '', 'Code: ' + manifest['git_sha'], '',
             'Completed new variants: ' + ', '.join(manifest['completed_new_variants']),
             'The full two-run queue is complete only when QUEUE_COMPLETE exists.', '',
             'Each layer averages heads; selected layers are multiplied element-wise and spatially normalized.',
             'Stable FP32 implementation: softmax(sum(log(head_mean))) over patch keys.',
             'Log arguments use machine tiny only for numerical zeros; no geometric mean, root, temperature or detach.',
             'Native CAM is unchanged: last3 mean C2P times ReLU patch map, sqrt, all-layer P2P.',
             'Fresh matched DeiT-S initialization per new run, seed 0, 45 epochs, 448, batch32.',
             'No change to class initialization, class/CCT losses, patch head, optimizer or augmentation.',
             'Classification: VOC val 1449, FP32 448 macro-class AP; CAM: train 1464, native three scales.',
             'Fixed threshold .45; best from the same 0:.01:.59 grid is diagnostic only.',
             'Existing three controls are immutable and are not retrained. One seed; no robustness claim.',
             'Exact commands/config/environment/source hashes: manifest.json and commands.sh / per-run commands.sh.', '',
             '| Pooling | Class mAP | Patch mAP | Class loss | Patch loss | CAM @.45 | Best CAM | Best t | FG P @.45 | FG R @.45 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['pooling'] + ' | ' + ' | '.join(
            f'{float(row[k]):.4f}' for k in rows[0] if k != 'pooling') + ' |')
    (output / 'C2P_PRODUCT_REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Tracked worktree must be clean')
    if subprocess.check_output(['git', 'branch', '--show-current']).decode().strip() != 'main':
        raise RuntimeError('Expected main')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = REPO / 'results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12'
    last3 = REPO / 'results/c2p_pooling/20260914-voc-s0'
    all_layers = REPO / 'results/c2p_pooling/20260914-voc-all-layers-s0'
    assert (baseline / 'RUN_COMPLETE').exists()
    assert all((p / 'QUEUE_COMPLETE').exists() for p in [last3, all_layers])
    sources = [all_layers / 'comparison.csv', last3 / 'manifest.json', all_layers / 'manifest.json']
    for directory in [baseline, last3 / 'VOC12', all_layers / 'VOC12']:
        sources.extend(directory / name for name in [
            'mctformerplus_final.pth', 'optimizer_spec.json', 'model_spec.json',
            'pretrained_load_report.json', 'raw_cam/metrics.json'])
    hashes = {str(p): sha256_file(p) for p in sources}
    spec = dataset_spec('VOC12')
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'command': shlex.join([sys.executable, '-m', 'experiments.ablations.run_c2p_product', *sys.argv[1:]]),
        'only_change_vs_matched_mean': 'c2p_pooling_reduction mean -> product',
        'queue': ['last3-product', 'all-product'], 'completed_new_variants': [],
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'source_sha256_before': hashes, 'dataset': audit(spec),
        'class_token_init': 'baseline', 'seed': 0, 'epochs': 45, 'input_size': 448,
        'batch_size': 32, 'nominal_lr': 5e-4, 'min_lr': 1e-5, 'checkpoint_policy': 'final',
        'pooling': 'softmax(sum_layers(log(mean_heads(actual A_c2p)))) over patch keys; raw M weighted sum',
        'numerical_floor': 'accumulation dtype finfo.tiny; FP32 under AMP; no tunable epsilon',
        'cam': 'unchanged native last3 mean C2P, sqrt, all-layer P2P; scales 1,.75,1.25',
        'evaluation': 'classification val1449 FP32 448; raw CAM train1464, fixed .45/grid 0:.01:.59',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_mctformerplus_c2p_pooling.py',
             'tests/test_mctformerplus_variants.py', 'tests/test_mctformerplus_patch_first.py',
             'tests/test_raw_cam_streaming.py', 'tests/test_width_scaling_aggregation.py'], output, 'tests')
        # Smoke both reductions' layer ranges before committing GPU time to full runs.
        for layers in ['last3', 'all']:
            smoke = output / (layers + '_product_smoke')
            experiment(spec, smoke, True, patch_pooling='c2p', c2p_pooling_layers=layers,
                       c2p_pooling_reduction='product')
            classification(smoke / 'mctformerplus_final.pth', smoke, spec, 'c2p',
                           'classification', limit=4, c2p_pooling_layers=layers,
                           c2p_pooling_reduction='product')
        with (all_layers / 'comparison.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row['pooling'] != 'gwrp':
                row['pooling'] += '-mean'
        for layers in ['last3', 'all']:
            directory = output / (layers + '_product')
            experiment(spec, directory, False, patch_pooling='c2p', c2p_pooling_layers=layers,
                       c2p_pooling_reduction='product')
            classification(directory / 'mctformerplus_final.pth', directory, spec, 'c2p',
                           'classification', c2p_pooling_layers=layers, c2p_pooling_reduction='product')
            for name in ['optimizer_spec.json', 'pretrained_load_report.json']:
                actual = json.loads((directory / name).read_text())
                assert all(actual == json.loads((ref / name).read_text())
                           for ref in [baseline, last3 / 'VOC12', all_layers / 'VOC12']), name
            rows.append(result_row('c2p-' + layers + '-product', directory))
            manifest['completed_new_variants'].append(layers + '-product')
            after = {p: sha256_file(Path(p)) for p in hashes}
            assert after == hashes, 'Source results/checkpoints changed'
            manifest['source_sha256_after'] = after
            manifest['source_integrity_unchanged'] = True
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            report(output, rows, manifest)
            (directory / 'VARIANT_COMPLETE').write_text('complete\n')
        (output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
