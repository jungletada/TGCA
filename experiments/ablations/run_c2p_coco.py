"""One fresh COCO all-product run; matched immutable GWRP control; native CAM only."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, audit, dataset_spec, experiment, run
from tools.evaluate_cam_threshold_grid import sha256_file


def classification(checkpoint, output, spec, pooling, stage, limit=0):
    run([sys.executable, '-u', 'tools/evaluate_mctformerplus_classification.py',
         '--checkpoint', checkpoint, '--model', 'mctformerplus', '--dataset', 'COCO',
         '--data-root', spec['root'], '--list-path', spec['val'], '--input-size', '448',
         '--batch-size', '16', '--num-workers', '8', '--bootstrap-resamples', '0',
         '--patch-pooling', pooling,
         *(['--c2p-pooling-layers', 'all', '--c2p-pooling-reduction', 'product'] if pooling == 'c2p' else []),
         '--limit', str(limit), '--output-dir', output / stage], output, stage)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Tracked worktree must be clean')
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'main':
        raise RuntimeError('Expected main')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = REPO / 'results/default_mctformerplus/20260912-voc-coco-s0-r2/COCO'
    assert (baseline / 'RUN_COMPLETE').exists()
    sources = [baseline / p for p in ['mctformerplus_final.pth', 'optimizer_spec.json',
               'pretrained_load_report.json', 'model_spec.json', 'commands.sh', 'raw_cam/metrics.json']]
    hashes = {str(p): sha256_file(p) for p in sources}
    assert hashes[str(baseline / 'mctformerplus_final.pth')] == (baseline / 'checkpoint_sha256.txt').read_text().split()[0]
    spec = dataset_spec('COCO')
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, '-u', '-m', 'experiments.ablations.run_c2p_coco', *sys.argv[1:]]),
        'baseline': str(baseline), 'source_sha256_before': hashes, 'dataset': audit(spec),
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'seed': 0, 'epochs': 45, 'input_size': 448, 'batch_size': 32,
        'nominal_lr': 5e-4, 'min_lr': 1e-5, 'class_token_init': 'baseline',
        'patch_pooling': 'c2p', 'c2p_pooling_layers': 'all', 'c2p_pooling_reduction': 'product',
        'c2p_pooling_affinity': False, 'checkpoint_policy': 'final',
        'cam': 'unchanged native three-scale 1,.75,1.25, train82783, fixed .45, grid0:.01:.59',
        'storage': 'online global confusion accumulation; no large full-run raw CAM dump (baseline dump 249 GiB)',
        'classification': 'both heads FP32 single448 val40504, macro-class AP; no bootstrap CI',
        'scope': 'explicit user-requested pooling transfer, not a TGCA generality gate claim',
    }
    baseline_manifest = json.loads((baseline.parent / 'manifest.json').read_text())
    assert manifest['dataset'] == baseline_manifest['datasets']['COCO']
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_c2p_coco.py',
             'tests/test_mctformerplus_c2p_pooling.py', 'tests/test_mctformerplus_variants.py',
             'tests/test_mctformerplus_patch_first.py', 'tests/test_raw_cam_streaming.py',
             'tests/test_width_scaling_aggregation.py'], output, 'tests')
        smoke = output / 'COCO_smoke'
        kwargs = dict(patch_pooling='c2p', c2p_pooling_layers='all', c2p_pooling_reduction='product')
        experiment(spec, smoke, True, **kwargs)
        # Same checkpoint/images: compare native disk and online evaluations exactly.
        run([sys.executable, '-u', 'make_cam.py', '--dataset', 'COCO', '--model', 'mctformerplus',
             '--coco_root', spec['root'], '--work_space', smoke, '--patch-pooling', 'c2p',
             '--c2p-pooling-layers', 'all', '--c2p-pooling-reduction', 'product',
             '--train_list', smoke / 'cam_train_id.txt', '--input_size', '448', '--scales', '1.0,0.75,1.25',
             '--checkpoint', smoke / 'mctformerplus_final.pth', '--cam_out_dir', 'cam_online',
             '--online-raw-eval', '--online-mask-dir', spec['masks'],
             '--online-output-dir', smoke / 'raw_cam_online'], smoke, 'online_cam_parity')
        a = np.load(smoke / 'raw_cam/aggregate_confusions.npz')['confusion']
        b = np.load(smoke / 'raw_cam_online/aggregate_confusions.npz')['confusion']
        np.testing.assert_array_equal(a, b)
        (smoke / 'ONLINE_PARITY_COMPLETE').write_text('all 60 thresholds: confusion arrays identical\n')
        classification(smoke / 'mctformerplus_final.pth', smoke, spec, 'c2p', 'classification', limit=4)
        classification(baseline / 'mctformerplus_final.pth', output, spec, 'gwrp', 'baseline_smoke_classification', limit=4)
        full = output / 'COCO'
        experiment(spec, full, False, online_raw_eval=True, **kwargs)
        for name in ['optimizer_spec.json', 'pretrained_load_report.json']:
            assert json.loads((full / name).read_text()) == json.loads((baseline / name).read_text()), name
        classification(full / 'mctformerplus_final.pth', full, spec, 'c2p', 'classification')
        classification(baseline / 'mctformerplus_final.pth', output, spec, 'gwrp', 'baseline_classification')
        rows = []
        for pooling, directory, cls_dir in [('gwrp', baseline, output / 'baseline_classification'),
                                           ('c2p-all-product', full, full / 'classification')]:
            cam = json.loads((directory / 'raw_cam/metrics.json').read_text())
            cls = json.loads((cls_dir / 'classification_metrics.json').read_text())
            branch = 'patch_gwrp' if pooling == 'gwrp' else 'patch_c2p'
            rows.append(dict(pooling=pooling,
                class_macro_mAP_percent=cls['metrics_percent']['class_token']['macro_class_ap'],
                patch_macro_mAP_percent=cls['metrics_percent'][branch]['macro_class_ap'],
                class_val_loss=cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
                patch_val_loss=cls['classification_loss'][branch + '_multilabel_soft_margin_mean'],
                fixed_t045_cam_mIoU_percent=100*cam['fixed']['mean_iou'],
                best_cam_mIoU_percent=100*cam['best']['mean_iou'], best_threshold=cam['best']['threshold'],
                fixed_FG_precision_percent=100*cam['fixed']['semantic_foreground_precision'],
                fixed_FG_recall_percent=100*cam['fixed']['semantic_foreground_recall']))
        with (output / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        manifest['source_sha256_after'] = {p: sha256_file(p) for p in hashes}
        assert manifest['source_sha256_after'] == hashes
        manifest['source_integrity_unchanged'] = True
        manifest['online_cam_smoke_parity'] = 'exact aggregate confusions, all60 thresholds'
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        lines = ['# COCO all-product versus GWRP', '', 'Code: ' + manifest['git_sha'], '',
                 'One fresh seed0 matched 45-epoch run. All12 head-mean C2P product; no pooling P2P affinity.',
                 'Initialization, class/CCT, patch head, optimizer, schedule and native CAM unchanged.',
                 'Classification: val40504, both heads macro-class AP, FP32 single448.',
                 'CAM: train82783 native three scales; fixed .45; best same-grid diagnostic only.',
                 'Online evaluation avoids large CAM dumps; smoke confirmed exact equality with disk evaluation.',
                 'No CRF, segmentation, extra variants or baseline retraining. One seed, no robustness claim.',
                 'Commands/config/checkpoint SHA/environment/tests/source integrity: adjacent logs and manifests.', '',
                 '| Pooling | Class AP | Patch AP | Class loss | Patch loss | CAM .45 | Best CAM | Best t | FG P | FG R |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            lines.append('| ' + row['pooling'] + ' | ' + ' | '.join(f'{v:.4f}' for k,v in row.items() if k != 'pooling') + ' |')
        (output / 'C2P_COCO_REPORT.md').write_text('\n'.join(lines) + '\n')
        (output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
