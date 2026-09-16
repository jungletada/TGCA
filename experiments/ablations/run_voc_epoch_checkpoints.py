"""Four fresh matched VOC runs, seeds0/11, GWRP/all-product, every epoch retained."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, audit, dataset_spec, run
from experiments.ablations.run_c2p_pooling import classification
from tools.epoch_checkpoints import sha256

MATRIX = [(seed, method) for seed in (0, 11) for method in ('gwrp', 'all_product')]
REFERENCES = {
    'gwrp': REPO/'results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12',
    'all_product': REPO/'results/c2p_pooling/20260914-voc-product-s0/all_product',
}


def canonical_model_spec(spec):
    # Older baseline records predate these flags. Fill only their known legacy
    # defaults, without changing source metadata or hiding non-default values.
    defaults = dict(patch_first=False, patch_pooling='gwrp', c2p_pooling_layers='last3',
                    c2p_pooling_reduction='mean', c2p_pooling_affinity=False)
    return {**defaults, **spec}


def pooling_flags(method):
    return ['--patch-pooling', 'gwrp'] if method == 'gwrp' else [
        '--patch-pooling', 'c2p', '--c2p-pooling-layers', 'all', '--c2p-pooling-reduction', 'product']


def training_command(spec, directory, method, seed, train=None, val=None, epochs=45, workers=10):
    return [sys.executable, '-u', 'train_model_v2.py', '--dataset', 'VOC12', '--model', 'mctformerplus',
            '--voc12_root', str(spec['root']), '--work_space', str(directory),
            '--train_list', str(train or spec['train']), '--val_list', str(val or spec['val']),
            '--input-size', '448', '--epochs', str(epochs), '--batch_size', '32', '--seed', str(seed),
            '--lr', '5e-4', '--min-lr', '1e-5', '--num_workers', str(workers),
            '--finetune', str(PRETRAIN), '--save-every-epoch', *pooling_flags(method)]


def verify_snapshots(directory, epochs):
    import torch
    checkpoints = directory/'epoch_checkpoints'
    rows = [json.loads(line) for line in (checkpoints/'index.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(epochs))
    assert len(list(checkpoints.glob('*.pth'))) == epochs
    for epoch, row in enumerate(rows):
        assert row['filename'] == f'mctformerplus_epoch_{epoch+1:03d}.pth'
        path = checkpoints/row['filename']
        assert path.stat().st_size == row['bytes'] and sha256(path) == row['sha256']
    saved = torch.load(checkpoints/rows[-1]['filename'], map_location='cpu')
    final = torch.load(directory/'mctformerplus_final.pth', map_location='cpu')
    assert saved['epoch'] == final['epoch'] == epochs-1
    assert all(k in saved for k in ['optimizer', 'lr_scheduler', 'scaler', 'rng_state', 'args', 'metrics'])
    assert saved['optimizer']['state']
    for key in final['model']:
        assert torch.equal(saved['model'][key], final['model'][key]), key
    (directory/'EPOCH_CHECKPOINTS_VERIFIED').write_text(f'{epochs} snapshots; hashes and final-state equality verified\n')


def evaluate(spec, directory, method, smoke=False):
    checkpoint = directory/('epoch_checkpoints/mctformerplus_epoch_002.pth' if smoke else 'mctformerplus_final.pth')
    cam_list = directory/'cam_train_id.txt' if smoke else spec['cam']
    run([sys.executable, '-u', 'make_cam.py', '--dataset', 'VOC12', '--model', 'mctformerplus',
         '--voc12_root', spec['root'], '--work_space', directory, '--train_list', cam_list,
         '--input_size', '448', '--scales', '1.0,0.75,1.25', '--checkpoint', checkpoint,
         '--cam_out_dir', 'cam_train', '--online-raw-eval', '--online-mask-dir', spec['masks'],
         '--online-output-dir', directory/'raw_cam', *pooling_flags(method)], directory, 'cam')
    classification(checkpoint, directory, spec, 'gwrp' if method=='gwrp' else 'c2p', 'classification',
                   limit=4 if smoke else 0, c2p_pooling_layers='last3' if method=='gwrp' else 'all',
                   c2p_pooling_reduction='mean' if method=='gwrp' else 'product')


def result(directory, method, seed):
    cam = json.loads((directory/'raw_cam/metrics.json').read_text())
    cls = json.loads((directory/'classification/classification_metrics.json').read_text())
    branch = 'patch_gwrp' if method == 'gwrp' else 'patch_c2p'
    assert cam['num_images'] == 1464 and cls['num_images'] == 1449 and cls['finite']
    return dict(method=method, seed=seed, class_macro_mAP_percent=cls['metrics_percent']['class_token']['macro_class_ap'],
                patch_macro_mAP_percent=cls['metrics_percent'][branch]['macro_class_ap'],
                class_val_loss=cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
                patch_val_loss=cls['classification_loss'][branch+'_multilabel_soft_margin_mean'],
                fixed_cam_mIoU_percent=100*cam['fixed']['mean_iou'], best_cam_mIoU_percent=100*cam['best']['mean_iou'],
                best_threshold=cam['best']['threshold'], fg_precision_percent=100*cam['fixed']['semantic_foreground_precision'],
                fg_recall_percent=100*cam['fixed']['semantic_foreground_recall'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(matrix=MATRIX, epochs=45, save_every_epoch=True, training_started=False)))
        return
    os.chdir(REPO)
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'main'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    assert shutil.disk_usage(REPO/'results').free > 60*1024**3
    output = args.output.resolve()
    assert output.is_relative_to(REPO/'results') and output != REPO/'results'
    output.mkdir(parents=True, exist_ok=False)
    spec = dataset_spec('VOC12')
    sources = [PRETRAIN, spec['train'], spec['val'], spec['cam'], spec['root']/'ImageLabel/cls_labels.npy']
    sources += [root/name for root in REFERENCES.values() for name in
                ['mctformerplus_final.pth', 'optimizer_spec.json', 'pretrained_load_report.json', 'model_spec.json']]
    manifest = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    matrix=MATRIX, epochs=45, dataset=audit(spec), save_every_epoch=True,
                    checkpoint_count_expected=180, completed=[],
                    epoch_filename_numbering='001..045; payload epoch=0..44',
                    checkpoint_contents='model, optimizer, scheduler, scaler, main-process RNGs, args, metrics, provenance',
                    resume_note='No change to legacy --resume; not claiming bitwise dataloader-worker replay',
                    source_sha256_before={str(p): sha256(p) for p in sources},
                    evaluation='final checkpoint classification val1449; online raw CAM train1464 native scales1,.75,1.25+flip fixed.45',
                    scope='GWRP and all-product only; no affinity, repair, probe, beta1 sweep, COCO, refinement or segmentation')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_epoch_checkpoints.py',
             'tests/test_mctformerplus_c2p_pooling.py', 'tests/test_raw_cam_streaming.py'], output, 'tests')
        for method in ['gwrp', 'all_product']:
            directory = output/(method+'_smoke')
            directory.mkdir()
            for key, count in [('train', 64), ('val', 4), ('cam', 2)]:
                name = 'cam_train_id.txt' if key=='cam' else key+'_id.txt'
                (directory/name).write_text('\n'.join(spec[key].read_text().splitlines()[:count])+'\n')
            run(training_command(spec, directory, method, 0, directory/'train_id.txt', directory/'val_id.txt', 2, 4), directory, 'train')
            verify_snapshots(directory, 2)
            evaluate(spec, directory, method, smoke=True)
            (directory/'SMOKE_COMPLETE').write_text('complete\n')
        rows = []
        for seed, method in MATRIX:
            name = f'{method}_s{seed}'
            directory = output/name
            directory.mkdir()
            run(training_command(spec, directory, method, seed), directory, 'train')
            verify_snapshots(directory, 45)
            checkpoint = directory/'mctformerplus_final.pth'
            (directory/'checkpoint_sha256.txt').write_text(f'{sha256(checkpoint)}  {checkpoint}\n')
            ref = REFERENCES[method]
            for filename in ['optimizer_spec.json', 'pretrained_load_report.json', 'model_spec.json']:
                actual = json.loads((directory/filename).read_text())
                baseline = json.loads((ref/filename).read_text())
                if filename == 'optimizer_spec.json':
                    assert actual.pop('seed') == seed
                    baseline.pop('seed')
                if filename == 'model_spec.json':
                    actual = canonical_model_spec(actual)
                    baseline = canonical_model_spec(baseline)
                assert actual == baseline, filename
            evaluate(spec, directory, method)
            rows.append(result(directory, method, seed))
            with (output/'comparison.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            manifest['completed'].append(name)
            manifest['source_sha256_after'] = {p: sha256(p) for p in manifest['source_sha256_before']}
            assert manifest['source_sha256_after'] == manifest['source_sha256_before']
            manifest['source_integrity_unchanged'] = True
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            (directory/'RUN_COMPLETE').write_text('complete\n')
        (output/'VOC_EPOCH_CHECKPOINT_REPORT.md').write_text(
            '# VOC matched seeds0/11 with epoch checkpoints\n\nFour45-epoch runs complete;180 immutable epoch snapshots verified.\n'
            'See comparison.csv for final-checkpoint classification and native raw CAM metrics.\n'
            'Epoch archives: each run/epoch_checkpoints/mctformerplus_epoch_001.pth through045, index.jsonl includes SHA256.\n'
            'Best/final files also retained. No per-image CAM dumps; no unrelated experiment launched.\n'
            'Source hashes unchanged. Exact commands, configuration, environments and tests are adjacent.\n')
        (output/'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output/'QUEUE_FAILED').write_text(repr(exc)+'\n')
        raise


if __name__ == '__main__':
    main()
