"""Fresh vanilla MCTformer+ VOC -> COCO training, ending at raw CAM evaluation.

Uses the existing MCTformer+ VOC baseline recipe for both datasets, rather
than the unrelated mcta defaults in the legacy COCO shell script.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from tools.evaluate_cam_threshold_grid import sha256_file

REPO = Path(__file__).resolve().parents[2]
PRETRAIN = Path('/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth')


def run(command, directory, stage):
    command = [str(x) for x in command]
    print(f'STAGE {stage} {datetime.datetime.now().isoformat()}', flush=True)
    with (directory / 'commands.sh').open('a') as stream:
        stream.write(shlex.join(command) + '\n')
    with (directory / f'{stage}.log').open('x') as stream:
        subprocess.run(command, cwd=REPO, check=True, stdout=stream,
                       stderr=subprocess.STDOUT)
    (directory / f'{stage.upper()}_COMPLETE').write_text('complete\n')


def dataset_spec(name):
    root = REPO / ('data/VOCdevkit/VOC2012' if name == 'VOC12' else 'data/MSCOCO')
    lists = root / 'ImageLists'
    return dict(name=name, root=root, root_flag='--voc12_root' if name == 'VOC12' else '--coco_root',
                train=lists / ('train_aug_id.txt' if name == 'VOC12' else 'train_id.txt'),
                val=lists / 'val_id.txt', cam=lists / 'train_id.txt',
                masks=root / ('SegmentationClass' if name == 'VOC12' else 'MaskSets/train2014'),
                classes=21 if name == 'VOC12' else 81)


def audit(spec):
    root = spec['root']
    label_file = root / 'ImageLabel' / ('cls_labels.npy' if spec['name'] == 'VOC12' else 'COCO_cls_labels.npy')
    labels = np.load(label_file, allow_pickle=True).item()
    report = {'label_sha256': sha256_file(label_file), 'lists': {}}
    for split in ('train', 'val', 'cam'):
        ids = spec[split].read_text().splitlines()
        assert ids and len(ids) == len(set(ids)), (split, 'invalid IDs')
        for name in ids:
            voc = spec['name'] == 'VOC12'
            image_dir = 'JPEGImages' if voc else ('val2014' if split == 'val' else 'train2014')
            assert (root / image_dir / f'{name}.jpg').is_file(), name
            label = labels[name if voc else name + '.jpg']
            assert np.shape(label) == (spec['classes'] - 1,), name
            if split == 'cam':
                assert (spec['masks'] / f'{name}.png').is_file(), name
        report['lists'][split] = {'path': str(spec[split]), 'count': len(ids),
                                  'sha256': sha256_file(spec[split])}
    return report


def experiment(spec, directory, smoke):
    directory.mkdir(exist_ok=False)
    train, val, cam = [spec[k] for k in ('train', 'val', 'cam')]
    if smoke:
        for key, count in (('train', 64), ('val', 4), ('cam', 2)):
            # The native loader routes train/val using the full list path.
            path = directory / ('cam_train_id.txt' if key == 'cam' else f'{key}_id.txt')
            path.write_text('\n'.join(spec[key].read_text().splitlines()[:count]) + '\n')
            if key == 'train':
                train = path
            elif key == 'val':
                val = path
            else:
                cam = path
    common = ['--dataset', spec['name'], '--model', 'mctformerplus',
              spec['root_flag'], spec['root'], '--work_space', directory]
    run([sys.executable, '-u', 'train_model_v2.py', *common,
         '--train_list', train, '--val_list', val, '--input-size', '448',
         '--epochs', '1' if smoke else '45', '--batch_size', '32',
         '--seed', '0', '--lr', '5e-4', '--min-lr', '1e-5',
         '--num_workers', '4' if smoke else '10', '--finetune', PRETRAIN],
        directory, 'train')
    checkpoint = directory / 'mctformerplus_final.pth'
    (directory / 'checkpoint_sha256.txt').write_text(sha256_file(checkpoint) + '  ' + str(checkpoint) + '\n')
    run([sys.executable, '-u', 'make_cam.py', *common, '--train_list', cam,
         '--input_size', '448', '--scales', '1.0,0.75,1.25',
         '--checkpoint', checkpoint, '--cam_out_dir', 'cam_train'], directory, 'cam')
    run([sys.executable, '-u', '-m', 'tools.evaluate_raw_cam_streaming',
         '--cam-dir', directory / 'cam_train', '--mask-dir', spec['masks'],
         '--id-list', cam, '--num-classes', spec['classes'],
         '--output-dir', directory / 'raw_cam'], directory, 'raw_cam_eval')
    (directory / 'RUN_COMPLETE').write_text('complete\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Tracked worktree must be clean')
    if subprocess.check_output(['git', 'branch', '--show-current']).decode().strip() != 'main':
        raise RuntimeError('Expected main checkout')
    import torch
    assert torch.cuda.is_available()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    specs = [dataset_spec(name) for name in ('VOC12', 'COCO')]
    metadata = {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'command': shlex.join([sys.executable, '-m', 'experiments.baselines.run_default_voc_coco', *sys.argv[1:]]),
                'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
                'seed': 0, 'epochs': 45, 'batch_size': 32, 'input_size': 448,
                'nominal_lr': 5e-4, 'min_lr': 1e-5, 'scales': [1.0, 0.75, 1.25],
                'checkpoint_policy': 'final epoch', 'cam_split': 'train',
                'refinement_or_segmentation': False,
                'datasets': {s['name']: audit(s) for s in specs}}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    run([sys.executable, '-m', 'pip', 'freeze'], args.output, 'pip_freeze')
    run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], args.output, 'conda_explicit')
    run([sys.executable, '-m', 'pytest', '-q', 'tests/test_raw_cam_streaming.py',
         'tests/test_width_scaling_aggregation.py'], args.output, 'tests')
    try:
        for spec in specs:
            experiment(spec, args.output / (spec['name'] + '_smoke'), True)
        for spec in specs:
            experiment(spec, args.output / spec['name'], False)
        rows = []
        for spec in specs:
            metrics = json.loads((args.output / spec['name'] / 'raw_cam/metrics.json').read_text())
            for policy in ('fixed', 'best'):
                rows.append({'dataset': spec['name'], 'split': 'train',
                             'policy': policy, **metrics[policy]})
        with (args.output / 'cam_summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        report = ['# Default MCTformer+ raw CAM rerun', '',
                  'Code: ' + metadata['git_sha'], '',
                  'Fresh DeiT-S initialization, seed 0, 45 epochs, batch 32, input 448.',
                  'Native three-scale CAM with training image-level labels; final checkpoint.',
                  'No CRF, downstream refinement, or segmentation training.',
                  'Best threshold is a diagnostic on the evaluation split, not an unbiased selected result.', '',
                  '| Dataset | Threshold policy | Threshold | mIoU (%) | Semantic FG precision (%) | Semantic FG recall (%) |',
                  '|---|---|---:|---:|---:|---:|']
        for row in rows:
            report.append(f"| {row['dataset']} | {row['policy']} | {row['threshold']:.2f} | "
                          f"{100*row['mean_iou']:.3f} | {100*row['semantic_foreground_precision']:.3f} | "
                          f"{100*row['semantic_foreground_recall']:.3f} |")
        (args.output / 'RAW_CAM_REPORT.md').write_text('\n'.join(report) + '\n')
        (args.output / 'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (args.output / 'QUEUE_FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
