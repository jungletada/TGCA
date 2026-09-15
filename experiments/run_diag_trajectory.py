"""Prepare/run matched diagnostic trajectories; never auto-start conditional arms.

Default invocation prints the matrix. --execute explicitly starts it. VOC is
DIAG-A/B x seeds0,1,2. COCO is the separate DIAG-D seed0 entry, not chained to VOC.
DIAG-C's unspecified single-CLS classifier/loss/CAM and conditional 3x training
are deliberately not synthesized by this infrastructure runner.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from analysis.diagnostics.probe_set import build_sets
from analysis.diagnostics.writer import REPO, result_path, sha256, write_json
from experiments.baselines.run_default_voc_coco import dataset_spec, PRETRAIN, run


def matrix(dataset):
    return [(seed, pooling) for seed in (0, 1, 2) for pooling in ('gwrp', 'c2p')] if dataset == 'VOC12' else [(0, 'c2p')]


def training_command(spec, directory, seed, pooling, probe, mini):
    return [sys.executable, '-u', 'train_model_v2.py', '--dataset', spec['name'],
            '--model', 'mctformerplus', spec['root_flag'], str(spec['root']),
            '--train_list', str(spec['train']), '--val_list', str(spec['val']),
            '--work_space', str(directory), '--input-size', '448', '--epochs', '45',
            '--batch_size', '32', '--seed', str(seed), '--lr', '5e-4', '--min-lr', '1e-5',
            '--num_workers', '10', '--finetune', str(PRETRAIN),
            '--patch-pooling', pooling, '--c2p-pooling-layers', 'all', '--c2p-pooling-reduction', 'product',
            '--probe', '--probe-set', str(probe), '--probe-mini-set', str(mini),
            '--probe-mini-mask-dir', str(spec['masks'])]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=['VOC12', 'COCO'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = result_path(args.output)
    if not args.execute:
        print(json.dumps({'dataset': args.dataset, 'matrix': matrix(args.dataset), 'output': str(output),
                          'training_started': False, 'note': 'Use --execute only in a new tmux queue after checking GPU/current runs.'}, indent=2))
        return
    os.chdir(REPO)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Commit implementation before launching a matched queue')
    output.mkdir(parents=True, exist_ok=False)
    spec = dataset_spec(args.dataset)
    baseline = REPO / 'results/default_mctformerplus/20260912-voc-coco-s0-r2' / args.dataset
    hashes = {str(p): sha256(p) for p in (PRETRAIN, baseline / 'optimizer_spec.json', baseline / 'pretrained_load_report.json', spec['train'], spec['val'])}
    write_json(output / 'manifest.json', {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                        'matrix': matrix(args.dataset), 'source_sha256': hashes,
                                        'counterfactual_c2p_for_gwrp': 'all-product, identical to DIAG-B; does not change GWRP training',
                                        'scope': 'DIAG-A/B or explicitly invoked D; no C/E, no automatic solution',
                                        'epochs': 45, 'fixed_set_seed': 0})
    try:
        probe, mini = build_sets(spec['root'], args.dataset, spec['train'], spec['masks'], output / 'sets')
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_diagnostics.py'], output, 'tests')
        diagnostic_paths = []
        for seed, pooling in matrix(args.dataset):
            directory = output / f'{pooling}_s{seed}'
            directory.mkdir(exist_ok=False)
            run(training_command(spec, directory, seed, pooling, probe, mini), directory, 'train')
            # Preserve matching optimizer/init proof apart from the registered seed.
            for name in ('optimizer_spec.json', 'pretrained_load_report.json'):
                observed = json.loads((directory / name).read_text())
                reference = json.loads((baseline / name).read_text())
                if name == 'optimizer_spec.json':
                    observed.pop('seed'); reference.pop('seed')
                if observed != reference:
                    raise RuntimeError(f'Matched recipe mismatch: {name}')
            write_json(directory / 'checkpoint_hashes.json', {p.name: sha256(p) for p in directory.glob('*.pth')})
            (directory / 'RUN_COMPLETE').touch(exist_ok=False)
            diagnostic_paths.append(str(directory / 'diagnostics'))
        run([sys.executable, '-m', 'analysis.diagnostics.plot_trajectory', *diagnostic_paths,
             '--output', output / 'trajectory_analysis'], output, 'trajectory_analysis')
        if any(sha256(path) != expected for path, expected in hashes.items()):
            raise RuntimeError('Source integrity changed')
        (output / 'QUEUE_COMPLETE').touch(exist_ok=False)
    except BaseException as exc:
        (output / 'QUEUE_FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
