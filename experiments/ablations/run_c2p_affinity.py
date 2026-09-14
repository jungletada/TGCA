"""Exactly two matched VOC product-pooling + all-layer P2P weight propagation runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.ablations.run_c2p_pooling import classification
from experiments.ablations.run_c2p_product import result_row
from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, audit, dataset_spec, experiment, run
from tools.evaluate_cam_threshold_grid import sha256_file


def report(output, rows, manifest):
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# C2P product + P2P affinity pooling', '', 'Code: ' + manifest['git_sha'], '',
             'Completed new variants: ' + ', '.join(manifest['completed_new_variants']),
             'Queue complete only when QUEUE_COMPLETE exists.', '',
             'w = softmax(sum_layers(log(mean_heads(A_c2p)))); P = sum_all12(mean_heads(A_p2p)).',
             'w_aff = normalize(w @ P.transpose(-1,-2)); patch logits = sum(w_aff * raw M).',
             'FP32 readout accumulation under AMP; no detach, no parameters, no new loss.',
             'P is the native unconditioned submatrix: its query-dependent patch-group mass is preserved.',
             'No P row normalization, symmetrization, extra self-loop or repeated propagation.',
             'Native CAM remains ReLU(M), last3 MEAN C2P multiplication, sqrt, all-layer P2P.',
             'Class initialization/branch, CCT, patch head, optimization and augmentation unchanged.',
             'Fresh matched DeiT-S initialization: seed0, 45 epochs, 448, batch32.',
             'Classification: VOC val1449 FP32 448; CAM: train1464 native three scales, fixed .45.',
             'Best threshold from the same 0:.01:.59 grid is diagnostic, not a tuned training setting.',
             'All five references reused read-only; main comparisons are corresponding products without affinity.',
             'Single seed: no robustness claim. Weight diagnostics do not establish semantic ownership/causality.',
             'Per-run weight_diagnostics/summary.csv: entropy, top1 mass, positive-pair top10% Jaccard.',
             'These use paired before/after readout on the SAME new checkpoint, not a training intervention control.',
             'CI: 5000 image-clustered paired resamples, equal image weights. Examples selected before inference.',
             'Exact commands, source hashes, config, environments, tests and checkpoint hashes saved alongside results.', '',
             '| Pooling | Class mAP | Patch mAP | Class loss | Patch loss | CAM @.45 | Best CAM | Best t | FG P | FG R |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['pooling'] + ' | ' + ' | '.join(
            f'{float(row[k]):.4f}' for k in rows[0] if k != 'pooling') + ' |')
    (output / 'C2P_AFFINITY_REPORT.md').write_text('\n'.join(lines) + '\n')


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
    products = REPO / 'results/c2p_pooling/20260914-voc-product-s0'
    assert (products / 'QUEUE_COMPLETE').exists()
    prior = json.loads((products / 'manifest.json').read_text())
    sources = list(map(Path, prior['source_sha256_before'])) + [products / 'comparison.csv', products / 'manifest.json']
    for layers in ['last3', 'all']:
        sources.extend(products / (layers + '_product') / name for name in [
            'mctformerplus_final.pth', 'optimizer_spec.json', 'model_spec.json',
            'pretrained_load_report.json', 'raw_cam/metrics.json', 'classification/classification_metrics.json'])
    hashes = {str(p): sha256_file(p) for p in sources}
    assert all(hashes[p] == digest for p, digest in prior['source_sha256_before'].items())
    spec = dataset_spec('VOC12')
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, '-m', 'experiments.ablations.run_c2p_affinity', *sys.argv[1:]]),
        'queue': ['last3-product-affinity', 'all-product-affinity'], 'completed_new_variants': [],
        'only_change_vs_corresponding_product': 'c2p_pooling_affinity False -> True',
        'pretrained': str(PRETRAIN), 'pretrained_sha256': sha256_file(PRETRAIN),
        'source_sha256_before': hashes, 'dataset': audit(spec),
        'class_token_init': 'baseline', 'seed': 0, 'epochs': 45, 'input_size': 448, 'batch_size': 32,
        'nominal_lr': 5e-4, 'min_lr': 1e-5, 'checkpoint_policy': 'final',
        'affinity': 'sum_all12(mean_heads(actual A_p2p)); normalize(w @ P.T); no P renormalization',
        'evaluation': 'classification val1449 FP32 448; CAM train1464 native 1,.75,1.25; fixed .45/grid 0:.01:.59',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    kwargs = dict(patch_pooling='c2p', c2p_pooling_reduction='product', c2p_pooling_affinity=True)
    cls_kwargs = dict(c2p_pooling_reduction='product', c2p_pooling_affinity=True)
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_mctformerplus_c2p_affinity.py',
             'tests/test_mctformerplus_c2p_pooling.py', 'tests/test_mctformerplus_variants.py',
             'tests/test_mctformerplus_patch_first.py', 'tests/test_raw_cam_streaming.py',
             'tests/test_width_scaling_aggregation.py'], output, 'tests')
        for layers in ['last3', 'all']:
            smoke = output / (layers + '_product_affinity_smoke')
            experiment(spec, smoke, True, c2p_pooling_layers=layers, **kwargs)
            classification(smoke / 'mctformerplus_final.pth', smoke, spec, 'c2p', 'classification',
                           limit=4, c2p_pooling_layers=layers, **cls_kwargs)
            run([sys.executable, '-u', '-m', 'analysis.c2p_affinity_readout', '--checkpoint',
                 smoke / 'mctformerplus_final.pth', '--layers', layers, '--limit', '4',
                 '--output', smoke / 'weight_diagnostics'], smoke, 'weight_diagnostics')
        with (products / 'comparison.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for layers in ['last3', 'all']:
            directory = output / (layers + '_product_affinity')
            experiment(spec, directory, False, c2p_pooling_layers=layers, **kwargs)
            classification(directory / 'mctformerplus_final.pth', directory, spec, 'c2p', 'classification',
                           c2p_pooling_layers=layers, **cls_kwargs)
            run([sys.executable, '-u', '-m', 'analysis.c2p_affinity_readout', '--checkpoint',
                 directory / 'mctformerplus_final.pth', '--layers', layers,
                 '--output', directory / 'weight_diagnostics'], directory, 'weight_diagnostics')
            ref = products / (layers + '_product')
            for name in ['optimizer_spec.json', 'pretrained_load_report.json']:
                assert json.loads((directory / name).read_text()) == json.loads((ref / name).read_text()), name
            model_spec = json.loads((directory / 'model_spec.json').read_text())
            old_spec = json.loads((ref / 'model_spec.json').read_text())
            assert model_spec.pop('c2p_pooling_affinity') is True
            assert model_spec == old_spec, 'Only affinity may differ in model configuration'
            rows.append(result_row('c2p-' + layers + '-product-affinity', directory))
            manifest['completed_new_variants'].append(layers + '-product-affinity')
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
