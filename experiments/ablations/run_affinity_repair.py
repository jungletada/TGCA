"""Ordered VOC-only repair queue: A1 -> A2/A3 -> conditional B (no other arms)."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from analysis.artifact_probe import CHECKPOINTS
from analysis.affinity_repair import DEFAULT, MINIMAL, CONSERVATIVE, config_id
from experiments.baselines.run_default_voc_coco import REPO, dataset_spec, experiment, run
from experiments.ablations.run_c2p_product import result_row
from tools.evaluate_cam_threshold_grid import sha256_file


def classification(checkpoint, directory, spec, config, limit=0):
    run([sys.executable, '-u', 'tools/evaluate_mctformerplus_classification.py',
         '--checkpoint', checkpoint, '--output-dir', directory/'classification', '--model', 'mctformerplus',
         '--dataset', 'VOC12', '--data-root', spec['root'], '--list-path', spec['val'],
         '--input-size', '448', '--batch-size', '16', '--num-workers', '8',
         '--bootstrap-resamples', '0',
         '--patch-pooling', 'c2p', '--c2p-pooling-layers', 'all', '--c2p-pooling-reduction', 'product',
         '--c2p-pooling-affinity', '--affinity-repair', json.dumps(config, sort_keys=True),
         *(['--limit', str(limit)] if limit else [])], directory, 'classification')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, default=Path('results/artifact_probe/20260916-voc-s0'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Dry run: artifact COMPLETE -> tests -> small A1/A2 smoke -> full A1/A2/A3 -> gated three B runs')
        return
    os.chdir(REPO)
    assert (args.artifact/'COMPLETE').exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'main'
    output = args.output.resolve()
    assert output.is_relative_to(REPO/'results') and output != REPO/'results'
    output.mkdir(parents=True, exist_ok=False)
    sources = [REPO/p/'mctformerplus_final.pth' for p in CHECKPOINTS.values()]
    sources += [args.artifact/'manifest.json', args.artifact/'decision.md']
    manifest = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256_before={str(p.resolve()): sha256_file(p) for p in sources},
                    order=['A1', 'A2 all_product', 'A3 gwrp', 'B only if A2 wins'],
                    scope='VOC only; no register/Gram/multiple seeds/COCO',
                    stage_b_protocol='repair training pooling only; native CAM remains original; 45epochs seed0 matched')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], output, 'conda_explicit')
        run([sys.executable, '-m', 'pytest', '-q', 'tests/test_affinity_repair.py',
             'tests/test_mctformerplus_c2p_affinity.py', 'tests/test_mctformerplus_c2p_pooling.py',
             'tests/test_artifact_probe.py', 'tests/test_raw_cam_streaming.py'], output, 'tests')
        for smoke in [True, False]:
            a1 = output/('a1_smoke' if smoke else 'a1')
            a2 = output/('a2_smoke' if smoke else 'a2')
            run([sys.executable, '-u', '-m', 'analysis.affinity_repair', 'a1',
                 '--output', a1, '--artifact', args.artifact,
                 *(['--limit', '8'] if smoke else [])], output, a1.name)
            run([sys.executable, '-u', '-m', 'analysis.affinity_repair', 'a2',
                 '--output', a2, '--a1', a1,
                 *(['--limit', '2'] if smoke else [])], output, a2.name)
        decision = json.loads((output/'a2/decision.json').read_text())
        manifest['screening_decision'] = decision
        rows = []
        if decision['stage_b_gate_passed']:
            spec = dataset_spec('VOC12')
            reference = REPO/CHECKPOINTS['all_product']
            configs = [MINIMAL, decision['best_config'], CONSERVATIVE]
            completed = {}
            for arm, config in zip(['minimal', 'a2_best', 'conservative'], configs):
                key = config_id(config)
                if key in completed:
                    # Same configuration is one experiment, not two independent replicates.
                    manifest[arm+'_alias'] = completed[key]
                    continue
                kwargs = dict(patch_pooling='c2p', c2p_pooling_layers='all', c2p_pooling_reduction='product',
                              c2p_pooling_affinity=True, affinity_repair=config, online_raw_eval=True)
                smoke = output/(arm+'_smoke')
                experiment(spec, smoke, True, **kwargs)
                classification(smoke/'mctformerplus_final.pth', smoke, spec, config, limit=4)
                directory = output/arm
                experiment(spec, directory, False, **kwargs)
                classification(directory/'mctformerplus_final.pth', directory, spec, config)
                for name in ['optimizer_spec.json', 'pretrained_load_report.json']:
                    assert json.loads((directory/name).read_text()) == json.loads((reference/name).read_text()), name
                observed = json.loads((directory/'model_spec.json').read_text())
                original = json.loads((reference/'model_spec.json').read_text())
                assert observed.pop('affinity_repair') == config
                assert observed.pop('c2p_pooling_affinity') is True
                original.pop('c2p_pooling_affinity', None)
                assert observed == original
                rows.append(result_row(arm, directory))
                completed[key] = arm
                (directory/'VARIANT_COMPLETE').write_text('complete\n')
                with (output/'stage_b_comparison.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        else:
            (output/'STAGE_B_NOT_RUN').write_text('A2 did not exceed historical fixed-threshold all-product; stop at gate.\n')
        manifest['source_sha256_after'] = {p: sha256_file(p) for p in manifest['source_sha256_before']}
        assert manifest['source_sha256_before'] == manifest['source_sha256_after']
        manifest['source_integrity_unchanged'] = True
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        lines = ['# Artifact and affinity repair final status', '',
                 'Artifact decision: '+str(args.artifact/'decision.md'),
                 'A1 screening: a1/summary.csv; A2/A3: a2/AFFINITY_SCREENING_REPORT.md.',
                 'Stage B gate passed: '+str(decision['stage_b_gate_passed']),
                 'Completed distinct B arms: '+', '.join(r['pooling'] for r in rows),
                 'B changes training pooling only; its reported CAM uses unchanged native refinement.',
                 'Inference screening and B training results are distinct. No causal claim from A2 alone.',
                 'All single-seed; no additional experiment is queued. Source hashes unchanged.',
                 'Exact commands, environment, tests, configs, checkpoints and metric JSONs are adjacent.']
        (output/'ARTIFACT_AFFINITY_REPORT.md').write_text('\n'.join(lines)+'\n')
        (output/'QUEUE_COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output/'QUEUE_FAILED').write_text(repr(exc)+'\n')
        raise


if __name__ == '__main__':
    main()
