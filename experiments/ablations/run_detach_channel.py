"""E1 detach and E4 A1-a: four fresh matched VOC runs, then frozen trajectories."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, dataset_spec, audit, run
from experiments.ablations.run_voc_epoch_checkpoints import training_command, verify_snapshots, canonical_model_spec
from experiments.ablations.evaluate_voc_epoch_trajectory import cam_command, read_result, cleanup_predictions, write_json
from analysis.detach_channel_diagnostics import classification_command
from tools.epoch_checkpoints import sha256

SOURCE=REPO/'results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2'
OLD_EVAL=REPO/'results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11'
MATRIX=[('detach',0),('detach',11),('a1',0),('a1',11)]


def extra_flags(variant):
    return ['--detach-weights'] if variant=='detach' else ['--channel-agg','--channel-agg-lr-mult','1']


def checkpoint_job(directory,variant,seed,epoch):
    index=[json.loads(x) for x in (directory/'epoch_checkpoints/index.jsonl').read_text().splitlines()]
    entry=index[epoch-1]
    assert entry['epoch']==epoch-1
    checkpoint=directory/'epoch_checkpoints'/entry['filename']
    assert checkpoint.name==f'mctformerplus_epoch_{epoch:03d}.pth'
    return dict(method='gwrp' if variant in ('a1','gwrp') else 'all_product',seed=seed,epoch=epoch,
                run=f'{variant}_s{seed}',checkpoint=str(checkpoint),checkpoint_sha256=entry['sha256'])


def evaluate_epoch(spec,directory,variant,job,smoke=False):
    if (directory/'COMPLETE').exists():
        row=json.loads((directory/'result.json').read_text())
        assert row['checkpoint_sha256']==job['checkpoint_sha256']
        return row
    directory.mkdir(parents=True,exist_ok=False)
    assert sha256(Path(job['checkpoint']))==job['checkpoint_sha256']
    cam_list=None
    if smoke:
        cam_list=directory/'cam_train_id.txt'
        cam_list.write_text('\n'.join(spec['cam'].read_text().splitlines()[:2])+'\n')
    command=cam_command(spec,directory,job,cam_list)
    command+=['--detach-weights'] if variant=='detach' else ['--channel-agg']
    run(command,directory,'cam')
    run(classification_command(job['checkpoint'],directory/'classification',variant,4 if smoke else 0),
        directory,'classification')
    row=read_result(directory,job,(4,2) if smoke else (1449,1464))
    row['method']=variant
    assert sha256(Path(job['checkpoint']))==job['checkpoint_sha256']
    write_json(directory/'result.json',row)
    cleanup_predictions(directory)
    (directory/'COMPLETE').write_text('complete\n')
    return row


def diagnostics(directory,job,smoke=False):
    if (directory/'diagnostics/COMPLETE').exists():
        data=json.loads((directory/'diagnostics/metrics.json').read_text())
        assert data['checkpoint_sha256']==job['checkpoint_sha256']
        return
    directory.mkdir(parents=True,exist_ok=True)
    flags=['--artifacts'] if smoke or job['epoch'] in (15,30,45) else []
    run([sys.executable,'-u','-m','analysis.detach_channel_diagnostics','--checkpoint',job['checkpoint'],
         '--output',directory/'diagnostics','--limit','4' if smoke else '0',*flags],directory,'diagnostics')


def gates(new_rows,old_rows):
    def value(rows,method,seed):
        return next(r['fixed_mean_iou_percent'] for r in rows if r['method']==method and r['seed']==seed)
    d0,d11=[value(new_rows,'detach',s) for s in (0,11)]
    base=[value(old_rows,'gwrp',s) for s in (0,11)]
    if d0<base[0] and d11<base[1]:e1='both_below_V0_stop'
    elif d11>=68.1 and d0>=71.1:e1='repair_gate_passed_requires_more_seeds'
    elif d11>=68.1:e1='s11_recovered_s0_gain_not_retained'
    else:e1='s11_not_recovered_fallback_candidate'
    delta=[value(new_rows,'a1',s)-b for s,b in zip((0,11),base)]
    if min(delta)>=.5:e4='both_at_least_plus_0.5_requires_more_seeds'
    elif max(delta)<0:e4='both_negative'
    elif min(delta)<0<max(delta):e4='mixed_seed_signs'
    else:e4='small_or_zero_effect'
    return dict(E1=e1,E1_thresholds=dict(s11=68.1,s0=71.1),E4=e4,E4_deltas=delta,
                interpretation='Two-seed descriptive gates, not proof of a causal feedback loop or population benefit',
                extensions='No automatic fallback/A1-b/5-seed launch; conditional ambiguity recorded for user decision')


def old_rows():
    rows=pd.read_csv(OLD_EVAL/'epoch_metrics.csv').to_dict('records')
    assert len(rows)==180
    return rows


def collect(output):
    rows=old_rows()
    for path in sorted(output.glob('*/evaluation/epoch_*/result.json')):
        if (path.parent/'COMPLETE').exists():rows.append(json.loads(path.read_text()))
    return rows


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=collect(output)
    pd.DataFrame(rows).sort_values(['method','seed','epoch']).to_csv(output/'epoch_metrics.csv',index=False)
    panels=[('class_macro_mAP_percent','Class macro mAP (%)'),('patch_macro_mAP_percent','Patch macro mAP (%)'),
            ('fixed_mean_iou_percent','Raw CAM mIoU @0.45 (%)'),('fixed_semantic_foreground_recall_percent','FG recall @0.45 (%)')]
    colors=dict(gwrp='tab:blue',all_product='tab:orange',detach='tab:green',a1='tab:red')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(key,title) in zip(axes.flat,panels):
        for method,color in colors.items():
            for seed in (0,11):
                points=sorted([r for r in rows if r['method']==method and r['seed']==seed],key=lambda r:r['epoch'])
                ax.plot([r['epoch'] for r in points],[r[key] for r in points],color=color,
                        linestyle='-' if seed==0 else '--',marker='.' if len(points)==1 else None,
                        label=f'{method} s{seed}')
        ax.set(title=title,xlabel='Epoch',xlim=(1,45));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);fig.tight_layout()
    fig.savefig(output/'epoch_curves.svg');fig.savefig(output/'epoch_curves.png',dpi=140);plt.close(fig)
    diag=[]
    for path in output.glob('*/evaluation/epoch_*/diagnostics/metrics.json'):
        d=json.loads(path.read_text());name=path.parents[3].name
        for key,item in d['metrics'].items():
            diag.append(dict(run=name,epoch=d['epoch'],metric=key,mean=item['mean'],n_images=item['n_images']))
        for key,value in d['channel'].items():
            diag.append(dict(run=name,epoch=d['epoch'],metric=key,mean=value,n_images=0))
    if diag:
        frame=pd.DataFrame(diag);frame.to_csv(output/'diagnostic_trajectories.csv',index=False)
        keys=['c2p_all_product_kappa','c2p_all_product_top1_mass','rho_cc',
              'channel_entropy','channel_max_min_ratio','channel_l1_uniform']
        fig,axes=plt.subplots(3,2,figsize=(13,11))
        for ax,key in zip(axes.flat,keys):
            for name,part in frame[frame.metric==key].groupby('run'):
                part=part.sort_values('epoch');method,seed=name.rsplit('_s',1)
                ax.plot(part.epoch,part['mean'],label=name,color=colors[method],linestyle='-' if seed=='0' else '--')
            ax.set(title=key,xlabel='Epoch',xlim=(1,45));ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8,ncol=2);fig.tight_layout()
        fig.savefig(output/'diagnostic_curves.svg');plt.close(fig)


def report(output,decision):
    rows=collect(output);final=[r for r in rows if r['epoch']==45]
    pd.DataFrame(final).to_csv(output/'final_comparison.csv',index=False)
    lines=['# MCTformer+ detach and channel aggregation','',
           'Classification: VOC val1449 FP32 single448 macro-class AP. CAM: VOC train1464 native scales/flip, threshold .45.',
           'Existing V0/V1 reused; four new45-epoch runs, all epoch checkpoints retained. No new loss or initialization changes.',
           'GWRP rank weights have zero derivative almost everywhere; sorted values remain differentiable. Detach cuts only the explicit pooling-weight path, not all gradients into attention.',
           'For IDENTICAL weights, detach changes no forward/CAM value. Differently TRAINED checkpoints need not produce identical CAMs.',
           'A1 theta0 exactly matches mean via mean(T)+sum((softmax(theta)-1/D)*T);384 zero-init parameters, no weight decay, LR multiplier1.',
           'A1 is GWRP-only; no BG token. CCT uses raw tokens; native CAM refinement remains unchanged.',
           'L1 is recorded as requested; total variation equals L1/2. Unit-sum channel weights preserve a convex readout, not a guarantee of identical trained logit distributions.',
           'Gates are descriptive with two seeds. No claim that all variance is caused by C2P, nor that feedback lock-in is proven.',
           'No literature novelty claim is inferred from this experiment. No unapproved follow-up launched.','',
           '| Method | Seed | Class mAP | Patch mAP | Fixed CAM mIoU | FG precision | FG recall |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(final,key=lambda x:(x['method'],x['seed'])):
        lines.append(f"| {r['method']} | {r['seed']} | {r['class_macro_mAP_percent']:.3f} | {r['patch_macro_mAP_percent']:.3f} | {r['fixed_mean_iou_percent']:.3f} | {r['fixed_semantic_foreground_precision_percent']:.3f} | {r['fixed_semantic_foreground_recall_percent']:.3f} |")
    lines+=['','## Preregistered decision','',f"E1: `{decision['E1']}`. E4: `{decision['E4']}`.",
            'Gate priority: both V2 seeds below V0 -> stop; otherwise fixed68.1/71.1 thresholds. No threshold retuning.',
            'A1 entropy >.98 is a near-uniform flag, not proof that the mean is optimal. Conditional A1-b/fallback requires the unresolved queue choice.',
            '',f'Epoch metrics available: {len(rows)}/360 (180 historical +180 new).',
            'See epoch_metrics.csv/epoch_curves.svg and diagnostic_trajectories.csv/diagnostic_curves.svg.',
            'M2/M4 are measured at epochs15/30/45, all12 layers; rho_cc uses multi-label images. No semantic masks in diagnostics.',
            'Full-evaluation inference predictions are cleaned only after verified numeric outputs are saved. Sources/checkpoints are immutable.',
            '', '## Appendix: GWRP gradient control', '',
            '`tests/test_detach_pooling.py::test_gwrp_weight_path_is_zero_but_value_path_differentiable`',
            'scatters the exact native rank weights back to spatial order. They have requires_grad=False and grad_fn=None.',
            'The native sorted-logit output equals their weighted sum within FP32 rounding; backward of the sum of logits',
            'equals the spatial rank weights. Thus the rank-weight derivative is zero almost everywhere (ties are nonsmooth),',
            'not the value derivative and not the total gradient into backbone attention. Leaf-attention tests separately',
            'verify C2P weight gradients are nonzero with detach off and absent with detach on; forward values are bitwise equal.']
    (output/'DETACH_AND_CHANNEL_AGG_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.chdir(REPO)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no']).strip()
    assert (SOURCE/'QUEUE_COMPLETE').exists() and (OLD_EVAL/'QUEUE_COMPLETE').exists()
    assert shutil.disk_usage(REPO/'results').free>65*1024**3
    output=args.output.resolve();assert output.is_relative_to(REPO/'results') and output!=REPO/'results'
    output.mkdir(parents=True,exist_ok=False)
    spec=dataset_spec('VOC12')
    sources=[PRETRAIN,SOURCE/'manifest.json',SOURCE/'comparison.csv',OLD_EVAL/'epoch_metrics.csv',
             REPO/'docs/MCTformerPlus_Detach_and_ChannelAgg_Plan.md']
    manifest=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  matrix=MATRIX,dataset=audit(spec),epochs=45,source_sha256_before={str(p):sha256(p) for p in sources},
                  completed_training=[],scope='Only E1 V2 and E4 A1-a; no automatic conditional extension',
                  E1_thresholds=dict(s11=68.1,s0=71.1),E4_delta_threshold=.5,early_warning=dict(epoch=20,macro_ap_below=80),
                  diagnostic_protocol='val1449 FP32; positive-class image averages; all-product weights for ALL hosts; M2/M4 ep15/30/45 all12 layers')
    write_json(output/'manifest.json',manifest)
    try:
        run([sys.executable,'-m','pip','freeze'],output,'pip_freeze')
        run(['/home/peng/anaconda3/bin/conda','list','-n','tgca-repro','--explicit'],output,'conda_explicit')
        run([sys.executable,'-m','pytest','-q','tests/test_detach_pooling.py','tests/test_channel_agg.py',
             'tests/test_detach_channel_experiment.py','tests/test_mctformerplus_c2p_pooling.py',
             'tests/test_epoch_checkpoints.py','tests/test_raw_cam_streaming.py'],output,'tests')
        for variant in ('detach','a1'):
            directory=output/(variant+'_smoke');directory.mkdir()
            for split,count in [('train',64),('val',4)]:
                (directory/(split+'_id.txt')).write_text('\n'.join(spec[split].read_text().splitlines()[:count])+'\n')
            command=training_command(spec,directory,'gwrp' if variant=='a1' else 'all_product',0,
                                     directory/'train_id.txt',directory/'val_id.txt',2,4)+extra_flags(variant)
            run(command,directory,'train');verify_snapshots(directory,2)
            job=checkpoint_job(directory,variant,0,2)
            evaluate_epoch(spec,directory/'evaluation',variant,job,True)
            diagnostics(directory/'evaluation',job,True)
            (directory/'SMOKE_COMPLETE').write_text('complete\n')
        final=[]
        for variant,seed in MATRIX:
            name=f'{variant}_s{seed}';directory=output/name;directory.mkdir()
            command=training_command(spec,directory,'gwrp' if variant=='a1' else 'all_product',seed)
            run(command+extra_flags(variant)+['--epoch20-classification-warning'],directory,'train')
            verify_snapshots(directory,45)
            base=SOURCE/f"{'gwrp' if variant=='a1' else 'all_product'}_s{seed}"
            actual=json.loads((directory/'optimizer_spec.json').read_text());expected=json.loads((base/'optimizer_spec.json').read_text())
            actual.pop('channel_agg_lr_mult',None);actual.pop('channel_theta_weight_decay',None)
            assert actual==expected,'Matched optimizer/training recipe differs'
            actual=json.loads((directory/'pretrained_load_report.json').read_text());expected=json.loads((base/'pretrained_load_report.json').read_text())
            actual.pop('zero_initialized_keys',None);assert actual==expected,'Matched DeiT initialization differs'
            actual=canonical_model_spec(json.loads((directory/'model_spec.json').read_text()))
            expected=canonical_model_spec(json.loads((base/'model_spec.json').read_text()))
            actual.pop('detach_weights',None);actual.pop('channel_agg',None)
            assert actual==expected,'Unrelated model change'
            checkpoint=directory/'mctformerplus_final.pth'
            (directory/'checkpoint_sha256.txt').write_text(f'{sha256(checkpoint)}  {checkpoint}\n')
            job=checkpoint_job(directory,variant,seed,45)
            evaluation=directory/'evaluation/epoch_045'
            final.append(evaluate_epoch(spec,evaluation,variant,job))
            diagnostics(evaluation,job)
            manifest['completed_training'].append(name);write_json(output/'manifest.json',manifest)
            (directory/'TRAINING_AND_FINAL_EVAL_COMPLETE').write_text('complete\n')
            render(output)
        decision=gates(final,[r for r in old_rows() if r['epoch']==45])
        decision['A1_near_uniform']={str(seed):json.loads((output/f'a1_s{seed}/evaluation/epoch_045/diagnostics/metrics.json').read_text())['channel']['channel_entropy']>.98 for seed in (0,11)}
        write_json(output/'decision.json',decision);report(output,decision)
        for epoch in range(1,46):
            for variant,seed in MATRIX:
                directory=output/f'{variant}_s{seed}';job=checkpoint_job(directory,variant,seed,epoch)
                evaluation=directory/f'evaluation/epoch_{epoch:03d}'
                evaluate_epoch(spec,evaluation,variant,job);diagnostics(evaluation,job)
            # Reuse historical classification/CAM; compute missing frozen diagnostics only.
            for variant in ('gwrp','all_product'):
                for seed in (0,11):
                    directory=SOURCE/f'{variant}_s{seed}'
                    job=checkpoint_job(directory,variant,seed,epoch)
                    evaluation=output/f'{variant}_s{seed}/evaluation/epoch_{epoch:03d}'
                    diagnostics(evaluation,job)
            render(output)
        manifest['source_sha256_after']={p:sha256(p) for p in manifest['source_sha256_before']}
        assert manifest['source_sha256_after']==manifest['source_sha256_before']
        manifest['source_integrity_unchanged']=True;manifest['status']='complete'
        write_json(output/'manifest.json',manifest);report(output,decision)
        (output/'QUEUE_COMPLETE').write_text('Four trainings;360 epoch metric rows;360 diagnostic checkpoints; no extension launched\n')
    except BaseException as exc:
        (output/'QUEUE_FAILED').write_text(repr(exc)+'\n');raise


if __name__=='__main__':main()
