"""Frozen FP32 H/A scans, native multi-scale CAM protocol; no training entry point."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from analysis.alpha_sweep import ALPHAS, blend, native_blend
from analysis.artifact_probe import dataset, REPO
from analysis.head_alpha_preflight import HOSTS, SOURCES, fp32_setup, load_host, write_json
from analysis.head_alpha_statistics import (stratified_hash_folds, paired_ci, crossfit,
                                          metric_rows, unlabeled_ranking)
from analysis.head_heatmap import CELLS, TOP_M, head_statistics
from datasets_cam import VOC12DatasetMS, build_transform
from models.adapter_modules import resize_input_minbound
from tools.epoch_checkpoints import sha256
from tools.evaluate_cam_threshold_grid import (cam_payload_winner, image_threshold_confusions,
                                              threshold_grid, confusion_metrics)
from experiments.baselines.run_default_voc_coco import run

ROOT = REPO/'data/VOCdevkit/VOC2012'


def alpha_crossfit_summary(fixed, folds, configs, reps=5000):
    result=crossfit(fixed,folds,reps=reps)
    result['selected_alphas']=[configs[k]['alpha'] for k in result['selected_candidate_indices']]
    result['stable_within_one_step']=abs(result['selected_alphas'][0]-result['selected_alphas'][1])<=.05000001
    result['reference_description']='EXACT native alpha.5 without floor; reselect alpha per bootstrap draw'
    return result


def configs_for(stage, ranking=None, head_rows=None):
    native=[dict(id='native', kind='native')]
    if stage=='alpha':
        return native+[dict(id=f'alpha_{a:.2f}',kind='alpha',alpha=a) for a in ALPHAS]
    if stage=='head':
        return native+[dict(id=f'L{l+1:02d}_H{h+1}',kind='head',layer=l,head=h) for l,h in CELLS]
    if stage=='control':
        order=sorted(head_rows,key=lambda r:(r['fixed_mean_iou_percent'],r['id']))
        chosen=[order[-1],order[len(order)//2],order[0]]
        cells=[(r['layer'],r['head']) for r in chosen]
        return native+[dict(id=f'L{l+1:02d}_H{h+1}_rownorm_{norm}',kind='head',layer=l,head=h,rownorm=norm)
                       for l,h in cells for norm in (False,True)]
    if stage=='topm':
        return native+[dict(id=f'{criterion}_top{m}',kind='topm',criterion=criterion,
                            cells=ranking[criterion][:m],m=m) for criterion in ('kappa','gamma') for m in TOP_M]
    raise ValueError(stage)


def extract(model, inputs, positive):
    """Only compact C2P/P2P summaries survive the once-per-view forward."""
    _,patches,records,_=model.forward_features(inputs)
    assert inputs.dtype==torch.float32 and all(a.dtype==torch.float32 for a in records)
    h,w=inputs.shape[-2]//16,inputs.shape[-1]//16
    maps=model.head(patches.transpose(1,2).reshape(len(inputs),384,h,w))
    means=torch.stack([r.mean(1) for r in records])
    a=means[-3:].mean(0)[:, :20,20:][:,positive]
    p=means[:,:,20:,20:].sum(0)
    cells=torch.stack([r[:,:, :20,20:][:,:,positive] for r in records])
    return maps[:,positive].flatten(2),a,p,cells,(h,w),maps,means


def seed_for(config, maps, native, cells):
    kind=config['kind']
    if kind=='alpha':
        return blend(maps,native,config['alpha'])
    if kind=='native':
        a=native
    elif kind=='head':
        a=cells[config['layer'],:,config['head']]
        if config.get('rownorm',False):
            a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
    elif kind=='topm':
        a=torch.stack([cells[CELLS[i][0],:,CELLS[i][1]] for i in config['cells']]).mean(0)
    else:
        raise ValueError(kind)
    return native_blend(maps,a)


def propagate(seeds,p):
    # seeds[K,B,C,N]. Each configuration shares exactly the SAME native P.
    return torch.matmul(p[None,:,None],seeds.unsqueeze(-1)).squeeze(-1)


def normalize(cams):
    low=cams.flatten(-2).amin(-1)[...,None,None]
    high=cams.flatten(-2).amax(-1)[...,None,None]
    return (cams-low)/(high-low+1e-8)


@torch.no_grad()
def unlabeled(host, output, limit=0):
    output.mkdir(parents=True,exist_ok=False)
    model,_=load_host(host)
    data=dataset()
    n=limit or len(data)
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,range(n)),batch_size=4,num_workers=4)
    values=[]
    minimum=np.full(72,np.inf)
    for images,labels in loader:
        _,_,records,_=model.forward_features(images.cuda())
        cells=torch.stack([r[:,:, :20,20:] for r in records]).permute(0,2,1,3,4).flatten(0,1)
        values.append(head_statistics(cells,labels.cuda()).cpu().numpy())
        nonzero=cells.masked_fill(cells<=0,float('inf')).flatten(1).amin(1).cpu().numpy()
        minimum=np.minimum(minimum,nonzero)
        if sum(len(v) for v in values)%100==0:
            print(f'unlabeled {host} {sum(len(v) for v in values)}/{n}',flush=True)
    values=np.concatenate(values)
    means=np.nanmean(values,0)
    ranking={c:unlabeled_ranking(means,c) for c in ('kappa','gamma')}
    np.savez_compressed(output/'per_image_metrics.npz',values=values,image_ids=data.img_name_list[:n])
    write_json(output/'rankings.json',ranking)
    # Whole image bootstrap, valid image counts can differ for gamma.
    valid=np.isfinite(values)
    rng=np.random.default_rng(0)
    samples=[]
    for start in range(0,5000,100):
        w=rng.multinomial(n,np.full(n,1/n),size=100)
        num=w@np.nan_to_num(values).reshape(n,-1)
        den=w@valid.reshape(n,-1)
        samples.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0).reshape(100,72,3))
    ci=np.nanquantile(np.concatenate(samples),[.025,.975],axis=0)
    rows=[]
    for i,(l,h) in enumerate(CELLS):
        row=dict(layer=l,head=h,id=f'L{l+1:02d}_H{h+1}',minimum_nonzero=float(minimum[i]),warning_below_1e30=bool(minimum[i]<1e-30))
        for j,name in enumerate(('kappa','gini','gamma')):
            row.update({name:float(means[i,j]),name+'_lo':float(ci[0,i,j]),name+'_hi':float(ci[1,i,j]),name+'_images':int(valid[:,i,j].sum())})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'head_statistics.csv',index=False)
    write_json(output/'protocol.json',dict(dataset='VOCval',images=n,semantic_gt_loaded=False,
               transform=str(data.transform),gamma_missing_for_single_label=True,
               ranking='ascending kappa, ascending gamma; lexicographic L/H tie-break; frozen before head mIoU',bootstrap=5000))
    del model,records,cells
    torch.cuda.empty_cache()
    return ranking,rows


@torch.no_grad()
def scan(host, stage, output, configs, folds, limit=0):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'configs.json',configs)
    model,_=load_host(host)
    data=VOC12DatasetMS(str(ROOT),str(ROOT/'ImageLists/train_id.txt'),(1.,.75,1.25),
                       build_transform(False,True,argparse.Namespace(input_size=448)))
    n=limit or len(data)
    thresholds=threshold_grid(0,.59,.01)
    totals=np.zeros((len(configs),60,21,21),np.int64)
    fixed=np.zeros((len(configs),n,21,21),np.int32)
    native_thresholds=np.zeros((60,n,21,21),np.int32) if stage=='alpha' else None
    areas=np.zeros((len(configs),2),np.int64)
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,range(n)),batch_size=1,num_workers=4)
    max_native_error=0.
    start=time.monotonic()
    for i,pack in enumerate(loader):
        size=[int(s.item()) for s in pack['size']]
        positive=pack['label'][0].nonzero().flatten().cuda()
        views=[]
        for image in pack['img']:
            inputs=resize_input_minbound(image[0].cuda(),min_size=448)
            maps,a,p,cells,grid,allmaps,means=extract(model,inputs,positive)
            view_parts=[]
            for start_config in range(0,len(configs),8):
                group=configs[start_config:start_config+8]
                seeds=torch.stack([seed_for(c,maps,a,cells) for c in group])
                cams=propagate(seeds,p).reshape(len(group),2,len(positive),*grid)
                if start_config==0 and i==0:
                    official=model.get_cam(allmaps,means)[:,positive]
                    torch.testing.assert_close(cams[0],official,rtol=1e-6,atol=1e-6)
                    max_native_error=max(max_native_error,float((cams[0]-official).abs().max()))
                up=F.interpolate(cams.flatten(0,1),size,mode='bilinear',align_corners=False)
                up=up.reshape(len(group),2,len(positive),*size)
                view_parts.append(up[:,0]+up[:,1].flip(-1))
            views.append(torch.cat(view_parts))
        normalized=normalize(torch.stack(views).sum(0)).cpu().numpy()
        image_id=pack['name'][0]
        with Image.open(ROOT/'SegmentationClass'/f'{image_id}.png') as image:
            target=np.asarray(image)
        positive=positive.cpu().tolist()
        for j in range(len(configs)):
            scores,classes=cam_payload_winner(dict(zip(positive,normalized[j])),target.shape,21)
            confusion=image_threshold_confusions(scores,classes,target,thresholds,21)
            totals[j]+=confusion
            fixed[j,i]=confusion[45]
            areas[j]+=[int((scores>.45).sum()),scores.size]
            if native_thresholds is not None and j==0:
                native_thresholds[:,i]=confusion
        if (i+1)%50==0 or i+1==n:
            print(f'{stage} {host} {i+1}/{n} elapsed={time.monotonic()-start:.1f}s',flush=True)
    absolute_ci,delta_ci=paired_ci(fixed)
    rows=[]
    for j,(config,metrics) in enumerate(zip(configs,metric_rows(totals))):
        rows.append(dict(host=host,stage=stage,**config,**metrics,
                         foreground_area_fraction_all=float(areas[j,0]/areas[j,1]),
                         ci95_lo_percent=float(100*absolute_ci[0,j]),ci95_hi_percent=float(100*absolute_ci[1,j]),
                         delta_pp=metrics['fixed_mean_iou_percent']-metric_rows(totals[:1])[0]['fixed_mean_iou_percent'],
                         delta_ci95_lo_pp=float(100*delta_ci[0,j]),delta_ci95_hi_pp=float(100*delta_ci[1,j])))
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    np.savez_compressed(output/'confusions.npz',total=totals,fixed=fixed,image_ids=data.img_name_list[:n],thresholds=thresholds)
    # Compact long-form curves (best threshold is diagnostic only).
    curve=[]
    for config,total in zip(configs,totals):
        m=confusion_metrics(total)
        for t,threshold in enumerate(thresholds):
            curve.append(dict(id=config['id'],threshold=threshold,miou_percent=float(100*m['mean_iou'][t])))
    pd.DataFrame(curve).to_csv(output/'threshold_curves.csv',index=False)
    if stage=='alpha':
        result=alpha_crossfit_summary(fixed,folds[:n],configs)
        # The literal clamped alpha.5 reference is reported separately too.
        with_clamped_ref=fixed.copy()
        with_clamped_ref[0]=fixed[1+ALPHAS.index(.5)]
        result['vs_clamped_alpha_half']=crossfit(with_clamped_ref,folds[:n])
        write_json(output/'alpha_crossfit.json',result)
        threshold_result=crossfit(np.concatenate([fixed[:1],native_thresholds]),folds[:n])
        threshold_result['selected_thresholds']=[float(thresholds[k-1]) for k in threshold_result['selected_candidate_indices']]
        write_json(output/'threshold_crossfit_diagnostic.json',threshold_result)
        np.savez_compressed(output/'native_threshold_confusions.npz',confusion=native_thresholds,image_ids=data.img_name_list[:n])
    reference=json.loads((SOURCES/f'{host}_s0/raw_cam/metrics.json').read_text())['fixed']['mean_iou']*100
    native_delta=rows[0]['fixed_mean_iou_percent']-reference
    write_json(output/'verification.json',dict(official_cam_max_abs_error=max_native_error,
               historical_native_delta_pp=native_delta,images=n,elapsed_seconds=time.monotonic()-start,
               precision='fp32',autocast=False,tf32=False))
    if not limit and abs(native_delta)>.01:
        raise RuntimeError(f'Native CAM reproduction mismatch {native_delta}pp: stop ranking')
    del model,means,allmaps,cells,views,p,normalized
    torch.cuda.empty_cache()
    (output/'COMPLETE').write_text('complete\n')
    return rows


def plots(output, all_rows, unlabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    correlations=[]
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,host in zip(axes,HOSTS):
        rows=all_rows[host]['head'][1:]
        numbers=np.array([r['fixed_mean_iou_percent'] for r in rows]).reshape(12,6)
        im=ax.imshow(numbers,aspect='auto',cmap='viridis')
        ax.set(xticks=range(6),xticklabels=range(1,7),yticks=range(12),yticklabels=range(1,13),
               xlabel='Head',ylabel='Layer',title=f'{host}; native={all_rows[host]["head"][0]["fixed_mean_iou_percent"]:.3f}%')
        for l in range(12):
            for h in range(6):
                ax.text(h,l,f'{numbers[l,h]:.1f}',ha='center',va='center',fontsize=7,color='white')
        fig.colorbar(im,ax=ax,label='fixed CAM mIoU (%)')
        for metric in ('kappa','gini','gamma'):
            rho=spearmanr([r[metric] for r in unlabel[host]],numbers.flatten()).statistic
            correlations.append(dict(host=host,metric=metric,spearman=float(rho),abs_rho_at_least_point7=bool(abs(rho)>=.7)))
    fig.savefig(output/'head_heatmap.svg');fig.savefig(output/'head_heatmap.png',dpi=180);plt.close(fig)
    pd.DataFrame(correlations).to_csv(output/'head_correlations.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for host in HOSTS:
        rows=all_rows[host]['alpha'][1:]
        axes[0].plot(ALPHAS,[r['fixed_mean_iou_percent'] for r in rows],'-o',label=host)
        axes[1].plot(ALPHAS,[r['foreground_area_fraction_all'] for r in rows],'-o',label=host)
    for ax in axes:
        ax.set_xlabel('alpha');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('fixed CAM mIoU (%)');axes[1].set_ylabel('foreground area / all pixels')
    fig.savefig(output/'alpha_curves.svg');fig.savefig(output/'alpha_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,host in zip(axes,HOSTS):
        rows=all_rows[host]['topm']
        for criterion in ('kappa','gamma'):
            ax.plot(TOP_M,[r['fixed_mean_iou_percent'] for r in rows if r.get('criterion')==criterion],'-o',label=criterion)
        ax.axhline(rows[0]['fixed_mean_iou_percent'],color='black',ls='--',label='native')
        ax.set(title=host,xlabel='top-m (unlabeled ranking)',ylabel='fixed mIoU (%)',xscale='log');ax.legend()
    fig.savefig(output/'topm_curves.svg');fig.savefig(output/'topm_curves.png',dpi=180);plt.close(fig)
    return correlations


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--preflight',required=True,type=Path)
    parser.add_argument('--limit',type=int,default=0,help='Smoke only; cannot produce full success gates')
    args=parser.parse_args()
    assert (args.preflight/'COMPLETE').exists() and json.loads((args.preflight/'gate.json').read_text())['passed']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no']).strip()
    output=args.output.resolve()
    assert output.is_relative_to(REPO/'results') and output!=REPO/'results'
    output.mkdir(parents=True,exist_ok=False)
    fp32_setup()
    manifest=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),precision='fp32',
                  autocast=False,tf32=False,training=False,seed=0,bootstrap=5000,scope='A then H then controls/topm only',
                  limit=args.limit,preflight=str(args.preflight.resolve()),
                  threshold_grid='0..0.59 step.01 (existing60 bins; plan estimate50 is not a new grid)',
                  epsilon='1e-8 interior alpha only; endpoints exact; native reference unclamped',
                  rownorm_control='view-dependent row scaling need not cancel after multi-scale sum or epsilon normalization',
                  multiplicity='72-cell selection not adjusted; cross-host sign replication is descriptive, not familywise error control',
                  gamma='positive pairs within image; single-label images excluded, not zero-filled',
                  crossfit='fixed hash/multilabel-stratified halves; image-weighted heldout mIoU; alpha reselected within each5000 paired image bootstrap')
    sources=[SOURCES/f'{host}_s0/mctformerplus_final.pth' for host in HOSTS]
    sources += [ROOT/'ImageLists/train_id.txt',ROOT/'ImageLists/val_id.txt',ROOT/'ImageLabel/cls_labels.npy',
                REPO/'docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md',args.preflight/'gate.json']
    ids=(ROOT/'ImageLists/train_id.txt').read_text().splitlines()
    labels=np.load(ROOT/'ImageLabel/cls_labels.npy',allow_pickle=True).item()
    y=np.stack([labels[i] for i in ids])
    folds=stratified_hash_folds(ids,y)
    if args.limit and len(set(folds[:args.limit]))<2:
        raise ValueError('Smoke limit must include both preassigned folds')
    pd.DataFrame(dict(image_id=ids,fold=folds)).to_csv(output/'folds.csv',index=False)
    manifest['fold_class_counts']=[y[folds==f].sum(0).astype(int).tolist() for f in (0,1)]
    sources += [ROOT/'SegmentationClass'/f'{i}.png' for i in ids[:args.limit or len(ids)]]
    manifest['source_sha256_before']={str(p):sha256(p) for p in sources}
    write_json(output/'manifest.json',manifest)
    (output/'commands.sh').write_text(shlex.join([sys.executable,'-m','analysis.head_alpha',*sys.argv[1:]])+'\n')
    run([sys.executable,'-m','pip','freeze'],output,'pip_freeze')
    run([sys.executable,'-m','pytest','-q','tests/test_head_alpha_preflight.py','tests/test_head_alpha.py'],output,'tests')
    all_rows={h:{} for h in HOSTS}
    unlabel={}
    rankings={}
    try:
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for host in HOSTS:
                all_rows[host]['alpha']=scan(host,'alpha',output/host/'alpha',configs_for('alpha'),folds,args.limit)
            for host in HOSTS:
                rankings[host],unlabel[host]=unlabeled(host,output/host/'unlabeled',max(args.limit,8) if args.limit else 0)
                all_rows[host]['head']=scan(host,'head',output/host/'head',configs_for('head'),folds,args.limit)
            for host in HOSTS:
                for stage in ('control','topm'):
                    configs=configs_for(stage,rankings[host],all_rows[host]['head'][1:])
                    all_rows[host][stage]=scan(host,stage,output/host/stage,configs,folds,args.limit)
        correlations=plots(output,all_rows,unlabel)
        from analysis.head_alpha_report import finish_report
        finish_report(output,all_rows,correlations,smoke=bool(args.limit))
        manifest['source_sha256_after']={p:sha256(Path(p)) for p in manifest['source_sha256_before']}
        assert manifest['source_sha256_before']==manifest['source_sha256_after']
        manifest['source_integrity_unchanged']=True
        write_json(output/'manifest.json',manifest)
        (output/('SMOKE_COMPLETE' if args.limit else 'COMPLETE')).write_text('complete; no follow-up training authorized\n')
    except BaseException as exc:
        (output/'FAILED').write_text(repr(exc)+'\n')
        raise


if __name__=='__main__':
    main()
