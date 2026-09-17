"""Frozen all-epoch E1/E4 diagnostics; no semantic masks or attention dumps."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from analysis.artifact_probe import dataset, capture_patch_stats
from analysis.diagnostics.metrics import class_token_affinity, gwrp_weights
from analysis.weight_stats import all_product_weights, normalized_entropy, positive_image_mean
from experiments.baselines.run_default_voc_coco import REPO, dataset_spec, run
from experiments.ablations.evaluate_voc_epoch_trajectory import write_json
from tools.epoch_checkpoints import sha256


def variant_flags(variant):
    if variant == 'detach':
        return ['--patch-pooling','c2p','--c2p-pooling-layers','all',
                '--c2p-pooling-reduction','product','--detach-weights']
    if variant == 'a1':
        return ['--patch-pooling','gwrp','--channel-agg']
    if variant == 'all_product':
        return ['--patch-pooling','c2p','--c2p-pooling-layers','all','--c2p-pooling-reduction','product']
    if variant == 'gwrp':
        return ['--patch-pooling','gwrp']
    raise ValueError(variant)


def classification_command(checkpoint, output, variant, limit=0):
    spec=dataset_spec('VOC12')
    return [sys.executable,'-u','tools/evaluate_mctformerplus_classification.py',
            '--checkpoint',str(checkpoint),'--model','mctformerplus','--voc-root',str(spec['root']),
            '--list-path',str(spec['val']),'--input-size','448','--batch-size','16','--num-workers','8',
            '--bootstrap-resamples','0','--limit',str(limit),'--output-dir',str(output),*variant_flags(variant)]


def discard_classification_predictions(output):
    # Only known newly generated outputs after the evaluator's success marker.
    assert (output/'CLASSIFICATION_COMPLETE').is_file()
    metrics=json.loads((output/'classification_metrics.json').read_text())
    assert metrics['finite']
    removed=[]
    for name in ('classification_predictions.npz','classification_per_image.csv'):
        path=output/name
        if path.exists():
            assert not path.is_symlink()
            removed.append(dict(path=name,bytes=path.stat().st_size,sha256=sha256(path)))
    write_json(output/'cleanup.json',dict(removed=removed,checkpoint_retained=True))
    for row in removed:(output/row['path']).unlink()


def epoch20_warning(args, checkpoint):
    """Separate process: parent training RNG/optimizer/model modes unaffected."""
    variant='detach' if args.detach_weights else 'a1'
    root=Path(args.work_space)/'epoch20_warning'
    root.mkdir(exist_ok=False)
    run(classification_command(checkpoint,root/'classification',variant),root,'classification')
    metrics=json.loads((root/'classification/classification_metrics.json').read_text())
    ap=metrics['metrics_percent']['class_token']['macro_class_ap']
    write_json(root/'warning.json',dict(epoch=20,class_macro_mAP_percent=ap,
               suspected_instability=ap<80,threshold=80,action='warn only; continue to epoch45'))
    discard_classification_predictions(root/'classification')
    print(f'EPOCH20 class macro mAP={ap:.4f}; suspect_instability={ap<80}; continue training',flush=True)


def diagnose(checkpoint, output, limit=0, artifacts=False):
    from models.mctformer_plus import build_mctformerplus
    digest=sha256(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu')
    spec=payload['model_spec']
    kwargs={k:spec[k] for k in ('patch_pooling','c2p_pooling_layers','c2p_pooling_reduction',
                               'detach_weights','channel_agg') if k in spec}
    model=build_mctformerplus('small',input_size=448,num_classes=20,**kwargs)
    model.load_state_dict(payload['model'],strict=True)
    epoch=payload['epoch']+1
    del payload
    model=model.cuda().eval()
    data=dataset()
    n=limit or len(data)
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,range(n)),batch_size=8,
                                      shuffle=False,num_workers=4)
    values={}
    def add(key,value):values.setdefault(key,[]).extend(value.detach().cpu().tolist())
    with torch.inference_mode(), (capture_patch_stats(model) if artifacts else nullcontext({})) as captured:
        for images,labels in loader:
            images,labels=images.cuda(),labels.cuda()
            cls,patches,records,_=model.forward_features(images)
            c2p=all_product_weights(records,20)
            maps=model.head(patches.reshape(len(images),28,28,384).permute(0,3,1,2).contiguous())
            pool=c2p if model.patch_pooling=='c2p' else gwrp_weights(maps.flatten(2),model.decay_parameter)
            for prefix,w in [('c2p_all_product',c2p),('actual_pool',pool)]:
                add(prefix+'_kappa',positive_image_mean(normalized_entropy(w),labels))
                add(prefix+'_top1_mass',positive_image_mean(w.max(-1).values,labels))
            add('rho_cc',class_token_affinity(cls,labels)['rho_cc'])
            if artifacts:
                for layer,s in captured.items():
                    add(f'L{layer}_M2_frac_hi',s['frac_hi'])
                    add(f'L{layer}_M4_top1pct_received_mass',s['top_share'])
            del records
    summary={}
    for key,items in values.items():
        a=np.asarray(items);finite=np.isfinite(a)
        summary[key]=dict(mean=float(a[finite].mean()) if finite.any() else None,
                          n_images=int(finite.sum()),missing_images=int((~finite).sum()))
    result=dict(epoch=epoch,num_images=n,checkpoint=str(checkpoint),checkpoint_sha256=digest,
                model_spec=spec,metrics=summary,channel=model.channel_aggregator.diagnostics(),
                channel_weights=(model.channel_aggregator.weights().detach().cpu().tolist()
                                 if model.channel_aggregator.enabled else None),
                aggregation='equal image; positive classes averaged within image; rho_cc on multi-label only',
                semantic_gt_loaded=False,artifacts=artifacts,
                artifact_definitions='M2: norm/median >3 fraction; M4: top round(.01*784)=8 P2P received-mass share',
                precision='FP32; TF32 disabled',source_unchanged=sha256(checkpoint)==digest)
    assert result['source_unchanged']
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'metrics.json',result)
    (output/'COMPLETE').write_text('complete\n')
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--artifacts',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    diagnose(args.checkpoint,args.output,args.limit,args.artifacts)


if __name__=='__main__':main()
