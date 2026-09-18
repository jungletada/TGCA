"""Head/alpha plan stage0: full FP32 final-checkpoint precision-boundary audit."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from analysis.artifact_probe import dataset, REPO
from models.mctformer_plus import build_mctformerplus
from tools.epoch_checkpoints import sha256


SOURCES = REPO / 'results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2'
HOSTS = ('gwrp', 'all_product')
TV_GATE = 1e-3  # Preregistered upper edge of the requested 1e-4 order of magnitude.


def fp32_setup():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_host(host):
    path = SOURCES / f'{host}_s0/mctformerplus_final.pth'
    payload = torch.load(path, map_location='cpu')
    assert payload['epoch'] == 44
    kwargs = {} if host == 'gwrp' else dict(patch_pooling='c2p', c2p_pooling_layers='all', c2p_pooling_reduction='product')
    model = build_mctformerplus('small', input_size=448, num_classes=20, cam=True, **kwargs).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    return model, path


def product_weights(c2p):
    return c2p.mean(2).clamp_min(torch.finfo(torch.float32).tiny).log().sum(0).softmax(-1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(REPO / 'results') and output != REPO / 'results'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO).strip()
    output.mkdir(parents=True, exist_ok=False)
    fp32_setup()
    data = dataset()
    assert len(data) == 1449
    sources = [SOURCES / f'{h}_s0/mctformerplus_final.pth' for h in HOSTS]
    sources += [REPO/'docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md',
                REPO/'data/VOCdevkit/VOC2012/ImageLists/val_id.txt',
                REPO/'data/VOCdevkit/VOC2012/ImageLabel/cls_labels.npy']
    manifest = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    precision='fp32', autocast=False, tf32=False, epoch_one_based=45, seed=0,
                    stage='0 precision gate; no training, no segmentation masks',
                    comparison='same FP32 forward probabilities vs explicit FP16 round-trip at probability boundary only; NOT an AMP backbone comparison',
                    gate='all_product positive-class TV (image-weighted and pooled-class-weighted means) < 0.001; gwrp hypothetical-product diagnostic only',
                    gate_limit=TV_GATE, bootstrap=5000, bootstrap_unit='image',
                    transform=str(data.transform),
                    native_c2p='mean heads then mean layers10,11,12; full softmax probabilities, not patch-conditional',
                    native_seed='sqrt(ReLU(M)*a_native); alpha=0.5',
                    native_p2p='sum all12 layers of head-mean patch-to-patch; P @ S',
                    normalization='native-size bilinear per view; unflip and sum views/scales; image-label gate; per-class min-max denominator range+1e-8',
                    source_sha256_before={str(p): sha256(p) for p in sources})
    write_json(output/'manifest.json', manifest)
    (output/'commands.sh').write_text(shlex.join([sys.executable, '-m', 'analysis.head_alpha_preflight', *sys.argv[1:]])+'\n')
    rows, summaries = [], []
    for host in HOSTS:
        model, path = load_host(host)
        loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False, num_workers=4)
        offset = 0
        with torch.autocast('cuda', enabled=False):
            for images, labels in loader:
                _, _, records, _ = model.forward_features(images.cuda())
                assert all(a.dtype == torch.float32 for a in records)
                cp = torch.stack([a[:, :, :20, 20:] for a in records])
                rounded = cp.half().float()
                tv = .5 * (product_weights(cp)-product_weights(rounded)).abs().sum(-1)
                dead = (rounded.mean(2).sum(-1) == 0).sum(0)
                for j in range(len(images)):
                    positive = labels[j].bool().cuda()
                    values = tv[j, positive]
                    rows.append(dict(host=host, image_id=data.img_name_list[offset+j],
                                     positive_classes=int(positive.sum()), tv_mean=float(values.mean()),
                                     tv_sum=float(values.sum()), tv_max=float(values.max()),
                                     all_class_tv_mean=float(tv[j].mean()),
                                     zero_layer_class_rows=int(dead[j, positive].sum())))
                offset += len(images)
                if offset % 100 == 0 or offset == len(data):
                    print(f'{host} precision {offset}/{len(data)}', flush=True)
        frame = pd.DataFrame([r for r in rows if r['host'] == host])
        rng = np.random.default_rng(0)
        values = frame.tv_mean.to_numpy()
        boot = np.concatenate([values[rng.integers(len(values), size=(100, len(values)))].mean(1) for _ in range(50)])
        summaries.append(dict(host=host, image_mean_tv=float(values.mean()),
                              pooled_positive_mean_tv=float(frame.tv_sum.sum()/frame.positive_classes.sum()),
                              ci95_lo=float(np.quantile(boot,.025)), ci95_hi=float(np.quantile(boot,.975)),
                              max_tv=float(frame.tv_max.max()),
                              affected_images=int((frame.zero_layer_class_rows>0).sum()),
                              zero_layer_class_rows=int(frame.zero_layer_class_rows.sum())))
        pd.DataFrame(rows).to_csv(output/'per_image_tv.csv', index=False)
        print(summaries[-1], flush=True)
        del model, records, cp, rounded
        torch.cuda.empty_cache()
    comparison = next(r for r in summaries if r['host'] == 'all_product')
    passed = comparison['image_mean_tv'] < TV_GATE and comparison['pooled_positive_mean_tv'] < TV_GATE
    write_json(output/'gate.json', dict(passed=passed, gate_limit=TV_GATE, summaries=summaries,
                                     next_action='A then H inference only' if passed else 'STOP: precondition fails; no scans or training'))
    manifest['source_sha256_after'] = {p: sha256(Path(p)) for p in manifest['source_sha256_before']}
    assert manifest['source_sha256_before'] == manifest['source_sha256_after']
    manifest['source_integrity_unchanged'] = True
    write_json(output/'manifest.json', manifest)
    (output/'COMPLETE').write_text('full val preflight completed; consult gate.json\n')


if __name__ == '__main__':
    main()
