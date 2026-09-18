"""Bounded FP16-path audit and two-epoch integration smoke; no full training."""
import argparse
import gc
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from analysis.artifact_probe import dataset
from experiments.ablations.run_voc_epoch_checkpoints import (
    pooling_flags, training_command, verify_snapshots,
)
from experiments.baselines.run_default_voc_coco import REPO, PRETRAIN, dataset_spec, run
from models.mctformer_plus import build_mctformerplus
from tools.epoch_checkpoints import sha256


TESTS = [
    'tests/test_c2p_pooling_fp32.py', 'tests/test_mctformerplus_c2p_pooling.py',
    'tests/test_tgca_normalization.py', 'tests/test_tgca_replication.py',
    'tests/test_detach_pooling.py', 'tests/test_channel_agg.py',
    'tests/test_epoch_checkpoints.py', 'tests/test_raw_cam_streaming.py',
]
SOURCE = REPO / 'results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def frozen_probe():
    torch.manual_seed(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    data = dataset()
    items = [data[i] for i in range(8)]
    images = torch.stack([item[0] for item in items]).cuda()
    positive = torch.stack([item[1] for item in items]).cuda() > 0
    results = []
    for seed in (0, 11):
        path = SOURCE / f'all_product_s{seed}/epoch_checkpoints/mctformerplus_epoch_012.pth'
        checkpoint = torch.load(path, map_location='cpu')
        m = build_mctformerplus('small', input_size=448, num_classes=20,
                               patch_pooling='c2p', c2p_pooling_layers='all',
                               c2p_pooling_reduction='product').cuda().eval()
        m.load_state_dict(checkpoint['model'], strict=True)
        precast = []

        def capture(module, args, output):
            # Independent reference from the exact SAME QK logits, not tokens
            # or rounded probabilities. Vanilla has no attention mask here.
            reference = args[0].float().softmax(-1)
            assert torch.equal(reference.half(), output)
            precast.append(reference[:, :, :20, 20:].mean(1))

        handles = [b.attn.normalizer.register_forward_hook(capture) for b in m.blocks]
        with torch.autocast('cuda', dtype=torch.float16):
            old = m.forward_features(images, return_aux=True)
        for handle in handles:
            handle.remove()
        m.c2p_pooling_fp32 = True
        with torch.autocast('cuda', dtype=torch.float16):
            new = m.forward_features(images, return_aux=True)
            maps = m.head(new[1].reshape(8, 28, 28, 384).permute(0, 3, 1, 2).contiguous())
        assert all(torch.equal(old[i], new[i]) for i in (0, 1))
        assert all(torch.equal(a, b) for i in (2, 3) for a, b in zip(old[i], new[i]))
        collected = new[4]['c2p_pooling_fp32_records']
        assert all(torch.equal(a, b) for a, b in zip(precast, collected))
        reference_w = torch.stack(precast).clamp_min(torch.finfo(torch.float32).tiny).log().sum(0).softmax(-1)
        old_w = m.c2p_spatial_weights(old[2], 784)
        repaired_logits = m.c2p_pool(maps, new[2], collected)
        reference_logits = (reference_w * maps.flatten(2).float()).sum(-1)
        assert torch.equal(repaired_logits, reference_logits)
        tv = (old_w - reference_w).abs().sum(-1) / 2
        layer_rows = []
        for layer, (native, fp32) in enumerate(zip(old[2], collected), 1):
            rounded = native[:, :, :20, 20:].float().mean(1)
            dead = (rounded.sum(-1) == 0) & positive
            layer_rows.append(dict(layer=layer, positive_all_zero_patch_rows=int(dead.sum()),
                                   positive_min_patch_mass_fp32=float(fp32.sum(-1)[positive].min())))
        # Raw CAM uses the original native attention and tokens, not the new readout.
        cam = build_mctformerplus('small', input_size=448, num_classes=20, cam=True,
                                 patch_pooling='c2p', c2p_pooling_layers='all',
                                 c2p_pooling_reduction='product').cuda().eval()
        cam.load_state_dict(checkpoint['model'], strict=True)
        with torch.autocast('cuda', dtype=torch.float16):
            old_cam = cam(images[:1])
            cam.c2p_pooling_fp32 = True
            assert torch.equal(old_cam, cam(images[:1]))
        results.append(dict(seed=seed, epoch_one_based=12, checkpoint=str(path),
                            positive_rows=int(positive.sum()), images=data.img_name_list[:8],
                            positive_weight_tv_mean=float(tv[positive].mean()),
                            positive_weight_tv_max=float(tv[positive].max()),
                            repaired_pool_logits_max_error=float((repaired_logits-reference_logits).abs().max()),
                            native_tokens_attention_cct_bitwise_equal=True, raw_cam_bitwise_equal=True,
                            precast_reference_bitwise_equal=True, layers=layer_rows))
        del m, cam, old, new, checkpoint, maps, collected, precast
        gc.collect()
        torch.cuda.empty_cache()
    return dict(scope='first8 VOC val images; frozen epoch12 seed0/11; not a performance evaluation',
                precision='AMP FP16 QK/native attention; pooling FP32 pre-cast softmax',
                transform=str(data.transform), results=results)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    output = args.output.resolve()
    assert output.is_relative_to(REPO / 'results')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'commands.sh').write_text(shlex.join([sys.executable, '-m', __spec__.name, *sys.argv[1:]]) + '\n')
    spec = dataset_spec('VOC12')
    sources = [PRETRAIN, spec['train'], spec['val'], spec['cam'], spec['root'] / 'ImageLabel/cls_labels.npy']
    sources += [SOURCE / f'all_product_s{s}/epoch_checkpoints/mctformerplus_epoch_012.pth' for s in (0, 11)]
    manifest = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256_before={str(p): sha256(p) for p in sources},
                    scope='precision repair verification only; no matched full retraining',
                    training_unchanged=['loss', 'GradScaler', 'seed', 'optimizer', 'schedule implementation'],
                    flag='--c2p-pooling-fp32', checkpoints_legacy_default=False)
    write_json(output / 'manifest.json', manifest)
    run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
    run([sys.executable, '-m', 'pytest', '-q', *TESTS], output, 'tests')
    torch.set_num_threads(2)
    write_json(output / 'frozen_probe.json', frozen_probe())
    smoke = output / 'smoke'
    smoke.mkdir()
    for key, count in [('train', 64), ('val', 4), ('cam', 2)]:
        name = 'cam_train_id.txt' if key == 'cam' else key + '_id.txt'
        (smoke / name).write_text('\n'.join(spec[key].read_text().splitlines()[:count]) + '\n')
    flags = [*pooling_flags('all_product'), '--c2p-pooling-fp32']
    run([*training_command(spec, smoke, 'all_product', 0, smoke/'train_id.txt',
                           smoke/'val_id.txt', epochs=2, workers=4), '--c2p-pooling-fp32'], smoke, 'train')
    verify_snapshots(smoke, 2)
    checkpoint = smoke / 'epoch_checkpoints/mctformerplus_epoch_002.pth'
    payload = torch.load(checkpoint, map_location='cpu')
    steps = sorted(set(int(s['step']) for s in payload['optimizer']['state'].values()))
    assert steps and min(steps) > 0
    write_json(output / 'smoke_checkpoint.json', dict(path=str(checkpoint), sha256=sha256(checkpoint),
               optimizer_steps=steps, scaler=payload['scaler'], model_spec=payload['model_spec']))
    del payload
    run([sys.executable, '-u', 'tools/evaluate_mctformerplus_classification.py',
         '--checkpoint', checkpoint, '--model', 'mctformerplus', '--voc-root', spec['root'],
         '--list-path', smoke/'val_id.txt', '--input-size', '448', '--batch-size', '4',
         '--num-workers', '2', '--bootstrap-resamples', '0', '--output-dir', smoke/'classification', *flags], smoke, 'classification')
    run([sys.executable, '-u', 'make_cam.py', '--dataset', 'VOC12', '--model', 'mctformerplus',
         '--voc12_root', spec['root'], '--work_space', smoke, '--train_list', smoke/'cam_train_id.txt',
         '--input_size', '448', '--scales', '1.0,0.75,1.25', '--checkpoint', checkpoint,
         '--cam_out_dir', 'cam_train', '--online-raw-eval', '--online-mask-dir', spec['masks'],
         '--online-output-dir', smoke/'raw_cam', *flags], smoke, 'cam')
    cls = json.loads((smoke/'classification/classification_metrics.json').read_text())
    cam = json.loads((smoke/'raw_cam/metrics.json').read_text())
    assert cls['num_images'] == 4 and cls['finite'] and cam['num_images'] == 2
    manifest['source_sha256_after'] = {p: sha256(Path(p)) for p in manifest['source_sha256_before']}
    assert manifest['source_sha256_after'] == manifest['source_sha256_before']
    manifest['source_integrity_unchanged'] = True
    write_json(output / 'manifest.json', manifest)
    (output / 'VALIDATION_COMPLETE').write_text('tests, frozen probe, smoke training, checkpoint, classification and CAM passed\n')


if __name__ == '__main__':
    main()
