"""Frozen VOC artifact probe. No semantic GT, model edits or attention dumps."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.c2p_pooling_attention import paired_bootstrap
from tools.evaluate_cam_threshold_grid import sha256_file

REPO = Path(__file__).resolve().parents[1]
SEED = 20260916
CHECKPOINTS = {
    'gwrp': 'results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12',
    'all_product': 'results/c2p_pooling/20260914-voc-product-s0/all_product',
    'all_product_affinity': 'results/c2p_pooling/20260914-voc-product-affinity-s0/all_product_affinity',
}
METRICS = ['ratio_max', 'frac_hi', 'p99_over_p50', 'gini', 'top_share',
           'symptom_jaccard', 'lowinfo_top1pct', 'lowinfo_hi', 'hi_empty']


@torch.no_grad()
def patch_norm_stats(patch_tokens, hi=3.0):
    norms = patch_tokens.float().norm(dim=-1)
    median = norms.median(-1, keepdim=True).values.clamp_min(1e-8)
    ratio = norms / median
    return dict(ratio=ratio, frac_hi=(ratio > hi).float().mean(-1),
                p99_over_p50=torch.quantile(norms, .99, dim=-1) / median.squeeze(-1))


@torch.no_grad()
def attractor_stats(attn_pp, topk_frac=.01):
    g = attn_pp.float().sum(1)
    g = g / g.sum(-1, keepdim=True).clamp_min(1e-12)
    n = g.shape[-1]
    x = g.sort(-1).values
    idx = torch.arange(1, n + 1, dtype=g.dtype, device=g.device)
    gini = ((2 * idx - n - 1) * x).sum(-1) / (n * x.sum(-1).clamp_min(1e-12))
    k = max(1, round(topk_frac * n))
    return dict(recv=g, gini=gini, top_share=g.topk(k, dim=-1).values.sum(-1), k=k)


@torch.no_grad()
def symptom_overlap(ratio, recv, topk_frac=.01):
    k = max(1, round(topk_frac * ratio.shape[-1]))
    a, b = torch.zeros_like(ratio, dtype=torch.bool), torch.zeros_like(recv, dtype=torch.bool)
    a.scatter_(1, ratio.topk(k, dim=-1).indices, True)
    b.scatter_(1, recv.topk(k, dim=-1).indices, True)
    return (a & b).sum(-1).float() / (a | b).sum(-1).clamp_min(1)


@torch.no_grad()
def lowinfo_scores(rgb, ratio, grid=28):
    # Input is inverse-ImageNet-normalized RGB, not differently scaled channels.
    x = rgb.float().mean(1, keepdim=True)
    gx, gy = x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]
    magnitude = (gx[..., :-1, :].square() + gy[..., :, :-1].square()).sqrt()
    gradient = F.adaptive_avg_pool2d(magnitude, (grid, grid)).flatten(1)
    median = gradient.median(-1).values.clamp_min(1e-8)
    k = max(1, round(.01 * ratio.shape[-1]))
    top = gradient.gather(1, ratio.topk(k, dim=-1).indices).mean(-1) / median
    hi = ratio > 3
    high_mean = (gradient * hi).sum(-1) / hi.sum(-1).clamp_min(1) / median
    high_mean = high_mean.masked_fill(~hi.any(-1), float('nan'))
    return top, high_mean, (~hi.any(-1)).float()


@contextmanager
def frozen_eval(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(next(model.parameters()).device.type, enabled=False):
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        assert all(module.training == mode for module, mode in modes)


@contextmanager
def capture_patch_stats(model):
    """Only compact patch norms/received mass, never cache N x N matrices."""
    captured = {}
    handles = []
    def hook(layer):
        def capture(module, inputs, output):
            tokens, attention = output
            norms = patch_norm_stats(tokens[:, model.num_classes:])
            attractor = attractor_stats(attention[:, :, model.num_classes:, model.num_classes:].mean(1))
            captured[layer] = {**norms, **attractor}
        return capture
    try:
        for layer, block in enumerate(model.blocks, 1):
            handles.append(block.register_forward_hook(hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def load_model(name, device='cuda'):
    from models.mctformer_plus import build_mctformerplus, validate_mctformerplus_patch_pooling_checkpoint
    path = REPO / CHECKPOINTS[name] / 'mctformerplus_final.pth'
    checkpoint = torch.load(path, map_location='cpu')
    pooling = 'gwrp' if name == 'gwrp' else 'c2p'
    layers, reduction = ('last3', 'mean') if name == 'gwrp' else ('all', 'product')
    affinity = name == 'all_product_affinity'
    validate_mctformerplus_patch_pooling_checkpoint(checkpoint, pooling, layers, reduction, affinity)
    assert not checkpoint.get('final_norm', False) and not checkpoint.get('patch_final_norm', False)
    spec = checkpoint.get('model_spec', {})
    assert spec.get('class_token_init', 'baseline') == 'baseline'
    assert spec.get('token_interaction', 'joint') == 'joint' and not spec.get('patch_first', False)
    model = build_mctformerplus('small', input_size=448, num_classes=20, patch_pooling=pooling,
                               c2p_pooling_layers=layers, c2p_pooling_reduction=reduction,
                               c2p_pooling_affinity=affinity)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device)


def dataset():
    from datasets_cam import VOC12Dataset, build_transform
    root = REPO / 'data/VOCdevkit/VOC2012'
    return VOC12Dataset(str(root), str(root / 'ImageLists/val_id.txt'),
                        transform=build_transform(False, False, argparse.Namespace(input_size=448)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def plot_example(path, image, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[0].set_title('RGB (no semantic GT)')
    for ax, key, title in zip(axes[1:], ['ratio', 'recv'], ['Patch norm / median', 'N * received mass']):
        values = stats[key].cpu().numpy().reshape(28, 28)
        im = ax.imshow(values * (784 if key == 'recv' else 1), cmap='magma')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    for ax in axes:
        ax.set_axis_off()
    for suffix in ['png', 'pdf']:
        fig.savefig(str(path) + '.' + suffix, dpi=130)
    plt.close(fig)


def summarize(output, all_values, counts):
    rows = []
    names = list(all_values)
    for stratum, mask in [('all', counts > 0), ('multi', counts >= 2)]:
        if not mask.any():
            continue
        for first, second in [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]:
            point, ci, n = paired_bootstrap(np.stack([all_values[first][mask], all_values[second][mask]]),
                                          repetitions=5000, seed=SEED)
            for layer in range(12):
                for j, metric in enumerate(METRICS):
                    for s, label in enumerate([first, second, second + '_minus_' + first]):
                        rows.append(dict(stratum=stratum, model=label, layer=layer + 1, metric=metric,
                                         n_images=int(n[layer, j]), mean=point[s, layer, j],
                                         ci95_lo=ci[0, s, layer, j], ci95_hi=ci[1, s, layer, j]))
    summary = pd.DataFrame(rows).drop_duplicates(['stratum', 'model', 'layer', 'metric'])
    summary.to_csv(output / 'summary.csv', index=False)
    lines = ['# Artifact probe decision', '',
             'Thresholds interpreted as proportions: M2 >= 0.005 (0.5%), M4 >= 0.05 (5%), M5 >= 0.2.',
             'Gates use equal-image L12 means on the full validation set, not a selected subset or CI endpoints.',
             'Each checkpoint is judged separately; no pooling of images/patches across checkpoints.',
             'M6 uses RGB grayscale gradient, not semantic GT: low information does NOT imply background.',
             'High-norm-specific M6 is missing if an image has no rho>3 patches; missing counts are reported.',
             'k=round(0.01*784)=8. Exact random independent-set expected Jaccard is about 0.00547, not 0.01.',
             'Image-clustered paired bootstrap 5000, seed20260916; no training-seed uncertainty.', '',
             '| Checkpoint | M2 fraction | M4 top1% mass | M5 Jaccard | All gates |',
             '|---|---:|---:|---:|---|']
    for name in names:
        means = np.nanmean(all_values[name][:, -1], axis=0)
        gates = [means[1] >= .005, means[4] >= .05, means[5] >= .2]
        lines.append(f'| {name} | {means[1]:.6f} | {means[4]:.6f} | {means[5]:.6f} | {gates}: ' +
                     ('register worth future consideration' if all(gates) else 'NO-GO register for this checkpoint') + ' |')
    lines.extend(['', 'No register implementation is authorized by this probe. Continue the planned affinity analysis.',
                  'Between-checkpoint differences are observations under matched seed0 training, not proof that affinity causes artifacts.'])
    (output / 'decision.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(REPO / 'results') and output != REPO / 'results'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO).strip()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = dataset()
    counts = np.asarray(data.label_list).sum(1)
    assert len(data) == 1449 and (counts >= 2).sum() == 522
    n = args.limit or len(data)
    assert 0 < n <= len(data)
    examples = np.random.default_rng(SEED).choice(np.flatnonzero(counts >= 2), 6, replace=False).tolist()
    sources = [REPO / p / 'mctformerplus_final.pth' for p in CHECKPOINTS.values()]
    sources += [REPO / 'data/VOCdevkit/VOC2012/ImageLists/val_id.txt',
                REPO / 'data/VOCdevkit/VOC2012/ImageLabel/cls_labels.npy']
    hashes = {str(p): sha256_file(p) for p in sources}
    for name, directory in CHECKPOINTS.items():
        assert hashes[str(REPO / directory / 'mctformerplus_final.pth')] == (REPO / directory / 'checkpoint_sha256.txt').read_text().split()[0]
    manifest = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    config=vars(args) | {'output': str(output)}, source_sha256_before=hashes,
                    seed=SEED, bootstrap=5000, unit='paired image; equal image weighting',
                    num_images=n, multi_images=int((counts[:n] >= 2).sum()),
                    transform=str(data.transform), precision='FP32; TF32 off; autocast off',
                    semantic_gt_loaded=False, gate_thresholds=dict(frac_hi=.005, top_share=.05, overlap=.2),
                    example_ids=[data.img_name_list[i] for i in examples],
                    M6='inverse-normalized RGB -> grayscale forward differences -> adaptive28x28; top1% and rho>3 sets',
                    model_mode_restored=True, checkpoint_policy='final epoch44')
    write_json(output / 'manifest.json', manifest)
    (output / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (output / 'conda_explicit.txt').write_text(subprocess.check_output(
        ['/home/peng/anaconda3/bin/conda', 'list', '-n', 'tgca-repro', '--explicit'], text=True))
    (output / 'checkpoint_sha256.txt').write_text(''.join(f'{hashes[str(p)]}  {p}\n' for p in sources[:3]))
    (output / 'commands.sh').write_text(shlex.join([sys.executable, '-m', 'analysis.artifact_probe', *sys.argv[1:]]) + '\n')
    all_values = {}
    try:
        for name in CHECKPOINTS:
            model = load_model(name)
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
            values = np.full((n, 12, len(METRICS)), np.nan)
            edges = np.geomspace(.01, 100, 161)
            histogram = np.zeros(160, dtype=np.int64)
            offset = 0
            with frozen_eval(model), capture_patch_stats(model) as captured:
                for images, _ in loader:
                    images = images.cuda()
                    captured.clear()
                    _, patches, records, _ = model.forward_features(images)
                    assert len(captured) == 12 and patches.shape[1] == 784
                    rgb = images * images.new_tensor([.229, .224, .225])[None, :, None, None] + images.new_tensor([.485, .456, .406])[None, :, None, None]
                    for layer in range(1, 13):
                        s = captured[layer]
                        lowtop, lowhi, empty = lowinfo_scores(rgb, s['ratio'])
                        measures = [s['ratio'].amax(-1), s['frac_hi'], s['p99_over_p50'], s['gini'], s['top_share'],
                                    symptom_overlap(s['ratio'], s['recv']), lowtop, lowhi, empty]
                        values[offset:offset + len(images), layer - 1] = torch.stack(measures, -1).cpu().numpy()
                    ratios = captured[12]['ratio'].cpu().numpy()
                    histogram += np.histogram(ratios.clip(edges[0], edges[-1]), bins=edges)[0]
                    for b in range(len(images)):
                        if offset + b in examples:
                            plot_example(output / f'attractor_map_{name}_{data.img_name_list[offset+b]}', rgb[b],
                                         {k: captured[12][k][b] for k in ['ratio', 'recv']})
                    offset += len(images)
                    del records, patches
                    if offset % 100 == 0 or offset == n:
                        print(f'{name} {offset}/{n}', flush=True)
            assert model.training
            all_values[name] = values
            rows = [dict(image_id=data.img_name_list[i], n_positive=int(counts[i]), layer=l+1,
                         **dict(zip(METRICS, values[i, l]))) for i in range(n) for l in range(12)]
            pd.DataFrame(rows).to_csv(output / f'per_image_{name}.csv', index=False)
            pd.DataFrame(dict(left=edges[:-1], right=edges[1:], count=histogram)).to_csv(output / f'norm_hist_{name}.csv', index=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.stairs(histogram, edges)
            ax.set(xscale='log', yscale='log', xlabel='L12 patch norm / image median', ylabel='Patch count (descriptive only)')
            for ext in ['png', 'pdf']:
                fig.savefig(output / f'norm_hist_{name}.{ext}', dpi=140)
            plt.close(fig)
            del model
            torch.cuda.empty_cache()
        summarize(output, all_values, counts[:n])
        manifest['source_sha256_after'] = {p: sha256_file(p) for p in hashes}
        assert manifest['source_sha256_after'] == hashes
        manifest['source_integrity_unchanged'] = True
        write_json(output / 'manifest.json', manifest)
        (output / 'COMPLETE').write_text('complete\n')
    except BaseException as exc:
        (output / 'FAILED').write_text(repr(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
