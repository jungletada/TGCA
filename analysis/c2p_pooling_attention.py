"""Frozen, paired layer-wise actual C2P attention audit (no segmentation GT).

Run with ``python -m analysis.c2p_pooling_attention --output NEW_DIRECTORY``.
Source checkpoints/results are read-only. Statistics weight images equally.
"""
import argparse
import hashlib
import itertools
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ('aeroplane bicycle bird boat bottle bus car cat chair cow diningtable '
         'dog horse motorbike person pottedplant sheep sofa train tvmonitor').split()
METRICS = ['pair_top1_same', 'all_top1_same', 'pair_top10pct_jaccard',
           'pair_top10_jaccard', 'pair_cosine', 'pair_pearson',
           'entropy_normalized', 'top1_mass', 'patch_group_mass', 'top1_tie_rate']
SOURCES = {
    'gwrp': Path('results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12'),
    'c2p': Path('results/c2p_pooling/20260914-voc-s0/VOC12'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def map_metrics(a):
    """[positive classes, patches] -> image-level metrics, never pair samples."""
    a = np.asarray(a, dtype=np.float64)
    c, n = a.shape
    assert c > 0 and n > 1 and np.isfinite(a).all() and (a >= 0).all()
    mass = a.sum(-1)
    assert (mass > 0).all()
    w = a / mass[:, None]
    order = np.argsort(-w, axis=-1, kind='stable')
    top1 = order[:, 0]
    pairs = list(itertools.combinations(range(c), 2))
    result = dict.fromkeys(METRICS, float('nan'))
    result.update(
        entropy_normalized=float(-(w * np.log(w.clip(1e-300))).sum(-1).mean() / np.log(n)),
        top1_mass=float(w.max(-1).mean()), patch_group_mass=float(mass.mean()),
        top1_tie_rate=float(((w == w.max(-1, keepdims=True)).sum(-1) > 1).mean()),
    )
    if pairs:
        result['pair_top1_same'] = np.mean([top1[i] == top1[j] for i, j in pairs])
        result['all_top1_same'] = float(len(set(top1)) == 1)
        for name, k in [('pair_top10pct_jaccard', math.ceil(n * .1)),
                        ('pair_top10_jaccard', min(10, n))]:
            sets = [set(row[:k]) for row in order]
            result[name] = np.mean([len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                                    for i, j in pairs])
        unit = w / np.linalg.norm(w, axis=-1, keepdims=True)
        result['pair_cosine'] = np.mean([unit[i] @ unit[j] for i, j in pairs])
        centered = w - w.mean(-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1)
        correlations = [centered[i] @ centered[j] / (norms[i] * norms[j])
                        for i, j in pairs if norms[i] > 1e-12 and norms[j] > 1e-12]
        if correlations:
            result['pair_pearson'] = float(np.mean(correlations))
    return np.array([result[key] for key in METRICS])


def paired_bootstrap(x, repetitions=5000, seed=20260914):
    """x [2, images, ...]; resample IMAGE IDs jointly, equal image weights.

    Finite intersection is used per metric for both models. The returned
    samples contain model 0, model 1, and paired (1 - 0), respectively.
    """
    shape = x.shape[2:]
    x = x.reshape(2, x.shape[1], -1)
    valid = np.isfinite(x).all(0)
    counts = valid.sum(0)
    clean = np.where(valid[None], x, 0)
    mean = clean.sum(1) / np.maximum(counts, 1)
    mean[:, counts == 0] = np.nan
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 100):
        weights = rng.multinomial(x.shape[1], np.full(x.shape[1], 1 / x.shape[1]),
                                  size=min(100, repetitions - start)).astype(float)
        denom = weights @ valid.astype(float)
        values = np.stack([weights @ clean[m] / np.where(denom > 0, denom, np.nan)
                           for m in range(2)])
        samples.append(np.concatenate([values, (values[1] - values[0])[None]], axis=0))
    samples = np.concatenate(samples, axis=1)
    ci = np.nanpercentile(samples, [2.5, 97.5], axis=1)
    point = np.concatenate([mean, (mean[1] - mean[0])[None]])
    return point.reshape((3,) + shape), ci.reshape((2, 3) + shape), counts.reshape(shape)


def make_figures(out, summary, examples, image_ids, labels, dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    shown = ['pair_top1_same', 'pair_top10pct_jaccard', 'pair_pearson',
             'all_top1_same', 'entropy_normalized', 'patch_group_mass']
    for ax, metric in zip(axes.flat, shown):
        data = summary[(summary.stratum == 'multi') & (summary.metric == metric)]
        for model, color in [('gwrp', '#2878b5'), ('c2p', '#d95319')]:
            ax.plot(data.layer, data[model], '.-', color=color, label=model.upper())
            ax.fill_between(data.layer, data[model + '_lo'], data[model + '_hi'],
                            color=color, alpha=.16)
        ax.set(title=metric.replace('_', ' '), xlabel='Transformer layer', xticks=range(1, 13))
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('VOC val: image-weighted positive-class attention; paired image bootstrap 95% CI')
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'layer_summary.{ext}', dpi=180)
    plt.close(fig)
    for idx, maps in examples.items():
        positive = np.flatnonzero(labels[idx])
        rgb = dataset[idx][0].numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * [.229, .224, .225] + [.485, .456, .406], 0, 1)
        for conditional in [False, True]:
            fig, axes = plt.subplots(2 * len(positive), 13,
                                     figsize=(26, 2.15 * 2 * len(positive)), squeeze=False)
            for row, (cls_pos, model_index) in enumerate(itertools.product(range(len(positive)), range(2))):
                axes[row, 0].imshow(rgb)
                axes[row, 0].set_ylabel(f'{NAMES[positive[cls_pos]]}\n{["GWRP", "C2P"][model_index]}', fontsize=11)
                for layer in range(12):
                    ax = axes[row, layer + 1]
                    a = maps[model_index][layer, cls_pos]
                    values = a / a.sum() * a.size if conditional else a
                    norm = LogNorm(1/16, 16, clip=True) if conditional else LogNorm(1e-6, .1, clip=True)
                    im = ax.imshow(values.reshape(28, 28), norm=norm, cmap='magma', interpolation='nearest')
                    peak = int(np.argmax(a))
                    ax.plot(peak % 28, peak // 28, marker='+', color='cyan', markersize=7, markeredgewidth=1)
                    if row == 0:
                        ax.set_title(f'L{layer + 1}')
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])
            mode = 'conditional' if conditional else 'raw'
            label = 'N * P(patch | patch keys), uniform = 1' if conditional else 'Actual head-mean A_c2p (not conditionalized)'
            fig.suptitle(f'{image_ids[idx]} | {label}\nAll layers/models share a fixed log color scale (clipped); cyan + = top-1', fontsize=14)
            fig.subplots_adjust(left=.055, right=.94, top=.91, bottom=.04, wspace=.04, hspace=.07)
            bar_ax = fig.add_axes([.952, .15, .008, .65])
            fig.colorbar(im, cax=bar_ax, extend='both')
            for ext in ['png', 'pdf']:
                fig.savefig(out / f'{image_ids[idx]}_{mode}.{ext}', dpi=130)
            plt.close(fig)


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from datasets_cam import VOC12Dataset, build_transform
    from models.mctformer_plus import build_mctformerplus

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    voc = Path('data/VOCdevkit/VOC2012').resolve()
    dataset = VOC12Dataset(str(voc), str(voc / 'ImageLists/val_id.txt'),
                           transform=build_transform(False, False, argparse.Namespace(input_size=448)))
    ids = list(dataset.img_name_list)
    labels = np.asarray(dataset.label_list) > 0
    nclass = labels.sum(1)
    assert len(ids) == 1449 and len(set(ids)) == 1449 and (nclass > 0).all()
    rng = np.random.default_rng(20260914)
    selected = sorted(np.concatenate([rng.choice(np.flatnonzero(mask), 3, replace=False)
                                      for mask in [nclass == 2, nclass >= 3]]).tolist())
    source_paths = [p for root in SOURCES.values() for p in root.iterdir()
                    if p.is_file() and (p.suffix in {'.json', '.pth'} or p.name == 'checkpoint_sha256.txt')]
    hashes = {str(p): sha256(p) for p in source_paths}
    manifest = {
        'command': shlex.join([sys.executable, '-m', 'analysis.c2p_pooling_attention'] + sys.argv[1:]),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
        'script_sha256': sha256(Path(__file__)), 'source_sha256_before': hashes,
        'torch': torch.__version__, 'python': sys.version,
        'device': torch.cuda.get_device_name(), 'precision': 'FP32; TF32 disabled; eval; no AMP',
        'transform': str(dataset.transform), 'input_size': 448, 'patch_grid': [28, 28],
        'bootstrap': {'repetitions': 5000, 'seed': 20260914, 'unit': 'paired image', 'weight': 'equal images'},
        'top10pct_k': 79, 'example_selection': '3 exactly-two + 3 three-or-more labels, seeded before inference',
        'example_ids': [ids[i] for i in selected],
        'counts': {'all': len(ids), 'single': int((nclass == 1).sum()),
                   'multi': int((nclass >= 2).sum()), 'two': int((nclass == 2).sum()),
                   'three_plus': int((nclass >= 3).sum())},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pd.DataFrame({'image_id': ids, 'n_positive': nclass,
                  'positive_classes': [','.join(np.array(NAMES)[l]) for l in labels]}).to_csv(out / 'images.csv', index=False)
    results = np.empty((2, len(ids), 12, len(METRICS)))
    examples = {i: [] for i in selected}
    checks = {}
    for model_index, (name, source) in enumerate(SOURCES.items()):
        checkpoint = torch.load(source / 'mctformerplus_final.pth', map_location='cpu', weights_only=False)
        model = build_mctformerplus('small', num_classes=20, input_size=448, patch_pooling=name,
                                    class_token_init='baseline').cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        loader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=False, pin_memory=True)
        position = 0
        max_equivalence_error = 0.
        saved_maps = []
        with torch.inference_mode():
            for images, batch_labels in loader:
                features = model.forward_features(images.cuda(non_blocking=True))
                records = features[2]
                assert len(records) == 12 and records[0].shape[1:] == (6, 804, 804)
                maps = torch.stack([a[:, :, :20, 20:].mean(1) for a in records], dim=1)
                native = model.c2p_spatial_weights(records, 784)
                derived = maps[:, -3:].mean(1)
                derived = derived / derived.sum(-1, keepdim=True)
                max_equivalence_error = max(max_equivalence_error, (native - derived).abs().max().item())
                torch.testing.assert_close(native, derived, rtol=1e-5, atol=1e-7)
                maps = maps.cpu().numpy()
                del features, records, native, derived
                for b in range(len(images)):
                    i = position + b
                    assert np.array_equal(batch_labels[b].numpy() > 0, labels[i])
                    # Index columns after selecting the image to retain [layer, class, patch].
                    positive_maps = maps[b][:, labels[i], :]
                    saved_maps.append(positive_maps)
                    for layer in range(12):
                        results[model_index, i, layer] = map_metrics(positive_maps[layer])
                    if i in examples:
                        examples[i].append(positive_maps.copy())
                position += len(images)
                if position % 100 == 0 or position == len(ids):
                    print(f'{name}: {position}/{len(ids)}', flush=True)
        assert position == len(ids)
        # Positive-only head-averaged attention, not full NxN tensors.
        np.savez_compressed(out / f'{name}_positive_attention.npz',
                            attention=np.concatenate(saved_maps, axis=1),
                            offsets=np.r_[0, np.cumsum(nclass)], image_ids=np.array(ids),
                            class_ids=np.concatenate([np.flatnonzero(l) for l in labels]))
        checks[name] = {'strict_load': True, 'images': position,
                        'native_last3_max_abs_error': max_equivalence_error}
        del model, saved_maps
        torch.cuda.empty_cache()
    rows = []
    for m, name in enumerate(SOURCES):
        for i in range(len(ids)):
            for layer in range(12):
                rows.append(dict(model=name, image_id=ids[i], n_positive=int(nclass[i]),
                                 layer=layer+1, **dict(zip(METRICS, results[m, i, layer]))))
    pd.DataFrame(rows).to_csv(out / 'per_image_layer_metrics.csv', index=False)
    summary_rows = []
    for stratum, mask in [('multi', nclass >= 2), ('two', nclass == 2), ('three_plus', nclass >= 3),
                          ('all', nclass >= 1), ('single', nclass == 1)]:
        # Singles have no class-pair metrics; avoid bootstrap all-NaN columns.
        metric_indices = np.arange(len(METRICS)) if stratum != 'single' else np.arange(6, 10)
        x = results[:, mask][:, :, :, metric_indices]
        means, ci, counts = paired_bootstrap(x)
        for l in range(12):
            for j, metric_index in enumerate(metric_indices):
                row = dict(stratum=stratum, layer=l+1, metric=METRICS[metric_index], n_images=int(counts[l, j]))
                for model_id, key in enumerate(['gwrp', 'c2p', 'delta']):
                    row[key] = means[model_id, l, j]
                    row[key + '_lo'] = ci[0, model_id, l, j]
                    row[key + '_hi'] = ci[1, model_id, l, j]
                summary_rows.append(row)
        print(f'Bootstrap complete: {stratum}', flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'layer_summary.csv', index=False)
    make_figures(out, summary, examples, ids, labels, dataset)
    after = {p: sha256(Path(p)) for p in hashes}
    assert after == hashes, 'Source files changed during read-only audit'
    manifest.update(source_sha256_after=after, source_integrity_unchanged=True, extraction_checks=checks)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = ['# GWRP vs C2P: layer-wise actual attention', '',
              f'Coverage: {manifest["counts"]}. Checkpoints and audited source metadata unchanged.', '',
              'Actual self-attention, averaged over all 6 heads, each of 12 layers. '
              'Map distribution metrics conditionalize over 784 patch keys; raw patch-group mass remains separate. '
              'No segmentation GT loaded. Positive labels are image-level; center crop can exclude labeled objects.', '',
              'Class-pair metrics are first averaged within each image, then equally over images. '
              '5,000 paired image bootstrap resamples; pointwise percentile 95% CIs, not simultaneous layerwise tests. '
              'This is one trained seed per method, not training-seed uncertainty. '
              'Top10% means 79 patches; absolute top10 is separately reported. Exact ties use stable raster-order selection.', '',
              '| Layer | Pair same top1 GWRP / C2P | Delta 95% CI | Top10% Jaccard GWRP / C2P | Pearson GWRP / C2P |',
              '|---|---|---|---|---|']
    for l in range(1, 13):
        d = summary[(summary.stratum == 'multi') & (summary.layer == l)].set_index('metric')
        t, j, p = [d.loc[k] for k in ['pair_top1_same', 'pair_top10pct_jaccard', 'pair_pearson']]
        report.append(f'| {l} | {t.gwrp:.4f} / {t.c2p:.4f} | {t.delta:.4f} [{t.delta_lo:.4f}, {t.delta_hi:.4f}] | '
                      f'{j.gwrp:.4f} / {j.c2p:.4f} | {p.gwrp:.4f} / {p.c2p:.4f} |')
    report += ['', '## Figures and reproduction', '',
               '`layer_summary.png/pdf`: all layers, multi-label images, pointwise confidence bands.',
               '`*_raw.png/pdf`: actual A_c2p, shared fixed logarithmic range 1e-6 to 0.1.',
               '`*_conditional.png/pdf`: 784 × conditional patch distribution, uniform=1; shared log range 1/16 to 16.',
               'Colorbar limits clip extremes for display only. Cyan + marks the argmax. '
               'Each example was selected from labels with a fixed seed before attention extraction; not representative-case cherry-picking.',
               'All numeric metrics use unclipped float64 calculations on extracted float32 maps.',
               'Agreement between different positive classes does not identify semantic ownership, background, or cause of CAM improvement.', '',
               'Exact command, code SHA, checkpoint/source hashes, precision and transform: `manifest.json`.',
               'Per-image measurements: `per_image_layer_metrics.csv`; strata/paired CIs: `layer_summary.csv`.',
               'Optional positive-only raw maps: `*_positive_attention.npz` (not needed to view compact results).']
    (out / 'ATTENTION_COMPARISON_REPORT.md').write_text('\n'.join(report) + '\n')
    (out / 'COMPLETE').write_text('All 1449 images, both models, all 12 layers; source hashes unchanged.\n')
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
