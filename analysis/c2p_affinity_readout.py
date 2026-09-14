"""Compact pre/post-affinity weight audit on VOC val, without semantic GT."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analysis.c2p_pooling_attention import METRICS, NAMES, map_metrics, paired_bootstrap
from datasets_cam import VOC12Dataset, build_transform
from models.mctformer_plus import build_mctformerplus, validate_mctformerplus_patch_pooling_checkpoint
from tools.evaluate_cam_threshold_grid import sha256_file


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layers', choices=['last3', 'all'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path('data/VOCdevkit/VOC2012').resolve()
    dataset = VOC12Dataset(str(root), str(root / 'ImageLists/val_id.txt'),
                           transform=build_transform(False, False, argparse.Namespace(input_size=448)))
    ids = list(dataset.img_name_list)
    labels = np.asarray(dataset.label_list) > 0
    counts = labels.sum(1)
    rng = np.random.default_rng(20260914)
    selected = sorted(np.concatenate([rng.choice(np.flatnonzero(mask), 3, replace=False)
                                      for mask in [counts == 2, counts >= 3]]).tolist())
    metadata = {'checkpoint_sha256': sha256_file(args.checkpoint), 'layers': args.layers,
                'transform': str(dataset.transform), 'precision': 'FP32, TF32 off',
                'examples_selected_before_inference': [ids[i] for i in selected],
                'bootstrap': '5000 paired image resamples, seed 20260914, equal image weighting',
                'top10pct_k': 79, 'semantic_gt_loaded': False, 'limit': args.limit}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_mctformerplus_patch_pooling_checkpoint(checkpoint, 'c2p', args.layers, 'product', True)
    model = build_mctformerplus('small', input_size=448, num_classes=20, patch_pooling='c2p',
                               c2p_pooling_layers=args.layers, c2p_pooling_reduction='product',
                               c2p_pooling_affinity=True).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    n = args.limit or len(ids)
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=4,
                        shuffle=False, num_workers=4)
    metrics = ['entropy_normalized', 'top1_mass', 'pair_top10pct_jaccard']
    columns = [METRICS.index(m) for m in metrics]
    values = np.empty((2, n, len(metrics)))
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, _ in loader:
            _, patches, records, _ = model.forward_features(images.cuda())
            w = model.c2p_spatial_weights(records, patches.shape[1])
            after = model.c2p_affinity_weights(w, records, patches.shape[1])
            before_after = [w.cpu().numpy(), after.cpu().numpy()]
            for b in range(len(images)):
                i = offset + b
                maps = [a[b, labels[i]] for a in before_after]
                for s, stage in enumerate(['before', 'after']):
                    values[s, i] = map_metrics(maps[s])[columns]
                    rows.append(dict(image_id=ids[i], n_positive=int(counts[i]), stage=stage,
                                     **dict(zip(metrics, values[s, i]))))
                if i in selected:
                    plot_example(args.output, ids[i], images[b], labels[i], maps)
            offset += len(images)
            if offset % 100 == 0:
                print(f'weight diagnostics {offset}/{n}', flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'per_image.csv', index=False)
    summary = []
    for stratum, mask in [('all', counts[:n] > 0), ('multi', counts[:n] >= 2)]:
        if not mask.any():
            continue
        point, ci, valid = paired_bootstrap(values[:, mask], repetitions=5000)
        for j, metric in enumerate(metrics):
            for s, stage in enumerate(['before', 'after', 'after_minus_before']):
                summary.append(dict(stratum=stratum, metric=metric, stage=stage, n_images=int(valid[j]),
                                    mean=point[s, j], ci95_lo=ci[0, s, j], ci95_hi=ci[1, s, j]))
    pd.DataFrame(summary).to_csv(args.output / 'summary.csv', index=False)
    assert sha256_file(args.checkpoint) == metadata['checkpoint_sha256']
    (args.output / 'COMPLETE').write_text('complete\n')


def plot_example(output, image_id, image, labels, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    positive = np.flatnonzero(labels)
    rgb = np.clip(image.numpy().transpose(1, 2, 0) * [.229, .224, .225] + [.485, .456, .406], 0, 1)
    fig, axes = plt.subplots(len(positive), 3, figsize=(9, 3 * len(positive)), squeeze=False,
                             constrained_layout=True)
    for c, cls in enumerate(positive):
        axes[c, 0].imshow(rgb)
        axes[c, 0].set_title(NAMES[cls])
        for s, name in enumerate(['product weights', 'after P2P']):
            im = axes[c, s + 1].imshow(maps[s][c].reshape(28, 28) * 784,
                                      norm=LogNorm(1/16, 16, clip=True), cmap='magma')
            axes[c, s + 1].set_title(name)
        for ax in axes[c]:
            ax.set_axis_off()
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), label='784 * weight (uniform=1)', extend='both')
    fig.suptitle(image_id + ': same fixed color scale; no semantic GT')
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'{image_id}.{ext}', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
