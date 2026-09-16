"""Pre-registered frozen affinity screening and native-protocol online CAM evaluation."""
import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.artifact_probe import (REPO, SEED, CHECKPOINTS, dataset, load_model,
                                     frozen_eval, write_json)
from analysis.c2p_pooling_attention import paired_bootstrap
from models.mctformer_plus import aggregate_p2p, propagate_c2p_weights
from tools.evaluate_cam_threshold_grid import sha256_file

DEFAULT = dict(p_layers='all', p_reduce='sum', p_alpha=1., p_sym=False, floor='none', beta=1.)
MINIMAL = DEFAULT | {'floor': 'min'}
CONSERVATIVE = dict(p_layers='last3', p_reduce='rownorm_mean', p_alpha=2., p_sym=True, floor='min', beta=.3)
WEIGHT_METRICS = ['entropy', 'top1', 'dc_share', 'pair_top10pct_jaccard', 'fallback_fraction']


def config_id(config):
    return '_'.join(str(config[k]).lower() for k in DEFAULT)


def preregistered_configs():
    """72 result-independent coverage configurations, not the full648 grid.

    Include current setting, every one-factor change and both registered arms;
    fill via deterministic maximin Hamming distance (then total distance).
    Lexicographic ID breaks ties. No measured quantity enters the design.
    """
    levels = [('all', 'last3', 'last6'), ('sum', 'mean', 'rownorm_mean'), (1., 2., 4.),
              (False, True), ('none', 'min', 'mean'), (1., .5, .3, .1)]
    full = [dict(zip(DEFAULT, values)) for values in itertools.product(*levels)]
    chosen = [DEFAULT.copy()]
    for key, values in zip(DEFAULT, levels):
        chosen.extend(DEFAULT | {key: value} for value in values if value != DEFAULT[key])
    chosen.append(CONSERVATIVE.copy())
    remaining = sorted([c for c in full if c not in chosen], key=config_id)
    def score(c):
        distances = [sum(c[k] != other[k] for k in DEFAULT) for other in chosen]
        return min(distances), sum(distances)
    while len(chosen) < 72:
        best = max(remaining, key=score)
        chosen.append(best)
        remaining.remove(best)
    assert MINIMAL in chosen and len({config_id(c) for c in chosen}) == 72
    return chosen


def operator(records, config):
    return aggregate_p2p(records, 20, **{k: config[k] for k in ['p_layers', 'p_reduce', 'p_alpha', 'p_sym']})


def weight_stats(w, labels, fallback=None):
    """Equal positive-class/pair weighting INSIDE image; image is sampling unit."""
    rows = []
    n = w.shape[-1]
    k = int(np.ceil(.1 * n))
    for i in range(len(w)):
        a = w[i, labels[i].bool()]
        p = a.clamp_min(1e-12)
        masks = torch.zeros_like(a, dtype=torch.bool)
        masks.scatter_(1, a.topk(k, -1).indices, True)
        pair_scores = []
        for c, d in itertools.combinations(range(len(a)), 2):
            pair_scores.append((masks[c] & masks[d]).sum() / (masks[c] | masks[d]).sum())
        jaccard = torch.stack(pair_scores).mean() if pair_scores else a.new_tensor(float('nan'))
        fail = a.new_tensor(0.) if fallback is None else fallback[i, labels[i].bool()].float().mean()
        rows.append(torch.stack([-(p * p.log()).sum(-1).mean() / np.log(n), a.amax(-1).mean(),
                                 (n * a.amin(-1)).mean(), jaccard, fail]))
    return torch.stack(rows)


def repaired_cam(seed, P, floor='none', beta=1., eps=1e-8):
    """CAM-only repair, never replace seed by pooling w or apply a TopK mask.

    S=sqrt(ReLU(M)*native-last3 A_c2p). For beta=1 use P*S then floor removal,
    preserving native absolute scale. For damping mix normalized S with normalized
    repaired response, restoring repaired response mass. Zero removed-floor mass
    falls back to S. Default is exactly native matrix multiplication.
    """
    r = torch.matmul(P.unsqueeze(1), seed.unsqueeze(-1)).squeeze(-1)
    if floor == 'min':
        r = r - r.amin(-1, keepdim=True)
    elif floor == 'mean':
        r = (r - r.mean(-1, keepdim=True)).clamp_min(0)
    elif floor != 'none':
        raise ValueError(floor)
    mass = r.sum(-1, keepdim=True)
    fallback = (mass.squeeze(-1) < eps) & (floor != 'none')
    if beta == 0:
        return seed, torch.zeros_like(fallback)
    if beta < 1:
        scaled_seed = seed / seed.sum(-1, keepdim=True).clamp_min(eps) * mass
        r = (1 - beta) * scaled_seed + beta * r
    if floor != 'none':
        r = torch.where(fallback[..., None], seed, r)
    return r, fallback


def start_output(args, stage):
    output = args.output.resolve()
    assert output.is_relative_to(REPO / 'results') and output != REPO / 'results'
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Tracked worktree must be clean')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sources = [REPO / p / 'mctformerplus_final.pth' for p in CHECKPOINTS.values()]
    sources += [REPO / 'data/VOCdevkit/VOC2012/ImageLists/val_id.txt',
                REPO / 'data/VOCdevkit/VOC2012/ImageLists/train_id.txt',
                REPO / 'data/VOCdevkit/VOC2012/ImageLabel/cls_labels.npy']
    if args.a1:
        assert (args.a1 / 'COMPLETE').exists()
        sources += [args.a1 / 'candidates.json', args.a1 / 'configs.json']
    manifest = dict(stage=stage, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256_before={str(p): sha256_file(p) for p in sources},
                    config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    precision='FP32 TF32off autocastoff', seed=SEED, bootstrap=5000,
                    checkpoint_policy='frozen final epoch44', semantic_gt_loaded=(stage == 'a2'))
    write_json(output / 'manifest.json', manifest)
    (output / 'commands.sh').write_text(shlex.join([sys.executable, '-m', 'analysis.affinity_repair', *sys.argv[1:]]) + '\n')
    (output / 'checkpoint_sha256.txt').write_text(''.join(f'{sha256_file(p)}  {p}\n' for p in sources[:3]))
    return output, manifest


def finish(output, manifest):
    manifest['source_sha256_after'] = {p: sha256_file(p) for p in manifest['source_sha256_before']}
    assert manifest['source_sha256_after'] == manifest['source_sha256_before']
    manifest['source_integrity_unchanged'] = True
    write_json(output / 'manifest.json', manifest)
    (output / 'COMPLETE').write_text('complete\n')


def a1(args):
    output, manifest = start_output(args, 'a1')
    assert (args.artifact / 'COMPLETE').exists()
    configs = preregistered_configs()
    write_json(output / 'configs.json', configs)
    manifest['design'] = '72 coverage configs: baseline+all OFAT+conservative; maximin Hamming fill; frozen before inference'
    manifest['selection'] = 'H in [.35,.65], DC<.5; sort abs(H-.5), config ID; max8 + minimal/conservative controls; <=10 unique'
    manifest['transform'] = str(dataset().transform)
    write_json(output / 'manifest.json', manifest)
    data = dataset()
    n = args.limit or len(data)
    labels = np.asarray(data.label_list[:n])
    assert len(data) == 1449 and (np.asarray(data.label_list).sum(1) >= 2).sum() == 522
    model = load_model('all_product')
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=2, shuffle=False, num_workers=4)
    values = np.empty((n, len(configs), len(WEIGHT_METRICS)))
    before = np.empty((n, len(WEIGHT_METRICS)))
    offset = 0
    with frozen_eval(model):
        for images, positive in loader:
            positive = positive.cuda()
            _, patches, records, _ = model.forward_features(images.cuda())
            w = model.c2p_spatial_weights(records, patches.shape[1])
            before[offset:offset + len(images)] = weight_stats(w, positive).cpu().numpy()
            # Compact head means only; release full 6-head records before config loop.
            reduced = [r.mean(1, keepdim=True) for r in records]
            del records
            block = []
            for config in configs:
                P = operator(reduced, config)
                repaired, fallback = propagate_c2p_weights(w, P, floor=config['floor'], beta=config['beta'], return_fallback=True)
                block.append(weight_stats(repaired, positive, fallback))
            values[offset:offset + len(images)] = torch.stack(block, 1).cpu().numpy()
            offset += len(images)
            if offset % 50 == 0 or offset == n:
                print(f'A1 {offset}/{n}', flush=True)
    pd.DataFrame([dict(image_id=data.img_name_list[i], config_id=config_id(c), n_positive=int(labels[i].sum()),
                       **dict(zip(WEIGHT_METRICS, values[i, j])))
                  for i in range(n) for j, c in enumerate(configs)]).to_csv(output / 'per_image.csv', index=False)
    rows = []
    for stratum, mask in [('all', labels.sum(1) > 0), ('multi', labels.sum(1) >= 2)]:
        if not mask.any():
            continue
        reference = np.broadcast_to(before[mask, None, :], values[mask].shape)
        point, ci, count = paired_bootstrap(np.stack([reference, values[mask]]), 5000, SEED)
        for j, config in enumerate(configs):
            for m, metric in enumerate(WEIGHT_METRICS):
                for s, stage in enumerate(['before', 'after', 'delta']):
                    rows.append(dict(config_id=config_id(config), stratum=stratum, metric=metric, stage=stage,
                                     n_images=int(count[j, m]), mean=point[s, j, m], ci95_lo=ci[0, s, j, m], ci95_hi=ci[1, s, j, m]))
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    means = np.nanmean(values, 0)
    passing = [i for i in range(len(configs)) if .35 <= means[i, 0] <= .65 and means[i, 2] < .5]
    passing.sort(key=lambda i: (abs(means[i, 0] - .5), config_id(configs[i])))
    selected = [MINIMAL, CONSERVATIVE]
    for i in passing:
        if configs[i] not in selected:
            selected.append(configs[i])
        if len(selected) == 10:
            break
    write_json(output / 'candidates.json', selected)
    write_json(output / 'screening.json', dict(passing_count=len(passing), selected=[dict(config=c,
               passed_band=any(c == configs[i] for i in passing), mandatory_control=c in [MINIMAL, CONSERVATIVE]) for c in selected]))
    finish(output, manifest)


def cam_bootstrap(confusions, reference, reps=5000):
    """Paired IMAGE resampling of fixed-threshold confusion, not image IoU means."""
    from tools.evaluate_cam_threshold_grid import confusion_metrics
    n = len(reference)
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, reps, 100):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(100, reps-start)).astype(float)
        a = (weights @ reference.reshape(n, -1)).reshape(-1, 21, 21)
        b = (weights @ confusions.reshape(n, -1)).reshape(-1, 21, 21)
        ma, mb = confusion_metrics(a)['mean_iou'], confusion_metrics(b)['mean_iou']
        samples.append(np.stack([ma, mb, mb-ma], 1))
    return np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0)


def a2(args):
    from datasets_cam import VOC12DatasetMS, build_transform
    from models.adapter_modules import resize_input_minbound
    from models.mctformer_plus import build_mctformerplus
    from tools.evaluate_cam_threshold_grid import cam_payload_winner, image_threshold_confusions, threshold_grid
    from tools.evaluate_raw_cam_streaming import save_summary
    from make_cam import normalize_cam, flip_cam
    from PIL import Image
    output, manifest = start_output(args, 'a2')
    configs = json.loads((args.a1 / 'candidates.json').read_text())
    configs = [DEFAULT] + [c for c in configs if c != DEFAULT]
    write_json(output / 'configs.json', configs)
    root = REPO / 'data/VOCdevkit/VOC2012'
    id_list = root / 'ImageLists/train_id.txt'
    data = VOC12DatasetMS(str(root), str(id_list), (1., .75, 1.25),
                          build_transform(False, True, argparse.Namespace(input_size=448)))
    assert len(data) == 1464
    n = args.limit or len(data)
    thresholds = threshold_grid(0, .59, .01)
    manifest['mask_sha256_before'] = {str(root/'SegmentationClass'/f'{name}.png'):
                                      sha256_file(root/'SegmentationClass'/f'{name}.png')
                                      for name in data.img_name_list[:n]}
    manifest['CAM_protocol'] = 'native three scales+flip/minbound448, original image size, sum then class-minmax, GT image-label gating; fixed.45'
    manifest['repair_CAM'] = repaired_cam.__doc__
    manifest['selection_bias'] = 'A2/A3 train1464 are screening, not independent generalization; 5000 image CI excludes model-selection uncertainty'
    write_json(output / 'manifest.json', manifest)
    rows = []
    for name in ['all_product', 'gwrp']:
        # Load exact trained state into ordinary CAM model for reference equivalence.
        source = load_model(name)
        model = build_mctformerplus('small', input_size=448, num_classes=20, cam=True).cuda()
        model.load_state_dict(source.state_dict(), strict=True)
        del source
        totals = np.zeros((len(configs), 60, 21, 21), dtype=np.int64)
        fixed = np.zeros((len(configs), n, 21, 21), dtype=np.int32)
        fallbacks = np.zeros(len(configs), dtype=np.int64)
        max_error = 0.
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=1, shuffle=False, num_workers=4)
        with frozen_eval(model):
            for i, pack in enumerate(loader):
                size = [int(x.item()) for x in pack['size']]
                positive = pack['label'][0].nonzero().flatten()
                multi_scale = [[] for _ in configs]
                for image in pack['img']:
                    inputs = resize_input_minbound(image[0].cuda(), min_size=448)
                    _, patches, records, _ = model.forward_features(inputs)
                    h, w = inputs.shape[-2]//16, inputs.shape[-1]//16
                    logits = model.head(patches.transpose(1, 2).reshape(2, 384, h, w))
                    means = torch.stack([r.mean(1) for r in records])
                    del records
                    seed = (logits.relu().flatten(2) * means[-3:].mean(0)[:, :20, 20:]).sqrt()
                    native_p = means[:, :, 20:, 20:].sum(0)
                    reduced = list(means.unsqueeze(2).unbind(0))
                    for j, config in enumerate(configs):
                        P = native_p if config == DEFAULT else operator(reduced, config)
                        cams, fallback = repaired_cam(seed, P, floor=config['floor'], beta=config['beta'])
                        cams = cams.reshape(2, 20, h, w)
                        if config == DEFAULT and i == 0:
                            official = model.get_cam(logits, means)
                            torch.testing.assert_close(cams, official, rtol=1e-6, atol=1e-6)
                            max_error = max(max_error, (cams-official).abs().max().item())
                        fallbacks[j] += int(fallback[:, positive].sum())
                        multi_scale[j].append(F.interpolate(cams, size, mode='bilinear', align_corners=False))
                image_id = pack['name'][0]
                with Image.open(root / 'SegmentationClass' / f'{image_id}.png') as mask:
                    target = np.asarray(mask)
                for j in range(len(configs)):
                    summed = torch.stack(flip_cam(multi_scale[j])).sum(0)[positive]
                    normalized = normalize_cam(summed).cpu().numpy()
                    payload = {int(cls): normalized[k] for k, cls in enumerate(positive)}
                    scores, classes = cam_payload_winner(payload, target.shape, 21)
                    conf = image_threshold_confusions(scores, classes, target, thresholds, 21)
                    totals[j] += conf
                    fixed[j, i] = conf[45]
                if (i+1) % 100 == 0 or i+1 == n:
                    print(f'A2/A3 {name} {i+1}/{n}', flush=True)
        for j, config in enumerate(configs):
            directory = output / name / config_id(config)
            directory.mkdir(parents=True, exist_ok=False)
            metrics = save_summary(totals[j], thresholds, n, 21, directory,
                                   '<online, not persisted>', root/'SegmentationClass', id_list)
            ci = cam_bootstrap(fixed[j], fixed[0])
            rows.append(dict(model=name, config_id=config_id(config), native=config==DEFAULT,
                             fixed_miou=metrics['fixed']['mean_iou'], best_miou=metrics['best']['mean_iou'],
                             best_threshold=metrics['best']['threshold'],
                             precision=metrics['fixed']['semantic_foreground_precision'],
                             recall=metrics['fixed']['semantic_foreground_recall'],
                             fixed_lo=ci[0, 1], fixed_hi=ci[1, 1],
                             delta_fixed=metrics['fixed']['mean_iou']-rows[-j]['fixed_miou'] if j else 0.,
                             delta_lo=ci[0, 2], delta_hi=ci[1, 2], fallback_class_rows=int(fallbacks[j])))
        np.savez_compressed(output / f'{name}_fixed_confusions.npz', confusion=fixed, image_ids=data.img_name_list[:n])
        manifest[name+'_native_cam_max_abs_error'] = max_error
        reference = json.loads((REPO/CHECKPOINTS[name]/'raw_cam/metrics.json').read_text())
        native = rows[-len(configs)]['fixed_miou']
        manifest[name+'_historical_native_delta'] = native - reference['fixed']['mean_iou']
        if not args.limit and abs(manifest[name+'_historical_native_delta']) > 1e-4:
            raise RuntimeError('Native protocol reproduction mismatch; no ranking/training allowed')
        pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
        del model
        torch.cuda.empty_cache()
    repaired_rows = [r for r in rows if r['model']=='all_product' and not r['native']]
    best = max(repaired_rows, key=lambda r: (r['fixed_miou'], r['config_id']))
    reference_value = json.loads((REPO/CHECKPOINTS['all_product']/'raw_cam/metrics.json').read_text())['fixed']['mean_iou']
    winner = next(c for c in configs if config_id(c)==best['config_id'])
    decision = dict(best=best, best_config=winner, full_dataset=not bool(args.limit),
                    stage_b_gate_passed=not args.limit and best['fixed_miou'] > reference_value,
                    reference_fixed_miou=reference_value,
                    training_configs=[MINIMAL, winner, CONSERVATIVE])
    write_json(output / 'decision.json', decision)
    report = ['# Affinity repair screening', '', 'Inference-only screening, NOT retraining results or causal proof.',
              'Fixed threshold.45 is the ranking metric; best threshold is diagnostic only.',
              'Paired5000 image bootstrap excludes selection bias and training-seed uncertainty.', '',
              pd.DataFrame(rows).to_csv(index=False), '',
              'Stage B gate: ' + str(decision['stage_b_gate_passed']),
              'No register, Gram, COCO or additional seeds authorized by this queue.']
    (output/'AFFINITY_SCREENING_REPORT.md').write_text('\n'.join(report))
    manifest['mask_sha256_after'] = {p: sha256_file(p) for p in manifest['mask_sha256_before']}
    assert manifest['mask_sha256_before'] == manifest['mask_sha256_after']
    finish(output, manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['a1', 'a2'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, default=Path('results/artifact_probe/20260916-voc-s0'))
    parser.add_argument('--a1', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    try:
        a1(args) if args.stage == 'a1' else a2(args)
    except BaseException as exc:
        if args.output.is_dir():
            (args.output/'FAILED').write_text(repr(exc)+'\n')
        raise


if __name__ == '__main__':
    main()
