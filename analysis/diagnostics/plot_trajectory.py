"""Descriptive epoch associations, not a validated mask-free selection rule."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .writer import append_csv, result_path, write_json


def association(epochs, diagnostic, miou, direction=None):
    epochs, x, y = map(np.asarray, (epochs, diagnostic, miou))
    cam_valid = np.isfinite(y)
    cam_peak = int(epochs[cam_valid][np.argmax(y[cam_valid])]) if cam_valid.any() else None
    diagnostic_valid = np.isfinite(x)
    opt_epochs, opt_values = epochs[diagnostic_valid], x[diagnostic_valid]
    valid = np.isfinite(x) & np.isfinite(y)
    e, x, y = epochs[valid], x[valid], y[valid]
    rho = float(spearmanr(x, y).statistic) if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else float('nan')
    gap, selected = float('nan'), None
    if len(opt_values) and direction in {'min', 'max'} and np.ptp(opt_values) > 0:
        selected = int(opt_epochs[np.argmin(opt_values) if direction == 'min' else np.argmax(opt_values)])
        gap = abs(selected - cam_peak) if cam_peak is not None else float('nan')
    return dict(n_epochs=len(e), spearman=rho, diagnostic_opt_epoch=selected,
                peak_epoch_gap=gap, direction=direction,
                cam_peak_epoch=cam_peak)


def registered_direction(metric):
    if metric.startswith(('rho_cc', 'E_hi')):
        return 'min'
    if metric == 'xi':
        return 'max'
    # kappa/Gini are not monotone quality scores; delta has no registered
    # optimum. Never choose their favorable sign from the same GT trajectory.
    return None


def analyze(paths, output):
    output = result_path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs, signatures = [], set()
    for directory in map(Path, paths):
        manifest = json.loads((directory / 'manifest.json').read_text())
        cfg = manifest['probe_config']
        signatures.add((manifest['probe_set_sha256'], manifest['mini_set_sha256'], manifest['labels_sha256'], cfg['dataset'], cfg['res_semantic'], cfg['res_spectral'],
                        tuple(cfg['layers']), cfg['n_radial_bins'], cfg['gwrp_decay'], cfg['c2p_agg'], cfg['c2p_layers'],
                        manifest['class_token_init'], json.dumps(manifest['model_config'], sort_keys=True)))
        rows = list(csv.DictReader((directory / 'metrics_wide.csv').open()))
        rows = [r for r in rows if (directory / f"epoch_{int(r['epoch']):03d}.COMPLETE").exists()]
        if not rows:
            raise ValueError(f'No complete epochs: {directory}')
        if any(not r.get('mini_cam_miou_fixed') for r in rows):
            raise ValueError('Association analysis requires separately enabled mini-CAM outcomes')
        data = {k: np.asarray([float(r[k]) if r[k] not in ('', 'True', 'False') else np.nan for r in rows]) for k in rows[0]}
        method = manifest['pooling']
        if method == 'c2p':
            method += '-' + cfg['c2p_layers'] + '-' + cfg['c2p_agg'] + ('-affinity' if manifest['affinity'] else '')
        runs.append((directory, manifest, method, data))
    if len(signatures) != 1:
        raise ValueError('Probe/mini sets, labels, dataset and resolutions must match across runs')
    identities = [(method, m['probe_config']['seed']) for _, m, method, _ in runs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate method/seed run supplied')
    correlations, epochs_summary = [], []
    for directory, manifest, method, data in runs:
        seed = manifest['probe_config']['seed']
        for metric, values in data.items():
            if metric.startswith(('rho_cc', 'kappa__', 'gini__', 'delta_', 'E_hi')) or metric == 'xi':
                correlations.append(dict(method=method, seed=seed, metric=metric,
                                         **association(data['epoch'], values, data['mini_cam_miou_fixed'], registered_direction(metric))))
        y = data['mini_cam_miou_fixed']
        epochs_summary.append(dict(method=method, seed=seed, n_epochs=len(y), cam_peak_epoch=int(data['epoch'][np.argmax(y)]),
                                   last_epoch=int(data['epoch'][-1]), last_is_peak=bool(y[-1] == y.max()),
                                   cam_nondecreasing=bool(np.all(np.diff(y) >= 0))))
        long = list(csv.DictReader((directory / 'metrics.csv').open()))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, metric in zip(axes, ('kappa', 'gini')):
            layers = sorted({int(r['layer']) for r in long if r['metric'] == metric and r['layer'].isdigit()})
            e = data['epoch'].astype(int).tolist()
            matrix = np.full((len(layers), len(e)), np.nan)
            for r in long:
                if r['metric'] == metric and r['class_id'] == '-1' and r['layer'].isdigit() and int(r['epoch']) in e:
                    matrix[layers.index(int(r['layer'])), e.index(int(r['epoch']))] = float(r['mean']) if r['mean'] else np.nan
            im = ax.imshow(matrix, origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set(yticks=range(len(layers)), yticklabels=layers, xlabel='Observed epoch index', ylabel='Layer', title=metric)
            ax.set_xticks(range(len(e))[::max(1, len(e) // 8)], e[::max(1, len(e) // 8)])
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'{method}, seed {seed}')
        fig.savefig(output / f'layer_profile_{method}_s{seed}.pdf'); plt.close(fig)
    append_csv(output / 'correlations_per_seed.csv', correlations, list(correlations[0]))
    summaries = []
    for method, metric in sorted({(r['method'], r['metric']) for r in correlations}):
        rows = [r for r in correlations if r['method'] == method and r['metric'] == metric]
        values = np.asarray([r['spearman'] for r in rows])
        finite = values[np.isfinite(values)]
        summaries.append(dict(method=method, metric=metric, n_seeds=len(rows), n_valid=len(finite),
                              mean=float(finite.mean()) if len(finite) else np.nan,
                              std=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan))
    append_csv(output / 'correlations_summary.csv', summaries, list(summaries[0]))
    candidates = []
    methods = sorted({r[2] for r in runs})
    for metric in sorted({r['metric'] for r in correlations}):
        rows = [r for r in correlations if r['metric'] == metric]
        grouped = {method: [r for r in rows if r['method'] == method] for method in methods}
        same_seeds = len({tuple(sorted(r['seed'] for r in group)) for group in grouped.values()}) == 1
        same_epochs = len({tuple(d['epoch']) for _, _, _, d in runs}) == 1
        has_controls = 'gwrp' in methods and any(m.startswith('c2p-') for m in methods)
        signs = {int(np.sign(r['spearman'])) for r in rows if np.isfinite(r['spearman'])}
        passed = (has_controls and same_seeds and same_epochs and all(len(group) >= 3 for group in grouped.values())
                  and len(signs) == 1 and registered_direction(metric) is not None
                  and all(np.isfinite(r['spearman']) and abs(r['spearman']) >= .7 and r['peak_epoch_gap'] <= 2 for r in rows))
        candidates.append(dict(metric=metric, meets_descriptive_preregistered_threshold=bool(passed)))
    panels = ('rho_cc__12', 'kappa__aggregate', 'gini__aggregate', 'delta_js__aggregate', 'E_hi__spectral', 'xi')
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for ax, metric in zip(axes.flat, panels):
        right = ax.twinx()
        for i, (_, manifest, method, data) in enumerate(runs):
            color = f'C{i % 10}'
            style = '--' if method == 'gwrp' else '-'
            label = f'{method}/s{manifest["probe_config"]["seed"]}'
            ax.plot(data['epoch'], data['mini_cam_miou_fixed'], style, color=color, label=label + ' CAM')
            for branch, marker in [('class', ':'), ('patch', '-.')]:
                ax.plot(data['epoch'], data[f'probe_{branch}_macro_ap'], marker, color=color, alpha=.45, label=label + ' ' + branch + ' AP')
            x = data.get(metric, np.full_like(data['epoch'], np.nan))
            finite = x[np.isfinite(x)]
            if len(finite) and finite.std() > 0:
                right.plot(data['epoch'], (x - finite.mean()) / finite.std(), style, color=color, alpha=.55)
        ax.set(title=metric, xlabel='Epoch (zero-based)', ylabel='Probe AP / mini-CAM mIoU (fraction)')
        right.set_ylabel('Diagnostic z-score')
    axes.flat[0].legend(fontsize=6)
    fig.savefig(output / 'trajectories.pdf'); plt.close(fig)
    write_json(output / 'analysis.json', {'sources': list(map(str, paths)), 'epoch_observations': epochs_summary, 'criteria': candidates,
                                          'limitations': ['Epoch correlations are descriptive and autocorrelated.', 'Mask-backed mini-CAM is not an unlabeled validation set.',
                                                         'No mask-free checkpoint-selection claim without held-out validation.', 'A late maximum does not prove monotonicity or H1.',
                                                         'Undefined xi and constant Spearman remain missing; no favorable direction fitted for kappa/Gini/delta.']})
    (output / 'COMPLETE').touch(exist_ok=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.runs, args.output)
