"""Independent forward-only probes; no engine or model edits required."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import class_token_affinity, conditional_attention, divergence, effective_support, gini, gwrp_weights
from .probe_set import ProbeDataset, label_path
from .spectral import spectrum_and_autocorrelation, high_freq_ratio, correlation_length
from .state import isolated_eval
from .writer import ProbeWriter, append_csv, sha256, state_sha256


@dataclass
class ProbeConfig:
    enabled: bool = False
    every_n_epochs: int = 1
    probe_set_path: str = ''
    probe_batch_size: int = 32
    res_semantic: int = 448
    res_spectral: int = 448
    layers: tuple = tuple(range(12))  # zero-based in config; one-based in tables
    gwrp_decay: float = None  # derive from actual model; reject mismatches
    c2p_agg: str = None
    c2p_layers: str = None
    n_radial_bins: int = 16
    dump_raw: bool = False
    out_dir: str = ''
    dataset: str = 'VOC12'
    data_root: str = 'data/VOCdevkit/VOC2012'
    mini_cam_set_path: str = ''
    mini_cam_mask_dir: str = ''
    minimum_per_class: int = 8
    seed: int = 0  # training seed; fixed set itself is always generated with seed 0
    checkpoint_path: str = ''  # optional offline file provenance, not fabricated online


def _loader(dataset, batch_size):
    generator = torch.Generator().manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return {'mean': float(valid.mean()) if len(valid) else float('nan'),
            'std': float(valid.std()) if len(valid) else float('nan'),
            'n_valid': len(valid), 'n_total': len(values)}


def _image_means(x):
    count = np.isfinite(x).sum(-1)
    return np.divide(np.nansum(x, -1), count, out=np.full(len(x), np.nan), where=count > 0)


class TokenProbe:
    def __init__(self, model, cfg: ProbeConfig, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.model = model.module if hasattr(model, 'module') else model
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        self.writer = None
        if not cfg.enabled:
            return
        error = None
        if self.rank == 0:
            try:
                self._initialize()
            except Exception as exc:
                if not self.distributed:
                    raise
                error = f'{type(exc).__name__}: {exc}'
        if self.distributed:
            status = [error]
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0]:
                raise RuntimeError('Rank0 probe initialization failed: ' + status[0])

    def _initialize(self):
        cfg = self.cfg
        m = self.model
        if cfg.every_n_epochs < 1 or cfg.probe_batch_size < 1:
            raise ValueError('Probe interval and batch size must be positive')
        if not cfg.layers or any(l < 0 or l >= len(m.blocks) for l in cfg.layers) or len(set(cfg.layers)) != len(cfg.layers):
            raise ValueError('Probe layers must be unique valid zero-based block indices')
        if (m.token_interaction != 'joint' or m.bcss_variant != 'e0' or m.psl_spec.enabled
                or m.cti_bgt or m.last_mct or m.class_stable_last):
            raise ValueError('Probe supports ordinary joint MCTformer+ GWRP/C2P, without auxiliary variants')
        if m.patch_first or m.final_norm or m.patch_final_norm:
            raise ValueError('First probe implementation requires class-first, raw final token readout')
        if min(cfg.res_spectral // s for s in m.patch_embed.patch_size) < 28:
            raise ValueError('Spectral resolution must produce a grid >= 28 in both dimensions')
        if any(r % s for r in (cfg.res_semantic, cfg.res_spectral) for s in m.patch_embed.patch_size):
            raise ValueError('Probe resolutions must be multiples of patch size')
        for field, actual in [('gwrp_decay', m.decay_parameter), ('c2p_agg', m.c2p_pooling_reduction), ('c2p_layers', m.c2p_pooling_layers)]:
            if getattr(cfg, field) is not None and getattr(cfg, field) != actual:
                raise ValueError(f'Probe {field} must match actual model configuration')
            setattr(cfg, field, actual)
        self.dataset = ProbeDataset(cfg.data_root, cfg.dataset, cfg.probe_set_path, cfg.res_semantic)
        self.spectral_dataset = ProbeDataset(cfg.data_root, cfg.dataset, cfg.probe_set_path, cfg.res_spectral)
        if self.dataset.labels.shape[1] != m.num_classes or (self.dataset.labels.sum(0) < cfg.minimum_per_class).any():
            raise ValueError('Probe labels/classes do not meet configured coverage')
        self.mini_dataset = None
        if cfg.mini_cam_set_path:
            if not cfg.mini_cam_mask_dir:
                raise ValueError('mini-CAM requires a separate mask directory')
            self.mini_dataset = ProbeDataset(cfg.data_root, cfg.dataset, cfg.mini_cam_set_path, cfg.res_semantic)
            if set(self.dataset.ids) & set(self.mini_dataset.ids):
                raise ValueError('Probe and mini-CAM sets must be disjoint')
            if any(not (Path(cfg.mini_cam_mask_dir) / f'{i}.png').is_file() for i in self.mini_dataset.ids):
                raise FileNotFoundError('Missing mini-CAM mask')
        counts = self.dataset.labels.sum(0)
        self.class_ids = list(range(m.num_classes)) if cfg.dataset == 'VOC12' else sorted(np.argsort(-counts, kind='stable')[:20].tolist())
        self.writer = ProbeWriter(cfg.out_dir, {
            'probe_config': asdict(cfg), 'probe_set_sha256': sha256(cfg.probe_set_path),
            'model_config': {'num_classes': m.num_classes, 'embed_dim': m.embed_dim,
                             'depth': len(m.blocks), 'num_heads': m.blocks[0].attn.num_heads,
                             'patch_size': list(m.patch_embed.patch_size),
                             'attention_normalization': m.attention_normalization},
            'mini_set_sha256': sha256(cfg.mini_cam_set_path) if cfg.mini_cam_set_path else None,
            'labels_sha256': sha256(label_path(cfg.data_root, cfg.dataset)),
            'checkpoint_sha256': sha256(cfg.checkpoint_path) if cfg.checkpoint_path else None,
            'checkpoint_note': 'Online epochs identify in-memory model_state_sha256; no per-epoch checkpoint is written.',
            'pooling': m.patch_pooling, 'affinity': m.c2p_pooling_affinity,
            'class_token_init': m.class_token_init, 'probe_class_counts': counts.tolist(),
            'saved_class_ids': self.class_ids, 'semantic_transform': repr(self.dataset.transform),
            'spectral_transform': repr(self.spectral_dataset.transform),
            'aggregation': 'equal-image means; per-class conditional on presence; std is population image std, NOT CI',
            'spectral_protocol': 'L2 tokens, channel demeaning, Hann; radial-bin mean E_hi; actual 2D autocorrelation; fixed radii1:6 R2>=.8; failed xi missing',
            'mini_cam_protocol': 'separate mask set; matched resize+center crop, nearest masks; single scale/no flip/native CAM, fixed.45/grid0:.01:.59; not official full-image multiscale evaluation',
            'D5': 'each layer versus FINAL raw classifier logits; configured aggregate uses native helper (including affinity when enabled)',
            'semantic_gt_loaded_for_diagnostics': False,
        })

    def run(self, epoch: int, global_step: int) -> dict:
        if not self.cfg.enabled or epoch % self.cfg.every_n_epochs:
            return {}
        error, result = None, {}
        if self.rank == 0:
            try:
                with isolated_eval(self.model, self.device):
                    result = self._run(epoch, global_step)
            except Exception as exc:
                if not self.distributed:
                    raise
                error = f'{type(exc).__name__}: {exc}'
        if self.distributed:
            status = [error]
            # Non-root ranks block here, without forwarding through DDP. Rank0
            # reports errors too, so peers are not left at a success-only barrier.
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0]:
                raise RuntimeError('Rank0 diagnostic probe failed: ' + status[0])
        return result

    def _run(self, epoch, global_step):
        from tools.evaluate_mctformerplus_classification import classification_metrics
        from tools.evaluate_mini_cam import evaluate_mini_cam
        start = time.perf_counter()
        self.writer.begin(epoch)
        cfg, m = self.cfg, self.model
        fingerprint = state_sha256(m)
        collected, token_sums, image_counts = {}, {}, 0
        scores_cls, scores_patch, labels_all = [], [], []
        spectra, acs = [], []

        def add(name, layer, values):
            collected.setdefault((name, layer), []).append(values.detach().float().cpu().numpy())

        def spatial(patches, images):
            h = images.shape[-2] // m.patch_embed.patch_size[0]
            w = images.shape[-1] // m.patch_embed.patch_size[1]
            return patches.reshape(len(images), h, w, -1)

        for batch_index, (images, labels, names) in enumerate(_loader(self.dataset, cfg.probe_batch_size)):
            images, y = images.to(self.device).float(), labels.to(self.device) > 0
            cls, patches, records, raw_cls = m.forward_features(images)
            maps = m.head(spatial(patches, images).permute(0, 3, 1, 2).contiguous())
            gw = gwrp_weights(maps.flatten(2), m.decay_parameter)
            cp = m.c2p_spatial_weights(records, patches.shape[1])
            if m.c2p_pooling_affinity:
                cp = m.c2p_affinity_weights(cp, records, patches.shape[1])
            pooled = m.c2p_pool(maps, records) if m.patch_pooling == 'c2p' else m.gwrp(maps)
            scores_cls.append(cls.mean(-1).cpu())
            scores_patch.append(pooled.cpu())
            labels_all.append(labels)
            image_counts += len(images)
            for layer in cfg.layers:
                affinity = class_token_affinity(raw_cls[layer], y)
                add('rho_cc', layer + 1, affinity['rho_cc_per_class'])
                add('rho_cc_all_image', layer + 1, affinity['rho_cc_all_image'])
                token_sums[layer] = token_sums.get(layer, 0) + raw_cls[layer].double().sum(0).cpu()
                head = conditional_attention(records[layer][:, :, :m.num_classes, m.num_classes:])
                mean = conditional_attention(records[layer][:, :, :m.num_classes, m.num_classes:].mean(1))
                for name, value in {'kappa': effective_support(mean), 'gini': gini(mean),
                                    'kappa_head_variance': effective_support(head).var(1, unbiased=False),
                                    **divergence(gw, mean)}.items():
                    add(name, layer + 1, value.masked_fill(~y, float('nan')))
            for name, value in {'kappa': effective_support(cp), 'gini': gini(cp), **divergence(gw, cp)}.items():
                add(name, 'aggregate', value.masked_fill(~y, float('nan')))
            if cfg.dump_raw:
                np.savez_compressed(self.writer.output / f'raw_{epoch:03d}_{batch_index:04d}.npz',
                                    ids=np.asarray(names), c2p=np.stack([a[:, :, :m.num_classes, m.num_classes:].cpu().numpy() for a in records]))
            if cfg.res_semantic == cfg.res_spectral:
                p, centers, counts, ac = spectrum_and_autocorrelation(spatial(patches, images), cfg.n_radial_bins)
                spectra.append(p.cpu().numpy()); acs.append(ac.cpu().numpy())
            del records, raw_cls, patches, maps, cp, gw

        if cfg.res_semantic != cfg.res_spectral:
            for images, _, _ in _loader(self.spectral_dataset, cfg.probe_batch_size):
                images = images.to(self.device).float()
                features = m.forward_features(images)
                p, centers, counts, ac = spectrum_and_autocorrelation(spatial(features[1], images), cfg.n_radial_bins)
                spectra.append(p.cpu().numpy()); acs.append(ac.cpu().numpy())
                del features

        spectra, acs = np.concatenate(spectra), np.concatenate(acs)
        labels = torch.cat(labels_all).numpy()
        hi = high_freq_ratio(torch.from_numpy(spectra)).numpy()
        # D4 is not a single-image quality score; conditional class summaries
        # simply group the scalar by image-level presence, not semantic regions.
        collected['E_hi', 'spectral'] = [np.where(labels > 0, hi[:, None], np.nan)]
        fit = correlation_length(torch.from_numpy(acs).double().mean(0, keepdim=True))
        rows, wide = [], {'epoch': epoch, 'global_step': global_step}
        for (metric, layer), batches in collected.items():
            values = np.concatenate(batches)
            aggregate = hi if metric == 'E_hi' else (values if values.ndim == 1 else _image_means(values))
            stats = _stats(aggregate)
            rows.append(dict(epoch=epoch, global_step=global_step, metric=metric, layer=layer, class_id=-1, **stats))
            wide[f'{metric}__{layer}'] = stats['mean']
            if values.ndim == 2:
                for c in self.class_ids:
                    rows.append(dict(epoch=epoch, global_step=global_step, metric=metric, layer=layer, class_id=c, **_stats(values[:, c])))
        for layer, sums in token_sums.items():
            # Outputs are INPUT-DEPENDENT; average across images FIRST for the
            # requested global control. Parameter-only similarity is separate.
            global_aff = class_token_affinity((sums / image_counts)[None], torch.ones(1, m.num_classes))
            value = float(global_aff['rho_cc'][0])
            wide[f'rho_cc_all__{layer + 1}'] = value
            rows.append(dict(epoch=epoch, global_step=global_step, metric='rho_cc_all', layer=layer + 1, class_id=-1, **_stats([value])))
            for c in self.class_ids:
                rows.append(dict(epoch=epoch, global_step=global_step, metric='rho_cc_all', layer=layer + 1, class_id=c, **_stats([float(global_aff['rho_cc_per_class'][0, c])])))
        parameter_aff = class_token_affinity(m.cls_token, torch.ones(1, m.num_classes, device=self.device))
        wide['rho_cc_parameter'] = float(parameter_aff['rho_cc'][0])
        rows.append(dict(epoch=epoch, global_step=global_step, metric='rho_cc_parameter', layer='parameter', class_id=-1, **_stats([wide['rho_cc_parameter']])))
        for c in self.class_ids:
            rows.append(dict(epoch=epoch, global_step=global_step, metric='rho_cc_parameter', layer='parameter', class_id=c, **_stats([float(parameter_aff['rho_cc_per_class'][0, c])])))
        for name in ('xi', 'xi_r2', 'xi_slope'):
            value = float(fit[name][0])
            wide[name] = value
            rows.append(dict(epoch=epoch, global_step=global_step, metric=name, layer='spectral', class_id=-1, **_stats([value])))
        wide['xi_valid'] = bool(fit['xi_valid'][0])
        for branch, parts in [('class', scores_cls), ('patch', scores_patch)]:
            scores = torch.cat(parts)
            summary = classification_metrics(labels, torch.sigmoid(scores).numpy())
            wide[f'probe_{branch}_macro_ap'] = summary['macro_class_ap']
            wide[f'probe_{branch}_val_loss'] = float(torch.nn.functional.multilabel_soft_margin_loss(scores, torch.from_numpy(labels)))
        if self.mini_dataset is not None:
            metrics, confusion, seen = evaluate_mini_cam(m, _loader(self.mini_dataset, cfg.probe_batch_size), cfg.mini_cam_mask_dir, cfg.res_semantic, self.device)
            if seen != self.mini_dataset.ids:
                raise RuntimeError('mini-CAM coverage mismatch')
            wide.update(metrics)
            np.savez_compressed(self.writer.output / f'mini_cam_{epoch:03d}.npz', confusion=confusion)
            from tools.evaluate_cam_threshold_grid import confusion_metrics, threshold_grid
            curve = confusion_metrics(confusion)
            curve_rows = [dict(threshold=float(t), **{k: float(v[i]) for k, v in curve.items() if v.ndim == 1}) for i, t in enumerate(threshold_grid(0, .59, .01))]
            append_csv(self.writer.output / f'mini_cam_curve_{epoch:03d}.csv', curve_rows, list(curve_rows[0]))
        if state_sha256(m) != fingerprint:
            raise RuntimeError('Probe unexpectedly changed model state')
        elapsed = time.perf_counter() - start
        wide['probe_seconds'] = elapsed
        self.writer.finish(epoch, rows, wide, {
            'epoch': epoch, 'global_step': global_step, 'global_step_unit': 'attempted optimizer update boundaries, not guaranteed non-skipped AMP steps',
            'model_state_sha256': fingerprint, 'checkpoint_sha256': sha256(cfg.checkpoint_path) if cfg.checkpoint_path else None,
            'num_images': image_counts, 'mini_cam_images': len(self.mini_dataset) if self.mini_dataset else 0,
            'seconds': elapsed, 'xi_valid': wide['xi_valid'], 'metrics': wide,
        }, spectra, acs, centers.cpu().numpy(), counts.cpu().numpy())
        return wide

    def close(self):
        # No persistent forward hooks, handles, worker processes, or model copy.
        pass


def add_probe_arguments(parser):
    parser.add_argument('--probe', action='store_true', help='Independent label-only diagnostics, disabled by default')
    parser.add_argument('--probe-set', default='')
    parser.add_argument('--probe-out-dir', default='')
    parser.add_argument('--probe-every', type=int, default=1)
    parser.add_argument('--probe-batch-size', type=int, default=32)
    parser.add_argument('--probe-spectral-size', type=int, default=448)
    parser.add_argument('--probe-layers', default=','.join(map(str, range(12))))
    parser.add_argument('--probe-dump-raw', action='store_true')
    parser.add_argument('--probe-mini-set', default='')
    parser.add_argument('--probe-mini-mask-dir', default='')


def from_training_args(model, args, device):
    return TokenProbe(model, ProbeConfig(
        enabled=True, every_n_epochs=args.probe_every, probe_set_path=args.probe_set,
        probe_batch_size=args.probe_batch_size, res_semantic=args.input_size,
        res_spectral=args.probe_spectral_size, layers=tuple(map(int, args.probe_layers.split(','))),
        dump_raw=args.probe_dump_raw, out_dir=args.probe_out_dir or str(Path(args.work_space) / 'diagnostics'),
        dataset=args.dataset, data_root=args.voc12_root if args.dataset == 'VOC12' else args.coco_root,
        mini_cam_set_path=args.probe_mini_set, mini_cam_mask_dir=args.probe_mini_mask_dir,
        seed=args.seed,
    ), device)
