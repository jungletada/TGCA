#!/usr/bin/env python
"""Layer-0--12 shared-presence diagnostics for the frozen CWP checkpoint."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analysis.lazy_assignment.experiment2.voc_semantic_dataset import VOCSemanticDataset
from analysis.relational_selector.analyze_frozen_relations import _pca_table, _task_a_summary
from analysis.relational_selector.layer_capture import (
    LayerRelationCollector,
    compute_token_relations,
)
from analysis.relational_selector.task_a import (
    evaluate_layer_relations,
    global_patch_regions,
)
from analysis.spatial_graph_stability.provenance import (
    EXPECTED_VOC_IMAGES,
    RunLog,
    command_line,
    create_output,
    csv_dump,
    git_metadata,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    timestamp,
    write_environment_manifests,
)
from models.mctformer_plus import (
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_class_token_init_checkpoint,
    validate_mctformerplus_final_norm_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cwp-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-task-summary', type=Path, required=True)
    parser.add_argument('--baseline-pca', type=Path, required=True)
    parser.add_argument('--official-pretrained', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--voc-root', type=Path, default=Path('data/VOCdevkit/VOC2012'))
    parser.add_argument('--list-path', type=Path, default=Path('data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--allow-tracked-dirty', action='store_true')
    return parser.parse_args()


def _resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _load_cwp(path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(checkpoint, 'mctformerplus')
    if resolution['variant'] != 'small':
        raise RuntimeError('CWP shared-presence analysis requires Small')
    validate_mctformerplus_class_token_init_checkpoint(checkpoint, 'cwp')
    validate_mctformerplus_final_norm_checkpoint(
        checkpoint, False, False, False, False
    )
    model = build_mctformerplus(
        'small', cam=True, num_classes=20, input_size=448,
        attention_normalization='vanilla', attention_gamma=1.0,
        bcss_variant='e0', psl_variant='baseline', cti_bgt=False,
        final_norm=False, patch_final_norm=False, last_mct=False,
        class_stable_last=False, class_token_init='cwp',
    )
    result = model.load_state_dict(checkpoint['model'], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'CWP strict checkpoint load failed: {result}')
    return model, checkpoint


def _validate_baseline(path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    resolve_mctformerplus_checkpoint_variant(checkpoint, 'mctformerplus')
    validate_mctformerplus_class_token_init_checkpoint(checkpoint, 'baseline')
    validate_mctformerplus_final_norm_checkpoint(
        checkpoint, False, False, False, False
    )
    return checkpoint


def _positive_pair_cosine(tokens: torch.Tensor, labels: torch.Tensor):
    tokens = F.normalize(tokens.float(), dim=-1)
    matrices = torch.matmul(tokens, tokens.transpose(1, 2))
    output = []
    for matrix, active in zip(matrices, labels > 0):
        ids = torch.nonzero(active, as_tuple=False).flatten().tolist()
        values = [matrix[first, second] for first, second in itertools.combinations(ids, 2)]
        output.append(float(torch.stack(values).mean()) if values else float('nan'))
    return output


def _initial_cwp_state(path: Path, device: torch.device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_mctformerplus(
            'small', num_classes=20, input_size=448,
            attention_normalization='vanilla', bcss_variant='e0',
            psl_variant='baseline', cti_bgt=False,
            class_token_init='cwp',
        )
    source = torch.load(path, map_location='cpu')
    adapted, report = adapt_deit_checkpoint_for_mctformerplus(
        source, model, num_classes=20
    )
    model.load_state_dict(adapted, strict=True)
    state = model.state_dict()
    keys = (
        'patch_embed.proj.weight', 'patch_embed.proj.bias', 'pos_embed_pat',
        'pos_embed_cls', 'class_token_pooler.class_queries',
    )
    return {key: state[key].to(device) for key in keys}, report


def _cwp_layer0_from_state(images, state):
    patches = F.conv2d(
        images, state['patch_embed.proj.weight'], state['patch_embed.proj.bias'],
        stride=16,
    ).flatten(2).transpose(1, 2) + state['pos_embed_pat']
    query = state['class_token_pooler.class_queries'].float()
    logits = torch.matmul(
        query.unsqueeze(0), patches.float().transpose(1, 2)
    ) / np.sqrt(384.0)
    attention = torch.softmax(logits, dim=-1)
    classes = torch.matmul(attention, patches.float()).to(patches.dtype)
    classes = classes + state['pos_embed_cls']
    return classes, patches, attention


class PoolingAccumulator:
    def __init__(self):
        self.count = 0
        self.sum = np.zeros((20, 784), dtype=np.float64)
        self.square = np.zeros((20, 784), dtype=np.float64)
        self.entropy = []
        self.interclass_cosine = []
        self.uniform_deviation = []

    def update(self, attention: torch.Tensor):
        values = attention.detach().float()
        count = values.shape[0]
        array = values.cpu().numpy().astype(np.float64)
        self.count += count
        self.sum += array.sum(axis=0)
        self.square += np.square(array).sum(axis=0)
        entropy = -(values * values.clamp_min(1e-30).log()).sum(-1) / np.log(784)
        normalized = F.normalize(values, dim=-1)
        cosine = torch.matmul(normalized, normalized.transpose(1, 2))
        mask = ~torch.eye(20, dtype=torch.bool, device=values.device).unsqueeze(0)
        self.entropy.extend(entropy.mean(-1).cpu().tolist())
        self.interclass_cosine.extend(
            cosine.masked_select(mask).reshape(count, -1).mean(-1).cpu().tolist()
        )
        self.uniform_deviation.extend(
            (values - 1.0 / 784).abs().mean(dim=(1, 2)).cpu().tolist()
        )

    def summary(self):
        variance = np.maximum(
            self.square / self.count - np.square(self.sum / self.count), 0.0
        )
        return {
            'num_images': self.count,
            'normalized_entropy_mean': float(np.mean(self.entropy)),
            'normalized_entropy_std_across_images': float(np.std(self.entropy)),
            'interclass_attention_cosine_mean': float(np.mean(self.interclass_cosine)),
            'interclass_attention_cosine_std_across_images': float(np.std(self.interclass_cosine)),
            'mean_absolute_deviation_from_uniform': float(np.mean(self.uniform_deviation)),
            'mean_attention_std_across_images': float(np.sqrt(variance).mean()),
            'max_attention_row_sum_error': None,
        }


def _pca_any(mu: np.ndarray):
    if mu.ndim != 3 or mu.shape[-1] != 384:
        raise ValueError(f'expected [N,L,384] shared means, got {mu.shape}')
    if mu.shape[1] == 12:
        return _pca_table(mu)[0]
    rows = []
    for layer in range(mu.shape[1]):
        values = np.asarray(mu[:, layer], dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        _u, singular, _vh = np.linalg.svd(centered, full_matrices=False)
        eigen = np.square(singular)
        total = float(eigen.sum())
        fractions = eigen / total if total else np.zeros_like(eigen)
        nonzero = fractions[fractions > 0]
        cumulative = np.cumsum(fractions)
        rows.append({
            'layer': layer,
            'pc1_fraction': float(fractions[0]),
            'pc2_fraction': float(fractions[1]),
            'pc4_cumulative': float(cumulative[min(3, len(cumulative) - 1)]),
            'pc8_cumulative': float(cumulative[min(7, len(cumulative) - 1)]),
            'pc16_cumulative': float(cumulative[min(15, len(cumulative) - 1)]),
            'participation_rank': float(1.0 / np.square(fractions).sum()) if total else float('nan'),
            'effective_rank': float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if len(nonzero) else float('nan'),
            'r90': int(np.searchsorted(cumulative, 0.90) + 1) if total else 0,
            'r95': int(np.searchsorted(cumulative, 0.95) + 1) if total else 0,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    os.chdir(root)
    require_environment()
    if args.limit < 0 or args.batch_size < 1:
        raise ValueError('limit must be non-negative and batch-size positive')
    if args.limit == 0 and args.allow_tracked_dirty:
        raise RuntimeError('full analysis requires a clean tracked checkout')
    source_git = git_metadata(root) if args.allow_tracked_dirty else require_clean_tracked(root)
    output = create_output(_resolve(root, args.output_dir))
    log = RunLog(output / 'run.log')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = {
        name: _resolve(root, value) for name, value in {
            'cwp_checkpoint': args.cwp_checkpoint,
            'baseline_checkpoint': args.baseline_checkpoint,
            'baseline_task_summary': args.baseline_task_summary,
            'baseline_pca': args.baseline_pca,
            'official_pretrained': args.official_pretrained,
            'voc_root': args.voc_root,
            'list_path': args.list_path,
        }.items()
    }
    for name, path in paths.items():
        if name != 'voc_root' and not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cwp_model, cwp_checkpoint = _load_cwp(paths['cwp_checkpoint'])
    baseline_checkpoint = _validate_baseline(paths['baseline_checkpoint'])
    initial_state, initial_pretrained_report = _initial_cwp_state(
        paths['official_pretrained'], device
    )
    cwp_model = cwp_model.to(device).eval()
    dataset = VOCSemanticDataset(
        paths['voc_root'], paths['list_path'], input_size=448, limit=args.limit
    )
    expected = args.limit or EXPECTED_VOC_IMAGES
    if len(dataset) != expected:
        raise RuntimeError(f'expected {expected} images, found {len(dataset)}')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    collector = LayerRelationCollector(
        cwp_model, num_classes=20, patch_count=784, width=384
    )
    cwp_rows = []
    cwp_mu = []
    pooling_accumulators = {
        'official_deit_initialization': PoolingAccumulator(),
        'trained_cwp': PoolingAccumulator(),
    }
    pooling_row_errors = {key: 0.0 for key in pooling_accumulators}
    index = 0
    before_hashes = {
        'cwp': sha256_file(paths['cwp_checkpoint']),
        'baseline': sha256_file(paths['baseline_checkpoint']),
    }
    log(f'start images={len(dataset)} device={device} batch_size={args.batch_size}')
    try:
        with torch.no_grad():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                masks = batch['mask']
                names = [str(name) for name in batch['name']]
                collector.set_positive_labels(labels)
                _cls, _patch, _attn, _all_cls, auxiliary = (
                    cwp_model.forward_features(images, return_aux=True)
                )
                records = collector.consume()
                cwp_l0 = compute_token_relations(
                    auxiliary['initial_class_tokens'],
                    auxiliary['initial_patch_tokens'], labels,
                )
                initial_classes, initial_patches, initial_attention = (
                    _cwp_layer0_from_state(images, initial_state)
                )
                del initial_classes, initial_patches
                final_attention = auxiliary['class_token_pooling_attention']
                pooling_accumulators['official_deit_initialization'].update(
                    initial_attention
                )
                pooling_accumulators['trained_cwp'].update(final_attention)
                pooling_row_errors['official_deit_initialization'] = max(
                    pooling_row_errors['official_deit_initialization'],
                    float((initial_attention.sum(-1) - 1).abs().max()),
                )
                pooling_row_errors['trained_cwp'] = max(
                    pooling_row_errors['trained_cwp'],
                    float((final_attention.sum(-1) - 1).abs().max()),
                )
                cwp_mu.append(torch.stack(
                    [cwp_l0.mean_class] + [item.mean_class for item in records],
                    dim=1,
                ).cpu().numpy())
                cwp_l0_cosine = _positive_pair_cosine(
                    auxiliary['initial_class_tokens'], labels
                )
                labels_np = labels.cpu().numpy()
                for local, image_id in enumerate(names):
                    regions = global_patch_regions(masks[local].numpy())
                    for layer, relation in enumerate([cwp_l0] + records):
                        row = evaluate_layer_relations(
                            image_id=image_id, image_index=index + local,
                            labels=labels_np[local], regions=regions, layer=layer,
                            raw=relation.raw[local].cpu().numpy(),
                            residual=relation.residual[local].cpu().numpy(),
                            common=relation.common[local].cpu().numpy(),
                            positive_common=relation.positive_common[local].cpu().numpy(),
                        )
                        row['initial_positive_class_token_pair_cosine'] = (
                            cwp_l0_cosine[local] if layer == 0 else float('nan')
                        )
                        cwp_rows.append(row)
                index += len(names)
                if batch_number == 1 or batch_number % 20 == 0 or index == len(dataset):
                    log(f'processed {index}/{len(dataset)}')
    finally:
        collector.close()
    after_hashes = {
        'cwp': sha256_file(paths['cwp_checkpoint']),
        'baseline': sha256_file(paths['baseline_checkpoint']),
    }
    if before_hashes != after_hashes:
        raise RuntimeError('source checkpoint changed during analysis')

    cwp_frame = pd.DataFrame(cwp_rows)
    cwp_summary = _task_a_summary(cwp_frame)
    for frame, source in ((cwp_summary, cwp_frame),):
        cosine = source.groupby('layer')[
            'initial_positive_class_token_pair_cosine'
        ].mean().rename('initial_positive_class_token_pair_cosine')
        frame['initial_positive_class_token_pair_cosine'] = frame.layer.map(cosine)

    cwp_mu_values = np.concatenate(cwp_mu, axis=0).astype(np.float32)
    cwp_pca = _pca_any(cwp_mu_values)
    baseline_pca = pd.read_csv(paths['baseline_pca'])
    baseline_summary = pd.read_csv(paths['baseline_task_summary'])
    summary_compare = baseline_summary.merge(
        cwp_summary, on='layer', how='outer',
        suffixes=('_baseline', '_cwp'), validate='one_to_one'
    )
    pca_compare = baseline_pca[['layer', 'effective_rank']].merge(
        cwp_pca[['layer', 'effective_rank']], on='layer',
        how='outer', suffixes=('_baseline', '_cwp'), validate='one_to_one'
    )
    comparison = summary_compare[[
        'layer', 'raw_pair_corr_baseline', 'raw_pair_corr_cwp',
        'raw_pair_jaccard_top05_baseline', 'raw_pair_jaccard_top05_cwp',
        'common_r2_baseline', 'common_r2_cwp',
    ]].merge(pca_compare, on='layer', validate='one_to_one')
    for metric in (
        'raw_pair_corr', 'raw_pair_jaccard_top05', 'common_r2', 'effective_rank'
    ):
        comparison[f'{metric}_delta_cwp_minus_baseline'] = (
            comparison[f'{metric}_cwp'] - comparison[f'{metric}_baseline']
        )

    pooling_summary = {
        key: accumulator.summary()
        for key, accumulator in pooling_accumulators.items()
    }
    for key, value in pooling_row_errors.items():
        pooling_summary[key]['max_attention_row_sum_error'] = value

    csv_dump(output / 'cwp_per_image_layer.csv', cwp_frame.to_dict('records'), list(cwp_frame.columns))
    csv_dump(output / 'cwp_layer_summary.csv', cwp_summary.to_dict('records'), list(cwp_summary.columns))
    csv_dump(output / 'cwp_pca.csv', cwp_pca.to_dict('records'), list(cwp_pca.columns))
    csv_dump(output / 'baseline_vs_cwp_layer_comparison.csv', comparison.to_dict('records'), list(comparison.columns))
    json_dump(output / 'cwp_layer0_pooling_summary.json', pooling_summary)
    environment = write_environment_manifests(output)
    metadata = {
        'schema': 'mctformerplus_cwp_shared_presence_v1',
        'created_at': timestamp(),
        'command': command_line(),
        'git': source_git,
        'num_images': len(dataset),
        'checkpoint_sha256': before_hashes,
        'checkpoint_epochs': {
            'cwp': cwp_checkpoint.get('epoch'),
            'baseline': baseline_checkpoint.get('epoch'),
        },
        'layers': list(range(13)),
        'layer_zero_definition': 'PatchEmbed + patch positional embedding; CWP/baseline initializer + class positional embedding; before Block 1',
        'source_results_immutable': True,
        'environment_manifests': environment,
        'initial_pretrained_load_report': initial_pretrained_report,
        'handoff_discrepancy': 'docs/CHAT_HANDOFF.md and docs/RESEARCH_PLAN_FULL.md were absent; live Git/tmux/results were used as operational truth.',
    }
    json_dump(output / 'metadata.json', metadata)
    json_dump(output / 'completion.json', {
        'status': 'complete', 'num_images': len(dataset),
        'cwp_rows': len(cwp_frame), 'created_at': timestamp(),
    })
    log(f'complete rows={len(cwp_frame)} checkpoint_sha256={before_hashes["cwp"]}')


if __name__ == '__main__':
    main()
