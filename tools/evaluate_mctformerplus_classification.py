#!/usr/bin/env python3
"""Evaluate frozen MCTformer+ class-token and patch-GWRP classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets_cam import VOC12Dataset, build_transform  # noqa: E402
from models.mctformer_plus import (  # noqa: E402
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
)


CLASS_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--model', choices=(
            'mctformerplus_tiny', 'mctformerplus', 'mctformerplus_base'),
        required=True,
    )
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--list-path', type=Path, required=True)
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2027)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _average_precision(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.sum() == 0:
        return np.nan
    return float(average_precision_score(labels, scores))


def classification_metrics(labels, scores):
    per_class = np.asarray([
        _average_precision(labels[:, class_id], scores[:, class_id])
        for class_id in range(labels.shape[1])
    ])
    per_image = np.asarray([
        _average_precision(labels[index], scores[index])
        for index in range(labels.shape[0])
    ])
    return {
        'macro_class_ap': float(np.nanmean(per_class)),
        'micro_ap': _average_precision(labels.reshape(-1), scores.reshape(-1)),
        'legacy_mean_image_ap': float(np.nanmean(per_image)),
        'per_class_ap': per_class,
        'per_image_ap': per_image,
    }


def _weighted_average_precision(labels, scores, weights):
    """Exact average precision for many bootstrap weight rows at once."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-scores, kind='mergesort')
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[:, order]
    cumulative_positive = np.cumsum(
        sorted_weights * sorted_labels[None], axis=1
    )
    cumulative_total = np.cumsum(sorted_weights, axis=1)
    group_ends = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    cumulative_positive = cumulative_positive[:, group_ends]
    cumulative_total = cumulative_total[:, group_ends]
    group_positive = np.diff(
        np.concatenate((
            np.zeros((len(weights), 1), dtype=np.float64),
            cumulative_positive,
        ), axis=1),
        axis=1,
    )
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total != 0,
    )
    total_positive = cumulative_positive[:, -1]
    return np.divide(
        (precision * group_positive).sum(axis=1),
        total_positive,
        out=np.full(len(weights), np.nan, dtype=np.float64),
        where=total_positive != 0,
    )


def bootstrap_metrics(labels, branch_scores, resamples, seed, batch_size=100):
    if resamples < 1:
        return {}, {}
    generator = np.random.default_rng(seed)
    names = tuple(branch_scores)
    samples = {
        f'{branch}_{metric}': np.empty(resamples, dtype=np.float64)
        for branch in names
        for metric in ('macro_class_ap', 'micro_ap', 'legacy_mean_image_ap')
    }
    probabilities = np.full(len(labels), 1.0 / len(labels))
    per_image_ap = {
        branch: np.asarray([
            _average_precision(labels[index], scores[index])
            for index in range(len(labels))
        ])
        for branch, scores in branch_scores.items()
    }
    for begin in range(0, resamples, batch_size):
        end = min(begin + batch_size, resamples)
        weights = generator.multinomial(
            len(labels), probabilities, size=end - begin
        ).astype(np.float64)
        for branch, scores in branch_scores.items():
            per_class = np.stack([
                _weighted_average_precision(
                    labels[:, class_id], scores[:, class_id], weights
                )
                for class_id in range(labels.shape[1])
            ], axis=1)
            samples[f'{branch}_macro_class_ap'][begin:end] = np.nanmean(
                per_class, axis=1
            )
            repeated_weights = np.repeat(weights, labels.shape[1], axis=1)
            samples[f'{branch}_micro_ap'][begin:end] = (
                _weighted_average_precision(
                    labels.reshape(-1), scores.reshape(-1), repeated_weights
                )
            )
            samples[f'{branch}_legacy_mean_image_ap'][begin:end] = (
                weights @ per_image_ap[branch] / len(labels)
            )
    intervals = {
        name: [float(value) for value in np.quantile(array, (0.025, 0.975))]
        for name, array in samples.items()
    }
    return intervals, samples


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    if args.input_size < 1 or args.input_size % 16:
        raise ValueError('--input-size must be a positive multiple of 16')
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0:
        raise ValueError('batch size must be positive; workers/limit non-negative')
    if args.bootstrap_resamples < 0:
        raise ValueError('--bootstrap-resamples must be non-negative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, args.model
    )
    attention = checkpoint.get('attention_normalization', {})
    bcss = checkpoint.get('bcss', {'variant': 'e0'})
    psl = checkpoint.get('psl', {'variant': 'baseline'})
    cti = checkpoint.get('cti_bgt', {'enabled': False})
    if attention.get('mode', 'vanilla') != 'vanilla':
        raise ValueError('Width-scaling classification requires vanilla attention')
    if bcss.get('variant', 'e0') != 'e0':
        raise ValueError('Width-scaling classification requires BCSS E0')
    if psl.get('variant', 'baseline') != 'baseline':
        raise ValueError('Width-scaling classification requires PSL baseline')
    if cti.get('enabled', False):
        raise ValueError('Width-scaling classification requires CTI-BGT disabled')

    model = build_mctformerplus(
        resolution['variant'],
        cam=False,
        num_classes=20,
        input_size=args.input_size,
        attention_normalization='vanilla',
        attention_gamma=1.0,
        bcss_variant='e0',
        psl_variant='baseline',
        cti_bgt=False,
    )
    state = checkpoint.get('model', checkpoint)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f'Unexpected strict-load result: {incompatibility}')
    model.to(device).eval()

    transform_args = argparse.Namespace(input_size=args.input_size)
    dataset = VOC12Dataset(
        voc12_root=str(args.voc_root),
        list_path=str(args.list_path),
        transform=build_transform(False, False, transform_args),
    )
    image_ids = list(dataset.img_name_list)
    if args.limit:
        image_ids = image_ids[:args.limit]
        dataset = torch.utils.data.Subset(dataset, range(len(image_ids)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    labels_parts = []
    class_parts = []
    patch_parts = []
    class_loss_sum = 0.0
    patch_loss_sum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            outputs = model(images.to(device, non_blocking=True))
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
                raise RuntimeError('Unexpected MCTformer+ classification output')
            class_logits = outputs[0]
            patch_logits = outputs[2]
            if class_logits.shape != patch_logits.shape or class_logits.shape[1] != 20:
                raise RuntimeError(
                    f'Invalid output shapes: {class_logits.shape}, {patch_logits.shape}'
                )
            labels_parts.append(labels.numpy().astype(np.uint8, copy=False))
            target = labels.to(device=device, dtype=class_logits.dtype)
            class_loss_sum += float(
                F.multilabel_soft_margin_loss(class_logits, target).item()
                * len(labels)
            )
            patch_loss_sum += float(
                F.multilabel_soft_margin_loss(patch_logits, target).item()
                * len(labels)
            )
            class_parts.append(torch.sigmoid(class_logits).cpu().numpy())
            patch_parts.append(torch.sigmoid(patch_logits).cpu().numpy())
            processed = min((batch_index + 1) * args.batch_size, len(image_ids))
            if processed % 100 == 0 or processed == len(image_ids):
                print(f'images={processed}/{len(image_ids)}', flush=True)

    labels = np.concatenate(labels_parts).astype(np.uint8, copy=False)
    class_scores = np.concatenate(class_parts).astype(np.float32, copy=False)
    patch_scores = np.concatenate(patch_parts).astype(np.float32, copy=False)
    if labels.shape != class_scores.shape or labels.shape != patch_scores.shape:
        raise RuntimeError('Prediction/label shape mismatch')
    if not np.isfinite(class_scores).all() or not np.isfinite(patch_scores).all():
        raise RuntimeError('Non-finite classification score')

    branch_scores = {
        'class_token': class_scores,
        'patch_gwrp': patch_scores,
    }
    point = {
        branch: classification_metrics(labels, scores)
        for branch, scores in branch_scores.items()
    }
    intervals, bootstrap = bootstrap_metrics(
        labels, branch_scores, args.bootstrap_resamples, args.bootstrap_seed
    )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'command.txt').write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    np.savez_compressed(
        args.output_dir / 'classification_predictions.npz',
        image_ids=np.asarray(image_ids),
        labels=labels,
        class_token_scores=class_scores,
        patch_gwrp_scores=patch_scores,
    )
    if bootstrap:
        np.savez_compressed(
            args.output_dir / 'bootstrap_samples.npz', **bootstrap
        )

    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_rows.append({
            'class_id': class_id,
            'class_name': class_name,
            'positive_images': int(labels[:, class_id].sum()),
            'class_token_ap_fraction': point[
                'class_token'
            ]['per_class_ap'][class_id],
            'class_token_ap_percent': 100.0 * point[
                'class_token'
            ]['per_class_ap'][class_id],
            'patch_gwrp_ap_fraction': point[
                'patch_gwrp'
            ]['per_class_ap'][class_id],
            'patch_gwrp_ap_percent': 100.0 * point[
                'patch_gwrp'
            ]['per_class_ap'][class_id],
        })
    with (args.output_dir / 'classification_per_class.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    per_image_rows = []
    for index, image_id in enumerate(image_ids):
        per_image_rows.append({
            'image_id': image_id,
            'positive_class_count': int(labels[index].sum()),
            'class_token_image_ap_fraction': point[
                'class_token'
            ]['per_image_ap'][index],
            'class_token_image_ap_percent': 100.0 * point[
                'class_token'
            ]['per_image_ap'][index],
            'patch_gwrp_image_ap_fraction': point[
                'patch_gwrp'
            ]['per_image_ap'][index],
            'patch_gwrp_image_ap_percent': 100.0 * point[
                'patch_gwrp'
            ]['per_image_ap'][index],
        })
    with (args.output_dir / 'classification_per_image.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)

    def scalar_metrics(values, scale):
        return {
            key: scale * float(values[key])
            for key in ('macro_class_ap', 'micro_ap', 'legacy_mean_image_ap')
        }

    metrics = {
        'schema_version': 1,
        'dataset': 'PASCAL VOC 2012',
        'split': args.list_path.stem,
        'num_images': len(image_ids),
        'class_names': list(CLASS_NAMES),
        'model_spec': model_spec_from_instance(model),
        'variant_resolution': resolution,
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': sha256_file(args.checkpoint),
            'epoch': checkpoint.get('epoch'),
            'policy': 'final' if args.checkpoint.name.endswith('_final.pth') else 'unspecified',
        },
        'protocol': {
            'input_size': args.input_size,
            'transform': 'datasets_cam.build_transform(False, False)',
            'single_scale': True,
            'horizontal_flip': False,
            'amp': False,
            'score_transform': 'sigmoid',
            'macro_definition': 'mean of 20 dataset-level one-vs-rest class AP values',
            'micro_definition': 'AP over flattened image-class pairs',
            'legacy_definition': 'mean AP over the 20-class vector within each image',
        },
        'metrics_fraction': {
            branch: scalar_metrics(values, 1.0)
            for branch, values in point.items()
        },
        'metrics_percent': {
            branch: scalar_metrics(values, 100.0)
            for branch, values in point.items()
        },
        'classification_loss': {
            'class_token_multilabel_soft_margin_mean': (
                class_loss_sum / len(image_ids)
            ),
            'patch_gwrp_multilabel_soft_margin_mean': (
                patch_loss_sum / len(image_ids)
            ),
            'training_objective_sum_mean': (
                (class_loss_sum + patch_loss_sum) / len(image_ids)
            ),
        },
        'bootstrap': {
            'unit': 'image',
            'resamples': args.bootstrap_resamples,
            'seed': args.bootstrap_seed,
            'ci95_fraction': intervals,
            'ci95_percent': {
                name: [100.0 * value for value in bounds]
                for name, bounds in intervals.items()
            },
            'training_seed_uncertainty_included': False,
        },
        'provenance': {
            'voc_root': str(args.voc_root.resolve()),
            'list_path': str(args.list_path.resolve()),
            'list_sha256': sha256_file(args.list_path),
            'labels_path': str((args.voc_root / 'ImageLabel/cls_labels.npy').resolve()),
            'labels_sha256': sha256_file(
                args.voc_root / 'ImageLabel/cls_labels.npy'
            ),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': str(device),
            'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
        },
        'elapsed_seconds': time.perf_counter() - started,
        'finite': True,
    }
    (args.output_dir / 'classification_metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'CLASSIFICATION_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
