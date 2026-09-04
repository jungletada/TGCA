#!/usr/bin/env python3
"""Exhaustively evaluate a fixed raw-CAM threshold grid on VOC masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


NUM_CLASSES = 21
CLASS_NAMES = (
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-dir', type=Path, required=True)
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--id-list', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold-start', type=float, default=0.0)
    parser.add_argument('--threshold-stop', type=float, default=0.59)
    parser.add_argument('--threshold-step', type=float, default=0.01)
    parser.add_argument('--fixed-threshold', type=float, default=0.45)
    parser.add_argument('--calibrated-threshold', type=float)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_grid(start, stop, step):
    if step <= 0 or start < 0 or stop > 1 or stop < start:
        raise ValueError('Invalid threshold interval')
    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if not np.isclose(values[-1], stop, rtol=0, atol=1e-10):
        raise ValueError('Threshold stop is not on the requested grid')
    return np.round(values, decimals=10)


def load_cam_winner(path, empty_spatial_shape=None):
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a CAM dictionary: {path}')
    if not payload:
        if empty_spatial_shape is None:
            raise ValueError(
                f'Empty CAM requires an explicit spatial shape: {path}'
            )
        shape = tuple(int(value) for value in empty_spatial_shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f'Invalid empty-CAM spatial shape: {shape}')
        # The native validation path can reject every foreground class through
        # its class/patch and flip intersection.  Such an image predicts only
        # background at every non-negative threshold.
        return (
            np.full(shape, -np.inf, dtype=np.float32),
            np.ones(shape, dtype=np.int64),
        )
    items = list(payload.items())
    class_ids = np.asarray([int(key) + 1 for key, _ in items], dtype=np.int64)
    if len(np.unique(class_ids)) != len(class_ids):
        raise ValueError(f'Duplicate CAM class IDs: {path}')
    if np.any((class_ids < 1) | (class_ids >= NUM_CLASSES)):
        raise ValueError(f'CAM class ID outside VOC range: {path}')
    cams = np.stack([np.asarray(value, dtype=np.float32) for _, value in items])
    if cams.ndim != 3 or not np.isfinite(cams).all():
        raise ValueError(f'Invalid CAM array in {path}: {cams.shape}')
    winner = np.argmax(cams, axis=0)
    scores = np.take_along_axis(cams, winner[None], axis=0)[0]
    classes = class_ids[winner]
    return scores, classes


def image_threshold_confusions(scores, classes, target, thresholds):
    if scores.shape != classes.shape or scores.shape != target.shape:
        raise ValueError(
            f'CAM/target shape mismatch: {scores.shape}, {target.shape}'
        )
    valid = target != 255
    target = target[valid].astype(np.int64, copy=False)
    scores = scores[valid]
    classes = classes[valid]
    if np.any((target < 0) | (target >= NUM_CLASSES)):
        raise ValueError('VOC target contains an invalid non-void label')
    # k is the count of thresholds strictly below the score.  Therefore a
    # pixel is foreground at grid index i exactly when k > i, matching score>t.
    passed_count = np.searchsorted(thresholds, scores, side='left')
    pair = target * (NUM_CLASSES - 1) + (classes - 1)
    width = len(thresholds) + 1
    histogram = np.bincount(
        pair * width + passed_count,
        minlength=NUM_CLASSES * (NUM_CLASSES - 1) * width,
    ).reshape(NUM_CLASSES, NUM_CLASSES - 1, width)
    foreground = np.cumsum(histogram[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    confusion = np.zeros(
        (len(thresholds), NUM_CLASSES, NUM_CLASSES), dtype=np.int32
    )
    confusion[:, :, 1:] = foreground.transpose(2, 0, 1).astype(np.int32)
    target_counts = np.bincount(target, minlength=NUM_CLASSES)
    confusion[:, :, 0] = (
        target_counts[None] - confusion[:, :, 1:].sum(axis=2)
    ).astype(np.int32)
    if np.any(confusion < 0):
        raise RuntimeError('Negative confusion count produced')
    if not np.all(confusion.sum(axis=(1, 2)) == valid.sum()):
        raise RuntimeError('Threshold confusion does not conserve valid pixels')
    return confusion


def confusion_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    target = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    union = target + predicted - true_positive
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / union
        class_precision = true_positive / predicted
        class_recall = true_positive / target
    semantic_tp = true_positive[..., 1:].sum(axis=-1)
    pred_fg = predicted[..., 1:].sum(axis=-1)
    target_fg = target[..., 1:].sum(axis=-1)
    foreground_overlap = confusion[..., 1:, 1:].sum(axis=(-2, -1))
    false_positive_background = confusion[..., 0, 1:].sum(axis=-1)
    target_background = target[..., 0]
    valid_pixels = confusion.sum(axis=(-2, -1))
    correct_pixels = true_positive.sum(axis=-1)

    def divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full(np.broadcast_shapes(
                np.shape(numerator), np.shape(denominator)), np.nan),
            where=np.asarray(denominator) != 0,
        )

    return {
        'mean_iou': np.nanmean(iou, axis=-1),
        'foreground_mean_iou': np.nanmean(iou[..., 1:], axis=-1),
        'semantic_foreground_precision': divide(semantic_tp, pred_fg),
        'semantic_foreground_recall': divide(semantic_tp, target_fg),
        'binary_foreground_precision': divide(foreground_overlap, pred_fg),
        'binary_foreground_recall': divide(foreground_overlap, target_fg),
        'background_false_positive_rate': divide(
            false_positive_background, target_background
        ),
        'pixel_accuracy': divide(correct_pixels, valid_pixels),
        'class_iou': iou,
        'class_precision': class_precision,
        'class_recall': class_recall,
    }


def _grid_index(thresholds, value, name):
    indices = np.flatnonzero(np.isclose(thresholds, value, rtol=0, atol=1e-9))
    if len(indices) != 1:
        raise ValueError(f'{name}={value} is not exactly on the threshold grid')
    return int(indices[0])


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    if args.limit < 0:
        raise ValueError('--limit must be non-negative')
    thresholds = threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    fixed_index = _grid_index(
        thresholds, args.fixed_threshold, 'fixed threshold'
    )
    calibrated_index = None
    if args.calibrated_threshold is not None:
        calibrated_index = _grid_index(
            thresholds, args.calibrated_threshold, 'calibrated threshold'
        )

    image_ids = [
        line.strip() for line in args.id_list.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise ValueError('ID list is empty')
    missing_cams = [
        image_id for image_id in image_ids
        if not (args.cam_dir / f'{image_id}.npy').is_file()
    ]
    missing_masks = [
        image_id for image_id in image_ids
        if not (args.voc_root / 'SegmentationClass' / f'{image_id}.png').is_file()
    ]
    if missing_cams or missing_masks:
        raise FileNotFoundError(
            f'Missing CAMs={missing_cams[:5]} ({len(missing_cams)} total), '
            f'masks={missing_masks[:5]} ({len(missing_masks)} total)'
        )

    started = time.perf_counter()
    per_image = np.empty(
        (len(image_ids), len(thresholds), NUM_CLASSES, NUM_CLASSES),
        dtype=np.int32,
    )
    empty_cam_ids = []
    for index, image_id in enumerate(image_ids):
        target = np.asarray(
            Image.open(
                args.voc_root / 'SegmentationClass' / f'{image_id}.png'
            ),
            dtype=np.uint8,
        )
        cam_path = args.cam_dir / f'{image_id}.npy'
        scores, classes = load_cam_winner(
            cam_path, empty_spatial_shape=target.shape
        )
        if not np.isfinite(scores).any():
            empty_cam_ids.append(image_id)
        per_image[index] = image_threshold_confusions(
            scores, classes, target, thresholds
        )
        if (index + 1) % 100 == 0 or index + 1 == len(image_ids):
            print(f'images={index + 1}/{len(image_ids)}', flush=True)

    total = per_image.astype(np.int64).sum(axis=0)
    curve = confusion_metrics(total)
    best_index = int(np.nanargmax(curve['mean_iou']))
    best_threshold = float(thresholds[best_index])
    best_miou = float(curve['mean_iou'][best_index])
    plateau = curve['mean_iou'] >= best_miou - 0.005
    plateau_thresholds = thresholds[plateau]
    threshold_span = float(thresholds[-1] - thresholds[0])
    curve_auc = float(np.trapz(curve['mean_iou'], thresholds))
    normalized_curve_auc = (
        curve_auc / threshold_span if threshold_span > 0 else np.nan
    )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'command.txt').write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    np.savez_compressed(
        args.output_dir / 'per_image_confusions.npz',
        image_ids=np.asarray(image_ids),
        thresholds=thresholds.astype(np.float32),
        confusions=per_image,
    )
    np.savetxt(
        args.output_dir / 'confusion_fixed_t045.csv',
        total[fixed_index], fmt='%d', delimiter=',',
    )
    if calibrated_index is not None:
        np.savetxt(
            args.output_dir / 'confusion_calibrated.csv',
            total[calibrated_index], fmt='%d', delimiter=',',
        )

    scalar_names = (
        'mean_iou', 'foreground_mean_iou',
        'semantic_foreground_precision', 'semantic_foreground_recall',
        'binary_foreground_precision', 'binary_foreground_recall',
        'background_false_positive_rate', 'pixel_accuracy',
    )
    curve_rows = []
    for index, threshold in enumerate(thresholds):
        row = {'threshold': float(threshold)}
        row.update({
            f'{name}_percent': _finite_or_none(
                100.0 * curve[name][index]
            )
            for name in scalar_names
        })
        curve_rows.append(row)
    with (args.output_dir / 'threshold_curve.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    (args.output_dir / 'threshold_curve.json').write_text(
        json.dumps(
            {
                'threshold_rule': (
                    'background wins ties; foreground iff max CAM > threshold'
                ),
                'rows': curve_rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + '\n',
        encoding='utf-8',
    )

    selected_indices = {'fixed_0.45': fixed_index, 'oracle': best_index}
    if calibrated_index is not None:
        selected_indices['small_calibrated'] = calibrated_index
    selected_metrics = {
        label: {
            'threshold': float(thresholds[index]),
            **{
                f'{name}_percent': _finite_or_none(
                    100.0 * curve[name][index]
                )
                for name in scalar_names
            },
        }
        for label, index in selected_indices.items()
    }

    class_rows = []
    for label, index in selected_indices.items():
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_rows.append({
                'selection': label,
                'threshold': float(thresholds[index]),
                'class_id': class_id,
                'class_name': class_name,
                'iou_percent': _finite_or_none(
                    100.0 * curve['class_iou'][index, class_id]
                ),
                'precision_percent': _finite_or_none(
                    100.0 * curve['class_precision'][index, class_id]
                ),
                'recall_percent': _finite_or_none(
                    100.0 * curve['class_recall'][index, class_id]
                ),
            })
    with (args.output_dir / 'metrics_by_class.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    split = 'train' if args.id_list.stem.startswith('train') else (
        'val' if args.id_list.stem.startswith('val') else args.id_list.stem
    )
    metrics = {
        'schema_version': 1,
        'dataset': 'PASCAL VOC 2012',
        'split': split,
        'num_images': len(image_ids),
        'empty_cam_images': {
            'count': len(empty_cam_ids),
            'ids': empty_cam_ids,
            'evaluation_rule': 'all-background prediction at every threshold',
        },
        'threshold_grid': [float(value) for value in thresholds],
        'threshold_rule': 'background wins ties; foreground iff max CAM > threshold',
        'curve_evaluation': 'exhaustive; no early stopping',
        'oracle_selection': {
            'role': 'sensitivity diagnostic only',
            'tie_break': 'lowest threshold',
            'threshold': best_threshold,
            'mean_iou_percent': 100.0 * best_miou,
        },
        'selected_metrics': selected_metrics,
        'threshold_curve_auc': curve_auc,
        'threshold_curve_auc_normalized': normalized_curve_auc,
        'plateau': {
            'definition': 'thresholds within 0.5 percentage point of oracle mIoU',
            'minimum_threshold': float(plateau_thresholds.min()),
            'maximum_threshold': float(plateau_thresholds.max()),
            'width': float(plateau_thresholds.max() - plateau_thresholds.min()),
            'grid_point_count': int(plateau.sum()),
        },
        'per_image_confusions': {
            'path': str((args.output_dir / 'per_image_confusions.npz').resolve()),
            'shape': list(per_image.shape),
            'dtype': str(per_image.dtype),
            'bootstrap_cluster_unit': 'image',
        },
        'provenance': {
            'cam_dir': str(args.cam_dir.resolve()),
            'cam_file_count': len(list(args.cam_dir.glob('*.npy'))),
            'voc_root': str(args.voc_root.resolve()),
            'id_list': str(args.id_list.resolve()),
            'id_list_sha256': sha256_file(args.id_list),
        },
        'elapsed_seconds': time.perf_counter() - started,
        'undefined_scalar_grid_points': {
            name: int((~np.isfinite(curve[name])).sum())
            for name in scalar_names
        },
        'undefined_rule': (
            'undefined denominators are null; they are never imputed or '
            'treated as numerical failures'
        ),
        'finite': bool(all(
            np.isfinite(curve[name]).all()
            for name in (
                'mean_iou', 'foreground_mean_iou',
                'semantic_foreground_recall', 'binary_foreground_recall',
                'background_false_positive_rate', 'pixel_accuracy',
            )
        )),
    }
    if not metrics['finite']:
        raise RuntimeError('Non-finite required scalar metric in threshold curve')
    (args.output_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'best_threshold.json').write_text(
        json.dumps(
            {
                'oracle_selection': metrics['oracle_selection'],
                'plateau': metrics['plateau'],
                'role': 'threshold-sensitivity diagnostic only',
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'THRESHOLD_EVALUATION_COMPLETE').write_text(
        'complete\n', encoding='utf-8'
    )
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
