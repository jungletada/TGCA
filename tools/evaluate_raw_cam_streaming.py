"""Dataset-global raw CAM metrics with constant memory (VOC or COCO)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.evaluate_cam_threshold_grid import (
    confusion_metrics, image_threshold_confusions, load_cam_winner, cam_payload_winner,
    sha256_file, threshold_grid,
)


def evaluate(cam_dir, mask_dir, id_list, num_classes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = id_list.read_text().splitlines()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate image IDs')
    thresholds = threshold_grid(0, 0.59, 0.01)
    total = np.zeros((len(thresholds), num_classes, num_classes), np.int64)
    for i, name in enumerate(ids):
        with Image.open(mask_dir / f'{name}.png') as image:
            target = np.asarray(image)
        scores, classes = load_cam_winner(
            cam_dir / f'{name}.npy', target.shape, num_classes)
        total += image_threshold_confusions(
            scores, classes, target, thresholds, num_classes)
        if i % 500 == 0:
            print(f'raw CAM evaluation {i + 1}/{len(ids)}', flush=True)
    return save_summary(total, thresholds, len(ids), num_classes, output_dir, cam_dir, mask_dir, id_list)


def save_summary(total, thresholds, num_images, num_classes, output_dir, cam_dir, mask_dir, id_list):
    metrics = confusion_metrics(total)
    rows = []
    for i, t in enumerate(thresholds):
        row = {'threshold': float(t)}
        row.update({k: float(v[i]) if np.isfinite(v[i]) else None
                    for k, v in metrics.items() if v.ndim == 1})
        rows.append(row)
    with (output_dir / 'threshold_curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    best = int(np.nanargmax(metrics['mean_iou']))
    summary = {'num_images': num_images, 'num_classes_including_bg': num_classes,
               'fixed': rows[45], 'best': rows[best],
               'cam_dir': str(cam_dir), 'mask_dir': str(mask_dir),
               'id_list': str(id_list), 'id_list_sha256': sha256_file(id_list),
               'protocol': 'dataset-global confusion; native CAM; no CRF; ties favor BG',
               'units': 'fractions, not percentages'}
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(output_dir / 'aggregate_confusions.npz',
                        thresholds=thresholds, confusion=total)
    (output_dir / 'EVAL_COMPLETE').write_text('complete\n')
    return summary


class OnlineCamEvaluator:
    """Accumulate native per-image payloads without persisting large CAM arrays."""
    def __init__(self, mask_dir, id_list, num_classes, output_dir):
        self.mask_dir, self.id_list, self.output_dir = map(Path, (mask_dir, id_list, output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.ids = self.id_list.read_text().splitlines()
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError('Empty or duplicate image IDs')
        self.num_classes = num_classes
        self.thresholds = threshold_grid(0, .59, .01)
        self.total = np.zeros((len(self.thresholds), num_classes, num_classes), np.int64)
        self.processed = []

    def update(self, name, payload):
        if name != self.ids[len(self.processed)]:
            raise ValueError('Online CAM image order/coverage mismatch')
        with Image.open(self.mask_dir / f'{name}.png') as image:
            target = np.asarray(image)
        scores, classes = cam_payload_winner(payload, target.shape, self.num_classes)
        self.total += image_threshold_confusions(scores, classes, target, self.thresholds, self.num_classes)
        self.processed.append(name)

    def finish(self):
        if self.processed != self.ids:
            raise ValueError('Incomplete online CAM coverage')
        return save_summary(self.total, self.thresholds, len(self.ids), self.num_classes,
                            self.output_dir, '<online: native CAMs not persisted>', self.mask_dir, self.id_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('cam-dir', 'mask-dir', 'id-list', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--num-classes', type=int, choices=(21, 81), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))
