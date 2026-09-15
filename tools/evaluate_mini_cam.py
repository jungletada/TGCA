"""Thin single-scale, no-flip diagnostic CAM wrapper; not formal evaluation."""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from analysis.diagnostics.probe_set import mask_transform
from models.mctformer_plus import MCTformerPlusCam
from tools.cam_utils import normalize_cam
from tools.evaluate_cam_threshold_grid import cam_payload_winner, image_threshold_confusions, confusion_metrics, threshold_grid


class _NativeReadout:
    """Read-only view of the SAME classifier/backbone; no second model."""
    n_layers = 3

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)


def native_cam(model, maps, records):
    # Calling the official model method preserves its exact ReLU/last3/sqrt/P2P.
    _, means = MCTformerPlusCam._stack_attention_records(model, records)
    return MCTformerPlusCam.get_cam(_NativeReadout(model), maps, means)


def evaluate_mini_cam(model, loader, mask_dir, resolution, device):
    from pathlib import Path
    thresholds = threshold_grid(0, .59, .01)
    classes = model.num_classes + 1
    confusion = np.zeros((len(thresholds), classes, classes), dtype=np.int64)
    seen = []
    for images, labels, names in loader:
        images = images.to(device, dtype=torch.float32)
        _, patches, records, _ = model.forward_features(images)
        h = images.shape[-2] // model.patch_embed.patch_size[0]
        w = images.shape[-1] // model.patch_embed.patch_size[1]
        maps = model.head(patches.reshape(len(images), h, w, -1).permute(0, 3, 1, 2).contiguous())
        cams = native_cam(model, maps, records)
        cams = F.interpolate(cams, (resolution, resolution), mode='bilinear', align_corners=False)
        for cam, label, name in zip(cams, labels, names):
            with Image.open(Path(mask_dir) / f'{name}.png') as image:
                target = mask_transform(image, resolution)
            positive = torch.nonzero(label > 0).flatten()
            selected = normalize_cam(cam[positive.to(cam.device)]) if len(positive) else cam[:0]
            payload = {int(c): v.cpu().numpy() for c, v in zip(positive, selected)}
            scores, winners = cam_payload_winner(payload, target.shape, classes)
            confusion += image_threshold_confusions(scores, winners, target, thresholds, classes)
            seen.append(name)
    metrics = confusion_metrics(confusion)
    best = int(np.nanargmax(metrics['mean_iou']))
    values = {'mini_cam_miou_fixed': float(metrics['mean_iou'][45]),
              'mini_cam_miou_best': float(metrics['mean_iou'][best]),
              'mini_cam_best_threshold': float(thresholds[best]),
              'mini_cam_fg_precision': float(metrics['semantic_foreground_precision'][45]),
              'mini_cam_fg_recall': float(metrics['semantic_foreground_recall'][45])}
    return values, confusion, seen
