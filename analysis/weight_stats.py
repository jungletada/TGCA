"""Compact spatial-weight statistics; positive classes averaged within image."""
import math
import torch


def normalized_entropy(weights):
    return -(weights * weights.clamp_min(1e-30).log()).sum(-1) / math.log(weights.shape[-1])


def all_product_weights(records, num_classes):
    responses = torch.stack([a[:, :, :num_classes, num_classes:] for a in records]).float()
    per_layer = responses.mean(2)
    return per_layer.clamp_min(torch.finfo(per_layer.dtype).tiny).log().sum(0).softmax(-1)


def positive_image_mean(values, labels):
    mask = labels.bool()
    count = mask.sum(-1)
    mean = (values*mask).sum(-1)/count.clamp_min(1)
    return mean.masked_fill(count==0, float('nan'))
