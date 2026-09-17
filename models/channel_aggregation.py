"""Shared positive channel readout; disabled path is the original mean."""
import math
import torch
from torch import nn


class ChannelAggregator(nn.Module):
    def __init__(self, dim, enabled=False):
        super().__init__()
        self.enabled = bool(enabled)
        self.theta = nn.Parameter(torch.zeros(dim)) if enabled else None

    def weights(self):
        return None if not self.enabled else self.theta.softmax(dim=0)

    def forward(self, tokens):
        baseline = tokens.mean(dim=-1)
        if not self.enabled:
            return baseline
        weights = self.weights().to(tokens.dtype)
        # Algebraically sum(a * tokens), but theta=0 gives EXACT native mean
        # rather than a different multiply/sum rounding path. Gradients to theta
        # are preserved; no data-dependent branch on theta is used.
        return baseline + (tokens * (weights - weights.new_tensor(1 / tokens.shape[-1]))).sum(-1)

    @torch.no_grad()
    def diagnostics(self):
        if not self.enabled:
            return {}
        w = self.weights().float()
        return dict(channel_entropy=float(-(w*w.clamp_min(1e-30).log()).sum()/math.log(w.numel())),
                    channel_max_min_ratio=float(w.max()/w.min()),
                    channel_l1_uniform=float((w-1/w.numel()).abs().sum()))


def configure_channel_optimizer(optimizer, model, multiplier=1.):
    """Separate theta's zero-decay group; no changes when A1 is disabled."""
    aggregator = model.channel_aggregator
    if not aggregator.enabled:
        if multiplier != 1.:
            raise ValueError('Channel LR multiplier requires channel aggregation')
        return
    theta = aggregator.theta
    for group in optimizer.param_groups:
        if any(p is theta for p in group['params']):
            assert group['weight_decay'] == 0
            group['params'] = [p for p in group['params'] if p is not theta]
            options = {k:v for k,v in group.items() if k!='params'}
            optimizer.add_param_group(dict(options, params=[theta], channel_lr_multiplier=multiplier))
            return
    raise ValueError('theta missing from optimizer')


def scale_channel_lr(optimizer):
    """Call once AFTER each native scheduler update (including creation).

    Scheduler base_values remain unscaled, so warmup/min-LR/cosine values are
    multiplied exactly once; backbone groups and their schedule are unchanged.
    """
    for group in optimizer.param_groups:
        if 'channel_lr_multiplier' in group:
            group['lr'] *= group['channel_lr_multiplier']
