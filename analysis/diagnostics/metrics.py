"""Label-only diagnostics. Missing/constant observations are NaN, not zero."""

import torch
import torch.nn.functional as F


def conditional_attention(a):
    mass = a.sum(-1, keepdim=True)
    return torch.where(mass > 0, a / mass.clamp_min(torch.finfo(a.dtype).tiny), torch.full_like(a, float('nan')))


def effective_support(p):
    entropy = -torch.where(p > 0, p * p.clamp_min(1e-30).log(), torch.zeros_like(p)).sum(-1)
    result = entropy.exp() / p.shape[-1]
    return result.masked_fill(~torch.isfinite(p).all(-1), float('nan'))


def gini(p):
    x = p.sort(dim=-1).values
    n = x.shape[-1]
    rank = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return ((2 * rank - n - 1) * x).sum(-1) / (n * x.sum(-1)).clamp_min(1e-12)


def class_token_affinity(tokens, labels):
    t = F.normalize(tokens, dim=-1)
    sim = t @ t.transpose(-1, -2)
    c = sim.shape[-1]
    off = ~torch.eye(c, dtype=torch.bool, device=t.device)
    pair = labels.bool().unsqueeze(-1) & labels.bool().unsqueeze(-2) & off
    counts = pair.sum((-1, -2))
    rho = (sim * pair).sum((-1, -2)) / counts.clamp_min(1)
    per_counts = pair.sum(-1)
    per = (sim * pair).sum(-1) / per_counts.clamp_min(1)
    return {
        'rho_cc': rho.masked_fill(counts == 0, float('nan')),
        'rho_cc_per_class': per.masked_fill(per_counts == 0, float('nan')),
        'rho_cc_all_image': sim.masked_fill(~off, 0).sum((-1, -2)) / max(c * (c - 1), 1),
    }


def gwrp_weights(logits, decay):
    """Scatter the EXACT native float32 logspace weights back to spatial order.

    Use native [B,N,C] sort, including its tie ordering. No new tie policy is
    imposed on GWRP; Spearman below nevertheless handles ties correctly.
    """
    if not 0 < decay <= 1:
        raise ValueError('GWRP decay must be in (0,1]')
    x = logits.transpose(1, 2)  # preserve native sort strides as well as axis
    order = torch.sort(x, dim=1, descending=True).indices
    n = x.shape[1]
    weights = torch.logspace(0, n - 1, n, base=decay, device=x.device, dtype=torch.float32)
    weights = weights / weights.sum()
    spatial = torch.empty_like(x, dtype=torch.float32)
    spatial.scatter_(1, order, weights[None, :, None].expand_as(x))
    return spatial.transpose(1, 2)


def average_ranks(x):
    """Vectorized average ranks for ties, including uniform/one-hot maps."""
    values, order = x.sort(dim=-1)
    starts = torch.ones_like(values, dtype=torch.bool)
    starts[..., 1:] = values[..., 1:] != values[..., :-1]
    groups = starts.long().cumsum(-1) - 1
    ranks = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(x)
    sums = torch.zeros_like(x).scatter_add_(-1, groups, ranks)
    counts = torch.zeros_like(x).scatter_add_(-1, groups, torch.ones_like(x))
    tied = (sums / counts.clamp_min(1)).gather(-1, groups)
    return torch.empty_like(x).scatter_(-1, order, tied)


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    denom = ra.norm(dim=-1) * rb.norm(dim=-1)
    value = (ra * rb).sum(-1) / denom.clamp_min(1e-30)
    valid = (denom > 0) & torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
    return value.masked_fill(~valid, float('nan'))


def js_divergence(p, q):
    """Jensen-Shannon divergence in nats, bounded by log(2)."""
    m = (p + q) / 2
    def kl(a):
        return torch.where(a > 0, a * (a.clamp_min(1e-30).log() - m.clamp_min(1e-30).log()), 0).sum(-1)
    return (kl(p) + kl(q)) / 2


def divergence(p, q):
    return {'delta_rank': 1 - spearman(p, q), 'delta_js': js_divergence(p, q)}
