"""Probability statistics. Inputs are already spatial maps, not token sequences.

FP32 diagnostics avoid unreliable low-precision entropy and second moments.
These are descriptions, not universal success/failure thresholds.
"""
import itertools
import math
import numpy as np
import torch


def conditional_attention(a):
    """Condition on patch mass; zero-mass rows remain NaN, not uniform. Computed in FP32."""
    a = a.float()
    mass = a.sum(-1, keepdim=True)
    return torch.where(mass > 0, a / mass.clamp_min(torch.finfo(a.dtype).tiny), torch.full_like(a, float('nan')))


def effective_support(p):
    """Kappa=exp(H)/N in [1/N,1]; larger means broader support, not necessarily better. The historical normalized-entropy table is NOT kappa. Computed in FP32."""
    p = p.float()
    entropy = -torch.where(p > 0, p * p.clamp_min(1e-30).log(), torch.zeros_like(p)).sum(-1)
    result = entropy.exp() / p.shape[-1]
    return result.masked_fill(~torch.isfinite(p).all(-1), float('nan'))


def gini(p):
    """Sorted concentration in [0,(N-1)/N]; higher is more concentrated. No universal gate. Computed in FP32."""
    p = p.float()
    x = p.sort(dim=-1).values
    n = x.shape[-1]
    rank = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return ((2 * rank - n - 1) * x).sum(-1) / (n * x.sum(-1)).clamp_min(1e-12)


def normalized_entropy(weights):
    """H/log(N), [0,1] for probabilities and N>1; high means diffuse. FP32.

    Source table values 0.2938 / 0.9719 describe this metric, not kappa.
    """
    weights = weights.float()
    return -(weights * weights.clamp_min(1e-30).log()).sum(-1) / math.log(weights.shape[-1])


def weight_stats(w, labels, fallback=None):
    """Equal positive-class/pair weighting INSIDE image; image is sampling unit."""
    w = w.float()
    rows = []
    n = w.shape[-1]
    k = int(np.ceil(.1 * n))
    for i in range(len(w)):
        a = w[i, labels[i].bool()]
        p = a.clamp_min(1e-12)
        masks = torch.zeros_like(a, dtype=torch.bool)
        masks.scatter_(1, a.topk(k, -1).indices, True)
        pair_scores = []
        for c, d in itertools.combinations(range(len(a)), 2):
            pair_scores.append((masks[c] & masks[d]).sum() / (masks[c] | masks[d]).sum())
        jaccard = torch.stack(pair_scores).mean() if pair_scores else a.new_tensor(float('nan'))
        fail = a.new_tensor(0.) if fallback is None else fallback[i, labels[i].bool()].float().mean()
        rows.append(torch.stack([-(p * p.log()).sum(-1).mean() / np.log(n), a.amax(-1).mean(),
                                 (n * a.amin(-1)).mean(), jaccard, fail]))
    return torch.stack(rows)
