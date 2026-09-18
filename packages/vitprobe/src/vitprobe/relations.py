"""Positive-query relations; aggregate within image before image bootstrap.

FP32 avoids fragile low-precision norms. Missing pairs remain NaN.
"""
import torch
import torch.nn.functional as F


def class_token_affinity(tokens, labels):
    """Positive-pair cosine [-1,1]; high means similar, not proof of semantic leakage."""
    tokens = tokens.float()
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


@torch.no_grad()
def symptom_overlap(ratio, recv, topk_frac=.01):
    """Top-support Jaccard [0,1], higher means greater symptom overlap.

    At N=784, k=round(.01*N)=8: random expected Jaccard is about 0.00547,
    not .01 (source decision.md). No universal causal interpretation. FP32.
    """
    ratio, recv = ratio.float(), recv.float()
    k = max(1, round(topk_frac * ratio.shape[-1]))
    a, b = torch.zeros_like(ratio, dtype=torch.bool), torch.zeros_like(recv, dtype=torch.bool)
    a.scatter_(1, ratio.topk(k, dim=-1).indices, True)
    b.scatter_(1, recv.topk(k, dim=-1).indices, True)
    return (a & b).sum(-1).float() / (a | b).sum(-1).clamp_min(1)


def head_statistics(a, labels):
    """a:[cells,B,C,N], returns [B,cells,3] kappa/Gini/gamma.

    Positive-class averaging within each image. Gamma is missing, NOT zero,
    for images with fewer than two positives or zero-norm positive maps.
    """
    a = a.float()
    mass = a.sum(-1, keepdim=True)
    p = a / mass.clamp_min(torch.finfo(a.dtype).tiny)
    kappa = (-(p * p.clamp_min(torch.finfo(a.dtype).tiny).log()).sum(-1)).exp() / a.shape[-1]
    sorted_p = p.sort(-1).values
    idx = torch.arange(1, a.shape[-1]+1, device=a.device, dtype=a.dtype)
    gini = ((2*idx-a.shape[-1]-1)*sorted_p).sum(-1) / a.shape[-1]
    valid = labels.bool()[None] & (mass.squeeze(-1) > 0)
    count = valid.sum(-1)
    def average(x):
        return (x*valid).sum(-1).div(count.clamp_min(1)).masked_fill(count==0, float('nan'))
    # Cosine is invariant to positive row scaling. Normalize mass BEFORE
    # squaring: raw1e-30 probabilities otherwise have zero FP32 L2 norm.
    unit = p / p.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)
    cosine = unit @ unit.transpose(-1,-2)
    pair = valid[..., :, None] & valid[..., None, :]
    pair &= ~torch.eye(a.shape[-2], device=a.device, dtype=torch.bool)
    pairs = pair.sum((-1,-2))
    gamma = (cosine*pair).sum((-1,-2)).div(pairs.clamp_min(1)).masked_fill(pairs==0, float('nan'))
    return torch.stack([average(kappa), average(gini), gamma], -1).transpose(0,1)


def class_agnosticism(a, labels):
    """Image-wise positive-query map cosine [0,1]; high means shared support.

    Uses the existing stable head_statistics gamma; single-query images are NaN.
    a is [cells,B,Q,N], labels [B,Q]. No semantic or causal claim follows.
    """
    return head_statistics(a, labels)[..., 2]
