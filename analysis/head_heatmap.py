"""Per-head spatial diagnostics. Image, not class-pair, is the sampling unit."""
import torch

CELLS = [(l, h) for l in range(12) for h in range(6)]
TOP_M = [1, 2, 4, 8, 16, 32, 72]


def c2p_single_head(attn_layers, n_class, l, h, row_normalize=False, eps=1e-8):
    a = attn_layers[l][:, h, :n_class, n_class:]
    if a.dtype != torch.float32:
        raise ValueError('Head analysis requires FP32 attention')
    return a / a.sum(-1, keepdim=True).clamp_min(eps) if row_normalize else a


def head_statistics(a, labels):
    """a:[cells,B,C,N], returns [B,cells,3] kappa/Gini/gamma.

    Positive-class averaging within each image. Gamma is missing, NOT zero,
    for images with fewer than two positives or zero-norm positive maps.
    """
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
