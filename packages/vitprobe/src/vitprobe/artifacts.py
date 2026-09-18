"""Artifact descriptors on patch-only tensors; no backbone or dataset coupling.

Norms, received mass and RGB gradient proxies are computed in FP32.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def patch_norm_stats(patch_tokens, hi=3.0):
    """Relative patch norms (nonnegative); high values indicate outliers.

    hi=3.0 is the source probe threshold, not a transferable semantic rule.
    Source M2 fractions: 0.013429 / 0.005396 / 0.016951. FP32 norms.
    """
    norms = patch_tokens.float().norm(dim=-1)
    median = norms.median(-1, keepdim=True).values.clamp_min(1e-8)
    ratio = norms / median
    return dict(ratio=ratio, frac_hi=(ratio > hi).float().mean(-1),
                p99_over_p50=torch.quantile(norms, .99, dim=-1) / median.squeeze(-1))


@torch.no_grad()
def attractor_stats(attn_pp, topk_frac=.01):
    """Incoming key mass, concentration and top-share; high means attraction.

    Input [B,N,N]: sum QUERY dimension. Source top-share values:
    0.081556 / 0.063604 / 0.088546; gate .05 was experiment-specific.
    """
    g = attn_pp.float().sum(1)
    g = g / g.sum(-1, keepdim=True).clamp_min(1e-12)
    n = g.shape[-1]
    x = g.sort(-1).values
    idx = torch.arange(1, n + 1, dtype=g.dtype, device=g.device)
    gini = ((2 * idx - n - 1) * x).sum(-1) / (n * x.sum(-1).clamp_min(1e-12))
    k = max(1, round(topk_frac * n))
    return dict(recv=g, gini=gini, top_share=g.topk(k, dim=-1).values.sum(-1), k=k)


@torch.no_grad()
def lowinfo_scores(rgb, ratio, grid=28, *, layout=None):
    """Relative RGB-gradient proxy; low means locally less image variation.

    RGB must be unnormalized; ratio is [B,N]. Source thresholds .01 and 3
    are retained. Empty high-norm sets return NaN. Not a background label.
    layout optionally supplies a rectangular grid; default preserves source.
    """
    # Input is inverse-ImageNet-normalized RGB, not differently scaled channels.
    x = rgb.float().mean(1, keepdim=True)
    gx, gy = x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]
    magnitude = (gx[..., :-1, :].square() + gy[..., :, :-1].square()).sqrt()
    grid_hw = (grid, grid) if layout is None else layout.grid_hw
    if min(grid_hw) < 1 or ratio.shape[-1] != grid_hw[0] * grid_hw[1]:
        raise ValueError('Patch ratios do not match the requested spatial grid')
    gradient = F.adaptive_avg_pool2d(magnitude, grid_hw).flatten(1)
    median = gradient.median(-1).values.clamp_min(1e-8)
    k = max(1, round(.01 * ratio.shape[-1]))
    top = gradient.gather(1, ratio.topk(k, dim=-1).indices).mean(-1) / median
    hi = ratio > 3
    high_mean = (gradient * hi).sum(-1) / hi.sum(-1).clamp_min(1) / median
    high_mean = high_mean.masked_fill(~hi.any(-1), float('nan'))
    return top, high_mean, (~hi.any(-1)).float()
