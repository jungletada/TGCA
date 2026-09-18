"""Generic patch-graph operators, copied without training or readout integration.

Layout replaces hard-coded token counts. Existing FP64 reference behavior is
preserved; other inputs accumulate in FP32 with autocast disabled. This is NOT
an implementation of a domain-specific repair pipeline.
"""
import torch

from .layout import TokenLayout


def aggregate_p2p(attn_layers, layout: TokenLayout, p_layers='all', p_reduce='sum',
                  p_alpha=1., p_sym=False, eps=1e-8):
    """Query i receives key j. FP32 accumulation, float64 for reference tests.

    Sequential sum exactly preserves the historical pooling-affinity path.
    Layout also preserves patch-first and register-token behavior.
    """
    if p_layers not in {'all', 'last3', 'last6'} or p_reduce not in {'sum', 'mean', 'rownorm_mean'}:
        raise ValueError('Invalid P2P layer/reduction setting')
    if p_alpha <= 0 or not attn_layers:
        raise ValueError('Positive alpha and nonempty attention required')
    records = attn_layers if p_layers == 'all' else attn_layers[-int(p_layers[4:]):]
    patch_slice = layout.patch_slice
    for record in records:
        if record.ndim != 4:
            raise ValueError('Attention records must be [B,H,T,T]')
        layout.validate_attention(record)
    dtype = torch.float64 if records[0].dtype == torch.float64 else torch.float32
    with torch.autocast(records[0].device.type, enabled=False):
        result = None
        for record in records:
            matrix = record[:, :, patch_slice, patch_slice].to(dtype).mean(1)
            if p_reduce == 'rownorm_mean':
                matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(eps)
            result = matrix if result is None else result + matrix
        if p_reduce != 'sum':
            result = result / len(records)
        if p_alpha != 1:
            result = result / result.sum(-1, keepdim=True).clamp_min(eps)
            result = result.clamp_min(0).pow(p_alpha)
        if p_sym:
            result = .5 * (result + result.transpose(-1, -2))
    return result


def propagate_weights(w, P, floor='none', beta=1., eps=1e-8, return_fallback=False):
    """Normalize w @ P.T, remove spatial floor, optionally mix with w.

    A removed-floor zero row falls back to original w; return its explicit mask
    for audit counting. Legacy floor=none keeps its original clamp behavior.
    """
    if floor not in {'none', 'min', 'mean'} or not 0 <= beta <= 1:
        raise ValueError('Invalid floor/beta')
    with torch.autocast(w.device.type, enabled=False):
        dtype = torch.float64 if w.dtype == torch.float64 else torch.float32
        w, P = w.to(dtype), P.to(dtype)
        if beta == 0:
            fallback = torch.zeros_like(w[..., 0], dtype=torch.bool)
            return (w, fallback) if return_fallback else w
        r = torch.bmm(w, P.transpose(-1, -2))
        if floor == 'min':
            r = r - r.amin(-1, keepdim=True)
        elif floor == 'mean':
            r = (r - r.mean(-1, keepdim=True)).clamp_min(0)
        mass = r.sum(-1, keepdim=True)
        fallback = (mass.squeeze(-1) < eps) & (floor != 'none')
        r = r / mass.clamp_min(eps)
        if floor != 'none':
            r = torch.where(fallback[..., None], w, r)
        if beta < 1:
            r = (1 - beta) * w + beta * r
            r = r / r.sum(-1, keepdim=True).clamp_min(eps)
    return (r, fallback) if return_fallback else r
