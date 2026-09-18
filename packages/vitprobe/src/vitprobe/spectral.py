"""Spatial (not embedding-axis) spectra. No smoothing is applied to the model.

    E_hi uses equal-weight radial-bin MEANS, as registered in the plan; it is
    not the annulus-count-weighted fraction of all 2D Fourier energy. xi uses
    the actual 2D inverse power transform, not irfft of a radial profile.
"""

import torch
import torch.nn.functional as F


def _radial_mean(values, bins, n_bins):
    flat = values.flatten(1)
    counts = torch.bincount(bins.flatten(), minlength=n_bins).to(values.dtype)
    total = values.new_zeros(values.shape[0], n_bins)
    total.scatter_add_(1, bins.flatten().expand_as(flat), flat)
    return total / counts.clamp_min(1), counts


def spectrum_and_autocorrelation(features, n_bins=16):
    if features.ndim != 4 or min(features.shape[1:3]) < 28:
        raise ValueError('Spectral grid must be [B,H,W,D] with both H,W >= 28')
    if n_bins < 4:
        raise ValueError('At least four radial bins required')
    f = F.normalize(features.float(), dim=-1)
    _, h, w, _ = f.shape
    varying = (f.amax((1, 2)) - f.amin((1, 2))).abs().amax(-1) > 0
    f = f - f.mean((1, 2), keepdim=True)
    window = torch.hann_window(h, periodic=False, device=f.device)[:, None] * torch.hann_window(w, periodic=False, device=f.device)[None]
    power = torch.fft.fft2(f * window[None, :, :, None], dim=(1, 2)).abs().square().sum(-1)
    yy = torch.arange(h, device=f.device) - h // 2
    xx = torch.arange(w, device=f.device) - w // 2
    radius = (yy[:, None].square() + xx[None].square()).float().sqrt()
    bins = (radius / radius.max() * (n_bins - 1)).round().long()
    radial, counts = _radial_mean(torch.fft.fftshift(power, dim=(1, 2)), bins, n_bins)
    ac = torch.fft.fftshift(torch.fft.ifft2(power).real, dim=(1, 2))
    ac_bins = radius.round().long()
    radial_ac, _ = _radial_mean(ac, ac_bins, int(ac_bins.max()) + 1)
    # Numerical residual after demeaning a constant field must not masquerade
    # as a meaningful spectrum or an enormous fitted correlation length.
    valid = varying & (f.square().sum((1, 2, 3)) > 1e-12)
    radial = torch.where(valid[:, None], radial, torch.zeros_like(radial))
    radial_ac = torch.where(valid[:, None], radial_ac, torch.zeros_like(radial_ac))
    centers = torch.linspace(0, float(radius.max()), n_bins, device=f.device)
    return radial, centers, counts, radial_ac


def radial_power_spectrum(features, n_bins=16):
    """FP32 radial power after L2, demeaning and Hann; constant fields are zero.

    White-noise high-half share is near .5; this is not a pathology threshold.
    """
    p, centers, _, _ = spectrum_and_autocorrelation(features, n_bins)
    return p, centers


def high_freq_ratio(p):
    """High-half radial-bin share [0,1], not annulus-weighted energy; zero is NaN."""
    p = p.float()
    total = p.sum(-1)
    value = p[..., p.shape[-1] // 2:].sum(-1) / total.clamp_min(1e-30)
    return value.masked_fill(total <= 0, float('nan'))


def correlation_length(autocorrelation, fit_range=(1, 6), min_r2=0.8):
    """Fixed integer radii 1..5 patch units. Failed fits stay missing.

    White noise has no positive exponential tail; xi is NOT forced to one.
    Return xi, R², slope, validity for auditing rather than clipping slopes.
    """
    autocorrelation = autocorrelation.float()
    lo, hi = fit_range
    if not 0 < lo < hi <= autocorrelation.shape[-1] or hi - lo < 3:
        raise ValueError('Invalid fixed autocorrelation fit range')
    norm = autocorrelation / autocorrelation[..., :1].clamp_min(1e-30)
    segment = norm[..., lo:hi]
    positive = (segment > 0).all(-1) & (autocorrelation[..., 0] > 0)
    y = segment.clamp_min(1e-30).log()
    r = torch.arange(lo, hi, device=y.device, dtype=y.dtype)
    r = r - r.mean()
    yc = y - y.mean(-1, keepdim=True)
    slope = (yc * r).sum(-1) / r.square().sum()
    ss = yc.square().sum(-1)
    r2 = 1 - (yc - slope[..., None] * r).square().sum(-1) / ss.clamp_min(1e-30)
    valid = positive & (slope < 0) & (r2 >= min_r2) & (ss > 0)
    xi = (-1 / slope).masked_fill(~valid, float('nan'))
    return {'xi': xi, 'xi_r2': r2.masked_fill(~positive, float('nan')), 'xi_slope': slope.masked_fill(~positive, float('nan')), 'xi_valid': valid}
