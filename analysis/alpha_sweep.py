"""Inference-only fusion exponent. Endpoints override the interior epsilon floor."""
import torch

ALPHAS = [round(.05 * k, 2) for k in range(21)]


def blend(M, a, alpha, eps=1e-8):
    if M.dtype != torch.float32 or a.dtype != torch.float32:
        raise ValueError('Fusion requires FP32 inputs')
    if alpha == 0:
        return M.relu()
    if alpha == 1:
        return a
    return M.relu().clamp_min(eps).pow(1-alpha) * a.clamp_min(eps).pow(alpha)


def native_blend(M, a):
    # Separate exact native reference; the registered interior floor can lift
    # zeros at alpha=.5, so do NOT silently call the clamped curve native.
    return (M.relu() * a).sqrt()
