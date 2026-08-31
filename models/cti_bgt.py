"""CTI BGT-only map refinement (official CTI commit 1c6fdb4).

No infusion, token swapping, memory bank, or class/background softmax is used.
The VOC branch propagates gradients through foreground, background and affinity.
"""
import math

import torch
import torch.nn.functional as F


def add_cti_bgt_arguments(parser):
    parser.add_argument('--cti-bgt', action='store_true',
                        help='enable the isolated CTI background-token baseline')
    parser.add_argument('--cti-bgt-weight', type=float, default=0.1,
                        help='weight of L_bcam; zero gives the token/head ablation')
    parser.add_argument('--cti-bgt-n-layers', type=int, default=6,
                        help='last layers summed for CTI class-to-patch attention')
    parser.add_argument('--cti-bgt-affinity-start', type=int, default=4,
                        help='first zero-based layer summed for CTI patch affinity')


def validate_cti_bgt(enabled, weight, n_layers, affinity_start, depth,
                     bcss_variant, psl_variant, attention_normalization):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('cti_bgt_weight must be finite and non-negative')
    if n_layers < 1 or affinity_start < 0:
        raise ValueError('CTI attention layers must be positive/start non-negative')
    if enabled:
        if bcss_variant.lower() != 'e0' or psl_variant.lower() != 'baseline':
            raise ValueError('CTI BGT requires BCSS E0 and PSL baseline')
        if attention_normalization != 'vanilla':
            raise ValueError('CTI BGT baseline requires vanilla attention')
        if n_layers > depth or affinity_start >= depth:
            raise ValueError('CTI attention layer selection exceeds model depth')


def cti_max_norm(maps, eps=1e-5):
    """Official max_norm: ReLU, spatial min/max, then epsilon-shifted ReLU.

    Unlike plain division by max, a constant positive map normalizes to zero.
    FP32 accumulation prevents epsilon underflow with autocast/half inputs.
    """
    maps = F.relu(maps.float())
    flat = maps.flatten(2)
    lo = flat.min(-1).values[..., None, None]
    hi = flat.max(-1).values[..., None, None]
    return F.relu(maps - lo - eps) / (hi - lo + eps)


def cti_bgt_maps(patch_cam, attentions, labels=None, n_layers=6, affinity_start=4):
    """Refine [BG, FG_1, ..., FG_C] CAMs with joint self-attention.

    attentions: [layers, batch, heads, tokens, tokens], before dropout.
    Foreground union is label-masked and max-reduced BEFORE propagation.
    The last-six / start-four defaults reproduce the official VOC BGT loss
    slices without importing its swap_idx control flow. No sqrt or re-softmax.
    """
    if patch_cam.ndim != 4 or patch_cam.shape[1] < 2:
        raise ValueError('patch_cam must be [B, C+1, H, W] including BG')
    batch, channels, height, width = patch_cam.shape
    patches = height * width
    if isinstance(attentions, (list, tuple)):
        attentions = torch.stack(attentions)
    if (attentions.ndim != 5 or attentions.shape[1] != batch or
            attentions.shape[-2:] != (channels + patches, channels + patches)):
        raise ValueError('attention shape does not match BG/FG/patch token layout')
    if not (1 <= n_layers <= len(attentions) and 0 <= affinity_start < len(attentions)):
        raise ValueError('invalid CTI attention layer selection')
    if labels is not None and labels.shape != (batch, channels - 1):
        raise ValueError('labels must contain exactly C foreground classes')

    # Explicitly disable autocast for propagation and normalization; do not
    # detach either branch (official VOC behavior). Upstream uses FP64 matmul.
    with torch.autocast(device_type=patch_cam.device.type, enabled=False):
        attention = attentions.float().mean(dim=2)
        class_to_patch = attention[-n_layers:, :, :channels, channels:].sum(0)
        patch_affinity = attention[affinity_start:, :, channels:, channels:].sum(0)
        fused = F.relu(patch_cam.float()) * class_to_patch.reshape_as(patch_cam)
        refined = torch.matmul(patch_affinity.unsqueeze(1),
                               fused.flatten(2).unsqueeze(-1)).reshape_as(fused)
        result = {
            'class_to_patch': class_to_patch,
            'background_to_patch': class_to_patch[:, :1],
            'patch_affinity': patch_affinity,
            'fused_cam': fused,
            'refined_cam': refined,
            'background': cti_max_norm(refined[:, :1]),
        }
        if labels is not None:
            foreground = (fused[:, 1:] * labels.float()[..., None, None]).max(
                dim=1, keepdim=True).values
            union = torch.matmul(patch_affinity, foreground.flatten(2).transpose(1, 2))
            union = union.transpose(1, 2).reshape(batch, 1, height, width)
            result['foreground_union'] = cti_max_norm(union)
        return result


def cti_bcam_loss(maps):
    """Mean absolute complement consistency; both branches receive gradients."""
    if 'foreground_union' not in maps:
        raise ValueError('L_bcam requires image-level foreground labels')
    return ((1.0 - maps['foreground_union']) - maps['background']).abs().mean()


def validate_cti_bgt_checkpoint(checkpoint, model):
    """Reject mismatched CAM settings, including old checkpoints without BGT."""
    actual = checkpoint.get('cti_bgt', {'enabled': False})
    expected = model.cti_bgt_configuration()
    keys = expected if expected['enabled'] else ('enabled',)
    mismatches = {key: (actual.get(key), expected[key]) for key in keys
                  if actual.get(key) != expected[key]}
    if mismatches:
        raise ValueError(f'CTI BGT checkpoint/config mismatch: {mismatches}')


def adapt_cti_bgt_finetune(checkpoint, model):
    """Initialize BGT from DeiT or an FG-only host, preserving FG head rows.

    Same-architecture BGT checkpoints retain every learned BG parameter.
    This helper is opt-in; the existing default loader remains untouched.
    """
    state = dict(checkpoint.get('model', checkpoint))
    target = model.state_dict()
    # Distinguish DeiT's 2-D classifier from an MCTformer+ convolutional head.
    is_deit = state.get('head.weight', torch.empty(0)).ndim == 2
    if is_deit:
        position = state['pos_embed']
        patch_position = position[:, 1:]
        side = math.isqrt(patch_position.shape[1])
        if side * side != patch_position.shape[1]:
            raise ValueError('expected a square DeiT patch-position grid')
        patch_position = F.interpolate(
            patch_position.reshape(1, side, side, -1).permute(0, 3, 1, 2),
            size=(model.Hp, model.Wp), mode='bicubic', align_corners=False)
        state['pos_embed_pat'] = patch_position.flatten(2).transpose(1, 2)
        state['pos_embed_cls'] = position[:, :1].repeat(1, model.num_classes, 1)
        state['cls_token'] = state['cls_token'].repeat(1, model.num_classes, 1)
        for key in ('head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias'):
            state.pop(key, None)
    elif 'bg_token' not in state:
        for key in ('head.weight', 'head.bias'):
            if key in state:
                if state[key].shape != target[key][1:].shape:
                    raise ValueError(f'incompatible foreground checkpoint shape: {key}')
                state[key] = torch.cat((target[key][:1], state[key]), dim=0)
    # Fail on unanticipated shape differences rather than silently dropping them.
    for key, value in state.items():
        if key in target and value.shape != target[key].shape:
            raise ValueError(f'incompatible checkpoint shape: {key}')
    return state
