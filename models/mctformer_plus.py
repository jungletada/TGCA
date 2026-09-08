import math
from collections import OrderedDict
from copy import deepcopy
from typing import Mapping

import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F
from models.vit import VisionTransformer, _cfg
from models.bcss import (
    SemanticSlotDecoder,
    bcss_schedule,
    infer_active_classes,
    ownership_calibrate_attention,
    semantic_slot_losses,
    validate_bcss_variant,
)
from models.persistent_semantic import (
    SemanticReadWrite,
    parse_interaction_layers,
    validate_psl_variant,
)

from models.cti_bgt import cti_bgt_maps, validate_cti_bgt
from models.class_token_pooling import (
    ClassWiseWeightedPooling,
    class_token_pooling_diagnostics,
)

__all__ = [
    'ClassStableLastPooler',
    'ClassWiseWeightedPooling',
    'LastPatchAggregator',
    'MCTFORMERPLUS_VARIANTS',
    'MCTformerPlus',
    'MCTformerPlusCam',
    'adapt_deit_checkpoint_for_mctformerplus',
    'build_mctformerplus',
    'get_mctformerplus_spec',
    'mctformerplus',
    'mctformerplus_base',
    'mctformerplus_tiny',
    'model_spec_from_instance',
    'checkpoint_class_stable_last_enabled',
    'checkpoint_class_token_init',
    'checkpoint_final_norm_enabled',
    'checkpoint_last_mct_enabled',
    'checkpoint_patch_final_norm_enabled',
    'resolve_mctformerplus_checkpoint_variant',
    'resolve_mctformerplus_variant',
    'validate_mctformerplus_final_norm_checkpoint',
    'validate_mctformerplus_class_token_init_checkpoint',
]


MCTFORMERPLUS_VARIANTS = {
    'tiny': {
        'family': 'MCTformer+',
        'variant': 'tiny',
        'model_name': 'mctformerplus_tiny',
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'head_dim': 64,
        'patch_size': 16,
        'mlp_ratio': 4,
        'pretrained_url': (
            'https://dl.fbaipublicfiles.com/deit/'
            'deit_tiny_patch16_224-a1311bcf.pth'
        ),
    },
    'small': {
        'family': 'MCTformer+',
        'variant': 'small',
        'model_name': 'mctformerplus',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'head_dim': 64,
        'patch_size': 16,
        'mlp_ratio': 4,
        'pretrained_url': (
            'https://dl.fbaipublicfiles.com/deit/'
            'deit_small_patch16_224-cd65a155.pth'
        ),
    },
    'base': {
        'family': 'MCTformer+',
        'variant': 'base',
        'model_name': 'mctformerplus_base',
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'head_dim': 64,
        'patch_size': 16,
        'mlp_ratio': 4,
        'pretrained_url': (
            'https://dl.fbaipublicfiles.com/deit/'
            'deit_base_patch16_224-b5f2ef4d.pth'
        ),
    },
}

_MCTFORMERPLUS_MODEL_TO_VARIANT = {
    spec['model_name']: variant
    for variant, spec in MCTFORMERPLUS_VARIANTS.items()
}


def resolve_mctformerplus_variant(model_name):
    """Resolve an exact MCTformer+ model/variant name without fuzzy matching."""
    normalized = str(model_name).strip().lower()
    if normalized in MCTFORMERPLUS_VARIANTS:
        return normalized
    if normalized in _MCTFORMERPLUS_MODEL_TO_VARIANT:
        return _MCTFORMERPLUS_MODEL_TO_VARIANT[normalized]
    supported = sorted(
        set(MCTFORMERPLUS_VARIANTS) | set(_MCTFORMERPLUS_MODEL_TO_VARIANT)
    )
    raise ValueError(
        f'Unknown MCTformer+ variant/model {model_name!r}; expected one of {supported}'
    )


def get_mctformerplus_spec(variant_or_model_name):
    """Return an isolated copy of the canonical architecture specification."""
    variant = resolve_mctformerplus_variant(variant_or_model_name)
    return deepcopy(MCTFORMERPLUS_VARIANTS[variant])


def _variant_constructor_kwargs(variant, kwargs):
    spec = get_mctformerplus_spec(variant)
    kwargs = dict(kwargs)
    fixed = {
        'patch_size': spec['patch_size'],
        'embed_dim': spec['embed_dim'],
        'depth': spec['depth'],
        'num_heads': spec['num_heads'],
        'mlp_ratio': spec['mlp_ratio'],
        'qkv_bias': True,
    }
    for key, expected in fixed.items():
        if key in kwargs and kwargs[key] != expected:
            raise ValueError(
                f'{spec["model_name"]} fixes {key}={expected}, got {kwargs[key]}'
            )
        kwargs[key] = expected
    if 'norm_layer' not in kwargs:
        kwargs['norm_layer'] = partial(nn.LayerNorm, eps=1e-6)
    return spec, kwargs


class LastPatchAggregator(nn.Module):
    """LaST-ViT repository patch aggregation over ``[B, N, D]`` tokens."""

    def __init__(self, embed_dim, topk=1, sigma=None, eps=1e-6):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)
        self.sigma = (
            math.sqrt(self.embed_dim) if sigma is None else float(sigma)
        )
        self.eps = float(eps)
        if self.embed_dim < 1:
            raise ValueError('Last-MCT embed_dim must be positive')
        if self.topk < 1:
            raise ValueError('Last-MCT topk must be positive')
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('Last-MCT sigma must be finite and positive')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('Last-MCT eps must be finite and positive')
        self.register_buffer('_last_kernel', torch.empty(0), persistent=False)

    def configuration(self):
        return {
            'topk': self.topk,
            'sigma': self.sigma,
            'eps': self.eps,
            'score_formula': 'P / abs(P_lowpass - P).clamp_min(eps)',
            'fft_dimension': 'embedding',
            'topk_dimension': 'patch',
            'selection_semantics': 'independent patch index per embedding channel',
        }

    def _get_last_kernel(self, patch_tokens):
        if patch_tokens.ndim != 3:
            raise ValueError(
                'Last-MCT patch tokens must have shape [B, N, D], got '
                f'{tuple(patch_tokens.shape)}'
            )
        width = patch_tokens.shape[-1]
        if width != self.embed_dim:
            raise ValueError(
                f'Last-MCT expected embedding width {self.embed_dim}, got {width}'
            )
        if self._last_kernel.numel() != width:
            positions = torch.arange(
                -width // 2 + 1,
                width // 2 + 1,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            kernel = torch.exp(-0.5 * (positions / self.sigma) ** 2)
            self._last_kernel = kernel / kernel.max()
        return self._last_kernel.view(1, 1, width).to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        )

    def low_pass(self, patch_tokens):
        original_dtype = patch_tokens.dtype
        if original_dtype in (torch.float16, torch.bfloat16):
            values = patch_tokens.float()
        else:
            values = patch_tokens
        kernel = self._get_last_kernel(values)
        spectrum = torch.fft.fft(values, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        spectrum = spectrum * kernel
        spectrum = torch.fft.ifftshift(spectrum, dim=-1)
        low_pass = torch.fft.ifft(spectrum, dim=-1).real
        return low_pass.to(original_dtype)

    def stability_score(self, patch_tokens, low_pass_tokens):
        if patch_tokens.shape != low_pass_tokens.shape:
            raise ValueError(
                'Last-MCT patch and low-pass token shapes must match, got '
                f'{tuple(patch_tokens.shape)} and {tuple(low_pass_tokens.shape)}'
            )
        if patch_tokens.dtype in (torch.float16, torch.bfloat16):
            values = patch_tokens.float()
            low_pass = low_pass_tokens.float()
        else:
            values = patch_tokens
            low_pass = low_pass_tokens
        denominator = (low_pass - values).abs().clamp_min(self.eps)
        return values / denominator

    def forward(self, patch_tokens):
        if patch_tokens.ndim != 3:
            raise ValueError(
                'Last-MCT patch tokens must have shape [B, N, D], got '
                f'{tuple(patch_tokens.shape)}'
            )
        if patch_tokens.shape[1] < 1:
            raise ValueError('Last-MCT requires at least one patch token')
        low_pass = self.low_pass(patch_tokens)
        stability = self.stability_score(patch_tokens, low_pass)
        k = min(self.topk, patch_tokens.shape[1])
        _, indices = torch.topk(stability, k=k, dim=1, largest=True)
        selected = torch.gather(patch_tokens, dim=1, index=indices)
        pooled = selected.mean(dim=1)
        return pooled, indices, stability


class ClassStableLastPooler(nn.Module):
    """Class-wise LaST selector over spatial response maps.

    Low-pass filtering operates on ``[B, N, D]`` patch tokens only along the
    embedding dimension. Pooling operates independently for every semantic
    class over flattened spatial locations and gathers values from the
    original response map, never from the low-pass map.
    """

    def __init__(self, embed_dim, topk=1, sigma=None, eps=1e-6):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)
        self.sigma = (
            math.sqrt(self.embed_dim) if sigma is None else float(sigma)
        )
        self.eps = float(eps)
        if self.embed_dim < 1:
            raise ValueError('Class-Stable LaST embed_dim must be positive')
        if self.topk < 1:
            raise ValueError('Class-Stable LaST topk must be positive')
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(
                'Class-Stable LaST sigma must be finite and positive'
            )
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('Class-Stable LaST eps must be finite and positive')
        self.register_buffer('_last_kernel', torch.empty(0), persistent=False)

    def configuration(self):
        return {
            'topk': self.topk,
            'sigma': self.sigma,
            'eps': self.eps,
            'score_formula': 'M / abs(M_lowpass - M).clamp_min(eps)',
            'fft_dimension': 'embedding',
            'topk_dimension': 'spatial',
            'selection_semantics': (
                'independent spatial index per semantic class'
            ),
            'gathered_values': 'original M',
            'lowpass_role': 'selector only',
        }

    def _get_last_kernel(self, patch_tokens):
        if patch_tokens.ndim != 3:
            raise ValueError(
                'Class-Stable LaST patch tokens must have shape [B, N, D], '
                f'got {tuple(patch_tokens.shape)}'
            )
        width = patch_tokens.shape[-1]
        if width != self.embed_dim:
            raise ValueError(
                'Class-Stable LaST expected embedding width '
                f'{self.embed_dim}, got {width}'
            )
        if self._last_kernel.numel() != width:
            positions = torch.arange(
                -width // 2 + 1,
                width // 2 + 1,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            kernel = torch.exp(-0.5 * (positions / self.sigma) ** 2)
            self._last_kernel = kernel / kernel.max()
        return self._last_kernel.view(1, 1, width).to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        )

    def low_pass(self, patch_tokens):
        original_dtype = patch_tokens.dtype
        values = (
            patch_tokens.float()
            if original_dtype in (torch.float16, torch.bfloat16)
            else patch_tokens
        )
        kernel = self._get_last_kernel(values)
        spectrum = torch.fft.fft(values, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1) * kernel
        low_pass = torch.fft.ifft(
            torch.fft.ifftshift(spectrum, dim=-1), dim=-1
        ).real
        return low_pass.to(original_dtype)

    def stability_score(self, class_map, low_pass_class_map):
        if class_map.ndim != 4:
            raise ValueError(
                'Class-Stable LaST maps must have shape [B, C, H, W], got '
                f'{tuple(class_map.shape)}'
            )
        if class_map.shape != low_pass_class_map.shape:
            raise ValueError(
                'Class-Stable LaST original and low-pass map shapes must '
                f'match, got {tuple(class_map.shape)} and '
                f'{tuple(low_pass_class_map.shape)}'
            )
        if class_map.dtype in (torch.float16, torch.bfloat16):
            values = class_map.float()
            low_pass_values = low_pass_class_map.float()
        else:
            values = class_map
            low_pass_values = low_pass_class_map
        denominator = (low_pass_values - values).abs().clamp_min(self.eps)
        return values / denominator

    def pool(self, class_map, stability):
        if class_map.shape != stability.shape:
            raise ValueError(
                'Class-Stable LaST map and stability shapes must match, got '
                f'{tuple(class_map.shape)} and {tuple(stability.shape)}'
            )
        original = class_map.flatten(2)
        scores = stability.flatten(2)
        if original.shape[-1] < 1:
            raise ValueError(
                'Class-Stable LaST requires at least one spatial location'
            )
        k = min(self.topk, original.shape[-1])
        indices = torch.topk(scores, k=k, dim=-1, largest=True).indices
        selected = torch.gather(original, dim=-1, index=indices)
        return selected.mean(dim=-1), indices, selected

    def forward(self, class_map, low_pass_class_map):
        stability = self.stability_score(class_map, low_pass_class_map)
        pooled, indices, selected = self.pool(class_map, stability)
        return pooled, indices, stability, selected


class MCTformerPlus(VisionTransformer):
    def __init__(
            self, decay_parameter=0.996, input_size=448,
            bcss_variant='e0', bcss_num_background_slots=1,
            bcss_tau=0.5, bcss_beta=0.5, bcss_cls_threshold=0.5,
            bcss_lambda_fg=0.5, bcss_lambda_bg=0.1,
            bcss_semantic_temperature=1.0, psl_variant='baseline',
            psl_interaction_layers=(11,), psl_relation_dim=384,
            psl_num_background_latents=1, cti_bgt=False, cti_bgt_weight=0.1,
            cti_bgt_n_layers=6, cti_bgt_affinity_start=4, final_norm=False,
            patch_final_norm=False, last_mct=False, class_stable_last=False,
            last_topk=1, last_sigma=None, last_eps=1e-6,
            class_token_init='baseline', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_token_init = str(class_token_init).strip().lower()
        if self.class_token_init not in {'baseline', 'cwp'}:
            raise ValueError(
                "class_token_init must be one of {'baseline', 'cwp'}"
            )
        self.final_norm = bool(final_norm)
        self.patch_final_norm = bool(patch_final_norm)
        self.last_mct = bool(last_mct)
        self.class_stable_last = bool(class_stable_last)
        if self.final_norm and self.patch_final_norm:
            raise ValueError(
                'final_norm and patch_final_norm are mutually exclusive'
            )
        if self.last_mct and self.class_stable_last:
            raise ValueError(
                'last_mct and class_stable_last are mutually exclusive'
            )
        if self.class_token_init == 'cwp':
            architecture = (
                self.embed_dim,
                len(self.blocks),
                int(self.blocks[0].attn.num_heads),
            )
            if architecture != (384, 12, 6):
                raise ValueError(
                    'CWP first-round support is restricted to '
                    'MCTformer+-Small (embed_dim=384, depth=12, heads=6)'
                )
            if self.final_norm or self.patch_final_norm:
                raise ValueError('CWP requires final_norm=False and patch_final_norm=False')
            if self.last_mct or self.class_stable_last:
                raise ValueError('CWP is incompatible with LaST pooling variants')
            if self.attention_normalization != 'vanilla':
                raise ValueError('CWP requires vanilla attention')
            if str(bcss_variant).lower() != 'e0':
                raise ValueError('CWP requires BCSS E0')
            if str(psl_variant).lower() != 'baseline':
                raise ValueError('CWP requires PSL baseline')
            if bool(cti_bgt):
                raise ValueError('CWP requires CTI-BGT disabled')
        if self.last_mct or self.class_stable_last:
            expected_sigma = math.sqrt(self.embed_dim)
            requested_sigma = (
                expected_sigma if last_sigma is None else float(last_sigma)
            )
            architecture = (
                self.embed_dim,
                len(self.blocks),
                int(self.blocks[0].attn.num_heads),
            )
            if architecture != (384, 12, 6):
                raise ValueError(
                    'LaST ablation first-round support is restricted to '
                    'MCTformer+-Small (embed_dim=384, depth=12, heads=6)'
                )
            if not self.patch_final_norm or self.final_norm:
                raise ValueError(
                    'LaST ablation requires patch_final_norm=True and '
                    'final_norm=False'
                )
            if self.attention_normalization != 'vanilla':
                raise ValueError('LaST ablations require vanilla attention')
            if str(bcss_variant).lower() != 'e0':
                raise ValueError('LaST ablations require BCSS E0')
            if str(psl_variant).lower() != 'baseline':
                raise ValueError('LaST ablations require PSL baseline')
            if bool(cti_bgt):
                raise ValueError('LaST ablations require CTI-BGT disabled')
            if int(last_topk) != 1:
                raise ValueError('LaST ablation first round fixes topk=1')
            if requested_sigma != expected_sigma:
                raise ValueError(
                    'LaST ablation first round fixes sigma=sqrt(embed_dim)'
                )
            if float(last_eps) != 1e-6:
                raise ValueError('LaST ablation first round fixes eps=1e-6')
        self.cti_bgt = bool(cti_bgt)
        self.cti_bgt_weight = cti_bgt_weight
        self.cti_bgt_n_layers = cti_bgt_n_layers
        self.cti_bgt_affinity_start = cti_bgt_affinity_start
        validate_cti_bgt(
            self.cti_bgt, cti_bgt_weight, cti_bgt_n_layers,
            cti_bgt_affinity_start, len(self.blocks), bcss_variant, psl_variant,
            self.attention_normalization)
        # num_classes continues to mean foreground labels, never C+1.
        self.num_class_tokens = self.num_classes + int(self.cti_bgt)
        if self.cti_bgt:
            for block in self.blocks:
                block.attn.num_classes = self.num_class_tokens
        if self.last_mct:
            self.head = nn.Conv2d(
                self.embed_dim, self.num_class_tokens,
                kernel_size=1, stride=1, padding=0,
            )
            self.last_patch_aggregator = LastPatchAggregator(
                embed_dim=self.embed_dim,
                topk=last_topk,
                sigma=last_sigma,
                eps=last_eps,
            )
        else:
            self.head = nn.Conv2d(
                self.embed_dim, self.num_class_tokens,
                kernel_size=3, stride=1, padding=1,
            )
            self.last_patch_aggregator = None
        self.class_stable_last_pooler = (
            ClassStableLastPooler(
                embed_dim=self.embed_dim,
                topk=last_topk,
                sigma=last_sigma,
                eps=last_eps,
            )
            if self.class_stable_last else None
        )
        self.head.apply(self._init_weights)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        self.Hp, self.Wp = math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1])
        self.num_patches = self.Hp * self.Wp

        if self.class_token_init == 'baseline':
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_classes, self.embed_dim)
            )
            self.class_token_pooler = None
        else:
            self.register_parameter('cls_token', None)
            self.class_token_pooler = ClassWiseWeightedPooling(
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
            )
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)

        if self.cti_bgt:
            self.bg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed_bg = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.register_parameter('bg_token', None)
            self.register_parameter('pos_embed_bg', None)

        self.bcss_variant = bcss_variant.lower()
        self.bcss_spec = validate_bcss_variant(self.bcss_variant)
        if bcss_tau <= 0:
            raise ValueError('bcss_tau must be positive')
        if not 0 <= bcss_beta <= 1:
            raise ValueError('bcss_beta must be in [0, 1]')
        if bcss_lambda_fg < 0 or bcss_lambda_bg < 0:
            raise ValueError('BCSS loss weights must be non-negative')
        if bcss_semantic_temperature <= 0:
            raise ValueError('bcss_semantic_temperature must be positive')
        self.bcss_final_tau = bcss_tau
        self.bcss_final_beta = bcss_beta
        self.bcss_cls_threshold = bcss_cls_threshold
        self.bcss_lambda_fg = bcss_lambda_fg
        self.bcss_lambda_bg = bcss_lambda_bg
        self.bcss_semantic_temperature = bcss_semantic_temperature
        self.bcss_runtime = bcss_schedule(0, bcss_tau, bcss_beta)

        if self.bcss_spec.backbone_register:
            self.register_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed_register = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.register_token, std=.02)
            trunc_normal_(self.pos_embed_register, std=.02)
        else:
            self.register_parameter('register_token', None)
            self.register_parameter('pos_embed_register', None)

        if self.bcss_spec.backbone_background:
            self.background_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed_background = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.background_token, std=.02)
            trunc_normal_(self.pos_embed_background, std=.02)
        else:
            self.register_parameter('background_token', None)
            self.register_parameter('pos_embed_background', None)

        if self.bcss_spec.competitive_ownership:
            self.semantic_slot_decoder = SemanticSlotDecoder(
                dim=self.embed_dim,
                num_classes=self.num_classes,
                num_background_slots=bcss_num_background_slots,
                enable_slot_update=self.bcss_spec.slot_update,
            )
            self.semantic_slot_decoder.apply(self._init_weights)
            trunc_normal_(self.semantic_slot_decoder.background_slots, std=.02)
        else:
            self.semantic_slot_decoder = None

        self.psl_variant = psl_variant.lower()
        self.psl_spec = validate_psl_variant(self.psl_variant)
        if (self.final_norm or self.patch_final_norm) and self.psl_spec.enabled:
            raise ValueError(
                'MCTformer+ FinalLN ablations are defined only for the native '
                'joint-token '
                'MCTformer+ path, not persistent-semantic variants'
            )
        self.psl_interaction_layers = parse_interaction_layers(
            psl_interaction_layers)
        self.psl_relation_dim = int(psl_relation_dim)
        self.psl_num_background_latents = int(psl_num_background_latents)
        if self.psl_spec.enabled:
            if self.bcss_variant != 'e0':
                raise ValueError('Persistent semantic variants require BCSS E0')
            if self.attention_normalization != 'vanilla':
                raise ValueError(
                    'Persistent semantic Phase 2 requires vanilla patch attention')
            if self.psl_relation_dim != self.embed_dim:
                raise ValueError(
                    'Phase 2 fixes relation_dim equal to the patch width')
            if self.psl_num_background_latents != 1:
                raise ValueError('Phase 2 requires exactly one background latent')
            if self.psl_interaction_layers[-1] >= len(self.blocks):
                raise ValueError('Persistent semantic interaction layer is out of range')
            self.background_semantic_latent = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.background_semantic_latent, std=.02)
            self.semantic_interactions = nn.ModuleDict({
                str(layer): SemanticReadWrite(
                    dim=self.embed_dim,
                    relation_dim=self.psl_relation_dim,
                    read=self.psl_spec.read,
                    write=self.psl_spec.write,
                )
                for layer in self.psl_interaction_layers
            })
            self.semantic_interactions.apply(self._init_weights)
        else:
            self.register_parameter('background_semantic_latent', None)
            self.semantic_interactions = nn.ModuleDict()
        
        self.decay_parameter=decay_parameter

    def interpolate_pos_encoding(self, x, w, h):
        npatch = (
            x.shape[1] if self.psl_spec.enabled
            else x.shape[1] - self.num_classes
        )
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                size=(w0, h0),
                mode='bicubic')

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def set_bcss_epoch(self, epoch):
        self.bcss_runtime = bcss_schedule(
            epoch, self.bcss_final_tau, self.bcss_final_beta)

    def _foreground_slice(self):
        return slice(int(self.cti_bgt), self.num_class_tokens)

    def _patch_slice(self, patch_count):
        return slice(self.num_class_tokens, self.num_class_tokens + patch_count)

    def cti_bgt_configuration(self):
        return {
            'enabled': self.cti_bgt,
            'weight': self.cti_bgt_weight,
            'n_layers': self.cti_bgt_n_layers,
            'affinity_start': self.cti_bgt_affinity_start,
        }

    def last_mct_configuration(self):
        if not self.last_mct:
            return {'enabled': False}
        return {
            'enabled': True,
            **self.last_patch_aggregator.configuration(),
            'patch_final_norm': True,
            'classifier': 'shared Conv2d(D, 20, kernel_size=1) / F.linear',
        }

    def class_stable_last_configuration(self):
        if not self.class_stable_last:
            return {'enabled': False}
        return {
            'enabled': True,
            **self.class_stable_last_pooler.configuration(),
            'patch_final_norm': True,
            'classifier': 'shared Conv2d(D, 20, kernel_size=3, padding=1)',
            'classification_map': 'original M',
            'cam_map': 'original M',
        }

    def class_token_initialization_configuration(self):
        if self.class_token_init == 'baseline':
            return {
                'class_token_init': 'baseline',
                'deit_cls_token_used': True,
                'class_token_source_policy': 'repeated_for_all_classes',
            }
        return {
            'class_token_init': 'cwp',
            'deit_cls_token_used': False,
            'deit_cls_token_policy': 'discarded',
            'class_pooling': 'class-wise weighted pooling',
            'pooling_softmax_axis': 'patch',
            'class_query_shape': [self.num_classes, self.embed_dim],
            'class_query_initialization': 'trunc_normal_std_0.02',
        }

    def _cti_bgt_maps(self, patch_cam, attentions, labels=None):
        return cti_bgt_maps(
            patch_cam, attentions, labels,
            n_layers=self.cti_bgt_n_layers,
            affinity_start=self.cti_bgt_affinity_start)


    def _attention_patch_slice(self, patch_count):
        if self.psl_spec.enabled:
            return slice(0, patch_count)
        return self._patch_slice(patch_count)

    @torch.no_grad()
    def initialize_psl_from_backbone(self):
        """Initialize Phase 2 relations from the corresponding pretrained block."""
        if not self.psl_spec.enabled:
            return
        for layer in self.psl_interaction_layers:
            self.semantic_interactions[str(layer)].initialize_from_backbone_attention(
                self.blocks[layer].attn)

    def psl_configuration(self):
        return {
            'variant': self.psl_variant,
            'interaction_layers_zero_based': list(self.psl_interaction_layers),
            'semantic_dim': self.embed_dim,
            'patch_dim': self.embed_dim,
            'relation_dim': self.psl_relation_dim,
            'num_background_latents': self.psl_num_background_latents,
            'relation': 'shared',
            'ordering': 'read_then_write',
            'write_gate_initialization': 0.0,
        }

    def _forward_psl_features(self, x, return_aux):
        batch, _, width, height = x.shape
        patches = self.patch_embed(x)
        if not self.training:
            patches = patches + self.interpolate_pos_encoding(
                patches, width, height)
        else:
            patches = patches + self.pos_embed_pat
        patches = self.pos_drop(patches)

        foreground = self.cls_token.expand(batch, -1, -1) + self.pos_embed_cls
        background = self.background_semantic_latent.expand(batch, -1, -1)
        semantic = torch.cat((foreground, background), dim=1)
        attentions = []
        all_foreground_latents = []
        relations = []
        for layer, block in enumerate(self.blocks):
            patches, weights = block(patches)
            attentions.append(weights)
            if layer in self.psl_interaction_layers:
                semantic, patches, relation = self.semantic_interactions[
                    str(layer)](semantic, patches)
                relation['layer'] = layer
                relations.append(relation)
            all_foreground_latents.append(semantic[:, :self.num_classes])

        auxiliary = {
            'variant': self.bcss_variant,
            'patch_count': patches.shape[1],
            'psl': self.psl_configuration(),
            'psl_relations': relations,
            'background_semantic_latents': semantic[:, self.num_classes:],
            'semantic_latents': semantic,
        }
        result = (
            semantic[:, :self.num_classes], patches, attentions,
            all_foreground_latents,
        )
        if return_aux:
            return result + (auxiliary,)
        return result

    def forward_features(self, x, n=12, active_labels=None, return_aux=False):
        if self.psl_spec.enabled:
            return self._forward_psl_features(x, return_aux)
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        pooling_attention = None
        initial_patch_tokens = x
        if self.class_token_init == 'baseline':
            cls_tokens = self.cls_token.expand(B, -1, -1)
        else:
            cls_tokens, pooling_attention = self.class_token_pooler(x)
        cls_tokens = cls_tokens + self.pos_embed_cls
        initial_class_tokens = cls_tokens

        patch_count = x.shape[1]
        # BGT joins every joint self-attention block in [BG, FG, patches] order.
        if self.cti_bgt:
            bg = self.bg_token.expand(B, -1, -1) + self.pos_embed_bg
            cls_tokens = torch.cat((bg, cls_tokens), dim=1)
        token_parts = [cls_tokens, x]
        if self.bcss_spec.backbone_register:
            register = self.register_token.expand(B, -1, -1) + self.pos_embed_register
            token_parts.append(register)
        elif self.bcss_spec.backbone_background:
            background = self.background_token.expand(B, -1, -1) + self.pos_embed_background
            token_parts.append(background)
        x = torch.cat(token_parts, dim=1)
        x = self.pos_drop(x)
        attn_weights = []
        all_x_cls = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
            all_x_cls.append(x[:, self._foreground_slice()])
            
        # MCTformer+-FinalLN changes only the readout of the complete final
        # token sequence. Per-block class tokens above intentionally remain
        # raw post-block values for the unchanged CCT loss.
        final_tokens = self.norm(x) if self.final_norm else x
        x_cls = final_tokens[:, self._foreground_slice()]
        x_patch = final_tokens[:, self._patch_slice(patch_count)]
        if self.patch_final_norm:
            # LayerNorm acts independently on the final embedding dimension,
            # so normalizing only this slice is exactly the patch readout of
            # full-sequence FinalLN while the class readout remains raw.
            x_patch = self.norm(x_patch)
        auxiliary = {
            'variant': self.bcss_variant,
            'patch_count': patch_count,
            'final_norm': self.final_norm,
            'patch_final_norm': self.patch_final_norm,
            'last_mct': self.last_mct,
            'class_stable_last': self.class_stable_last,
            'class_token_init': self.class_token_init,
        }
        if pooling_attention is not None:
            auxiliary.update({
                'class_token_pooling_attention': pooling_attention,
                'initial_class_tokens': initial_class_tokens,
                'initial_patch_tokens': initial_patch_tokens,
                'class_token_pooling_diagnostics': (
                    class_token_pooling_diagnostics(
                        pooling_attention, initial_class_tokens
                    )
                ),
            })
        if self.bcss_spec.backbone_register:
            auxiliary['register_tokens'] = final_tokens[
                :, self.num_classes + patch_count:
            ]
        elif self.bcss_spec.backbone_background:
            auxiliary['background_tokens'] = final_tokens[
                :, self.num_classes + patch_count:
            ]

        if self.semantic_slot_decoder is not None:
            cls_logits = x_cls.mean(-1)
            if active_labels is None:
                active_classes = infer_active_classes(
                    cls_logits, self.bcss_cls_threshold)
            else:
                active_classes = active_labels > 0
            slot_outputs = self.semantic_slot_decoder(
                class_tokens=x_cls,
                patch_tokens=x_patch,
                active_classes=active_classes,
                tau=self.bcss_runtime['tau'],
                competitive=self.bcss_spec.competitive_ownership,
                refinement_strength=self.bcss_runtime['refinement_strength'],
            )
            auxiliary.update(slot_outputs)

        result = (x_cls, x_patch, attn_weights, all_x_cls)
        if return_aux:
            return result + (auxiliary,)
        return result
    
    def gwrp(self, x_patch):
        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)
        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter,
                                  device=x_patch.device)
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        return x_patch_logits

    def last_low_pass(self, patch_tokens):
        if not self.last_mct:
            raise RuntimeError('last_low_pass requires last_mct=True')
        return self.last_patch_aggregator.low_pass(patch_tokens)

    def last_stability_score(self, patch_tokens, low_pass_tokens):
        if not self.last_mct:
            raise RuntimeError('last_stability_score requires last_mct=True')
        return self.last_patch_aggregator.stability_score(
            patch_tokens, low_pass_tokens
        )

    def last_aggregate(self, patch_tokens):
        if not self.last_mct:
            raise RuntimeError('last_aggregate requires last_mct=True')
        return self.last_patch_aggregator(patch_tokens)

    def last_classify(self, pooled_token):
        if not self.last_mct:
            raise RuntimeError('last_classify requires last_mct=True')
        if self.head.kernel_size != (1, 1):
            raise RuntimeError('Last-MCT classifier must be a 1x1 convolution')
        return F.linear(
            pooled_token,
            self.head.weight[:, :, 0, 0],
            self.head.bias,
        )

    def class_stable_last_low_pass(self, patch_tokens):
        if not self.class_stable_last:
            raise RuntimeError(
                'class_stable_last_low_pass requires class_stable_last=True'
            )
        return self.class_stable_last_pooler.low_pass(patch_tokens)

    def class_stable_last_pool(self, patch_tokens, original_class_map):
        """Pool original 3x3 class maps using low-pass response stability."""
        if not self.class_stable_last:
            raise RuntimeError(
                'class_stable_last_pool requires class_stable_last=True'
            )
        if patch_tokens.ndim != 3 or original_class_map.ndim != 4:
            raise ValueError(
                'Expected patch tokens [B,N,D] and class map [B,C,H,W]'
            )
        batch, patch_count, width = patch_tokens.shape
        if original_class_map.shape[0] != batch:
            raise ValueError('Patch-token and class-map batch sizes must match')
        height, grid_width = original_class_map.shape[-2:]
        if patch_count != height * grid_width or width != self.embed_dim:
            raise ValueError(
                'Patch-token shape does not match the original class-map grid'
            )
        low_pass_tokens = self.class_stable_last_low_pass(patch_tokens)
        low_pass_grid = low_pass_tokens.reshape(
            batch, height, grid_width, width
        ).permute(0, 3, 1, 2).contiguous()
        # The same 3x3 classifier object produces M and M_lp. M_lp only
        # selects indices; selected classification values come from M.
        low_pass_class_map = self.head(low_pass_grid)[
            :, self._foreground_slice()
        ]
        return self.class_stable_last_pooler(
            original_class_map, low_pass_class_map
        )

    def forward(self, x, active_labels=None):
        w, h = x.shape[2:]
        x_cls, x_patch, attentions, all_x_cls, auxiliary = self.forward_features(
            x, active_labels=active_labels, return_aux=True)

        if self.last_mct:
            pooled_token, _, _ = self.last_aggregate(x_patch)
            x_patch_logits = self.last_classify(pooled_token)
        else:
            x_patch_tokens = x_patch
            n, p, c = x_patch.shape
            if w != h:
                w0 = w // self.patch_embed.patch_size[0]
                h0 = h // self.patch_embed.patch_size[0]
                x_patch = torch.reshape(x_patch, [n, w0, h0, c])
            else:
                x_patch = torch.reshape(
                    x_patch, [n, int(p ** 0.5), int(p ** 0.5), c]
                )
            x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
            x_patch = self.head(x_patch)
            if self.cti_bgt:
                if active_labels is not None:
                    auxiliary['cti_bgt'] = self._cti_bgt_maps(
                        x_patch, attentions, active_labels)
                # BG has no image-level label and no GWRP classification loss.
                x_patch = x_patch[:, 1:]
            if self.class_stable_last:
                x_patch_logits, _, _, _ = self.class_stable_last_pool(
                    # forward_features returns normalized tokens when the
                    # required patch-only FinalLN is enabled.
                    patch_tokens=x_patch_tokens,
                    original_class_map=x_patch[:, self._foreground_slice()],
                )
            else:
                x_patch_flattened = x_patch.view(
                    x_patch.shape[0], x_patch.shape[1], -1
                ).permute(0, 2, 1)
                sorted_patch_token, indices = torch.sort(
                    x_patch_flattened, -2, descending=True
                )
                weights = torch.logspace(
                    start=0, end=x_patch_flattened.size(-2) - 1,
                    steps=x_patch_flattened.size(-2), base=self.decay_parameter,
                    device=x_patch.device,
                )
                x_patch_logits = torch.sum(
                    sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1),
                    dim=-2,
                ) / weights.sum()
        x_cls_logits = x_cls.mean(-1)

        output = []
        output.append(x_cls_logits)
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)
        if (self.bcss_spec.competitive_ownership or 'cti_bgt' in auxiliary
                or self.class_token_init == 'cwp'):
            output.append(auxiliary)
        return output

    def bcss_losses(self, auxiliary, targets):
        if not self.bcss_spec.competitive_ownership:
            return {}
        classifier_weight = self.head.weight.mean(dim=(2, 3))
        return semantic_slot_losses(
            auxiliary=auxiliary,
            classifier_weight=classifier_weight,
            classifier_bias=self.head.bias,
            targets=targets,
            use_foreground_anchor=self.bcss_spec.foreground_anchor,
            use_background_null=self.bcss_spec.background_null,
            retain_foreground_ownership_mass=self.bcss_spec.foreground_mass_anchor,
            semantic_temperature=self.bcss_semantic_temperature,
        )


class MCTformerPlusCam(MCTformerPlus):
    """
        CAM Model for MCTformerPlus
    """
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
        """
        Basic Initialization
        """
        super().__init__(decay_parameter, input_size, *args, **kwargs)
        self.n_layers = 3

    def _psl_class_to_patch(self, auxiliary):
        relations = auxiliary.get('psl_relations', ())
        if not relations:
            raise RuntimeError('Persistent semantic CAM requires relation outputs')
        return torch.stack([
            item['read_attention'][:, :self.num_classes]
            for item in relations[-self.n_layers:]
        ]).mean(dim=0)
    
    @torch.no_grad()
    def get_cam(self, x_patch, attn_weights, auxiliary=None):
        feature_map = x_patch[:, self._foreground_slice()].detach().clone()  # FG only
        feature_map = F.relu(feature_map)
        
        n, c, h, w = feature_map.shape
        patch_slice = self._attention_patch_slice(h * w)
        if self.psl_spec.enabled:
            cls2pat = self._psl_class_to_patch(auxiliary)
        else:
            cls2pat = attn_weights[-self.n_layers:].mean(0)\
                [:, self._foreground_slice(), patch_slice]
        if auxiliary is not None and 'class_ownership' in auxiliary:
            cls2pat = ownership_calibrate_attention(
                cls2pat,
                auxiliary['class_ownership'],
                self.bcss_runtime['beta'],
            )
        elif auxiliary is not None and 'background_tokens' in auxiliary:
            background_index = self.num_classes + h * w
            background_attention = attn_weights[-self.n_layers:].mean(0)[
                :, background_index, patch_slice]
            background_attention = background_attention / background_attention.amax(
                dim=-1, keepdim=True).clamp_min(1e-6)
            foreground_gate = (1.0 - background_attention).unsqueeze(1).expand_as(cls2pat)
            cls2pat = ownership_calibrate_attention(
                cls2pat, foreground_gate, self.bcss_runtime['beta'])
        cls2pat = cls2pat.reshape([n, c, h, w])
        cams = cls2pat * feature_map  # B * C * 14 * 14
        cams = torch.sqrt(cams)
        
        patch_attn = attn_weights[:, :, patch_slice, patch_slice]
        patch_attn = torch.sum(patch_attn, dim=0) # B x Np x Np
        B, _, hp, wp = cams.shape
        cams = torch.matmul(
                patch_attn.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, self.num_classes, -1, 1) # B x Cls x Np x 1
        ).reshape(B, self.num_classes, hp, wp)
        return cams
    
    @torch.no_grad()
    def get_cls2pat(self, x_patch, attn_weights, auxiliary=None):
        feature_map = x_patch[:, self._foreground_slice()].detach().clone()  # FG only
        feature_map = F.relu(feature_map)
        n, c, h, w = feature_map.shape
        if self.psl_spec.enabled:
            cls2pat = self._psl_class_to_patch(auxiliary)
        else:
            cls2pat = attn_weights[-self.n_layers:].mean(0)[
                :, self._foreground_slice(), self._patch_slice(h * w)]
        if auxiliary is not None and 'class_ownership' in auxiliary:
            cls2pat = ownership_calibrate_attention(
                cls2pat, auxiliary['class_ownership'], self.bcss_runtime['beta'])
        elif auxiliary is not None and 'background_tokens' in auxiliary:
            background_index = self.num_classes + h * w
            background_attention = attn_weights[-self.n_layers:].mean(0)[
                :, background_index, self._patch_slice(h * w)]
            background_attention = background_attention / background_attention.amax(
                dim=-1, keepdim=True).clamp_min(1e-6)
            foreground_gate = (1.0 - background_attention).unsqueeze(1).expand_as(cls2pat)
            cls2pat = ownership_calibrate_attention(
                cls2pat, foreground_gate, self.bcss_runtime['beta'])
        cls2pat = cls2pat.reshape([n, c, h, w])
        return cls2pat
    
    @torch.no_grad()
    def forward_with_label(self, x, active_labels=None):
        b, _, w, h = x.shape
        x_cls_last, x_patch_tokens, attn_weights, _, auxiliary = self.forward_features(
            x, active_labels=active_labels, return_aux=True)
        cls_logits = x_cls_last.mean(-1) # [B, K]
        if self.last_mct:
            pooled_token, _, _ = self.last_aggregate(x_patch_tokens)
            x_logits = self.last_classify(pooled_token)

        n, p, c = x_patch_tokens.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch_tokens, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(
                x_patch_tokens, [n, int(p ** 0.5), int(p ** 0.5), c]
            )
        
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.head(x_patch)

        attn_weights = torch.mean(
            torch.stack(attn_weights), dim=2).detach()
        
        cls_label = torch.ones(b, self.num_classes).to(x.device)
        cls_label[cls_logits <= 0] = 0

        if self.class_stable_last:
            x_logits, _, _, _ = self.class_stable_last_pool(
                x_patch_tokens,
                x_patch[:, self._foreground_slice()],
            )
        elif not self.last_mct:
            x_logits = self.gwrp(x_patch[:, self._foreground_slice()])
        patch_label = torch.ones(b, self.num_classes).to(x.device)
        patch_label[x_logits <= 0] = 0
        outputs = self.get_cam(x_patch, attn_weights, auxiliary)

        return cls_label, patch_label, outputs
    
    @torch.no_grad()
    def forward(self, x, return_attn=False, return_token=False,
                active_labels=None, return_diagnostics=False):
        w, h = x.shape[2:]
        x_cls_last, x_patch_tokens, attn_weights, class_embeddings, auxiliary = self.forward_features(
            x, active_labels=active_labels, return_aux=True)
        # 12 * B * H * N * N -> 12 * B * N * N
        head_attention = torch.stack(attn_weights)
        attn_weights = torch.mean(head_attention, dim=2)
        if return_attn:
            return attn_weights
        if return_token:
            return class_embeddings

        n, p, c = x_patch_tokens.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            patch_grid = torch.reshape(x_patch_tokens, [n, w0, h0, c])
        else:
            patch_grid = torch.reshape(x_patch_tokens, [n, int(p ** 0.5), int(p ** 0.5), c])
        
        patch_grid = patch_grid.permute([0, 3, 1, 2]).contiguous()
        patch_cam = self.head(patch_grid)
        outputs = self.get_cam(patch_cam, attn_weights, auxiliary)
        if return_diagnostics:
            if self.cti_bgt:
                auxiliary['cti_bgt'] = self._cti_bgt_maps(
                    patch_cam, head_attention, active_labels)
            return self._diagnostic_outputs(
                x_cls_last, x_patch_tokens, patch_cam, attn_weights,
                auxiliary, outputs, patch_grid.shape[-2:], head_attention)
        return outputs

    def _diagnostic_outputs(self, x_cls, x_patch, patch_cam, attn_weights,
                            auxiliary, final_cam, grid_size, head_attention):
        hp, wp = grid_size
        patch_slice = self._attention_patch_slice(hp * wp)
        if self.psl_spec.enabled:
            relations = auxiliary['psl_relations']
            class_to_patch = self._psl_class_to_patch(auxiliary)
        else:
            class_to_patch = attn_weights[-self.n_layers:].mean(0)[
                :, self._foreground_slice(), patch_slice]
        result = {
            'class_logits': x_cls.mean(-1),
            'patch_cam': F.relu(patch_cam[:, self._foreground_slice()]),
            'class_to_patch': class_to_patch.reshape(
                x_cls.shape[0], self.num_classes, hp, wp),
            'final_cam': final_cam,
            'patch_feature_norm': x_patch.norm(dim=-1).reshape(x_cls.shape[0], hp, wp),
        }
        if self.psl_spec.enabled:
            result.update({
                'psl_relations': relations,
                'class_to_patch_layers': torch.stack([
                    item['read_attention'][:, :self.num_classes]
                    for item in relations
                ]),
                'patch_to_class_layers': torch.stack([
                    item['write_attention'][:, :, :self.num_classes]
                    for item in relations
                ]),
                'background_read_layers': torch.stack([
                    item['read_attention'][:, self.num_classes:]
                    for item in relations
                ]),
                'patch_to_background_layers': torch.stack([
                    item['write_attention'][:, :, self.num_classes:]
                    for item in relations
                ]),
                'semantic_latents': auxiliary['semantic_latents'],
                'write_gates': torch.stack([
                    item['write_gate'] for item in relations
                ]),
            })
        else:
            result.update({
                'class_to_patch_heads': head_attention[
                    :, :, :, self._foreground_slice(), patch_slice],
                'patch_to_class_heads': head_attention[
                    :, :, :, patch_slice, self._foreground_slice()],
            })
        if 'cti_bgt' in auxiliary:
            result['cti_bgt'] = auxiliary['cti_bgt']
            result['background_cam'] = F.relu(patch_cam[:, :1])
            result['background_to_patch'] = head_attention[:, :, :, 0, patch_slice]
            result['patch_to_background'] = head_attention[:, :, :, patch_slice, 0]
        if 'register_tokens' in auxiliary:
            register_index = self.num_classes + hp * wp
            result['register_to_patch'] = head_attention[
                :, :, :, register_index, patch_slice]
            result['patch_to_register'] = head_attention[
                :, :, :, patch_slice, register_index]
        if 'background_tokens' in auxiliary:
            background_index = self.num_classes + hp * wp
            result['background_to_patch'] = head_attention[
                :, :, :, background_index, patch_slice]
            result['patch_to_background'] = head_attention[
                :, :, :, patch_slice, background_index]
            result['background_attention'] = head_attention[
                -self.n_layers:, :, :, background_index, patch_slice
            ].mean(dim=(0, 2)).reshape(x_cls.shape[0], 1, hp, wp)
        if 'background_raw_score' in auxiliary:
            result['background_raw_score'] = auxiliary['background_raw_score'].reshape(
                x_cls.shape[0], -1, hp, wp)
            result['background_attention'] = auxiliary['background_attention'].reshape(
                x_cls.shape[0], -1, hp, wp)
        if 'class_ownership' in auxiliary:
            result['background_raw_score'] = auxiliary['energies'][
                :, self.num_classes:].reshape(x_cls.shape[0], -1, hp, wp)
            result['class_ownership'] = auxiliary['class_ownership'].reshape(
                x_cls.shape[0], self.num_classes, hp, wp)
            result['background_ownership'] = auxiliary['background_ownership'].reshape(
                x_cls.shape[0], hp, wp)
            result['ownership'] = auxiliary['ownership'].reshape(
                x_cls.shape[0], -1, hp, wp)
        return result
    
    @torch.no_grad()
    def forward_ablation(self, x, return_type='all'):
        """
        One can choose return_type as:
            'cam': return cam for testing
            'all': whole attention map
            'cls_token': the class token
            'cls2cls': class-to-class attention map
            'cls2pat': class-to-patch attention map
            'pat2cls': patch-to-class attention map
            'pat2pat': patch-to-patch attention map
        """
        b, _, w, h = x.shape
        x_cls_last, x_patch, attn_weights, class_embeddings, auxiliary = (
            self.forward_features(x, return_aux=True)
        )
        # 12 * B * H * N * N -> 12 * B * N * N
        attn_weights = torch.mean(torch.stack(attn_weights), dim=2)
        if return_type == 'all':
            return attn_weights
        
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.head(x_patch)
        outputs = self.get_cam(x_patch, attn_weights, auxiliary)
        return outputs

        
def model_spec_from_instance(model):
    """Describe a concrete MCTformer+ instance for checkpoint provenance."""
    if not isinstance(model, MCTformerPlus):
        raise TypeError(f'Expected MCTformerPlus, got {type(model).__name__}')
    depth = len(model.blocks)
    if depth < 1:
        raise ValueError('MCTformer+ must contain at least one transformer block')
    num_heads = int(model.blocks[0].attn.num_heads)
    if any(int(block.attn.num_heads) != num_heads for block in model.blocks):
        raise ValueError('MCTformer+ blocks have inconsistent attention head counts')
    embed_dim = int(model.embed_dim)
    patch_size = [int(value) for value in model.patch_embed.patch_size]
    mlp_ratio = model.blocks[0].mlp.fc1.out_features / embed_dim
    if not float(mlp_ratio).is_integer():
        raise ValueError(f'Non-integral MLP ratio: {mlp_ratio}')
    matches = [
        variant for variant, candidate in MCTFORMERPLUS_VARIANTS.items()
        if candidate['embed_dim'] == embed_dim
        and candidate['depth'] == depth
        and candidate['num_heads'] == num_heads
        and [candidate['patch_size'], candidate['patch_size']] == patch_size
        and candidate['mlp_ratio'] == int(mlp_ratio)
    ]
    if len(matches) != 1:
        raise ValueError(
            'Concrete model does not match exactly one registered MCTformer+ variant: '
            f'embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, '
            f'patch_size={patch_size}, mlp_ratio={mlp_ratio}'
        )
    variant = matches[0]
    registry = MCTFORMERPLUS_VARIANTS[variant]
    return {
        'family': 'MCTformer+',
        'variant': variant,
        'model_name': registry['model_name'],
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'head_dim': embed_dim // num_heads,
        'mlp_ratio': int(mlp_ratio),
        'cam_class_to_patch_layers': int(
            getattr(model, 'n_layers', 3)
        ),
        'cam_patch_to_patch_layers': depth,
        'class_token_init': model.class_token_init,
    }


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    state = checkpoint.get('model', checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise TypeError('Checkpoint model state must be a non-empty mapping')
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise TypeError('Checkpoint state must map string keys to tensors')
    prefixes = [key.startswith('module.') for key in state]
    if any(prefixes):
        if not all(prefixes):
            raise ValueError(
                'Checkpoint mixes module.-prefixed and unprefixed state keys'
            )
        state = OrderedDict(
            (key[len('module.'):], value) for key, value in state.items()
        )
    return state


def checkpoint_final_norm_enabled(checkpoint):
    """Return the recorded FinalLN state, defaulting legacy checkpoints off."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    value = checkpoint.get('final_norm', False)
    if not isinstance(value, bool):
        raise TypeError('checkpoint final_norm metadata must be a boolean')
    return value


def checkpoint_patch_final_norm_enabled(checkpoint):
    """Return the recorded patch-only FinalLN state, defaulting legacy off."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    value = checkpoint.get('patch_final_norm', False)
    if not isinstance(value, bool):
        raise TypeError(
            'checkpoint patch_final_norm metadata must be a boolean'
        )
    return value


def checkpoint_last_mct_enabled(checkpoint):
    """Return the recorded Last-MCT state, defaulting legacy checkpoints off."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    value = checkpoint.get('last_mct', False)
    if not isinstance(value, bool):
        raise TypeError('checkpoint last_mct metadata must be a boolean')
    return value


def checkpoint_class_stable_last_enabled(checkpoint):
    """Return recorded Class-Stable LaST state, defaulting legacy off."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    value = checkpoint.get('class_stable_last', False)
    if not isinstance(value, bool):
        raise TypeError(
            'checkpoint class_stable_last metadata must be a boolean'
        )
    return value


def checkpoint_class_token_init(checkpoint):
    """Return and validate the class-token initialization checkpoint contract."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f'Checkpoint must be a mapping, got {type(checkpoint).__name__}'
        )
    state = _checkpoint_state_dict(checkpoint)
    has_baseline = 'cls_token' in state
    has_cwp = 'class_token_pooler.class_queries' in state
    if has_baseline == has_cwp:
        raise ValueError(
            'Checkpoint must contain exactly one of cls_token or '
            'class_token_pooler.class_queries'
        )
    state_value = 'cwp' if has_cwp else 'baseline'
    top_value = checkpoint.get('class_token_init')
    model_spec = checkpoint.get('model_spec')
    spec_value = (
        model_spec.get('class_token_init')
        if isinstance(model_spec, Mapping) else None
    )
    recorded = [value for value in (top_value, spec_value) if value is not None]
    if state_value == 'cwp' and (top_value is None or spec_value is None):
        raise ValueError(
            'CWP checkpoints require explicit class_token_init=cwp in both '
            'top-level metadata and model_spec'
        )
    for value in recorded:
        if not isinstance(value, str) or value not in {'baseline', 'cwp'}:
            raise ValueError(
                'checkpoint class_token_init must be baseline or cwp'
            )
        if value != state_value:
            raise ValueError(
                f'checkpoint class_token_init={value!r} conflicts with '
                f'state architecture {state_value!r}'
            )
    return state_value


def validate_mctformerplus_class_token_init_checkpoint(checkpoint, expected):
    expected = str(expected).strip().lower()
    if expected not in {'baseline', 'cwp'}:
        raise ValueError('expected class_token_init must be baseline or cwp')
    observed = checkpoint_class_token_init(checkpoint)
    if observed != expected:
        raise ValueError(
            f'Checkpoint class_token_init={observed!r} does not match '
            f'requested {expected!r}'
        )
    return observed


def validate_mctformerplus_final_norm_checkpoint(
        checkpoint, expected, expected_patch=False, expected_last_mct=False,
        expected_class_stable_last=False):
    """Require checkpoint and requested FinalLN/LaST scopes to match."""

    if not all(isinstance(value, bool) for value in (
            expected, expected_patch, expected_last_mct,
            expected_class_stable_last)):
        raise TypeError('expected FinalLN and LaST states must be booleans')
    if expected_last_mct and expected_class_stable_last:
        raise ValueError(
            'requested last_mct and class_stable_last are mutually exclusive'
        )
    if expected and expected_patch:
        raise ValueError(
            'requested final_norm and patch_final_norm are mutually exclusive'
        )
    observed = checkpoint_final_norm_enabled(checkpoint)
    observed_patch = checkpoint_patch_final_norm_enabled(checkpoint)
    observed_last_mct = checkpoint_last_mct_enabled(checkpoint)
    observed_class_stable = checkpoint_class_stable_last_enabled(checkpoint)
    if observed_last_mct and observed_class_stable:
        raise ValueError(
            'checkpoint last_mct and class_stable_last cannot both be true'
        )
    if observed and observed_patch:
        raise ValueError(
            'checkpoint final_norm and patch_final_norm cannot both be true'
        )
    if observed_last_mct and (observed or not observed_patch):
        raise ValueError(
            'checkpoint Last-MCT requires patch_final_norm=true and '
            'final_norm=false'
        )
    if expected_last_mct and (expected or not expected_patch):
        raise ValueError(
            'requested Last-MCT requires patch_final_norm=True and '
            'final_norm=False'
        )
    if observed_class_stable and (observed or not observed_patch):
        raise ValueError(
            'checkpoint Class-Stable LaST requires patch_final_norm=true and '
            'final_norm=false'
        )
    if expected_class_stable_last and (expected or not expected_patch):
        raise ValueError(
            'requested Class-Stable LaST requires patch_final_norm=True and '
            'final_norm=False'
        )
    if (observed, observed_patch, observed_last_mct,
            observed_class_stable) != (
            expected, expected_patch, expected_last_mct,
            expected_class_stable_last):
        raise ValueError(
            'Checkpoint FinalLN/LaST state '
            f'(final_norm={observed}, patch_final_norm={observed_patch}, '
            f'last_mct={observed_last_mct}, '
            f'class_stable_last={observed_class_stable}) '
            'does not match requested state '
            f'(final_norm={expected}, patch_final_norm={expected_patch}, '
            f'last_mct={expected_last_mct}, '
            f'class_stable_last={expected_class_stable_last})'
        )
    return observed


def _state_architecture(state):
    required = ('patch_embed.proj.weight', 'blocks.0.attn.qkv.weight')
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f'Checkpoint lacks architecture keys: {missing}')
    has_baseline = 'cls_token' in state
    has_cwp = 'class_token_pooler.class_queries' in state
    if has_baseline == has_cwp:
        raise ValueError(
            'Checkpoint must contain exactly one class-token initializer key'
        )
    embed_dim = int(state['patch_embed.proj.weight'].shape[0])
    patch_weight = state['patch_embed.proj.weight']
    if patch_weight.ndim != 4:
        raise ValueError('patch_embed.proj.weight must be four-dimensional')
    blocks = sorted({
        int(key.split('.')[1]) for key in state
        if key.startswith('blocks.') and key.split('.')[1].isdigit()
    })
    return {
        'embed_dim': embed_dim,
        'depth': len(blocks),
        'block_indices': blocks,
        'patch_size': [int(patch_weight.shape[-2]), int(patch_weight.shape[-1])],
        'class_token_init': 'cwp' if has_cwp else 'baseline',
        'class_token_count': int(
            state['class_token_pooler.class_queries'].shape[0]
            if has_cwp else state['cls_token'].shape[1]
        ),
        'class_token_initializer_shape': list(
            state['class_token_pooler.class_queries'].shape
            if has_cwp else state['cls_token'].shape
        ),
        'qkv_shape': list(state['blocks.0.attn.qkv.weight'].shape),
    }


def resolve_mctformerplus_checkpoint_variant(checkpoint, model_name):
    """Validate CLI/checkpoint architecture and return import provenance.

    A checkpoint without ``model_spec`` is accepted only for the canonical
    legacy Small CLI name.  Tiny/Base are never inferred from a bare width.
    """
    requested_variant = resolve_mctformerplus_variant(model_name)
    requested_name = str(model_name).strip().lower()
    state = _checkpoint_state_dict(checkpoint)
    observed = _state_architecture(state)
    observed_class_token_init = checkpoint_class_token_init(checkpoint)
    metadata = checkpoint.get('model_spec') if isinstance(checkpoint, Mapping) else None
    legacy = metadata is None
    if legacy:
        if requested_name != MCTFORMERPLUS_VARIANTS['small']['model_name']:
            raise ValueError(
                'MCTformer+ checkpoint lacks model_spec; legacy import is allowed '
                'only with --model mctformerplus'
            )
        resolved_variant = 'small'
    else:
        if not isinstance(metadata, Mapping):
            raise TypeError('checkpoint model_spec must be a mapping')
        if metadata.get('family') != 'MCTformer+':
            raise ValueError(
                f'checkpoint model_spec.family={metadata.get("family")!r}, '
                'expected MCTformer+'
            )
        resolved_variant = resolve_mctformerplus_variant(
            metadata.get('variant', metadata.get('model_name', ''))
        )
        canonical_name = MCTFORMERPLUS_VARIANTS[resolved_variant]['model_name']
        if metadata.get('model_name') != canonical_name:
            raise ValueError(
                f'checkpoint model_spec.model_name={metadata.get("model_name")!r} '
                f'does not match canonical {canonical_name!r}'
            )
        if resolved_variant != requested_variant:
            raise ValueError(
                f'Checkpoint variant {resolved_variant!r} does not match requested '
                f'{requested_variant!r}'
            )
    spec = get_mctformerplus_spec(resolved_variant)
    expected_observed = {
        'embed_dim': spec['embed_dim'],
        'depth': spec['depth'],
        'block_indices': list(range(spec['depth'])),
        'patch_size': [spec['patch_size'], spec['patch_size']],
        'class_token_count': 20,
        'qkv_shape': [3 * spec['embed_dim'], spec['embed_dim']],
        'class_token_init': observed_class_token_init,
        'class_token_initializer_shape': (
            [20, spec['embed_dim']]
            if observed_class_token_init == 'cwp'
            else [1, 20, spec['embed_dim']]
        ),
    }
    mismatches = {
        key: {'observed': observed[key], 'expected': value}
        for key, value in expected_observed.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f'Checkpoint state architecture mismatch: {mismatches}')
    if metadata is not None:
        expected_metadata = {
            'variant': resolved_variant,
            'model_name': spec['model_name'],
            'patch_size': [spec['patch_size'], spec['patch_size']],
            'embed_dim': spec['embed_dim'],
            'depth': spec['depth'],
            'num_heads': spec['num_heads'],
            'head_dim': spec['head_dim'],
            'mlp_ratio': spec['mlp_ratio'],
            'cam_class_to_patch_layers': 3,
            'cam_patch_to_patch_layers': spec['depth'],
        }
        if ('class_token_init' in metadata
                or observed_class_token_init == 'cwp'):
            expected_metadata['class_token_init'] = observed_class_token_init
        metadata_mismatches = {
            key: {'observed': metadata.get(key), 'expected': value}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if metadata_mismatches:
            raise ValueError(
                f'Checkpoint model_spec mismatch: {metadata_mismatches}'
            )
    return {
        'variant': resolved_variant,
        'model_name': spec['model_name'],
        'legacy_small_import': legacy,
        'model_spec_present': not legacy,
        'state_architecture': observed,
        'class_token_init': observed_class_token_init,
        'legacy_small_import_manifest': (
            {
                'status': 'legacy_small_import',
                'reason': 'checkpoint lacks model_spec',
                'required_cli_model': 'mctformerplus',
                'resolved_variant': 'small',
                'observed_state_architecture': observed,
            }
            if legacy else None
        ),
    }


def adapt_deit_checkpoint_for_mctformerplus(checkpoint, model, num_classes=20):
    """Adapt one official non-distilled DeiT state to a baseline MCTformer+."""
    if not isinstance(model, MCTformerPlus):
        raise TypeError(f'Expected MCTformerPlus, got {type(model).__name__}')
    source = _checkpoint_state_dict(checkpoint)
    target = model.state_dict()
    model_spec = model_spec_from_instance(model)
    variant_spec = get_mctformerplus_spec(model_spec['variant'])
    if num_classes != model.num_classes:
        raise ValueError(
            f'num_classes={num_classes} does not match model.num_classes={model.num_classes}'
        )
    required_source = {'cls_token', 'pos_embed', 'patch_embed.proj.weight'}
    absent = sorted(required_source - set(source))
    if absent:
        raise ValueError(f'DeiT checkpoint lacks required keys: {absent}')
    source_embed_dim = int(source['cls_token'].shape[-1])
    source_blocks = sorted({
        int(key.split('.')[1]) for key in source
        if key.startswith('blocks.') and key.split('.')[1].isdigit()
    })
    source_depth = len(source_blocks)
    failures = []
    if source_embed_dim != model.embed_dim:
        failures.append(
            f'source embed_dim {source_embed_dim} != target {model.embed_dim}'
        )
    if source_blocks != list(range(len(model.blocks))):
        failures.append(
            f'source block indices {source_blocks} != target '
            f'{list(range(len(model.blocks)))}'
        )
    cls_token = source['cls_token']
    position = source['pos_embed']
    if list(cls_token.shape) != [1, 1, model.embed_dim]:
        failures.append(f'invalid source cls_token shape {list(cls_token.shape)}')
    if position.ndim != 3 or position.shape[0] != 1 or position.shape[2] != model.embed_dim:
        failures.append(f'invalid source pos_embed shape {list(position.shape)}')
    source_patch_count = int(position.shape[1] - 1)
    source_side = math.isqrt(source_patch_count)
    if source_side * source_side != source_patch_count:
        failures.append(
            f'source positional patch count {source_patch_count} is not square'
        )
    if failures:
        raise ValueError('; '.join(failures))

    source_cls_position = position[:, :1].repeat(1, num_classes, 1)
    source_patch_position = position[:, 1:].reshape(
        1, source_side, source_side, model.embed_dim
    ).permute(0, 3, 1, 2)
    source_patch_position = F.interpolate(
        source_patch_position,
        size=(model.Hp, model.Wp),
        mode='bicubic',
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(1, 2)
    repeated_cls_token = cls_token.repeat(1, num_classes, 1)

    derived = {
        'pos_embed_cls': source_cls_position,
        'pos_embed_pat': source_patch_position,
    }
    if model.class_token_init == 'baseline':
        derived['cls_token'] = repeated_cls_token
    random_keys = {'head.weight', 'head.bias'}
    if model.class_token_init == 'cwp':
        random_keys.add('class_token_pooler.class_queries')
    ignored_classifier_keys = {
        'head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias'
    }
    ignored_architecture_keys = (
        {'cls_token'} if model.class_token_init == 'cwp' else set()
    )
    adapted = OrderedDict()
    shape_mismatches = []
    missing_source_keys = []
    loaded_numel = 0
    for key, target_value in target.items():
        if key in random_keys:
            adapted[key] = target_value
            continue
        if key in derived:
            value = derived[key]
        elif key in source:
            value = source[key]
        else:
            missing_source_keys.append(key)
            continue
        if value.shape != target_value.shape:
            shape_mismatches.append({
                'key': key,
                'source_shape': list(value.shape),
                'target_shape': list(target_value.shape),
            })
            continue
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite tensor in pretrained source key {key}')
        adapted[key] = value
        loaded_numel += value.numel()
    unexpected_keys = sorted(
        key for key in source
        if key not in target
        and key not in ignored_classifier_keys
        and key not in ignored_architecture_keys
    )
    if missing_source_keys or shape_mismatches or unexpected_keys:
        raise ValueError(
            'Official DeiT adaptation failed: '
            f'missing={missing_source_keys}, shape_mismatches={shape_mismatches}, '
            f'unexpected={unexpected_keys}'
        )
    if set(adapted) != set(target):
        raise RuntimeError(
            f'Adapted state keys differ from model keys: '
            f'missing={sorted(set(target) - set(adapted))}, '
            f'extra={sorted(set(adapted) - set(target))}'
        )
    report = {
        'variant': model_spec['variant'],
        'model_name': model_spec['model_name'],
        'source_url': variant_spec['pretrained_url'],
        'source_embed_dim': source_embed_dim,
        'target_embed_dim': model.embed_dim,
        'source_depth': source_depth,
        'target_depth': len(model.blocks),
        'source_patch_position_shape': list(position[:, 1:].shape),
        'target_patch_position_shape': list(source_patch_position.shape),
        'target_class_token_shape': (
            list(repeated_cls_token.shape)
            if model.class_token_init == 'baseline' else None
        ),
        'loaded_key_count': len(adapted) - len(random_keys),
        'loaded_numel': int(loaded_numel),
        'randomly_initialized_keys': sorted(random_keys),
        'ignored_source_classifier_keys': sorted(
            key for key in source if key in ignored_classifier_keys
        ),
        'missing_source_keys': [],
        'unexpected_keys': [],
        'shape_mismatches': [],
        **model.class_token_initialization_configuration(),
        'passed': True,
    }
    return adapted, report


def build_mctformerplus(variant, cam=False, pretrained=False, **kwargs):
    """Build a registered-width MCTformer+ training or CAM model."""
    spec, constructor_kwargs = _variant_constructor_kwargs(variant, kwargs)
    model_class = MCTformerPlusCam if cam else MCTformerPlus
    model = model_class(**constructor_kwargs)
    model.default_cfg = _cfg(url=spec['pretrained_url'])
    model.mctformerplus_variant = spec['variant']
    model.mctformerplus_model_name = spec['model_name']
    model.mctformerplus_pretrained_url = spec['pretrained_url']
    model.pretrained_load_report = None
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=spec['pretrained_url'], map_location='cpu', check_hash=True
        )
        adapted, report = adapt_deit_checkpoint_for_mctformerplus(
            checkpoint, model, num_classes=model.num_classes
        )
        incompatible = model.load_state_dict(adapted, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Unexpected strict-load result: {incompatible}')
        model.pretrained_load_report = report
    return model


@register_model
def mctformerplus_tiny(pretrained=False, **kwargs):
    return build_mctformerplus('tiny', pretrained=pretrained, **kwargs)


@register_model
def mctformerplus(pretrained=False, **kwargs):
    """Canonical legacy name for the MCTformer+-Small architecture."""
    return build_mctformerplus('small', pretrained=pretrained, **kwargs)


@register_model
def mctformerplus_base(pretrained=False, **kwargs):
    return build_mctformerplus('base', pretrained=pretrained, **kwargs)
