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

__all__ = [
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
    'resolve_mctformerplus_checkpoint_variant',
    'resolve_mctformerplus_variant',
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

class MCTformerPlus(VisionTransformer):
    def __init__(
            self, decay_parameter=0.996, input_size=448,
            bcss_variant='e0', bcss_num_background_slots=1,
            bcss_tau=0.5, bcss_beta=0.5, bcss_cls_threshold=0.5,
            bcss_lambda_fg=0.5, bcss_lambda_bg=0.1,
            bcss_semantic_temperature=1.0, psl_variant='baseline',
            psl_interaction_layers=(11,), psl_relation_dim=384,
            psl_num_background_latents=1, cti_bgt=False, cti_bgt_weight=0.1,
            cti_bgt_n_layers=6, cti_bgt_affinity_start=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.head = nn.Conv2d(self.embed_dim, self.num_class_tokens, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        self.Hp, self.Wp = math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1])
        self.num_patches = self.Hp * self.Wp

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

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

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

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
            
        x_cls = x[:, self._foreground_slice()]
        x_patch = x[:, self._patch_slice(patch_count)]
        auxiliary = {
            'variant': self.bcss_variant,
            'patch_count': patch_count,
        }
        if self.bcss_spec.backbone_register:
            auxiliary['register_tokens'] = x[:, self.num_classes + patch_count:]
        elif self.bcss_spec.backbone_background:
            auxiliary['background_tokens'] = x[:, self.num_classes + patch_count:]

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

    def forward(self, x, active_labels=None):
        w, h = x.shape[2:]
        x_cls, x_patch, attentions, all_x_cls, auxiliary = self.forward_features(
            x, active_labels=active_labels, return_aux=True)

        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.head(x_patch)
        if self.cti_bgt:
            if active_labels is not None:
                auxiliary['cti_bgt'] = self._cti_bgt_maps(
                    x_patch, attentions, active_labels)
            # BG has no image-level label and no GWRP classification loss.
            x_patch = x_patch[:, 1:]
        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)
        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter,
                                  device=x_patch.device)
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        x_cls_logits = x_cls.mean(-1)

        output = []
        output.append(x_cls_logits)
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)
        if self.bcss_spec.competitive_ownership or 'cti_bgt' in auxiliary:
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
        x_cls_last, x_patch, attn_weights, _, auxiliary = self.forward_features(
            x, active_labels=active_labels, return_aux=True)
        cls_logits = x_cls_last.mean(-1) # [B, K]
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.head(x_patch)

        attn_weights = torch.mean(
            torch.stack(attn_weights), dim=2).detach()
        
        cls_label = torch.ones(b, self.num_classes).to(x.device)
        cls_label[cls_logits <= 0] = 0

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


def _state_architecture(state):
    required = ('cls_token', 'patch_embed.proj.weight', 'blocks.0.attn.qkv.weight')
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f'Checkpoint lacks architecture keys: {missing}')
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
        'class_token_count': int(state['cls_token'].shape[1]),
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
        'cls_token': repeated_cls_token,
        'pos_embed_cls': source_cls_position,
        'pos_embed_pat': source_patch_position,
    }
    random_keys = {'head.weight', 'head.bias'}
    ignored_classifier_keys = {
        'head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias'
    }
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
        key for key in source if key not in target and key not in ignored_classifier_keys
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
        'target_class_token_shape': list(repeated_cls_token.shape),
        'loaded_key_count': len(adapted) - len(random_keys),
        'loaded_numel': int(loaded_numel),
        'randomly_initialized_keys': sorted(random_keys),
        'ignored_source_classifier_keys': sorted(
            key for key in source if key in ignored_classifier_keys
        ),
        'missing_source_keys': [],
        'unexpected_keys': [],
        'shape_mismatches': [],
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
