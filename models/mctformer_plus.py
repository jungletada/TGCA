import math
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

__all__ = ['mctformerplus']

class MCTformerPlus(VisionTransformer):
    def __init__(
            self, decay_parameter=0.996, input_size=448,
            bcss_variant='e0', bcss_num_background_slots=1,
            bcss_tau=0.5, bcss_beta=0.5, bcss_cls_threshold=0.5,
            bcss_lambda_fg=0.5, bcss_lambda_bg=0.1,
            bcss_semantic_temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
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
        
        self.decay_parameter=decay_parameter

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
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

    def _patch_slice(self, patch_count):
        return slice(self.num_classes, self.num_classes + patch_count)

    def forward_features(self, x, n=12, active_labels=None, return_aux=False):
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
            all_x_cls.append(x[:, :self.num_classes])
            
        x_cls = x[:, :self.num_classes]
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
        x_cls, x_patch, _, all_x_cls, auxiliary = self.forward_features(
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
        if self.bcss_spec.competitive_ownership:
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
    
    @torch.no_grad()
    def get_cam(self, x_patch, attn_weights, auxiliary=None):
        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)
        
        n, c, h, w = feature_map.shape
        patch_slice = self._patch_slice(h * w)
        cls2pat = attn_weights[-self.n_layers:].mean(0)\
            [:, 0:self.num_classes, patch_slice]
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
        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)
        n, c, h, w = feature_map.shape
        cls2pat = attn_weights[-self.n_layers:].mean(0)[
            :, 0:self.num_classes, self._patch_slice(h * w)]
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

        x_logits = self.gwrp(x_patch)
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
            return self._diagnostic_outputs(
                x_cls_last, x_patch_tokens, patch_cam, attn_weights,
                auxiliary, outputs, patch_grid.shape[-2:], head_attention)
        return outputs

    def _diagnostic_outputs(self, x_cls, x_patch, patch_cam, attn_weights,
                            auxiliary, final_cam, grid_size, head_attention):
        hp, wp = grid_size
        patch_slice = self._patch_slice(hp * wp)
        class_to_patch = attn_weights[-self.n_layers:].mean(0)[
            :, :self.num_classes, patch_slice]
        result = {
            'class_logits': x_cls.mean(-1),
            'patch_cam': F.relu(patch_cam),
            'class_to_patch': class_to_patch.reshape(
                x_cls.shape[0], self.num_classes, hp, wp),
            'final_cam': final_cam,
            'patch_feature_norm': x_patch.norm(dim=-1).reshape(x_cls.shape[0], hp, wp),
            'class_to_patch_heads': head_attention[
                :, :, :, :self.num_classes, patch_slice],
            'patch_to_class_heads': head_attention[
                :, :, :, patch_slice, :self.num_classes],
        }
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
        x_cls_last, x_patch, attn_weights, class_embeddings = self.forward_features(x)
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
        outputs = self.get_cam(x_patch, attn_weights)
        return outputs

        
@register_model
def mctformerplus(pretrained=False, **kwargs):
    """Creates a MCTformerPlus model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments passed to the model
        
    Returns:
        MCTformerPlus: The constructed model
    """
    model = MCTformerPlus(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
