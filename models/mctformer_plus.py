import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F
from models.vit import VisionTransformer, _cfg

__all__ = ['mctformerplus']

class MCTformerPlus(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
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

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        attn_weights = []
        all_x_cls = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
            all_x_cls.append(x[:, :self.num_classes])
            
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, all_x_cls
    
    def gwrp(self, x_patch):
        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)
        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        return x_patch_logits

    def forward(self, x):
        w, h = x.shape[2:]
        x_cls, x_patch, _, all_x_cls = self.forward_features(x)

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
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        x_cls_logits = x_cls.mean(-1)

        output = []
        output.append(x_cls_logits)
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)
        return output


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
    def get_cam(self, x_patch, attn_weights):
        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)
        n, c, h, w = feature_map.shape
        
        cls2pat = attn_weights[-self.n_layers:].mean(0)\
            [:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        cams = cls2pat * feature_map  # B * C * 14 * 14
        cams = torch.sqrt(cams)
        
        patch_attn = torch.sum(patch_attn, dim=0) # B x Np x Np
        B, _, hp, wp = cams.shape
        cams = torch.matmul(
                patch_attn.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, self.num_classes, -1, 1) # B x Cls x Np x 1
        ).reshape(B, self.num_classes, hp, wp)
        return cams
    
    @torch.no_grad()
    def get_cls2pat(self, x_patch, attn_weights):
        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)
        n, c, h, w = feature_map.shape
        cls2pat = attn_weights[-self.n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([
            n, c, h, w])
        return cls2pat
    
    @torch.no_grad()
    def forward_with_label(self, x):
        b, _, w, h = x.shape
        x_cls_last, x_patch, attn_weights, _ = self.forward_features(x)
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
        outputs = self.get_cam(x_patch, attn_weights)

        return cls_label, patch_label, outputs
    
    @torch.no_grad()
    def forward(self, x, return_attn=False, return_token=False):
        w, h = x.shape[2:]
        x_cls_last, x_patch, attn_weights, class_embeddings = self.forward_features(x)
        # 12 * B * H * N * N -> 12 * B * N * N
        attn_weights = torch.mean(torch.stack(attn_weights), dim=2)
        if return_attn:
            return attn_weights
        if return_token:
            return class_embeddings

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