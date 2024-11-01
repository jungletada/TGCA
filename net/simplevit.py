import math
import torch
import torch.nn as nn
from functools import partial
from net.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F


__all__ = ['simplevit']


class SimpleViT(VisionTransformer):
    def __init__(self, input_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        self.Hp, self.Wp = math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1])
        self.num_patches = self.Hp * self.Wp
        if self.cls_token is not None:
            del self.cls_token
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        trunc_normal_(self.pos_embed_pat, std=.02)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def interpolate_pos_encoding(self, patch_tokens, token_size):
        if self.Hp == token_size[0] and self.Wp == token_size[1]:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = patch_tokens.shape[-1]
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, self.Hp, self.Wp, dim).permute(0, 3, 1, 2),
            size=token_size,
            mode='bilinear',
            align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x, n=12):
        H, W = x.shape[2:]   # B x 3 x H x W
        x = self.patch_embed(x)
        token_size = (H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x,token_size)
            x = x + pos_embed_pat
        else: 
            x = x + self.pos_embed_pat

        x = self.pos_drop(x) # B x (N') x C, where N' = Nc + Np
        attn_weights = []
        for blk in self.blocks:
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
        # [B * Nc * C], [B * Np * C], list[B * Hd * N' * N']
        return x, attn_weights

    def reshape_patch_tokens(self, patch_tokens, H, W):
        B, _, C = patch_tokens.shape
        Hp = H // self.patch_embed.patch_size[0]
        Wp = W // self.patch_embed.patch_size[1]
        patch_tokens = torch.reshape(patch_tokens, [B, Hp, Wp, C])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2]).contiguous() # B x C x Hp x Wp
        return patch_tokens
    
    def forward(self, x):
        H, W = x.shape[2:]
        x, _ = self.forward_features(x) # basic forward
        x = self.reshape_patch_tokens(x, H, W) # B x C x Hp x Wp 
        x = self.head(x) # Make predictions based on patch-tokens
        x_logits = self.avgpool(x).squeeze(3).squeeze(2) # return predictive logits
        return x_logits


class SimpleViTCAM(SimpleViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_attention(self, x, attn_weights, fuse_layers=12):
        attn_weights = torch.mean(torch.stack(attn_weights), dim=2)  # L * B * N' * N' with L = 12
        cams = F.relu(x.detach().clone()) # With ReLU Activation
        # patch-to-patch attention L x B x Np x Np
        pat2pat = torch.sum(attn_weights[-fuse_layers:], dim=0) # B x Np x Np
        B, _, Hf, Wf = cams.shape
        cams = torch.matmul(
                pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, self.num_classes, -1, 1) # B x Cls x Np x 1
        ).reshape(B, self.num_classes, Hf, Wf)
        
        return cams
    
    def forward(self, x, fuse_layers=3):
        H, W = x.shape[2:]
        x, attn_weights = self.forward_features(x)
        x = self.reshape_patch_tokens(x, H, W) # B x C x Hp x Wp 
        x = self.head(x) # Make predictions based on patch-tokens
        cams = self.forward_attention(
            x, attn_weights, fuse_layers)
        return cams


@register_model
def simplevit(pretrained=True, **kwargs):
    model = SimpleViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True)['model']
        
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
