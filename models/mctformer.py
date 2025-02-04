import math
import torch
import torch.nn as nn
from functools import partial
# 
from models.vit import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F


__all__ = ['mctformerv2']


class MCTformerV2(VisionTransformer):
    def __init__(self, input_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.Hp, self.Wp = math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1])
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)

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
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic')

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x, n=12):
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else: x = x + self.pos_embed_pat

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x) # B x (N') x C, where N' = Nc + Np
        
        attn_weights = []
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
        # [B * Nc * C], [B * Np * C], list[B * Hd * N' * N']
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights

    def forward(self, x):
        Wp, Hp = x.shape[2:]
        # class-tokens, patch-tokens, attention weights
        x_cls, x_patch, attn_weights = self.forward_features(x)
        x_cls_logits = x_cls.mean(-1) # B x Nc
        B, Np, Nc = x_patch.shape
        if Wp != Hp:
            num_w_patches = Wp // self.patch_embed.patch_size[1]
            num_h_patches = Hp // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [B, num_w_patches, num_h_patches, Nc])
        else:
            x_patch = torch.reshape(x_patch, [B, int(Np ** 0.5), int(Np ** 0.5), Nc])
        
        # patch-tokens
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous() # B x C x Wp x Hp
        x_patch = self.head(x_patch) # Make predictions based on patch-tokens
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) # return predictive logits
        
        return x_cls_logits, x_patch_logits


class MCTformerV2Cam(MCTformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_attention(self, x_patch, attn_weights, fuse_layers=3):
        attn_weights = torch.stack(attn_weights)  # L * [B * Hd * N' * N'] with L = 12
        attn_weights = torch.mean(attn_weights, dim=2)  # L * B * N' * N' with L = 12

        feature_map = x_patch.detach().clone()  # B x Nc x Wp x Hp
        feature_map = F.relu(feature_map) # With ReLU Activation

        B, Nc, Hp, Wp = feature_map.shape# [L x B x N' x N'] -> (sum) [B x N' x N'] 
        
        attn_maps = attn_weights[-fuse_layers:].sum(0)
        cls2pat = attn_maps[:, 0:self.num_classes, self.num_classes:].reshape([B, Nc, Hp, Wp])
    
        cams = cls2pat * feature_map  #  B x Nc x Hp x Wp
        # patch-to-patch attention L x B x Np x Np
        pat2pat = attn_weights[:, :, self.num_classes:, self.num_classes:]
        pat2pat = torch.sum(pat2pat, dim=0) # B x Np x Np
        B, _, Hf, Wf = cams.shape
        
        cams = torch.matmul(
                pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, self.num_classes, -1, 1) # B x Cls x Np x 1
        ).reshape(B, self.num_classes, Hf, Wf)
        
        return cams
    
    def forward(self, x, fuse_layers=3):
        Wp, Hp = x.shape[2:]
        # class-tokens, patch-tokens, attention weights
        x_cls, x_patch, attn_weights = self.forward_features(x)
        
        B, Np, Nc = x_patch.shape
        if Wp != Hp:
            num_w_patches = Wp // self.patch_embed.patch_size[1]
            num_h_patches = Hp // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [B, num_w_patches, num_h_patches, Nc])
        else:
            x_patch = torch.reshape(x_patch, [B, int(Np ** 0.5), int(Np ** 0.5), Nc])
        
        # patch-tokens
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous() # B x C x Wp x Hp
        x_patch = self.head(x_patch) # Make predictions based on patch-tokens
        
        cams = self.forward_attention(
            x_patch, attn_weights, fuse_layers=fuse_layers)
        return cams


def convert_weights_update(checkpoint, num_classes=80):
    # print(checkpoint.keys())
    pos_embed = checkpoint["pos_embed"]
    checkpoint["pos_embed_cls"] = pos_embed[:, :num_classes, :]
    checkpoint["pos_embed_pat"] = pos_embed[:, num_classes:, :]
    del checkpoint["pos_embed"]
    return checkpoint


@register_model
def mctformerv2(pretrained=False, **kwargs):
    model = MCTformerV2(
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

if __name__ == '__main__':
    # checkpoint = torch.load('checkpoints/MCTformerV2_coco.pth')['model']
    # updated = convert_weights_update(checkpoint, num_classes=80)
    # torch.save(updated, 'checkpoints/MCTformerV2_coco_update.pth')

    model = MCTformerV2(input_size=224)
    model_dict = torch.load('checkpoints/MCTformerV2_coco_update.pth', map_location='cpu')
    model.load_state_dict(model_dict)