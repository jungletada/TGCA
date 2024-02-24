import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F

# models.
from net.modules import DownConv, SpatialPriorModule, SpatialFuseModule
from net.base_vit import VisionTransformer, _cfg


__all__ = ['deit_small_MCTG']

    
class MCTG(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.num_patches = num_patches
        self.stage_indices = (0, 3, 6, 9, 12)
        self.stages = len(self.stage_indices) - 1
        mask_ratios = [0.4, 0.3, 0.2, 0.1]
        self.spatial_dims = [self.embed_dim] * self.stages
       
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
         
        #================== Extra Part ==================#
        self.proj_cls_embed = nn.Linear(self.stages, self.num_classes)
        self.decay_parameter = decay_parameter
        
        self.spatial_prior = SpatialPriorModule(
            inplanes=64, embed_dims=self.spatial_dims)
        
        self.spatial_fuse = nn.ModuleList([
            SpatialFuseModule(query_dim=self.spatial_dims[i], key_dim=self.embed_dim, num_heads=6, 
                qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., 
                num_classes=self.num_classes, norm_layer=nn.LayerNorm, mask_ratio=mask_ratios[i])
            for i in range(self.stages)])
        
        self.spatial_downsamples = nn.ModuleList([
            DownConv(in_dim=self.spatial_dims[i], out_dim=self.spatial_dims[i+1])
                for i in range(self.stages-1)])
        
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(self.embed_dim * 5, self.embed_dim, 1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU())
        
    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and h == w:
            return self.pos_embed_pat
        
        patch_pos_embed = self.pos_embed_pat
        h0 = h // self.patch_embed.patch_size[0]
        w0 = w // self.patch_embed.patch_size[1]
        dim = x.shape[-1]
        Np = int(math.sqrt(N))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, Np, Np, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False)
        assert h0 == patch_pos_embed.shape[-2] and w0 == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def build_class_tokens(self, x):
        """
        Input: x -> list[B x C x H^ x W^]
        Return: cls-tokens -> B x Cls x C
        """
        x_cls = [self.globalweightedpooling(f).unsqueeze(-1) for f in x]  # list [B x C x 1]
        x_cls = torch.cat(x_cls, dim=-1)        # B x C x 4
        x_cls = self.proj_cls_embed(x_cls)      # B x C x Cls
        x_cls = x_cls.permute(0, 2, 1).contiguous()  # B x Cls x C
        return x_cls
         
    def forward_features(self, x):
        """
        Input:
            x: B x 3 x H x W
        Return:
            x_cls: [B x Cls x C] 
            x_pat: [B x Np x C]
            attn_weights: list[B x Hd x N' x N']
            x_spatial: list[B x C x H^ x W^]
        """
        B, _, H, W = x.shape                # B x 3 x H x W
        x_spatial = self.spatial_prior(x)   # list [B x C x H^ x W^]
        x = self.patch_embed(x)
        
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, H, W)
            x = x + pos_embed_pat
            
        else: x = x + self.pos_embed_pat
        
        nn_cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        cls_tokens = self.build_class_tokens(x_spatial) + nn_cls_tokens
        
        x = torch.cat((cls_tokens, x), dim=1) # Concat input with Nc class tokens
        x = self.pos_drop(x)                  # B x (N') x C, where N' = Nc + Np
        
        attn_weights = []
        for i in range(self.stages):
            for j in range(self.stage_indices[i], self.stage_indices[i+1]):# for each layer
                x, weights_j = self.blocks[j](x) # weights_j: the j-th layer attention weights
                attn_weights.append(weights_j)
            # spatial fusion 
            x_spatial[i] = self.spatial_fuse[i](x_query=x_spatial[i], x_key=x) 
            # downsample add
            if i != self.stages - 1:
                z = self.spatial_downsamples[i](x_spatial[i])
                x_spatial[i + 1] = x_spatial[i + 1] + z
    
        return x[:, :self.num_classes], x[:, self.num_classes:], attn_weights, x_spatial
    
    def reshape_patch_tokens(self, patch_tokens, H, W):
        B, _, C = patch_tokens.shape
        Hp = H // self.patch_embed.patch_size[0]
        Wp = W // self.patch_embed.patch_size[1]
        patch_tokens = torch.reshape(patch_tokens, [B, Hp, Wp, C])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2]).contiguous() # B x C x Hp x Wp
        return patch_tokens
    
    def globalweightedpooling(self, x):
        """
        Input:
            x->B x C x Hp x Wp
        Return
            out-> B x C
        """
        B, C, Hp, Wp = x.shape; N = Hp * Wp
        flatten_x = x.view(B, C, -1).permute(0, 2, 1) # B x (Hp x Wp) x C
        sorted_x, _ = torch.sort(flatten_x, -2, descending=True)
        weights = torch.logspace(start=0, end=N-1, steps=N, base=self.decay_parameter).cuda()
        out = torch.sum(sorted_x * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        
        return out
        
    def foward_tokens(self, patch_tokens):
        """ MCTformer Plus: Weighted Patch Tokens """
        patch_tokens = self.head(patch_tokens) # B x Cls x Hp x Wp
        patch_logits = self.globalweightedpooling(patch_tokens)

        return patch_logits
    
    def forward(self, x):
        H, W = x.shape[2:]
        # basic forward
        cls_tokens, patch_tokens, _, x_spatials = self.forward_features(x)
        cls_logits = cls_tokens.mean(-1)
        # patch tokens
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W) # B x C x Hp x Wp  
        out_spatial = [patch_tokens]
        # spatial tokens
        out_size = patch_tokens.shape[2:]
        for x in x_spatials:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
            out_spatial.append(x)
        # concat spatial and patch tokens        
        out_spatial = torch.cat(out_spatial, dim=1)
        out_spatial = self.spatial_reduce(out_spatial)
        # classification head and GWP
        pat_logits = self.foward_tokens(out_spatial)
        
        return cls_logits, pat_logits


class MCTGCAM(MCTG):
    def __init__(self, *args, **kwargs):
        super().__init__(decay_parameter=0.996, input_size=448, *args, **kwargs)
    
    def forward_attention(self, tokens, attn_weights, fuse_layers=12):
        """
        Input: 
            patch_tokens: patch tokens from the last backbone layer
            x_spatial: Spatial features from the spatial model
            attn_weights: attention weights from last L layers -> L x B x d x (Cls+Np) x (Cls+Np)
            fuse_layers: The attention of the last L layers to fuse
        Output: 
            Refined class activation maps -> B x Cls x Hp x Wp
        """
        
        B, Cls, Hp, Wp = tokens.shape
        
        attn_weights = torch.mean(torch.stack(attn_weights), dim=2)     # L x B x (Cls+Np) x (Cls+Np) 
        attn_maps = attn_weights[-fuse_layers:].mean(0)                 # B x (Cls+Np) x (Cls+Np)
        
        cls2pat = attn_maps[:, :Cls, Cls:].reshape([B, Cls, Hp, Wp])    # B x Cls x Hp x Wp
        
        patch_cam = tokens.detach().clone()   # B x Cls x Wp x Hp
        patch_cam = F.relu(patch_cam)   # With ReLU Activation
        
        cams = torch.pow(cls2pat * patch_cam, 1/2)
    
        # Apply pat2pat affinity refinement
        pat2pat = attn_weights[:, :, Cls:, Cls:] #  L x B x Np x Np
        pat2pat = torch.sum(pat2pat, dim=0)      # B x Np x Np

        cams = torch.matmul(
                pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, Cls, -1, 1) # B x Cls x Np x 1
            ).reshape(B, Cls, Hp, Wp)
        
        return cams
    
    def forward(self, x):
        H, W = x.shape[2:]
        # basic forward
        _, patch_tokens, attn_weights, x_spatials = self.forward_features(x)
        # patch tokens
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W) # B x C x Hp x Wp  
        out_spatial = [patch_tokens]
        # spatial tokens
        out_size = patch_tokens.shape[2:]
        for x in x_spatials:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
            out_spatial.append(x)
        # concat spatial and patch tokens        
        out_spatial = torch.cat(out_spatial, dim=1)
        out_spatial = self.spatial_reduce(out_spatial)
        # class activation map
        out_spatial = self.head(out_spatial)  # B x Cls x Hp x Wp
        cams = self.forward_attention(
            out_spatial, attn_weights, fuse_layers=3)
        
        return cams


@register_model
def deit_small_MCTG(pretrained=False, **kwargs):
    model = MCTG(
        patch_size=16, embed_dim=384, depth=12, 
        num_heads=6, mlp_ratio=4, qkv_bias=True,
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


if __name__ == "__main__":
    x = torch.ones(2, 3, 527, 481).cuda()
    # model = MCTGCAM(num_classes=20).cuda()
    # out = model(x)
    # print(f"CAM: {out.shape}")
    
    from timm.models import create_model
    model = create_model(
        "deit_small_MCTG",
        pretrained=False,
        num_classes=20,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        input_size=512).cuda()
    
    model.eval()
    
    return_att = False
    if return_att:
        cls_logits, cams, pat2pat, attn_maps = model(
            x, n_layers=12)
        print(f"cls_logits: {cls_logits.shape}")
        print(f"cams: {cams.shape}")
        print(f"pat2pat: {pat2pat.shape}")
        
    else:
        output_logits = model(x)
        for i in range(len(output_logits)):
            print(f"{i}-logits shape: {output_logits[i].shape}")