import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple


from net.modules import GraphConvolution, DownConv, TopKMaxPooling
from net.modules import SpatialPriorModule, SpatialFuseModule
from net.base_vit import VisionTransformer, _cfg


__all__ = ['deit_small_MCTGCN']

    
class MCTG(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.num_patches = num_patches
        
        self.stage_indices = (0, 3, 6, 9, 12)
        self.stages = len(self.stage_indices) - 1
        self.spatial_dims = [self.embed_dim] * self.stages
        spatial_mask_ratios = [0.4, 0.3, 0.2, 0.1]
        
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        
        # ================= GCN ================= #
        self.gcn_dim = 384
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.topk_pool = TopKMaxPooling(kmax=0.05)
        
        self.gcn_dim_transform = nn.Conv2d(self.embed_dim, self.gcn_dim, kernel_size=1)
        self.guidance_transform = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.guidance_module = nn.Sequential(
            nn.Conv1d(self.embed_dim * 3, self.embed_dim * 3, kernel_size=1),
            nn.BatchNorm1d(self.embed_dim * 3),
            nn.LeakyReLU(0.2))
        
        self.matrix_transform = nn.Conv1d(
            self.embed_dim * 4 + self.gcn_dim, self.num_classes, kernel_size=1)
        self.forward_gcn = GraphConvolution(self.embed_dim + self.gcn_dim)
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.gcn_classifier = nn.Conv1d(self.embed_dim + self.gcn_dim, self.num_classes, 1)
        self.gcn_classifier.apply(self._init_weights)
        
        # ================= Structure ================= #
        self.proj_cls_embed = nn.Linear(self.stages, self.num_classes)
        self.decay_parameter = decay_parameter
        
        self.spatial_prior = SpatialPriorModule(
            inplanes=64, embed_dims=self.spatial_dims)
        
        self.spatial_fuse = nn.ModuleList([
            SpatialFuseModule(
                query_dim=self.spatial_dims[i], 
                key_dim=self.embed_dim, 
                num_heads=6, 
                qkv_bias=True, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0., 
                drop_path=0., 
                num_classes=self.num_classes, 
                norm_layer=nn.LayerNorm, 
                mask_ratio=spatial_mask_ratios[i])
            for i in range(self.stages)])
        
        self.spatial_downsamples = nn.ModuleList([
            DownConv(in_dim=self.spatial_dims[i], 
                     out_dim=self.spatial_dims[i+1])
            for i in range(self.stages-1)])
        
        self.trans_classifier = nn.Linear(self.embed_dim * 4, self.num_classes)
        
    def get_parameter_gropus(self):
        spatial_modules = [
            self.proj_cls_embed,
            self.spatial_prior,
            self.spatial_fuse,
            self.spatial_downsamples,
        ]
        gcn_modules = [
            self.gcn_dim_transform,
            self.guidance_transform,
            self.guidance_module,
            self.matrix_transform,
            self.forward_gcn,
            self.gcn_classifier,
        ]
        
        spatial_params = [p for module in spatial_modules for p in module.parameters()]
        gcn_params = [p for module in gcn_modules for p in module.parameters()] + [self.mask_mat]
        
        all_params = set(self.parameters())   
        base_params = list(all_params - set(spatial_params)-set(gcn_params))
        
        return base_params, spatial_params, gcn_params
    
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

    def build_class_tokens(self, z):
        """
        Input: feats -> list [B x C x H^ x W^]
        Return: cls-tokens -> B x Cls x C
        """
        B = z[0].shape[0]
        cls_tokens = []
        for i, feat in enumerate(z):
            out = self.avgpool2d(feat).reshape(B, self.spatial_dims[i], -1) # B x Ct x 1
            # out = self.spatial_heads[i](out)
            cls_tokens.append(out)
        cls_tokens = torch.cat(cls_tokens, dim=-1)   # B x C x 4
        cls_tokens = self.proj_cls_embed(cls_tokens) # B x C x Cls
        cls_tokens = cls_tokens.permute(0, 2, 1).contiguous() # B x Cls x C
        return cls_tokens
    
    def forward_features(self, x):
        """
        Input: 
            image -> B x 3 x H x W
        Output:
            x_cls -> B x Cls x C [], list[]
            x_pat -> B * Np * C
            attn_weights -> B * Hd * N' * N' (N'=Cls+Np)
            x_spatial -> list(B x C x H^ x W^)
        """
        B, _, H, W = x.shape
        x_spatial = self.spatial_prior(x) # list [B x C x H^ x W^]
        x = self.patch_embed(x)
        
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, H, W)
            x = x + pos_embed_pat
            
        else: x = x + self.pos_embed_pat
        
        nn_cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        cls_tokens = self.build_class_tokens(x_spatial) + nn_cls_tokens
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        
        attn_weights = []
        for i in range(self.stages):
            
            for j in range(self.stage_indices[i], self.stage_indices[i+1]):
                x, weights_j = self.blocks[j](x) 
                attn_weights.append(weights_j) 
                
            x_spatial[i] = self.spatial_fuse[i](
                x_query=x_spatial[i], x_key=x) 
            
            if i != self.stages - 1:
                x_spatial[i + 1] = x_spatial[i + 1] + self.spatial_downsamples[i](x_spatial[i])
        
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, x_spatial
        
    def reshape_patch_tokens(self, patch_tokens, Wp, Hp):
        B, Np, C = patch_tokens.shape
        if Wp != Hp:
            num_h_patches = Hp // self.patch_embed.patch_size[0]
            num_w_patches = Wp // self.patch_embed.patch_size[1]
            patch_tokens = torch.reshape(patch_tokens, [B, num_w_patches, num_h_patches, C])
        else:
            patch_tokens = torch.reshape(patch_tokens, [B, int(Np ** 0.5), int(Np ** 0.5), C])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2]).contiguous() # B x C x Wp x Hp
        return patch_tokens

    def foward_patch_tokens(self, patch_tokens):
        """ MCTformer Plus Weighted Patch Tokens """
        patch_tokens = self.head(patch_tokens) # B x Cls x Hp x Wp
        B, Cls, Hp, Wp = patch_tokens.shape
        N = Hp * Wp
        flattened_tokens = patch_tokens.view(B, Cls, -1).permute(0, 2, 1) # B x (Hp x Wp) x Cls
        sorted_tokens, _ = torch.sort(flattened_tokens, -2, descending=True)
        weights = torch.logspace(start=0, end=N-1, steps=N, base=self.decay_parameter).cuda()
        patch_logits = torch.sum(sorted_tokens * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        return patch_logits
    
    def build_nodes(self, x_pat, x_spatial):
        # build nodes for GCN
        Cls = self.num_classes
        # patch tokens
        B, _, H, W = x_pat.shape # N = H x W
        mask = self.head(x_pat).view(B, Cls, -1)    # B x Cls x N
        mask = torch.sigmoid(mask).transpose(1, 2)  # B x N x Cls

        x_pat = self.gcn_dim_transform(x_pat)       # B x Cg x H x W
        x_pat = x_pat.view(B, x_pat.shape[1], -1)   # B x Cg x N
        vg = torch.matmul(x_pat, mask)              # B x Cg x Cls

        x_spatial = F.interpolate(  # Upsample the last spatial tokens
                input=x_spatial,
                size=(H, W),
                mode='bilinear',
                align_corners=False)
        
        x_spatial = x_spatial.reshape(B, x_spatial.shape[1], -1) # B x Ct x N
        vt = torch.matmul(x_spatial, mask).detach()  # B x Ct x Cls
        vt = self.guidance_transform(vt)        # B x Ct x Cls
        nodes = torch.cat((vg, vt), dim=1)      # B x (Cg+Ct) x Cls
        
        return nodes
    
    def build_joint_correlation_matrix(self, x_spatials, nodes):
        """5. build joint correlation matrix"""
        c2, c3, c4 = x_spatials[1:]
        B, C = c2.shape[:2] # B x Ct x H x W
        c2 = self.avgpool1d(c2.view(B, C, -1))
        c3 = self.avgpool1d(c3.view(B, C, -1))
        c4 = self.avgpool1d(c4.view(B, C, -1))
        trans_guid = torch.cat((c2, c3, c4), dim=1)    # B x 3Ct x 1
        trans_guid = self.guidance_module(trans_guid)  # B x 3Ct x 1
        trans_guid = trans_guid.expand(
            trans_guid.shape[0], trans_guid.shape[1], nodes.shape[2])  # B x 3Ct x Cls
        nodes = torch.cat((trans_guid, nodes), dim=1)                  # B x (Cg+4Ct) x Cls
        joint_correlation = self.matrix_transform(nodes)               # B x Cls x Cls
        joint_correlation = torch.sigmoid(joint_correlation)           # B x Cls x Cls
        
        return joint_correlation
    
    def forward(self, x):
        H, W = x.shape[2:]
        class_tokens, patch_tokens, _, x_spatial = self.forward_features(x)
        cls_logits = class_tokens.mean(-1)
        
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W) # B x C x Hp x Wp
        patch_logits = self.foward_patch_tokens(patch_tokens)
        
        V = self.build_nodes(patch_tokens, x_spatial[-1])       # B x (Cg+Ct) x Cls
        As = self.build_joint_correlation_matrix(x_spatial, V)  # B x Cls x Cls
        
        G = self.forward_gcn(As, V) + V        # B x (Cg+Ct) x Cls
        gcn_logits = self.gcn_classifier(G)    # B x Cls x Cls
        mask_mat = self.mask_mat.detach()      # idendity Cls x Cls
        gcn_logits = (gcn_logits * mask_mat).sum(-1)  
       
        return cls_logits, patch_logits, gcn_logits


class MCTGCAM(MCTG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward_attention(self, x_patch, attn_weights, fuse_layers=12):
        """
        Input: 
            patch_tokens: patch tokens from the last backbone layer
            x_spatial: Spatial features from the spatial model
            attn_weights: attention weights from last L layers -> L x B x d x (Cls+Np) x (Cls+Np)
            fuse_layers: The attention of the last L layers to fuse
        Output: 
            Refined class activation maps -> B x Cls x Hp x Wp
        """
        Cls = self.num_classes # simplify code
        B, _, Hp, Wp = x_patch.shape
        
        attn_weights = torch.mean(torch.stack(attn_weights), dim=2)     # L x B x (Cls+Np) x (Cls+Np) 
        attn_maps = attn_weights[-fuse_layers:].mean(0)                 # B x (Cls+Np) x (Cls+Np)
        
        cls2pat = attn_maps[:, :Cls, Cls:].reshape([B, Cls, Hp, Wp])    # B x Cls x Hp x Wp
    
        patch_cam = x_patch.detach().clone()   # B x Cls x Wp x Hp
        patch_cam = F.relu(patch_cam)               # With ReLU Activation
        
        cams = cls2pat * patch_cam           #  B x Nc x Hp x Wp
        cams = torch.pow(cams, 1/2)
    
        # Apply pat2pat affinity refinement
        pat2pat = attn_weights[:, :, Cls:, Cls:]         #  L x B x Np x Np
        pat2pat = torch.sum(pat2pat, dim=0) # B x Np x Np

        cams = torch.matmul(
                pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, Cls, -1, 1) # B x Cls x Np x 1
            ).reshape(B, Cls, Hp, Wp)
        
        return cams
    
    def forward(self, x):
        B, _, H, W = x.shape   # batch size=2
        _, patch_tokens, attn_weights, x_spatial = self.forward_features(x)
        
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W)  # B x C x Hp x Wp
        patch_tokens = self.head(patch_tokens)  # B x Cls x Hp x Wp
        
        x_spatial = self.spatial_head(x_spatial)
        
        cams = self.forward_attention(
            patch_tokens, attn_weights, fuse_layers=12)
        
        return cams
    


@register_model
def deit_small_MCTGCN(pretrained=False, **kwargs):
    model = MCTG(
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


if __name__ == "__main__":
    from timm.models import create_model
    
    model = create_model(
        "deit_small_MCTGCN",
        pretrained=False,
        num_classes=20).cuda()
    x = torch.ones(2, 3, 644, 644).cuda()
    model.eval()
    
    with torch.no_grad():
        output_logits = model(x)
        for i in range(len(output_logits)):
            print(f"{i}-logits shape: {output_logits[i].shape}")
    