import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from net.mctgformer_plus import MCTGFormer
from net.modules import get_indices_of_pairs


class MCTG_Aff(MCTGFormer):
    def __init__(self):
        super().__init__(
            patch_size=16, 
            embed_dim=384, 
            depth=12, 
            num_heads=6, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if self.head is not None:
            del self.head
        if self.channel_reduction is not None:
            del self.channel_reduction
            
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim * 5, 448, 1),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU())
        
        self.predefined_size = int(448 // 8)
        self.ind_from, self.ind_to = get_indices_of_pairs(
            radius=5, size=(self.predefined_size, self.predefined_size))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)
    
        scratch_parameters = list(self.fuse_conv.parameters())
        freeze_parameters = [p for p in self.parameters() if p not in scratch_parameters]
        for p in freeze_parameters:
            p.require_grads = False
    
    def forward(self, x, to_dense=False):
        H, W = x.shape[2:]
        feat_dict = self.forward_features(x) # basic forward
        
        patch_tokens = self.reshape_patch_tokens(feat_dict['x_pat'], H, W) # B x C x Hp x Wp  
        out_spatial = [patch_tokens]
        out_size = patch_tokens.shape[2:]
        aff_size = feat_dict['x_stru'][0].shape[2:]
        for x in feat_dict['x_stru']:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
            out_spatial.append(x)
          
        out_spatial = torch.cat(out_spatial, dim=1)
        x = self.fuse_conv(out_spatial) # B x 448 x H/16 x W/16 
         
        x = F.interpolate(x, size=aff_size, mode="bilinear", align_corners=False) # B x 448 x H/8 x W/8 
        
        if x.size(2) == self.predefined_size and x.size(3) == self.predefined_size:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = get_indices_of_pairs(radius=5, size=(x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))
        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))# B x 34 x 2496
        
        if to_dense:
            aff = aff.view(-1).cpu()
            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])
            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])
            aff_mat = torch.sparse_coo_tensor(
                torch.cat([indices, indices_id, indices_tp], dim=1),
                torch.cat([aff, torch.ones([area]), aff]), dtype=torch.float).to_dense().cuda()
            return aff_mat
        else:
            return aff
    