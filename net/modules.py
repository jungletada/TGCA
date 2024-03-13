import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from net.gcn_lib import Grapher, act_layer


"""
    The code is modified based on 
    TDRG ()
    and MCTformer ()
"""


def nlc2nchw(x, d_size):
    _, N, C = x.shape
    assert d_size[0] * d_size[1] == N, f"{d_size} not equal to {N}"
    x = x.permute(0, 2, 1).reshape(-1, C, d_size[0], d_size[1]).contiguous()
    return x


def nchw2nlc(x):
    B, C = x.shape[:2]
    x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
    return x
  
    
class DownConv(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.BatchNorm2d, stride=2):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.norm = norm_layer(out_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
        
    
class TopKMaxPooling(nn.Module):
    """
        Top-K Maxpooling
        Input: B x C x H x W
        Return: B x C
    """

    def __init__(self, kmax=1.0):
        super(TopKMaxPooling, self).__init__()
        self.kmax = kmax

    @staticmethod
    def get_positive_k(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions
        num_kmax = self.get_positive_k(self.kmax, n)
        sorted, _ = torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True)
        region_max = sorted.narrow(dim=2, start=0, length=num_kmax) # B x C x K
        output = region_max.sum(2).div_(num_kmax)   # B x C
        return output.view(batch_size, num_channels)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, dim):
        super(GraphConvolution, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(dim, dim, 1)

    def forward(self, adj, nodes):
        """
            adj->B x Cls x Cls
            nodes->B x (Cg+Ct) x Cls
        """
        nodes = torch.matmul(nodes, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes
    

class SpatialPriorModule(nn.Module):
    def __init__(self, 
                 inplanes=64, 
                 embed_dims=[96, 192, 384, 768], 
                 norm_layer=nn.BatchNorm2d, 
                 act_layer=nn.GELU):
        super().__init__()
        self.stem = nn.Sequential(*[ # downsample by 4
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]) 
        self.conv2 = nn.Sequential(*[# downsample by 2
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(2 * inplanes),
            act_layer(),
        ])
        self.conv3 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer(),
        ])
        self.conv4 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer(),
        ])
        self.conv5 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8 * inplanes),
            act_layer(),
        ])
        
        # self.fc1 = nn.Conv2d(inplanes, embed_dims[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dims[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dims[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dims[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc5 = nn.Conv2d(8 * inplanes, embed_dims[3], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        # c1 = self.fc1(c1) # 4s
        c2 = self.fc2(c2) # 8s
        c3 = self.fc3(c3) # 16s
        c4 = self.fc4(c4) # 32s
        c5 = self.fc5(c5) # 32s
    
        return [c2, c3, c4, c5]


class Conv1dProj(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, 1, bias=bias)
        self.act = act
    
    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
        

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_classes=20, num_heads=8, qkv_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0., mask_ratio=0.3):
        super().__init__()
        self.num_heads = num_heads
        dim = min(query_dim, key_dim)
        self.head_dim = dim // num_heads
        self.mask_ratio = mask_ratio
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.proj_q = nn.Linear(query_dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(key_dim, dim * 2, bias=qkv_bias)
        self.proj_cls = nn.Linear(key_dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Cls = num_classes

    def forward(self, input_query, input_key):
        B, Nt, _ = input_query.shape
        N = input_key.shape[1]
        
        # Concat class tokens to query
        cls_tokens = self.proj_cls(input_key[:, :self.Cls, :])
        input_query = self.proj_q(input_query)
        input_query = torch.cat((cls_tokens, input_query), dim=1)
        q = input_query.reshape(
            B, (self.Cls+Nt), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
       
        kv = self.proj_kv(input_key).reshape(
            B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # B x Nd x (Cls+N) x d
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x Nd x (Cls+Nt) x (Cls+N)
        #== Split class and patch tokens ============================================#               
        attn_cls, attn_pat = torch.split(attn, [self.Cls, N-self.Cls], dim=-1)
        attn_pat = attn_pat.softmax(dim=-1)
        attn_cls = attn_cls.softmax(dim=-1)
        attn = torch.cat((attn_cls, attn_pat), dim=-1)
        #======================================================================#     
        # DropKey
        m_r = torch.ones_like(attn) * self.mask_ratio 
        attn = attn + torch.bernoulli(m_r) * -1e12
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape( # B x Nd x (Cls+Nt) x d
            B, (self.Cls+Nt), -1)               # B x (Cls+Nt) x Ct
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x[:, self.Cls:, :]
        return x

       
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class SpatialFuseModule(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., drop_path=0., num_classes=20, norm_layer=nn.LayerNorm, mask_ratio=0.3):
        super().__init__()
        self.norm1 = norm_layer(query_dim)
        self.norm2 = norm_layer(key_dim)
        self.norm3 = norm_layer(query_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Cls = num_classes
        self.cross_attn = CrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mask_ratio=mask_ratio)

        self.mlp = MLP(in_features=query_dim, hidden_features=query_dim * 4)

    def forward(self, x_query, x_key):
        """
        z: spatial features
        x: transformer features
        """
        H, W = x_query.shape[2:]
        x_query = nchw2nlc(x_query)
        x_query = x_query + self.drop_path(self.cross_attn(self.norm1(x_query), self.norm2(x_key)))
        x_query = x_query + self.drop_path(self.mlp(self.norm3(x_query)))
        x_query = nlc2nchw(x_query, d_size=(H, W))
        
        return x_query
                      
            
class SemanticGraph(nn.Module):
    def __init__(self, kernel_size=9, num_classes=20, mid_channels=64, drop_path=0., dilation=1):
        super().__init__()
        
        self.grapher = Grapher(
            in_channels=num_classes, 
            mid_channels=mid_channels,
            kernel_size=kernel_size, 
            dilation=dilation, 
            conv='mr', 
            act='gelu', 
            norm='instance',
            bias=True, 
            stochastic=False, 
            epsilon=0.2, 
            r=1, 
            drop_path=drop_path)
    
    def forward(self, x):
        x = self.grapher(x)
        return x
        
      
class SemanticSpatialModule(nn.Module):
    def __init__(self, 
                 query_dim, 
                 key_dim, 
                 num_classes=20, 
                 mid_channels=64,
                 attn_drop=0., 
                 proj_drop=0., 
                 qkv_bias=True,
                 drop_path=0., 
                 mask_ratio=0.1, 
                 kernel_dict=9, 
                 dilation_dict=1,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.norm1 = norm_layer(query_dim)
        self.norm2 = norm_layer(key_dim)
        self.norm3 = norm_layer(query_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Cls = num_classes
        self.mask_ratio = mask_ratio
        self.dim = min(query_dim, key_dim)
        self.scale = self.Cls ** -0.5
        
        self.proj_q = nn.Linear(query_dim, self.dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(key_dim, self.dim * 2, bias=qkv_bias)
        self.proj_cls = nn.Linear(key_dim, self.dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.graph_b = SemanticGraph(
            kernel_size=kernel_dict,
            num_classes=num_classes,
            mid_channels=mid_channels,
            drop_path=drop_path,
            dilation=dilation_dict)
        
        self.graph_s = SemanticGraph(
            kernel_size=kernel_dict,
            num_classes=num_classes,
            mid_channels=mid_channels,
            drop_path=drop_path,
            dilation=dilation_dict)
        
        self.mlp = MLP(
            in_features=self.dim, 
            hidden_features=self.dim * 4,
            out_features=self.dim)

    def weighted_tokens(self, x, base=0.9):
        C, N = x.shape[1:]
        weights = torch.logspace(start=0, end=C-1, steps=C, base=base).unsqueeze(-1)
        x = x.softmax(dim=1)
        x_sorted, indices = torch.sort(x, dim=1, descending=True)
        weighted_sorted = x_sorted * weights.to(x.device)
        result = torch.empty_like(x)
        inv_indices = indices.argsort(dim=1)
        result.scatter_(1, inv_indices, weighted_sorted)
        return result
    
    def forward_semantic_gnn(self, x_spatial, x_backbone, token_size, spatial_size):
        """
        Input:
            x_spatial: spatial features from prior module->[B, (Hi * Wi), Cq]
            x_backbone: multi-class token transformer features->[B, (Cls+N'), Ck]
        Output:
            x_spatial: updated spatial features 
        """
        B, N, _ = x_backbone.shape
        Ni = spatial_size[0] * spatial_size[1]
        # Linear projection 
        q = self.proj_q(x_spatial)  # B, (Hi * Wi), C
        x_cls, x_pat = torch.split(x_backbone, [self.Cls, N-self.Cls], dim=1)
        x_cls = self.proj_cls(x_cls)    # B, Cls, C
        # Build semantic relation
        relation_s = (x_cls @ q.transpose(-2, -1)) * self.scale # (B, Cls, C) (B, Ni, C) 
        # relation_s = self.weighted_tokens(relation_s, base=0.9)
        
        relation_s = relation_s.view(B, -1, spatial_size[0], spatial_size[1])
        relation_s = self.graph_s(relation_s)
        relation_s = relation_s.view(B, self.Cls, -1)
        
        kv = self.proj_kv(x_pat).reshape(B, -1, 2, self.dim).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1] # [B, N', C]
        # Build semantic relation
        relation_b = (x_cls @ k.transpose(-2, -1)) * self.scale # (B, Cls, C) (B, N', C) 
        # relation_b = self.weighted_tokens(relation_b, base=0.9)
        
        relation_b = relation_b.view(B, -1, token_size[0], token_size[1]) # (B, Cls, H', W')
        relation_b = self.graph_b(relation_b)
        relation_b = relation_b.view(B, self.Cls, -1) 
        # GNN feature aggregator
        spatial_guide = (relation_s.transpose(-2, -1) @ relation_b) * self.scale
        spatial_guide = spatial_guide.softmax(dim=-1)
        m_r = torch.ones_like(spatial_guide) * self.mask_ratio 
        spatial_guide = spatial_guide + torch.bernoulli(m_r) * -1e12
        spatial_guide = spatial_guide.softmax(dim=-1)
        
        spatial_guide = self.attn_drop(spatial_guide)
        output = spatial_guide @ v # Fuse backbone relation and spatial relation
       
        return output
    
        
    def forward(self, x_spatial, x_backbone, token_size):
        """
        Input:
            x_spatial: spatial features from prior module->[B, Cq, Hi, Wi]
            x_backbone: multi-class token transformer features->[B, (Cls+N'), Ck]
        Output:
            x_spatial: updated spatial features 
        """
        H, W = x_spatial.shape[2:]
        x_spatial = nchw2nlc(x_spatial)
        idendity = x_spatial.clone()

        x_spatial = self.forward_semantic_gnn(
            self.norm1(x_spatial), self.norm2(x_backbone), 
            token_size=token_size, spatial_size=(H, W))
        
        x_spatial = idendity + self.drop_path(x_spatial)
        x_spatial = x_spatial + self.drop_path(self.mlp(self.norm3(x_spatial)))
        
        x_spatial = nlc2nchw(x_spatial, d_size=(H, W))
        return x_spatial
        
        
    
if __name__ == "__main__":
    H, W = 14, 14
    Hi, Wi = 35, 24
    model = SemanticSpatialModule(
        query_dim=64, key_dim=384, num_classes=20,)
    x = torch.ones(3, 64, Hi, Wi)
    y = torch.ones(3, int(H * W + 20), 384)
    out = model(x, y, token_size=(H, W))
    # print(out.shape)
