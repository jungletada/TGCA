import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from psa.resnet38d import ResNet38d


def get_indices_of_pairs(radius, size):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))
    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))
    radius_floor = radius - 1
    full_indices = np.reshape(
        np.arange(0, size[0]*size[1], dtype=np.int64),
        (size[0], size[1]))
    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor
    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])
    indices_to_list = []
    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])
        indices_to_list.append(indices_to)
    concat_indices_to = np.concatenate(indices_to_list, axis=0)
    return indices_from, concat_indices_to


class ResNet38d_Aff(ResNet38d):
    def __init__(self):
        super(ResNet38d_Aff, self).__init__()
        if self.head is not None:
            del self.head
        self.f8_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = nn.Conv2d(4096, 256, 1, bias=False)
        self.f9 = nn.Conv2d(448, 448, 1, bias=False)
        
        nn.init.kaiming_normal_(self.f8_3.weight)
        nn.init.kaiming_normal_(self.f8_4.weight)
        nn.init.kaiming_normal_(self.f8_5.weight)
        nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_size = int(448 // 8)
        self.ind_from, self.ind_to = get_indices_of_pairs(
            radius=5, size=(self.predefined_size, self.predefined_size))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

    def forward(self, x, to_dense=False):
        d = super().forward_as_dict(x)
        f8_3 = F.elu(self.f8_3(d['conv4'])) # B x 512 x (H/8) x (W/8)  -> 64
        f8_4 = F.elu(self.f8_4(d['conv5'])) # B x 1024 x (H/8) x (W/8) -> 128
        f8_5 = F.elu(self.f8_5(d['conv6'])) # B x 4096 x (H/8) x (W/8) -> 256
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1))) # B x 448 x (H/8) x (W/8)

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

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)  # 10xlr & weight dacay
                    else:
                        groups[0].append(m.weight)  # 1xlr & weight dacay
                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)    # 20xlr & no weight dacay
                    else:
                        groups[1].append(m.bias)    # 2xlr & no weight dacay
        return groups
    