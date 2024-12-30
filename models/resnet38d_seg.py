import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet38d import ResNet38d


class ResNet38d_Seg(ResNet38d):
    def __init__(self, num_classes):
        super().__init__()
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        self.from_scratch_layers = [self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x):
        x = self.forward_as_dict(x)['conv6']
        x_seg = F.relu(self.fc8_seg_conv1(x))
        x_seg = self.fc8_seg_conv2(x_seg)
        return x_seg


    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


if __name__ == '__main__':
    model = ResNet38d_Seg(num_classes=21)
    images = torch.randn(4, 3, 321, 321)
    with torch.no_grad():
        pred = model(images)
        print(pred.shape)