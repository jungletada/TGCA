import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x) # (b, c, 1, 1)
        avgout = self.avgpool(x) # (b, c, 1, 1)
        
        maxout = self.mlp(maxout) # (b, c, 1, 1)
        avgout = self.mlp(avgout) # (b, c, 1, 1)
        attention = self.sigmoid(maxout + avgout) # (b, c, 1, 1)
        return attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1,
                kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): # (b, c, h, w)
        maxpool = x.argmax(dim=1, keepdim=True) # (b, 1, h, w)
        avgpool = x.mean(dim=1, keepdim=True)   # (b, 1, h, w)
        out = torch.cat([maxpool, avgpool], dim=1) # (b, 2, h, w)
        out = self.conv(out)  # (b, 1, h, w)
        attention = self.sigmoid(out) # (b, 1, h, w)
        return attention


if __name__ == "__main__":
    ca = ChannelAttention(64)
    sa = SpatialAttention()

    x = torch.randn(3, 64, 56, 56)

    channel = ca(x)  # (3, 64, 1, 1)
    x = channel * x  # (3, 64, 56, 56)

    spatial = sa(x)  # (3, 1, 56, 56)
    x = spatial * x  # (3, 64, 56, 56)