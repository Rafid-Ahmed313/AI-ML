import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            Swish()
        )

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio):
        super().__init__()
        reduced_ch = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, reduced_ch, 1),
            Swish(),
            nn.Conv2d(reduced_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride, se_ratio):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = in_ch == out_ch and stride == 1
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, 1, 1))
        layers.append(ConvBNAct(mid_ch, mid_ch, kernel_size, stride, groups=mid_ch))
        layers.append(SEBlock(mid_ch, se_ratio))
        layers.append(nn.Conv2d(mid_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            return x + out
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        cfg = [
            (1,  16, 1, 1, 3),
            (6,  24, 2, 2, 3),
            (6,  40, 2, 2, 5),
            (6,  80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]
        in_ch = 32
        self.stem = ConvBNAct(3, in_ch, 3, 2)
        layers = []
        for t, c, n, s, k in cfg:
            out_ch = c
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConvBlock(in_ch, out_ch, t, k, stride, 0.25))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.head = ConvBNAct(in_ch, 1280, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

