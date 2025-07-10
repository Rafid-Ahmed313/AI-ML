import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(self.norm1(x_windows))
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.reduction(self.norm(x))
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, depth, input_resolution, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, input_resolution, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(input_resolution, dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000, embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.stages = nn.ModuleList()
        resolution = img_size // patch_size
        for i in range(len(depths)):
            stage = SwinStage(embed_dim * 2**i, depths[i], (resolution // 2**i, resolution // 2**i), num_heads[i], window_size)
            self.stages.append(stage)
        self.norm = nn.LayerNorm(embed_dim * 2**(len(depths) - 1))
        self.head = nn.Linear(embed_dim * 2**(len(depths) - 1), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.head(x.mean(dim=1))
        return x

model = SwinTransformer()

