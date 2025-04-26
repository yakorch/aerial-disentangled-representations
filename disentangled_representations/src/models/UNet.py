from typing import Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet_parts import Down, Up, DoubleNonLinearConv
from .abstract_models import I2IModel


def _adaptive_instance_norm(features: torch.Tensor, params: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    B, C, H, W = features.size()
    assert params.size(1) == 2 * C, f"Expected params of shape (B, {2 * C}), got {tuple(params.size())}"

    gamma, beta = params.chunk(2, dim=1)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)

    normalized = F.instance_norm(features, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, eps=eps)
    return gamma * normalized + beta


class ConvNeXtFuse(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9, expansion: int = 4, layer_scale: float | None = None):
        super().__init__()
        padding = kernel_size // 2

        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        hidden_dim = in_ch * expansion
        layers = [nn.Conv2d(in_ch, hidden_dim, kernel_size=1), nn.GELU(), nn.Conv2d(hidden_dim, out_ch, kernel_size=1), ]
        self.pw = nn.Sequential(*layers)

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual = nn.Identity()

        if layer_scale is not None:
            self.gamma = nn.Parameter(torch.ones(out_ch) * layer_scale)
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.residual(x) + y


class EnhancedStyleFuse(nn.Module):
    def __init__(self, in_ch, out_ch, se_reduction=4):
        super().__init__()
        self.dw1 = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=4, dilation=2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.dw2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        from timm.layers import SqueezeExcite
        self.se = SqueezeExcite(out_ch, rd_ratio=se_reduction)
        self.act = nn.ReLU(inplace=True)
        self.res = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        shortcut = self.res(x)
        x = self.act(self.bn1(self.dw1(x)))
        x = self.act(self.bn2(self.dw2(x)))
        x = self.bn3(self.pw(x))
        x = self.se(x)
        return self.act(x + shortcut)


class SuperFuse(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        from timm.layers import MixedConv2d
        self.mix = MixedConv2d(in_ch, in_ch, kernel_size=[3, 5, 7], bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)

        from timm.layers.eca import EfficientChannelAttn
        self.eca = EfficientChannelAttn(in_ch)

        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.mix(x)))
        x = self.eca(x)
        x = self.act(self.bn3(self.pw(x)))
        return x


class ResDW5x5ECA(nn.Module):
    def __init__(self, in_ch, out_ch):
        from timm.layers.eca import EfficientChannelAttn

        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.eca = EfficientChannelAttn(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        shortcut = self.res(x)
        x = self.act(self.bn1(self.dw(x)))
        x = self.eca(x)
        x = self.bn2(self.pw(x))
        return self.act(x + shortcut)


class UNet(I2IModel):
    def __init__(self, in_channels: int, out_channels: int, channels: Sequence[int], conv_block_down: Type[nn.Module], conv_block_up: Type[nn.Module],
                 latent_dim: int):
        super().__init__()
        assert len(channels) >= 1

        self.inc = DoubleNonLinearConv(in_channels, channels[0])

        self.downs = nn.ModuleList([Down(channels[i], channels[i + 1], conv_block=conv_block_down) for i in range(len(channels) - 1)])
        rev_ch = list(reversed(channels))
        self.ups = nn.ModuleList(
            Up(in_channels=rev_ch[i] + rev_ch[i + 1], out_channels=rev_ch[i + 1], conv_block=conv_block_up) for i in range(len(rev_ch) - 1))

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        in_ch = channels[-1] + latent_dim
        out_ch = channels[-1]

        # self.style_fuse = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False),
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        #     )
        # self.style_fuse = ConvNeXtFuse(in_ch=in_ch, out_ch=out_ch, kernel_size=9, expansion=4)
        self.style_fuse = EnhancedStyleFuse(in_ch, out_ch)
        # self.style_fuse = SuperFuse(in_ch, out_ch)
        # self.style_fuse = ResDW5x5ECA(in_ch, out_ch)

    def compute_structural_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips.pop()
        return x, tuple(skips)

    def enrich_structural_embedding(self, feature_map: torch.Tensor, style_params: torch.Tensor) -> torch.Tensor:
        # return _adaptive_instance_norm(feature_map, style_params)
        B, _, H, W = feature_map.shape
        _, L = style_params.shape
        style_map = style_params.view(B, L, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([feature_map, style_map], dim=1)
        return self.style_fuse(x)

    def reconstruct(self, x: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        i = 0
        for up, skip in zip(self.ups, reversed(skips)):
            i += 1
            x = up(x, skip=skip)
        return torch.sigmoid(self.out_conv(x))

    def forward(self, x: torch.Tensor, style_params: torch.Tensor) -> torch.Tensor:
        feature_map, skips = self.compute_structural_embedding(x)
        styled_feature_map = self.enrich_structural_embedding(feature_map, style_params)
        return self.reconstruct(styled_feature_map, skips)
