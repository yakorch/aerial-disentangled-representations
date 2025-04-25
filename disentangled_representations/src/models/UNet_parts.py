from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleNonLinearConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect', groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_block: Type[nn.Module]) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), conv_block(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_block: Type[nn.Module]) -> None:
        super().__init__()

        decoder_channels = in_channels - out_channels
        skip_channels = out_channels
        self.built_for_in = 2 * skip_channels

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(decoder_channels, skip_channels, kernel_size=1, bias=False), )
        self.conv = conv_block(2 * skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
