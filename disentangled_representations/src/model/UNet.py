from typing import override, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_models import I2IModel


class DoubleNonLinearConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.max_pool_conv = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleNonLinearConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, padding_mode="reflect"))

        self.conv = DoubleNonLinearConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2, ])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def adaptive_instance_norm(features: torch.Tensor, params: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    B, C, H, W = features.size()
    assert params.size(1) == 2 * C, f"Expected params of shape (B, {2 * C}), got {tuple(params.size())}"

    gamma, beta = params.chunk(2, dim=1)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)

    normalized = F.instance_norm(features, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, eps=eps)
    return gamma * normalized + beta


class UNet(I2IModel):
    def __init__(self, n_channels: int, out_channels: int) -> None:
        super().__init__()

        self.inc = DoubleNonLinearConv(n_channels, 64)

        self.down_1 = Down(64, 128)
        self.down_2 = Down(128, 256)
        self.down_3 = Down(256, 512)
        self.down_4 = Down(512, 512)
        self.down_5 = Down(512, 512)

        self.up_1 = Up(512 + 512, 512)
        self.up_2 = Up(512 + 512, 256)
        self.up_3 = Up(256 + 256, 128)
        self.up_4 = Up(128 + 128, 64)
        self.up_5 = Up(64 + 64, 64)

        self.out_conv = OutConv(64, out_channels)

    @override
    def compute_structural_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        x_1 = self.inc(x)

        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)
        x_5 = self.down_4(x_4)

        x_6 = self.down_5(x_5)
        return x_6, (x_5, x_4, x_3, x_2, x_1)

    @override
    def enrich_structural_embedding(self, feature_map: torch.Tensor, style_params: torch.Tensor) -> torch.Tensor:
        return adaptive_instance_norm(feature_map, style_params)

    @override
    def reconstruct(self, styled_feature_map: torch.Tensor, auxiliary_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        x_5, x_4, x_3, x_2, x_1 = auxiliary_tensors

        x = self.up_1(styled_feature_map, skip=x_5)
        x = self.up_2(x, skip=x_4)
        x = self.up_3(x, skip=x_3)
        x = self.up_4(x, skip=x_2)
        x = self.up_5(x, skip=x_1)

        return torch.tanh(self.out_conv(x))

    def forward(self, x: torch.Tensor, style_params: torch.Tensor) -> torch.Tensor:
        feature_map, auxiliary_tensors = self.compute_structural_embedding(x)
        enriched_feature_map = self.enrich_structural_embedding(feature_map, style_params)
        return self.reconstruct(enriched_feature_map, auxiliary_tensors)