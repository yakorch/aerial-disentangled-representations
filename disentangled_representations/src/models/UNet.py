from typing import Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet_parts import Down, Up, DoubleNonLinearConv, SeparableConv
from .abstract_models import I2IModel


def _adaptive_instance_norm(features: torch.Tensor, params: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    B, C, H, W = features.size()
    assert params.size(1) == 2 * C, f"Expected params of shape (B, {2 * C}), got {tuple(params.size())}"

    gamma, beta = params.chunk(2, dim=1)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)

    normalized = F.instance_norm(features, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, eps=eps)
    return gamma * normalized + beta

class UNet(I2IModel):
    def __init__(self, in_channels: int, out_channels: int, channels: Sequence[int], conv_block_down: Type[nn.Module] = DoubleNonLinearConv, conv_block_up: Type[nn.Module] = SeparableConv):
        super().__init__()
        assert len(channels) >= 1

        self.inc = DoubleNonLinearConv(in_channels, channels[0])

        self.downs = nn.ModuleList([Down(channels[i], channels[i + 1], conv_block=conv_block_down) for i in range(len(channels) - 1)])
        rev_ch = list(reversed(channels))
        self.ups = nn.ModuleList(
            Up(
                in_channels=rev_ch[i] + rev_ch[i + 1],
                out_channels=rev_ch[i + 1],
                conv_block=conv_block_up
            )
            for i in range(len(rev_ch) - 1)
        )

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

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
        return _adaptive_instance_norm(feature_map, style_params)

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
