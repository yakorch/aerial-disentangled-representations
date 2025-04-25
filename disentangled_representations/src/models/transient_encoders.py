from typing import override

import torch
from torch import nn

from .abstract_models import VariationalTransientEncoder


class EfficientNetB0VariationalTransientEncoder(VariationalTransientEncoder):
    def __init__(self, in_channels: int, latent_dimensionality: int) -> None:
        super().__init__()

        self.latent_dimensionality = latent_dimensionality

        import timm
        self.model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=in_channels, )
        self.model.classifier = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, 2 * latent_dimensionality), )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
