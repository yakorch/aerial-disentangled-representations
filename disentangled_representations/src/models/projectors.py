from typing import override

import torch
from torch import nn

from .abstract_models import DeterministicProjector, VariationalProjector


class SimpleDeterministicProjector(DeterministicProjector):
    def __init__(self, input_dimensionality: int, hidden_features: list[int], output_dimensionality: int) -> None:
        super().__init__()

        self.latent_dimensionality = output_dimensionality

        dims = [input_dimensionality] + hidden_features + [output_dimensionality]

        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(nn.ReLU(inplace=True))
        self.MLP = nn.Sequential(*layers)

    @override
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.MLP(embedding)


class SimpleVariationalProjector(SimpleDeterministicProjector, VariationalProjector):
    def __init__(self, input_dimensionality: int, hidden_features: list[int], latent_dimensionality: int) -> None:
        super(SimpleDeterministicProjector, self).__init__(input_dimensionality, hidden_features, 2 * latent_dimensionality)