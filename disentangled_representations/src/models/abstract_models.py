from abc import ABC, abstractmethod
from typing import final

import torch
import torch.nn as nn

class ImageEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: image.
        :return: image embedding.
        """
        pass

class DeterministicProjector(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        pass

class VariationalProjector(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        :param embedding: image embedding.
        :return: z_params -- variational (unsampled) latent vector parameters.
        """
        pass

    @staticmethod
    @final
    def sample_from_multivariate_normal(z_params: torch.Tensor) -> torch.Tensor:
        mu, log_variance = VariationalProjector.multivariate_params_from_vector(z_params)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    @staticmethod
    @final
    def multivariate_params_from_vector(z_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_variance = z_params.chunk(2, dim=1)
        return mu, log_variance
