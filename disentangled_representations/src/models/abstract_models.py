from abc import ABC, abstractmethod
from typing import final, Optional, Sequence

import torch
import torch.nn as nn


class I2IModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        :param x: image
        :param z: AdaIN parameters.
        """
        pass

    @abstractmethod
    def compute_structural_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[Sequence[torch.Tensor]]]:
        """
        :param x: image
        :return: feature map of the lowest spatial resolution; some intermediate auxiliary tensors could be the 2nd argument, such as skip connections.
        """
        pass

    @abstractmethod
    def enrich_structural_embedding(self, feature_map: torch.Tensor, style_params: torch.Tensor) -> torch.Tensor:
        """
        :param feature_map: low-res feature map.
        :param style_params: parameters used to enrich the feature map. for example, AdaIN parameters.
        :return:
        """
        pass

    @abstractmethod
    def reconstruct(self, styled_feature_map: torch.Tensor, auxiliary_tensors: Optional[Sequence[torch.Tensor]]) -> torch.Tensor:
        pass


class VariationalTransientEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: image.
        :return: z_params -- variational (unsampled) latent vector parameters.
        """
        pass

    @staticmethod
    @final
    def sample_from_multivariate_normal(z_params: torch.Tensor) -> torch.Tensor:
        mu, log_variance = VariationalTransientEncoder.multivariate_params_from_vector(z_params)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    @staticmethod
    @final
    def multivariate_params_from_vector(z_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_variance = z_params.chunk(2, dim=1)
        return mu, log_variance