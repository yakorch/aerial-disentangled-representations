from dataclasses import dataclass
from typing import Sequence, Optional

import torch
import torch.nn as nn

from .abstract_models import I2IModel, VariationalTransientEncoder


@dataclass
class HiddenParams:
    transient_params: torch.Tensor
    transient_sample: Optional[torch.Tensor]
    structural_feature_map: torch.Tensor


@dataclass
class ReconstructionMetadata:
    hidden_params: HiddenParams

    style_params: torch.Tensor
    auxiliary: Sequence[torch.Tensor]
    reconstruction: torch.Tensor


@dataclass
class AllReconstructionsIntermediateMetadata:
    a_recon_metadata: ReconstructionMetadata
    b_recon_metadata: ReconstructionMetadata

    a_cycled_hidden_params: HiddenParams
    b_cycled_hidden_params: HiddenParams

    a_hat: torch.Tensor
    b_hat: torch.Tensor

    a_hat_hidden_params: HiddenParams
    b_hat_hidden_params: HiddenParams


class Kapellmeister:
    def __init__(self, I2I_model: I2IModel, variational_transient_encoder: VariationalTransientEncoder, style_params_MLP: nn.Module) -> None:
        self.I2I_model = I2I_model

        self.variational_transient_encoder = variational_transient_encoder
        self.style_params_MLP = style_params_MLP

    def half_forward_pass(self, x: torch.Tensor) -> tuple[HiddenParams, Sequence[torch.Tensor]]:
        x_transient_params = self.variational_transient_encoder(x)
        x_structure, x_auxiliary = self.I2I_model.compute_structural_embedding(x)
        return HiddenParams(transient_params=x_transient_params, transient_sample=None, structural_feature_map=x_structure), x_auxiliary

    def self_reconstruction(self, x: torch.Tensor) -> ReconstructionMetadata:
        """
        :param x: image.
        """
        hidden_params, x_auxiliary = self.half_forward_pass(x)
        z = VariationalTransientEncoder.sample_from_multivariate_normal(hidden_params.transient_params)
        hidden_params.transient_sample = z
        style_params = self.style_params_MLP(z)

        x_structure_enriched = self.I2I_model.enrich_structural_embedding(hidden_params.structural_feature_map, style_params)
        x_hat = self.I2I_model.reconstruct(x_structure_enriched, x_auxiliary)

        return ReconstructionMetadata(hidden_params=hidden_params, style_params=style_params, auxiliary=x_auxiliary, reconstruction=x_hat)

    def all_reconstructions(self, a: torch.Tensor, b: torch.Tensor) -> AllReconstructionsIntermediateMetadata:
        a_recon_metadata = self.self_reconstruction(a)
        b_recon_metadata = self.self_reconstruction(b)

        a_cycled_hidden_params, _ = self.half_forward_pass(a_recon_metadata.reconstruction)
        b_cycled_hidden_params, _ = self.half_forward_pass(b_recon_metadata.reconstruction)

        a_structure_enriched_with_b_style = self.I2I_model.enrich_structural_embedding(a_recon_metadata.hidden_params.structural_feature_map,
                                                                                       b_recon_metadata.style_params)
        b_hat = self.I2I_model.reconstruct(a_structure_enriched_with_b_style, a_recon_metadata.auxiliary)

        b_structure_enriched_with_a_style = self.I2I_model.enrich_structural_embedding(b_recon_metadata.hidden_params.structural_feature_map,
                                                                                       a_recon_metadata.style_params)
        a_hat = self.I2I_model.reconstruct(b_structure_enriched_with_a_style, b_recon_metadata.auxiliary)

        a_hat_structure, _ = self.I2I_model.compute_structural_embedding(a_hat)
        b_hat_structure, _ = self.I2I_model.compute_structural_embedding(b_hat)

        a_hat_latent_params = self.variational_transient_encoder(a_hat)
        b_hat_latent_params = self.variational_transient_encoder(b_hat)

        return AllReconstructionsIntermediateMetadata(a_recon_metadata=a_recon_metadata, b_recon_metadata=b_recon_metadata,
                                                      a_cycled_hidden_params=a_cycled_hidden_params, b_cycled_hidden_params=b_cycled_hidden_params, a_hat=a_hat,
                                                      b_hat=b_hat, a_hat_hidden_params=HiddenParams(transient_params=a_hat_latent_params, transient_sample=None,
                                                                                                    structural_feature_map=a_hat_structure),
                                                      b_hat_hidden_params=HiddenParams(transient_params=b_hat_latent_params, transient_sample=None,
                                                                                       structural_feature_map=b_hat_structure))
