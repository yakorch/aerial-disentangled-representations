import torch

from disentangled_representations.src.models.abstract_models import VariationalTransientEncoder
from disentangled_representations.src.models.model_kapellmeister import ReconstructionMetadata, HiddenParams
from .loss_functions import L1_loss, KL_divergence_from_multivariate_standard_normal_loss, DISTSPerceptualLoss, Wasserstein_distance_between_normals, compute_feature_attention, weighted_perceptual_loss, VGGFeaturesExtractor

Perceptual_loss = DISTSPerceptualLoss()
vgg_feature_extractor = VGGFeaturesExtractor()

def compute_reconstruction_losses(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, ...]:
    l1_image_loss = L1_loss(A, B)
    perceptual_loss = Perceptual_loss(A, B)
    return l1_image_loss, perceptual_loss


def compute_KL_loss(transient_params: torch.Tensor):
    mu, log_variance = VariationalTransientEncoder.multivariate_params_from_vector(transient_params)
    return KL_divergence_from_multivariate_standard_normal_loss(mu, log_variance)


def compute_self_losses(X: torch.Tensor, recon_metadata: ReconstructionMetadata, cycled_hidden_params: HiddenParams) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_losses = compute_reconstruction_losses(recon_metadata.reconstruction, X)

    mu, log_variance = VariationalTransientEncoder.multivariate_params_from_vector(recon_metadata.hidden_params.transient_params)
    KL_loss = KL_divergence_from_multivariate_standard_normal_loss(mu, log_variance)

    structure_consistency_loss = L1_loss(recon_metadata.hidden_params.structural_feature_map, cycled_hidden_params.structural_feature_map)
    transient_consistency_loss = Wasserstein_distance_between_normals(mu, log_variance, *VariationalTransientEncoder.multivariate_params_from_vector(
        cycled_hidden_params.transient_params))

    return recon_losses, KL_loss, structure_consistency_loss, transient_consistency_loss


def compute_cross_losses(A: torch.Tensor, recon_metadata: ReconstructionMetadata, A_hat: torch.Tensor, A_hat_hidden_params: HiddenParams,
                         B_hat_hidden_params: HiddenParams) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    cross_recon_losses = compute_reconstruction_losses(A, A_hat)

    structure_cross_consistency_loss = L1_loss(recon_metadata.hidden_params.structural_feature_map, B_hat_hidden_params.structural_feature_map)

    transient_cross_consistency_loss = Wasserstein_distance_between_normals(
        *VariationalTransientEncoder.multivariate_params_from_vector(recon_metadata.hidden_params.transient_params),
        *VariationalTransientEncoder.multivariate_params_from_vector(A_hat_hidden_params.transient_params))

    return cross_recon_losses, structure_cross_consistency_loss, transient_cross_consistency_loss


def compute_attention_based_perceptual_loss(A: torch.Tensor, B: torch.Tensor, A_hat: torch.Tensor, B_hat: torch.Tensor):
    features = [vgg_feature_extractor(X) for X in [A, B]]
    features_hat = [vgg_feature_extractor(X) for X in [A_hat, B_hat]]

    mask = compute_feature_attention(*features)

    total = 0.0
    layer_weights = [1.0, 0.75, 0.5, 0.5]
    for i in range(2):
        total += weighted_perceptual_loss(mask, features[i], features_hat[i], layer_weights)
    return total
