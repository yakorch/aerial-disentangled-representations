import torch

from disentangled_representations.src.models.abstract_models import VariationalProjector
from .loss_functions import KL_divergence_from_multivariate_standard_normal_loss


def compute_KL_loss(transient_params: torch.Tensor):
    mu, log_variance = VariationalProjector.multivariate_params_from_vector(transient_params)
    return KL_divergence_from_multivariate_standard_normal_loss(mu, log_variance)
