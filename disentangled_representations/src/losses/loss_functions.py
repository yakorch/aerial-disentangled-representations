import torch
import torch.nn.functional as F


def L1_loss(x_1: torch.Tensor, x_2: torch.Tensor):
    return F.l1_loss(x_1, x_2, reduction='mean')


def KL_divergence_from_multivariate_standard_normal_loss(mu: torch.Tensor, log_variance: torch.Tensor):
    kl_per_sample = 0.5 * (mu.pow(2) + log_variance.exp() - 1 - log_variance)
    return kl_per_sample.mean()


def Wasserstein_distance_between_normals(mu_1: torch.Tensor, log_variance_1: torch.Tensor, mu_2: torch.Tensor, log_variance_2: torch.Tensor):
    var_1 = log_variance_1.exp()
    var_2 = log_variance_2.exp()

    term_1 = (mu_1 - mu_2).pow(2)
    term_2 = (var_1 + var_2 - 2 * (var_1 * var_2).sqrt())

    return (term_1 + term_2).mean()
