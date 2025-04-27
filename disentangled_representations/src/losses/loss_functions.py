import timm
import torch
import torch.nn.functional as F
from DISTS_pytorch import DISTS


def L1_loss(x_1: torch.Tensor, x_2: torch.Tensor):
    return F.l1_loss(x_1, x_2, reduction='mean')


def KL_divergence_from_multivariate_standard_normal_loss(mu: torch.Tensor, log_variance: torch.Tensor):
    kl_per_sample = 0.5 * (mu.pow(2) + log_variance.exp() - 1 - log_variance)
    return kl_per_sample.mean()


class DISTSPerceptualLoss:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DISTS_loss = DISTS().to(device)

    def __call__(self, A: torch.Tensor, B: torch.Tensor):
        return self.DISTS_loss(A, B, require_grad=True, batch_average=True)


class VGGFeaturesExtractor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model('vgg19', pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), in_chans=1).to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)


def compute_feature_attention(features_A: list[torch.Tensor], features_B: torch.Tensor, min_val = 0.1) -> list[torch.Tensor]:
    masks = []
    for f_A, f_B in zip(features_A, features_B):
        diff = (f_A - f_B).abs().mean(dim=1, keepdim=True)
        mask = diff  # .pow(2)
        mask = torch.clamp(mask, min=min_val)
        masks.append(mask.detach())
    return masks


def weighted_perceptual_loss(masks, features_A, features_A_hat, layer_weights):
    total = 0.0
    for w, f_hat, f_ref, mask in zip(layer_weights, features_A_hat, features_A, masks):
        diff = (f_hat - f_ref)  # .pow(2)
        mask_exp = mask.expand_as(diff)
        total += w * (mask_exp * diff).mean()
    return total


def Wasserstein_distance_between_normals(mu_1: torch.Tensor, log_variance_1: torch.Tensor, mu_2: torch.Tensor, log_variance_2: torch.Tensor):
    var_1 = log_variance_1.exp()
    var_2 = log_variance_2.exp()

    term_1 = (mu_1 - mu_2).pow(2)
    term_2 = (var_1 + var_2 - 2 * (var_1 * var_2).sqrt())

    return (term_1 + term_2).mean()
