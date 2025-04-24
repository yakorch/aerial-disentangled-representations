import torch
import torch.nn.functional as F
from DISTS_pytorch import DISTS


def L1_loss(x: torch.Tensor, x_hat: torch.Tensor):
    return F.l1_loss(x, x_hat, reduction='sum') / x.shape[0]

def KL_loss(mu: torch.Tensor, log_variance: torch.Tensor):
    return 0.5 * torch.sum((mu ** 2 + torch.exp(log_variance) - 1 - log_variance), dim=1).mean()


_DISTS_loss = DISTS()

def perceptual_loss(A: torch.Tensor, B: torch.Tensor):
    return _DISTS_loss(A, B, require_grad=True, batch_average=True)
