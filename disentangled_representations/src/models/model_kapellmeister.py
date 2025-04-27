import torch

from .abstract_models import ImageEncoder, DeterministicProjector, VariationalProjector


class Kapellmeister:
    def __init__(self, image_encoder: ImageEncoder, projector: DeterministicProjector | VariationalProjector):
        self.image_encoder = image_encoder
        self.projector = projector
        self.is_projector_variational = isinstance(projector, VariationalProjector)

    def compute_z_params(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: image
        :return: latent vector z.
        """
        embedding = self.image_encoder(x)
        return self.projector(embedding)
