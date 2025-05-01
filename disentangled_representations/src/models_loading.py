import pathlib
# NOTE: work-around for pickle loading.
import __main__
from disentangled_representations.src.training_procedure import LitKapellmeister, LossWeights
__main__.LitKapellmeister = LitKapellmeister
__main__.LossWeights      = LossWeights

from disentangled_representations.src.training_procedure import LitKapellmeister, LossWeights
from disentangled_representations.src.models.projectors import SimpleDeterministicProjector, SimpleVariationalProjector
from disentangled_representations.src.models.image_encoders import EfficientNetB0
import torch
from loguru import logger


def load_models_by_checkpoint_dir_path(ckpt_dir: pathlib.Path, out_dim: int, variational: bool, device: str | torch.device):
    assert ckpt_dir.is_dir()

    _ckpt_paths = list(ckpt_dir.glob("*.ckpt"))
    if len(_ckpt_paths) != 1:
        logger.warning(f"{_ckpt_paths=}")

    ckpt_path = _ckpt_paths[0]

    encoder = EfficientNetB0(in_channels=1)
    embedding_dim = int(encoder.feature_dim)

    if variational:
        projector = SimpleVariationalProjector(
            input_dimensionality=embedding_dim,
            hidden_features=[512],
            latent_dimensionality=out_dim // 2
        )
    else:
        projector = SimpleDeterministicProjector(
            input_dimensionality=embedding_dim,
            hidden_features=[512],
            output_dimensionality=out_dim
        )

    loss_weights = LossWeights(w_NTXent=1.0, w_KL=0.5)

    model = LitKapellmeister.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        image_encoder=encoder,
        projector=projector,
        loss_weights=loss_weights,
        map_location=device,
        strict=True
    )
    return model