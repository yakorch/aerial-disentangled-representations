import warnings

warnings.filterwarnings('ignore', message=r'.*deprecated since 0\.13.*', category=UserWarning, module=r'torchvision\.models\._utils')

import torch

torch.set_float32_matmul_precision('high')

from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import defaultdict
from dataclasses import dataclass

import click
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .data_processing.aerial_dataset_loaders import create_train_data_loader_for_image_pairs, create_val_data_loader_for_image_pairs
from lightly.loss import NTXentLoss
from .losses.objective_components import compute_KL_loss
from .models.abstract_models import ImageEncoder, DeterministicProjector, VariationalProjector
from .models.model_kapellmeister import Kapellmeister


@dataclass
class LossWeights:
    w_NTXent: float
    w_KL: float


class LitKapellmeister(pl.LightningModule):
    def __init__(self, image_encoder: ImageEncoder, projector: DeterministicProjector | VariationalProjector, loss_weights: LossWeights, lr: float = 1e-4,
                 anneal_epochs: int = 10, temperature: float = 0.1):
        super().__init__()

        self.image_encoder = image_encoder
        self.projector = projector
        self.is_projector_variational = isinstance(projector, VariationalProjector)

        self.kapellmeister = Kapellmeister(image_encoder=image_encoder, projector=projector)

        self.anneal_epochs = anneal_epochs

        self.loss_weights = loss_weights
        self.lr = lr
        self.temperature = temperature
        self.loss_criterion = NTXentLoss(temperature=temperature)

        self.save_hyperparameters(ignore=["image_encoder", "projector"])

    def forward(self, X):
        return self.kapellmeister.compute_z_params(X)

    def _shared_step(self, A: torch.Tensor, B: torch.Tensor):
        losses = defaultdict(float)

        z_params_A = self.kapellmeister.compute_z_params(A)
        z_params_B = self.kapellmeister.compute_z_params(B)

        if self.is_projector_variational:
            losses["KL_loss"] += compute_KL_loss(z_params_A) * 0.5
            losses["KL_loss"] += compute_KL_loss(z_params_B) * 0.5

            z_A, z_B = VariationalProjector.sample_from_multivariate_normal(z_params_A), VariationalProjector.sample_from_multivariate_normal(z_params_B)
        else:
            z_A, z_B = z_params_A, z_params_B

        losses["NTXent"] = self.loss_criterion(z_A, z_B)

        anneal_factor = min(1.0, self.current_epoch / float(self.anneal_epochs)) if self.anneal_epochs > 0 else 1.0
        total = losses["KL_loss"] * self.loss_weights.w_KL * anneal_factor + losses["NTXent"] * self.loss_weights.w_NTXent

        assert len(losses.keys()) == 2, "Only KL and NTXent were expected."

        losses['loss'] = total
        return losses

    def _log_losses(self, losses: dict, prefix: str = '', on_step: bool = True, on_epoch: bool = True, prog_bar: bool = False):
        for name, value in losses.items():
            key = f"{prefix}{name}"
            show_bar = prog_bar if name == 'loss' else False
            self.log(key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar)

    def training_step(self, batch, batch_idx):
        A, B = batch
        losses = self._shared_step(A, B)
        self._log_losses(losses, prefix='train/', on_step=True, on_epoch=True, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        A, B = batch
        losses = self._shared_step(A, B)
        self._log_losses(losses, prefix='val/', on_step=False, on_epoch=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos', div_factor=5.0, final_div_factor=50)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1, }]


def parse_hidden_features(ctx, param, val):
    return [int(x) for x in val.split(',')]


@click.command()
@click.option('--batch_size', default=16, help='Training batch size')
@click.option('--val_batch_size', default=8, help='Validation batch size')
@click.option('--num_workers', default=4, help='DataLoader workers')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--max_epochs', default=50, help='Number of epochs')
@click.option('--hidden_features', default='512', callback=parse_hidden_features, help='CSV, e.g. `--hidden_features 512`.')
@click.option('--total_z_dimensionality', default=128, help='Projected latent dimensionality.')
@click.option("--variational", is_flag=True, default=False, help="Whether to use a variational z.")
@click.option('--temperature', default=0.1, help='NTXent loss temperature.')
@click.option('--w_ntxent', default=1.0, help='Cross consistency weight')
@click.option('--w_kl', default=0.5, help='KL loss weight')
@click.option('--accelerator', default='auto', help="Accelerator: 'cpu', 'gpu', 'mps', or 'auto'")
@click.option('--devices', default=1, type=int, help='Number of devices (e.g. GPUs or MPS)')
@click.option('--anneal_epochs', default=10, type=int, help='Number of epochs over which to linearly anneal KL.')
@click.option("--resume_from_checkpoint", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to a Lightning checkpoint to resume training from.", )
def main(batch_size, val_batch_size, num_workers, lr, max_epochs, hidden_features, total_z_dimensionality, variational, temperature, w_ntxent, w_kl,
         accelerator, devices, anneal_epochs, resume_from_checkpoint):
    loss_weights = LossWeights(w_NTXent=w_ntxent, w_KL=w_kl)

    train_loader = create_train_data_loader_for_image_pairs(batch_size=batch_size, num_workers=num_workers, )
    val_loader = create_val_data_loader_for_image_pairs(batch_size=val_batch_size, num_workers=2, )

    from .models.image_encoders import EfficientNetB0
    image_encoder = EfficientNetB0(in_channels=3)
    image_encoders_embedding_dim = image_encoder.feature_dim

    if variational:
        assert total_z_dimensionality % 2 == 0
        from .models.projectors import SimpleVariationalProjector
        projector = SimpleVariationalProjector(input_dimensionality=image_encoders_embedding_dim, hidden_features=hidden_features,
                                               latent_dimensionality=total_z_dimensionality // 2)
    else:
        from .models.projectors import SimpleDeterministicProjector
        projector = SimpleDeterministicProjector(input_dimensionality=image_encoders_embedding_dim, hidden_features=hidden_features,
                                                 output_dimensionality=total_z_dimensionality)

    model = LitKapellmeister(image_encoder=image_encoder, projector=projector, loss_weights=loss_weights, lr=lr, anneal_epochs=anneal_epochs,
                             temperature=temperature)

    logger = TensorBoardLogger("tb_logs", name="disent_rep")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=logger, callbacks=[lr_monitor])  # TODO: `precision=16` ?
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    main()
