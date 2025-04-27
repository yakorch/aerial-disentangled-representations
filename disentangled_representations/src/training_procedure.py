import warnings

warnings.filterwarnings('ignore', message=r'.*deprecated since 0\.13.*', category=UserWarning, module=r'torchvision\.models\._utils')

import torch

torch.set_float32_matmul_precision('high')

from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision.utils as vutils
from collections import defaultdict
from dataclasses import dataclass

import click
import pytorch_lightning as pl

import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from .data_processing.aerial_dataset_loaders import create_train_data_loader_for_image_pairs, create_val_data_loader_for_image_pairs
from .losses.objective_components import compute_cross_losses, compute_self_losses, compute_reconstruction_losses, compute_KL_loss
from .models.model_kapellmeister import I2IModel, Kapellmeister, VariationalTransientEncoder, SimplifiedCrossReconstructionMeta


@dataclass
class LossWeights:
    w_L1_image: float = 0.25
    w_perceptual: float = 1.0
    w_cross_recon: float = 2.0
    w_KL: float = 1.0
    w_struct_consistency: float = 0.5
    w_transient_consistency: float = 0.5
    w_cross_consistency: float = 0.5


class LitKapellmeister(pl.LightningModule):
    def __init__(self, I2I_model: I2IModel, variational_transient_encoder: VariationalTransientEncoder, style_params_MLP: torch.nn.Module,
                 loss_weights: LossWeights, lr: float = 1e-4, anneal_epochs: int = 10, simplified_training: bool = True):
        super().__init__()

        self.I2I_model = I2I_model
        self.variational_transient_encoder = variational_transient_encoder
        self.style_params_MLP = style_params_MLP

        self.anneal_epochs = anneal_epochs

        self.kapellmeister = Kapellmeister(I2I_model=self.I2I_model, variational_transient_encoder=self.variational_transient_encoder,
                                           style_params_MLP=self.style_params_MLP, )

        self.loss_weights = loss_weights
        self.lr = lr

        self.simplified_training = simplified_training

        self.save_hyperparameters(ignore=["I2I_model", "variational_transient_encoder", "style_params_MLP"])

    def forward(self, A, B):
        return self.kapellmeister.all_reconstructions(A, B)

    def _shared_step_complex(self, A: torch.Tensor, B: torch.Tensor):
        meta = self.kapellmeister.all_reconstructions(A, B)
        originals = [A, B]
        self_metas = [meta.a_recon_metadata, meta.b_recon_metadata]
        cycled = [meta.a_cycled_hidden_params, meta.b_cycled_hidden_params]
        hats = [meta.a_hat, meta.b_hat]
        hat_params = [meta.a_hat_hidden_params, meta.b_hat_hidden_params]

        losses = defaultdict(float)

        for i in range(2):
            recon_losses, KL_loss, struct_loss, trans_loss = compute_self_losses(X=originals[i], recon_metadata=self_metas[i], cycled_hidden_params=cycled[i])
            cross_losses, struct_cross, trans_cross = compute_cross_losses(A=originals[i], recon_metadata=self_metas[i], A_hat=hats[i],
                                                                           A_hat_hidden_params=hat_params[i], B_hat_hidden_params=hat_params[1 - i])

            losses['self_recon_L1'] += recon_losses[0]
            losses['self_recon_perceptual'] += recon_losses[1]
            losses['cross_recon_L1'] += cross_losses[0]
            losses['cross_recon_perceptual'] += cross_losses[1]
            losses['KL'] += KL_loss
            losses["self_struct_consistency"] += struct_loss
            losses["cross_struct_consistency"] += struct_cross
            losses["self_transient_consistency"] += trans_loss
            losses["cross_transient_consistency"] += trans_cross

        anneal_factor = min(1.0, self.current_epoch / float(self.anneal_epochs)) if self.anneal_epochs > 0 else 1.0

        total = ((losses['self_recon_L1'] + losses['cross_recon_L1'] * self.loss_weights.w_cross_recon) * self.loss_weights.w_L1_image + (
                losses['self_recon_perceptual'] + losses[
            'cross_recon_perceptual'] * self.loss_weights.w_cross_recon) * self.loss_weights.w_perceptual + anneal_factor * (
                         losses['KL'] * self.loss_weights.w_KL + (losses["self_struct_consistency"] + self.loss_weights.w_cross_consistency * losses[
                     "cross_struct_consistency"]) * self.loss_weights.w_struct_consistency + ((losses["self_transient_consistency"] + losses[
                     "cross_transient_consistency"] * self.loss_weights.w_cross_consistency) * self.loss_weights.w_transient_consistency)))
        losses['total'] = total
        return losses

    def _shared_step_simplified(self, A: torch.Tensor, B: torch.Tensor):
        losses = defaultdict(float)

        simple_meta: SimplifiedCrossReconstructionMeta = self.kapellmeister.cross_reconstructions(A, B)
        A_hat, B_hat = simple_meta.a_hat, simple_meta.b_hat

        recon_losses_A = compute_reconstruction_losses(A, A_hat)
        recon_losses_B = compute_reconstruction_losses(B, B_hat)

        A_KL_loss, B_KL_loss = compute_KL_loss(simple_meta.a_transient_params), compute_KL_loss(simple_meta.b_transient_params)

        anneal_factor = min(1.0, self.current_epoch / float(self.anneal_epochs)) if self.anneal_epochs > 0 else 1.0

        losses["cross_recon_L1"] += (recon_losses_A[0] + recon_losses_B[0]) * 0.5
        losses["cross_recon_perceptual"] += (recon_losses_A[1] + recon_losses_B[1]) * 0.5
        losses["KL"] += (A_KL_loss + B_KL_loss) * 0.5
        losses["total"] = losses["cross_recon_L1"] * self.loss_weights.w_L1_image + losses["cross_recon_perceptual"] * self.loss_weights.w_perceptual + losses[
            "KL"] * self.loss_weights.w_KL * anneal_factor
        return losses

    def _log_losses(self, losses: dict, prefix: str = '', on_step: bool = True, on_epoch: bool = True, prog_bar: bool = False):
        for name, value in losses.items():
            key = f"{prefix}{name}"
            show_bar = prog_bar if name == 'total' else False
            self.log(key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar)

    def training_step(self, batch, batch_idx):
        A, B = batch
        if self.simplified_training:
            losses = self._shared_step_simplified(A, B)
        else:
            losses = self._shared_step_complex(A, B)
        self._log_losses(losses, prefix='train/', on_step=True, on_epoch=True, prog_bar=True)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        A, B = batch
        if self.simplified_training:
            losses = self._shared_step_simplified(A, B)
        else:
            losses = self._shared_step_complex(A, B)
        self._log_losses(losses, prefix='val/', on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx != 0:
            return losses

        with torch.no_grad():
            meta = self.kapellmeister.cross_reconstructions(A.to(self.device), B.to(self.device))

            cross_images = torch.cat([A, B, meta.a_hat, meta.b_hat], dim=0)
            grid_cross = vutils.make_grid(cross_images, nrow=A.size(0), normalize=False, value_range=(0, 1))
            self.logger.experiment.add_image("val/cross_reconstructions", grid_cross, self.current_epoch)

        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos', div_factor=5.0, final_div_factor=50)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1, }]


def parse_channels(ctx, param, val):
    return [int(x) for x in val.split(',')]


@click.command()
@click.option('--batch_size', default=16, help='Training batch size')
@click.option('--val_batch_size', default=8, help='Validation batch size')
@click.option('--num_workers', default=4, help='DataLoader workers')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_epochs', default=50, help='Number of epochs')
@click.option('--unet_channels', default='32,64,128,256', callback=parse_channels, help='CSV, e.g. `--unet-channels 32,64,128,256`.')
@click.option('--latent_d', default=128, help='Transient latent dimensionality.')
@click.option('--w_l1_image', default=1, help='L1 loss weight')
@click.option('--w_perceptual', default=2.5, help='Perceptual loss weight')
@click.option('--w_cross_recon', default=5.0, help='How much cross reconstruction is more important than self reconstruction.')
@click.option('--w_kl', default=0.5, help='KL loss weight')
@click.option('--w_struct_consistency', default=0.5, help='Structural consistency weight')
@click.option('--w_transient_consistency', default=0.75, help='Transient consistency weight')
@click.option('--w_cross_consistency', default=0.5, help='Cross consistency weight')
@click.option('--accelerator', default='auto', help="Accelerator: 'cpu', 'gpu', 'mps', or 'auto'")
@click.option('--devices', default=1, type=int, help='Number of devices (e.g. GPUs or MPS)')
@click.option('--anneal_epochs', default=10, type=int, help='Number of epochs over which to linearly anneal KL & consistency terms')
@click.option("--resume_from_checkpoint", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to a Lightning checkpoint to resume training from.", )
def main(batch_size, val_batch_size, num_workers, lr, max_epochs, unet_channels, latent_d, w_l1_image, w_perceptual, w_cross_recon, w_kl, w_struct_consistency,
         w_transient_consistency, w_cross_consistency, accelerator, devices, anneal_epochs, resume_from_checkpoint):
    loss_weights = LossWeights(w_L1_image=w_l1_image, w_perceptual=w_perceptual, w_cross_recon=w_cross_recon, w_KL=w_kl,
                               w_struct_consistency=w_struct_consistency, w_transient_consistency=w_transient_consistency,
                               w_cross_consistency=w_cross_consistency)

    train_loader = create_train_data_loader_for_image_pairs(batch_size=batch_size, num_workers=num_workers, )
    val_loader = create_val_data_loader_for_image_pairs(batch_size=val_batch_size, num_workers=2, )

    from .models.transient_encoders import EfficientNetB0VariationalTransientEncoder
    from .models.UNet import UNet
    from .models.UNet_parts import DoubleNonLinearConv

    # efficient_net_b0 = EfficientNetB0VariationalTransientEncoder(in_channels=1, latent_dimensionality=latent_d)

    variational_transient_encoder = EfficientNetB0VariationalTransientEncoder(in_channels=1, latent_dimensionality=latent_d)
    # style_params_MLP = nn.Sequential(nn.Linear(latent_d, latent_d), nn.ReLU(inplace=True), nn.Linear(latent_d, 2 * unet_channels[-1]))
    MLP_out = latent_d
    style_params_MLP = nn.Sequential(nn.Linear(latent_d, latent_d), nn.ReLU(inplace=True), nn.Linear(latent_d, MLP_out))
    I2I_model = UNet(in_channels=1, out_channels=1, channels=unet_channels, conv_block_down=DoubleNonLinearConv, conv_block_up=DoubleNonLinearConv,
                     latent_dim=MLP_out)

    model = LitKapellmeister(I2I_model=I2I_model, variational_transient_encoder=variational_transient_encoder, style_params_MLP=style_params_MLP,
                             loss_weights=loss_weights, lr=lr, anneal_epochs=anneal_epochs, simplified_training=True)

    logger = TensorBoardLogger("tb_logs", name="disent_rep")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=logger, callbacks=[lr_monitor])  # TODO: `precision=16` ?
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    main()
