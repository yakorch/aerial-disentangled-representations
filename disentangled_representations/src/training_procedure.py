from collections import defaultdict
from dataclasses import dataclass

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from .data_processing.aerial_dataset_loaders import create_train_data_loader_for_image_pairs, create_val_data_loader_for_image_pairs
from .losses.objective_components import compute_cross_losses, compute_self_losses
from .models.model_kapellmeister import I2IModel, Kapellmeister, VariationalTransientEncoder


@dataclass
class LossWeights:
    w_L1_image: float = 0.25
    w_perceptual: float = 1.0
    w_KL: float = 1.0
    w_struct_consistency: float = 0.5
    w_transient_consistency: float = 0.5
    w_cross_consistency: float = 0.5


class LitKapellmeister(pl.LightningModule):
    def __init__(self, I2I_model: I2IModel, variational_transient_encoder: VariationalTransientEncoder, style_params_MLP: torch.nn.Module,
                 loss_weights: LossWeights, lr: float = 1e-4, ):
        super().__init__()

        self.I2I_model = I2I_model
        self.variational_transient_encoder = variational_transient_encoder
        self.style_params_MLP = style_params_MLP

        self.kapellmeister = Kapellmeister(I2I_model=self.I2I_model, variational_transient_encoder=self.variational_transient_encoder,
                                           style_params_MLP=self.style_params_MLP, )

        self.loss_weights = loss_weights
        self.lr = lr

        self.save_hyperparameters(ignore=["i2i_model", "variational_transient_encoder", "style_params_MLP"])

    def forward(self, A, B):
        return self.kapellmeister.all_reconstructions(A, B)

    def _shared_step(self, A: torch.Tensor, B: torch.Tensor):
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

            losses['self_L1'] += recon_losses[0]
            losses['self_perceptual'] += recon_losses[1]
            losses['cross_L1'] += cross_losses[0]
            losses['cross_perceptual'] += cross_losses[1]
            losses['KL'] += KL_loss
            losses['struct_consistency'] += struct_loss + struct_cross * self.loss_weights.w_cross_consistency
            losses['transient_consistency'] += trans_loss + trans_cross * self.loss_weights.w_cross_consistency

        total = ((losses['self_L1'] + losses['cross_L1']) * self.loss_weights.w_L1_image + (
                losses['self_perceptual'] + losses['cross_perceptual']) * self.loss_weights.w_perceptual + losses['KL'] * self.loss_weights.w_KL + losses[
                     'struct_consistency'] * self.loss_weights.w_struct_consistency + losses[
                     'transient_consistency'] * self.loss_weights.w_transient_consistency)
        losses['total'] = total
        return losses

    def _log_losses(self, losses: dict, prefix: str = '', on_step: bool = True, on_epoch: bool = True, prog_bar: bool = False):
        for name, value in losses.items():
            key = f"{prefix}{name}"
            show_bar = prog_bar if name == 'total' else False
            self.log(key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar)

    def training_step(self, batch, batch_idx):
        A, B = batch
        losses = self._shared_step(A, B)
        self._log_losses(losses, prefix='train/', on_step=True, on_epoch=True, prog_bar=True)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        A, B = batch
        losses = self._shared_step(A, B)
        self._log_losses(losses, prefix='val/', on_step=False, on_epoch=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
@click.option('--w_l1_image', default=0.25, help='L1 loss weight')
@click.option('--w_perceptual', default=1.0, help='Perceptual loss weight')
@click.option('--w_kl', default=1.0, help='KL loss weight')
@click.option('--w_struct_consistency', default=0.5, help='Structural consistency weight')
@click.option('--w_transient_consistency', default=0.5, help='Transient consistency weight')
@click.option('--w_cross_consistency', default=0.5, help='Cross consistency weight')
@click.option('--accelerator', default='auto', help="Accelerator: 'cpu', 'gpu', 'mps', or 'auto'")
@click.option('--devices', default=1, type=int, help='Number of devices (e.g. GPUs or MPS)')
def main(batch_size, val_batch_size, num_workers, lr, max_epochs, unet_channels, latent_d, w_l1_image, w_perceptual, w_kl, w_struct_consistency,
         w_transient_consistency, w_cross_consistency, accelerator, devices):
    loss_weights = LossWeights(w_L1_image=w_l1_image, w_perceptual=w_perceptual, w_KL=w_kl, w_struct_consistency=w_struct_consistency,
                               w_transient_consistency=w_transient_consistency, w_cross_consistency=w_cross_consistency)

    train_loader = create_train_data_loader_for_image_pairs(batch_size=batch_size, num_workers=num_workers, )
    val_loader = create_val_data_loader_for_image_pairs(batch_size=val_batch_size, num_workers=2, )

    from .models.transient_encoders import EfficientNetB0VariationalTransientEncoder
    from .models.UNet import UNet

    unet = UNet(in_channels=1, out_channels=1, channels=unet_channels)
    efficient_net_b0 = EfficientNetB0VariationalTransientEncoder(in_channels=1, latent_dimensionality=latent_d)

    I2I_model = unet
    variational_transient_encoder = efficient_net_b0
    style_params_MLP = nn.Sequential(nn.Linear(latent_d, latent_d), nn.ReLU(), nn.Linear(latent_d, 2 * unet_channels[-1]))

    model = LitKapellmeister(I2I_model=I2I_model, variational_transient_encoder=variational_transient_encoder, style_params_MLP=style_params_MLP,
                             loss_weights=loss_weights, lr=lr)

    logger = TensorBoardLogger("tb_logs", name="disent_rep")

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=logger, )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
