"""PyTorch Lightning modules for BNNs."""

from typing import Any, Callable
from functools import partial

import lightning as L
import torch

from fastbnns.bnn import base, losses, types


class BNNLightning(L.LightningModule):
    """PyTorch Lightning wrapper for BNN class."""

    def __init__(
        self,
        bnn: base.BNN,
        loss: losses.BNNLoss,
        optimizer: Callable = partial(torch.optim.AdamW, lr=1.0e-3),
    ) -> None:
        """Initialize Lightning wrapper.

        Args:
            bnn: Bayesian neural network to wrap in Lightning.
            loss: Loss function to call in training/validation.
            optimizer: Partially initialized optimizer that will be given parameters
                to optimize in self.configure_optimizers().
        """
        super().__init__()

        self.bnn = bnn
        self.loss = loss
        self.optimizer_fxn = optimizer

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(*args, **kwargs)

    def configure_optimizers(self):
        return self.optimizer_fxn(self.parameters())

    def training_step(self, batch, batch_idx):
        """Training step for a single batch."""
        # Compute forward pass through model.
        out = self.bnn(types.MuVar(batch[0]))

        # Compute loss.
        loss = self.loss(model=self.bnn, input=out[0], target=batch[1], var=out[1])

        # Log results.
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        # Compute forward pass through model.
        out = self.bnn(types.MuVar(batch[0]))

        # Compute loss.
        loss = self.loss(model=self.bnn, input=out[0], target=batch[1], var=out[1])

        # Log results.
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss
