"""Bayesian neural network base module(s) and utilities."""

from typing import Any, Iterator

import lightning as L
import torch

from bnn.types import MuVar
from bnn.wrappers import convert_to_bnn_


class BNN(torch.nn.Module):
    """Bayesian neural network base class."""

    def __init__(self, nn: torch.nn.Module, *args, **kwargs):
        """Initialize Bayesian neural network.


        WARNINGS:
            (1): Some functionality of this class relies on parameter names
                containing the suffixes "_mean" and "_rho".  If the input `nn`
                has parameters containing these strings, this class may not
                behave as expected!

        Args:
            nn: Neural network to be converted to a Bayesian neural network.
            args, kwargs: Passed as
                bnn.utils.convert_to_bnn_(model=nn, *args, **kwargs)
        """
        super().__init__()

        # Convert the neural network to a Bayesian neural network.
        convert_to_bnn_(model=nn, *args, **kwargs)
        self.bnn = nn

    def named_parameters_tagged(self, tag: str) -> Iterator:
        """Return named parameters whose name contains `tag`."""
        for name, param in self.named_parameters():
            if tag in name:
                yield name, param

    def load_state_dict(
        self, state_dict: dict, strict: bool = True, assign: bool = False
    ):
        """Overloaded load_state_dict() that can mean parameters from a non-Bayesian network."""
        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(*args, **kwargs)


class BNNLightning(L.LightningModule):
    """PyTorch Lightning wrapper for BNN class."""

    def __init__(self, bnn: BNN, loss: torch.nn.Module, *args, **kwargs):
        """Initialize Lightning wrapper..

        Args:
            bnn: Bayesian neural network to wrap in Lightning.
            loss: Loss function to call in training/validation.
        """
        super().__init__()

        self.bnn = bnn
        self.loss = loss

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for a single batch."""
        # Compute forward pass through model.
        out = self.bnn(MuVar(batch["input"]["x"]))

        # Compute loss.
        loss = self.loss(
            model=self.bnn, input=out[0], target=batch["output"], var=out[1]
        )

        # Log results.
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        # Compute forward pass through model.
        out = self.bnn(MuVar(batch["input"]["x"]))

        # Compute loss.
        loss = self.loss(
            model=self.bnn, input=out[0], target=batch["output"], var=out[1]
        )

        # Log results.
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss


if __name__ == "__main__":
    from models import mlp

    # Convert a model to a Bayesian counterpart.
    in_features = 3
    out_features = 1
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        n_hidden_layers=3,
        activation=torch.nn.LeakyReLU,
    )
    convert_to_bnn_(model=model)
    out = model(torch.ones(1, in_features))

    # Create a BNN wrapper for our model.
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        n_hidden_layers=3,
        activation=torch.nn.LeakyReLU,
    )
    bnn = BNN(nn=model)
    bnn(torch.randn((1, 3)))
