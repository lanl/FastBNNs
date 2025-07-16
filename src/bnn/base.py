"""Bayesian neural network base module(s) and utilities."""

import copy
from functools import partial
from typing import Any, Callable, Iterator, Union

import laplace
import lightning as L
import torch

from bnn.losses import BNNLoss
from bnn.types import MuVar
from bnn.wrappers import convert_to_bnn_, convert_to_nn


class BNN(torch.nn.Module):
    """Bayesian neural network base class."""

    def __init__(
        self, nn: torch.nn.Module, convert_in_place: bool = True, *args, **kwargs
    ):
        """Initialize Bayesian neural network.


        WARNINGS:
            (1): Some functionality of this class relies on parameter names
                containing the suffixes "_mean" and "_rho".  If the input `nn`
                has parameters containing these strings, this class may not
                behave as expected!
            (2): The forward pass of `nn` is assumed to accept a single tensor
                representing the input.  The conversion to a BNN will hijack
                the forward pass through `nn` by changing the type of this
                input tensor to bnn.types.MuVar.

        Args:
            nn: Neural network to be converted to a Bayesian neural network.
            convert_in_place: Flag indicating input `nn` should be converted to
                a BNN in place.
            args, kwargs: Passed as
                bnn.utils.convert_to_bnn_(model=nn, *args, **kwargs)
        """
        super().__init__()

        # Convert the neural network to a Bayesian neural network.
        if not convert_in_place:
            nn = copy.deepcopy(nn)
        convert_to_bnn_(model=nn, *args, **kwargs)
        self.bnn = nn

    def named_parameters_tagged(self, tag: str) -> Iterator:
        """Return named parameters whose name contains `tag`."""
        for name, param in self.named_parameters():
            if tag in name:
                yield name, param

    def laplace_init(
        self, dataloader: torch.utils.data.DataLoader, laplace_kwargs: dict = {}
    ) -> None:
        """Initialize model parameter variances to Laplace approximated values."""
        # Define default arguments for DiagLaplace.
        laplace_kwargs_default = {"likelihood": "regression", "prior_precision": 0.0}
        laplace_kwargs = laplace_kwargs_default | laplace_kwargs

        # Compute Laplace approximation.
        model = convert_to_nn(self.bnn)
        la = laplace.DiagLaplace(model, **laplace_kwargs)
        la.fit(dataloader)

        # Define an inverse scale transform to convert scale parameters
        # (st. dev. from Laplace approximation) to `rho` parameters learned by the BNN.
        def inv_scale_tform(scale: torch.Tensor) -> torch.Tensor:
            return torch.log(torch.exp(scale) - 1.0)

        # Remap LA parameters to model parameters (la.params shares ordering of model.named_parameters()).
        param_dict = {}
        var_ind = 0  # pointer to track start of variances for each parameter
        for n, param in enumerate(model.named_parameters()):
            name_split = param[0].split(".")
            base_name = (
                f'bnn.{".".join(name_split[:-1])}._module_params.{name_split[-1]}'
            )
            param_dict[base_name + "_mean"] = la.params[n]
            param_dict[base_name + "_rho"] = inv_scale_tform(
                la.posterior_scale[var_ind : (var_ind + la.params[n].numel())]
            ).reshape(la.params[n].shape)
            var_ind += la.params[n].numel()

        # Load LA parameters into model.
        self.bnn.load_state_dict(param_dict, strict=False)

    def forward(self, input: Union[MuVar, torch.Tensor], *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(input, *args, **kwargs)


class BNNLightning(L.LightningModule):
    """PyTorch Lightning wrapper for BNN class."""

    def __init__(
        self,
        bnn: BNN,
        loss: BNNLoss,
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
        out = self.bnn(MuVar(batch[0]))

        # Compute loss.
        loss = self.loss(model=self.bnn, input=out[0], target=batch[1], var=out[1])

        # Log results.
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        # Compute forward pass through model.
        out = self.bnn(MuVar(batch[0]))

        # Compute loss.
        loss = self.loss(model=self.bnn, input=out[0], target=batch[1], var=out[1])

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
