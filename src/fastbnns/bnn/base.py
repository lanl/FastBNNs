"""Bayesian neural network base module(s) and utilities."""

import copy
from typing import Any, Iterator, Union

import torch

from .types import MuVar
from .wrappers import convert_to_bnn_


class BNN(torch.nn.Module):
    """Bayesian neural network base class."""

    def __init__(
        self,
        nn: torch.nn.Module,
        convert_in_place: bool = False,
        *args,
        **kwargs,
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

    def forward(self, input: Union[MuVar, torch.Tensor], *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(input, *args, **kwargs)


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
