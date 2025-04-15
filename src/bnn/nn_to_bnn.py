"""Functionality for converting neural networks to Bayesian neural networks."""

import re
from typing import Any

import torch

from bnn.layers import BayesianLayer, ForwardPassMean
import bnn.inference


# Define available Bayesian layers.
BAYESIAN_LAYERS = ("Linear", "Conv2d")


def convert_to_bnn_(
    model: torch.nn.Module,
    bayesian_layer_kwargs: dict = {},
):
    """Convert layers of `model` to Bayesian counterparts.

    Args:
        model: Model to be converted to Bayesian counterpart.
        bayesian_layer_kwargs: Additional keyword arguments passed to
            initialization of each Bayesian layer.
    """
    # Search for modules of `model` to convert, removing stem modules from the
    # list (we just want the leaf modules that contain parameters).
    modules = [m for m in model.named_modules()]
    leaf_names = []
    for module in modules[::-1]:
        # If other modules are a prefix of this modules name, we'll assume they
        # are this modules parent (hence not a leaf module).
        children = []
        for leaf in leaf_names:
            matches = re.match(f"{module[0]}.*", leaf)
            if matches is not None:
                children.append(matches)
        if len(children) == 0:
            leaf_names.append(module[0])

    # Replace leaf modules with Bayesian counterparts or compatible passthroughs.
    for leaf in leaf_names:
        layer = model.get_submodule(leaf)
        layer_name = layer.__class__.__name__
        if layer_name in BAYESIAN_LAYERS:
            # Search for a good default moment propagator.
            if "moment_propagator" not in bayesian_layer_kwargs.keys():
                if hasattr(bnn.inference, layer_name):
                    propagator = getattr(bnn.inference, layer_name)
                    bayesian_layer_kwargs["moment_propagator"] = propagator()

            # Create Bayesian version of this layer.
            bayesian_layer = BayesianLayer(module=layer, **bayesian_layer_kwargs)

            # Replace module with Bayesian version.
            model.set_submodule(leaf, bayesian_layer)
        else:
            # Use generic mean passthrough layer for compatibility with Bayesian layers.
            passthrough_layer = ForwardPassMean(layer=layer)
            model.set_submodule(leaf, passthrough_layer)


class BNN(torch.nn.Module):
    """Bayesian neural network basic module."""

    def __init__(self, model: torch.nn.Module):
        """Initialize Bayesian neural network class.

        Args:
            model: Standard (non-Bayesian) neural network to be converted to
                a corresponding Bayesian neural network.

        """
        super().__init__()

        # Convert model to a Bayesian neural network.
        convert_to_bnn_(model)
        self.model = model

    def forward(self, input: torch.tensor) -> Any:
        """Forward pass through model."""
        return self.model(input)


if __name__ == "__main__":
    from models import mlp

    # Convert a model to a Bayesian counterpart.
    in_features = 3
    out_features = 1
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        activation=torch.nn.LeakyReLU,
    )
    bnn = BNN(model=model)
    out = bnn(torch.ones(1, in_features))
