"""Bayesian neural network base module(s) and utilities."""

import re
from typing import Any, Iterator

import torch

import bnn.modules
from bnn.modules import BayesianLayer, BayesianLayerSafe
import bnn.inference


# List out torch.nn modules known to be compatbiel with BayesianLayer.  Other
# layers will default to using BayesianLayerSafe.
BAYESIAN_LAYER_COMPATIBLE = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ReLU",
    "LeakyReLU",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
]


def convert_to_bnn_(
    model: torch.nn.Module,
    converter_kwargs: dict = {},
    converter_kwargs_global: dict = {},
    named_modules_to_convert: list = None,
) -> None:
    """Convert layers of `model` to Bayesian counterparts.

    Args:
        model: Model to be converted to Bayesian counterpart.
        converter_kwargs: Additional keyword arguments passed to
            initialization of named Bayesian layers.  For example, if `model`
            has a module named "module1", we'll convert "module1" as
            Converter(module1, **converter_kwargs["module1]) where
            Converter is a module converter.
        converter_kwargs_global: Keyword arguments that we'll merge
            with values of bayesian_module_kwargs as, e.g.,
            Converter(module1, **(converter_kwargs_global | converter_kwargs["module1]))
        named_modules_to_convert: List of named modules that we wish to
            convert to Bayesian modules (i.e., modules whose learnable parameters
            we wish to treat as distributions).
    """
    # Search for modules of `model` to convert, removing stem modules from the
    # list (we just want the leaf modules that contain parameters).
    if named_modules_to_convert is None:
        modules = [m for m in model.named_modules()]
    else:
        modules = named_modules_to_convert
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
        module = model.get_submodule(leaf)
        module_name = module.__class__.__name__

        # Search for an appropriate module converter in order: custom converters
        # for this named module, BayesianLayer for modules listed in
        # BAYESIAN_LAYER_COMPATIBLE (NOTE: this list was manually defined and many
        # modules in torch.nn that are compatible with BayesianLayer haven't been
        # added)., or finally using the BayesianLayerSafe converter which should
        # work for most modules but may be less efficient than custom or
        # BayesianLayer conversions.
        module_kwargs = converter_kwargs_global | converter_kwargs.pop(module_name, {})
        if hasattr(bnn.modules, module_name):
            # If a custom layer exists for this named layer, we'll use that by default.
            custom_layer = getattr(bnn.modules, module_name)(
                module=module, **module_kwargs
            )
            model.set_submodule(leaf, custom_layer)
        elif module.__class__.__name__ in BAYESIAN_LAYER_COMPATIBLE:
            # Search for a good default moment propagator.
            kwarg_overrides = {}
            if "moment_propagator" not in module_kwargs.keys():
                if hasattr(bnn.inference, module_name):
                    propagator = getattr(bnn.inference, module_name)
                    kwarg_overrides["moment_propagator"] = propagator()

            # Create Bayesian version of this layer.
            bayesian_layer = BayesianLayer(
                module=module, **(module_kwargs | kwarg_overrides)
            )
            model.set_submodule(leaf, bayesian_layer)
        else:
            # Use the "safe" converter which is expected to work for more
            # modules than `BayesianLayer`.
            bayesian_layer = BayesianLayerSafe(module=module, **module_kwargs)
            model.set_submodule(leaf, bayesian_layer)


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

    def named_parameters_mean(self) -> Iterator:
        """Return named mean parameters."""
        for name, param in self.named_parameters():
            if "_mean" in name:
                yield name, param

    def named_parameters_rho(self) -> Iterator:
        """Return named rho parameters."""
        for name, param in self.named_parameters():
            if "_rho" in name:
                yield name, param

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(*args, **kwargs)


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
