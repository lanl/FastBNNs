"""Bayesian neural network base module(s) and utilities."""

import re
from typing import Any

import torch

import bnn.modules
from bnn.modules import BayesianLayer, ForwardPassMean
import bnn.inference
from utils.misc import get_torch_functional


def set_requires_grad_loc_(model: torch.nn.Module, requires_grad: bool) -> list[str]:
    """Change requires_grad of loc parameters of Bayesian layers in `model`.

    Args:
        model: Model for which we'll call .set_requires_grad_loc() on all
            BayesianLayer modules.
        requires_grad: Value we wish to set for requires_grad property of
            Bayesian layer loc parameters.

    Return:
        set: Lists of strings containing module names that
            had .set_requires_grad_loc() called.
    """
    set = []
    for module in model.named_modules():
        if hasattr(module[1], "set_requires_grad_loc"):
            module[1].set_requires_grad_loc(requires_grad=requires_grad)
            set.append(module[0])

    return set


def set_requires_grad_scale_(model: torch.nn.Module, requires_grad: bool) -> list[str]:
    """Change requires_grad of scale parameters of Bayesian layers in `model`.

    Args:
        model: Model for which we'll call .set_requires_grad_scale() on all
            BayesianLayer modules.
        requires_grad: Value we wish to set for requires_grad property of
            Bayesian layer scale parameters.

    Return:
        set: Lists of strings containing module names that
            had .set_requires_grad_scale() called.
    """
    set = []
    for module in model.named_modules():
        if hasattr(module[1], "set_requires_grad_scale"):
            module[1].set_requires_grad_scale(requires_grad=requires_grad)
            set.append(module[0])

    return set


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

        # Use default Bayesian layer constructor (which works for layers with
        # functionals in torch.nn.functional), otherwise fallback to a mean
        # passthrough layer.
        functional = get_torch_functional(module.__class__)
        module_kwargs = converter_kwargs_global | converter_kwargs.pop(module_name, {})
        if hasattr(bnn.modules, module_name):
            # If a custom layer exists for this named layer, we'll use that by default.
            custom_layer = getattr(bnn.modules, module_name)(
                module=module, **module_kwargs
            )
            model.set_submodule(leaf, custom_layer)
        elif functional is not None:
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
            # Use generic mean passthrough layer for compatibility with Bayesian layers.
            passthrough_layer = ForwardPassMean(module=module, **module_kwargs)
            model.set_submodule(leaf, passthrough_layer)


class BNN(torch.nn.Module):
    """Bayesian neural network base class."""

    def __init__(self, nn: torch.nn.Module, *args, **kwargs):
        """Initialize Bayesian neural network.

        Args:
            nn: Neural network to be converted to a Bayesian neural network.
            args, kwargs: Passed as
                bnn.utils.convert_to_bnn_(model=nn, *args, **kwargs)
        """
        super().__init__()

        # Convert the neural network to a Bayesian neural network.
        convert_to_bnn_(model=nn, *args, **kwargs)
        self.bnn = nn

    def set_requires_grad(self, property: str, requires_grad: bool) -> None:
        """Modify requires grad property of BNN parameters."""
        if property == "loc":
            set_requires_grad_loc_(model=self, requires_grad=requires_grad)
        elif property == "scale":
            set_requires_grad_scale_(model=self, requires_grad=requires_grad)
        else:
            ValueError("Input `property` must be `loc` or `scale`.")

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
