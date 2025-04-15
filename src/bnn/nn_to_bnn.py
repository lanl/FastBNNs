"""Functionality for converting neural networks to Bayesian neural networks."""

import functools
import re
from typing import Any

import torch

import bnn.layers
from bnn.layers import BayesianLayer, ForwardPassMean
import bnn.inference
from utils.misc import get_torch_functional


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
        module = model.get_submodule(leaf)
        module_name = module.__class__.__name__

        # Use default Bayesian layer constructor (which works for layers with
        # functionals in torch.nn.functional), otherwise fallback to a mean
        # passthrough layer.
        functional = get_torch_functional(module.__class__)
        if functional is not None:
            # Search for a good default moment propagator.
            kwarg_overrides = {}
            if "moment_propagator" not in bayesian_layer_kwargs.keys():
                if hasattr(bnn.inference, module_name):
                    propagator = getattr(bnn.inference, module_name)
                    kwarg_overrides["moment_propagator"] = propagator()

            # Create Bayesian version of this layer.
            bayesian_layer = BayesianLayer(
                module=module, **(bayesian_layer_kwargs | kwarg_overrides)
            )
            model.set_submodule(leaf, bayesian_layer)
        elif hasattr(bnn.layers, module_name):
            # If a custom layer exists for this named layer, we'll use that by default.
            custom_layer = getattr(bnn.layers, module_name)(
                module=module, **bayesian_layer_kwargs
            )
            model.set_submodule(leaf, custom_layer)
        else:
            # Use generic mean passthrough layer for compatibility with Bayesian layers.
            passthrough_layer = ForwardPassMean(module=module)
            model.set_submodule(leaf, passthrough_layer)


if __name__ == "__main__":
    from models import mlp

    # Convert a model to a Bayesian counterpart.
    in_features = 3
    out_features = 1
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        n_layers=3,
        activation=torch.nn.LeakyReLU,
    )
    convert_to_bnn_(model=model)
    out = model(torch.ones(1, in_features))
