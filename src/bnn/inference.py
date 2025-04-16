"""Inference modules for Bayesian neural network layers.

Custom propagators for specific layers (e.g., "Linear" for Bayesian analog of
torch.nn.Linear) should share a name with the layer such that
getattr(inference, layer.__class__.__name__) will return the custom propagator
for that layer if available.
"""

from typing import Union

import torch
import torch.distributions as dist


class MomentPropagator(torch.nn.Module):
    """Base class for layer propagators."""

    def __init__(self):
        """Initializer for MomentPropagator module."""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input_mu: Union[tuple, torch.tensor],
        input_var: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """Forward method for MomentPropagator modules.

        Args:
            module: Instance of the layer through which we will propagate moments.
            input_mu: Mean of input, or optionally, tuple organizing mean and variance
                of the input (in which case `input_var` is ignored).
            input_var: Variance of input (ignored if `input_mu` is a tuple).
        """
        pass


class MonteCarlo(MomentPropagator):
    """Monte Carlo propagation of mean and variance through `module`.

    This propagator is designed to perform Monte Carlo sampling of BOTH
    the input and the layer being propagated through.  This would generally
    only be used for Bayesian/stochastic layers through which we cannot
    propagate moments through analytically.
    """

    def __init__(
        self, n_samples: int = 10, input_sampler: dist.Distribution = dist.Normal
    ):
        """Initializer for MonteCarlo inference module.

        Args:
            n_samples: Number of Monte Carlo samples used to estimate mean
                and variance after passing through this layer.
            input_sampler: Distribution template used to sample the input to
                a layer in the forward pass as
                input_sample = input_sampler(loc=input_mu, scale=input_var.sqrt()).sample()

        """
        super().__init__()
        self.n_samples = n_samples
        self.input_sampler = input_sampler

    def forward(
        self,
        module: torch.nn.Module,
        input_mu: Union[tuple, torch.tensor],
        input_var: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """Propagate moments by averaging over n_samples forward passes of module."""
        # Repackage `input_mu` if passed as a tuple (this allows the input mean
        # and variance to be passed as forward(x) where x = (mu, var)).
        if isinstance(input_mu, tuple):
            input_mu, input_var = input_mu

        # Temporarily turn off moment propagation in module to avoid recursion.
        if hasattr(module, "propagate_moments"):
            propagate_moments_init = module.propagate_moments
            module.propagate_moments = False

        # Compute forward passes.
        if input_var is None:
            samples = torch.stack([module(input_mu)[0] for _ in range(self.n_samples)])
        else:
            # If input_var is provided, we also need to sample the input distribution.
            input_dist = self.input_sampler(loc=input_mu, scale=input_var.sqrt())
            samples = torch.stack(
                [module(input_dist.sample())[0] for _ in range(self.n_samples)]
            )

        # Restore initial setting of `propagate_moments` (this may always be
        # be True but using initial state to account for custom use cases).
        if hasattr(module, "propagate_moments"):
            module.propagate_moments = propagate_moments_init

        return samples.mean(dim=0), samples.var(dim=0)


class Linear(MomentPropagator):
    """Deterministic moment propagation of mean, variance through a Linear layer."""

    def __init__(self):
        """Initializer for PropagateLinear inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input_mu: Union[tuple, torch.tensor],
        input_var: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """Analytical moment propagation through layer."""
        # Repackage `input_mu` if passed as a tuple (this allows the input mean
        # and variance to be passed as forward(x) where x = (mu, var)).
        if isinstance(input_mu, tuple):
            input_mu, input_var = input_mu

        # Compute analytical result under mean-field approximation following
        # https://doi.org/10.48550/arXiv.2402.14532
        if "bias" in module._module_params.keys():
            bias_params = module._module_params["bias"]
        else:
            bias_params = (None, None)
        mu = torch.nn.functional.linear(
            input=input_mu,
            weight=module._module_params["weight"][0],
            bias=bias_params[0],
        )
        var = torch.nn.functional.linear(
            input=input_mu**2,
            weight=module.scale_tform(module._module_params["weight"][1]) ** 2,
            bias=(
                None
                if bias_params[1] is None
                else module.scale_tform(bias_params[1]) ** 2
            ),
        )

        # Add input variance contribution.
        if input_var is not None:
            var += torch.nn.functional.linear(
                input=input_var,
                weight=module._module_params["weight"][0] ** 2
                + module.scale_tform(module._module_params["weight"][1]) ** 2,
                bias=None,
            )

        return mu, var
