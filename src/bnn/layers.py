"""Collections of Bayesian neural network layers."""

from typing import Union

import numpy as np
import torch
import torch.distributions as dist

from bnn.inference import MomentPropagator, MonteCarlo, Linear
from utils.misc import get_torch_functional


class BayesianLayer(torch.nn.Module):
    """Bayesian implementation of a generic PyTorch module."""

    def __init__(
        self,
        module: torch.nn.Module,
        samplers: dict = None,
        samplers_init: dict = None,
        moment_propagator: MomentPropagator = MonteCarlo(n_samples=10),
        propagate_moments: bool = True,
    ):
        """BayesianLayer initializer.

        Args:
            module: Module with a functional version present in torch.nn.functional.
                For such modules, we'll search for learnable parameters, create
                their Bayesian counterparts under the mean-field approximation for
                two parameter distributions (i.e., param -> (param_mean, param_var))
            samplers: Dictionary of distributions that are sampled to return
                parameters of this layer (i.e., the variational distributions).
                These should be uninitialized or partially initialized
                distributions that accept inputs `loc` and `scale` and return
                an object with a .sample() method which returns tensors
                corresponding to sampled parameters.  For example,
                samplers["weight"](loc=0.0, scale=1.0).sample()
                must return a tensor of size (out_features, in_features) if
                module = torch.nn.Linear(in_features, out_features)
            samplers_init: Dictionary of samplers that can be sampled to reset
                parameters of this layer.  Each value is a tuple corresponding
                to the mean and (unscaled, see `var_tform` usage) variance
                parameters (e.g., samplers_init["weight"][0].sample() should
                return a shape (out_features, in_features) tensor defining
                initial weight parameters).
            moment_propagator: Propagation module to propagate mean and variance
                through this layer.
            propagate_moments: Flag indicating forward pass should use
                `moment_propagator` to compute the output mean and variance from
                this layer.
        """
        super().__init__()

        # Validate `module` compatability with this class and find associated
        # functional from torch.nn.functional.
        functional = get_torch_functional(module.__class__)
        assert (
            functional is not None
        ), "`module` must have a functional version in torch.nn.functional to use this class!"
        self.functional = functional

        # Set moment propagator.
        self.propagate_moments = propagate_moments
        self.moment_propagator = moment_propagator

        # Parametrize the distributions of the Bayesian counterparts of each
        # named parameter.
        _module_params = torch.nn.ParameterDict()
        for param in module.named_parameters():
            if param[1] is None:
                _module_params[param[0]] = torch.nn.Parameter(None)
            else:
                _module_params[param[0]] = torch.nn.Parameter(
                    torch.empty(
                        (2, *param[1].shape),
                        device=param[1].device,
                        dtype=param[1].dtype,
                    )  # (mean, unscaled variance)
                )
        self._module_params = _module_params

        # Create samplers for each named parameter.
        if samplers_init is None:
            # Prepare (relatively arbitrary) default samplers.
            samplers_init = {
                key: (
                    dist.Normal(
                        loc=torch.zeros_like(_module_params[key][0]),
                        scale=1.0 / _module_params[key][0].numel(),
                    ),
                    dist.Normal(
                        loc=-5.0 * torch.ones_like(_module_params[key][1]),
                        scale=1.0 / _module_params[key][0].numel(),
                    ),
                )
                for key in _module_params.keys()
            }
        self.samplers_init = samplers_init
        self.reset_parameters()

        # Initialize variational distribution samplers.
        var_tform = torch.nn.functional.softplus
        self.var_tform = var_tform
        if samplers is None:
            samplers = {key: dist.Normal for key in _module_params.keys()}
        for key, value in samplers.items():
            samplers[key] = value(
                loc=_module_params[key][0],
                scale=torch.sqrt(var_tform(_module_params[key][1])),
            )
        self.samplers = samplers

    def reset_parameters(self) -> None:
        """Resample layer parameters from initial distributions."""
        for key, value in self._module_params.items():
            if value is not None:
                value.data = torch.stack(
                    [sampler.sample() for sampler in self.samplers_init[key]]
                )

    @property
    def module_params(self):
        """Return a sample of this module's parameters."""
        return {
            key: None if self._module_params[key] is None else value.sample()
            for key, value in self.samplers.items()
        }

    def forward(
        self,
        input_mu: Union[tuple, torch.tensor],
        input_var: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """Forward pass through layer."""
        # Repackage `input_mu` if passed as a tuple (this allows the input mean
        # and variance to be passed as forward(x) where x = (mu, var)).
        if isinstance(input_mu, tuple):
            input_mu, input_var = input_mu

        # Propagate mean and variance through layer.
        if self.propagate_moments:
            mu, var = self.moment_propagator(
                module=self, input_mu=input_mu, input_var=input_var
            )
        else:
            # Compute forward pass with a random sample of parameters.
            mu = self.functional(input=input_mu, **self.module_params)
            var = None

        return mu, var


class ForwardPassMean(torch.nn.Module):
    """General layer wrapper that passes mean as input and ignores variance."""

    def __init__(self, module: torch.nn.Module):
        """Initializer for ForwardPassMean wrapper.

        This module is designed to wrap modules whose forward call accepts a
        tensor and returns a tensor.  The intention is that we can use the
        wrapped module in a BNN whose forward pass through each layer instead
        accepts and returns two tensors corresponding to mean and variance.
        For example, if out = layer(input), then
        layer_w = ForwardPassMean(module=module) can be called as out_w = layer_w(input, Any)
        with out_w[0] == out

        Args:
            layer: Layer to be wrapped to accomodate two input forward pass.
        """
        super().__init__()
        self.layer = module

    def forward(
        self, input_mu: Union[tuple, torch.tensor], input_var: torch.tensor = None
    ) -> tuple[torch.tensor, torch.tensor]:
        """Forward pass through the wrapped layer."""
        # Repackage `input_mu` if passed as a tuple (this allows calling as either
        # forward(input_mu, input_var) or forward((input_mu, input_var)))
        if isinstance(input_mu, tuple):
            input_mu, input_var = input_mu

        return self.layer(input_mu), input_var


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Basic usage example of BayesianLinear.
    in_features = 3
    out_features = 1
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    bayesian_linear = BayesianLayer(
        module=linear,
        propagate_moments=False,
    )
    n_samples = 10000
    bayesian_linear_mc = BayesianLayer(
        module=linear,
        propagate_moments=True,
        moment_propagator=MonteCarlo(n_samples=n_samples),
    )  # computes moments from Monte Carlo without returning actual samples
    bayesian_linear_mc.load_state_dict(bayesian_linear.state_dict())
    bayesian_linear_det = BayesianLayer(
        module=linear,
        propagate_moments=True,
        moment_propagator=Linear(),
    )  # analytically computes moments
    bayesian_linear_det.load_state_dict(bayesian_linear.state_dict())
    batch_size = 1
    input = torch.randn(batch_size, in_features)
    output_samples = torch.stack([bayesian_linear(input)[0] for _ in range(n_samples)])
    output_mc = bayesian_linear_mc(input)
    output_det = bayesian_linear_det(input)

    # Overlay deterministic predictions on Monte Carlo samples.
    plt.hist(output_samples.detach().cpu().squeeze(), density=True)
    x = torch.linspace(output_samples.min(), output_samples.max(), 1000)
    plt.plot(
        x,
        (
            torch.exp(-0.5 * (x - output_det[0]) ** 2 / output_det[1])
            / torch.sqrt(2.0 * np.pi * output_det[1])
        )
        .detach()
        .cpu()
        .squeeze(),
        label="Analytical",
    )
    plt.plot(
        x,
        (
            torch.exp(-0.5 * (x - output_mc[0]) ** 2 / output_mc[1])
            / torch.sqrt(2.0 * np.pi * output_mc[1])
        )
        .detach()
        .cpu()
        .squeeze(),
        label="Monte Carlo",
    )
    plt.legend()
    plt.show()
