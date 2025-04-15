"""Collections of Bayesian neural network layers."""

import copy
from typing import Any, Union

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.parameter import Parameter

from bnn.inference import MomentPropagator, MonteCarlo, PropagateLinear


class BayesianLayer(torch.nn.Module):
    """Base class for Bayesian layers."""

    def __init__(self, propagate_moments: bool = True):
        """BayesianLayer initializer.

        Args:
            propagate_moments: Flag indicating forward pass should compute the mean
                and variance determinstically (i.e., from parameter means and
                variances without random sampling).
        """
        super().__init__()
        self.propagate_moments = propagate_moments

    @staticmethod
    def init_from(layer: torch.nn.Module, **kwargs) -> Any:
        pass

    def forward(
        self, input_mu: Union[torch.tensor, tuple], input_var: torch.tensor = None
    ) -> tuple[torch.tensor, torch.tensor]:
        """Forward pass for Bayesian layers.

        Args:
            input_mu: Mean of input, or optionally, a tuple defining the mean
                and variance of the input.
            input_var: Variance of input (ignored if `input_mu` is passed as a tuple).
        """
        pass


class Linear(BayesianLayer):
    """Bayesian parametrization of torch.nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        samplers: dict = None,
        samplers_init: dict = None,
        moment_propagator: MomentPropagator = PropagateLinear(),
        **kwargs,
    ):
        """Initializer for Bayesian Linear class.

        Args:
            in_features: Number of input features to layer.
            out_features: Number of output features from layer.
            bias: Flag indicating whether or not to learn a bias term.
            device: Device on which to initialize parameters.
            dtype: Datatype of learnable parameters.
            samplers: Dictionary of distributions that are sampled to return
                weights of this layer (i.e., the variational distributions).
                These should be uninitialized or partially initialized
                distributions that accept inputs `loc` and `scale` and return
                an object with a .sample() method which returns tensors
                corresponding to sampled parameters.  For example,
                samplers["weight"](loc=0.0, scale=1.0).sample()
                must return a tensor of size (out_features, in_features)
            samplers_init: Dictionary of samplers that can be sampled to reset
                parameters of this layer.  Each value is a tuple corresponding
                to the mean and (unscaled, see `var_tform` usage) variance
                parameters (e.g., samplers_init["weight"][0].sample() should
                return a shape (out_features, in_features) tensor defining
                initial weight parameters).
            moment_propagator: Propagation module to propagate mean and variance
                through this layer.
            kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        # Set moment propagator.
        self.moment_propagator = moment_propagator

        # Define layer parameters (dimension 2 corresponds to mean and variance).
        factory_kwargs = {"device": device, "dtype": dtype}
        layer_params = torch.nn.ParameterDict(
            {
                "weight": Parameter(
                    torch.empty((2, out_features, in_features), **factory_kwargs)
                ),
                "bias": Parameter(None),
            }
        )
        if bias:
            layer_params["bias"] = Parameter(
                torch.empty((2, out_features), **factory_kwargs)
            )
        self._layer_params = layer_params

        # Initialize parameters.
        if samplers_init is None:
            samplers_init = {
                "weight": (
                    dist.Normal(
                        loc=torch.zeros_like(layer_params["weight"][0]),
                        scale=1.0 / in_features,
                    ),
                    dist.Normal(
                        loc=-5.0 * torch.ones_like(layer_params["weight"][1]),
                        scale=1.0 / in_features,
                    ),
                ),
                "bias": (
                    dist.Normal(
                        loc=torch.zeros_like(layer_params["bias"][0]),
                        scale=1.0 / out_features,
                    ),
                    dist.Normal(
                        loc=-5.0 * torch.ones_like(layer_params["bias"][1]),
                        scale=1.0 / out_features,
                    ),
                ),
            }
        self.samplers_init = samplers_init
        self.reset_parameters()

        # Initialize samplers (variational distribution samplers).
        var_tform = torch.nn.functional.softplus
        self.var_tform = var_tform
        if samplers is None:
            samplers = {"weight": dist.Normal, "bias": dist.Normal}
        for key, value in samplers.items():
            samplers[key] = value(
                loc=layer_params[key][0],
                scale=torch.sqrt(var_tform(layer_params[key][1])),
            )
        self.samplers = samplers

    @property
    def layer_params(self):
        """Return a sample of this layer's parameters."""
        return {
            key: None if self._layer_params[key] is None else value.sample()
            for key, value in self.samplers.items()
        }

    @staticmethod
    def init_from(layer: torch.nn.Module, **kwargs) -> Any:
        return Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        """Resample layer parameters from initial distributions."""
        self._layer_params["weight"].data = torch.stack(
            [sampler.sample() for sampler in self.samplers_init["weight"]]
        )
        if self._layer_params["bias"] is not None:
            self._layer_params["bias"].data = torch.stack(
                [sampler.sample() for sampler in self.samplers_init["bias"]]
            )

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
            mu = torch.nn.functional.linear(input=input_mu, **self.layer_params)
            var = None

        return mu, var


class ForwardPassMean(torch.nn.Module):
    """General layer wrapper that passes mean as input and ignores variance."""

    def __init__(self, layer: torch.nn.Module):
        """Initializer for ForwardPassMean wrapper.

        This module is designed to wrap modules whose forward call accepts a
        tensor and returns a tensor.  The intention is that we can use the
        wrapped module in a BNN whose forward pass through each layer instead
        accepts and returns two tensors corresponding to mean and variance.
        For example, if out = layer(input), then
        layer_w = ForwardPassMean(layer=layer) can be called as out_w = layer_w(input, Any)
        with out_w[0] == out

        Args:
            layer: Layer to be wrapped to accomodate two input forward pass.
        """
        super().__init__()
        self.layer = layer

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
    bayesian_linear = Linear(in_features=3, out_features=1, propagate_moments=False)
    n_samples = 10000
    bayesian_linear_mc = Linear(
        in_features=3,
        out_features=1,
        propagate_moments=True,
        moment_propagator=MonteCarlo(n_samples=n_samples),
    )  # computes moments from Monte Carlo without returning actual samples
    bayesian_linear_mc.load_state_dict(bayesian_linear.state_dict())
    bayesian_linear_det = Linear(
        in_features=3,
        out_features=1,
        propagate_moments=True,
        moment_propagator=PropagateLinear(),
    )  # analytically computes moments
    bayesian_linear_det.load_state_dict(bayesian_linear.state_dict())
    batch_size = 1
    input = torch.randn(batch_size, 3)
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
