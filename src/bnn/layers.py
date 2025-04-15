"""Collections of Bayesian neural network layers."""

from typing import Any, Union

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.parameter import Parameter


class BayesianLayer(torch.nn.Module):
    """Base class for Bayesian layers."""

    def __init__(
        self,
        deterministic: bool = True,
    ):
        """BayesianLayer initializer.

        Args:
            deterministic: Flag indicating forward pass should compute the mean
                and variance determinstically (i.e., from parameter means and
                variances without random sampling).
        """
        super().__init__()
        self.deterministic = deterministic

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
            kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        # Define layer parameters.
        factory_kwargs = {"device": device, "dtype": dtype}
        self._weight_params = Parameter(
            torch.empty((2, out_features, in_features), **factory_kwargs)
        )  # (mean, variance)
        if bias:
            self._bias_params = Parameter(
                torch.empty((2, out_features), **factory_kwargs)
            )  # (mean, variance)
        else:
            self._bias_params = self.register_parameter("_bias_params", None)

        # Initialize parameters.
        if samplers_init is None:
            samplers_init = {
                "weight": (
                    dist.Normal(
                        loc=torch.zeros_like(self._weight_params[0]),
                        scale=1.0 / in_features,
                    ),
                    dist.Normal(
                        loc=-5.0 * torch.ones_like(self._weight_params[1]),
                        scale=1.0 / in_features,
                    ),
                ),
                "bias": (
                    dist.Normal(
                        loc=torch.zeros_like(self._bias_params[0]),
                        scale=1.0 / out_features,
                    ),
                    dist.Normal(
                        loc=-5.0 * torch.ones_like(self._bias_params[1]),
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
            params = getattr(self, f"_{key}_params")
            samplers[key] = value(loc=params[0], scale=torch.sqrt(var_tform(params[1])))
        self.samplers = samplers

    @property
    def weight(self):
        """Return a sample from the `weight` sampler distribution."""
        return self.samplers["weight"].sample()

    @property
    def bias(self):
        """Return a sample from the `bias` sampler distribution."""
        if self._bias_params is None:
            return None
        else:
            return self.samplers["bias"].sample()

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
        """Resample layer parameters."""
        self._weight_params.data = torch.stack(
            [sampler.sample() for sampler in self.samplers_init["weight"]]
        )
        if self._bias_params is not None:
            self._bias_params.data = torch.stack(
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

        # Pass input through layer.
        if self.deterministic:
            # Compute analytical result under mean-field approximation following
            # https://doi.org/10.48550/arXiv.2402.14532
            mu = torch.nn.functional.linear(
                input=input_mu, weight=self._weight_params[0], bias=self._bias_params[0]
            )
            var = torch.nn.functional.linear(
                input=input_mu**2,
                weight=self.var_tform(self._weight_params[1]),
                bias=(
                    None
                    if self._bias_params is None
                    else self.var_tform(self._bias_params[1])
                ),
            )

            # Add input variance contribution.
            if input_var is not None:
                var += torch.nn.functional.linear(
                    input=input_var,
                    weight=self._weight_params[0] ** 2
                    + self.var_tform(self._weight_params[1]),
                    bias=None,
                )
        else:
            # Compute forward pass with a random sample of parameters.
            mu = torch.nn.functional.linear(
                input=input_mu, weight=self.weight, bias=self.bias
            )
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
    linear = torch.nn.Linear(in_features=3, out_features=1)
    bayesian_linear = Linear(in_features=3, out_features=1)
    bl_test = Linear.init_from(layer=linear)
    batch_size = 1
    n_samples = 10000
    input = torch.randn(batch_size, 3)
    output_samples = torch.stack(
        [bayesian_linear(input, deterministic=False)[0] for _ in range(n_samples)]
    )
    output_mc = (output_samples.mean(dim=0), output_samples.var(dim=0))
    output_det = bayesian_linear(input, deterministic=True)

    # Overlay deterministic predictions on Monte Carlo samples.
    plt.hist(output_samples.detach().cpu().squeeze(), density=True)
    x = torch.linspace(-5.0, 5.0, 1000)
    plt.plot(
        x,
        (
            torch.exp(-0.5 * (x - output_det[0]) ** 2 / output_det[1])
            / torch.sqrt(2.0 * np.pi * output_det[1])
        )
        .detach()
        .cpu()
        .squeeze(),
    )
    plt.show()
