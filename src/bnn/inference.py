"""Inference modules for Bayesian neural network layers.

Custom propagators for specific layers (e.g., "Linear" for Bayesian analog of
torch.nn.Linear) should share a name with the layer such that
getattr(inference, layer.__class__.__name__) will return the custom propagator
for that layer if available.
"""

from collections.abc import Callable, Iterable
import functools
from typing import List, Optional, Union

import torch
import torch.distributions as dist


def select_default_sigmas(
    mu: torch.Tensor, var: torch.Tensor, kappa: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sigma points for use in the unscented transform as in [1].

    [1] https://doi.org/10.1117/12.280797

    Args:
        mu: Mean values that will be propagated through the function of interest.
        var: Variances that will be propagated through the function of interest.
        kappa: Spread parameter for sigma point selection.
    """
    # Compute the sigma points.
    scaled_stdev = ((kappa + 1) * var).sqrt()
    sigma_points = torch.stack((mu, mu - scaled_stdev, mu + scaled_stdev))

    # Compute the weights.
    weights = torch.empty(3, device=mu.device, dtype=mu.dtype)
    denom = kappa + 1
    weights[0] = kappa / denom
    weights[1:] = 0.5 / denom

    return sigma_points, weights


class MomentPropagator(torch.nn.Module):
    """Base class for layer propagators."""

    def __init__(self):
        """Initializer for MomentPropagator module."""
        super().__init__()

    def forward(
        self,
        module: Callable,
        input: Iterable,
    ) -> Union[Iterable, tuple]:
        """Forward method for MomentPropagator modules.

        Args:
            module: Instance of the layer through which we will propagate moments.
            input: Input passed to layer, which will typically be a torch.Tensor or
                types.MuVar.
        """
        raise NotImplementedError


class BasicPropagator(MomentPropagator):
    """Propagate mean and variance through `module`.

    This propagator can be used with modules that have "simple" forward passes
    for which propagation rules are already defined by methods in types.MuVar
    (e.g., forward pass is just input*param1 + param2).
    """

    def __init__(self):
        """Initializer for BasicPropagator inference module."""
        super().__init__()

    def forward(
        self,
        module: Callable,
        input: Iterable,
    ) -> Union[Iterable, tuple]:
        """Propagate moments by relying intrinsically on methods in types.MuVar."""
        return module.module(input)


class UnscentedTransform(MomentPropagator):
    """Unscented transform propagation of mean and variance through `module`.

    This propagator uses the unscented transform to propagate mean and variance
    through a deterministic layer.
    """

    def __init__(
        self, sigma_selector: Optional[Callable] = None, n_module_samples: int = 1
    ):
        """Initializer for UnscentedTransform inference module.

        Args:
            sigma_selector: Callable that accepts input mean and variance and
                returns the corresponding sigma points and weights used in the
                unscented transform.
            n_module_samples: Number of samples to make of the module itself.
                This allows us to use the unscented transform through modules
                which themselves are parametrized by random variables.
        """
        super().__init__()

        if sigma_selector is None:
            sigma_selector = functools.partial(select_default_sigmas, kappa=2)
        self.sigma_selector = sigma_selector

        self.n_module_samples = n_module_samples

    def forward(
        self,
        module: Callable,
        input: Iterable,
        return_samples: bool = False,
    ) -> Union[Iterable, tuple]:
        """Propagate moments using the unscented transform."""
        # Select sigma points and reshape along batch dimension for batched eval.
        sigma_points, weights = self.sigma_selector(mu=input[0], var=input[1])
        sp_shape = sigma_points.shape
        sigma_points = sigma_points.reshape(sp_shape[0] * sp_shape[1], *sp_shape[2:])

        # Propagate mean and variance.
        if self.n_module_samples > 1:
            # Perform n_module_samples unscented transforms and combine results.
            mu_samples = []
            var_samples = []
            for n in range(self.n_module_samples):
                # Prepare a sampled instance of the module.  For stochastic modules,
                # module.module returns a new sample of weights each time, so
                # we need to prepare the instance before running .forward() on each
                # sigma point.
                if hasattr(module, "module"):
                    module_sample = module.module
                else:
                    module_sample = module

                # Forward pass through module and use unscented transform.
                samples = module_sample(sigma_points)
                samples = samples.reshape(sp_shape[0], sp_shape[1], *samples.shape[1:])
                if n == 0:
                    weights = weights.reshape(
                        (weights.shape[0],) + (1,) * (samples.ndim - 1)
                    )
                mu_samples.append((weights * samples).sum(dim=0))
                var_samples.append(
                    (weights * ((samples - mu_samples[-1]) ** 2)).sum(dim=0)
                )

            # Combine estimates from each unscented transform using law of total
            # expectation and law of total variance.
            mu = torch.stack(mu_samples).mean(dim=0)
            var = torch.stack(var_samples).mean(dim=0) + torch.stack(mu_samples).var(
                dim=0
            )
        else:
            # Prepare a sampled instance of the module.  For stochastic modules,
            # module.module returns a new sample of weights each time, so
            # we need to prepare the instance before running .forward() on each
            # sigma point.
            if hasattr(module, "module"):
                module_sample = module.module
            else:
                module_sample = module
            samples = module_sample(sigma_points)

            # Compute output mean and variance.
            samples = samples.reshape(sp_shape[0], sp_shape[1], *samples.shape[1:])
            weights = weights.reshape((weights.shape[0],) + (1,) * (samples.ndim - 1))
            mu = (weights * samples).sum(dim=0)
            var = (weights * ((samples - mu) ** 2)).sum(dim=0)

        if return_samples:
            return type(input)([mu, var]), samples
        else:
            return type(input)([mu, var])


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
        module: Callable,
        input: Iterable,
        return_samples: bool = False,
    ) -> Union[Iterable, tuple]:
        """Propagate moments by averaging over n_samples forward passes of module."""
        # If the input variance is greater than zero, we'll need to sample the
        # input as well.
        if (input[1] > 0.0).all():
            input_dist = self.input_sampler(loc=input[0], scale=input[1].sqrt())
            samples = torch.stack(
                [module(input_dist.sample()) for _ in range(self.n_samples)]
            )
        else:
            samples = torch.stack([module(input[0]) for _ in range(self.n_samples)])

        if return_samples:
            return type(input)([samples.mean(dim=0), samples.var(dim=0)]), samples
        else:
            return type(input)([samples.mean(dim=0), samples.var(dim=0)])


class Linear(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a Linear layer."""

    functional = torch.nn.functional.linear

    def __init__(self):
        """Initializer for Linear inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input: Iterable,
    ) -> Iterable:
        """Analytical moment propagation through layer."""
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        module_params = module._module_params
        bias_mean = module_params.get("bias_mean", None)
        bias_rho = module_params.get("bias_rho", None)
        weight_mean = module_params["weight_mean"]
        weight_rho = module_params["weight_rho"]

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=weight_mean,
            bias=bias_mean,
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if weight_rho is None:
            # No "first term" since there is no parameter variance.
            var = self.functional(
                input=input[1],
                weight=weight_mean**2,
            )
        else:
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input[0] ** 2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
            ) + self.functional(
                input=input[1],
                weight=weight_mean**2 + weight_var,
            )

        return type(input)([mu, var])


class ConvNd(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a ConvNd layer."""

    def __init__(self):
        """Initializer for ConvNd inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input: Iterable,
    ) -> Iterable:
        """Analytical moment propagation through layer."""
        # Modify input and prepare functional arguments.
        if module._module.padding_mode != "zeros":
            # This is added to mimic torch.nn.modules.conv _conv_forward methods.
            input = torch.nn.functional.pad(
                input,
                module._module._reversed_padding_repeated_twice,
                mode=module._module.padding_mode,
            )
            functional_kwargs = {
                "stride": module._module.stride,
                "padding": torch.nn.modules.utils._triple(0),
                "dilation": module._module.dilation,
                "groups": module._module.groups,
            }
        else:
            functional_kwargs = {
                "stride": module._module.stride,
                "padding": module._module.padding,
                "dilation": module._module.dilation,
                "groups": module._module.groups,
            }

        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        module_params = module._module_params
        bias_mean = module_params.get("bias_mean", None)
        bias_rho = module_params.get("bias_rho", None)
        weight_mean = module_params["weight_mean"]
        weight_rho = module_params["weight_rho"]

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=weight_mean,
            bias=bias_mean,
            **functional_kwargs,
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if weight_rho is None:
            # No "first term" since there is no parameter variance.
            var = self.functional(
                input=input[1],
                weight=weight_mean**2,
                bias=None,
                **functional_kwargs,
            )
        else:
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input[0] ** 2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
                **functional_kwargs,
            ) + self.functional(
                input=input[1],
                weight=weight_mean**2 + weight_var,
                bias=None,
                **functional_kwargs,
            )

        return type(input)([mu, var])


class Conv1d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv1d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    functional = torch.nn.functional.conv1d

    def __init__(self):
        """Initializer for Conv1d inference module"""
        super().__init__()


class Conv2d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv2d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    functional = torch.nn.functional.conv2d

    def __init__(self):
        """Initializer for Conv2d inference module"""
        super().__init__()


class Conv3d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv3d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    functional = torch.nn.functional.conv3d

    def __init__(self):
        """Initializer for Conv3d inference module"""
        super().__init__()


class ConvTransposeNd(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a ConvTransposeNd layer."""

    def __init__(self):
        """Initializer for ConvTransposeNd inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input: Iterable,
        output_size: Optional[List[int]] = None,
    ) -> Iterable:
        """Analytical moment propagation through layer."""
        # Prepare functional arguments.
        output_padding = module._module._output_padding(
            input,
            output_size,
            module._module.stride,
            module._module.padding,
            module._module.kernel_size,
            self.num_spatial_dims,
            module._module.dilation,
        )
        functional_kwargs = {
            "stride": module._module.stride,
            "padding": module._module.padding,
            "dilation": module._module.dilation,
            "groups": module._module.groups,
            "output_padding": output_padding,
        }

        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        module_params = module._module_params
        bias_mean = module_params.get("bias_mean", None)
        bias_rho = module_params.get("bias_rho", None)
        weight_mean = module_params["weight_mean"]
        weight_rho = module_params["weight_rho"]

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=weight_mean,
            bias=bias_mean,
            **functional_kwargs,
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if weight_rho is None:
            # No "first term" since there is no parameter variance.
            var = self.functional(
                input=input[1],
                weight=weight_mean**2,
                bias=None,
                **functional_kwargs,
            )
        else:
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input[0] ** 2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
                **functional_kwargs,
            ) + self.functional(
                input=input[1],
                weight=weight_mean**2 + weight_var,
                bias=None,
                **functional_kwargs,
            )

        return type(input)([mu, var])


class ConvTranspose1d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose1d layer."""

    functional = torch.nn.functional.conv_transpose1d

    def __init__(self):
        """Initializer for ConvTranspose1d inference module"""
        super().__init__()
        self.num_spatial_dims = 1


class ConvTranspose2d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose2d layer."""

    functional = torch.nn.functional.conv_transpose2d

    def __init__(self):
        """Initializer for ConvTranspose2d inference module"""
        super().__init__()
        self.num_spatial_dims = 2


class ConvTranspose3d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose3d layer."""

    functional = torch.nn.functional.conv_transpose3d

    def __init__(self):
        """Initializer for ConvTranspose3d inference module"""
        super().__init__()
        self.num_spatial_dims = 3


if __name__ == "__main__":
    """Example usages of inference modules."""
    import matplotlib.pyplot as plt

    # Define a nonlinearity to propagate through.
    layer = torch.nn.LeakyReLU()

    # Define some propagators.
    mc = MonteCarlo(n_samples=100)
    ut = UnscentedTransform()

    # Propagate example data through layer.
    input = (torch.tensor([1.23])[None, :], torch.tensor([3.21])[None, :])
    out_mc, samples_mc = mc(module=layer, input=input, return_samples=True)
    out_ut, samples_ut = ut(module=layer, input=input, return_samples=True)

    # Plot results.
    x = torch.linspace(
        (input[0] - 3.0 * torch.sqrt(input[1])).squeeze(),
        (input[0] + 3.0 * torch.sqrt(input[1])).squeeze(),
        1000,
    )
    fig, ax = plt.subplots()
    ax.hist(samples_mc.squeeze(), density=True, label="Monte Carlo samples")
    ax.plot(
        x,
        torch.distributions.Normal(
            loc=out_mc[0].squeeze(), scale=out_mc[1].squeeze().sqrt()
        )
        .log_prob(x)
        .exp(),
        label="Monte Carlo estimated PDF",
    )
    ax.plot(
        x,
        torch.distributions.Normal(
            loc=out_ut[0].squeeze(), scale=out_ut[1].squeeze().sqrt()
        )
        .log_prob(x)
        .exp(),
        label="Unscented transform estimated PDF",
    )
    plt.legend()
    plt.show()
