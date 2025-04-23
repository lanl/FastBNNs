"""Inference modules for Bayesian neural network layers.

Custom propagators for specific layers (e.g., "Linear" for Bayesian analog of
torch.nn.Linear) should share a name with the layer such that
getattr(inference, layer.__class__.__name__) will return the custom propagator
for that layer if available.
"""

from collections.abc import Callable
import functools
from typing import List, Optional, Union

import torch
import torch.distributions as dist

from bnn.types import MuVar


def select_default_sigmas(
    mu: torch.Tensor, var: torch.Tensor, n_sigma_points: int = 3, kappa: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sigma points for use in the unscented transform as in [1].

    [1] https://doi.org/10.1117/12.280797

    Args:
        mu: Mean values that will be propagated through the function of interest.
        var: Variances that will be propagated through the function of interest.
        n_sigma_points: Number of sigma points to return.  This must be an odd number.
        kappa: Spread parameter for sigma point selection.
    """

    # Compute sigma points.
    assert (n_sigma_points // 2) != 0, "Input `n_sigma_points` must be an odd number."
    sigma_points = torch.empty(
        (n_sigma_points, *mu.shape), device=mu.device, dtype=mu.dtype
    )
    sigma_points[0] = mu.detach()
    n = (n_sigma_points - 1) // 2
    n_vec = torch.arange(1, 1 + n, device=mu.device)
    sigma_points[1 : 1 + n] = mu + torch.sqrt(
        torch.einsum("ij,j...->i...", n_vec[:, None] + kappa, var[None, ...])
    )
    sigma_points[1 + n :] = mu - torch.sqrt(
        torch.einsum("ij,j...->i...", n_vec[:, None] + kappa, var[None, ...])
    )

    # Compute corresponding weights.
    weights = torch.empty(n_sigma_points, device=mu.device, dtype=mu.dtype)
    weights[0] = kappa / (n + kappa)
    weights[1 : 1 + n] = 0.5 / (n_vec + kappa)
    weights[1 + n :] = 0.5 / (n_vec + kappa)

    return sigma_points, weights


class MomentPropagator(torch.nn.Module):
    """Base class for layer propagators."""

    def __init__(self):
        """Initializer for MomentPropagator module."""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input_mu: Union[tuple, torch.Tensor],
        input_var: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for MomentPropagator modules.

        Args:
            module: Instance of the layer through which we will propagate moments.
            input_mu: Mean of input, or optionally, tuple organizing mean and variance
                of the input (in which case `input_var` is ignored).
            input_var: Variance of input (ignored if `input_mu` is a tuple).
        """
        raise NotImplementedError


class UnscentedTransform(MomentPropagator):
    """Unscented transform propagation of mean and variance through `module`.

    This propagator uses the unscented transform to propagate mean and variance
    through a deterministic layer.
    """

    def __init__(self, sigma_selector: Callable = None, n_module_samples: int = 1):
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
            sigma_selector = functools.partial(
                select_default_sigmas, n_sigma_points=3, kappa=2
            )
        self.sigma_selector = sigma_selector

        self.n_module_samples = n_module_samples

    def forward(
        self,
        module: torch.nn.Module,
        input: MuVar,
        return_samples: bool = False,
    ) -> Union[MuVar, tuple]:
        """Propagate moments using the unscented transform."""
        # Select sigma points.
        sigma_points, weights = self.sigma_selector(mu=input[0], var=input[1])

        # Propagate mean and variance.
        if self.n_module_samples > 1:
            # Perform n_module_samples unscented transforms and combine results.
            mu_samples = []
            var_samples = []
            for s in range(self.n_module_samples):
                samples = torch.stack([module(s) for s in sigma_points])
                mu_samples.append(torch.einsum("i,i...->...", weights, samples))
                var_samples.append(
                    torch.einsum(
                        "i,i...->...", weights, (samples - mu_samples[-1]) ** 2
                    )
                )

            # Combine estimates from each unscented transform
            mu = torch.stack(mu_samples).mean(dim=0)
            var = torch.stack(var_samples).mean(dim=0) + torch.stack(mu_samples).var(
                dim=0
            )
        else:
            samples = torch.stack([module(s) for s in sigma_points])
            mu = torch.einsum("i,i...->...", weights, samples)
            var = torch.einsum("i,i...->...", weights, (samples - mu) ** 2)

        if return_samples:
            return MuVar(mu, var), samples
        else:
            return MuVar(mu, var)


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
        input: MuVar,
        return_samples: bool = False,
    ) -> Union[MuVar, tuple]:
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
            return MuVar(samples.mean(dim=0), samples.var(dim=0)), samples
        else:
            return MuVar(samples.mean(dim=0), samples.var(dim=0))


class Linear(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a Linear layer."""

    def __init__(self):
        """Initializer for Linear inference module"""
        super().__init__()
        self.functional = torch.nn.functional.linear

    def forward(
        self,
        module: torch.nn.Module,
        input: MuVar,
    ) -> MuVar:
        """Analytical moment propagation through layer."""
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        if "bias_mean" in module._module_params.keys():
            bias_params = (
                module._module_params["bias_mean"],
                module._module_params["bias_rho"],
            )
        else:
            bias_params = (None, None)

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=module._module_params["weight_mean"],
            bias=bias_params[0],
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if module._module_params["weight_rho"] is None:
            weight_var = torch.zeros_like(module._module_params["weight_mean"])
        else:
            weight_var = module.scale_tform(module._module_params["weight_rho"]) ** 2
        var = self.functional(
            input=input[0] ** 2,
            weight=weight_var,
            bias=(
                None
                if bias_params[1] is None
                else module.scale_tform(bias_params[1]) ** 2
            ),
        ) + self.functional(
            input=input[1],
            weight=module._module_params["weight_mean"] ** 2 + weight_var,
            bias=None,
        )

        return MuVar(mu, var)


class ConvNd(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a ConvNd layer."""

    def __init__(self):
        """Initializer for ConvNd inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input: MuVar,
    ) -> MuVar:
        """Analytical moment propagation through layer."""
        # Modify input and prepare functional arguments.
        if module.mu.padding_mode != "zeros":
            # This is added to mimic torch.nn.modules.conv _conv_forward methods.
            input = torch.nn.functional.pad(
                input,
                module.mu._reversed_padding_repeated_twice,
                mode=module.mu.padding_mode,
            )
            functional_kwargs = {
                "stride": module.mu.stride,
                "padding": torch.nn.modules.utils._triple(0),
                "dilation": module.mu.dilation,
                "groups": module.mu.groups,
            }
        else:
            functional_kwargs = {
                "stride": module.mu.stride,
                "padding": module.mu.padding,
                "dilation": module.mu.dilation,
                "groups": module.mu.groups,
            }
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        if "bias_mean" in module._module_params.keys():
            bias_params = (
                module._module_params["bias_mean"],
                module._module_params["bias_rho"],
            )
        else:
            bias_params = (None, None)

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=module._module_params["weight_mean"],
            bias=bias_params[0],
            **functional_kwargs,
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if module._module_params["weight_rho"] is None:
            weight_var = torch.zeros_like(module._module_params["weight_mean"])
        else:
            weight_var = module.scale_tform(module._module_params["weight_rho"]) ** 2
        var = self.functional(
            input=input[0] ** 2,
            weight=weight_var,
            bias=(
                None
                if bias_params[1] is None
                else module.scale_tform(bias_params[1]) ** 2
            ),
            **functional_kwargs,
        ) + self.functional(
            input=input[1],
            weight=module._module_params["weight_mean"] ** 2 + weight_var,
            bias=None,
            **functional_kwargs,
        )

        return MuVar(mu, var)


class Conv1d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv1d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    def __init__(self):
        """Initializer for Conv1d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv1d


class Conv2d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv2d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    def __init__(self):
        """Initializer for Conv2d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv2d


class Conv3d(ConvNd):
    """Deterministic moment propagation of mean and variance through a Conv3d layer.

    The internal logic is identical to ConvNd so we just create this class for compatibility
    with module-name-based searches.
    """

    def __init__(self):
        """Initializer for Conv3d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv3d


class ConvTransposeNd(MomentPropagator):
    """Deterministic moment propagation of mean and variance through a ConvTransposeNd layer."""

    def __init__(self):
        """Initializer for ConvTransposeNd inference module"""
        super().__init__()

    def forward(
        self,
        module: torch.nn.Module,
        input: MuVar,
        output_size: Optional[List[int]] = None,
    ) -> MuVar:
        """Analytical moment propagation through layer."""
        # Prepare functional arguments.
        output_padding = module.mu._output_padding(
            input,
            output_size,
            module.mu.stride,
            module.mu.padding,
            module.mu.kernel_size,
            self.num_spatial_dims,
            module.mu.dilation,
        )
        functional_kwargs = {
            "stride": module.mu.stride,
            "padding": module.mu.padding,
            "dilation": module.mu.dilation,
            "groups": module.mu.groups,
            "output_padding": output_padding,
        }
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        # Reorganize parameters.
        if "bias_mean" in module._module_params.keys():
            bias_params = (
                module._module_params["bias_mean"],
                module._module_params["bias_rho"],
            )
        else:
            bias_params = (None, None)

        # Propagate mean.
        mu = self.functional(
            input=input[0],
            weight=module._module_params["weight_mean"],
            bias=bias_params[0],
            **functional_kwargs,
        )

        # Propagate variance: first term accounts for parameter variance, second
        # term propagates input variance.
        if module._module_params["weight_rho"] is None:
            weight_var = torch.zeros_like(module._module_params["weight_mean"])
        else:
            weight_var = module.scale_tform(module._module_params["weight_rho"]) ** 2
        var = self.functional(
            input=input[0] ** 2,
            weight=weight_var,
            bias=(
                None
                if bias_params[1] is None
                else module.scale_tform(bias_params[1]) ** 2
            ),
            **functional_kwargs,
        ) + self.functional(
            input=input[1],
            weight=module._module_params["weight_mean"] ** 2 + weight_var,
            bias=None,
            **functional_kwargs,
        )

        return MuVar(mu, var)


class ConvTranspose1d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose1d layer."""

    def __init__(self):
        """Initializer for ConvTranspose1d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv_transpose1d
        self.num_spatial_dims = 1


class ConvTranspose2d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose2d layer."""

    def __init__(self):
        """Initializer for ConvTranspose2d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv_transpose2d
        self.num_spatial_dims = 2


class ConvTranspose3d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose3d layer."""

    def __init__(self):
        """Initializer for ConvTranspose3d inference module"""
        super().__init__()
        self.functional = torch.nn.functional.conv_transpose3d
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
    input = MuVar(torch.tensor([1.23])[None, :], torch.tensor([3.21])[None, :])
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
