"""Inference modules for Bayesian neural network layers.

Custom propagators for specific layers (e.g., "Linear" for Bayesian analog of
torch.nn.Linear) should share a name with the layer such that
getattr(inference, layer.__class__.__name__) will return the custom propagator
for that layer if available.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import itertools
import math
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.distributions as dist

from . import types


if TYPE_CHECKING:
    from fastbnns.bnn.wrappers import BayesianModule


class MomentPropagator(torch.nn.Module):
    """Base class for layer propagators."""

    def __init__(self):
        """Initializer for MomentPropagator module."""
        super().__init__()

    def forward(
        self,
        module: Callable,
        input: Iterable[torch.Tensor],
    ) -> Iterable[torch.Tensor]:
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
        input: Iterable[torch.Tensor],
    ) -> Iterable[torch.Tensor]:
        """Propagate moments by relying intrinsically on methods in types.MuVar."""
        return module(input)


class UnscentedTransform(MomentPropagator):
    """Unscented transform propagation of mean and variance through `module`.

    This propagator uses the unscented transform to propagate mean and variance
    through a deterministic layer.
    """

    def __init__(
        self,
        sigma_scale: Optional[torch.Tensor] = None,
        sigma_weights: Optional[torch.Tensor] = None,
        n_module_samples: int = 1,
    ):
        """Initializer for UnscentedTransform inference module.

        Args:
            sigma_scale: Scale factor for sigma points mu -+ sigma_scale*var.sqrt().
            sigma_weights: Weighting factors corresponding to sigma points.
            n_module_samples: Number of samples to make of the module itself.
                This allows us to use the unscented transform through modules
                which themselves are parametrized by random variables.
        """
        super().__init__()

        # Set defaults as needed (defaults chosen as in https://doi.org/10.1117/12.280797)
        kappa = 2.0
        if sigma_scale is None:
            sigma_scale = torch.sqrt(torch.tensor(kappa + 1.0))
        self.register_buffer("_scale", sigma_scale)

        if sigma_weights is None:
            sigma_weights = torch.tensor(
                [kappa / (kappa + 1.0), 0.5 / (kappa + 1.0), 0.5 / (kappa + 1.0)]
            )
        self.register_buffer("_weights", sigma_weights)

        self.n_module_samples = n_module_samples

    def forward(
        self,
        module: Callable,
        input: types.MuVar,
        return_samples: bool = False,
    ) -> types.MuVar:
        """Propagate moments using the unscented transform."""
        if input.var is None:
            # Input variance is None (zero) so we only need to account for module samples.
            # Propagate mean and variance.
            if self.n_module_samples > 1:
                # Perform n_module_samples unscented transforms and combine results.
                mu_samples = []
                var_samples = []
                for n in range(self.n_module_samples):
                    # Forward pass through module.
                    mu_samples.append(module(input.mu))

                # Combine estimates from each unscented transform using law of total
                # expectation and law of total variance.
                mu = torch.stack(mu_samples).mean(dim=0)
                var = torch.stack(mu_samples).var(dim=0)
            else:
                # Forward pass through module.
                mu = module(input.mu)
                var = None
        else:
            # Select sigma points and reshape along batch dimension for batched eval.
            scaled_stdev = self._scale * input.var.sqrt()
            sigma_points = torch.stack(
                (input.mu, input.mu - scaled_stdev, input.mu + scaled_stdev)
            )
            sp_shape = sigma_points.shape
            sigma_points = sigma_points.reshape(
                sp_shape[0] * sp_shape[1], *sp_shape[2:]
            )
            weights = self._weights

            # Propagate mean and variance.
            if self.n_module_samples > 1:
                # Perform n_module_samples unscented transforms and combine results.
                mu_samples = []
                var_samples = []
                for n in range(self.n_module_samples):
                    # Forward pass through module and use unscented transform.
                    samples = module(sigma_points)
                    samples = samples.reshape(
                        sp_shape[0], sp_shape[1], *samples.shape[1:]
                    )
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
                var = torch.stack(var_samples).mean(dim=0) + torch.stack(
                    mu_samples
                ).var(dim=0)
            else:
                # Compute output mean and variance.
                samples = module(sigma_points)
                samples = samples.reshape(sp_shape[0], sp_shape[1], *samples.shape[1:])
                weights = weights.reshape(
                    (weights.shape[0],) + (1,) * (samples.ndim - 1)
                )
                mu = (weights * samples).sum(dim=0)
                var = (weights * ((samples - mu) ** 2)).sum(dim=0)

        if return_samples:
            return type(input)([mu, var]), samples
        else:
            return type(input)([mu, var])


class JointUnscentedTransform(MomentPropagator):
    """Unscented transform propagation of mean and variance through Bayesian `module`.

    This propagator uses the unscented transform to propagate mean and variance
    through a Bayesian layer, jointly applying an unscented transform that accounts
    for the input distribution and the parameter distributions of the layer.
    """

    def __init__(
        self,
        sigma_scale: Optional[float | torch.Tensor] = None,
        sigma_weights: Optional[list[float] | tuple[float, ...] | torch.Tensor] = None,
    ):
        """Initializer for JointUnscentedTransform inference module.

        Args:
            sigma_scale: Scale factor for sigma points mu -+ sigma_scale*var.sqrt().
            sigma_weights: Weighting factors corresponding to sigma points.
        """
        super().__init__()

        # Set defaults as needed (defaults chosen as in https://doi.org/10.1117/12.280797)
        kappa = 2.0
        if sigma_scale is None:
            sigma_scale = math.sqrt(kappa + 1.0)
        self._scale = sigma_scale

        if sigma_weights is None:
            sigma_weights = [
                kappa / (kappa + 1.0),
                0.5 / (kappa + 1.0),
                0.5 / (kappa + 1.0),
            ]
        self._weights = sigma_weights

    def forward(
        self,
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
        # If input and layer are both deterministic, we can just return the module call
        # without doing the unscented transform.
        if not (module._learn_var or input.var):
            return type(input)(module._module(input.mu), None)

        # Prepare input distribution sigma points.
        if input.var is None:
            sigma_point_and_weight = [(input.mu, 1.0)]
        else:
            offset = self._scale * input.var.sqrt()
            sigma_point_and_weight = [
                (input.mu, self._weights[0]),
                (input.mu - offset, self._weights[1]),
                (input.mu + offset, self._weights[2]),
            ]
        sigma_points_all = [sigma_point_and_weight]
        sigma_point_names = ["input"]

        # Prepare parameter distribution sigma points.
        for name in module._module._parameters:
            mu = module._module_params[f"{name}_mean"]
            rho = module._module_params[f"{name}_rho"]
            if rho is None:
                sigma_point_and_weight = [(mu, 1.0)]
            else:
                offset = self._scale * module.scale_tform(rho)
                sigma_point_and_weight = [
                    (mu, self._weights[0]),
                    (mu - offset, self._weights[1]),
                    (mu + offset, self._weights[2]),
                ]

            sigma_points_all.append(sigma_point_and_weight)
            sigma_point_names.append(name)

        # Loop over sigma points and evaluate.
        outputs = []
        output_weights = []
        for sigma_points in itertools.product(*sigma_points_all):
            # For each set of sigma points, separate input sigma points from
            # parameter distribution sigma points and pass through the module.
            sigma_point_input = sigma_points[0][0]
            total_weight = sigma_points[0][1]
            parameters = {}
            for name, sigma_point_param in zip(
                sigma_point_names[1:],
                sigma_points[1:],
            ):
                parameter_point, parameter_weight = sigma_point_param
                parameters[name] = parameter_point
                total_weight *= parameter_weight

            output = torch.func.functional_call(
                module._module,
                parameters,
                (sigma_point_input,),
            )
            outputs.append(output)
            output_weights.append(total_weight)

        # Estimate mean and variance.
        outputs = torch.stack(outputs)
        weights = outputs.new_tensor(output_weights)
        for _ in range(outputs.ndim - 1):
            weights = weights.unsqueeze(-1)
        output_mu = (weights * outputs).sum(dim=0)
        output_var = (weights * (outputs - output_mu) ** 2).sum(dim=0)

        return types.MuVar(output_mu, output_var)


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
        input: types.MuVar,
        return_samples: bool = False,
    ) -> types.MuVar:
        """Propagate moments by averaging over n_samples forward passes of module."""
        # If the input variance is not None, we'll need to sample the input as well.
        if input.var is None:
            samples = torch.stack([module(input.mu) for _ in range(self.n_samples)])
        else:
            input_dist = self.input_sampler(loc=input.mu, scale=input.var.sqrt())
            samples = torch.stack(
                [module(input_dist.sample()) for _ in range(self.n_samples)]
            )

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
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
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
            input=input.mu,
            weight=weight_mean,
            bias=bias_mean,
        )

        # Propagate variance.
        if weight_rho is None:
            if input.var is None:
                # No parameter variance and no input variance, so output has no variance.
                var = None
            else:
                var = self.functional(
                    input=input.var,
                    weight=weight_mean**2,
                )
        else:
            # First term accounts for parameter variance, second term propagates input variance.
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input.mu**2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
            )
            if input.var is not None:
                var += self.functional(
                    input=input.var,
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
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
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
            input=input.mu,
            weight=weight_mean,
            bias=bias_mean,
            **functional_kwargs,
        )

        # Propagate variance.
        if weight_rho is None:
            if input.var is None:
                # No parameter variance and no input variance, so output has no variance.
                var = None
            else:
                var = self.functional(
                    input=input.var,
                    weight=weight_mean**2,
                    bias=None,
                    **functional_kwargs,
                )
        else:
            # First term accounts for parameter variance, second term propagates input variance.
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input.mu**2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
                **functional_kwargs,
            )
            if input.var is not None:
                var += self.functional(
                    input=input.var,
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
        module: BayesianModule,
        input: types.MuVar,
        output_size: Optional[List[int]] = None,
    ) -> types.MuVar:
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
            input=input.mu,
            weight=weight_mean,
            bias=bias_mean,
            **functional_kwargs,
        )

        # Propagate variance.
        if weight_rho is None:
            if input.var is None:
                # No parameter variance and no input variance, so output has no variance.
                var = None
            else:
                var = self.functional(
                    input=input.var,
                    weight=weight_mean**2,
                    bias=None,
                    **functional_kwargs,
                )
        else:
            # First term accounts for parameter variance, second term propagates input variance.
            weight_var = module.scale_tform(weight_rho) ** 2
            var = self.functional(
                input=input.mu**2,
                weight=weight_var,
                bias=(None if bias_rho is None else module.scale_tform(bias_rho) ** 2),
                **functional_kwargs,
            )
            if input.var is not None:
                var += self.functional(
                    input=input.var,
                    weight=weight_mean**2 + weight_var,
                    bias=None,
                    **functional_kwargs,
                )

        return type(input)([mu, var])


class ConvTranspose1d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose1d layer."""

    functional = torch.nn.functional.conv_transpose1d
    num_spatial_dims = 1

    def __init__(self):
        """Initializer for ConvTranspose1d inference module"""
        super().__init__()


class ConvTranspose2d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose2d layer."""

    functional = torch.nn.functional.conv_transpose2d
    num_spatial_dims = 2

    def __init__(self):
        """Initializer for ConvTranspose2d inference module"""
        super().__init__()


class ConvTranspose3d(ConvTransposeNd):
    """Deterministic moment propagation of mean and variance through a ConvTranspose3d layer."""

    functional = torch.nn.functional.conv_transpose3d
    num_spatial_dims = 3

    def __init__(self):
        """Initializer for ConvTranspose3d inference module"""
        super().__init__()


class AvgPoolNd(MomentPropagator):
    """Deterministic moment propagation of mean and variance through AvgPoolNd layers."""

    def __init__(self):
        """Initializer for AvgPoolNd inference module"""
        super().__init__()

    def forward(
        self,
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
        """Analytical moment propagation through layer."""
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        kernel_size = module._module.kernel_size
        mu = self.functional(
            input=input.mu,
            kernel_size=kernel_size,
            stride=module._module.stride,
            padding=module._module.padding,
        )
        if input.var is None:
            var = None
        else:
            n_pool = (
                kernel_size**self.n_dim
                if isinstance(kernel_size, int)
                else torch.prod(torch.tensor(kernel_size))
            )
            var = (
                self.functional(
                    input=input.var,
                    kernel_size=kernel_size,
                    stride=module._module.stride,
                    padding=module._module.padding,
                )
                / n_pool
            )

        return type(input)([mu, var])


class AvgPool1d(AvgPoolNd):
    """Deterministic moment propagation of mean and variance through AvgPool1d layers."""

    functional = torch.nn.functional.avg_pool1d
    n_dim = 1

    def __init__(self):
        """Initializer for AvgPool1d inference module"""
        super().__init__()


class AvgPool2d(AvgPoolNd):
    """Deterministic moment propagation of mean and variance through AvgPool2d layers."""

    functional = torch.nn.functional.avg_pool2d
    n_dim = 2

    def __init__(self):
        """Initializer for AvgPool2d inference module"""
        super().__init__()


class AvgPool3d(AvgPoolNd):
    """Deterministic moment propagation of mean and variance through AvgPool3d layers."""

    functional = torch.nn.functional.avg_pool3d
    n_dim = 3

    def __init__(self):
        """Initializer for AvgPool3d inference module"""
        super().__init__()


class ReLUa(MomentPropagator):
    """Deterministic moment propagation of a normal random variable through a ReLU.

    NOTE: the suffix "a" is added to prevent fastbnns.bnn.wrappers.covert_to_bnn_() from
    automatically selecting this propagator for ReLU. This is currently desirable
    since using UnscentedTransform is faster (though less accurate) than analytical propagation.
    """

    def __init__(self):
        """Initializer for ReLU inference module"""
        super().__init__()

    def forward(
        self,
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
        """Analytical moment propagation through layer."""
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        if input.var is None:
            # With no input variance we can apply the ReLU directly.
            mu = module(input.mu)
            var = None
        else:
            # Compute the mean of the output assuming input independent normal random variables.
            s_input = input.var.sqrt()
            alpha = torch.clamp(-input.mu / s_input, min=-3.0, max=3.0)
            phi = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))  # P(input<0)
            psi = torch.exp(-0.5 * (alpha.pow(2))) / math.sqrt(2.0 * math.pi)
            ev_gt0 = input.mu + s_input * psi / (1.0 - phi)
            mu = (1.0 - phi) * ev_gt0

            # Compute the variance of the output assuming input independent normal random variables.
            var_gt0 = input.var * (
                1.0 + (alpha * psi / (1.0 - phi)) - (psi / (1.0 - phi)).pow(2)
            )
            var = (1 - phi) * var_gt0 + phi * (1 - phi) * ev_gt0.pow(2)

        if module._module.inplace:
            out = list(input)
            out[0].copy_(mu)
            out[1].copy_(var)
            return type(input)(out)
        else:
            return type(input)([mu, var])


class LeakyReLUa(MomentPropagator):
    """Deterministic moment propagation of a normal random variable through a leaky-ReLU.

    NOTE: the suffix "a" is added to prevent fastbnns.bnn.wrappers.covert_to_bnn_() from
    automatically selecting this propagator for LeakyReLU. This is currently desirable
    since using UnscentedTransform is faster (though less accurate) than analytical propagation.
    """

    def __init__(self):
        """Initializer for leaky-ReLU inference module"""
        super().__init__()

    def forward(
        self,
        module: BayesianModule,
        input: types.MuVar,
    ) -> types.MuVar:
        """Analytical moment propagation through layer."""
        ## Compute analytical result under mean-field approximation following
        ## https://doi.org/10.48550/arXiv.2402.14532
        if input.var is None:
            # With no input variance we can apply the ReLU directly.
            mu = module(input.mu)
            var = None
        else:
            # Compute the mean of the output assuming input independent normal random variables.
            l = -module._module.negative_slope
            s_input = input.var.sqrt()
            alpha = torch.clamp(-input.mu / s_input, min=-3.0, max=3.0)
            phi = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))  # P(input<0)
            psi = torch.exp(-0.5 * (alpha.pow(2))) / math.sqrt(2.0 * math.pi)
            ev_lt0 = input.mu - s_input * psi / phi
            ev_gt0 = input.mu + s_input * psi / (1.0 - phi)
            mu = l * phi * ev_lt0 + (1.0 - phi) * ev_gt0

            # Compute the variance of the output assuming input independent normal random variables.
            var_lt0 = input.var * (1.0 - (alpha * psi / phi) - (psi / phi).pow(2))
            var_gt0 = input.var * (
                1.0 + (alpha * psi / (1.0 - phi)) - (psi / (1.0 - phi)).pow(2)
            )
            var = (
                (l**2) * phi * var_lt0
                + (1 - phi) * var_gt0
                + phi * (1 - phi) * (l * ev_lt0 - ev_gt0).pow(2)
            )

        if module._module.inplace:
            out = list(input)
            out[0].copy_(mu)
            out[1].copy_(var)
            return type(input)(out)
        else:
            return type(input)([mu, var])


if __name__ == "__main__":
    """Example usages of inference modules."""
    import matplotlib.pyplot as plt
    from fastbnns.bnn.wrappers import BayesianModule

    # Define a nonlinearity to propagate through.
    layer = BayesianModule(module=torch.nn.LeakyReLU())

    # Define some propagators.
    mc = MonteCarlo(n_samples=100)
    ut = UnscentedTransform()
    lr = LeakyReLUa()

    # Propagate example data through layer.
    input = types.MuVar(torch.tensor([1.23])[None, :], torch.tensor([3.21])[None, :])
    out_mc, samples_mc = mc(module=layer, input=input, return_samples=True)
    out_ut, samples_ut = ut(module=layer, input=input, return_samples=True)
    out_lr = lr(module=layer, input=input)

    # Plot results.
    x = torch.linspace(
        (input.mu - 3.0 * torch.sqrt(input.var)).squeeze(),
        (input.mu + 3.0 * torch.sqrt(input.var)).squeeze(),
        1000,
    )
    fig, ax = plt.subplots()
    ax.hist(samples_mc.squeeze(), density=True, label="Monte Carlo samples")
    ax.plot(
        x,
        torch.distributions.Normal(
            loc=out_mc.mu.squeeze(), scale=out_mc.var.squeeze().sqrt()
        )
        .log_prob(x)
        .exp(),
        label="Monte Carlo estimated PDF",
    )
    ax.plot(
        x,
        torch.distributions.Normal(
            loc=out_ut.mu.squeeze(), scale=out_ut.var.squeeze().sqrt()
        )
        .log_prob(x)
        .exp(),
        label="Unscented transform estimated PDF",
    )
    ax.plot(
        x,
        torch.distributions.Normal(
            loc=out_lr.mu.squeeze(), scale=out_lr.var.squeeze().sqrt()
        )
        .log_prob(x)
        .exp(),
        label="Analytical PDF",
    )
    plt.legend()
    plt.show()
