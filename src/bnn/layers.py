"""Collections of Bayesian neural network layers."""

import copy
from typing import Union

import numpy as np
import torch
import torch.distributions as dist

from bnn.inference import MomentPropagator, MonteCarlo, UnscentedTransform
from bnn.losses import kl_divergence_sampled
from utils.misc import get_torch_functional


class BayesianLayer(torch.nn.Module):
    """Bayesian implementation of a generic PyTorch module."""

    def __init__(
        self,
        module: torch.nn.Module,
        samplers: dict = None,
        samplers_init: dict = None,
        priors: dict = None,
        moment_propagator: MomentPropagator = None,
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
                to the mean and (unscaled, see `scale_tform` usage) standard deviation
                parameters (e.g., samplers_init["weight"][0].sample() should
                return a shape (out_features, in_features) tensor defining
                initial weight parameters).  Keys should match those in `samplers`.
            priors: Dictionary of prior distributions over layer parameters.
                These distributions should be initialized (like `samplers_init`)
                and have a .sample() method returning a tensor of the same
                size as the corresponding layer parameter and a .log_prob(x) method
                returning the log of the PDF at input x in the distributions domain.
                These distributions are used in self.compute_kl_divergence().
                Keys should match those in `samplers`.
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

        # Parametrize the distributions of the Bayesian counterparts of each
        # named parameter.  Each distribution will be parametrized by a
        # mean and unscaled st. dev. parameter.  These are stored as
        # a ParameterList to simplify advanced training strategies (e.g.,
        # freezing mean parameters and only training st. dev. parameters).
        _module_params = torch.nn.ParameterDict()
        for param in module.named_parameters():
            _module_params[param[0]] = torch.nn.ParameterList(
                [
                    torch.empty(
                        param[1].shape,
                        device=param[1].device,
                        dtype=param[1].dtype,
                    ),
                    torch.empty(
                        param[1].shape,
                        device=param[1].device,
                        dtype=param[1].dtype,
                    ),
                ]
            )  # (mean, unscaled standard deviation)
        self._module_params = _module_params

        # Set moment propagator.
        self.propagate_moments = propagate_moments
        if moment_propagator is None:
            if len(_module_params) == 0:
                # If this module doesn't have learnable parameters we'll
                # default use the unscented transform.
                moment_propagator = UnscentedTransform()
            else:
                # With learnable Bayesian parameters, we'll default to
                # Monte Carlo sampling.
                moment_propagator = MonteCarlo()
        self.moment_propagator = moment_propagator

        # Create samplers for each named parameter.
        if samplers_init is None:
            # Prepare (relatively arbitrary) default samplers.
            samplers_init = {}
            for key, val in _module_params.items():
                samplers_init[key] = (
                    dist.Uniform(
                        low=-np.sqrt(val[0].numel()),
                        high=np.sqrt(val[0].numel()),
                    ),
                    dist.Uniform(
                        low=-8.0,
                        high=-2.0,
                    ),
                )
        self.samplers_init = samplers_init
        self.reset_parameters()

        # Initialize variational distribution samplers.
        scale_tform = torch.nn.functional.softplus
        self.scale_tform = scale_tform
        if samplers is None:
            samplers = {key: dist.Normal for key in _module_params.keys()}
        self.samplers = samplers

        # Set priors.
        if priors is None:
            # Default to same distributions as `samplers_init` mean samplers.
            priors = {key: copy.deepcopy(val[0]) for key, val in samplers_init.items()}
        self.priors = priors

    @property
    def module_params(self):
        """Return a sample of this module's parameters."""
        # Initialize the samplers on the fly to ensure we're on the same device
        # as self._module_params.
        samplers = {
            key: val(
                loc=self._module_params[key][0],
                scale=self.scale_tform(self._module_params[key][1]),
            )
            for key, val in self.samplers.items()
        }

        return {
            key: (None if self._module_params[key] is None else val.sample())
            for key, val in samplers.items()
        }

    def reset_parameters(self) -> None:
        """Resample layer parameters from initial distributions."""
        for key, params in self._module_params.items():
            if params is not None:
                for param, sampler in zip(params, self.samplers_init[key]):
                    param.data = sampler.sample(sample_shape=param.shape)

    def set_requires_grad_loc(self, requires_grad: bool) -> None:
        """Set requires_grad property of all loc parameters."""
        for loc_scale in self._module_params.values():
            loc_scale[0].requires_grad = requires_grad

    def set_requires_grad_scale(self, requires_grad: bool) -> None:
        """Set requires_grad property of all scale parameters."""
        for loc_scale in self._module_params.values():
            loc_scale[1].requires_grad = requires_grad

    def compute_kl_divergence(
        self, priors: Union[dict, dist.Distribution] = None, n_samples: int = 1
    ) -> torch.tensor:
        """Compute the KL divergence between self.prior and module parameters.

        Args:
            priors: Prior distribution over parameters.  This can be a single
                distribution for all parameters or a dictionary whose keys
                correspond to named parameters of this layer.  By default, None
                will use self.priors.  This argument is used to allow external
                passing of priors not defined at initialization of this layer.
            n_samples: Number of Monte Carlo samples used to estimate KL
                divergence for distributions not compatible with
                torch.nn.distributions.kl_divergence()
        """
        # Return 0.0 if no learnable parameters are in this layer.
        if len([p for p in self.parameters()]) == 0:
            return torch.tensor(0.0)

        # Reorganize priors if needed.
        if priors is None:
            priors = self.priors
        elif not isinstance(priors, dict):
            priors = {key: priors for key in self.samplers.keys()}

        # Loop over parameters and compute KL contribution.
        kl_divergence = []
        for param_name, param_prior in priors.items():
            # Compute KL divergence for this parameter.
            try:
                # Attempt to use PyTorch kl_divergence.  If our distributions
                # aren't compatible, dist.kl_divergence kicks us to the except
                # clause below.
                kl_divergence.append(
                    dist.kl_divergence(
                        self.samplers[param_name](
                            loc=self._module_params[param_name][0],
                            scale=self.scale_tform(self._module_params[param_name][1]),
                        ),
                        param_prior,
                    ).sum()
                )
            except NotImplementedError:
                # Compute Monte Carlo KL divergence.
                kl_divergence.append(
                    kl_divergence_sampled(
                        dist0=self.samplers[param_name](
                            loc=self._module_params[param_name][0],
                            scale=self.scale_tform(self._module_params[param_name][1]),
                        ),
                        dist1=param_prior,
                        n_samples=n_samples,
                    ).sum()
                )

        return torch.stack(kl_divergence).sum()

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

    from bnn.inference import Linear

    # Basic usage example of BayesianLinear.
    in_features = 3
    out_features = 1
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    bayesian_linear = BayesianLayer(
        module=linear,
        propagate_moments=False,
    )

    # Usage with MonteCarlo moment propagation through layer.
    n_samples = 100
    bayesian_linear_mc = BayesianLayer(
        module=linear,
        propagate_moments=True,
        moment_propagator=MonteCarlo(n_samples=n_samples),
    )  # computes moments from Monte Carlo without returning actual samples
    bayesian_linear_mc.load_state_dict(bayesian_linear.state_dict())

    # Usage with analytical moment propagation through layer.
    bayesian_linear_det = BayesianLayer(
        module=linear,
        propagate_moments=True,
        moment_propagator=Linear(),
    )  # analytically computes moments
    bayesian_linear_det.load_state_dict(bayesian_linear.state_dict())

    # Generate example outputs.
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
