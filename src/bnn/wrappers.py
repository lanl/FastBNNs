"""Collections of Bayesian neural network layers."""

from abc import ABC, abstractmethod
import copy
import functools
import re
import sys
from typing import Union

import numpy as np
import torch
import torch.distributions as dist

import bnn.inference
from bnn.inference import MomentPropagator
from bnn.losses import kl_divergence_sampled
from bnn.priors import Distribution
from bnn.types import MuVar
from utils.torch_utils import get_torch_functional


# List out torch.nn modules whose functionals in torch.nn.functional can directly
# replicate the corresponding torch.nn behavior (see torch.nn.Conv2d for an
# example of when this is NOT the case, since the forward pass contains additional
# logic/processing before the call to torch.nn.functional.conv2d()).
HAS_COMPATIBLE_FUNCTIONAL = [
    "Linear",
    "Bilinear",
]

# Define layers that can be applied to input mean and variance without additional
# processing (e.g., a flatten layer, which only changes the shape of the input).
PASSTHROUGH = [
    "ChannelShuffle",
    "Identity",
    "Flatten",
    "Unflatten",
    *[f"ReflectionPad{n+1}d" for n in range(3)],
    *[f"ReplicationPad{n+1}d" for n in range(3)],
    *[f"ZeroPad{n+1}d" for n in range(3)],
    *[f"ConstantPad{n+1}d" for n in range(3)],
    *[f"CircularPad{n+1}d" for n in range(3)],
]


CURRENT_MODULE = sys.modules[__name__]


def select_default_propagator(
    module: torch.nn.Module, is_bayesian: bool = True
) -> MomentPropagator:
    """Select a compatible moment propagator for `module`.

    Args:
        module: Module that we'll choose a propagator for.
        is_bayesian: Flag indicating parameters of `module` will be treated
            as distributions.
    """
    if hasattr(bnn.inference, module.__class__.__name__):
        # A custom propagator exists for this module so we'll use that.
        propagator = getattr(bnn.inference, module.__class__.__name__)
        moment_propagator = propagator()
    # elif not is_bayesian or (
    elif not is_bayesian or (
        len([p for p in module.parameters() if p.requires_grad]) == 0
    ):
        # If this module doesn't have learnable parameters we'll
        # default to the unscented transform.
        moment_propagator = bnn.inference.UnscentedTransform()
    else:
        # With learnable Bayesian parameters, we'll default to
        # Monte Carlo sampling.
        moment_propagator = bnn.inference.MonteCarlo()

    return moment_propagator


def isolate_leaf_module_names(module_names: list[str]) -> list[str]:
    """Prepare a list of leaf modules of `model`.

    This function filters `module_names` to eliminate the names of parent modules.
    For example, if we have a model: torch.nn.Module and call [p]
    """
    leaf_names = []
    for module in module_names[::-1]:
        # If other modules are a prefix of this modules name, we'll assume they
        # are this modules parent (hence not a leaf module).
        children = []
        for leaf in leaf_names:
            matches = re.match(f"{module}.*", leaf)
            if matches is not None:
                children.append(matches)
        if len(children) == 0:
            leaf_names.append(module)

    return leaf_names


def convert_to_bnn_(
    model: torch.nn.Module,
    wrapper_kwargs: dict = {},
    wrapper_kwargs_global: dict = {},
    passthrough_module_tags: list = [],
) -> None:
    """Convert layers of `model` to Bayesian counterparts.

    Args:
        model: Model to be converted to Bayesian counterpart.
        wrapper_kwargs: Additional keyword arguments passed to
            initialization of named Bayesian layers.  For example, if `model`
            has a module named "module1", we'll convert "module1" as
            Converter(module1, **wrapper_kwargs["module1"]) where
            Converter is a module converter.
        wrapper_kwargs_global: Keyword arguments that we'll merge
            with values of bayesian_module_kwargs as, e.g.,
            Converter(module1, **(wrapper_kwargs_global | wrapper_kwargs["module1"]))
        passthrough_module_tags: List of strings that, if present in the class
            name of a module, will indicate the module should be treated as a
            passthrough module: i.e., apply forward method to mean and variance
            directly without additional logic.
    """
    # Search for modules of `model` to convert, removing stem modules from the
    # list (we just want the leaf modules that contain parameters).
    module_names = [n for n, _ in model.named_modules()]
    leaf_names = isolate_leaf_module_names(module_names)

    # Replace leaf modules with Bayesian counterparts or compatible passthroughs.
    for leaf in leaf_names:
        # module = get_submodule_custom(model=model, target=leaf)
        module = model.get_submodule(leaf)
        module_name = module.__class__.__name__

        # Prepare module arguments.
        module_kwargs = wrapper_kwargs_global | wrapper_kwargs.pop(leaf, {})

        # Search for an appropriate module converter, in the following order of
        # priority:
        #   (1) Passthrough layer if tagged by passthrough_module_tags or listed
        #       in PASSTHROUGH list.
        #   (2) Named converters if a wrapper exists with the same name as the
        #       module class.
        #   (3) BayesianModule
        if (module_name in PASSTHROUGH) or any(
            [tag in module_name for tag in passthrough_module_tags]
        ):
            # This module can be broadcast along (mu, var) without additional
            # processing (e.g., a flatten layer, which only changes shapes).
            bayesian_layer = PassthroughModule(module=module, **module_kwargs)
        elif hasattr(CURRENT_MODULE, module_name):
            # If a custom converter exists for this named layer, we'll use that by default.
            bayesian_layer = getattr(CURRENT_MODULE, module_name)(
                module=module, **module_kwargs
            )
        else:
            bayesian_layer = BayesianModule(module=module, **module_kwargs)

        # Reset submodule to the converted module.
        model.set_submodule(leaf, bayesian_layer)


class BayesianModuleBase(ABC, torch.nn.Module):
    """Abstract base class for Bayesian modules."""

    @property
    @abstractmethod
    def learn_var(self, *args, **kwargs) -> bool:
        """Flag indicating variance of module parameters should be learnable."""
        pass

    @property
    @abstractmethod
    def module_params(self, *args, **kwargs) -> torch.ParameterDict:
        """Dictionary organizing parameters of this module."""
        pass

    @property
    @abstractmethod
    def module(self, *args, **kwargs) -> torch.nn.Module:
        """Return a sampled instance of this module."""
        pass

    @property
    @abstractmethod
    def moment_propagator(self, *args, **kwargs) -> MomentPropagator:
        """Function used to propagate mean and variance through this module."""
        pass

    @abstractmethod
    def compute_kl_divergence(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Method that computes the KL divergence of this module w.r.t. some prior."""
        pass

    @abstractmethod
    def forward(
        self,
        input: Union[MuVar, torch.Tensor],
        *args,
        **kwargs,
    ) -> Union[MuVar, torch.Tensor]:
        """Method that computes a forward pass through this module."""
        pass


class BayesianModule(BayesianModuleBase):
    """Base class for BayesianModule modules to make PyTorch modules BNN compatible."""

    def __init__(
        self,
        module: torch.nn.Module,
        samplers: dict = None,
        samplers_init: dict = None,
        priors: dict = None,
        moment_propagator: MomentPropagator = None,
        learn_var: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """BayesianModule initializer.

        Args:
            module: Module with parameters that we wish to treat as distributions
                (or in some cases, a module between Bayesian layers that we'll
                wrap for compatibility purposes).
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
                parameters (e.g., samplers_init["weight_mean"].sample() should
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
            learn_var: Flag indicating learnable parameters should be treated as
                distributions (with learned variance).  This flag allows us to
                wrap a layer with this module and its functionality without
                changing the behavior of its parameters.
        """
        super().__init__(*args, **kwargs)
        self.__name__ = module.__class__.__name__

        # Store `module` and a copy of `module` to act as the mean and unscaled
        # st. dev. parameters (rho), respectively.
        mu = module
        module_params = [p for p in module.named_parameters()]
        _module_params = torch.nn.ParameterDict()
        self._learn_var = learn_var
        if (len(module_params) > 0) and learn_var:
            # Prepare a copy of `module` to act as the unscaled st. dev. parameters.
            rho = copy.deepcopy(module)
            for key, _ in module_params:
                _module_params[key + "_mean"] = getattr(mu, key)
                _module_params[key + "_rho"] = getattr(rho, key)
        else:
            rho = None
            for key, _ in module_params:
                _module_params[key + "_mean"] = getattr(mu, key)
                _module_params[key + "_rho"] = None
        self.mu = mu
        self.rho = rho
        self._module_params = _module_params

        # Prepare a module copy for sampling (we'll actually initialize this
        # on first use so that we don't waste memory if it's never needed).
        self._module = None

        # Store a functional version of this module if possible (some modules
        # won't have a functional, or their action can't be directly replicated
        # through the functional).
        if module.__class__.__name__ in HAS_COMPATIBLE_FUNCTIONAL:
            self._functional = get_torch_functional(module.__class__)
        else:
            self._functional = None

        # Validate and set moment propagator.
        # Set moment propagator.
        if moment_propagator is None:
            moment_propagator = select_default_propagator(
                module=module, is_bayesian=learn_var
            )
        self._moment_propagator = moment_propagator

        # Create samplers for each named parameter.
        if samplers_init is None:
            # Prepare (relatively arbitrary) default samplers.
            samplers_init = {}
            for key, val in _module_params.items():
                if "_mean" in key:
                    samplers_init[key] = dist.Uniform(
                        low=-1.0 / np.sqrt(val.shape[-1]),
                        high=1.0 / np.sqrt(val.shape[-1]),
                    )
                else:
                    samplers_init[key] = dist.Uniform(
                        low=-8.0,
                        high=-2.0,
                    )
        self.samplers_init = samplers_init
        self.reset_parameters()

        # Initialize variational distribution samplers.
        scale_tform = torch.nn.functional.softplus
        self.scale_tform = scale_tform
        if samplers is None:
            samplers = {key: dist.Normal for key, _ in module.named_parameters()}
        self.samplers = samplers

        # Set priors.
        if priors is None:
            # Default to same distributions as `samplers_init` mean samplers.
            priors = {key: samplers_init[key + "_mean"] for key in samplers.keys()}
        self.priors = priors

    @property
    def learn_var(self) -> bool:
        """Return property `learn_var`.

        This property is written as an @property method for compatibility with the
        abstract parent class.
        """
        return self._learn_var

    @property
    def moment_propagator(self) -> bool:
        """Return property `moment_propagator`.

        This property is written as an @property method for compatibility with the
        abstract parent class.
        """
        return self._moment_propagator

    @property
    def module_params(self) -> dict:
        """Return a sample of this module's parameters."""
        # Initialize the samplers on the fly to ensure we're on the same device
        # as self._module_params.
        samplers = {
            key: val(
                loc=self._module_params[key + "_mean"],
                scale=self.scale_tform(self._module_params[key + "_rho"]),
            )
            for key, val in self.samplers.items()
        }

        return {key: val.sample() for key, val in samplers.items()}

    @property
    def module(self) -> torch.nn.Module:
        """Prepare a callable that acts like input `module` with random parameters."""
        if self._functional is None:
            # No functional is available/compatible with this converter so we'll use
            # copies of the input `module` instead.
            if self.rho is None:
                # If rho is None, we don't have any parameters to sample so we can
                # just return self.mu.
                return self.mu
            else:
                # In this case, we want to resample parameters of self._module.
                if self._module is None:
                    self._module = copy.deepcopy(self.mu)
                params = {key: val for key, val in self._module.named_parameters()}
                params_sampled = self.module_params
                for param_name, param in params.items():
                    param.data = params_sampled[param_name]

                return self._module
        else:
            # Return the functional with sampled parameters pre-populated.
            return functools.partial(self._functional, **self.module_params)

    def reset_parameters(self) -> None:
        """Resample layer parameters from initial distributions."""
        for key, param in self._module_params.items():
            if param is not None:
                param.data = self.samplers_init[key].sample(sample_shape=param.shape)

    def compute_kl_divergence(
        self, priors: Union[dict, Distribution] = None, n_samples: int = 1
    ) -> torch.Tensor:
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
                            loc=self._module_params[param_name + "_mean"],
                            scale=self.scale_tform(
                                self._module_params[param_name + "_rho"]
                            ),
                        ),
                        param_prior.distribution,
                    ).sum()
                )
            except NotImplementedError:
                # Compute Monte Carlo KL divergence.
                kl_divergence.append(
                    kl_divergence_sampled(
                        dist0=self.samplers[param_name](
                            loc=self._module_params[param_name + "_mean"],
                            scale=self.scale_tform(
                                self._module_params[param_name + "_rho"]
                            ),
                        ),
                        dist1=param_prior.distribution,
                        n_samples=n_samples,
                    ).sum()
                )

        return torch.stack(kl_divergence).sum()

    def forward(
        self,
        input: Union[MuVar, torch.Tensor],
        *args,
        **kwargs,
    ) -> Union[MuVar, torch.Tensor]:
        """Forward pass through layer."""
        if isinstance(input, MuVar):
            # Propagate mean and variance through layer.
            out = self.moment_propagator(
                module=self,
                input=input,
                *args,
                **kwargs,
            )
        else:
            # Compute forward pass with a random sample of parameters.
            out = self.module(input, *args, **kwargs)

        return out


class PassthroughModule(torch.nn.Module):
    """PassthroughModule for compatibility with other Bayesian layers."""

    def __init__(
        self,
        module: torch.nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """PassthroughModule initializer.

        Args:
            module: Module that will be applied in forward pass.  When the
                output of the previous layer is a MuVar type, module is
                simply broadcast across both elements of MuVar.
        """
        super().__init__(*args, **kwargs)
        self.__name__ = module.__class__.__name__
        self.module = module

    def forward(
        self,
        input: Union[MuVar, torch.Tensor],
        *args,
        **kwargs,
    ) -> Union[MuVar, torch.Tensor]:
        """Forward pass through layer."""
        if isinstance(input, MuVar):
            # Propagate mean and variance through layer.
            out = MuVar(self.module(input[0]), self.module(input[1]))
        else:
            # Compute forward pass with a random sample of parameters.
            out = self.module(input)

        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bnn.inference import Linear

    # Basic usage example of BayesianLinear.
    in_features = 3
    out_features = 1
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    bayesian_linear = BayesianModule(
        module=linear,
    )

    # Usage with MonteCarlo moment propagation through layer.
    n_samples = 100
    bayesian_linear_mc = BayesianModule(
        module=linear,
        moment_propagator=bnn.inference.MonteCarlo(n_samples=n_samples),
    )  # computes moments from Monte Carlo without returning actual samples
    bayesian_linear_mc.load_state_dict(bayesian_linear.state_dict())

    # Usage with analytical moment propagation through layer.
    bayesian_linear_det = BayesianModule(
        module=linear,
        moment_propagator=Linear(),
    )  # analytically computes moments
    bayesian_linear_det.load_state_dict(bayesian_linear.state_dict())

    # Generate example outputs.  Wrapping input in type MuVar will signal the
    # layers to use their moment_propagator modules instead of sampling.
    batch_size = 1
    input = torch.randn(batch_size, in_features)
    output_samples = torch.stack([bayesian_linear(input)[0] for _ in range(n_samples)])
    output_mc = bayesian_linear_mc(MuVar(input))
    output_det = bayesian_linear_det(MuVar(input))

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
