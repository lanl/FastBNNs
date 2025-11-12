"""Collections of Bayesian neural network layers."""

from abc import ABC, abstractmethod
import copy
import functools
import re
import sys
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributions as dist

from . import inference
from .inference import MomentPropagator
from .losses import kl_divergence_sampled
from .priors import Distribution
from .types import MuVar


# Define layers that can be applied to input mean and variance without additional
# processing (e.g., a flatten layer, which only changes the shape of the input).
BROADCAST = [
    "ChannelShuffle",
    "Identity",
    "Flatten",
    "Unflatten",
    *[f"ReflectionPad{n + 1}d" for n in range(3)],
    *[f"ReplicationPad{n + 1}d" for n in range(3)],
    *[f"ZeroPad{n + 1}d" for n in range(3)],
    *[f"ConstantPad{n + 1}d" for n in range(3)],
    *[f"CircularPad{n + 1}d" for n in range(3)],
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
    if hasattr(inference, module.__class__.__name__):
        # A custom propagator exists for this module so we'll use that.
        propagator = getattr(inference, module.__class__.__name__)
        moment_propagator = propagator()
    elif not is_bayesian or (
        len([p for p in module.parameters() if p.requires_grad]) == 0
    ):
        # If this module doesn't have learnable parameters we'll
        # default to the unscented transform.
        moment_propagator = inference.UnscentedTransform()
    else:
        # With learnable Bayesian parameters, we'll default to
        # Monte Carlo sampling.
        moment_propagator = inference.MonteCarlo()

    return moment_propagator


def isolate_leaf_module_names(module_names: list[str]) -> list[str]:
    """Prepare a list of leaf modules of `model`.

    This function filters `module_names` to eliminate the names of parent modules.
    For example, if we have a model: torch.nn.Module with named modules
    m = ["", "module1", "module2", "module1.submodule", "module2.submodule"],
    isolate_leaf_module_names(m) == ["module1.submodule", "module2.submodule"]
    """
    leaf_names = []
    module_names.remove("")  # remove root module empty string
    for module in module_names[::-1]:
        # If other modules are a prefix of this modules name, we'll assume they
        # are this modules parent (hence not a leaf module).
        children = []
        for leaf in leaf_names:
            matches = re.match(f"{module}\..*", leaf)
            if matches is not None:
                children.append(matches)
        if len(children) == 0:
            leaf_names.append(module)

    return leaf_names


def convert_to_bnn_(
    model: torch.nn.Module,
    layer_wrappers: dict = {},
    layer_wrappers_tag: dict = {},
    layer_wrappers_type: dict = {},
    wrapper_kwargs: dict = {},
    wrapper_kwargs_tag: dict = {},
    wrapper_kwargs_type: dict = {},
    wrapper_kwargs_global: dict = {},
) -> None:
    """Convert layers of `model` to Bayesian counterparts.

    Args:
        model: Model to be converted to Bayesian counterpart.
        layer_wrappers: Dictionary of manually-specified wrappers for specific
            module layers. The keys are names of leaf modules (e.g.,
            "module1.layer1") and the values are the corresponding wrapper
            class present in this module (e.g., "BroadcastModule").
        layer_wrappers_tag: Extends functionality of `layer_wrappers` where
            keys don't have to be exact layer names but instead can be tags,
            e.g., {"encoder": "BroadcastModule"} specifies that any
            submodule in [name for name, _ in model.named_modules()] satisfying
            `"encoder" in name` will be wrapped with a BroadcastModule.
        layer_wrappers_type: Extends functionality of `layer_wrappers` where
            keys are now type names, e.g., {"BatchNorm2d": "BroadcastModule"}
            specifies that any submodule in [m for m in model.modules()]
            satisfying `"BatchNorm2d" == type(m).__name__"` will be wrapped
            with a BroadcastModule.
        wrapper_kwargs: Additional keyword arguments passed to
            initialization of named Bayesian layers.  For example, if `model`
            has a module named "module1", we'll convert "module1" as
            Converter(module1, **wrapper_kwargs["module1"]) where
            Converter is a module converter.
        wrapper_kwargs_tag: Extends functionality of `wrapper_kwargs` where
            keys don't have to be exact layer names but instead can be tags,
            e.g., {"encoder": encoder_kwargs} specifies that any
            submodule in [name for name, _ in model.named_modules()] satisfying
            `"encoder" in name` will use `encoder_kwargs` for their wrapper.
        wrapper_kwargs_type: Extends functionality of `wrapper_kwargs` where
            keys are now type names, e.g., {"BatchNorm2d": batchnorm_kwargs}
            specifies that any submodule in [m for m in model.modules()]
            satisfying `"BatchNorm2d" == type(m).__name__"` will use
            `batchnorm_kwargs` for their wrapper.
        wrapper_kwargs_global: Keyword arguments that we'll merge
            with values of wrapper_kwargs, wrapper_kwargs_tag, and
            wrapper_kwargs_type as appropriate for each module.
    """
    # Search for modules of `model` to convert, removing stem modules from the
    # list (we just want the leaf modules that contain parameters).
    modules = {k: v for k, v in model.named_modules()}
    leaf_names = isolate_leaf_module_names(list(modules.keys()))

    # Replace leaf modules with Bayesian counterparts or compatible passthroughs.
    for leaf in leaf_names:
        module = modules[leaf]
        module_name = type(module).__name__

        # Prepare module arguments in the following order of priority:
        #   (1) User-specified kwargs in `wrapper_kwargs`
        #   (2) User-specified kwargs in `wrapper_kwargs_tag`
        #   (3) User-specified kwargs in `wrapper_kwargs_type`
        #   (4) Global default kwargs in `wrapper_kwargs_global`
        if custom_kwargs := wrapper_kwargs.pop(leaf, {}):
            pass
        elif custom_kwargs := [v for k, v in wrapper_kwargs_tag.items() if k in leaf]:
            # If there are multiple matches we'll just use the first one.
            custom_kwargs = custom_kwargs[0]
        elif custom_kwargs := [
            v for k, v in wrapper_kwargs_type.items() if module_name == k
        ]:
            # If there are multiple matches we'll just use the first one.
            custom_kwargs = custom_kwargs[0]
        else:
            custom_kwargs = {}
        module_kwargs = dict(wrapper_kwargs_global) | dict(custom_kwargs)

        ## Search for an appropriate module converter, in the following order of
        ## priority:
        #   (1) User-specified wrapper designated in `layer_wrappers`
        #   (2) User-specified wrapper designated in `layer_wrappers_tag`
        #   (3) User-specified wrapper designated in `layer_wrappers_type`
        #   (4) Named converters if a wrapper exists with the same name as the
        #       module class
        #   (5) BroadcastModule if the module class is listed in BROADCAST
        #   (6) BayesianModule

        if custom_wrapper := layer_wrappers.pop(leaf, None):
            pass
        elif custom_wrapper := [v for k, v in layer_wrappers_tag.items() if k in leaf]:
            # If there are multiple matches we'll just use the first one.
            custom_wrapper = custom_wrapper[0]
        elif custom_wrapper := [
            v for k, v in layer_wrappers_type.items() if module_name == k
        ]:
            # If there are multiple matches we'll just use the first one.
            custom_wrapper = custom_wrapper[0]
        else:
            custom_wrapper = None

        if custom_wrapper is not None:
            # Wrap module in user-specified wrapper.
            bayesian_layer = getattr(CURRENT_MODULE, custom_wrapper)(
                module=module, **module_kwargs
            )
        elif hasattr(CURRENT_MODULE, module_name):
            # If a custom converter exists for this named layer, we'll use that by default.
            bayesian_layer = getattr(CURRENT_MODULE, module_name)(
                module=module, **module_kwargs
            )
        elif module_name in BROADCAST:
            # This module can be broadcast along (mu, var) without additional
            # processing (e.g., a flatten layer, which only changes shapes).
            bayesian_layer = BroadcastModule(module=module, **module_kwargs)
        else:
            bayesian_layer = BayesianModule(module=module, **module_kwargs)

        # Reset submodule to the converted module.
        model.set_submodule(leaf, bayesian_layer)


def convert_to_nn(
    bnn: torch.nn.Module,
) -> None:
    """Inverse of convert_to_bnn_ to convert a BNN back to a standard NN

    Args:
        model: Bayesian NN to be converted back to a standard NN.
    """
    # Search for modules of `model` to convert, removing stem modules from the
    # list (we just want the leaf modules that contain parameters).
    model = copy.deepcopy(bnn)
    module_names = [n for n, _ in model.named_modules()]
    leaf_names = isolate_leaf_module_names(module_names)

    # Remove BNN-specific modules from list.
    bnn_module_names = ["_module_params", "_moment_propagator"]
    leaf_names = [
        leaf for leaf in leaf_names if not any([bn in leaf for bn in bnn_module_names])
    ]

    # Replace Bayesian leaf modules with standard counterparts.
    for leaf in leaf_names:
        # If `leaf` is a named `_module` parameter, we'll reset the module to the `_module` leaf.
        # If the module is a BroadcastModule, we just need to remove the wrapper.
        module = model.get_submodule(leaf)
        leaf_split = leaf.split(".")
        if leaf_split[-1] == "_module":
            model.set_submodule(".".join(leaf_split[:-1]), module)
        elif isinstance(module, BroadcastModule):
            model.set_submodule(leaf, module.module)

    return model


class BayesianModuleBase(ABC, torch.nn.Module):
    """Abstract base class for Bayesian modules."""

    # Define a scale transform to transform learnable `rho` parameters to scale
    # parameters (e.g., sigma = torch.nn.functional.softplus(rho)) where `rho` is
    # learned and `sigma` is the standard deviation of the Gaussian defining
    # a module's parameters.
    scale_tform = torch.nn.functional.softplus

    def scale_tform_inv(x: torch.Tensor) -> torch.Tensor:
        # Inverse of self.scale_tform.
        return torch.log(torch.exp(x) - 1.0)

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
    def module_map(self, *args, **kwargs) -> torch.nn.Module:
        """Return a non-Bayesian instance of self with parameters set to learned means."""
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
        samplers: Optional[dict] = None,
        samplers_init: Optional[dict] = None,
        resample_mean: bool = True,
        priors: Optional[dict] = None,
        moment_propagator: Optional[MomentPropagator] = None,
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
                an object with a .rsample() method which returns tensors
                corresponding to sampled parameters.  For example,
                samplers["weight"](loc=0.0, scale=1.0).rsample()
                must return a tensor of size (out_features, in_features) if
                module = torch.nn.Linear(in_features, out_features)
            samplers_init: Dictionary of samplers that can be sampled to reset
                parameters of this layer.  Each value is a tuple corresponding
                to the mean and (unscaled, see `scale_tform` usage) standard deviation
                parameters (e.g., samplers_init["weight_mean"].rsample() should
                return a shape (out_features, in_features) tensor defining
                initial weight parameters).  Keys should match those in `samplers`.
            resample_mean: Flag indicating parameter means should be resampled using
                appropriate samplers from `samplers_init` before training. This can
                be set to False to initialize mean values to parameter values of the
                input module (e.g., if pretrained).
            priors: Dictionary of prior distributions over layer parameters.
                These distributions should be initialized (like `samplers_init`)
                and have a .rsample() method returning a tensor of the same
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

        # Define mean and unscaled standard deviation parameters.
        module_params = [p for p in module.named_parameters()]
        _module_params = torch.nn.ParameterDict()
        self._learn_var = learn_var
        if (len(module_params) > 0) and learn_var:
            for key, _ in module_params:
                _module_params[key + "_mean"] = getattr(module, key)
                _module_params[key + "_rho"] = torch.nn.Parameter(
                    torch.empty_like(_module_params[key + "_mean"])
                )
        else:
            for key, _ in module_params:
                _module_params[key + "_mean"] = getattr(module, key)
                _module_params[key + "_rho"] = None
        self._module = module
        self._module_params = _module_params

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
        self.resample_mean = resample_mean
        if learn_var:
            self.reset_parameters()

        # Define variational distribution sampler types.
        if samplers is None:
            samplers = {}
            for key, _ in module.named_parameters():
                if self._module_params[key + "_rho"] is None:
                    samplers[key] = None
                else:
                    samplers[key] = dist.Normal
        self._samplers = samplers

        # Set priors.
        if priors is None:
            # Default to same distributions as `samplers_init` mean samplers.
            priors = {
                key: Distribution(samplers_init[key + "_mean"])
                for key in samplers.keys()
            }
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

    def get_named_sampler(self, name: str) -> torch.distributions.Distribution:
        """Initialize and return the requested module parameter sampler."""
        if self._samplers[name] is None:
            return None
        else:
            return self._samplers[name](
                loc=self._module_params[name + "_mean"],
                scale=self.scale_tform(self._module_params[name + "_rho"]),
            )

    @property
    def samplers(self) -> dict:
        """Return initialized variational distribution samplers."""
        # Initialize the samplers on the fly to ensure we're on the same device
        # as self._module_params.
        return {key: self.get_named_sampler(name=key) for key in self._samplers.keys()}

    @property
    def module_map(self) -> torch.nn.Module:
        """Return module instance with parameters set to learned means."""
        return self._module

    @property
    def module_params(self) -> dict:
        """Return a sample of this module's parameters."""
        params = {}
        for key, val in self.samplers.items():
            if val is None:
                # This parameter is not treated as a distribution so we can
                # directly return the mean value.
                params[key] = self._module_params[key + "_mean"]
            else:
                params[key] = val.rsample()

        return params

    @property
    def module(self) -> torch.nn.Module:
        """Prepare a callable that acts like input `module` with random parameters."""
        return functools.partial(
            torch.func.functional_call, self._module, self.module_params
        )

    def __getattr__(self, name: str) -> Any:
        """Custom getattr to return samples of named parameters."""
        # Check for a sampler associated with `name` and return a sample if it exists.
        inst_modules = self.__dict__.get("_modules", {})
        if f"{name}_mean" in inst_modules.get("_module_params", {}).keys():
            return self.get_named_sampler(name=name).sample()

        # Check remaining modules and parameters for requested attribute.
        inst_params = self.__dict__.get("_parameters", {})
        if name in inst_modules.keys():
            return inst_modules[name]
        elif name in inst_params.keys():
            return inst_params[name]

        # Raise attribute error if we reach this point.
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def reset_parameters(self) -> None:
        """Resample layer parameters from initial distributions."""
        for key, param in self._module_params.items():
            if param is not None:
                # If this is a parameter mean, verify self.resample_mean flag
                # before resampling.
                if ("_mean" not in key) or self.resample_mean:
                    param.data = self.samplers_init[key].sample(
                        sample_shape=param.shape
                    )

    def compute_kl_divergence(
        self, priors: Optional[Union[dict, Distribution]] = None, n_samples: int = 1
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
            priors = {key: priors for key in self._samplers.keys()}

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
                        self.get_named_sampler(name=param_name),
                        param_prior.distribution,
                    ).sum()
                )
            except NotImplementedError:
                # Compute Monte Carlo KL divergence.
                kl_divergence.append(
                    kl_divergence_sampled(
                        dist0=self.get_named_sampler(name=param_name),
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


class BroadcastModule(torch.nn.Module):
    """BroadcastModule for compatibility with other Bayesian layers."""

    def __init__(
        self,
        module: torch.nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """BroadcastModule initializer.

        Args:
            module: Module that will be applied in forward pass.  When the
                output of the previous layer is a MuVar type, module is
                simply broadcast across both elements of MuVar.
        """
        super().__init__()
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
            if input[1] is None:
                # No input variance so we only need to operate on mean.
                out = MuVar(self.module(input[0]), None)
            else:
                out = MuVar(self.module(input[0]), self.module(input[1]))
        else:
            # Compute forward pass with a random sample of parameters.
            out = self.module(input)

        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from inference import Linear

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
        moment_propagator=inference.MonteCarlo(n_samples=n_samples),
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
