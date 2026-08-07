"""Bayesian neural network base module(s) and utilities."""

from __future__ import annotations

import copy
from typing import Any, Iterator, Union, TYPE_CHECKING

import torch

from .types import MuVar
from .wrappers import convert_to_bnn_, convert_to_nn


if TYPE_CHECKING:
    # laplace-torch is used for specialized functionality so we don't want to import it for general users.

    import laplace


def bnn_params_from_laplace(laplace_model: laplace.DiagLaplace) -> dict:
    """Create dictionary of parameters for a BNN from a diagonal Laplace approximation.

    Args:
        laplace_model: Diagonal Laplace approximation instance whose parameters
            will be reorganized for ingestion into a BNN instance.
    """

    # Define an inverse scale transform to convert scale parameters
    # (st. dev. from Laplace approximation) to `rho` parameters learned by the BNN.
    def inv_scale_tform(scale: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(scale) - 1.0)

    # Remap LA parameters to BNN parameters
    # (laplace_model.params shares ordering of model.named_parameters()).
    param_dict = {}
    var_ind = 0  # pointer to track start of variances for each parameter
    for n, param in enumerate(laplace_model.model.named_parameters()):
        name_split = param[0].split(".")
        base_name = f"{'.'.join(name_split[:-1])}._module_params.{name_split[-1]}"
        param_dict[base_name + "_mean"] = laplace_model.params[n]
        param_dict[base_name + "_rho"] = inv_scale_tform(
            laplace_model.posterior_scale[
                var_ind : (var_ind + laplace_model.params[n].numel())
            ]
        ).reshape(laplace_model.params[n].shape)
        var_ind += laplace_model.params[n].numel()

    return param_dict


class BNN(torch.nn.Module):
    """Bayesian neural network base class."""

    def __init__(
        self,
        nn: Union[torch.nn.Module, laplace.DiagLaplace],
        convert_in_place: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize Bayesian neural network.


        WARNINGS:
            (1): Some functionality of this class relies on parameter names
                containing the suffixes "_mean" and "_rho".  If the input `nn`
                has parameters containing these strings, this class may not
                behave as expected!
            (2): The forward pass of `nn` is assumed to accept a single tensor
                representing the input.  The conversion to a BNN will hijack
                the forward pass through `nn` by changing the type of this
                input tensor to bnn.types.MuVar.

        Args:
            nn: PyTorch module to be converted to its Bayesian counterpart.
                Alternatively, this can be a laplace.DiagLaplace instance, in
                which case we will remap the diagonal Laplace approximation
                parameters therein for compatibility with this class.
            convert_in_place: Flag indicating input `nn` should be converted to
                a BNN in place.
            args, kwargs: Passed as
                bnn.wrappers.convert_to_bnn_(model=nn, *args, **kwargs)
        """
        super().__init__()

        # Convert the neural network to a Bayesian neural network.
        if isinstance(nn, torch.nn.Module):
            bnn = nn if convert_in_place else copy.deepcopy(nn)
            convert_to_bnn_(model=bnn, *args, **kwargs)
        elif isinstance(nn, laplace.DiagLaplace):
            # Convert nn.model to a BNN.
            bnn = nn.model if convert_in_place else copy.deepcopy(nn.model)
            convert_to_bnn_(model=bnn, *args, **kwargs)

            # Update relevant parameters from Laplace approximation.
            param_dict = bnn_params_from_laplace(laplace_model=nn)
            bnn.load_state_dict(param_dict, strict=False)
        else:
            raise (TypeError(f"Unknown network type {type(model)}"))
        self.bnn = bnn

    def named_parameters_tagged(self, tag: str) -> Iterator:
        """Return named parameters whose name contains `tag`."""
        for name, param in self.named_parameters():
            if tag in name:
                yield name, param

    def laplace_init(
        self,
        dataloader: torch.utils.data.DataLoader,
        laplace_kwargs: dict = {},
        laplace_prior_opt_kwargs: dict = {},
    ) -> None:
        """Initialize model parameter variances to Laplace approximated values."""
        # Import laplace-torch and let user know if it needs to be installed.
        try:
            import laplace
        except ImportError as e:
            raise ImportError(
                "Initialization from the Laplace approximation requires laplace-torch https://pypi.org/project/laplace-torch/."
            ) from e

        # Define default arguments for DiagLaplace.
        laplace_kwargs_default = {
            "likelihood": "regression",
            "prior_precision": torch.inf,
        }
        laplace_kwargs = laplace_kwargs_default | laplace_kwargs

        # Compute Laplace approximation.
        model = convert_to_nn(self.bnn)
        la = laplace.DiagLaplace(model, **laplace_kwargs)
        la.fit(dataloader)

        # Optimize prior precision for LA.
        laplace_prior_opt_kwargs_default = {"pred_type": "nn", "link_approx": "mc"}
        laplace_prior_opt_kwargs = (
            laplace_prior_opt_kwargs_default | laplace_prior_opt_kwargs
        )
        la.optimize_prior_precision(**laplace_prior_opt_kwargs)

        # Load LA parameters into model.
        param_dict = bnn_params_from_laplace(laplace_model=la)
        self.bnn.load_state_dict(param_dict, strict=False)

    def forward(self, input: Union[MuVar, torch.Tensor], *args, **kwargs) -> Any:
        """Forward pass through BNN."""
        return self.bnn(input, *args, **kwargs)


if __name__ == "__main__":
    from models import mlp

    # Convert a model to a Bayesian counterpart.
    in_features = 3
    out_features = 1
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        n_hidden_layers=3,
        activation=torch.nn.LeakyReLU,
    )
    convert_to_bnn_(model=model)
    out = model(torch.ones(1, in_features))

    # Create a BNN wrapper for our model.
    model = mlp.MLP(
        in_features=in_features,
        out_features=out_features,
        n_hidden_layers=3,
        activation=torch.nn.LeakyReLU,
    )
    bnn = BNN(nn=model)
    bnn(torch.randn((1, 3)))
