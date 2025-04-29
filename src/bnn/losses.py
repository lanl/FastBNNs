"""Losses and helpers useful for Bayesian neural network training/evaluation."""

from typing import Union

import torch
import torch.distributions as dist
from torch.nn.modules.loss import _Loss

from bnn import priors


def kl_divergence_sampled(
    dist0: dist.Distribution, dist1: dist.Distribution, n_samples: int = 1
):
    """KL divergence KL(dist0||dist1) approximated by Monte Carlo sampling."""
    kl_divergence = []
    for _ in range(n_samples):
        sample = dist0.sample()
        kl_divergence.append(dist0.log_prob(sample) - dist1.log_prob(sample))

    return torch.stack(kl_divergence).mean(dim=0)


class KLDivergence(_Loss):
    """KL divergence loss for Bayesian neural networks."""

    def __init__(self, prior: Union[dict, priors.Distribution] = None):
        """Initialize KL divergence loss.
        Args:
            prior: Prior distribution over parameters.  This can be a single
                distribution for all parameters or a dictionary of dictionaries whose
                primary keys correspond to named modules and whose secondary keys
                correspond to parameters of that module.  The list indices
                correspond to model.named_modules(). By default, None
                will use priors set within each Bayesian layer on
                initialization.
        """
        super().__init__()
        self.prior = prior

    def forward(
        self,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute KL divergence for Bayesian sub-modules of `model`.

        Args:
            model: torch.nn.Module that may have some Bayesian layers
                as sub-modules, for which we'll compute the KL divergence w.r.t
                their prior.
        """
        kl = []
        for module in model.named_modules():
            module_parameters = [p for p in module[1].parameters()]
            if (
                hasattr(module[1], "compute_kl_divergence")
                and (len(module_parameters) > 0)
                and module[1].learn_var
            ):
                if isinstance(self.prior, dict):
                    # Pass the input prior dictionary for this module.
                    kl.append(
                        module[1].compute_kl_divergence(priors=self.prior[module[0]])
                    )
                else:
                    kl.append(module[1].compute_kl_divergence(priors=self.prior))

        return torch.stack(kl).sum()


class ELBO(_Loss):
    """Evidence lower bound with scaled KL."""

    def __init__(
        self,
        neg_log_likelihood: _Loss = None,
        kl_divergence: _Loss = KLDivergence(),
        beta: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        """Initialize ELBO loss.

        Args:
            neg_log_likelihood: Initialized neg_log_likelihood loss (e.g.,
                torch.nn.GaussianNLLLoss()).  This will be called in the forward
                pass of this loss as neg_log_likelihood(**kwargs) where **kwargs are
                the keyword arguments passed as ELBO()(**kwargs).
            kl_divergence: Initialized kl_divergence loss whose forward pass
                takes a torch.nn.Module `model` as input and returns a tensor
                corresponding to the KL divergence between parameters of `model`
                and their prior distribution.
            beta: Scaling parameter for KL loss term in the ELBO.
        """
        super().__init__(reduction=reduction)

        # Set default log likelihood calculator.
        if neg_log_likelihood is None:
            neg_log_likelihood = torch.nn.GaussianNLLLoss(
                full=True, reduction=reduction
            )
        self.neg_log_likelihood = neg_log_likelihood

        self.kl_divergence = kl_divergence
        self.beta = beta

    def forward(self, model: torch.nn.Module = None, **kwargs) -> torch.Tensor:
        """Compute the ELBO loss.

        Args:
            model: torch.nn.Module that may have some layers.BayesianLayers
                as sub-modules, for which we'll compute the KL divergence w.r.t
                their prior.  Passing None is treated as no model, i.e., KL = 0.0
            kwargs: Keyword arguments to pass to self.log_likelihood(**kwargs)
        """
        if (model is None) or (self.beta == 0.0) or (self.kl_divergence is None):
            return self.neg_log_likelihood(**kwargs)
        else:
            return self.neg_log_likelihood(**kwargs) + self.beta * self.kl_divergence(
                model
            )
