"""Losses and helpers useful for Bayesian neural network training/evaluation."""

import torch
import torch.distributions as dist
from torch.nn.modules.loss import _Loss

from bnn import layers


def kl_divergence_sampled(
    dist0: dist.Distribution, dist1: dist.Distribution, n_samples: int = 1
):
    """KL divergence KL(dist0||dist1) approximated by Monte Carlo sampling."""
    kl_divergence = torch.tensor(0.0)
    for _ in range(n_samples):
        sample = dist0.sample()
        kl_divergence += dist0.log_prob(sample) - dist1.log_prob(sample)

    return kl_divergence / n_samples


class KLDivergence(_Loss):
    """KL divergence loss for Bayesian neural networks."""

    def __init__(self):
        """Initialize KL divergence loss."""
        super().__init__()

    def forward(self, model: torch.nn.Module) -> torch.tensor:
        """Compute KL divergence for Bayesian sub-modules of `model`."""
        kl = torch.tensor(0.0)
        for module in model.named_modules():
            if isinstance(module, layers.BayesianLayer):
                kl += module.compute_kl_divergence()

        return kl


class ELBO(_Loss):
    """Evidence lower bound with scaled KL."""

    def __init__(
        self,
        log_likelihood: _Loss = None,
        kl_divergence: _Loss = KLDivergence(),
        beta: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        """Initialize ELBO loss.

        Args:
            log_likelihood: Initialized log_likelihood loss (e.g.,
                torch.nn.GaussianNLLLoss()).  This will be called in the forward
                pass of this loss as log_likelihood(**kwargs) where **kwargs are
                the keyword arguments passed as ELBO()(**kwargs).
                NOTE: It's best to use reduction="sum" for the log_likelihood
                so that we don't have to rescale the KL term by the batch size
                (see Graves 2011 NeurIPS paper discussion around eqn. 18)
            kl_divergence: Initialized kl_divergence loss whose forward pass
                takes a torch.nn.Module `model` as input and returns a tensor
                corresponding to the KL divergence between parameters of `model`
                and their prior distribution.
            beta: Scaling parameter for KL loss term in the ELBO.
        """
        super().__init__(reduction=reduction)

        # Set default log likelihood calculator.
        if log_likelihood is None:
            log_likelihood = torch.nn.GaussianNLLLoss(full=True, reduction=reduction)
        self.log_likelihood = log_likelihood

        self.kl_divergence = kl_divergence
        self.beta = beta

    def forward(self, model: torch.nn.Module = None, **kwargs) -> torch.tensor:
        """Compute the ELBO loss.

        Args:
            model: torch.nn.Module that may have some layers.BayesianLayers
                as sub-modules, for which we'll compute the KL divergence w.r.t
                their prior.  Passing None is treated as no model, i.e., KL = 0.0
            kwargs: Keyword arguments to pass to self.log_likelihood(**kwargs)
        """
        if model is None:
            return self.log_likelihood(**kwargs)
        else:
            return self.log_likelihood(**kwargs) + self.beta * self.kl_divergence(model)
