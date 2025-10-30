"""Test custom Bayesian neural network losses."""

import torch

from fastbnns.bnn.base import BNN
from fastbnns.bnn.losses import ELBO, KLDivergence
from fastbnns.bnn.priors import Distribution, SpikeSlab


def test_losses() -> None:
    """Test basic functionality of the custom BNN losses."""
    # Test KL-divergence loss when closed-form result is available.
    prior = Distribution(
        distribution=torch.distributions.Normal(
            loc=torch.tensor(0.0), scale=torch.tensor(1.0)
        )
    )
    nn = torch.nn.Sequential(
        torch.nn.Linear(1, 1), torch.nn.LeakyReLU(), torch.nn.Linear(1, 1)
    )
    model = BNN(nn=nn)
    loss = KLDivergence(prior=prior)
    loss(model)

    # Test sampled KL-divergence loss.
    prior = SpikeSlab(
        loc=torch.tensor([0.0, 0.0]),
        scale=torch.tensor([0.5, 1.0]),
        probs=torch.tensor([0.5, 0.5]),
    )
    loss = KLDivergence(prior=prior)
    loss(model)

    # Test the ELBO loss.
    loss = ELBO(
        kl_divergence=KLDivergence(prior=prior),
        neg_log_likelihood=torch.nn.GaussianNLLLoss(full=True, reduction="sum"),
    )
    loss(
        model=model,
        input=torch.randn(1),
        target=torch.randn(1),
        var=torch.randn(1).abs(),
    )
