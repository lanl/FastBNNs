"""Test prior distributions."""

import torch

from fastbnns.bnn.priors import Distribution, SpikeSlab


def test_priors() -> None:
    """Test custom prior distributions and types."""
    # Test distribution wrapper.
    loc = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    dist = Distribution(distribution=torch.distributions.Normal(loc=loc, scale=scale))
    dist.sample()
    assert dist.log_prob(x=loc) == torch.log(
        1.0 / torch.sqrt(2.0 * torch.pi * (scale**2))
    ), """log_prob(x=loc) not returning expected result!"""

    # Test the SpikeSlab prior.
    dist = SpikeSlab(
        loc=torch.tensor([0.0, 0.0]),
        scale=torch.tensor([0.1, 1.0]),
        probs=torch.tensor([0.5, 0.5]),
    )
    dist.sample()
    dist.log_prob(x=torch.tensor(0.0))
