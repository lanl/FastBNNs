"""Definitions of prior distributions over neural network parameters."""

from collections.abc import Callable, Iterable

import numpy as np
import torch
import torch.distributions as dist


class Prior(torch.nn.Module):
    def __init__(self):
        """Initializer for Prior base class."""
        super().__init__()

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """Compute the log PDF of the prior at points `x`."""
        pass

    def sample(self, sample_shape: Iterable) -> torch.tensor:
        """Generate samples from the prior of size `sample_shape`."""
        pass


class SpikeSlab(Prior):
    """Spike-slab Gaussian Mixture Model prior."""

    def __init__(
        self,
        mu: torch.tensor = torch.tensor([0.0, 0.0]),
        sigma: torch.tensor = torch.tensor([0.1, 1.0]),
        probs: torch.tensor = torch.tensor([0.5, 0.5]),
    ):
        super().__init__()
        loc = torch.nn.Parameter(mu, requires_grad=False)
        scale = torch.nn.Parameter(sigma, requires_grad=False)
        mixture_distribution = dist.Categorical(probs=probs)
        self.distribution = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=dist.Normal(loc=loc, scale=scale),
        )

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """Compute the log PDF of the prior at points `x`."""
        return self.distribution.log_prob(x)

    def sample(self, sample_shape: Iterable = (1,)) -> torch.tensor:
        """Generate samples from the prior of size `sample_shape`."""
        return self.distribution.sample(sample_shape=sample_shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example using an IsotropicNormal prior.
    prior = SpikeSlab()
    sample = prior.sample(sample_shape=(100, 1))
    x = torch.linspace(-5.0, 5.0, 1000)
    pdf = torch.exp(prior.log_prob(x))
    fig, ax = plt.subplots()
    ax.hist(sample, density=True)
    ax.plot(x, pdf)
    plt.show()
