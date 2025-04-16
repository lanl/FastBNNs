"""Definitions of prior distributions over neural network parameters."""

from collections.abc import Iterable

import torch
import torch.distributions as dist


class SpikeSlab(dist.Distribution):
    """Spike-slab Gaussian Mixture Model prior."""

    def __init__(
        self,
        loc: torch.tensor = torch.tensor([0.0, 0.0]),
        scale: torch.tensor = torch.tensor([0.1, 1.0]),
        probs: torch.tensor = torch.tensor([0.5, 0.5]),
    ):
        super().__init__()
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
