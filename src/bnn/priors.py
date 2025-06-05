"""Definitions of prior distributions over neural network parameters."""

from collections.abc import Iterable
import copy
from typing import Any, Optional

import torch
import torch.distributions as dist


class Distribution(torch.nn.Module):
    """Distribution wrapper to facilitate device transfers."""

    def __init__(
        self,
        distribution: Optional[dist.Distribution] = None,
        *args: Any,
        **kwargs: Any
    ):
        """Initialize wrapper."""
        super().__init__()

        # If `distribution` is passed, we'll build the wrapper automatically.
        if distribution is not None:
            self._distribution = copy.deepcopy(distribution)
            for key, val in distribution.__dict__.items():
                if isinstance(val, torch.Tensor):
                    self.register_buffer(key, val)

    @property
    def distribution(self) -> dist.Distribution:
        """Prepare an instance of the distribution."""
        for key, val in self._distribution.__dict__.items():
            if isinstance(val, torch.Tensor):
                buffer_val = getattr(self, key)
                setattr(self._distribution, key, buffer_val)
        return self._distribution

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log PDF of the prior at points `x`."""
        return self.distribution.log_prob(x)

    def sample(self, sample_shape: Iterable = (1,)) -> torch.Tensor:
        """Generate samples from the prior of size `sample_shape`."""
        return self.distribution.sample(sample_shape=sample_shape)


class SpikeSlab(Distribution):
    """Spike-slab Gaussian Mixture Model prior."""

    def __init__(
        self,
        loc: torch.Tensor = torch.tensor([0.0, 0.0]),
        scale: torch.Tensor = torch.tensor([0.1, 1.0]),
        probs: torch.Tensor = torch.tensor([0.5, 0.5]),
    ):
        super().__init__()

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)
        self.register_buffer("probs", probs)

    @property
    def distribution(self) -> dist.Distribution:
        """Prepare an instance of the distribution."""
        mixture_distribution = dist.Categorical(probs=self.probs)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=dist.Normal(loc=self.loc, scale=self.scale),
        )


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
