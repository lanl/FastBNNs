"""Definitions of prior distributions over neural network parameters."""

from collections.abc import Callable, Iterable

import numpy as np
import torch


class Prior(torch.nn.Module):
    def __init__(self, generator: Callable, generator_kwargs: dict = {}):
        """Initializer for Prior base class

        Args:
            generator: Random number generator that samples from the desired prior.
            generator_kwargs: Keyword arguments passed to the generators call.
        """
        super().__init__()

        self.generator = generator
        self.generator_kwargs = generator_kwargs

    def forward(self, x: torch.tensor):
        """Compute the PDF of the prior at points `x`."""
        pass

    def sample(self, size: Iterable):
        """Generate samples from the prior of size `size`."""
        pass


class IsotropicNormal(Prior):
    def __init__(
        self,
        mu: torch.tensor = torch.tensor(0.0),
        sigma: torch.tensor = torch.tensor(1.0),
    ):
        super().__init__(generator=lambda size: mu + sigma * torch.randn(size=size))
        self.mu = torch.nn.Parameter(mu, requires_grad=False)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=False)

    def forward(self, x: torch.tensor):
        """Compute the Normal PDF at points `x`."""
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (
            np.sqrt(2.0 * torch.pi) * self.sigma
        )

    def sample(self, size: Iterable = [1]):
        """Sample from the Normal prior."""
        return self.generator(size)


if __name__ == "__main__":
    # Example using an IsotropicNormal prior.
    prior = IsotropicNormal()
    sample = prior.sample(size=(100, 10))
