"""Custom activation functions."""

import torch
from torch.distributions import Distribution


def scaled_sigmoid(
    x: torch.Tensor, alpha: torch.Tensor = torch.tensor(1.0)
) -> torch.Tensor:
    """Compute the scaled sigmoid function \frac{1.0}{1.0+\exp(-\alpha*x)}"""
    return 1.0 / (1.0 + torch.exp(-alpha * x))


class InverseTransformSampling(torch.nn.Module):
    """Activation to mimic inverse transform sampling from some distribution."""

    def __init__(
        self,
        distribution: Distribution = torch.distributions.Normal(loc=0.0, scale=1.0),
        learn_alpha: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Initialize InverseTransformSampling class.

        Args:
            distribution: Torch distribution with a defined .icdf() method.
            learn_alpha: Flag indicating we should learn the alpha scale in
                the domain transform f(x) = 1.0 / (1.0 + exp(-alpha*x)),
                otherwise alpha=1.0 will always be used.
        """
        super().__init__(*args, **kwargs)

        # Define the domain transform to convert inputs in
        # (-\inf, \inf) to [0, 1]
        self._alpha = torch.nn.Parameter(
            torch.tensor([0.5413]), requires_grad=learn_alpha
        )  # self.alpha=softplus(self._alpha)
        self.domain_tform = scaled_sigmoid

        # Define the Normal distribution of interest.
        self.distribution = distribution

    @property
    def alpha(self) -> torch.Tensor:
        """Scale self._alpha to ensure positivity and return."""
        return torch.nn.functional.softplus(self._alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through activation."""
        # Transform inputs from (-\inf, \inf) to [0, 1]
        x_prime = self.domain_tform(x, alpha=self.alpha)

        # Treat transformed inputs as samples from U[0, 1] and pass through
        # inverse CDF of Normal.
        return self.distribution.icdf(x_prime)
