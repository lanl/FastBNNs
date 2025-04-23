"""Custom types and associated functionality."""

from __future__ import annotations
from typing import Union

import torch


class MuVar:
    """Custom tuple-like object holding mean and variance of some distribution."""

    def __init__(
        self,
        mu: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        var: torch.Tensor = None,
    ) -> None:
        """Initialize MuVar instance.

        Args:
            mu: Mean of input distribution, or optionally, a tuple containing
                both mean and variance (to allow calling this method on a tuple
                without unpacking arguments).
        """
        if isinstance(mu, tuple):
            self.mu_var = mu
        else:
            self.mu_var = (mu, var)

    def __repr__(self):
        """Custom display functionality."""
        return f"MuVar({self.mu_var})"

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Access requested index of self.mu_var"""
        return self.mu_var[idx]

    def __add__(self, input: Union[float, torch.Tensor, tuple, list, MuVar]) -> MuVar:
        """Custom add functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Adding a float or tensor is like adding a delta R.V., so
            # variance does not change.
            return MuVar(self.mu_var[0] + input, self.mu_var[1])
        elif isinstance(input, (tuple, list, MuVar)):
            # Means and variances both add (assuming independent distributions).
            return MuVar(self.mu_var[0] + input[0], self.mu_var[1] + input[1])
        else:
            raise NotImplementedError

    def __radd__(self, input: Union[float, torch.Tensor, tuple, list]) -> MuVar:
        """Custom add functionality for MuVar types."""
        return self.__add__(input)

    def __sub__(self, input: Union[float, torch.Tensor, tuple, list, MuVar]) -> MuVar:
        """Custom subtract functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Adding a float or tensor is like adding a delta R.V., so
            # variance does not change.
            return MuVar(self.mu_var[0] - input, self.mu_var[1])
        elif isinstance(input, (tuple, list, MuVar)):
            # Means can be subtracted but variances still add.
            return MuVar(self.mu_var[0] - input[0], self.mu_var[1] + input[1])
        else:
            raise NotImplementedError

    def __rsub__(self, input: Union[float, torch.Tensor, tuple, list]) -> MuVar:
        """Custom subtract functionality for MuVar types."""
        return self.__sub__(input)

    def __mul__(self, input: Union[float, torch.Tensor, tuple, list, MuVar]) -> MuVar:
        """Custom multiply functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Multiplication by scalar: E[aX] = aE[x], V[aX]=a**2 V[X]
            return MuVar(input * self.mu_var[0], (input**2) * self.mu_var[1])
        elif isinstance(input, (tuple, list, MuVar)):
            # Multiplication of two random independent variables:
            #   E[XY] = E[X]E[Y]
            #   V[XY] = (V[X] + E[X]**2)(V[Y] + E[Y]**2) - (E[X]E[Y])**2
            mu = self.mu_var[0] * input[0]
            var = (
                (self.mu_var[0] ** 2 + self.mu_var[1]) * (input[0] ** 2 + input[1])
            ) - (self.mu_var[0] * input[0]) ** 2
            return MuVar(mu, var)
        else:
            raise NotImplementedError

    def __rmul__(self, input: Union[float, torch.Tensor, tuple, list]) -> MuVar:
        """Custom multiply functionality for MuVar types."""
        return self.__mul__(input)

    def __pow__(self, input: int) -> MuVar:
        """Custom exponentiation functionality for MuVar types."""
        if isinstance(input, int):
            out = self
            for _ in range(input - 1):
                out *= self
            return out
        else:
            raise NotImplementedError


if __name__ == "__main__":
    a = MuVar(1.0, 2.0)
    b = MuVar(1.1, 0.5)
    print(a + b)

    print(a + 1.0)
    print(1.0 + a)

    print(a**2)
