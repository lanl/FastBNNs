"""Functionality for simulating polynomial data."""

from collections.abc import Iterable

import torch


def polynomial(x: torch.tensor, coefficients: Iterable = [0.0, 1.0]) -> torch.tensor:
    """Basic polynomial.

    Args:
        x: Points at which to evaluate the polynomial.
        coefficients: Polynomial coefficients in ascending order.
    """
    return torch.stack([c * (x**n) for n, c in enumerate(coefficients)]).sum(dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.linspace(-1.0, 1.0, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, polynomial(x))
    plt.show()
