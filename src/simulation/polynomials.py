"""Functionality for simulating polynomial data."""

from collections.abc import Iterable
from typing import Union

import torch


def polynomial(
    x: Union[float, torch.tensor], order: int = 1, coefficients: Iterable = [0.0, 1.0]
) -> Union[float, torch.tensor]:
    """Basic polynomial.

    Args:
        x: Points at which to evaluate the polynomial.
        order: Order of the polynomial.
        coefficients: Polynomial coefficients in ascending order.
    """
    return torch.stack(
        [c * (x**n) for c, n in zip(coefficients, range(order + 1))]
    ).sum(dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.linspace(-1.0, 1.0, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, polynomial(x))
    plt.show()
