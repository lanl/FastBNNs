"""Functionality for simulating polynomial data."""

from collections.abc import Iterable
from typing import Union

import numpy as np


def polynomial(
    x: Union[float, np.array], order: int = 1, coefficients: Iterable = [0.0, 1.0]
) -> Union[float, np.array]:
    """Basic polynomial.

    Args:
        x: Points at which to evaluate the polynomial.
        order: Order of the polynomial.
        coefficients: Polynomial coefficients in ascending order.
    """
    return np.sum([c * (x**n) for c, n in zip(coefficients, range(order + 1))], axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, polynomial(x))
    plt.show()
