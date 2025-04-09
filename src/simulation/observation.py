"""Functionality for simulating observations of random variables."""

from typing import Union

import numpy as np


def add_read_noise(
    signal: Union[float, np.array], sigma: Union[float, np.array]
) -> Union[float, np.array]:
    """Noisy realization of `signal` (read noise).

    Args:
        signal: Clean signal to which we add zero-mean Normal read noise.
        sigma: Standard deviation of zero-mean Normally distributed read noise. Can be
            homoscedastic (scalar) or heteroscedastic (array matching len(signal)).
    """
    return signal + sigma * np.random.randn(*signal.shape)


def sensor_noise(
    signal: Union[float, np.array], sigma: Union[float, np.array]
) -> Union[float, np.array]:
    """Noisy realization of `signal` (read noise + shot noise).

    Args:
        signal: Clean signal to which we add read noise and shot noise.
        sigma: Standard deviation of zero mean Normally distributed read noise. Can be
            homoscedastic (scalar) or heteroscedastic (array matching len(signal)).
    """
    return np.random.poisson(add_read_noise(signal=signal, sigma=sigma))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import polynomials

    x = np.linspace(-1.0, 1.0, 100)
    signal = polynomials.polynomial(x, order=1, coefficients=[0.0, 1.0])

    # Homoscedastic noise:
    fig, ax = plt.subplots()
    ax.plot(x, signal, color="r", linewidth=2, label="clean signal")
    for _ in range(5):
        ax.plot(
            x,
            add_read_noise(
                signal=signal,
                sigma=0.1,
            ),
            color="k",
            alpha=0.2,
        )

    ax.plot([], color="k", alpha=0.2, label="noisy realizations")
    ax.legend()
    plt.show()

    # Heteroscedastic noise:
    fig, ax = plt.subplots()
    ax.plot(x, signal, color="r", linewidth=2, label="clean signal")
    for _ in range(5):
        ax.plot(
            x,
            add_read_noise(
                signal=signal,
                sigma=0.1 + 0.1 * (1.0 + np.sin(2.0 * np.pi * x)),
            ),
            color="k",
            alpha=0.2,
        )

    ax.plot([], color="k", alpha=0.2, label="noisy realizations")
    ax.legend()
    plt.show()
