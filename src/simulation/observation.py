"""Functionality for simulating observations of random variables."""

from collections.abc import Callable
from typing import Union

import numpy as np
import torch


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
    return add_read_noise(signal=np.random.poisson(signal), sigma=sigma)


class NoiseTransform(torch.nn.Module):
    """Wrapper to facilitate using noise functions with torch transform functionality."""

    def __init__(
        self,
        noise_fxn: Callable,
        noise_fxn_kwargs: dict = {},
        noise_fxn_kwargs_generator: dict = {},
    ) -> None:
        """Initializer for stochastic simulator dataset.

        Args:
            noise_fxn: Callable that noises an input signal.
            noise_fxn_kwargs: Keyword arguments passed to noise_fxn as
                noise_fxn(x, **noise_fxn_kwargs)
            noise_fxn_kwargs_generator: Keyword argument generator whose values
                can be called on the forward(x) pass to generate x-dependent
                kwargs that override noise_fxn_kwargs.  For example,
                if noise_fxn accepts an input argument `sigma` that can be a
                len(x) array, we can generate heteroscedastic (x-dependent)
                noise defined by `sigma` as
                noise_fxn_kwargs_generator={"sigma": lambda x: 0.1 * x**2}
        """
        super().__init__()
        self.noise_fxn = noise_fxn
        self.noise_fxn_kwargs = noise_fxn_kwargs
        self.noise_fxn_kwargs_generator = noise_fxn_kwargs_generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate noisy `x`."""
        # Generate x-dependent arguments and merge with noise_fxn_kwargs.
        generated_kwargs = {
            key: value_gen(x)
            for key, value_gen in self.noise_fxn_kwargs_generator.items()
        }
        noise_fxn_kwargs = self.noise_fxn_kwargs | generated_kwargs
        return self.noise_fxn(x, **noise_fxn_kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import polynomials

    x = torch.linspace(-1.0, 1.0, 100)
    signal = polynomials.polynomial(x, order=1, coefficients=[0.0, 1.0])

    # Homoscedastic noise:
    noise_tform = NoiseTransform(
        noise_fxn=add_read_noise, noise_fxn_kwargs={"sigma": 0.1}
    )
    fig, ax = plt.subplots()
    ax.plot(x, signal, color="r", linewidth=2, label="clean signal")
    for _ in range(5):
        ax.plot(
            x,
            noise_tform(x),
            color="k",
            alpha=0.2,
        )
    ax.plot([], color="k", alpha=0.2, label="noisy realizations")
    ax.legend()
    plt.show()

    # Heteroscedastic noise generated from input x:
    noise_tform = NoiseTransform(
        noise_fxn=add_read_noise,
        noise_fxn_kwargs_generator={
            "sigma": lambda x: 0.1 + 0.1 * (1.0 + np.sin(2.0 * np.pi * x))
        },
    )
    fig, ax = plt.subplots()
    ax.plot(x, signal, color="r", linewidth=2, label="clean signal")
    for _ in range(5):
        ax.plot(
            x,
            noise_tform(x),
            color="k",
            alpha=0.2,
        )
    ax.plot([], color="k", alpha=0.2, label="noisy realizations")
    ax.legend()
    plt.show()
