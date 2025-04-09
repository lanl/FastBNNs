"""Functionality for simulating image data."""

from collections.abc import Iterable
from typing import Union

import numpy as np
import scipy.stats


def gaussian_blobs(
    mu: np.array,
    sigma: np.array,
    amplitude: Union[float, np.array],
    im_size: Iterable,
) -> np.array:
    """Noise-free image of a (possibly unnormalized) 2D isotropic Gaussian mixture model.

    Generate a noise-free image of the sum of 2D isotropic Gaussians in the domain
    x in [-(im_size[1]-1)/2, (im_size[1]+1)/2] and y in [-(im_size[0]-1)/2, (im_size[0]+1)/2].

    Args:
        mu: Locations of individual Gaussians. [n_gaussians, y, x]
        sigma: Standard deviation(s) of the len(mu) Gaussians. [1 or n_gaussians, y, x]
        amplitude: Amplitude(s) of the len(mu) Gaussians.  [1 or n_gaussians]
        im_size: Size of the output image in pixels.  [y, x]
    """
    # Reshape inputs.
    if len(sigma) != len(mu):
        sigma = sigma * np.ones_like(mu)
    if isinstance(amplitude, float):
        amplitude = amplitude * np.ones(mu.shape[0])

    # Compute output image.
    y = np.arange(im_size[0]) - im_size[0] / 2 + 0.5
    x = np.arange(im_size[1]) - im_size[1] / 2 + 0.5
    out = np.zeros(im_size)
    for n in range(len(mu)):
        out += (
            2.0
            * np.pi
            * np.prod(sigma[n])
            * amplitude[n]
            * (
                scipy.stats.norm.pdf(y, loc=mu[n][0], scale=sigma[n][0])[:, None]
                @ scipy.stats.norm.pdf(x, loc=mu[n][1], scale=sigma[n][1])[None, :]
            )
        )

    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    im = gaussian_blobs(
        mu=np.array([[0.0, 0.0], [-1.0, 3.0]]),
        sigma=np.array([[0.7, 1.0], [0.5, 0.5]]),
        amplitude=np.array([1.0, 0.7]),
        im_size=(8, 8),
    )
    fig, ax = plt.subplots()
    plt.imshow(im, cmap="gray", extent=([-3.5, 3.5, 3.5, -3.5]))
    plt.show()

    im = gaussian_blobs(
        mu=np.array([[0.0, 0.0]]),
        sigma=np.array([[1.0, 1.0]]),
        amplitude=np.array([1.0]),
        im_size=(8, 8),
    )
    print(im.max())
