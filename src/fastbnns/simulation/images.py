"""Functionality for simulating image data."""

from collections.abc import Iterable
import math
from typing import Union

import scipy.stats
import torch


def gaussian_blobs(
    mu: torch.tensor,
    sigma: torch.tensor,
    amplitude: Union[float, torch.tensor],
    im_size: Iterable,
) -> torch.tensor:
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
        sigma = sigma * torch.ones_like(mu)
    if isinstance(amplitude, float):
        amplitude = amplitude * torch.ones(mu.shape[0])

    # Compute output image.
    y = torch.arange(im_size[0]) - im_size[0] / 2 + 0.5
    x = torch.arange(im_size[1]) - im_size[1] / 2 + 0.5
    out = torch.zeros((1, *im_size), dtype=torch.float32)
    for n in range(len(mu)):
        out += (
            2.0
            * torch.pi
            * math.prod(sigma[n])
            * amplitude[n]
            * (
                scipy.stats.norm.pdf(y, loc=mu[n][0], scale=sigma[n][0])[:, None]
                @ scipy.stats.norm.pdf(x, loc=mu[n][1], scale=sigma[n][1])[None, :]
            )
        )

    return out


class GridSamples:
    def __init__(self, n_per_pixel: int = 10, im_size: Iterable = (8, 8)) -> None:
        """Initialize stateful location sampler"""
        self.n_per_pixel = n_per_pixel
        grid = torch.meshgrid(
            torch.arange(-im_size[0] // 2, im_size[0] // 2) + 0.5,
            torch.arange(-im_size[0] // 2, im_size[0] // 2) + 0.5,
            indexing="ij",
        )
        self.xy = torch.stack((grid[0].flatten(), grid[1].flatten())).T.repeat(
            (n_per_pixel, 1)
        )
        self._counter = 0

    def __len__(self) -> int:
        """Return sampler length."""
        return len(self.xy)

    def __call__(self) -> torch.tensor:
        """Return xy location."""
        xy = self.xy[self._counter % len(self)][None, :]
        self._counter += 1
        return xy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    im = gaussian_blobs(
        mu=torch.tensor([[0.0, 0.0], [-1.0, 3.0]]),
        sigma=torch.tensor([[0.7, 1.0], [0.5, 0.5]]),
        amplitude=torch.tensor([1.0, 0.7]),
        im_size=(8, 8),
    )
    fig, ax = plt.subplots()
    plt.imshow(im[0], cmap="gray", extent=([-3.5, 3.5, 3.5, -3.5]))
    plt.show()

    im = gaussian_blobs(
        mu=torch.tensor([[0.0, 0.0]]),
        sigma=torch.tensor([[1.0, 1.0]]),
        amplitude=torch.tensor([1.0]),
        im_size=(8, 8),
    )
    print(im.max())
