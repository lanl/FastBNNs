"""Generic dataset for simulated data."""

from collections.abc import Callable
from typing import Optional

import torch

from simulation import generators, observation


class SimulatedData(torch.utils.data.Dataset):
    """Dataset for stochastic simulators with fixed input arguments."""

    def __init__(
        self,
        data_generator: Callable,
        dataset_length: int = 1,
        cache: bool = False,
        transform: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initializer for stochastic simulator dataset.

        Args:
            data_generator:  Callable that can be called to generate data as data = data_generator()
            dataset_length: Desired dataset length (set manually since `simulator` is stochastic).
            cache: Flag indicating data should be cached in memory (as opposed to sampling new
                data every __getitem__()).
            transform: Transform applied to simulated data before returning.
        """
        super().__init__()
        self.data_generator = data_generator
        self._len = dataset_length
        self.cache = cache
        self.transform = transform
        self.data = [None for _ in range(dataset_length)]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        if self.cache and self.data[idx] is not None:
            return self.data[idx]
        else:
            data_dict = self.data_generator()
            if self.transform is not None:
                data_dict["output"] = self.transform(data_dict["output"])
            if self.cache:
                self.data[idx] = data_dict

        return data_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from simulation import generators, images, observation

    # Example dataset: simulated noisy images.
    data_generator = generators.Generator(
        simulator=images.gaussian_blobs,
        simulator_kwargs={
            "sigma": np.array([[1.0, 1.0]]),
            "amplitude": 100.0,
            "im_size": (8, 8),
        },
        simulator_kwargs_generator={
            "mu": lambda: 8 * (np.random.random(size=(1, 2)) - 0.5),
        },
    )
    simulator_kwargs = {"mu": np.array([[]])}
    noise_tform = observation.NoiseTransform(
        noise_fxn=observation.add_read_noise,
        noise_fxn_kwargs={"sigma": 0.1},
    )
    n_data = 3
    ds = SimulatedData(
        data_generator=data_generator, dataset_length=n_data, transform=noise_tform
    )
    for n in range(n_data):
        data = ds[n]
        fig, ax = plt.subplots()
        plt.imshow(
            data["output"],
        )
        plt.show()
