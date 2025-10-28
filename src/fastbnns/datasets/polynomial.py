"""Dataset for simulated polynomial data."""

from collections.abc import Callable
from typing import Optional

import torch


class Polynomial(torch.utils.data.Dataset):
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
            transform: Transform applied to simulated data before returning.
        """
        super().__init__()
        self.data_generator = data_generator
        self._len = dataset_length
        self.transform = transform
        self.cache = cache
        self.data = [None for _ in range(dataset_length)]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple:
        if self.cache and self.data[idx] is not None:
            return self.data[idx]
        else:
            data_dict = self.data_generator()
            if self.transform is not None:
                data_dict["output"] = self.transform(data_dict["output"])
            if self.cache:
                self.data[idx] = (data_dict["input"]["x"], data_dict["output"])

        return data_dict["input"]["x"], data_dict["output"]
