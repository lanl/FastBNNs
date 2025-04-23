"""Simple MLP PyTorch models."""

from typing import Union

import torch


class CNN(torch.nn.Module):
    """Basic multi-layer CNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_features: int = 8,
        n_hidden_layers: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = "same",
        activation: type = torch.nn.LeakyReLU,
    ):
        super().__init__()

        modules = [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        ]
        if activation is not None:
            modules.append(activation())
        for _ in range(n_hidden_layers):
            modules.append(
                torch.nn.Conv2d(
                    in_channels=hidden_features,
                    out_channels=hidden_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if activation is not None:
                modules.append(activation())
        modules.append(
            torch.nn.Conv2d(
                in_channels=hidden_features,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.module_list:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Basic MLP usage example.
    in_channels = 1
    out_channels = 1
    network = CNN(in_channels=in_channels, out_channels=out_channels)
    batch_size = 8
    out = network(torch.ones((batch_size, in_channels, 128, 128)))
