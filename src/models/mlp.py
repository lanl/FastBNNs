"""Simple MLP PyTorch models."""

import torch


class MLP(torch.nn.Module):
    """Basic multi-layer perceptron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        n_layers: int = 3,
        activation: type = torch.nn.LeakyReLU,
    ):
        super().__init__()
        modules = [
            torch.nn.Linear(in_features=in_features, out_features=hidden_features),
        ]
        if activation is not None:
            modules.append(activation())
        for _ in range(n_layers - 2):
            modules.append(
                torch.nn.Linear(
                    in_features=hidden_features, out_features=hidden_features
                )
            )
            if activation is not None:
                modules.append(activation())
        modules.append(
            torch.nn.Linear(in_features=hidden_features, out_features=out_features)
        )
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.module_list:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Basic MLP usage example.
    in_features = 1
    out_features = 1
    network = MLP(in_features=in_features, out_features=out_features)
    batch_size = 8
    out = network(torch.ones((batch_size, in_features)))
