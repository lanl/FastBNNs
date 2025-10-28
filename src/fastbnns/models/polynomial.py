"""Torch module for basic polynomials."""

from typing import Union

import torch

from fastbnns.bnn import types


class PolyModule(torch.nn.Module):
    """Polynomial model."""

    def __init__(self, poly_order: int = 1) -> None:
        """Initialize module."""
        super().__init__()
        self.poly_order = poly_order
        self.coeffs = torch.nn.Parameter(
            torch.randn(poly_order + 1, dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, input: Union[torch.Tensor, types.MuVar]) -> torch.Tensor:
        """Forward pass through module."""
        return torch.stack([c * (input**n) for n, c in enumerate(self.coeffs)]).sum(
            dim=0
        )
