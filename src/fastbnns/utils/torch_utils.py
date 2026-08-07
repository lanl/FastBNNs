"""Miscellaneous utility functions useful throughout repository."""

import torch


def set_requires_grad_(module: torch.nn.Module, requires_grad: bool, tag: str) -> None:
    """Set requires_grad property of all parameters whose name contains `tag`."""
    for param_name, param_value in module.named_parameters():
        if tag in param_name:
            param_value.requires_grad = requires_grad
