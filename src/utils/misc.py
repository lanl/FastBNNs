"""Miscellaneous utiliy functions useful throughout repository."""

from collections.abc import Callable
from typing import Union

import inspect
import re

import torch
import torch.nn.functional as F


def set_requires_grad(module: torch.nn.Module, requires_grad: bool, tag: str) -> None:
    """Set requires_grad property of all parameters whose name contains `tag`."""
    for param_name, param_value in module.named_parameters():
        if tag in param_name:
            param_value.requires_grad = requires_grad


def get_torch_functional(torch_module_class: type) -> Union[Callable, None]:
    """Search for a torch.nn.functional call in module.forward().

    Args:
        torch_module_class: Class of a PyTorch module that we wish to find a
            functional for in torch.nn.functional.
            For example, if torch_module_class = torch.nn.Linear,
            get_torch_functional(torch_module_class) returns torch.nn.functional.linear
    """
    # Search for any functional call that starts with F.
    # (import torch.nn.functional as F is used in PyTorch source), starts with
    # any lowercase character, is followed by any number of characters, and
    # finally is terminated by an opening parenthesis to start the functional call.
    forward = inspect.getsource(torch_module_class.forward)
    re_matches = re.search("F.(?P<functional>[a-z].+)\(", forward)
    if re_matches is None:
        return None
    else:
        functional_name = re_matches["functional"]
        return getattr(F, functional_name)
