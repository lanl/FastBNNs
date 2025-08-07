"""Test functionality of utils."""

import torch

from utils.torch_utils import set_requires_grad_


def test_set_requires_grad_() -> None:
    """Test functionality of set_requires_grad_."""
    example_module = torch.nn.ParameterDict(
        {"mean": torch.randn(2), "rho": torch.rand(2)}
    )

    # Set requires_grad to False and verify.
    tag = "rho"
    set_requires_grad_(example_module, requires_grad=False, tag=tag)
    for name, param in example_module.named_parameters():
        if tag in name:
            assert not param.requires_grad
        else:
            # Make sure other parameters weren't affected.
            assert param.requires_grad

    # Set requires_grad to True and verify.
    set_requires_grad_(example_module, requires_grad=True, tag=tag)
    for name, param in example_module.named_parameters():
        assert param.requires_grad
