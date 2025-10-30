"""Test functionality of custom activations."""

import torch

from fastbnns.models.activations import InverseTransformSampling, scaled_sigmoid


def test_scaled_sigmoid() -> None:
    """Test functionality of scaled_sigmoid."""
    # Test some values to make sure the right function is applied.
    x = torch.tensor([0.0, 1.0, 2.0])
    alpha = torch.tensor([1.0, 2.0, 3.0])
    y_expected = torch.tensor([0.5000, 0.8808, 0.9975])
    y = scaled_sigmoid(x=x, alpha=alpha)
    assert (torch.abs(y_expected - y) < 1.0e-3).all(), (
        "Output of `scaled_sigmoid` not matching expected results!"
    )


def test_inverse_tform_activation() -> None:
    "Test basic functionality of InverseTransformSampling custom activation."
    # Test basic usage of activation.
    activation = InverseTransformSampling(
        alpha_init=torch.tensor([1.0]),
        distribution=torch.distributions.Normal(loc=0.0, scale=1.0),
    )
    x = torch.tensor([0.0000, 0.4001, 1.2276])
    y_expected = torch.tensor([0.0000, 0.2500, 0.7501])
    y = activation(x)
    assert (torch.abs(y_expected - y) < 1.0e-3).all(), (
        "Output of `InverseTransformSampling` forward pass not matching expected results!"
    )
