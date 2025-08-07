"""Test custom data types."""

import torch

import bnn.types


def test_types() -> None:
    """Test custom data types."""
    # Verify that desired operations run on bnn.types.MuVar without error.
    a = bnn.types.MuVar(torch.randn((2, 2)), torch.ones((2, 2)))
    b = bnn.types.MuVar(torch.randn((2, 2)), 1.23 * torch.ones((2, 2)))
    a + b
    a - b
    a**3
    a.shape
    a.size(1)
    assert isinstance(
        a.to("cpu"), bnn.types.MuVar
    ), "MuVar.to() is not returning data as type MuVar!"
    assert isinstance(
        a.detach(), bnn.types.MuVar
    ), "MuVar.detach() is not returning data as type MuVar!"
    a.numel()
    a.sum()
    a @ b
    torch.cat([a, b], dim=-1)
    torch.nn.functional.pad(a, [0, 1, 2, 0])

    # Verify the math of some basic operations.
    ab_sum = a + b
    tol = 1.0e-4
    assert (
        (ab_sum[0] - (a[0] + b[0])).abs() < tol
    ).all(), "Addition on type `MuVar` not producing expected mean!"
    assert (
        (ab_sum[1] - (a[1] + b[1])).abs() < tol
    ).all(), "Addition on type `MuVar` not producing expected variance!"
    ab_mul = a @ b
    assert (
        (ab_mul[0] - (a[0] @ b[0])).abs() < tol
    ).all(), "Matrix multiplication on type `MuVar` not producing expected mean!"
    assert (
        (ab_mul[1] - (a[1] @ b[1] + (a[0] ** 2) @ b[1] + a[1] @ (b[0] ** 2))).abs()
        < tol
    ).all(), "Matrix multiplication on type `MuVar` not producing expected mean!"
