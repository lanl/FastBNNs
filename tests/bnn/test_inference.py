"""Test BNN inference modules."""

import torch

import bnn.inference
import bnn.types
import bnn.wrappers
import models.polynomial


def test_inference() -> None:
    """Test inference wrappers with simple example modules."""
    # Test the generic inference modules.
    module = models.polynomial.PolyModule(poly_order=2)
    bayes_module = bnn.wrappers.BayesianModule(module, learn_var=False)
    batch_size = 4
    in_features = 3
    x = bnn.types.MuVar(torch.randn((batch_size, in_features)))
    propagators = [
        bnn.inference.BasicPropagator(),
        bnn.inference.MonteCarlo(),
        bnn.inference.UnscentedTransform(),
    ]
    for propagator in propagators:
        out = propagator(module=bayes_module, input=x)
        assert list(out.shape) == [
            batch_size,
            in_features,
        ], f"Outputs of `{type(propagator).__name__}` not expected shape!"

    # Test the Linear layer propagator.
    in_features = 3
    out_features = 2
    x = bnn.types.MuVar(torch.randn((batch_size, in_features)))
    module = torch.nn.Linear(in_features=in_features, out_features=out_features)
    bayes_module = bnn.wrappers.BayesianModule(module, learn_var=True)
    propagator = bnn.inference.Linear()
    out = propagator(module=bayes_module, input=x)
    assert list(out.shape) == [
        batch_size,
        out_features,
    ], f"Outputs of `{type(propagator).__name__}` not expected shape!"

    # Test convolutional propagators.
    n_dim = [1, 2, 3]
    propagators = [getattr(bnn.inference, f"Conv{n}d")() for n in n_dim]
    kernel_size = 3
    for n, propagator in enumerate(propagators):
        x = bnn.types.MuVar(
            torch.randn(
                (batch_size, in_features, *[kernel_size for _ in range(n_dim[n])])
            )
        )
        module = getattr(torch.nn, f"Conv{n_dim[n]}d")(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
        )
        bayes_module = bnn.wrappers.BayesianModule(module, learn_var=True)
        out = propagator(module=bayes_module, input=x)
        assert list(out.shape) == [
            batch_size,
            out_features,
            *[1 for _ in range(n_dim[n])],
        ], f"Outputs of `{type(propagator).__name__}` not expected shape!"

    # Test transposed convolution propagators.
    n_dim = [1, 2, 3]
    propagators = [getattr(bnn.inference, f"ConvTranspose{n}d")() for n in n_dim]
    kernel_size = 3
    for n, propagator in enumerate(propagators):
        x = bnn.types.MuVar(
            torch.randn(
                (batch_size, in_features, *[kernel_size for _ in range(n_dim[n])])
            )
        )
        module = getattr(torch.nn, f"ConvTranspose{n_dim[n]}d")(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
        )
        bayes_module = bnn.wrappers.BayesianModule(module, learn_var=True)
        out = propagator(module=bayes_module, input=x)
        assert list(out.shape) == [
            batch_size,
            out_features,
            *[kernel_size + in_features - 1 for _ in range(n_dim[n])],
        ], f"Outputs of `{type(propagator).__name__}` not expected shape!"
