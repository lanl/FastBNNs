"""Test BNN inference modules."""

import torch

import bnn.inference
import bnn.types
import bnn.wrappers
import models.polynomial


def test_inference() -> None:
    """Test inference wrappers with simple example modules."""
    # Test the generic inference modules on a basic module.
    torch.manual_seed(12)
    module = models.polynomial.PolyModule(poly_order=2)
    bayes_module = bnn.wrappers.BayesianModule(module, learn_var=False)
    batch_size = 4
    in_features = 3
    x = bnn.types.MuVar(
        torch.randn((batch_size, in_features)),
        torch.zeros((batch_size, in_features)),
    )
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

    # Test generic inference modules for a module with learnable variance.
    module = models.polynomial.PolyModule(poly_order=2)
    bayes_module = bnn.wrappers.BayesianModule(module, learn_var=True)
    batch_size = 1
    in_features = 1
    x = bnn.types.MuVar(
        torch.randn((batch_size, in_features)),
        torch.zeros((batch_size, in_features)),
    )
    n_samples = 100
    propagators = [
        bnn.inference.BasicPropagator(),
        bnn.inference.MonteCarlo(n_samples=n_samples),
        bnn.inference.UnscentedTransform(),
    ]
    out_mc_manual = torch.stack([bayes_module(x[0]) for _ in range(n_samples)])
    out_mc_mean = out_mc_manual.mean()
    out_mc_stdev = out_mc_manual.std()
    out = []
    for propagator in propagators:
        out = propagator(module=bayes_module, input=x)

        # Verify correct shape is returned.
        assert list(out.shape) == [
            batch_size,
            in_features,
        ], f"Outputs of `{type(propagator).__name__}` not expected shape!"

        # Verify outputs are consistent with manual Monte Carlo result.
        tol = 1.0e-1  # chosen empirically
        assert (out[0] - out_mc_mean).abs() < tol, (
            f"{type(propagator).__name__} not returning expected mean!"
        )
        assert (out[1].sqrt() - out_mc_stdev).abs() < tol, (
            f"{type(propagator).__name__} not returning expected variance!"
        )

    # Test the Linear layer propagator.
    in_features = 3
    out_features = 2
    x = bnn.types.MuVar(
        torch.randn((batch_size, in_features)),
        torch.zeros((batch_size, in_features)),
    )
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
            ),
            torch.zeros(
                (batch_size, in_features, *[kernel_size for _ in range(n_dim[n])])
            ),
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
            ),
            torch.zeros(
                (batch_size, in_features, *[kernel_size for _ in range(n_dim[n])])
            ),
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


if __name__ == "__main__":
    test_inference()
