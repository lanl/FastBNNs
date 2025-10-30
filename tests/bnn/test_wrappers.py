"""Test BNN module wrappers."""

import torch

from fastbnns.bnn.types import MuVar
from fastbnns.bnn.wrappers import BayesianModule


def test_wrappers() -> None:
    """Test wrappers with simple example modules."""
    # Test the generic wrapper with a simple module.
    in_features = 3
    out_features = 2
    module = torch.nn.Linear(in_features=in_features, out_features=out_features)
    bayes_module = BayesianModule(module, learn_var=True)

    # Verify that additional learnable parameters were added to learn variance.
    learnable_params = [p for p in module.parameters() if p.requires_grad]
    learnable_params_bnn = [p for p in bayes_module.parameters() if p.requires_grad]
    assert len(learnable_params_bnn) == 2 * len(learnable_params), (
        "`BayesianModule` wrapper is not using `learn_var` as expected!"
    )

    # Verify that the KL-divergence calculator runs.
    kl = bayes_module.compute_kl_divergence()
    assert isinstance(kl, torch.Tensor) and kl.numel() == 1, (
        "KL divergence returned by compute_kl_divergence() not expected type and shape!"
    )

    # Test the forward pass.
    batch_size = 4
    x = MuVar(torch.randn((batch_size, in_features)))
    out_orig = module(x[0])
    out_bayes = bayes_module(x)
    assert out_orig.shape == out_bayes.shape, (
        "Forward pass of `BayesianModule` not returning expected output shape!"
    )
