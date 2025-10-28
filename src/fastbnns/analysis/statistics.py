"""Helpers for statistical analysis of BNNs."""

import torch


def compute_coverage(
    observations: torch.Tensor,
    mu: torch.Tensor = torch.tensor(0.0),
    sigma: torch.Tensor = torch.tensor(1.0),
    alphas: torch.Tensor = torch.tensor([1.0]),
) -> torch.Tensor:
    """Compute the coverage of `alpha` confidence intervals of a Gaussian."""
    coverage = []
    for alpha in alphas:
        in_interval = (observations - mu).abs() <= (sigma * alpha)
        coverage.append(in_interval.sum() / in_interval.numel())

    return torch.stack(coverage)
