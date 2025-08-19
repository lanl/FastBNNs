# FastBNNs

## Introduction
FastBNNs implements fast and flexible Bayesian inference of neural networks based on propagation of statistical moments [1] and the unscented transform [2] in PyTorch.
FastBNNs enables one-line conversion of many PyTorch-based models to Bayesian counterparts, whereby learnable model parameters are treated as Normal random variables.

## Basic usage
A neural network `nn: torch.nn.Module` can be converted to a Bayesian neural network using the
`bnn.base.BNN` wrapper class (see [Caveats and known limitations](#caveats-and-known-limitations) for exceptions):

```
n_features = 1
out_features = 1
hidden_features = 32
n_hidden_layers = 1
in_features = 1
out_features = 1
nn = mlp.MLP(
    in_features=in_features,
    out_features=out_features,
    n_hidden_layers=n_hidden_layers,
    hidden_features=hidden_features,
    activation=torch.nn.LeakyReLU,
)
bnn = bnn.base.BNN(nn=nn, convert_in_place=False)
```

Forward calls through `bnn` can be made identically to `nn`:

```
data = torch.randn((1, n_features), dtype=torch.float32)
out_nn = nn(data)
out_bnn = bnn(data)
```

In this usage, a single sample of each network parameter is made in `bnn` before computing the forward computation identically to `nn`.
As such, multiple network samples can be made to characterize the output distribution as

```
n_samples = 100
out_bnn_mc = torch.stack([bnn(data) for _ in range(n_samples)])
out_bnn_mc_mean = out_bnn_mc.mean(dim=0)
out_bnn_mc_var = out_bnn_mc.var(dim=0)
```

Alternatively, to leverage the fast inference methods (i.e., non-sampling-based), the network input can be wrapped in the custom type `bnn.types.MuVar`:

```
out_bnn_fast = out_bnn(bnn.types.MuVar(data))
out_bnn_fast_mean = out_bnn_fast[0]
out_bnn_fast_var = out_bnn_fast[1]
```


## Bayesian treatment
The wrapped model `bnn: torch.nn.Module` is still an instance of `torch.nn.Module` and can be trained using standard PyTorch or PyTorch Lightning strategies.
However, a Bayesian treatment of `bnn` training requires use of a custom loss function, such as the evidence lower bound (ELBO) used in Bayes-by-backprop [3].
Examples of training the Bayesian MLP from [Basic Usage](#basic-usage) using the ELBO loss are provided in [PyTorch](examples/mlp.py) and [PyTorch Lightning](examples/mlp_lightning.py).

## Caveats and known limitations
The base model wrapper `bnn.base.BNN` attempts to convert a neural network `nn: torch.nn.Module` to a Bayesian neural network with Normally distributed parameters assuming a mean-field approximation (i.e., all parameters are conditionally independent of the data).
This is done by wrapping sub-modules of `nn` with an appropriate wrapper from `bnn.wrappers`.
The Bayesian neural network `bnn = bnn.base.BNN(nn)` should behave as a stochastic version of `nn` on a typical forward pass `output = bnn(data)`, and no issues are known at this time.
However, to leverage the fast inference methods of [1, 2] while maintaining the flexibility of the wrapper (i.e., one line conversion to/from a Bayesian version of `nn`), we introduce a custom type that wraps `data` for a forward call as `bnn(bnn.types.MuVar(data))`.
This allows us to propagate the predictive mean and variance through each sub-module of `nn`.
To accommodate non-`torch.nn.Module` operations in the neural network, `bnn.types.MuVar` implements several common tensor operations (e.g., addition, concatenation, ...) that act on the mean and variance as needed for the operation.
Unfortunately, some operations, such as those using external calls to C, are not accounted for and hence neural networks using such operations may not be compatible with `bnn.base.BNN` (e.g., `torch.nn.Transformer`) without monkey-patching those operations to accommodate the `bnn.types.MuVar` type.

## References
[1] David J. Schodt, Ryan Brown, Michael Merritt, Samuel Park, Delsin Menolascino, and Mark A.
Peot. A framework for variational inference of lightweight bayesian neural networks with
heteroscedastic uncertainties. 2024. arXiv:2402.14532 [cs].

[2] David J. Schodt. Few-sample Variational Inference of Bayesian Neural Networks with Arbitrary Nonlinearities. 2024. arXiv:2405.02063 [cs].

[3] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight Uncertainty
in Neural Networks, May 2015. arXiv:1505.05424 [cs, stat]

## Copyright

LANL O4956

&copy; 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
