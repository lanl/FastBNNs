# FastBNNs developer guide

## Introduction
FastBNNs was designed to enable automatic conversion of PyTorch-based neural networks (NNs) to Bayesian neural networks (BNNs), coupled with fast, approximate variational inference using the algorithms presented in [1] and [2].
To accomplish this, FastBNNs searches for leaf modules of a PyTorch neural network `nn`, wraps them in a module-dependent wrapper, and pairs each wrapper with an appropriate inference class.
FastBNNs additionally implements a custom data type `MuVar` found in [types.py](../../src/fastbnns/bnn/types.py).
The `MuVar` type wraps two PyTorch Tensors `mu` and `var`, corresponding to the mean and variance of a Normal distribution, respectively.
At inference, FastBNNs leverages a combination of the selected inference classes and custom-implemented `MuVar` operations to propagate network inputs and their uncertainties through the Bayesian neural network.

In the following sections, we provide additional details about this process that developers may find useful.
In particular, we expand on the mechanics of the automated NN-to-BNN conversion process, leaf module wrappers, and inference classes/the `MuVar` type, as well as provide useful tips for developers that need to expand or modify the behavior of key components of FastBNNs.

## Automated NN-to-BNN conversion
A PyTorch-based NN `nn` can be converted to a BNN using 

```
from fastbnns.bnn.base import BNN
bnn = BNN(nn=nn, convert_in_place=False)
```

Internally, the base class `BNN` will call the helper function `convert_to_bnn_()` found in [wrappers.py](../../src/fastbnns/bnn/wrappers.py).
`convert_to_bnn_()` attempts to find all unique leaf modules of `nn` (i.e., `nn.named_modules()` that do not have any child modules).
Using a combination of default arguments as well as user-defined `args` and `kwargs` passed through `BNN`, `convert_to_bnn_()` will wrap each leaf module with a FastBNNs wrapper found in [wrappers.py](../../src/fastbnns/bnn/wrappers.py).
At instantiation, each wrapper will additionally select an inference class using the helper function `select_default_propagator()` found in [wrappers.py](../../src/fastbnns/bnn/wrappers.py).

### `convert_to_bnn_()`
The `convert_to_bnn_()` utility was designed to reroute forward passes through `nn` through `bnn` by hijacking the forward pass `nn.forward()` of the base `nn`.
The intention is to capture standard operations made through a forward pass of a NN, e.g., the actions of each distinct module in a `torch.nn.Sequential()` module.
To do this, module wrappers store a copy of the original leaf module (used for its `forward()` call) and its learnable parameters (used to define the mean of the parameter distributions).
If appropriate (e.g., for a `BayesianModule` wrapper in [wrappers.py](../../src/fastbnns/bnn/wrappers.py) with argument `learn_var=True` wrapping a module with learnable parameters), the wrapper will create another set of parameters associated with the variances of learnable parameters.
In most cases, `convert_to_bnn_()` itself should not be modified.
Instead, users should leverage the arguments of `convert_to_bnn_()` passed through the `BNN` base class to modify which wrappers and inference classes are used by each wrapper.
We suggest developers carefully review `convert_to_bnn_()` to understand the selection process of wrappers and corresponding inference classes.


## Module wrappers
A basic set of `torch.nn.Module` wrappers have been defined in [wrappers.py](../../src/fastbnns/bnn/wrappers.py).
The wrapper `BayesianModule` is the primary wrapper for arbitrary leaf modules of `nn`.
To write a custom wrapper for arbitrary leaf modules, we recommend inheriting from the base class `BayesianModuleBase` to ensure compatibility with the rest of the FastBNNs package.
An additional wrapper `BroadcastModule` was added to reduce overhead when wrapping modules whose action applies independently to an input distribution (e.g., the `torch.nn.Identity()` module, which should leave an input distribution unchanged).
A set of modules that are automatically wrapped by `BroadcastModule` is maintained in the variable `BROADCAST` in [wrappers.py](../../src/fastbnns/bnn/wrappers.py).
Developers are encouraged to expand this list if appropriate modules from `torch.nn` are identified that should be present in this list.


## Inference and the MuVar type
FastBNNs essentially hijacks the forward pass of the underlying `nn` to avoid custom mapping of the computational graph.
This is achieved through the combined action of custom inference classes [inference.py](../../src/fastbnns/bnn/inference.py) and the datatype MuVar found in [types.py](../../src/fastbnns/bnn/types.py).

Inference classes are intended to reroute module inputs to the underlying modules forward call as appropriate for each inference algorithm.
For example, the custom inference algorithm `Linear` in [inference.py](../../src/fastbnns/bnn/inference.py) will use analytic moment propagation rules to compute the mean and variance of `torch.nn.Linear` when called on an input random variable.
More general algorithms found in [inference.py](../../src/fastbnns/bnn/inference.py) include `UnscentedTransform`, `JointUnscentedTransform`, and `MonteCarlo`, which similarly route inputs through forward calls to underlying `torch.nn.Modules` and estimate the output mean and variance from the results.
If the user does not specify a specific inference algorithm for each module when using `BNN()` or `convert_to_bnn_()`, a default is selected to based on the module name and whether or not it has learnable parameters (see `select_default_propagator()` in [wrappers.py](../../src/fastbnns/bnn/wrappers.py) understand the selection process).
Analytic propagation algorithms for layers such as `torch.nn.Linear`, `torch.nn.Conv(1,2,3)d`, `torch.nn.ConvTranspose(1,2,3)d`, and `torch.nn.AvgPool(1,2,3)d` have already been implemented.
Custom analytic propagation algorithms for modules in `torch.nn` should be added to [inference.py](../../src/fastbnns/bnn/inference.py) with the algorithm defined as a `MomentPropagator` class whose name matches the name of the `torch.nn.Module`.
Analytic propagation algorithms for `torch.nn.ReLU` and `torch.nn.LeakyReLU` are included as reference under the aliases `ReLUa` and `LeakyReLUa`.
The suffix `a` was added to intentionally prevent their selection by `select_default_propagator()`, as we have found the `UnscentedTransform` to be faster with minimal loss in accuracy (unpublished result).

Since many PyTorch NN workflows include additional operations not encompassed in a subclass of `torch.nn.Module`, FastBNNs implements a custom type `MuVar` to carry the mean and variance of a distribution through such operations.
The `MuVar` type implements custom handlers in [types.py](../../src/fastbnns/bnn/types.py) which apply analytic propagation rules (e.g., for the addition of two random variables) or fallbacks like the unscented transform (e.g., for arbitrary torch function calls whose computational graph is unknown).
If operations that work for `torch.Tensor` instances do not work (i.e., throw an error) for the `MuVar` type, developers should review the sets `SIMPLE_TORCH_FUNCS` and `TENSOR_METHODS` to determine if the operation can be added to these sets.
If instead an operation works for `torch.Tensor` and works with `MuVar` without throwing an error, yet the output differs from the result expected by the user, developers should consider implementing a custom handler for that operation and register it with the `MUVAR_HANDLERS` in [types.py](../../src/fastbnns/bnn/types.py).

To review the FastBNNs inference workflow with a simple example, consider a NN

```
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = torch.nn.ModuleList([
            torch.nn.Linear(1, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2, 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.module_list:
            x = layer(x)
        
        return x + x**2
```

We then convert to a BNN and run a test pass through the model:
```
from fastbnns.bnn.base import BNN
from fastbnns.bnn.types import MuVar

nn = model()
bnn = BNN(nn=nn, convert_in_place=False)
out = bnn(MuVar(torch.randn(1, 1)))
```
At conversion, the `torch.nn.Linear` layers will be wrapped in a `BayesianModule` and assigned an inference class `Linear`, while the `torch.nn.LeakyReLU` layer will be wrapped in a `BayesianModule` with inference class `UnscentedTransform`.
When calling `bnn` on a `MuVar` wrapped input, the action `x=layer(x)` will be rerouted through the inference module of each layer to produce an output `MuVar` corresponding to moment propagation through that layer.
In contrast, the final set of operations `x + x**2`, which are not contained within a `torch.nn.Module`, will rely on handlers for `add` and `pow` defined in the `MuVar` type.

## References
[1] David J. Schodt, Ryan Brown, Michael Merritt, Samuel Park, Delsin Menolascino, and Mark A.
Peot. A framework for variational inference of lightweight bayesian neural networks with
heteroscedastic uncertainties. 2024. arXiv:2402.14532 [cs].

[2] David J. Schodt. Few-sample Variational Inference of Bayesian Neural Networks with Arbitrary Nonlinearities. 2024. arXiv:2405.02063 [cs].