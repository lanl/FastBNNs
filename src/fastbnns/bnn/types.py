"""Custom types and associated functionality."""

from __future__ import annotations
import functools
from typing import Any, Callable, Optional, Union

import math
import torch


# List torch functions that we can apply independently to mean and variance.
SIMPLE_TORCH_FUNCS = [
    torch.cat,
    torch.chunk,
    torch.dsplit,
    torch.column_stack,
    torch.dstack,
    torch.gather,
    torch.hsplit,
    torch.hstack,
    torch.index_select,
    torch.masked_select,
    torch.movedim,
    torch.permute,
    torch.reshape,
    torch.select,
    torch.split,
    torch.stack,
    torch.take_along_dim,
    torch.tensor_split,
    torch.tile,
    torch.transpose,
    torch.unbind,
    torch.unravel_index,
    torch.squeeze,
    torch.unsqueeze,
    torch.vsplit,
    torch.vstack,
    torch.where,
    torch.sum,  # assumes independence: E[a+b]=E[a]+E[b], V[a+b]=V[a]+V[b]
    torch.detach,
    torch.clone,
    torch.nn.functional.pad,
    torch.nn.functional.interpolate,
    torch.nn.functional.upsample,
    torch.nn.functional.upsample_nearest,
    torch.nn.functional.upsample_bilinear,
    torch.nn.functional.grid_sample,
    torch.nn.functional.affine_grid,
]

# Define additional tensor-specific methods that can only be called as x.method(), not torch.method(x).
TENSOR_METHODS = ["cpu", "cuda", "to", "requires_grad_", "view"]


class MuVar:
    """Custom object holding mean and variance of some distribution.

    WARNING: Some functionality, like __pow__(), assumes the normal distribution!"""

    def __init__(
        self,
        mu: Union[
            torch.Tensor,
            list[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            MuVar,
        ],
        var: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize MuVar instance.

        Args:
            mu: Mean of input distribution, or optionally, a list containing
                both mean and variance (to allow calling this method on a list
                without unpacking arguments).
            var: Variance of input distribution.
        """
        if isinstance(mu, (list, tuple)):
            self.mu = mu[0]
            self.var = mu[1]
        elif isinstance(mu, MuVar):
            # Repackage for compatibility.
            self.mu = mu.mu
            self.var = mu.var
        elif var is None:
            # Only mean was passed, default variance to None
            # (which will be treated as zero when possible/appropriate).
            self.mu = mu
            self.var = None
        else:
            # Mu and var passed individually.
            self.mu = mu
            self.var = var

    @classmethod
    def __torch_function__(
        self,
        func: Callable,
        types: list,
        args: Any = (),
        kwargs: dict = {},
    ) -> Any:
        """General overloading function for torch functions."""

        # Ensure this is the __torch_function__ we need to call.
        # See https://pytorch.org/docs/stable/notes/extending.html
        if not any(issubclass(t, MuVar) for t in types):
            return NotImplemented

        # For "simple" functions like torch.cat, we'll route mu and var separately
        # through the function.  Otherwise, we don't want to use these
        # __torch_function__ implementations.
        if func in SIMPLE_TORCH_FUNCS:

            def split_muvar(args: Union[list, tuple]) -> Union[list, tuple]:
                """Recursively split MuVar instances into separate lists of mu and lists of var."""
                if isinstance(args, Union[list, tuple]):
                    # Loop through arguments and split.
                    args_mu = []
                    args_var = []
                    for arg in args:
                        if isinstance(arg, MuVar):
                            # Split into mu and var, explicitly materializing None var's to zeros
                            # (to enable generic function applications expecting a tensor valued var).
                            args_mu.append(arg.mu)
                            args_var.append(
                                torch.zeros_like(arg.mu) if arg.var is None else arg.var
                            )
                        elif isinstance(arg, (list, tuple)):
                            # Split MuVar items if needed.
                            splits = split_muvar(arg)
                            args_mu.append(splits[0])
                            args_var.append(splits[1])
                        else:
                            # Append to both argument lists.
                            args_mu.append(arg)
                            args_var.append(arg)
                    return args_mu, args_var
                else:
                    # This is for, e.g., scalar arguments.
                    return args, args

            # Separate MuVar types into mu and var for split calls to `func`.
            args_mu, args_var = split_muvar(args)
            kwargs_mu = {}
            kwargs_var = {}
            for k, v in kwargs.items():
                kwargs_mu[k], kwargs_var[k] = split_muvar(v)
            return MuVar(func(*args_mu, **kwargs_mu), func(*args_var, **kwargs_var))
        elif hasattr(self, func.__name__):
            # A custom implementation of this torch function was defined for this type.
            return getattr(self, func.__name__)(*args, **kwargs)
        else:
            # Return NotImplemented to allow other overrides to be used.
            return NotImplemented

    def __repr__(self):
        """Custom display functionality."""
        return f"MuVar({self.mu}, {self.var})"

    def __getattr__(self, name: str) -> Any:
        """Custom getattr fallback handler."""
        # If a torch function exists with name `name` (e.g., x.sum()), return that.
        # Otherwise we'll return the requested attribute for self.mu.
        torch_fxn = getattr(torch, name, None)
        if name in TENSOR_METHODS:
            # These methods can only be called as tensor.method() but otherwise can be applied
            # to mu and var independently.
            return functools.partial(getattr(self, "apply_method"), name)
        elif (torch_fxn is not None) and callable(torch_fxn):
            return functools.partial(torch_fxn, self)
        else:
            if hasattr(self.mu, name):
                return getattr(self.mu, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, idx: int) -> MuVar:
        """Access requested index of self.mu and self.var"""
        return MuVar(self.mu[idx], self.var[idx])

    def __add__(self, input: Union[float, torch.Tensor, list, MuVar]) -> MuVar:
        """Custom add functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Adding a float or tensor is like adding a delta R.V., so
            # variance does not change.
            return MuVar(self.mu + input, self.var)
        elif isinstance(input, (list, MuVar)):
            # Means and variances both add (assuming independent distributions).
            mu_in = input.mu if isinstance(input, MuVar) else input[0]
            var_in = input.var if isinstance(input, MuVar) else input[1]
            if (self.var is None) and (var_in is None):
                return MuVar(self.mu + mu_in, None)
            elif self.var is None:
                return MuVar(self.mu + mu_in, var_in)
            elif var_in is None:
                return MuVar(self.mu + mu_in, self.var)
            else:
                return MuVar(self.mu + mu_in, self.var + var_in)
        else:
            raise NotImplementedError

    def __radd__(self, input: Union[float, torch.Tensor, list]) -> MuVar:
        """Custom add functionality for MuVar types."""
        return self.__add__(input)

    def __sub__(self, input: Union[float, torch.Tensor, list, MuVar]) -> MuVar:
        """Custom subtract functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Adding a float or tensor is like adding a delta R.V., so
            # variance does not change.
            return MuVar(self.mu - input, self.var)
        elif isinstance(input, (list, MuVar)):
            # Means can be subtracted but variances still add.
            mu_in = input.mu if isinstance(input, MuVar) else input[0]
            var_in = input.var if isinstance(input, MuVar) else input[1]
            if (self.var is None) and (var_in is None):
                return MuVar(self.mu - mu_in, None)
            elif self.var is None:
                return MuVar(self.mu - mu_in, var_in)
            elif var_in is None:
                return MuVar(self.mu - mu_in, self.var)
            else:
                return MuVar(self.mu - mu_in, self.var + var_in)
        else:
            raise NotImplementedError

    def __rsub__(self, input: Union[float, torch.Tensor, list]) -> MuVar:
        """Custom subtract functionality for MuVar types."""
        return self.__sub__(input)

    def __mul__(self, input: Union[float, torch.Tensor, list, MuVar]) -> MuVar:
        """Custom multiply functionality for MuVar types."""
        if isinstance(input, (float, torch.Tensor)):
            # Multiplication by scalar: E[aX] = aE[x], V[aX]=a**2 V[X]
            if self.var is None:
                return MuVar(input * self.mu, None)
            else:
                return MuVar(input * self.mu, (input**2) * self.var)
        elif isinstance(input, (list, MuVar)):
            # Multiplication of two random independent variables:
            #   E[XY] = E[X]E[Y]
            #   V[XY] = (E[X]**2)*V[Y] + V[X]*(E[Y]**2) + V[X]*V[Y]
            mu_in = input.mu if isinstance(input, MuVar) else input[0]
            var_in = input.var if isinstance(input, MuVar) else input[1]
            mu = mu_in * self.mu
            if (self.var is None) and (var_in is None):
                var = None
            elif self.var is None:
                var = (self.mu**2) * var_in
            elif var_in is None:
                var = self.var * (mu_in**2)
            else:
                var = (self.mu**2) * var_in + self.var * ((mu_in**2) + var_in)
            return MuVar(mu, var)
        else:
            raise NotImplementedError

    def __rmul__(self, input: Union[float, torch.Tensor, list]) -> MuVar:
        """Custom multiply functionality for MuVar types."""
        return self.__mul__(input)

    def __matmul__(self, input: Union[torch.Tensor, list, MuVar]) -> MuVar:
        """Custom matrix multiply functionality for MuVar types."""
        # NOTE: MuVar is NOT holding multivariate distributions.  Each scalar entry
        # represents (mu, var) of an independent distribution, so matrix multiplication
        # is not multiplication of multivariate random variables!
        if isinstance(input, torch.Tensor):
            # Multiplication by scalar: E[a@X] = a @ E[x], V[a@X]=a**2 @ V[X]
            if self.var is None:
                return MuVar(input @ self.mu, None)
            else:
                return MuVar(input @ self.mu, input @ self.var @ input.T)
        elif isinstance(input, (list, MuVar)):
            # Multiplication of two random independent variables:
            #   E[X@Y] = E[X] @ E[Y]
            #   V[X@Y] = E[X]**2 @ V[Y] + V[X] @  E[Y]**2 + V[X] @ V[Y]
            mu_in = input.mu if isinstance(input, MuVar) else input[0]
            var_in = input.var if isinstance(input, MuVar) else input[1]
            mu = self.mu @ mu_in
            if (self.var is None) and (var_in is None):
                var = None
            elif self.var is None:
                var = (self.mu**2) @ var_in
            elif var_in is None:
                var = self.var @ (mu_in**2)
            else:
                var = (self.mu**2) @ var_in + self.var @ ((mu_in**2) + var_in)
            return MuVar(mu, var)
        else:
            raise NotImplementedError

    def __rmatmul__(self, input: Union[torch.Tensor, list]) -> MuVar:
        """Custom matrix multiply functionality for MuVar types."""
        return self.__matmul__(input)

    def __pow__(self, input: int) -> MuVar:
        """Custom exponentiation functionality for MuVar types.

        WARNING: This implementation assumes independent Normally distributed random variables!
        """
        if isinstance(input, int):
            # Exponentiation of a Normal random variable: see
            # https://en.wikipedia.org/wiki/Normal_distribution#Moments
            def normal_moment(
                mu: torch.Tensor,
                v: torch.Tensor,
                n: int,
            ) -> torch.Tensor:
                """Compute E[X^n] for X ~ Normal(mu, v)."""
                moment = torch.zeros_like(mu)
                for m in range(n // 2 + 1):
                    coeff = math.comb(n, 2 * m) * math.prod(range(1, 2 * m, 2))
                    moment += coeff * (v**m) * (mu ** (n - 2 * m))
                return moment

            mu = normal_moment(mu=self.mu, v=self.var, n=input)
            if self.var is None:
                var = None
            else:
                var = (
                    normal_moment(mu=self.mu, v=self.var, n=2 * input)
                    - normal_moment(mu=self.mu, v=self.var, n=input) ** 2
                )
            return MuVar(mu, var)
        else:
            raise NotImplementedError

    def apply(self, func: Callable, *args, **kwargs) -> MuVar:
        """Generic apply() for functions that act separately on mu and var."""
        if self.var is None:
            # If self.var is None (zero variance), we need to explicitly materialize the
            # zeros tensor to accommodate arbitrary func().
            return MuVar(
                func(self.mu, *args, **kwargs),
                func(torch.zeros_like(self.mu), *args, **kwargs),
            )
        else:
            return MuVar(
                func(self.mu, *args, **kwargs),
                func(self.var, *args, **kwargs),
            )

    def apply_method(self, name: str, *args, **kwargs) -> MuVar:
        """Generic apply_method() for methods that act separately on mu and var."""
        if self.var is None:
            # If self.var is None (zero variance), we need to explicitly materialize the
            # zeros tensor to accommodate arbitrary func().
            return MuVar(
                getattr(self.mu, name)(*args, **kwargs),
                getattr(torch.zeros_like(self.mu), name)(*args, **kwargs),
            )
        else:
            return MuVar(
                getattr(self.mu, name)(*args, **kwargs),
                getattr(self.var, name)(*args, **kwargs),
            )

    def numel(self) -> int:
        """Custom numel() to avoid complicated logic in __getattr__ above."""
        return self.mu.numel()

    def mean(self, *args, **kwargs) -> MuVar:
        """Custom replacement of torch.mean() for MuVar type."""
        # Assuming independence, we can directly apply mean to self.mu.
        x_mean = self.mu.mean(*args, **kwargs)

        # To compute variance, we call the torch version of mean() with
        # keepdim=True so we can account for the scaling factor:
        # assuming independence, V[(x_0+x_1) / 2] = (V[x_0]+V[x_1]) / 4
        if self.var is None:
            x_var = None
        else:
            if (args == ()) and (kwargs == {}):
                x_var = self.var.mean()
                x_var /= self.var.numel() ** 2
            else:
                x_var = self.var.mean(*args, **(kwargs | {"keepdim": True}))
                x_var /= (
                    torch.tensor(self.var.shape) / torch.tensor(x_var.shape)
                ).prod().squeeze() ** 2
        if kwargs.pop("keepdim", None) is None:
            return MuVar(x_mean, x_var)
        else:
            return MuVar(x_mean.squeeze(), None if x_var is None else x_var.squeeze())


if __name__ == "__main__":
    # Scalar operations.
    a = MuVar(torch.tensor(1.0), torch.tensor(2.0))
    b = MuVar(torch.tensor(1.1), torch.tensor(0.5))

    print(a + b)
    print(a + 1.0)
    print(1.0 + a)
    print(a**2)

    # Torch/tensor operations and attributes.
    a = MuVar(torch.randn((2, 2)), torch.ones((2, 2)))
    b = MuVar(torch.randn((2, 2)), 1.1 * torch.ones((2, 2)))

    print(a.shape)
    print(a.size(1))
    print(a.to("cpu"))
    print(a.numel())
    print(a.sum())
    print(a.mean())
    print(a.mean(dim=1, keepdim=True))
    print(a @ b)
    print(torch.cat([a, b], dim=-1))
    print(torch.nn.functional.pad(a, [0, 1, 2, 0]))
