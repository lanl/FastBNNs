"""This example demonstrates FastBNNs ability to infer arbitrary torch parameters."""

import copy
import timeit

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import laplace
import matplotlib.pyplot as plt
import torch

from fastbnns.analysis import statistics
from fastbnns.bnn import base, losses, priors, types
from fastbnns.datasets import polynomial
from fastbnns.simulation import generators, observation

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class LearnableNonlinearity(torch.nn.Module):
    """PyTorch module for a custom, learnable nonlinearity."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize module."""
        super().__init__()

        self.c = torch.nn.Parameter(torch.zeros((in_features, out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through module."""
        return (x**3) @ self.c


class CustomMLP(torch.nn.Module):
    """PyTorch module to model a nonlinear function."""

    def __init__(self, in_features: int, out_features: int, n_hidden: int) -> None:
        """Initialize module."""
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features,
                out_features=n_hidden,
                bias=True,
            ),
            LearnableNonlinearity(
                in_features=n_hidden,
                out_features=n_hidden,
            ),
            torch.nn.Linear(
                in_features=n_hidden,
                out_features=out_features,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through module."""
        return self.mlp(x)


# Define a dataset.
a = torch.tensor([[1.0]])
c = torch.tensor([[2.0]])
in_features = 1
out_features = 1
data_generator = generators.Generator(
    simulator=lambda x: x @ a + (x**3) @ c,
    simulator_kwargs={},
    simulator_kwargs_generator={"x": lambda: torch.rand(in_features) - 0.5},
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.add_read_noise,
    noise_fxn_kwargs_generator={
        "sigma": lambda x: 0.1 + 0.2 * (torch.cos(2.0 * torch.pi * x) ** 2)
    },
)
n_data = 1024
batch_size = 128
ds_train = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=noise_tform,
    cache=True,  # set False to use fresh data every epoch
)
dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size)
n_data_val = 128
ds_val = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data_val,
    transform=noise_tform,
    cache=True,
)
dl_val = torch.utils.data.DataLoader(dataset=ds_val, batch_size=batch_size)

# Create a torch model for the dataset.
n_hidden = 8
nn = CustomMLP(
    in_features=in_features,
    out_features=out_features,
    n_hidden=n_hidden,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn.to(device)

# Train a standard NN.
lr = 1.0e-2
optimizer = torch.optim.AdamW(nn.parameters(), lr=lr)
best_model_state_dict_nn = copy.deepcopy(nn.state_dict())
best_loss = torch.inf
loss_fn_nn = torch.nn.MSELoss()
n_epochs_max = 5000  # maximum allowed epochs
n_epochs_print_status = 500  # number of epochs to print status update
patience = 500  # early stopping patience in epochs
n_batches = n_data // batch_size
print("______________________________________________________________________")
print("Training base neural network...")
epochs_without_val_loss_decrease = 0
for epoch in range(n_epochs_max):
    # Training set:
    nn.train()
    loss_epoch_train = []
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = nn(batch[0].to(device))

        # Compute loss.
        loss = loss_fn_nn(input=out, target=batch[1].to(device))

        # Update model.
        loss.backward()
        optimizer.step()
        loss_epoch_train.append(loss)

    # Validation set:
    nn.eval()
    with torch.no_grad():
        loss_epoch_val = []
        within_1sigma_epoch_val = []
        for batch in dl_val:
            # Forward pass through model.
            out = nn(batch[0].to(device))

            # Compute loss.
            loss = loss_fn_nn(input=out, target=batch[1].to(device))
            loss_epoch_val.append(loss)

    avg_loss_train = torch.mean(torch.stack(loss_epoch_train))
    avg_loss_val = torch.mean(torch.stack(loss_epoch_val))

    if avg_loss_val < best_loss:
        epochs_without_val_loss_decrease = 0
        best_loss = avg_loss_val
        best_model_state_dict_nn = copy.deepcopy(nn.state_dict())
    else:
        epochs_without_val_loss_decrease += 1

    stop_early = epochs_without_val_loss_decrease >= patience
    if (not (epoch % n_epochs_print_status)) or stop_early:
        print(
            f"epoch {epoch + 1} of {n_epochs_max}: train mse = {avg_loss_train}, val mse = {avg_loss_val}"
        )
        if stop_early:
            print(f"Stopping early at epoch {epoch + 1} due to val loss plateau")
            break

nn.load_state_dict(best_model_state_dict_nn)

# Train a BNN with FastBNNs.
posterior_mu_init = 0.0
posterior_rho_init = -3.0
nn_aleatoric = CustomMLP(
    in_features=in_features,
    out_features=out_features * 2,  # extra output node for aleatoric uncertainty
    n_hidden=n_hidden,
)
bnn = base.BNN(
    nn=nn_aleatoric,
    convert_in_place=False,
    wrapper_kwargs_global={
        "resample_mean": True,  # set False to initialize parameter means from `nn`
        "samplers_init": {
            "mean": torch.distributions.Normal(
                loc=posterior_mu_init, scale=0.1
            ),  # match initialization behavior of bayesian_torch
            "rho": torch.distributions.Normal(
                loc=posterior_rho_init, scale=0.1
            ),  # match initialization behavior of bayesian_torch
        },
    },
)
prior_mu = 0.0
prior_sigma = 1.0
bnn.to(device)
prior = priors.Distribution(
    torch.distributions.Normal(
        loc=torch.tensor([prior_mu]), scale=torch.tensor([prior_sigma])
    )
).to(device)
kl_beta = 1.0 / n_batches  # see Graves 2011
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=kl_beta,
)
optimizer = torch.optim.AdamW(bnn.parameters(), lr=lr)
best_model_state_dict_fastbnns = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
print("______________________________________________________________________")
print("Training Bayesian neural network with fastbnns...")
epochs_without_val_loss_decrease = 0
for epoch in range(n_epochs_max):
    # Training set:
    bnn.train()
    loss_epoch_train = []
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = bnn(types.MuVar(batch[0].to(device)))

        # Compute loss: this model provides an additional output node that
        # we'll treat as the unscaled aleatoric uncertainty (variance inherent to
        # the data).
        aleatoric_var = torch.nn.functional.softplus(out.mu[:, 1]) ** 2
        epistemic_var = out.var[:, 0]
        loss = loss_fn(
            model=bnn,
            input=out.mu[:, 0],
            target=batch[1][:, 0].to(device),
            var=aleatoric_var + epistemic_var,
        )

        # Update model.
        loss.backward()
        optimizer.step()
        loss_epoch_train.append(loss)

    # Validation set:
    bnn.eval()
    with torch.no_grad():
        loss_epoch_val = []
        within_1sigma_epoch_val = []
        for batch in dl_val:
            # Forward pass through model.
            out = bnn(types.MuVar(batch[0].to(device)))

            # Compute loss: this model provides an additional output node that
            # we'll treat as the unscaled aleatoric uncertainty (variance inherent to
            # the data).
            predictive_var = (
                out.var[:, 0] + torch.nn.functional.softplus(out.mu[:, 1]) ** 2
            )
            loss = loss_fn(
                model=bnn,
                input=out.mu[:, 0],
                target=batch[1][:, 0].to(device),
                var=predictive_var,
            )
            loss_epoch_val.append(loss)

            # Check predictive variance.
            within_1sigma_epoch_val.append(
                statistics.compute_coverage(
                    observations=batch[1][:, 0].to(device),
                    mu=out.mu[:, 0],
                    sigma=predictive_var.sqrt(),
                    alphas=torch.tensor([1.0]),
                )
            )

    avg_loss_train = torch.mean(torch.stack(loss_epoch_train))
    avg_loss_val = torch.mean(torch.stack(loss_epoch_val))

    if avg_loss_val < best_loss:
        epochs_without_val_loss_decrease = 0
        best_loss = avg_loss_val
        best_model_state_dict_fastbnns = copy.deepcopy(bnn.state_dict())
    else:
        epochs_without_val_loss_decrease += 1

    stop_early = epochs_without_val_loss_decrease >= patience
    if (not (epoch % n_epochs_print_status)) or stop_early:
        print(
            f"epoch {epoch + 1} of {n_epochs_max}: train -ELBO = {avg_loss_train}, val -ELBO = {avg_loss_val}, {100.0 * torch.mean(torch.stack(within_1sigma_epoch_val)):.2f}% within 1 st. dev."
        )
        if stop_early:
            print(f"Stopping early at epoch {epoch + 1} due to val loss plateau")
            break

bnn.load_state_dict(best_model_state_dict_fastbnns)

# Train a BNN with Bayesian-Torch.
bayestorch_prior = {
    "prior_mu": prior_mu,
    "prior_sigma": prior_sigma,
    "posterior_mu_init": posterior_mu_init,
    "posterior_rho_init": posterior_rho_init,
    "type": "Reparameterization",
    "moped_enable": False,  # set True to initialize parameter means from `nn`
    "moped_delta": 1.0,
}
bnn_bayestorch = copy.deepcopy(nn_aleatoric)
dnn_to_bnn(bnn_bayestorch, bayestorch_prior)
bnn_bayestorch.to(device)
nll_loss = torch.nn.GaussianNLLLoss(reduction="sum")
optimizer = torch.optim.AdamW(bnn_bayestorch.parameters(), lr=lr)
best_model_state_dict_bayestorch = copy.deepcopy(bnn_bayestorch.state_dict())
best_loss = torch.inf
n_mc_samples_train = 30
print("______________________________________________________________________")
print("Training Bayesian neural network with bayesian_torch...")
epochs_without_val_loss_decrease = 0
for epoch in range(n_epochs_max):
    # Training set:
    bnn_bayestorch.train()
    loss_epoch_train = []
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = torch.stack(
            [bnn_bayestorch(batch[0].to(device)) for _ in range(n_mc_samples_train)]
        )

        # Compute loss: this model provides an additional output node that
        # we'll treat as the unscaled aleatoric uncertainty (variance inherent to
        # the data).
        kl = get_kl_loss(bnn_bayestorch)
        predictive_var = (
            torch.var(out[..., 0], dim=0)
            + torch.nn.functional.softplus(torch.mean(out[..., 1], dim=0)) ** 2
        )
        loss = (
            nll_loss(
                input=torch.mean(out[..., 0], dim=0),
                target=batch[1][:, 0].to(device),
                var=predictive_var,
            )
            + kl_beta * kl
        )

        # Update model.
        loss.backward()
        optimizer.step()
        loss_epoch_train.append(loss)

    # Validation set:
    bnn_bayestorch.eval()
    with torch.no_grad():
        loss_epoch_val = []
        within_1sigma_epoch_val = []
        for batch in dl_val:
            # Forward pass through model.
            out = torch.stack(
                [bnn_bayestorch(batch[0].to(device)) for _ in range(n_mc_samples_train)]
            )

            # Compute loss.
            kl = get_kl_loss(bnn_bayestorch)
            out_mu = torch.mean(out[..., 0], dim=0)
            predictive_var = (
                torch.var(out[..., 0], dim=0)
                + torch.nn.functional.softplus(torch.mean(out[..., 1], dim=0)) ** 2
            )
            loss = (
                nll_loss(
                    input=out_mu,
                    target=batch[1][:, 0].to(device),
                    var=predictive_var,
                )
                + kl / n_batches
            )
            loss_epoch_val.append(loss)

            # Check predictive variance.
            within_1sigma_epoch_val.append(
                statistics.compute_coverage(
                    observations=batch[1][:, 0].to(device),
                    mu=out_mu,
                    sigma=predictive_var.sqrt(),
                    alphas=torch.tensor([1.0]),
                )
            )

    avg_loss_train = torch.mean(torch.stack(loss_epoch_train))
    avg_loss_val = torch.mean(torch.stack(loss_epoch_val))

    if avg_loss_val < best_loss:
        epochs_without_val_loss_decrease = 0
        best_loss = avg_loss_val
        best_model_state_dict_bayestorch = copy.deepcopy(bnn_bayestorch.state_dict())
    else:
        epochs_without_val_loss_decrease += 1

    stop_early = epochs_without_val_loss_decrease >= patience
    if (not (epoch % n_epochs_print_status)) or stop_early:
        print(
            f"epoch {epoch + 1} of {n_epochs_max}: train -ELBO = {avg_loss_train}, val -ELBO = {avg_loss_val}, {100.0 * torch.mean(torch.stack(within_1sigma_epoch_val)):.2f}% within 1 st. dev."
        )
        if stop_early:
            print(f"Stopping early at epoch {epoch + 1} due to val loss plateau")
            break

bnn_bayestorch.load_state_dict(best_model_state_dict_bayestorch)

# Use Laplace to get a BNN from pretrained NN.
prior_precision = 1.0 / prior_sigma**2
la = laplace.DiagLaplace(
    nn,
    likelihood="regression",
    prior_precision=prior_precision,
)
print("______________________________________________________________________")
print("Training Bayesian neural network with laplace...")
la.fit(dl_train)
log_sigma_noise = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.AdamW([log_sigma_noise], lr=0.1)
best_loss = torch.inf
epochs_without_val_loss_decrease = 0
for epoch in range(n_epochs_max):
    # Early stopping.
    if epochs_without_val_loss_decrease >= patience:
        print(f"Stopping early at epoch {epoch} due to val loss plateau")
        break

    # Optimize observation noise.
    # NOTE: we don't optimize prior precision here since we want to make a
    # one-to-one comparison with bayesian_torch and fastbnns, which are also
    # not optimizing the prior in this example script.
    opt.zero_grad()
    loss = -la.log_marginal_likelihood(
        prior_precision=prior_precision,
        sigma_noise=log_sigma_noise.exp(),
    )
    loss.backward()
    opt.step()
    if loss < best_loss:
        epochs_without_val_loss_decrease = 0
        best_loss = loss.item()
    else:
        epochs_without_val_loss_decrease += 1

# Plot test predictions.
inference_device = "cuda"
bnn = bnn.to(inference_device).eval()
bnn_bayestorch.to(inference_device).eval()
la.model.to(inference_device).eval()
model_input = []
observations = []
n_data_test = 1024
ds_test = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data_test,
    transform=noise_tform,
    cache=True,
)
ds_test.data_generator.simulator_kwargs_generator["x"] = lambda: (
    2.0 * (torch.rand(1) - 0.5)
)  # factor of 2.0 so we test out-of-distribution samples
for n in range(n_data_test):
    data = ds_test[n]
    model_input.append(data[0])
    observations.append(data[1])
model_input = torch.stack(model_input, dim=0).to(inference_device)
observations = torch.stack(observations, dim=0)
n_mc_samples_test = 30
la_pred_params = {
    "pred_type": "glm",
    "link_approx": "bridge_norm",
}
with torch.no_grad():
    output = bnn(types.MuVar(model_input))
    n_timing_samples = 100
    t_fastbnns = timeit.timeit(
        lambda: bnn(types.MuVar(model_input)), number=n_timing_samples
    )
    output_bayestorch = torch.stack(
        [bnn_bayestorch(model_input) for _ in range(n_mc_samples_test)]
    )
    t_bayestorch = timeit.timeit(
        lambda: torch.stack(
            [bnn_bayestorch(model_input) for _ in range(n_mc_samples_test)]
        ),
        number=n_timing_samples,
    )
    output_bayestorch = (
        torch.mean(output_bayestorch, dim=0),
        torch.var(output_bayestorch, dim=0),
    )
    output_laplace = la(model_input, **la_pred_params)
    t_laplace = timeit.timeit(lambda: la(model_input), number=n_timing_samples)

## Visualize results.
plt.rcParams.update(
    {
        "font.size": 12,
        "font.weight": "normal",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

# FastBNNs:
x, sort_inds = torch.sort(model_input.squeeze().cpu())
y_fastbnns = output.mu[:, 0].detach().cpu().squeeze()[sort_inds]
y_var_aleatoric_fastbnns = (
    torch.nn.functional.softplus(output.mu[:, 1]).detach().cpu().squeeze()[sort_inds]
    ** 2
)
y_var_epistemic_fastbnns = output.var[:, 0].detach().cpu().squeeze()[sort_inds]
yerr_fastbnns = (y_var_aleatoric_fastbnns + y_var_epistemic_fastbnns).sqrt()
y_gt = data_generator.simulator(
    x=x[:, None], **data_generator.simulator_kwargs
).squeeze()
yerr_gt = noise_tform.noise_fxn_kwargs_generator["sigma"](x)
observations = observations.cpu().squeeze()[sort_inds]
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4), constrained_layout=True)
ax[0].plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax[0].fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax[0].fill_between(
    x=x,
    y1=y_fastbnns - yerr_fastbnns,
    y2=y_fastbnns + yerr_fastbnns,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax[0].plot(
    x,
    y_fastbnns,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax[0].set_title(f"FastBNNs: inference time {t_fastbnns:.2f} s")
ax[0].set_xlabel("input")
ax[0].set_ylabel("prediction")
ax[0].set_xlim((x.min(), x.max()))
ylim = (y_gt.min(), y_gt.max())
ax[0].set_ylim(ylim)
ax[0].plot([-0.5, -0.5], ylim, "--", color="b")
ax[0].plot([0.5, 0.5], ylim, "--", color="b")
ax[0].text(0.0, -2.75, "I.D.", weight="bold", ha="center", va="center")
ax[0].text(0.75, -2.75, "O.O.D.", weight="bold", ha="center", va="center")
ax[0].text(-0.75, -2.75, "O.O.D.", weight="bold", ha="center", va="center")
ax[0].legend(loc="upper left", framealpha=1)

# Bayesian-Torch:
x, sort_inds = torch.sort(model_input.cpu().squeeze())
y_bayestorch = output_bayestorch[0][:, 0].detach().cpu().squeeze()[sort_inds]
y_var_aleatoric_bayestorch = (
    torch.nn.functional.softplus(output_bayestorch[0][:, 1])
    .detach()
    .cpu()
    .squeeze()[sort_inds]
    ** 2
)
y_var_epistemic_bayestorch = (
    output_bayestorch[1][:, 0].detach().cpu().squeeze()[sort_inds]
)
yerr_bayestorch = (y_var_aleatoric_bayestorch + y_var_epistemic_bayestorch).sqrt()
ax[1].plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax[1].fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax[1].fill_between(
    x=x,
    y1=y_bayestorch - yerr_bayestorch,
    y2=y_bayestorch + yerr_bayestorch,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax[1].plot(
    x,
    y_bayestorch,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax[1].set_title(f"Bayesian-Torch: inference time {t_bayestorch:.2f} s")
ax[1].set_xlabel("input")
ax[1].set_ylabel("prediction")
ax[1].set_xlim((x.min(), x.max()))
ax[1].set_ylim(ylim)
ax[1].plot([-0.5, -0.5], ylim, "--", color="b")
ax[1].plot([0.5, 0.5], ylim, "--", color="b")

# Laplace approximation:
y_laplace = output_laplace[0].cpu().squeeze()[sort_inds]
yerr_laplace = torch.sqrt(
    output_laplace[1].cpu().squeeze()[sort_inds]
    + log_sigma_noise.detach().exp().cpu() ** 2
)
ax[2].plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax[2].fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax[2].fill_between(
    x=x,
    y1=y_laplace - yerr_laplace,
    y2=y_laplace + yerr_laplace,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax[2].plot(
    x,
    y_laplace,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax[2].set_title(f"Laplace: inference time {t_laplace:.2f} s")
ax[2].set_xlabel("input")
ax[2].set_ylabel("prediction")
ax[2].set_xlim((x.min(), x.max()))
ax[2].set_ylim(ylim)
ax[2].plot([-0.5, -0.5], ylim, "--", color="b")
ax[2].plot([0.5, 0.5], ylim, "--", color="b")
fig.savefig("polynomial.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# Test set MSEs:
loss_mse = torch.nn.MSELoss(reduction="mean")
print(f"test set MSE FastBNNs: {loss_mse(input=y_fastbnns, target=observations)}")
print(
    f"test set MSE Bayesian-Torch: {loss_mse(input=y_bayestorch, target=observations)}"
)
print(f"test set MSE Laplace: {loss_mse(input=y_laplace, target=observations)}")

# Test set likelihoods:
loss_gnll = torch.nn.GaussianNLLLoss(reduction="mean")
print(
    f"test set GNLL FastBNNs: {loss_gnll(input=y_fastbnns, target=observations, var=yerr_fastbnns**2)}"
)
print(
    f"test set GNLL Bayesian-Torch: {loss_gnll(input=y_bayestorch, target=observations, var=yerr_bayestorch**2)}"
)
print(
    f"test set GNLL Laplace: {loss_gnll(input=y_laplace, target=observations, var=yerr_laplace**2)}"
)

# Time comparisons:
print(f"inference runtime FastBNNs: {t_fastbnns}")
print(f"inference runtime Bayesian-Torch: {t_bayestorch}")
print(f"inference runtime Laplace: {t_laplace}")
