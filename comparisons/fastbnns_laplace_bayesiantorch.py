"""Example comparing Bayesian MLP trained with fastbnns, laplace, and bayesian_torch."""

import copy
import time
import timeit

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import laplace
import matplotlib.pyplot as plt
import torch

from fastbnns.analysis import statistics
from fastbnns.bnn import base, losses, priors, types
from fastbnns.datasets import polynomial
from fastbnns.models import mlp
from fastbnns.simulation import generators, polynomials, observation

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# Create a Bayesian multilayer perceptron to model a linear function y=mx+b.
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn.to(device)

# Define a dataset.
data_generator = generators.Generator(
    simulator=polynomials.polynomial,
    simulator_kwargs={"coefficients": torch.tensor([0.0, 1.0])},
    simulator_kwargs_generator={"x": lambda: torch.rand(1) - 0.5},
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.add_read_noise,
    noise_fxn_kwargs_generator={
        "sigma": lambda x: 0.1 + 0.2 * (torch.cos(2.0 * torch.pi * x) ** 2)
    },
)
n_data = 1024 * 10
batch_size = 128
ds_train = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=noise_tform,
    cache=True,
)
ds_val = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=noise_tform,
    cache=True,
)
ds_test = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=noise_tform,
    cache=True,
)
dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size)
dl_val = torch.utils.data.DataLoader(dataset=ds_val, batch_size=batch_size)

# Train a standard NN.
optimizer = torch.optim.AdamW(nn.parameters(), lr=1.0e-2)
best_model_state_dict_nn = copy.deepcopy(nn.state_dict())
best_loss = torch.inf
loss_fn_nn = torch.nn.MSELoss()
n_epochs_max = 300  # maximum allowed epochs
patience = 20  # early stopping patience in epochs
n_batches = n_data // batch_size
print("______________________________________________________________________")
print("Training base neural network...")
epochs_without_val_loss_decrease = 0
t0 = time.time()
for epoch in range(n_epochs_max):
    # Early stopping.
    if epochs_without_val_loss_decrease >= patience:
        print(f"Stopping early at epoch {epoch} due to val loss plateau")
        break

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
    print(
        f"epoch {epoch + 1} of {n_epochs_max}: train mse = {avg_loss_train}, val mse = {avg_loss_val}"
    )

t_train_nn = time.time() - t0
nn.load_state_dict(best_model_state_dict_nn)

# Train a BNN with FastBNNs.
bnn = base.BNN(
    nn=nn,
    convert_in_place=False,
    wrapper_kwargs_global={"resample_mean": False},  # init parameter means to `nn`
)
prior_mu = 0.0
prior_sigma = 0.5
bnn.to(device)
prior = priors.Distribution(
    torch.distributions.Normal(
        loc=torch.tensor([prior_mu]), scale=torch.tensor([prior_sigma])
    )
).to(device)
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=1.0 / n_batches,  # see Graves 2011
)
optimizer = torch.optim.AdamW(bnn.parameters(), lr=1.0e-2)
best_model_state_dict = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
print("______________________________________________________________________")
print("Training Bayesian neural network with fastbnns...")
epochs_without_val_loss_decrease = 0
t0 = time.time()
for epoch in range(n_epochs_max):
    # Early stopping.
    if epochs_without_val_loss_decrease >= patience:
        print(f"Stopping early at epoch {epoch} due to val loss plateau")
        break

    # Training set:
    bnn.train()
    loss_epoch_train = []
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = bnn(types.MuVar(batch[0].to(device)))

        # Compute loss.
        loss = loss_fn(model=bnn, input=out.mu, target=batch[1].to(device), var=out.var)

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

            # Compute loss.
            loss = loss_fn(
                model=bnn, input=out.mu, target=batch[1].to(device), var=out.var
            )
            loss_epoch_val.append(loss)

            # Check predictive variance.
            within_1sigma_epoch_val.append(
                statistics.compute_coverage(
                    observations=batch[1].to(device),
                    mu=out.mu,
                    sigma=out.var.sqrt(),
                    alphas=torch.tensor([1.0]),
                )
            )

    avg_loss_train = torch.mean(torch.stack(loss_epoch_train))
    avg_loss_val = torch.mean(torch.stack(loss_epoch_val))

    if avg_loss_val < best_loss:
        epochs_without_val_loss_decrease = 0
        best_loss = avg_loss_val
        best_model_state_dict = copy.deepcopy(bnn.state_dict())
    else:
        epochs_without_val_loss_decrease += 1
    print(
        f"epoch {epoch + 1} of {n_epochs_max}: train ELBO = {avg_loss_train}, val ELBO = {avg_loss_val}, {100.0 * torch.mean(torch.stack(within_1sigma_epoch_val)):.2f}% within 1 st. dev."
    )
t_train_fastbnns = time.time() - t0
bnn.load_state_dict(best_model_state_dict)

# Train a BNN with Bayesian-Torch.
bayestorch_prior = {
    "prior_mu": prior_mu,
    "prior_sigma": prior_sigma,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": True,
    "moped_delta": 1.0,
}
bnn_bayestorch = copy.deepcopy(nn)
dnn_to_bnn(bnn_bayestorch, bayestorch_prior)
bnn_bayestorch.to(device)
nll_loss = torch.nn.GaussianNLLLoss(reduction="sum")
optimizer = torch.optim.AdamW(bnn_bayestorch.parameters(), lr=1.0e-2)
best_model_state_dict_bayestorch = copy.deepcopy(bnn_bayestorch.state_dict())
best_loss = torch.inf
n_mc_samples_train = 30
print("______________________________________________________________________")
print("Training Bayesian neural network with bayesian_torch...")
epochs_without_val_loss_decrease = 0
t0 = time.time()
for epoch in range(n_epochs_max):
    # Early stopping.
    if epochs_without_val_loss_decrease >= patience:
        print(f"Stopping early at epoch {epoch} due to val loss plateau")
        break

    # Training set:
    bnn_bayestorch.train()
    loss_epoch_train = []
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = torch.stack(
            [bnn_bayestorch(batch[0].to(device)) for _ in range(n_mc_samples_train)]
        )

        # Compute loss.
        kl = get_kl_loss(bnn_bayestorch)
        loss = (
            nll_loss(
                input=torch.mean(out, dim=0),
                target=batch[1].to(device),
                var=torch.var(out, dim=0),
            )
            + kl / n_batches
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
            out_mu = torch.mean(out, dim=0)
            out_var = torch.var(out, dim=0)
            loss = (
                nll_loss(
                    input=out_mu,
                    target=batch[1].to(device),
                    var=out_var,
                )
                + kl / n_batches
            )
            loss_epoch_val.append(loss)

            # Check predictive variance.
            within_1sigma_epoch_val.append(
                statistics.compute_coverage(
                    observations=batch[1].to(device),
                    mu=out_mu,
                    sigma=out_var.sqrt(),
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

    print(
        f"epoch {epoch + 1} of {n_epochs_max}: train ELBO = {avg_loss_train}, val ELBO = {avg_loss_val}, {100.0 * torch.mean(torch.stack(within_1sigma_epoch_val)):.2f}% within 1 st. dev."
    )
t_train_bayestorch = time.time() - t0
bnn_bayestorch.load_state_dict(best_model_state_dict_bayestorch)

# Use laplace-torch to get a BNN from pretrained NN.
prior_precision = 1.0 / prior_sigma**2
la = laplace.DiagLaplace(
    nn,
    likelihood="regression",
    prior_precision=prior_precision,
)
print("______________________________________________________________________")
print("Training Bayesian neural network with laplace...")
t0 = time.time()
la.fit(dl_train)
log_sigma_noise = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.AdamW([log_sigma_noise], lr=0.1)
n_steps = 200
for _ in range(n_steps):
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
t_train_laplace = time.time() - t0

# Plot some examples.
inference_device = "cuda"
bnn = bnn.to(inference_device).eval()
bnn_bayestorch.to(inference_device).eval()
la.model.to(inference_device).eval()
input = []
observations = []
n_examples = 1000
ds_test.data_generator.simulator_kwargs_generator["x"] = lambda: (
    2.0 * (torch.rand(1) - 0.5)
)
for n in range(n_examples):
    data = ds_test[n]
    input.append(data[0])
    observations.append(data[1])
input = torch.stack(input, dim=0).to(inference_device)
observations = torch.stack(observations, dim=0)
n_mc_samples_test = 30
la_pred_params = {
    "pred_type": "glm",
    "link_approx": "bridge_norm",
}
with torch.no_grad():
    output = bnn(types.MuVar(input))
    n_timing_samples = 100
    t_fastbnns = timeit.timeit(lambda: bnn(types.MuVar(input)), number=n_timing_samples)
    output_bayestorch = torch.stack(
        [bnn_bayestorch(input) for _ in range(n_mc_samples_test)]
    )
    t_bayestorch = timeit.timeit(
        lambda: torch.stack([bnn_bayestorch(input) for _ in range(n_mc_samples_test)]),
        number=n_timing_samples,
    )
    output_bayestorch = (
        torch.mean(output_bayestorch, dim=0),
        torch.var(output_bayestorch, dim=0),
    )
    output_laplace = la(input, **la_pred_params)
    t_laplace = timeit.timeit(lambda: la(input), number=n_timing_samples)

# FastBNNs:
x, sort_inds = torch.sort(input.cpu().squeeze())
y_fastbnns = output.mu.cpu().squeeze()[sort_inds]
yerr_fastbnns = output.var.cpu().sqrt().squeeze()[sort_inds]
y_gt = data_generator.simulator(x=x, **data_generator.simulator_kwargs)
yerr_gt = noise_tform.noise_fxn_kwargs_generator["sigma"](x)
observations = observations.cpu().squeeze()[sort_inds]

plt.rcParams.update(
    {
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)
fig, ax = plt.subplots()
ax.plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax.fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax.fill_between(
    x=x,
    y1=y_fastbnns - yerr_fastbnns,
    y2=y_fastbnns + yerr_fastbnns,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax.plot(
    x,
    y_fastbnns,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax.set_xlabel("input")
ax.set_ylabel("prediction")
ax.set_xlim((x.min(), x.max()))
ax.set_ylim((y_gt.min(), y_gt.max()))
# fig.legend(loc="upper left")
fig.savefig("fastbnns.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# Bayesian-Torch:
x, sort_inds = torch.sort(input.cpu().squeeze())
y_bayestorch = output_bayestorch[0].cpu().squeeze()[sort_inds]
yerr_bayestorch = output_bayestorch[1].cpu().sqrt().squeeze()[sort_inds]

fig, ax = plt.subplots()
ax.plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax.fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax.fill_between(
    x=x,
    y1=y_bayestorch - yerr_bayestorch,
    y2=y_bayestorch + yerr_bayestorch,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax.plot(
    x,
    y_bayestorch,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax.set_xlabel("input")
ax.set_ylabel("prediction")
ax.set_xlim((x.min(), x.max()))
ax.set_ylim((y_gt.min(), y_gt.max()))
# fig.legend(loc="upper left")
fig.savefig("bayesian_torch.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# Laplace approximation:
y_laplace = output_laplace[0].cpu().squeeze()[sort_inds]
yerr_laplace = torch.sqrt(
    output_laplace[1].cpu().squeeze()[sort_inds]
    + log_sigma_noise.detach().exp().cpu() ** 2
)

fig, ax = plt.subplots()
ax.plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax.fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax.fill_between(
    x=x,
    y1=y_laplace - yerr_laplace,
    y2=y_laplace + yerr_laplace,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax.plot(
    x,
    y_laplace,
    color="g",
    marker="",
    linewidth=3,
    linestyle="-",
    label="predicted mean",
)
ax.set_xlabel("input")
ax.set_ylabel("prediction")
ax.set_xlim((x.min(), x.max()))
ax.set_ylim((y_gt.min(), y_gt.max()))
# fig.legend(loc="upper left")
fig.savefig("laplace.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# Test set MSEs:
loss_mse = torch.nn.MSELoss(reduction="mean")
print(f"test set MSE fastbnns: {loss_mse(input=y_fastbnns, target=observations)}")
print(
    f"test set MSE bayesian-torch: {loss_mse(input=y_bayestorch, target=observations)}"
)
print(f"test set MSE laplace: {loss_mse(input=y_laplace, target=observations)}")

# Test set likelihoods:
loss_gnll = torch.nn.GaussianNLLLoss(reduction="mean")
print(
    f"test set GNLL fastbnns: {loss_gnll(input=y_fastbnns, target=observations, var=yerr_fastbnns**2)}"
)
print(
    f"test set GNLL bayesian-torch: {loss_gnll(input=y_bayestorch, target=observations, var=yerr_bayestorch**2)}"
)
print(
    f"test set GNLL laplace: {loss_gnll(input=y_laplace, target=observations, var=yerr_laplace**2)}"
)

# Time comparisons:
print(f"training runtime fastbnns: {t_train_fastbnns}")
print(f"training runtime bayesian-torch: {t_train_bayestorch}")
print(f"training runtime laplace: {t_train_laplace}")
print(f"inference runtime fastbnns: {t_fastbnns}")
print(f"inference runtime bayesian-torch: {t_bayestorch}")
print(f"inference runtime laplace: {t_laplace}")
