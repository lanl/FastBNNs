"""Example of training a Bayesian MLP."""

import copy

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
bnn = base.BNN(nn=nn, convert_in_place=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnn = bnn.to(device)

# Define a prior (this one applies to all parameters in the model).
prior = priors.Distribution(
    torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
).to(device)

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
n_data = 1024 * 5
batch_size = 128
ds_train = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=noise_tform,
    cache=False,  # use fresh data every epoch
)
dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size)
n_data_val = 1024 * 2
ds_val = polynomial.Polynomial(
    data_generator=data_generator,
    dataset_length=n_data_val,
    transform=noise_tform,
    cache=True,
)
dl_val = torch.utils.data.DataLoader(dataset=ds_val, batch_size=batch_size)

# Define optimizer and loss.
n_batches = n_data // batch_size
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=1.0 / n_batches,  # see Graves 2011
)
n_epochs = 100
optimizer = torch.optim.AdamW(bnn.parameters(), lr=1.0e-2)

# Train.
loss_train = []
loss_val = []
best_model_state_dict = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
for epoch in range(n_epochs):
    loss_epoch = []
    bnn.train()
    for batch in dl_train:
        # Forward pass through model.
        optimizer.zero_grad()
        out = bnn(types.MuVar(batch[0].to(device)))

        # Compute loss.
        loss = loss_fn(model=bnn, input=out.mu, target=batch[1].to(device), var=out.var)

        # Update model.
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss)

    avg_loss_train = torch.mean(torch.stack(loss_epoch))
    loss_train.append(avg_loss_train)

    # Evaluate on validation set.
    with torch.no_grad():
        loss_epoch_val = []
        within_1sigma_val = []
        bnn.eval()
        for batch in dl_val:
            # Forward pass through model.
            out = bnn(types.MuVar(batch[0].to(device)))

            # Compute loss.
            loss = loss_fn(
                model=bnn,
                input=out.mu,
                target=batch[1].to(device),
                var=out.var,
            )
            loss_epoch_val.append(loss)

            # Check predictive variance.
            within_1sigma_val.append(
                statistics.compute_coverage(
                    observations=batch[1].to(device),
                    mu=out.mu,
                    sigma=out.var.sqrt(),
                    alphas=torch.tensor([1.0]),
                )
            )

    avg_loss_val = torch.mean(torch.stack(loss_epoch_val))
    loss_val.append(avg_loss_val)
    if avg_loss_val < best_loss:
        best_loss = avg_loss_val
        best_model_state_dict = copy.deepcopy(bnn.state_dict())
    print(
        f"epoch {epoch + 1} of {n_epochs}: loss = {avg_loss_val}, {100.0 * torch.mean(torch.stack(within_1sigma_val)):.2f}% within 1 st. dev."
    )

# Plot some examples.
final_model_state_dict = copy.deepcopy(bnn.state_dict())
bnn.load_state_dict(best_model_state_dict)
bnn = bnn.to("cpu")
input = []
observations = []
n_examples = 1000
data_generator_test = copy.deepcopy(data_generator)
data_generator_test.simulator_kwargs_generator["x"] = lambda: (
    2.0 * (torch.rand(1) - 0.5)
)
ds_test = polynomial.Polynomial(
    data_generator=data_generator_test,
    dataset_length=n_examples,
    transform=noise_tform,
    cache=True,
)
for n in range(n_examples):
    data = ds_test[n]
    input.append(data[0])
    observations.append(data[1])
input = torch.stack(input, dim=0)
observations = torch.stack(observations, dim=0)
with torch.no_grad():
    output = bnn(types.MuVar(input))

x, sort_inds = torch.sort(input.cpu().squeeze())
y = output.mu.cpu().squeeze()[sort_inds]
yerr = output.var.cpu().sqrt().squeeze()[sort_inds]
y_gt = data_generator.simulator(x=x, **data_generator.simulator_kwargs)
yerr_gt = (
    (noise_tform.noise_fxn_kwargs_generator["sigma"](x)) ** 2 + (y - y_gt) ** 2
).sqrt()
observations = observations.cpu().squeeze()[sort_inds]
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
    y1=y - yerr,
    y2=y + yerr,
    alpha=0.5,
    color="m",
    label="predicted uncertainty",
)
ax.plot(x, y, marker=".", linestyle="", label="predicted mean")
ax.set_ylim((y_gt.min(), y_gt.max()))
ylim = ax.get_ylim()
ax.plot([-0.5, -0.5], ylim, "--", color="g")
ax.plot([0.5, 0.5], ylim, "--", color="g")
ax.text(0.0, -0.75, "I.D.")
ax.text(0.75, -0.75, "O.O.D.")
ax.set_xlabel("input")
ax.set_ylabel("prediction")
fig.legend(loc="upper left")
fig.savefig("mlp.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Done")
