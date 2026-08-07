"""Example of training a Bayesian MLP in Lightning."""

import copy
from functools import partial

import lightning as L
import matplotlib.pyplot as plt
import torch

from fastbnns.bnn import base, losses, priors, types
from fastbnns.datasets import polynomial
from fastbnns.models import lightning_wrappers, mlp
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
bnn = base.BNN(nn=nn)

# Define a prior (this one applies to all parameters in the model).
prior = priors.Distribution(
    torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
)

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

# Prepare a Lightning module.
optimizer = partial(torch.optim.AdamW, lr=1.0e-2)
bnn_lightning = lightning_wrappers.BNNLightning(
    bnn=bnn,
    loss=loss_fn,
    optimizer=optimizer,
)

# Train.
n_epochs = 100
trainer = L.Trainer(
    max_epochs=n_epochs,
    check_val_every_n_epoch=1,
    accelerator="auto",
    devices="auto",
)
trainer.fit(model=bnn_lightning, train_dataloaders=dl_train, val_dataloaders=dl_val)

# Plot some examples.
input = []
output = []
n_examples = 1000
bnn = bnn.to("cpu")
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
input = torch.stack(input, dim=0)
with torch.no_grad():
    output = bnn(types.MuVar(input))

x, sort_inds = torch.sort(input.cpu().squeeze())
y = output.mu.cpu().squeeze()[sort_inds]
yerr = output.var.cpu().sqrt().squeeze()[sort_inds]
y_gt = data_generator.simulator(x=x, **data_generator.simulator_kwargs)
yerr_gt = noise_tform.noise_fxn_kwargs_generator["sigma"](x)
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
fig.savefig("mlp_lightning.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Done")
